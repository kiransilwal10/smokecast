"""Training loop.

    python -m smokecast.train --config configs/smokecast.yaml
    python -m smokecast.train --config configs/smokecast.yaml --set model.use_graph=false --name gru
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
import yaml

from .dataset import SMOKE, USG, Batcher, SmokeData
from .model import SmokeCast


def load_config(path: Path, overrides: list[str] = ()) -> dict:
    cfg = yaml.safe_load(Path(path).read_text())
    for item in overrides:
        key, value = item.split("=", 1)
        node = cfg
        *parents, leaf = key.split(".")
        for p in parents:
            node = node[p]
        node[leaf] = yaml.safe_load(value)
    return cfg


def pick_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def event_weights(data: SmokeData, t0: np.ndarray, boost: float) -> np.ndarray:
    """Oversample hours when smoke or high PM2.5 is widespread. Uses only what is known at t0."""
    pm = np.asarray(data.pm[t0])
    high = (np.nan_to_num(pm) > USG).mean(1) >= 0.05
    smoky = (np.asarray(data.x[t0, :, SMOKE[0]], dtype=np.float32) >= 0.6).mean(1) >= 0.10
    return 1.0 + boost * (high | smoky)


def loss_weights(batch: dict) -> torch.Tensor:
    """Every valid target at leads >= 1; at lead 0 only the hidden sites (the rest would be trivial)."""
    w = batch["y_valid"].float()
    w[:, 0] *= batch["node_mask"].float()
    return w


def masked_mse(pred, y, w):
    return ((pred - y) ** 2 * w).sum() / w.sum().clamp_min(1.0)


def build_model(cfg: dict, data: SmokeData) -> SmokeCast:
    return SmokeCast(data.lat, data.lon, in_channels=data.x.shape[-1], horizon=cfg["horizon"],
                     history=cfg["history"], **cfg["model"])


@torch.no_grad()
def predict(model, batcher, t0, leads=None, node_mask=None, batch_size: int = 64) -> np.ndarray:
    """Normalised predictions (len(t0), len(leads), N); all leads 0..H when ``leads`` is None."""
    model.eval()
    out = []
    for i in range(0, len(t0), batch_size):
        mask = None if node_mask is None else node_mask.expand(len(t0[i:i + batch_size]), -1)
        p = model(batcher(t0[i:i + batch_size], node_mask=mask))
        out.append((p if leads is None else p[:, list(leads)]).float().cpu().numpy())
    return np.concatenate(out)


@torch.no_grad()
def validation_loss(model, batcher, t0, batch_size: int = 64) -> float:
    model.eval()
    num, den = 0.0, 0.0
    gen = torch.Generator(device="cpu").manual_seed(1)
    for i in range(0, len(t0), batch_size):
        rows = t0[i:i + batch_size]
        mask = (torch.rand(len(rows), batcher.x.shape[1], generator=gen) < batcher.mask_frac).to(batcher.device)
        b = batcher(rows, node_mask=mask)
        w = loss_weights(b)
        num += float(((model(b) - b["y"]) ** 2 * w).sum())
        den += float(w.sum())
    return num / max(den, 1.0)


def train(cfg: dict) -> Path:
    tc = cfg["train"]
    torch.manual_seed(tc["seed"])
    np.random.seed(tc["seed"])
    device = pick_device()
    run_dir = Path(cfg.get("run_dir", "runs")) / cfg["name"]
    run_dir.mkdir(parents=True, exist_ok=True)

    data = SmokeData(Path(cfg["data_dir"]))
    splits = data.meta["splits"]
    mask_frac = tc["mask_frac"] if cfg["model"].get("use_graph", True) else 0.0
    batcher = Batcher(data, cfg["history"], cfg["horizon"], device, mask_frac=mask_frac)
    t_train = data.forecast_times(tuple(splits["train"]), cfg["history"], cfg["horizon"])
    t_val = data.forecast_times(tuple(splits["val"]), cfg["history"], cfg["horizon"], tc["val_stride"])
    weights = torch.as_tensor(event_weights(data, t_train, tc["event_boost"]))
    print(f"{data.n_nodes} sites | {len(t_train)} train windows "
          f"({float((weights > 1).float().mean()):.1%} smoke/high-PM hours) | {len(t_val)} val | {device}")

    model = build_model(cfg, data).to(device)
    print(f"{sum(p.numel() for p in model.parameters()):,} parameters")
    opt = torch.optim.AdamW(model.parameters(), lr=tc["lr"], weight_decay=tc["weight_decay"])
    total = tc["epochs"] * tc["steps_per_epoch"]
    warmup = min(500, total // 10)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, (s + 1) / warmup) * 0.5 * (1 + math.cos(math.pi * min(s, total) / total)))

    history, best, best_state, start, stale = [], float("inf"), None, 0, 0
    resume = run_dir / "last.pt"
    if resume.exists():  # pick up where an interrupted run stopped
        state = torch.load(resume, map_location=device, weights_only=False)
        if state["config"] == cfg:
            model.load_state_dict(state["model"])
            opt.load_state_dict(state["opt"])
            sched.load_state_dict(state["sched"])
            history, best, best_state, start, stale = (state["history"], state["best"], state["best_state"],
                                                       state["epoch"], state["stale"])
            torch.set_rng_state(state["rng"].cpu())
            print(f"resuming after epoch {start}")

    for epoch in range(start, tc["epochs"]):
        if stale >= tc["patience"]:
            print(f"no improvement for {stale} epochs, stopping")
            break
        model.train()
        t_start, running = time.time(), 0.0
        for _ in range(tc["steps_per_epoch"]):
            idx = torch.multinomial(weights, tc["batch_size"], replacement=True).numpy()
            b = batcher(t_train[idx], train=True)
            loss = masked_mse(model(b), b["y"], loss_weights(b))
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            running += loss.item()
        val = validation_loss(model, batcher, t_val)
        history.append({"epoch": epoch + 1, "train": running / tc["steps_per_epoch"], "val": val,
                        "seconds": time.time() - t_start})
        flag = ""
        if val < best:
            best, best_state, stale, flag = val, copy.deepcopy(model.state_dict()), 0, " *"
        else:
            stale += 1
        print(f"epoch {epoch + 1:3d}  train {history[-1]['train']:.4f}  val {val:.4f}  "
              f"({history[-1]['seconds']:.0f}s){flag}", flush=True)
        torch.save({"config": cfg, "epoch": epoch + 1, "model": model.state_dict(), "opt": opt.state_dict(),
                    "sched": sched.state_dict(), "history": history, "best": best, "best_state": best_state,
                    "stale": stale, "rng": torch.get_rng_state()}, resume)

    torch.save({"config": cfg, "state_dict": best_state}, run_dir / "model.pt")
    (run_dir / "history.json").write_text(json.dumps(history, indent=1))
    resume.unlink()
    return run_dir


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--set", nargs="*", default=[], help="overrides like model.use_graph=false")
    ap.add_argument("--name", default=None)
    args = ap.parse_args()
    cfg = load_config(args.config, args.set)
    if args.name:
        cfg["name"] = args.name
    train(cfg)


if __name__ == "__main__":
    main()
