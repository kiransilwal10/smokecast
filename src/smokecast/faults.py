"""Sensor fault detection with the virtual sensor.

Faults are injected into the test-period record of real monitors, five days each:

    offset   reads 10 µg/m³ high from the start (a calibration error)
    drift    error grows linearly to 20 µg/m³ (a fouled inlet or ageing optics)
    stuck    repeats the reading from the first hour (a frozen logger)
    noise    random error with a 10 µg/m³ spread (a failing laser counter)

Two detectors watch the corrupted record:

    own history     robust z-score of each reading against the site's own trailing 14 days
    virtual sensor  each reading against SmokeCast's estimate of it from the neighbours,
                    with the site itself hidden from the model

Both flag a site when the 24 h root-mean-square of their score passes a threshold. The
thresholds are set on clean data so both detectors raise the same number of false alarms,
which makes the detection rates directly comparable.

    python -m smokecast.faults --run smokecast
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import torch

from .dataset import FILL_HOURS, SMOKE, Batcher, SmokeData, ffill
from .train import build_model, pick_device

KINDS = ("offset", "drift", "stuck", "noise")
EPISODE_HOURS = 120
FALSE_ALARM_RATE = 0.005  # fraction of clean site-hours flagged
WINDOW = 24


def inject(pm: np.ndarray, rows: np.ndarray, rng: np.random.Generator, per_site: int = 2):
    """Corrupt ``per_site`` non-overlapping episodes at every site. Returns the copy and the episode list."""
    out = pm.copy()
    episodes = []
    span = np.arange(EPISODE_HOURS)
    for n in range(pm.shape[1]):
        starts = []
        while len(starts) < per_site:
            s = int(rng.integers(rows[0] + 24 * 14, rows[-1] - EPISODE_HOURS))
            if all(abs(s - o) > EPISODE_HOURS + 24 * 10 for o in starts):
                starts.append(s)
        for s in starts:
            kind = KINDS[int(rng.integers(len(KINDS)))]
            t = s + span
            ok = np.isfinite(out[t, n])
            if ok.mean() < 0.6:
                continue
            if kind == "offset":
                out[t, n] += 10.0
            elif kind == "drift":
                out[t, n] += 20.0 * span / (EPISODE_HOURS - 1)
            elif kind == "stuck":
                out[t, n] = out[t[ok][0], n]
            else:
                out[t, n] += rng.normal(0, 10.0, len(t))
            out[t, n] = np.where(ok, np.maximum(out[t, n], 0.0), np.nan)
            episodes.append({"site": n, "start": s, "kind": kind})
    return out, episodes


def greedy_colouring(src: np.ndarray, dst: np.ndarray, n: int) -> np.ndarray:
    """Groups of sites with no two neighbours in the same group, so each can be hidden at once."""
    nbrs = [set() for _ in range(n)]
    for a, b in zip(src, dst):
        nbrs[a].add(b)
    colour = -np.ones(n, int)
    for v in np.argsort([-len(s) for s in nbrs]):
        used = {colour[u] for u in nbrs[v]}
        colour[v] = next(c for c in range(n) if c not in used)
    return colour


@torch.no_grad()
def virtual_readings(model, view, rows: np.ndarray, colour: np.ndarray, device, batch_size: int = 96):
    """(len(rows), N) µg/m³ estimate of every site's current reading with that site hidden."""
    batcher = Batcher(view, 48, 0, device)  # only the current hour is needed as a target
    out = np.zeros((len(rows), len(colour)), np.float32)
    model.eval()
    for c in range(colour.max() + 1):
        mask = torch.as_tensor(colour == c, device=device)
        for i in range(0, len(rows), batch_size):
            t0 = rows[i:i + batch_size]
            pred = model(batcher(t0, node_mask=mask.expand(len(t0), -1)))[:, 0].float().cpu().numpy()
            out[i:i + batch_size, colour == c] = pred[:, colour == c]
    return np.maximum(view.mu + view.sigma * out, 0.0)


def rolling_rms(score: np.ndarray) -> np.ndarray:
    df = pd.DataFrame(score ** 2)
    return np.sqrt(df.rolling(WINDOW, min_periods=WINDOW // 2).mean().to_numpy())


def own_history_score(pm: np.ndarray, rows: np.ndarray) -> np.ndarray:
    """Robust z-score against the trailing 14 days of the same site (strictly before each hour)."""
    start = rows[0] - 24 * 15
    df = pd.DataFrame(pm[start:rows[-1] + 1])
    med = df.shift(1).rolling(24 * 14, min_periods=24 * 3).median()
    mad = (df.shift(1) - med).abs().rolling(24 * 14, min_periods=24 * 3).median()
    z = (df - med) / (1.4826 * mad).clip(lower=1.0)
    return z.to_numpy()[rows - start]


def detection(score: np.ndarray, threshold: np.ndarray, episodes, rows) -> dict:
    first = {k: [] for k in KINDS}
    for ep in episodes:
        i = ep["start"] - rows[0]
        flagged = np.nonzero(score[i:i + EPISODE_HOURS, ep["site"]] > threshold[ep["site"]])[0]
        first[ep["kind"]].append(int(flagged[0]) if len(flagged) else None)
    return {k: {"detected": float(np.mean([f is not None for f in v])),
                "median_hours_to_flag": float(np.median([f for f in v if f is not None])) if any(
                    f is not None for f in v) else None, "episodes": len(v)}
            for k, v in first.items()}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="smokecast")
    ap.add_argument("--run-root", type=Path, default=Path("runs"))
    ap.add_argument("--data", type=Path, default=Path("data/processed"))
    ap.add_argument("--out", type=Path, default=Path("results"))
    args = ap.parse_args()

    data = SmokeData(args.data)
    device = pick_device()
    ckpt = torch.load(args.run_root / args.run / "model.pt", map_location="cpu", weights_only=False)
    model = build_model(ckpt["config"], data).to(device)
    model.load_state_dict(ckpt["state_dict"])
    rows = data.forecast_times(tuple(data.meta["splits"]["test"]), 48, 0)

    pm_clean = np.asarray(data.pm)
    pm_bad, episodes = inject(pm_clean, rows, np.random.default_rng(0))
    x_bad = np.array(data.x)
    x_bad[..., 0] = np.nan_to_num((ffill(pm_bad, FILL_HOURS) - data.mu) / data.sigma).astype(np.float16)
    colour = greedy_colouring(model.src.cpu().numpy(), model.dst.cpu().numpy(), data.n_nodes)
    print(f"{len(episodes)} fault episodes, {colour.max() + 1} site groups", flush=True)

    views = {"clean": SimpleNamespace(x=data.x, pm=pm_clean, mu=data.mu, sigma=data.sigma, index=data.index),
             "faulty": SimpleNamespace(x=x_bad, pm=pm_bad, mu=data.mu, sigma=data.sigma, index=data.index)}
    scores = {}
    for name, view in views.items():
        print(f"virtual sensor on {name} data", flush=True)
        virtual = virtual_readings(model, view, rows, colour, device)
        obs = view.pm[rows]
        resid = obs - virtual
        if name == "clean":
            spread = np.nanmedian(np.abs(resid - np.nanmedian(resid, 0)), 0) * 1.4826
            np.save(args.run_root / args.run / "virtual_spread.npy", spread)
            np.save(args.run_root / args.run / "virtual_clean.npy", virtual.astype(np.float16))
        else:
            np.save(args.run_root / args.run / "virtual_faulty.npy", virtual.astype(np.float16))
        scores[name] = {"Virtual sensor (SmokeCast)": rolling_rms(resid / np.maximum(spread, 1.0)),
                        "Own history": rolling_rms(own_history_score(view.pm, rows))}

    smoky = (np.asarray(data.x[rows, :, SMOKE[0]], dtype=np.float32) >= 0.6)
    results = {"episodes": len(episodes), "false_alarm_rate": FALSE_ALARM_RATE, "detectors": {}}
    for det in scores["clean"]:
        clean = np.nan_to_num(scores["clean"][det])
        threshold = np.quantile(clean, 1 - FALSE_ALARM_RATE, axis=0)
        flags = clean > threshold
        results["detectors"][det] = {
            "by_fault": detection(np.nan_to_num(scores["faulty"][det]), threshold, episodes, rows),
            "false_alarms_in_smoke": float(flags[smoky].mean()),
            "false_alarms_outside_smoke": float(flags[~smoky].mean()),
        }
        np.save(args.run_root / args.run / f"score_{det.split()[0].lower()}.npy",
                np.stack([scores["clean"][det], scores["faulty"][det]]).astype(np.float16))
        np.save(args.run_root / args.run / f"threshold_{det.split()[0].lower()}.npy", threshold)
    (args.run_root / args.run / "fault_episodes.json").write_text(json.dumps(episodes))
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "faults.json").write_text(json.dumps(results, indent=1))
    print(json.dumps(results, indent=1))


if __name__ == "__main__":
    main()
