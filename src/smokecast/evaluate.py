"""Test-set evaluation: baselines, the ablation, smoke and high-PM subsets, exceedance skill,
and the virtual sensor with sites hidden.

    python -m smokecast.evaluate --runs gru gnn_static nosmoke smokecast
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from numpy.lib.stride_tricks import sliding_window_view

from .dataset import SMOKE, USG, Batcher, SmokeData, time_features
from .geo import haversine_km
from .train import build_model, pick_device, predict

LEADS = (1, 3, 6, 12, 24, 48, 72)
SAVED = (0,) + LEADS
LABELS = {"gru": "GRU per site (no graph)", "gnn_static": "GNN, static edges",
          "nosmoke": "SmokeCast, no satellite smoke", "smokecast": "SmokeCast"}
HIDDEN_FRAC = 0.3


def climatology(data: SmokeData, rows: np.ndarray) -> np.ndarray:
    """Mean PM2.5 by (month, hour of day, site) over ``rows``."""
    pm = np.asarray(data.pm[rows])
    month, hour = data.index.month[rows] - 1, data.index.hour[rows]
    clim = np.zeros((12, 24, data.n_nodes), np.float32)
    for m in range(12):
        for h in range(24):
            sel = (month == m) & (hour == h)
            clim[m, h] = np.nanmean(pm[sel], 0)
    return clim


def ridge_baseline(data: SmokeData, history: int, t_fit: np.ndarray, t_val: np.ndarray,
                   t_out: list[np.ndarray]) -> list[np.ndarray]:
    """Per-site ridge on the site's own flattened inputs, penalty picked on ``t_val``.

    Returns forecasts in µg/m³, (len(t), L, N), for each array of issue times in ``t_out``.
    """
    x = np.asarray(data.x, dtype=np.float32)
    tf = time_features(data.index)
    pm = np.asarray(data.pm)
    outs = [np.zeros((len(t), len(LEADS), data.n_nodes), np.float32) for t in t_out]
    leads = np.array(LEADS)
    for n in range(data.n_nodes):
        win = sliding_window_view(x[:, n], history, axis=0)  # (T - history + 1, C, history)
        feats = lambda t0: np.concatenate([win[t0 - history + 1].reshape(len(t0), -1), tf[t0],
                                           np.ones((len(t0), 1), np.float32)], 1)
        base = lambda t0: x[t0, n, 0] * x[t0, n, 1]
        target = lambda t0: (pm[t0[:, None] + leads, n] - data.mu[n]) / data.sigma[n] - base(t0)[:, None]
        Xf, Yf = feats(t_fit), target(t_fit)
        ok = np.isfinite(Yf).all(1)
        Xf, Yf = Xf[ok], Yf[ok]
        Xv, Yv = feats(t_val), target(t_val)
        okv = np.isfinite(Yv)
        gram, cross = Xf.T @ Xf, Xf.T @ Yf
        best = None
        for lam in (1.0, 10.0, 100.0, 1000.0):
            W = np.linalg.solve(gram + lam * len(Xf) * 1e-3 * np.eye(len(gram)), cross)
            err = np.nanmean(((Xv @ W - Yv) ** 2)[okv])
            if best is None or err < best[0]:
                best = (err, W)
        for t, out in zip(t_out, outs):
            out[:, :, n] = data.mu[n] + data.sigma[n] * (feats(t) @ best[1] + base(t)[:, None])
    return [np.maximum(out, 0.0) for out in outs]


def idw_virtual(data: SmokeData, t0: np.ndarray, hidden: np.ndarray, k: int = 8) -> np.ndarray:
    """Inverse-distance interpolation of the current reading at hidden sites from visible neighbours."""
    d = haversine_km(data.lat[:, None], data.lon[:, None], data.lat[None], data.lon[None])
    pm = np.asarray(data.x[t0, :, 0], dtype=np.float32) * data.sigma + data.mu
    valid = np.asarray(data.x[t0, :, 1], dtype=np.float32) > 0.5
    out = np.full((len(t0), data.n_nodes), np.nan, np.float32)
    for i in range(len(t0)):
        for n in np.nonzero(hidden[i])[0]:
            ok = valid[i] & ~hidden[i]
            ok[n] = False
            cand = np.nonzero(ok)[0]
            if len(cand) == 0:
                continue
            near = cand[np.argsort(d[n, cand])[:k]]
            w = 1.0 / np.maximum(d[n, near], 1.0) ** 2
            out[i, n] = (w * pm[i, near]).sum() / w.sum()
    return out


def rmse(pred, obs, mask) -> float:
    m = mask & np.isfinite(obs) & np.isfinite(pred)
    return float(np.sqrt(np.mean((pred[m] - obs[m]) ** 2))) if m.any() else float("nan")


def exceedance(pred, obs, mask, level: float = USG) -> dict:
    """Skill of warning when the forecast exceeds ``level`` for observed PM2.5 above USG."""
    m = mask & np.isfinite(obs)
    p, o = (pred > level) & m, (obs > USG) & m
    hits, misses, false = int((p & o).sum()), int((~p & o).sum()), int((p & ~o).sum())
    return {"hit_rate": hits / max(hits + misses, 1), "false_alarm_ratio": false / max(hits + false, 1),
            "csi": hits / max(hits + misses + false, 1), "events": hits + misses}


def tune_warning_level(pred, obs) -> float:
    """Forecast level above which to warn, chosen to maximise CSI on validation data.

    Models trained on mean-squared error under-forecast peaks, so warning only when the forecast
    itself passes 35.4 misses most events; every method, baselines included, gets its own level.
    """
    m = np.isfinite(obs)
    p, o = pred[m], obs[m] > USG
    best = (USG, -1.0)
    for level in np.arange(10.0, 45.01, 0.5):
        w = p > level
        hits, misses, false = (w & o).sum(), (~w & o).sum(), (w & ~o).sum()
        csi = hits / max(hits + misses + false, 1)
        if csi > best[1]:
            best = (float(level), csi)
    return best[0]


def evaluate(runs: list[str], run_root: Path, data_dir: Path, out: Path) -> dict:
    data = SmokeData(data_dir)
    splits = data.meta["splits"]
    device = pick_device()
    history, horizon = 48, 72
    t_train = data.forecast_times(tuple(splits["train"]), history, horizon)
    t_val = data.forecast_times(tuple(splits["val"]), history, horizon, 6)
    t_test = data.forecast_times(tuple(splits["test"]), history, horizon)
    t_warn = data.forecast_times(tuple(splits["val"]), history, horizon)  # every hour, for warning levels
    pm = np.asarray(data.pm)
    obs_warn = pm[t_warn[:, None] + np.array(LEADS)]
    obs = pm[t_test[:, None] + np.array(LEADS)]                                   # (T0, L, N)
    x_t0 = np.asarray(data.x[t_test], dtype=np.float32)
    smoky = ((x_t0[..., SMOKE[0]] >= 0.6) | (x_t0[..., SMOKE[1]] >= 0.6))[:, None, :]
    subsets = {"all": np.ones_like(obs, bool), "smoke": np.broadcast_to(smoky, obs.shape),
               "high": np.nan_to_num(obs) > USG}

    forecasts, warn = {}, {}
    clim = climatology(data, np.arange(t_train[0], t_train[-1]))

    def climatology_at(t0):
        vt = (t0[:, None] + np.array(LEADS)).ravel()
        return clim[data.index.month[vt] - 1, data.index.hour[vt]].reshape(len(t0), len(LEADS), -1)

    def persistence_at(t0):
        x = np.asarray(data.x[t0, :, :2], dtype=np.float32)
        current = np.where(x[..., 1] > 0.5, x[..., 0] * data.sigma + data.mu, data.mu)
        return np.broadcast_to(current[:, None], (len(t0), len(LEADS), data.n_nodes))

    forecasts["Climatology"], warn["Climatology"] = climatology_at(t_test), climatology_at(t_warn)
    forecasts["Persistence"], warn["Persistence"] = persistence_at(t_test), persistence_at(t_warn)
    print("fitting ridge baseline", flush=True)
    warn["Ridge per site"], forecasts["Ridge per site"] = ridge_baseline(data, history, t_train[::4], t_val,
                                                                         [t_warn, t_test])
    np.save(run_root / "ridge_test_pred.npy", forecasts["Ridge per site"].astype(np.float16))

    models = {}
    batcher = Batcher(data, history, horizon, device)
    for name in runs:
        print(f"predicting {name}", flush=True)
        ckpt = torch.load(run_root / name / "model.pt", map_location="cpu", weights_only=False)
        model = build_model(ckpt["config"], data).to(device)
        model.load_state_dict(ckpt["state_dict"])
        models[name] = (model, ckpt["config"])
        norm = predict(model, batcher, t_test, leads=SAVED)
        pred = np.maximum(data.mu + data.sigma * norm, 0.0)
        np.save(run_root / name / "test_pred.npy", pred.astype(np.float16))
        forecasts[LABELS.get(name, name)] = pred[:, 1:]
        warn[LABELS.get(name, name)] = np.maximum(data.mu + data.sigma * predict(model, batcher, t_warn, leads=LEADS), 0.0)

    results = {"leads": list(LEADS), "n_points": {k: int((v & np.isfinite(obs)).sum()) for k, v in subsets.items()},
               "rmse": {}, "exceedance": {}, "warning_level": {}}
    for subset, m in subsets.items():
        results["rmse"][subset] = {name: {"by_lead": {l: rmse(f[:, i], obs[:, i], m[:, i]) for i, l in enumerate(LEADS)},
                                          "mean": float(np.mean([rmse(f[:, i], obs[:, i], m[:, i])
                                                                 for i in range(len(LEADS))]))}
                                   for name, f in forecasts.items()}
    for name, f in forecasts.items():
        levels = {l: tune_warning_level(warn[name][:, i], obs_warn[:, i]) for i, l in enumerate(LEADS)}
        results["warning_level"][name] = levels
        results["exceedance"][name] = {l: exceedance(f[:, i], obs[:, i], np.ones_like(obs[:, i], bool), levels[l])
                                       for i, l in enumerate(LEADS)}

    # virtual sensor: hide 30% of sites and estimate their current reading from the rest
    rng = np.random.default_rng(0)
    t_vs = t_test[::24]
    hidden = rng.random((len(t_vs), data.n_nodes)) < HIDDEN_FRAC
    truth = pm[t_vs]
    virtual = {"Inverse-distance interpolation": idw_virtual(data, t_vs, hidden),
               "Climatology": clim[data.index.month[t_vs] - 1, data.index.hour[t_vs]]}
    for name, (model, cfg) in models.items():
        if not cfg["model"].get("use_graph", True):
            continue
        out_ = []
        for i in range(0, len(t_vs), 64):
            mask = torch.as_tensor(hidden[i:i + 64], device=device)
            with torch.no_grad():
                model.eval()
                out_.append(model(batcher(t_vs[i:i + 64], node_mask=mask))[:, 0].float().cpu().numpy())
        virtual[LABELS.get(name, name)] = np.maximum(data.mu + data.sigma * np.concatenate(out_), 0)
    smoky_vs = (np.asarray(data.x[t_vs, :, SMOKE[0]], dtype=np.float32) >= 0.6)
    results["virtual_sensor"] = {name: {"all": rmse(v, truth, hidden), "smoke": rmse(v, truth, hidden & smoky_vs)}
                                 for name, v in virtual.items()}
    results["virtual_sensor_points"] = int((hidden & np.isfinite(truth)).sum())

    out.mkdir(parents=True, exist_ok=True)
    (out / "metrics.json").write_text(json.dumps(results, indent=1))
    (out / "metrics.md").write_text(markdown_tables(results))
    print(markdown_tables(results))
    return results


def markdown_tables(results: dict) -> str:
    leads = results["leads"]
    titles = {"all": "All test hours", "smoke": "Satellite smoke over or upwind of the site at issue time",
              "high": f"Observed PM2.5 above {USG} µg/m³"}
    lines = []
    for subset, table in results["rmse"].items():
        lines += [f"\n**{titles[subset]}** ({results['n_points'][subset]:,} forecasts), RMSE in µg/m³\n",
                  "| Model | " + " | ".join(f"{l} h" for l in leads) + " | mean |",
                  "|---|" + "---:|" * (len(leads) + 1)]
        best = {l: min(v["by_lead"][l] for v in table.values()) for l in leads}
        for name, v in table.items():
            cells = [f"**{v['by_lead'][l]:.2f}**" if np.isclose(v["by_lead"][l], best[l]) else f"{v['by_lead'][l]:.2f}"
                     for l in leads]
            lines.append(f"| {name} | " + " | ".join(cells) + f" | {v['mean']:.2f} |")
    lines += [f"\n**Early warning: hourly PM2.5 above {USG} µg/m³** (critical success index / hit rate / false-alarm ratio; "
              "each method warns above its own forecast level, tuned on 2022)\n",
              "| Model | " + " | ".join(f"{l} h" for l in leads) + " |", "|---|" + "---:|" * len(leads)]
    for name, by_lead in results["exceedance"].items():
        lines.append(f"| {name} | " + " | ".join(
            f"{v['csi']:.2f} / {v['hit_rate']:.2f} / {v['false_alarm_ratio']:.2f}" for v in by_lead.values()) + " |")
    lines += [f"\n**Virtual sensor**: {HIDDEN_FRAC:.0%} of sites hidden, their current PM2.5 estimated from the rest "
              f"({results['virtual_sensor_points']:,} estimates), RMSE in µg/m³\n",
              "| Method | all hours | smoke over the site |", "|---|---:|---:|"]
    for name, v in results["virtual_sensor"].items():
        lines.append(f"| {name} | {v['all']:.2f} | {v['smoke']:.2f} |")
    return "\n".join(lines) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--run-root", type=Path, default=Path("runs"))
    ap.add_argument("--data", type=Path, default=Path("data/processed"))
    ap.add_argument("--out", type=Path, default=Path("results"))
    args = ap.parse_args()
    evaluate(args.runs, args.run_root, args.data, args.out)


if __name__ == "__main__":
    main()
