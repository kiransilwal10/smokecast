"""HTTP API and dashboard.

    uvicorn smokecast.serve.app:app            # or: docker compose up

The service replays the 2023-2025 test period from a bundle written by
scripts/export_serving.py: pick an issue time and it returns what SmokeCast
would have forecast then, which sites it would have warned, and which sensors
it would have flagged. Swapping the bundle for a live feed (AirNow, OpenAQ)
only changes where the inputs come from.
"""

from __future__ import annotations

import json
import os
from functools import lru_cache
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import torch
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse

from ..dataset import SMOKE, USG, Batcher
from ..model import SmokeCast

BUNDLE = Path(os.environ.get("SMOKECAST_BUNDLE", "serving"))
STATIC = Path(__file__).parent / "static"
HEALTH_WINDOW = 24

app = FastAPI(title="SmokeCast", version="0.1.0")


@lru_cache(maxsize=1)
def bundle():
    meta = json.loads((BUNDLE / "meta.json").read_text())
    arrays = np.load(BUNDLE / "inputs.npz")
    ckpt = torch.load(BUNDLE / "model.pt", map_location="cpu", weights_only=False)
    cfg = ckpt["config"]
    lat = np.array([s["lat"] for s in meta["sites"]])
    lon = np.array([s["lon"] for s in meta["sites"]])
    model = SmokeCast(lat, lon, in_channels=arrays["x"].shape[-1], horizon=cfg["horizon"],
                      history=cfg["history"], **cfg["model"])
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    index = pd.date_range(meta["first_hour"], periods=arrays["x"].shape[0], freq="h")
    view = SimpleNamespace(x=arrays["x"], pm=arrays["pm"].astype(np.float32), mu=arrays["mu"],
                           sigma=arrays["sigma"], index=index, lat=lat, lon=lon)
    return SimpleNamespace(meta=meta, model=model, view=view, index=index, colour=arrays["colour"],
                           spread=arrays["spread"], threshold=arrays["threshold"], history=cfg["history"],
                           horizon=cfg["horizon"], batcher=Batcher(view, cfg["history"], 0, "cpu"))


def issue_row(issued: str) -> int:
    b = bundle()
    t = pd.Timestamp(issued).tz_localize(None).floor("h")
    row = b.index.get_indexer([t])[0]
    if row < b.history - 1:
        raise HTTPException(400, f"issue time must be between {b.index[b.history - 1]} and {b.index[-1]} UTC")
    return int(row)


@lru_cache(maxsize=256)
def forecast(row: int) -> np.ndarray:
    """(H + 1, N) µg/m³ for one issue time."""
    b = bundle()
    with torch.no_grad():
        out = b.model(b.batcher(np.array([row])))[0].numpy()
    return np.maximum(b.view.mu + b.view.sigma * out, 0.0)


def sensor_scores(row: int, simulate: int | None = None) -> np.ndarray:
    """Per-site alarm score over the 24 h ending at ``row``; above 1 means the sensor is flagged.

    The deviation is (reading - virtual reading) / typical spread. Its |median| catches offsets
    and drift, the median hour-to-hour jump catches noise; each has its own threshold, one for
    clear air and one for hours under satellite smoke.
    """
    b = bundle()
    rows = np.arange(row - HEALTH_WINDOW + 1, row + 1)
    view = b.view
    if simulate is not None:  # pretend one sensor started reading 15 µg/m³ high two days ago
        x, pm = view.x.copy(), view.pm.copy()
        bad = slice(row - 48, row + 1)
        pm[bad, simulate] += 15.0
        x[bad, simulate, 0] = (np.nan_to_num(pm[bad, simulate]) - view.mu[simulate]) / view.sigma[simulate]
        view = SimpleNamespace(x=x, pm=pm, mu=view.mu, sigma=view.sigma, index=view.index, lat=view.lat, lon=view.lon)
    batcher = Batcher(view, b.history, 0, "cpu") if simulate is not None else b.batcher
    virtual = np.zeros((len(rows), len(b.colour)), np.float32)
    with torch.no_grad():
        for c in range(b.colour.max() + 1):
            group = torch.as_tensor(b.colour == c).expand(len(rows), -1)
            out = b.model(batcher(rows, node_mask=group))[:, 0].numpy()
            virtual[:, b.colour == c] = out[:, b.colour == c]
    virtual = np.maximum(view.mu + view.sigma * virtual, 0.0)
    resid = (view.pm[rows] - virtual) / np.maximum(b.spread, 1.0)
    bias = np.abs(np.nanmedian(resid, 0))
    jitter = np.nanmedian(np.abs(np.diff(resid, axis=0)), 0)
    smoky = np.asarray(view.x[row, :, SMOKE[0]], dtype=np.float32) >= 0.6  # yesterday's HMS, known now
    thr = np.where(smoky[None], b.threshold[1], b.threshold[0])          # (statistic, site)
    return np.nan_to_num(np.fmax(bias / thr[0], jitter / thr[1]))


@app.get("/api/health")
def health():
    b = bundle()
    return {"status": "ok", "sites": len(b.meta["sites"]),
            "replay_from": str(b.index[b.history - 1]), "replay_to": str(b.index[-1])}


@app.get("/api/sites")
def sites():
    return bundle().meta["sites"]


@app.get("/api/forecast")
def get_forecast(issued: str = Query(..., description="issue time, UTC, e.g. 2023-06-07T00:00")):
    b, row = bundle(), issue_row(issued)
    fc = forecast(row)
    now = b.view.pm[row]
    return {"issued": str(b.index[row]), "leads_h": list(range(fc.shape[0])),
            "sites": [{"site": s["site"], "observed_now": None if np.isnan(now[i]) else round(float(now[i]), 1),
                       "forecast": [round(float(v), 1) for v in fc[:, i]]} for i, s in enumerate(b.meta["sites"])]}


@app.get("/api/alerts")
def alerts(issued: str = Query(...), within_h: int = Query(24, ge=1, le=72)):
    """Sites forecast to exceed the AQI 'unhealthy for sensitive groups' level within ``within_h`` hours."""
    b, row = bundle(), issue_row(issued)
    fc = forecast(row)[1:within_h + 1]
    out = []
    for i, s in enumerate(b.meta["sites"]):
        lead = int(np.argmax(fc[:, i]))
        if fc[lead, i] > USG:
            out.append({**s, "peak_ugm3": round(float(fc[lead, i]), 1), "peak_in_h": lead + 1,
                        "first_exceedance_in_h": int(np.argmax(fc[:, i] > USG)) + 1})
    return {"issued": str(b.index[row]), "threshold_ugm3": USG,
            "alerts": sorted(out, key=lambda a: a["first_exceedance_in_h"])}


@app.get("/api/sensors")
def sensors(issued: str = Query(...), simulate_fault: str | None = Query(None, description="site id to break")):
    b, row = bundle(), issue_row(issued)
    ids = [s["site"] for s in b.meta["sites"]]
    sim = ids.index(simulate_fault) if simulate_fault in ids else None
    score = sensor_scores(row, sim)
    return {"issued": str(b.index[row]), "alarm_level": 1.0,
            "sensors": [{"site": ids[i], "score": round(float(score[i]), 2), "flagged": bool(score[i] > 1.0)}
                        for i in range(len(ids))]}


@app.get("/")
def dashboard():
    return FileResponse(STATIC / "index.html")
