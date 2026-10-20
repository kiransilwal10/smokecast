"""Model tensors from the raw downloads, and on-device batching.

Per site and hour the inputs are:

    pm              PM2.5 anomaly, (pm - site mean) / site std (train years)
    pm_valid        1 where the monitor reported (gaps up to 2 h are forward-filled)
    u, v            wind at the nearest airport, m/s / 10
    temp            (T - 15 C) / 10
    dewpoint_dep    (T - Td) / 10; humid air near the ground traps smoke, dry air mixes it up
    slp             (p - 1015 hPa) / 10
    precip          asinh(1 h rain in mm); rain washes particles out
    met_valid
    smoke_here      NOAA HMS smoke density over the site from yesterday's analysis, 0..1
    smoke_upwind    HMS density 100-300 km upwind of the site along the current wind, 0..1

Yesterday's smoke analysis is used so every input is available at forecast time.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from .geo import haversine_km, offset_point, wind_components

CHANNELS = ("pm", "pm_valid", "u", "v", "temp", "dewpoint_dep", "slp", "precip", "met_valid",
            "smoke_here", "smoke_upwind")
PM, PM_VALID, U, V = 0, 1, 2, 3
SMOKE = (9, 10)
UPWIND_KM = (100.0, 200.0, 300.0)
FILL_HOURS = 2
IDW_NEIGHBOURS = 12  # about 8 stay visible when 30% of sites are hidden
USG = 35.4  # µg/m³: hourly PM2.5 above this is "unhealthy for sensitive groups" on the AQI scale


def ffill(a: np.ndarray, limit: int) -> np.ndarray:
    return pd.DataFrame(a.reshape(len(a), -1)).ffill(limit=limit).to_numpy().reshape(a.shape)


def smoke_features(grid: np.ndarray, hms: dict, index: pd.DatetimeIndex, lat, lon, u, v):
    """(T, N) smoke over each site and upwind of it, from the previous day's HMS raster."""
    day = (index.normalize() - pd.Timestamp(hms["first_day"])).days.to_numpy() - 1
    res, lon0, lat0, nx, ny = hms["res"], hms["lon0"], hms["lat0"], hms["nx"], hms["ny"]

    def sample(la, lo, rows):
        i = np.floor((lo - lon0) / res).astype(int)
        j = np.floor((la - lat0) / res).astype(int)
        inside = (i >= 0) & (i < nx) & (j >= 0) & (j < ny)
        d = np.broadcast_to(day[rows, None], la.shape)
        return np.where(inside, grid[d, j.clip(0, ny - 1), i.clip(0, nx - 1)], 0) / 3.0

    here = np.zeros((len(index), len(lat)), np.float32)
    up = np.zeros_like(here)
    for a in range(0, len(index), 8760):
        rows = np.arange(a, min(a + 8760, len(index)))
        la = np.broadcast_to(lat, (len(rows), len(lat)))
        lo = np.broadcast_to(lon, (len(rows), len(lat)))
        here[rows] = sample(la, lo, rows)
        uu, vv = u[rows], v[rows]
        speed = np.hypot(uu, vv)
        calm = ~(speed >= 1.0)
        ux, uy = -uu / np.maximum(speed, 1e-6), -vv / np.maximum(speed, 1e-6)  # unit vector pointing upwind
        total = np.zeros_like(la, dtype=np.float32)
        for dist in UPWIND_KM:
            total += sample(*offset_point(la, lo, ux * dist, uy * dist), rows)
        ring = np.zeros_like(total)
        for ang in np.deg2rad(np.arange(0, 360, 45)):  # calm or no wind data: look all around
            ring += sample(*offset_point(la, lo, 200 * np.sin(ang), 200 * np.cos(ang)), rows)
        up[rows] = np.where(calm, ring / 8, total / len(UPWIND_KM))
    return here, up


def build(raw: Path, out: Path, splits: dict) -> None:
    out.mkdir(parents=True, exist_ok=True)
    info = json.loads((raw / "sites.json").read_text())
    sites = info["sites"]
    index = pd.date_range(f"{info['start']}-01-01", info.get("last_hour", f"{info['end']}-12-31 23:00"), freq="h")
    lat = np.array([s["lat"] for s in sites])
    lon = np.array([s["lon"] for s in sites])

    pm = np.load(raw / "pm25.npy").astype(np.float32)
    pm[(pm < -10) | (pm > 1000)] = np.nan
    pm = np.where(np.isnan(pm), np.nan, np.maximum(pm, 0.0)).astype(np.float32)
    train = (index.year >= splits["train"][0]) & (index.year <= splits["train"][1])
    mu, sigma = np.nanmean(pm[train], 0), np.nanstd(pm[train], 0)
    p99 = np.nanpercentile(pm[train], 99, axis=0)
    filled = ffill(pm, FILL_HOURS)
    pm_valid = np.isfinite(filled)

    met = np.load(raw / "met.npy")
    temp, dew, slp, wdir, wspd, rain = (met[..., i] for i in range(6))
    temp = np.where((temp > -50) & (temp < 55), temp, np.nan)
    dew = np.where((dew > -60) & (dew < 40) & (dew <= temp + 0.5), dew, np.nan)
    slp = np.where((slp > 900) & (slp < 1080), slp, np.nan)
    ok_wind = (wspd >= 0) & (wspd < 60) & (wdir >= 0) & (wdir <= 360)
    u, v = wind_components(np.where(ok_wind, wspd, np.nan), np.where(ok_wind, wdir, np.nan))
    rain = np.where((rain >= 0) & (rain < 200), rain, np.nan)
    temp, dew, slp, u, v, rain = (ffill(a, 3) for a in (temp, dew, slp, u, v, rain))
    met_valid = np.isfinite(temp) & np.isfinite(u)

    grid = np.load(raw / "hms_grid.npy", mmap_mode="r")
    hms = json.loads((raw / "hms.json").read_text())
    smoke_here, smoke_up = smoke_features(np.asarray(grid), hms, index, lat, lon,
                                          np.nan_to_num(u), np.nan_to_num(v))

    z = lambda a: np.nan_to_num(a).astype(np.float32)
    x = np.stack([
        z((filled - mu) / sigma), pm_valid.astype(np.float32),
        z(u / 10), z(v / 10), z((temp - 15) / 10), z((temp - dew) / 10), z((slp - 1015) / 10),
        z(np.arcsinh(rain)), met_valid.astype(np.float32), smoke_here, smoke_up,
    ], -1).astype(np.float16)

    np.save(out / "x.npy", x)
    np.save(out / "pm.npy", pm)
    meta = {"start": info["start"], "end": info["end"], "last_hour": str(index[-1]), "splits": splits, "channels": list(CHANNELS),
            "mu": mu.tolist(), "sigma": sigma.tolist(), "p99": p99.tolist(), "sites": sites,
            "hms_missing": hms["missing"]}
    (out / "meta.json").write_text(json.dumps(meta))
    print(f"{len(sites)} sites, {len(index)} hours, {x.nbytes / 1e6:.0f} MB of inputs; "
          f"pm coverage {np.isfinite(pm).mean():.1%}, weather coverage {met_valid.mean():.1%}")


def time_features(index: pd.DatetimeIndex) -> np.ndarray:
    """(T, 6): hour of day, day of year and day of week as sin/cos pairs (local solar effects are learned)."""
    h = 2 * np.pi * index.hour.to_numpy() / 24
    d = 2 * np.pi * index.dayofyear.to_numpy() / 365.25
    w = 2 * np.pi * index.dayofweek.to_numpy() / 7
    return np.stack([np.sin(h), np.cos(h), np.sin(d), np.cos(d), np.sin(w), np.cos(w)], 1).astype(np.float32)


class SmokeData:
    def __init__(self, root: Path):
        self.root = Path(root)
        self.meta = json.loads((self.root / "meta.json").read_text())
        self.x = np.load(self.root / "x.npy", mmap_mode="r")
        self.pm = np.load(self.root / "pm.npy", mmap_mode="r")
        self.index = pd.date_range(f"{self.meta['start']}-01-01", self.meta["last_hour"], freq="h")
        self.sites = self.meta["sites"]
        self.lat = np.array([s["lat"] for s in self.sites])
        self.lon = np.array([s["lon"] for s in self.sites])
        self.mu = np.array(self.meta["mu"], dtype=np.float32)
        self.sigma = np.array(self.meta["sigma"], dtype=np.float32)
        self.n_nodes = len(self.sites)

    def forecast_times(self, years: tuple[int, int], history: int, horizon: int, stride: int = 1) -> np.ndarray:
        """Forecast issue times t0 whose input and target windows stay inside ``years``."""
        inside = np.nonzero((self.index.year >= years[0]) & (self.index.year <= years[1]))[0]
        return np.arange(inside[0] + history - 1, inside[-1] - horizon + 1, stride)


class Batcher:
    """Gathers (input window, targets) for a set of issue times, entirely on the device.

    ``mask_frac`` hides that fraction of sites from the model (their PM2.5 history is zeroed and
    flagged invalid). The model must then reconstruct their current reading from the neighbours,
    which is the virtual sensor used for fault detection, and forecast them without their own data.

    Two starting points are returned for the model's residual: ``base`` is the site's own current
    reading (0, the site mean, when it is hidden or offline), and ``base_spatial`` replaces that 0
    with an inverse-distance-weighted estimate from the visible neighbours. Graph models start from
    ``base_spatial``, so their virtual sensor is a learned correction to plain interpolation.
    """

    def __init__(self, data: SmokeData, history: int, horizon: int, device, mask_frac: float = 0.0):
        self.history, self.horizon, self.device, self.mask_frac = history, horizon, device, mask_frac
        self.x = torch.as_tensor(np.array(data.x), device=device)                       # (T, N, C) half
        pm = np.array(data.pm)
        self.y = torch.as_tensor(np.nan_to_num((pm - data.mu) / data.sigma), device=device)
        self.y_valid = torch.as_tensor(np.isfinite(pm), device=device)
        self.time = torch.as_tensor(time_features(data.index), device=device)
        self.mu = torch.as_tensor(data.mu, device=device)
        self.sigma = torch.as_tensor(data.sigma, device=device)
        d = haversine_km(data.lat[:, None], data.lon[:, None], data.lat[None], data.lon[None])
        np.fill_diagonal(d, np.inf)
        rows = np.arange(len(d))[:, None]
        near = np.argsort(d, 1)[:, :IDW_NEIGHBOURS]
        w = np.zeros_like(d)
        w[rows, near] = 1.0 / np.maximum(d[rows, near], 1.0) ** 2
        self.idw = torch.as_tensor(w.T.astype(np.float32), device=device)  # pm (B, N) @ idw -> estimates

    def __call__(self, t0: np.ndarray, node_mask: torch.Tensor | None = None, train: bool = False) -> dict:
        t0 = torch.as_tensor(t0, device=self.device)
        past = t0[:, None] + torch.arange(-self.history + 1, 1, device=self.device)
        ahead = t0[:, None] + torch.arange(0, self.horizon + 1, device=self.device)
        x = self.x[past].float()                                                       # (B, T, N, C)
        B, _, N, _ = x.shape
        if node_mask is None:
            node_mask = (torch.rand(B, N, device=self.device) < self.mask_frac) if train else \
                torch.zeros(B, N, dtype=torch.bool, device=self.device)
        keep = (~node_mask).float()[:, None, :]
        x[..., PM] *= keep
        x[..., PM_VALID] *= keep
        now = x[:, -1, :, PM_VALID]
        base = x[:, -1, :, PM] * now
        weight = now @ self.idw
        nearby = ((x[:, -1, :, PM] * self.sigma + self.mu) * now) @ self.idw / weight.clamp_min(1e-12)
        nearby = torch.where(weight > 0, nearby, self.mu)
        base_spatial = torch.where(now > 0.5, base, (nearby - self.mu) / self.sigma)
        return {
            "x": x, "base": base, "base_spatial": base_spatial, "node_mask": node_mask, "time": self.time[t0],
            "y": self.y[ahead],                     # (B, H + 1, N), lead 0 is the current hour
            "y_valid": self.y_valid[ahead], "t0": t0,
        }
