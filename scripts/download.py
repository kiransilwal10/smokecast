"""Download PM2.5, weather and smoke data, and choose the monitoring sites.

    python scripts/download.py --start 2017 --end 2025

Writes to data/raw/: sites.json, pm25.npy (T, N), met.npy (T, N, 6), hms_grid.npy (days, ny, nx),
hms.json. The zip and gz downloads are cached, so an interrupted run picks up where it stopped.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib.path import Path as PolygonPath

from smokecast import sources
from smokecast.geo import haversine_km

BBOX = (-98.0, -66.5, 24.0, 49.5)  # eastern United States
HMS_GRID = {"lon0": -112.0, "lat0": 18.0, "res": 0.25, "nx": 240, "ny": 168}  # reaches into Canada


def choose_sites(pm: dict, meta: dict, n_hours: int, min_coverage: float, target: int) -> list[str]:
    """Best-covered sites, thinned so no two are closer than the smallest spacing that gives <= target."""
    cover = {s: np.isfinite(v).sum() / n_hours for s, v in pm.items()}
    good = sorted((s for s in pm if cover[s] >= min_coverage), key=lambda s: -cover[s])
    for spacing in (15, 20, 25, 30, 35, 40, 50, 60, 75):
        chosen = []
        for s in good:
            if all(haversine_km(meta[s]["lat"], meta[s]["lon"], meta[c]["lat"], meta[c]["lon"]) >= spacing
                   for c in chosen):
                chosen.append(s)
        if len(chosen) <= target:
            print(f"{len(good)} sites with >= {min_coverage:.0%} coverage; {len(chosen)} kept at {spacing} km spacing")
            return chosen
    return chosen


def rasterize(polygons, grid: dict) -> np.ndarray:
    out = np.zeros((grid["ny"], grid["nx"]), dtype=np.uint8)
    res, lon0, lat0 = grid["res"], grid["lon0"], grid["lat0"]
    lon_c = lon0 + res * (np.arange(grid["nx"]) + 0.5)
    lat_c = lat0 + res * (np.arange(grid["ny"]) + 0.5)
    for density, ring in polygons:
        (x0, y0), (x1, y1) = ring.min(0), ring.max(0)
        i0, i1 = max(int((x0 - lon0) / res), 0), min(int((x1 - lon0) / res) + 1, grid["nx"])
        j0, j1 = max(int((y0 - lat0) / res), 0), min(int((y1 - lat0) / res) + 1, grid["ny"])
        if i1 <= i0 or j1 <= j0:
            continue
        gx, gy = np.meshgrid(lon_c[i0:i1], lat_c[j0:j1])
        inside = PolygonPath(ring).contains_points(np.c_[gx.ravel(), gy.ravel()]).reshape(gx.shape)
        sub = out[j0:j1, i0:i1]
        np.maximum(sub, np.where(inside, density, 0).astype(np.uint8), out=sub)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", type=int, default=2017)
    ap.add_argument("--end", type=int, default=2025)
    ap.add_argument("--last", default="2025-08-26 23:00",
                    help="last hour (UTC); NOAA stopped updating ISD-Lite at the end of August 2025")
    ap.add_argument("--raw", type=Path, default=Path("data/raw"))
    ap.add_argument("--sites", type=int, default=160, help="target number of monitoring sites")
    ap.add_argument("--min-coverage", type=float, default=0.75)
    args = ap.parse_args()
    raw = args.raw
    index = pd.date_range(f"{args.start}-01-01", args.last, freq="h")

    # 1. PM2.5 monitors
    years = range(args.start, args.end + 1)
    sources.fetch_many([(sources.AQS_URL.format(year=y), raw / "aqs" / f"hourly_88101_{y}.zip") for y in years]
                       + [(sources.AQS_SITES_URL, raw / "aqs" / "aqs_sites.zip")], workers=3)
    pm, meta = {}, {}
    for y in years:
        df = sources.read_aqs(raw / "aqs" / f"hourly_88101_{y}.zip", BBOX)
        for site, g in df.groupby("site"):
            if site not in pm:
                pm[site] = np.full(len(index), np.nan, dtype=np.float32)
                meta[site] = {"site": site, "lat": float(g.lat.iloc[0]), "lon": float(g.lon.iloc[0])}
            rows = index.get_indexer(g.time)
            ok = rows >= 0
            pm[site][rows[ok]] = g.pm.to_numpy()[ok]
        print(f"AQS {y}: {df.site.nunique()} sites", flush=True)
    names = sources.aqs_site_names(raw / "aqs" / "aqs_sites.zip")
    chosen = choose_sites(pm, meta, len(index), args.min_coverage, args.sites)

    # 2. nearest airport weather station with a full record
    sources.fetch(sources.ISD_HISTORY_URL, raw / "isd" / "isd-history.csv")
    stations = sources.isd_stations(raw / "isd" / "isd-history.csv")
    stations = stations[(stations.begin <= pd.Timestamp(f"{args.start}-01-01"))
                        & (stations.end >= index[-1] - pd.Timedelta(days=7))
                        & ~stations.station.str.endswith("99999")]
    candidates = {}
    for s in chosen:
        d = haversine_km(meta[s]["lat"], meta[s]["lon"], stations.lat.to_numpy(), stations.lon.to_numpy())
        order = np.argsort(d)[:4]
        candidates[s] = [(stations.station.iloc[i], stations.name.iloc[i], float(d[i])) for i in order if d[i] < 75]
    wanted = sorted({c[0] for cs in candidates.values() for c in cs})
    sources.fetch_many([(sources.ISD_LITE_URL.format(year=y, station=st), raw / "isd" / str(y) / f"{st}-{y}.gz")
                        for st in wanted for y in years], workers=8)
    cache = {}

    def station_series(st):
        if st not in cache:
            merged = None
            for y in years:
                if (p := raw / "isd" / str(y) / f"{st}-{y}.gz").exists():
                    part = sources.read_isd_lite(p, index)
                    merged = part if merged is None else np.where(np.isnan(merged), part, merged)
            cache[st] = merged
        return cache[st]

    kept, met = [], []
    for s in chosen:
        for st, name, dist in candidates[s]:
            series = station_series(st)
            if series is not None and np.isfinite(series[:, 4]).mean() > 0.7:
                kept.append({**meta[s], **names.get(s, {}), "met_station": st, "met_name": name,
                             "met_km": round(dist, 1), "coverage": float(np.isfinite(pm[s]).mean())})
                met.append(series)
                break
    print(f"{len(kept)} of {len(chosen)} sites matched to a weather station within 75 km")
    raw.mkdir(parents=True, exist_ok=True)
    np.save(raw / "pm25.npy", np.stack([pm[k["site"]] for k in kept], 1))
    np.save(raw / "met.npy", np.stack(met, 1))
    (raw / "sites.json").write_text(json.dumps({"start": args.start, "end": args.end, "last_hour": str(index[-1]),
                                                "sites": kept}, indent=1))

    # 3. satellite smoke analysis, one raster per day (the day before start covers the one-day lag)
    days = pd.date_range(f"{args.start - 1}-12-31", index[-1].normalize(), freq="D")
    paths = [raw / "hms" / f"{d:%Y}" / f"hms_smoke{d:%Y%m%d}.zip" for d in days]
    found = sources.fetch_many([(sources.HMS_URL.format(d=d), p) for d, p in zip(days, paths)], workers=6)
    grids = np.zeros((len(days), HMS_GRID["ny"], HMS_GRID["nx"]), dtype=np.uint8)
    for i, (p, ok) in enumerate(zip(paths, found)):
        if ok:
            try:
                grids[i] = rasterize(sources.read_hms(p), HMS_GRID)
            except Exception as err:  # a handful of daily files are malformed
                print(f"skipping {p.name}: {err}")
                found[i] = False
    np.save(raw / "hms_grid.npy", grids)
    (raw / "hms.json").write_text(json.dumps({**HMS_GRID, "first_day": str(days[0].date()),
                                              "missing": [str(d.date()) for d, ok in zip(days, found) if not ok]}))
    print(f"HMS: {sum(found)} of {len(days)} days")


if __name__ == "__main__":
    main()
