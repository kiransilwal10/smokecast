"""README figures from the evaluation outputs: python scripts/make_figures.py"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from smokecast import viz
from smokecast.dataset import SmokeData
from smokecast.evaluate import LABELS, LEADS, SAVED
from smokecast.faults import EPISODE_HOURS, inject
from smokecast.geo import haversine_km
from smokecast.graph import knn_edges

LEAD = 24  # hours ahead shown in the case studies
# (place, first day, last day, lat, lon): the nearest site with data in the window is used
CASES = [
    ("New York City", "2023-06-05", "2023-06-10", 40.74, -73.82),
    ("Chicago", "2023-06-25", "2023-06-30", 41.88, -87.63),
    ("Detroit", "2023-06-25", "2023-06-30", 42.33, -83.05),
]
ANIMATION = ("2023-06-06 00:00", "2023-06-09 00:00", (-84.0, -68.0, 36.0, 46.5), (12, 24))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, default=Path("data/processed"))
    ap.add_argument("--raw", type=Path, default=Path("data/raw"))
    ap.add_argument("--runs", type=Path, default=Path("runs"))
    ap.add_argument("--results", type=Path, default=Path("results"))
    ap.add_argument("--figures", type=Path, default=Path("figures"))
    args = ap.parse_args()
    FIG = args.figures
    FIG.mkdir(exist_ok=True)
    data = SmokeData(args.data)
    results = json.loads((args.results / "metrics.json").read_text())
    t_test = data.forecast_times(tuple(data.meta["splits"]["test"]), 48, 72)
    pm = np.asarray(data.pm)
    x0 = lambda rows: np.asarray(data.x[rows, :, :2], dtype=np.float32)
    preds = {LABELS[r]: np.load(args.runs / r / "test_pred.npy", mmap_mode="r")
             for r in LABELS if (args.runs / r / "test_pred.npy").exists()}
    ridge = np.load(args.runs / "ridge_test_pred.npy", mmap_mode="r")

    def at_lead(name, valid_rows, lead):
        rows = valid_rows - lead - t_test[0]
        if name == "Persistence":
            cur = x0(valid_rows - lead)
            return np.where(cur[..., 1] > 0.5, cur[..., 0] * data.sigma + data.mu, data.mu)
        if name == "Ridge per site":
            return np.asarray(ridge[rows, LEADS.index(lead)], np.float32)
        return np.asarray(preds[name][rows, SAVED.index(lead)], np.float32)

    # 1. network
    src, dst = knn_edges(data.lat, data.lon, 8)
    viz.network_map(data.lat, data.lon, src, dst, FIG / "network.png")

    # 2. error against lead
    order = ["Climatology", "Persistence", "Ridge per site"] + list(preds)
    viz.skill_curves(results, FIG / "skill_vs_lead.png", [m for m in order if m in results["rmse"]["all"]])

    # 3. case studies at a fixed lead
    cases, peaks = [], []
    for place, first, last, plat, plon in CASES:
        window = np.arange(data.index.get_loc(pd.Timestamp(first)), data.index.get_loc(pd.Timestamp(last)) + 24)
        has = np.isfinite(pm[window]).mean(0) > 0.8
        n = int(np.argmin(np.where(has, haversine_km(plat, plon, data.lat, data.lon), np.inf)))
        shown = ["Persistence", "Ridge per site"] + [m for m in ("GRU per site (no graph)", "SmokeCast") if m in preds]
        forecasts = {m: at_lead(m, window, LEAD)[:, n] for m in shown}
        obs = pm[window, n]
        site = data.sites[n]
        cases.append({"title": f"{place}: {site.get('name') or site['site']} ({first[:4]})",
                      "times": data.index[window], "observed": obs, "forecasts": forecasts})
        i_peak = int(np.nanargmax(obs))
        peaks.append({"place": place, "site": site.get("name") or site["site"], "peak_time": str(data.index[window[i_peak]]),
                      "observed_peak": float(obs[i_peak]),
                      **{f"{m} {lead} h": float(at_lead(m, window[i_peak:i_peak + 1], lead)[0, n])
                         for m in ("Persistence", "Ridge per site", "SmokeCast") for lead in (6, 12, 24)}})
    viz.case_studies(cases, FIG / "case_studies.png", LEAD)
    (args.results / "peaks.json").write_text(json.dumps(peaks, indent=1))

    # 4. animation of the June 2023 smoke across the northeast
    t_start, t_end, extent, leads = ANIMATION
    frames = np.arange(data.index.get_loc(pd.Timestamp(t_start)), data.index.get_loc(pd.Timestamp(t_end)) + 1)
    hms = json.loads((args.raw / "hms.json").read_text())
    grid = np.load(args.raw / "hms_grid.npy", mmap_mode="r")
    day = (data.index[frames].normalize() - pd.Timestamp(hms["first_day"])).days.to_numpy()
    forecasts = {f"SmokeCast, {lead} h ahead": at_lead("SmokeCast", frames, lead) for lead in leads}
    viz.smoke_animation(data.index[frames], data.lat, data.lon, pm[frames], forecasts, np.asarray(grid[day]),
                        hms, "Canadian wildfire smoke, June 2023", FIG / "smoke_june2023.gif", extent)

    # 5. one injected sensor fault, and what each detector made of it
    rows = data.forecast_times(tuple(data.meta["splits"]["test"]), 48, 0)
    faulty, episodes = inject(pm, rows, np.random.default_rng(0))
    score_v = np.load(args.runs / "smokecast" / "score_virtual.npy", mmap_mode="r")[1]
    score_o = np.load(args.runs / "smokecast" / "score_own.npy", mmap_mode="r")[1]
    virtual = np.load(args.runs / "smokecast" / "virtual_faulty.npy", mmap_mode="r")
    # a typical drift, not the best one: the episode with the median peak score relative to the alarm level
    drifts = [e for e in episodes if e["kind"] == "drift"]
    peak = [np.nanmax(np.asarray(score_v[e["start"] - rows[0]:e["start"] - rows[0] + EPISODE_HOURS, e["site"]],
                                 np.float32)) for e in drifts]
    ep = drifts[int(np.argsort(peak)[len(peak) // 2])]
    n, i0 = ep["site"], ep["start"] - rows[0]
    span = np.arange(i0 - 72, i0 + EPISODE_HOURS + 48)
    site = data.sites[n]
    viz.fault_example(data.index[rows[span]], pm[rows[span], n], faulty[rows[span], n],
                      np.asarray(virtual[span, n], np.float32),
                      {"Virtual sensor (SmokeCast)": np.asarray(score_v[span, n], np.float32),
                       "Own history": np.asarray(score_o[span, n], np.float32)},
                      {"Virtual sensor (SmokeCast)": 1.0, "Own history": 1.0}, (72, 72 + EPISODE_HOURS),
                      f"Injected drift at {site.get('name') or site['site']}, {site.get('city', '')} {site.get('state', '')}",
                      FIG / "fault_example.png")
    print(json.dumps(peaks, indent=1))


if __name__ == "__main__":
    main()
