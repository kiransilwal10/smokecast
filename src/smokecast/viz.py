"""Figures for the README: network map, skill curves, smoke case studies, animation, fault detection."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.colors import BoundaryNorm, ListedColormap

try:
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature
except ImportError:  # maps fall back to plain lat/lon axes
    ccrs = None

EXTENT = (-98.5, -66.0, 24.0, 49.5)
INK, MUTED, GRID = "#1f2328", "#6e7781", "#d0d7de"
COLORS = {
    "Climatology": "#8c959f", "Persistence": "#bf8700", "Ridge per site": "#9a6700",
    "GRU per site (no graph)": "#0969da", "GNN, static edges": "#8250df",
    "SmokeCast, no satellite smoke": "#54aeff", "SmokeCast": "#cf222e",
}
# EPA AQI categories for PM2.5 (2024 breakpoints), µg/m³
AQI_BOUNDS = [0, 9.0, 35.4, 55.4, 125.4, 225.4, 1000]
AQI_COLORS = ["#00e400", "#ffff00", "#ff7e00", "#ff0000", "#8f3f97", "#7e0023"]
AQI_NAMES = ["Good", "Moderate", "Unhealthy for\nsensitive groups", "Unhealthy", "Very unhealthy", "Hazardous"]
AQI_CMAP, AQI_NORM = ListedColormap(AQI_COLORS), BoundaryNorm(AQI_BOUNDS, len(AQI_COLORS))

plt.rcParams.update({
    "font.size": 10, "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": MUTED,
    "ytick.color": MUTED, "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "figure.dpi": 130,
    "savefig.bbox": "tight", "legend.frameon": False,
})


def map_axes(fig, rect=111, extent=EXTENT):
    if ccrs is None:
        ax = fig.add_subplot(rect)
        ax.set_xlim(extent[:2]), ax.set_ylim(extent[2:])
        ax.set_aspect(1 / np.cos(np.deg2rad(38)))
        return ax, {}
    ax = fig.add_subplot(rect, projection=ccrs.PlateCarree())
    ax.set_extent(extent, crs=ccrs.PlateCarree())
    ax.add_feature(cfeature.LAND.with_scale("50m"), facecolor="#f3f4f6", edgecolor="none")
    ax.add_feature(cfeature.OCEAN.with_scale("50m"), facecolor="#ffffff")
    ax.add_feature(cfeature.LAKES.with_scale("50m"), facecolor="#ffffff", edgecolor="#8c959f", linewidth=0.3)
    ax.coastlines("50m", linewidth=0.5, color="#8c959f")
    ax.add_feature(cfeature.STATES.with_scale("50m"), linewidth=0.3, edgecolor="#d0d7de")
    ax.add_feature(cfeature.BORDERS.with_scale("50m"), linewidth=0.5, edgecolor="#8c959f")
    ax.grid(False)
    return ax, {"transform": ccrs.PlateCarree()}


def network_map(lat, lon, src, dst, path: Path, labels=None):
    fig = plt.figure(figsize=(7.5, 6))
    ax, tf = map_axes(fig)
    for s, d in zip(src, dst):
        if s < d:
            ax.plot([lon[s], lon[d]], [lat[s], lat[d]], color="#8250df", lw=0.5, alpha=0.45, **tf)
    ax.scatter(lon, lat, s=12, color=INK, zorder=3, **tf)
    for name, (la, lo) in (labels or {}).items():
        ax.text(lo + 0.3, la + 0.25, name, fontsize=8, color=INK, zorder=4, **tf)
    ax.set_title(f"{len(lat)} EPA PM2.5 monitors, {len(src) // 2} graph edges", loc="left", color=INK)
    fig.savefig(path)
    plt.close(fig)


def skill_curves(results: dict, path: Path, models: list[str]):
    leads = results["leads"]
    titles = {"all": "All test hours (2023–2025)", "smoke": "Satellite smoke over or upwind",
              "high": "PM2.5 above 35.4 µg/m³"}
    fig, axes = plt.subplots(1, 3, figsize=(12.5, 3.6), sharex=True)
    for ax, subset in zip(axes, titles):
        table = results["rmse"][subset]
        for name in models:
            by_lead = table[name]["by_lead"]
            ys = [by_lead[str(l)] if str(l) in by_lead else by_lead[l] for l in leads]
            ax.plot(leads, ys, marker="o", ms=3.5, lw=1.8 if name == "SmokeCast" else 1.2,
                    color=COLORS.get(name, INK), label=name)
        ax.set_title(titles[subset], loc="left", fontsize=10, color=INK)
        ax.set_xlabel("forecast lead (h)")
        ax.set_xscale("log")
        ax.set_xticks(leads)
        ax.set_xticklabels([str(l) for l in leads])
        ax.minorticks_off()
    axes[0].set_ylabel("RMSE of PM2.5 (µg/m³)")
    axes[-1].legend(loc="upper left", bbox_to_anchor=(1.02, 1.0), fontsize=8.5)
    fig.savefig(path)
    plt.close(fig)


def case_studies(cases: list[dict], path: Path, lead: int):
    """Each case: title, times, observed, and a dict of forecasts valid at ``times``."""
    fig, axes = plt.subplots(len(cases), 1, figsize=(9, 2.5 * len(cases)), squeeze=False)
    for ax, case in zip(axes[:, 0], cases):
        for lo, hi, c in zip(AQI_BOUNDS[2:4], AQI_BOUNDS[3:5], AQI_COLORS[2:4]):
            ax.axhspan(lo, hi, color=c, alpha=0.08, lw=0)
        ax.axhline(35.4, color=AQI_COLORS[2], lw=0.8, ls=":")
        ax.plot(case["times"], case["observed"], color=INK, lw=2.2, label="observed")
        for name, series in case["forecasts"].items():
            main = name == "SmokeCast"
            ax.plot(case["times"], series, color=COLORS.get(name, MUTED), lw=2.0 if main else 1.0,
                    ls="-" if main else "--", alpha=1.0 if main else 0.9, label=name)
        ax.set_title(case["title"], loc="left", fontsize=10, color=INK)
        ax.set_ylabel("PM2.5 (µg/m³)")
        ax.set_ylim(bottom=0)
        ax.xaxis.set_major_locator(mdates.DayLocator())
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
    axes[0, 0].legend(ncol=5, fontsize=8, loc="lower left", bbox_to_anchor=(0, 1.12))
    fig.suptitle(f"Forecasts issued {lead} h before the time shown", x=0.01, ha="left",
                 fontsize=9, color=MUTED, y=1.0)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def smoke_animation(times, lat, lon, observed, forecasts: dict, smoke, smoke_grid, title, path: Path, extent):
    """Observed PM2.5 next to forecasts valid at the same hour, over the HMS smoke analysis."""
    panels = {"Observed PM2.5": observed, **forecasts}
    n = len(panels)
    fig = plt.figure(figsize=(4.6 * n + 1.2, 4.2))
    axes, tfs = [], []
    for k in range(n):
        ax, tf = map_axes(fig, 100 + 10 * n + k + 1, extent)
        axes.append(ax), tfs.append(tf)
    width = fig.get_figwidth()
    right = 1 - 1.5 / width
    fig.subplots_adjust(left=0.01, right=right, top=0.86, bottom=0.08, wspace=0.04)
    for ax, name in zip(axes, panels):
        ax.set_title(name, loc="left", color=INK)
    lon0, lat0, res = smoke_grid["lon0"], smoke_grid["lat0"], smoke_grid["res"]
    gx = lon0 + res * np.arange(smoke.shape[-1] + 1)
    gy = lat0 + res * np.arange(smoke.shape[-2] + 1)
    shade = [ax.pcolormesh(gx, gy, np.ma.masked_equal(smoke[0], 0), cmap="Greys", vmin=0, vmax=4,
                           alpha=0.35, zorder=1, **tf) for ax, tf in zip(axes, tfs)]
    scat = [ax.scatter(lon, lat, c=np.zeros_like(lat), cmap=AQI_CMAP, norm=AQI_NORM, s=34,
                       edgecolor=INK, linewidth=0.3, zorder=3, **tf) for ax, tf in zip(axes, tfs)]
    stamp = fig.text(0.01, 0.02, "", color=MUTED)
    fig.text(0.25, 0.02, "grey: NOAA HMS satellite smoke analysis for the day", color=MUTED, fontsize=8)
    cb = fig.colorbar(scat[-1], cax=fig.add_axes([right + 0.15 / width, 0.12, 0.13 / width, 0.7]))
    cb.set_ticks([(a + b) / 2 for a, b in zip(AQI_BOUNDS[:-1], AQI_BOUNDS[1:])])
    cb.set_ticklabels(AQI_NAMES, fontsize=7)
    fig.suptitle(title, x=0.01, ha="left", color=INK, y=0.98)

    def draw(i):
        for k, values in enumerate(v[i] for v in panels.values()):
            scat[k].set_array(np.ma.masked_invalid(values))
            shade[k].set_array(np.ma.masked_equal(smoke[i], 0))
        stamp.set_text(times[i].strftime("%Y-%m-%d %H:%M UTC"))
        return scat + shade + [stamp]

    anim = FuncAnimation(fig, draw, frames=len(times), blit=False)
    anim.save(path, writer=PillowWriter(fps=6))
    plt.close(fig)


def fault_example(times, clean, faulty, virtual, scores: dict, thresholds: dict, episode, title, path: Path):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9, 5), sharex=True, gridspec_kw={"height_ratios": [3, 2]})
    for ax in (ax1, ax2):
        ax.axvspan(times[episode[0]], times[episode[1] - 1], color="#cf222e", alpha=0.07, lw=0)
    ax1.plot(times, clean, color=MUTED, lw=1.0, label="true reading")
    ax1.plot(times, faulty, color=INK, lw=1.6, label="what the faulty sensor reports")
    ax1.plot(times, virtual, color="#cf222e", lw=1.4, label="virtual sensor (from neighbours)")
    ax1.set_ylabel("PM2.5 (µg/m³)")
    ax1.set_title(title, loc="left", fontsize=10, color=INK)
    ax1.legend(fontsize=8, ncol=3, loc="upper left")
    for name, s in scores.items():
        color = "#cf222e" if name.startswith("Virtual") else "#0969da"
        ax2.plot(times, s / thresholds[name], color=color, lw=1.3, label=name)
    ax2.axhline(1.0, color=INK, lw=0.8, ls=":")
    ax2.set_ylabel("score / alarm level")
    ax2.legend(fontsize=8, loc="upper left")
    ax2.xaxis.set_major_locator(mdates.DayLocator(interval=2))
    ax2.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
