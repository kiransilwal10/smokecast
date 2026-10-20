"""Public data sources, none of which need an API key.

- EPA AQS hourly PM2.5 (parameter 88101), one zip per year
- NOAA ISD-Lite hourly surface weather at airports, one small file per station-year
- NOAA HMS smoke polygons, drawn by analysts from satellite imagery, one zip per day
"""

from __future__ import annotations

import gzip
import io
import time
import zipfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import requests

AQS_URL = "https://aqs.epa.gov/aqsweb/airdata/hourly_88101_{year}.zip"
AQS_SITES_URL = "https://aqs.epa.gov/aqsweb/airdata/aqs_sites.zip"
ISD_HISTORY_URL = "https://www.ncei.noaa.gov/pub/data/noaa/isd-history.csv"
ISD_LITE_URL = "https://noaa-isd-pds.s3.amazonaws.com/isd-lite/data/{year}/{station}-{year}.gz"
HMS_URL = ("https://satepsanone.nesdis.noaa.gov/pub/FIRE/web/HMS/Smoke_Polygons/Shapefile/"
           "{d:%Y}/{d:%m}/hms_smoke{d:%Y%m%d}.zip")

HMS_DENSITY = {"light": 1, "medium": 2, "heavy": 3, "5.000": 1, "16.000": 2, "27.000": 3}


def fetch(url: str, path: Path, retries: int = 5) -> bool:
    """Download ``url`` to ``path`` unless it is already there. False if the server has no such file."""
    if path.exists():
        return True
    if path.with_suffix(path.suffix + ".missing").exists():
        return False
    path.parent.mkdir(parents=True, exist_ok=True)
    for attempt in range(retries):
        try:
            with requests.get(url, stream=True, timeout=120) as r:
                if r.status_code == 404:
                    path.with_suffix(path.suffix + ".missing").touch()
                    return False
                r.raise_for_status()
                tmp = path.with_suffix(path.suffix + ".part")
                with open(tmp, "wb") as f:
                    for chunk in r.iter_content(1 << 20):
                        f.write(chunk)
                tmp.rename(path)
                return True
        except requests.RequestException:
            time.sleep(min(10 * 2 ** attempt, 120))
    raise RuntimeError(f"failed to download {url}")


def fetch_many(jobs: list[tuple[str, Path]], workers: int = 6) -> list[bool]:
    with ThreadPoolExecutor(workers) as pool:
        return list(pool.map(lambda job: fetch(*job), jobs))


# ---------------------------------------------------------------- EPA AQS

def read_aqs(path: Path, bbox: tuple[float, float, float, float]) -> pd.DataFrame:
    """One year of hourly PM2.5 inside ``bbox`` (lon0, lon1, lat0, lat1), averaged over co-located monitors."""
    cols = ["State Code", "County Code", "Site Num", "Latitude", "Longitude",
            "Date GMT", "Time GMT", "Sample Measurement"]
    df = pd.read_csv(path, usecols=cols, dtype={"State Code": str, "County Code": str, "Site Num": str})
    lon0, lon1, lat0, lat1 = bbox
    df = df[df.Longitude.between(lon0, lon1) & df.Latitude.between(lat0, lat1)]
    df = pd.DataFrame({
        "site": df["State Code"] + "-" + df["County Code"] + "-" + df["Site Num"],
        "time": pd.to_datetime(df["Date GMT"] + " " + df["Time GMT"]),
        "pm": df["Sample Measurement"].astype(np.float32),
        "lat": df.Latitude, "lon": df.Longitude,
    })
    return df.groupby(["site", "time"], as_index=False).agg(pm=("pm", "mean"), lat=("lat", "first"),
                                                            lon=("lon", "first"))


def aqs_site_names(path: Path) -> dict[str, dict]:
    df = pd.read_csv(path, dtype=str, usecols=["State Code", "County Code", "Site Number",
                                                "Local Site Name", "City Name", "State Name"])
    df["site"] = df["State Code"] + "-" + df["County Code"] + "-" + df["Site Number"]
    return {r.site: {"name": r["Local Site Name"], "city": r["City Name"], "state": r["State Name"]}
            for _, r in df.fillna("").iterrows()}


# ---------------------------------------------------------------- NOAA ISD-Lite

def isd_stations(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, dtype={"USAF": str, "WBAN": str})
    df = df.dropna(subset=["LAT", "LON"])
    df["station"] = df.USAF + "-" + df.WBAN
    df["begin"] = pd.to_datetime(df.BEGIN.astype(str), format="%Y%m%d", errors="coerce")
    df["end"] = pd.to_datetime(df.END.astype(str), format="%Y%m%d", errors="coerce")
    return df[["station", "STATION NAME", "LAT", "LON", "begin", "end"]].rename(
        columns={"STATION NAME": "name", "LAT": "lat", "LON": "lon"})


ISD_FIELDS = ("temp_c", "dewpoint_c", "slp_hpa", "wind_dir_deg", "wind_speed_ms", "precip_1h_mm")


def read_isd_lite(path: Path, index: pd.DatetimeIndex) -> np.ndarray:
    """Hourly (len(index), 6) array of ISD_FIELDS, NaN where missing. Times are UTC."""
    out = np.full((len(index), len(ISD_FIELDS)), np.nan, dtype=np.float32)
    raw = np.loadtxt(gzip.open(path, "rt"), dtype=np.int64, ndmin=2)
    if raw.size == 0:
        return out
    t = pd.to_datetime(pd.DataFrame({"year": raw[:, 0], "month": raw[:, 1], "day": raw[:, 2],
                                     "hour": raw[:, 3]}))
    rows = index.get_indexer(t)
    ok = rows >= 0
    values = raw[ok][:, [4, 5, 6, 7, 8, 10]].astype(np.float32)
    values[values == -9999] = np.nan
    values[:, [0, 1, 2, 4, 5]] /= 10.0  # tenths of a unit
    values[:, 5] = np.where(values[:, 5] < 0, 0.0, values[:, 5])  # -1 means a trace
    out[rows[ok]] = values
    return out


# ---------------------------------------------------------------- NOAA HMS smoke

def read_hms(path: Path) -> list[tuple[int, np.ndarray]]:
    """Smoke polygons for one day as (density 1..3, (K, 2) lon/lat ring) pairs."""
    import shapefile  # pyshp

    with zipfile.ZipFile(path) as z:
        names = {Path(n).suffix.lower(): n for n in z.namelist()}
        if ".shp" not in names:
            return []
        reader = shapefile.Reader(shp=io.BytesIO(z.read(names[".shp"])),
                                  dbf=io.BytesIO(z.read(names[".dbf"])))
        fields = [f[0].lower() for f in reader.fields[1:]]
        polygons = []
        for shape_rec in reader.iterShapeRecords():
            record = dict(zip(fields, shape_rec.record))
            density = HMS_DENSITY.get(str(record.get("density", "")).strip().lower(), 1)
            points = np.asarray(shape_rec.shape.points, dtype=np.float64)
            parts = list(shape_rec.shape.parts) + [len(points)]
            for a, b in zip(parts[:-1], parts[1:]):
                if b - a >= 3:
                    polygons.append((density, points[a:b]))
        return polygons
