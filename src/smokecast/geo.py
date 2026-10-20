"""Spherical geometry and wind helpers (numpy only)."""

from __future__ import annotations

import numpy as np

EARTH_RADIUS_KM = 6371.0


def haversine_km(lat1, lon1, lat2, lon2):
    lat1, lon1, lat2, lon2 = map(np.deg2rad, (lat1, lon1, lat2, lon2))
    a = (np.sin((lat2 - lat1) / 2) ** 2
         + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2)
    return 2 * EARTH_RADIUS_KM * np.arcsin(np.sqrt(np.clip(a, 0.0, 1.0)))


def east_north_km(lat0, lon0, lat1, lon1):
    """Local east/north offset (km) of point 1 from point 0, fine below a few hundred km."""
    lat_mid = np.deg2rad((np.asarray(lat0) + np.asarray(lat1)) / 2)
    east = np.deg2rad(np.asarray(lon1) - np.asarray(lon0)) * EARTH_RADIUS_KM * np.cos(lat_mid)
    north = np.deg2rad(np.asarray(lat1) - np.asarray(lat0)) * EARTH_RADIUS_KM
    return east, north


def offset_point(lat, lon, east_km, north_km):
    """Point displaced by (east_km, north_km) from (lat, lon)."""
    lat2 = lat + np.rad2deg(north_km / EARTH_RADIUS_KM)
    lon2 = lon + np.rad2deg(east_km / (EARTH_RADIUS_KM * np.cos(np.deg2rad(lat))))
    return lat2, lon2


def wind_components(speed, direction_deg):
    """Meteorological convention: direction is where the wind blows *from*. Returns (u east, v north)."""
    theta = np.deg2rad(direction_deg)
    return -speed * np.sin(theta), -speed * np.cos(theta)
