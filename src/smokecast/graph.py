"""Monitoring-site graph and its wind-aware edge features.

Sites are linked to their k nearest neighbours (great-circle distance) and the
graph is made symmetric. Static edge features give the sender's position
relative to the receiver. Dynamic edge features come from the wind at the
sender: smoke is carried from j to i only when the wind blows from j toward i,
and it takes roughly distance / (wind component along the edge) to get there.
"""

from __future__ import annotations

import numpy as np
import torch

from .geo import east_north_km, haversine_km

EDGE_SCALE_KM = 300.0
WIND_EDGE_DIM = 3


def knn_edges(lat: np.ndarray, lon: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
    """Symmetric k-NN graph. Returns (src, dst) arrays of directed edges src -> dst."""
    n = len(lat)
    d = haversine_km(lat[:, None], lon[:, None], lat[None], lon[None])
    np.fill_diagonal(d, np.inf)
    nbrs = np.argsort(d, axis=1)[:, :k]
    adj = np.zeros((n, n), dtype=bool)
    adj[np.repeat(np.arange(n), k), nbrs.ravel()] = True
    adj |= adj.T
    dst, src = np.nonzero(adj)
    return src.astype(np.int64), dst.astype(np.int64)


def edge_geometry(lat, lon, src, dst) -> np.ndarray:
    """(E, 3): unit vector from sender to receiver (east, north) and distance in km."""
    east, north = east_north_km(lat[src], lon[src], lat[dst], lon[dst])
    dist = haversine_km(lat[src], lon[src], lat[dst], lon[dst])
    return np.stack([east / dist, north / dist, dist], 1).astype(np.float32)


def static_edge_features(geometry: np.ndarray) -> np.ndarray:
    """Sender offset relative to the receiver, [east, north, distance] / 300 km."""
    ux, uy, dist = geometry.T
    return (np.stack([-ux * dist, -uy * dist, dist], 1) / EDGE_SCALE_KM).astype(np.float32)


def wind_edge_features(wind: torch.Tensor, geometry: torch.Tensor, src: torch.Tensor) -> torch.Tensor:
    """Advection features per edge from the sender's wind.

    wind: (B, N, 2) recent mean wind in m/s; geometry: (E, 3) from ``edge_geometry``.
    Returns (B, E, 3): along-edge wind / 10, wind speed / 10, and exp(-transit / 24 h),
    which is near 1 when air from the sender arrives within hours and 0 when it blows the other way.
    """
    w = wind[:, src]                                            # (B, E, 2)
    along = w[..., 0] * geometry[:, 0] + w[..., 1] * geometry[:, 1]
    speed = w.norm(dim=-1)
    transit_h = geometry[:, 2] / (along.clamp_min(0.1) * 3.6)
    reach = torch.exp(-transit_h / 24.0) * (along > 0.5)
    return torch.stack([along / 10, speed / 10, reach], -1)


def node_positions(lat: np.ndarray, lon: np.ndarray) -> np.ndarray:
    """Unit-sphere xyz coordinates."""
    la, lo = np.deg2rad(lat), np.deg2rad(lon)
    return np.stack([np.cos(la) * np.cos(lo), np.cos(la) * np.sin(lo), np.sin(la)], 1).astype(np.float32)
