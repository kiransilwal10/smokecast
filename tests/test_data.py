import numpy as np
import pandas as pd

from smokecast.faults import greedy_colouring, inject
from smokecast.geo import haversine_km, offset_point, wind_components
from smokecast.dataset import smoke_features


def test_wind_components_follow_meteorological_convention():
    u, v = wind_components(np.array([10.0, 10.0]), np.array([270.0, 0.0]))  # from the west, from the north
    assert np.allclose([u[0], v[0]], [10.0, 0.0], atol=1e-6)
    assert np.allclose([u[1], v[1]], [0.0, -10.0], atol=1e-6)


def test_offset_point_moves_the_requested_distance():
    lat, lon = offset_point(40.0, -80.0, 300.0, 400.0)
    assert abs(haversine_km(40.0, -80.0, lat, lon) - 500.0) < 5.0


def test_smoke_upwind_looks_into_the_wind():
    hms = {"first_day": "2023-06-06", "lon0": -90.0, "lat0": 35.0, "res": 0.25, "nx": 80, "ny": 60}
    grid = np.zeros((3, 60, 80), np.uint8)
    grid[0, 42:, :] = 3  # heavy smoke north of 45.5 N on the first day only
    index = pd.date_range("2023-06-07 00:00", periods=2, freq="h")  # uses the previous day's raster
    lat, lon = np.array([45.0]), np.array([-80.0])
    from_north = np.array([[0.0], [0.0]]), np.array([[-8.0], [-8.0]])  # wind blowing toward the south
    from_south = np.array([[0.0], [0.0]]), np.array([[8.0], [8.0]])
    here, up_n = smoke_features(grid, hms, index, lat, lon, *from_north)
    _, up_s = smoke_features(grid, hms, index, lat, lon, *from_south)
    assert np.all(here == 0)
    assert np.all(up_n > 0.6) and np.all(up_s == 0)


def test_fault_injection_offsets_and_keeps_gaps():
    rng = np.random.default_rng(0)
    pm = rng.uniform(5, 15, (24 * 120, 3)).astype(np.float32)
    pm[100:110, 0] = np.nan
    bad, episodes = inject(pm, np.arange(len(pm)), np.random.default_rng(1))
    assert np.isnan(bad[100:110, 0]).all()
    for ep in episodes:
        rows = slice(ep["start"], ep["start"] + 120)
        diff = bad[rows, ep["site"]] - pm[rows, ep["site"]]
        if ep["kind"] == "offset":
            assert np.allclose(diff[np.isfinite(diff)], 10.0)
        if ep["kind"] == "stuck":
            assert np.nanstd(bad[rows, ep["site"]]) == 0


def test_colouring_never_hides_two_neighbours_together():
    src = np.array([0, 1, 1, 2, 2, 3, 3, 0])
    dst = np.array([1, 0, 2, 1, 3, 2, 0, 3])
    colour = greedy_colouring(src, dst, 4)
    assert all(colour[s] != colour[d] for s, d in zip(src, dst))
