import json

import numpy as np
import pytest
import torch

fastapi = pytest.importorskip("fastapi")
from fastapi.testclient import TestClient  # noqa: E402

from smokecast.dataset import CHANNELS  # noqa: E402
from smokecast.model import SmokeCast  # noqa: E402


@pytest.fixture()
def client(tmp_path, monkeypatch):
    n, hours = 6, 24 * 5
    rng = np.random.default_rng(0)
    lat, lon = rng.uniform(38, 44, n), rng.uniform(-85, -72, n)
    cfg = {"history": 12, "horizon": 6,
           "model": {"dim": 16, "layers": 1, "heads": 2, "k": 2, "use_graph": True, "wind_edges": True, "use_smoke": True}}
    model = SmokeCast(lat, lon, len(CHANNELS), horizon=6, history=12, **cfg["model"])
    torch.save({"config": cfg, "state_dict": model.state_dict()}, tmp_path / "model.pt")
    pm = rng.uniform(5, 15, (hours, n))
    pm[-30:, 0] = 80.0  # one site in a smoke episode
    sigma, mu = np.full(n, 5.0, np.float32), np.full(n, 10.0, np.float32)
    x = np.zeros((hours, n, len(CHANNELS)), np.float16)
    x[..., 0], x[..., 1] = (pm - mu) / sigma, 1.0
    np.savez(tmp_path / "inputs.npz", x=x, pm=pm.astype(np.float16), mu=mu, sigma=sigma,
             colour=np.arange(n) % 2, spread=np.full(n, 3.0), threshold=np.full((2, n), 2.0))
    sites = [{"site": f"s{i}", "name": f"Site {i}", "city": "", "state": "", "lat": float(lat[i]), "lon": float(lon[i])}
             for i in range(n)]
    (tmp_path / "meta.json").write_text(json.dumps({"first_hour": "2023-06-01 00:00:00", "sites": sites}))

    import smokecast.serve.app as app_module
    monkeypatch.setattr(app_module, "BUNDLE", tmp_path)
    app_module.bundle.cache_clear()
    app_module.forecast.cache_clear()
    return TestClient(app_module.app)


def test_health_and_forecast_shape(client):
    assert client.get("/api/health").json()["sites"] == 6
    body = client.get("/api/forecast", params={"issued": "2023-06-04T00:00"}).json()
    assert len(body["sites"]) == 6 and all(len(s["forecast"]) == 7 for s in body["sites"])
    assert all(v >= 0 for s in body["sites"] for v in s["forecast"])


def test_issue_time_outside_the_replay_window_is_rejected(client):
    assert client.get("/api/forecast", params={"issued": "2023-06-01T03:00"}).status_code == 400


def test_alerts_and_sensor_health(client):
    alerts = client.get("/api/alerts", params={"issued": "2023-06-05T20:00", "within_h": 6}).json()
    assert alerts["threshold_ugm3"] == 35.4
    sensors = client.get("/api/sensors", params={"issued": "2023-06-05T20:00", "simulate_fault": "s3"}).json()
    assert len(sensors["sensors"]) == 6 and all("flagged" in s for s in sensors["sensors"])
