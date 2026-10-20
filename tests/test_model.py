import numpy as np
import torch

from smokecast.dataset import CHANNELS, U, V
from smokecast.graph import edge_geometry, knn_edges, wind_edge_features
from smokecast.layers import GraphAttentionLayer, segment_softmax
from smokecast.model import SmokeCast


def test_segment_softmax_sums_to_one_per_receiver():
    torch.manual_seed(0)
    dst = torch.tensor([0, 0, 1, 1, 1, 3])
    w = segment_softmax(torch.randn(2, 6, 4), dst, num_segments=4)
    sums = torch.zeros(2, 4, 4).index_add(1, dst, w)
    assert torch.allclose(sums[:, [0, 1, 3]], torch.ones(2, 3, 4), atol=1e-6)
    assert torch.all(sums[:, 2] == 0)


def test_message_passing_is_permutation_equivariant():
    torch.manual_seed(0)
    n, d = 7, 16
    src = torch.tensor([0, 1, 1, 2, 3, 4, 5, 6, 2, 0])
    dst = torch.tensor([1, 0, 2, 1, 4, 3, 6, 5, 0, 2])
    layer = GraphAttentionLayer(d, heads=4).eval()
    h, e = torch.randn(2, n, d), torch.randn(2, len(src), d)
    out, _, _ = layer(h, e, src, dst)
    perm = torch.randperm(n)
    inv = torch.argsort(perm)
    out_p, _, _ = layer(h[:, perm], e, inv[src], inv[dst])
    assert torch.allclose(out_p, out[:, perm], atol=1e-5)


def test_knn_graph_is_symmetric():
    rng = np.random.default_rng(0)
    lat, lon = rng.uniform(30, 45, 30), rng.uniform(-95, -70, 30)
    src, dst = knn_edges(lat, lon, 4)
    pairs = set(zip(src.tolist(), dst.tolist()))
    assert all((d, s) in pairs for s, d in pairs)
    assert not any(s == d for s, d in pairs)


def test_wind_edges_only_reach_downwind():
    # site 1 is due east of site 0; a westerly wind (blowing toward the east) carries 0 -> 1, not 1 -> 0
    lat, lon = np.array([40.0, 40.0]), np.array([-80.0, -79.0])
    src, dst = np.array([0, 1]), np.array([1, 0])
    geometry = torch.tensor(edge_geometry(lat, lon, src, dst))
    wind = torch.tensor([[[10.0, 0.0], [10.0, 0.0]]])
    feats = wind_edge_features(wind, geometry, torch.tensor(src))
    along, reach = feats[0, :, 0], feats[0, :, 2]
    assert along[0] > 0.9 and along[1] < -0.9
    assert reach[0] > 0.5 and reach[1] == 0  # 85 km at 36 km/h is about 2.4 h


def _batch(b, t, n):
    torch.manual_seed(1)
    return {"x": torch.randn(b, t, n, len(CHANNELS)), "base": torch.randn(b, n),
            "time": torch.randn(b, 6), "node_mask": torch.zeros(b, n, dtype=torch.bool)}


def test_shapes_and_ablation_switches():
    n = 12
    rng = np.random.default_rng(0)
    lat, lon = rng.uniform(35, 42, n), rng.uniform(-85, -75, n)
    batch = _batch(3, 12, n)
    for use_graph in (True, False):
        for wind_edges in (True, False):
            for use_smoke in (True, False):
                m = SmokeCast(lat, lon, len(CHANNELS), horizon=6, dim=32, layers=2, heads=4, k=3,
                              use_graph=use_graph, wind_edges=wind_edges, use_smoke=use_smoke, history=12)
                out = m(batch)
                assert out.shape == (3, 7, n) and torch.isfinite(out).all()


def test_without_graph_a_site_ignores_its_neighbours():
    n = 8
    rng = np.random.default_rng(1)
    lat, lon = rng.uniform(35, 42, n), rng.uniform(-85, -75, n)
    m = SmokeCast(lat, lon, len(CHANNELS), horizon=4, dim=32, layers=2, heads=4, k=3,
                  use_graph=False, history=12).eval()
    batch = _batch(1, 12, n)
    other = {**batch, "x": batch["x"].clone()}
    other["x"][:, :, 1:] += 5.0  # change every site except site 0, including the wind it would see
    other["x"][:, :, 0, [U, V]] = batch["x"][:, :, 0, [U, V]]
    assert torch.allclose(m(batch)[..., 0], m(other)[..., 0], atol=1e-5)
