"""Bundle the trained model and the test-period inputs for the API: python scripts/export_serving.py"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from smokecast.dataset import SmokeData
from smokecast.faults import greedy_colouring
from smokecast.graph import knn_edges

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", type=Path, default=Path("runs/smokecast"))
    ap.add_argument("--data", type=Path, default=Path("data/processed"))
    ap.add_argument("--out", type=Path, default=Path("serving"))
    args = ap.parse_args()

    data = SmokeData(args.data)
    ckpt = torch.load(args.run / "model.pt", map_location="cpu", weights_only=False)
    history = ckpt["config"]["history"]
    first = data.forecast_times(tuple(data.meta["splits"]["test"]), history, 0)[0] - history + 1
    src, dst = knn_edges(data.lat, data.lon, ckpt["config"]["model"]["k"])
    args.out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.out / "inputs.npz", x=np.asarray(data.x[first:]), pm=np.asarray(data.pm[first:], np.float16),
                        mu=data.mu, sigma=data.sigma, colour=greedy_colouring(src, dst, data.n_nodes),
                        spread=np.load(args.run / "virtual_spread.npy"),
                        threshold=np.load(args.run / "threshold_virtual.npy"))
    torch.save(ckpt, args.out / "model.pt")
    keys = ("site", "name", "city", "state", "lat", "lon")
    (args.out / "meta.json").write_text(json.dumps({
        "first_hour": str(data.index[first]), "sites": [{k: s.get(k) for k in keys} for s in data.sites]}))
    size = sum(p.stat().st_size for p in args.out.iterdir()) / 1e6
    print(f"wrote {args.out}/ ({size:.0f} MB), replay from {data.index[first + history - 1]}")
