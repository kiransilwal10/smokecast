"""Turn data/raw into model tensors: python scripts/build_dataset.py"""

import argparse
from pathlib import Path

from smokecast.dataset import build

SPLITS = {"train": [2017, 2021], "val": [2022, 2022], "test": [2023, 2025]}

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", type=Path, default=Path("data/raw"))
    ap.add_argument("--out", type=Path, default=Path("data/processed"))
    args = ap.parse_args()
    build(args.raw, args.out, SPLITS)
