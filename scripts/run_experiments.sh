#!/usr/bin/env bash
# End to end: download -> build -> ablation -> evaluate -> fault detection -> figures -> serving bundle.
set -euo pipefail

python scripts/download.py --start 2017 --end 2025
python scripts/build_dataset.py

CFG=configs/smokecast.yaml
python -m smokecast.train --config $CFG --name gru        --set model.use_graph=false
python -m smokecast.train --config $CFG --name gnn_static --set model.wind_edges=false
python -m smokecast.train --config $CFG --name smokecast

python -m smokecast.evaluate --runs gru gnn_static smokecast
python -m smokecast.faults --run smokecast
python scripts/make_figures.py
python scripts/export_serving.py
