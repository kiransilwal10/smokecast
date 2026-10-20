"""Train, evaluate and package SmokeCast on a Kaggle GPU.

Inputs come from the private dataset holding data/processed and the code;
everything written under /kaggle/working is kept as the notebook's output.
"""

import glob
import os
import shutil
import subprocess
import sys
import time

meta = glob.glob("/kaggle/input/**/processed/meta.json", recursive=True)[0]
data = os.path.dirname(meta)
# Kaggle unpacks uploaded archives, so the code arrives as a folder
code = glob.glob("/kaggle/input/**/code/src/smokecast/train.py", recursive=True)[0].rsplit("/src/", 1)[0]
work = "/kaggle/working/smokecast"
shutil.copytree(code, work, ignore=shutil.ignore_patterns("._*", "*.egg-info"), dirs_exist_ok=True)
os.chdir(work)
env = {**os.environ, "PYTHONPATH": os.path.join(work, "src")}
subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv"])


def run(*args):
    start = time.time()
    print(f"\n$ python {' '.join(args)}", flush=True)
    subprocess.run([sys.executable, *args], check=True, env=env)
    print(f"({(time.time() - start) / 60:.1f} min)", flush=True)


cfg = "configs/smokecast.yaml"
for name, extra in [("smokecast", []), ("gnn_static", ["model.wind_edges=false"]), ("gru", ["model.use_graph=false"])]:
    run("-m", "smokecast.train", "--config", cfg, "--name", name, "--set", f"data_dir={data}", *extra)
run("-m", "smokecast.evaluate", "--runs", "gru", "gnn_static", "smokecast", "--data", data)
run("-m", "smokecast.faults", "--run", "smokecast", "--data", data)
run("scripts/export_serving.py", "--data", data)
for junk in ("src", "configs", "scripts"):  # keep the output to results only
    subprocess.run(["rm", "-rf", os.path.join(work, junk)])
print("ALL DONE", flush=True)
