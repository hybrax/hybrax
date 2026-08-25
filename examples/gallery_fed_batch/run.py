"""Gallery: feeds, boluses and samples.

A continuous feed, two boluses and sampling events in one run, and a reaction
module (custom.py) that reads the feed rate and a controlled process variable
as real biological inputs, not just transport bookkeeping.

See docs/source/gallery/fed_batch.md for the narrated version.
"""

import os
import subprocess
import sys
from pathlib import Path

import pandas as pd

import hybrax.format as hxf

HERE = Path(__file__).parent
ENV = {
    **os.environ,
    "JAX_PLATFORMS": "cpu",
    "HYBRAX_TRAIN_DEVICES": "1",
    "MPLBACKEND": "Agg",
}


def hxt_cli(*args):
    proc = subprocess.run(
        [sys.executable, "-m", "hybrax.train.cli", *args],
        cwd=HERE,
        env=ENV,
        capture_output=True,
        text=True,
    )
    if proc.returncode != 0:
        raise RuntimeError(proc.stdout + proc.stderr)
    return proc.stdout + proc.stderr


collection = hxf.serialization.load_process_collection(HERE / "data.json")
process = collection.processes["fedbatch_1"]
for name, vc in process.volume.volume_changes.items():
    print(f"{name:15s} {type(vc).__name__:20s} continuous={vc.is_continuous}")
hxf.print_rhs_ode(process)

hxt_cli(
    "prepare",
    "--config",
    "prepare-config.json",
    "--output-dir",
    "prepared",
    "--overwrite",
)
hxt_cli("train", "--config", "train-config.json", "--overwrite")
final_loss = pd.read_csv(HERE / "run/metrics.csv")["mean_loss"].iloc[-1]
print(f"final mean_loss = {final_loss:.5g}")

hxt_cli(
    "forward",
    "--config",
    "forward-config.json",
    "--output-dir",
    "run/forward",
    "--overwrite",
)
print(f"forward plot: {HERE / 'run/forward/plots/fedbatch_1.png'}")
