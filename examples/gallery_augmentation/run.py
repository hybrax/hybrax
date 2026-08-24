"""Gallery: augmentation.

Generates synthetic sibling processes from a single fed-batch run via
prepare.augmentation, shows the automatic diagnostic plot, and demonstrates
augment_state_values (per-state control over what gets generated: here,
repairing product's monotonicity after default noise).

See docs/source/gallery/augmentation.md for the narrated version.
"""

import importlib.util
import os
import subprocess
import sys
from pathlib import Path

import numpy as np

import hybrax.format as hxf
from hybrax.train import augmentation, prepare, run_config

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


out = hxt_cli(
    "prepare",
    "--config",
    "prepare-config.json",
    "--output-dir",
    "prepared",
    "--overwrite",
)
for line in out.splitlines():
    if "UserWarning: " in line:
        print("UserWarning:", line.split("UserWarning: ", 1)[1])
print(f"augmentation diagnostic: {HERE / 'prepared/augmented-data.png'}")

# --- Direct Python: inspect what augmentation actually generated -----------
spec = importlib.util.spec_from_file_location("custom", str(HERE / "custom.py"))
custom = importlib.util.module_from_spec(spec)
spec.loader.exec_module(custom)

raw = prepare.load_raw_collection(str(HERE / "data.json"))
raw = custom.transform_process_collection(raw, None)
cfg = run_config.RunConfig(
    prepare=run_config.PrepareConfig(
        raw_input=str(HERE / "data.json"),
        augmentation=run_config.AugmentationConfig(
            n_children_per_process=5,
            n_time_points=11,
            noise_std={
                "biomass": 0.05,
                "glucose": 0.05,
                "lactate": 0.05,
                "product": 0.05,
            },
        ),
    )
)
augmented = augmentation.augment_process_collection(
    raw, cfg, custom.augment_state_values
)

children = [n for n, p in augmented.processes.items() if hasattr(p, "parent_process")]
print(f"{len(children)} synthetic children from 1 parent")
for name in children[:3]:
    values = np.asarray(
        augmented.processes[name]
        .reactor_medium.components["product"]
        .concentration.values
    )
    print(
        f"  {name}: parent={augmented.processes[name].parent_process}"
        f"  product monotone={bool(np.all(np.diff(values) >= 0))}"
    )

# --- Train on the enlarged dataset ------------------------------------------
hxt_cli("train", "--config", "train-config.json", "--overwrite")
print(f"run directory: {HERE / 'run'}")
