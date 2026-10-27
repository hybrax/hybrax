"""Build, train and plot the synthetic HEK293 literature-model example."""

import os
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ENV = os.environ | {
    "JAX_PLATFORMS": "cpu",
    "JAX_ENABLE_X64": "1",
    "HYBRAX_TRAIN_DEVICES": "1",
    "MPLBACKEND": "Agg",
}

if __name__ == "__main__":
    commands = [
        [sys.executable, "import_from_mat.py"],
        [sys.executable, "plot_data.py"],
        [
            sys.executable,
            "-m",
            "hybrax.train.cli",
            "prepare",
            "--config",
            "prepare-config.json",
            "--overwrite",
        ],
        [
            sys.executable,
            "-m",
            "hybrax.train.cli",
            "train",
            "--config",
            "train-config.json",
            "--overwrite",
        ],
        [sys.executable, "results.py"],
    ]
    for command in commands:
        subprocess.run(command, cwd=HERE, env=ENV, check=True)
