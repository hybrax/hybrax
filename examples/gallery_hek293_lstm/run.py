"""Build, train and plot the synthetic HEK293 literature-model example."""

import os
import subprocess
import sys
from pathlib import Path

from results import select_checkpoint

HERE = Path(__file__).resolve().parent
ENV = os.environ | {
    "JAX_PLATFORMS": "cpu",
    "JAX_ENABLE_X64": "1",
    "HYBRAX_TRAIN_DEVICES": "1",
    "MPLBACKEND": "Agg",
}


def run(*args):
    subprocess.run([sys.executable, *args], cwd=HERE, env=ENV, check=True)


def hybrax(command, config, *args):
    run("-m", "hybrax.train.cli", command, "--config", config, "--overwrite", *args)


if __name__ == "__main__":
    run("import_from_mat.py")
    run("plot_data.py")
    hybrax("prepare", "prepare-config.json")
    hybrax("train", "train-config.json")
    # Fine-tune the checkpoint with the lowest Br7 loss, not the final weights.
    epoch = select_checkpoint(HERE)
    hybrax(
        "train",
        "train-config-nonneg.json",
        "--init-from",
        f"run/checkpoints/step_{epoch:05d}",
    )
    run("results.py")
