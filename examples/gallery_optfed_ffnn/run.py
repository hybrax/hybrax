"""Fit the neural OptFed variant and select a concentration-fit checkpoint."""

import json
import os
import subprocess
import sys
from pathlib import Path

import equinox as eqx
import jax
import pandas as pd

from hybrax.format.serialization import load_process_collection
from hybrax.train import model_load

HERE = Path(__file__).parent
# Reuse the data, scores, and checkpoint selection of the mechanistic example.
OPTFED = HERE.parent / "gallery_optfed"
sys.path.insert(0, str(OPTFED))
from run import sampling_predictions, scores, select_checkpoint


def run_cli(*arguments):
    subprocess.run(
        [sys.executable, "-m", "hybrax.train.cli", *arguments],
        cwd=HERE,
        env=os.environ | {"JAX_PLATFORMS": "cpu"},
        check=True,
    )


if __name__ == "__main__":
    run_cli(
        "prepare",
        "--config",
        "prepare-config.json",
        "--output-dir",
        "prepared",
        "--overwrite",
    )
    run_cli("train", "--config", "train-config.json", "--overwrite")
    collection = load_process_collection(OPTFED / "data.json")
    checkpoint = select_checkpoint(HERE / "run", collection)
    print(f"Selected checkpoint: {checkpoint.name}")
    forward_config = json.loads((HERE / "forward-config.json").read_text())
    forward_config["models"] = [str(checkpoint.resolve())]
    selected_config = HERE / "run/selected-forward-config.json"
    selected_config.write_text(json.dumps(forward_config, indent=2) + "\n")
    run_cli(
        "forward",
        "--config",
        str(selected_config.resolve()),
        "--output-dir",
        "run/forward",
        "--overwrite",
    )
    predictions = pd.read_csv(HERE / "run/selected-predictions.csv")
    points = sampling_predictions(collection, predictions)
    for name, value in scores(points).items():
        print(f"{name:10s} R2 = {value:.4f}")
    model, _ = model_load(checkpoint)
    count = sum(
        x.size
        for x in jax.tree.leaves(
            eqx.filter(model.reaction_module.network, eqx.is_array)
        )
    )
    print(f"Network parameters: {count}")
