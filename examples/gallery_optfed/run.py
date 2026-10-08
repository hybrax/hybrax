"""Fit the eleven-parameter OptFed model to published concentrations."""

import os
import subprocess
import sys
from pathlib import Path

import jax
import numpy as np
import pandas as pd

from hybrax.format.serialization import load_process_collection
from hybrax.train import model_load

HERE = Path(__file__).parent


def run_cli(*arguments):
    subprocess.run(
        [sys.executable, "-m", "hybrax.train.cli", *arguments],
        cwd=HERE,
        env=os.environ | {"JAX_PLATFORMS": "cpu", "HYBRAX_TRAIN_DEVICES": "1"},
        check=True,
    )


def sampling_predictions(collection, predictions):
    """Align concentration predictions with the experimental measurements."""
    rows = []
    for name, process in collection.processes.items():
        predicted = predictions[predictions.process == name]
        components = process.reactor_medium.components
        frame = pd.DataFrame({"t": components["biomass"].concentration.times})
        frame["process"] = name
        for species in ("biomass", "glycerol", "product"):
            frame[species] = components[species].concentration.values
            frame[f"predicted_{species}"] = np.interp(
                frame.t, predicted.t, predicted[f"c_{species}"]
            )
        frame["P_X"] = frame["product"] / frame["biomass"]
        frame["predicted_P_X"] = frame.predicted_product / frame.predicted_biomass
        rows.append(frame)
    return pd.concat(rows, ignore_index=True)


def scores(points):
    return {
        name: float(
            1
            - np.sum((points[name] - points[f"predicted_{name}"]) ** 2)
            / np.sum((points[name] - points[name].mean()) ** 2)
        )
        for name in ("biomass", "glycerol", "product", "P_X")
    }


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
    run_cli(
        "forward",
        "--config",
        "forward-config.json",
        "--output-dir",
        "run/forward",
        "--overwrite",
    )
    collection = load_process_collection(HERE / "data.json")
    points = sampling_predictions(collection, pd.read_csv(HERE / "run/predictions.csv"))
    for name, value in scores(points).items():
        print(f"{name:10s} R2 = {value:.4f}")
    model, _ = model_load(HERE / "run")
    count = sum(x.size for x in jax.tree.leaves(model.reaction_module.raw_parameters))
    print(f"Fitted parameters: {count}")
