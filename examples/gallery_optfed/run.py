"""Fit eleven OptFed parameters and select a concentration-fit checkpoint."""

import json
import os
import subprocess
import sys
from pathlib import Path

import jax
import numpy as np
import pandas as pd

from hybrax.format.serialization import load_process_collection
from hybrax.train import model_load, model_predict, model_reload

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
        indices = np.searchsorted(predicted.t, frame.t)
        np.testing.assert_allclose(
            predicted.t.iloc[indices], frame.t, atol=1e-10, rtol=0
        )
        for species in ("biomass", "glycerol", "product"):
            frame[species] = components[species].concentration.values
            frame[f"predicted_{species}"] = (
                predicted[f"c_{species}"].iloc[indices].to_numpy()
            )
        frame["P_X"] = frame["product"] / frame["biomass"]
        frame["predicted_P_X"] = frame.predicted_product / frame.predicted_biomass
        rows.append(frame)
    return pd.concat(rows, ignore_index=True)


def normalized_scores(points, fields):
    """Pool residuals after scaling each target by its observed maximum."""
    observed = points[fields].to_numpy()
    predicted = points[[f"predicted_{name}" for name in fields]].to_numpy()
    scales = np.abs(observed).max(axis=0)
    normalized = observed / scales
    residual = (predicted - observed) / scales
    rss = np.sum(residual**2)
    tss = np.sum((normalized - normalized.mean(axis=0)) ** 2)
    return float(1 - rss / tss), float(np.sqrt(np.mean(residual**2)))


def select_checkpoint(run, collection):
    """Select a checkpoint with lowest training concentration NRMSE."""
    records = []
    model = None
    best_nrmse = np.inf
    for checkpoint in sorted((run / "checkpoints").glob("step_*")):
        if model is None:
            model, config = model_load(checkpoint)
        else:
            model, config = model_reload(checkpoint, model)
        exports = model_predict(model, config, collection)
        frames = []
        for name, export in exports.items():
            assert export.prediction_valid.all(), name
            frame = pd.DataFrame(
                {"process": name, "t": export.t, "V_real": export.v_real}
            )
            for column, species in enumerate(model.modeled_RMC_names):
                frame[f"c_{species}"] = export.c_species[:, column]
            frames.append(frame)
        predictions = pd.concat(frames, ignore_index=True)
        points = sampling_predictions(collection, predictions)
        r2, nrmse = normalized_scores(points, ["biomass", "glycerol", "product"])
        records.append(
            {
                "step": int(checkpoint.name.split("_")[1]),
                "concentration_r2": r2,
                "concentration_nrmse": nrmse,
            }
        )
        if nrmse < best_nrmse:
            best_nrmse = nrmse
            selected = checkpoint
            predictions.to_csv(run / "selected-predictions.csv", index=False)
    pd.DataFrame(records).to_csv(run / "checkpoint-scores.csv", index=False)
    (run / "selection.json").write_text(
        json.dumps(
            {
                "checkpoint": str(selected.resolve()),
                "concentration_nrmse": best_nrmse,
            },
            indent=2,
        )
        + "\n"
    )
    return selected


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
    collection = load_process_collection(HERE / "data.json")
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
    count = sum(x.size for x in jax.tree.leaves(model.reaction_module.raw_parameters))
    print(f"Fitted parameters: {count}")
