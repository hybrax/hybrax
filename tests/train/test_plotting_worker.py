from __future__ import annotations

import time
from pathlib import Path

import pandas as pd

from bp_train.plotting_worker import BackgroundPlotter
from bp_train.postprocessing import (
    export_observations_csv,
    render_process_plots_from_csv,
)
from bp_train.training_data import TrainingDataStore

# Reuse the tiny single-process collection fixture.
from test_serialization import _collection


# Module-level (picklable) jobs for the spawned worker.
def _slow_write(path: str, seconds: float) -> None:
    time.sleep(seconds)
    Path(path).write_text("done", encoding="utf-8")


def _make_prediction_csvs(tmp_path: Path) -> tuple[Path, Path]:
    pred = pd.DataFrame(
        {
            "process": ["p1"] * 4,
            "t": [0.0, 0.5, 1.0, 2.0],
            "c_biomass": [1.0, 0.9, 0.8, 0.64],
            "V_cont": [1.0, 1.0, 1.0, 1.0],
            "V_real": [1.0, 1.0, 0.9, 0.9],
            "q_biomass": [-0.1, -0.1, -0.1, -0.1],
        }
    )
    obs = pd.DataFrame(
        {
            "process": ["p1", "p1", "p1"],
            "variable": ["biomass", "biomass", "biomass"],
            "t": [0.0, 1.0, 2.0],
            "value": [1.0, 0.8, 0.64],
        }
    )
    pred_path = tmp_path / "predictions.csv"
    obs_path = tmp_path / "observations.csv"
    pred.to_csv(pred_path, index=False)
    obs.to_csv(obs_path, index=False)
    return pred_path, obs_path


def test_render_process_plots_from_csv_direct(tmp_path: Path):
    pred_path, obs_path = _make_prediction_csvs(tmp_path)
    out = tmp_path / "plots"
    render_process_plots_from_csv(
        pred_path, obs_path, out, process_names=("p1",)
    )
    png = out / "p1.png"
    assert png.is_file()
    assert png.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"


def test_render_tolerates_missing_observations(tmp_path: Path):
    pred_path, _ = _make_prediction_csvs(tmp_path)
    out = tmp_path / "plots"
    render_process_plots_from_csv(pred_path, tmp_path / "nope.csv", out)
    assert (out / "p1.png").is_file()


def test_export_observations_csv_long_format(tmp_path: Path):
    collection = _collection()
    store = TrainingDataStore.from_collection(
        collection,
        target_variable_order=["biomass"],
        target_source="reactor_components",
    )
    out = tmp_path / "observations.csv"
    export_observations_csv(collection, store, out)
    df = pd.read_csv(out)
    assert list(df.columns) == ["process", "variable", "t", "value"]
    assert set(df["variable"]) == {"biomass"}
    assert (df["process"] == "p1").all()
    assert len(df) == 3  # biomass measured at t in {0,1,2}


def test_background_plotter_runs_and_drains(tmp_path: Path):
    marker = tmp_path / "marker.txt"
    plotter = BackgroundPlotter(max_pending=2)
    plotter.submit(_slow_write, str(marker), 0.01)
    plotter.close()
    assert marker.read_text() == "done"


def test_background_plotter_drops_when_backed_up(tmp_path: Path):
    plotter = BackgroundPlotter(max_pending=1)
    # First job occupies the single worker slot; subsequent rapid submits with a
    # full queue are dropped (non-blocking guarantee).
    for i in range(6):
        plotter.submit(_slow_write, str(tmp_path / f"m{i}.txt"), 0.2)
    plotter.close()
    assert plotter._dropped > 0
    # At least the first job completed.
    written = list(tmp_path.glob("m*.txt"))
    assert len(written) >= 1
