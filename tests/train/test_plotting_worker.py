from __future__ import annotations

import time
from pathlib import Path

import pandas as pd

from bp_train.plotting_worker import BackgroundPlotter
from bp_train.postprocessing import (
    measured_points_records,
    render_process_plots_from_csv,
)
from bp_train.training_data import TrainingDataStore

# Reuse the tiny single-process collection fixture.
from test_serialization import _collection


# Module-level (picklable) jobs for the spawned worker.
def _slow_write(path: str, seconds: float) -> None:
    time.sleep(seconds)
    Path(path).write_text("done", encoding="utf-8")


def _make_prediction_csv(
    tmp_path: Path,
) -> tuple[Path, list[tuple[str, str, float, float]]]:
    pred = pd.DataFrame(
        {
            "process": ["p1"] * 4,
            "t": [0.0, 0.5, 1.0, 2.0],
            "c_biomass": [1.0, 0.9, 0.8, 0.64],
            "V_real": [1.0, 1.0, 0.9, 0.9],
            "q_biomass": [-0.1, -0.1, -0.1, -0.1],
        }
    )
    pred_path = tmp_path / "predictions.csv"
    pred.to_csv(pred_path, index=False)
    # measured overlay points handed to the worker as picklable records
    records = [
        ("p1", "biomass", 0.0, 1.0),
        ("p1", "biomass", 1.0, 0.8),
        ("p1", "biomass", 2.0, 0.64),
    ]
    return pred_path, records


def test_render_process_plots_from_csv_direct(tmp_path: Path):
    pred_path, records = _make_prediction_csv(tmp_path)
    out = tmp_path / "plots"
    render_process_plots_from_csv(pred_path, records, out, process_names=("p1",))
    png = out / "p1.png"
    assert png.is_file()
    assert png.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"


def test_render_tolerates_missing_observations(tmp_path: Path):
    pred_path, _ = _make_prediction_csv(tmp_path)
    out = tmp_path / "plots"
    render_process_plots_from_csv(pred_path, None, out)
    assert (out / "p1.png").is_file()


def test_measured_points_records_long_format(tmp_path: Path):
    collection = _collection()
    store = TrainingDataStore.from_collection(
        collection,
        target_variable_order=["biomass"],
        target_source="reactor_components",
    )
    records = measured_points_records(collection, store)
    df = pd.DataFrame(records, columns=["process", "variable", "t", "value"])
    assert set(df["variable"]) == {"biomass"}
    assert (df["process"] == "p1").all()
    assert len(df) == 3  # biomass measured at t in {0,1,2}


def test_background_plotter_runs_and_drains(tmp_path: Path):
    marker = tmp_path / "marker.txt"
    plotter = BackgroundPlotter()
    plotter.submit(_slow_write, str(marker), 0.01)
    plotter.close()
    assert marker.read_text() == "done"


def test_background_plotter_is_non_lossy(tmp_path: Path):
    # Every submitted job runs and is drained at close() — nothing is dropped.
    plotter = BackgroundPlotter()
    for i in range(6):
        plotter.submit(_slow_write, str(tmp_path / f"m{i}.txt"), 0.02)
    plotter.close()
    written = sorted(p.name for p in tmp_path.glob("m*.txt"))
    assert written == [f"m{i}.txt" for i in range(6)]
