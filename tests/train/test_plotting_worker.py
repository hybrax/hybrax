from __future__ import annotations

import time
from pathlib import Path

import numpy as np

from bp_train.plotting_worker import BackgroundPlotter
from bp_train.postprocessing import ProcessPlotData, render_process_figures


# Module-level (picklable) jobs for the spawned worker.
def _slow_write(path: str, seconds: float) -> None:
    time.sleep(seconds)
    Path(path).write_text("done", encoding="utf-8")


def _plot_data(name: str = "p1") -> ProcessPlotData:
    """A minimal picklable per-process plot payload (1 species, 1 rate, no feeds)."""
    t = np.linspace(0.0, 2.0, 5)
    return ProcessPlotData(
        process_name=name,
        is_train=True,
        time_unit="h",
        t_start=0.0,
        t_end=2.0,
        v_unit="L",
        modeled_RMC_names=("biomass",),
        modeled_PV_names=(),
        modeled_FVC_names=(),
        rate_names=("biomass",),
        fvc_units=(),
        t_dense=t,
        c_dense=np.linspace(1.0, 0.6, 5).reshape(5, 1),
        q_dense=np.full((5, 1), -0.1),
        v_real_pred=np.ones(5),
        b_modeled_pred=np.zeros((5, 0)),
        c_std=None,
        q_std=None,
        v_std=None,
        v_real_true_dense=np.ones(5),
        b_modeled_true_dense=np.zeros((5, 0)),
        measured_series=(
            ("biomass", "g/L", np.array([0.0, 1.0, 2.0]), np.array([1.0, 0.8, 0.64])),
        ),
        volume_changes=(
            ("sample_1", "sample", False, np.array([1.0]), np.array([-0.1])),
        ),
        named_losses={"biomass": 0.01},
        total_loss=0.01,
    )


def test_render_process_figures_direct(tmp_path: Path):
    out = tmp_path / "plots"
    render_process_figures([_plot_data()], out)
    png = out / "p1.png"
    assert png.is_file()
    assert png.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"


def test_render_process_figures_in_background_worker(tmp_path: Path):
    # The whole reason ProcessPlotData exists: render_process_figures is picklable
    # (plain numpy, no JAX), so the SAME renderer used for run-root/forward plots
    # also runs in the spawn BackgroundPlotter for per-checkpoint plots.
    out = tmp_path / "plots"
    plotter = BackgroundPlotter()
    plotter.submit(render_process_figures, [_plot_data()], out)
    plotter.close()
    assert (out / "p1.png").is_file()


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
