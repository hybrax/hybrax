"""Self-contained training checkpoint writer."""

from __future__ import annotations

import gzip
import json
import shutil
import time
from pathlib import Path
from typing import Any, Callable, Sequence

from .postprocessing import (
    ProcessPlotData,
    plot_grad_norm_curve,
    plot_loss_curve,
    render_process_figures,
)
from .serialization import save_model, save_opt_state


def _bundle_prepared_gz(src: Path, dst: Path) -> None:
    if src.suffix == ".gz" or src.name.endswith(".json.gz"):
        shutil.copyfile(src, dst)
        return
    with open(src, "rb") as source, gzip.open(dst, "wb") as destination:
        shutil.copyfileobj(source, destination)


class CheckpointWriter:
    def __init__(
        self,
        checkpoints_dir: Path,
        *,
        plotter: Any | None = None,
        plots_enabled: bool = True,
        prepared_src: Path | None = None,
    ) -> None:
        self._dir = Path(checkpoints_dir)
        self._dir.mkdir(parents=True, exist_ok=True)
        self._plotter = plotter
        self._plots_enabled = bool(plots_enabled)
        self._prepared_src = Path(prepared_src) if prepared_src is not None else None

    def write(
        self,
        *,
        step: int,
        samples_seen: int,
        wrapper: Any,
        opt_state: Any,
        mean_loss: float,
        holdout_loss: float | None,
        render_predictions_fn: Callable[[Path], list[ProcessPlotData] | None],
        loss_by_step: Sequence[float],
        grad_norm_by_step: Sequence[float] | None = None,
        per_target_loss_by_step: Sequence[tuple[float, ...]] | None = None,
        target_names: Sequence[str] | None = None,
        holdout_loss_by_step: dict[int, float] | None = None,
        holdout_per_target_by_step: dict[int, tuple[float, ...]] | None = None,
        holdout_label: str | None = None,
    ) -> Path:
        d = self._dir / f"step_{step:05d}"
        d.mkdir(parents=True, exist_ok=True)
        save_model(wrapper, d / "params.eqx")
        save_opt_state(opt_state, d / "opt_state.eqx")
        (d / "train_state.json").write_text(
            json.dumps(
                {
                    "step": int(step),
                    "samples_seen": int(samples_seen),
                    "mean_loss": float(mean_loss),
                    "holdout_loss": (
                        float(holdout_loss) if holdout_loss is not None else None
                    ),
                    "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z", time.localtime()),
                },
                indent=2,
            ),
            encoding="utf-8",
        )

        run_dir = self._dir.parent
        for name in ("config.json", "custom.py"):
            source = run_dir / name
            if source.is_file():
                shutil.copyfile(source, d / name)
        if self._prepared_src is not None and self._prepared_src.is_file():
            _bundle_prepared_gz(self._prepared_src, d / "prepared.json.gz")

        plot_data = render_predictions_fn(d / "predictions.csv")
        if self._plots_enabled and self._plotter is not None:
            self._plotter.submit(
                plot_loss_curve,
                list(loss_by_step),
                d / "loss_curve.png",
                title=f"Training loss (through step {step})",
                per_target_loss_by_step=(
                    list(per_target_loss_by_step) if per_target_loss_by_step else None
                ),
                target_names=tuple(target_names) if target_names else None,
                monitor_loss_by_step=(
                    dict(holdout_loss_by_step) if holdout_loss_by_step else None
                ),
                monitor_per_target_by_step=(
                    dict(holdout_per_target_by_step)
                    if holdout_per_target_by_step
                    else None
                ),
                monitor_label=holdout_label,
            )
            if grad_norm_by_step:
                self._plotter.submit(
                    plot_grad_norm_curve,
                    list(grad_norm_by_step),
                    d / "grad_norm_curve.png",
                    title=f"Gradient norm (through step {step})",
                )
            if plot_data:
                self._plotter.submit(render_process_figures, plot_data, d)
        self._update_latest(d)
        return d

    def _update_latest(self, step_dir: Path) -> None:
        link = self._dir / "latest"
        if link.is_symlink() or link.exists():
            link.unlink()
        link.symlink_to(step_dir.name)
