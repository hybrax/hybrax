"""Periodic training checkpoints: resumable state + background-rendered plots.

A checkpoint is the **lightweight resumable state** written synchronously
(``params.eqx`` = trainable leaves, ``opt_state.eqx``, ``train_state.json``)
plus per-checkpoint plots submitted to a :class:`BackgroundPlotter` so training
never blocks on matplotlib. Retention follows ``checkpoint.keep``: ``best+latest``
prunes every other ``step_*`` dir after each write; ``all`` keeps everything.
"""

from __future__ import annotations

import gzip
import json
import logging
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
from .run_config import CheckpointConfig
from .serialization import save_model, save_opt_state

logger = logging.getLogger(__name__)


def _bundle_prepared_gz(src: Path, dst: Path) -> None:
    """Copy the prepared artifact to ``dst`` as gzip. If ``src`` is already
    gzipped, copy it verbatim; otherwise gzip-compress the plain JSON."""
    if src.suffix == ".gz" or src.name.endswith(".json.gz"):
        shutil.copyfile(src, dst)
        return
    with open(src, "rb") as fi, gzip.open(dst, "wb") as fo:
        shutil.copyfileobj(fi, fo)


class CheckpointWriter:
    """Write resumable state + submit plot jobs at a fixed step cadence.

    Driven by the pydantic :class:`~bp_train.run_config.CheckpointConfig`
    (``every``, ``keep``) + the ``checkpoints/`` directory — no second config
    type, no fused ``log_every``.
    """

    def __init__(
        self,
        checkpoints_dir: Path,
        cfg: CheckpointConfig,
        *,
        plotter: Any | None = None,
        plots_enabled: bool = True,
        prepared_src: Path | None = None,
    ) -> None:
        self._dir = Path(checkpoints_dir)
        self._cfg = cfg
        self._enabled = int(cfg.every) > 0
        self._plotter = plotter
        self._plots_enabled = bool(plots_enabled)
        # Resolved prepared.json(.gz) path; bundled into every checkpoint so each
        # folder is self-contained (config.json + custom.py come from the run dir).
        self._prepared_src = Path(prepared_src) if prepared_src is not None else None
        self._best_ckpt_loss = float("inf")
        if self._enabled:
            self._dir.mkdir(parents=True, exist_ok=True)
            logger.info(
                "checkpointing enabled: dir=%s every=%d keep=%s",
                self._dir,
                int(cfg.every),
                cfg.keep,
            )

    @property
    def enabled(self) -> bool:
        return self._enabled

    def maybe_write(
        self,
        *,
        step: int,
        wrapper: Any,
        opt_state: Any,
        mean_loss: float,
        best_loss: float,
        render_predictions_fn: Callable[[Path], list[ProcessPlotData] | None],
        loss_by_step: Sequence[float],
        grad_norm_by_step: Sequence[float] | None = None,
        per_target_loss_by_step: Sequence[tuple[float, ...]] | None = None,
        target_names: Sequence[str] | None = None,
        monitor_loss_by_step: dict[int, float] | None = None,
        monitor_per_target_by_step: dict[int, tuple[float, ...]] | None = None,
        monitor_label: str | None = None,
    ) -> Path | None:
        """Write a checkpoint if ``step`` is a positive multiple of ``every``.

        Order: resumable state (sync) → predictions.csv + per-process plot data
        (sync; may raise) → plot jobs (background) → publish ``latest``/``best``
        symlinks → prune. ``render_predictions_fn`` writes predictions.csv and
        returns the picklable per-process :class:`ProcessPlotData` for the
        background renderer. If it raises, the partial state remains but no
        symlink is published and no pruning happens (so a failed export never
        clobbers a good ``latest``/``best``).
        """
        if not self._enabled or step <= 0 or step % int(self._cfg.every) != 0:
            return None

        d = self._dir / f"step_{step:05d}"
        d.mkdir(parents=True, exist_ok=True)

        save_model(wrapper, d / "params.eqx")  # trainable leaves only
        save_opt_state(opt_state, d / "opt_state.eqx")
        (d / "train_state.json").write_text(
            json.dumps(
                {
                    "step": int(step),
                    "mean_loss": float(mean_loss),
                    "best_loss": float(best_loss),
                    "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z", time.localtime()),
                },
                indent=2,
            ),
            encoding="utf-8",
        )

        # Make the checkpoint self-contained: bundle config.json + custom.py (from
        # the run dir) + prepared.json.gz, so the folder loads on its own.
        run_dir = self._dir.parent
        for fname in ("config.json", "custom.py"):
            src = run_dir / fname
            if src.is_file():
                shutil.copyfile(src, d / fname)
        if self._prepared_src is not None and self._prepared_src.is_file():
            _bundle_prepared_gz(self._prepared_src, d / "prepared.json.gz")

        # JAX forward sim, main process. Writes predictions.csv and returns the
        # picklable per-process plot data. Raising here aborts before publishing.
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
                    dict(monitor_loss_by_step) if monitor_loss_by_step else None
                ),
                monitor_per_target_by_step=(
                    dict(monitor_per_target_by_step)
                    if monitor_per_target_by_step
                    else None
                ),
                monitor_label=monitor_label,
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

        self._update_symlink("latest", d)
        if float(mean_loss) <= self._best_ckpt_loss:
            self._best_ckpt_loss = float(mean_loss)
            self._update_symlink("best", d)

        if self._cfg.keep == "best+latest":
            self._prune_except({"latest", "best"})
        return d

    def _update_symlink(self, name: str, step_dir: Path) -> None:
        link = self._dir / name
        try:
            if link.is_symlink() or link.exists():
                link.unlink()
            link.symlink_to(step_dir.name)
        except OSError as exc:
            logger.warning("could not update %r symlink at %s: %s", name, link, exc)

    def _prune_except(self, keep_links: set[str]) -> None:
        """Remove every ``step_*`` dir not pointed to by a kept symlink."""
        keep_targets: set[str] = set()
        for name in keep_links:
            link = self._dir / name
            if link.is_symlink():
                keep_targets.add(Path(link.resolve()).name)
        for child in self._dir.iterdir():
            if child.is_symlink() or not child.is_dir():
                continue
            if not child.name.startswith("step_"):
                continue
            if child.name in keep_targets:
                continue
            shutil.rmtree(child, ignore_errors=True)
