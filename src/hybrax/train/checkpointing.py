"""Periodic training checkpoints: model snapshot + sidecar metadata + loss curve."""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

from .postprocessing import plot_loss_curve, save_model, save_model_metadata

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class CheckpointConfig:
    """Configuration for periodic checkpoints during training.

    ``every <= 0`` disables checkpointing entirely.
    """

    output_dir: Path
    every: int


class CheckpointWriter:
    """Write wrapper snapshot + sidecar + loss curve at a fixed step cadence."""

    def __init__(self, cfg: CheckpointConfig) -> None:
        self._cfg = cfg
        self._enabled = int(cfg.every) > 0
        if self._enabled:
            cfg.output_dir.mkdir(parents=True, exist_ok=True)
            logger.info(
                "checkpointing enabled: dir=%s every=%d steps",
                cfg.output_dir,
                int(cfg.every),
            )

    @property
    def enabled(self) -> bool:
        return self._enabled

    def maybe_write(
        self,
        *,
        step: int,
        wrapper: Any,
        mean_loss_by_step: Sequence[float],
        per_target_loss_by_step: Sequence[tuple[float, ...]] | None = None,
        target_names: Sequence[str] | None = None,
        monitor_loss_by_step: dict[int, float] | None = None,
        monitor_label: str | None = None,
    ) -> Path | None:
        """Write a checkpoint if ``step`` is a multiple of ``every``."""
        if not self._enabled:
            return None
        if step <= 0 or step % int(self._cfg.every) != 0:
            return None
        return self._write(
            step=step,
            wrapper=wrapper,
            mean_loss_by_step=mean_loss_by_step,
            per_target_loss_by_step=per_target_loss_by_step,
            target_names=target_names,
            monitor_loss_by_step=monitor_loss_by_step,
            monitor_label=monitor_label,
        )

    def publish_latest(self, step_dir: Path) -> None:
        """Publish ``step_dir`` as the latest complete checkpoint.

        Call this only after all per-step artifacts have been written.
        """
        self._update_latest_symlink(step_dir)

    def _write(
        self,
        *,
        step: int,
        wrapper: Any,
        mean_loss_by_step: Sequence[float],
        per_target_loss_by_step: Sequence[tuple[float, ...]] | None = None,
        target_names: Sequence[str] | None = None,
        monitor_loss_by_step: dict[int, float] | None = None,
        monitor_label: str | None = None,
    ) -> Path:
        step_dir = self._cfg.output_dir / f"step_{step:05d}"
        step_dir.mkdir(parents=True, exist_ok=True)

        save_model(wrapper, step_dir / "trained_wrapper.eqx")

        latest_loss = (
            float(mean_loss_by_step[-1]) if len(mean_loss_by_step) else float("nan")
        )
        save_model_metadata(
            step_dir / "trained_wrapper.meta.json",
            {
                "step": int(step),
                "mean_loss": latest_loss,
                "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z", time.localtime()),
            },
        )

        plot_loss_curve(
            list(mean_loss_by_step),
            step_dir / "loss_curve.png",
            title=f"Training loss (through step {step})",
            per_target_loss_by_step=(
                list(per_target_loss_by_step) if per_target_loss_by_step else None
            ),
            target_names=tuple(target_names) if target_names else None,
            monitor_loss_by_step=dict(monitor_loss_by_step) if monitor_loss_by_step else None,
            monitor_label=monitor_label,
        )

        return step_dir

    def _update_latest_symlink(self, step_dir: Path) -> None:
        latest = self._cfg.output_dir / "latest"
        try:
            if latest.is_symlink() or latest.exists():
                latest.unlink()
            latest.symlink_to(step_dir.name)
        except OSError as exc:
            logger.warning("could not update 'latest' symlink at %s: %s", latest, exc)
