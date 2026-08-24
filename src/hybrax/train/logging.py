"""Per-update console, in-memory, CSV, and JSONL training telemetry."""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import pandas as pd

from .serialization import dumps_json

__all__ = ["StepRecord", "RunLogger"]


@dataclass(frozen=True)
class StepRecord:
    """One training step's loss/timing/diagnostics, as handed to :class:`RunLogger`."""

    step: int
    total_updates: int
    epoch: int
    batch_in_epoch: int
    samples_seen: int
    mean_loss: float
    per_target_loss: tuple[float, ...]
    per_process_loss: tuple[float, ...]
    target_names: tuple[str, ...]
    process_names: tuple[str, ...]  # batch composition for this step
    step_dt: float  # wall time (seconds)
    rebuild_count: int  # cumulative
    # Holdout / validation loss, populated only when a holdout is evaluated at a
    # checkpoint boundary; None on intermediate steps.
    holdout_loss: float | None = None
    holdout_label: str | None = None
    # Set only on the final batch row of each epoch; None on intermediate batches.
    epoch_mean_loss: float | None = None
    epoch_time_seconds: float | None = None
    # Global L2 norm of the gradient pytree at this step (pre-clipping).
    grad_norm: float | None = None
    # Names of processes in this batch whose ODE solve bailed mid-trajectory
    # (post-failure measurement points were dropped from the loss). Empty when the
    # whole batch solved cleanly; its length is the per-step failed-segment count.
    failed_process_names: tuple[str, ...] = ()


class _ConsoleTableFormatter:
    _MIN_NUMERIC_WIDTH = 10

    def __init__(
        self,
        target_names: Sequence[str],
        total_updates: int,
        decimals: int = 4,
        header_every: int = 10,
    ) -> None:
        self._decimals = int(decimals)
        self._header_every = int(header_every)
        self._row_count = 0
        target_widths = [
            max(len(name), self._MIN_NUMERIC_WIDTH) for name in target_names
        ]
        self._columns: list[tuple[str, int]] = [
            ("time", 8),
            ("step", max(len(str(int(total_updates))), 4)),
            ("loss", self._MIN_NUMERIC_WIDTH),
            *list(zip(target_names, target_widths)),
            ("dt", 6),
        ]

    @property
    def columns(self) -> tuple[tuple[str, int], ...]:
        return tuple(self._columns)

    def header_lines(self) -> tuple[str, str]:
        header = "|".join(f" {name:>{width}} " for name, width in self._columns)
        separator = "|".join("-" * (width + 2) for _, width in self._columns)
        return header, separator

    def format_row(
        self,
        *,
        clock: str,
        step: int,
        mean_loss: float,
        per_target_loss: Sequence[float],
        step_dt: float,
    ) -> str:
        if len(per_target_loss) + 4 != len(self._columns):
            raise ValueError(
                f"per_target_loss has {len(per_target_loss)} entries, "
                f"formatter expects {len(self._columns) - 4}"
            )
        cells = [
            f" {clock:>{self._columns[0][1]}} ",
            f" {step:>{self._columns[1][1]}d} ",
            f" {mean_loss:>{self._columns[2][1]}.{self._decimals}f} ",
        ]
        cells.extend(
            f" {float(value):>{width}.{self._decimals}f} "
            for (_, width), value in zip(self._columns[3:-1], per_target_loss)
        )
        cells.append(f" {float(step_dt):>{self._columns[-1][1]}.2f} ")
        return "|".join(cells)

    def emit(self, logger: logging.Logger, *, row: str) -> None:
        if self._header_every and self._row_count % self._header_every == 0:
            for line in self.header_lines():
                logger.info(line)
        logger.info(row)
        self._row_count += 1


def _process_loss_line(record: StepRecord, decimals: int) -> str:
    values = "  ".join(
        f"{name}={float(value):.{decimals}f}"
        for name, value in zip(record.process_names, record.per_process_loss)
    )
    return "            \u21b3 per-process: " + values


def _row(record: StepRecord, *, strings: bool) -> dict[str, Any]:
    def number(value):
        return f"{value:.10g}" if strings else value

    def optional(value):
        return "" if value is None and strings else value

    return {
        "step": record.step,
        "total_updates": record.total_updates,
        "epoch": record.epoch,
        "batch_in_epoch": record.batch_in_epoch,
        "samples_seen": record.samples_seen,
        "mean_loss": number(record.mean_loss),
        "per_target_loss": (
            ";".join(number(v) for v in record.per_target_loss)
            if strings
            else list(record.per_target_loss)
        ),
        "per_process_loss": (
            ";".join(number(v) for v in record.per_process_loss)
            if strings
            else list(record.per_process_loss)
        ),
        "target_names": (
            ";".join(record.target_names) if strings else list(record.target_names)
        ),
        "process_names": (
            ";".join(record.process_names) if strings else list(record.process_names)
        ),
        "step_dt": f"{record.step_dt:.6f}" if strings else record.step_dt,
        "rebuild_count": record.rebuild_count,
        "holdout_loss": (
            number(record.holdout_loss)
            if record.holdout_loss is not None
            else optional(None)
        ),
        "holdout_label": (
            (record.holdout_label or "") if strings else record.holdout_label
        ),
        "epoch_mean_loss": (
            number(record.epoch_mean_loss)
            if record.epoch_mean_loss is not None
            else optional(None)
        ),
        "epoch_time_seconds": (
            f"{record.epoch_time_seconds:.6f}"
            if strings and record.epoch_time_seconds is not None
            else optional(record.epoch_time_seconds)
        ),
        "grad_norm": (
            number(record.grad_norm) if record.grad_norm is not None else optional(None)
        ),
        "n_failed_samples": len(record.failed_process_names),
        "failed_processes": (
            ";".join(record.failed_process_names)
            if strings
            else list(record.failed_process_names)
        ),
    }


class RunLogger:
    """Console table + in-memory history + ``metrics.csv``/``metrics.jsonl`` writer.

    One instance covers one training run: :meth:`start` opens the output files
    and prints the run banner, :meth:`record_step` is called once per training
    step, and :meth:`finalize`/:meth:`close` flush and release the file
    handles. Usable as a context manager (``__enter__``/``__exit__`` call
    :meth:`close`).
    """

    def __init__(
        self,
        *,
        log_process_losses: bool = False,
        metrics_csv: str | Path | None = None,
        metrics_jsonl: str | Path | None = None,
        log_decimals: int = 4,
        logger_name: str = "hybrax.train.harness",
    ) -> None:
        """Configure output destinations; no I/O happens until :meth:`start`.

        Args:
            log_process_losses: Also log a per-process loss breakdown line
                after every step's table row.
            metrics_csv: Path to write ``metrics.csv`` incrementally to, or
                ``None`` to skip CSV output.
            metrics_jsonl: Path to write ``metrics.jsonl`` incrementally to,
                or ``None`` to skip JSONL output.
            log_decimals: Decimal places used when formatting loss/gradient
                values in the console table.
            logger_name: Name of the ``logging.Logger`` to emit console lines
                through.
        """
        self._log_process_losses = bool(log_process_losses)
        self._metrics_csv_path = Path(metrics_csv) if metrics_csv is not None else None
        self._metrics_jsonl_path = (
            Path(metrics_jsonl) if metrics_jsonl is not None else None
        )
        self._decimals = int(log_decimals)
        self._logger = logging.getLogger(logger_name)
        self._formatter: _ConsoleTableFormatter | None = None
        self._jsonl_file = None
        self._history: dict[str, Any] = {
            "mean_loss_by_step": [],
            "per_target_loss_by_step": [],
            "step_time_seconds": [],
            "batch_process_names_by_step": [],
            "per_process_loss_by_step": [],
            "holdout_loss_by_step": {},
            "holdout_label": None,
            "train_step_rebuild_count": 0,
            "grad_norm_by_step": [],
        }
        self._target_names: tuple[str, ...] = ()

    def __enter__(self) -> RunLogger:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def start(
        self,
        *,
        target_names: Sequence[str],
        process_names: Sequence[str],
        total_updates: int,
        compile_warmup_seconds: float,
    ) -> None:
        """Open the CSV/JSONL files (if configured) and log the run banner.

        Args:
            target_names: Loss-term names, in the order every step's
                ``per_target_loss`` is reported in.
            process_names: Every process in the training run, for the banner
                only.
            total_updates: Total optimizer steps the run will take, used to
                size the console table's step column.
            compile_warmup_seconds: JIT warmup time to report in the banner.
        """
        self._target_names = tuple(target_names)
        self._formatter = _ConsoleTableFormatter(
            target_names, total_updates, decimals=self._decimals
        )
        if self._metrics_csv_path is not None:
            self._metrics_csv_path.parent.mkdir(parents=True, exist_ok=True)
            pd.DataFrame(columns=_row(_DUMMY_RECORD, strings=True)).to_csv(
                self._metrics_csv_path, index=False
            )
        if self._metrics_jsonl_path is not None:
            self._metrics_jsonl_path.parent.mkdir(parents=True, exist_ok=True)
            self._jsonl_file = self._metrics_jsonl_path.open("w")
        self._logger.info(
            "run start  targets=[%s]  processes=%d  jit_warmup=%.1fs",
            ", ".join(self._target_names),
            len(process_names),
            compile_warmup_seconds,
        )

    def record_rebuild(self, step_index: int) -> None:
        """Log and count a train-step JIT rebuild at ``step_index``."""
        self._history["train_step_rebuild_count"] += 1
        self._logger.warning(
            "train-step rebuilt at step=%d (rebuild_count=%d)",
            step_index,
            self._history["train_step_rebuild_count"],
        )

    def record_step(self, record: StepRecord) -> None:
        """Log ``record``'s console row, append it to history, and write it to
        ``metrics.csv``/``metrics.jsonl`` (whichever are configured).

        Raises:
            RuntimeError: If called before :meth:`start`.
        """
        if self._formatter is None:
            raise RuntimeError("RunLogger.start() must be called before record_step()")
        self._history["mean_loss_by_step"].append(float(record.mean_loss))
        self._history["per_target_loss_by_step"].append(record.per_target_loss)
        self._history["step_time_seconds"].append(float(record.step_dt))
        self._history["batch_process_names_by_step"].append(record.process_names)
        self._history["per_process_loss_by_step"].append(record.per_process_loss)
        if record.grad_norm is not None:
            self._history["grad_norm_by_step"].append(float(record.grad_norm))
        if record.holdout_loss is not None:
            self._history["holdout_loss_by_step"][record.step] = record.holdout_loss
            self._history["holdout_label"] = record.holdout_label

        # Failed-segment warning: a finite fail_time means a sample's ODE solve
        # bailed mid-trajectory and its post-failure points were dropped from the
        # loss. Surface it every time it happens so the rate is visible.
        if record.failed_process_names:
            self._logger.warning(
                "step %d: %d/%d samples hit a failed ODE segment: [%s]",
                record.step,
                len(record.failed_process_names),
                len(record.process_names),
                ", ".join(record.failed_process_names),
            )

        self._formatter.emit(
            self._logger,
            row=self._formatter.format_row(
                clock=time.strftime("%H:%M:%S"),
                step=record.step,
                mean_loss=record.mean_loss,
                per_target_loss=record.per_target_loss,
                step_dt=record.step_dt,
            ),
        )
        if self._log_process_losses:
            self._logger.info(_process_loss_line(record, self._decimals))
        if record.holdout_loss is not None:
            self._logger.info(
                "            \u21b3 holdout (%s): %.*f",
                record.holdout_label or "holdout",
                self._decimals,
                record.holdout_loss,
            )
        if record.epoch_mean_loss is not None:
            self._logger.info(
                "epoch %d complete: mean_loss=%.*f duration=%.2fs",
                record.epoch,
                self._decimals,
                record.epoch_mean_loss,
                record.epoch_time_seconds,
            )
        if self._metrics_csv_path is not None:
            pd.DataFrame([_row(record, strings=True)]).to_csv(
                self._metrics_csv_path, mode="a", header=False, index=False
            )
        if self._jsonl_file is not None:
            self._jsonl_file.write(dumps_json(_row(record, strings=False)) + "\n")

    def finalize(self) -> dict[str, Any]:
        """Flush the JSONL file and return the run's full in-memory history.

        Returns:
            Every ``_history`` series (loss, timing, per-process/per-target
            breakdowns, holdout results) plus ``target_names``.
        """
        if self._jsonl_file is not None:
            self._jsonl_file.flush()
        return self.snapshot()

    def snapshot(self) -> dict[str, Any]:
        """Return a snapshot of the history recorded so far."""
        history = {
            key: tuple(value) if isinstance(value, list) else value
            for key, value in self._history.items()
        }
        history["holdout_loss_by_step"] = dict(history["holdout_loss_by_step"])
        history["target_names"] = self._target_names
        return history

    def close(self) -> None:
        """Close the JSONL file handle, if one is open."""
        if self._jsonl_file is not None:
            self._jsonl_file.close()
            self._jsonl_file = None


_DUMMY_RECORD = StepRecord(
    step=0,
    total_updates=0,
    epoch=0,
    batch_in_epoch=0,
    samples_seen=0,
    mean_loss=0.0,
    per_target_loss=(),
    per_process_loss=(),
    target_names=(),
    process_names=(),
    step_dt=0.0,
    rebuild_count=0,
)
