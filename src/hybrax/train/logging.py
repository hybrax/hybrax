"""Per-run telemetry logger for the bp_train training harness.

This module owns three concerns that used to be tangled inside the harness
training loop:

1. Console output (a fixed-width tabular format with periodic header
   re-emission).
2. In-memory history (the per-step lists that the harness exposes via
   ``TrainHarnessResult``).
3. Optional structured persistence (CSV / JSONL) for downstream analysis.

The training loop only needs to construct one ``StepRecord`` per step and
hand it to ``RunLogger.record_step``.  Everything else — formatting,
file I/O, history bookkeeping — is owned by ``RunLogger``.

No JAX dependency: all values arriving in a ``StepRecord`` are plain Python
``float`` / ``tuple`` / ``str``.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import pandas as pd


__all__ = ["StepRecord", "RunLogger"]


@dataclass(frozen=True)
class StepRecord:
    """Immutable per-step telemetry record.

    Mirrors the spec section 16.6 (Batch Telemetry Contract) one-row-per-step
    fields.  All numeric values are plain Python ``float`` / ``int`` (the
    harness converts them out of JAX arrays before constructing the record).
    """

    step: int  # 1-indexed
    total_steps: int
    mean_loss: float
    per_target_loss: tuple[float, ...]  # length == n_targets
    per_process_loss: tuple[float, ...]  # length == batch_size
    target_names: tuple[str, ...]
    process_names: tuple[str, ...]  # batch composition for this step
    step_dt: float  # wall time (seconds)
    rebuild_count: int  # cumulative
    # Optional monitor / validation-loss column (see TrainHarnessConfig.monitor_processes).
    # Populated only at log-step cadence; None on intermediate steps.
    monitor_loss: float | None = None
    monitor_label: str | None = None


class _ConsoleTableFormatter:
    """Build the fixed-width tabular console output.

    Layout (one row per step):

        | time | step | loss | tgt_1 | tgt_2 | ... | dt |

    Decimal places are fixed (default 4) so the decimal points line up across
    rows.  The header is two lines (column names + separator) and is
    re-emitted every ``header_every`` rows so a long-running scroll stays
    readable.
    """

    _MIN_NUMERIC_WIDTH = 10  # one negative sign + digit + '.' + 4 decimals + slack

    def __init__(
        self,
        target_names: Sequence[str],
        total_steps: int,
        decimals: int = 4,
        header_every: int = 30,
    ) -> None:
        self._decimals = int(decimals)
        self._header_every = int(header_every)
        self._row_count = 0

        time_width = 8  # HH:MM:SS
        dt_width = 6  # 9999.99 max
        step_width = max(len(str(int(total_steps))), 4)
        loss_width = max(len("loss"), self._MIN_NUMERIC_WIDTH)
        target_widths = [
            max(len(name), self._MIN_NUMERIC_WIDTH) for name in target_names
        ]

        # Each entry is (column_name, width).  The width is the inner content
        # width (not counting the surrounding spaces or `|` separators).
        self._columns: list[tuple[str, int]] = [
            ("time", time_width),
            ("step", step_width),
            ("loss", loss_width),
            *list(zip(target_names, target_widths)),
            ("dt", dt_width),
        ]

    @property
    def columns(self) -> tuple[tuple[str, int], ...]:
        return tuple(self._columns)

    def header_lines(self) -> tuple[str, str]:
        cells = [f" {name:>{w}} " for name, w in self._columns]
        header = "|".join(cells)
        sep_cells = ["-" * (w + 2) for _, w in self._columns]
        sep = "|".join(sep_cells)
        return header, sep

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
        cells: list[str] = []
        cells.append(f" {clock:>{self._columns[0][1]}} ")
        cells.append(f" {step:>{self._columns[1][1]}d} ")
        cells.append(f" {mean_loss:>{self._columns[2][1]}.{self._decimals}f} ")
        for (_, w), v in zip(self._columns[3:-1], per_target_loss):
            cells.append(f" {float(v):>{w}.{self._decimals}f} ")
        cells.append(f" {float(step_dt):>{self._columns[-1][1]}.2f} ")
        return "|".join(cells)

    def emit(self, logger: logging.Logger, *, row: str) -> None:
        if self._header_every and self._row_count % self._header_every == 0:
            header, sep = self.header_lines()
            logger.info(header)
            logger.info(sep)
        logger.info(row)
        self._row_count += 1


def _log_step_indent_line(
    record: StepRecord,
    process_loss_decimals: int,
) -> str:
    """Indented one-line summary of per-process losses for a log step."""
    cells = [
        f"{name}={float(val):.{process_loss_decimals}f}"
        for name, val in zip(record.process_names, record.per_process_loss)
    ]
    return "            \u21b3 per-process: " + "  ".join(cells)


def _csv_row_dict(record: StepRecord) -> dict[str, Any]:
    """Flatten a StepRecord into a single CSV-friendly row dict.

    Vector fields are joined as semicolon-separated strings so the file stays
    valid CSV regardless of batch_size.
    """
    return {
        "step": record.step,
        "total_steps": record.total_steps,
        "mean_loss": f"{record.mean_loss:.10g}",
        "per_target_loss": ";".join(f"{v:.10g}" for v in record.per_target_loss),
        "per_process_loss": ";".join(f"{v:.10g}" for v in record.per_process_loss),
        "target_names": ";".join(record.target_names),
        "process_names": ";".join(record.process_names),
        "step_dt": f"{record.step_dt:.6f}",
        "rebuild_count": record.rebuild_count,
        "monitor_loss": (
            f"{record.monitor_loss:.10g}" if record.monitor_loss is not None else ""
        ),
        "monitor_label": record.monitor_label or "",
    }


def _jsonl_row_dict(record: StepRecord) -> dict[str, Any]:
    """Convert a StepRecord into a JSON-serialisable dict."""
    return {
        "step": record.step,
        "total_steps": record.total_steps,
        "mean_loss": record.mean_loss,
        "per_target_loss": list(record.per_target_loss),
        "per_process_loss": list(record.per_process_loss),
        "target_names": list(record.target_names),
        "process_names": list(record.process_names),
        "step_dt": record.step_dt,
        "rebuild_count": record.rebuild_count,
        "monitor_loss": record.monitor_loss,
        "monitor_label": record.monitor_label,
    }


class RunLogger:
    """Per-run telemetry sink: console + in-memory history + optional files.

    Lifecycle:

        with RunLogger(log_every=20, metrics_csv="run.csv") as run_log:
            run_log.start(
                target_names=...,
                process_names=...,
                total_steps=...,
                compile_warmup_seconds=...,
            )
            for step in ...:
                run_log.record_step(StepRecord(...))
            history = run_log.finalize()

    The ``history`` dict can be unpacked into the ``TrainHarnessResult``
    history fields directly.
    """

    def __init__(
        self,
        *,
        log_every: int,
        log_process_losses: bool = False,
        metrics_csv: str | Path | None = None,
        metrics_jsonl: str | Path | None = None,
        log_decimals: int = 4,
        log_header_every: int = 30,
        logger_name: str = "bp_train.harness",
    ) -> None:
        self._log_every = max(int(log_every), 1)
        self._log_process_losses = bool(log_process_losses)
        self._metrics_csv_path = Path(metrics_csv) if metrics_csv is not None else None
        self._metrics_jsonl_path = (
            Path(metrics_jsonl) if metrics_jsonl is not None else None
        )
        self._log_decimals = int(log_decimals)
        self._log_header_every = int(log_header_every)
        self._logger = logging.getLogger(logger_name)

        self._formatter: _ConsoleTableFormatter | None = None
        self._csv_header_written = False
        self._jsonl_file = None

        self._mean_loss_by_step: list[float] = []
        self._step_time_seconds: list[float] = []
        self._batch_process_names_by_step: list[tuple[str, ...]] = []
        self._per_process_loss_by_step: list[tuple[float, ...]] = []
        self._sampled_loss_by_process_at_log_steps: dict[
            int, tuple[tuple[str, float], ...]
        ] = {}
        self._monitor_loss_by_log_step: dict[int, float] = {}
        self._monitor_label: str | None = None
        self._rebuild_count: int = 0

        self._target_names: tuple[str, ...] = ()
        self._total_steps: int = 0
        self._closed = False

    # ----- context manager -----

    def __enter__(self) -> "RunLogger":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    # ----- lifecycle -----

    def start(
        self,
        *,
        target_names: Sequence[str],
        process_names: Sequence[str],
        total_steps: int,
        compile_warmup_seconds: float,
    ) -> None:
        self._target_names = tuple(target_names)
        self._total_steps = int(total_steps)

        self._formatter = _ConsoleTableFormatter(
            target_names=self._target_names,
            total_steps=self._total_steps,
            decimals=self._log_decimals,
            header_every=self._log_header_every,
        )

        if self._metrics_csv_path is not None:
            self._metrics_csv_path.parent.mkdir(parents=True, exist_ok=True)
            fieldnames = list(_csv_row_dict(_DUMMY_RECORD).keys())
            pd.DataFrame(columns=fieldnames).to_csv(self._metrics_csv_path, index=False)
            self._csv_header_written = True

        if self._metrics_jsonl_path is not None:
            self._metrics_jsonl_path.parent.mkdir(parents=True, exist_ok=True)
            self._jsonl_file = self._metrics_jsonl_path.open("w")

        self._logger.info(
            "run start  targets=[%s]  processes=%d  jit_warmup=%.1fs",
            ", ".join(self._target_names),
            len(process_names),
            float(compile_warmup_seconds),
        )

    def record_rebuild(self, step_index: int) -> None:
        self._rebuild_count += 1
        self._logger.warning(
            "train-step rebuilt at step=%d (rebuild_count=%d)",
            int(step_index),
            self._rebuild_count,
        )

    def record_step(self, record: StepRecord) -> None:
        if self._formatter is None:
            raise RuntimeError("RunLogger.start() must be called before record_step()")

        # In-memory history
        self._mean_loss_by_step.append(float(record.mean_loss))
        self._step_time_seconds.append(float(record.step_dt))
        self._batch_process_names_by_step.append(tuple(record.process_names))
        self._per_process_loss_by_step.append(tuple(record.per_process_loss))

        # Console row
        clock = time.strftime("%H:%M:%S")
        row = self._formatter.format_row(
            clock=clock,
            step=record.step,
            mean_loss=record.mean_loss,
            per_target_loss=record.per_target_loss,
            step_dt=record.step_dt,
        )
        self._formatter.emit(self._logger, row=row)

        # Monitor (validation) loss column — populated only on log steps.
        if record.monitor_loss is not None:
            self._monitor_loss_by_log_step[record.step] = float(record.monitor_loss)
            if record.monitor_label is not None:
                self._monitor_label = record.monitor_label

        # Per-process indented line at log-step cadence
        is_log_step = record.step % self._log_every == 0
        if is_log_step:
            self._sampled_loss_by_process_at_log_steps[record.step] = tuple(
                (name, float(val))
                for name, val in zip(record.process_names, record.per_process_loss)
            )
            # Always show the indented per-process line at log steps;
            # `--log-process-losses` is reserved for "show every step".
            self._logger.info(_log_step_indent_line(record, self._log_decimals))
            if record.monitor_loss is not None:
                self._logger.info(
                    "            \u21b3 monitor (%s): %.*f",
                    record.monitor_label or "validation",
                    self._log_decimals,
                    float(record.monitor_loss),
                )
        elif self._log_process_losses:
            # If the user opted in to per-step per-process losses, also emit
            # the indented line on non-log-steps.
            self._logger.info(_log_step_indent_line(record, self._log_decimals))

        # File sinks
        if self._metrics_csv_path is not None:
            pd.DataFrame([_csv_row_dict(record)]).to_csv(
                self._metrics_csv_path,
                mode="a",
                header=not self._csv_header_written,
                index=False,
            )
            self._csv_header_written = True
        if self._jsonl_file is not None:
            self._jsonl_file.write(json.dumps(_jsonl_row_dict(record)) + "\n")

    def finalize(self) -> dict[str, Any]:
        """Flush file sinks and return the history dict for TrainHarnessResult."""
        if self._jsonl_file is not None:
            self._jsonl_file.flush()
        return {
            "mean_loss_by_step": tuple(self._mean_loss_by_step),
            "step_time_seconds": tuple(self._step_time_seconds),
            "batch_process_names_by_step": tuple(self._batch_process_names_by_step),
            "per_process_loss_by_step": tuple(self._per_process_loss_by_step),
            "sampled_loss_by_process_at_log_steps": dict(
                self._sampled_loss_by_process_at_log_steps
            ),
            "monitor_loss_by_log_step": dict(self._monitor_loss_by_log_step),
            "monitor_label": self._monitor_label,
            "train_step_rebuild_count": int(self._rebuild_count),
        }

    def close(self) -> None:
        if self._closed:
            return
        if self._jsonl_file is not None:
            try:
                self._jsonl_file.close()
            except Exception:
                pass
            self._jsonl_file = None
        self._csv_header_written = False
        self._closed = True


# A throwaway record only used to derive the CSV column names at start().
_DUMMY_RECORD = StepRecord(
    step=0,
    total_steps=0,
    mean_loss=0.0,
    per_target_loss=(),
    per_process_loss=(),
    target_names=(),
    process_names=(),
    step_dt=0.0,
    rebuild_count=0,
)
