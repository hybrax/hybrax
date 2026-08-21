from __future__ import annotations

import json
import logging
from dataclasses import replace

import pandas as pd
import pytest

from hybrax.train.logging import RunLogger, StepRecord, _ConsoleTableFormatter


def _record(
    step: int,
    *,
    epoch_end: bool = False,
    holdout: float | None = None,
    failed_process_names: tuple[str, ...] = (),
):
    return StepRecord(
        step=step,
        total_updates=4,
        epoch=(step - 1) // 2 + 1,
        batch_in_epoch=(step - 1) % 2 + 1,
        samples_seen=step * 2,
        mean_loss=float(step),
        per_target_loss=(float(step),),
        per_process_loss=(float(step), float(step)),
        target_names=("biomass",),
        process_names=("p1", "p2"),
        step_dt=0.1,
        rebuild_count=0,
        holdout_loss=holdout,
        holdout_label="holdout" if holdout is not None else None,
        epoch_mean_loss=float(step) - 0.5 if epoch_end else None,
        epoch_time_seconds=0.2 if epoch_end else None,
        grad_norm=0.3,
        failed_process_names=failed_process_names,
    )


def test_formatter_alignment_and_target_count_validation():
    formatter = _ConsoleTableFormatter(("biomass",), total_updates=20)
    header, separator = formatter.header_lines()
    row = formatter.format_row(
        clock="12:00:00",
        step=1,
        mean_loss=float("inf"),
        per_target_loss=(float("nan"),),
        step_dt=0.1,
    )
    assert [i for i, value in enumerate(header) if value == "|"] == [
        i for i, value in enumerate(row) if value == "|"
    ]
    assert len(separator) == len(header)
    with pytest.raises(ValueError, match="formatter expects"):
        formatter.format_row(
            clock="12:00:00",
            step=1,
            mean_loss=1.0,
            per_target_loss=(),
            step_dt=0.1,
        )


def test_runlogger_persists_batch_and_epoch_fields(tmp_path, caplog):
    caplog.set_level(logging.INFO, logger="hybrax.train.harness")
    csv_path = tmp_path / "metrics.csv"
    jsonl_path = tmp_path / "metrics.jsonl"
    with RunLogger(metrics_csv=csv_path, metrics_jsonl=jsonl_path) as run:
        run.start(
            target_names=("biomass",),
            process_names=("p1", "p2"),
            total_updates=4,
            compile_warmup_seconds=0.0,
        )
        run.record_step(_record(1))
        run.record_step(_record(2, epoch_end=True, holdout=0.25))
        history = run.finalize()

    rows = pd.read_csv(csv_path)
    assert rows[["step", "epoch", "batch_in_epoch", "samples_seen"]].to_dict(
        orient="records"
    ) == [
        {"step": 1, "epoch": 1, "batch_in_epoch": 1, "samples_seen": 2},
        {"step": 2, "epoch": 1, "batch_in_epoch": 2, "samples_seen": 4},
    ]
    assert pd.isna(rows.loc[0, "epoch_mean_loss"])
    assert rows.loc[1, "epoch_mean_loss"] == pytest.approx(1.5)
    assert rows.loc[1, "epoch_time_seconds"] == pytest.approx(0.2)
    assert pd.isna(rows.loc[0, "holdout_loss"])
    assert rows.loc[1, "holdout_loss"] == pytest.approx(0.25)
    assert history["holdout_loss_by_step"] == {2: 0.25}
    assert any("epoch 1 complete" in record.message for record in caplog.records)
    assert json.loads(jsonl_path.read_text().splitlines()[1])["samples_seen"] == 4


def test_runlogger_csv_pins_exact_metrics_schema(tmp_path):
    csv_path = tmp_path / "metrics.csv"
    with RunLogger(metrics_csv=csv_path) as run:
        run.start(
            target_names=("biomass",),
            process_names=("p1", "p2"),
            total_updates=1,
            compile_warmup_seconds=0.0,
        )
        run.record_step(_record(1))

    rows = pd.read_csv(csv_path)
    assert rows.columns.tolist() == [
        "step",
        "total_updates",
        "epoch",
        "batch_in_epoch",
        "samples_seen",
        "mean_loss",
        "per_target_loss",
        "per_process_loss",
        "target_names",
        "process_names",
        "step_dt",
        "rebuild_count",
        "holdout_loss",
        "holdout_label",
        "epoch_mean_loss",
        "epoch_time_seconds",
        "grad_norm",
        "n_failed_samples",
        "failed_processes",
    ]


def test_runlogger_writes_every_batch_and_truncates_existing_csv(tmp_path):
    path = tmp_path / "metrics.csv"
    path.write_text("old,data\n1,2\n")
    with RunLogger(metrics_csv=path) as run:
        run.start(
            target_names=("biomass",),
            process_names=("p1", "p2"),
            total_updates=3,
            compile_warmup_seconds=0.0,
        )
        for step in range(1, 4):
            run.record_step(_record(step))
    assert len(pd.read_csv(path)) == 3


def test_runlogger_reports_failed_segments(caplog):
    """A step whose batch had a bailing ODE solve emits a warning naming the failed
    processes with the per-step k/B count; clean steps emit no warning."""
    with RunLogger() as run:
        run.start(
            target_names=("biomass",),
            process_names=("p1", "p2"),
            total_updates=3,
            compile_warmup_seconds=0.0,
        )
        with caplog.at_level(logging.WARNING):
            run.record_step(_record(1))
            run.record_step(_record(2, failed_process_names=("p2",)))
            run.record_step(_record(3, failed_process_names=("p1", "p2")))
        run.finalize()
    warnings = [r.getMessage() for r in caplog.records if r.levelno == logging.WARNING]
    fail_warnings = [w for w in warnings if "failed ODE segment" in w]
    assert len(fail_warnings) == 2  # only the two steps with failures
    assert "p2" in fail_warnings[0] and "1/2" in fail_warnings[0]
    assert "2/2" in fail_warnings[1]


def test_runlogger_csv_and_jsonl_carry_failed_segment_columns(tmp_path):
    """The failed-segment count + names surface in both file sinks so the failure
    rate is analyzable after the run."""
    csv_path = tmp_path / "metrics.csv"
    jsonl_path = tmp_path / "metrics.jsonl"
    with RunLogger(metrics_csv=csv_path, metrics_jsonl=jsonl_path) as run:
        run.start(
            target_names=("biomass",),
            process_names=("p1", "p2"),
            total_updates=2,
            compile_warmup_seconds=0.0,
        )
        run.record_step(_record(1))
        run.record_step(_record(2, failed_process_names=("p2",)))
        run.finalize()

    df = pd.read_csv(csv_path)
    assert list(df["n_failed_samples"]) == [0, 1]
    # Empty on the clean step; the failed process name on the second. CSV joins
    # names into a scalar string, so the clean row reads back as "" (NaN in pandas).
    failed_col = df["failed_processes"].fillna("").astype(str).tolist()
    assert failed_col[0] in ("", "nan")
    assert failed_col[1] == "p2"

    # JSONL keeps failed_processes as a list, matching every other sequence column.
    records = [json.loads(x) for x in jsonl_path.read_text().strip().splitlines()]
    assert records[0]["n_failed_samples"] == 0
    assert records[0]["failed_processes"] == []
    assert records[1]["n_failed_samples"] == 1
    assert records[1]["failed_processes"] == ["p2"]


def test_runlogger_jsonl_normalizes_nonfinite_metrics(tmp_path):
    path = tmp_path / "metrics.jsonl"
    record = replace(
        _record(1),
        mean_loss=float("inf"),
        per_target_loss=(float("nan"),),
        per_process_loss=(-float("inf"),),
        process_names=("p1",),
    )
    with RunLogger(metrics_jsonl=path) as run:
        run.start(
            target_names=("biomass",),
            process_names=("p1",),
            total_updates=1,
            compile_warmup_seconds=0.0,
        )
        run.record_step(record)

    text = path.read_text(encoding="utf-8")
    assert "Infinity" not in text and "NaN" not in text
    row = json.loads(text)
    assert row["mean_loss"] is None
    assert row["per_target_loss"] == [None]
    assert row["per_process_loss"] == [None]


def test_runlogger_close_is_idempotent(tmp_path):
    csv_path = tmp_path / "metrics.csv"
    run = RunLogger(metrics_csv=csv_path)
    run.start(
        target_names=("biomass",),
        process_names=("p1", "p2"),
        total_updates=1,
        compile_warmup_seconds=0.0,
    )
    run.record_step(_record(1))
    run.finalize()
    run.close()
    run.close()  # a second close must be a no-op, not raise


def test_runlogger_record_rebuild_and_start_guard(caplog):
    run = RunLogger()
    with pytest.raises(RuntimeError, match="start"):
        run.record_step(_record(1))
    caplog.set_level(logging.WARNING, logger="hybrax.train.harness")
    with run:
        run.start(
            target_names=("biomass",),
            process_names=("p1",),
            total_updates=1,
            compile_warmup_seconds=0.0,
        )
        run.record_rebuild(1)
        assert run.finalize()["train_step_rebuild_count"] == 1
