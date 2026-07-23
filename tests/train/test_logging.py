"""Unit tests for bp_train.logging (RunLogger, formatter, file sinks)."""

from __future__ import annotations

import json
import logging
import re

import pandas as pd
import pytest

from bp_train.logging import (
    RunLogger,
    StepRecord,
    _ConsoleTableFormatter,
    _csv_row_dict,
    _DUMMY_RECORD,
)


def _make_record(
    step: int,
    *,
    total_steps: int = 4,
    target_names: tuple[str, ...] = ("biomass", "glycerol"),
    process_names: tuple[str, ...] = ("p1", "p2"),
    mean_loss: float = 1.5,
    per_target: tuple[float, ...] = (1.0, 2.0),
    per_process: tuple[float, ...] = (1.2, 1.8),
    step_dt: float = 0.05,
    rebuild_count: int = 0,
    failed_process_names: tuple[str, ...] = (),
) -> StepRecord:
    return StepRecord(
        step=step,
        total_steps=total_steps,
        mean_loss=mean_loss,
        per_target_loss=per_target,
        per_process_loss=per_process,
        target_names=target_names,
        process_names=process_names,
        step_dt=step_dt,
        rebuild_count=rebuild_count,
        failed_process_names=failed_process_names,
    )


# ----- _ConsoleTableFormatter -----


def test_formatter_header_and_row_have_matching_pipe_positions():
    fmt = _ConsoleTableFormatter(
        target_names=("biomass", "glycerol"),
        total_steps=200,
        decimals=4,
        header_every=30,
    )
    header, sep = fmt.header_lines()
    row = fmt.format_row(
        clock="12:00:00",
        step=1,
        mean_loss=0.5,
        per_target_loss=(0.5, 0.4),
        step_dt=0.1,
    )
    # Pipe positions must be identical between header / separator / row.
    pipes_header = [i for i, c in enumerate(header) if c == "|"]
    pipes_row = [i for i, c in enumerate(row) if c == "|"]
    assert pipes_header == pipes_row
    # Separator row is the same width as the header.
    assert len(sep) == len(header)


def test_formatter_handles_inf_and_nan_without_breaking_alignment():
    fmt = _ConsoleTableFormatter(
        target_names=("biomass",),
        total_steps=10,
        decimals=4,
    )
    row_normal = fmt.format_row(
        clock="00:00:00",
        step=1,
        mean_loss=1.0,
        per_target_loss=(1.0,),
        step_dt=0.1,
    )
    row_inf = fmt.format_row(
        clock="00:00:00",
        step=2,
        mean_loss=float("inf"),
        per_target_loss=(float("nan"),),
        step_dt=0.1,
    )
    pipes_a = [i for i, c in enumerate(row_normal) if c == "|"]
    pipes_b = [i for i, c in enumerate(row_inf) if c == "|"]
    assert pipes_a == pipes_b


def test_formatter_rejects_per_target_length_mismatch():
    fmt = _ConsoleTableFormatter(
        target_names=("a", "b"),
        total_steps=10,
    )
    with pytest.raises(ValueError):
        fmt.format_row(
            clock="00:00:00",
            step=1,
            mean_loss=0.0,
            per_target_loss=(1.0,),  # only one value, expects two
            step_dt=0.1,
        )


# ----- RunLogger lifecycle / history -----


def test_runlogger_history_matches_input_stream(caplog):
    caplog.set_level(logging.INFO, logger="bp_train.harness")
    with RunLogger(log_every=2, log_header_every=0) as run:
        run.start(
            target_names=("biomass", "glycerol"),
            process_names=("p1", "p2"),
            total_steps=4,
            compile_warmup_seconds=1.5,
        )
        for i in range(1, 5):
            run.record_step(_make_record(i, mean_loss=float(i)))
        history = run.finalize()
    assert history["mean_loss_by_step"] == (1.0, 2.0, 3.0, 4.0)
    assert len(history["step_time_seconds"]) == 4
    assert len(history["batch_process_names_by_step"]) == 4
    assert len(history["per_process_loss_by_step"]) == 4
    # Sampled per-process losses populated only at log-step cadence.
    assert set(history["sampled_loss_by_process_at_log_steps"].keys()) == {2, 4}
    assert history["train_step_rebuild_count"] == 0


def test_runlogger_console_emits_one_row_per_step_plus_indented_log_step(caplog):
    caplog.set_level(logging.INFO, logger="bp_train.harness")
    with RunLogger(log_every=2, log_header_every=0) as run:
        run.start(
            target_names=("biomass",),
            process_names=("p1", "p2"),
            total_steps=3,
            compile_warmup_seconds=0.0,
        )
        for i in range(1, 4):
            run.record_step(
                _make_record(
                    i,
                    total_steps=3,
                    target_names=("biomass",),
                    per_target=(0.5,),
                )
            )
        run.finalize()
    row_re = re.compile(r"^\s\d{2}:\d{2}:\d{2}\s\|")
    rows = [r.message for r in caplog.records if row_re.match(r.message)]
    indented = [r.message for r in caplog.records if "per-process:" in r.message]
    assert len(rows) == 3
    assert len(indented) == 1  # only step 2 is a log-step (3 not divisible by 2)


def test_runlogger_header_reemitted_every_n_rows(caplog):
    caplog.set_level(logging.INFO, logger="bp_train.harness")
    with RunLogger(log_every=100, log_header_every=2) as run:
        run.start(
            target_names=("biomass",),
            process_names=("p1",),
            total_steps=5,
            compile_warmup_seconds=0.0,
        )
        for i in range(1, 6):
            run.record_step(
                _make_record(
                    i,
                    total_steps=5,
                    target_names=("biomass",),
                    process_names=("p1",),
                    per_target=(0.5,),
                    per_process=(0.5,),
                )
            )
        run.finalize()
    # Header has the literal column name "biomass" embedded.
    headers = [
        r.message
        for r in caplog.records
        if "biomass" in r.message and "|" in r.message and "↳" not in r.message
    ]
    # Filter out the data rows: data rows have the clock prefix.
    header_only = [m for m in headers if not re.match(r"^\s\d{2}:", m)]
    # 5 rows with header_every=2 → emitted before rows 1, 3, 5 → 3 headers.
    assert len(header_only) == 3


def test_runlogger_record_rebuild_logs_warning_and_increments(caplog):
    caplog.set_level(logging.WARNING, logger="bp_train.harness")
    with RunLogger(log_every=10, log_header_every=0) as run:
        run.start(
            target_names=("biomass",),
            process_names=("p1",),
            total_steps=2,
            compile_warmup_seconds=0.0,
        )
        run.record_rebuild(1)
        run.record_rebuild(2)
        run.record_step(
            _make_record(
                1,
                total_steps=2,
                target_names=("biomass",),
                process_names=("p1",),
                per_target=(0.5,),
                per_process=(0.5,),
                rebuild_count=2,
            )
        )
        run.record_step(
            _make_record(
                2,
                total_steps=2,
                target_names=("biomass",),
                process_names=("p1",),
                per_target=(0.5,),
                per_process=(0.5,),
                rebuild_count=2,
            )
        )
        history = run.finalize()
    warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
    assert len(warnings) == 2
    assert history["train_step_rebuild_count"] == 2


def test_runlogger_record_step_before_start_raises():
    run = RunLogger(log_every=1)
    with pytest.raises(RuntimeError):
        run.record_step(_make_record(1))


# ----- file sinks -----


def test_runlogger_csv_sink_has_header_and_one_row_per_step(tmp_path):
    csv_path = tmp_path / "metrics.csv"
    with RunLogger(log_every=2, metrics_csv=csv_path, log_header_every=0) as run:
        run.start(
            target_names=("biomass",),
            process_names=("p1", "p2"),
            total_steps=3,
            compile_warmup_seconds=0.0,
        )
        for i in range(1, 4):
            run.record_step(
                _make_record(
                    i,
                    total_steps=3,
                    target_names=("biomass",),
                    per_target=(0.5,),
                )
            )
        run.finalize()
    rows = pd.read_csv(csv_path).to_dict(orient="records")
    assert len(rows) == 3
    assert {"step", "mean_loss", "per_target_loss", "per_process_loss"} <= set(
        rows[0].keys()
    )
    # Vector fields are semicolon-joined.
    assert ";" in rows[0]["per_process_loss"]


def test_runlogger_csv_sink_truncates_existing_file_on_start(tmp_path):
    csv_path = tmp_path / "metrics.csv"
    csv_path.write_text("legacy,data\nx,y\n")
    with RunLogger(log_every=1, metrics_csv=csv_path, log_header_every=0) as run:
        run.start(
            target_names=("biomass",),
            process_names=("p1",),
            total_steps=1,
            compile_warmup_seconds=0.0,
        )
        run.record_step(
            _make_record(
                1,
                total_steps=1,
                target_names=("biomass",),
                process_names=("p1",),
                per_target=(0.5,),
                per_process=(0.5,),
            )
        )
        run.finalize()
    rows = pd.read_csv(csv_path)
    assert len(rows) == 1
    assert rows.columns.tolist()[0] == "step"


def test_runlogger_csv_sink_writes_header_for_zero_step_run(tmp_path):
    csv_path = tmp_path / "metrics.csv"
    with RunLogger(log_every=1, metrics_csv=csv_path, log_header_every=0) as run:
        run.start(
            target_names=("biomass",),
            process_names=("p1",),
            total_steps=1,
            compile_warmup_seconds=0.0,
        )
        run.finalize()
    rows = pd.read_csv(csv_path)
    assert rows.columns.tolist() == [
        "step",
        "total_steps",
        "mean_loss",
        "per_target_loss",
        "per_process_loss",
        "target_names",
        "process_names",
        "step_dt",
        "rebuild_count",
        "monitor_loss",
        "monitor_label",
        "grad_norm",
        "n_failed_samples",
        "failed_processes",
    ]
    assert rows.empty


def test_runlogger_jsonl_sink_round_trips_records(tmp_path):
    jsonl_path = tmp_path / "metrics.jsonl"
    with RunLogger(log_every=2, metrics_jsonl=jsonl_path, log_header_every=0) as run:
        run.start(
            target_names=("biomass",),
            process_names=("p1", "p2"),
            total_steps=2,
            compile_warmup_seconds=0.0,
        )
        run.record_step(
            _make_record(1, total_steps=2, target_names=("biomass",), per_target=(0.5,))
        )
        run.record_step(
            _make_record(2, total_steps=2, target_names=("biomass",), per_target=(0.4,))
        )
        run.finalize()
    lines = jsonl_path.read_text().strip().splitlines()
    assert len(lines) == 2
    records = [json.loads(line) for line in lines]
    assert records[0]["step"] == 1
    assert records[1]["step"] == 2
    assert records[0]["per_process_loss"] == [1.2, 1.8]
    assert records[0]["target_names"] == ["biomass"]


def test_runlogger_reports_failed_segments(caplog):
    """A step whose batch had a bailing ODE solve emits a warning naming the failed
    processes with the per-step k/B count; clean steps emit no warning."""
    with RunLogger(log_every=100, log_header_every=0) as run:
        run.start(
            target_names=("biomass",),
            process_names=("p1", "p2"),
            total_steps=3,
            compile_warmup_seconds=0.0,
        )
        with caplog.at_level(logging.WARNING):
            run.record_step(
                _make_record(1, target_names=("biomass",), per_target=(1.0,))
            )
            run.record_step(
                _make_record(
                    2,
                    target_names=("biomass",),
                    per_target=(1.0,),
                    failed_process_names=("p2",),
                )
            )
            run.record_step(
                _make_record(
                    3,
                    target_names=("biomass",),
                    per_target=(1.0,),
                    failed_process_names=("p1", "p2"),
                )
            )
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
    with RunLogger(
        log_every=100,
        metrics_csv=csv_path,
        metrics_jsonl=jsonl_path,
        log_header_every=0,
    ) as run:
        run.start(
            target_names=("biomass",),
            process_names=("p1", "p2"),
            total_steps=2,
            compile_warmup_seconds=0.0,
        )
        run.record_step(_make_record(1, target_names=("biomass",), per_target=(1.0,)))
        run.record_step(
            _make_record(
                2,
                target_names=("biomass",),
                per_target=(1.0,),
                failed_process_names=("p2",),
            )
        )
        run.finalize()

    df = pd.read_csv(csv_path)
    assert list(df["n_failed_samples"]) == [0, 1]
    # Empty on the clean step; the failed process name on the second.
    failed_col = df["failed_processes"].fillna("").astype(str).tolist()
    assert failed_col[0] in ("", "nan")
    assert failed_col[1] == "p2"

    records = [json.loads(x) for x in jsonl_path.read_text().strip().splitlines()]
    assert records[0]["n_failed_samples"] == 0
    assert records[0]["failed_processes"] == []
    assert records[1]["n_failed_samples"] == 1
    assert records[1]["failed_processes"] == ["p2"]


def test_runlogger_resume_rejects_metrics_csv_schema_mismatch(tmp_path):
    """Resuming against a metrics.csv written with an older, narrower schema (e.g. one
    predating the failed-segment columns) must fail loudly, not silently append
    misaligned columns."""
    csv_path = tmp_path / "metrics.csv"
    old_cols = [
        c
        for c in _csv_row_dict(_DUMMY_RECORD)
        if c not in ("n_failed_samples", "failed_processes")
    ]
    pd.DataFrame(columns=old_cols).to_csv(csv_path, index=False)
    run = RunLogger(log_every=1, metrics_csv=csv_path, resume=True, log_header_every=0)
    with pytest.raises(ValueError, match="schema mismatch"):
        run.start(
            target_names=("biomass",),
            process_names=("p1",),
            total_steps=1,
            compile_warmup_seconds=0.0,
        )


def test_runlogger_resume_appends_when_schema_matches(tmp_path):
    """A current-schema resume appends cleanly (the guard does not over-fire) and the
    columns stay aligned across the restart."""
    csv_path = tmp_path / "metrics.csv"
    with RunLogger(log_every=1, metrics_csv=csv_path, log_header_every=0) as run:
        run.start(
            target_names=("biomass",),
            process_names=("p1",),
            total_steps=2,
            compile_warmup_seconds=0.0,
        )
        run.record_step(
            _make_record(1, target_names=("biomass",), per_target=(1.0,))
        )
        run.finalize()
    with RunLogger(
        log_every=1, metrics_csv=csv_path, resume=True, log_header_every=0
    ) as run:
        run.start(
            target_names=("biomass",),
            process_names=("p1",),
            total_steps=2,
            compile_warmup_seconds=0.0,
        )
        run.record_step(
            _make_record(
                2,
                target_names=("biomass",),
                per_target=(1.0,),
                failed_process_names=("p1",),
            )
        )
        run.finalize()
    df = pd.read_csv(csv_path)
    assert list(df["step"]) == [1, 2]
    assert "n_failed_samples" in df.columns
    assert list(df["n_failed_samples"]) == [0, 1]


def test_runlogger_close_is_idempotent(tmp_path):
    csv_path = tmp_path / "metrics.csv"
    run = RunLogger(log_every=1, metrics_csv=csv_path)
    run.start(
        target_names=("biomass",),
        process_names=("p1",),
        total_steps=1,
        compile_warmup_seconds=0.0,
    )
    run.record_step(
        _make_record(
            1,
            total_steps=1,
            target_names=("biomass",),
            process_names=("p1",),
            per_target=(0.5,),
            per_process=(0.5,),
        )
    )
    run.finalize()
    run.close()
    run.close()  # second close must not raise
