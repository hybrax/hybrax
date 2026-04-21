"""Tests for the `bp-train forward` CLI path and forward harness plumbing.

These tests exercise the pieces that do not require a real trained model:

* the loss-table formatter,
* the metadata sidecar helpers,
* the CLI dispatch for `forward` (via monkeypatching ``forward_from_collection``),
* the sidecar write performed by ``_handle_train``,
* a light-touch check that ``plot_process_simulations`` is plumbed through and
  accepts the new ``training_process_names`` / ``timeseries_csv_path`` /
  ``filename_suffix`` parameters.

End-to-end forward (with a real ODE solve) is covered by
``test_forward_end_to_end`` which runs on the kittler example fixture when
available. It is marked ``integration`` so it can be skipped in fast suites.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from bp_train import cli, postprocessing
from bp_train.harness import (
    ForwardConfig,
    ForwardResult,
    TrainHarnessResult,
)


# ---------------------------------------------------------------------------
# metadata sidecar helpers
# ---------------------------------------------------------------------------


def test_save_and_load_model_metadata_roundtrip(tmp_path: Path):
    meta = {
        "prepared_input": "/abs/path/prepared.json",
        "solver": {"rtol": 1e-5, "atol": 1e-7, "max_steps": 2048},
        "training_processes": ["p1", "p2"],
    }
    path = tmp_path / "trained_wrapper.meta.json"
    postprocessing.save_model_metadata(path, meta)

    loaded = postprocessing.load_model_metadata(path)
    assert loaded == meta


def test_load_model_metadata_missing_returns_empty(tmp_path: Path):
    assert postprocessing.load_model_metadata(tmp_path / "nope.json") == {}


# ---------------------------------------------------------------------------
# loss table formatter
# ---------------------------------------------------------------------------


def _make_forward_result(
    *,
    process_names=("p1", "p2"),
    target_names=("X", "S"),
    training_process_names=("p1",),
    per_process_total=None,
    per_process_per_target=None,
) -> ForwardResult:
    per_process_total = per_process_total or {"p1": 0.25, "p2": 0.75}
    per_process_per_target = per_process_per_target or {
        "p1": (0.1, 0.4),
        "p2": (0.3, 1.2),
    }
    return ForwardResult(
        trained_wrapper=None,
        store=None,
        process_names=tuple(process_names),
        target_names=tuple(target_names),
        modeled_flow_names=(),
        training_process_names=tuple(training_process_names),
        per_process_total_loss=per_process_total,
        per_process_per_target_loss=per_process_per_target,
    )


def test_format_loss_table_has_expected_rows_columns():
    result = _make_forward_result()
    table_str, csv_rows = cli._format_loss_table(result)

    assert "LOSSES (forward evaluation)" in table_str
    assert "p1" in table_str
    assert "p2" in table_str
    assert "train" in table_str
    assert "holdout" in table_str
    # header + 2 data rows + mean row
    assert len(csv_rows) == 1 + 2 + 1
    header = csv_rows[0]
    assert header == ["process", "total", "X", "S", "split"]
    # mean row: total=(0.25+0.75)/2 = 0.5
    mean_row = csv_rows[-1]
    assert mean_row[0] == "total (mean)"
    assert float(mean_row[1]) == pytest.approx(0.5)
    assert float(mean_row[2]) == pytest.approx(0.2)
    assert float(mean_row[3]) == pytest.approx(0.8)
    # train/holdout classification
    data_rows = {row[0]: row for row in csv_rows[1:-1]}
    assert data_rows["p1"][-1] == "train"
    assert data_rows["p2"][-1] == "holdout"


def test_format_loss_table_all_holdout_when_training_empty():
    result = _make_forward_result(training_process_names=())
    _, csv_rows = cli._format_loss_table(result)
    for row in csv_rows[1:-1]:
        assert row[-1] == "holdout"


# ---------------------------------------------------------------------------
# CLI: _handle_train writes a sidecar next to trained_wrapper.eqx
# ---------------------------------------------------------------------------


def test_train_cli_writes_sidecar_with_solver_and_training_context(
    monkeypatch, tmp_path: Path
):
    """After training, a .meta.json sidecar must sit next to the .eqx file."""
    sentinel_collection = object()

    def fake_load(_path):
        return sentinel_collection

    def fake_train_from_collection(collection, *, config, custom_py, runtime_config):
        return TrainHarnessResult(
            trained_wrapper=None,
            mean_loss_by_step=(1.0, 0.5),
            sampled_loss_by_process_at_log_steps={1: (("p1", 1.0),)},
            batch_process_names_by_step=(("p1",),),
            per_process_loss_by_step=((0.5,),),
            compile_warmup_seconds=0.0,
            step_time_seconds=(0.0,),
            train_step_input_signature=(),
            train_step_rebuild_count=0,
        )

    monkeypatch.setattr(cli, "train_from_collection", fake_train_from_collection)
    monkeypatch.setattr(cli, "load_process_collection_json", fake_load)
    monkeypatch.setattr(cli, "save_model", lambda wrapper, path: None)

    output_dir = tmp_path / "out"
    exit_code = cli.main(
        [
            "train",
            "--input",
            "prepared.json",
            "--process",
            "p1,p2",
            "--target",
            "X",
            "--solver-rtol",
            "1e-4",
            "--solver-atol",
            "1e-6",
            "--solver-max-steps",
            "1234",
            "--output-dir",
            str(output_dir),
            "--no-plot",
        ]
    )
    assert exit_code == 0

    sidecar = output_dir / "trained_wrapper.meta.json"
    assert sidecar.exists()
    meta = json.loads(sidecar.read_text())
    assert meta["training_processes"] == ["p1", "p2"]
    assert meta["targets"] == ["X"]
    assert meta["solver"] == {
        "max_steps": 1234,
        "rtol": 1e-4,
        "atol": 1e-6,
        "use_jump_ts": True,
    }
    assert "final_mean_loss" in meta["training"]


def test_train_cli_sidecar_defaults_to_none_training_processes_when_not_set(
    monkeypatch, tmp_path: Path
):
    """When --process is omitted, training_processes must be None, not missing."""

    def fake_load(_path):
        return object()

    def fake_train_from_collection(collection, *, config, custom_py, runtime_config):
        return TrainHarnessResult(
            trained_wrapper=None,
            mean_loss_by_step=(1.0,),
            sampled_loss_by_process_at_log_steps={},
            batch_process_names_by_step=((),),
            per_process_loss_by_step=((),),
            compile_warmup_seconds=0.0,
            step_time_seconds=(0.0,),
            train_step_input_signature=(),
            train_step_rebuild_count=0,
        )

    monkeypatch.setattr(cli, "train_from_collection", fake_train_from_collection)
    monkeypatch.setattr(cli, "load_process_collection_json", fake_load)
    monkeypatch.setattr(cli, "save_model", lambda w, p: None)

    output_dir = tmp_path / "out"
    cli.main(
        [
            "train",
            "--input",
            "prepared.json",
            "--output-dir",
            str(output_dir),
            "--no-plot",
        ]
    )
    meta = json.loads((output_dir / "trained_wrapper.meta.json").read_text())
    assert meta["training_processes"] is None


# ---------------------------------------------------------------------------
# CLI: _handle_forward dispatch
# ---------------------------------------------------------------------------


class _DummyStore:
    process_order = ("p1", "p2", "p3")


def _make_fake_collection():
    class _Coll:
        processes = {"p1": object(), "p2": object(), "p3": object()}

    return _Coll()


def _stub_forward_result(**kwargs) -> ForwardResult:
    defaults = dict(
        trained_wrapper=object(),
        store=_DummyStore(),
        process_names=("p1", "p2"),
        target_names=("X", "S"),
        modeled_flow_names=(),
        training_process_names=("p1", "p2"),
        per_process_total_loss={"p1": 0.1, "p2": 0.2},
        per_process_per_target_loss={"p1": (0.05, 0.15), "p2": (0.1, 0.3)},
    )
    defaults.update(kwargs)
    return ForwardResult(**defaults)


def test_forward_cli_dispatches_and_writes_losses_csv(monkeypatch, tmp_path: Path):
    captured: dict[str, object] = {}

    model_path = tmp_path / "trained_wrapper.eqx"
    model_path.write_bytes(b"")
    sidecar = tmp_path / "trained_wrapper.meta.json"
    sidecar.write_text(
        json.dumps(
            {
                "prepared_input": str(tmp_path / "prepared.json"),
                "custom_py": None,
                "training_processes": ["p1", "p2"],
                "targets": ["X", "S"],
                "target_source": "reactor_components",
                "solver": {
                    "max_steps": 2048,
                    "rtol": 1e-5,
                    "atol": 1e-7,
                    "use_jump_ts": True,
                },
            }
        )
    )

    fake_collection = _make_fake_collection()
    monkeypatch.setattr(cli, "load_process_collection_json", lambda p: fake_collection)

    def fake_forward(
        collection,
        *,
        model_path,
        config,
        custom_py,
        runtime_config,
        training_process_names,
    ):
        captured["collection"] = collection
        captured["model_path"] = Path(model_path)
        captured["config"] = config
        captured["custom_py"] = custom_py
        captured["training_process_names"] = training_process_names
        return _stub_forward_result()

    monkeypatch.setattr(cli, "forward_from_collection", fake_forward)

    # Stub out plotting — we don't want to launch matplotlib here.
    plot_calls: dict[str, object] = {}

    def fake_plot(
        trained_wrapper,
        collection,
        store,
        output_dir,
        process_names=None,
        *,
        solver_max_steps,
        solver_rtol,
        solver_atol,
        training_process_names=None,
        timeseries_csv_path=None,
        filename_suffix="",
    ):
        plot_calls["solver_rtol"] = solver_rtol
        plot_calls["training_process_names"] = training_process_names
        plot_calls["process_names"] = process_names
        plot_calls["output_dir"] = Path(output_dir)
        plot_calls["timeseries_csv_path"] = timeseries_csv_path

    monkeypatch.setattr(cli, "plot_process_simulations", fake_plot)

    output_dir = tmp_path / "fwd"
    exit_code = cli.main(
        [
            "forward",
            "--model",
            str(model_path),
            "--process",
            "p1,p2",
            "--output-dir",
            str(output_dir),
        ]
    )
    assert exit_code == 0

    # Config received by harness matches sidecar values.
    cfg = captured["config"]
    assert isinstance(cfg, ForwardConfig)
    assert cfg.solver_rtol == 1e-5
    assert cfg.solver_atol == 1e-7
    assert cfg.solver_max_steps == 2048
    assert cfg.process_names == ("p1", "p2")
    assert cfg.target_variable_order == ("X", "S")
    assert cfg.target_source == "reactor_components"
    assert captured["training_process_names"] == ("p1", "p2")

    # Loss CSV written to the default location.
    losses_csv = output_dir / "losses.csv"
    assert losses_csv.exists()
    rows = list(csv.reader(losses_csv.open()))
    assert rows[0] == ["process", "total", "X", "S", "split"]
    assert any(row[0] == "p1" and row[-1] == "train" for row in rows[1:])

    # Plotting invoked with sidecar-derived solver settings.
    assert plot_calls["solver_rtol"] == 1e-5
    assert plot_calls["training_process_names"] == ("p1", "p2")


def test_forward_cli_overrides_beat_sidecar(monkeypatch, tmp_path: Path):
    model_path = tmp_path / "m.eqx"
    model_path.write_bytes(b"")
    (tmp_path / "m.meta.json").write_text(
        json.dumps(
            {
                "prepared_input": str(tmp_path / "prepared.json"),
                "solver": {
                    "max_steps": 10,
                    "rtol": 1e-5,
                    "atol": 1e-7,
                    "use_jump_ts": True,
                },
            }
        )
    )

    monkeypatch.setattr(
        cli, "load_process_collection_json", lambda p: _make_fake_collection()
    )

    captured_cfg: dict[str, ForwardConfig] = {}

    def fake_forward(collection, **kwargs):
        captured_cfg["cfg"] = kwargs["config"]
        return _stub_forward_result()

    monkeypatch.setattr(cli, "forward_from_collection", fake_forward)
    monkeypatch.setattr(cli, "plot_process_simulations", lambda *a, **k: None)

    cli.main(
        [
            "forward",
            "--model",
            str(model_path),
            "--solver-rtol",
            "1e-3",
            "--solver-max-steps",
            "99",
            "--no-jump-ts",
            "--output-dir",
            str(tmp_path / "fwd"),
            "--no-plot",
        ]
    )
    cfg = captured_cfg["cfg"]
    assert cfg.solver_rtol == 1e-3
    assert cfg.solver_max_steps == 99
    assert cfg.solver_use_jump_ts is False
    # Non-overridden atol still comes from the sidecar
    assert cfg.solver_atol == 1e-7


def test_forward_cli_requires_input_when_no_sidecar(monkeypatch, tmp_path: Path):
    model_path = tmp_path / "m.eqx"
    model_path.write_bytes(b"")

    with pytest.raises(SystemExit, match="prepared_input"):
        cli.main(["forward", "--model", str(model_path)])


def test_forward_cli_missing_model_errors(tmp_path: Path):
    with pytest.raises(SystemExit, match="does not exist"):
        cli.main(["forward", "--model", str(tmp_path / "nope.eqx")])


def test_forward_cli_unknown_sidecar_marks_everything_holdout(
    monkeypatch, tmp_path: Path
):
    """Pre-sidecar models: treat every eval process as holdout."""
    model_path = tmp_path / "m.eqx"
    model_path.write_bytes(b"")
    # Sidecar with no training_processes key at all.
    (tmp_path / "m.meta.json").write_text(
        json.dumps({"prepared_input": str(tmp_path / "prepared.json")})
    )

    monkeypatch.setattr(
        cli, "load_process_collection_json", lambda p: _make_fake_collection()
    )

    captured_tpn: dict[str, object] = {}

    def fake_forward(collection, **kwargs):
        captured_tpn["tpn"] = kwargs["training_process_names"]
        return _stub_forward_result(training_process_names=())

    monkeypatch.setattr(cli, "forward_from_collection", fake_forward)
    monkeypatch.setattr(cli, "plot_process_simulations", lambda *a, **k: None)

    cli.main(
        [
            "forward",
            "--model",
            str(model_path),
            "--output-dir",
            str(tmp_path / "fwd"),
            "--no-plot",
        ]
    )
    assert captured_tpn["tpn"] == ()


# ---------------------------------------------------------------------------
# Integration: end-to-end forward on the kittler example (slow)
# ---------------------------------------------------------------------------


KITTLER_PREPARED = (
    Path(__file__).resolve().parents[1]
    / "examples"
    / "01_kittler_2022"
    / "prepared.json"
)
KITTLER_CUSTOM = (
    Path(__file__).resolve().parents[1] / "examples" / "01_kittler_2022" / "custom.py"
)


@pytest.mark.integration
@pytest.mark.skipif(
    not KITTLER_PREPARED.exists() or not KITTLER_CUSTOM.exists(),
    reason="kittler example fixture not available",
)
def test_forward_end_to_end_on_kittler(tmp_path: Path):
    """Full round-trip: train 3 steps → forward → check outputs."""
    out_dir = tmp_path / "out"
    exit_code = cli.main(
        [
            "train",
            "--input",
            str(KITTLER_PREPARED),
            "--custom",
            str(KITTLER_CUSTOM),
            "--target-source",
            "reactor_components",
            "--steps",
            "3",
            "--log-every",
            "3",
            "--seed",
            "42",
            "--solver-max-steps",
            "2048",
            "--solver-rtol",
            "1e-3",
            "--solver-atol",
            "1e-5",
            "--output-dir",
            str(out_dir),
            "--no-plot",
        ]
    )
    assert exit_code == 0
    model = out_dir / "trained_wrapper.eqx"
    sidecar = out_dir / "trained_wrapper.meta.json"
    assert model.exists()
    assert sidecar.exists()

    fwd_dir = out_dir / "forward"
    exit_code = cli.main(
        [
            "forward",
            "--model",
            str(model),
            "--process",
            "DoE1_R1",
            "--output-dir",
            str(fwd_dir),
            "--no-plot",
            "--timeseries-csv",
            str(fwd_dir / "ts.csv"),
        ]
    )
    assert exit_code == 0
    losses_csv = fwd_dir / "losses.csv"
    assert losses_csv.exists()
    rows = list(csv.reader(losses_csv.open()))
    # header + 1 data row + mean row
    assert len(rows) == 3
    assert rows[0][0] == "process"
    assert rows[1][0] == "DoE1_R1"
    assert rows[1][-1] == "train"  # default training covers all kittler processes


# ---------------------------------------------------------------------------
# plot_process_simulations: light signature / stub test
# ---------------------------------------------------------------------------


def test_plot_process_simulations_is_exported_with_new_kwargs():
    """Guard against accidental signature regressions."""
    import inspect

    sig = inspect.signature(postprocessing.plot_process_simulations)
    assert "training_process_names" in sig.parameters
    assert "timeseries_csv_path" in sig.parameters
    assert "filename_suffix" in sig.parameters
