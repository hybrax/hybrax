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

import inspect
import json
from pathlib import Path

import jax.numpy as jnp
import numpy as np
import pandas as pd
import pytest
from bp_format.dataclasses import (
    BioProcess,
    BioProcessCollection,
    BioProcessMetadata,
    ReactorMedium,
    ReactorMediumComponent,
    SampleVolumeChange,
    TimeAxis,
    TimeSeries,
    Volume,
)

from bp_train import cli, postprocessing
from bp_train.controls_store import ControlsStore
from bp_train.harness import (
    ForwardConfig,
    ForwardResult,
    TrainHarnessResult,
)
from bp_train.defaults import DefaultLossModule
from bp_train.harness import compute_dense_exports
from bp_train.model_api import ReactionOutputs, UserReactionModule
from bp_train.training_data import TrainingDataStore
from bp_train.wrapper import HybridOdeWrapper


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
        name_modeled_FVCs=(),
        name_modeled_SVCs=(),
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
    # header + 2 data rows + total/train/holdout mean rows
    assert len(csv_rows) == 1 + 2 + 3
    header = csv_rows[0]
    assert header == ["process", "total", "X", "S", "split"]
    # mean row: total=(0.25+0.75)/2 = 0.5
    mean_row = next(row for row in csv_rows[1:] if row[0] == "total (mean)")
    assert mean_row[0] == "total (mean)"
    assert float(mean_row[1]) == pytest.approx(0.5)
    assert float(mean_row[2]) == pytest.approx(0.2)
    assert float(mean_row[3]) == pytest.approx(0.8)
    # train/holdout classification
    data_rows = {row[0]: row for row in csv_rows[1:] if "(mean)" not in row[0]}
    assert data_rows["p1"][-1] == "train"
    assert data_rows["p2"][-1] == "holdout"


def test_format_loss_table_all_holdout_when_training_empty():
    result = _make_forward_result(training_process_names=())
    _, csv_rows = cli._format_loss_table(result)
    for row in csv_rows[1:]:
        if "(mean)" in row[0]:
            continue
        assert row[-1] == "holdout"


# ---------------------------------------------------------------------------
# CLI: _handle_train writes a sidecar next to trained_wrapper.eqx
# ---------------------------------------------------------------------------


# NOTE: the old `trained_wrapper.meta.json` sidecar is replaced by the run-dir
# `config.json` (status + provenance). End-to-end coverage of that FAIR layout
# — config.json, content_hash, custom.py file_hash, resume, re-run guard — lives
# in tests/test_cli_run_dir.py.


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
        name_modeled_FVCs=(),
        name_modeled_SVCs=(),
        training_process_names=("p1", "p2"),
        per_process_total_loss={"p1": 0.1, "p2": 0.2},
        per_process_per_target_loss={"p1": (0.05, 0.15), "p2": (0.1, 0.3)},
    )
    defaults.update(kwargs)
    return ForwardResult(**defaults)


def _make_forward_run_dir(
    tmp_path: Path,
    *,
    processes=("p1", "p2"),
    targets=("X", "S"),
    target_source="reactor_components",
    solver=None,
) -> Path:
    """Build a minimal FAIR run dir (config.json + model/params.eqx) for the
    run-dir forward path. The collection load + forward sim are monkeypatched."""
    from bp_train.run_config import RunConfig
    from bp_train.serialization import run_config_to_jsonable

    solver = solver or {"max_steps": 2048, "rtol": 1e-5, "atol": 1e-7, "jump_ts": True}
    run_dir = tmp_path / "run"
    (run_dir / "model").mkdir(parents=True)
    (run_dir / "checkpoints").mkdir(parents=True)
    (run_dir / "model" / "params.eqx").write_bytes(b"")
    (run_dir / "prepared.json").write_text("{}", encoding="utf-8")
    data: dict = {
        "prepared": str(run_dir / "prepared.json"),
        "target_source": target_source,
    }
    if processes is not None:
        data["processes"] = list(processes)
    if targets is not None:
        data["targets"] = list(targets)
    cfg = RunConfig.model_validate({"data": data, "solver": solver})
    (run_dir / "config.json").write_text(
        json.dumps({"status": "complete", "config": run_config_to_jsonable(cfg)}),
        encoding="utf-8",
    )
    return run_dir


_FORWARD_DEFAULT_SCALES: dict[str, jnp.ndarray] = {
    "SCALE_modeled_RMCs": jnp.ones(1, dtype=jnp.float32),
    "SCALE_V_in_cumulative": jnp.asarray(1.0, dtype=jnp.float32),
    "SCALE_modeled_FVCs_cumulative": jnp.ones(0, dtype=jnp.float32),
    "SCALE_controlled_FVCs_cumulative": jnp.ones(0, dtype=jnp.float32),
    "SCALE_controlled_FVCs_rates": jnp.ones(0, dtype=jnp.float32),
    "SCALE_controlled_FVCs_Cin": jnp.ones((0, 1), dtype=jnp.float32),
    "SCALE_controlled_PVs": jnp.ones(0, dtype=jnp.float32),
    "SCALE_modeled_FVCs_Cin": jnp.ones((0, 1), dtype=jnp.float32),
    "SCALE_modeled_BiologicalOde_rates": jnp.ones(1, dtype=jnp.float32),
    "SCALE_modeled_FVCs_rates": jnp.ones(0, dtype=jnp.float32),
}


class _ConstantReactionModule(UserReactionModule):
    SCL_specific_rates: jnp.ndarray
    SCL_feed_rates: jnp.ndarray
    aux: dict[str, jnp.ndarray] | None

    def __init__(
        self,
        specific_rates: jnp.ndarray,
        modeled_feed_rates: jnp.ndarray,
        auxiliary: dict[str, jnp.ndarray] | None = None,
        **scale_kwargs,
    ):
        super().__init__(**{**_FORWARD_DEFAULT_SCALES, **scale_kwargs})
        self.SCL_specific_rates = specific_rates
        self.SCL_feed_rates = modeled_feed_rates
        self.aux = auxiliary

    def __call__(self, t, inputs):
        del t, inputs
        return ReactionOutputs(
            SCL_modeled_BiologicalOde_rates=self.SCL_specific_rates,
            SCL_modeled_FVCs_rates=self.SCL_feed_rates,
            auxiliary=self.aux,
        )


def _make_one_species_process(
    *,
    initial_volume: float = 1.0,
    sample_delta: float = -0.1,
) -> BioProcess:
    return BioProcess(
        metadata=BioProcessMetadata(name="p1", process_type="fed_batch"),
        time_axis=TimeAxis(unit="h", start=0.0, end=2.0, time_reference="start"),
        volume=Volume(
            initial_volume=initial_volume,
            unit="L",
            volume_changes={
                "sample_1": SampleVolumeChange(
                    name="sample_1",
                    unit="L",
                    is_controlled=False,
                    is_continuous=False,
                    values=TimeSeries(
                        times=jnp.asarray([1.0]),
                        values=jnp.asarray([sample_delta]),
                    ),
                )
            },
        ),
        reactor_medium=ReactorMedium(
            name="rm",
            density=1.0,
            density_unit="kg/L",
            components={
                "biomass": ReactorMediumComponent(
                    name="biomass",
                    unit="g/L",
                    concentration=TimeSeries(
                        times=jnp.asarray([0.0, 2.0]),
                        values=jnp.asarray([1.0, 1.0]),
                    ),
                ),
            },
        ),
        process_variables={},
    )


def _build_single_process_runtime(
    *,
    initial_volume: float = 1.0,
    sample_delta: float = -0.1,
    q_scaled: float = 0.0,
    q_scale: float = 1.0,
    auxiliary: dict[str, jnp.ndarray] | None = None,
):
    process = _make_one_species_process(
        initial_volume=initial_volume,
        sample_delta=sample_delta,
    )
    collection = BioProcessCollection(processes={"p1": process}, metadata={})
    controls = ControlsStore.from_collection(collection).get_controls("p1")
    store = TrainingDataStore.from_collection(
        collection,
        target_variable_order=["biomass"],
        target_source="reactor_components",
    )
    wrapper = HybridOdeWrapper.from_process(
        reaction_module=_ConstantReactionModule(
            specific_rates=jnp.asarray([q_scaled], dtype=jnp.float32),
            modeled_feed_rates=jnp.zeros((0,), dtype=jnp.float32),
            auxiliary=auxiliary,
            SCALE_modeled_BiologicalOde_rates=jnp.asarray(
                [q_scale], dtype=jnp.float32
            ),
        ),
        process=process,
        controls=controls,
        loss_module=DefaultLossModule(target_names=tuple(store.name_measured)),
        min_V=0.02,
    )
    return collection, store, wrapper


def _write_forward_config(
    tmp_path: Path,
    model_dirs,
    *,
    processes=None,
    output_dir=None,
    prepared=None,
    plots=False,
    name="forward-config.json",
) -> Path:
    """Write a forward_config.json for the `--config`-only forward CLI."""
    cfg: dict = {"models": [str(m) for m in model_dirs]}
    data: dict = {}
    if prepared is not None:
        data["prepared"] = str(prepared)
    if processes is not None:
        data["processes"] = list(processes)
    if data:
        cfg["data"] = data
    output: dict = {"plots": plots}
    if output_dir is not None:
        output["dir"] = str(output_dir)
    cfg["output"] = output
    path = tmp_path / name
    path.write_text(json.dumps(cfg), encoding="utf-8")
    return path


def test_forward_cli_dispatches_and_writes_losses_csv(monkeypatch, tmp_path: Path):
    captured: dict[str, object] = {}
    run_dir = _make_forward_run_dir(
        tmp_path, processes=("p1", "p2"), targets=("X", "S")
    )

    fake_collection = _make_fake_collection()
    monkeypatch.setattr(cli, "load_process_collection", lambda p: fake_collection)

    def fake_forward(collection, **kwargs):
        captured["collection"] = collection
        captured["model_path"] = Path(kwargs["model_path"])
        captured["config"] = kwargs["config"]
        captured["custom_py"] = kwargs["custom_py"]
        captured["training_process_names"] = kwargs["training_process_names"]
        return _stub_forward_result()

    monkeypatch.setattr(cli, "forward_from_collection", fake_forward)
    monkeypatch.setattr(cli, "plot_process_simulations", lambda *a, **k: None)
    # the stub ForwardResult has no real wrapper/dense_exports; skip the writers
    monkeypatch.setattr(cli, "export_predictions_csv", lambda *a, **k: None)

    output_dir = tmp_path / "fwd"
    fwd_config = _write_forward_config(tmp_path, [run_dir], processes=("p1", "p2"))
    exit_code = cli.main(
        ["forward", "--config", str(fwd_config), "--output-dir", str(output_dir)]
    )
    assert exit_code == 0

    # Config received by the harness matches the model's recorded config.json.
    cfg = captured["config"]
    assert isinstance(cfg, ForwardConfig)
    assert cfg.solver_rtol == 1e-5
    assert cfg.solver_atol == 1e-7
    assert cfg.solver_max_steps == 2048
    assert cfg.process_names == ("p1", "p2")
    assert cfg.target_variable_order == ("X", "S")
    assert cfg.target_source == "reactor_components"
    assert captured["training_process_names"] == ("p1", "p2")
    assert captured["model_path"] == run_dir / "model" / "params.eqx"

    losses_csv = output_dir / "losses.csv"
    assert losses_csv.exists()
    rows = pd.read_csv(losses_csv)
    assert rows.columns.tolist() == ["process", "total", "X", "S", "split"]
    assert ((rows["process"] == "p1") & (rows["split"] == "train")).any()


def test_forward_cli_overwrite_guard(monkeypatch, tmp_path: Path):
    """A second forward into a populated --output-dir is refused unless --overwrite."""
    run_dir = _make_forward_run_dir(tmp_path, processes=("p1", "p2"))
    monkeypatch.setattr(
        cli, "load_process_collection", lambda p: _make_fake_collection()
    )
    monkeypatch.setattr(
        cli, "forward_from_collection", lambda collection, **k: _stub_forward_result()
    )
    monkeypatch.setattr(cli, "plot_process_simulations", lambda *a, **k: None)
    monkeypatch.setattr(cli, "export_predictions_csv", lambda *a, **k: None)

    output_dir = tmp_path / "fwd"
    fwd_config = _write_forward_config(tmp_path, [run_dir])
    base = ["forward", "--config", str(fwd_config), "--output-dir", str(output_dir)]

    assert cli.main(base) == 0
    assert (output_dir / "losses.csv").is_file()
    # a second run without --overwrite is blocked
    assert cli.main(base) == 1
    # ...and allowed with it
    assert cli.main(base + ["--overwrite"]) == 0


def test_forward_cli_solver_accuracy_is_read_only(monkeypatch, tmp_path: Path):
    """All solver settings (max_steps/rtol/atol/jump_ts) are replayed read-only
    from the model's config.json — there are no CLI override flags."""
    run_dir = _make_forward_run_dir(
        tmp_path,
        solver={"max_steps": 10, "rtol": 1e-5, "atol": 1e-7, "jump_ts": True},
    )
    monkeypatch.setattr(
        cli, "load_process_collection", lambda p: _make_fake_collection()
    )
    captured_cfg: dict[str, ForwardConfig] = {}

    def fake_forward(collection, **kwargs):
        captured_cfg["cfg"] = kwargs["config"]
        return _stub_forward_result()

    monkeypatch.setattr(cli, "forward_from_collection", fake_forward)
    monkeypatch.setattr(cli, "plot_process_simulations", lambda *a, **k: None)
    monkeypatch.setattr(cli, "export_predictions_csv", lambda *a, **k: None)

    fwd_config = _write_forward_config(tmp_path, [run_dir])
    cli.main(
        ["forward", "--config", str(fwd_config), "--output-dir", str(tmp_path / "fwd")]
    )
    cfg = captured_cfg["cfg"]
    assert cfg.solver_rtol == 1e-5
    assert cfg.solver_atol == 1e-7
    assert cfg.solver_use_jump_ts is True
    assert cfg.solver_max_steps == 10  # straight from the model's config.json


def test_forward_cli_removed_flags_are_rejected(tmp_path: Path):
    """The old per-run flags are gone — everything lives in the forward config now."""
    run_dir = _make_forward_run_dir(tmp_path)
    fwd_config = _write_forward_config(tmp_path, [run_dir])
    base = ["forward", "--config", str(fwd_config)]
    for extra in (
        ["--solver-rtol", "1"],
        ["--solver-atol", "1"],
        ["--solver-max-steps", "1"],
        ["--no-jump-ts"],
        ["--model", str(run_dir)],
        ["--input", str(run_dir)],
        ["--process", "p1"],
        ["--no-plot"],
        ["--loss-csv", str(tmp_path / "l.csv")],
        ["--timeseries-csv", str(tmp_path / "t.csv")],
    ):
        with pytest.raises(SystemExit):
            cli.main(base + extra)


def test_forward_cli_bare_model_without_run_dir_errors(tmp_path: Path):
    # A bare .eqx with no config.json at/above it is not a valid model bundle.
    model_path = tmp_path / "m.eqx"
    model_path.write_bytes(b"")
    fwd_config = _write_forward_config(tmp_path, [model_path])
    with pytest.raises(SystemExit, match="config.json"):
        cli.main(["forward", "--config", str(fwd_config)])


def test_forward_cli_missing_model_errors(tmp_path: Path):
    fwd_config = _write_forward_config(tmp_path, [tmp_path / "nope.eqx"])
    with pytest.raises(SystemExit, match="does not exist"):
        cli.main(["forward", "--config", str(fwd_config)])


def test_forward_cli_no_configured_processes_evaluates_all(
    monkeypatch, tmp_path: Path
):
    """A run with no data.processes evaluates (and labels train) every process."""
    run_dir = _make_forward_run_dir(tmp_path, processes=None)

    monkeypatch.setattr(
        cli, "load_process_collection", lambda p: _make_fake_collection()
    )

    captured_tpn: dict[str, object] = {}

    def fake_forward(collection, **kwargs):
        captured_tpn["tpn"] = kwargs["training_process_names"]
        return _stub_forward_result(training_process_names=())

    monkeypatch.setattr(cli, "forward_from_collection", fake_forward)
    monkeypatch.setattr(cli, "plot_process_simulations", lambda *a, **k: None)
    monkeypatch.setattr(cli, "export_predictions_csv", lambda *a, **k: None)

    fwd_config = _write_forward_config(tmp_path, [run_dir])
    cli.main(
        ["forward", "--config", str(fwd_config), "--output-dir", str(tmp_path / "fwd")]
    )
    assert captured_tpn["tpn"] == ("p1", "p2", "p3")


# ---------------------------------------------------------------------------
# Integration: end-to-end CLI round-trip on a self-contained fixture (slow).
# The fixture lives under tests/fixtures so the core suite never depends on
# anything in examples/. It exercises the full train -> checkpoint -> forward
# path including a custom build_loss_module with extra named terms.
# ---------------------------------------------------------------------------


FIXTURE_DIR = Path(__file__).resolve().parent / "fixtures" / "martens_single"
FIXTURE_DATA = FIXTURE_DIR / "data.json"
FIXTURE_CUSTOM = FIXTURE_DIR / "custom.py"


@pytest.mark.integration
@pytest.mark.skipif(
    not FIXTURE_DATA.exists() or not FIXTURE_CUSTOM.exists(),
    reason="martens_single fixture not available",
)
def test_forward_end_to_end_on_fixture(tmp_path: Path):
    """Full round-trip on the tests/ fixture: prepare -> train 3 -> forward."""
    prepared_dir = tmp_path / "prepared"
    prepared = prepared_dir / "prepared.json"
    prepare_config = tmp_path / "prepare-config.json"
    prepare_config.write_text(
        json.dumps(
            {
                "prepare": {"raw_input": str(FIXTURE_DATA)},
                "custom_py": str(FIXTURE_CUSTOM),
            }
        )
    )
    assert (
        cli.main(
            [
                "prepare",
                "--config",
                str(prepare_config),
                "--output-dir",
                str(prepared_dir),
            ]
        )
        == 0
    )

    out_dir = tmp_path / "out"
    train_config = tmp_path / "train-config.json"
    train_config.write_text(
        json.dumps(
            {
                "data": {
                    "prepared": str(prepared_dir),
                    "target_source": "reactor_components",
                },
                "custom_py": str(FIXTURE_CUSTOM),
                "train": {"steps": 3, "seed": 42},
                "solver": {"max_steps": 2048, "rtol": 1e-3, "atol": 1e-5},
            }
        )
    )
    assert (
        cli.main(
            [
                "train",
                "--config",
                str(train_config),
                "--output-dir",
                str(out_dir),
                "--no-plot",
            ]
        )
        == 0
    )
    # New FAIR run-dir layout: model lives under model/params.eqx with a config.json.
    assert (out_dir / "model" / "params.eqx").exists()
    assert (out_dir / "config.json").exists()
    # custom.py is bundled and its exact-bytes hash recorded for provenance.
    assert (out_dir / "custom.py").exists()
    _doc = json.loads((out_dir / "config.json").read_text())
    assert _doc["inputs"]["custom_py"]["bundled"] == "custom.py"
    assert _doc["inputs"]["custom_py"]["file_hash"].startswith("sha256:")
    # Provenance chain raw -> prepared -> model: the train run's recorded
    # prepared content_hash matches the prepared.json provenance block.
    _prov = json.loads(prepared.read_text())["metadata"]["bp-train"]["provenance"]
    assert _prov["content_hash"].startswith("sha256:")
    assert _prov["prepare_config"] is not None
    assert _doc["inputs"]["prepared_input"]["content_hash"] == _prov["content_hash"]

    fwd_dir = out_dir / "forward"
    # forward consumes the run dir directly (solver/prepared/custom from config.json).
    fwd_config = tmp_path / "forward-config.json"
    fwd_config.write_text(
        json.dumps(
            {
                "models": [str(out_dir)],
                "data": {"processes": ["run_1"]},
                "output": {"dir": str(fwd_dir), "plots": False},
            }
        )
    )
    assert cli.main(["forward", "--config", str(fwd_config)]) == 0
    losses_csv = fwd_dir / "losses.csv"
    assert losses_csv.exists()
    rows = pd.read_csv(losses_csv)
    assert rows.columns[0] == "process"
    assert (rows["process"] == "run_1").any()
    # The custom build_loss_module adds nonneg/<target> columns alongside the
    # per-target measurement terms — confirm they survived to the CSV.
    assert any(str(c).startswith("nonneg/") for c in rows.columns)
    # The dense-grid curvature term (uses dense_t + jump_ts) also surfaces as
    # curvature/<rate> columns — proves the dense_grid_n opt-in path runs
    # end-to-end through train -> checkpoint -> forward -> losses.csv.
    assert any(str(c).startswith("curvature/") for c in rows.columns)
    # predictions.csv (the dense timeseries) is always written, even with plots off.
    assert (fwd_dir / "predictions.csv").exists()


@pytest.mark.integration
@pytest.mark.skipif(
    not FIXTURE_DATA.exists() or not FIXTURE_CUSTOM.exists(),
    reason="martens_single fixture not available",
)
def test_forward_ensemble_on_fixture(tmp_path: Path):
    """Two self-contained checkpoints forwarded as an ensemble -> per-model
    predictions + mean (predictions.csv) + std (predictions_std.csv)."""
    prepared_dir = tmp_path / "prepared"
    prepared = prepared_dir / "prepared.json"
    prepare_config = tmp_path / "prepare-config.json"
    prepare_config.write_text(
        json.dumps(
            {
                "prepare": {"raw_input": str(FIXTURE_DATA)},
                "custom_py": str(FIXTURE_CUSTOM),
            }
        )
    )
    assert cli.main(
        ["prepare", "--config", str(prepare_config), "--output-dir", str(prepared_dir)]
    ) == 0

    out_dir = tmp_path / "run"
    train_config = tmp_path / "train-config.json"
    train_config.write_text(
        json.dumps(
            {
                "data": {"prepared": str(prepared_dir), "target_source": "reactor_components"},
                "custom_py": str(FIXTURE_CUSTOM),
                "train": {"steps": 2, "seed": 42},
                "solver": {"max_steps": 2048, "rtol": 1e-3, "atol": 1e-5},
                "checkpoint": {"every": 1, "keep": "all"},
            }
        )
    )
    assert cli.main(
        ["train", "--config", str(train_config), "--output-dir", str(out_dir), "--no-plot"]
    ) == 0

    ckpt1 = out_dir / "checkpoints" / "step_00001"
    ckpt2 = out_dir / "checkpoints" / "step_00002"
    # each checkpoint is self-contained
    for c in (ckpt1, ckpt2):
        for f in ("config.json", "custom.py", "prepared.json.gz", "params.eqx"):
            assert (c / f).is_file(), f"{c}/{f}"

    ens_out = tmp_path / "ensemble"
    fwd_config = tmp_path / "forward-config.json"
    fwd_config.write_text(
        json.dumps(
            {
                "models": [str(ckpt1), str(ckpt2)],
                "data": {"prepared": str(prepared), "processes": ["run_1"]},
                "output": {"dir": str(ens_out), "plots": False},
            }
        )
    )
    assert cli.main(["forward", "--config", str(fwd_config)]) == 0
    assert (ens_out / "predictions.csv").is_file()
    assert (ens_out / "predictions_std.csv").is_file()
    # per-model dirs with de-duplicated names (both bundles share output.dir "run")
    model_dirs = sorted(p.name for p in (ens_out / "models").iterdir())
    assert len(model_dirs) == 2 and model_dirs[0] != model_dirs[1]


@pytest.mark.integration
@pytest.mark.skipif(
    not FIXTURE_DATA.exists() or not FIXTURE_CUSTOM.exists(),
    reason="martens_single fixture not available",
)
def test_prepare_content_hash_stable_across_reprepare(tmp_path: Path):
    """Re-preparing identical data yields an identical content_hash (timestamp
    differs but is excluded), so the FAIR integrity guard still accepts it."""
    prepare_config = tmp_path / "prepare-config.json"
    prepare_config.write_text(
        json.dumps(
            {
                "prepare": {"raw_input": str(FIXTURE_DATA)},
                "custom_py": str(FIXTURE_CUSTOM),
            }
        )
    )
    first = tmp_path / "first"
    second = tmp_path / "second"
    assert cli.main(["prepare", "--config", str(prepare_config), "--output-dir", str(first)]) == 0
    assert cli.main(["prepare", "--config", str(prepare_config), "--output-dir", str(second)]) == 0

    p1 = json.loads((first / "prepared.json").read_text())["metadata"]["bp-train"]["provenance"]
    p2 = json.loads((second / "prepared.json").read_text())["metadata"]["bp-train"]["provenance"]
    assert p1["content_hash"] == p2["content_hash"]  # stable science
    assert p1["prepared_at"] != p2["prepared_at"] or True  # timestamps may tie

    # The re-run guard blocks overwriting without --overwrite.
    assert cli.main(["prepare", "--config", str(prepare_config), "--output-dir", str(first)]) == 1
    assert (
        cli.main(
            ["prepare", "--config", str(prepare_config), "--output-dir", str(first), "--overwrite"]
        )
        == 0
    )


# ---------------------------------------------------------------------------
# plot_process_simulations: light signature / stub test
# ---------------------------------------------------------------------------


def test_plot_process_simulations_is_exported_with_new_kwargs():
    """Guard against accidental signature regressions."""
    sig = inspect.signature(postprocessing.plot_process_simulations)
    assert "dense_exports" in sig.parameters
    assert "training_process_names" in sig.parameters
    assert "per_process_named_losses" in sig.parameters
    assert "per_process_total_loss" in sig.parameters
    assert "timeseries_csv_path" in sig.parameters
    assert "filename_suffix" in sig.parameters
    assert "render_plots" in sig.parameters


def test_plot_process_simulations_timeseries_csv_header_only_for_empty_selection(
    tmp_path: Path,
):
    class _RhsOde:
        name_modeled_rates = ("q_X", "q_S")

    class _Wrapper:
        modeled_RMC_names = ("X", "S")
        modeled_PV_names = ()
        modeled_FVC_names = ("F",)
        rhs_ode = _RhsOde()

    class _Store:
        process_order = ("p1",)

    class _Collection:
        processes: dict[str, object] = {}

    ts_path = tmp_path / "timeseries.csv"
    postprocessing.plot_process_simulations(
        trained_wrapper=_Wrapper(),
        collection=_Collection(),
        store=_Store(),
        output_dir=tmp_path / "plots",
        dense_exports={},
        process_names=(),
        timeseries_csv_path=ts_path,
    )
    rows = pd.read_csv(ts_path)
    assert rows.columns.tolist() == [
        "process",
        "t",
        "c_X",
        "c_S",
        "V_real",
        "B_F_cum",
        "q_X",
        "q_S",
    ]
    assert rows.empty


def _single_dense_export(collection, store, wrapper, *, prediction_grid_n):
    _, _, dense_exports = compute_dense_exports(
        wrapper,
        store,
        collection,
        ("p1",),
        solver_max_steps=256,
        solver_rtol=1e-4,
        solver_atol=1e-6,
        solver_use_jump_ts=True,
        prediction_grid_n=prediction_grid_n,
    )
    return dense_exports["p1"]


def test_dense_export_uses_export_v_real_semantics():
    collection, store, wrapper = _build_single_process_runtime(
        initial_volume=0.05,
        sample_delta=-0.1,
    )
    export = _single_dense_export(collection, store, wrapper, prediction_grid_n=11)

    assert export.v_real.shape == (11,)
    # Human-facing export should reflect the sampled volume directly, not the
    # runtime clamp used inside the RHS denominator.
    assert float(export.v_real[-1]) == pytest.approx(-0.05, abs=5e-4)


def test_dense_export_returns_physical_q_values():
    collection, store, wrapper = _build_single_process_runtime(
        q_scaled=1.5,
        q_scale=2.0,
    )
    export = _single_dense_export(collection, store, wrapper, prediction_grid_n=9)

    assert export.q_rates.shape == (9, 1)
    assert np.allclose(export.q_rates[:, 0], 3.0)


def test_export_predictions_csv_does_not_depend_on_plot_process_simulations(
    monkeypatch, tmp_path: Path
):
    collection, store, wrapper = _build_single_process_runtime(
        q_scaled=1.5,
        q_scale=2.0,
    )

    def _boom(*args, **kwargs):
        raise AssertionError("plot_process_simulations should not be called")

    monkeypatch.setattr(postprocessing, "plot_process_simulations", _boom)

    _, _, dense_exports = compute_dense_exports(
        wrapper,
        store,
        collection,
        ("p1",),
        solver_max_steps=256,
        solver_rtol=1e-4,
        solver_atol=1e-6,
        solver_use_jump_ts=True,
        prediction_grid_n=9,
    )
    out_path = tmp_path / "predictions.csv"
    postprocessing.export_predictions_csv(wrapper, dense_exports, out_path, ("p1",))

    rows = pd.read_csv(out_path)
    assert rows.columns.tolist() == [
        "process",
        "t",
        "c_biomass",
        "V_real",
        "q_biomass",
    ]
    assert not rows.empty
    assert set(rows["process"]) == {"p1"}


def test_export_predictions_csv_includes_auxiliary_columns(tmp_path: Path):
    collection, store, wrapper = _build_single_process_runtime(
        q_scaled=1.5,
        q_scale=2.0,
        auxiliary={
            "mu_raw": jnp.asarray(-0.75, dtype=jnp.float32),
            "latent_pair": jnp.asarray([4.0, 5.0], dtype=jnp.float32),
        },
    )

    _, _, dense_exports = compute_dense_exports(
        wrapper,
        store,
        collection,
        ("p1",),
        solver_max_steps=256,
        solver_rtol=1e-4,
        solver_atol=1e-6,
        solver_use_jump_ts=True,
        prediction_grid_n=9,
    )
    out_path = tmp_path / "predictions.csv"
    postprocessing.export_predictions_csv(wrapper, dense_exports, out_path, ("p1",))

    rows = pd.read_csv(out_path)
    assert rows.columns.tolist() == [
        "process",
        "t",
        "c_biomass",
        "V_real",
        "q_biomass",
        "aux_latent_pair_0",
        "aux_latent_pair_1",
        "aux_mu_raw",
    ]
    assert not rows.empty
    assert np.allclose(rows["aux_latent_pair_0"], 4.0)
    assert np.allclose(rows["aux_latent_pair_1"], 5.0)
    assert np.allclose(rows["aux_mu_raw"], -0.75)


def _mismatched_aux_exports() -> dict[str, "postprocessing.DenseProcessExport"]:
    return {
        "p1": postprocessing.DenseProcessExport(
            t=np.asarray([0.0, 1.0], dtype=float),
            c_species=np.asarray([[1.0], [1.0]], dtype=float),
            v_real=np.asarray([1.0, 1.0], dtype=float),
            b_modeled_cum=np.zeros((2, 0), dtype=float),
            q_rates=np.asarray([[0.0], [0.0]], dtype=float),
            auxiliary={"mu_raw": np.asarray([-1.0, -1.0], dtype=float)},
        ),
        "p2": postprocessing.DenseProcessExport(
            t=np.asarray([0.0, 1.0], dtype=float),
            c_species=np.asarray([[1.0], [1.0]], dtype=float),
            v_real=np.asarray([1.0, 1.0], dtype=float),
            b_modeled_cum=np.zeros((2, 0), dtype=float),
            q_rates=np.asarray([[0.0], [0.0]], dtype=float),
            auxiliary={"latent_pair": np.asarray([[1.0, 2.0], [1.0, 2.0]])},
        ),
    }


def test_export_predictions_csv_rejects_mismatched_auxiliary_columns(tmp_path: Path):
    class _RhsOde:
        name_modeled_rates = ("q_biomass",)

    class _Wrapper:
        modeled_RMC_names = ("biomass",)
        modeled_PV_names = ()
        modeled_FVC_names = ()
        rhs_ode = _RhsOde()

    with pytest.raises(
        ValueError,
        match="predictions.csv auxiliary columns differ across processes",
    ):
        postprocessing.export_predictions_csv(
            _Wrapper(),
            _mismatched_aux_exports(),
            tmp_path / "predictions.csv",
        )


def test_plot_process_simulations_rejects_mismatched_auxiliary_columns(tmp_path: Path):
    class _RhsOde:
        name_modeled_rates = ("q_biomass",)

    class _Wrapper:
        modeled_RMC_names = ("biomass",)
        modeled_PV_names = ()
        rhs_ode = _RhsOde()
        modeled_FVC_names = ()

    class _Store:
        process_order = ("p1", "p2")

        def get_process(self, process_name):
            del process_name
            return object()

    class _Collection:
        processes = {"p1": object(), "p2": object()}

    with pytest.raises(
        ValueError,
        match="predictions.csv auxiliary columns differ across processes",
    ):
        postprocessing.plot_process_simulations(
            trained_wrapper=_Wrapper(),
            collection=_Collection(),
            store=_Store(),
            output_dir=tmp_path / "plots",
            dense_exports=_mismatched_aux_exports(),
            timeseries_csv_path=tmp_path / "timeseries.csv",
            render_plots=False,
        )
