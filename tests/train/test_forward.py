"""Tests for the `bp-train forward` CLI path and forward harness plumbing.

These tests exercise the pieces that do not require a real trained model:

* the loss-table formatter,
* the metadata sidecar helpers,
* the CLI dispatch for `forward` (via monkeypatching ``forward_from_collection``),
* the sidecar write performed by ``_handle_train``,
* selective dense prediction export.

End-to-end forward (with a real ODE solve) is covered by
``test_forward_end_to_end`` which runs on the kittler example fixture when
available. It is marked ``integration`` so it can be skipped in fast suites.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import jax.numpy as jnp
import numpy as np
import pandas as pd
import pytest
from matplotlib.figure import Figure
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
import bp_train.forward_plotting as forward_plotting
import bp_train.harness as harness_module
from bp_train.controls_store import ControlsStore
from bp_train.harness import ForwardConfig, ForwardResult
from bp_train.defaults import DefaultLossModule
from bp_train.forward_plotting import plot_forward_predictions
from bp_train.harness import compute_dense_exports, evaluate_trained_wrapper
from bp_train.model_api import AffineScaler, ReactionOutputs, UserReactionModule
from bp_train.training_data import TrainingDataStore
from bp_train.wrapper import HybridOdeWrapper


@pytest.mark.parametrize(
    ("mode", "expected"),
    [
        ("none", ()),
        ("parents", ("parent-2", "parent-1")),
        ("all", ("parent-2", "child", "parent-1")),
    ],
)
def test_select_prediction_processes_preserves_evaluation_order(mode, expected):
    evaluated = ("parent-2", "child", "parent-1")
    parents = ("parent-1", "parent-2")

    assert cli._select_prediction_processes(mode, evaluated, parents) == expected


def test_select_prediction_processes_rejects_unknown_scope():
    with pytest.raises(ValueError, match="unknown prediction scope"):
        cli._select_prediction_processes("typo", ("p1",), ("p1",))


def test_evaluate_trained_wrapper_preserves_requested_order_and_labels(monkeypatch):
    store = SimpleNamespace(
        process_order=("train", "holdout"),
        name_modeled_FVCs=("F",),
        name_modeled_SVCs=("S",),
    )
    wrapper = object()

    def fake_dense(trained_wrapper, received_store, process_names, **kwargs):
        assert trained_wrapper is wrapper
        assert received_store is store
        assert process_names == ("holdout", "train")
        assert kwargs["solver_use_jump_ts"] is False
        return (
            np.asarray([2.0, 1.0]),
            np.asarray([[20.0, 21.0], [10.0, 11.0]]),
            {"holdout": object(), "train": object()},
        )

    monkeypatch.setattr("bp_train.harness.compute_dense_exports", fake_dense)
    result = evaluate_trained_wrapper(
        wrapper,
        store,
        config=ForwardConfig(
            process_names=("holdout", "train"), solver_use_jump_ts=False
        ),
        target_names=("loss-b", "loss-a"),
        training_process_names=("train",),
    )

    assert result.process_names == ("holdout", "train")
    assert result.target_names == ("loss-b", "loss-a")
    assert result.training_process_names == ("train",)
    assert list(result.per_process_total_loss) == ["holdout", "train"]
    assert result.per_process_per_target_loss == {
        "holdout": (20.0, 21.0),
        "train": (10.0, 11.0),
    }
    assert list(result.dense_exports) == ["holdout", "train"]


def test_evaluate_trained_wrapper_skips_dense_solve_for_no_predictions(monkeypatch):
    store = SimpleNamespace(
        process_order=("p1",),
        name_modeled_FVCs=(),
        name_modeled_SVCs=(),
        Cin_controlled_FVCs=jnp.zeros((1, 0, 1)),
        Cin_modeled_FVCs=jnp.zeros((1, 0, 1)),
        gather_batch=lambda indices: SimpleNamespace(process_indices=indices),
    )
    monkeypatch.setattr(
        harness_module,
        "compute_dense_exports",
        lambda *_args, **_kwargs: pytest.fail("dense solve should be skipped"),
    )
    monkeypatch.setattr(
        harness_module,
        "_iter_batched_loss_outputs",
        lambda *_args, **_kwargs: iter(
            [
                (
                    ("p1",),
                    (
                        None,
                        np.asarray([[2.0]]),
                        np.asarray([2.0]),
                        None,
                        None,
                        None,
                    ),
                )
            ]
        ),
    )

    result = evaluate_trained_wrapper(
        object(),
        store,
        config=ForwardConfig(process_names=("p1",)),
        target_names=("loss",),
        prediction_process_names=(),
    )

    assert result.per_process_total_loss == {"p1": 2.0}
    assert result.dense_exports == {}


def test_evaluate_trained_wrapper_loss_only_batches_exclude_padding(monkeypatch):
    process_names = tuple(f"p{i}" for i in range(35))

    class _Store:
        process_order = process_names
        name_modeled_FVCs = ()
        name_modeled_SVCs = ()
        Cin_controlled_FVCs = jnp.zeros((35, 0, 1))
        Cin_modeled_FVCs = jnp.zeros((35, 0, 1))

        @staticmethod
        def gather_batch(indices):
            return SimpleNamespace(process_indices=indices)

    seen_indices = []

    def fake_batched_loss(wrapper, batch, *args, **kwargs):
        del wrapper, args, kwargs
        indices = jnp.asarray(batch.process_indices)
        seen_indices.append(np.asarray(indices))
        return (
            jnp.mean(indices),
            indices[:, None],
            indices,
            None,
            None,
            None,
        )

    monkeypatch.setattr(harness_module, "_BATCHED_LOSS_FN_JIT", fake_batched_loss)
    monkeypatch.setattr(
        harness_module,
        "compute_dense_exports",
        lambda *_args, **_kwargs: pytest.fail("dense solve should be skipped"),
    )

    result = evaluate_trained_wrapper(
        object(),
        _Store(),
        config=ForwardConfig(process_names=process_names, solver_use_jump_ts=False),
        target_names=("loss",),
        prediction_process_names=(),
    )

    assert [len(indices) for indices in seen_indices] == [32, 32]
    np.testing.assert_array_equal(seen_indices[1][:3], [32, 33, 34])
    np.testing.assert_array_equal(seen_indices[1][3:], np.full(29, 34))
    assert list(result.per_process_total_loss) == list(process_names)
    assert list(result.per_process_total_loss.values()) == list(range(35))


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
    strict_output = path.read_text(encoding="utf-8")
    assert json.loads(strict_output) == meta
    path.write_text("// model metadata\n" + strict_output, encoding="utf-8")

    loaded = postprocessing.load_model_metadata(path)
    assert loaded == meta


def test_save_model_metadata_normalizes_nonfinite_loss(tmp_path: Path):
    path = tmp_path / "trained_wrapper.meta.json"
    postprocessing.save_model_metadata(
        path, {"training": {"final_mean_loss": float("inf")}}
    )

    text = path.read_text(encoding="utf-8")
    assert "Infinity" not in text
    assert json.loads(text)["training"]["final_mean_loss"] is None


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


def test_write_train_results_consumes_forward_result_without_reconstruction(
    monkeypatch, tmp_path
):
    result = _make_forward_result()
    exported = {}
    monkeypatch.setattr(
        cli,
        "forward_from_collection",
        lambda *_a, **_k: pytest.fail("redundant forward reconstruction called"),
    )
    monkeypatch.setattr(
        cli,
        "export_predictions_csv",
        lambda wrapper, dense, path, *, process_names: exported.update(
            wrapper=wrapper,
            dense=dense,
            path=path,
            process_names=process_names,
        ),
    )

    cli._write_train_results(
        output_dir=tmp_path,
        forward_result=result,
        prediction_processes=("p1", "p2"),
    )

    assert pd.read_csv(tmp_path / "losses.csv")["process"].iloc[:2].tolist() == [
        "p1",
        "p2",
    ]
    assert exported["process_names"] == ("p1", "p2")


def test_write_train_results_removes_stale_predictions_for_empty_scope(tmp_path):
    predictions_path = tmp_path / "predictions.csv"
    predictions_path.write_text("stale", encoding="utf-8")

    cli._write_train_results(
        output_dir=tmp_path,
        forward_result=_make_forward_result(),
        prediction_processes=(),
    )

    assert not predictions_path.exists()
    assert (tmp_path / "losses.csv").is_file()


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
        processes = {
            "p1": SimpleNamespace(),
            "p2": SimpleNamespace(),
            "p3": SimpleNamespace(),
        }

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
    "SCALE_modeled_RMCs": jnp.ones(1),
    "SCALE_V_in_cumulative": jnp.asarray(1.0),
    "SCALE_modeled_FVCs_cumulative": jnp.ones(0),
    "SCALE_controlled_FVCs_cumulative": jnp.ones(0),
    "SCALE_controlled_FVCs_rates": jnp.ones(0),
    "SCALE_controlled_FVCs_Cin": jnp.ones((0, 1)),
    "SCALE_controlled_PVs": jnp.ones(0),
    "SCALE_modeled_FVCs_Cin": jnp.ones((0, 1)),
    "SCALE_modeled_BiologicalOde_rates": jnp.ones(1),
    "SCALE_modeled_FVCs_rates": jnp.ones(0),
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
    biomass_times: jnp.ndarray | None = None,
) -> BioProcess:
    biomass_times = (
        jnp.asarray([0.0, 2.0]) if biomass_times is None else jnp.asarray(biomass_times)
    )
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
                        times=biomass_times,
                        values=jnp.ones_like(biomass_times),
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
    biomass_times: jnp.ndarray | None = None,
    modeled_rmc_scaler=None,
):
    process = _make_one_species_process(
        initial_volume=initial_volume,
        sample_delta=sample_delta,
        biomass_times=biomass_times,
    )
    collection = BioProcessCollection(processes={"p1": process}, metadata={})
    controls = ControlsStore.from_collection(collection).get_controls("p1")
    store = TrainingDataStore.from_collection(
        collection,
        target_variable_order=["biomass"],
        target_source="reactor_components",
    )
    scale_kwargs = {"SCALE_modeled_BiologicalOde_rates": jnp.asarray([q_scale])}
    if modeled_rmc_scaler is not None:
        scale_kwargs["SCALE_modeled_RMCs"] = modeled_rmc_scaler
    wrapper = HybridOdeWrapper.from_process(
        reaction_module=_ConstantReactionModule(
            specific_rates=jnp.asarray([q_scaled]),
            modeled_feed_rates=jnp.zeros((0,)),
            auxiliary=auxiliary,
            **scale_kwargs,
        ),
        process=process,
        controls=controls,
        loss_module=DefaultLossModule(target_names=tuple(store.name_measured)),
    )
    return collection, store, wrapper


def _write_forward_config(
    tmp_path: Path,
    model_dirs,
    *,
    processes=None,
    output_dir=None,
    prepared=None,
    predictions="parents",
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
    output: dict = {"predictions": predictions, "plots": plots}
    if output_dir is not None:
        output["dir"] = str(output_dir)
    cfg["output"] = output
    path = tmp_path / name
    path.write_text(json.dumps(cfg), encoding="utf-8")
    return path


def test_forward_cli_rejects_unknown_process_before_evaluation(monkeypatch, tmp_path):
    run_dir = _make_forward_run_dir(tmp_path, processes=("p1", "p2"))
    monkeypatch.setattr(
        cli, "load_process_collection", lambda _p: _make_fake_collection()
    )
    monkeypatch.setattr(
        cli,
        "forward_from_collection",
        lambda *_args, **_kwargs: pytest.fail("unknown process reached evaluation"),
    )
    config = _write_forward_config(tmp_path, [run_dir], processes=("missing",))

    assert cli.main(["forward", "--config", str(config)]) == 1


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
        captured["prediction_process_names"] = kwargs["prediction_process_names"]
        return _stub_forward_result()

    monkeypatch.setattr(cli, "forward_from_collection", fake_forward)
    # the stub ForwardResult has no real wrapper/dense_exports; skip the writers
    monkeypatch.setattr(cli, "export_predictions_csv", lambda *a, **k: None)

    output_dir = tmp_path / "fwd"
    output_dir.mkdir()
    reference_dir = output_dir / "reference-dir"
    reference_dir.mkdir()
    unrelated = output_dir / "unrelated.txt"
    unrelated.write_text("keep", encoding="utf-8")
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
    assert captured["prediction_process_names"] == ("p1", "p2")
    assert captured["model_path"] == run_dir / "model" / "params.eqx"
    assert unrelated.read_text(encoding="utf-8") == "keep"

    results_dir = output_dir / "forward-results"
    assert (results_dir.stat().st_mode & 0o7777) == (
        reference_dir.stat().st_mode & 0o7777
    )
    losses_csv = results_dir / "losses.csv"
    assert losses_csv.exists()
    rows = pd.read_csv(losses_csv)
    assert rows.columns.tolist() == ["process", "total", "X", "S", "split"]
    assert ((rows["process"] == "p1") & (rows["split"] == "train")).any()


def test_forward_preserves_prior_results_when_result_writing_fails(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    run_dir = _make_forward_run_dir(tmp_path)
    output_dir = tmp_path / "forward"
    results_dir = output_dir / "forward-results"
    results_dir.mkdir(parents=True)
    prior = results_dir / "prior.txt"
    prior.write_text("prior", encoding="utf-8")
    unrelated = output_dir / "unrelated.txt"
    unrelated.write_text("keep", encoding="utf-8")
    config = _write_forward_config(tmp_path, [run_dir], output_dir=output_dir)

    monkeypatch.setattr(
        cli, "load_process_collection", lambda _path: _make_fake_collection()
    )
    monkeypatch.setattr(
        cli, "forward_from_collection", lambda *_args, **_kwargs: _stub_forward_result()
    )

    def fail_writing(*_args, **_kwargs):
        raise OSError("disk full")

    monkeypatch.setattr(cli, "_write_forward_results", fail_writing)

    with pytest.raises(OSError, match="disk full"):
        cli.main(["forward", "--config", str(config), "--overwrite"])

    assert list(output_dir.glob(".forward-results-*")) == []
    assert prior.read_text(encoding="utf-8") == "prior"
    assert unrelated.read_text(encoding="utf-8") == "keep"


def test_publish_forward_results_restores_prior_results(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    results_dir = tmp_path / "forward-results"
    results_dir.mkdir()
    (results_dir / "prior.txt").write_text("prior", encoding="utf-8")
    staging_dir = tmp_path / ".forward-results-new"
    staging_dir.mkdir()
    (staging_dir / "new.txt").write_text("new", encoding="utf-8")
    replace = Path.replace

    def fail_staged_publication(path, target):
        if path == staging_dir:
            raise OSError("publication failed")
        return replace(path, target)

    monkeypatch.setattr(Path, "replace", fail_staged_publication)

    with pytest.raises(OSError, match="publication failed"):
        cli._publish_forward_results(staging_dir, results_dir)

    assert (results_dir / "prior.txt").read_text(encoding="utf-8") == "prior"
    assert staging_dir.is_dir()
    assert list(tmp_path.glob(".forward-results-old-*")) == []


def test_publish_forward_results_reports_backup_cleanup_failure(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    results_dir = tmp_path / "forward-results"
    results_dir.mkdir()
    (results_dir / "prior.txt").write_text("prior", encoding="utf-8")
    staging_dir = tmp_path / ".forward-results-new"
    staging_dir.mkdir()
    (staging_dir / "new.txt").write_text("new", encoding="utf-8")
    rmtree = cli.shutil.rmtree

    def fail_backup_cleanup(path, *args, **kwargs):
        if Path(path).name.startswith(".forward-results-old-"):
            raise OSError("cleanup failed")
        return rmtree(path, *args, **kwargs)

    monkeypatch.setattr(cli.shutil, "rmtree", fail_backup_cleanup)

    with pytest.raises(OSError, match="cleanup failed"):
        cli._publish_forward_results(staging_dir, results_dir)

    assert (results_dir / "new.txt").read_text(encoding="utf-8") == "new"
    backups = list(tmp_path.glob(".forward-results-old-*"))
    assert len(backups) == 1
    assert (backups[0] / "forward-results" / "prior.txt").read_text(
        encoding="utf-8"
    ) == "prior"


def test_forward_cli_plots_selected_predictions(monkeypatch, tmp_path: Path):
    run_dir = _make_forward_run_dir(tmp_path)
    collection = _make_fake_collection()
    dense_exports = {"p1": object(), "p2": object()}
    result = _stub_forward_result(dense_exports=dense_exports)
    monkeypatch.setattr(cli, "load_process_collection", lambda _path: collection)
    monkeypatch.setattr(
        cli, "forward_from_collection", lambda *_args, **_kwargs: result
    )
    monkeypatch.setattr(cli, "export_predictions_csv", lambda *_args, **_kwargs: None)

    calls = []
    monkeypatch.setattr(
        cli,
        "plot_forward_predictions",
        lambda *args: calls.append(args),
    )
    output_dir = tmp_path / "forward"
    config = _write_forward_config(
        tmp_path,
        [run_dir],
        output_dir=output_dir,
        plots=True,
    )

    assert cli.main(["forward", "--config", str(config)]) == 0
    assert len(calls) == 1
    assert calls[0][:-1] == (
        collection,
        result.trained_wrapper,
        dense_exports,
        None,
        {
            "p1": (0.1, {"X": 0.05, "S": 0.15}),
            "p2": (0.2, {"X": 0.1, "S": 0.3}),
        },
    )
    staged_results = Path(calls[0][-1])
    assert staged_results.name == "forward-results"
    assert staged_results.parent.parent == output_dir
    assert staged_results.parent.name.startswith(".forward-results-")


def test_forward_model_names_are_unique_across_explicit_suffixes():
    refs = (
        SimpleNamespace(name="foo", path="first"),
        SimpleNamespace(name="foo#2", path="second"),
        SimpleNamespace(name="foo", path="third"),
    )

    assert cli._resolve_model_names(refs) == ["foo", "foo#2", "foo#3"]


@pytest.mark.parametrize("unsafe_name", ["../..", "bad\0name"])
def test_forward_rejects_unsafe_model_name_before_writing(
    monkeypatch, tmp_path, unsafe_name
):
    run_dir = _make_forward_run_dir(tmp_path / "model")
    output_dir = tmp_path / "output"
    results_dir = output_dir / "forward-results"
    results_dir.mkdir(parents=True)
    unrelated = output_dir / "losses.csv"
    unrelated.write_text("keep", encoding="utf-8")
    config = _write_forward_config(
        tmp_path,
        [run_dir],
        output_dir=output_dir,
        name="unsafe-name-config.json",
    )
    document = json.loads(config.read_text(encoding="utf-8"))
    document["models"] = [{"name": unsafe_name, "path": str(run_dir)}]
    config.write_text(json.dumps(document), encoding="utf-8")
    monkeypatch.setattr(
        cli,
        "forward_from_collection",
        lambda *_args, **_kwargs: pytest.fail("unsafe model reached evaluation"),
    )

    assert cli.main(["forward", "--config", str(config), "--overwrite"]) == 1
    assert unrelated.read_text(encoding="utf-8") == "keep"
    assert results_dir.is_dir()


@pytest.mark.parametrize("input_kind", ["prepared", "custom_py"])
def test_forward_overwrite_preserves_model_config_input(tmp_path, input_kind):
    run_dir = _make_forward_run_dir(tmp_path / "model")
    output_dir = tmp_path / "output"
    results_dir = output_dir / "forward-results"
    results_dir.mkdir(parents=True)
    nested_input = results_dir / f"input-{input_kind}"
    nested_input.write_text("{}", encoding="utf-8")
    run_config = run_dir / "config.json"
    document = json.loads(run_config.read_text(encoding="utf-8"))
    if input_kind == "prepared":
        (run_dir / "prepared.json").unlink()
        document["config"]["data"]["prepared"] = str(nested_input)
    else:
        document["config"]["custom_py"] = str(nested_input)
    run_config.write_text(json.dumps(document), encoding="utf-8")
    config = _write_forward_config(
        tmp_path,
        [run_dir],
        output_dir=output_dir,
        name=f"nested-{input_kind}-config.json",
    )

    with pytest.raises(ValueError, match="contains input file"):
        cli.main(["forward", "--config", str(config), "--overwrite"])
    assert nested_input.is_file()


@pytest.mark.parametrize("input_kind", ["prepared", "custom_py"])
def test_forward_overwrite_normalizes_input_spelled_through_results(
    monkeypatch, tmp_path, input_kind
):
    run_dir = _make_forward_run_dir(tmp_path / "model")
    output_dir = tmp_path / "output"
    results_dir = output_dir / "forward-results"
    results_dir.mkdir(parents=True)
    retained_input = output_dir / f"input-{input_kind}"
    retained_input.write_text("{}", encoding="utf-8")
    indirect_input = results_dir / ".." / retained_input.name
    run_config = run_dir / "config.json"
    document = json.loads(run_config.read_text(encoding="utf-8"))
    if input_kind == "prepared":
        (run_dir / "prepared.json").unlink()
        document["config"]["data"]["prepared"] = str(indirect_input)
    else:
        document["config"]["custom_py"] = str(indirect_input)
    run_config.write_text(json.dumps(document), encoding="utf-8")
    config = _write_forward_config(
        tmp_path,
        [run_dir],
        output_dir=output_dir,
        predictions="none",
        name=f"indirect-{input_kind}-config.json",
    )
    seen = {}

    def fake_load(path):
        if input_kind == "prepared":
            seen["input"] = Path(path)
        return _make_fake_collection()

    def fake_forward(*_args, **kwargs):
        if input_kind == "custom_py":
            seen["input"] = Path(kwargs["custom_py"])
        return _stub_forward_result()

    monkeypatch.setattr(cli, "load_process_collection", fake_load)
    monkeypatch.setattr(cli, "forward_from_collection", fake_forward)

    assert cli.main(["forward", "--config", str(config), "--overwrite"]) == 0
    assert seen["input"] == retained_input
    assert retained_input.is_file()


def test_forward_overwrite_preserves_prepared_resolved_from_directory(tmp_path):
    run_dir = _make_forward_run_dir(tmp_path / "model")
    output_dir = tmp_path / "output"
    results_dir = output_dir / "forward-results"
    results_dir.mkdir(parents=True)
    nested_prepared = results_dir / "prepared-target.json"
    nested_prepared.write_text("{}", encoding="utf-8")
    external_prepared_dir = tmp_path / "external-prepared"
    external_prepared_dir.mkdir()
    (external_prepared_dir / "prepared.json").symlink_to(nested_prepared)
    (run_dir / "prepared.json").unlink()
    run_config = run_dir / "config.json"
    document = json.loads(run_config.read_text(encoding="utf-8"))
    document["config"]["data"]["prepared"] = str(external_prepared_dir)
    run_config.write_text(json.dumps(document), encoding="utf-8")
    config = _write_forward_config(
        tmp_path,
        [run_dir],
        output_dir=output_dir,
        name="prepared-directory-config.json",
    )

    with pytest.raises(ValueError, match="contains input file"):
        cli.main(["forward", "--config", str(config), "--overwrite"])
    assert nested_prepared.is_file()


def test_forward_overwrite_preserves_resolved_params_input(tmp_path):
    run_dir = _make_forward_run_dir(tmp_path / "model")
    output_dir = tmp_path / "output"
    results_dir = output_dir / "forward-results"
    results_dir.mkdir(parents=True)
    nested_params = results_dir / "params-input.eqx"
    params = run_dir / "model" / "params.eqx"
    nested_params.write_bytes(params.read_bytes())
    params.unlink()
    params.symlink_to(nested_params)
    config = _write_forward_config(
        tmp_path,
        [run_dir],
        output_dir=output_dir,
        name="nested-params-config.json",
    )

    with pytest.raises(ValueError, match="contains input file"):
        cli.main(["forward", "--config", str(config), "--overwrite"])
    assert nested_params.is_file()
    assert params.is_file()


def test_forward_overwrite_preserves_input_symlink_in_results(tmp_path):
    run_dir = _make_forward_run_dir(tmp_path / "model")
    output_dir = tmp_path / "output"
    results_dir = output_dir / "forward-results"
    results_dir.mkdir(parents=True)
    external_prepared = tmp_path / "external-prepared.json"
    external_prepared.write_text("{}", encoding="utf-8")
    nested_link = results_dir / "prepared-link.json"
    nested_link.symlink_to(external_prepared)
    (run_dir / "prepared.json").unlink()
    run_config = run_dir / "config.json"
    document = json.loads(run_config.read_text(encoding="utf-8"))
    document["config"]["data"]["prepared"] = str(nested_link)
    run_config.write_text(json.dumps(document), encoding="utf-8")
    config = _write_forward_config(
        tmp_path,
        [run_dir],
        output_dir=output_dir,
        name="nested-symlink-config.json",
    )

    with pytest.raises(ValueError, match="contains input file"):
        cli.main(["forward", "--config", str(config), "--overwrite"])
    assert nested_link.is_symlink()
    assert external_prepared.is_file()


def test_forward_overwrite_preserves_input_through_results_symlink(tmp_path):
    run_dir = _make_forward_run_dir(tmp_path / "model")
    output_dir = tmp_path / "output"
    output_dir.mkdir()
    external_results = tmp_path / "existing-forward-results"
    external_results.mkdir()
    prepared = external_results / "prepared.json"
    prepared.write_text("{}", encoding="utf-8")
    results_dir = output_dir / "forward-results"
    results_dir.symlink_to(external_results, target_is_directory=True)
    (run_dir / "prepared.json").unlink()
    run_config = run_dir / "config.json"
    document = json.loads(run_config.read_text(encoding="utf-8"))
    document["config"]["data"]["prepared"] = str(results_dir / "prepared.json")
    run_config.write_text(json.dumps(document), encoding="utf-8")
    config = _write_forward_config(
        tmp_path,
        [run_dir],
        output_dir=output_dir,
        name="results-symlink-config.json",
    )

    with pytest.raises(ValueError, match="contains input file"):
        cli.main(["forward", "--config", str(config), "--overwrite"])
    assert results_dir.is_symlink()
    assert prepared.is_file()


def test_forward_plot_losses_keep_named_total_separate():
    result = _stub_forward_result(
        target_names=("total",),
        per_process_per_target_loss={"p1": (0.05,), "p2": (0.1,)},
    )

    assert cli._aggregate_forward_plot_losses([result]) == {
        "p1": (0.1, {"total": 0.05}),
        "p2": (0.2, {"total": 0.1}),
    }


def test_forward_plot_losses_align_ensemble_members_by_name():
    first = _stub_forward_result(
        process_names=("p1",),
        target_names=("X", "S"),
        per_process_total_loss={"p1": 3.0},
        per_process_per_target_loss={"p1": (1.0, 5.0)},
    )
    second = _stub_forward_result(
        process_names=("p1",),
        target_names=("S", "X"),
        per_process_total_loss={"p1": 3.0},
        per_process_per_target_loss={"p1": (5.0, 1.0)},
    )

    assert cli._aggregate_forward_plot_losses([first, second]) == {
        "p1": (3.0, {"X": 1.0, "S": 5.0})
    }


def test_forward_omits_plots_when_scope_selects_no_process(monkeypatch, tmp_path: Path):
    run_dir = _make_forward_run_dir(tmp_path)
    collection = SimpleNamespace(
        processes={
            "p1": SimpleNamespace(parent_process="parent"),
            "p2": SimpleNamespace(parent_process="parent"),
        }
    )
    monkeypatch.setattr(cli, "load_process_collection", lambda _path: collection)
    monkeypatch.setattr(
        cli,
        "forward_from_collection",
        lambda *_args, **_kwargs: _stub_forward_result(dense_exports={}),
    )
    monkeypatch.setattr(
        cli,
        "export_predictions_csv",
        lambda *_args, **_kwargs: pytest.fail("empty scope exported predictions"),
    )
    monkeypatch.setattr(
        cli,
        "plot_forward_predictions",
        lambda *_args: pytest.fail("empty scope rendered plots"),
    )
    output_dir = tmp_path / "forward"
    config = _write_forward_config(
        tmp_path,
        [run_dir],
        output_dir=output_dir,
        predictions="parents",
        plots=True,
    )

    assert cli.main(["forward", "--config", str(config)]) == 0
    assert not (output_dir / "forward-results" / "plots").exists()


def test_forward_cli_plot_failure_is_nonfatal(monkeypatch, tmp_path: Path):
    run_dir = _make_forward_run_dir(tmp_path)
    monkeypatch.setattr(
        cli, "load_process_collection", lambda _path: _make_fake_collection()
    )
    monkeypatch.setattr(
        cli,
        "forward_from_collection",
        lambda *_args, **_kwargs: _stub_forward_result(
            dense_exports={"p1": object(), "p2": object()}
        ),
    )
    monkeypatch.setattr(cli, "export_predictions_csv", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(
        cli,
        "plot_forward_predictions",
        lambda *_args: (_ for _ in ()).throw(OSError("disk full")),
    )
    output_dir = tmp_path / "forward"
    config = _write_forward_config(
        tmp_path,
        [run_dir],
        output_dir=output_dir,
        plots=True,
    )

    assert cli.main(["forward", "--config", str(config)]) == 0
    assert (output_dir / "forward-results" / "losses.csv").is_file()


@pytest.mark.parametrize(
    ("mode", "expected_processes"),
    [
        ("none", ()),
        ("parents", ("p1", "p3")),
        ("all", ("p1", "p2", "p3")),
    ],
)
def test_ensemble_forward_applies_prediction_scope(
    monkeypatch, tmp_path: Path, mode, expected_processes
):
    run_dirs = (
        _make_forward_run_dir(tmp_path / "first", processes=("p1", "p2", "p3")),
        _make_forward_run_dir(tmp_path / "second", processes=("p1", "p2", "p3")),
    )
    collection = SimpleNamespace(
        processes={
            "p1": SimpleNamespace(parent_process=None),
            "p2": SimpleNamespace(parent_process="p1"),
            "p3": SimpleNamespace(parent_process=None),
        }
    )
    monkeypatch.setattr(cli, "load_process_collection", lambda _path: collection)

    forwarded = []

    def fake_forward(_collection, **kwargs):
        forwarded.append(kwargs["prediction_process_names"])
        return _stub_forward_result(
            process_names=("p1", "p2", "p3"),
            training_process_names=("p1", "p2", "p3"),
            per_process_total_loss={"p1": 0.1, "p2": 0.2, "p3": 0.3},
            per_process_per_target_loss={
                "p1": (0.05, 0.15),
                "p2": (0.1, 0.3),
                "p3": (0.15, 0.45),
            },
            dense_exports={name: object() for name in expected_processes},
        )

    monkeypatch.setattr(cli, "forward_from_collection", fake_forward)
    monkeypatch.setattr(
        cli,
        "aggregate_dense_exports",
        lambda exports: (exports[0], exports[0]),
    )
    exported = []
    monkeypatch.setattr(
        cli,
        "export_predictions_csv",
        lambda _wrapper, _dense, _path, *, process_names: exported.append(
            process_names
        ),
    )
    prepared = tmp_path / "prepared.json"
    prepared.write_text("{}", encoding="utf-8")
    config = _write_forward_config(
        tmp_path,
        run_dirs,
        processes=("p1", "p2", "p3"),
        output_dir=tmp_path / "output",
        prepared=prepared,
        predictions=mode,
    )

    output_dir = tmp_path / "output"
    assert cli.main(["forward", "--config", str(config)]) == 0
    assert forwarded == [expected_processes, expected_processes]
    assert exported == ([expected_processes] * 4 if expected_processes else [])
    results_dir = output_dir / "forward-results"
    assert (results_dir / "losses.csv").is_file()
    assert len(list((results_dir / "models").glob("*/losses.csv"))) == 2
    if not expected_processes:
        assert not (results_dir / "predictions.csv").exists()
        assert not (results_dir / "predictions_std.csv").exists()
        assert not list((results_dir / "models").glob("*/predictions.csv"))


def test_forward_cli_overwrite_guard(monkeypatch, tmp_path: Path):
    """Only forward-results is replaced by --overwrite."""
    run_dir = _make_forward_run_dir(tmp_path, processes=("p1", "p2"))
    monkeypatch.setattr(
        cli, "load_process_collection", lambda p: _make_fake_collection()
    )
    monkeypatch.setattr(
        cli, "forward_from_collection", lambda collection, **k: _stub_forward_result()
    )
    monkeypatch.setattr(cli, "export_predictions_csv", lambda *a, **k: None)

    output_dir = tmp_path / "fwd"
    fwd_config = _write_forward_config(tmp_path, [run_dir])
    base = ["forward", "--config", str(fwd_config), "--output-dir", str(output_dir)]

    assert cli.main(base) == 0
    results_dir = output_dir / "forward-results"
    assert (results_dir / "losses.csv").is_file()
    unrelated = output_dir / "keep.txt"
    unrelated.write_text("keep", encoding="utf-8")

    assert cli.main(base) == 1
    stale = results_dir / "stale.txt"
    stale.write_text("stale", encoding="utf-8")
    write_results = cli._write_forward_results

    def write_while_prior_results_exist(*args, **kwargs):
        assert stale.is_file()
        write_results(*args, **kwargs)

    monkeypatch.setattr(cli, "_write_forward_results", write_while_prior_results_exist)
    assert cli.main(base + ["--overwrite"]) == 0
    assert not stale.exists()
    assert unrelated.read_text(encoding="utf-8") == "keep"


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


def test_forward_cli_no_configured_processes_evaluates_all(monkeypatch, tmp_path: Path):
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
    monkeypatch.setattr(cli, "export_predictions_csv", lambda *a, **k: None)

    fwd_config = _write_forward_config(tmp_path, [run_dir])
    cli.main(
        ["forward", "--config", str(fwd_config), "--output-dir", str(tmp_path / "fwd")]
    )
    assert captured_tpn["tpn"] == ("p1", "p2", "p3")


def test_export_predictions_csv_header_only_for_empty_selection(
    tmp_path: Path,
):
    class _RhsOde:
        name_modeled_rates = ("q_X", "q_S")

    class _Wrapper:
        modeled_RMC_names = ("X", "S")
        modeled_PV_names = ()
        modeled_FVC_names = ("F",)
        rhs_ode = _RhsOde()

    ts_path = tmp_path / "timeseries.csv"
    postprocessing.export_predictions_csv(
        _Wrapper(),
        {},
        ts_path,
        process_names=(),
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


@pytest.mark.parametrize("process_count", [3, 35])
def test_dense_exports_batch_and_exclude_padded_tail(monkeypatch, process_count):
    process_names = tuple(f"p{i}" for i in range(process_count))

    class _Store:
        process_order = process_names
        Cin_controlled_FVCs = jnp.zeros((process_count, 0, 1))
        Cin_modeled_FVCs = jnp.zeros((process_count, 0, 1))

        @staticmethod
        def gather_batch(indices):
            return SimpleNamespace(process_indices=indices)

    seen_indices = []

    def fake_batched_loss(
        wrapper,
        batch,
        cin,
        cin_modeled,
        jump_ts_rows,
        **kwargs,
    ):
        del wrapper, cin, cin_modeled, jump_ts_rows, kwargs
        indices = jnp.asarray(batch.process_indices)
        seen_indices.append(np.asarray(indices))
        return (
            jnp.mean(indices),
            jnp.stack((indices, indices + 100), axis=1),
            indices,
            indices[:, None],
            indices[:, None],
            jnp.full(indices.shape, jnp.inf),
        )

    def fake_dense_exports(prediction_t, save_outputs, wrapper, names):
        del wrapper
        np.testing.assert_array_equal(prediction_t[:, 0], save_outputs[:, 0])
        assert prediction_t.shape[0] == len(names)
        return {name: int(prediction_t[i, 0]) for i, name in enumerate(names)}

    monkeypatch.setattr(harness_module, "_BATCHED_LOSS_FN_JIT", fake_batched_loss)
    monkeypatch.setattr(
        harness_module,
        "dense_exports_from_save_outputs",
        fake_dense_exports,
    )

    totals, per_target, dense = harness_module.compute_dense_exports(
        object(),
        _Store(),
        process_names,
        solver_max_steps=10,
        solver_rtol=1e-4,
        solver_atol=1e-6,
        solver_use_jump_ts=False,
    )

    assert [len(indices) for indices in seen_indices] == [32] * (
        (process_count + 31) // 32
    )
    for batch_index, indices in enumerate(seen_indices):
        start = batch_index * 32
        valid = min(32, process_count - start)
        np.testing.assert_array_equal(indices[:valid], np.arange(start, start + valid))
        np.testing.assert_array_equal(
            indices[valid:], np.full(32 - valid, process_count - 1)
        )
    np.testing.assert_array_equal(totals, np.arange(process_count))
    np.testing.assert_array_equal(per_target[:, 0], np.arange(process_count))
    assert list(dense) == list(process_names)
    assert list(dense.values()) == list(range(process_count))


def test_dense_exports_reject_duplicate_process_names():
    class _Store:
        process_order = ("p1",)

    with pytest.raises(ValueError, match="duplicate process names.*p1"):
        harness_module.compute_dense_exports(
            object(),
            _Store(),
            ("p1", "p1"),
            solver_max_steps=10,
            solver_rtol=1e-4,
            solver_atol=1e-6,
            solver_use_jump_ts=False,
        )


def _single_dense_export(store, wrapper, *, prediction_grid_n):
    _, _, dense_exports = compute_dense_exports(
        wrapper,
        store,
        ("p1",),
        solver_max_steps=256,
        solver_rtol=1e-4,
        solver_atol=1e-6,
        solver_use_jump_ts=True,
        prediction_grid_n=prediction_grid_n,
    )
    return dense_exports["p1"]


def test_plot_forward_predictions_writes_process_png(tmp_path: Path):
    collection, store, wrapper = _build_single_process_runtime(q_scaled=0.5)
    export = _single_dense_export(store, wrapper, prediction_grid_n=7)

    plot_forward_predictions(
        collection,
        wrapper,
        {"p1": export},
        None,
        {"p1": (0.1, {"biomass": 0.1})},
        tmp_path,
    )

    path = tmp_path / "plots" / "p1.png"
    assert path.read_bytes().startswith(b"\x89PNG\r\n\x1a\n")


def test_plot_forward_predictions_plots_every_rate(monkeypatch, tmp_path: Path):
    collection, store, wrapper = _build_single_process_runtime(q_scaled=0.5)
    export = _single_dense_export(store, wrapper, prediction_grid_n=7)
    rate_values = np.column_stack(
        [np.full(export.t.shape, value) for value in (1.0, 2.0, 3.0)]
    )
    export = postprocessing.DenseProcessExport(
        t=export.t,
        c_species=export.c_species,
        v_real=export.v_real,
        b_modeled_cum=export.b_modeled_cum,
        q_rates=rate_values,
    )
    plot_wrapper = SimpleNamespace(
        modeled_RMC_names=wrapper.modeled_RMC_names,
        modeled_PV_names=wrapper.modeled_PV_names,
        modeled_FVC_names=wrapper.modeled_FVC_names,
        rhs_ode=SimpleNamespace(name_modeled_rates=("r1", "r2", "r3")),
    )
    plotted = []
    original = forward_plotting._plot_prediction

    def record_prediction(axis, t, mean, std, measured, **kwargs):
        plotted.append(np.asarray(mean))
        return original(axis, t, mean, std, measured, **kwargs)

    monkeypatch.setattr(forward_plotting, "_plot_prediction", record_prediction)

    plot_forward_predictions(
        collection,
        plot_wrapper,
        {"p1": export},
        None,
        {"p1": (float("nan"), {})},
        tmp_path,
    )

    assert len(plotted) == 4  # one modeled state plus three independent rates
    for index, value in enumerate((1.0, 2.0, 3.0), start=1):
        assert np.all(plotted[index] == value)


@pytest.mark.parametrize("measured_y", [np.array([1.0]), np.array([1.0, 1.0])])
def test_fit_title_marks_undefined_r_squared(measured_y):
    axis = SimpleNamespace(
        plot=lambda *_args, **_kwargs: None,
        scatter=lambda *_args, **_kwargs: None,
        legend=lambda *_args, **_kwargs: None,
    )
    measured_t = np.arange(len(measured_y), dtype=float)

    r_squared = forward_plotting._plot_prediction(
        axis,
        np.array([0.0, 1.0]),
        np.array([1.0, 1.0]),
        None,
        (measured_t, measured_y),
    )

    assert np.isnan(r_squared)
    title = forward_plotting._fit_title("biomass", loss=0.1, r_squared=r_squared)
    assert "R²=undefined" in title
    assert "loss[biomass]=0.1" in title


def test_plot_prediction_keeps_measured_points():
    figure = Figure()
    axis = figure.subplots()

    forward_plotting._plot_prediction(
        axis,
        np.array([0.0, 1.0]),
        np.array([1.0, 2.0]),
        None,
        (np.array([0.0, 1.0]), np.array([1.1, 1.9])),
    )

    assert len(axis.collections) == 1


def test_continuous_volume_change_is_relative_to_process_start():
    change = SimpleNamespace(
        is_continuous=True,
        values=SimpleNamespace(
            breaks=None,
            times=np.array([0.0, 10.0]),
            values=np.array([100.0, 102.0]),
        ),
    )

    cumulative = forward_plotting._volume_change_cumulative(
        change, np.array([5.0, 10.0]), process_start=5.0
    )

    np.testing.assert_allclose(cumulative, [0.0, 1.0])


def test_plot_volume_changes_samples_nonlinear_spline_on_dense_grid():
    t = np.linspace(0.0, 2.0, 5)
    change = SimpleNamespace(
        is_continuous=True,
        values=SimpleNamespace(
            breaks=np.array([0.0, 1.0, 2.0]),
            times=None,
            values=None,
            evaluate_many=lambda values: np.asarray(values) ** 2,
        ),
    )
    process = SimpleNamespace(
        time_axis=SimpleNamespace(start=0.0, end=2.0),
        volume=SimpleNamespace(volume_changes={"feed": change}, unit="L"),
    )
    axis = Figure().subplots()

    forward_plotting._plot_volume_changes(axis, process, t)

    np.testing.assert_array_equal(axis.lines[0].get_xdata(), t)
    np.testing.assert_allclose(axis.lines[0].get_ydata(), t**2)


def test_plot_forward_predictions_removes_partial_png_on_failure(
    monkeypatch, tmp_path: Path
):
    collection, store, wrapper = _build_single_process_runtime(q_scaled=0.5)
    export = _single_dense_export(store, wrapper, prediction_grid_n=7)
    plot_dir = tmp_path / "plots"
    output_path = plot_dir / "p1.png"

    def fail_savefig(_figure, path, **_kwargs):
        Path(path).write_bytes(b"partial")
        raise OSError("disk full")

    monkeypatch.setattr(Figure, "savefig", fail_savefig)

    plot_forward_predictions(
        collection,
        wrapper,
        {"p1": export},
        None,
        {"p1": (0.1, {"biomass": 0.1})},
        tmp_path,
    )

    assert not output_path.exists()
    assert not (plot_dir / ".p1.png.tmp").exists()


def test_plot_forward_predictions_continues_after_process_failure(
    monkeypatch, tmp_path: Path
):
    calls = []

    def fake_plot(process_name, *_args):
        calls.append(process_name)
        if process_name == "p1":
            raise OSError("disk full")

    monkeypatch.setattr(forward_plotting, "_plot_process", fake_plot)
    collection = SimpleNamespace(processes={"p1": object(), "p2": object()})

    plot_forward_predictions(
        collection,
        object(),
        {"p1": object(), "p2": object()},
        None,
        {},
        tmp_path,
    )

    assert calls == ["p1", "p2"]


@pytest.mark.parametrize("unsafe_name", ["../escaped", "/tmp/escaped"])
def test_plot_forward_predictions_rejects_unsafe_process_filename(
    monkeypatch, tmp_path: Path, unsafe_name: str
):
    calls = []

    def fake_plot(process_name, *_args):
        calls.append(process_name)

    monkeypatch.setattr(forward_plotting, "_plot_process", fake_plot)
    collection = SimpleNamespace(processes={unsafe_name: object(), "safe": object()})

    plot_forward_predictions(
        collection,
        object(),
        {unsafe_name: object(), "safe": object()},
        None,
        {},
        tmp_path,
    )

    assert calls == ["safe"]


def test_affine_state_offset_keeps_zero_rhs_stationary_through_forward():
    # Test 2 solver path + end-to-end forward: q=0 means dRAW/dt=0. A wrong
    # value-scale on the derivative would subtract offset/scale and drift the
    # biomass; scale_derivative leaves it stationary while value unscale carries
    # the offset into/out of the solver.
    _, store, wrapper = _build_single_process_runtime(
        q_scaled=0.0,
        modeled_rmc_scaler=AffineScaler(
            jnp.asarray([2.0]),
            jnp.asarray([10.0]),
        ),
    )
    export = _single_dense_export(store, wrapper, prediction_grid_n=7)
    assert export.c_species.shape == (7, 1)
    assert np.allclose(export.c_species[:, 0], 1.0, rtol=0.0, atol=2e-6)


def test_dense_export_returns_physical_q_values():
    collection, store, wrapper = _build_single_process_runtime(
        q_scaled=1.5,
        q_scale=2.0,
    )
    export = _single_dense_export(store, wrapper, prediction_grid_n=9)

    assert export.q_rates.shape == (9, 1)
    assert np.allclose(export.q_rates[:, 0], 3.0)


def test_dense_export_grid_includes_measurement_times():
    """Measurement times must be exact nodes in the exported grid.

    The holdout scorers (``bp_train.loo_metrics``, ``bp_bench.metrics``) evaluate
    predictions by ``np.interp(meas_t, pred_t, pred_y)``. If a measurement falls
    between two uniform grid points that straddle a bolus/feed discontinuity, the
    interpolant is a straight ramp across the jump. Splicing the measurement grid
    into the export makes each measurement an exact node, so np.interp returns the
    exact solve value there.
    """
    # 0.7 is interior and lands strictly between linspace(0, 2, 11) nodes, so it can
    # only appear as a node because the measurement grid was spliced in.
    collection, store, wrapper = _build_single_process_runtime(
        q_scaled=0.5, biomass_times=[0.0, 0.7, 2.0]
    )
    export = _single_dense_export(store, wrapper, prediction_grid_n=11)

    assert np.all(np.diff(export.t) > 0)  # sorted + de-duplicated
    for tm in (0.0, 0.7, 2.0):
        assert np.any(np.isclose(export.t, tm, atol=1e-9)), f"missing node {tm}"
    assert not np.any(np.isclose(np.linspace(0.0, 2.0, 11), 0.7, atol=1e-9))
    # every exported array is aligned with the (grown) grid
    assert export.c_species.shape[0] == export.t.shape[0]
    assert export.v_real.shape == export.t.shape
    assert export.q_rates.shape[0] == export.t.shape[0]


def test_loo_scored_value_equals_training_framework_solve():
    """The value the LOO scorer reads at a measurement must equal the model's exact
    ODE solution there -- i.e. what the training framework evaluates the loss
    against -- not a straight line interpolated across the coarse prediction grid.

    Coarse prediction grid (nodes at 0, 1, 2) plus an off-grid measurement at 0.7.
    Biomass grows exponentially, so the old linspace-only interpolation lands ~6%
    off the true value; splicing the measurement in as a node makes np.interp exact.
    """
    collection, store, wrapper = _build_single_process_runtime(
        q_scaled=0.8, biomass_times=[0.0, 0.7, 2.0]
    )
    export = _single_dense_export(store, wrapper, prediction_grid_n=3)
    c = export.c_species[:, 0]

    node_mask = np.isclose(export.t, 0.7, atol=1e-9)
    assert node_mask.sum() == 1
    # training-framework value == the exact solve at the measurement time; the
    # export now carries it as a node (same sample-grid value the loss module uses).
    train_val = float(c[node_mask][0])
    # LOO-metric value == np.interp over the exported grid (what bp_train.loo_metrics
    # and bp_bench.metrics do). Identical, because the measurement is now a node.
    loo_val = float(np.interp(0.7, export.t, c))
    assert loo_val == pytest.approx(train_val, rel=1e-6)
    # the pre-fix behaviour (interpolating the uniform grid only) is materially off.
    lin_t = np.linspace(0.0, 2.0, 3)
    lin_c = np.interp(lin_t, export.t, c)
    ramp_val = float(np.interp(0.7, lin_t, lin_c))
    assert abs(ramp_val - train_val) > 0.01 * abs(train_val)


def test_export_predictions_csv_includes_auxiliary_columns(tmp_path: Path):
    collection, store, wrapper = _build_single_process_runtime(
        q_scaled=1.5,
        q_scale=2.0,
        auxiliary={
            "mu_raw": jnp.asarray(-0.75),
            "latent_pair": jnp.asarray([4.0, 5.0]),
        },
    )

    _, _, dense_exports = compute_dense_exports(
        wrapper,
        store,
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


def test_aggregate_dense_exports_rejects_mismatched_time_grids():
    def export(t):
        rows = len(t)
        return postprocessing.DenseProcessExport(
            t=np.asarray(t, dtype=float),
            c_species=np.zeros((rows, 1)),
            v_real=np.zeros(rows),
            b_modeled_cum=np.zeros((rows, 0)),
            q_rates=np.zeros((rows, 1)),
        )

    with pytest.raises(ValueError, match="different time grids for process 'p1'"):
        postprocessing.aggregate_dense_exports(
            [{"p1": export([0.0, 1.0])}, {"p1": export([0.0, 1.5])}]
        )


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
