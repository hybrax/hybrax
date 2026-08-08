from __future__ import annotations

from dataclasses import replace
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import jax.numpy as jnp
import numpy as np
import pytest
from bp_format.dataclasses import SampleVolumeChange
from bp_format.serialization import load_process_collection

import bp_train.runtime_artifact as runtime_artifact
from bp_train.harness import _resolve_estimated_scales
from bp_train.model_api import AffineScaler
from bp_train.runtime_artifact import (
    RhsOdeDescriptor,
    RuntimeArtifactFold,
    _rhs,
    load_runtime_artifact,
    write_runtime_artifact,
)
from bp_train.runtime_context import RuntimeContext, RuntimeDataContext
from bp_train.training_data import TrainingDataStore
from bp_train.utils import load_custom_module


_KITTLER = Path("examples/01_kittler_2022/prepared/prepared.json")
_CUSTOM = Path("examples/01_kittler_2022/structured/custom.py")


def _write_manifest(artifact: Path, manifest: dict) -> None:
    manifest.pop("identity", None)
    encoded = json.dumps(
        manifest, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode()
    manifest["identity"] = "sha256:" + hashlib.sha256(encoded).hexdigest()
    (artifact / "manifest.json").write_text(
        json.dumps(manifest, sort_keys=True, separators=(",", ":"), allow_nan=False)
    )


def _rewrite_array(artifact: Path, name: str, array: np.ndarray) -> None:
    manifest = json.loads((artifact / "manifest.json").read_text())
    record = manifest["arrays"][name]
    array_path = artifact / record["file"]
    np.save(array_path, array, allow_pickle=False)
    record.update(
        dtype=array.dtype.str,
        shape=list(array.shape),
        sha256="sha256:" + hashlib.sha256(array_path.read_bytes()).hexdigest(),
    )
    _write_manifest(artifact, manifest)


@pytest.fixture(scope="module")
def runtime_context() -> RuntimeContext:
    collection = load_process_collection(_KITTLER)
    store = TrainingDataStore.from_collection(
        collection, target_source="reactor_components"
    )
    data = RuntimeDataContext.from_collection(store, collection)
    scales = _resolve_estimated_scales(
        custom_module=load_custom_module(_CUSTOM),
        runtime_data=data,
        custom_cfg=SimpleNamespace(
            custom=SimpleNamespace(ratios_softmax_temp=2.0, Y_XS=0.627, Y_PS=0.652)
        ),
    )
    return RuntimeContext(data, scales)


@pytest.fixture(scope="module")
def descriptor(runtime_context: RuntimeContext) -> RhsOdeDescriptor:
    rhs = runtime_context.training_data.rhs_ode
    return RhsOdeDescriptor(
        rhs.name_modeled_rates,
        rhs.name_modeled_algebraic,
        rhs.name_modeled_RMCs,
        rhs.name_modeled_PVs,
        rhs.name_modeled_FVCs,
        rhs.name_modeled_SVCs,
        rhs.name_controlled_PVs,
        rhs.name_controlled_FVCs,
        rhs.name_controlled_SVCs,
        (),
        ("q_biomass", "q_glycerol", "q_product"),
    )


def test_round_trip_affine_scales_and_selected_fold(
    tmp_path, runtime_context, descriptor
):
    scales = replace(
        runtime_context.scales,
        SCALE_modeled_RMCs=AffineScaler(jnp.array([2.0, 3.0, 4.0]), jnp.ones(3)),
    )
    process_order = tuple(runtime_context.training_data.process_order)
    folds = (
        RuntimeArtifactFold(3, (process_order[0],), (process_order[1],), "a", 11),
        RuntimeArtifactFold(4, (process_order[1],), (process_order[0],), "b", 12),
    )
    artifact = tmp_path / "artifact"

    selected_scales = replace(
        scales,
        SCALE_modeled_RMCs=AffineScaler(jnp.array([5.0, 6.0, 7.0]), jnp.full(3, 2.0)),
    )
    identity = write_runtime_artifact(
        artifact,
        runtime_data=runtime_context.data,
        folds=((folds[0], scales), (folds[1], selected_scales)),
        rhs_descriptor=descriptor,
    )
    loaded = load_runtime_artifact(artifact, fold_id=4)

    assert loaded.identity == identity
    assert loaded.fold == folds[1]
    assert loaded.context.training_data.process_order == tuple(
        runtime_context.training_data.process_order
    )
    np.testing.assert_array_equal(
        np.asarray(loaded.context.training_data.y_measured),
        np.asarray(runtime_context.training_data.y_measured),
    )
    scaler = loaded.context.scales.SCALE_modeled_RMCs
    np.testing.assert_array_equal(np.asarray(scaler.scale), [5.0, 6.0, 7.0])
    np.testing.assert_array_equal(np.asarray(scaler.offset), [2.0, 2.0, 2.0])
    derivative = loaded.context.training_data.rhs_ode.derivative_funcs[0]
    assert float(derivative(jnp.array([0.0, 0.0, 0.0, 0.0, 1.0, 2.0, 3.0]))) == 1.0
    trace = loaded.context.data.sample_volume_event_traces[0][0]
    assert not trace.flags.writeable
    with pytest.raises(ValueError, match="read-only"):
        trace[0] = 0.0


def test_round_trip_multiple_overlapping_sample_streams(
    tmp_path, runtime_context, descriptor
):
    collection = load_process_collection(_KITTLER)
    process = next(iter(collection.processes.values()))
    name, sample = next(
        (name, change)
        for name, change in process.volume.volume_changes.items()
        if isinstance(change, SampleVolumeChange)
    )
    duplicate = replace(sample, name=f"{name}_overlap")
    process.volume.volume_changes[duplicate.name] = duplicate

    store = TrainingDataStore.from_collection(
        collection, target_source="reactor_components"
    )
    data = RuntimeDataContext.from_collection(store, collection)
    process_order = tuple(store.process_order)
    fold = RuntimeArtifactFold(
        1, (process_order[0],), (process_order[1],), "overlap", 1
    )
    artifact = tmp_path / "artifact"

    write_runtime_artifact(
        artifact,
        runtime_data=data,
        folds=((fold, runtime_context.scales),),
        rhs_descriptor=descriptor,
    )
    loaded = load_runtime_artifact(artifact, fold_id=1)

    expected_times, expected_values = data.sample_volume_event_traces[0]
    loaded_times, loaded_values = loaded.context.data.sample_volume_event_traces[0]
    assert np.any(np.diff(expected_times) == 0)
    np.testing.assert_array_equal(loaded_times, expected_times)
    np.testing.assert_array_equal(loaded_values, expected_values)


def test_identity_is_deterministic_and_publication_never_overwrites(
    tmp_path, runtime_context, descriptor
):
    process_order = tuple(runtime_context.training_data.process_order)
    folds = (
        RuntimeArtifactFold(1, (process_order[0],), (process_order[1],), "one", 1),
    )
    first = tmp_path / "first"
    second = tmp_path / "second"

    folds_with_scales = ((folds[0], runtime_context.scales),)
    first_identity = write_runtime_artifact(
        first,
        runtime_data=runtime_context.data,
        folds=folds_with_scales,
        rhs_descriptor=descriptor,
    )
    second_identity = write_runtime_artifact(
        second,
        runtime_data=runtime_context.data,
        folds=folds_with_scales,
        rhs_descriptor=descriptor,
    )

    assert first_identity == second_identity
    with pytest.raises(FileExistsError):
        write_runtime_artifact(
            first,
            runtime_data=runtime_context.data,
            folds=folds_with_scales,
            rhs_descriptor=descriptor,
        )


def test_publication_race_preserves_destination(
    tmp_path, runtime_context, descriptor, monkeypatch
):
    artifact = tmp_path / "artifact"
    process_order = tuple(runtime_context.training_data.process_order)
    fold = RuntimeArtifactFold(1, (process_order[0],), (process_order[1],), "one", 1)
    publish = runtime_artifact._publish_directory

    def create_destination_before_publish(source, destination):
        destination.mkdir()
        (destination / "owner").write_text("other producer")
        publish(source, destination)

    monkeypatch.setattr(
        runtime_artifact, "_publish_directory", create_destination_before_publish
    )
    with pytest.raises(FileExistsError):
        write_runtime_artifact(
            artifact,
            runtime_data=runtime_context.data,
            folds=((fold, runtime_context.scales),),
            rhs_descriptor=descriptor,
        )

    assert (artifact / "owner").read_text() == "other producer"
    assert list(tmp_path.glob(".artifact.*.tmp")) == []


def test_selected_fold_checksum_isolated(tmp_path, runtime_context, descriptor):
    artifact = tmp_path / "artifact"
    process_order = tuple(runtime_context.training_data.process_order)
    folds = (
        RuntimeArtifactFold(1, (process_order[0],), (process_order[1],), "one", 1),
        RuntimeArtifactFold(2, (process_order[1],), (process_order[0],), "two", 2),
    )
    write_runtime_artifact(
        artifact,
        runtime_data=runtime_context.data,
        folds=(
            (folds[0], runtime_context.scales),
            (folds[1], runtime_context.scales),
        ),
        rhs_descriptor=descriptor,
    )
    manifest = json.loads((artifact / "manifest.json").read_text())
    unselected = next(name for name in manifest["arrays"] if name.startswith("fold.1."))
    (artifact / manifest["arrays"][unselected]["file"]).write_bytes(b"corrupt")

    assert load_runtime_artifact(artifact, fold_id=2).fold == folds[1]

    selected = next(name for name in manifest["arrays"] if name.startswith("fold.2."))
    selected_path = artifact / manifest["arrays"][selected]["file"]
    corrupted = bytearray(selected_path.read_bytes())
    corrupted[-1] ^= 1
    selected_path.write_bytes(corrupted)
    with pytest.raises(ValueError, match="checksum mismatch"):
        load_runtime_artifact(artifact, fold_id=2)


def test_rejects_checksum_consistent_semantic_corruption(
    tmp_path, runtime_context, descriptor
):
    artifact = tmp_path / "artifact"
    process_order = tuple(runtime_context.training_data.process_order)
    fold = RuntimeArtifactFold(1, (process_order[0],), (process_order[1],), "one", 1)
    write_runtime_artifact(
        artifact,
        runtime_data=runtime_context.data,
        folds=((fold, runtime_context.scales),),
        rhs_descriptor=descriptor,
    )
    manifest = json.loads((artifact / "manifest.json").read_text())
    for suffix in ("times", "values"):
        name = f"shared.trace.sample.0.{suffix}"
        original = np.load(
            artifact / manifest["arrays"][name]["file"], allow_pickle=False
        )
        _rewrite_array(artifact, name, original[None, :])
        manifest = json.loads((artifact / "manifest.json").read_text())

    with pytest.raises(ValueError, match="invalid sample trace shape"):
        load_runtime_artifact(artifact, fold_id=1)


def test_rejects_invalid_measurement_counts(tmp_path, runtime_context, descriptor):
    artifact = tmp_path / "artifact"
    process_order = tuple(runtime_context.training_data.process_order)
    fold = RuntimeArtifactFold(1, (process_order[0],), (process_order[1],), "one", 1)
    write_runtime_artifact(
        artifact,
        runtime_data=runtime_context.data,
        folds=((fold, runtime_context.scales),),
        rhs_descriptor=descriptor,
    )
    original = np.asarray(runtime_context.training_data.n_measured)
    padded_width = runtime_context.training_data.t_measured.shape[1]

    for invalid in (-1, padded_width + 1):
        corrupted = original.copy()
        corrupted[0] = invalid
        _rewrite_array(artifact, "shared.store.n_measured", corrupted)
        with pytest.raises(ValueError, match="values exceed padded dimensions"):
            load_runtime_artifact(artifact, fold_id=1)


def test_rejects_invalid_loaded_scaler_values(tmp_path, runtime_context, descriptor):
    artifact = tmp_path / "artifact"
    process_order = tuple(runtime_context.training_data.process_order)
    fold = RuntimeArtifactFold(1, (process_order[0],), (process_order[1],), "one", 1)
    scales = replace(
        runtime_context.scales,
        SCALE_modeled_RMCs=AffineScaler(jnp.ones(3), jnp.zeros(3)),
    )
    write_runtime_artifact(
        artifact,
        runtime_data=runtime_context.data,
        folds=((fold, scales),),
        rhs_descriptor=descriptor,
    )

    scale_name = "fold.1.scale.SCALE_modeled_RMCs.scale"
    offset_name = "fold.1.scale.SCALE_modeled_RMCs.offset"
    for invalid in (0.0, np.nan, np.inf):
        corrupted = np.ones(3)
        corrupted[0] = invalid
        _rewrite_array(artifact, scale_name, corrupted)
        with pytest.raises(ValueError, match="invalid semantic scale values"):
            load_runtime_artifact(artifact, fold_id=1)

    _rewrite_array(artifact, scale_name, np.ones(3))
    for invalid in (np.nan, np.inf):
        corrupted = np.zeros(3)
        corrupted[0] = invalid
        _rewrite_array(artifact, offset_name, corrupted)
        with pytest.raises(ValueError, match="invalid semantic scale values"):
            load_runtime_artifact(artifact, fold_id=1)


def test_rejects_invalid_in_memory_scalers(runtime_context, descriptor, tmp_path):
    process_order = tuple(runtime_context.training_data.process_order)
    fold = RuntimeArtifactFold(1, (process_order[0],), (process_order[1],), "one", 1)
    invalid_scalers = (
        AffineScaler(jnp.array([0.0, 1.0, 1.0]), jnp.zeros(3)),
        AffineScaler(jnp.ones(3), jnp.array([jnp.inf, 0.0, 0.0])),
    )

    for index, scaler in enumerate(invalid_scalers):
        scales = replace(runtime_context.scales, SCALE_modeled_RMCs=scaler)
        with pytest.raises(ValueError, match="invalid semantic scale values"):
            write_runtime_artifact(
                tmp_path / f"artifact-{index}",
                runtime_data=runtime_context.data,
                folds=((fold, scales),),
                rhs_descriptor=descriptor,
            )


@pytest.mark.parametrize("corruption", ["shape", "nonfinite"])
def test_rejects_semantically_invalid_context(
    runtime_context, descriptor, tmp_path, corruption
):
    times, values = runtime_context.data.sample_volume_event_traces[0]
    traces = list(runtime_context.data.sample_volume_event_traces)
    if corruption == "shape":
        traces[0] = (times[None, :], values)
        message = "invalid sample trace shape"
    else:
        invalid = times.copy()
        invalid[0] = np.nan
        traces[0] = (invalid, values)
        message = "invalid sample trace values"
    data = replace(runtime_context.data, sample_volume_event_traces=tuple(traces))
    process_order = tuple(runtime_context.training_data.process_order)
    fold = RuntimeArtifactFold(1, (process_order[0],), (process_order[1],), "one", 1)

    with pytest.raises(ValueError, match=message):
        write_runtime_artifact(
            tmp_path / "artifact",
            runtime_data=data,
            folds=((fold, runtime_context.scales),),
            rhs_descriptor=descriptor,
        )


def test_reconstructed_rhs_supports_algebraic_expressions():
    descriptor = RhsOdeDescriptor(
        ("q",),
        ("active",),
        ("biomass", "product"),
        (),
        (),
        (),
        (),
        (),
        (),
        ("biomass - product",),
        ("q * active", "0"),
    )
    rhs = _rhs(
        descriptor,
        {
            "shared.rhs.Cin_controlled_FVCs": np.zeros((0, 2)),
            "shared.rhs.Cin_modeled_FVCs": np.zeros((0, 2)),
        },
    )

    actual = rhs(
        jnp.array([3.0, 1.0, 1.0]),
        jnp.array([2.0]),
        jnp.zeros(0),
        jnp.zeros(0),
        jnp.zeros(0),
    )

    np.testing.assert_allclose(actual, [4.0, 0.0, 0.0])


@pytest.mark.parametrize(
    "expression",
    ("Max()", "Min()", "sqrt(1, 2)", "log(1, 2)", "1 / 0"),
)
def test_rejects_invalid_rhs_expression_semantics(expression):
    with pytest.raises(ValueError):
        runtime_artifact._parse_expression(expression, ())


@pytest.mark.parametrize(
    "expression",
    ("Abs(-1)", "Max(1, 2)", "Min(1, 2)", "sqrt(4)", "log(1)"),
)
def test_accepts_supported_rhs_function_arities(expression):
    parsed = runtime_artifact._parse_expression(expression, ())
    assert parsed.is_finite is not False


def test_rejects_unsafe_rhs_expression():
    descriptor = RhsOdeDescriptor(
        ("q",),
        (),
        ("biomass",),
        (),
        (),
        (),
        (),
        (),
        (),
        (),
        ("__import__('os').system('true')",),
    )

    with pytest.raises(ValueError, match="unsupported syntax"):
        _rhs(
            descriptor,
            {
                "shared.rhs.Cin_controlled_FVCs": np.zeros((0, 1)),
                "shared.rhs.Cin_modeled_FVCs": np.zeros((0, 1)),
            },
        )


def test_rejects_invalid_fold_membership(tmp_path, runtime_context, descriptor):
    process_order = tuple(runtime_context.training_data.process_order)
    artifact = tmp_path / "artifact"
    fold = RuntimeArtifactFold(
        1,
        (process_order[0],),
        (process_order[0],),
        "one",
        1,
    )

    with pytest.raises(ValueError, match="invalid fold metadata"):
        write_runtime_artifact(
            artifact,
            runtime_data=runtime_context.data,
            folds=((fold, runtime_context.scales),),
            rhs_descriptor=descriptor,
        )


def test_rejects_active_nonfinite_values(tmp_path, runtime_context, descriptor):
    artifact = tmp_path / "artifact"
    process_order = tuple(runtime_context.training_data.process_order)
    fold = RuntimeArtifactFold(1, (process_order[0],), (process_order[1],), "one", 1)
    write_runtime_artifact(
        artifact,
        runtime_data=runtime_context.data,
        folds=((fold, runtime_context.scales),),
        rhs_descriptor=descriptor,
    )
    mutations = (
        ("shared.store.y0_measured", (0, 0)),
        ("shared.controls.sample_event_times", (0, 0)),
        ("shared.controls.dense_grid", (0, 0)),
        ("shared.store.t_measured", (0, 0)),
    )
    manifest = json.loads((artifact / "manifest.json").read_text())
    for name, index in mutations:
        original = np.load(
            artifact / manifest["arrays"][name]["file"], allow_pickle=False
        )
        corrupted = original.copy()
        corrupted[index] = np.nan
        _rewrite_array(artifact, name, corrupted)
        with pytest.raises(ValueError, match="non-finite"):
            load_runtime_artifact(artifact, fold_id=1)
        _rewrite_array(artifact, name, original)
        manifest = json.loads((artifact / "manifest.json").read_text())


def test_rejects_invalid_control_lengths_and_masks(
    tmp_path, runtime_context, descriptor
):
    artifact = tmp_path / "artifact"
    process_order = tuple(runtime_context.training_data.process_order)
    fold = RuntimeArtifactFold(1, (process_order[0],), (process_order[1],), "one", 1)
    write_runtime_artifact(
        artifact,
        runtime_data=runtime_context.data,
        folds=((fold, runtime_context.scales),),
        rhs_descriptor=descriptor,
    )
    cases = (
        ("shared.controls.grid_lengths", -1),
        (
            "shared.controls.grid_lengths",
            runtime_context.training_data.controls_store.dense_grid.shape[1] + 1,
        ),
        ("shared.controls.jump_ts_lengths", -1),
        (
            "shared.controls.jump_ts_lengths",
            runtime_context.training_data.controls_store.jump_ts.shape[1] + 1,
        ),
    )
    manifest = json.loads((artifact / "manifest.json").read_text())
    for name, invalid in cases:
        original = np.load(
            artifact / manifest["arrays"][name]["file"], allow_pickle=False
        )
        corrupted = original.copy()
        corrupted[0] = invalid
        _rewrite_array(artifact, name, corrupted)
        with pytest.raises(ValueError, match="invalid active lengths"):
            load_runtime_artifact(artifact, fold_id=1)
        _rewrite_array(artifact, name, original)
        manifest = json.loads((artifact / "manifest.json").read_text())

    name = "shared.controls.sample_event_mask"
    original = np.load(artifact / manifest["arrays"][name]["file"], allow_pickle=False)
    if original.shape[1] >= 2:
        corrupted = original.copy()
        corrupted[0, :2] = (False, True)
        _rewrite_array(artifact, name, corrupted)
        with pytest.raises(ValueError, match="active prefix"):
            load_runtime_artifact(artifact, fold_id=1)


def test_rejects_invalid_solver_window_bounds(tmp_path, runtime_context, descriptor):
    process_order = tuple(runtime_context.training_data.process_order)
    fold = RuntimeArtifactFold(1, (process_order[0],), (process_order[1],), "one", 1)
    for index, (name, value) in enumerate(
        (
            ("max_event_gap_fraction", -0.1),
            ("max_event_gap_fraction", 1.1),
            ("max_measurements_per_event_gap", -1),
        )
    ):
        artifact = tmp_path / f"artifact-{index}"
        write_runtime_artifact(
            artifact,
            runtime_data=runtime_context.data,
            folds=((fold, runtime_context.scales),),
            rhs_descriptor=descriptor,
        )
        manifest = json.loads((artifact / "manifest.json").read_text())
        manifest["base"]["controls"][name] = value
        _write_manifest(artifact, manifest)
        with pytest.raises(ValueError, match="invalid runtime controls metadata"):
            load_runtime_artifact(artifact, fold_id=1)


def test_loaded_metadata_is_immutable(tmp_path, runtime_context, descriptor):
    artifact = tmp_path / "artifact"
    process_order = tuple(runtime_context.training_data.process_order)
    fold = RuntimeArtifactFold(1, (process_order[0],), (process_order[1],), "one", 1)
    write_runtime_artifact(
        artifact,
        runtime_data=runtime_context.data,
        folds=((fold, runtime_context.scales),),
        rhs_descriptor=descriptor,
    )
    loaded = load_runtime_artifact(artifact, fold_id=1)
    store = loaded.context.training_data
    controls = store.controls_store

    with pytest.raises(AttributeError):
        store.process_order.append("evil")
    with pytest.raises(TypeError):
        controls.shape_metadata["n_processes"] = 0
    with pytest.raises(TypeError):
        controls._process_md_by_name[process_order[0]]["control_metadata"] = {}

    metadata = controls.get_controls(process_order[0]).control_metadata
    with pytest.raises(TypeError):
        metadata["evil"] = {}
    first = next(iter(metadata))
    with pytest.raises(TypeError):
        metadata[first]["evil"] = True


def test_rejects_invalid_ordered_time_axes(tmp_path, runtime_context, descriptor):
    artifact = tmp_path / "artifact"
    process_order = tuple(runtime_context.training_data.process_order)
    fold = RuntimeArtifactFold(1, (process_order[0],), (process_order[1],), "one", 1)
    write_runtime_artifact(
        artifact,
        runtime_data=runtime_context.data,
        folds=((fold, runtime_context.scales),),
        rhs_descriptor=descriptor,
    )
    names = (
        "shared.controls.dense_grid",
        "shared.controls.spline_breaks",
        "shared.store.t_measured",
        "shared.trace.modeled.0.0.times",
    )
    for name in names:
        manifest = json.loads((artifact / "manifest.json").read_text())
        original = np.load(
            artifact / manifest["arrays"][name]["file"], allow_pickle=False
        )
        corrupted = original.copy()
        if corrupted.ndim == 2:
            corrupted[0, 1] = corrupted[0, 0]
        else:
            corrupted[1] = corrupted[0]
        _rewrite_array(artifact, name, corrupted)
        with pytest.raises(ValueError, match="time axis|strictly increasing"):
            load_runtime_artifact(artifact, fold_id=1)
        _rewrite_array(artifact, name, original)

    name = "shared.trace.sample.0.times"
    manifest = json.loads((artifact / "manifest.json").read_text())
    original = np.load(artifact / manifest["arrays"][name]["file"], allow_pickle=False)
    corrupted = original.copy()
    corrupted[1] = corrupted[0] - 1.0
    _rewrite_array(artifact, name, corrupted)
    with pytest.raises(ValueError, match="time axis"):
        load_runtime_artifact(artifact, fold_id=1)


def test_rejects_invalid_runtime_metadata(tmp_path, runtime_context, descriptor):
    process_order = tuple(runtime_context.training_data.process_order)
    fold = RuntimeArtifactFold(1, (process_order[0],), (process_order[1],), "one", 1)

    missing_metadata = tmp_path / "missing-metadata"
    write_runtime_artifact(
        missing_metadata,
        runtime_data=runtime_context.data,
        folds=((fold, runtime_context.scales),),
        rhs_descriptor=descriptor,
    )
    manifest = json.loads((missing_metadata / "manifest.json").read_text())
    control_metadata = manifest["base"]["controls"]["_process_md_by_name"]
    control_metadata[process_order[0]].pop("control_metadata")
    _write_manifest(missing_metadata, manifest)
    with pytest.raises(ValueError, match="per-process control metadata"):
        load_runtime_artifact(missing_metadata, fold_id=1)

    inverted_bounds = tmp_path / "inverted-bounds"
    write_runtime_artifact(
        inverted_bounds,
        runtime_data=runtime_context.data,
        folds=((fold, runtime_context.scales),),
        rhs_descriptor=descriptor,
    )
    manifest = json.loads((inverted_bounds / "manifest.json").read_text())
    for snapshots in manifest["base"]["runtime"]["bound_snapshots"]:
        snapshots[0][3:] = [2.0, 1.0]
    _write_manifest(inverted_bounds, manifest)
    with pytest.raises(ValueError, match="bounds snapshot"):
        load_runtime_artifact(inverted_bounds, fold_id=1)


def test_rejects_invalid_active_sample_events(tmp_path, runtime_context, descriptor):
    artifact = tmp_path / "artifact"
    process_order = tuple(runtime_context.training_data.process_order)
    fold = RuntimeArtifactFold(1, (process_order[0],), (process_order[1],), "one", 1)
    write_runtime_artifact(
        artifact,
        runtime_data=runtime_context.data,
        folds=((fold, runtime_context.scales),),
        rhs_descriptor=descriptor,
    )
    manifest = json.loads((artifact / "manifest.json").read_text())
    for name, invalid, message in (
        ("shared.controls.sample_event_volumes", -1.0, "values must be positive"),
        ("shared.controls.sample_event_times", -1.0, "invalid active time axis"),
    ):
        original = np.load(
            artifact / manifest["arrays"][name]["file"], allow_pickle=False
        )
        corrupted = original.copy()
        corrupted[0, 0] = invalid
        _rewrite_array(artifact, name, corrupted)
        with pytest.raises(ValueError, match=message):
            load_runtime_artifact(artifact, fold_id=1)
        _rewrite_array(artifact, name, original)
        manifest = json.loads((artifact / "manifest.json").read_text())


def test_rejects_manifest_schema_changes(tmp_path, runtime_context, descriptor):
    process_order = tuple(runtime_context.training_data.process_order)
    artifact = tmp_path / "artifact"
    fold = RuntimeArtifactFold(
        1,
        (process_order[0],),
        (process_order[1],),
        "one",
        1,
    )
    write_runtime_artifact(
        artifact,
        runtime_data=runtime_context.data,
        folds=((fold, runtime_context.scales),),
        rhs_descriptor=descriptor,
    )
    manifest_path = artifact / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["unexpected"] = True
    manifest_path.write_text(json.dumps(manifest))

    with pytest.raises(ValueError, match="manifest schema"):
        load_runtime_artifact(artifact, fold_id=1)
