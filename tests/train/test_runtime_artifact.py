from __future__ import annotations

import dataclasses
from copy import deepcopy
from dataclasses import replace
from functools import cache
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import jax.numpy as jnp
import numpy as np
import pytest
from bp_format.dataclasses import AugmentedBioProcess
from bp_format.mechanistic import build_rhs_ode
from bp_format.serialization import load_process_collection, save_process_collection

import bp_train.runtime_artifact as runtime_artifact
from bp_train.defaults import default_build_reaction_module
from bp_train.harness import (
    TrainHarnessConfig,
    _resolve_estimated_scales,
    prepare_training_from_runtime_artifact,
)
from bp_train.model_api import AffineScaler, EstimatedScales
from bp_train.runtime_artifact import (
    RhsNames,
    RuntimeArtifactFold,
    load_runtime_artifact,
    read_runtime_artifact_metadata,
    write_runtime_artifact as _write_runtime_artifact,
)
from bp_train.runtime_context import (
    ProducerCollectionData,
    canonical_training_parents,
    original_parent_processes,
    select_parent_collection,
)
from bp_train.training_data import TrainingDataStore
from bp_train.utils import load_custom_module


_KITTLER = Path("examples/01_kittler_2022/prepared/prepared.json")
_CUSTOM = Path("examples/01_kittler_2022/structured/custom.py")


@cache
def _source_collection():
    return load_process_collection(_KITTLER)


def write_runtime_artifact(
    path,
    *,
    producer_data,
    folds,
    rhs_names,
    parent_collection=None,
    **kwargs,
):
    """Write a test artifact with all canonical original parents."""
    if parent_collection is None:
        parent_collection = select_parent_collection(
            _source_collection(),
            original_parent_processes(
                producer_data.process_order, producer_data.augmentation_parents
            ),
        )
    return _write_runtime_artifact(
        path,
        training_data=producer_data.training_data,
        parent_collection=parent_collection,
        augmentation_parents=producer_data.augmentation_parents,
        folds=tuple(folds),
        rhs_names=rhs_names,
        **kwargs,
    )


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


def _rewrite_parent_collection(artifact: Path, collection) -> None:
    parent_path = artifact / "training-parents.json"
    save_process_collection(collection, parent_path)
    manifest = json.loads((artifact / "manifest.json").read_text())
    manifest["training_parent_collection"]["sha256"] = runtime_artifact._file_digest(
        parent_path
    )
    _write_manifest(artifact, manifest)


@pytest.fixture(scope="module")
def producer_data() -> ProducerCollectionData:
    collection = load_process_collection(_KITTLER)
    store = TrainingDataStore.from_collection(
        collection, target_source="reactor_components"
    )
    return ProducerCollectionData.from_collection(store, collection)


@pytest.fixture(scope="module")
def scales(producer_data: ProducerCollectionData) -> EstimatedScales:
    collection = load_process_collection(_KITTLER)
    return _resolve_estimated_scales(
        custom_module=load_custom_module(_CUSTOM),
        runtime_data=producer_data.select_training_parents(
            collection, producer_data.training_data.process_order
        ),
        custom_cfg=SimpleNamespace(
            custom=SimpleNamespace(ratios_softmax_temp=2.0, Y_XS=0.627, Y_PS=0.652)
        ),
    )


@pytest.fixture(scope="module")
def rhs_names(producer_data: ProducerCollectionData) -> RhsNames:
    return RhsNames.from_rhs_ode(producer_data.training_data.rhs_ode)


def test_round_trip_parent_collection_is_filtered(
    tmp_path, producer_data, scales, rhs_names
):
    source = deepcopy(_source_collection())
    process_order = producer_data.process_order
    child = "DoE1_R1__aug_000"
    other_parent = "DoE1_R2"
    source.metadata["trusted-test-metadata"] = {
        "process-shaped": list(process_order),
        "held-out": process_order[-1],
    }
    trusted_metadata = deepcopy(source.metadata["trusted-test-metadata"])
    parent_collection = select_parent_collection(
        source,
        original_parent_processes(process_order, producer_data.augmentation_parents),
    )
    fold = RuntimeArtifactFold(
        0,
        (process_order[-1],),
        (child, other_parent),
        "selected-parents",
        0,
    )
    expected_parents = ("DoE1_R1", "DoE1_R2")
    assert (
        canonical_training_parents(
            process_order,
            producer_data.augmentation_parents,
            fold.train,
        )
        == expected_parents
    )
    artifact = tmp_path / "artifact"

    write_runtime_artifact(
        artifact,
        producer_data=producer_data,
        folds=((fold, scales),),
        rhs_names=rhs_names,
        parent_collection=parent_collection,
    )
    serialized = load_process_collection(artifact / "training-parents.json")
    loaded = load_runtime_artifact(artifact, fold_id=0)

    assert tuple(serialized.processes) == original_parent_processes(
        process_order, producer_data.augmentation_parents
    )
    assert tuple(loaded.training_parent_collection.processes) == expected_parents
    assert loaded.training_parent_collection.metadata["bp-train"][
        "process_order"
    ] == list(expected_parents)
    assert (
        tuple(loaded.training_parent_collection.metadata["bp-train"]["processes"])
        == expected_parents
    )
    assert (
        loaded.training_parent_collection.metadata["trusted-test-metadata"]
        == trusted_metadata
    )


def test_artifact_fold_parents_are_accepted_by_the_harness(
    tmp_path, producer_data, scales, rhs_names
):
    """The fold-narrowed sidecar is exactly what the training harness expects.

    Covers the junction in `prepare_single_fold_from_runtime_artifact`, where a
    loaded artifact's parent collection is handed to
    `prepare_training_from_runtime_artifact`. The writer stores every original
    parent while the harness derives only the parents represented by the fold's
    training processes, so the two agree only because the loader re-narrows.
    Parent names are literal here so the assertion does not re-derive them
    through `canonical_training_parents`.
    """
    parent_collection = select_parent_collection(
        _source_collection(),
        original_parent_processes(
            producer_data.process_order,
            producer_data.augmentation_parents,
        ),
    )
    fold = RuntimeArtifactFold(
        0, ("DoE3_R4",), ("DoE1_R1__aug_000", "DoE1_R2"), "junction", 0
    )
    artifact = tmp_path / "artifact"
    write_runtime_artifact(
        artifact,
        producer_data=producer_data,
        folds=((fold, scales),),
        rhs_names=rhs_names,
        parent_collection=parent_collection,
    )
    loaded = load_runtime_artifact(artifact, fold_id=0)
    seen = []

    class _CustomModule:
        @staticmethod
        def build_reaction_module(*, training_parent_collection, **kwargs):
            seen.append(tuple(training_parent_collection.processes))
            return default_build_reaction_module(
                training_parent_collection=training_parent_collection, **kwargs
            )

    prepared = prepare_training_from_runtime_artifact(
        loaded,
        config=TrainHarnessConfig(
            process_names=fold.train, holdout_processes=fold.test, epochs=1
        ),
        custom_module=_CustomModule,
        custom_cfg={},
    )

    assert seen == [("DoE1_R1", "DoE1_R2")]
    assert prepared.config.process_names == ("DoE1_R1__aug_000", "DoE1_R2")


def test_round_trip_affine_scales_and_selected_fold(
    tmp_path, producer_data, scales, rhs_names
):
    scales = replace(
        scales,
        SCALE_modeled_RMCs=AffineScaler(jnp.array([2.0, 3.0, 4.0]), jnp.ones(3)),
    )
    process_order = tuple(producer_data.training_data.process_order)
    # Train on a parent whose feed composition differs from the canonical row 0,
    # so the loader substituting the training parent's own Cin instead of row 0
    # would show up below rather than being masked by equal rows.
    canonical_cin = np.asarray(producer_data.training_data.Cin_controlled_FVCs)
    other = next(
        index
        for index in range(1, len(process_order))
        if not np.array_equal(canonical_cin[0], canonical_cin[index])
    )
    folds = (
        RuntimeArtifactFold(3, (process_order[other],), (process_order[0],), "a", 11),
        RuntimeArtifactFold(4, (process_order[0],), (process_order[other],), "b", 12),
    )
    artifact = tmp_path / "artifact"

    selected_scales = replace(
        scales,
        SCALE_modeled_RMCs=AffineScaler(jnp.array([5.0, 6.0, 7.0]), jnp.full(3, 2.0)),
    )
    identity = write_runtime_artifact(
        artifact,
        producer_data=producer_data,
        folds=((folds[0], scales), (folds[1], selected_scales)),
        rhs_names=rhs_names,
    )
    loaded = load_runtime_artifact(artifact, fold_id=4)

    assert loaded.identity == identity
    assert loaded.fold == folds[1]
    assert loaded.training_data.process_order == tuple(
        producer_data.training_data.process_order
    )
    np.testing.assert_array_equal(
        np.asarray(loaded.training_data.y_measured),
        np.asarray(producer_data.training_data.y_measured),
    )
    scaler = loaded.scales.SCALE_modeled_RMCs
    np.testing.assert_array_equal(np.asarray(scaler.scale), [5.0, 6.0, 7.0])
    np.testing.assert_array_equal(np.asarray(scaler.offset), [2.0, 2.0, 2.0])

    # Cin comes back from the canonical store arrays, whose row 0 is the reference
    # process the producer's own `rhs_ode` was built from.
    np.testing.assert_array_equal(
        np.asarray(loaded.training_data.rhs_ode.Cin_controlled_FVCs),
        np.asarray(producer_data.training_data.Cin_controlled_FVCs[0]),
    )
    np.testing.assert_array_equal(
        np.asarray(loaded.training_data.rhs_ode.Cin_modeled_FVCs),
        np.asarray(producer_data.training_data.Cin_modeled_FVCs[0]),
    )


def test_metadata_inspection_never_parses_parent_collection_or_reads_arrays(
    tmp_path, producer_data, scales, rhs_names, monkeypatch
):
    artifact = tmp_path / "artifact"
    process_order = tuple(producer_data.training_data.process_order)
    folds = (
        RuntimeArtifactFold(1, (process_order[0],), (process_order[1],), "one", 1),
        RuntimeArtifactFold(2, (process_order[1],), (process_order[0],), "two", 2),
    )
    identity = write_runtime_artifact(
        artifact,
        producer_data=producer_data,
        folds=tuple((fold, scales) for fold in folds),
        rhs_names=rhs_names,
        identity_inputs={"run_fingerprint": "sha256:expected"},
    )

    monkeypatch.setattr(
        runtime_artifact,
        "_read_array",
        lambda *_args: pytest.fail("metadata inspection read a numeric array"),
    )
    monkeypatch.setattr(
        runtime_artifact,
        "load_process_collection",
        lambda *_args, **_kwargs: pytest.fail(
            "metadata inspection parsed the parent collection"
        ),
    )
    metadata = read_runtime_artifact_metadata(artifact)

    assert metadata.identity == identity
    assert dict(metadata.identity_inputs) == {"run_fingerprint": "sha256:expected"}
    assert metadata.folds == folds
    with pytest.raises(TypeError):
        metadata.identity_inputs["evil"] = "value"


def test_parent_collection_checksum_is_checked_before_parsing(
    tmp_path, producer_data, scales, rhs_names, monkeypatch
):
    artifact = tmp_path / "artifact"
    fold = RuntimeArtifactFold(
        0,
        (producer_data.process_order[0],),
        (producer_data.process_order[1],),
        "fold",
        0,
    )
    write_runtime_artifact(
        artifact,
        producer_data=producer_data,
        folds=((fold, scales),),
        rhs_names=rhs_names,
    )
    with (artifact / "training-parents.json").open("a") as stream:
        stream.write("\n")
    calls = 0

    def fail_if_called(*_args, **_kwargs):
        nonlocal calls
        calls += 1
        pytest.fail("parent collection parsed before checksum validation")

    monkeypatch.setattr(runtime_artifact, "load_process_collection", fail_if_called)

    with pytest.raises(ValueError, match="collection checksum mismatch"):
        load_runtime_artifact(artifact, fold_id=0)
    assert calls == 0


def test_missing_parent_collection_is_rejected(
    tmp_path, producer_data, scales, rhs_names
):
    artifact = tmp_path / "artifact"
    fold = RuntimeArtifactFold(
        0,
        (producer_data.process_order[0],),
        (producer_data.process_order[1],),
        "fold",
        0,
    )
    write_runtime_artifact(
        artifact,
        producer_data=producer_data,
        folds=((fold, scales),),
        rhs_names=rhs_names,
    )
    (artifact / "training-parents.json").unlink()

    with pytest.raises(ValueError, match="collection checksum mismatch"):
        load_runtime_artifact(artifact, fold_id=0)


def test_parent_collection_symlink_is_rejected(
    tmp_path, producer_data, scales, rhs_names
):
    artifact = tmp_path / "artifact"
    fold = RuntimeArtifactFold(
        0,
        (producer_data.process_order[0],),
        (producer_data.process_order[1],),
        "fold",
        0,
    )
    write_runtime_artifact(
        artifact,
        producer_data=producer_data,
        folds=((fold, scales),),
        rhs_names=rhs_names,
    )
    parent_path = artifact / "training-parents.json"
    outside_path = tmp_path / "outside-training-parents.json"
    parent_path.rename(outside_path)
    parent_path.symlink_to(outside_path)

    with pytest.raises(ValueError, match="collection checksum mismatch"):
        load_runtime_artifact(artifact, fold_id=0)


def test_parent_collection_is_parsed_once(
    tmp_path, producer_data, scales, rhs_names, monkeypatch
):
    artifact = tmp_path / "artifact"
    fold = RuntimeArtifactFold(
        0,
        (producer_data.process_order[0],),
        (producer_data.process_order[1],),
        "fold",
        0,
    )
    write_runtime_artifact(
        artifact,
        producer_data=producer_data,
        folds=((fold, scales),),
        rhs_names=rhs_names,
    )
    original_loader = runtime_artifact.load_process_collection
    calls = 0

    def counting_loader(*args, **kwargs):
        nonlocal calls
        calls += 1
        return original_loader(*args, **kwargs)

    monkeypatch.setattr(runtime_artifact, "load_process_collection", counting_loader)

    load_runtime_artifact(artifact, fold_id=0)
    assert calls == 1


def test_writer_requires_all_original_parents(
    tmp_path, producer_data, scales, rhs_names
):
    parent_names = original_parent_processes(
        producer_data.process_order,
        producer_data.augmentation_parents,
    )
    parent_collection = select_parent_collection(_source_collection(), parent_names)
    parent_collection.processes.pop(parent_names[-1])
    fold = RuntimeArtifactFold(
        0,
        (producer_data.process_order[0],),
        (producer_data.process_order[1],),
        "fold",
        0,
    )

    with pytest.raises(ValueError, match="exactly all original parents"):
        write_runtime_artifact(
            tmp_path / "artifact",
            producer_data=producer_data,
            folds=((fold, scales),),
            rhs_names=rhs_names,
            parent_collection=parent_collection,
        )


@pytest.mark.parametrize("mode", ["missing", "extra"])
def test_loader_rejects_parent_collection_key_mismatch(
    tmp_path, producer_data, scales, rhs_names, mode
):
    artifact = tmp_path / "artifact"
    process_order = tuple(producer_data.training_data.process_order)
    fold = RuntimeArtifactFold(1, (process_order[0],), (process_order[1],), "one", 1)
    write_runtime_artifact(
        artifact,
        producer_data=producer_data,
        folds=((fold, scales),),
        rhs_names=rhs_names,
    )
    parent_names = original_parent_processes(
        producer_data.process_order,
        producer_data.augmentation_parents,
    )
    collection = select_parent_collection(_source_collection(), parent_names)
    if mode == "missing":
        collection.processes.pop(parent_names[0])
    else:
        process_name, process = next(iter(collection.processes.items()))
        collection.processes[f"{process_name}_extra"] = deepcopy(process)
    _rewrite_parent_collection(artifact, collection)

    with pytest.raises(ValueError, match="must contain exactly all original parents"):
        load_runtime_artifact(artifact, fold_id=1)


@pytest.mark.parametrize(
    "mode",
    [
        "augmented",
        "process_order",
        "process_order_type",
        "processes",
        "processes_type",
    ],
)
def test_loader_rejects_invalid_parent_collection_identity(
    tmp_path, producer_data, scales, rhs_names, mode
):
    artifact = tmp_path / "artifact"
    process_order = producer_data.process_order
    augmentation_parents = producer_data.augmentation_parents
    parent_names = original_parent_processes(process_order, augmentation_parents)
    fold = RuntimeArtifactFold(0, (parent_names[0],), (parent_names[1],), "one", 1)
    write_runtime_artifact(
        artifact,
        producer_data=producer_data,
        folds=((fold, scales),),
        rhs_names=rhs_names,
    )
    collection = select_parent_collection(_source_collection(), parent_names)
    if mode == "augmented":
        child_name = process_order[
            next(i for i, p in enumerate(augmentation_parents) if p)
        ]
        child = deepcopy(_source_collection().processes[child_name])
        assert isinstance(child, AugmentedBioProcess)
        collection.processes[parent_names[0]] = child
        error = "contains an augmented process"
    elif mode == "process_order":
        collection.metadata["bp-train"]["process_order"] = list(reversed(parent_names))
        error = "structural metadata"
    elif mode == "process_order_type":
        collection.metadata["bp-train"]["process_order"] = None
        error = "structural metadata"
    elif mode == "processes":
        process_metadata = collection.metadata["bp-train"]["processes"]
        process_metadata["wrong-parent"] = process_metadata.pop(parent_names[0])
        error = "structural metadata"
    else:
        collection.metadata["bp-train"]["processes"] = None
        error = "structural metadata"
    _rewrite_parent_collection(artifact, collection)

    with pytest.raises(ValueError, match=error):
        load_runtime_artifact(artifact, fold_id=0)


def test_parent_collection_changes_artifact_identity(
    tmp_path, producer_data, scales, rhs_names
):
    parent_names = original_parent_processes(
        producer_data.process_order,
        producer_data.augmentation_parents,
    )
    first = select_parent_collection(_source_collection(), parent_names)
    second = select_parent_collection(_source_collection(), parent_names)
    first.metadata["trusted-test-value"] = "first"
    second.metadata["trusted-test-value"] = "second"
    fold = RuntimeArtifactFold(
        0,
        (producer_data.process_order[0],),
        (producer_data.process_order[1],),
        "fold",
        0,
    )

    first_identity = write_runtime_artifact(
        tmp_path / "first",
        producer_data=producer_data,
        folds=((fold, scales),),
        rhs_names=rhs_names,
        parent_collection=first,
    )
    second_identity = write_runtime_artifact(
        tmp_path / "second",
        producer_data=producer_data,
        folds=((fold, scales),),
        rhs_names=rhs_names,
        parent_collection=second,
    )

    assert first_identity != second_identity


@pytest.mark.parametrize("extra_path", ["unexpected.json", "arrays/unexpected.npy"])
def test_rejects_extra_artifact_files(
    tmp_path, producer_data, scales, rhs_names, extra_path
):
    artifact = tmp_path / "artifact"
    fold = RuntimeArtifactFold(
        0,
        (producer_data.process_order[0],),
        (producer_data.process_order[1],),
        "fold",
        0,
    )
    write_runtime_artifact(
        artifact,
        producer_data=producer_data,
        folds=((fold, scales),),
        rhs_names=rhs_names,
    )
    extra_file = artifact / extra_path
    extra_file.parent.mkdir(parents=True, exist_ok=True)
    extra_file.write_text("{}")

    with pytest.raises(ValueError, match="missing or extra files"):
        read_runtime_artifact_metadata(artifact)
    with pytest.raises(ValueError, match="missing or extra files"):
        load_runtime_artifact(artifact, fold_id=0)


def test_identity_is_deterministic_and_publication_never_overwrites(
    tmp_path, producer_data, scales, rhs_names
):
    process_order = tuple(producer_data.training_data.process_order)
    folds = (
        RuntimeArtifactFold(1, (process_order[0],), (process_order[1],), "one", 1),
    )
    first = tmp_path / "first"
    second = tmp_path / "second"

    folds_with_scales = ((folds[0], scales),)
    first_identity = write_runtime_artifact(
        first,
        producer_data=producer_data,
        folds=folds_with_scales,
        rhs_names=rhs_names,
    )
    second_identity = write_runtime_artifact(
        second,
        producer_data=producer_data,
        folds=folds_with_scales,
        rhs_names=rhs_names,
    )

    assert first_identity == second_identity
    with pytest.raises(FileExistsError):
        write_runtime_artifact(
            first,
            producer_data=producer_data,
            folds=folds_with_scales,
            rhs_names=rhs_names,
        )


def test_publication_race_preserves_destination(
    tmp_path, producer_data, scales, rhs_names, monkeypatch
):
    artifact = tmp_path / "artifact"
    process_order = tuple(producer_data.training_data.process_order)
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
            producer_data=producer_data,
            folds=((fold, scales),),
            rhs_names=rhs_names,
        )

    assert (artifact / "owner").read_text() == "other producer"
    assert list(tmp_path.glob(".artifact.*.tmp")) == []


def test_selected_fold_checksum_isolated(tmp_path, producer_data, scales, rhs_names):
    artifact = tmp_path / "artifact"
    process_order = tuple(producer_data.training_data.process_order)
    folds = (
        RuntimeArtifactFold(1, (process_order[0],), (process_order[1],), "one", 1),
        RuntimeArtifactFold(2, (process_order[1],), (process_order[0],), "two", 2),
    )
    write_runtime_artifact(
        artifact,
        producer_data=producer_data,
        folds=(
            (folds[0], scales),
            (folds[1], scales),
        ),
        rhs_names=rhs_names,
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
    tmp_path, producer_data, scales, rhs_names
):
    artifact = tmp_path / "artifact"
    process_order = tuple(producer_data.training_data.process_order)
    fold = RuntimeArtifactFold(1, (process_order[0],), (process_order[1],), "one", 1)
    write_runtime_artifact(
        artifact,
        producer_data=producer_data,
        folds=((fold, scales),),
        rhs_names=rhs_names,
    )
    # Rewriting through `_rewrite_array` keeps the record's dtype, shape and
    # checksum internally consistent, so only the semantic pass can catch it.
    original = np.asarray(producer_data.training_data.n_measured)
    _rewrite_array(artifact, "shared.store.n_measured", original[None, :])

    with pytest.raises(ValueError, match="invalid semantic dtype or shape"):
        load_runtime_artifact(artifact, fold_id=1)


def test_rejects_invalid_measurement_counts(tmp_path, producer_data, scales, rhs_names):
    artifact = tmp_path / "artifact"
    process_order = tuple(producer_data.training_data.process_order)
    fold = RuntimeArtifactFold(1, (process_order[0],), (process_order[1],), "one", 1)
    write_runtime_artifact(
        artifact,
        producer_data=producer_data,
        folds=((fold, scales),),
        rhs_names=rhs_names,
    )
    original = np.asarray(producer_data.training_data.n_measured)
    padded_width = producer_data.training_data.t_measured.shape[1]

    for invalid in (-1, padded_width + 1):
        corrupted = original.copy()
        corrupted[0] = invalid
        _rewrite_array(artifact, "shared.store.n_measured", corrupted)
        with pytest.raises(ValueError, match="values exceed padded dimensions"):
            load_runtime_artifact(artifact, fold_id=1)


def test_rejects_invalid_loaded_scaler_values(
    tmp_path, producer_data, scales, rhs_names
):
    artifact = tmp_path / "artifact"
    process_order = tuple(producer_data.training_data.process_order)
    fold = RuntimeArtifactFold(1, (process_order[0],), (process_order[1],), "one", 1)
    scales = replace(
        scales,
        SCALE_modeled_RMCs=AffineScaler(jnp.ones(3), jnp.zeros(3)),
    )
    write_runtime_artifact(
        artifact,
        producer_data=producer_data,
        folds=((fold, scales),),
        rhs_names=rhs_names,
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


def test_rejects_invalid_in_memory_scalers(producer_data, scales, rhs_names, tmp_path):
    process_order = tuple(producer_data.training_data.process_order)
    fold = RuntimeArtifactFold(1, (process_order[0],), (process_order[1],), "one", 1)
    invalid_scalers = (
        AffineScaler(jnp.array([0.0, 1.0, 1.0]), jnp.zeros(3)),
        AffineScaler(jnp.ones(3), jnp.array([jnp.inf, 0.0, 0.0])),
    )

    for index, scaler in enumerate(invalid_scalers):
        scales = replace(scales, SCALE_modeled_RMCs=scaler)
        with pytest.raises(ValueError, match="invalid semantic scale values"):
            write_runtime_artifact(
                tmp_path / f"artifact-{index}",
                producer_data=producer_data,
                folds=((fold, scales),),
                rhs_names=rhs_names,
            )


def test_rejects_invalid_fold_membership(tmp_path, producer_data, scales, rhs_names):
    process_order = tuple(producer_data.training_data.process_order)
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
            producer_data=producer_data,
            folds=((fold, scales),),
            rhs_names=rhs_names,
        )


def test_rejects_active_nonfinite_values(tmp_path, producer_data, scales, rhs_names):
    artifact = tmp_path / "artifact"
    process_order = tuple(producer_data.training_data.process_order)
    fold = RuntimeArtifactFold(1, (process_order[0],), (process_order[1],), "one", 1)
    write_runtime_artifact(
        artifact,
        producer_data=producer_data,
        folds=((fold, scales),),
        rhs_names=rhs_names,
    )
    mutations = (
        ("shared.store.y0_measured", (0, 0)),
        ("shared.controls.sample_event_times", (0, 0)),
        ("shared.controls.linear_grid", (0, 0)),
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
    tmp_path, producer_data, scales, rhs_names
):
    artifact = tmp_path / "artifact"
    process_order = tuple(producer_data.training_data.process_order)
    fold = RuntimeArtifactFold(1, (process_order[0],), (process_order[1],), "one", 1)
    write_runtime_artifact(
        artifact,
        producer_data=producer_data,
        folds=((fold, scales),),
        rhs_names=rhs_names,
    )
    cases = (
        ("shared.controls.grid_lengths", -1),
        (
            "shared.controls.grid_lengths",
            producer_data.training_data.controls_store.linear_grid.shape[1] + 1,
        ),
        ("shared.controls.jump_ts_lengths", -1),
        (
            "shared.controls.jump_ts_lengths",
            producer_data.training_data.controls_store.jump_ts.shape[1] + 1,
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


def test_rejects_invalid_solver_window_bounds(
    tmp_path, producer_data, scales, rhs_names
):
    process_order = tuple(producer_data.training_data.process_order)
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
            producer_data=producer_data,
            folds=((fold, scales),),
            rhs_names=rhs_names,
        )
        manifest = json.loads((artifact / "manifest.json").read_text())
        manifest["base"]["controls"][name] = value
        _write_manifest(artifact, manifest)
        with pytest.raises(ValueError, match="invalid runtime controls metadata"):
            load_runtime_artifact(artifact, fold_id=1)


def test_loaded_metadata_is_immutable(tmp_path, producer_data, scales, rhs_names):
    artifact = tmp_path / "artifact"
    process_order = tuple(producer_data.training_data.process_order)
    fold = RuntimeArtifactFold(1, (process_order[0],), (process_order[1],), "one", 1)
    write_runtime_artifact(
        artifact,
        producer_data=producer_data,
        folds=((fold, scales),),
        rhs_names=rhs_names,
    )
    loaded = load_runtime_artifact(artifact, fold_id=1)
    store = loaded.training_data
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


def test_rejects_invalid_ordered_time_axes(tmp_path, producer_data, scales, rhs_names):
    artifact = tmp_path / "artifact"
    process_order = tuple(producer_data.training_data.process_order)
    fold = RuntimeArtifactFold(1, (process_order[0],), (process_order[1],), "one", 1)
    write_runtime_artifact(
        artifact,
        producer_data=producer_data,
        folds=((fold, scales),),
        rhs_names=rhs_names,
    )
    names = (
        "shared.controls.linear_grid",
        "shared.controls.spline_breaks",
        "shared.store.t_measured",
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


def test_rejects_invalid_runtime_metadata(tmp_path, producer_data, scales, rhs_names):
    process_order = tuple(producer_data.training_data.process_order)
    fold = RuntimeArtifactFold(1, (process_order[0],), (process_order[1],), "one", 1)

    missing_metadata = tmp_path / "missing-metadata"
    write_runtime_artifact(
        missing_metadata,
        producer_data=producer_data,
        folds=((fold, scales),),
        rhs_names=rhs_names,
    )
    manifest = json.loads((missing_metadata / "manifest.json").read_text())
    control_metadata = manifest["base"]["controls"]["_process_md_by_name"]
    control_metadata[process_order[0]].pop("control_metadata")
    _write_manifest(missing_metadata, manifest)
    with pytest.raises(ValueError, match="per-process control metadata"):
        load_runtime_artifact(missing_metadata, fold_id=1)

    chained_parent = tmp_path / "chained-parent"
    write_runtime_artifact(
        chained_parent,
        producer_data=producer_data,
        folds=((fold, scales),),
        rhs_names=rhs_names,
    )
    manifest = json.loads((chained_parent / "manifest.json").read_text())
    parents = manifest["base"]["augmentation_parents"]
    child = next(index for index, parent in enumerate(parents) if parent is not None)
    parents[parents.index(None)] = process_order[child]
    _write_manifest(chained_parent, manifest)
    with pytest.raises(ValueError, match="invalid augmentation parent"):
        load_runtime_artifact(chained_parent, fold_id=1)


def test_rejects_invalid_active_sample_events(
    tmp_path, producer_data, scales, rhs_names
):
    artifact = tmp_path / "artifact"
    process_order = tuple(producer_data.training_data.process_order)
    fold = RuntimeArtifactFold(1, (process_order[0],), (process_order[1],), "one", 1)
    write_runtime_artifact(
        artifact,
        producer_data=producer_data,
        folds=((fold, scales),),
        rhs_names=rhs_names,
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


@pytest.mark.parametrize(
    ("record", "message"),
    [
        ({"file": "training-parents.json"}, "record must contain exactly"),
        (
            {"file": "other.json", "sha256": "sha256:" + "0" * 64},
            "must use file",
        ),
        (
            {"file": "../training-parents.json", "sha256": "sha256:" + "0" * 64},
            "must use file",
        ),
        (
            {"file": "training-parents.json", "sha256": "bad"},
            "invalid digest",
        ),
        (
            {"file": "training-parents.json", "sha256": f"sha256:{'g' * 64}"},
            "invalid digest",
        ),
    ],
)
def test_rejects_invalid_parent_collection_record(
    tmp_path, producer_data, scales, rhs_names, record, message
):
    artifact = tmp_path / "artifact"
    process_order = tuple(producer_data.training_data.process_order)
    fold = RuntimeArtifactFold(1, (process_order[0],), (process_order[1],), "one", 1)
    write_runtime_artifact(
        artifact,
        producer_data=producer_data,
        folds=((fold, scales),),
        rhs_names=rhs_names,
    )
    manifest_path = artifact / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["training_parent_collection"] = record
    manifest.pop("identity")
    manifest["identity"] = runtime_artifact._digest(
        runtime_artifact._canonical_json(manifest)
    )
    manifest_path.write_text(json.dumps(manifest))

    with pytest.raises(ValueError, match=message):
        load_runtime_artifact(artifact, fold_id=1)


def test_rejects_unsupported_artifact_format(
    tmp_path, producer_data, scales, rhs_names
):
    process_order = tuple(producer_data.training_data.process_order)
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
        producer_data=producer_data,
        folds=((fold, scales),),
        rhs_names=rhs_names,
    )
    manifest_path = artifact / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["format_version"] = 2
    _write_manifest(artifact, manifest)

    with pytest.raises(ValueError, match="unsupported runtime artifact format"):
        read_runtime_artifact_metadata(artifact)
    with pytest.raises(ValueError, match="unsupported runtime artifact format"):
        load_runtime_artifact(artifact, fold_id=1)


def test_rejects_manifest_schema_changes(tmp_path, producer_data, scales, rhs_names):
    process_order = tuple(producer_data.training_data.process_order)
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
        producer_data=producer_data,
        folds=((fold, scales),),
        rhs_names=rhs_names,
    )
    manifest_path = artifact / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["unexpected"] = True
    manifest_path.write_text(json.dumps(manifest))

    with pytest.raises(ValueError, match="manifest schema"):
        load_runtime_artifact(artifact, fold_id=1)


_E2E_SIM = Path("examples/00_e2e_sim/prepared/prepared.json")


def _piecewise_collection():
    """A small real collection whose algebraic law needs a `Piecewise` branch.

    Kittler carries no `Piecewise`, and the point of this fixture is that the
    artifact never serializes the expression at all: bp-format rebuilds it from
    the parents, so an expression bp-train could not have parsed still works.
    """
    collection = load_process_collection(_E2E_SIM)
    for process in collection.processes.values():
        process.biological_ode.algebraic["X_active"] = (
            "Piecewise((biomass - product_intracellular, biomass > 1), (0.0, True))"
        )
    return collection


def _artifact_from_collection(path, collection, holdout, train):
    store = TrainingDataStore.from_collection(
        collection, target_source="reactor_components"
    )
    producer = ProducerCollectionData.from_collection(store, collection)
    process_order = tuple(store.process_order)
    scales = _resolve_estimated_scales(
        custom_module=None,
        runtime_data=producer.select_training_parents(collection, train),
        custom_cfg={},
    )
    _write_runtime_artifact(
        path,
        training_data=store,
        parent_collection=select_parent_collection(
            collection,
            original_parent_processes(process_order, producer.augmentation_parents),
        ),
        augmentation_parents=producer.augmentation_parents,
        folds=((RuntimeArtifactFold(0, holdout, train, "fold", 0), scales),),
        rhs_names=RhsNames.from_rhs_ode(store.rhs_ode),
    )
    return store


def test_reconstructed_rhs_matches_direct_build_including_piecewise(tmp_path):
    collection = _piecewise_collection()
    names = tuple(collection.processes)
    artifact = tmp_path / "artifact"
    store = _artifact_from_collection(artifact, collection, names[:1], names[1:])

    loaded = load_runtime_artifact(artifact, fold_id=0)
    reconstructed = loaded.training_data.rhs_ode
    direct = build_rhs_ode(collection.processes[names[0]])

    for field in dataclasses.fields(RhsNames):
        assert getattr(reconstructed, field.name) == getattr(direct, field.name)

    n_state = (
        len(direct.name_modeled_RMCs)
        + len(direct.name_modeled_PVs)
        + 1
        + len(direct.name_modeled_FVCs)
        + len(direct.name_modeled_SVCs)
    )
    n_u = (
        len(direct.name_controlled_FVCs)
        + len(direct.name_controlled_SVCs)
        + len(direct.name_controlled_PVs)
    )
    # Straddle the Piecewise branch point (biomass > 1) so both arms are exercised.
    for biomass in (0.5, 5.0):
        c = jnp.full(n_state, 2.0).at[1].set(biomass)
        rates = jnp.linspace(0.1, 0.9, len(direct.name_modeled_rates))
        u = jnp.full(n_u, 0.25)
        f_fvc = jnp.full(len(direct.name_modeled_FVCs), 0.1)
        f_svc = jnp.full(len(direct.name_modeled_SVCs), 0.1)
        np.testing.assert_allclose(
            np.asarray(reconstructed(c, rates, u, f_fvc, f_svc)),
            np.asarray(
                dataclasses.replace(
                    direct,
                    Cin_controlled_FVCs=store.Cin_controlled_FVCs[0],
                    Cin_modeled_FVCs=store.Cin_modeled_FVCs[0],
                )(c, rates, u, f_fvc, f_svc)
            ),
        )


def test_exact_inventory_carries_no_expression_or_trace_payload(
    tmp_path, producer_data, scales, rhs_names
):
    """The whole point of the format: canonical arrays, and nothing per-child.

    The counts are derived from the field sets rather than hard-coded, so adding
    a canonical array updates them, while any reappearance of a per-process trace,
    descriptor Cin, bounds snapshot or expression payload fails.
    """
    process_order = tuple(producer_data.training_data.process_order)
    folds = tuple(
        RuntimeArtifactFold(
            idx, (process_order[idx],), (process_order[idx - 1],), f"f{idx}", idx
        )
        for idx in (1, 2)
    )
    artifact = tmp_path / "artifact"
    write_runtime_artifact(
        artifact,
        producer_data=producer_data,
        folds=tuple((fold, scales) for fold in folds),
        rhs_names=rhs_names,
    )

    files = {
        item.relative_to(artifact).as_posix()
        for item in artifact.rglob("*")
        if item.is_file()
    }
    shared = {name for name in files if name.startswith("arrays/shared/")}
    fold_files = {name for name in files if name.startswith("arrays/folds/")}
    n_affine = sum(
        1
        for name in runtime_artifact._SCALE_NAMES
        if isinstance(getattr(scales, name), AffineScaler)
    )

    assert files == {"manifest.json", "training-parents.json"} | shared | fold_files
    assert len(shared) == len(runtime_artifact._CONTROL_ARRAYS) + len(
        runtime_artifact._STORE_ARRAYS
    )
    assert "arrays/shared/controls.min_V.npy" in shared
    assert len(fold_files) == len(folds) * (
        len(runtime_artifact._SCALE_NAMES) + n_affine
    )

    manifest = json.loads((artifact / "manifest.json").read_text())
    encoded = json.dumps(manifest)
    for forbidden in ("trace", "bound_snapshot", "expression", "plot", "shared.rhs."):
        assert forbidden not in encoded
    assert set(manifest["base"]) == {
        "identity_inputs",
        "augmentation_parents",
        "rhs",
        "store",
        "controls",
    }
    # Every rhs entry is a list of bare semantic names. Flatten before testing:
    # iterating the dict's values yields lists, so a per-value isinstance check
    # would pass no matter what the payload contained.
    declared_names = [
        name for group in manifest["base"]["rhs"].values() for name in group
    ]
    assert declared_names
    assert all(isinstance(name, str) for name in declared_names)
    assert not any(token in name for name in declared_names for token in "+-*/()<>, ")


def test_loader_rejects_parent_derived_control_partition_mismatch(
    tmp_path, producer_data, scales, rhs_names
):
    """The stored `ControlsStore` statics are re-derived, not trusted.

    Loading arrays straight from `.npy` bypasses `ControlsStore.__post_init__`,
    so without this check a tampered partition or side would silently change how
    every control is evaluated. Both corruptions keep every array shape valid,
    which is exactly why nothing else catches them.
    """
    process_order = tuple(producer_data.training_data.process_order)
    fold = RuntimeArtifactFold(1, (process_order[0],), (process_order[1],), "one", 1)

    def permute_spline_columns(controls):
        """Reorder the spline columns; every array shape stays valid."""
        controls["spline_indices"] = list(reversed(controls["spline_indices"]))

    def flip_continuity_side(controls):
        controls["continuity_side"] = (
            "left" if controls["continuity_side"] == "right" else "right"
        )

    for label, corrupt, message in (
        ("partition", permute_spline_columns, "control partition"),
        ("side", flip_continuity_side, "continuity side"),
    ):
        artifact = tmp_path / f"artifact-{label}"
        write_runtime_artifact(
            artifact,
            producer_data=producer_data,
            folds=((fold, scales),),
            rhs_names=rhs_names,
        )
        manifest = json.loads((artifact / "manifest.json").read_text())
        corrupt(manifest["base"]["controls"])
        _write_manifest(artifact, manifest)

        with pytest.raises(ValueError, match=message):
            load_runtime_artifact(artifact, fold_id=1)


def test_loader_rejects_parent_collection_with_different_rhs_axes(tmp_path):
    """A parent collection whose equations moved must not silently reshape the RHS."""
    collection = _piecewise_collection()
    names = tuple(collection.processes)
    artifact = tmp_path / "artifact"
    _artifact_from_collection(artifact, collection, names[:1], names[1:])

    tampered = load_process_collection(artifact / "training-parents.json")
    for process in tampered.processes.values():
        process.biological_ode.algebraic["extra"] = "biomass"
        process.biological_ode.derivatives["biomass"] = "q_biomass * extra"
    _rewrite_parent_collection(artifact, tampered)

    with pytest.raises(ValueError, match="reconstructed RhsOde axes differ"):
        load_runtime_artifact(artifact, fold_id=0)
