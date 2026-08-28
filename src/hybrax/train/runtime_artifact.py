"""Strict runtime artifact format."""

from __future__ import annotations

import errno
import hashlib
import json
import os
import shutil
import uuid
from dataclasses import dataclass, fields
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from hybrax.format.dataclasses import AugmentedBioProcess, BioProcessCollection
from hybrax.format.mechanistic import RhsOde, build_rhs_ode
from hybrax.format.validate import validate_reaction_ode_equivalence
from hybrax.format.serialization import load_process_collection, save_process_collection

from .controls_store import ControlsStore, derive_control_partition
from .model_api import AffineScaler, EstimatedScales, LinearScaler, Scaler
from .runtime_context import (
    canonical_training_parents,
    original_parent_processes,
    select_parent_collection,
)
from .training_data import TrainingDataStore

FORMAT_VERSION = 4
_CONTROL_ARRAYS = (
    "spline_breaks",
    "spline_coeffs",
    "linear_grid",
    "control_values",
    "control_derivatives",
    "jump_ts",
    "grid_lengths",
    "jump_ts_lengths",
    "min_V",
    "sample_event_times",
    "sample_event_volumes",
    "sample_event_mask",
    "bolus_event_times",
    "bolus_event_volumes",
    "bolus_event_Cin",
    "bolus_event_mask",
)
_PROCESS_MATRIX_NAMES = (
    "Cin_controlled_Inflows",
    "Cin_modeled_Inflows",
    "retention_controlled_Outflows",
    "retention_modeled_Outflows",
)
_STORE_ARRAYS = (
    *_PROCESS_MATRIX_NAMES,
    "t_measured",
    "y_measured",
    "mask_measured",
    "n_measured",
    "y0_measured",
)
_SCALE_NAMES = tuple(field.name for field in fields(EstimatedScales))
_SLUG_CHARACTERS = frozenset("+-._")


@dataclass(frozen=True)
class RhsNames:
    """Semantic RHS axes: enough to validate array shapes, never the equations.

    The equations come back from hybrax.format's ``build_rhs_ode()`` on a
    training parent, so no biological expression is ever serialized.
    """

    name_modeled_rates: tuple[str, ...]
    name_modeled_algebraic: tuple[str, ...]
    name_modeled_RMCs: tuple[str, ...]
    name_modeled_PVs: tuple[str, ...]
    name_modeled_Inflows: tuple[str, ...]
    name_modeled_Outflows: tuple[str, ...]
    name_controlled_PVs: tuple[str, ...]
    name_controlled_Inflows: tuple[str, ...]
    name_controlled_Outflows: tuple[str, ...]

    @classmethod
    def from_rhs_ode(cls, rhs_ode) -> RhsNames:
        """Extract every :class:`RhsNames` field from an ``RhsOde`` by name."""
        return cls(
            **{field.name: tuple(getattr(rhs_ode, field.name)) for field in fields(cls)}
        )


@dataclass(frozen=True)
class RuntimeArtifactFold:
    """One LOO fold's identity: its index, test/train process sets, slug, and seed."""

    idx: int
    test: tuple[str, ...]
    train: tuple[str, ...]
    slug: str
    seed: int


@dataclass(frozen=True)
class RuntimeArtifact:
    """One fold's loaded runtime inputs.

    ``training_parent_collection`` is already filtered to the parents represented
    by ``fold.train``; ``augmentation_parents`` is the canonical full mapping, kept
    so consumers can re-derive that selection independently instead of trusting
    the filtered collection's own keys.
    """

    identity: str
    training_data: TrainingDataStore
    scales: EstimatedScales
    fold: RuntimeArtifactFold
    training_parent_collection: BioProcessCollection
    augmentation_parents: tuple[str | None, ...]


@dataclass(frozen=True)
class RuntimeArtifactMetadata:
    """Validated manifest data available without reading numeric arrays."""

    identity: str
    identity_inputs: Mapping[str, str]
    folds: tuple[RuntimeArtifactFold, ...]


def _canonical_json(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode()


def _digest(value: bytes) -> str:
    return "sha256:" + hashlib.sha256(value).hexdigest()


def _file_digest(path: Path) -> str:
    with path.open("rb") as file:
        return "sha256:" + hashlib.file_digest(file, "sha256").hexdigest()


def _json_value(value: Any) -> Any:
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not np.isfinite(value):
            raise TypeError("manifest metadata must be finite JSON values")
        return value
    if isinstance(value, np.generic):
        return _json_value(value.item())
    if isinstance(value, (tuple, list)):
        return [_json_value(item) for item in value]
    if isinstance(value, dict) and all(isinstance(key, str) for key in value):
        return {key: _json_value(item) for key, item in value.items()}
    raise TypeError(f"manifest metadata has unsupported type {type(value).__name__}")


def _array_record(root: Path, name: str, value: jax.Array) -> dict[str, Any]:
    array = np.asarray(value)
    if array.dtype.hasobject or array.dtype.kind not in "biuf?":
        raise TypeError(f"{name}: unsupported dtype {array.dtype}")
    filename = f"arrays/{name}.npy"
    path = root / filename
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, array, allow_pickle=False)
    return {
        "file": filename,
        "dtype": array.dtype.str,
        "shape": list(array.shape),
        "sha256": _file_digest(path),
    }


def _scaler(value: Scaler, name: str, arrays: dict[str, jax.Array]) -> dict[str, str]:
    if type(value) is LinearScaler:
        arrays[f"scale.{name}.scale"] = value.scale
        return {"kind": "linear", "scale": f"scale.{name}.scale"}
    if type(value) is AffineScaler:
        arrays[f"scale.{name}.scale"] = value.scale
        arrays[f"scale.{name}.offset"] = value.offset
        return {
            "kind": "affine",
            "scale": f"scale.{name}.scale",
            "offset": f"scale.{name}.offset",
        }
    raise TypeError(f"{name}: unsupported scaler {type(value).__name__}")


def _shared_arrays(store: TrainingDataStore) -> dict[str, jax.Array]:
    """Every canonical array the artifact stores once for all folds."""
    arrays = {
        f"controls.{name}": getattr(store.controls_store, name)
        for name in _CONTROL_ARRAYS
    }
    arrays.update({f"store.{name}": getattr(store, name) for name in _STORE_ARRAYS})
    return arrays


def _validate_rhs_names(names: RhsNames) -> None:
    groups = tuple(getattr(names, field.name) for field in fields(names))
    if any(
        not isinstance(group, tuple)
        or not all(isinstance(name, str) and name for name in group)
        for group in groups
    ):
        raise ValueError("RhsOde names must be non-empty strings")
    flat = tuple(name for group in groups for name in group)
    if len(flat) != len(set(flat)):
        raise ValueError("RhsOde names must be unique")


def _rhs_names_payload(names: RhsNames) -> dict[str, list[str]]:
    _validate_rhs_names(names)
    return {field.name: list(getattr(names, field.name)) for field in fields(names)}


def _rhs_names_from_payload(raw: Any) -> RhsNames:
    expected = {field.name for field in fields(RhsNames)}
    if (
        not isinstance(raw, dict)
        or set(raw) != expected
        or any(not isinstance(value, list) for value in raw.values())
    ):
        raise ValueError("invalid RhsOde names")
    names = RhsNames(**{name: tuple(value) for name, value in raw.items()})
    _validate_rhs_names(names)
    return names


def _base_metadata(
    store: TrainingDataStore,
    rhs_names: RhsNames,
    augmentation_parents: tuple[str | None, ...],
    identity_inputs: dict[str, str],
) -> dict[str, Any]:
    controls = store.controls_store
    return {
        "identity_inputs": _json_value(identity_inputs),
        "augmentation_parents": _json_value(augmentation_parents),
        "rhs": _rhs_names_payload(rhs_names),
        "store": _json_value(
            {
                "process_order": store.process_order,
                "name_measured_RMCs": store.name_measured_RMCs,
                "name_measured_PVs": store.name_measured_PVs,
                "name_modeled_Inflows": store.name_modeled_Inflows,
                "name_modeled_Outflows": store.name_modeled_Outflows,
            }
        ),
        "controls": _json_value(
            {
                name: getattr(controls, name)
                for name in (
                    "process_order",
                    "name_controlled_Inflows",
                    "name_controlled_Outflows",
                    "name_controlled_PVs",
                    "shape_metadata",
                    "spline_indices",
                    "linear_indices",
                    "continuity_side",
                    "max_event_gap_fraction",
                    "max_measurements_per_event_gap",
                    "_process_md_by_name",
                )
            }
        ),
    }


def _validate_fold(
    fold: RuntimeArtifactFold,
    process_order: tuple[str, ...],
    augmentation_parents: tuple[str | None, ...],
) -> None:
    process_names = set(process_order)
    if (
        type(fold.idx) is not int
        or fold.idx < 0
        or type(fold.seed) is not int
        or not isinstance(fold.slug, str)
        or not fold.slug
        or fold.slug in {".", ".."}
        or any(
            not (character.isalnum() or character in _SLUG_CHARACTERS)
            for character in fold.slug
        )
        or not fold.test
        or not fold.train
        or any(not isinstance(name, str) or not name for name in fold.test + fold.train)
        or len(set(fold.test)) != len(fold.test)
        or len(set(fold.train)) != len(fold.train)
        or not set(fold.test + fold.train) <= process_names
        or set(fold.test) & set(fold.train)
    ):
        raise ValueError("invalid fold metadata")
    groups = {
        name: parent or name
        for name, parent in zip(
            process_order,
            augmentation_parents,
            strict=True,
        )
    }
    test_groups = {groups[name] for name in fold.test}
    if any(groups[name] in test_groups for name in fold.train):
        raise ValueError("fold train/test sets leak an augmentation group")


def _validate_shared_payload(
    store: TrainingDataStore,
    rhs_names: RhsNames,
    augmentation_parents: tuple[str | None, ...],
) -> None:
    _validate_rhs_names(rhs_names)
    if RhsNames.from_rhs_ode(store.rhs_ode) != rhs_names:
        raise ValueError("RhsOde names differ from the training store")
    base = _base_metadata(store, rhs_names, augmentation_parents, {})
    _validate_base_metadata(base)
    _validate_semantic_arrays(
        base,
        {
            f"shared.{name}": np.asarray(value)
            for name, value in _shared_arrays(store).items()
        },
    )


def _validate_scales(scales: EstimatedScales, rhs_names: RhsNames) -> None:
    if not isinstance(scales, EstimatedScales):
        raise TypeError("fold scales must be EstimatedScales")
    _validate_scale_arrays(
        rhs_names,
        {
            name: (
                np.asarray(getattr(scales, name).scale),
                None
                if type(getattr(scales, name)) is LinearScaler
                else np.asarray(getattr(scales, name).offset),
            )
            for name in _SCALE_NAMES
        },
    )


def _publish_directory(source: Path, destination: Path) -> None:
    """Atomically publish a completed, non-empty artifact directory."""
    try:
        os.replace(source, destination)
    except OSError as error:
        if error.errno in (errno.EEXIST, errno.ENOTEMPTY):
            raise FileExistsError(
                error.errno, os.strerror(error.errno), destination
            ) from error
        raise


def write_runtime_artifact(
    path: str | Path,
    *,
    training_data: TrainingDataStore,
    parent_collection: BioProcessCollection,
    augmentation_parents: tuple[str | None, ...],
    folds: tuple[tuple[RuntimeArtifactFold, EstimatedScales], ...],
    rhs_names: RhsNames,
    identity_inputs: dict[str, str] | None = None,
) -> str:
    """Atomically write shared runtime data and independently loadable scales."""
    if not isinstance(training_data, TrainingDataStore):
        raise TypeError("training_data must be a TrainingDataStore")
    if not isinstance(rhs_names, RhsNames):
        raise TypeError("rhs_names must be an RhsNames")
    if not folds:
        raise ValueError("folds must be non-empty")
    if not all(
        isinstance(fold, RuntimeArtifactFold) and isinstance(scales, EstimatedScales)
        for fold, scales in folds
    ):
        raise TypeError("folds must contain (RuntimeArtifactFold, EstimatedScales)")
    fold_metadata = tuple(fold for fold, _ in folds)
    identity_inputs = {} if identity_inputs is None else identity_inputs
    if not isinstance(identity_inputs, dict) or not all(
        isinstance(key, str) and isinstance(value, str)
        for key, value in identity_inputs.items()
    ):
        raise TypeError("identity_inputs must contain string keys and values")
    process_order = tuple(training_data.process_order)
    augmentation_parents = tuple(augmentation_parents)
    _validate_shared_payload(training_data, rhs_names, augmentation_parents)
    expected_parent_names = original_parent_processes(
        process_order, augmentation_parents
    )
    _validate_training_parent_collection_identity(
        parent_collection, expected_parent_names
    )
    _validate_process_matrices(
        {
            name: np.asarray(getattr(training_data, name))
            for name in _PROCESS_MATRIX_NAMES
        },
        parent_collection,
        process_order,
        augmentation_parents,
        rhs_names,
    )
    for fold, scales in folds:
        _validate_fold(fold, process_order, augmentation_parents)
        _validate_scales(scales, rhs_names)
    if len({fold.idx for fold in fold_metadata}) != len(fold_metadata) or len(
        {fold.slug for fold in fold_metadata}
    ) != len(fold_metadata):
        raise ValueError("fold IDs and slugs must be unique")
    path = Path(path)
    if path.exists():
        raise FileExistsError(path)
    temporary = path.parent / f".{path.name}.{uuid.uuid4().hex}.tmp"
    temporary.mkdir(parents=True)
    try:
        records = {
            f"shared.{name}": _array_record(temporary, f"shared/{name}", value)
            for name, value in sorted(_shared_arrays(training_data).items())
        }
        fold_records = []
        for fold, estimated_scales in folds:
            scale_arrays: dict[str, jax.Array] = {}
            scales = {
                name: _scaler(getattr(estimated_scales, name), name, scale_arrays)
                for name in _SCALE_NAMES
            }
            records.update(
                {
                    f"fold.{fold.idx}.{name}": _array_record(
                        temporary, f"folds/{fold.idx}/{name}", value
                    )
                    for name, value in sorted(scale_arrays.items())
                }
            )
            fold_records.append({"fold": _json_value(fold.__dict__), "scales": scales})
        base = _base_metadata(
            training_data, rhs_names, augmentation_parents, identity_inputs
        )
        parent_collection_path = temporary / "training-parents.json"
        save_process_collection(parent_collection, parent_collection_path)
        parent_collection_record = {
            "file": parent_collection_path.name,
            "sha256": _file_digest(parent_collection_path),
        }
        manifest = {
            "format_version": FORMAT_VERSION,
            "base": base,
            "arrays": records,
            "training_parent_collection": parent_collection_record,
            "folds": fold_records,
        }
        manifest["identity"] = _digest(_canonical_json(manifest))
        (temporary / "manifest.json").write_bytes(_canonical_json(manifest))
        _publish_directory(temporary, path)
        return manifest["identity"]
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise


def _read_manifest(root: Path) -> dict[str, Any]:
    try:
        manifest = json.loads((root / "manifest.json").read_text())
    except Exception as error:
        raise ValueError("invalid runtime artifact manifest") from error
    if not isinstance(manifest, dict) or set(manifest) != {
        "format_version",
        "base",
        "arrays",
        "training_parent_collection",
        "folds",
        "identity",
    }:
        raise ValueError("invalid runtime artifact manifest schema")
    if manifest["format_version"] != FORMAT_VERSION:
        raise ValueError("unsupported runtime artifact format")
    identity = manifest.pop("identity")
    if not isinstance(identity, str) or identity != _digest(_canonical_json(manifest)):
        raise ValueError("runtime artifact identity mismatch")
    manifest["identity"] = identity
    return manifest


def _verified_parent_collection_path(root: Path, record: Any) -> Path:
    if not isinstance(record, dict) or set(record) != {"file", "sha256"}:
        raise ValueError(
            "training parent collection record must contain exactly file and sha256"
        )
    if record["file"] != "training-parents.json":
        raise ValueError(
            "training parent collection record must use file training-parents.json"
        )
    digest = record["sha256"]
    if (
        not isinstance(digest, str)
        or not digest.startswith("sha256:")
        or len(digest) != 71
        or any(character not in "0123456789abcdef" for character in digest[7:])
    ):
        raise ValueError("training parent collection record has invalid digest")
    path = root / record["file"]
    if (
        path.is_symlink()
        or not path.is_file()
        or _file_digest(path) != record["sha256"]
    ):
        raise ValueError("training parent collection checksum mismatch")
    return path


def _validate_training_parent_collection_identity(
    collection: BioProcessCollection,
    required_parent_names: tuple[str, ...],
) -> None:
    if tuple(collection.processes) != required_parent_names:
        raise ValueError(
            "training parent collection must contain exactly all original parents "
            "in canonical order"
        )
    if any(
        isinstance(process, AugmentedBioProcess)
        for process in collection.processes.values()
    ):
        raise ValueError("training parent collection contains an augmented process")

    hybrax_train_metadata = (collection.metadata or {}).get("hybrax.train")
    if hybrax_train_metadata is None:
        return
    if not isinstance(hybrax_train_metadata, dict):
        raise ValueError("invalid training parent collection structural metadata")
    if "process_order" in hybrax_train_metadata:
        process_order = hybrax_train_metadata["process_order"]
        if (
            not isinstance(process_order, list)
            or tuple(process_order) != required_parent_names
        ):
            raise ValueError("invalid training parent collection structural metadata")
    if "processes" in hybrax_train_metadata:
        process_metadata = hybrax_train_metadata["processes"]
        if (
            not isinstance(process_metadata, dict)
            or tuple(process_metadata) != required_parent_names
        ):
            raise ValueError("invalid training parent collection structural metadata")


def _load_training_parent_collection(
    path: Path,
    *,
    process_order: tuple[str, ...],
    augmentation_parents: tuple[str | None, ...],
) -> BioProcessCollection:
    try:
        collection = load_process_collection(path)
    except Exception as error:
        raise ValueError("invalid training parent collection") from error
    _validate_training_parent_collection_identity(
        collection, original_parent_processes(process_order, augmentation_parents)
    )
    return collection


def _array_filename(name: str, record: Any) -> str:
    if not isinstance(record, dict) or set(record) != {
        "file",
        "dtype",
        "shape",
        "sha256",
    }:
        raise ValueError(f"{name}: invalid array record")
    filename = record["file"]
    try:
        dtype = np.dtype(record["dtype"])
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name}: invalid array record") from error
    if (
        not isinstance(filename, str)
        or not filename.startswith("arrays/")
        or not filename.endswith(".npy")
        or Path(filename).is_absolute()
        or ".." in Path(filename).parts
        or dtype.hasobject
        or dtype.kind not in "biuf?"
        or not isinstance(record["shape"], list)
        or any(type(size) is not int or size < 0 for size in record["shape"])
        or not isinstance(record["sha256"], str)
        or not record["sha256"].startswith("sha256:")
        or len(record["sha256"]) != 71
    ):
        raise ValueError(f"{name}: invalid array record")
    return filename


def _read_array(root: Path, name: str, record: Any) -> np.ndarray:
    filename = _array_filename(name, record)
    path = root / filename
    if not path.is_file() or _file_digest(path) != record["sha256"]:
        raise ValueError(f"{name}: checksum mismatch")
    try:
        array = np.load(path, allow_pickle=False)
    except Exception as error:
        raise ValueError(f"{name}: invalid array") from error
    if array.dtype.str != record["dtype"] or list(array.shape) != record["shape"]:
        raise ValueError(f"{name}: dtype or shape mismatch")
    array.setflags(write=False)
    return array


def _validate_exact_file_inventory(root: Path, expected_files: set[str]) -> None:
    actual_files = {
        item.relative_to(root).as_posix() for item in root.rglob("*") if item.is_file()
    }
    if actual_files != expected_files:
        raise ValueError("runtime artifact has missing or extra files")


def _rhs_process_matrices(rhs_ode: RhsOde) -> tuple[np.ndarray, ...]:
    return tuple(np.asarray(getattr(rhs_ode, name)) for name in _PROCESS_MATRIX_NAMES)


def _validate_process_matrices(
    matrix_rows: dict[str, np.ndarray],
    parent_collection: BioProcessCollection,
    process_order: tuple[str, ...],
    augmentation_parents: tuple[str | None, ...],
    rhs_names: RhsNames,
) -> None:
    equivalent, message = validate_reaction_ode_equivalence(parent_collection)
    if not equivalent:
        raise ValueError(message)

    parent_rhs: dict[str, RhsOde] = {}
    for process_name in original_parent_processes(process_order, augmentation_parents):
        rhs_ode = build_rhs_ode(parent_collection.processes[process_name])
        if RhsNames.from_rhs_ode(rhs_ode) != rhs_names:
            raise ValueError(
                f"{process_name}: reconstructed RhsOde axes differ from the runtime "
                "artifact"
            )
        parent_rhs[process_name] = rhs_ode

    for row, (process_name, parent_name) in enumerate(
        zip(process_order, augmentation_parents, strict=True)
    ):
        expected = _rhs_process_matrices(
            parent_rhs[process_name if parent_name is None else parent_name]
        )
        actual = tuple(matrix_rows[name][row] for name in _PROCESS_MATRIX_NAMES)
        if any(
            not np.array_equal(got, want)
            for got, want in zip(actual, expected, strict=True)
        ):
            raise ValueError(
                f"{process_name}: cached process matrices differ from its canonical "
                "parent RhsOde"
            )


def _reconstruct_rhs_ode(
    parent_collection: BioProcessCollection,
    rhs_names: RhsNames,
    arrays: dict[str, np.ndarray],
    process_row: int,
) -> RhsOde:
    """Build equations from the selected canonical parent and cached matrices."""
    try:
        process = next(iter(parent_collection.processes.values()))
    except StopIteration as error:
        raise ValueError(
            "runtime artifact requires a non-empty parent collection"
        ) from error
    rhs_ode = build_rhs_ode(process)
    if RhsNames.from_rhs_ode(rhs_ode) != rhs_names:
        raise ValueError("reconstructed RhsOde axes differ from the runtime artifact")
    return eqx.tree_at(
        lambda rhs: tuple(getattr(rhs, name) for name in _PROCESS_MATRIX_NAMES),
        rhs_ode,
        tuple(
            jnp.asarray(arrays[f"shared.store.{name}"][process_row])
            for name in _PROCESS_MATRIX_NAMES
        ),
    )


def _validate_control_partition(
    parent_collection: BioProcessCollection,
    controls: dict[str, Any],
) -> None:
    """Re-derive the control layout from the parents and reject a stored mismatch.

    Loading ``.npy`` arrays straight into
    :class:`~hybrax.train.controls_store.ControlsStore`
    bypasses its ``__post_init__``, so these statics are otherwise taken on trust.
    """
    partition = derive_control_partition(parent_collection)
    stored = (
        tuple(controls["name_controlled_Inflows"]),
        tuple(controls["name_controlled_Outflows"]),
        tuple(controls["name_controlled_PVs"]),
        tuple(controls["spline_indices"]),
        tuple(controls["linear_indices"]),
    )
    derived = (
        partition.name_controlled_Inflows,
        partition.name_controlled_Outflows,
        partition.name_controlled_PVs,
        partition.spline_indices,
        partition.linear_indices,
    )
    if stored != derived:
        raise ValueError(
            "runtime artifact control partition differs from its parent collection"
        )
    if (
        partition.continuity_side is not None
        and partition.continuity_side != controls["continuity_side"]
    ):
        raise ValueError(
            "runtime artifact continuity side differs from its parent collection"
        )


def _validate_base_metadata(
    base: Any,
) -> tuple[tuple[str, ...], tuple[str | None, ...]]:
    required_base_keys = {
        "identity_inputs",
        "augmentation_parents",
        "rhs",
        "store",
        "controls",
    }
    if not isinstance(base, dict) or set(base) != required_base_keys:
        raise ValueError("invalid runtime artifact schema")
    if not isinstance(base["identity_inputs"], dict) or not all(
        isinstance(key, str) and isinstance(value, str)
        for key, value in base["identity_inputs"].items()
    ):
        raise ValueError("invalid runtime artifact identity inputs")
    _rhs_names_from_payload(base["rhs"])
    store = base["store"]
    controls = base["controls"]
    if not isinstance(store, dict) or set(store) != {
        "process_order",
        "name_measured_RMCs",
        "name_measured_PVs",
        "name_modeled_Inflows",
        "name_modeled_Outflows",
    }:
        raise ValueError("invalid runtime store metadata")
    process_order = store["process_order"]
    if (
        not isinstance(process_order, list)
        or not process_order
        or any(not isinstance(name, str) or not name for name in process_order)
        or len(process_order) != len(set(process_order))
    ):
        raise ValueError("invalid runtime process order")
    for key, values in store.items():
        if (
            not isinstance(values, list)
            or any(not isinstance(value, str) or not value for value in values)
            or len(values) != len(set(values))
        ):
            raise ValueError(f"invalid runtime store metadata {key}")
    if not isinstance(controls, dict) or set(controls) != {
        "process_order",
        "name_controlled_Inflows",
        "name_controlled_Outflows",
        "name_controlled_PVs",
        "shape_metadata",
        "spline_indices",
        "linear_indices",
        "continuity_side",
        "max_event_gap_fraction",
        "max_measurements_per_event_gap",
        "_process_md_by_name",
    }:
        raise ValueError("invalid runtime controls metadata")
    if controls["process_order"] != process_order:
        raise ValueError("runtime control process order mismatch")
    for key in (
        "name_controlled_Inflows",
        "name_controlled_Outflows",
        "name_controlled_PVs",
    ):
        values = controls[key]
        if (
            not isinstance(values, list)
            or any(not isinstance(value, str) or not value for value in values)
            or len(values) != len(set(values))
        ):
            raise ValueError(f"invalid runtime controls metadata {key}")
    if (
        not isinstance(controls["shape_metadata"], dict)
        or not isinstance(controls["spline_indices"], list)
        or not isinstance(controls["linear_indices"], list)
        or any(
            type(index) is not int
            for index in controls["spline_indices"] + controls["linear_indices"]
        )
        or controls["continuity_side"] not in {"left", "right"}
        or type(controls["max_measurements_per_event_gap"]) is not int
        or not isinstance(controls["max_event_gap_fraction"], (int, float))
        or not np.isfinite(controls["max_event_gap_fraction"])
        or not 0 <= controls["max_event_gap_fraction"] <= 1
        or controls["max_measurements_per_event_gap"] < 0
        or not isinstance(controls["_process_md_by_name"], dict)
        or set(controls["_process_md_by_name"]) != set(process_order)
    ):
        raise ValueError("invalid runtime controls metadata")
    control_name_keys = (
        "name_controlled_Inflows",
        "name_controlled_Outflows",
        "name_controlled_PVs",
    )
    canonical_control_names = set(
        controls["name_controlled_Inflows"]
        + controls["name_controlled_Outflows"]
        + controls["name_controlled_PVs"]
    )
    for process_name in process_order:
        process_metadata = controls["_process_md_by_name"][process_name]
        if (
            not isinstance(process_metadata, dict)
            or set(process_metadata)
            != {*control_name_keys, "control_metadata", "control_supports"}
            or any(process_metadata[key] != controls[key] for key in control_name_keys)
            or not isinstance(process_metadata["control_metadata"], dict)
            or set(process_metadata["control_metadata"]) != canonical_control_names
            or not _valid_json_metadata(process_metadata["control_metadata"])
            or not isinstance(process_metadata["control_supports"], dict)
            or set(process_metadata["control_supports"]) != canonical_control_names
        ):
            raise ValueError("invalid per-process control metadata")
        for support in process_metadata["control_supports"].values():
            if (
                not isinstance(support, list)
                or len(support) != 2
                or any(
                    bound is not None
                    and (not isinstance(bound, (int, float)) or not np.isfinite(bound))
                    for bound in support
                )
                or (support[0] is None) != (support[1] is None)
                or (
                    support[0] is not None
                    and support[1] is not None
                    and support[0] > support[1]
                )
            ):
                raise ValueError("invalid per-process control metadata")
    parents = base["augmentation_parents"]
    if not isinstance(parents, list) or len(parents) != len(process_order):
        raise ValueError("augmentation parent count differs from process order")
    if any(
        parent is not None
        and (not isinstance(parent, str) or parent not in process_order)
        for parent in parents
    ):
        raise ValueError("invalid augmentation parent")
    parent_by_name = dict(zip(process_order, parents, strict=True))
    if any(
        parent == name or (parent is not None and parent_by_name[parent] is not None)
        for name, parent in parent_by_name.items()
    ):
        raise ValueError("invalid augmentation parent")
    return tuple(process_order), tuple(parents)


def _fold_from_manifest(
    raw: Any,
    process_order: tuple[str, ...],
    augmentation_parents: tuple[str | None, ...],
) -> RuntimeArtifactFold:
    try:
        if not isinstance(raw, dict) or set(raw) != {
            "idx",
            "test",
            "train",
            "slug",
            "seed",
        }:
            raise ValueError
        if not isinstance(raw["test"], list) or not isinstance(raw["train"], list):
            raise ValueError
        fold = RuntimeArtifactFold(
            raw["idx"],
            tuple(raw["test"]),
            tuple(raw["train"]),
            raw["slug"],
            raw["seed"],
        )
        _validate_fold(fold, process_order, augmentation_parents)
        return fold
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("invalid fold manifest") from error


def _scale_array_keys(raw: Any, fold_id: int) -> set[str]:
    if not isinstance(raw, dict) or set(raw) != set(_SCALE_NAMES):
        raise ValueError("invalid scale manifest")
    keys = set()
    for name, value in raw.items():
        linear = {"kind": "linear", "scale": f"scale.{name}.scale"}
        affine = {
            "kind": "affine",
            "scale": f"scale.{name}.scale",
            "offset": f"scale.{name}.offset",
        }
        if value not in (linear, affine):
            raise ValueError("invalid scale manifest")
        keys.update(
            f"fold.{fold_id}.{array_name}"
            for key, array_name in value.items()
            if key != "kind"
        )
    return keys


def _freeze_metadata(value: Any) -> Any:
    if isinstance(value, dict):
        return MappingProxyType(
            {key: _freeze_metadata(item) for key, item in value.items()}
        )
    if isinstance(value, list):
        return tuple(_freeze_metadata(item) for item in value)
    return value


def _tuple_metadata(raw: dict[str, Any]) -> dict[str, Any]:
    return {key: _freeze_metadata(value) for key, value in raw.items()}


def _strictly_increasing(values: np.ndarray) -> bool:
    return values.size < 2 or bool(np.all(np.diff(values) > 0))


def _nondecreasing(values: np.ndarray) -> bool:
    return values.size < 2 or bool(np.all(np.diff(values) >= 0))


def _outside_time_window(
    values: np.ndarray,
    start: float,
    end: float,
    *,
    include_end: bool = True,
) -> bool:
    if values.size == 0:
        return False
    tolerance = (
        4
        * max(np.finfo(values.dtype).eps, np.finfo(np.float32).eps)
        * max(1.0, abs(float(start)), abs(float(end)))
    )
    below_start = np.any(values < start - tolerance)
    above_end = (
        np.any(values > end + tolerance) if include_end else np.any(values >= end)
    )
    return bool(below_start or above_end)


def _polynomial_has_wrong_sign(
    coeffs: np.ndarray, width: float, *, positive: bool
) -> bool:
    candidates = [0.0, width]
    for root in np.polynomial.polynomial.polyroots(
        np.polynomial.polynomial.polyder(coeffs)
    ):
        if np.isreal(root) and 0.0 <= float(np.real(root)) <= width:
            candidates.append(float(np.real(root)))
    values = np.polynomial.polynomial.polyval(candidates, coeffs)
    return bool(np.any(values < 0 if positive else values > 0))


def _validate_flow_control_signs(
    controls: dict[str, Any],
    arrays: dict[str, np.ndarray],
    *,
    n_inflows: int,
    n_outflows: int,
) -> None:
    flow_end = n_inflows + n_outflows
    grid_lengths = np.asarray(arrays["shared.controls.grid_lengths"])
    values = np.asarray(arrays["shared.controls.control_values"])
    derivatives = np.asarray(arrays["shared.controls.control_derivatives"])
    for column, control_index in enumerate(controls["linear_indices"]):
        if control_index >= flow_end:
            continue
        positive = control_index < n_inflows
        for row, length in enumerate(grid_lengths):
            active = slice(0, int(length))
            for data in (values[row, active, column], derivatives[row, active, column]):
                if np.any(data < 0 if positive else data > 0):
                    raise ValueError("runtime controls contain sign-invalid flows")

    breaks = np.asarray(arrays["shared.controls.spline_breaks"])
    coeffs = np.asarray(arrays["shared.controls.spline_coeffs"])
    for column, control_index in enumerate(controls["spline_indices"]):
        if control_index >= flow_end:
            continue
        positive = control_index < n_inflows
        for row, row_breaks in enumerate(breaks):
            finite = row_breaks[np.isfinite(row_breaks)]
            for segment, width in enumerate(np.diff(finite)):
                polynomial = coeffs[row, segment, column]
                if _polynomial_has_wrong_sign(
                    polynomial, float(width), positive=positive
                ) or _polynomial_has_wrong_sign(
                    np.polynomial.polynomial.polyder(polynomial),
                    float(width),
                    positive=positive,
                ):
                    raise ValueError("runtime controls contain sign-invalid flows")


def _valid_json_metadata(value: Any) -> bool:
    if value is None or isinstance(value, (str, bool)):
        return True
    if type(value) is int:
        return True
    if isinstance(value, float):
        return bool(np.isfinite(value))
    if isinstance(value, list):
        return all(_valid_json_metadata(item) for item in value)
    if isinstance(value, dict):
        return all(
            isinstance(key, str) and _valid_json_metadata(item)
            for key, item in value.items()
        )
    return False


def _expect_array(
    arrays: dict[str, np.ndarray],
    name: str,
    shape: tuple[int, ...],
    kind: str,
) -> np.ndarray:
    try:
        array = np.asarray(arrays[name])
    except KeyError as error:
        raise ValueError(f"{name}: missing runtime array") from error
    if array.shape != shape or array.dtype.kind not in kind:
        raise ValueError(f"{name}: invalid semantic dtype or shape")
    return array


def _validate_semantic_arrays(
    base: dict[str, Any],
    arrays: dict[str, np.ndarray],
) -> None:
    descriptor = _rhs_names_from_payload(base["rhs"])
    store = base["store"]
    controls = base["controls"]
    n_processes = len(store["process_order"])
    n_rmc = len(descriptor.name_modeled_RMCs)
    n_pv = len(descriptor.name_modeled_PVs)
    n_modeled_inflow = len(descriptor.name_modeled_Inflows)
    n_modeled_outflow = len(descriptor.name_modeled_Outflows)
    n_controlled_inflow = len(descriptor.name_controlled_Inflows)
    n_controlled_outflow = len(descriptor.name_controlled_Outflows)
    n_controlled_pv = len(descriptor.name_controlled_PVs)
    n_controls = n_controlled_inflow + n_controlled_outflow + n_controlled_pv
    n_targets = len(store["name_measured_RMCs"]) + len(store["name_measured_PVs"])
    n_y = n_targets + n_modeled_inflow + n_modeled_outflow

    if (
        not set(store["name_measured_RMCs"]) <= set(descriptor.name_modeled_RMCs)
        or not set(store["name_measured_PVs"]) <= set(descriptor.name_modeled_PVs)
        or tuple(store["name_modeled_Inflows"]) != descriptor.name_modeled_Inflows
        or tuple(store["name_modeled_Outflows"]) != descriptor.name_modeled_Outflows
        or tuple(controls["name_controlled_Inflows"])
        != descriptor.name_controlled_Inflows
        or tuple(controls["name_controlled_Outflows"])
        != descriptor.name_controlled_Outflows
        or tuple(controls["name_controlled_PVs"]) != descriptor.name_controlled_PVs
    ):
        raise ValueError("runtime metadata differs from RhsOde descriptor")

    spline_indices = tuple(controls["spline_indices"])
    linear_indices = tuple(controls["linear_indices"])
    if sorted(spline_indices + linear_indices) != list(range(n_controls)):
        raise ValueError("invalid runtime control indices")

    spline_breaks = np.asarray(arrays["shared.controls.spline_breaks"])
    linear_grid = np.asarray(arrays["shared.controls.linear_grid"])
    jump_ts = np.asarray(arrays["shared.controls.jump_ts"])
    sample_times = np.asarray(arrays["shared.controls.sample_event_times"])
    bolus_times = np.asarray(arrays["shared.controls.bolus_event_times"])
    if any(
        array.ndim != 2 or array.shape[0] != n_processes
        for array in (
            spline_breaks,
            linear_grid,
            jump_ts,
            sample_times,
            bolus_times,
        )
    ):
        raise ValueError("invalid runtime control process axes")
    n_spline_breaks = spline_breaks.shape[1]
    n_grid = linear_grid.shape[1]
    n_jump = jump_ts.shape[1]
    n_sample = sample_times.shape[1]
    n_bolus = bolus_times.shape[1]
    expected_shape_metadata = {
        "n_processes": n_processes,
        "max_grid_length": n_grid,
        "max_spline_breaks": n_spline_breaks,
        "max_controls": n_controls,
        "max_jump_ts_length": n_jump,
        "max_sample_events": n_sample,
        "max_bolus_events": n_bolus,
    }
    if controls["shape_metadata"] != expected_shape_metadata:
        raise ValueError("invalid runtime control shape metadata")

    float_arrays = {
        "spline_breaks": (n_processes, n_spline_breaks),
        "spline_coeffs": (
            n_processes,
            max(n_spline_breaks - 1, 0),
            len(spline_indices),
            4,
        ),
        "linear_grid": (n_processes, n_grid),
        "control_values": (n_processes, n_grid, len(linear_indices)),
        "control_derivatives": (n_processes, n_grid, len(linear_indices)),
        "jump_ts": (n_processes, n_jump),
        "min_V": (n_processes,),
        "sample_event_times": (n_processes, n_sample),
        "sample_event_volumes": (n_processes, n_sample),
        "bolus_event_times": (n_processes, n_bolus),
        "bolus_event_volumes": (n_processes, n_bolus),
        "bolus_event_Cin": (n_processes, n_bolus, n_rmc),
    }
    validated_controls = {
        name: _expect_array(arrays, f"shared.controls.{name}", shape, "f")
        for name, shape in float_arrays.items()
    }
    if any(
        not np.all(np.isfinite(validated_controls[name]))
        for name in float_arrays
        if name != "spline_breaks"
    ):
        raise ValueError("runtime controls contain non-finite values")
    if np.any(validated_controls["min_V"] <= 0):
        raise ValueError("shared.controls.min_V: values must be positive")
    breaks = validated_controls["spline_breaks"]
    if np.any(np.isnan(breaks)) or np.any(np.isneginf(breaks)):
        raise ValueError("runtime spline breaks contain invalid values")
    for row in breaks:
        infinite = np.isposinf(row)
        if np.any(infinite) and not np.all(infinite[np.argmax(infinite) :]):
            raise ValueError("runtime spline break padding is invalid")
        finite = row[np.isfinite(row)]
        if not _strictly_increasing(finite):
            raise ValueError("runtime spline breaks must be strictly increasing")

    grid_lengths = _expect_array(
        arrays, "shared.controls.grid_lengths", (n_processes,), "iu"
    )
    jump_lengths = _expect_array(
        arrays, "shared.controls.jump_ts_lengths", (n_processes,), "iu"
    )
    if np.any(grid_lengths < 1) or np.any(grid_lengths > n_grid):
        raise ValueError("shared.controls.grid_lengths: invalid active lengths")
    if np.any(jump_lengths < 0) or np.any(jump_lengths > n_jump):
        raise ValueError("shared.controls.jump_ts_lengths: invalid active lengths")

    event_masks = {
        name: _expect_array(arrays, f"shared.controls.{name}", shape, "b")
        for name, shape in {
            "sample_event_mask": (n_processes, n_sample),
            "bolus_event_mask": (n_processes, n_bolus),
        }.items()
    }
    for name, mask in event_masks.items():
        expected = np.arange(mask.shape[1]) < np.sum(mask, axis=1, keepdims=True)
        if not np.array_equal(mask, expected):
            raise ValueError(f"shared.controls.{name}: mask must be an active prefix")

    # Each process's solve window is not stored: the linear grid is built from the
    # process start/end plus that process's own raw knots, so its active endpoints
    # are the window. Deriving it here keeps every other time axis checked against
    # data the artifact already had to get right, with nothing extra to trust.
    process_bounds = [
        (
            float(linear_grid[row, 0]),
            float(linear_grid[row, int(grid_lengths[row]) - 1]),
        )
        for row in range(n_processes)
    ]
    for row, (start, end) in enumerate(process_bounds):
        active_grid = linear_grid[row, : int(grid_lengths[row])]
        active_jumps = jump_ts[row, : int(jump_lengths[row])]
        if not _strictly_increasing(active_grid):
            raise ValueError("shared.controls.linear_grid: invalid active time axis")
        if not _strictly_increasing(active_jumps) or _outside_time_window(
            active_jumps, start, end
        ):
            raise ValueError("shared.controls.jump_ts: invalid active time axis")
        for kind, times, strict_end in (
            ("sample", sample_times, False),
            ("bolus", bolus_times, True),
        ):
            mask = event_masks[f"{kind}_event_mask"][row]
            active_times = times[row, mask]
            volumes = validated_controls[f"{kind}_event_volumes"][row, mask]
            ordered = (
                _strictly_increasing(active_times)
                if kind == "sample"
                else _nondecreasing(active_times)
            )
            if not ordered or _outside_time_window(
                active_times, start, end, include_end=not strict_end
            ):
                raise ValueError(
                    f"shared.controls.{kind}_event_times: invalid active time axis"
                )
            if np.any(volumes <= 0):
                raise ValueError(
                    f"shared.controls.{kind}_event_volumes: values must be positive"
                )

    _expect_array(
        arrays,
        "shared.store.Cin_controlled_Inflows",
        (n_processes, n_controlled_inflow, n_rmc),
        "f",
    )
    _expect_array(
        arrays,
        "shared.store.Cin_modeled_Inflows",
        (n_processes, n_modeled_inflow, n_rmc),
        "f",
    )
    _expect_array(
        arrays,
        "shared.store.retention_controlled_Outflows",
        (n_processes, n_controlled_outflow, n_rmc),
        "f",
    )
    _expect_array(
        arrays,
        "shared.store.retention_modeled_Outflows",
        (n_processes, n_modeled_outflow, n_rmc),
        "f",
    )
    for name in _PROCESS_MATRIX_NAMES:
        values = np.asarray(arrays[f"shared.store.{name}"])
        if not np.all(np.isfinite(values)):
            raise ValueError(f"shared.store.{name}: non-finite values")
        if name.startswith("retention_") and (np.any(values < 0) or np.any(values > 1)):
            raise ValueError(f"shared.store.{name}: values must be within [0, 1]")
    measured_times = np.asarray(arrays["shared.store.t_measured"])
    if measured_times.ndim != 2 or measured_times.shape[0] != n_processes:
        raise ValueError("shared.store.t_measured: invalid semantic dtype or shape")
    n_measured_times = measured_times.shape[1]
    _expect_array(
        arrays,
        "shared.store.t_measured",
        (n_processes, n_measured_times),
        "f",
    )
    for name, kind in (("y_measured", "f"), ("mask_measured", "b")):
        _expect_array(
            arrays,
            f"shared.store.{name}",
            (n_processes, n_measured_times, n_y),
            kind,
        )
    n_measured = _expect_array(arrays, "shared.store.n_measured", (n_processes,), "iu")
    if np.any(n_measured < 0) or np.any(n_measured > n_measured_times):
        raise ValueError("shared.store.n_measured: values exceed padded dimensions")
    y0 = _expect_array(
        arrays,
        "shared.store.y0_measured",
        (n_processes, n_rmc + n_pv + 1 + n_modeled_inflow + n_modeled_outflow),
        "f",
    )
    if not np.all(np.isfinite(measured_times)) or not np.all(
        np.isfinite(arrays["shared.store.y_measured"])
    ):
        raise ValueError("runtime measurements contain non-finite values")
    if not np.all(np.isfinite(y0)):
        raise ValueError("shared.store.y0_measured: non-finite values")
    y_measured = np.asarray(arrays["shared.store.y_measured"])
    inflow_slice = slice(n_targets, n_targets + n_modeled_inflow)
    outflow_slice = slice(n_targets + n_modeled_inflow, n_y)
    y0_flow_start = n_rmc + n_pv + 1
    y0_inflow_slice = slice(y0_flow_start, y0_flow_start + n_modeled_inflow)
    y0_outflow_slice = slice(y0_flow_start + n_modeled_inflow, y0.shape[1])
    if (
        np.any(y_measured[..., inflow_slice] < 0)
        or np.any(y_measured[..., outflow_slice] > 0)
        or np.any(y0[..., y0_inflow_slice] < 0)
        or np.any(y0[..., y0_outflow_slice] > 0)
    ):
        raise ValueError("runtime measurements contain sign-invalid flows")
    _validate_flow_control_signs(
        controls,
        arrays,
        n_inflows=n_controlled_inflow,
        n_outflows=n_controlled_outflow,
    )
    mask_measured = np.asarray(arrays["shared.store.mask_measured"])
    for row, length in enumerate(n_measured):
        active_times = measured_times[row, : int(length)]
        start, end = process_bounds[row]
        if not _strictly_increasing(active_times) or _outside_time_window(
            active_times, start, end
        ):
            raise ValueError("shared.store.t_measured: invalid active time axis")
        if np.any(mask_measured[row, int(length) :]):
            raise ValueError("shared.store.mask_measured: active values after row end")


def _validate_scale_arrays(
    descriptor: RhsNames,
    scales: dict[str, tuple[np.ndarray, np.ndarray | None]],
) -> None:
    n_rmc = len(descriptor.name_modeled_RMCs)
    shapes = {
        "SCALE_modeled_RMCs": (n_rmc,),
        "SCALE_V_in_cumulative": (),
        "SCALE_modeled_Inflows_cumulative": (len(descriptor.name_modeled_Inflows),),
        "SCALE_modeled_Outflows_cumulative": (len(descriptor.name_modeled_Outflows),),
        "SCALE_controlled_Inflows_cumulative": (
            len(descriptor.name_controlled_Inflows),
        ),
        "SCALE_controlled_Outflows_cumulative": (
            len(descriptor.name_controlled_Outflows),
        ),
        "SCALE_controlled_Inflows_rates": (len(descriptor.name_controlled_Inflows),),
        "SCALE_controlled_Outflows_rates": (len(descriptor.name_controlled_Outflows),),
        "SCALE_controlled_Inflows_Cin": (
            len(descriptor.name_controlled_Inflows),
            n_rmc,
        ),
        "SCALE_controlled_PVs": (len(descriptor.name_controlled_PVs),),
        "SCALE_modeled_Inflows_Cin": (
            len(descriptor.name_modeled_Inflows),
            n_rmc,
        ),
        "SCALE_modeled_ReactionOde_rates": (len(descriptor.name_modeled_rates),),
        "SCALE_modeled_Inflows_rates": (len(descriptor.name_modeled_Inflows),),
        "SCALE_modeled_Outflows_rates": (len(descriptor.name_modeled_Outflows),),
        "SCALE_modeled_PVs": (len(descriptor.name_modeled_PVs),),
    }
    if set(scales) != set(_SCALE_NAMES) or set(shapes) != set(_SCALE_NAMES):
        raise ValueError("invalid scale arrays")
    for name, (scale, offset) in scales.items():
        scale_array = np.asarray(scale)
        offset_array = None if offset is None else np.asarray(offset)
        if (
            scale_array.dtype.kind != "f"
            or scale_array.shape != shapes[name]
            or not np.all(np.isfinite(scale_array))
            or np.any(scale_array <= 0)
            or (
                offset_array is not None
                and (
                    offset_array.dtype.kind != "f"
                    or offset_array.shape != scale_array.shape
                    or not np.all(np.isfinite(offset_array))
                )
            )
        ):
            raise ValueError(f"{name}: invalid semantic scale values or shape")


def load_runtime_artifact(path: str | Path, *, fold_id: int) -> RuntimeArtifact:
    """Load one fold; unselected fold scale files are never opened or checksummed."""
    if type(fold_id) is not int:
        raise ValueError("fold_id must be an integer")
    root = Path(path)
    manifest = _read_manifest(root)
    records = manifest["arrays"]
    base = manifest["base"]
    if not isinstance(records, dict):
        raise ValueError("invalid runtime artifact schema")
    process_order, augmentation_parents = _validate_base_metadata(base)
    raw_folds = manifest["folds"]
    if not isinstance(raw_folds, list) or not raw_folds:
        raise ValueError("invalid fold manifest")
    parsed_folds: list[tuple[RuntimeArtifactFold, dict[str, Any]]] = []
    expected_scale_keys: set[str] = set()
    for item in raw_folds:
        if not isinstance(item, dict) or set(item) != {"fold", "scales"}:
            raise ValueError("invalid fold manifest")
        fold = _fold_from_manifest(item["fold"], process_order, augmentation_parents)
        expected_scale_keys |= _scale_array_keys(item["scales"], fold.idx)
        parsed_folds.append((fold, item["scales"]))
    folds = tuple(fold for fold, _ in parsed_folds)
    if len({fold.idx for fold in folds}) != len(folds) or len(
        {fold.slug for fold in folds}
    ) != len(folds):
        raise ValueError("fold IDs and slugs must be unique")
    selected = next(
        ((fold, scales) for fold, scales in parsed_folds if fold.idx == fold_id),
        None,
    )
    if selected is None:
        raise ValueError(f"unknown fold ID {fold_id}")
    fold, scales_raw = selected
    selected_scale_keys = _scale_array_keys(scales_raw, fold_id)
    expected_all = {
        f"shared.{name}" for name in _shared_array_names()
    } | expected_scale_keys
    if set(records) != expected_all:
        raise ValueError("runtime artifact has missing or extra arrays")
    filenames = {
        name: _array_filename(name, record) for name, record in records.items()
    }
    if len(set(filenames.values())) != len(filenames):
        raise ValueError("runtime artifact arrays must use distinct files")
    parent_collection_path = _verified_parent_collection_path(
        root, manifest["training_parent_collection"]
    )
    _validate_exact_file_inventory(
        root,
        {
            "manifest.json",
            parent_collection_path.name,
            *filenames.values(),
        },
    )
    full_parent_collection = _load_training_parent_collection(
        parent_collection_path,
        process_order=process_order,
        augmentation_parents=augmentation_parents,
    )
    # Fold isolation is intentional: validate every declaration and filename,
    # but open and checksum only shared arrays plus the selected fold's scales.
    required = {
        name for name in expected_all if name.startswith("shared.")
    } | selected_scale_keys
    arrays = {name: _read_array(root, name, records[name]) for name in required}
    _validate_control_partition(full_parent_collection, base["controls"])
    _validate_semantic_arrays(base, arrays)
    rhs_names = _rhs_names_from_payload(base["rhs"])
    _validate_process_matrices(
        {
            name: np.asarray(arrays[f"shared.store.{name}"])
            for name in _PROCESS_MATRIX_NAMES
        },
        full_parent_collection,
        process_order,
        augmentation_parents,
        rhs_names,
    )
    selected_parent_names = canonical_training_parents(
        process_order, augmentation_parents, fold.train
    )
    training_parent_collection = select_parent_collection(
        full_parent_collection, selected_parent_names
    )
    _validate_scale_arrays(
        rhs_names,
        {
            name: (
                arrays[f"fold.{fold_id}.{value['scale']}"],
                None
                if value["kind"] == "linear"
                else arrays[f"fold.{fold_id}.{value['offset']}"],
            )
            for name, value in scales_raw.items()
        },
    )
    controls = ControlsStore(
        **_tuple_metadata(base["controls"]),
        **{
            name: jnp.asarray(arrays[f"shared.controls.{name}"])
            for name in _CONTROL_ARRAYS
        },
    )
    store_metadata = {name: tuple(value) for name, value in base["store"].items()}
    store = TrainingDataStore(
        **store_metadata,
        controls_store=controls,
        rhs_ode=_reconstruct_rhs_ode(
            training_parent_collection,
            rhs_names,
            arrays,
            process_order.index(fold.train[0]),
        ),
        **{name: jnp.asarray(arrays[f"shared.store.{name}"]) for name in _STORE_ARRAYS},
    )
    scales = {}
    for name, value in scales_raw.items():
        scale = jnp.asarray(arrays[f"fold.{fold_id}.{value['scale']}"])
        scales[name] = (
            LinearScaler(scale)
            if value["kind"] == "linear"
            else AffineScaler(
                scale, jnp.asarray(arrays[f"fold.{fold_id}.{value['offset']}"])
            )
        )
    return RuntimeArtifact(
        manifest["identity"],
        store,
        EstimatedScales(**scales),
        fold,
        training_parent_collection,
        augmentation_parents,
    )


def read_runtime_artifact_metadata(path: str | Path) -> RuntimeArtifactMetadata:
    """Validate artifact declarations without opening any numeric array file."""
    root = Path(path)
    manifest = _read_manifest(root)
    records = manifest["arrays"]
    if not isinstance(records, dict):
        raise ValueError("invalid runtime artifact schema")
    process_order, parents = _validate_base_metadata(manifest["base"])
    parsed: list[RuntimeArtifactFold] = []
    scale_keys: set[str] = set()
    if not isinstance(manifest["folds"], list) or not manifest["folds"]:
        raise ValueError("invalid fold manifest")
    for item in manifest["folds"]:
        if not isinstance(item, dict) or set(item) != {"fold", "scales"}:
            raise ValueError("invalid fold manifest")
        fold = _fold_from_manifest(item["fold"], process_order, parents)
        parsed.append(fold)
        scale_keys |= _scale_array_keys(item["scales"], fold.idx)
    if len({fold.idx for fold in parsed}) != len(parsed) or len(
        {fold.slug for fold in parsed}
    ) != len(parsed):
        raise ValueError("fold IDs and slugs must be unique")
    expected = {f"shared.{name}" for name in _shared_array_names()} | scale_keys
    if set(records) != expected:
        raise ValueError("runtime artifact has missing or extra arrays")
    filenames = {
        name: _array_filename(name, record) for name, record in records.items()
    }
    if len(set(filenames.values())) != len(filenames):
        raise ValueError("runtime artifact arrays must use distinct files")
    parent_collection_path = _verified_parent_collection_path(
        root, manifest["training_parent_collection"]
    )
    _validate_exact_file_inventory(
        root,
        {
            "manifest.json",
            parent_collection_path.name,
            *filenames.values(),
        },
    )
    return RuntimeArtifactMetadata(
        manifest["identity"],
        MappingProxyType(dict(manifest["base"]["identity_inputs"])),
        tuple(parsed),
    )


def _shared_array_names() -> set[str]:
    """Expected shared names; the canonical set is fixed by the store's fields."""
    return {f"controls.{name}" for name in _CONTROL_ARRAYS} | {
        f"store.{name}" for name in _STORE_ARRAYS
    }
