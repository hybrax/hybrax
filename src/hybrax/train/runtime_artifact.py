"""Strict, collection-free runtime artifact format."""

from __future__ import annotations

import ast
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

import jax.numpy as jnp
import numpy as np
import sympy
from bp_format.mechanistic import RhsOde

from .controls_store import ControlsStore
from .model_api import AffineScaler, EstimatedScales, LinearScaler, Scaler
from .runtime_context import RuntimeContext, RuntimeDataContext
from .training_data import TrainingDataStore

FORMAT_VERSION = 2
_CONTROL_ARRAYS = (
    "spline_breaks",
    "spline_coeffs",
    "dense_grid",
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
_RHS_ARRAYS = ("Cin_controlled_FVCs", "Cin_modeled_FVCs")
_STORE_ARRAYS = (
    "Cin_controlled_FVCs",
    "Cin_modeled_FVCs",
    "t_measured",
    "y_measured",
    "mask_measured",
    "n_measured",
    "y0_measured",
)
_SCALE_NAMES = tuple(field.name for field in fields(EstimatedScales))
_ALLOWED_FUNCTIONS = {
    name: getattr(sympy, name)
    for name in (
        "Abs",
        "Max",
        "Min",
        "cos",
        "cosh",
        "exp",
        "log",
        "sign",
        "sin",
        "sinh",
        "sqrt",
        "tan",
        "tanh",
    )
}
_VARIADIC_FUNCTIONS = {"Max", "Min"}
_ALLOWED_BINARY_OPERATORS = (ast.Add, ast.Sub, ast.Mult, ast.Div, ast.Pow)
_ALLOWED_UNARY_OPERATORS = (ast.UAdd, ast.USub)
_SLUG_CHARACTERS = frozenset("+-._")


@dataclass(frozen=True)
class RhsOdeDescriptor:
    name_modeled_rates: tuple[str, ...]
    name_modeled_algebraic: tuple[str, ...]
    name_modeled_RMCs: tuple[str, ...]
    name_modeled_PVs: tuple[str, ...]
    name_modeled_FVCs: tuple[str, ...]
    name_modeled_SVCs: tuple[str, ...]
    name_controlled_PVs: tuple[str, ...]
    name_controlled_FVCs: tuple[str, ...]
    name_controlled_SVCs: tuple[str, ...]
    algebraic_expressions: tuple[str, ...]
    derivative_expressions: tuple[str, ...]


@dataclass(frozen=True)
class RuntimeArtifactFold:
    idx: int
    test: tuple[str, ...]
    train: tuple[str, ...]
    slug: str
    seed: int


@dataclass(frozen=True)
class RuntimeArtifact:
    identity: str
    context: RuntimeContext
    fold: RuntimeArtifactFold


@dataclass(frozen=True)
class RuntimeArtifactMetadata:
    """Validated manifest data available without reading numeric arrays."""

    identity: str
    identity_inputs: Mapping[str, str]
    folds: tuple[RuntimeArtifactFold, ...]


@dataclass(frozen=True)
class _ExpressionFunction:
    """Adapter from SymPy's positional callable to RhsOde's vector callable."""

    function: Any

    def __call__(self, values):
        return self.function(*values)


def _canonical_json(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode()


def _digest(value: bytes) -> str:
    return "sha256:" + hashlib.sha256(value).hexdigest()


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


def _array_record(root: Path, name: str, value: Any) -> dict[str, Any]:
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
        "sha256": _digest(path.read_bytes()),
    }


def _scaler(value: Scaler, name: str, arrays: dict[str, Any]) -> dict[str, str]:
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


def _context_arrays(data: RuntimeDataContext) -> dict[str, Any]:
    store = data.training_data
    arrays = {
        f"controls.{name}": getattr(store.controls_store, name)
        for name in _CONTROL_ARRAYS
    }
    arrays.update({f"rhs.{name}": getattr(store.rhs_ode, name) for name in _RHS_ARRAYS})
    arrays.update({f"store.{name}": getattr(store, name) for name in _STORE_ARRAYS})
    for kind, rows in (
        ("modeled", data.modeled_volume_change_traces),
        ("state", data.raw_state_traces),
    ):
        for row, traces in enumerate(rows):
            for column, (times, values) in enumerate(traces):
                arrays[f"trace.{kind}.{row}.{column}.times"] = times
                arrays[f"trace.{kind}.{row}.{column}.values"] = values
    for row, (times, values) in enumerate(data.sample_volume_event_traces):
        arrays[f"trace.sample.{row}.times"] = times
        arrays[f"trace.sample.{row}.values"] = values
    return arrays


def _validate_descriptor(descriptor: RhsOdeDescriptor) -> None:
    groups = tuple(
        getattr(descriptor, field.name)
        for field in fields(descriptor)
        if field.name.startswith("name_")
    )
    if any(
        not all(isinstance(name, str) and name for name in group) for group in groups
    ):
        raise ValueError("RhsOde names must be non-empty strings")
    names = tuple(name for group in groups for name in group)
    if len(names) != len(set(names)):
        raise ValueError("RhsOde names must be unique")
    if len(descriptor.algebraic_expressions) != len(descriptor.name_modeled_algebraic):
        raise ValueError("RhsOde algebraic expression count differs from names")
    states = descriptor.name_modeled_RMCs + descriptor.name_modeled_PVs
    if len(descriptor.derivative_expressions) != len(states):
        raise ValueError("RhsOde derivative expression count differs from states")
    base = states + descriptor.name_controlled_PVs
    for index, expression in enumerate(descriptor.algebraic_expressions):
        _parse_expression(
            expression,
            base
            + descriptor.name_modeled_algebraic[:index]
            + descriptor.name_modeled_rates,
        )
    all_symbols = (
        base + descriptor.name_modeled_algebraic + descriptor.name_modeled_rates
    )
    for expression in descriptor.derivative_expressions:
        _parse_expression(expression, all_symbols)


def _parse_expression(expression: str, names: tuple[str, ...]) -> sympy.Expr:
    if not isinstance(expression, str):
        raise ValueError("RhsOde expression must be a string")
    try:
        tree = ast.parse(expression, mode="eval")
    except SyntaxError as error:
        raise ValueError("invalid RhsOde expression") from error

    def validate(node: ast.AST) -> None:
        if isinstance(node, ast.Expression):
            validate(node.body)
        elif isinstance(node, ast.BinOp) and isinstance(
            node.op, _ALLOWED_BINARY_OPERATORS
        ):
            validate(node.left)
            validate(node.right)
        elif isinstance(node, ast.UnaryOp) and isinstance(
            node.op, _ALLOWED_UNARY_OPERATORS
        ):
            validate(node.operand)
        elif isinstance(node, ast.Name):
            if node.id not in names:
                raise ValueError("RhsOde expression uses an unknown symbol")
        elif isinstance(node, ast.Constant):
            if type(node.value) not in (int, float) or not np.isfinite(node.value):
                raise ValueError("RhsOde expression has an invalid constant")
        elif (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id in _ALLOWED_FUNCTIONS
            and not node.keywords
        ):
            if (
                node.func.id in _VARIADIC_FUNCTIONS
                and not node.args
                or node.func.id not in _VARIADIC_FUNCTIONS
                and len(node.args) != 1
            ):
                raise ValueError("RhsOde expression uses an invalid function arity")
            for argument in node.args:
                validate(argument)
        else:
            raise ValueError("RhsOde expression uses unsupported syntax")

    validate(tree)
    symbols = {name: sympy.Symbol(name) for name in names}
    try:
        parsed = sympy.sympify(
            expression,
            locals={**_ALLOWED_FUNCTIONS, **symbols},
        )
    except Exception as error:
        raise ValueError("invalid RhsOde expression") from error
    if not isinstance(parsed, sympy.Expr) or parsed.free_symbols - set(
        symbols.values()
    ):
        raise ValueError("RhsOde expression uses an unknown symbol")
    if parsed.has(sympy.nan, sympy.oo, -sympy.oo, sympy.zoo, sympy.I):
        raise ValueError("RhsOde expression must be finite and real")
    return parsed


def _descriptor_payload(descriptor: RhsOdeDescriptor) -> dict[str, list[str]]:
    _validate_descriptor(descriptor)
    return {
        field.name: list(getattr(descriptor, field.name))
        for field in fields(descriptor)
    }


def _base_metadata(
    data: RuntimeDataContext,
    descriptor: RhsOdeDescriptor,
    identity_inputs: dict[str, str],
) -> dict[str, Any]:
    store = data.training_data
    controls = store.controls_store
    return {
        "identity_inputs": _json_value(identity_inputs),
        "rhs": _descriptor_payload(descriptor),
        "store": _json_value(
            {
                "process_order": store.process_order,
                "name_measured_RMCs": store.name_measured_RMCs,
                "name_measured_PVs": store.name_measured_PVs,
                "name_modeled_FVCs": store.name_modeled_FVCs,
                "name_modeled_SVCs": store.name_modeled_SVCs,
            }
        ),
        "controls": _json_value(
            {
                name: getattr(controls, name)
                for name in (
                    "process_order",
                    "name_controlled_FVCs",
                    "name_controlled_SVCs",
                    "name_controlled_PVs",
                    "shape_metadata",
                    "spline_indices",
                    "fallback_indices",
                    "spline_side",
                    "max_event_gap_fraction",
                    "max_measurements_per_event_gap",
                    "_process_md_by_name",
                )
            }
        ),
        "runtime": _json_value(
            {
                "augmentation_parents": data.augmentation_parents,
                "process_time_bounds": data.process_time_bounds,
                "bound_snapshots": data.bound_snapshots,
                "modeled_trace_widths": [
                    len(row) for row in data.modeled_volume_change_traces
                ],
                "state_trace_widths": [len(row) for row in data.raw_state_traces],
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


def _validate_runtime_data(
    data: RuntimeDataContext,
    descriptor: RhsOdeDescriptor,
) -> None:
    _validate_descriptor(descriptor)
    rhs = data.training_data.rhs_ode
    for field in fields(descriptor):
        if field.name.startswith("name_") and tuple(
            getattr(rhs, field.name)
        ) != getattr(descriptor, field.name):
            raise ValueError(
                f"RhsOde descriptor {field.name} differs from runtime data"
            )
    base = _base_metadata(data, descriptor, {})
    arrays = {
        f"shared.{name}": np.asarray(value)
        for name, value in _context_arrays(data).items()
    }
    _validate_semantic_arrays(base, arrays)


def _validate_scales(scales: EstimatedScales, descriptor: RhsOdeDescriptor) -> None:
    if not isinstance(scales, EstimatedScales):
        raise TypeError("fold scales must be EstimatedScales")
    _validate_scale_arrays(
        descriptor,
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
    runtime_data: RuntimeDataContext,
    folds: tuple[tuple[RuntimeArtifactFold, EstimatedScales], ...],
    rhs_descriptor: RhsOdeDescriptor,
    identity_inputs: dict[str, str] | None = None,
) -> str:
    """Atomically write shared runtime data and independently loadable scales."""
    if not isinstance(runtime_data, RuntimeDataContext):
        raise TypeError("runtime_data must be a RuntimeDataContext")
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
    _validate_runtime_data(runtime_data, rhs_descriptor)
    for fold, scales in folds:
        _validate_fold(
            fold,
            runtime_data.process_order,
            runtime_data.augmentation_parents,
        )
        _validate_scales(scales, rhs_descriptor)
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
        shared = _context_arrays(runtime_data)
        records = {
            f"shared.{name}": _array_record(temporary, f"shared/{name}", value)
            for name, value in sorted(shared.items())
        }
        fold_records = []
        for fold, estimated_scales in folds:
            scale_arrays: dict[str, Any] = {}
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
        base = _base_metadata(runtime_data, rhs_descriptor, identity_inputs)
        manifest = {
            "format_version": FORMAT_VERSION,
            "base": base,
            "arrays": records,
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
    if not path.is_file() or _digest(path.read_bytes()) != record["sha256"]:
        raise ValueError(f"{name}: checksum mismatch")
    try:
        array = np.load(path, allow_pickle=False)
    except Exception as error:
        raise ValueError(f"{name}: invalid array") from error
    if array.dtype.str != record["dtype"] or list(array.shape) != record["shape"]:
        raise ValueError(f"{name}: dtype or shape mismatch")
    array.setflags(write=False)
    return array


def _descriptor_from_payload(raw: Any) -> RhsOdeDescriptor:
    names = {field.name for field in fields(RhsOdeDescriptor)}
    if (
        not isinstance(raw, dict)
        or set(raw) != names
        or any(not isinstance(value, list) for value in raw.values())
    ):
        raise ValueError("invalid RhsOde descriptor")
    descriptor = RhsOdeDescriptor(**{name: tuple(value) for name, value in raw.items()})
    _validate_descriptor(descriptor)
    return descriptor


def _rhs(raw: Any, arrays: dict[str, np.ndarray]) -> RhsOde:
    descriptor = (
        raw if isinstance(raw, RhsOdeDescriptor) else _descriptor_from_payload(raw)
    )
    _validate_descriptor(descriptor)
    base = (
        descriptor.name_modeled_RMCs
        + descriptor.name_modeled_PVs
        + descriptor.name_controlled_PVs
    )
    symbols = base + descriptor.name_modeled_algebraic + descriptor.name_modeled_rates
    algebraic = tuple(
        _ExpressionFunction(
            sympy.lambdify(
                tuple(sympy.Symbol(name) for name in symbols),
                _parse_expression(
                    expr,
                    base
                    + descriptor.name_modeled_algebraic[:i]
                    + descriptor.name_modeled_rates,
                ),
                modules="jax",
            )
        )
        for i, expr in enumerate(descriptor.algebraic_expressions)
    )
    derivatives = tuple(
        _ExpressionFunction(
            sympy.lambdify(
                tuple(sympy.Symbol(name) for name in symbols),
                _parse_expression(expr, symbols),
                modules="jax",
            )
        )
        for expr in descriptor.derivative_expressions
    )
    controlled = arrays["shared.rhs.Cin_controlled_FVCs"]
    modeled = arrays["shared.rhs.Cin_modeled_FVCs"]
    n_rmc = len(descriptor.name_modeled_RMCs)
    if controlled.shape != (
        len(descriptor.name_controlled_FVCs),
        n_rmc,
    ) or modeled.shape != (len(descriptor.name_modeled_FVCs), n_rmc):
        raise ValueError("RhsOde Cin array shape mismatch")
    return RhsOde(
        *[
            getattr(descriptor, field.name)
            for field in fields(descriptor)
            if field.name.startswith("name_")
        ],
        algebraic,
        derivatives,
        jnp.asarray(controlled),
        jnp.asarray(modeled),
    )


def _validate_base_metadata(
    base: Any,
) -> tuple[tuple[str, ...], tuple[str | None, ...]]:
    required_base_keys = {"identity_inputs", "rhs", "store", "controls", "runtime"}
    if not isinstance(base, dict) or set(base) != required_base_keys:
        raise ValueError("invalid runtime artifact schema")
    if not isinstance(base["identity_inputs"], dict) or not all(
        isinstance(key, str) and isinstance(value, str)
        for key, value in base["identity_inputs"].items()
    ):
        raise ValueError("invalid runtime artifact identity inputs")
    _descriptor_from_payload(base["rhs"])
    store = base["store"]
    controls = base["controls"]
    runtime = base["runtime"]
    if not isinstance(store, dict) or set(store) != {
        "process_order",
        "name_measured_RMCs",
        "name_measured_PVs",
        "name_modeled_FVCs",
        "name_modeled_SVCs",
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
        "name_controlled_FVCs",
        "name_controlled_SVCs",
        "name_controlled_PVs",
        "shape_metadata",
        "spline_indices",
        "fallback_indices",
        "spline_side",
        "max_event_gap_fraction",
        "max_measurements_per_event_gap",
        "_process_md_by_name",
    }:
        raise ValueError("invalid runtime controls metadata")
    if controls["process_order"] != process_order:
        raise ValueError("runtime control process order mismatch")
    for key in (
        "name_controlled_FVCs",
        "name_controlled_SVCs",
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
        or not isinstance(controls["fallback_indices"], list)
        or any(
            type(index) is not int
            for index in controls["spline_indices"] + controls["fallback_indices"]
        )
        or controls["spline_side"] not in {"left", "right"}
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
        "name_controlled_FVCs",
        "name_controlled_SVCs",
        "name_controlled_PVs",
    )
    canonical_control_names = set(
        controls["name_controlled_FVCs"]
        + controls["name_controlled_SVCs"]
        + controls["name_controlled_PVs"]
    )
    for process_name in process_order:
        process_metadata = controls["_process_md_by_name"][process_name]
        if (
            not isinstance(process_metadata, dict)
            or set(process_metadata) != {*control_name_keys, "control_metadata"}
            or any(process_metadata[key] != controls[key] for key in control_name_keys)
            or not isinstance(process_metadata["control_metadata"], dict)
            or set(process_metadata["control_metadata"]) != canonical_control_names
            or not _valid_json_metadata(process_metadata["control_metadata"])
        ):
            raise ValueError("invalid per-process control metadata")
    if not isinstance(runtime, dict) or set(runtime) != {
        "augmentation_parents",
        "process_time_bounds",
        "bound_snapshots",
        "modeled_trace_widths",
        "state_trace_widths",
    }:
        raise ValueError("invalid runtime context metadata")
    n_processes = len(process_order)
    parents = runtime["augmentation_parents"]
    row_metadata = (
        parents,
        runtime["process_time_bounds"],
        runtime["bound_snapshots"],
        runtime["modeled_trace_widths"],
        runtime["state_trace_widths"],
    )
    if any(
        not isinstance(value, list) or len(value) != n_processes
        for value in row_metadata
    ):
        raise ValueError("runtime metadata process count mismatch")
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
    for bounds in runtime["process_time_bounds"]:
        if (
            not isinstance(bounds, list)
            or len(bounds) != 2
            or any(not isinstance(value, (int, float)) for value in bounds)
            or not all(np.isfinite(value) for value in bounds)
            or bounds[0] > bounds[1]
        ):
            raise ValueError("invalid process time bounds")
    for widths in (
        runtime["modeled_trace_widths"],
        runtime["state_trace_widths"],
    ):
        if any(type(width) is not int or width < 0 for width in widths):
            raise ValueError("invalid runtime trace width")
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
    descriptor = _descriptor_from_payload(base["rhs"])
    store = base["store"]
    controls = base["controls"]
    runtime = base["runtime"]
    n_processes = len(store["process_order"])
    n_rmc = len(descriptor.name_modeled_RMCs)
    n_pv = len(descriptor.name_modeled_PVs)
    n_rates = len(descriptor.name_modeled_rates)
    n_modeled_fvc = len(descriptor.name_modeled_FVCs)
    n_modeled_svc = len(descriptor.name_modeled_SVCs)
    n_controlled_fvc = len(descriptor.name_controlled_FVCs)
    n_controlled_svc = len(descriptor.name_controlled_SVCs)
    n_controlled_pv = len(descriptor.name_controlled_PVs)
    n_controls = n_controlled_fvc + n_controlled_svc + n_controlled_pv
    n_targets = len(store["name_measured_RMCs"]) + len(store["name_measured_PVs"])
    n_y = n_targets + n_modeled_fvc + n_modeled_svc

    if (
        not set(store["name_measured_RMCs"]) <= set(descriptor.name_modeled_RMCs)
        or not set(store["name_measured_PVs"]) <= set(descriptor.name_modeled_PVs)
        or tuple(store["name_modeled_FVCs"]) != descriptor.name_modeled_FVCs
        or tuple(store["name_modeled_SVCs"]) != descriptor.name_modeled_SVCs
        or tuple(controls["name_controlled_FVCs"]) != descriptor.name_controlled_FVCs
        or tuple(controls["name_controlled_SVCs"]) != descriptor.name_controlled_SVCs
        or tuple(controls["name_controlled_PVs"]) != descriptor.name_controlled_PVs
    ):
        raise ValueError("runtime metadata differs from RhsOde descriptor")

    spline_indices = tuple(controls["spline_indices"])
    fallback_indices = tuple(controls["fallback_indices"])
    if sorted(spline_indices + fallback_indices) != list(range(n_controls)):
        raise ValueError("invalid runtime control indices")

    spline_breaks = np.asarray(arrays["shared.controls.spline_breaks"])
    dense_grid = np.asarray(arrays["shared.controls.dense_grid"])
    jump_ts = np.asarray(arrays["shared.controls.jump_ts"])
    sample_times = np.asarray(arrays["shared.controls.sample_event_times"])
    bolus_times = np.asarray(arrays["shared.controls.bolus_event_times"])
    if any(
        array.ndim != 2 or array.shape[0] != n_processes
        for array in (
            spline_breaks,
            dense_grid,
            jump_ts,
            sample_times,
            bolus_times,
        )
    ):
        raise ValueError("invalid runtime control process axes")
    n_spline_breaks = spline_breaks.shape[1]
    n_grid = dense_grid.shape[1]
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
        "dense_grid": (n_processes, n_grid),
        "control_values": (n_processes, n_grid, len(fallback_indices)),
        "control_derivatives": (n_processes, n_grid, len(fallback_indices)),
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

    process_bounds = runtime["process_time_bounds"]
    for row, (start, end) in enumerate(process_bounds):
        active_grid = dense_grid[row, : int(grid_lengths[row])]
        active_jumps = jump_ts[row, : int(jump_lengths[row])]
        if not _strictly_increasing(active_grid) or _outside_time_window(
            active_grid, start, end
        ):
            raise ValueError("shared.controls.dense_grid: invalid active time axis")
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
                else active_times.size < 2 or bool(np.all(np.diff(active_times) >= 0))
            )
            outside = _outside_time_window(
                active_times, start, end, include_end=not strict_end
            )
            if not ordered or outside:
                raise ValueError(
                    f"shared.controls.{kind}_event_times: invalid active time axis"
                )
            if np.any(volumes <= 0):
                raise ValueError(
                    f"shared.controls.{kind}_event_volumes: values must be positive"
                )

    _expect_array(
        arrays,
        "shared.rhs.Cin_controlled_FVCs",
        (n_controlled_fvc, n_rmc),
        "f",
    )
    _expect_array(
        arrays,
        "shared.rhs.Cin_modeled_FVCs",
        (n_modeled_fvc, n_rmc),
        "f",
    )
    for name in _RHS_ARRAYS:
        if not np.all(np.isfinite(arrays[f"shared.rhs.{name}"])):
            raise ValueError(f"shared.rhs.{name}: non-finite values")
    _expect_array(
        arrays,
        "shared.store.Cin_controlled_FVCs",
        (n_processes, n_controlled_fvc, n_rmc),
        "f",
    )
    _expect_array(
        arrays,
        "shared.store.Cin_modeled_FVCs",
        (n_processes, n_modeled_fvc, n_rmc),
        "f",
    )
    for name in ("Cin_controlled_FVCs", "Cin_modeled_FVCs"):
        if not np.all(np.isfinite(arrays[f"shared.store.{name}"])):
            raise ValueError(f"shared.store.{name}: non-finite values")
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
        (n_processes, n_rmc + n_pv + 1 + n_modeled_fvc + n_modeled_svc),
        "f",
    )
    if not np.all(np.isfinite(measured_times)) or not np.all(
        np.isfinite(arrays["shared.store.y_measured"])
    ):
        raise ValueError("runtime measurements contain non-finite values")
    if not np.all(np.isfinite(y0)):
        raise ValueError("shared.store.y0_measured: non-finite values")
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

    expected_widths = {
        "modeled": n_modeled_fvc + n_modeled_svc,
        "state": n_rmc + n_pv,
    }
    for kind, expected_width in expected_widths.items():
        widths = runtime[f"{kind}_trace_widths"]
        if widths != [expected_width] * n_processes:
            raise ValueError(f"invalid {kind} trace widths")
        for row in range(n_processes):
            for column in range(expected_width):
                times = _expect_array(
                    arrays,
                    f"shared.trace.{kind}.{row}.{column}.times",
                    np.asarray(
                        arrays[f"shared.trace.{kind}.{row}.{column}.times"]
                    ).shape,
                    "f",
                )
                if times.ndim != 1:
                    raise ValueError(f"invalid {kind} trace shape")
                values = _expect_array(
                    arrays,
                    f"shared.trace.{kind}.{row}.{column}.values",
                    times.shape,
                    "f",
                )
                if not np.all(np.isfinite(times)) or not np.all(np.isfinite(values)):
                    raise ValueError(f"invalid {kind} trace values")
                if not _strictly_increasing(times):
                    raise ValueError(f"invalid {kind} trace time axis")
    for row in range(n_processes):
        times = np.asarray(arrays[f"shared.trace.sample.{row}.times"])
        if times.ndim != 1 or times.dtype.kind != "f":
            raise ValueError("invalid sample trace shape")
        values = _expect_array(
            arrays,
            f"shared.trace.sample.{row}.values",
            times.shape,
            "f",
        )
        if not np.all(np.isfinite(times)) or not np.all(np.isfinite(values)):
            raise ValueError("invalid sample trace values")
        if not _nondecreasing(times):
            raise ValueError("invalid sample trace time axis")

    for snapshots in runtime["bound_snapshots"]:
        for record in snapshots:
            if (
                not isinstance(record, list)
                or len(record) != 5
                or not isinstance(record[0], str)
                or record[1] not in {"state", "volume", "rate"}
                or type(record[2]) is not int
                or any(
                    value is not None
                    and (not isinstance(value, (int, float)) or not np.isfinite(value))
                    for value in record[3:]
                )
            ):
                raise ValueError("invalid runtime bounds snapshot")
            limits = {"state": n_rmc + n_pv, "volume": 1, "rate": n_rates}
            expected_axis = n_rmc + n_pv if record[1] == "volume" else record[2]
            if record[1] == "volume":
                valid_axis = record[2] == expected_axis
            else:
                valid_axis = 0 <= record[2] < limits[record[1]]
            if not valid_axis or (
                record[3] is not None
                and record[4] is not None
                and record[3] > record[4]
            ):
                raise ValueError("invalid runtime bounds snapshot")


def _validate_scale_arrays(
    descriptor: RhsOdeDescriptor,
    scales: dict[str, tuple[np.ndarray, np.ndarray | None]],
) -> None:
    shapes = {
        "SCALE_modeled_RMCs": (len(descriptor.name_modeled_RMCs),),
        "SCALE_modeled_PVs": (len(descriptor.name_modeled_PVs),),
        "SCALE_V_in_cumulative": (),
        "SCALE_modeled_FVCs_cumulative": (len(descriptor.name_modeled_FVCs),),
        "SCALE_controlled_FVCs_cumulative": (len(descriptor.name_controlled_FVCs),),
        "SCALE_controlled_FVCs_rates": (len(descriptor.name_controlled_FVCs),),
        "SCALE_controlled_FVCs_Cin": (
            len(descriptor.name_controlled_FVCs),
            len(descriptor.name_modeled_RMCs),
        ),
        "SCALE_controlled_PVs": (len(descriptor.name_controlled_PVs),),
        "SCALE_modeled_FVCs_Cin": (
            len(descriptor.name_modeled_FVCs),
            len(descriptor.name_modeled_RMCs),
        ),
        "SCALE_modeled_BiologicalOde_rates": (len(descriptor.name_modeled_rates),),
        "SCALE_modeled_FVCs_rates": (len(descriptor.name_modeled_FVCs),),
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
            or np.any(scale_array == 0)
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
        f"shared.{name}" for name in _context_arrays_placeholder(base)
    } | expected_scale_keys
    if set(records) != expected_all:
        raise ValueError("runtime artifact has missing or extra arrays")
    filenames = {
        name: _array_filename(name, record) for name, record in records.items()
    }
    if len(set(filenames.values())) != len(filenames):
        raise ValueError("runtime artifact arrays must use distinct files")
    expected_files = {"manifest.json", *filenames.values()}
    actual_files = {
        item.relative_to(root).as_posix() for item in root.rglob("*") if item.is_file()
    }
    if actual_files != expected_files:
        raise ValueError("runtime artifact has missing or extra files")
    required = {
        name for name in expected_all if name.startswith("shared.")
    } | selected_scale_keys
    arrays = {name: _read_array(root, name, records[name]) for name in required}
    _validate_semantic_arrays(base, arrays)
    descriptor = _descriptor_from_payload(base["rhs"])
    _validate_scale_arrays(
        descriptor,
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
    runtime = base["runtime"]
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
        rhs_ode=_rhs(base["rhs"], arrays),
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
    try:
        modeled = tuple(
            tuple(
                (
                    arrays[f"shared.trace.modeled.{i}.{j}.times"],
                    arrays[f"shared.trace.modeled.{i}.{j}.values"],
                )
                for j in range(width)
            )
            for i, width in enumerate(runtime["modeled_trace_widths"])
        )
        states = tuple(
            tuple(
                (
                    arrays[f"shared.trace.state.{i}.{j}.times"],
                    arrays[f"shared.trace.state.{i}.{j}.values"],
                )
                for j in range(width)
            )
            for i, width in enumerate(runtime["state_trace_widths"])
        )
        samples = tuple(
            (
                arrays[f"shared.trace.sample.{i}.times"],
                arrays[f"shared.trace.sample.{i}.values"],
            )
            for i in range(len(store.process_order))
        )
        data = RuntimeDataContext(
            store,
            tuple(runtime["augmentation_parents"]),
            tuple(tuple(value) for value in runtime["process_time_bounds"]),
            modeled,
            states,
            samples,
            tuple(
                tuple(tuple(value) for value in row)
                for row in runtime["bound_snapshots"]
            ),
        )
    except (KeyError, TypeError) as error:
        raise ValueError("invalid runtime trace metadata") from error
    return RuntimeArtifact(
        manifest["identity"],
        RuntimeContext(data, EstimatedScales(**scales)),
        fold,
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
    expected = {
        f"shared.{name}" for name in _context_arrays_placeholder(manifest["base"])
    } | scale_keys
    if set(records) != expected:
        raise ValueError("runtime artifact has missing or extra arrays")
    filenames = {
        name: _array_filename(name, record) for name, record in records.items()
    }
    if len(set(filenames.values())) != len(filenames):
        raise ValueError("runtime artifact arrays must use distinct files")
    actual_files = {
        item.relative_to(root).as_posix() for item in root.rglob("*") if item.is_file()
    }
    if actual_files != {"manifest.json", *filenames.values()}:
        raise ValueError("runtime artifact has missing or extra files")
    return RuntimeArtifactMetadata(
        manifest["identity"],
        MappingProxyType(dict(manifest["base"]["identity_inputs"])),
        tuple(parsed),
    )


def _context_arrays_placeholder(base: dict[str, Any]) -> set[str]:
    """Expected shared names, derived solely from strict runtime metadata."""
    names = (
        {f"controls.{name}" for name in _CONTROL_ARRAYS}
        | {f"rhs.{name}" for name in _RHS_ARRAYS}
        | {f"store.{name}" for name in _STORE_ARRAYS}
    )
    runtime = base["runtime"]
    try:
        for kind, widths in (
            ("modeled", runtime["modeled_trace_widths"]),
            ("state", runtime["state_trace_widths"]),
        ):
            for row, width in enumerate(widths):
                if type(width) is not int or width < 0:
                    raise ValueError
                for column in range(width):
                    names |= {
                        f"trace.{kind}.{row}.{column}.times",
                        f"trace.{kind}.{row}.{column}.values",
                    }
        for row in range(len(base["store"]["process_order"])):
            names |= {f"trace.sample.{row}.times", f"trace.sample.{row}.values"}
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("invalid runtime trace metadata") from error
    return names
