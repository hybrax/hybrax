from __future__ import annotations

import sys
from typing import Any

import equinox as eqx
import jax
import jax.tree_util as jtu


TRAINABLE_METADATA_KEY = "bp_train_trainable"


def trainable_field(**kwargs: Any) -> Any:
    """eqx.field whose array leaves are included in optimizer updates.

    An explicit tag on a parent field overrides every descendant's own tag.
    Untagged array leaves default to frozen.
    """
    metadata = dict(kwargs.pop("metadata", {}) or {})
    metadata[TRAINABLE_METADATA_KEY] = True
    return eqx.field(metadata=metadata, **kwargs)


def frozen_field(**kwargs: Any) -> Any:
    """eqx.field whose leaves are masked out of optimizer updates."""
    metadata = dict(kwargs.pop("metadata", {}) or {})
    metadata[TRAINABLE_METADATA_KEY] = False
    return eqx.field(metadata=metadata, **kwargs)


def _resolve_effective_tag(module: eqx.Module, path: tuple) -> bool | None:
    """Walk ``module`` along ``path`` and return the effective trainable tag.

    Returns True (trainable), False (frozen), or None (untagged anywhere on
    the path; caller maps None to frozen for arrays).

    Inheritance rule: the first explicit tag encountered along the path wins;
    deeper tags are ignored.
    """
    inherited: bool | None = None
    current: Any = module
    for entry in path:
        if isinstance(entry, jtu.GetAttrKey):
            attr_name = entry.name
            if isinstance(current, eqx.Module):
                field = current.__dataclass_fields__.get(attr_name)
                if field is not None:
                    field_tag = field.metadata.get(TRAINABLE_METADATA_KEY)
                    if inherited is None and field_tag is not None:
                        inherited = field_tag
            current = getattr(current, attr_name)
        elif isinstance(entry, jtu.SequenceKey):
            current = current[entry.idx]
        elif isinstance(entry, jtu.DictKey):
            current = current[entry.key]
        else:
            current = None
    return inherited


def _partition_trainable_from_metadata(
    module: eqx.Module,
) -> tuple[eqx.Module, eqx.Module]:
    """Partition a module into (trainable, static) pytrees from field metadata."""
    leaves_with_paths, treedef = jtu.tree_flatten_with_path(module)
    bool_leaves: list[bool] = []
    for path, leaf in leaves_with_paths:
        if not eqx.is_inexact_array(leaf):
            # Activation callables, ints, dtypes, etc. cannot receive gradients;
            # always static regardless of any inherited trainable tag.
            bool_leaves.append(False)
            continue
        tag = _resolve_effective_tag(module, path)
        bool_leaves.append(bool(tag) if tag is not None else False)
    filter_spec = jtu.tree_unflatten(treedef, bool_leaves)
    return eqx.partition(module, filter_spec)


def _validate_partition_pair(
    module: eqx.Module,
    trainable: eqx.Module,
    static: eqx.Module,
) -> tuple[eqx.Module, eqx.Module]:
    """Validate trainable/static pytrees can reconstruct the original module."""
    module_leaves, module_treedef = jtu.tree_flatten(
        module,
        is_leaf=lambda value: value is None,
    )
    trainable_leaves, trainable_treedef = jtu.tree_flatten(
        trainable,
        is_leaf=lambda value: value is None,
    )
    static_leaves, static_treedef = jtu.tree_flatten(
        static,
        is_leaf=lambda value: value is None,
    )

    if trainable_treedef != module_treedef or static_treedef != module_treedef:
        raise ValueError("partition_trainable outputs must match module structure")

    for module_leaf, trainable_leaf, static_leaf in zip(
        module_leaves,
        trainable_leaves,
        static_leaves,
        strict=False,
    ):
        trainable_is_none = trainable_leaf is None
        static_is_none = static_leaf is None
        if trainable_is_none and static_is_none:
            if module_leaf is None:
                continue
            raise ValueError(
                "partition_trainable leaves must appear in exactly one partition"
            )
        if not trainable_is_none and not static_is_none:
            raise ValueError(
                "partition_trainable leaves must appear in exactly one partition"
            )

    try:
        combined = eqx.combine(trainable, static)
    except Exception as exc:  # pragma: no cover - defensive guard
        raise ValueError(
            "partition_trainable must return compatible trainable/static pytrees"
        ) from exc
    if not eqx.tree_equal(combined, module):
        raise ValueError(
            "partition_trainable outputs must reconstruct the original module"
        )
    return trainable, static


def partition_trainable(module: eqx.Module) -> tuple[eqx.Module, eqx.Module]:
    """Return trainable/static pytrees for a user reaction module.

    If the module exposes ``partition_trainable()``, that method is used.
    Otherwise the metadata-based default applies: array leaves under
    ``trainable_field()`` are trainable; everything else (including untagged
    array leaves) is static.
    """
    method = getattr(module, "partition_trainable", None)
    if callable(method):
        trainable, static = method()
    else:
        trainable, static = _partition_trainable_from_metadata(module)
    return _validate_partition_pair(module, trainable, static)


# ---------------------------------------------------------------------------
# Structure print-out
# ---------------------------------------------------------------------------


_ANSI_RED = "\x1b[31m"
_ANSI_RESET = "\x1b[0m"


def _shape_str(leaf: Any) -> str:
    if eqx.is_array(leaf):
        return str(tuple(leaf.shape))
    return "()"


def _collect_structure_rows(
    value: Any,
    prefix: str,
    inherited_tag: bool | None,
) -> list[tuple[str, str, str]]:
    """Walk recursively and emit one row per ``jax.Array``-like leaf.

    Module / list / tuple containers do not emit their own row; their
    structure is visible through the dotted/indexed names of the leaves
    they contain. Non-array leaves (ints, callables, dtypes, etc.) are
    skipped entirely.
    """
    rows: list[tuple[str, str, str]] = []

    if isinstance(value, eqx.Module):
        for fname, finfo in value.__dataclass_fields__.items():
            child = getattr(value, fname)
            child_name = f"{prefix}.{fname}" if prefix else fname
            field_tag = finfo.metadata.get(TRAINABLE_METADATA_KEY)
            child_inherited = (
                inherited_tag if inherited_tag is not None else field_tag
            )
            rows.extend(_collect_structure_rows(child, child_name, child_inherited))
        return rows

    if isinstance(value, (list, tuple)):
        for i, item in enumerate(value):
            rows.extend(
                _collect_structure_rows(item, f"{prefix}[{i}]", inherited_tag)
            )
        return rows

    if eqx.is_array(value):
        leaf_tag = inherited_tag if eqx.is_inexact_array(value) else False
        status = "trainable" if leaf_tag is True else "frozen"
        rows.append((prefix, _shape_str(value), status))

    return rows


def format_trainable_structure(
    module: eqx.Module,
    *,
    name_width: int | None = None,
    shape_width: int | None = None,
    status_width: int | None = None,
    color: bool = False,
) -> str:
    """Return a tabular string of (name, shape, status) for the full module.

    Submodules render as ``(Module)`` rows and their fields nest below. Status
    is ``trainable`` or ``frozen``. With ``color=True``, only the status cell
    of trainable rows is wrapped in ANSI red so the table borders stay plain.
    """
    rows = _collect_structure_rows(module, "", None)

    header = ("name", "shape", "status")
    width_floor = 3
    if name_width is None:
        name_width = max([len(header[0])] + [len(r[0]) for r in rows] + [width_floor])
    if shape_width is None:
        shape_width = max([len(header[1])] + [len(r[1]) for r in rows] + [width_floor])
    if status_width is None:
        status_width = max(
            [len(header[2])] + [len(r[2]) for r in rows] + [width_floor]
        )

    total_width = name_width + shape_width + status_width + 10

    def _wrap(text: str) -> str:
        return f"{_ANSI_RED}{text}{_ANSI_RESET}"

    def _fmt_row(name: str, shape: str, status: str, *, colorize: bool) -> str:
        name_pad = " " * (name_width - len(name))
        shape_pad = " " * (shape_width - len(shape))
        status_pad = " " * (status_width - len(status))
        if colorize and status == "trainable":
            name_cell = _wrap(name) + name_pad
            shape_cell = shape_pad + _wrap(shape)
            status_cell = status_pad + _wrap(status)
        else:
            name_cell = name + name_pad
            shape_cell = shape_pad + shape
            status_cell = status_pad + status
        return f"| {name_cell} | {shape_cell} | {status_cell} |"

    divider = "+" + "-" * (total_width - 2) + "+"
    title = " UserReactionModule Structure "
    title_pad_total = total_width - 2 - len(title)
    title_left = title_pad_total // 2
    title_right = title_pad_total - title_left
    title_line = "+" + "-" * title_left + title + "-" * title_right + "+"

    lines: list[str] = [title_line]
    lines.append(_fmt_row(header[0], header[1], header[2], colorize=False))
    lines.append(divider)
    if rows:
        for name, shape, status in rows:
            lines.append(_fmt_row(name, shape, status, colorize=color))
    else:
        msg = "no array leaves"
        pad = total_width - 4 - len(msg)
        lines.append(f"| {msg}{' ' * pad} |")
    lines.append(divider)
    return "\n".join(lines)


def print_trainable_structure(
    module: eqx.Module,
    *,
    color: bool | None = None,
) -> None:
    """Print the trainable-structure table.

    ``color=None`` auto-detects with ``sys.stdout.isatty()``.
    """
    if color is None:
        color = bool(getattr(sys.stdout, "isatty", lambda: False)())
    print(format_trainable_structure(module, color=color))


# ---------------------------------------------------------------------------
# Public types
# ---------------------------------------------------------------------------


class ReactionOutputs(eqx.Module):
    """Structured return value for user reaction modules.

    Attributes
    ----------
    specific_rates:
        Specific rates ``q_i`` for each species, aligned with the species state
        vector.  These are multiplied by ``X_active`` inside the mechanistic
        ODE (``dc_i/dt = q_i * X_active + transport``).
    modeled_feed_rates:
        Volumetric flow rates for uncontrolled (modeled) feed streams, aligned
        with the modeled-flow ordering from the mechanistic ODE module.
        Use a zero-length array when there are no modeled flows.
    auxiliary:
        Optional model-defined observables that should follow the solver-time
        save path. Conservative V1 contract: ``None`` or ``dict[str, array]``
        with scalar or 1D-array leaves and stable keys across calls.
    """

    specific_rates: jax.Array
    modeled_feed_rates: jax.Array
    auxiliary: dict[str, jax.Array] | None = None


class UserReactionModule(eqx.Module):
    """Base abstraction for user-defined reaction modules."""

    def __call__(
        self,
        t: jax.Array,
        c_species: jax.Array,
        controls_vector: jax.Array,
    ) -> ReactionOutputs:
        raise NotImplementedError

    def observe(self, states: jax.Array) -> jax.Array:
        """Optional observation map; default identity."""
        return states
