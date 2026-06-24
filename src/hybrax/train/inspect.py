"""Read-only introspection helpers for trained modules and wrappers.

Two responsibilities:

- ``format_trainable_structure`` / ``print_trainable_structure``: render a
  table of every array leaf on a :class:`UserReactionModule` subclass,
  tagged ``trainable`` or ``frozen``. Used by the training harness at
  startup so the trainable surface is visible at a glance.

- ``format_reaction_schema`` / ``print_reaction_schema``: render labeled
  tables for the :class:`ReactionInputs` / :class:`ReactionOutputs` axes
  of a wrapper, with names sourced from ``rhs_ode`` and ``controls``.
  Lets a user cross-reference each SCL vector slot with the underlying
  bp-format entity it represents.
"""

from __future__ import annotations

import sys
from typing import TYPE_CHECKING, Any

import equinox as eqx

from .model_api import TRAINABLE_METADATA_KEY

if TYPE_CHECKING:
    from bp_format.mechanistic import RhsOde

    from .controls_store import PerProcessControls
    from .wrapper import HybridOdeWrapper


# ---------------------------------------------------------------------------
# Trainable-structure printer
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
    title: str = "UserReactionModule",
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
    title = f" {title} Structure "
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
    title: str = "UserReactionModule",
) -> None:
    """Print the trainable-structure table.

    ``color=None`` auto-detects with ``sys.stdout.isatty()``.
    """
    if color is None:
        color = bool(getattr(sys.stdout, "isatty", lambda: False)())
    print(format_trainable_structure(module, color=color, title=title), flush=True)


# ---------------------------------------------------------------------------
# Reaction-schema printer
# ---------------------------------------------------------------------------


_NAMES_CELL_WIDTH = 60  # max width of the wrappable "names" column


def _shape_tuple_str(*dims: int) -> str:
    if len(dims) == 0:
        return "()"
    if len(dims) == 1:
        return f"({dims[0]},)"
    return "(" + ", ".join(str(d) for d in dims) + ")"


def _wrap_names_to_width(
    names: tuple[str, ...], width: int, hang_indent: int = 0
) -> list[str]:
    """Wrap a comma-joined name list to fit ``width`` characters per line.

    ``hang_indent`` adds spaces to continuation lines (used for ``rows:`` /
    ``cols:`` follow-up labels so the wrapped tail aligns under the first
    name rather than the label prefix).
    """
    if not names:
        # Empty cell — the (0,)/(0, N) shape already signals "no entries".
        return [""]
    lines: list[str] = []
    current = ""
    for i, name in enumerate(names):
        piece = name if i == len(names) - 1 else f"{name}, "
        current_width = width if not lines else (width - hang_indent)
        if current and len(current) + len(piece) > current_width:
            # Continuation: keep the trailing comma so readers see the list
            # continues onto the next line.
            lines.append(current.rstrip())
            current = piece
        else:
            current += piece
    if current:
        lines.append(current.rstrip().rstrip(","))
    if hang_indent and len(lines) > 1:
        lines = [lines[0]] + [" " * hang_indent + ln for ln in lines[1:]]
    return lines


def _names_cell_lines(
    names: tuple[str, ...],
    cin_followup: tuple[tuple[str, ...], tuple[str, ...]] | None,
    width: int,
) -> list[str]:
    """Produce the wrapped lines that should appear in the names cell.

    For 2-D Cin matrices ``cin_followup`` carries ``(row_names, col_names)``
    and the cell renders ``rows:`` and ``cols:`` sub-blocks instead of the
    flat name list.
    """
    if cin_followup is None:
        return _wrap_names_to_width(names, width)

    # 2-D Cin: always emit both `rows:` and `cols:` labels so the matrix
    # axes are visible even when one dimension is empty.
    row_names, col_names = cin_followup
    rows_block = _wrap_names_to_width(row_names, width - 6, hang_indent=6)
    rows_block[0] = "rows: " + rows_block[0]
    cols_block = _wrap_names_to_width(col_names, width - 6, hang_indent=6)
    cols_block[0] = "cols: " + cols_block[0]
    return rows_block + cols_block


def _reaction_schema_rows(
    rhs_ode: "RhsOde",
    controls: "PerProcessControls",
) -> tuple[list[tuple[str, str, tuple[str, ...]]], list[tuple[str, str, tuple[str, ...]]], list[tuple[str, tuple[str, ...], tuple[str, ...]]]]:
    """Build the (inputs, outputs, cin_followups) row collections.

    `cin_followups` carries (axis_name, row_names, col_names) for each
    2-D Cin matrix so the renderer can emit indented "rows:"/"cols:"
    lines underneath the table row.
    """
    name_RMCs = tuple(rhs_ode.name_modeled_RMCs)
    name_modeled_FVCs = tuple(rhs_ode.name_modeled_FVCs)
    name_controlled_FVCs = tuple(rhs_ode.name_controlled_FVCs)
    name_controlled_PVs = tuple(rhs_ode.name_controlled_PVs)
    name_modeled_rates = tuple(rhs_ode.name_modeled_rates)

    n_RMCs = len(name_RMCs)
    n_modeled_FVCs = len(name_modeled_FVCs)
    n_controlled_FVCs = len(name_controlled_FVCs)
    n_controlled_PVs = len(name_controlled_PVs)
    n_modeled_rates = len(name_modeled_rates)

    inputs: list[tuple[str, str, tuple[str, ...]]] = [
        ("SCL_modeled_RMCs", _shape_tuple_str(n_RMCs), name_RMCs),
        ("SCL_modeled_V", "()", ("V_real",)),
        (
            "SCL_modeled_FVCs_cumulative",
            _shape_tuple_str(n_modeled_FVCs),
            name_modeled_FVCs,
        ),
        (
            "SCL_controlled_FVCs_cumulative",
            _shape_tuple_str(n_controlled_FVCs),
            name_controlled_FVCs,
        ),
        (
            "SCL_controlled_FVCs_rates",
            _shape_tuple_str(n_controlled_FVCs),
            name_controlled_FVCs,
        ),
        (
            "SCL_controlled_FVCs_Cin",
            _shape_tuple_str(n_controlled_FVCs, n_RMCs),
            (),  # 2-D: detail rendered as follow-up lines
        ),
        (
            "SCL_controlled_PVs",
            _shape_tuple_str(n_controlled_PVs),
            name_controlled_PVs,
        ),
        (
            "SCL_modeled_FVCs_Cin",
            _shape_tuple_str(n_modeled_FVCs, n_RMCs),
            (),
        ),
    ]

    outputs: list[tuple[str, str, tuple[str, ...]]] = [
        (
            "SCL_modeled_BiologicalOde_rates",
            _shape_tuple_str(n_modeled_rates),
            name_modeled_rates,
        ),
        (
            "SCL_modeled_FVCs_rates",
            _shape_tuple_str(n_modeled_FVCs),
            name_modeled_FVCs,
        ),
    ]

    cin_followups: list[tuple[str, tuple[str, ...], tuple[str, ...]]] = [
        ("SCL_controlled_FVCs_Cin", name_controlled_FVCs, name_RMCs),
        ("SCL_modeled_FVCs_Cin", name_modeled_FVCs, name_RMCs),
    ]

    return inputs, outputs, cin_followups


def _render_schema_table(
    title: str,
    rows: list[tuple[str, str, tuple[str, ...]]],
    cin_followups_by_name: dict[str, tuple[tuple[str, ...], tuple[str, ...]]],
) -> str:
    """Render a bordered schema table matching the ``RhsOde Structure`` style.

    Names that exceed the names column wrap onto additional rows that
    repeat the same outer ``|`` borders, with the axis + shape cells blank
    on continuation lines. 2-D Cin matrices put their ``rows:`` / ``cols:``
    sub-blocks inside the names cell so the table stays rectangular.
    """
    header = ("axis", "shape", "names")
    width_floor = 3

    pre_lines: list[list[str]] = []
    for axis, shape, names in rows:
        cell_lines = _names_cell_lines(
            names, cin_followups_by_name.get(axis), _NAMES_CELL_WIDTH
        )
        pre_lines.append(cell_lines)

    axis_width = max(
        [len(header[0])] + [len(r[0]) for r in rows] + [width_floor]
    )
    shape_width = max(
        [len(header[1])] + [len(r[1]) for r in rows] + [width_floor]
    )
    names_width = max(
        [len(header[2])]
        + [len(line) for block in pre_lines for line in block]
        + [width_floor]
    )

    total_width = axis_width + shape_width + names_width + 10  # borders + separators

    def _fmt_row(a: str, b: str, c: str) -> str:
        return (
            f"| {a}{' ' * (axis_width - len(a))}"
            f" | {b}{' ' * (shape_width - len(b))}"
            f" | {c}{' ' * (names_width - len(c))} |"
        )

    divider = "+" + "-" * (total_width - 2) + "+"
    title_pad_total = total_width - 2 - len(title) - 2  # two spaces around title
    title_left = title_pad_total // 2
    title_right = title_pad_total - title_left
    title_line = (
        "+" + "-" * title_left + " " + title + " " + "-" * title_right + "+"
    )

    out: list[str] = [title_line]
    out.append(_fmt_row(header[0], header[1], header[2]))
    out.append(divider)
    for (axis, shape, _names), cell_lines in zip(rows, pre_lines, strict=True):
        out.append(_fmt_row(axis, shape, cell_lines[0]))
        for cont in cell_lines[1:]:
            out.append(_fmt_row("", "", cont))
    out.append(divider)
    return "\n".join(out)


def format_reaction_schema(
    rhs_ode: "RhsOde",
    controls: "PerProcessControls",
) -> str:
    """Return labeled tables for ReactionInputs + ReactionOutputs axes.

    Each table row names the SCL_* axis, its shape, and the bp-format
    entity names that map to each slot. 2-D Cin matrices render as one
    row with shape ``(n_rows, n_cols)`` followed by indented
    ``rows:``/``cols:`` lines naming the matrix axes.
    """
    inputs_rows, outputs_rows, cin_followups = _reaction_schema_rows(
        rhs_ode, controls
    )
    cin_followups_by_name = {name: (rows, cols) for name, rows, cols in cin_followups}

    inputs_table = _render_schema_table(
        "ReactionInputs Schema", inputs_rows, cin_followups_by_name
    )
    outputs_table = _render_schema_table(
        "ReactionOutputs Schema", outputs_rows, {}
    )
    return inputs_table + "\n\n" + outputs_table


def print_reaction_schema(wrapper: "HybridOdeWrapper") -> None:
    """Print the labeled ReactionInputs / ReactionOutputs tables.

    Sources names from ``wrapper.rhs_ode`` and ``wrapper.controls``.
    """
    print(format_reaction_schema(wrapper.rhs_ode, wrapper.controls), flush=True)
