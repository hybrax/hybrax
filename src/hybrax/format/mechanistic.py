"""
Mechanistic API for bp-format.

JAX/Equinox-compatible modules for building continuous-time control functions
and ODE right-hand sides directly from a :class:`~bp_format.BioProcess`. All
modules are fully JAX-jittable via ``equinox.filter_jit``.

Public API
----------
get_process_ordering(process) -> ProcessOrdering
    Single source of truth for canonical name ordering across all derived
    modules (states, controls, rates, algebraic, Inflows, Outflows).

get_control_splines(process, ordering=None) -> ControlSplines
    ``ControlSplines.__call__(t)`` evaluates all controlled signals at *t*.
    The output layout is
    ``[Inflow_flows | Outflow_flows | PV_values]``: the first
    ``len(Inflows)+len(Outflows)`` entries are flow rates (spline derivatives), the
    remaining ``len(PVs)`` entries are direct values. Outflow flow rates carry
    the storage sign (negative cumulative volume → negative flow rate); the
    feed-dilution machinery interprets them as signed quantities.

build_rhs_ode(process, ordering=None) -> RhsOde
    Build the :class:`RhsOde` for a process. ``BiologicalOde`` is required
    (auto-generated in ``BioProcess.__post_init__`` when not user-supplied).
    Call signature::

        dc_dt = rhs_ode(c, rates, u, f_modeled_Inflows, f_modeled_Outflows)

    where ``u`` is the full control vector (output of ``ControlSplines``)
    and ``f_modeled_*`` are uncontrolled (modeled) flow vectors.

extract_discrete_events(process, ordering) -> list[dict]
    Extract sampling and bolus-feed events. ``Cin`` arrays are aligned with
    ``ordering.name_modeled_RMCs``.

build_state_splines(process, ordering) -> dict
    Spline callables for every non-volume state, built directly from the
    stored real-concentration TimeSeries.

build_algebraic_func(process) -> Callable
    Returns ``f(state_values, ctrl_pv_values, rates) -> {name: scalar}`` for
    inspecting algebraic quantities (e.g. ``X_active``).

Forward integration of the process lives in ``bp-train``; this module does
not integrate. Rate inversion (recovering rate values from state splines)
is not implemented.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional, Tuple

import equinox as eqx
import jax.numpy as jnp

from .dataclasses import (
    BioProcess,
    Inflow,
    Outflow,
    ProcessOrdering,
    StaticVariable,
    TimeSeries,
)
from .splines import (
    _MIN_REACTOR_VOLUME,
    make_cubic_ppoly,
)
from .time_series import PPoly


# ---------------------------------------------------------------------------
# Spline helpers
# ---------------------------------------------------------------------------


def _require_reactor_volume_above_threshold(
    volume: jnp.ndarray,
    *,
    context: str,
    V_min: float | jnp.ndarray = _MIN_REACTOR_VOLUME,
) -> jnp.ndarray:
    """Fail when reactor volume reaches its minimum valid value."""
    volume_arr = jnp.asarray(volume)
    return eqx.error_if(
        volume_arr,
        jnp.any(volume_arr <= V_min),
        f"{context} reached minimum reactor volume.",
    )


def _timeseries_to_ppoly(series: TimeSeries) -> PPoly:
    """Return a bp_format PPoly for a TimeSeries carrier.

    When the carrier already stores spline state, return that PPoly directly
    (no copy, no refit) so mechanistic consumers use the same canonical
    representation that was fit/serialized. Fall back to a cubic refit only
    for sample-only series without spline coefficients.
    """
    if series.poly is not None:
        return series.poly
    if series.times is not None and series.values is not None:
        return make_cubic_ppoly(
            jnp.asarray(series.times, dtype=float),
            jnp.asarray(series.values, dtype=float),
        )
    raise ValueError("TimeSeries must provide spline state or discrete samples.")


def _value_to_ppoly(
    value: TimeSeries | StaticVariable,
    *,
    t_start: float,
    t_end: float,
) -> PPoly:
    """Return a bp_format PPoly for a dynamic or static state carrier."""
    if isinstance(value, TimeSeries):
        return _timeseries_to_ppoly(value)
    v = float(value.value)
    return PPoly(
        jnp.array([t_start, t_end], dtype=float),
        jnp.array([[v, 0.0, 0.0, 0.0]], dtype=float),
    )


# ---------------------------------------------------------------------------
# Reactor mass balance: feed inflow + sample outflow + dilution
# ---------------------------------------------------------------------------


def _apply_feed_dilution(
    c_RMCs: jnp.ndarray,
    V: jnp.ndarray,
    u_controlled_Inflows: jnp.ndarray,
    u_controlled_Outflows: jnp.ndarray,
    f_modeled_Inflows: jnp.ndarray,
    f_modeled_Outflows: jnp.ndarray,
    Cin_controlled_Inflows: jnp.ndarray,
    Cin_modeled_Inflows: jnp.ndarray,
    retention_controlled_Outflows: jnp.ndarray,
    retention_modeled_Outflows: jnp.ndarray,
    n_RMCs: int,
    V_min: float | jnp.ndarray = _MIN_REACTOR_VOLUME,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Reactor-component mass balance from feed/sample flows + ``dV/dt``.

    Convention: every flow-rate vector carries the storage sign of its
    underlying volume change. Inflow values are stored as cumulative volumes
    with ``values >= 0``, so ``u_controlled_Inflows`` and ``f_modeled_Inflows``
    are non-negative. Outflow values are stored as ``values <= 0``, so
    ``u_controlled_Outflows`` and ``f_modeled_Outflows`` are non-positive.

    Standard well-mixed reactor mass balance: a species leaving with an
    outflow at the reactor's own bulk concentration does not, by itself,
    change that concentration — only feeding (which adds material at a
    *different* concentration, ``Cin``) does. So an unretained outflow
    (retention sigma=0, the default) contributes no dilution term of its
    own; it only shrinks ``V``, which concentrates whatever is left. Each
    Outflow's per-component ``retention`` (sigma in [0, 1]) inverts this:
    the retained fraction of a component does *not* leave with the flow,
    so as volume drops around it, its concentration rises — sigma=1 means
    that component is fully retained (e.g. cells in perfusion; solutes in
    evaporation) and concentrates exactly in step with the volume loss.

    - Inflows contribute dilution (``-q/V·c``) **and** species addition
      (``q·Cin/V``); they push ``dV`` upward.
    - Outflows contribute a dilution term only through what they *retain*
      (``+retained_q/V·c``); an outflow with sigma=0 everywhere has no
      effect on ``dilution`` at all. They push ``dV`` downward regardless
      of retention — retention changes what leaves with the flow, not how
      much volume the flow removes.
    """
    V = _require_reactor_volume_above_threshold(V, context="ODE state", V_min=V_min)

    total_in = jnp.sum(u_controlled_Inflows) + jnp.sum(f_modeled_Inflows)  # >= 0
    total_out = -(jnp.sum(u_controlled_Outflows) + jnp.sum(f_modeled_Outflows))  # >= 0
    dV = total_in - total_out

    retained_out_per_rmc = jnp.sum(
        retention_controlled_Outflows * (-u_controlled_Outflows)[:, None], axis=0
    ) + jnp.sum(retention_modeled_Outflows * (-f_modeled_Outflows)[:, None], axis=0)

    dilution = -(total_in - retained_out_per_rmc) * c_RMCs / V

    addition = jnp.zeros(n_RMCs)
    if Cin_controlled_Inflows.shape[0] > 0:
        addition = (
            addition
            + jnp.sum(u_controlled_Inflows[:, None] * Cin_controlled_Inflows, axis=0)
            / V
        )
    if Cin_modeled_Inflows.shape[0] > 0:
        addition = (
            addition
            + jnp.sum(f_modeled_Inflows[:, None] * Cin_modeled_Inflows, axis=0) / V
        )

    return dilution + addition, dV


# ---------------------------------------------------------------------------
# Sympy helpers used by the RhsOde builder
# ---------------------------------------------------------------------------


def _lambdify_with_array_arg(expr, ordered_names: Tuple[str, ...]) -> Callable:
    """Return ``f(args)`` that evaluates *expr* with the symbol values
    supplied as a flat 1-D array indexed by *ordered_names*.

    Uses a small Python wrapper around :func:`sympy.lambdify` so each call
    site passes a single array (whose contents we build by concatenation)
    rather than unpacking it positionally — `*array` does not work under
    ``jax.jit`` when the array is traced.
    """
    import sympy

    syms = [sympy.Symbol(n) for n in ordered_names]
    fn_raw = sympy.lambdify(syms, expr, modules="jax")
    n = len(ordered_names)

    def fn(args):
        return fn_raw(*[args[i] for i in range(n)])

    return fn


def _topo_sort_algebraic(algebraic_exprs: Dict[str, Any]) -> List[str]:
    """Topologically sort algebraic names by mutual dependencies. Assumes
    expressions have already been parsed (sympy)."""
    algebraic_names = set(algebraic_exprs.keys())
    deps = {
        name: {str(s) for s in expr.free_symbols} & algebraic_names
        for name, expr in algebraic_exprs.items()
    }
    order: List[str] = []
    remaining = set(algebraic_names)
    while remaining:
        ready = sorted(n for n in remaining if not (deps[n] & remaining))
        if not ready:
            raise ValueError(
                "Cyclic algebraic dependencies detected during topo-sort: "
                f"{sorted(remaining)}"
            )
        order.extend(ready)
        remaining -= set(ready)
    return order


def _parse_biological_ode_expressions(
    bo,
    allowed_names: set,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Sympify ``BiologicalOde.algebraic`` and ``derivatives`` strings.

    Returns ``(algebraic_exprs, derivative_exprs)`` keyed by name. Raises
    ``ValueError`` on parse failure.
    """
    import sympy

    symbol_table = {n: sympy.Symbol(n) for n in allowed_names}

    algebraic_exprs: Dict[str, Any] = {}
    for name, expr_str in bo.algebraic.items():
        try:
            algebraic_exprs[name] = sympy.sympify(expr_str, locals=symbol_table)
        except Exception as exc:
            raise ValueError(
                f"biological_ode.algebraic[{name!r}] failed to parse: {exc}"
            ) from exc

    derivative_exprs: Dict[str, Any] = {}
    for name, expr_str in bo.derivatives.items():
        try:
            derivative_exprs[name] = sympy.sympify(expr_str, locals=symbol_table)
        except Exception as exc:
            raise ValueError(
                f"biological_ode.derivatives[{name!r}] failed to parse: {exc}"
            ) from exc

    return algebraic_exprs, derivative_exprs


# ---------------------------------------------------------------------------
# ProcessOrdering factory
# ---------------------------------------------------------------------------


def get_process_ordering(process: BioProcess) -> ProcessOrdering:
    """Build the canonical :class:`ProcessOrdering` for a process.

    All other mechanistic factories (``get_control_splines``,
    ``build_rhs_ode``, ``extract_discrete_events``, ``build_state_splines``)
    consume this object so every state/control/rate vector has the same
    layout.

    Validations performed here:

    - Every continuous ``Inflow`` must define ``feed_medium`` and
      every component in that medium must exist in
      ``reactor_medium.components``.
    - Every ``Outflow.retention`` key must name a reactor component, every
      value must be in ``[0, 1]``, and discrete Outflows cannot set retention.
    - Every non-controlled ``ProcessVariable`` must have a ``TimeSeries``
      value carrier (static PVs must be ``is_controlled=True``).
    - The ``BiologicalOde.algebraic`` graph must be acyclic.
    - All names across every group must be unique (no shared names between
      states, rates, algebraic, controlled PVs, Inflows, Outflows).
    """
    # ---- Reactor components (RMCs) — alphabetical
    name_modeled_RMCs = tuple(sorted(process.reactor_medium.components.keys()))

    # ---- Process variables — partition into controlled vs modeled, alphabetical
    name_modeled_PVs = tuple(
        sorted(n for n, pv in process.process_variables.items() if not pv.is_controlled)
    )
    name_controlled_PVs = tuple(
        sorted(n for n, pv in process.process_variables.items() if pv.is_controlled)
    )

    # Static PVs must be is_controlled=True; modeled PVs must carry a TimeSeries.
    for pv_name in name_modeled_PVs:
        pv = process.process_variables[pv_name]
        if isinstance(pv.values, StaticVariable):
            raise ValueError(
                f"Process variable {pv_name!r} has StaticVariable values but "
                "is_controlled=False. Static (non-time-varying) process "
                "variables must be flagged is_controlled=True so they live "
                "in name_controlled_PVs, not name_modeled_PVs."
            )

    # ---- Volume changes: Inflow/Outflow × controlled/modeled, alphabetical
    name_modeled_Inflows = tuple(
        sorted(
            n
            for n, vc in process.volume.volume_changes.items()
            if (isinstance(vc, Inflow) and vc.is_continuous and not vc.is_controlled)
        )
    )
    name_controlled_Inflows = tuple(
        sorted(
            n
            for n, vc in process.volume.volume_changes.items()
            if isinstance(vc, Inflow) and vc.is_continuous and vc.is_controlled
        )
    )
    name_modeled_Outflows = tuple(
        sorted(
            n
            for n, vc in process.volume.volume_changes.items()
            if (isinstance(vc, Outflow) and vc.is_continuous and not vc.is_controlled)
        )
    )
    name_controlled_Outflows = tuple(
        sorted(
            n
            for n, vc in process.volume.volume_changes.items()
            if isinstance(vc, Outflow) and vc.is_continuous and vc.is_controlled
        )
    )

    # Inflow feed-medium validation (across modeled and controlled)
    rmc_set = set(name_modeled_RMCs)
    for vc_name in name_modeled_Inflows + name_controlled_Inflows:
        vc = process.volume.volume_changes[vc_name]
        if vc.feed_medium is None:
            raise ValueError(f"Inflow {vc_name!r} has no feed_medium defined.")
        unknown = [c for c in vc.feed_medium.components.keys() if c not in rmc_set]
        if unknown:
            raise ValueError(
                f"Inflow {vc_name!r} references unknown reactor "
                f"component(s) in its feed_medium: {unknown}. All feed "
                "components must exist in process.reactor_medium.components."
            )

    # Validate retention here because mechanistic factories must not silently
    # turn invalid component names into zero-retention rows.
    for vc_name, vc in process.volume.volume_changes.items():
        if not isinstance(vc, Outflow) or not vc.retention:
            continue
        if not vc.is_continuous:
            raise ValueError(
                f"Outflow {vc_name!r} sets retention {vc.retention!r} but is "
                "discrete (is_continuous=False). Retention is only "
                "implemented for continuous Outflows; setting it on a "
                "discrete Outflow would be silently ignored by the RHS ODE."
            )
        unknown = [name for name in vc.retention if name not in rmc_set]
        if unknown:
            raise ValueError(
                f"Outflow {vc_name!r} retention references unknown reactor "
                f"component(s): {unknown}."
            )
        out_of_range = {
            name: value
            for name, value in vc.retention.items()
            if not 0.0 <= value <= 1.0
        }
        if out_of_range:
            raise ValueError(
                f"Outflow {vc_name!r} retention value(s) out of [0, 1]: {out_of_range}."
            )

    # ---- Biological ODE — rates (insertion order) and algebraic (topo-sorted)
    bo = process.biological_ode
    if bo is None:
        name_modeled_rates: Tuple[str, ...] = ()
        name_modeled_algebraic: Tuple[str, ...] = ()
    else:
        name_modeled_rates = tuple(bo.rates.keys())
        # Need parsed algebraic expressions for topo-sort. Build a minimal
        # allowed-symbol set that lets sympify recognise every name — the
        # full validation is done by validate_biological_ode.
        allowed = (
            set(name_modeled_RMCs)
            | set(name_modeled_PVs)
            | set(name_controlled_PVs)
            | set(bo.algebraic.keys())
            | set(name_modeled_rates)
        )
        algebraic_exprs, _ = _parse_biological_ode_expressions(bo, allowed)
        name_modeled_algebraic = tuple(_topo_sort_algebraic(algebraic_exprs))

    # ---- Cross-group name-collision check
    groups = {
        "name_modeled_RMCs": name_modeled_RMCs,
        "name_modeled_PVs": name_modeled_PVs,
        "name_controlled_PVs": name_controlled_PVs,
        "name_modeled_Inflows": name_modeled_Inflows,
        "name_controlled_Inflows": name_controlled_Inflows,
        "name_modeled_Outflows": name_modeled_Outflows,
        "name_controlled_Outflows": name_controlled_Outflows,
        "name_modeled_rates": name_modeled_rates,
        "name_modeled_algebraic": name_modeled_algebraic,
    }
    seen: Dict[str, str] = {}
    duplicates: List[str] = []
    for group_name, names in groups.items():
        for n in names:
            if n in seen:
                duplicates.append(f"{n!r} appears in both {seen[n]} and {group_name}")
            else:
                seen[n] = group_name
    if duplicates:
        raise ValueError(
            "Name collisions across ProcessOrdering groups:\n  - "
            + "\n  - ".join(duplicates)
        )

    return ProcessOrdering(
        name_modeled_rates=name_modeled_rates,
        name_modeled_algebraic=name_modeled_algebraic,
        name_modeled_RMCs=name_modeled_RMCs,
        name_modeled_PVs=name_modeled_PVs,
        name_modeled_Inflows=name_modeled_Inflows,
        name_modeled_Outflows=name_modeled_Outflows,
        name_controlled_PVs=name_controlled_PVs,
        name_controlled_Inflows=name_controlled_Inflows,
        name_controlled_Outflows=name_controlled_Outflows,
    )


# ---------------------------------------------------------------------------
# ControlSplines module
# ---------------------------------------------------------------------------


class ControlSplines(eqx.Module):
    """JAX/Equinox module that evaluates all controlled signals at time *t*.

    Created by :func:`get_control_splines`.

    Output layout of ``__call__(t)``::

        u = [Inflow_flow_rates... | Outflow_flow_rates... | PV_values...]

    The first ``len(name_controlled_Inflows) + len(name_controlled_Outflows)``
    entries are spline derivatives (flow rates carrying the storage sign of
    the underlying cumulative-volume series). The remaining entries are
    direct PV values.
    """

    name_controlled_Inflows: tuple[str, ...] = eqx.field(static=True)
    name_controlled_Outflows: tuple[str, ...] = eqx.field(static=True)
    name_controlled_PVs: tuple[str, ...] = eqx.field(static=True)
    # Original control PPolys in canonical [Inflow | Outflow | PV] order.
    _splines: tuple[PPoly, ...]

    def __call__(self, t: jnp.ndarray) -> jnp.ndarray:
        if not self._splines:
            return jnp.zeros(jnp.shape(t) + (0,))
        n_flows = len(self.name_controlled_Inflows) + len(self.name_controlled_Outflows)
        values = [
            spline(t, nu=1) if i < n_flows else spline(t)
            for i, spline in enumerate(self._splines)
        ]
        return jnp.stack(values, axis=-1)


def get_control_splines(
    process: BioProcess,
    ordering: Optional[ProcessOrdering] = None,
) -> ControlSplines:
    """Build a :class:`ControlSplines` module for *process*.

    Block layout (Inflows → Outflows → PVs) is enforced by the canonical ordering
    in :class:`ProcessOrdering`.
    """
    if ordering is None:
        ordering = get_process_ordering(process)

    t_start = float(process.time_axis.start)
    t_end = float(process.time_axis.end)
    splines: list[PPoly] = []

    for vc_name in ordering.name_controlled_Inflows:
        vc = process.volume.volume_changes[vc_name]
        splines.append(_timeseries_to_ppoly(vc.values))
    for vc_name in ordering.name_controlled_Outflows:
        vc = process.volume.volume_changes[vc_name]
        splines.append(_timeseries_to_ppoly(vc.values))
    for pv_name in ordering.name_controlled_PVs:
        pv = process.process_variables[pv_name]
        splines.append(_value_to_ppoly(pv.values, t_start=t_start, t_end=t_end))

    return ControlSplines(
        name_controlled_Inflows=ordering.name_controlled_Inflows,
        name_controlled_Outflows=ordering.name_controlled_Outflows,
        name_controlled_PVs=ordering.name_controlled_PVs,
        _splines=tuple(splines),
    )


# ---------------------------------------------------------------------------
# Cin matrix builder
# ---------------------------------------------------------------------------


def _build_cin(
    process: BioProcess,
    vc_names: Tuple[str, ...],
    name_modeled_RMCs: Tuple[str, ...],
) -> jnp.ndarray:
    """Build a ``(len(vc_names), len(RMCs))`` feed-composition matrix.

    Each row is the static feed concentration of every reactor component in
    the corresponding ``Inflow.feed_medium``. Components omitted from the
    sparse feed mapping remain zero in the preinitialized row.
    """
    n = len(vc_names)
    n_RMCs = len(name_modeled_RMCs)
    Cin = jnp.zeros((n, n_RMCs), dtype=float)
    for k, vc_name in enumerate(vc_names):
        vc = process.volume.volume_changes[vc_name]
        if not isinstance(vc, Inflow) or vc.feed_medium is None:
            continue
        feed = vc.feed_medium
        for j, sp_name in enumerate(name_modeled_RMCs):
            if sp_name not in feed.components:
                continue
            conc = feed.components[sp_name].concentration
            if isinstance(conc, StaticVariable):
                Cin = Cin.at[k, j].set(float(conc.value))
            else:
                raise NotImplementedError(
                    "TimeSeries feed concentrations are not yet supported. "
                    f"Found TimeSeries for species {sp_name!r} in feed "
                    f"{feed.name!r} of volume change {vc_name!r}."
                )
    return Cin


def _build_retention(
    process: BioProcess,
    vc_names: Tuple[str, ...],
    name_modeled_RMCs: Tuple[str, ...],
) -> jnp.ndarray:
    """Build a ``(len(vc_names), len(RMCs))`` retention matrix.

    Each row is the per-component retention fraction (sigma in [0, 1]) of
    the corresponding ``Outflow.retention``. Unlike ``_build_cin``,
    a missing entry here is deliberately left at 0 (not an error) — an
    empty/partial ``retention`` is the ordinary, correct state for
    the overwhelming majority of processes (no perfusion/evaporation
    modeling), not a data gap standing in for something unknown.
    """
    n = len(vc_names)
    n_RMCs = len(name_modeled_RMCs)
    retention = jnp.zeros((n, n_RMCs), dtype=float)
    for k, vc_name in enumerate(vc_names):
        vc = process.volume.volume_changes[vc_name]
        if not isinstance(vc, Outflow):
            continue
        for j, sp_name in enumerate(name_modeled_RMCs):
            if sp_name in vc.retention:
                retention = retention.at[k, j].set(float(vc.retention[sp_name]))
    return retention


# ---------------------------------------------------------------------------
# RhsOde module
# ---------------------------------------------------------------------------


class RhsOde(eqx.Module):
    """JAX/Equinox module that evaluates the biological RHS for a process.

    Built by :func:`build_rhs_ode` from ``process.biological_ode``
    (auto-generated in :meth:`BioProcess.__post_init__` when not
    user-supplied). The biological ``dc/dt`` per state comes from
    user-written expression strings; bp-format adds the physical
    contributions (feed, dilution, dV) on top.

    Call signature::

        dc_dt = rhs_ode(c, rates, u, f_modeled_Inflows, f_modeled_Outflows)

    where:

    - ``c = [name_modeled_RMCs... | name_modeled_PVs... | V]``.
    - ``rates`` aligns with :attr:`name_modeled_rates`.
    - ``u`` is the full control vector (output of ``ControlSplines``):
      ``[Inflow_flows | Outflow_flows | PV_values]``.
    - ``f_modeled_Inflows`` are uncontrolled Inflow flow rates aligned with
      :attr:`name_modeled_Inflows` (non-negative).
    - ``f_modeled_Outflows`` are uncontrolled Outflow flow rates aligned with
      :attr:`name_modeled_Outflows` (non-positive, signed).

    Returns ``dc/dt`` of shape ``(len(RMCs) + len(PVs) + 1,)``.
    """

    # --- Names (deterministic ordering)
    name_modeled_rates: tuple = eqx.field(static=True)
    name_modeled_algebraic: tuple = eqx.field(static=True)
    name_modeled_RMCs: tuple = eqx.field(static=True)
    name_modeled_PVs: tuple = eqx.field(static=True)
    name_modeled_Inflows: tuple = eqx.field(static=True)
    name_modeled_Outflows: tuple = eqx.field(static=True)
    name_controlled_PVs: tuple = eqx.field(static=True)
    name_controlled_Inflows: tuple = eqx.field(static=True)
    name_controlled_Outflows: tuple = eqx.field(static=True)

    # --- Compiled callables
    algebraic_funcs: tuple = eqx.field(static=True)
    derivative_funcs: tuple = eqx.field(static=True)

    # --- Feed compositions
    Cin_controlled_Inflows: jnp.ndarray
    Cin_modeled_Inflows: jnp.ndarray

    # --- Outflow component retention (sigma in [0, 1], default 0)
    retention_controlled_Outflows: jnp.ndarray
    retention_modeled_Outflows: jnp.ndarray

    def __call__(
        self,
        c: jnp.ndarray,
        rates: jnp.ndarray,
        u: jnp.ndarray,
        f_modeled_Inflows: jnp.ndarray,
        f_modeled_Outflows: jnp.ndarray,
        V_min: float | jnp.ndarray = _MIN_REACTOR_VOLUME,
    ) -> jnp.ndarray:
        n_RMCs = len(self.name_modeled_RMCs)
        n_PVs = len(self.name_modeled_PVs)
        n_inflow_ctrl = len(self.name_controlled_Inflows)
        n_outflow_ctrl = len(self.name_controlled_Outflows)

        # Unpack state
        c_RMCs = c[:n_RMCs]
        c_PVs = c[n_RMCs : n_RMCs + n_PVs]
        V = c[n_RMCs + n_PVs]

        # Unpack control vector (Inflow flows | Outflow flows | PV values)
        u_controlled_Inflows = u[:n_inflow_ctrl]
        u_controlled_Outflows = u[n_inflow_ctrl : n_inflow_ctrl + n_outflow_ctrl]
        ctrl_PVs = u[n_inflow_ctrl + n_outflow_ctrl :]

        state_and_ctrl = jnp.concatenate([c_RMCs, c_PVs, ctrl_PVs])

        # 1. Algebraic variables in topo order
        n_algebraic = len(self.name_modeled_algebraic)
        algebraic_arr = jnp.zeros(n_algebraic) if n_algebraic > 0 else jnp.zeros(0)
        for i, fn in enumerate(self.algebraic_funcs):
            args = jnp.concatenate([state_and_ctrl, algebraic_arr, rates])
            algebraic_arr = algebraic_arr.at[i].set(fn(args))

        # 2. Biological derivatives per dynamic state
        full_args = jnp.concatenate([state_and_ctrl, algebraic_arr, rates])
        biol_dc_list = [fn(full_args) for fn in self.derivative_funcs]
        biol_dc = jnp.stack(biol_dc_list) if biol_dc_list else jnp.zeros(0)

        # 3. Physical contributions on RMC states (PV states are biological-only).
        feed_term, dV = _apply_feed_dilution(
            c_RMCs,
            V,
            u_controlled_Inflows,
            u_controlled_Outflows,
            f_modeled_Inflows,
            f_modeled_Outflows,
            self.Cin_controlled_Inflows,
            self.Cin_modeled_Inflows,
            self.retention_controlled_Outflows,
            self.retention_modeled_Outflows,
            n_RMCs,
            V_min,
        )

        dc_RMCs = biol_dc[:n_RMCs] + feed_term
        dc_PVs = biol_dc[n_RMCs : n_RMCs + n_PVs] if n_PVs > 0 else jnp.zeros(0)

        return jnp.concatenate([dc_RMCs, dc_PVs, jnp.array([dV])])


# ---------------------------------------------------------------------------
# RhsOde builder
# ---------------------------------------------------------------------------


def build_rhs_ode(
    process: BioProcess,
    ordering: Optional[ProcessOrdering] = None,
) -> RhsOde:
    """Build a :class:`RhsOde` from a process whose ``biological_ode`` is
    set (auto-generated in :meth:`BioProcess.__post_init__` when not
    user-supplied). Raises ``ValueError`` if the block is missing or
    fails validation.
    """
    import sympy

    bo = process.biological_ode
    if bo is None:
        raise ValueError("build_rhs_ode requires process.biological_ode to be set.")

    if ordering is None:
        ordering = get_process_ordering(process)

    # Canonical args order — every lambdified expression consumes a single
    # flat array indexed in this exact order. Must match the concatenation
    # order in RhsOde.__call__.
    args_order: Tuple[str, ...] = (
        ordering.name_modeled_RMCs
        + ordering.name_modeled_PVs
        + ordering.name_controlled_PVs
        + ordering.name_modeled_algebraic
        + ordering.name_modeled_rates
    )

    allowed_names = set(args_order)
    algebraic_exprs, derivative_exprs = _parse_biological_ode_expressions(
        bo, allowed_names
    )

    algebraic_funcs = tuple(
        _lambdify_with_array_arg(algebraic_exprs[n], args_order)
        for n in ordering.name_modeled_algebraic
    )

    state_names = ordering.name_modeled_RMCs + ordering.name_modeled_PVs
    zero_expr = sympy.Integer(0)
    derivative_funcs = tuple(
        _lambdify_with_array_arg(derivative_exprs.get(n, zero_expr), args_order)
        for n in state_names
    )

    Cin_controlled_Inflows = _build_cin(
        process, ordering.name_controlled_Inflows, ordering.name_modeled_RMCs
    )
    Cin_modeled_Inflows = _build_cin(
        process, ordering.name_modeled_Inflows, ordering.name_modeled_RMCs
    )
    retention_controlled_Outflows = _build_retention(
        process, ordering.name_controlled_Outflows, ordering.name_modeled_RMCs
    )
    retention_modeled_Outflows = _build_retention(
        process, ordering.name_modeled_Outflows, ordering.name_modeled_RMCs
    )

    return RhsOde(
        name_modeled_rates=ordering.name_modeled_rates,
        name_modeled_algebraic=ordering.name_modeled_algebraic,
        name_modeled_RMCs=ordering.name_modeled_RMCs,
        name_modeled_PVs=ordering.name_modeled_PVs,
        name_modeled_Inflows=ordering.name_modeled_Inflows,
        name_modeled_Outflows=ordering.name_modeled_Outflows,
        name_controlled_PVs=ordering.name_controlled_PVs,
        name_controlled_Inflows=ordering.name_controlled_Inflows,
        name_controlled_Outflows=ordering.name_controlled_Outflows,
        algebraic_funcs=algebraic_funcs,
        derivative_funcs=derivative_funcs,
        Cin_controlled_Inflows=Cin_controlled_Inflows,
        Cin_modeled_Inflows=Cin_modeled_Inflows,
        retention_controlled_Outflows=retention_controlled_Outflows,
        retention_modeled_Outflows=retention_modeled_Outflows,
    )


# ---------------------------------------------------------------------------
# Algebraic-variable inspector
# ---------------------------------------------------------------------------


def build_algebraic_func(
    process: BioProcess,
    ordering: Optional[ProcessOrdering] = None,
) -> Callable[[jnp.ndarray, jnp.ndarray, jnp.ndarray], Dict[str, jnp.ndarray]]:
    """Build a callable that evaluates the ``biological_ode.algebraic``
    quantities from concrete ``(state_values, ctrl_pv_values, rates)``
    arrays.

    ``state_values`` is the non-volume state in canonical order
    (``name_modeled_RMCs`` followed by ``name_modeled_PVs``);
    ``ctrl_pv_values`` aligns with ``name_controlled_PVs``; ``rates``
    aligns with ``name_modeled_rates``.

    Returns ``{name: scalar}`` keyed by ``name_modeled_algebraic`` (topo
    order). Useful for plotting / loss computation.
    """
    bo = process.biological_ode
    if bo is None:
        raise ValueError(
            "build_algebraic_func requires process.biological_ode to be set."
        )
    rhs_ode = build_rhs_ode(process, ordering=ordering)
    name_modeled_algebraic = rhs_ode.name_modeled_algebraic
    n_algebraic = len(name_modeled_algebraic)
    algebraic_funcs = rhs_ode.algebraic_funcs

    def algebraic_func(state_values, ctrl_pv_values, rates):
        algebraic_arr = jnp.zeros(n_algebraic) if n_algebraic > 0 else jnp.zeros(0)
        for i, fn in enumerate(algebraic_funcs):
            args = jnp.concatenate([state_values, ctrl_pv_values, algebraic_arr, rates])
            algebraic_arr = algebraic_arr.at[i].set(fn(args))
        return {name: algebraic_arr[i] for i, name in enumerate(name_modeled_algebraic)}

    return algebraic_func


# ---------------------------------------------------------------------------
# Discrete-event extraction
# ---------------------------------------------------------------------------


def extract_discrete_events(
    process: BioProcess,
    ordering: ProcessOrdering,
) -> List[Dict[str, Any]]:
    """Extract sampling and bolus-feed events from *process*.

    Each event dict carries:

    - ``t`` (float): event time
    - ``kind`` (str): ``'sample'`` or ``'bolus_feed'``
    - ``dV`` (float): signed volume change (positive = add, negative = remove)
    - ``Cin`` (jnp.ndarray | None): feed composition aligned with
      ``ordering.name_modeled_RMCs`` (None for sampling events)
    - ``source`` (str): originating volume-change name

    Events are sorted by time; sampling precedes bolus feeds at the same
    timestamp. At most one sample and one bolus per timestamp.
    """
    events: List[Dict[str, Any]] = []
    n_RMCs = len(ordering.name_modeled_RMCs)

    for vc_name, vc in process.volume.volume_changes.items():
        if vc.is_continuous:
            continue

        tp = jnp.asarray(vc.values.times, dtype=float)
        vv = jnp.asarray(vc.values.values, dtype=float)

        for t_event, dV_event in zip(tp, vv):
            if abs(dV_event) < 1e-15:
                continue

            if dV_event > 0 and isinstance(vc, Inflow):
                if vc.feed_medium is None:
                    raise ValueError(
                        f"Inflow {vc_name!r} has a positive discrete (bolus) "
                        "event but no feed_medium defined. There's no "
                        "reasonable way to fabricate an entire medium's "
                        "identity from nothing — define feed_medium explicitly."
                    )
                Cin_event = jnp.zeros(n_RMCs)
                for j, sp_name in enumerate(ordering.name_modeled_RMCs):
                    if sp_name not in vc.feed_medium.components:
                        continue
                    conc = vc.feed_medium.components[sp_name].concentration
                    if isinstance(conc, StaticVariable):
                        Cin_event = Cin_event.at[j].set(float(conc.value))
                    else:
                        raise NotImplementedError(
                            "TimeSeries feed concentrations are not yet "
                            f"supported. Found TimeSeries for species "
                            f"{sp_name!r} in feed {vc.feed_medium.name!r} of "
                            f"volume change {vc_name!r}."
                        )
                events.append(
                    dict(
                        t=float(t_event),
                        kind="bolus_feed",
                        dV=float(dV_event),
                        Cin=Cin_event,
                        source=vc_name,
                    )
                )
            else:
                events.append(
                    dict(
                        t=float(t_event),
                        kind="sample",
                        dV=float(dV_event),
                        Cin=None,
                        source=vc_name,
                    )
                )

    events.sort(
        key=lambda e: (
            e["t"],
            0 if e["kind"] == "sample" else 1,
        )
    )

    # At a single timestamp at most one sample and one bolus.
    per_time_counts: Dict[float, Dict[str, int]] = {}
    for ev in events:
        t = float(ev["t"])
        kind = str(ev["kind"])
        per_time_counts.setdefault(t, {"sample": 0, "bolus_feed": 0})
        per_time_counts[t][kind] += 1

    duplicate_kinds = []
    for t, counts in per_time_counts.items():
        if counts["sample"] > 1:
            duplicate_kinds.append((t, "sample", counts["sample"]))
        if counts["bolus_feed"] > 1:
            duplicate_kinds.append((t, "bolus_feed", counts["bolus_feed"]))
    if duplicate_kinds:
        details = ", ".join(
            f"t={t}: {kind} x{count}" for t, kind, count in duplicate_kinds
        )
        raise ValueError(
            "At most one discrete event per kind is allowed at a given time "
            f"(allowed: one sample and one bolus). Found duplicates: {details}."
        )
    return events


# ---------------------------------------------------------------------------
# State splines
# ---------------------------------------------------------------------------


def _timeseries_samples_match(left: TimeSeries, right: TimeSeries) -> bool:
    """Compare TimeSeries sample anchors used by mechanistic initial states."""
    import numpy as np

    if left.times is None or left.values is None:
        return False
    if right.times is None or right.values is None:
        return False
    left_times = np.asarray(left.times, dtype=float)
    right_times = np.asarray(right.times, dtype=float)
    left_values = np.asarray(left.values, dtype=float)
    right_values = np.asarray(right.values, dtype=float)
    return (
        left_times.shape == right_times.shape
        and left_values.shape == right_values.shape
        and np.allclose(left_times, right_times, rtol=0.0, atol=1e-12)
        and np.allclose(left_values, right_values, rtol=1e-10, atol=1e-12)
    )


def build_state_splines(
    process: BioProcess,
    ordering: ProcessOrdering,
) -> Dict[str, Any]:
    """Build state splines from stored TimeSeries spline state.

    Reactor-component and process-variable states are converted directly
    from their TimeSeries or StaticVariable carriers.

    Returns ``{state_name: spline_callable}`` for every non-volume state
    in ``ordering.name_modeled_RMCs + ordering.name_modeled_PVs``.
    """
    state_splines: Dict[str, Any] = {}

    for sp_name in ordering.name_modeled_RMCs:
        comp = process.reactor_medium.components[sp_name]
        state_splines[sp_name] = _value_to_ppoly(
            comp.concentration,
            t_start=float(process.time_axis.start),
            t_end=float(process.time_axis.end),
        )

    for pv_name in ordering.name_modeled_PVs:
        pv = process.process_variables[pv_name]
        state_splines[pv_name] = _value_to_ppoly(
            pv.values,
            t_start=float(process.time_axis.start),
            t_end=float(process.time_axis.end),
        )

    return state_splines
