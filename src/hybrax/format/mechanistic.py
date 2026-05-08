"""
Mechanistic API for bp-format.

JAX/Equinox-compatible modules for building continuous-time control functions
and ODE right-hand sides directly from a :class:`~bp_format.BioProcess`. All
modules are fully JAX-jittable via ``equinox.filter_jit``.

Public API
----------
get_control_splines(process) -> ControlSplines
    ``ControlSplines.__call__(t)`` evaluates all controlled signals at ``t``.
    Continuous volume-change feeds are returned as **flow rates** (derivative
    of the cumulative-volume spline).

get_rhs_ode(process) -> RhsOde
    Build the :class:`RhsOde` for a process whose ``biological_ode`` block
    is set (auto-generated in :meth:`BioProcess.__post_init__` when not
    user-supplied). ``RhsOde.__call__(c, rates, u_flow, f_modeled, ctrl_pv_values)``
    computes ``dc/dt`` (including ``dV/dt``).

extract_discrete_events(process, rhs_ode) -> list[dict]
    Extract discrete events (sampling, bolus feeds) from a BioProcess.

build_state_splines(process, rhs_ode) -> dict
    Build spline callables for all non-volume states.

integrate_process(process, ctrl, rhs_ode, rates_func, t_eval) -> dict
    Full hybrid ODE integration with discrete event handling. Honest forward
    integration: state read directly from the integrator at every step.

Rates protocol
--------------
``rates_func(t, state, controls) -> jnp.ndarray`` of shape ``(rhs_ode.rate_size,)``
aligned with ``rhs_ode.rate_names`` (= the insertion order of
``process.biological_ode.rates``). The integrator handles feed, dilution, sample
and volume dynamics on top of the biological ``dc/dt`` returned by ``rhs_ode``.

For the auto-generated :class:`BiologicalOde` produced by
``BioProcess.__post_init__``, the rate name layout is
``q_<rmc_biomass_first>... + r_<dynamic_pv>...``.

The legacy spline-based rate inversion helpers (``build_q_func``,
``build_rates_func``, ``estimate_specific_rates``, ``integrate_process_pseudospace``)
were removed as part of the P3 refactor. They will be replaced by
``build_rates_func_analytical`` (see ``documentation/_analytical_rates_spec.md``).
"""

from __future__ import annotations

from dataclasses import dataclass as _py_dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple

import diffrax
import equinox as eqx
import interpax
import jax
import jax.numpy as jnp
import numpy as np

from .dataclasses import (
    BioProcess,
    FeedVolumeChange,
    StaticVariable,
    TimeSeries,
)
from .splines import (
    _DEFAULT_BATCH_KNOTS,
    _MIN_REACTOR_VOLUME,
    _adf_for_division,
    _constant_timeseries,
    _evaluate_with_boundary_start,
    _require_reactor_volume_scalar,
    build_backtransform_spline,
    make_interpax_spline,
)


# ---------------------------------------------------------------------------
# Batched spline helpers
# ---------------------------------------------------------------------------

_MIN_ACTIVE_BIOMASS = 1e-6
_SEGMENT_BOUNDARY_TOL = 1e-12
_MIN_SOLVER_DT0 = 1e-6


def _require_reactor_volume_above_threshold(
    volume: jnp.ndarray,
    *,
    context: str,
) -> jnp.ndarray:
    """Fail when reactor volume reaches a physically invalid near-zero value."""
    volume_arr = jnp.asarray(volume)
    return eqx.error_if(
        volume_arr,
        jnp.any(volume_arr <= _MIN_REACTOR_VOLUME),
        f"{context} reached zero or near-zero reactor volume.",
    )


def _batch_splines(
    spline_list: List[interpax.CubicSpline],
    t_start: float,
    t_end: float,
    n_knots: int = _DEFAULT_BATCH_KNOTS,
) -> interpax.PPoly:
    """Resample splines onto a shared uniform grid and stack into a single PPoly.

    The returned PPoly has coefficients of shape ``(4, n_knots-1, n_splines)``
    so that evaluating at scalar ``t`` returns ``(n_splines,)`` in one call.
    """
    x_common = jnp.linspace(t_start, t_end, n_knots)
    resampled = [
        interpax.CubicSpline(x_common, sp(x_common), bc_type="natural", check=False)
        for sp in spline_list
    ]
    c_stacked = jnp.stack([s.c for s in resampled], axis=-1)  # (4, m, n)
    return interpax.PPoly.construct_fast(c_stacked, x_common, extrapolate=True)


def _timeseries_to_interpax_spline(series: TimeSeries) -> interpax.PPoly:
    """Build an interpax spline from a TimeSeries carrier.

    Prefer stored spline state when available so mechanistic consumers use the
    same canonical representation that was fit/serialized. Fall back to a
    cubic refit only for sample-only series without spline coefficients.
    """
    if (
        getattr(series, "breaks", None) is not None
        and getattr(series, "coeffs", None) is not None
    ):
        coeffs = jnp.asarray(series.coeffs, dtype=float).T[::-1]
        breaks = jnp.asarray(series.breaks, dtype=float)
        return interpax.PPoly.construct_fast(coeffs, breaks, extrapolate=True)
    if (
        getattr(series, "times", None) is not None
        and getattr(series, "values", None) is not None
    ):
        return make_interpax_spline(
            jnp.asarray(series.times, dtype=float),
            jnp.asarray(series.values, dtype=float),
        )
    raise ValueError("TimeSeries must provide spline state or discrete samples.")


def _value_to_interpax_spline(
    value: TimeSeries | StaticVariable,
    *,
    t_start: float,
    t_end: float,
) -> interpax.CubicSpline:
    """Build an interpax spline from a dynamic or static state carrier."""
    if isinstance(value, TimeSeries):
        return _timeseries_to_interpax_spline(value)
    v = float(value.value)
    return make_interpax_spline(
        jnp.array([t_start, t_end], dtype=float),
        jnp.array([v, v], dtype=float),
    )


# ---------------------------------------------------------------------------
# Intracellular mass-balance helpers
# ---------------------------------------------------------------------------


def _apply_feed_dilution(
    c_reactor: jnp.ndarray,
    V: jnp.ndarray,
    u_flow: jnp.ndarray,
    f_modeled: jnp.ndarray,
    Cin: jnp.ndarray,
    Cin_modeled: jnp.ndarray,
    u_flow_size: int,
    f_modeled_size: int,
    n_reactor: int,
) -> tuple:
    """Compute the feed/dilution contribution on reactor-component states and
    the volume derivative from controlled and modeled flows.

    Returns ``(feed_term, dV)`` where ``feed_term`` has shape ``(n_reactor,)``
    and ``dV`` is a scalar. Used by both the auto-generated :class:`RhsOde`
    and the user-defined RHS so the physical-contributions code stays in one
    place.
    """
    V = _require_reactor_volume_above_threshold(V, context="ODE state")
    feed_term = jnp.zeros(n_reactor)
    dV = jnp.zeros(())
    if u_flow_size > 0:
        feed_term = (
            feed_term
            + jnp.sum(u_flow[:, None] * (Cin - c_reactor[None, :]), axis=0) / V
        )
        dV = dV + jnp.sum(u_flow)
    if f_modeled_size > 0:
        feed_term = (
            feed_term
            + jnp.sum(f_modeled[:, None] * (Cin_modeled - c_reactor[None, :]), axis=0)
            / V
        )
        dV = dV + jnp.sum(f_modeled)
    return feed_term, dV


# ---------------------------------------------------------------------------
# ControlSplines module
# ---------------------------------------------------------------------------


class ControlSplines(eqx.Module):
    """JAX/Equinox module that evaluates all controlled signals at time *t*.

    Created by :func:`get_control_splines`; do not instantiate directly.

    Attributes
    ----------
    control_names : tuple[str, ...]
        Deterministic ordering of all controlled signals included in
        ``__call__``.  Continuous volume-change flows come first (in
        insertion order), followed by controlled process variables.
    flow_indices : tuple[int, ...]
        Positions inside *control_names* (and the output of ``__call__``)
        that correspond to continuous volume-change **flow rates**.
    ctrl_indices : tuple[int, ...]
        Positions that correspond to controlled process variables (pH,
        temperature, initial conditions, etc.).

    Notes
    -----
    Discrete volume changes (``is_continuous=False``) are **excluded** from
    the module because they represent jump events (sampling, bolus additions)
    that must be handled as state discontinuities, not smooth RHS terms.

    JIT usage::

        import equinox as eqx
        ctrl  = get_control_splines(process)
        u     = eqx.filter_jit(ctrl)(t)         # shape (n_controls,)
        u_flow = u[list(ctrl.flow_indices)]      # continuous-flow subset
    """

    control_names: tuple = eqx.field(static=True)
    flow_indices: tuple = eqx.field(static=True)
    ctrl_indices: tuple = eqx.field(static=True)
    _batched: interpax.PPoly  # batched PPoly with shape (4, m, n_controls)
    _deriv_mask: jnp.ndarray  # boolean mask: True → return d/dt
    _n_controls: int = eqx.field(static=True)

    def __call__(self, t: jnp.ndarray) -> jnp.ndarray:
        """Evaluate all controlled signals at scalar time *t*.

        Parameters
        ----------
        t:
            Scalar time (same unit as the underlying :class:`~bp_format.TimeSeries`).

        Returns
        -------
        jnp.ndarray, shape (n_controls,)
            Stacked array of controlled-signal values in the order given by
            :attr:`control_names`.  Continuous volume-change entries are
            **flow rates** (first derivative of the cumulative-volume spline).
        """
        if self._n_controls == 0:
            return jnp.zeros(0)
        vals = self._batched(t)  # (n_controls,)
        dvals = self._batched(t, nu=1)  # (n_controls,) — derivative
        return jnp.where(self._deriv_mask, dvals, vals)


# ---------------------------------------------------------------------------
# Sympy helpers used by the RhsOde builder
# ---------------------------------------------------------------------------


def _lambdify_with_array_arg(expr, ordered_names: Tuple[str, ...]) -> Callable:
    """Return ``f(args)`` that evaluates *expr* with the symbol values supplied
    as a flat 1-D array indexed by *ordered_names*.

    Uses a small Python wrapper around :func:`sympy.lambdify` so each call site
    passes a single array (whose contents we build by concatenation) rather
    than unpacking it positionally — `*array` does not work under ``jax.jit``
    when the array is traced.
    """
    import sympy

    syms = [sympy.Symbol(n) for n in ordered_names]
    fn_raw = sympy.lambdify(syms, expr, modules="jax")
    n = len(ordered_names)

    def fn(args):
        return fn_raw(*[args[i] for i in range(n)])

    return fn


def _topo_sort_algebraic(algebraic_exprs: Dict[str, Any]) -> List[str]:
    """Topologically sort algebraic names by mutual dependencies. Assumes the
    expressions have already been parsed and validated as acyclic."""
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


class RhsOde(eqx.Module):
    """JAX/Equinox module that evaluates the biological RHS for a process.

    Built by :func:`get_rhs_ode` from ``process.biological_ode`` (auto-generated
    in :meth:`BioProcess.__post_init__` when not user-supplied). The biological
    ``dc/dt`` per state comes from user-written expression strings; bp-format
    adds the physical contributions (feed, dilution, dV) on top.

    Call signature::

        dc_dt = rhs_ode(c, rates, u_flow, f_modeled, ctrl_pv_values)

    where:

    - ``c`` is the state vector ``[reactor..., pv..., V]``.
    - ``rates`` is the user-declared rate vector, shape ``(rate_size,)``,
      aligned with :attr:`rate_names`.
    - ``u_flow`` are continuous controlled-feed flow rates,
      shape ``(u_flow_size,)``.
    - ``f_modeled`` are continuous uncontrolled (modeled) flow rates,
      shape ``(f_modeled_size,)``. Pass ``jnp.zeros(0)`` when none.
    - ``ctrl_pv_values`` is an array of controlled-PV values at the current
      time, aligned with :attr:`controlled_pv_names`. Pass ``jnp.zeros(0)``
      when there are no controlled PVs.
    """

    c_size: int = eqx.field(static=True)
    rate_size: int = eqx.field(static=True)
    r_size: int = eqx.field(static=True)
    u_flow_size: int = eqx.field(static=True)
    f_modeled_size: int = eqx.field(static=True)
    output_size: int = eqx.field(static=True)
    n_reactor_states: int = eqx.field(static=True)
    n_pv_states: int = eqx.field(static=True)
    n_controlled_pv: int = eqx.field(static=True)
    reactor_indices: tuple = eqx.field(static=True)
    pv_indices: tuple = eqx.field(static=True)
    volume_idx: int = eqx.field(static=True)
    static_pv_indices: tuple = eqx.field(static=True)

    # --- Names (deterministic ordering) ---
    reactor_component_state_names: tuple = eqx.field(static=True)
    process_variable_state_names: tuple = eqx.field(static=True)
    controlled_pv_names: tuple = eqx.field(static=True)
    flow_names: tuple = eqx.field(static=True)
    modeled_flow_names: tuple = eqx.field(static=True)
    name_modeled_algebraic: tuple = eqx.field(static=True)
    rate_names: tuple = eqx.field(static=True)

    # --- Compiled callables ---
    # Each takes a flat args array (state | ctrl_pv | algebraic | rates) and
    # returns a scalar. Stored as static so equinox treats them as Python
    # config rather than pytree leaves.
    algebraic_funcs: tuple = eqx.field(static=True)
    derivative_funcs: tuple = eqx.field(static=True)

    # --- Feed-dilution data ---
    Cin: jnp.ndarray
    Cin_modeled: jnp.ndarray

    def __call__(
        self,
        c: jnp.ndarray,
        rates: jnp.ndarray,
        u_flow: jnp.ndarray,
        f_modeled: jnp.ndarray,
        ctrl_pv_values: jnp.ndarray,
    ) -> jnp.ndarray:
        n_non_volume = self.c_size - 1
        n_reactor = self.n_reactor_states
        state_values = c[:n_non_volume]
        c_reactor = c[:n_reactor]
        V = c[self.volume_idx]

        # 1. Compute algebraic variables in topo order (already sorted at build).
        n_algebraic = len(self.name_modeled_algebraic)
        algebraic_arr = jnp.zeros(n_algebraic) if n_algebraic > 0 else jnp.zeros(0)
        for i, fn in enumerate(self.algebraic_funcs):
            args = jnp.concatenate([state_values, ctrl_pv_values, algebraic_arr, rates])
            algebraic_arr = algebraic_arr.at[i].set(fn(args))

        # 2. Compute biological derivatives per dynamic state.
        full_args = jnp.concatenate([state_values, ctrl_pv_values, algebraic_arr, rates])
        biol_dc_list = [fn(full_args) for fn in self.derivative_funcs]
        biol_dc = jnp.stack(biol_dc_list) if biol_dc_list else jnp.zeros(0)

        # 3. Reactor states get feed/dilution on top; PV states are
        # biological-only (PV physical dynamics are out of scope here).
        feed_term, dV = _apply_feed_dilution(
            c_reactor,
            V,
            u_flow,
            f_modeled,
            self.Cin,
            self.Cin_modeled,
            self.u_flow_size,
            self.f_modeled_size,
            n_reactor,
        )

        dc_reactor = biol_dc[:n_reactor] + feed_term
        if self.n_pv_states > 0:
            dc_pv = biol_dc[n_reactor:n_non_volume]
        else:
            dc_pv = jnp.zeros(0)

        return jnp.append(jnp.append(dc_reactor, dc_pv), dV)


def build_rhs_ode(process: BioProcess) -> RhsOde:
    """Build a :class:`RhsOde` from a process whose ``biological_ode`` block is
    set (auto-generated in :meth:`BioProcess.__post_init__` when not
    user-supplied). Raises :class:`ValueError` if the block is missing or
    fails validation.
    """
    import sympy

    bo = process.biological_ode
    if bo is None:
        raise ValueError(
            "build_rhs_ode requires process.biological_ode to be set."
        )

    meta = _build_process_metadata(process)

    reactor_state_names = meta.reactor_component_state_names
    pv_state_names = meta.process_variable_state_names
    controlled_pv_names = tuple(
        name for name, pv in process.process_variables.items() if pv.is_controlled
    )

    state_derivative_names = tuple(reactor_state_names) + tuple(pv_state_names)
    algebraic_names_set = set(bo.algebraic.keys())
    rate_names_tuple = tuple(bo.rates.keys())
    allowed_names = (
        set(state_derivative_names)
        | set(controlled_pv_names)
        | algebraic_names_set
        | set(rate_names_tuple)
    )
    symbol_table = {n: sympy.Symbol(n) for n in allowed_names}

    # Parse all expressions; surface any error from validation for clarity.
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

    # Topo-sort algebraic; build ordered tuple of names.
    algebraic_order = _topo_sort_algebraic(algebraic_exprs)
    name_modeled_algebraic_ordered = tuple(algebraic_order)

    # Build the canonical args ordering used by every lambdified expression.
    # Concatenation order: state_derivative_names | controlled_pv_names
    #                      | name_modeled_algebraic_ordered | rate_names_tuple.
    args_order = (
        tuple(state_derivative_names)
        + tuple(controlled_pv_names)
        + name_modeled_algebraic_ordered
        + rate_names_tuple
    )

    algebraic_funcs = tuple(
        _lambdify_with_array_arg(algebraic_exprs[n], args_order)
        for n in name_modeled_algebraic_ordered
    )

    # Build per-state derivative callables in state order; missing state
    # entries should have been caught by validate_biological_ode but we
    # default to zero defensively.
    zero_expr = sympy.Integer(0)
    derivative_funcs = tuple(
        _lambdify_with_array_arg(derivative_exprs.get(n, zero_expr), args_order)
        for n in state_derivative_names
    )

    return RhsOde(
        c_size=meta.c_size,
        rate_size=len(rate_names_tuple),
        r_size=meta.r_size,
        u_flow_size=meta.u_flow_size,
        f_modeled_size=meta.f_modeled_size,
        output_size=meta.output_size,
        n_reactor_states=meta.n_reactor_states,
        n_pv_states=meta.n_pv_states,
        n_controlled_pv=len(controlled_pv_names),
        reactor_indices=meta.reactor_indices,
        pv_indices=meta.pv_indices,
        volume_idx=meta.volume_idx,
        static_pv_indices=meta.static_pv_indices,
        reactor_component_state_names=meta.reactor_component_state_names,
        process_variable_state_names=meta.process_variable_state_names,
        controlled_pv_names=controlled_pv_names,
        flow_names=meta.flow_names,
        modeled_flow_names=meta.modeled_flow_names,
        name_modeled_algebraic=name_modeled_algebraic_ordered,
        rate_names=rate_names_tuple,
        algebraic_funcs=algebraic_funcs,
        derivative_funcs=derivative_funcs,
        Cin=meta.Cin,
        Cin_modeled=meta.Cin_modeled,
    )


def build_algebraic_func(
    process: BioProcess,
) -> Callable[[jnp.ndarray, jnp.ndarray, jnp.ndarray], Dict[str, jnp.ndarray]]:
    """Build a callable that evaluates the ``biological_ode.algebraic``
    variables given concrete state, controlled-PV, and rate arrays.

    Returns ``f(state_values, ctrl_pv_values, rates) -> {name: scalar}``.
    Useful to expose algebraic quantities like ``X_active`` as observables for
    plotting or loss computation. Raises ``ValueError`` if the process has no
    ``biological_ode`` block.
    """
    bo = process.biological_ode
    if bo is None:
        raise ValueError(
            "build_algebraic_func requires process.biological_ode to be set."
        )
    rhs_ode = build_rhs_ode(process)
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
# Factory functions
# ---------------------------------------------------------------------------


def get_control_splines(process: BioProcess) -> ControlSplines:
    """Build a :class:`ControlSplines` module from a :class:`BioProcess`.

    Controlled signals are collected in a deterministic order:

    1. Continuous controlled volume changes (``is_controlled=True``,
       ``is_continuous=True``), in insertion order.
    2. Controlled process variables (``is_controlled=True``), in insertion order.

    Discrete volume changes (``is_continuous=False``) are **not** included in
    the returned module.

    Parameters
    ----------
    process:
        A :class:`~bp_format.BioProcess` instance.

    Returns
    -------
    ControlSplines
        An ``eqx.Module`` whose ``__call__(t)`` returns a 1-D JAX array of
        all controlled-signal values at time *t*.
    """
    control_names: List[str] = []
    flow_indices: List[int] = []
    ctrl_indices: List[int] = []
    splines: List[interpax.CubicSpline] = []
    is_derivative_list: List[bool] = []

    idx = 0

    # 1) Continuous controlled volume changes → flow rates (spline derivative)
    for vc_name, vc in process.volume.volume_changes.items():
        if not (vc.is_controlled and vc.is_continuous):
            continue
        sp = _timeseries_to_interpax_spline(vc.values)
        control_names.append(vc_name)
        flow_indices.append(idx)
        splines.append(sp)
        is_derivative_list.append(True)
        idx += 1

    # 2) Controlled process variables → direct spline value
    t_start = float(process.time_axis.start)
    t_end = float(process.time_axis.end)
    for pv_name, pv in process.process_variables.items():
        if not pv.is_controlled:
            continue
        sp = _value_to_interpax_spline(pv.values, t_start=t_start, t_end=t_end)
        control_names.append(pv_name)
        ctrl_indices.append(idx)
        splines.append(sp)
        is_derivative_list.append(False)
        idx += 1

    # Batch all splines into a single PPoly for vectorized evaluation
    if splines:
        batched = _batch_splines(splines, t_start, t_end)
    else:
        # Dummy 1-interval PPoly for empty case
        batched = interpax.PPoly.construct_fast(
            jnp.zeros((4, 1, 0)), jnp.array([0.0, 1.0]), extrapolate=True
        )
    deriv_mask = jnp.array(is_derivative_list, dtype=bool)

    return ControlSplines(
        control_names=tuple(control_names),
        flow_indices=tuple(flow_indices),
        ctrl_indices=tuple(ctrl_indices),
        _batched=batched,
        _deriv_mask=deriv_mask,
        _n_controls=len(splines),
    )


def get_rhs_ode(process: BioProcess) -> RhsOde:
    """Build the :class:`RhsOde` module for *process*.

    ``process.biological_ode`` must be set (auto-generated in
    :meth:`BioProcess.__post_init__` when not user-supplied).
    """
    return build_rhs_ode(process)


@_py_dataclass(frozen=True)
class _ProcessMetadata:
    """Static, RHS-independent process geometry derived from a :class:`BioProcess`.

    Single source of truth for state/flow ordering, sizes, and feed
    composition matrices. Consumed by :func:`build_rhs_ode`.
    """

    reactor_component_state_names: Tuple[str, ...]
    process_variable_state_names: Tuple[str, ...]
    static_pv_indices: Tuple[int, ...]
    flow_names: Tuple[str, ...]
    modeled_flow_names: Tuple[str, ...]
    n_reactor_states: int
    n_pv_states: int
    c_size: int
    r_size: int
    u_flow_size: int
    f_modeled_size: int
    output_size: int
    reactor_indices: Tuple[int, ...]
    pv_indices: Tuple[int, ...]
    volume_idx: int
    Cin: jnp.ndarray
    Cin_modeled: jnp.ndarray


def _build_process_metadata(process: BioProcess) -> _ProcessMetadata:
    """Build :class:`_ProcessMetadata` for *process* (biomass-first ordering,
    feed validation, Cin matrices)."""
    # --- Reactor component ordering: biomass always at index 0 ---
    all_component_names = list(process.reactor_medium.components.keys())
    biomass_name: str = ""
    for name in all_component_names:
        if name.strip().lower() == "biomass":
            biomass_name = name
            break
    if not biomass_name:
        raise ValueError(
            "No 'biomass' component found in process.reactor_medium.components. "
            f"Available components: {all_component_names}"
        )
    other_names = [n for n in all_component_names if n != biomass_name]
    reactor_component_state_names: Tuple[str, ...] = (biomass_name,) + tuple(
        other_names
    )
    n_reactor = len(reactor_component_state_names)

    process_variable_state_names: Tuple[str, ...] = tuple(
        pv_name
        for pv_name, pv in process.process_variables.items()
        if not pv.is_controlled
    )
    static_pv_indices: List[int] = []
    for i, pv_name in enumerate(process_variable_state_names):
        if isinstance(process.process_variables[pv_name].values, StaticVariable):
            static_pv_indices.append(i)
    n_pv = len(process_variable_state_names)
    n_non_volume = n_reactor + n_pv
    reactor_name_to_idx = {
        name: i for i, name in enumerate(reactor_component_state_names)
    }

    # Strict check: feed components must be known reactor components.
    for vc_name, vc in process.volume.volume_changes.items():
        if not isinstance(vc, FeedVolumeChange):
            continue
        if vc.feed_medium is None:
            raise ValueError(
                "FeedVolumeChange must define feed_medium in get_rhs_ode strict "
                f"validation. Missing for volume change '{vc_name}'."
            )
        unknown = [
            name
            for name in vc.feed_medium.components.keys()
            if name not in reactor_name_to_idx
        ]
        if unknown:
            raise ValueError(
                "Unknown feed component(s) in volume change "
                f"'{vc_name}': {unknown}. "
                "All feed components must exist in "
                "process.reactor_medium.components."
            )

    def _build_cin(vc_names):
        n = len(vc_names)
        Cin = jnp.zeros((n, n_reactor), dtype=float)
        for k, vc_name in enumerate(vc_names):
            vc = process.volume.volume_changes[vc_name]
            if not isinstance(vc, FeedVolumeChange):
                continue
            if vc.feed_medium is None:
                raise ValueError(
                    "FeedVolumeChange must define feed_medium for Cin construction. "
                    f"Missing for volume change '{vc_name}'."
                )
            feed = vc.feed_medium
            for j, sp_name in enumerate(reactor_component_state_names):
                if sp_name not in feed.components:
                    continue
                conc = feed.components[sp_name].concentration
                if isinstance(conc, StaticVariable):
                    Cin = Cin.at[k, j].set(float(conc.value))
                else:
                    raise NotImplementedError(
                        "TimeSeries feed concentrations are not yet supported "
                        "in get_rhs_ode. Found TimeSeries for species "
                        f"{sp_name!r} in feed {feed.name!r} of volume change "
                        f"{vc_name!r}."
                    )
        return Cin

    flow_names: List[str] = [
        vc_name
        for vc_name, vc in process.volume.volume_changes.items()
        if vc.is_controlled and vc.is_continuous
    ]
    modeled_flow_names: List[str] = [
        vc_name
        for vc_name, vc in process.volume.volume_changes.items()
        if (not vc.is_controlled) and vc.is_continuous
    ]

    Cin = _build_cin(flow_names)
    Cin_modeled = _build_cin(modeled_flow_names)

    return _ProcessMetadata(
        reactor_component_state_names=reactor_component_state_names,
        process_variable_state_names=process_variable_state_names,
        static_pv_indices=tuple(static_pv_indices),
        flow_names=tuple(flow_names),
        modeled_flow_names=tuple(modeled_flow_names),
        n_reactor_states=n_reactor,
        n_pv_states=n_pv,
        c_size=n_non_volume + 1,
        r_size=n_non_volume,
        u_flow_size=len(flow_names),
        f_modeled_size=len(modeled_flow_names),
        output_size=n_non_volume + 1,
        reactor_indices=tuple(range(n_reactor)),
        pv_indices=tuple(range(n_reactor, n_non_volume)),
        volume_idx=n_non_volume,
        Cin=Cin,
        Cin_modeled=Cin_modeled,
    )


# ---------------------------------------------------------------------------
# Discrete event handling
# ---------------------------------------------------------------------------


def extract_discrete_events(
    process: BioProcess,
    rhs_ode: RhsOde,
) -> List[Dict[str, Any]]:
    """Extract discrete events (sampling, bolus feeds) from a BioProcess.

    Parameters
    ----------
    process:
        A :class:`~bp_format.BioProcess` instance.
    rhs_ode:
        A :class:`RhsOde` module (used to align ``Cin`` with species ordering).

    Returns
    -------
    list[dict]
        Sorted list of event dicts, each with keys:
        - ``t`` (float): event time
        - ``kind`` (str): ``'sample'`` or ``'bolus_feed'``
        - ``dV`` (float): signed volume change (positive = add, negative = remove)
        - ``Cin`` (jnp.ndarray | None): feed composition aligned with
          ``rhs_ode.reactor_component_state_names`` (None for sampling)
        - ``source`` (str): name of the originating VolumeChange
    """
    events: List[Dict[str, Any]] = []
    n_reactor = rhs_ode.n_reactor_states

    for vc_name, vc in process.volume.volume_changes.items():
        if vc.is_continuous:
            continue

        tp = jnp.asarray(vc.values.times, dtype=float)
        vv = jnp.asarray(vc.values.values, dtype=float)

        for t_event, dV_event in zip(tp, vv):
            if abs(dV_event) < 1e-15:
                continue

            if dV_event > 0 and isinstance(vc, FeedVolumeChange):
                Cin_event = jnp.zeros(n_reactor)
                if vc.feed_medium is not None:
                    for j, sp_name in enumerate(rhs_ode.reactor_component_state_names):
                        if sp_name in vc.feed_medium.components:
                            conc = vc.feed_medium.components[sp_name].concentration
                            if isinstance(conc, StaticVariable):
                                Cin_event = Cin_event.at[j].set(float(conc.value))
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

    # At a single timestamp we only allow:
    # - one sampling event, and/or
    # - one bolus feed event.
    # If both are present, sampling must be applied first (sorting above).
    per_time_counts: Dict[float, Dict[str, int]] = {}
    for ev in events:
        t = float(ev["t"])
        kind = str(ev["kind"])
        if t not in per_time_counts:
            per_time_counts[t] = {"sample": 0, "bolus_feed": 0}
        per_time_counts[t][kind] = per_time_counts[t][kind] + 1

    duplicate_kinds = []
    for t, counts in per_time_counts.items():
        if counts["sample"] > 1:
            duplicate_kinds.append((t, "sample", counts["sample"]))
        if counts["bolus_feed"] > 1:
            duplicate_kinds.append((t, "bolus_feed", counts["bolus_feed"]))
    if duplicate_kinds:
        details = ", ".join(
            [f"t={t}: {kind} x{count}" for t, kind, count in duplicate_kinds]
        )
        raise ValueError(
            "At most one discrete event per kind is allowed at a given time "
            f"(allowed: one sample and one bolus). Found duplicates: {details}."
        )
    return events


def build_state_splines(
    process: BioProcess,
    rhs_ode: "RhsOde",
) -> Dict[str, Any]:
    """Build state splines from stored TimeSeries spline state.

    Pseudobatch-transformed reactor components are identified through the
    process-level ``pseudobatch_transform`` bundle and converted into a
    real-space backtransform spline. Other reactor-component and
    process-variable states are converted directly from their TimeSeries or
    StaticVariable carrier.

    Parameters
    ----------
    process:
        A :class:`~bp_format.BioProcess` instance.
    rhs_ode:
        A :class:`RhsOde` module (provides species ordering).

    Returns
    -------
    dict
        Mapping non-volume state name -> callable spline.
    """
    state_splines: Dict[str, Any] = {}
    pseudobatch_transform = _validate_process_pseudobatch_transform(process, rhs_ode)

    for sp_name in rhs_ode.reactor_component_state_names:
        comp = process.reactor_medium.components[sp_name]
        concentration = comp.concentration
        if (
            pseudobatch_transform is not None
            and sp_name in pseudobatch_transform.species
        ):
            state_splines[sp_name] = build_backtransform_spline(
                pseudobatch_transform,
                sp_name,
            )
        else:
            _reject_orphan_pseudobatch_metadata(concentration, sp_name)
            state_splines[sp_name] = _value_to_interpax_spline(
                concentration,
                t_start=float(process.time_axis.start),
                t_end=float(process.time_axis.end),
            )

    for pv_name in rhs_ode.process_variable_state_names:
        pv = process.process_variables[pv_name]
        state_splines[pv_name] = _value_to_interpax_spline(
            pv.values,
            t_start=float(process.time_axis.start),
            t_end=float(process.time_axis.end),
        )

    return state_splines


def build_conc_splines(
    process: BioProcess,
    rhs_ode: "RhsOde",
) -> Dict[str, Any]:
    """Compatibility alias for :func:`build_state_splines`."""
    return build_state_splines(process, rhs_ode)


def _resolve_state_splines(
    *,
    state_splines: Optional[Dict[str, Any]],
    conc_splines: Optional[Dict[str, Any]] = None,
) -> Optional[Dict[str, Any]]:
    """Resolve state spline arguments while supporting the legacy name."""
    if state_splines is not None and conc_splines is not None:
        raise ValueError("Use either state_splines or conc_splines, not both.")
    if state_splines is not None:
        return state_splines
    return conc_splines


# ---------------------------------------------------------------------------
# Full hybrid ODE integration
# ---------------------------------------------------------------------------


def _require_rates_func(rates_func: Optional[Callable]) -> Callable:
    """Require the runtime rates callback."""
    if rates_func is None:
        raise ValueError(
            "rates_func is required and must have signature "
            "rates_func(t, state, controls) -> rates_array, "
            "shape (rhs_ode.rate_size,) aligned with rhs_ode.rate_names."
        )
    return rates_func


def _build_segment_rhs(
    rhs_ode,
    ctrl,
    rates_func,
    batched_mod,
):
    """Build the ODE right-hand side function for a segment.

    The RHS reads everything from the integrator's current state and from
    user-supplied ``rates_func(t, state, controls) -> rates_array``.

    Parameters
    ----------
    batched_mod : interpax.PPoly or None
        Batched PPoly for modeled (uncontrolled) cumulative volume splines.
        Evaluate with ``batched_mod(t, nu=1)`` to get flow rates.
    """
    flow_idx = jnp.array(list(ctrl.flow_indices))
    ctrl_idx = jnp.array(list(ctrl.ctrl_indices))

    def rhs(t, state, args):
        u = ctrl(t)
        u_flow = u[flow_idx] if len(flow_idx) > 0 else jnp.zeros(rhs_ode.u_flow_size)
        rates = rates_func(t, state, u)
        f_mod = (
            batched_mod(t, nu=1)
            if batched_mod is not None
            else jnp.zeros(rhs_ode.f_modeled_size)
        )
        ctrl_pv_values = u[ctrl_idx] if len(ctrl_idx) > 0 else jnp.zeros(0)
        return rhs_ode(state, rates, u_flow, f_mod, ctrl_pv_values)

    return rhs


def _validate_rates_output_shapes(
    rhs_ode: "RhsOde",
    rates_func: Callable,
    *,
    t: float,
    state: jnp.ndarray,
    controls: jnp.ndarray,
) -> None:
    """Validate ``rates_func`` output shape before JIT solve."""
    rates_probe = jnp.asarray(rates_func(float(t), state, controls), dtype=float)
    if rates_probe.shape != (rhs_ode.rate_size,):
        raise ValueError(
            f"rates_func must return shape ({rhs_ode.rate_size},), "
            f"got {rates_probe.shape}."
        )


def _compute_scale_factors(process: BioProcess, rhs_ode: "RhsOde") -> jnp.ndarray:
    """Compute non-volume state scale factors for numerical conditioning."""
    scales = jnp.ones(rhs_ode.r_size)
    for i, sp_name in enumerate(rhs_ode.reactor_component_state_names):
        vals = jnp.asarray(
            process.reactor_medium.components[sp_name].concentration.values, dtype=float
        )
        s = float(jnp.max(jnp.abs(vals)))
        if s > 1.0:
            scales = scales.at[i].set(s)
    offset = rhs_ode.n_reactor_states
    for j, pv_name in enumerate(rhs_ode.process_variable_state_names):
        pv = process.process_variables[pv_name]
        if isinstance(pv.values, TimeSeries):
            vals = jnp.asarray(pv.values.values, dtype=float)
            s = float(jnp.max(jnp.abs(vals)))
        else:
            s = abs(float(pv.values.value))
        if s > 1.0:
            scales = scales.at[offset + j].set(s)
    return scales


def _piecewise_linear_value_and_slope(
    t: jnp.ndarray,
    knot_t: jnp.ndarray,
    knot_v: jnp.ndarray,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Evaluate piecewise-linear value and slope at scalar time t."""
    knot_t = jnp.asarray(knot_t, dtype=float)
    knot_v = jnp.asarray(knot_v, dtype=float)

    if len(knot_t) < 2:
        return knot_v[0], jnp.array(0.0)

    t0 = knot_t[0]
    tN = knot_t[-1]
    t_clip = jnp.clip(jnp.asarray(t, dtype=float), t0, tN)

    right = jnp.searchsorted(knot_t, t_clip, side="right")
    left_idx = jnp.clip(right - 1, 0, len(knot_t) - 2)
    right_idx = left_idx + 1

    x0 = knot_t[left_idx]
    x1 = knot_t[right_idx]
    y0 = knot_v[left_idx]
    y1 = knot_v[right_idx]

    denom = jnp.maximum(x1 - x0, 1e-12)
    alpha = (t_clip - x0) / denom
    val = y0 + alpha * (y1 - y0)
    slope = (y1 - y0) / denom

    outside = (t < t0) | (t > tN)
    slope = jnp.where(outside, 0.0, slope)
    return val, slope


def _snap_times_to_discrete_events(
    times: jnp.ndarray,
    event_times: jnp.ndarray | None,
) -> jnp.ndarray:
    """Snap near-exact event-grid values back to exact event timestamps."""
    times = jnp.asarray(times, dtype=float)
    if event_times is None or int(event_times.shape[0]) == 0:
        return times
    events = jnp.asarray(event_times, dtype=float)
    dist = jnp.abs(times[:, None] - events[None, :])
    idx = jnp.argmin(dist, axis=1)
    nearest = events[idx]
    scale = jnp.maximum(1.0, jnp.maximum(jnp.abs(times), jnp.abs(nearest)))
    tol = 16.0 * jnp.finfo(times.dtype).eps * scale
    return jnp.where(dist[jnp.arange(times.shape[0]), idx] <= tol, nearest, times)


def _is_pseudobatch_carrier(value: Any) -> bool:
    """Return whether a TimeSeries carries lightweight pseudobatch metadata."""
    if not isinstance(value, TimeSeries) or not isinstance(value.metadata, dict):
        return False
    transform = value.metadata.get("transform")
    return isinstance(transform, dict) and transform.get("name") == "pseudo_batch"


def _reject_orphan_pseudobatch_metadata(value: Any, species_name: str) -> None:
    """Fail when c* metadata exists without process-level transform bundle."""
    if _is_pseudobatch_carrier(value):
        raise ValueError(
            f"Species {species_name!r} carries pseudobatch c* metadata but is not "
            "present in process.pseudobatch_transform."
        )


def _timeseries_samples_match(left: TimeSeries, right: TimeSeries) -> bool:
    """Compare TimeSeries sample anchors used by mechanistic initial states."""
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


def _validate_process_pseudobatch_transform(
    process: BioProcess,
    rhs_ode: "RhsOde",
):
    """Validate process-level pseudobatch bundle before runtime use."""
    transform = getattr(process, "pseudobatch_transform", None)
    if transform is None:
        for sp_name in rhs_ode.reactor_component_state_names:
            comp = process.reactor_medium.components[sp_name]
            _reject_orphan_pseudobatch_metadata(comp.concentration, sp_name)
        return None

    for species_key, species_transform in transform.species.items():
        if species_transform.species != species_key:
            raise ValueError(
                f"Pseudobatch species key {species_key!r} does not match stored "
                f"species {species_transform.species!r}."
            )
        if species_key not in process.reactor_medium.components:
            raise ValueError(
                f"Pseudobatch species {species_key!r} is not a reactor component."
            )
        concentration = process.reactor_medium.components[species_key].concentration
        if not isinstance(concentration, TimeSeries):
            raise TypeError(
                f"Pseudobatch species {species_key!r} concentration must be a "
                "TimeSeries c* carrier."
            )
        if not _timeseries_samples_match(
            concentration,
            species_transform.c_star_ts,
        ):
            raise ValueError(
                f"Pseudobatch species {species_key!r} reactor concentration does "
                "not match transform c_star_ts. Assign the bundle c_star_ts to "
                "the reactor component before mechanistic runtime use."
            )

    return transform


def _build_pseudobatch_transforms(
    process: BioProcess,
    rhs_ode: "RhsOde",
) -> List[Dict[str, Any]]:
    """Build per-species pseudo-batch transform descriptors.

    Each descriptor supports:
      c* = adf(t) * c - fc(t)
      c  = (c* + fc(t)) / adf(t)
      dc*/dt = adf * dc/dt + d(adf)/dt * c - d(fc)/dt
    """
    t_start = float(process.time_axis.start)
    t_end = float(process.time_axis.end)
    pseudobatch_transform = _validate_process_pseudobatch_transform(process, rhs_ode)
    transforms: List[Dict[str, Any]] = []
    for sp_name in rhs_ode.reactor_component_state_names:
        if (
            pseudobatch_transform is None
            or sp_name not in pseudobatch_transform.species
        ):
            adf_ts = _constant_timeseries(1.0, t_start, t_end)
            fc_ts = _constant_timeseries(0.0, t_start, t_end)
            transforms.append(
                {
                    "kind": "identity",
                    "adf_ts": adf_ts,
                    "dadf_ts": adf_ts.deriv(),
                    "feed_corr_ts": fc_ts,
                    "dfc_ts": fc_ts.deriv(),
                }
            )
            continue

        species_transform = pseudobatch_transform.species[sp_name]
        adf_ts = pseudobatch_transform.adf_ts
        feed_corr_ts = species_transform.feed_corr_ts
        fc_metadata = (
            feed_corr_ts.metadata if isinstance(feed_corr_ts.metadata, dict) else {}
        )
        fc_interp = str(fc_metadata.get("interp", "piecewise_polynomial"))
        if fc_interp in {"linear", "linear_plus_step"}:
            raise ValueError(
                f"Legacy feed_corr_interp={fc_interp!r} unsupported for "
                "pseudobatch mechanistic path; regenerate transformed "
                "TimeSeries payloads."
            )
        if fc_interp not in {
            "cubic",
            "piecewise_polynomial",
            "piecewise_constant",
        }:
            raise ValueError(
                f"Unknown feed_corr_interp={fc_interp!r}; expected 'cubic', "
                "'piecewise_polynomial', or 'piecewise_constant'."
            )
        if adf_ts.breaks is None or adf_ts.coeffs is None:
            raise ValueError("ADF transform TimeSeries must provide spline state.")
        if feed_corr_ts.breaks is None or feed_corr_ts.coeffs is None:
            raise ValueError(
                "Feed-correction transform TimeSeries must provide spline state."
            )

        transforms.append(
            {
                "kind": "pb",
                "adf_ts": adf_ts,
                "dadf_ts": adf_ts.deriv(),
                "feed_corr_ts": feed_corr_ts,
                "dfc_ts": feed_corr_ts.deriv(),
                "fc_interp": fc_interp,
            }
        )

    for _ in rhs_ode.process_variable_state_names:
        adf_ts = _constant_timeseries(1.0, t_start, t_end)
        fc_ts = _constant_timeseries(0.0, t_start, t_end)
        transforms.append(
            {
                "kind": "identity",
                "adf_ts": adf_ts,
                "dadf_ts": adf_ts.deriv(),
                "feed_corr_ts": fc_ts,
                "dfc_ts": fc_ts.deriv(),
            }
        )

    return transforms


def integrate_process(
    process: BioProcess,
    ctrl: ControlSplines,
    rhs_ode: RhsOde,
    rates_func: Callable,
    t_eval: jnp.ndarray,
    *,
    rtol: float = 1e-4,
    atol: float = 1e-6,
    max_steps: int = 16384,
) -> Dict[str, Any]:
    """Full hybrid ODE integration with discrete event handling.

    Integrates the ODE segment-by-segment using ``jax.lax.scan`` over
    segments separated by discrete events.  The entire scan is JIT-compiled
    once via ``eqx.filter_jit``; subsequent calls reuse the compiled code.

    Between events the ODE is solved with ``diffrax.Tsit5``.  At event
    boundaries, discrete state updates (sampling, bolus feeds) are applied.

    Concentrations that span large magnitudes (e.g. cell counts at 1e9)
    are automatically normalized for numerical conditioning, then
    un-normalized before returning.

    Parameters
    ----------
    process:
        A :class:`~bp_format.BioProcess` instance.
    ctrl:
        :class:`ControlSplines` module.
    rhs_ode:
        :class:`RhsOde` module.
    rates_func:
        Callable ``rates_func(t, state, controls) -> jnp.ndarray`` of shape
        ``(rhs_ode.rate_size,)`` aligned with ``rhs_ode.rate_names``
        (= the insertion order of ``process.biological_ode.rates``).
    t_eval:
        1-D array of time points at which to record the solution.
    rtol, atol:
        Relative and absolute tolerances for the ODE solver.
    max_steps:
        Maximum number of ODE solver steps per segment.

    Notes
    -----
    Process-variable states are treated as additive-only in this integration
    path: ``dc_pv/dt = r_pv`` (with configured static PV indices clamped to
    zero). They are not subjected to reactor feed/dilution terms.
    For discrete events, this segmented API is left-continuous at explicit
    boundary samples (a sample exactly at an event time is reported pre-event).
    The post-event state is visible from the next output point after `t_b`.

    Returns
    -------
    dict
        ``{'t': jnp.ndarray, 'c': jnp.ndarray, 'V': jnp.ndarray}``
        where ``c`` has shape ``(len(t_eval), n_non_volume_states)`` and
        ``V`` has shape ``(len(t_eval),)``.
    """
    t_eval = jnp.asarray(t_eval, dtype=float)
    t_start = float(process.time_axis.start)
    t_end = float(process.time_axis.end)
    n_reactor = rhs_ode.n_reactor_states
    n_non_volume = rhs_ode.r_size
    rates_func = _require_rates_func(rates_func)

    # Per-state scale factors for numerical conditioning
    scales = _compute_scale_factors(process, rhs_ode)
    scale_vec = jnp.array(scales)  # (n_non_volume,)
    state_scale = jnp.append(scale_vec, 1.0)  # [scales..., 1.0]

    # Build modeled flow splines (batched)
    cum_splines_mod_list = []
    for fn in rhs_ode.modeled_flow_names:
        vc = process.volume.volume_changes[fn]
        sp = _timeseries_to_interpax_spline(vc.values)
        cum_splines_mod_list.append(sp)
    batched_mod = (
        _batch_splines(cum_splines_mod_list, t_start, t_end)
        if cum_splines_mod_list
        else None
    )

    # Extract discrete events and build segment boundaries
    events = extract_discrete_events(process, rhs_ode)
    event_times = sorted(set(ev["t"] for ev in events))
    event_times_in_range = [t for t in event_times if t_start < t < t_end]
    boundaries = [t_start] + event_times_in_range + [t_end]
    n_seg = len(boundaries) - 1

    # Build event lookup
    event_lookup: Dict[float, List[Dict]] = {}
    for ev in events:
        event_lookup.setdefault(ev["t"], []).append(ev)

    # Initial state (in original coordinates)
    c0_reactor = jnp.array(
        [
            float(
                jnp.asarray(
                    process.reactor_medium.components[s].concentration.values[0]
                )
            )
            for s in rhs_ode.reactor_component_state_names
        ]
    )
    c0_pv = jnp.array(
        [
            float(jnp.asarray(pv.values.values[0]))
            if isinstance(pv.values, TimeSeries)
            else float(pv.values.value)
            for pv_name, pv in process.process_variables.items()
            if pv_name in rhs_ode.process_variable_state_names
        ],
        dtype=float,
    )
    c0 = jnp.concatenate([c0_reactor, c0_pv])
    c0 = c0.at[:n_reactor].set(jnp.maximum(c0[:n_reactor], 0.0))
    V0 = _require_reactor_volume_scalar(
        process.volume.initial_volume,
        context="initial reactor volume",
    )

    # Save pre-event initial state for left-continuous output at t_start.
    c0_pre_event = c0
    V0_pre_event = V0

    # Apply events at t_start directly to the initial state so the first
    # segment begins from the correct post-event initial condition.
    for ev in event_lookup.get(t_start, []):
        dV = float(ev["dV"])
        V_new = _require_reactor_volume_scalar(
            V0 + dV,
            context="initial discrete event",
        )
        if ev["kind"] == "bolus_feed" and ev["Cin"] is not None:
            Cin = jnp.asarray(ev["Cin"], dtype=float)
            c0_reactor = (c0[:n_reactor] * V0 + Cin * dV) / V_new
            c0_reactor = jnp.maximum(c0_reactor, 0.0)
            c0 = c0.at[:n_reactor].set(c0_reactor)
        V0 = V_new

    state0_probe = jnp.append(c0, V0)
    controls0_probe = ctrl(jnp.array(t_start))
    _validate_rates_output_shapes(
        rhs_ode,
        rates_func,
        t=t_start,
        state=state0_probe,
        controls=controls0_probe,
    )

    # Build RHS in original coordinates, then wrap for normalized state
    rhs_original = _build_segment_rhs(
        rhs_ode,
        ctrl,
        rates_func,
        batched_mod,
    )

    def rhs_normalized(t, state_norm, args):
        state_orig = state_norm * state_scale
        dc_dt_orig = rhs_original(t, state_orig, args)
        return dc_dt_orig / state_scale

    term = diffrax.ODETerm(rhs_normalized)
    solver = diffrax.Tsit5()
    stepsize_controller = diffrax.PIDController(rtol=rtol, atol=atol)

    # Normalized initial state
    state_norm_init = jnp.append(c0, V0) / state_scale

    # ---------------------------------------------------------------
    # Pre-build padded segment time arrays
    # ---------------------------------------------------------------
    seg_t_arrays = []
    for seg_idx in range(n_seg):
        t_lo = boundaries[seg_idx]
        t_hi = boundaries[seg_idx + 1]
        mask = (t_eval >= t_lo) & (t_eval <= t_hi)
        t_seg = t_eval[mask]
        if len(t_seg) == 0:
            t_seg = jnp.array([t_lo, t_hi])
        else:
            if t_seg[0] > t_lo + _SEGMENT_BOUNDARY_TOL:
                t_seg = jnp.concatenate([jnp.array([t_lo]), t_seg])
            if t_seg[-1] < t_hi - _SEGMENT_BOUNDARY_TOL:
                t_seg = jnp.concatenate([t_seg, jnp.array([t_hi])])
        seg_t_arrays.append(t_seg)

    max_ts_len = max(len(ts) for ts in seg_t_arrays)

    seg_t_valid_len = []
    seg_t_padded_list = []
    for ts in seg_t_arrays:
        n_valid = len(ts)
        seg_t_valid_len.append(n_valid)
        if n_valid < max_ts_len:
            pad = jnp.full(max_ts_len - n_valid, ts[-1])
            seg_t_padded_list.append(jnp.concatenate([ts, pad]))
        else:
            seg_t_padded_list.append(ts)

    # Stack into JAX arrays for lax.scan
    seg_t_lo = jnp.array([boundaries[i] for i in range(n_seg)])
    seg_t_hi = jnp.array([boundaries[i + 1] for i in range(n_seg)])
    seg_ts_padded = jnp.stack(seg_t_padded_list)  # (n_seg, max_ts_len)
    seg_n_valid = jnp.array(seg_t_valid_len)  # (n_seg,)

    # ---------------------------------------------------------------
    # Pre-build padded event arrays for each segment boundary
    # ---------------------------------------------------------------
    max_ev = max(
        (len(event_lookup.get(boundaries[i + 1], [])) for i in range(n_seg)),
        default=0,
    )
    max_ev = max(max_ev, 1)  # at least 1 slot for padding

    ev_n_arr = jnp.zeros(n_seg, dtype=jnp.int32)
    ev_dV_arr = jnp.zeros((n_seg, max_ev))
    ev_is_bolus_arr = jnp.zeros((n_seg, max_ev), dtype=bool)
    ev_Cin_arr = jnp.zeros((n_seg, max_ev, n_reactor))

    for i in range(n_seg):
        evs = event_lookup.get(boundaries[i + 1], [])
        ev_n_arr = ev_n_arr.at[i].set(len(evs))
        for j, ev in enumerate(evs):
            ev_dV_arr = ev_dV_arr.at[i, j].set(ev["dV"])
            if ev["kind"] == "bolus_feed" and ev["Cin"] is not None:
                ev_is_bolus_arr = ev_is_bolus_arr.at[i, j].set(True)
                ev_Cin_arr = ev_Cin_arr.at[i, j].set(jnp.asarray(ev["Cin"]))
    # Last segment has no events (ev_n_arr[-1] stays 0)

    # ---------------------------------------------------------------
    # JIT-compiled scan over segments
    # ---------------------------------------------------------------
    @eqx.filter_jit
    def _run_scan(
        y0_norm,
        s_t_lo,
        s_t_hi,
        s_ts,
        s_n_valid,
        s_ev_n,
        s_ev_dV,
        s_ev_is_bolus,
        s_ev_Cin,
    ):
        def _scan_body(carry, x):
            state_n = carry
            t_lo, t_hi, ts, n_val, n_ev, e_dV, e_bolus, e_Cin = x

            dt0 = jnp.minimum(0.1, (t_hi - t_lo) / 10.0)

            sol = diffrax.diffeqsolve(
                term,
                solver,
                t0=t_lo,
                t1=t_hi,
                dt0=dt0,
                y0=state_n,
                saveat=diffrax.SaveAt(ts=ts),
                stepsize_controller=stepsize_controller,
                max_steps=max_steps,
            )
            ys_norm = sol.ys  # (max_ts_len, state_dim)

            # Apply discrete events at boundary (in original coords)
            state_orig = ys_norm[-1] * state_scale

            def _apply_event(state, j):
                dV = e_dV[j]
                is_bolus = e_bolus[j]
                Cin = e_Cin[j]
                c_reactor = state[:n_reactor]
                c_pv = (
                    state[n_reactor:n_non_volume]
                    if rhs_ode.n_pv_states > 0
                    else jnp.zeros(0, dtype=state.dtype)
                )
                V = state[rhs_ode.volume_idx]
                V = _require_reactor_volume_above_threshold(
                    V, context="pre-event integration volume"
                )
                V_new = _require_reactor_volume_above_threshold(
                    V + dV,
                    context="discrete event volume",
                )
                c_reactor_bolus = (c_reactor * V + Cin * dV) / V_new
                c_reactor_new = jnp.where(is_bolus, c_reactor_bolus, c_reactor)
                c_reactor_new = jnp.maximum(c_reactor_new, 0.0)
                new_state = jnp.append(
                    jnp.append(c_reactor_new, c_pv),
                    V_new,
                )
                active = j < n_ev
                return jnp.where(active, new_state, state)

            state_orig = jax.lax.fori_loop(
                0, max_ev, lambda j, s: _apply_event(s, j), state_orig
            )
            state_n_next = state_orig / state_scale

            n_steps = sol.stats["num_steps"]
            return state_n_next, (ys_norm, n_steps)

        xs = (s_t_lo, s_t_hi, s_ts, s_n_valid, s_ev_n, s_ev_dV, s_ev_is_bolus, s_ev_Cin)
        _, (all_ys, all_steps) = jax.lax.scan(_scan_body, y0_norm, xs)
        return all_ys, all_steps  # (n_seg, max_ts_len, state_dim), (n_seg,)

    all_ys_norm, all_steps = _run_scan(
        state_norm_init,
        seg_t_lo,
        seg_t_hi,
        seg_ts_padded,
        seg_n_valid,
        ev_n_arr,
        ev_dV_arr,
        ev_is_bolus_arr,
        ev_Cin_arr,
    )

    # ---------------------------------------------------------------
    # Post-process: un-normalize, extract valid points, concatenate
    # ---------------------------------------------------------------
    all_ys_orig = (
        all_ys_norm * state_scale[None, None, :]
    )  # (n_seg, max_ts_len, state_dim)

    # Left-continuous: segment 0's first point is at t_start, solved from
    # the post-event IC. If events existed at t_start, override it with the
    # pre-event state so the output at t_start is pre-event.
    if event_lookup.get(t_start, []):
        pre_event_state = jnp.append(c0_pre_event, jnp.array(V0_pre_event))
        all_ys_orig = all_ys_orig.at[0, 0, :].set(pre_event_state)

    t_segments = []
    c_segments = []
    V_segments = []

    for seg_idx in range(n_seg):
        n_valid = seg_t_valid_len[seg_idx]
        ys_seg = all_ys_orig[seg_idx, :n_valid, :]

        c_seg = ys_seg[:, :n_non_volume]
        c_seg = c_seg.at[:, :n_reactor].set(jnp.maximum(c_seg[:, :n_reactor], 0.0))
        V_seg = _require_reactor_volume_above_threshold(
            ys_seg[:, rhs_ode.volume_idx],
            context="segmented integration output volume",
        )
        t_seg = seg_t_arrays[seg_idx]

        # Each segment starts at the previous boundary t_b. For non-first
        # segments, that first point carries the post-event initial condition
        # and duplicates the pre-event t_b already kept by the previous
        # segment, so drop it. Segment 0 keeps all its points.
        if seg_idx > 0:
            t_segments.append(t_seg[1:])
            c_segments.append(c_seg[1:])
            V_segments.append(V_seg[1:])
        else:
            t_segments.append(t_seg)
            c_segments.append(c_seg)
            V_segments.append(V_seg)

    t_out = jnp.concatenate(t_segments)
    c_out = jnp.vstack(c_segments)
    V_out = jnp.concatenate(V_segments)

    return {
        "t": t_out,
        "c": c_out,
        "V": V_out,
        "stats": {
            "num_steps": int(jnp.sum(all_steps)),
            "steps_per_segment": all_steps,
        },
    }
