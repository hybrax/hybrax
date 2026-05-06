"""
Mechanistic API for bp-format.

Provides JAX/Equinox-compatible modules for building continuous-time control
functions and ODE right-hand sides directly from a :class:`~bp_format.BioProcess`.

All modules are fully JAX-jittable via ``equinox.filter_jit``.  Spline
evaluation uses ``interpax.CubicSpline``, which is itself an
``equinox.Module`` and therefore a valid JAX pytree.

Public API
----------
get_control_splines(process) -> ControlSplines
    Returns an ``eqx.Module`` whose ``__call__(t)`` evaluates all controlled
    signals at time ``t``.  Continuous volume-change feeds are returned as
    **flow rates** (derivative of the cumulative-volume spline).

get_rhs_ode(process) -> RhsOde
    Returns an ``eqx.Module`` whose ``__call__(c, q, u_flow, f_modeled, r)``
    computes the ODE RHS ``dc/dt`` (including ``dV/dt``).
    Uncontrolled continuous volume changes (modeled feeds) are supported via
    the optional ``f_modeled`` argument.

extract_discrete_events(process, mb) -> list[dict]
    Extract discrete events (sampling, bolus feeds) from a BioProcess.

build_state_splines(process, mb) -> dict
    Build spline callables for all non-volume states.

build_q_func(process, ctrl, mb, state_splines) -> Callable
    Build an analytical, JIT-compilable q(t) callable from state splines.

build_rates_func(process, ctrl, mb, state_splines) -> Callable
    Build a ``rates_func(t, state, controls) -> (q, r)`` from state splines.

estimate_specific_rates(process, ctrl, mb, state_splines, t_eval) -> jnp.ndarray
    Estimate specific rates q(t) via ODE RHS inversion (convenience wrapper).

integrate_process(process, ctrl, mb, rates_func, t_eval) -> dict
    Full hybrid ODE integration with discrete event handling. Honest
    forward integration: state read directly from the integrator at every
    step.
integrate_process_pseudospace(process, ctrl, mb, rates_func, t_eval) -> dict
    Single-pass integration in pseudo-batch ``c*`` coordinates. Useful as
    a post-processing / spline-fitting helper. The state lives in
    spline-derived coordinates so the trajectory is biased toward the
    reference splines — do not use for honest forward prediction with
    externally-supplied q.

Rates Pipeline
--------------
The intended workflow is:

1. Build ``state_splines = build_state_splines(process, mb)``.
2. Build inversion-side rates with ``q_func = build_q_func(...)``.
3. Build integration-side callback with
   ``rates_func = build_rates_func(..., r_func=...)`` or provide your own
   callable with signature ``rates_func(t, state, controls) -> (q, r)``.
4. Integrate via ``integrate_process*``.

Default assumptions:

- All reactor-component states are treated as biologically driven and therefore
  represented in biomass-specific ``q``.
- ``r`` defaults depend on state block when ``r_func`` is not supplied:
  reactor-component entries are zero, process-variable entries come from PV
  spline derivatives.
- Reactor-component states are represented in both vectors: ``q`` has one entry
  per reactor-component state, and the first ``n_reactor_states`` entries of
  ``r`` align to those same reactor-component states.
- Process-variable states are represented only in ``r`` (tail entries after the
  reactor-component block).
- Process-variable states are additive-only in the ODE path:
  ``dc_pv/dt = r_pv`` (no dilution/feed term). Reactor-component states receive
  reaction + feed + ``r_reactor`` terms.

``q_state_indices`` and ``r_state_indices`` let callers split reactor-component
state dynamics between ``q`` and ``r`` during inversion. They are optional; if
omitted, all reactor-component states are ``q``-states.

Supported Runtime Scenarios
---------------------------
1. Data-driven default (no explicit physical rates):

   - Build spline-backed rates with ``build_rates_func(..., r_func=None)``.
   - Internally this uses :func:`build_q_func` for all reactor-component states.
   - Default ``r`` behavior is:
     - reactor-component block: zeros,
     - process-variable block: inferred from PV spline derivatives.
   - Integration then applies reaction + feed/dilution for reactor states, and
     additive PV rates from inferred derivatives.

2. Data-driven + explicit physical rates:

   - Build spline-backed rates with ``build_rates_func(..., r_func=...)``.
   - In this mode, callers must pass both ``q_state_indices`` and
     ``r_state_indices`` explicitly, so inversion knows where physical terms
     should be treated as ``r`` (including overlap subtraction when needed).

3. Fully custom runtime rates callback:

   - Provide your own ``rates_func(t, state, controls) -> (q, r)`` directly to
     ``integrate_process*``.
   - The integrator still handles feed/dilution, volume dynamics, and discrete
     events. Callers provide only the biological ``q`` and additive physical
     ``r`` terms.

Usage with JIT
--------------
Both modules are equinox Modules (JAX pytrees).  Use ``eqx.filter_jit``
to compile them::

    import equinox as eqx
    ctrl = get_control_splines(process)
    mb   = get_rhs_ode(process)

    u      = eqx.filter_jit(ctrl)(t)
    r      = jnp.zeros(mb.r_size)
    dc_dt  = eqx.filter_jit(mb)(c, q, u_flow, jnp.zeros(0), r)
    # With modeled flows (e.g. base feed):
    dc_dt  = eqx.filter_jit(mb)(c, q, u_flow, f_modeled, r)
"""

from __future__ import annotations

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

_DEFAULT_BATCH_KNOTS = 128
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


def _add_intracellular_to_biomass(
    reaction: jnp.ndarray,
    q: jnp.ndarray,
    X_active: jnp.ndarray,
    biomass_idx: int,
    intracellular_indices: tuple,
) -> jnp.ndarray:
    """Add intracellular q contributions to the measured-biomass derivative.

    ``X_measured = X_active + Σ P_intracellular``, so
    ``dX_meas/dt = dX_active/dt + Σ dP_intra/dt``. The per-state reaction term
    ``q[i] * X_active`` is the intracellular accumulation rate for index *i*;
    summing over intracellular indices and adding to the biomass entry keeps
    the measured-biomass equation consistent.
    """
    if not intracellular_indices:
        return reaction
    intra_arr = jnp.array(intracellular_indices, dtype=int)
    return reaction.at[biomass_idx].add(jnp.sum(q[intra_arr]) * X_active)


def _subtract_intracellular_from_biomass_q(
    q_all: jnp.ndarray,
    biomass_idx: int,
    intracellular_indices: tuple,
) -> jnp.ndarray:
    """Inverse of :func:`_add_intracellular_to_biomass` for q inversion.

    Inversion yields ``q_all[i] = (dc_i/dt - feed_i) / X_active``. For
    ``i == biomass_idx`` this is the *apparent* specific growth rate of
    measured biomass; subtracting the intracellular q values recovers the
    specific growth rate of active biomass.
    """
    if not intracellular_indices:
        return q_all
    intra_arr = jnp.array(intracellular_indices, dtype=int)
    return q_all.at[biomass_idx].add(-jnp.sum(q_all[intra_arr]))


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
# RhsOde module
# ---------------------------------------------------------------------------


class RhsOde(eqx.Module):
    """JAX/Equinox module implementing the generalized fed-batch ODE RHS.

    Created by :func:`get_rhs_ode`; do not instantiate directly.

    The state vector is ``c = [reactor_components..., pv_states..., V]`` where
    the last element is reactor volume. Biomass is always index 0 in the
    reactor-component block.

    Attributes
    ----------
    c_size : int
        ``n_non_volume_states + 1``.
    q_size : int
        ``n_reactor_states`` — number of specific rates (reactor block only).
    u_flow_size : int
        Number of continuous controlled flow streams.
    f_modeled_size : int
        Number of continuous uncontrolled (modeled) flow streams.
    output_size : int
        Same as :attr:`c_size`.
    reactor_component_state_names : tuple[str, ...]
        Ordering of reactor-component states in *c* and *q*. Biomass is
        always first.
    process_variable_state_names : tuple[str, ...]
        Ordering of process-variable states in *c*.
    flow_names : tuple[str, ...]
        Ordering of continuous controlled flow streams in *u_flow*.
    modeled_flow_names : tuple[str, ...]
        Ordering of continuous uncontrolled (modeled) flow streams in
        *f_modeled*.
    biomass_idx : int
        Index of ``"biomass"`` in reactor-component ordering (always 0).
    intracellular_indices : tuple[int, ...]
        Indices of intracellular states in reactor-component ordering.
        Intracellular components (e.g., intracellular product) accumulate
        inside the cells.  Active biomass is therefore:
        ``X_active = c[biomass_idx] - sum(c[i] for i in intracellular_indices)``.
    Cin : jnp.ndarray, shape (n_flows, n_reactor_states)
        Feed composition matrix for controlled flows: ``Cin[k, i]`` is the
        concentration of species *i* in controlled feed stream *k*.
    Cin_modeled : jnp.ndarray, shape (n_modeled_flows, n_reactor_states)
        Feed composition matrix for modeled (uncontrolled) flows.

    Notes
    -----
    JIT usage::

        import equinox as eqx
        mb    = get_rhs_ode(process)
        r     = jnp.zeros(mb.r_size)
        dc_dt = eqx.filter_jit(mb)(c, q, u_flow, jnp.zeros(0), r)
        # With modeled flows:
        dc_dt = eqx.filter_jit(mb)(c, q, u_flow, f_modeled, r)
    """

    c_size: int = eqx.field(static=True)
    q_size: int = eqx.field(static=True)
    r_size: int = eqx.field(static=True)
    u_flow_size: int = eqx.field(static=True)
    f_modeled_size: int = eqx.field(static=True)
    output_size: int = eqx.field(static=True)
    n_reactor_states: int = eqx.field(static=True)
    n_pv_states: int = eqx.field(static=True)
    reactor_indices: tuple = eqx.field(static=True)
    pv_indices: tuple = eqx.field(static=True)
    volume_idx: int = eqx.field(static=True)
    reactor_component_state_names: tuple = eqx.field(static=True)
    process_variable_state_names: tuple = eqx.field(static=True)
    static_pv_indices: tuple = eqx.field(static=True)
    flow_names: tuple = eqx.field(static=True)
    modeled_flow_names: tuple = eqx.field(static=True)
    biomass_idx: int = eqx.field(static=True)
    intracellular_indices: tuple = eqx.field(static=True)
    Cin: jnp.ndarray
    Cin_modeled: jnp.ndarray

    def __call__(
        self,
        c: jnp.ndarray,
        q: jnp.ndarray,
        u_flow: jnp.ndarray,
        f_modeled: jnp.ndarray,
        r: jnp.ndarray,
    ) -> jnp.ndarray:
        """Compute the ODE RHS ``dc/dt``.

        Parameters
        ----------
        c:
            State vector ``[reactor_components..., pv_states..., V]``,
            shape ``(c_size,)``.
            Biomass (measured) is at index 0; intracellular components follow
            in the positions given by :attr:`intracellular_indices`.
        q:
            Specific rates aligned with reactor-component ordering, shape
            ``(q_size,)``.
        u_flow:
            Volumetric flow rates for each continuous controlled feed stream
            (volume / time, matching the units of the stored
            ``VolumeChange``), shape ``(u_flow_size,)``.
        f_modeled:
            Volumetric flow rates for each continuous uncontrolled (modeled)
            feed stream, shape ``(f_modeled_size,)``.  Pass
            ``jnp.zeros(0)`` when there are no modeled flows.
        r:
            Additive physical-rate vector for all non-volume states, shape
            ``(r_size,)``. The first ``n_reactor_states`` entries align to
            reactor-component states and the remaining entries align to
            process-variable states.

        Returns
        -------
        jnp.ndarray, shape ``(output_size,)``
            ``dc/dt`` with ``dV/dt`` as the last element.

        Notes
        -----
        ODE RHS implemented:

        .. math::

            X_{active} = c_{biomass} - \\sum_{i \\in intracellular} c_i

            \\frac{dc_i}{dt} = q_i \\cdot X_{active} + r_i
                + \\sum_k \\frac{f_k}{V}\\,(C_{in,k,i} - c_i)
                \\quad (i \\notin \\{biomass\\})

            \\frac{dc_{biomass}}{dt} = \\Big(q_{biomass}
                + \\sum_{j \\in intracellular} q_j\\Big) \\cdot X_{active}
                + r_{biomass}
                + \\sum_k \\frac{f_k}{V}\\,(C_{in,k,biomass} - c_{biomass})

            \\frac{dc_{pv,j}}{dt} = r_{pv,j}

            \\frac{dV}{dt} = \\sum_k f_k

        where :math:`X_{active}` is the active biomass concentration
        (measured biomass minus intracellular component concentrations),
        and the sums over *k* include both controlled and modeled flows.
        The extra :math:`\\sum_{j} q_j` term in the biomass derivative
        comes from :math:`X_{measured} = X_{active} + \\sum_j P_j` so that
        :math:`dX_{measured}/dt = dX_{active}/dt + \\sum_j dP_j/dt`. With
        no intracellular components the sum is empty and
        :math:`dc_{biomass}/dt` collapses to the same form as the other
        states. ``q_{biomass}`` is therefore the specific growth rate of
        *active* biomass, not of measured biomass.
        """
        n_reactor = self.n_reactor_states
        n_non_volume = self.r_size
        c_reactor = c[:n_reactor]
        V = c[self.volume_idx]
        if self.n_pv_states > 0:
            c_pv = c[n_reactor:n_non_volume]
        else:
            c_pv = jnp.zeros(0)
        r_non_volume = r
        r_reactor = r_non_volume[:n_reactor]
        r_pv = r_non_volume[n_reactor:]
        if self.n_pv_states > 0 and len(self.static_pv_indices) > 0:
            static_idx = jnp.array(self.static_pv_indices, dtype=int)
            r_pv = r_pv.at[static_idx].set(0.0)

        # Active biomass: measured biomass minus intracellular components
        X_measured = c[self.biomass_idx]
        if len(self.intracellular_indices) > 0:
            intracellular_sum = jnp.sum(
                c_reactor[jnp.array(self.intracellular_indices)]
            )
        else:
            intracellular_sum = jnp.zeros(())
        X_active = X_measured - intracellular_sum

        # Reaction contribution: q_i * X_active, plus intracellular accumulation
        # added back into the measured-biomass entry for mass balance.
        reaction = q * X_active
        reaction = _add_intracellular_to_biomass(
            reaction, q, X_active, self.biomass_idx, self.intracellular_indices
        )

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

        dc_reactor = reaction + r_reactor + feed_term
        dc_pv = c_pv * 0.0 + r_pv

        return jnp.append(jnp.append(dc_reactor, dc_pv), dV)


# ---------------------------------------------------------------------------
# User-defined biological ODE (RhsOde alternative when
# ``process.biological_ode`` is set)
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


def _topo_sort_derived(derived_exprs: Dict[str, Any]) -> List[str]:
    """Topologically sort derived names by mutual dependencies. Assumes the
    expressions have already been parsed and validated as acyclic."""
    derived_names = set(derived_exprs.keys())
    deps = {
        name: {str(s) for s in expr.free_symbols} & derived_names
        for name, expr in derived_exprs.items()
    }
    order: List[str] = []
    remaining = set(derived_names)
    while remaining:
        ready = sorted(n for n in remaining if not (deps[n] & remaining))
        if not ready:
            raise ValueError(
                "Cyclic derived dependencies detected during topo-sort: "
                f"{sorted(remaining)}"
            )
        order.extend(ready)
        remaining -= set(ready)
    return order


class UserDefinedRhsOde(eqx.Module):
    """JAX/Equinox module that evaluates a user-defined biological RHS.

    Created by :func:`build_user_defined_rhs_ode` when
    ``process.biological_ode is not None``. The biological ``dc/dt`` per
    state is given by user-written expression strings; bp-format adds the
    physical contributions (feed, dilution, dV) on top.

    Call signature::

        dc_dt = mb(c, rates, u_flow, f_modeled, ctrl_pv_values)

    where:

    - ``c`` is the state vector ``[reactor..., pv..., V]``.
    - ``rates`` is the user-declared rate vector, shape ``(rate_size,)``,
      aligned with :attr:`rate_names`.
    - ``u_flow``, ``f_modeled`` are continuous flow rates (same as
      :class:`RhsOde`).
    - ``ctrl_pv_values`` is an array of controlled-PV values at the current
      time, aligned with :attr:`controlled_pv_names`. Pass
      ``jnp.zeros(0)`` when there are no controlled PVs.
    """

    # --- Sizes / indices (mirror RhsOde so downstream introspection works) ---
    c_size: int = eqx.field(static=True)
    rate_size: int = eqx.field(static=True)
    # Alias for `rate_size` so code paths that currently expect ``mb.q_size``
    # (e.g. _validate_rates_output_shapes, _build_segment_rhs dispatch) keep
    # working when given a UserDefinedRhsOde. With the user-defined path,
    # ``q_size == rate_size`` and may differ from ``n_reactor_states``.
    q_size: int = eqx.field(static=True)
    r_size: int = eqx.field(static=True)
    u_flow_size: int = eqx.field(static=True)
    f_modeled_size: int = eqx.field(static=True)
    output_size: int = eqx.field(static=True)
    n_reactor_states: int = eqx.field(static=True)
    n_pv_states: int = eqx.field(static=True)
    n_controlled_pv: int = eqx.field(static=True)
    n_derived: int = eqx.field(static=True)
    reactor_indices: tuple = eqx.field(static=True)
    pv_indices: tuple = eqx.field(static=True)
    volume_idx: int = eqx.field(static=True)
    biomass_idx: int = eqx.field(static=True)
    intracellular_indices: tuple = eqx.field(static=True)
    static_pv_indices: tuple = eqx.field(static=True)

    # --- Names (deterministic ordering) ---
    reactor_component_state_names: tuple = eqx.field(static=True)
    process_variable_state_names: tuple = eqx.field(static=True)
    controlled_pv_names: tuple = eqx.field(static=True)
    flow_names: tuple = eqx.field(static=True)
    modeled_flow_names: tuple = eqx.field(static=True)
    derived_names: tuple = eqx.field(static=True)
    rate_names: tuple = eqx.field(static=True)

    # --- Compiled callables ---
    # Each takes a flat args array (state | ctrl_pv | derived | rates) and
    # returns a scalar. Stored as static so equinox treats them as Python
    # config rather than pytree leaves.
    derived_funcs: tuple = eqx.field(static=True)
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

        # 1. Compute derived variables in topo order (already sorted at build).
        derived_arr = jnp.zeros(self.n_derived) if self.n_derived > 0 else jnp.zeros(0)
        for i, fn in enumerate(self.derived_funcs):
            args = jnp.concatenate([state_values, ctrl_pv_values, derived_arr, rates])
            derived_arr = derived_arr.at[i].set(fn(args))

        # 2. Compute biological derivatives per dynamic state.
        full_args = jnp.concatenate([state_values, ctrl_pv_values, derived_arr, rates])
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


def build_user_defined_rhs_ode(process: BioProcess) -> UserDefinedRhsOde:
    """Build a :class:`UserDefinedRhsOde` from a process whose
    ``biological_ode`` block is set. Raises :class:`ValueError` if the block
    is missing or fails validation.
    """
    import sympy

    bo = process.biological_ode
    if bo is None:
        raise ValueError(
            "build_user_defined_rhs_ode requires process.biological_ode to be set."
        )

    # Reuse the auto path's symbol/index discovery so biomass-at-0 and
    # intracellular ordering match exactly. We construct a transient RhsOde
    # for its metadata and discard the rest.
    auto = _build_auto_rhs_ode(process)

    reactor_state_names = auto.reactor_component_state_names
    pv_state_names = auto.process_variable_state_names
    controlled_pv_names = tuple(
        name for name, pv in process.process_variables.items() if pv.is_controlled
    )

    state_derivative_names = tuple(reactor_state_names) + tuple(pv_state_names)
    derived_names_set = set(bo.derived.keys())
    rate_names_tuple = tuple(bo.rates.keys())
    allowed_names = (
        set(state_derivative_names)
        | set(controlled_pv_names)
        | derived_names_set
        | set(rate_names_tuple)
    )
    symbol_table = {n: sympy.Symbol(n) for n in allowed_names}

    # Parse all expressions; surface any error from validation for clarity.
    derived_exprs: Dict[str, Any] = {}
    for name, expr_str in bo.derived.items():
        try:
            derived_exprs[name] = sympy.sympify(expr_str, locals=symbol_table)
        except Exception as exc:
            raise ValueError(
                f"biological_ode.derived[{name!r}] failed to parse: {exc}"
            ) from exc

    derivative_exprs: Dict[str, Any] = {}
    for name, expr_str in bo.derivatives.items():
        try:
            derivative_exprs[name] = sympy.sympify(expr_str, locals=symbol_table)
        except Exception as exc:
            raise ValueError(
                f"biological_ode.derivatives[{name!r}] failed to parse: {exc}"
            ) from exc

    # Topo-sort derived; build ordered tuple of names.
    derived_order = _topo_sort_derived(derived_exprs)
    derived_names_ordered = tuple(derived_order)

    # Build the canonical args ordering used by every lambdified expression.
    # Concatenation order: state_derivative_names | controlled_pv_names
    #                      | derived_names_ordered | rate_names_tuple.
    args_order = (
        tuple(state_derivative_names)
        + tuple(controlled_pv_names)
        + derived_names_ordered
        + rate_names_tuple
    )

    derived_funcs = tuple(
        _lambdify_with_array_arg(derived_exprs[n], args_order)
        for n in derived_names_ordered
    )

    # Build per-state derivative callables in state order; missing state
    # entries should have been caught by validate_biological_ode but we
    # default to zero defensively.
    zero_expr = sympy.Integer(0)
    derivative_funcs = tuple(
        _lambdify_with_array_arg(derivative_exprs.get(n, zero_expr), args_order)
        for n in state_derivative_names
    )

    return UserDefinedRhsOde(
        c_size=auto.c_size,
        rate_size=len(rate_names_tuple),
        q_size=len(rate_names_tuple),
        r_size=auto.r_size,
        u_flow_size=auto.u_flow_size,
        f_modeled_size=auto.f_modeled_size,
        output_size=auto.output_size,
        n_reactor_states=auto.n_reactor_states,
        n_pv_states=auto.n_pv_states,
        n_controlled_pv=len(controlled_pv_names),
        n_derived=len(derived_names_ordered),
        reactor_indices=auto.reactor_indices,
        pv_indices=auto.pv_indices,
        volume_idx=auto.volume_idx,
        biomass_idx=auto.biomass_idx,
        intracellular_indices=auto.intracellular_indices,
        static_pv_indices=auto.static_pv_indices,
        reactor_component_state_names=auto.reactor_component_state_names,
        process_variable_state_names=auto.process_variable_state_names,
        controlled_pv_names=controlled_pv_names,
        flow_names=auto.flow_names,
        modeled_flow_names=auto.modeled_flow_names,
        derived_names=derived_names_ordered,
        rate_names=rate_names_tuple,
        derived_funcs=derived_funcs,
        derivative_funcs=derivative_funcs,
        Cin=auto.Cin,
        Cin_modeled=auto.Cin_modeled,
    )


def build_derived_func(
    process: BioProcess,
) -> Callable[[jnp.ndarray, jnp.ndarray, jnp.ndarray], Dict[str, jnp.ndarray]]:
    """Build a callable that evaluates the ``biological_ode.derived`` variables
    given concrete state, controlled-PV, and rate arrays.

    Returns ``f(state_values, ctrl_pv_values, rates) -> {name: scalar}``.
    Useful to expose derived quantities like ``X_active`` as observables for
    plotting or loss computation. Raises ``ValueError`` if the process has no
    ``biological_ode`` block.
    """
    bo = process.biological_ode
    if bo is None:
        raise ValueError(
            "build_derived_func requires process.biological_ode to be set."
        )
    mb = build_user_defined_rhs_ode(process)
    n_derived = mb.n_derived
    derived_names = mb.derived_names
    derived_funcs = mb.derived_funcs

    def derived_func(state_values, ctrl_pv_values, rates):
        derived_arr = jnp.zeros(n_derived) if n_derived > 0 else jnp.zeros(0)
        for i, fn in enumerate(derived_funcs):
            args = jnp.concatenate([state_values, ctrl_pv_values, derived_arr, rates])
            derived_arr = derived_arr.at[i].set(fn(args))
        return {name: derived_arr[i] for i, name in enumerate(derived_names)}

    return derived_func


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


def get_rhs_ode(process: BioProcess):
    """Build the appropriate RHS ODE module for *process*.

    Dispatches based on the presence of ``process.biological_ode``:

    - When set: returns :class:`UserDefinedRhsOde` built from the
      user-declared per-state biological expressions, derived variables,
      and abstract rate symbols.
    - When ``None`` (default / backward compatible): returns the
      auto-generated :class:`RhsOde` whose biological term is
      ``q_i * X_active`` per reactor component (with the intracellular
      mass-balance correction applied to the biomass entry).

    Parameters
    ----------
    process:
        A :class:`~bp_format.BioProcess` instance.  The reactor medium must
        contain a component named ``"biomass"`` (case-insensitive).

    Returns
    -------
    RhsOde or UserDefinedRhsOde
        See above.

    Raises
    ------
    ValueError
        If no ``"biomass"`` component is found in the reactor medium, or if
        ``biological_ode`` is set but malformed (see
        :func:`bp_format.validate.validate_biological_ode` for the rules).
    """
    if process.biological_ode is not None:
        return build_user_defined_rhs_ode(process)
    return _build_auto_rhs_ode(process)


def _build_auto_rhs_ode(process: BioProcess) -> RhsOde:
    """Build the auto-generated :class:`RhsOde` (the pre-Phase-2 behavior).

    Used by :func:`get_rhs_ode` as the fallback when no
    ``biological_ode`` block is present, and by
    :func:`build_user_defined_rhs_ode` for shared metadata discovery.
    """
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
    biomass_idx: int = 0  # always 0 by construction

    # --- Intracellular indices ---
    intracellular_indices: List[int] = []
    for i, name in enumerate(reactor_component_state_names):
        comp = process.reactor_medium.components[name]
        if comp.is_intracellular:
            intracellular_indices.append(i)

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

    # --- Helper to build a Cin matrix for a list of volume-change names ---
    def _build_cin(vc_names):
        n = len(vc_names)
        Cin = jnp.zeros((n, n_reactor), dtype=float)
        for k, vc_name in enumerate(vc_names):
            vc = process.volume.volume_changes[vc_name]
            if not isinstance(vc, FeedVolumeChange):
                continue  # SampleVolumeChange has no feed medium
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

    # --- Controlled continuous flows ---
    flow_names: List[str] = []
    for vc_name, vc in process.volume.volume_changes.items():
        if vc.is_controlled and vc.is_continuous:
            flow_names.append(vc_name)

    # --- Modeled (uncontrolled) continuous flows ---
    modeled_flow_names: List[str] = []
    for vc_name, vc in process.volume.volume_changes.items():
        if (not vc.is_controlled) and vc.is_continuous:
            modeled_flow_names.append(vc_name)

    Cin = _build_cin(flow_names)
    Cin_modeled = _build_cin(modeled_flow_names)

    return RhsOde(
        c_size=n_non_volume + 1,
        q_size=n_reactor,
        r_size=n_non_volume,
        u_flow_size=len(flow_names),
        f_modeled_size=len(modeled_flow_names),
        output_size=n_non_volume + 1,
        n_reactor_states=n_reactor,
        n_pv_states=n_pv,
        reactor_indices=tuple(range(n_reactor)),
        pv_indices=tuple(range(n_reactor, n_non_volume)),
        volume_idx=n_non_volume,
        reactor_component_state_names=tuple(reactor_component_state_names),
        process_variable_state_names=tuple(process_variable_state_names),
        static_pv_indices=tuple(static_pv_indices),
        flow_names=tuple(flow_names),
        modeled_flow_names=tuple(modeled_flow_names),
        biomass_idx=biomass_idx,
        intracellular_indices=tuple(intracellular_indices),
        Cin=Cin,
        Cin_modeled=Cin_modeled,
    )


# ---------------------------------------------------------------------------
# Discrete event handling
# ---------------------------------------------------------------------------


def extract_discrete_events(
    process: BioProcess,
    mb: RhsOde,
) -> List[Dict[str, Any]]:
    """Extract discrete events (sampling, bolus feeds) from a BioProcess.

    Parameters
    ----------
    process:
        A :class:`~bp_format.BioProcess` instance.
    mb:
        A :class:`RhsOde` module (used to align ``Cin`` with species ordering).

    Returns
    -------
    list[dict]
        Sorted list of event dicts, each with keys:
        - ``t`` (float): event time
        - ``kind`` (str): ``'sample'`` or ``'bolus_feed'``
        - ``dV`` (float): signed volume change (positive = add, negative = remove)
        - ``Cin`` (jnp.ndarray | None): feed composition aligned with
          ``mb.reactor_component_state_names`` (None for sampling)
        - ``source`` (str): name of the originating VolumeChange
    """
    events: List[Dict[str, Any]] = []
    n_reactor = mb.n_reactor_states

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
                    for j, sp_name in enumerate(mb.reactor_component_state_names):
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
    mb: "RhsOde",
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
    mb:
        A :class:`RhsOde` module (provides species ordering).

    Returns
    -------
    dict
        Mapping non-volume state name -> callable spline.
    """
    state_splines: Dict[str, Any] = {}
    pseudobatch_transform = _validate_process_pseudobatch_transform(process, mb)

    for sp_name in mb.reactor_component_state_names:
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

    for pv_name in mb.process_variable_state_names:
        pv = process.process_variables[pv_name]
        state_splines[pv_name] = _value_to_interpax_spline(
            pv.values,
            t_start=float(process.time_axis.start),
            t_end=float(process.time_axis.end),
        )

    return state_splines


def build_conc_splines(
    process: BioProcess,
    mb: "RhsOde",
) -> Dict[str, Any]:
    """Compatibility alias for :func:`build_state_splines`."""
    return build_state_splines(process, mb)


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
# Analytical q(t) builder and specific rate estimation
# ---------------------------------------------------------------------------


def build_q_func(
    process: BioProcess,
    ctrl: ControlSplines,
    mb: RhsOde,
    state_splines: Optional[Dict[str, Any]] = None,
    *,
    conc_splines: Optional[Dict[str, Any]] = None,
    q_state_indices: Optional[List[int]] = None,
    r_state_indices: Optional[List[int]] = None,
    r_func: Optional[Callable] = None,
) -> Callable:
    """Build an analytical, JIT-compilable ``q(t)`` callable.

    Returns a function ``q(t) -> jnp.ndarray`` that evaluates specific rates
    at *any* time ``t`` using the analytical spline derivatives:

    .. math::
        q_i(t) = \\frac{dc_i/dt - \\text{feed\\_term}_i}{X_{active}}
        \\quad (i \\notin \\{biomass\\})

        q_{biomass}(t) = \\frac{dc_{biomass}/dt
            - \\sum_{j \\in intracellular} dc_j/dt
            - \\text{feed\\_term}_{biomass}}{X_{active}}

    The biomass-specific case subtracts the intracellular concentration
    derivatives so the returned ``q[biomass]`` is the specific growth rate
    of *active* biomass (the inverse of the forward RHS in :class:`RhsOde`).
    Without intracellular components the sum is empty and the formula
    collapses to the same form as the other states.

    All components (concentration derivatives, volume, flow rates, active
    biomass) are evaluated analytically from splines and discrete-event
    data via ``jnp.searchsorted``.  The returned callable is compatible
    with ``jax.jit``.

    Parameters
    ----------
    process:
        A :class:`~bp_format.BioProcess` instance.
    ctrl:
        :class:`ControlSplines` module for evaluating control signals.
    mb:
        :class:`RhsOde` module (provides species ordering, Cin, etc.).
    state_splines:
        Dict mapping non-volume state name -> callable spline that supports
        ``spline(t)`` and ``spline.derivative()(t)``.
    q_state_indices, r_state_indices:
        Optional reactor-component index partition for inversion:
        ``dc/dt = q*X_active + r + feed``.

        - If both are omitted and ``r_func`` is not provided, all
          reactor-component states are treated as
          ``q``-states (default biological assumption).
        - If ``r_func`` is provided, both ``q_state_indices`` and
          ``r_state_indices`` must be provided explicitly.
        - If either is supplied, both must be supplied.
        - The union of ``q_state_indices`` and ``r_state_indices`` must cover
          all reactor-component states.
        - Overlap (indicating that biological and physical both affect some of the state
          variables) is allowed only when ``r_func`` is provided, so overlap ``r`` can
          be subtracted before solving for ``q``.
    r_func:
        Optional ``r_func(t) -> r`` used during inversion partitioning.
        ``r`` is interpreted as additive physical rates over all non-volume
        states; only reactor-component entries are used by ``build_q_func``.

    Returns
    -------
    Callable
        ``q_func(t) -> jnp.ndarray`` of shape ``(n_reactor_states,)``.
    """
    state_splines = _resolve_state_splines(
        state_splines=state_splines,
        conc_splines=conc_splines,
    )
    if state_splines is None:
        raise ValueError("state_splines is required.")
    n_reactor = mb.n_reactor_states
    t_start = float(process.time_axis.start)
    t_end = float(process.time_axis.end)

    # -- Cumulative volume splines for continuous flows --
    cum_splines_ctrl = []
    for fn in mb.flow_names:
        vc = process.volume.volume_changes[fn]
        sp = _timeseries_to_interpax_spline(vc.values)
        cum_splines_ctrl.append(sp)

    cum_splines_mod = []
    for fn in mb.modeled_flow_names:
        vc = process.volume.volume_changes[fn]
        sp = _timeseries_to_interpax_spline(vc.values)
        cum_splines_mod.append(sp)

    # -- Batch all volume splines into a single PPoly --
    all_vol_splines = cum_splines_ctrl + cum_splines_mod
    n_vol = len(all_vol_splines)
    if n_vol > 0:
        batched_vol = _batch_splines(all_vol_splines, t_start, t_end)
    else:
        batched_vol = None

    # -- Discrete events: precompute sorted times and cumulative dV --
    events = extract_discrete_events(process, mb)
    if events:
        ev_sorted = sorted(events, key=lambda e: e["t"])
        ev_times = jnp.array([e["t"] for e in ev_sorted])
        ev_dV_cum = jnp.cumsum(jnp.array([e["dV"] for e in ev_sorted]))
    else:
        ev_times = None
        ev_dV_cum = None

    V0 = _require_reactor_volume_above_threshold(
        jnp.array(float(process.volume.initial_volume)),
        context="initial reactor volume",
    )
    Cin = jnp.array(mb.Cin)
    Cin_mod = jnp.array(mb.Cin_modeled)

    q_idx_arg = None if q_state_indices is None else list(q_state_indices)
    r_idx_arg = None if r_state_indices is None else list(r_state_indices)

    if r_func is not None and (q_idx_arg is None or r_idx_arg is None):
        raise ValueError(
            "r_func requires explicit q_state_indices and r_state_indices "
            "so q/r inversion partitioning is unambiguous."
        )

    if q_idx_arg is None and r_idx_arg is None:
        q_idx = list(range(n_reactor))
        r_idx = []
    else:
        if q_idx_arg is None or r_idx_arg is None:
            raise ValueError(
                "q_state_indices and r_state_indices must both be provided "
                "when using runtime q/r partitioning."
            )
        q_idx = sorted(set(int(i) for i in q_idx_arg))
        r_idx = sorted(set(int(i) for i in r_idx_arg))
        if any(i < 0 or i >= n_reactor for i in q_idx + r_idx):
            raise ValueError(
                f"q/r indices must be in [0, {n_reactor - 1}] for reactor states."
            )
        overlap = set(q_idx).intersection(r_idx)
        if overlap and r_func is None:
            raise ValueError(
                "Overlapping q/r state indices require r_func(t) so r can be "
                "subtracted during q inversion."
            )
        if set(q_idx).union(r_idx) != set(range(n_reactor)):
            raise ValueError(
                "q_state_indices and r_state_indices must together cover all "
                "reactor states."
            )

    q_mask = jnp.zeros(n_reactor, dtype=bool).at[jnp.array(q_idx, dtype=int)].set(True)
    overlap_mask = jnp.zeros(n_reactor, dtype=bool)
    if len(r_idx) > 0:
        overlap_idx = sorted(set(q_idx).intersection(r_idx))
        if overlap_idx:
            overlap_mask = overlap_mask.at[jnp.array(overlap_idx, dtype=int)].set(True)

    # -- Per-species concentration spline evaluators --
    # Uses original BacktransformSpline evaluations (exact piecewise-linear fc)
    # to avoid Gibbs oscillation from cubic resampling of step-like fc data.
    conc_evals = [state_splines[s] for s in mb.reactor_component_state_names]
    deriv_evals = [
        state_splines[s].derivative() for s in mb.reactor_component_state_names
    ]

    biomass_idx = mb.biomass_idx
    intra_idx = mb.intracellular_indices
    u_flow_size = mb.u_flow_size
    f_mod_size = mb.f_modeled_size

    def q_func(t):
        # Concentrations — per-species spline evaluation
        c_t = jnp.stack([conc_evals[i](t) for i in range(n_reactor)])
        c_t = jnp.maximum(c_t, 0.0)

        # Concentration derivatives (analytical) — per-species
        dc_dt = jnp.stack([deriv_evals[i](t) for i in range(n_reactor)])

        # Volume: V0 + continuous flows + discrete events
        V_t = V0
        if batched_vol is not None:
            V_t = V_t + jnp.sum(batched_vol(t))  # single batched eval
        if ev_times is not None:
            idx = jnp.searchsorted(ev_times, t, side="left")
            V_t = V_t + jnp.where(idx > 0, ev_dV_cum[jnp.clip(idx - 1, 0)], 0.0)
        V_t = _require_reactor_volume_above_threshold(
            V_t, context="q-function reconstructed volume"
        )

        # Flow rates (derivatives of cumulative volume splines) — batched
        if u_flow_size > 0 or f_mod_size > 0:
            all_flow_rates = batched_vol(t, nu=1)  # (n_vol,)
            u_flow = all_flow_rates[:u_flow_size] if u_flow_size > 0 else jnp.zeros(0)
            f_mod = all_flow_rates[u_flow_size:] if f_mod_size > 0 else jnp.zeros(0)
        else:
            u_flow = jnp.zeros(0)
            f_mod = jnp.zeros(0)

        # Feed term: sum_k (f_k / V) * (C_in_k - c)
        feed_term = jnp.zeros(n_reactor)
        if u_flow_size > 0:
            feed_term = feed_term + jnp.sum(
                (u_flow[:, None] / V_t) * (Cin - c_t[None, :]), axis=0
            )
        if f_mod_size > 0:
            feed_term = feed_term + jnp.sum(
                (f_mod[:, None] / V_t) * (Cin_mod - c_t[None, :]), axis=0
            )

        # Active biomass
        X_active = c_t[biomass_idx]
        if len(intra_idx) > 0:
            X_active = X_active - jnp.sum(c_t[jnp.array(intra_idx)])
        X_active = jnp.maximum(X_active, jnp.array(_MIN_ACTIVE_BIOMASS))

        r_reactor = jnp.zeros(n_reactor)
        if r_func is not None:
            r_full = jnp.asarray(r_func(t), dtype=float)
            r_reactor = r_full[:n_reactor]

        q_all = (dc_dt - feed_term - jnp.where(overlap_mask, r_reactor, 0.0)) / X_active
        # Intracellular correction: q[biomass] from this division is the
        # *apparent* specific rate of measured biomass; subtract the
        # intracellular q values to recover the active-biomass specific rate.
        q_all = _subtract_intracellular_from_biomass_q(q_all, biomass_idx, intra_idx)
        return jnp.where(q_mask, q_all, 0.0)

    return q_func


def build_rates_func(
    process: BioProcess,
    ctrl: ControlSplines,
    mb: RhsOde,
    state_splines: Optional[Dict[str, Any]] = None,
    *,
    conc_splines: Optional[Dict[str, Any]] = None,
    q_state_indices: Optional[List[int]] = None,
    r_state_indices: Optional[List[int]] = None,
    r_func: Optional[Callable] = None,
) -> Callable:
    """Build integration callback ``rates_func(t, state, controls) -> (q, r)``.

    This is the bridge between inversion-side spline reconstruction and
    runtime integration APIs.

    Pipeline semantics:

    - ``q`` is obtained from :func:`build_q_func` and has one biomass-specific
      rate per reactor-component state.
    - ``r`` is an additive physical-rate vector over all non-volume states.
      Its first ``n_reactor_states`` entries align to the same reactor-component
      states covered by ``q``; any remaining entries correspond to
      process-variable states. If ``r_func`` is not supplied, reactor-component
      ``r`` entries default to zero and process-variable entries are inferred
      from state-spline derivatives.
    - ``state`` and ``controls`` are accepted to satisfy integration callback
      signature, but this spline-derived wrapper currently depends on ``t``
      only.

    Parameters
    ----------
    process, ctrl, mb, state_splines, conc_splines:
        Same meaning as :func:`build_q_func`.
    q_state_indices, r_state_indices:
        Forwarded to :func:`build_q_func`; see its docstring for partitioning
        rules.
    r_func:
        Optional ``r_func(t) -> r`` used to populate additive physical rates.

    Returns
    -------
    Callable
        ``rates_func(t, state, controls) -> tuple[q, r]`` with shapes
        ``q.shape == (mb.q_size,)`` and ``r.shape == (mb.r_size,)``.
    """
    state_splines = _resolve_state_splines(
        state_splines=state_splines,
        conc_splines=conc_splines,
    )
    q_only = build_q_func(
        process,
        ctrl,
        mb,
        state_splines,
        conc_splines=None,
        q_state_indices=q_state_indices,
        r_state_indices=r_state_indices,
        r_func=r_func,
    )
    pv_derivs: List[Callable] = []
    if r_func is None and mb.n_pv_states > 0:
        pv_derivs = [
            state_splines[pv_name].derivative()
            for pv_name in mb.process_variable_state_names
        ]

    def rates_func(t, _state, _controls):
        # default rates_func only uses inverted splines: we can ignore state and
        # controls
        q = q_only(t)
        if r_func is None:
            if mb.n_pv_states == 0:
                r = jnp.zeros(mb.r_size, dtype=float)
            else:
                r_reactor = jnp.zeros(mb.n_reactor_states, dtype=float)
                r_pv = jnp.stack([pv_derivs[i](t) for i in range(mb.n_pv_states)])
                r = jnp.concatenate([r_reactor, r_pv])
        else:
            r = jnp.asarray(r_func(t), dtype=float)
        return q, r

    return rates_func


def estimate_specific_rates(
    process: BioProcess,
    ctrl: ControlSplines,
    mb: RhsOde,
    state_splines: Optional[Dict[str, Any]] = None,
    t_eval: Optional[jnp.ndarray] = None,
    *,
    conc_splines: Optional[Dict[str, Any]] = None,
    q_state_indices: Optional[List[int]] = None,
    r_state_indices: Optional[List[int]] = None,
    r_func: Optional[Callable] = None,
) -> jnp.ndarray:
    """Estimate specific rates q(t) via ODE RHS inversion.

    Convenience wrapper around :func:`build_q_func` that evaluates the
    analytical rate function at the given time points.

    This helper is inversion-only and returns ``q`` values. Integration APIs
    consume ``rates_func(t, state, controls) -> (q, r)``; use
    :func:`build_rates_func` to assemble that callback from spline-derived
    ``q`` and optional ``r_func``.

    Parameters
    ----------
    process:
        A :class:`~bp_format.BioProcess` instance.
    ctrl:
        :class:`ControlSplines` module for evaluating control signals.
    mb:
        :class:`RhsOde` module (provides species ordering, Cin, etc.).
    state_splines:
        Dict mapping non-volume state name -> callable spline that supports
        ``spline(t)`` and ``spline.derivative()(t)``.
    t_eval:
        1-D array of time points at which to estimate q.
    q_state_indices, r_state_indices:
        Forwarded to :func:`build_q_func`; see its docstring for partitioning
        rules.
    r_func:
        Optional ``r_func(t)`` forwarded to :func:`build_q_func` for overlap
        handling in q/r partitioning.

    Returns
    -------
    jnp.ndarray, shape (len(t_eval), n_reactor_states)
        Estimated specific rates at each time point.
    """
    if t_eval is None:
        raise ValueError("t_eval is required.")
    t_eval = jnp.asarray(t_eval, dtype=float)
    q_func = build_q_func(
        process,
        ctrl,
        mb,
        state_splines,
        conc_splines=conc_splines,
        q_state_indices=q_state_indices,
        r_state_indices=r_state_indices,
        r_func=r_func,
    )
    q_func_jit = eqx.filter_jit(q_func)
    return jax.vmap(q_func_jit)(t_eval)


# ---------------------------------------------------------------------------
# Full hybrid ODE integration
# ---------------------------------------------------------------------------


def _require_rates_func(rates_func: Optional[Callable]) -> Callable:
    """Require the mixed-state runtime rates callback."""
    if rates_func is None:
        raise ValueError(
            "rates_func is required and must have signature "
            "rates_func(t, state, controls) -> (q, r)."
        )
    return rates_func


def _build_segment_rhs(
    mb,
    ctrl,
    rates_func,
    batched_mod,
):
    """Build the ODE right-hand side function for a segment.

    Honest forward integration: the RHS reads everything (including
    ``X_active`` for the reaction term) from the integrator's current state.
    No spline-substitution trick. UserDefinedRhsOde takes a separate
    branch that also passes controlled-PV values evaluated from
    ``ControlSplines``.

    Parameters
    ----------
    batched_mod : interpax.PPoly or None
        Batched PPoly for modeled (uncontrolled) cumulative volume splines.
        Evaluate with ``batched_mod(t, nu=1)`` to get flow rates.
    """
    flow_idx = jnp.array(list(ctrl.flow_indices))
    ctrl_idx = jnp.array(list(ctrl.ctrl_indices))
    is_user_defined = isinstance(mb, UserDefinedRhsOde)

    def rhs(t, state, args):
        u = ctrl(t)
        u_flow = u[flow_idx] if len(flow_idx) > 0 else jnp.zeros(mb.u_flow_size)
        q, r = rates_func(t, state, u)

        if batched_mod is not None:
            f_mod = batched_mod(t, nu=1)
        else:
            f_mod = jnp.zeros(mb.f_modeled_size)

        if is_user_defined:
            ctrl_pv_values = u[ctrl_idx] if len(ctrl_idx) > 0 else jnp.zeros(0)
            return mb(state, q, u_flow, f_mod, ctrl_pv_values)
        return mb(state, q, u_flow, f_mod, r)

    return rhs


def _validate_rates_output_shapes(
    mb: "RhsOde",
    rates_func: Callable,
    *,
    t: float,
    state: jnp.ndarray,
    controls: jnp.ndarray,
) -> None:
    """Validate ``rates_func`` output shapes before JIT solve."""
    q_probe, r_probe = rates_func(float(t), state, controls)
    q_probe = jnp.asarray(q_probe, dtype=float)
    r_probe = jnp.asarray(r_probe, dtype=float)
    if q_probe.shape != (mb.q_size,):
        raise ValueError(
            f"rates_func must return q with shape ({mb.q_size},), got {q_probe.shape}."
        )
    if r_probe.shape != (mb.r_size,):
        raise ValueError(
            f"rates_func must return r with shape ({mb.r_size},), got {r_probe.shape}."
        )


def _compute_scale_factors(process: BioProcess, mb: "RhsOde") -> jnp.ndarray:
    """Compute non-volume state scale factors for numerical conditioning."""
    scales = jnp.ones(mb.r_size)
    for i, sp_name in enumerate(mb.reactor_component_state_names):
        vals = jnp.asarray(
            process.reactor_medium.components[sp_name].concentration.values, dtype=float
        )
        s = float(jnp.max(jnp.abs(vals)))
        if s > 1.0:
            scales = scales.at[i].set(s)
    offset = mb.n_reactor_states
    for j, pv_name in enumerate(mb.process_variable_state_names):
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
    mb: "RhsOde",
):
    """Validate process-level pseudobatch bundle before runtime use."""
    transform = getattr(process, "pseudobatch_transform", None)
    if transform is None:
        for sp_name in mb.reactor_component_state_names:
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
    mb: "RhsOde",
) -> List[Dict[str, Any]]:
    """Build per-species pseudo-batch transform descriptors.

    Each descriptor supports:
      c* = adf(t) * c - fc(t)
      c  = (c* + fc(t)) / adf(t)
      dc*/dt = adf * dc/dt + d(adf)/dt * c - d(fc)/dt
    """
    t_start = float(process.time_axis.start)
    t_end = float(process.time_axis.end)
    pseudobatch_transform = _validate_process_pseudobatch_transform(process, mb)
    transforms: List[Dict[str, Any]] = []
    for sp_name in mb.reactor_component_state_names:
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

    for _ in mb.process_variable_state_names:
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


def integrate_process_pseudospace(
    process: BioProcess,
    ctrl: ControlSplines,
    mb: RhsOde,
    rates_func: Callable,
    t_eval: jnp.ndarray,
    *,
    state_splines: Optional[Dict[str, Any]] = None,
    conc_splines: Optional[Dict[str, Any]] = None,
    rtol: float = 1e-6,
    atol: float = 1e-8,
    use_jump_ts: bool = True,
    max_steps: int = 16384,
) -> Dict[str, Any]:
    """Integrate in c* (pseudo-batch) space with a single ``diffeqsolve``.

    Parameters
    ----------
    rates_func:
        Callable ``rates_func(t, state, controls) -> tuple[q, r]``.
        ``q`` covers reactor-component biomass-specific rates and ``r`` covers
        additive physical rates over all non-volume states.

    Notes
    -----
    Process-variable states are integrated as additive-only:
    ``dc_pv/dt = r_pv``. They do not receive dilution/feed terms.
    """
    state_splines = _resolve_state_splines(
        state_splines=state_splines,
        conc_splines=conc_splines,
    )
    t_eval = jnp.asarray(t_eval, dtype=float)
    t_start = float(process.time_axis.start)
    t_end = float(process.time_axis.end)
    n_reactor = mb.n_reactor_states
    n_non_volume = mb.r_size
    rates_func = _require_rates_func(rates_func)

    transforms = _build_pseudobatch_transforms(process, mb)

    # Per-state pseudobatch transforms. ADF/feed-correction are canonical
    # TimeSeries objects; derivatives ignore instantaneous jump impulses.
    adf_ts_list: List[TimeSeries] = []
    dadf_ts_list: List[TimeSeries] = []
    fc_ts_list: List[TimeSeries] = []
    dfc_ts_list: List[TimeSeries] = []
    for tr in transforms:
        adf_ts_list.append(tr["adf_ts"])
        dadf_ts_list.append(tr["dadf_ts"])
        fc_ts_list.append(tr["feed_corr_ts"])
        dfc_ts_list.append(tr["dfc_ts"])

    if state_splines is not None:
        bio_spline = state_splines[mb.reactor_component_state_names[mb.biomass_idx]]
        intra_splines = [
            state_splines[mb.reactor_component_state_names[i]]
            for i in mb.intracellular_indices
        ]
    else:
        bio_spline = None
        intra_splines = []

    # Modeled (uncontrolled) continuous flows.
    cum_splines_mod: List[interpax.CubicSpline] = []
    for fn in mb.modeled_flow_names:
        vc = process.volume.volume_changes[fn]
        sp = _timeseries_to_interpax_spline(vc.values)
        cum_splines_mod.append(sp)
    batched_mod = (
        _batch_splines(cum_splines_mod, t_start, t_end)
        if len(cum_splines_mod) > 0
        else None
    )

    events = extract_discrete_events(process, mb)
    if events:
        ev_sorted = sorted(events, key=lambda e: e["t"])
        ev_times = jnp.asarray([e["t"] for e in ev_sorted], dtype=float)
        ev_dV_cum = jnp.cumsum(jnp.asarray([e["dV"] for e in ev_sorted], dtype=float))
        jump_times = sorted(
            set(float(e["t"]) for e in ev_sorted if t_start < float(e["t"]) < t_end)
        )
    else:
        ev_times = None
        ev_dV_cum = None
        jump_times = []
    jump_ts = (
        jnp.asarray(jump_times, dtype=float) if (use_jump_ts and jump_times) else None
    )

    flow_idx = jnp.array(list(ctrl.flow_indices))
    Cin = jnp.array(mb.Cin)
    Cin_mod = jnp.array(mb.Cin_modeled)
    V0 = _require_reactor_volume_above_threshold(
        jnp.array(float(process.volume.initial_volume)),
        context="initial reactor volume",
    )

    # Initial state: [c*_0, V_cont_0], where c*_0 == c_0.
    c0_reactor = jnp.array(
        [
            float(
                jnp.asarray(
                    process.reactor_medium.components[s].concentration.values[0]
                )
            )
            for s in mb.reactor_component_state_names
        ]
    )
    c0_pv = jnp.array(
        [
            float(jnp.asarray(pv.values.values[0]))
            if isinstance(pv.values, TimeSeries)
            else float(pv.values.value)
            for pv_name, pv in process.process_variables.items()
            if pv_name in mb.process_variable_state_names
        ],
        dtype=float,
    )
    c0 = jnp.concatenate([c0_reactor, c0_pv])
    c0 = c0.at[:n_reactor].set(jnp.maximum(c0[:n_reactor], 0.0))
    y0 = jnp.append(c0, jnp.array(0.0))
    state0_probe = jnp.append(c0, V0)
    controls0_probe = ctrl(jnp.array(t_start))
    _validate_rates_output_shapes(
        mb,
        rates_func,
        t=t_start,
        state=state0_probe,
        controls=controls0_probe,
    )

    scales = _compute_scale_factors(process, mb)
    state_scale = jnp.append(jnp.array(scales), 1.0)

    def _eval_adf(t):
        adf_vals = []
        for i in range(n_non_volume):
            adf_vals.append(
                _evaluate_with_boundary_start(adf_ts_list[i], t, side="left")
            )
        return jnp.stack(adf_vals)

    def _eval_dadf(t):
        dadf_vals = []
        for i in range(n_non_volume):
            dadf_vals.append(dadf_ts_list[i].evaluate(t, side="left"))
        return jnp.stack(dadf_vals)

    def _eval_fc_and_dfc(t):
        fc_vals = []
        dfc_vals = []
        for i in range(n_non_volume):
            fc_i = _evaluate_with_boundary_start(fc_ts_list[i], t, side="left")
            dfc_i = dfc_ts_list[i].evaluate(t, side="left")
            fc_vals.append(fc_i)
            dfc_vals.append(dfc_i)
        return jnp.stack(fc_vals), jnp.stack(dfc_vals)

    def rhs_cstar(t, state, args):
        c_star = state[:n_non_volume]
        V_cont = state[n_non_volume]

        adf = _adf_for_division(_eval_adf(t))
        fc, dfc = _eval_fc_and_dfc(t)
        c = (c_star + fc) / adf
        c_reactor = jnp.maximum(c[:n_reactor], 0.0)
        if mb.n_pv_states > 0:
            c_pv = c[n_reactor:]
        else:
            c_pv = jnp.zeros(0, dtype=c.dtype)

        V_disc = jnp.zeros(())
        if ev_times is not None:
            idx = jnp.searchsorted(ev_times, t, side="left")
            V_disc = jnp.where(idx > 0, ev_dV_cum[jnp.clip(idx - 1, 0)], 0.0)
        V = _require_reactor_volume_above_threshold(
            V0 + V_cont + V_disc,
            context="pseudospace integration volume",
        )

        u = ctrl(t)
        u_flow = u[flow_idx] if len(flow_idx) > 0 else jnp.zeros(mb.u_flow_size)
        state_rates = jnp.append(c, V)
        q, r = rates_func(t, state_rates, u)

        if bio_spline is not None:
            X_active = bio_spline(t)
            for spl in intra_splines:
                X_active = X_active - spl(t)
        else:
            X_active = c_reactor[mb.biomass_idx]
            for i in mb.intracellular_indices:
                X_active = X_active - c_reactor[i]
        X_active = jnp.maximum(X_active, _MIN_ACTIVE_BIOMASS)

        reaction = q * X_active
        reaction = _add_intracellular_to_biomass(
            reaction, q, X_active, mb.biomass_idx, mb.intracellular_indices
        )
        feed_term = jnp.zeros(n_reactor)
        dV_cont = jnp.zeros(())
        if mb.u_flow_size > 0:
            feed_term = feed_term + jnp.sum(
                (u_flow[:, None] / V) * (Cin - c_reactor[None, :]), axis=0
            )
            dV_cont = dV_cont + jnp.sum(u_flow)
        if mb.f_modeled_size > 0:
            f_mod = (
                batched_mod(t, nu=1)
                if batched_mod is not None
                else jnp.zeros(mb.f_modeled_size)
            )
            feed_term = feed_term + jnp.sum(
                (f_mod[:, None] / V) * (Cin_mod - c_reactor[None, :]), axis=0
            )
            dV_cont = dV_cont + jnp.sum(f_mod)

        r_reactor = r[:n_reactor]
        r_pv = r[n_reactor:]
        if mb.n_pv_states > 0 and len(mb.static_pv_indices) > 0:
            static_idx = jnp.array(mb.static_pv_indices, dtype=int)
            r_pv = r_pv.at[static_idx].set(0.0)
        dc_reactor = reaction + feed_term + r_reactor
        dc_pv = c_pv * 0.0 + r_pv
        dc = jnp.concatenate([dc_reactor, dc_pv])
        # c_star = c * adf - fc, so
        #   dc_star/dt = adf * dc/dt + c * d(adf)/dt - dfc/dt
        # The c · d(adf)/dt term matters whenever ADF varies smoothly between
        # events (continuous feed / sample-compensation factor); without it
        # the pseudospace integrator drifts off reference.
        dadf = _eval_dadf(t)
        dc_star = adf * dc + c * dadf - dfc
        return jnp.append(dc_star, dV_cont)

    def rhs_normalized(t, state_n, args):
        state = state_n * state_scale
        dstate = rhs_cstar(t, state, args)
        return dstate / state_scale

    term = diffrax.ODETerm(rhs_normalized)
    solver = diffrax.Tsit5()
    controller = diffrax.PIDController(rtol=rtol, atol=atol, jump_ts=jump_ts)
    dt0 = min(0.1, max((t_end - t_start) / 100.0, _MIN_SOLVER_DT0))

    @eqx.filter_jit
    def _solve(ts):
        return diffrax.diffeqsolve(
            term,
            solver,
            t0=t_start,
            t1=t_end,
            dt0=dt0,
            y0=y0 / state_scale,
            saveat=diffrax.SaveAt(ts=ts),
            stepsize_controller=controller,
            max_steps=max_steps,
        )

    sol = _solve(t_eval)
    ys = sol.ys * state_scale[None, :]
    c_star_out = ys[:, :n_non_volume]
    V_cont_out = ys[:, n_non_volume]
    t_output = _snap_times_to_discrete_events(t_eval, ev_times)

    adf_out = jax.vmap(_eval_adf)(t_output)
    fc_out, _ = jax.vmap(_eval_fc_and_dfc)(t_output)
    c_out = (c_star_out + fc_out) / _adf_for_division(adf_out)
    c_out = c_out.at[:, :n_reactor].set(jnp.maximum(c_out[:, :n_reactor], 0.0))

    if ev_times is not None:
        idx = jax.vmap(lambda t: jnp.searchsorted(ev_times, t, side="left"))(t_output)
        V_disc_out = jnp.where(idx > 0, ev_dV_cum[jnp.clip(idx - 1, 0)], 0.0)
    else:
        V_disc_out = jnp.zeros_like(t_eval)
    V_out = _require_reactor_volume_above_threshold(
        V0 + V_cont_out + V_disc_out,
        context="pseudospace output volume",
    )

    return {
        "t": t_eval,
        "c": c_out,
        "V": V_out,
        "stats": {"num_steps": int(sol.stats["num_steps"])},
        "_solve": _solve,
    }


def integrate_process(
    process: BioProcess,
    ctrl: ControlSplines,
    mb: RhsOde,
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
    mb:
        :class:`RhsOde` module.
    rates_func:
        Callable
        ``rates_func(t, state, controls) -> tuple[q, r]`` where
        ``q`` has one biomass-specific rate per reactor-component state and
        ``r`` has one additive physical rate per non-volume state. The first
        ``n_reactor_states`` entries of ``r`` align to the same
        reactor-component states as ``q``; tail entries cover process-variable
        states.
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
    n_reactor = mb.n_reactor_states
    n_non_volume = mb.r_size
    rates_func = _require_rates_func(rates_func)

    # Per-state scale factors for numerical conditioning
    scales = _compute_scale_factors(process, mb)
    scale_vec = jnp.array(scales)  # (n_non_volume,)
    state_scale = jnp.append(scale_vec, 1.0)  # [scales..., 1.0]

    # Build modeled flow splines (batched)
    cum_splines_mod_list = []
    for fn in mb.modeled_flow_names:
        vc = process.volume.volume_changes[fn]
        sp = _timeseries_to_interpax_spline(vc.values)
        cum_splines_mod_list.append(sp)
    batched_mod = (
        _batch_splines(cum_splines_mod_list, t_start, t_end)
        if cum_splines_mod_list
        else None
    )

    # Extract discrete events and build segment boundaries
    events = extract_discrete_events(process, mb)
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
            for s in mb.reactor_component_state_names
        ]
    )
    c0_pv = jnp.array(
        [
            float(jnp.asarray(pv.values.values[0]))
            if isinstance(pv.values, TimeSeries)
            else float(pv.values.value)
            for pv_name, pv in process.process_variables.items()
            if pv_name in mb.process_variable_state_names
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
        mb,
        rates_func,
        t=t_start,
        state=state0_probe,
        controls=controls0_probe,
    )

    # Build RHS in original coordinates, then wrap for normalized state
    rhs_original = _build_segment_rhs(
        mb,
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
                    if mb.n_pv_states > 0
                    else jnp.zeros(0, dtype=state.dtype)
                )
                V = state[mb.volume_idx]
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
            ys_seg[:, mb.volume_idx],
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
