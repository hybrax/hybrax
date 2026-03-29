"""
Mechanistic API for BPbench.

Provides JAX/Equinox-compatible modules for building continuous-time control
functions and ODE right-hand sides directly from a :class:`~bpbench.BioProcess`.

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
    Returns an ``eqx.Module`` whose ``__call__(c, q, u_flow, f_modeled)``
    computes the ODE RHS ``dc/dt`` (including ``dV/dt``).
    Uncontrolled continuous volume changes (modeled feeds) are supported via
    the optional ``f_modeled`` argument.

extract_discrete_events(process, mb) -> list[dict]
    Extract discrete events (sampling, bolus feeds) from a BioProcess.

build_q_func(process, ctrl, mb, conc_splines) -> Callable
    Build an analytical, JIT-compilable q(t) callable from splines.

estimate_specific_rates(process, ctrl, mb, conc_splines, t_eval) -> jnp.ndarray
    Estimate specific rates q(t) via ODE RHS inversion (convenience wrapper).

integrate_process(process, ctrl, mb, q_func, t_eval) -> dict
integrate_process_pseudospace(process, ctrl, mb, q_func, t_eval) -> dict
    Full hybrid ODE integration with discrete event handling.

Usage with JIT
--------------
Both modules are equinox Modules (JAX pytrees).  Use ``eqx.filter_jit``
to compile them::

    import equinox as eqx
    ctrl = get_control_splines(process)
    mb   = get_rhs_ode(process)

    u      = eqx.filter_jit(ctrl)(t)
    dc_dt  = eqx.filter_jit(mb)(c, q, u_flow)
    # With modeled flows (e.g. base feed):
    dc_dt  = eqx.filter_jit(mb)(c, q, u_flow, f_modeled)
"""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional, Tuple

import diffrax
import equinox as eqx
import interpax
import jax
import jax.numpy as jnp

from .dataclasses import (
    BioProcess,
    FeedVolumeChange,
    SampleVolumeChange,
    StaticVariable,
    TimeSeries,
)
from .splines import (
    make_interpax_spline,
    build_interpax_spline,
    build_backtransform_spline,
)


# ---------------------------------------------------------------------------
# Batched spline helpers
# ---------------------------------------------------------------------------

_DEFAULT_BATCH_KNOTS = 128


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
            Scalar time (same unit as the underlying :class:`~bpbench.TimeSeries`).

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

    The state vector is ``c = [c_species..., V]`` where the last element is
    the reactor volume.  Biomass is always at index 0.

    Attributes
    ----------
    c_size : int
        ``n_species + 1`` (species concentrations + volume).
    q_size : int
        ``n_species`` — number of specific rates (aligned with
        :attr:`species_names`).
    u_flow_size : int
        Number of continuous controlled flow streams.
    f_modeled_size : int
        Number of continuous uncontrolled (modeled) flow streams.
    output_size : int
        Same as :attr:`c_size`.
    species_names : tuple[str, ...]
        Ordering of species in *c* and *q*.  Biomass is always first.
    flow_names : tuple[str, ...]
        Ordering of continuous controlled flow streams in *u_flow*.
    modeled_flow_names : tuple[str, ...]
        Ordering of continuous uncontrolled (modeled) flow streams in
        *f_modeled*.
    biomass_idx : int
        Index of ``"biomass"`` in :attr:`species_names` (always 0).
    intracellular_indices : tuple[int, ...]
        Indices of intracellular species in :attr:`species_names`.
        Intracellular components (e.g., intracellular product) accumulate
        inside the cells.  Active biomass is therefore:
        ``X_active = c[biomass_idx] - sum(c[i] for i in intracellular_indices)``.
    Cin : jnp.ndarray, shape (n_flows, n_species)
        Feed composition matrix for controlled flows: ``Cin[k, i]`` is the
        concentration of species *i* in controlled feed stream *k*.
    Cin_modeled : jnp.ndarray, shape (n_modeled_flows, n_species)
        Feed composition matrix for modeled (uncontrolled) flows.

    Notes
    -----
    JIT usage::

        import equinox as eqx
        mb    = get_rhs_ode(process)
        dc_dt = eqx.filter_jit(mb)(c, q, u_flow)
        # With modeled flows:
        dc_dt = eqx.filter_jit(mb)(c, q, u_flow, f_modeled)
    """

    c_size: int = eqx.field(static=True)
    q_size: int = eqx.field(static=True)
    u_flow_size: int = eqx.field(static=True)
    f_modeled_size: int = eqx.field(static=True)
    output_size: int = eqx.field(static=True)
    species_names: tuple = eqx.field(static=True)
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
    ) -> jnp.ndarray:
        """Compute the ODE RHS ``dc/dt``.

        Parameters
        ----------
        c:
            State vector ``[c_species..., V]``, shape ``(c_size,)``.
            Biomass (measured) is at index 0; intracellular components follow
            in the positions given by :attr:`intracellular_indices`.
        q:
            Specific rates aligned with :attr:`species_names`, shape
            ``(q_size,)``.
        u_flow:
            Volumetric flow rates for each continuous controlled feed stream
            (volume / time, matching the units of the stored
            ``VolumeChange``), shape ``(u_flow_size,)``.
        f_modeled:
            Volumetric flow rates for each continuous uncontrolled (modeled)
            feed stream, shape ``(f_modeled_size,)``.  Pass
            ``jnp.zeros(0)`` when there are no modeled flows.

        Returns
        -------
        jnp.ndarray, shape ``(output_size,)``
            ``dc/dt`` with ``dV/dt`` as the last element.

        Notes
        -----
        ODE RHS implemented:

        .. math::

            X_{active} = c_{biomass} - \\sum_{i \\in intracellular} c_i

            \\frac{dc_i}{dt} = q_i \\cdot X_{active}
                + \\sum_k \\frac{f_k}{V}\\,(C_{in,k,i} - c_i)

            \\frac{dV}{dt} = \\sum_k f_k

        where :math:`X_{active}` is the active biomass concentration
        (measured biomass minus intracellular component concentrations),
        and the sums over *k* include both controlled and modeled flows.
        """
        c_species = c[: self.q_size]
        V = c[self.q_size]

        # Active biomass: measured biomass minus intracellular components
        X_measured = c[self.biomass_idx]
        if len(self.intracellular_indices) > 0:
            intracellular_sum = jnp.sum(
                c_species[jnp.array(self.intracellular_indices)]
            )
        else:
            intracellular_sum = jnp.zeros(())
        X_active = X_measured - intracellular_sum

        # Reaction contribution: q_i * X_active
        reaction = q * X_active

        # Controlled feed / dilution contribution (zero when u_flow_size == 0)
        # u_flow: (n_flows,),  Cin: (n_flows, n_species)
        feed_contrib = u_flow[:, None] * (self.Cin - c_species[None, :])
        feed_term = jnp.sum(feed_contrib, axis=0) / V
        dV = jnp.sum(u_flow)

        # Modeled (uncontrolled) feed contribution
        if self.f_modeled_size > 0:
            modeled_contrib = f_modeled[:, None] * (
                self.Cin_modeled - c_species[None, :]
            )
            feed_term = feed_term + jnp.sum(modeled_contrib, axis=0) / V
            dV = dV + jnp.sum(f_modeled)

        dc_species = reaction + feed_term

        return jnp.append(dc_species, dV)


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
        A :class:`~bpbench.BioProcess` instance.

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
        if vc.interpolator is not None:
            sp = build_interpax_spline(vc.interpolator)[0][0]
        else:
            sp = make_interpax_spline(
                jnp.asarray(vc.values.times),
                jnp.asarray(vc.values.values),
            )
        control_names.append(vc_name)
        flow_indices.append(idx)
        splines.append(sp)
        is_derivative_list.append(True)
        idx += 1

    # 2) Controlled process variables → direct spline value
    for pv_name, pv in process.process_variables.items():
        if not pv.is_controlled:
            continue
        if pv.interpolator is not None:
            sp = build_interpax_spline(pv.interpolator)[0][0]
        elif isinstance(pv.values, TimeSeries):
            sp = make_interpax_spline(
                jnp.asarray(pv.values.times),
                jnp.asarray(pv.values.values),
            )
        else:
            # StaticVariable: constant spline over the full process time span
            t_start = float(process.time_axis.start)
            t_end = float(process.time_axis.end)
            sp = make_interpax_spline(
                jnp.array([t_start, t_end]),
                jnp.array([float(pv.values.value), float(pv.values.value)]),
            )
        control_names.append(pv_name)
        ctrl_indices.append(idx)
        splines.append(sp)
        is_derivative_list.append(False)
        idx += 1

    # Batch all splines into a single PPoly for vectorized evaluation
    t_start = float(process.time_axis.start)
    t_end = float(process.time_axis.end)
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
    """Build a :class:`RhsOde` module from a :class:`BioProcess`.

    Parameters
    ----------
    process:
        A :class:`~bpbench.BioProcess` instance.  The reactor medium must
        contain a component named ``"biomass"`` (case-insensitive).

    Returns
    -------
    RhsOde
        An ``eqx.Module`` whose ``__call__(c, q, u_flow, f_modeled)``
        computes the ODE RHS ``dc/dt``.

    Raises
    ------
    ValueError
        If no ``"biomass"`` component is found in the reactor medium.
    """
    # --- Species ordering: biomass always at index 0 ---
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
    species_names: Tuple[str, ...] = (biomass_name,) + tuple(other_names)
    n_species = len(species_names)
    biomass_idx: int = 0  # always 0 by construction

    # --- Intracellular indices ---
    intracellular_indices: List[int] = []
    for i, name in enumerate(species_names):
        comp = process.reactor_medium.components[name]
        if comp.is_intracellular:
            intracellular_indices.append(i)

    # --- Helper to build a Cin matrix for a list of volume-change names ---
    def _build_cin(vc_names):
        n = len(vc_names)
        Cin = jnp.zeros((n, n_species), dtype=float)
        for k, vc_name in enumerate(vc_names):
            vc = process.volume.volume_changes[vc_name]
            if not isinstance(vc, FeedVolumeChange):
                continue  # SampleVolumeChange has no feed medium
            feed = vc.feed_medium
            for j, sp_name in enumerate(species_names):
                if sp_name not in feed.components:
                    continue
                conc = feed.components[sp_name].concentration
                if isinstance(conc, StaticVariable):
                    Cin = Cin.at[k, j].set(float(conc.value))
                else:
                    raise NotImplementedError(
                        "TimeSeries feed concentrations are not yet supported in get_rhs_ode. "
                        f"Found TimeSeries for species '{sp_name}' in feed '{feed.name}' of volume change '{vc_name}'."
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
        c_size=n_species + 1,
        q_size=n_species,
        u_flow_size=len(flow_names),
        f_modeled_size=len(modeled_flow_names),
        output_size=n_species + 1,
        species_names=tuple(species_names),
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
        A :class:`~bpbench.BioProcess` instance.
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
          ``mb.species_names`` (None for sampling)
        - ``source`` (str): name of the originating VolumeChange
    """
    events: List[Dict[str, Any]] = []
    n_sp = len(mb.species_names)

    for vc_name, vc in process.volume.volume_changes.items():
        if vc.is_continuous:
            continue

        tp = jnp.asarray(vc.values.times, dtype=float)
        vv = jnp.asarray(vc.values.values, dtype=float)

        for t_event, dV_event in zip(tp, vv):
            if abs(dV_event) < 1e-15:
                continue

            if dV_event > 0 and isinstance(vc, FeedVolumeChange):
                Cin_event = jnp.zeros(n_sp)
                if vc.feed_medium is not None:
                    for j, sp_name in enumerate(mb.species_names):
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

    events.sort(key=lambda e: e["t"])
    return events


def build_conc_splines(
    process: BioProcess,
    mb: "RhsOde",
) -> Dict[str, Any]:
    """Build concentration splines from stored backtransform splines or raw data.

    Uses the pre-fitted ``comp.interpolator`` (backtransform spline built during
    the pseudobatch step) when available.  The backtransform spline maps
    from the continuous pseudobatch domain back to reactor concentrations,
    correctly handling bolus feeds and sampling events without requiring
    per-segment splines.

    Falls back to fitting a raw ``interpax.CubicSpline`` from the
    concentration time series when no stored spline is available.

    Parameters
    ----------
    process:
        A :class:`~bpbench.BioProcess` instance.
    mb:
        A :class:`RhsOde` module (provides species ordering).

    Returns
    -------
    dict
        Mapping species name → callable spline.
    """
    conc_splines: Dict[str, Any] = {}

    for sp_name in mb.species_names:
        comp = process.reactor_medium.components[sp_name]
        if (
            comp.interpolator is not None
            and comp.interpolator.interpolator_metadata
            and "transform" in comp.interpolator.interpolator_metadata
        ):
            # Backtransform spline: continuous in pseudobatch domain,
            # correctly accounts for bolus feeds / sampling.
            conc_splines[sp_name] = build_backtransform_spline(comp.interpolator)
        elif comp.interpolator is not None:
            conc_splines[sp_name] = build_interpax_spline(comp.interpolator)[0][0]
        else:
            ts = comp.concentration
            conc_splines[sp_name] = make_interpax_spline(
                jnp.asarray(ts.times, dtype=float),
                jnp.asarray(ts.values, dtype=float),
            )

    return conc_splines


# ---------------------------------------------------------------------------
# Analytical q(t) builder and specific rate estimation
# ---------------------------------------------------------------------------


def build_q_func(
    process: BioProcess,
    ctrl: ControlSplines,
    mb: RhsOde,
    conc_splines: Dict[str, Any],
) -> Callable:
    """Build an analytical, JIT-compilable ``q(t)`` callable.

    Returns a function ``q(t) -> jnp.ndarray`` that evaluates specific rates
    at *any* time ``t`` using the analytical spline derivatives:

    .. math::
        q_i(t) = \\frac{dc_i/dt - \\text{feed\\_term}_i}{X_{active}}

    All components (concentration derivatives, volume, flow rates, active
    biomass) are evaluated analytically from splines and discrete-event
    data via ``jnp.searchsorted``.  The returned callable is compatible
    with ``jax.jit``.

    Parameters
    ----------
    process:
        A :class:`~bpbench.BioProcess` instance.
    ctrl:
        :class:`ControlSplines` module for evaluating control signals.
    mb:
        :class:`RhsOde` module (provides species ordering, Cin, etc.).
    conc_splines:
        Dict mapping species name → callable spline that supports
        ``spline(t)`` and ``spline.derivative()(t)``.

    Returns
    -------
    Callable
        ``q_func(t) -> jnp.ndarray`` of shape ``(n_species,)``.
    """
    n_sp = mb.q_size
    t_start = float(process.time_axis.start)
    t_end = float(process.time_axis.end)

    # -- Cumulative volume splines for continuous flows --
    cum_splines_ctrl = []
    for fn in mb.flow_names:
        vc = process.volume.volume_changes[fn]
        if vc.interpolator is not None:
            sp = build_interpax_spline(vc.interpolator)[0][0]
        else:
            sp = make_interpax_spline(
                jnp.asarray(vc.values.times),
                jnp.asarray(vc.values.values),
            )
        cum_splines_ctrl.append(sp)

    cum_splines_mod = []
    for fn in mb.modeled_flow_names:
        vc = process.volume.volume_changes[fn]
        if vc.interpolator is not None:
            sp = build_interpax_spline(vc.interpolator)[0][0]
        else:
            sp = make_interpax_spline(
                jnp.asarray(vc.values.times),
                jnp.asarray(vc.values.values),
            )
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

    V0 = jnp.array(float(process.volume.initial_volume))
    Cin = jnp.array(mb.Cin)
    Cin_mod = jnp.array(mb.Cin_modeled)

    # -- Per-species concentration spline evaluators --
    # Uses original BacktransformSpline evaluations (exact piecewise-linear fc)
    # to avoid Gibbs oscillation from cubic resampling of step-like fc data.
    conc_evals = [conc_splines[s] for s in mb.species_names]
    deriv_evals = [conc_splines[s].derivative() for s in mb.species_names]

    biomass_idx = mb.biomass_idx
    intra_idx = mb.intracellular_indices
    u_flow_size = mb.u_flow_size
    f_mod_size = mb.f_modeled_size

    def q_func(t):
        # Concentrations — per-species spline evaluation
        c_t = jnp.stack([conc_evals[i](t) for i in range(n_sp)])
        c_t = jnp.maximum(c_t, 0.0)

        # Concentration derivatives (analytical) — per-species
        dc_dt = jnp.stack([deriv_evals[i](t) for i in range(n_sp)])

        # Volume: V0 + continuous flows + discrete events
        V_t = V0
        if batched_vol is not None:
            V_t = V_t + jnp.sum(batched_vol(t))  # single batched eval
        if ev_times is not None:
            idx = jnp.searchsorted(ev_times, t, side="right")
            V_t = V_t + jnp.where(idx > 0, ev_dV_cum[jnp.clip(idx - 1, 0)], 0.0)
        V_t = jnp.maximum(V_t, jnp.array(1e-10))

        # Flow rates (derivatives of cumulative volume splines) — batched
        if u_flow_size > 0 or f_mod_size > 0:
            all_flow_rates = batched_vol(t, nu=1)  # (n_vol,)
            u_flow = all_flow_rates[:u_flow_size] if u_flow_size > 0 else jnp.zeros(0)
            f_mod = all_flow_rates[u_flow_size:] if f_mod_size > 0 else jnp.zeros(0)
        else:
            u_flow = jnp.zeros(0)
            f_mod = jnp.zeros(0)

        # Feed term: sum_k (f_k / V) * (C_in_k - c)
        feed_term = jnp.zeros(n_sp)
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
        X_active = jnp.maximum(X_active, jnp.array(1e-6))

        return (dc_dt - feed_term) / X_active

    return q_func


def estimate_specific_rates(
    process: BioProcess,
    ctrl: ControlSplines,
    mb: RhsOde,
    conc_splines: Dict[str, Any],
    t_eval: jnp.ndarray,
) -> jnp.ndarray:
    """Estimate specific rates q(t) via ODE RHS inversion.

    Convenience wrapper around :func:`build_q_func` that evaluates the
    analytical rate function at the given time points.

    Parameters
    ----------
    process:
        A :class:`~bpbench.BioProcess` instance.
    ctrl:
        :class:`ControlSplines` module for evaluating control signals.
    mb:
        :class:`RhsOde` module (provides species ordering, Cin, etc.).
    conc_splines:
        Dict mapping species name → callable spline that supports
        ``spline(t)`` and ``spline.derivative()(t)``.
    t_eval:
        1-D array of time points at which to estimate q.

    Returns
    -------
    jnp.ndarray, shape (len(t_eval), n_species)
        Estimated specific rates at each time point.
    """
    t_eval = jnp.asarray(t_eval, dtype=float)
    q_func = build_q_func(process, ctrl, mb, conc_splines)
    q_func_jit = eqx.filter_jit(q_func)
    return jax.vmap(q_func_jit)(t_eval)


# ---------------------------------------------------------------------------
# Full hybrid ODE integration
# ---------------------------------------------------------------------------


def _build_segment_rhs(mb, ctrl, q_func, batched_mod, conc_eval_list=None):
    """Build the ODE right-hand side function for a segment.

    Parameters
    ----------
    batched_mod : interpax.PPoly or None
        Batched PPoly for modeled (uncontrolled) cumulative volume splines.
        Evaluate with ``batched_mod(t, nu=1)`` to get flow rates.
    conc_eval_list : list of callables or None
        Per-species concentration spline callables (e.g. BacktransformSpline).
        If provided, the RHS uses spline-evaluated concentrations for the
        active-biomass term (``X_active``) instead of the ODE state.
    """
    flow_idx = jnp.array(list(ctrl.flow_indices))

    if conc_eval_list is not None:
        biomass_idx = mb.biomass_idx
        intra_idx = (
            jnp.array(mb.intracellular_indices) if mb.intracellular_indices else None
        )

    def rhs(t, state, args):
        u = ctrl(t)
        u_flow = u[flow_idx] if len(flow_idx) > 0 else jnp.zeros(mb.u_flow_size)
        q = q_func(t)

        # Modeled flow rates via batched PPoly derivative
        if batched_mod is not None:
            f_mod = batched_mod(t, nu=1)  # (n_modeled_flows,)
        else:
            f_mod = jnp.zeros(mb.f_modeled_size)

        if conc_eval_list is not None:
            all_conc = jnp.stack([sp(t) for sp in conc_eval_list])
            X_active = all_conc[biomass_idx]
            if intra_idx is not None:
                X_active = X_active - jnp.sum(all_conc[intra_idx])
            X_active = jnp.maximum(X_active, 1e-6)

            c_species = state[: mb.q_size]
            V = state[mb.q_size]

            reaction = q * X_active

            feed_term = jnp.zeros(mb.q_size)
            dV = jnp.zeros(())
            if mb.u_flow_size > 0:
                feed_contrib = u_flow[:, None] * (mb.Cin - c_species[None, :])
                feed_term = feed_term + jnp.sum(feed_contrib, axis=0) / V
                dV = dV + jnp.sum(u_flow)
            if mb.f_modeled_size > 0:
                mod_contrib = f_mod[:, None] * (mb.Cin_modeled - c_species[None, :])
                feed_term = feed_term + jnp.sum(mod_contrib, axis=0) / V
                dV = dV + jnp.sum(f_mod)

            dc_species = reaction + feed_term
            return jnp.append(dc_species, dV)
        else:
            return mb(state, q, u_flow, f_mod)

    return rhs


def _compute_scale_factors(process: BioProcess, mb: "RhsOde") -> jnp.ndarray:
    """Compute per-species scale factors for numerical conditioning.

    Returns an array of shape (n_species,) where each entry is the
    max absolute concentration for that species (clamped to >= 1.0 so
    species with small values are not artificially inflated).
    """
    scales = jnp.ones(mb.q_size)
    for i, sp_name in enumerate(mb.species_names):
        vals = jnp.asarray(
            process.reactor_medium.components[sp_name].concentration.values,
            dtype=float,
        )
        s = float(jnp.max(jnp.abs(vals)))
        if s > 1.0:
            scales = scales.at[i].set(s)
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
    transforms: List[Dict[str, Any]] = []
    for sp_name in mb.species_names:
        comp = process.reactor_medium.components[sp_name]
        rep = comp.interpolator
        tr = (
            rep.interpolator_metadata.get("transform")
            if rep is not None and rep.interpolator_metadata is not None
            else None
        )
        if tr is None:
            transforms.append({"kind": "identity"})
            continue

        adf_t = jnp.asarray(tr["adf_times"], dtype=float)
        adf_v = jnp.asarray(tr["adf_values"], dtype=float)
        fc_t = jnp.asarray(tr["feed_corr_times"], dtype=float)
        fc_v = jnp.asarray(tr["feed_corr_values"], dtype=float)
        fc_interp = str(tr.get("feed_corr_interp", "linear"))

        fc_spline = None
        if fc_interp == "cubic":
            fc_spline = make_interpax_spline(fc_t, fc_v)

        transforms.append(
            {
                "kind": "pb",
                "adf_t": adf_t,
                "adf_v": adf_v,
                "fc_t": fc_t,
                "fc_v": fc_v,
                "fc_interp": fc_interp,
                "fc_spline": fc_spline,
            }
        )
        if fc_interp != "cubic":
            if len(fc_t) < 2:
                transforms[-1]["fc_slopes"] = jnp.array([0.0], dtype=float)
            else:
                fc_dt = jnp.diff(fc_t)
                fc_slopes = jnp.diff(fc_v) / jnp.maximum(fc_dt, 1e-12)
                median_dt = jnp.median(fc_dt)
                fc_slopes = jnp.where(fc_dt < 0.1 * median_dt, 0.0, fc_slopes)
                transforms[-1]["fc_slopes"] = fc_slopes

    return transforms


def integrate_process_pseudospace(
    process: BioProcess,
    ctrl: ControlSplines,
    mb: RhsOde,
    q_func: Callable,
    t_eval: jnp.ndarray,
    *,
    conc_splines: Optional[Dict[str, Any]] = None,
    rtol: float = 1e-6,
    atol: float = 1e-8,
    use_jump_ts: bool = True,
    max_steps: int = 16384,
) -> Dict[str, Any]:
    """Integrate in c* (pseudo-batch) space with a single ``diffeqsolve``."""
    t_eval = jnp.asarray(t_eval, dtype=float)
    t_start = float(process.time_axis.start)
    t_end = float(process.time_axis.end)
    n_sp = mb.q_size

    transforms = _build_pseudobatch_transforms(process, mb)

    # Shared ADF is process-level and identical across species in pseudobatch.
    adf_t = None
    adf_v = None
    for tr in transforms:
        if tr["kind"] == "pb":
            adf_t = tr["adf_t"]
            adf_v = tr["adf_v"]
            break
    if adf_t is None:
        adf_t = jnp.asarray([t_start, t_end], dtype=float)
        adf_v = jnp.asarray([1.0, 1.0], dtype=float)

    # Per-species feed-correction representations.
    fc_is_cubic: List[bool] = []
    fc_cubic: List[Optional[interpax.CubicSpline]] = []
    fc_cubic_deriv: List[Optional[interpax.PPoly]] = []
    fc_t_list: List[jnp.ndarray] = []
    fc_v_list: List[jnp.ndarray] = []
    for tr in transforms:
        if tr["kind"] == "pb" and tr.get("fc_interp") == "cubic":
            fc_is_cubic.append(True)
            fc_cubic.append(tr["fc_spline"])
            fc_cubic_deriv.append(tr["fc_spline"].derivative())
            fc_t_list.append(jnp.asarray([t_start, t_end], dtype=float))
            fc_v_list.append(jnp.asarray([0.0, 0.0], dtype=float))
        elif tr["kind"] == "pb":
            fc_is_cubic.append(False)
            fc_cubic.append(None)
            fc_cubic_deriv.append(None)
            fc_t_list.append(jnp.asarray(tr["fc_t"], dtype=float))
            fc_v_list.append(jnp.asarray(tr["fc_v"], dtype=float))
        else:
            fc_is_cubic.append(False)
            fc_cubic.append(None)
            fc_cubic_deriv.append(None)
            fc_t_list.append(jnp.asarray([t_start, t_end], dtype=float))
            fc_v_list.append(jnp.asarray([0.0, 0.0], dtype=float))

    if conc_splines is not None:
        bio_spline = conc_splines[mb.species_names[mb.biomass_idx]]
        intra_splines = [
            conc_splines[mb.species_names[i]] for i in mb.intracellular_indices
        ]
    else:
        bio_spline = None
        intra_splines = []

    # Modeled (uncontrolled) continuous flows.
    cum_splines_mod: List[interpax.CubicSpline] = []
    for fn in mb.modeled_flow_names:
        vc = process.volume.volume_changes[fn]
        if vc.interpolator is not None:
            sp = build_interpax_spline(vc.interpolator)[0][0]
        else:
            sp = make_interpax_spline(
                jnp.asarray(vc.values.times),
                jnp.asarray(vc.values.values),
            )
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
    V0 = jnp.array(float(process.volume.initial_volume))

    # Initial state: [c*_0, V_cont_0], where c*_0 == c_0.
    c0 = jnp.array(
        [
            float(
                jnp.asarray(
                    process.reactor_medium.components[s].concentration.values[0]
                )
            )
            for s in mb.species_names
        ]
    )
    c0 = jnp.maximum(c0, 0.0)
    y0 = jnp.append(c0, jnp.array(0.0))

    scales = _compute_scale_factors(process, mb)
    state_scale = jnp.append(jnp.array(scales), 1.0)

    def _eval_fc_and_dfc(t):
        fc_vals = []
        dfc_vals = []
        for i in range(n_sp):
            if fc_is_cubic[i]:
                fc_i = fc_cubic[i](t)
                dfc_i = fc_cubic_deriv[i](t)
            else:
                ft = fc_t_list[i]
                fv = fc_v_list[i]
                fc_i = jnp.interp(t, ft, fv)
                tr = transforms[i]
                if tr["kind"] == "pb" and "fc_slopes" in tr:
                    sl = tr["fc_slopes"]
                    idx = jnp.clip(jnp.searchsorted(ft, t) - 1, 0, sl.shape[0] - 1)
                    dfc_i = sl[idx]
                else:
                    _, dfc_i = _piecewise_linear_value_and_slope(t, ft, fv)
            fc_vals.append(fc_i)
            dfc_vals.append(dfc_i)
        return jnp.stack(fc_vals), jnp.stack(dfc_vals)

    def rhs_cstar(t, state, args):
        c_star = state[:n_sp]
        V_cont = state[n_sp]

        adf = jnp.interp(t, adf_t, adf_v)
        adf = jnp.maximum(adf, 1e-12)
        fc, dfc = _eval_fc_and_dfc(t)
        c = (c_star + fc) / adf
        c = jnp.maximum(c, 0.0)

        V_disc = jnp.zeros(())
        if ev_times is not None:
            idx = jnp.searchsorted(ev_times, t, side="right")
            V_disc = jnp.where(idx > 0, ev_dV_cum[jnp.clip(idx - 1, 0)], 0.0)
        V = jnp.maximum(V0 + V_cont + V_disc, 1e-10)

        u = ctrl(t)
        u_flow = u[flow_idx] if len(flow_idx) > 0 else jnp.zeros(mb.u_flow_size)
        q = q_func(t)

        if bio_spline is not None:
            X_active = bio_spline(t)
            for spl in intra_splines:
                X_active = X_active - spl(t)
        else:
            X_active = c[mb.biomass_idx]
            for i in mb.intracellular_indices:
                X_active = X_active - c[i]
        X_active = jnp.maximum(X_active, 1e-6)

        reaction = q * X_active
        feed_term = jnp.zeros(n_sp)
        dV_cont = jnp.zeros(())
        if mb.u_flow_size > 0:
            feed_term = feed_term + jnp.sum(
                (u_flow[:, None] / V) * (Cin - c[None, :]), axis=0
            )
            dV_cont = dV_cont + jnp.sum(u_flow)
        if mb.f_modeled_size > 0:
            f_mod = (
                batched_mod(t, nu=1)
                if batched_mod is not None
                else jnp.zeros(mb.f_modeled_size)
            )
            feed_term = feed_term + jnp.sum(
                (f_mod[:, None] / V) * (Cin_mod - c[None, :]), axis=0
            )
            dV_cont = dV_cont + jnp.sum(f_mod)

        dc = reaction + feed_term
        dc_star = adf * dc - dfc
        return jnp.append(dc_star, dV_cont)

    def rhs_normalized(t, state_n, args):
        state = state_n * state_scale
        dstate = rhs_cstar(t, state, args)
        return dstate / state_scale

    term = diffrax.ODETerm(rhs_normalized)
    solver = diffrax.Tsit5()
    controller = diffrax.PIDController(rtol=rtol, atol=atol, jump_ts=jump_ts)
    dt0 = min(0.1, max((t_end - t_start) / 100.0, 1e-6))

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
    c_star_out = ys[:, :n_sp]
    V_cont_out = ys[:, n_sp]

    adf_out = jnp.interp(t_eval, adf_t, adf_v)
    fc_out, _ = jax.vmap(_eval_fc_and_dfc)(t_eval)
    c_out = (c_star_out + fc_out) / jnp.maximum(adf_out[:, None], 1e-12)
    c_out = jnp.maximum(c_out, 0.0)

    if ev_times is not None:
        idx = jax.vmap(lambda t: jnp.searchsorted(ev_times, t, side="right"))(t_eval)
        V_disc_out = jnp.where(idx > 0, ev_dV_cum[jnp.clip(idx - 1, 0)], 0.0)
    else:
        V_disc_out = jnp.zeros_like(t_eval)
    V_out = jnp.maximum(V0 + V_cont_out + V_disc_out, 1e-10)

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
    q_func: Callable,
    t_eval: jnp.ndarray,
    *,
    conc_splines: Optional[Dict[str, Any]] = None,
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
        A :class:`~bpbench.BioProcess` instance.
    ctrl:
        :class:`ControlSplines` module.
    mb:
        :class:`RhsOde` module.
    q_func:
        Callable ``q_func(t) -> jnp.ndarray`` returning specific rates
        aligned with ``mb.species_names``.
    t_eval:
        1-D array of time points at which to record the solution.
    conc_splines:
        Optional dict mapping species name → callable spline.  When
        provided, the ODE RHS uses the spline-evaluated biomass (and
        intracellular) concentrations for computing ``X_active`` instead
        of the ODE state.  This prevents exponential error amplification
        when biomass spans many orders of magnitude.
    rtol, atol:
        Relative and absolute tolerances for the ODE solver.
    max_steps:
        Maximum number of ODE solver steps per segment.

    Returns
    -------
    dict
        ``{'t': jnp.ndarray, 'c': jnp.ndarray, 'V': jnp.ndarray}``
        where ``c`` has shape ``(len(t_eval), n_species)`` and ``V`` has
        shape ``(len(t_eval),)``.
    """
    t_eval = jnp.asarray(t_eval, dtype=float)
    t_start = float(process.time_axis.start)
    t_end = float(process.time_axis.end)
    n_sp = mb.q_size

    # Per-species scale factors for numerical conditioning
    scales = _compute_scale_factors(process, mb)
    scale_vec = jnp.array(scales)  # (n_sp,)
    state_scale = jnp.append(scale_vec, 1.0)  # [scales..., 1.0]
    state_dim = n_sp + 1

    # Build modeled flow splines (batched)
    cum_splines_mod_list = []
    for fn in mb.modeled_flow_names:
        vc = process.volume.volume_changes[fn]
        if vc.interpolator is not None:
            sp = build_interpax_spline(vc.interpolator)[0][0]
        else:
            sp = make_interpax_spline(
                jnp.asarray(vc.values.times),
                jnp.asarray(vc.values.values),
            )
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
    c0 = jnp.array(
        [
            float(
                jnp.asarray(
                    process.reactor_medium.components[s].concentration.values[0]
                )
            )
            for s in mb.species_names
        ]
    )
    c0 = jnp.maximum(c0, 0.0)
    V0 = float(process.volume.initial_volume)

    # Build per-species concentration spline evaluators for X_active
    # (avoids cubic resampling of piecewise-linear fc in batched approach)
    if conc_splines is not None:
        conc_eval_list = [conc_splines[s] for s in mb.species_names]
    else:
        conc_eval_list = None

    # Build RHS in original coordinates, then wrap for normalized state
    rhs_original = _build_segment_rhs(mb, ctrl, q_func, batched_mod, conc_eval_list)

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
            if t_seg[0] > t_lo + 1e-12:
                t_seg = jnp.concatenate([jnp.array([t_lo]), t_seg])
            if t_seg[-1] < t_hi - 1e-12:
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
    ev_Cin_arr = jnp.zeros((n_seg, max_ev, n_sp))

    for i in range(n_seg - 1):
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
                c = state[:n_sp]
                V = state[n_sp]
                V_new = V + dV
                c_bolus = (c * V + Cin * dV) / jnp.maximum(V_new, 1e-10)
                c_new = jnp.where(is_bolus, c_bolus, c)
                V_new = jnp.maximum(V_new, 1e-10)
                c_new = jnp.maximum(c_new, 0.0)
                new_state = jnp.append(c_new, V_new)
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

    t_segments = []
    c_segments = []
    V_segments = []

    for seg_idx in range(n_seg):
        n_valid = seg_t_valid_len[seg_idx]
        ys_seg = all_ys_orig[seg_idx, :n_valid, :]

        c_seg = jnp.maximum(ys_seg[:, :n_sp], 0.0)
        V_seg = jnp.maximum(ys_seg[:, n_sp], 1e-10)
        t_seg = seg_t_arrays[seg_idx]

        # Skip last point of non-final segments to avoid duplication
        if seg_idx < n_seg - 1:
            t_segments.append(t_seg[:-1])
            c_segments.append(c_seg[:-1])
            V_segments.append(V_seg[:-1])
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
