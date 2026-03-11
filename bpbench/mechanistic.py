"""
Mechanistic API for BPbench.

Provides JAX/Equinox-compatible modules for building continuous-time control
functions and mass-balance ODEs directly from a :class:`~bpbench.BioProcess`.

All modules are fully JAX-jittable via ``equinox.filter_jit``.  Spline
evaluation uses ``interpax.CubicSpline``, which is itself an
``equinox.Module`` and therefore a valid JAX pytree.

Public API
----------
get_control_splines(process) -> ControlSplines
    Returns an ``eqx.Module`` whose ``__call__(t)`` evaluates all controlled
    signals at time ``t``.  Continuous volume-change feeds are returned as
    **flow rates** (derivative of the cumulative-volume spline).

get_mass_balance(process) -> MassBalance
    Returns an ``eqx.Module`` whose ``__call__(c, q, u_flow, f_modeled)``
    computes the full mass-balance RHS ``dc/dt`` (including ``dV/dt``).
    Uncontrolled continuous volume changes (modeled feeds) are supported via
    the optional ``f_modeled`` argument.

extract_discrete_events(process, mb) -> list[dict]
    Extract discrete events (sampling, bolus feeds) from a BioProcess.

apply_discrete_event(state, event) -> jnp.ndarray
    Apply a single discrete event to the ODE state vector.

estimate_specific_rates(process, ctrl, mb, conc_splines, t_eval) -> np.ndarray
    Estimate specific rates q(t) via mass-balance inversion.

integrate_process(process, ctrl, mb, q_func, t_eval) -> dict
    Full hybrid ODE integration with discrete event handling.

Usage with JIT
--------------
Both modules are equinox Modules (JAX pytrees).  Use ``eqx.filter_jit``
to compile them::

    import equinox as eqx
    ctrl = get_control_splines(process)
    mb   = get_mass_balance(process)

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
import jax.numpy as jnp
import numpy as np

from .dataclasses import BioProcess, FeedVolumeChange, SampleVolumeChange, SplineRepresentation, StaticVariable, TimeSeries
from .splines import make_interpax_spline, build_interpax_spline, build_backtransform_spline


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
    _splines: list          # list of interpax.CubicSpline (each an eqx.Module)
    _is_derivative: tuple = eqx.field(static=True)  # True → return d/dt

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
        if len(self._is_derivative) == 0:
            return jnp.zeros(0)
        values = []
        for spline, is_deriv in zip(self._splines, self._is_derivative):
            if is_deriv:
                val = spline.derivative()(t)
            else:
                val = spline(t)
            values.append(val)
        return jnp.stack(values)


# ---------------------------------------------------------------------------
# MassBalance module
# ---------------------------------------------------------------------------

class MassBalance(eqx.Module):
    """JAX/Equinox module implementing the generalized fed-batch mass balance.

    Created by :func:`get_mass_balance`; do not instantiate directly.

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
        mb    = get_mass_balance(process)
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
        """Compute the mass-balance RHS ``dc/dt``.

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
        Mass balance implemented:

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
        if vc.spline is not None:
            sp = build_interpax_spline(vc.spline)[0][0]
        else:
            sp = make_interpax_spline(
                np.asarray(vc.values.timepoints),
                np.asarray(vc.values.values),
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
        if pv.spline is not None:
            sp = build_interpax_spline(pv.spline)[0][0]
        elif isinstance(pv.values, TimeSeries):
            sp = make_interpax_spline(
                np.asarray(pv.values.timepoints),
                np.asarray(pv.values.values),
            )
        else:
            # StaticVariable: constant spline over the full process time span
            t_start = float(process.time_axis.start)
            t_end = float(process.time_axis.end)
            sp = make_interpax_spline(
                np.array([t_start, t_end]),
                np.array([float(pv.values.value), float(pv.values.value)]),
            )
        control_names.append(pv_name)
        ctrl_indices.append(idx)
        splines.append(sp)
        is_derivative_list.append(False)
        idx += 1

    return ControlSplines(
        control_names=tuple(control_names),
        flow_indices=tuple(flow_indices),
        ctrl_indices=tuple(ctrl_indices),
        _splines=splines,
        _is_derivative=tuple(is_derivative_list),
    )

def get_mass_balance(process: BioProcess) -> MassBalance:
    """
    This is a backwards compatability function to illustrate that I changed this function name from
    get_mass_balance to get_rhs_ode, since it is more descriptive of what the function actually does.
    The old name is still supported for now but will eventually be removed.
    """
    return get_rhs_ode(process)

def get_rhs_ode(process: BioProcess) -> MassBalance:
    """Build a :class:`MassBalance` module from a :class:`BioProcess`.

    Parameters
    ----------
    process:
        A :class:`~bpbench.BioProcess` instance.  The reactor medium must
        contain a component named ``"biomass"`` (case-insensitive).

    Returns
    -------
    MassBalance
        An ``eqx.Module`` whose ``__call__(c, q, u_flow, f_modeled)``
        computes the mass-balance RHS ``dc/dt``.

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
        Cin_np = np.zeros((n, n_species), dtype=float)
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
                    Cin_np[k, j] = float(conc.value)
                else:
                    raise NotImplementedError(
                        "TimeSeries feed concentrations are not yet supported in get_mass_balance. "
                        f"Found TimeSeries for species '{sp_name}' in feed '{feed.name}' of volume change '{vc_name}'."
                    )
        return Cin_np

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

    Cin_np = _build_cin(flow_names)
    Cin_modeled_np = _build_cin(modeled_flow_names)

    return MassBalance(
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
        Cin=jnp.array(Cin_np),
        Cin_modeled=jnp.array(Cin_modeled_np),
    )


# ---------------------------------------------------------------------------
# Discrete event handling
# ---------------------------------------------------------------------------

def extract_discrete_events(
    process: BioProcess,
    mb: MassBalance,
) -> List[Dict[str, Any]]:
    """Extract discrete events (sampling, bolus feeds) from a BioProcess.

    Parameters
    ----------
    process:
        A :class:`~bpbench.BioProcess` instance.
    mb:
        A :class:`MassBalance` module (used to align ``Cin`` with species ordering).

    Returns
    -------
    list[dict]
        Sorted list of event dicts, each with keys:
        - ``t`` (float): event time
        - ``kind`` (str): ``'sample'`` or ``'bolus_feed'``
        - ``dV`` (float): signed volume change (positive = add, negative = remove)
        - ``Cin`` (np.ndarray | None): feed composition aligned with
          ``mb.species_names`` (None for sampling)
        - ``source`` (str): name of the originating VolumeChange
    """
    events: List[Dict[str, Any]] = []
    n_sp = len(mb.species_names)

    for vc_name, vc in process.volume.volume_changes.items():
        if vc.is_continuous:
            continue

        tp = np.asarray(vc.values.timepoints, dtype=float)
        vv = np.asarray(vc.values.values, dtype=float)

        for t_event, dV_event in zip(tp, vv):
            if abs(dV_event) < 1e-15:
                continue

            if dV_event > 0 and isinstance(vc, FeedVolumeChange):
                Cin_event = np.zeros(n_sp)
                if vc.feed_medium is not None:
                    for j, sp_name in enumerate(mb.species_names):
                        if sp_name in vc.feed_medium.components:
                            conc = vc.feed_medium.components[sp_name].concentration
                            if isinstance(conc, StaticVariable):
                                Cin_event[j] = float(conc.value)
                events.append(dict(
                    t=float(t_event),
                    kind='bolus_feed',
                    dV=float(dV_event),
                    Cin=Cin_event,
                    source=vc_name,
                ))
            else:
                events.append(dict(
                    t=float(t_event),
                    kind='sample',
                    dV=float(dV_event),
                    Cin=None,
                    source=vc_name,
                ))

    events.sort(key=lambda e: e['t'])
    return events


def apply_discrete_event(
    state: jnp.ndarray,
    event: Dict[str, Any],
) -> jnp.ndarray:
    """Apply a single discrete event to the ODE state vector.

    Parameters
    ----------
    state:
        Current state ``[c_species..., V]``.
    event:
        Event dict as returned by :func:`extract_discrete_events`.

    Returns
    -------
    jnp.ndarray
        Updated state vector after applying the event.
    """
    n_sp = state.shape[0] - 1
    c = state[:n_sp]
    V = state[n_sp]
    dV = event['dV']

    if event['kind'] == 'bolus_feed' and event['Cin'] is not None:
        V_new = V + dV
        c_new = (c * V + jnp.asarray(event['Cin']) * dV) / V_new
    else:
        # Sampling: concentrations unchanged, volume changes
        V_new = V + dV
        c_new = c

    V_new = jnp.maximum(V_new, 1e-10)
    c_new = jnp.maximum(c_new, 0.0)
    return jnp.append(c_new, V_new)


def build_conc_splines(
    process: BioProcess,
    mb: "MassBalance",
) -> Dict[str, Any]:
    """Build concentration splines from stored backtransform splines or raw data.

    Uses the pre-fitted ``comp.spline`` (backtransform spline built during
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
        A :class:`MassBalance` module (provides species ordering).

    Returns
    -------
    dict
        Mapping species name → callable spline.
    """
    conc_splines: Dict[str, Any] = {}

    for sp_name in mb.species_names:
        comp = process.reactor_medium.components[sp_name]
        if (
            comp.spline is not None
            and comp.spline.spline_metadata
            and "transform" in comp.spline.spline_metadata
        ):
            # Backtransform spline: continuous in pseudobatch domain,
            # correctly accounts for bolus feeds / sampling.
            conc_splines[sp_name] = build_backtransform_spline(comp.spline)
        elif comp.spline is not None:
            conc_splines[sp_name] = build_interpax_spline(comp.spline)[0][0]
        else:
            ts = comp.concentration
            conc_splines[sp_name] = make_interpax_spline(
                np.asarray(ts.timepoints, dtype=float),
                np.asarray(ts.values, dtype=float),
            )

    return conc_splines


# ---------------------------------------------------------------------------
# Specific rate estimation (mass-balance inversion)
# ---------------------------------------------------------------------------

def estimate_specific_rates(
    process: BioProcess,
    ctrl: ControlSplines,
    mb: MassBalance,
    conc_splines: Dict[str, Any],
    t_eval: np.ndarray,
) -> np.ndarray:
    """Estimate specific rates q(t) via mass-balance inversion.

    Uses the concentration splines and their derivatives to invert the
    mass-balance equation for q(t):

    .. math::
        q_i = \\frac{dc_i/dt - \\text{feed\\_term}}{X_{active}}

    Parameters
    ----------
    process:
        A :class:`~bpbench.BioProcess` instance.
    ctrl:
        :class:`ControlSplines` module for evaluating control signals.
    mb:
        :class:`MassBalance` module (provides species ordering, Cin, etc.).
    conc_splines:
        Dict mapping species name → callable spline that supports
        ``spline(t)`` and ``spline.derivative()(t)`` (e.g.,
        :class:`~bpbench.splines.BacktransformSpline` or
        ``interpax.CubicSpline``).
    t_eval:
        1-D array of time points at which to estimate q.

    Returns
    -------
    np.ndarray, shape (len(t_eval), n_species)
        Estimated specific rates at each time point.
    """
    t_eval = np.asarray(t_eval, dtype=float)
    n_sp = mb.q_size
    n_pts = len(t_eval)
    q_out = np.zeros((n_pts, n_sp))

    # Build cumulative volume splines for continuous flows
    cum_splines_ctrl = []
    for fn in mb.flow_names:
        vc = process.volume.volume_changes[fn]
        if vc.spline is not None:
            sp = build_interpax_spline(vc.spline)[0][0]
        else:
            sp = make_interpax_spline(
                np.asarray(vc.values.timepoints),
                np.asarray(vc.values.values),
            )
        cum_splines_ctrl.append(sp)

    cum_splines_mod = []
    for fn in mb.modeled_flow_names:
        vc = process.volume.volume_changes[fn]
        if vc.spline is not None:
            sp = build_interpax_spline(vc.spline)[0][0]
        else:
            sp = make_interpax_spline(
                np.asarray(vc.values.timepoints),
                np.asarray(vc.values.values),
            )
        cum_splines_mod.append(sp)

    # Cumulative discrete volume changes
    events = extract_discrete_events(process, mb)

    V0 = float(process.volume.initial_volume)
    Cin = np.array(mb.Cin)
    Cin_mod = np.array(mb.Cin_modeled)

    # Normalization: scale concentrations so all species are O(1)
    scales = _compute_scale_factors(process, mb)
    Cin_norm = Cin / scales[None, :]
    Cin_mod_norm = Cin_mod / scales[None, :] if Cin_mod.size > 0 else Cin_mod

    flow_idx = list(ctrl.flow_indices)

    for k, t in enumerate(t_eval):
        tj = jnp.array(t)

        # Concentrations from splines (then normalize)
        c_t = np.array([float(conc_splines[s](tj)) for s in mb.species_names])
        c_t = np.maximum(c_t, 0.0)
        c_norm = c_t / scales

        # Volume at time t
        V_t = V0
        for sp in cum_splines_ctrl:
            V_t += float(sp(tj))
        for sp in cum_splines_mod:
            V_t += float(sp(tj))
        for ev in events:
            if ev['t'] <= t:
                V_t += ev['dV']
        V_t = max(V_t, 1e-10)

        # Flow rates
        u_flow = np.zeros(mb.u_flow_size)
        for i, sp in enumerate(cum_splines_ctrl):
            u_flow[i] = float(sp.derivative()(tj))

        f_mod = np.zeros(mb.f_modeled_size)
        for i, sp in enumerate(cum_splines_mod):
            f_mod[i] = float(sp.derivative()(tj))

        # Feed contributions (in normalized coordinates)
        feed_norm = np.zeros(n_sp)
        if mb.u_flow_size > 0:
            feed_ctrl = np.sum(
                (u_flow[:, None] / V_t) * (Cin_norm - c_norm[None, :]), axis=0
            )
            feed_norm += feed_ctrl
        if mb.f_modeled_size > 0:
            feed_modeled = np.sum(
                (f_mod[:, None] / V_t) * (Cin_mod_norm - c_norm[None, :]), axis=0
            )
            feed_norm += feed_modeled

        # Concentration derivatives from splines (in normalized coordinates)
        dc_norm = np.zeros(n_sp)
        for i, s in enumerate(mb.species_names):
            sp = conc_splines[s]
            if hasattr(sp, 'derivative'):
                dc_norm[i] = float(sp.derivative()(tj)) / scales[i]
            else:
                # BacktransformSpline doesn't have .derivative()
                # Use finite differences
                dt = 1e-4
                dc_norm[i] = (float(sp(tj + dt)) - float(sp(tj - dt))) / (2 * dt * scales[i])

        # Active biomass (compute in original coordinates, then normalize)
        X_active = c_t[mb.biomass_idx]
        for idx in mb.intracellular_indices:
            X_active -= c_t[idx]
        X_active = max(X_active, 1e-6)
        X_active_norm = X_active / scales[mb.biomass_idx]

        # q_i = (dc_i/dt - feed_i) / X_active
        #      = (scales[i] * dc_norm_i - scales[i] * feed_norm_i) / (scales[bio] * X_active_norm)
        #      = (scales[i] / scales[bio]) * (dc_norm_i - feed_norm_i) / X_active_norm
        q_out[k] = (scales / scales[mb.biomass_idx]) * (dc_norm - feed_norm) / X_active_norm

    return q_out


# ---------------------------------------------------------------------------
# Full hybrid ODE integration
# ---------------------------------------------------------------------------

def _build_segment_rhs(mb, ctrl, q_func, cum_splines_mod, conc_splines=None):
    """Build the ODE right-hand side function for a segment.

    Parameters
    ----------
    conc_splines : dict or None
        If provided, a dict mapping species name → callable spline.
        When given, the RHS uses spline-evaluated concentrations for the
        active-biomass term (``X_active``) instead of the ODE state.  This
        prevents the exponential error amplification that occurs when
        biomass spans many orders of magnitude (e.g. 1e6 → 1e9 cells/mL).
    """
    flow_idx = jnp.array(list(ctrl.flow_indices))

    if conc_splines is not None:
        # Build list of splines aligned with species_names for biomass
        # and intracellular species
        bio_spline = conc_splines[mb.species_names[mb.biomass_idx]]
        intra_splines = [conc_splines[mb.species_names[i]]
                         for i in mb.intracellular_indices]

    def rhs(t, state, args):
        u = ctrl(t)
        u_flow = u[flow_idx] if len(flow_idx) > 0 else jnp.zeros(mb.u_flow_size)
        q = q_func(t)

        f_mod = jnp.zeros(mb.f_modeled_size)
        for i, sp in enumerate(cum_splines_mod):
            f_mod = f_mod.at[i].set(sp.derivative()(t))

        if conc_splines is not None:
            # Override X_active from splines for numerical stability
            X_active = bio_spline(t)
            for isp in intra_splines:
                X_active = X_active - isp(t)
            X_active = jnp.maximum(X_active, 1e-6)

            c_species = state[:mb.q_size]
            V = state[mb.q_size]

            # Reaction term with spline-based X_active
            reaction = q * X_active

            # Feed terms (same as MassBalance.__call__)
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


def _compute_scale_factors(process: BioProcess, mb: "MassBalance") -> np.ndarray:
    """Compute per-species scale factors for numerical conditioning.

    Returns an array of shape (n_species,) where each entry is the
    max absolute concentration for that species (clamped to >= 1.0 so
    species with small values are not artificially inflated).
    """
    scales = np.ones(mb.q_size)
    for i, sp_name in enumerate(mb.species_names):
        vals = np.asarray(
            process.reactor_medium.components[sp_name].concentration.values,
            dtype=float,
        )
        s = np.max(np.abs(vals))
        if s > 1.0:
            scales[i] = s
    return scales


def integrate_process(
    process: BioProcess,
    ctrl: ControlSplines,
    mb: MassBalance,
    q_func: Callable,
    t_eval: np.ndarray,
    *,
    conc_splines: Optional[Dict[str, Any]] = None,
    rtol: float = 1e-6,
    atol: float = 1e-8,
    max_steps: int = 16384,
) -> Dict[str, Any]:
    """Full hybrid ODE integration with discrete event handling.

    Integrates the mass-balance ODE segment-by-segment.  Segments are
    separated by discrete events (sampling, bolus feeds).  Between events
    the ODE is solved with ``diffrax.Dopri5``.  At event boundaries,
    :func:`apply_discrete_event` is applied.

    ``diffeqsolve`` is called directly (not wrapped in ``eqx.filter_jit``)
    so that diffrax handles its own internal JIT compilation of the step
    function.  This avoids tracing the entire solve loop into a single XLA
    program, which would cause extremely slow compilation when splines have
    many knots.

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
        :class:`MassBalance` module.
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
    t_eval = np.asarray(t_eval, dtype=float)
    t_start = float(process.time_axis.start)
    t_end = float(process.time_axis.end)
    n_sp = mb.q_size

    # Per-species scale factors for numerical conditioning
    scales = _compute_scale_factors(process, mb)
    scale_vec = jnp.array(scales)  # (n_sp,)
    # Full state scale: [scales..., 1.0] (volume is not scaled)
    state_scale = jnp.append(scale_vec, 1.0)

    # Build modeled flow splines
    cum_splines_mod = []
    for fn in mb.modeled_flow_names:
        vc = process.volume.volume_changes[fn]
        if vc.spline is not None:
            sp = build_interpax_spline(vc.spline)[0][0]
        else:
            sp = make_interpax_spline(
                np.asarray(vc.values.timepoints),
                np.asarray(vc.values.values),
            )
        cum_splines_mod.append(sp)

    # Extract discrete events and build segment boundaries
    events = extract_discrete_events(process, mb)
    event_times = sorted(set(ev['t'] for ev in events))
    event_times_in_range = [t for t in event_times if t_start < t < t_end]
    boundaries = [t_start] + event_times_in_range + [t_end]

    # Build event lookup
    event_lookup: Dict[float, List[Dict]] = {}
    for ev in events:
        event_lookup.setdefault(ev['t'], []).append(ev)

    # Initial state (in original coordinates)
    c0 = np.array([
        float(np.asarray(process.reactor_medium.components[s].concentration.values[0]))
        for s in mb.species_names
    ])
    c0 = np.maximum(c0, 0.0)
    V0 = float(process.volume.initial_volume)

    # Build RHS in original coordinates, then wrap for normalized state
    rhs_original = _build_segment_rhs(mb, ctrl, q_func, cum_splines_mod, conc_splines)

    def rhs_normalized(t, state_norm, args):
        state_orig = state_norm * state_scale
        dc_dt_orig = rhs_original(t, state_orig, args)
        return dc_dt_orig / state_scale

    term = diffrax.ODETerm(rhs_normalized)
    solver = diffrax.Dopri5()
    stepsize_controller = diffrax.PIDController(rtol=rtol, atol=atol)

    # Normalized initial state
    state_norm = jnp.array(np.append(c0, V0)) / state_scale

    # Segment-by-segment integration
    t_segments = []
    c_segments = []
    V_segments = []

    for seg_idx in range(len(boundaries) - 1):
        t_lo = boundaries[seg_idx]
        t_hi = boundaries[seg_idx + 1]

        # Select eval points in this segment
        mask = (t_eval >= t_lo) & (t_eval <= t_hi)
        t_seg = t_eval[mask]
        if len(t_seg) == 0:
            t_seg = np.array([t_lo, t_hi])
        else:
            if t_seg[0] > t_lo + 1e-12:
                t_seg = np.concatenate([[t_lo], t_seg])
            if t_seg[-1] < t_hi - 1e-12:
                t_seg = np.concatenate([t_seg, [t_hi]])

        dt0 = min(0.1, (t_hi - t_lo) / 10.0)

        sol = diffrax.diffeqsolve(
            term,
            solver,
            t0=t_lo,
            t1=t_hi,
            dt0=dt0,
            y0=state_norm,
            saveat=diffrax.SaveAt(ts=jnp.array(t_seg)),
            stepsize_controller=stepsize_controller,
            max_steps=max_steps,
        )
        ys_norm = sol.ys

        # Un-normalize
        ys_orig = ys_norm * state_scale[None, :]

        c_seg = jnp.maximum(ys_orig[:, :n_sp], 0.0)
        V_seg = jnp.maximum(ys_orig[:, n_sp], 1e-10)

        # Store results (skip last point if not final segment to avoid duplication)
        if seg_idx < len(boundaries) - 2:
            t_segments.append(np.asarray(t_seg[:-1]))
            c_segments.append(np.asarray(c_seg[:-1]))
            V_segments.append(np.asarray(V_seg[:-1]))
        else:
            t_segments.append(np.asarray(t_seg))
            c_segments.append(np.asarray(c_seg))
            V_segments.append(np.asarray(V_seg))

        # Apply discrete events at boundary (in original coordinates)
        if seg_idx < len(boundaries) - 2:
            state_orig = jnp.array(
                [float(c_seg[-1, i]) for i in range(n_sp)] + [float(V_seg[-1])]
            )
            t_event = boundaries[seg_idx + 1]
            for ev in event_lookup.get(t_event, []):
                state_orig = apply_discrete_event(state_orig, ev)
            state_norm = state_orig / state_scale

    t_out = np.concatenate(t_segments)
    c_out = np.vstack(c_segments)
    V_out = np.concatenate(V_segments)

    return {'t': t_out, 'c': c_out, 'V': V_out}
