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
    Returns an ``eqx.Module`` whose ``__call__(c, q, u_flow)`` computes the
    full mass-balance RHS ``dc/dt`` (including ``dV/dt``).

Usage with JIT
--------------
Both modules are equinox Modules (JAX pytrees).  Use ``eqx.filter_jit``
to compile them::

    import equinox as eqx
    ctrl = get_control_splines(process)
    mb   = get_mass_balance(process)

    u      = eqx.filter_jit(ctrl)(t)
    dc_dt  = eqx.filter_jit(mb)(c, q, u_flow)
"""

from __future__ import annotations

from typing import List, Tuple

import equinox as eqx
import interpax
import jax.numpy as jnp
import numpy as np

from .dataclasses import BioProcess, StaticVariable, TimeSeries


# ---------------------------------------------------------------------------
# Internal helper: build an interpax.CubicSpline from a TimeSeries
# ---------------------------------------------------------------------------

def _make_spline(timepoints: jnp.ndarray,
                 values: jnp.ndarray) -> interpax.CubicSpline:
    """Fit a natural cubic spline to a 1-D time series.

    Parameters
    ----------
    timepoints:
        Strictly increasing 1-D time array (≥ 2 points).
    values:
        Corresponding values (same length as *timepoints*).

    Returns
    -------
    interpax.CubicSpline
        An ``eqx.Module`` that evaluates the spline at arbitrary *t* via
        ``spline(t)`` and its first derivative via
        ``spline.derivative()(t)``.  Both calls are fully JAX-jittable.
    """
    t = jnp.asarray(timepoints, dtype=float)
    v = jnp.asarray(values, dtype=float)
    if t.shape[0] < 2:
        # Single observation: extend to two identical points so interpax can
        # build a valid (constant) spline.  The offset is arbitrary because all
        # derivative coefficients are zero for a constant spline, so spline(t)
        # returns v[0] for any query time regardless of breakpoint spacing.
        t = jnp.array([t[0], t[0] + 1.0])
        v = jnp.array([v[0], v[0]])
    return interpax.CubicSpline(t, v, bc_type="natural", check=False)


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
    the reactor volume.

    Attributes
    ----------
    c_size : int
        ``n_species + 1`` (species concentrations + volume).
    q_size : int
        ``n_species`` — number of specific rates (aligned with
        :attr:`species_names`).
    u_flow_size : int
        Number of continuous controlled flow streams.
    output_size : int
        Same as :attr:`c_size`.
    species_names : tuple[str, ...]
        Ordering of species in *c* and *q*.
    flow_names : tuple[str, ...]
        Ordering of continuous flow streams in *u_flow*.
    biomass_idx : int
        Index of ``"biomass"`` in :attr:`species_names`.
    Cin : jnp.ndarray, shape (n_flows, n_species)
        Feed composition matrix: ``Cin[k, i]`` is the concentration of
        species *i* in feed stream *k*.

    Notes
    -----
    JIT usage::

        import equinox as eqx
        mb    = get_mass_balance(process)
        dc_dt = eqx.filter_jit(mb)(c, q, u_flow)
    """

    c_size: int = eqx.field(static=True)
    q_size: int = eqx.field(static=True)
    u_flow_size: int = eqx.field(static=True)
    output_size: int = eqx.field(static=True)
    species_names: tuple = eqx.field(static=True)
    flow_names: tuple = eqx.field(static=True)
    biomass_idx: int = eqx.field(static=True)
    Cin: jnp.ndarray

    def __call__(
        self,
        c: jnp.ndarray,
        q: jnp.ndarray,
        u_flow: jnp.ndarray,
    ) -> jnp.ndarray:
        """Compute the mass-balance RHS ``dc/dt``.

        Parameters
        ----------
        c:
            State vector ``[c_species..., V]``, shape ``(c_size,)``.
        q:
            Specific rates aligned with :attr:`species_names`, shape
            ``(q_size,)``.
        u_flow:
            Volumetric flow rates for each continuous feed stream (volume /
            time, matching the units of the stored ``VolumeChange``),
            shape ``(u_flow_size,)``.

        Returns
        -------
        jnp.ndarray, shape ``(output_size,)``
            ``dc/dt`` with ``dV/dt`` as the last element.

        Notes
        -----
        Mass balance implemented:

        .. math::

            \\frac{dc_i}{dt} = q_i \\cdot X
                + \\sum_k \\frac{f_k}{V}\\,(C_{in,k,i} - c_i)

            \\frac{dV}{dt} = \\sum_k f_k

        where :math:`X = c[\\text{biomass\\_idx}]`.
        """
        c_species = c[: self.q_size]
        V = c[self.q_size]
        X = c[self.biomass_idx]

        # Reaction contribution: q_i * X
        reaction = q * X

        # Feed / dilution contribution (zero when u_flow_size == 0)
        # u_flow: (n_flows,),  Cin: (n_flows, n_species)
        feed_contrib = u_flow[:, None] * (self.Cin - c_species[None, :])
        feed_term = jnp.sum(feed_contrib, axis=0) / V

        dc_species = reaction + feed_term
        dV = jnp.sum(u_flow)

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
        sp = _make_spline(vc.values.timepoints, vc.values.values)
        control_names.append(vc_name)
        flow_indices.append(idx)
        splines.append(sp)
        is_derivative_list.append(True)
        idx += 1

    # 2) Controlled process variables → direct spline value
    for pv_name, pv in process.process_variables.items():
        if not pv.is_controlled:
            continue
        if isinstance(pv.values, TimeSeries):
            sp = _make_spline(pv.values.timepoints, pv.values.values)
        else:
            # StaticVariable: constant spline over the full process time span
            t_start = float(process.time_axis.start)
            t_end = float(process.time_axis.end)
            sp = _make_spline(
                jnp.array([t_start, t_end]),
                jnp.array([float(pv.values.value), float(pv.values.value)]),
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
    """Build a :class:`MassBalance` module from a :class:`BioProcess`.

    Parameters
    ----------
    process:
        A :class:`~bpbench.BioProcess` instance.  The reactor medium must
        contain a component named ``"biomass"`` (case-insensitive).

    Returns
    -------
    MassBalance
        An ``eqx.Module`` whose ``__call__(c, q, u_flow)`` computes the
        mass-balance RHS ``dc/dt``.

    Raises
    ------
    ValueError
        If no ``"biomass"`` component is found in the reactor medium.
    """
    # --- Species ordering (from reactor medium, insertion order) ---
    species_names: Tuple[str, ...] = tuple(
        process.reactor_medium.components.keys()
    )
    n_species = len(species_names)

    biomass_idx: int = -1
    for i, name in enumerate(species_names):
        if name.strip().lower() == "biomass":
            biomass_idx = i
            break
    if biomass_idx < 0:
        raise ValueError(
            "No 'biomass' component found in process.reactor_medium.components. "
            f"Available components: {list(species_names)}"
        )

    # --- Flow ordering (continuous controlled volume changes) ---
    flow_names: List[str] = []
    for vc_name, vc in process.volume.volume_changes.items():
        if vc.is_controlled and vc.is_continuous:
            flow_names.append(vc_name)
    n_flows = len(flow_names)

    # --- Feed composition matrix Cin: (n_flows, n_species) ---
    Cin_np = np.zeros((n_flows, n_species), dtype=float)
    for k, vc_name in enumerate(flow_names):
        feed = process.volume.volume_changes[vc_name].feed_medium
        for j, sp_name in enumerate(species_names):
            if sp_name not in feed.components:
                continue
            conc = feed.components[sp_name].concentration
            if isinstance(conc, StaticVariable):
                Cin_np[k, j] = float(conc.value)
            else:
                # TimeSeries feed concentration → use the mean
                Cin_np[k, j] = float(jnp.mean(jnp.asarray(conc.values)))

    return MassBalance(
        c_size=n_species + 1,
        q_size=n_species,
        u_flow_size=n_flows,
        output_size=n_species + 1,
        species_names=tuple(species_names),
        flow_names=tuple(flow_names),
        biomass_idx=biomass_idx,
        Cin=jnp.array(Cin_np),
    )
