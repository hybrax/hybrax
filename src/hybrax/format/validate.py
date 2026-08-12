"""
Validation utilities for bioprocess data
"""

import jax.numpy as jnp
from typing import Dict, List, Optional, Tuple
from .dataclasses import (
    AugmentedBioProcess,
    BioProcess,
    BioProcessCollection,
    Bounds,
    StaticVariable,
    TimeSeries,
    FeedVolumeChange,
    SampleVolumeChange,
)


def _check_bounds_tuple(bounds: Bounds, label: str) -> Tuple[bool, str]:
    """Sanity-check a bounds tuple: lo <= hi when both are set."""
    if bounds is None:
        return True, f"Bounds {label} ok (unset)"
    lo, hi = bounds
    if lo is not None and hi is not None and lo > hi:
        return (
            False,
            f"Bounds {label} invalid: lower {lo} > upper {hi}",
        )
    return True, f"Bounds {label} ok"


def validate_biological_ode(process: BioProcess) -> Tuple[bool, str]:
    """Validate ``process.biological_ode`` if present.

    Checks (no-op when ``biological_ode is None``):

    - Every reactor-medium component and uncontrolled process variable
      (the dynamic states) must have an entry in ``derivatives``.
    - Every key in ``derivatives`` must correspond to a dynamic state.
    - All expressions parse successfully via ``sympy.sympify``.
    - Every free symbol in any expression resolves to a dynamic state, a
      controlled process variable (input), an algebraic name, or a declared
      rate name.
    - Algebraic-variable dependencies are acyclic.
    - Rate names do not collide with state, algebraic, or controlled-PV names.
    - All bounds tuples are sane (``lo <= hi`` when both set).
    """
    bo = process.biological_ode
    if bo is None:
        return True, "biological_ode: not set"

    import sympy

    state_names = set()
    if process.reactor_medium:
        state_names.update(process.reactor_medium.components.keys())
    state_names.update(
        name for name, pv in process.process_variables.items() if not pv.is_controlled
    )
    controlled_pv_names = {
        name for name, pv in process.process_variables.items() if pv.is_controlled
    }
    algebraic_names = set(bo.algebraic.keys())
    rate_names = set(bo.rates.keys())

    errors: List[str] = []

    # Name-collision checks
    overlap_rate_state = rate_names & state_names
    if overlap_rate_state:
        errors.append(
            f"rate names collide with state names: {sorted(overlap_rate_state)}"
        )
    overlap_rate_algebraic = rate_names & algebraic_names
    if overlap_rate_algebraic:
        errors.append(
            f"rate names collide with algebraic names: {sorted(overlap_rate_algebraic)}"
        )
    overlap_rate_ctrl = rate_names & controlled_pv_names
    if overlap_rate_ctrl:
        errors.append(
            f"rate names collide with controlled PV names: {sorted(overlap_rate_ctrl)}"
        )
    overlap_algebraic_state = algebraic_names & state_names
    if overlap_algebraic_state:
        errors.append(
            "algebraic names collide with state names: "
            f"{sorted(overlap_algebraic_state)}"
        )
    overlap_algebraic_ctrl = algebraic_names & controlled_pv_names
    if overlap_algebraic_ctrl:
        errors.append(
            "algebraic names collide with controlled PV names: "
            f"{sorted(overlap_algebraic_ctrl)}"
        )

    # Coverage of derivatives
    deriv_keys = set(bo.derivatives.keys())
    missing = state_names - deriv_keys
    if missing:
        errors.append(
            f"derivatives missing entries for dynamic state(s): {sorted(missing)}. "
            'Use "0" to declare no biological dynamics.'
        )
    extra = deriv_keys - state_names
    if extra:
        errors.append(
            f"derivatives keys must be dynamic states; extras: {sorted(extra)}"
        )

    # Expression parsing + symbol resolution
    allowed = state_names | controlled_pv_names | algebraic_names | rate_names
    symbol_table = {n: sympy.Symbol(n) for n in allowed}

    def _parse(name: str, expr_str: str, kind: str):
        try:
            expr = sympy.sympify(expr_str, locals=symbol_table)
        except (sympy.SympifyError, SyntaxError, TypeError) as exc:
            errors.append(f"{kind} {name!r} expression failed to parse: {exc}")
            return None
        unknown = {str(s) for s in expr.free_symbols} - allowed
        if unknown:
            errors.append(
                f"{kind} {name!r} expression references undeclared symbol(s): "
                f"{sorted(unknown)}"
            )
        return expr

    algebraic_exprs = {n: _parse(n, e, "algebraic") for n, e in bo.algebraic.items()}
    derivative_exprs = {
        n: _parse(n, e, "derivatives") for n, e in bo.derivatives.items()
    }

    # Unit consistency: a sympy ``Add`` whose operands collectively reference
    # two or more dynamic states (reactor components or uncontrolled PVs) must
    # have those states share a unit. Catches things like ``biomass - product``
    # when biomass is g/L and product is mg/L (raw numerical subtraction would
    # be meaningless). This generalises the legacy intracellular unit check.
    def _state_unit(name: str) -> Optional[str]:
        if process.reactor_medium and name in process.reactor_medium.components:
            return process.reactor_medium.components[name].unit
        pv = process.process_variables.get(name)
        return pv.unit if pv is not None else None

    for expr_name, expr in {**algebraic_exprs, **derivative_exprs}.items():
        if expr is None:
            continue
        for node in sympy.preorder_traversal(expr):
            if not isinstance(node, sympy.Add):
                continue
            state_syms = {str(s) for s in node.free_symbols} & state_names
            if len(state_syms) < 2:
                continue
            units = {s: _state_unit(s) for s in state_syms}
            if len(set(units.values())) > 1:
                pretty = ", ".join(f"{s}={u!r}" for s, u in sorted(units.items()))
                errors.append(
                    f"{expr_name!r}: state variables combined additively with "
                    f"mismatched units ({pretty})"
                )
                break  # one error per expression is enough

    # Cycle detection on algebraic-variable graph
    deps = {
        n: {str(s) for s in (expr.free_symbols if expr is not None else set())}
        & algebraic_names
        for n, expr in algebraic_exprs.items()
    }
    visiting: set = set()
    visited: set = set()

    def _dfs(node: str, stack: List[str]) -> bool:
        if node in visiting:
            cycle = stack[stack.index(node) :] + [node]
            errors.append(f"algebraic dependency cycle: {' -> '.join(cycle)}")
            return False
        if node in visited:
            return True
        visiting.add(node)
        for dep in deps.get(node, ()):
            _dfs(dep, stack + [node])
        visiting.discard(node)
        visited.add(node)
        return True

    for n in algebraic_exprs:
        _dfs(n, [])

    # Bounds sanity on rates
    for rname, bounds in bo.rates.items():
        ok, msg = _check_bounds_tuple(bounds, f"rate {rname!r}")
        if not ok:
            errors.append(msg)

    if errors:
        bullets = "\n  - ".join(errors)
        return False, f"biological_ode invalid:\n  - {bullets}"
    return True, "biological_ode ok"


def validate_biological_ode_equivalence(
    container: "BioProcessCollection",
) -> Tuple[bool, str]:
    """Verify all processes in *container* share the same ``biological_ode``.

    Equivalence requires identical ``algebraic``, ``derivatives``, and
    ``rates`` dicts (same keys mapped to the same values). Containers with
    zero or one process trivially pass.

    Returns ``(is_valid, message)``.
    """
    if not isinstance(container, BioProcessCollection):
        raise TypeError(
            "validate_biological_ode_equivalence() expects a "
            f"BioProcessCollection, got {type(container).__name__!r}"
        )

    procs = list(container.processes.items())
    if len(procs) <= 1:
        return True, (
            f"biological_ode equivalence trivially holds ({len(procs)} processes)"
        )

    first_name, first_proc = procs[0]
    ref_bo = first_proc.biological_ode

    errors: List[str] = []
    for name, proc in procs[1:]:
        bo = proc.biological_ode
        if (ref_bo is None) != (bo is None):
            errors.append(
                f"Process '{name}' biological_ode presence differs from '{first_name}'"
            )
            continue
        if ref_bo is None:
            continue
        if dict(bo.algebraic) != dict(ref_bo.algebraic):
            errors.append(
                f"Process '{name}' biological_ode.algebraic differs from '{first_name}'"
            )
        if dict(bo.derivatives) != dict(ref_bo.derivatives):
            errors.append(
                f"Process '{name}' biological_ode.derivatives differs from "
                f"'{first_name}'"
            )
        if dict(bo.rates) != dict(ref_bo.rates):
            errors.append(
                f"Process '{name}' biological_ode.rates differs from '{first_name}'"
            )

    if errors:
        return False, "biological_ode mismatch across processes:\n  - " + "\n  - ".join(
            errors
        )
    return True, f"biological_ode equivalent across {len(procs)} processes"


def validate_bounds(process: BioProcess) -> Tuple[bool, str]:
    """Sanity-check all bounds tuples on the process (states, PVs, volume)."""
    errors: List[str] = []
    if process.reactor_medium:
        for cname, comp in process.reactor_medium.components.items():
            ok, msg = _check_bounds_tuple(comp.bounds, f"reactor component {cname!r}")
            if not ok:
                errors.append(msg)
    for pname, pv in process.process_variables.items():
        ok, msg = _check_bounds_tuple(pv.bounds, f"process variable {pname!r}")
        if not ok:
            errors.append(msg)
    if process.volume is not None:
        ok, msg = _check_bounds_tuple(process.volume.bounds, "volume")
        if not ok:
            errors.append(msg)
    if errors:
        bullets = "\n  - ".join(errors)
        return False, f"bounds invalid:\n  - {bullets}"
    return True, "bounds ok"


def _check_data_against_bounds(
    value: object, bounds: Bounds, label: str
) -> Tuple[bool, str]:
    """Compare a raw measured value (StaticVariable or TimeSeries) against its
    own declared Bounds tuple; report count/min/max of violations."""
    if bounds is None or (bounds[0] is None and bounds[1] is None):
        return True, f"{label}: bounds unset — data check skipped"
    lo, hi = bounds

    if isinstance(value, StaticVariable):
        v = float(value.value)
        if lo is not None and v < lo:
            return False, f"{label}: value {v:g} is below lower bound {lo:g}"
        if hi is not None and v > hi:
            return False, f"{label}: value {v:g} is above upper bound {hi:g}"
        return True, f"{label}: value within bounds"

    if not _is_dynamic_series(value):
        return True, f"{label}: not a TimeSeries/StaticVariable — data check skipped"

    vals = jnp.asarray(value.values)
    errors: List[str] = []
    if lo is not None:
        n_below = int(jnp.sum(vals < lo))
        if n_below:
            errors.append(
                f"{n_below} datapoint(s) below lower bound {lo:g} "
                f"(min observed {float(jnp.min(vals)):g})"
            )
    if hi is not None:
        n_above = int(jnp.sum(vals > hi))
        if n_above:
            errors.append(
                f"{n_above} datapoint(s) above upper bound {hi:g} "
                f"(max observed {float(jnp.max(vals)):g})"
            )
    if errors:
        return False, f"{label}: " + "; ".join(errors)
    return True, f"{label}: all datapoints within bounds"


def validate_bounds_against_data(process: BioProcess) -> Tuple[bool, str]:
    """Check measured/raw data values against their own declared Bounds.

    For every ``ReactorMediumComponent.concentration``,
    ``ProcessVariable.values``, and (when present) ``Volume.total_volume``
    that carries a *set* Bounds tuple (either side non-``None``), compares
    the actual scalar (``StaticVariable``) or ``TimeSeries.values`` array
    against ``(lo, hi)`` and reports how many datapoints violate the bound.

    Out of scope: ``BiologicalOde.rates`` bounds — no rate-inversion
    machinery exists to compute a measured rate value to check against them
    (see ``validate_biological_ode``'s tuple-only sanity check on rates).
    """
    errors: List[str] = []

    if process.reactor_medium:
        for cname, comp in process.reactor_medium.components.items():
            ok, msg = _check_data_against_bounds(
                comp.concentration, comp.bounds, f"reactor component {cname!r}"
            )
            if not ok:
                errors.append(msg)

    for pname, pv in process.process_variables.items():
        ok, msg = _check_data_against_bounds(
            pv.values, pv.bounds, f"process variable {pname!r}"
        )
        if not ok:
            errors.append(msg)

    if process.volume is not None and process.volume.total_volume is not None:
        ok, msg = _check_data_against_bounds(
            process.volume.total_volume, process.volume.bounds, "volume"
        )
        if not ok:
            errors.append(msg)

    if errors:
        bullets = "\n  - ".join(errors)
        return False, f"bounds-vs-data invalid:\n  - {bullets}"
    return True, "bounds-vs-data ok"


def _is_dynamic_series(value: object) -> bool:
    """Return True when *value* looks like a TimeSeries-like dynamic object."""
    times = getattr(value, "times", None)
    values = getattr(value, "values", None)
    return times is not None and values is not None


def validate_timeseries_shape(ts: TimeSeries, name: str = "") -> Tuple[bool, str]:
    """
    Check that a TimeSeries has consistent shapes and ordered times.

    Verifies:
    - ``times`` and ``values`` are 1-D arrays.
    - Both arrays have the same length.
    - ``times`` are strictly monotonically increasing (no duplicates).

    Args:
        ts: TimeSeries object to validate.
        name: Optional label used in error messages (e.g. the variable name).

    Returns:
        A tuple ``(is_valid, message)`` where ``is_valid`` is ``True`` when all
        checks pass and ``message`` contains a human-readable summary.
    """
    label = f"'{name}' " if name else ""
    errors: List[str] = []

    if not _is_dynamic_series(ts):
        return (
            False,
            f"TimeSeries {label}invalid:\n  - missing discrete times/values arrays",
        )

    tp = jnp.asarray(ts.times)
    vals = jnp.asarray(ts.values)

    if tp.ndim != 1:
        errors.append(f"times must be 1-D, got shape {tp.shape}")
    if vals.ndim != 1:
        errors.append(f"values must be 1-D, got shape {vals.shape}")

    if tp.ndim == 1 and vals.ndim == 1:
        if tp.shape[0] != vals.shape[0]:
            errors.append(
                f"times length ({tp.shape[0]}) does not match "
                f"values length ({vals.shape[0]})"
            )
        if tp.shape[0] > 1:
            diffs = jnp.diff(tp)
            if not bool(jnp.all(diffs > 0)):
                errors.append("times are not strictly monotonically increasing")

    if errors:
        return False, f"TimeSeries {label}invalid:\n" + "\n".join(
            f"  - {e}" for e in errors
        )
    return True, f"TimeSeries {label}OK"


def validate_discrete_events(process: BioProcess) -> Tuple[bool, str]:
    """Check event time shape, ordering, bounds, and label count."""
    events = process.discrete_events
    if events is None:
        return True, "No discrete events — check skipped"

    times = jnp.asarray(events.times)
    errors = []
    if times.ndim != 1:
        errors.append(f"times must be 1-D, got shape {times.shape}")
    else:
        if times.shape[0] > 1 and not bool(jnp.all(jnp.diff(times) > 0)):
            errors.append("times are not strictly monotonically increasing")

        start = process.time_axis.start
        end = process.time_axis.end
        outside = ~((times >= start) & (times <= end))
        count = int(jnp.sum(outside))
        if count:
            errors.append(f"{count} timestamp(s) outside [{start}, {end}]")

        if events.labels is not None and len(events.labels) != times.shape[0]:
            errors.append(
                f"labels length ({len(events.labels)}) does not match "
                f"times length ({times.shape[0]})"
            )

    if errors:
        return False, "Discrete events invalid:\n  - " + "\n  - ".join(errors)
    return True, "Discrete events valid — OK"


def validate_mapping_names(process: BioProcess) -> Tuple[bool, str]:
    """Check that mapping keys match the embedded object names."""
    errors = []

    def check(mapping, label):
        errors.extend(
            f"{label} key {key!r} does not match object name {value.name!r}"
            for key, value in mapping.items()
            if key != value.name
        )

    check(process.reactor_medium.components, "Reactor component")
    check(process.process_variables, "Process variable")
    check(process.volume.volume_changes, "Volume change")
    for key, change in process.volume.volume_changes.items():
        if isinstance(change, FeedVolumeChange) and change.feed_medium:
            check(change.feed_medium.components, f"Feed components for {key!r}")

    if errors:
        return False, "Mapping name mismatches:\n  - " + "\n  - ".join(errors)
    return True, "Mapping keys match object names — OK"


def validate_time_axis(process: BioProcess) -> Tuple[bool, str]:
    """Check that the process time axis starts no later than it ends."""
    axis = process.time_axis
    if axis.start > axis.end:
        return False, (
            f"Time axis invalid: start {axis.start} is after end {axis.end} {axis.unit}"
        )
    return True, f"Time axis [{axis.start}, {axis.end}] {axis.unit} — OK"


def validate_timestamp_bounds(process: BioProcess) -> Tuple[bool, str]:
    """Check that process timestamps fall within its inclusive time-axis bounds."""
    series = []
    if process.reactor_medium:
        for name, component in process.reactor_medium.components.items():
            if _is_dynamic_series(component.concentration):
                series.append((f"reactor component {name!r}", component.concentration))
            if _is_dynamic_series(component.c_star_concentration):
                series.append(
                    (
                        f"reactor component {name!r} c_star",
                        component.c_star_concentration,
                    )
                )

    for name, variable in process.process_variables.items():
        if _is_dynamic_series(variable.values):
            series.append((f"process variable {name!r}", variable.values))

    for name, change in process.volume.volume_changes.items():
        if _is_dynamic_series(change.values):
            series.append((f"volume change {name!r}", change.values))

    if _is_dynamic_series(process.volume.total_volume):
        series.append(("measured total volume", process.volume.total_volume))

    start = process.time_axis.start
    end = process.time_axis.end
    errors = []
    for label, time_series in series:
        times = jnp.asarray(time_series.times)
        invalid = ~((times >= start) & (times <= end))
        count = int(jnp.sum(invalid))
        if count:
            errors.append(f"{label}: {count} timestamp(s) outside [{start}, {end}]")

    if errors:
        return False, "Timestamp bounds invalid:\n  - " + "\n  - ".join(errors)
    return True, (
        f"All timestamps within [{start}, {end}] {process.time_axis.unit} — OK"
    )


def validate_volume_change_sign(
    volume_change,
) -> Tuple[bool, str]:
    """
    Verify that a volume change has correct sign for its type.

    For a ``FeedVolumeChange`` all values must be ≥ 0.
    For a ``SampleVolumeChange`` all values must be ≤ 0.
    If the concrete type is unknown, fall back to verifying that the change is
    purely positive or purely negative (never mixed).

    Args:
        volume_change: FeedVolumeChange or SampleVolumeChange object.

    Returns:
        A tuple ``(is_valid, message)``.
    """
    vals = jnp.asarray(volume_change.values.values)
    eps = 1e-12

    if isinstance(volume_change, FeedVolumeChange):
        if bool(jnp.all(vals >= -eps)):
            return True, (
                f"Volume change '{volume_change.name}' (FeedVolumeChange) has all "
                "non-negative values — OK"
            )
        return False, (
            f"Volume change '{volume_change.name}' (FeedVolumeChange) contains "
            "negative values. Feed volume changes must have all values >= 0."
        )
    elif isinstance(volume_change, SampleVolumeChange):
        if bool(jnp.all(vals <= eps)):
            return True, (
                f"Volume change '{volume_change.name}' (SampleVolumeChange) has all "
                "non-positive values — OK"
            )
        return False, (
            f"Volume change '{volume_change.name}' (SampleVolumeChange) contains "
            "positive values. Sample volume changes must have all values <= 0."
        )
    else:
        # Fallback for unknown types
        all_non_negative = bool(jnp.all(vals >= 0))
        all_non_positive = bool(jnp.all(vals <= 0))

        if all_non_negative or all_non_positive:
            sign = "positive" if all_non_negative else "negative"
            return True, (f"Volume change '{volume_change.name}' is purely {sign} — OK")
        return False, (
            f"Volume change '{volume_change.name}' contains mixed positive and "
            "negative values. Each volume change must be purely positive or purely "
            "negative."
        )


def validate_volume_change_states(
    process: BioProcess,
) -> Tuple[bool, str]:
    """
    For every *positive* volume change, verify that all dynamic state
    variables defined in the reactor medium are also present as components
    in the referenced feed medium and use the same unit string.

    A "state variable" is a reactor-medium component whose concentration is
    a TimeSeries (i.e. it is measured dynamically over time), as opposed to a
    StaticVariable.

    Args:
        process: BioProcess object to validate.

    Returns:
        A tuple ``(is_valid, message)`` where ``is_valid`` is ``True`` when
        every positive volume change covers all dynamic state variables with
        matching unit strings.
    """
    # Collect names of dynamic state variables in the reactor medium
    state_names: List[str] = []
    if process.reactor_medium and process.reactor_medium.components:
        for comp_name, comp in process.reactor_medium.components.items():
            if _is_dynamic_series(comp.concentration):
                state_names.append(comp_name)

    if not state_names:
        return (
            True,
            "No dynamic state variables found in reactor medium — check skipped",
        )

    errors: List[str] = []

    for vc_name, vc in process.volume.volume_changes.items():
        vals = jnp.asarray(vc.values.values)
        all_non_negative = bool(jnp.all(vals >= 0))
        has_positive = bool(jnp.any(vals > 0))
        is_positive = all_non_negative and has_positive
        if not is_positive:
            continue  # only check positive (inflowing) volume changes

        # Check that the feed medium defines all state variables with matching units
        if not isinstance(vc, FeedVolumeChange):
            continue  # SampleVolumeChange has no feed medium
        feed = vc.feed_medium
        if feed is None:
            errors.append(
                f"Volume change '{vc_name}' is positive but has no feed medium defined."
            )
            continue

        feed_component_names = set(feed.components.keys()) if feed.components else set()
        missing = [s for s in state_names if s not in feed_component_names]
        if missing:
            errors.append(
                f"Volume change '{vc_name}' (feed: '{feed.name}') is missing "
                f"feed components for state variable(s): {missing}"
            )

        for state_name in state_names:
            if state_name not in feed_component_names:
                continue
            reactor_unit = process.reactor_medium.components[state_name].unit
            feed_unit = feed.components[state_name].unit
            if feed_unit != reactor_unit:
                errors.append(
                    f"Volume change '{vc_name}' (feed: '{feed.name}') component "
                    f"'{state_name}' uses unit {feed_unit!r}; reactor medium uses "
                    f"{reactor_unit!r}."
                )

    if errors:
        return False, "State-variable/feed consistency errors:\n" + "\n".join(
            f"  - {e}" for e in errors
        )
    return True, (
        "All positive volume changes cover all dynamic state variables with matching "
        "units — OK"
    )


def validate_biomass_in_reactor_medium(process: BioProcess) -> Tuple[bool, str]:
    """
    Check that the reactor medium contains a component whose name is
    ``'biomass'`` (case-insensitive).

    Args:
        process: BioProcess object to validate.

    Returns:
        A tuple ``(is_valid, message)`` where ``is_valid`` is ``True`` when a
        biomass component is found and ``message`` contains a human-readable
        summary.
    """
    if not process.reactor_medium or not process.reactor_medium.components:
        return False, (
            "Reactor medium has no components — cannot verify biomass presence"
        )

    biomass_keys = [
        k for k in process.reactor_medium.components if k.strip().lower() == "biomass"
    ]

    if biomass_keys:
        return True, f"Biomass found in reactor medium as '{biomass_keys[0]}' — OK"
    return False, (
        "Reactor medium does not contain a 'biomass' component. "
        f"Found components: {list(process.reactor_medium.components.keys())}"
    )


def validate_volume_units(process: BioProcess) -> Tuple[bool, str]:
    """Require every volume change to use the process volume unit."""
    mismatches = [
        f"{name!r} uses {change.unit!r}"
        for name, change in process.volume.volume_changes.items()
        if change.unit != process.volume.unit
    ]
    if mismatches:
        return False, (
            f"Volume changes must use volume unit {process.volume.unit!r}: "
            + ", ".join(mismatches)
        )
    return True, f"Volume changes use volume unit {process.volume.unit!r} — OK"


def validate_process(process: BioProcess) -> Tuple[bool, List[str]]:
    """
    Run all available validation checks on a single BioProcess.

    Checks performed:
    - Discrete-event time shape, ordering, bounds, and label count.
    - TimeSeries shape and ordering for every reactor-medium component,
      process variable, volume change, and measured total volume.
    - Mapping keys match embedded object names.
    - The process time axis starts no later than it ends.
    - Every timestamp falls within the process time-axis bounds.
    - Every volume change uses the process volume unit.
    - Sign consistency for every volume change.
    - State-variable / feed-medium coverage for positive volume changes.
    - Presence of a ``biomass`` component in the reactor medium.
    - Measured data falls within its own declared ``Bounds``
      (:func:`validate_bounds_against_data`).

    Args:
        process: BioProcess object to validate.

    Returns:
        A tuple ``(all_valid, messages)`` where ``all_valid`` is ``True`` only
        when every individual check passes and ``messages`` is a list of
        human-readable result strings (one per check).

    Raises:
        TypeError: If ``process`` is not a :class:`BioProcess` instance.
    """
    if not isinstance(process, BioProcess):
        raise TypeError(
            f"validate_process() expects a BioProcess instance, "
            f"got {type(process).__name__!r}"
        )
    all_valid = True
    messages: List[str] = []

    ok, msg = validate_mapping_names(process)
    messages.append(msg)
    all_valid = all_valid and ok

    ok, msg = validate_discrete_events(process)
    messages.append(msg)
    all_valid = all_valid and ok

    # --- TimeSeries shape checks ---
    # Reactor medium components
    if process.reactor_medium:
        for comp_name, comp in process.reactor_medium.components.items():
            if _is_dynamic_series(comp.concentration):
                ok, msg = validate_timeseries_shape(comp.concentration, name=comp_name)
                messages.append(msg)
                all_valid = all_valid and ok
            if _is_dynamic_series(comp.c_star_concentration):
                ok, msg = validate_timeseries_shape(
                    comp.c_star_concentration, name=f"{comp_name} c_star"
                )
                messages.append(msg)
                all_valid = all_valid and ok

    # Process variables
    for pv_name, pv in process.process_variables.items():
        if _is_dynamic_series(pv.values):
            ok, msg = validate_timeseries_shape(pv.values, name=pv_name)
            messages.append(msg)
            all_valid = all_valid and ok

    # Volume changes
    if process.volume and process.volume.volume_changes:
        for vc_name, vc in process.volume.volume_changes.items():
            if vc.values is not None:
                ok, msg = validate_timeseries_shape(vc.values, name=vc_name)
                messages.append(msg)
                all_valid = all_valid and ok

    # Measured total volume
    if _is_dynamic_series(process.volume.total_volume):
        ok, msg = validate_timeseries_shape(
            process.volume.total_volume, name="measured total volume"
        )
        messages.append(msg)
        all_valid = all_valid and ok

    # --- Time-axis checks ---
    ok, msg = validate_time_axis(process)
    messages.append(msg)
    all_valid = all_valid and ok

    ok, msg = validate_timestamp_bounds(process)
    messages.append(msg)
    all_valid = all_valid and ok

    ok, msg = validate_volume_units(process)
    messages.append(msg)
    all_valid = all_valid and ok

    if process.volume and process.volume.volume_changes:
        # --- Volume change sign checks ---
        for vc_name, vc in process.volume.volume_changes.items():
            if vc.values is not None:
                ok, msg = validate_volume_change_sign(vc)
                messages.append(msg)
                all_valid = all_valid and ok

        # --- State-variable / feed-medium coverage ---
        ok, msg = validate_volume_change_states(process)
        messages.append(msg)
        all_valid = all_valid and ok

    # --- Biomass check ---
    ok, msg = validate_biomass_in_reactor_medium(process)
    messages.append(msg)
    all_valid = all_valid and ok

    # --- Measurement/sampling alignment check ---
    ok, msg = validate_measurement_sampling_alignment(process)
    messages.append(msg)
    all_valid = all_valid and ok

    # --- Bounds sanity ---
    ok, msg = validate_bounds(process)
    messages.append(msg)
    all_valid = all_valid and ok

    # --- Bounds vs. actual data ---
    ok, msg = validate_bounds_against_data(process)
    messages.append(msg)
    all_valid = all_valid and ok

    # --- User-defined biological ODE ---
    ok, msg = validate_biological_ode(process)
    messages.append(msg)
    all_valid = all_valid and ok

    return all_valid, messages


def validate_measurement_sampling_alignment(
    process: BioProcess,
    rel_threshold: float = 1e-4,
) -> Tuple[bool, str]:
    """
    Check that reactor medium measurement times are not slightly offset from
    sampling times.

    When a concentration measurement is taken just *after* a sampling event
    (e.g. 0.0003 h later), the accumulated dilution factor (ADF) in the
    pseudobatch transform may use the wrong reactor volume, corrupting the
    normalisation and downstream spline calculations.

    This function flags every measurement time point that is close to (but not
    exactly at) a sampling time point, where "close" means within
    ``rel_threshold`` of the total process length.

    Args:
        process: BioProcess object to validate.
        rel_threshold: Maximum relative deviation (fraction of process length)
            that is considered "suspiciously close".  Default is ``1e-4``
            (0.01 %).

    Returns:
        A tuple ``(is_valid, message)`` where ``is_valid`` is ``False`` when
        at least one near-miss is detected.
    """
    # Collect sampling times from SampleVolumeChange objects
    sampling_times_list: List[float] = []
    if process.volume and process.volume.volume_changes:
        for vc in process.volume.volume_changes.values():
            if isinstance(vc, SampleVolumeChange) and _is_dynamic_series(vc.values):
                sampling_times_list.extend(
                    float(t) for t in jnp.asarray(vc.values.times)
                )

    if not sampling_times_list:
        return True, "No sampling events — measurement/sampling alignment check skipped"

    sampling_times = jnp.array(sorted(sampling_times_list))
    proc_length = float(process.time_axis.end - process.time_axis.start)
    if proc_length <= 0:
        return (
            True,
            "Process length is zero — measurement/sampling alignment check skipped",
        )
    abs_threshold = rel_threshold * proc_length

    warnings: List[str] = []

    if process.reactor_medium and process.reactor_medium.components:
        for comp_name, comp in process.reactor_medium.components.items():
            if not _is_dynamic_series(comp.concentration):
                continue
            meas_times = jnp.asarray(comp.concentration.times)
            for mt in meas_times:
                mt_f = float(mt)
                idx = int(jnp.argmin(jnp.abs(sampling_times - mt_f)))
                nearest_st = float(sampling_times[idx])
                delta = mt_f - nearest_st
                if 0 < delta <= abs_threshold:
                    warnings.append(
                        f"  '{comp_name}': measurement at t={mt_f:.6f} is "
                        f"{delta:.6f} {process.time_axis.unit} after sampling "
                        f"at t={nearest_st:.6f} "
                        f"({delta / proc_length * 100:.4f}% of process length)"
                    )

    if warnings:
        header = (
            "Measurement times are slightly offset from sampling times. "
            "This can lead to incorrect ADF values in the pseudobatch "
            "normalisation and errors in the spline calculation.\n"
        )
        return False, header + "\n".join(warnings)
    return True, "Measurement/sampling time alignment — OK"


def validate_volume_consistency(
    process: BioProcess, final_volume: Optional[float] = None
) -> Tuple[bool, str, float]:
    """
    Validate that volume changes sum to expected final volume.

    This function checks whether the sum of all volume changes (feeds, sampling, etc.)
    is consistent with the expected final volume. It handles both continuous
    (cumulative time series) and discrete volume changes.

    Note: as these values may be on different time-scale and this check is
    supposed to be run _before_ any modeling or spline interpolation happens,
    here only the last time points are considered.

    Args:
        process: BioProcess object whose volume changes are validated.
        final_volume: Expected final volume in the process's volume unit.  When
            provided the function checks that the sum of all volume changes plus
            ``process.volume.initial_volume`` is within 5 % of this value.

    Returns:
        A tuple ``(is_valid, message, total_change)`` where:

        - ``is_valid`` is ``True`` when the relative deviation between the
          calculated and expected final volume is ≤ 5 %.
        - ``message`` is a human-readable summary of the volume balance.
        - ``total_change`` is the net volume change (sum of all individual
          volume changes, in the process's volume unit).
    """

    volume = process.volume

    # Calculate total volume change and collect data for plotting
    total_change = 0.0
    messages = []

    for name, change in volume.volume_changes.items():
        if change.is_continuous:
            # For continuous changes, data should be cumulative
            values = change.values.values
            # Cumulative volume: final - initial
            change_vol = float(values[-1] - values[0])
            total_change += change_vol
            messages.append(
                f"  {name:15}: {change_vol:+8.2f} {volume.unit} (continuous)"
            )
        elif not change.is_continuous:
            # For discrete changes, sum all values from the timeseries
            values = change.values.values
            change_vol = float(jnp.sum(values))
            total_change += change_vol
            messages.append(f"  {name:15}: {change_vol:+8.2f} {volume.unit} (discrete)")

    calculated_final = volume.initial_volume + total_change

    diff = abs(calculated_final - final_volume)
    delta = total_change
    rel_diff = diff / final_volume if final_volume > 0 else 0

    messages.insert(0, f"Initial volume   : {volume.initial_volume:8.2f} {volume.unit}")
    messages.append(f"Total change     : {total_change:8.2f} {volume.unit}")
    messages.append(f"Calculated final : {calculated_final:8.2f} {volume.unit}")
    messages.append(f"Expected final   : {final_volume:8.2f} {volume.unit}")
    messages.append(
        f"Difference       : {diff:8.2f} {volume.unit} ({rel_diff * 100:.1f}%)"
    )

    if rel_diff > 0.05:  # More than 5% difference
        return (False, "Volume inconsistency detected:\n" + "\n".join(messages), delta)
    else:
        return (True, "Volume balance OK:\n" + "\n".join(messages), delta)


def validate_cross_process_consistency(
    collection: BioProcessCollection,
) -> Tuple[bool, List[str]]:
    """Verify every process in *collection* shares identical structure.

    Compares, against the first process as reference:

    - The same reactor-medium component names, each with the same concentration
      type (``TimeSeries`` or ``StaticVariable``) and unit.
    - The same process-variable names, each with the same value type
      (``TimeSeries`` or ``StaticVariable``) and unit.
    - The same volume unit.
    - The same volume-change names and units.
    - The same time-axis unit and reference. Start and end may differ.

    Collections with zero or one process trivially pass.

    Returns ``(is_valid, messages)`` — ``messages`` lists each mismatch found,
    empty when consistent.
    """
    processes = collection.processes
    if len(processes) <= 1:
        return True, []

    consistency_errors: List[str] = []

    # Build a reference signature from the first process
    first_name, first_process = next(iter(processes.items()))

    def _reactor_signature(process: BioProcess) -> Dict[str, Tuple[str, str]]:
        """Map each reactor medium component name to (concentration type name, unit)."""
        if not process.reactor_medium or not process.reactor_medium.components:
            return {}
        return {
            name: (type(comp.concentration).__name__, comp.unit)
            for name, comp in process.reactor_medium.components.items()
        }

    def _pv_signature(process: BioProcess) -> Dict[str, Tuple[str, str]]:
        """Map each process variable name to (value type name, unit)."""
        return {
            name: (type(pv.values).__name__, pv.unit)
            for name, pv in process.process_variables.items()
        }

    def _vc_signature(process: BioProcess) -> Dict[str, str]:
        """Map each volume change name to its unit."""
        if not process.volume or not process.volume.volume_changes:
            return {}
        return {name: vc.unit for name, vc in process.volume.volume_changes.items()}

    ref_reactor = _reactor_signature(first_process)
    ref_pv = _pv_signature(first_process)
    ref_volume_unit = first_process.volume.unit
    ref_vc = _vc_signature(first_process)
    ref_time_axis = (
        first_process.time_axis.unit,
        first_process.time_axis.time_reference,
    )

    for proc_name, process in processes.items():
        if proc_name == first_name:
            continue

        time_axis = (process.time_axis.unit, process.time_axis.time_reference)
        if time_axis != ref_time_axis:
            consistency_errors.append(
                f"Process '{proc_name}' time axis differs from '{first_name}': "
                f"expected {ref_time_axis}, got {time_axis}"
            )

        if process.volume.unit != ref_volume_unit:
            consistency_errors.append(
                f"Process '{proc_name}' volume unit differs from '{first_name}': "
                f"expected {ref_volume_unit!r}, got {process.volume.unit!r}"
            )

        reactor_sig = _reactor_signature(process)
        if reactor_sig != ref_reactor:
            consistency_errors.append(
                f"Process '{proc_name}' reactor medium components differ from "
                f"'{first_name}': expected {ref_reactor}, got {reactor_sig}"
            )

        pv_sig = _pv_signature(process)
        if pv_sig != ref_pv:
            consistency_errors.append(
                f"Process '{proc_name}' process variables differ from "
                f"'{first_name}': expected {ref_pv}, got {pv_sig}"
            )

        vc_sig = _vc_signature(process)
        if vc_sig != ref_vc:
            consistency_errors.append(
                f"Process '{proc_name}' volume changes differ from "
                f"'{first_name}': expected {ref_vc}, got {vc_sig}"
            )

    return not consistency_errors, consistency_errors


def validate_for_publication(
    collection: BioProcessCollection,
) -> Tuple[bool, Dict[str, List[str]]]:
    """
    Validate a collection for storage/publication as a coherent case study.

    This is bp-format's own concern — is this collection well-formed and
    internally coherent — distinct from bp-train's training-readiness
    concern (``bp_train.validation.validate_for_training``). Runs
    :func:`validate_process` for every process, then
    :func:`validate_cross_process_consistency` and
    :func:`validate_augmented_parent_refs`.

    Args:
        collection: :class:`BioProcessCollection` object to validate.

    Returns:
        A tuple ``(all_valid, report)`` where ``all_valid`` is ``True`` only
        when every per-process validation passes *and* the cross-process
        structure is consistent, and ``report`` is a dict mapping each process
        name to its list of validation messages.  Cross-process consistency
        errors are stored under the key ``"__consistency__"``, augmented-parent
        findings under the key ``"__augmented__"``.

    Raises:
        TypeError: If ``collection`` is not a :class:`BioProcessCollection`.
    """
    if not isinstance(collection, BioProcessCollection):
        raise TypeError(
            f"validate_for_publication() expects a BioProcessCollection "
            f"instance, got {type(collection).__name__!r}"
        )

    all_valid = True
    report: Dict[str, List[str]] = {}

    # --- Per-process validation ---
    for proc_name, process in collection.processes.items():
        ok, messages = validate_process(process)
        report[proc_name] = messages
        all_valid = all_valid and ok

    if not collection.processes:
        return all_valid, report

    # --- Cross-process consistency ---
    consistency_ok, consistency_errors = validate_cross_process_consistency(collection)
    report["__consistency__"] = (
        consistency_errors
        if consistency_errors
        else ["Cross-process structure is consistent — OK"]
    )
    all_valid = all_valid and consistency_ok

    # --- Augmented parent-reference validation ---
    aug_ok, aug_messages = validate_augmented_parent_refs(collection)
    report["__augmented__"] = aug_messages
    all_valid = all_valid and aug_ok

    return all_valid, report


def validate_augmented_parent_refs(
    container: "BioProcessCollection",
) -> Tuple[bool, List[str]]:
    """Verify ``AugmentedBioProcess.parent_process`` references in a container.

    For every :class:`AugmentedBioProcess` in ``container.processes``, the
    referenced ``parent_process`` must be a key in the same ``processes``
    dict and must point to a non-augmented :class:`BioProcess` (chained
    augmentation is not supported in v1).

    Args:
        container: A :class:`BioProcessCollection`.

    Returns:
        ``(all_valid, messages)``. ``messages`` always contains at least one
        line summarising the result; on failure each problem is reported
        individually.
    """
    if not isinstance(container, BioProcessCollection):
        raise TypeError(
            "validate_augmented_parent_refs() expects a "
            f"BioProcessCollection instance, got {type(container).__name__!r}"
        )

    processes = container.processes
    messages: List[str] = []
    all_valid = True

    for child_name, child in processes.items():
        if not isinstance(child, AugmentedBioProcess):
            continue
        parent_name = child.parent_process
        if parent_name not in processes:
            all_valid = False
            messages.append(
                f"AugmentedBioProcess '{child_name}' references unknown "
                f"parent_process '{parent_name}'"
            )
            continue
        parent = processes[parent_name]
        if isinstance(parent, AugmentedBioProcess):
            all_valid = False
            messages.append(
                f"AugmentedBioProcess '{child_name}' references parent "
                f"'{parent_name}', which is itself augmented; "
                "chained augmentation is not supported"
            )

    if all_valid and not messages:
        messages.append("Augmented parent references are consistent — OK")
    return all_valid, messages
