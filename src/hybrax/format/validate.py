"""
Validation utilities for bioprocess data
"""

import jax.numpy as jnp
from typing import Dict, List, Literal, Optional, Sequence, Tuple
from .dataclasses import (
    AugmentedBioProcess,
    BioProcess,
    BioProcessCollection,
    Bounds,
    StaticVariable,
    TimeSeries,
    Inflow,
    Outflow,
    _check_outflow_retention,
)


_VOLUME_SIGN_EPS = 1e-12
_TIMESTAMP_BOUNDS_RTOL = 1e-7


def _timestamp_bounds_tolerance(start: float, end: float) -> float:
    """Allow for legacy float32 timestamps widened during deserialization."""
    return _TIMESTAMP_BOUNDS_RTOL * max(1.0, abs(start), abs(end))


_Verdict = Literal["PASS", "FAIL", "SKIP"]


def _check_result(verdict: _Verdict, check_name: str, detail: str) -> Tuple[bool, str]:
    """Build a check's ``(ok, message)`` result from a single template.

    ``ok`` is derived from ``verdict`` rather than passed separately, so the
    boolean and the message text can never disagree. ``SKIP`` counts as
    ``ok=True`` — skipping a check that doesn't apply isn't a failure.
    """
    ok = verdict != "FAIL"
    return ok, f"{verdict} {check_name}: {detail}"


def _join_details(details: Sequence[str], *, bulleted: bool = False) -> str:
    """Join short failure fragments into one detail string for `_check_result`."""
    if bulleted:
        return "\n  - " + "\n  - ".join(details)
    return "; ".join(details)


def _check_bounds_tuple(bounds: Bounds, label: str) -> Tuple[bool, str]:
    """Sanity-check a bounds tuple: lo <= hi when both are set.

    Returns a bare detail fragment (empty string on pass/unset) for the
    caller to assemble into its own top-level check result.
    """
    if bounds is None:
        return True, ""
    lo, hi = bounds
    if lo is not None and hi is not None and lo > hi:
        return False, f"{label} has lo={lo} > hi={hi} (lo must be <= hi)"
    return True, ""


def validate_biological_ode(process: BioProcess) -> Tuple[bool, str]:
    """Validate ``process.biological_ode`` if present.

    Checks (skipped when ``biological_ode is None``):

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
        return _check_result(
            "SKIP", "biological_ode",
            "process.biological_ode is None — structural checks skipped",
        )

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
    # have those states share a unit, UNLESS a term is individually scaled by
    # a declared rate. A lone ``rate * state`` (a ``Mul``, never inspected
    # here at all) is already trusted to bridge units via the rate; the same
    # trust extends per-addend so ``-q_a * a - r_b * b`` (each term its own
    # rate) is fine even when ``a`` and ``b`` differ, while a genuinely bare
    # ``a - b`` (no rate anywhere) still isn't. Catches things like
    # ``biomass - product`` when biomass is g/L and product is mg/L (raw
    # numerical subtraction would be meaningless). This generalises the
    # legacy intracellular unit check.
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
            bare_units = {}
            for addend in node.args:
                addend_syms = {str(s) for s in addend.free_symbols}
                if addend_syms & rate_names:
                    continue  # this term's unit is rate-mediated; exempt
                for s in addend_syms & state_names:
                    bare_units[s] = _state_unit(s)
            if len(set(bare_units.values())) > 1:
                pretty = ", ".join(f"{s}={u!r}" for s, u in sorted(bare_units.items()))
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
        ok, detail = _check_bounds_tuple(bounds, f"rate {rname!r}")
        if not ok:
            errors.append(detail)

    if errors:
        return _check_result("FAIL", "biological_ode", _join_details(errors, bulleted=True))
    return _check_result(
        "PASS", "biological_ode",
        "derivatives/algebraic/rates parse, resolve, and are acyclic with consistent units",
    )


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
        return _check_result(
            "PASS", "biological_ode_equivalence",
            f"{len(procs)} process(es) — trivially equivalent",
        )

    first_name, first_proc = procs[0]
    ref_bo = first_proc.biological_ode

    errors: List[str] = []
    for name, proc in procs[1:]:
        bo = proc.biological_ode
        if (ref_bo is None) != (bo is None):
            errors.append(
                f"process '{name}' biological_ode presence differs from '{first_name}'"
            )
            continue
        if ref_bo is None:
            continue
        if dict(bo.algebraic) != dict(ref_bo.algebraic):
            errors.append(
                f"process '{name}' biological_ode.algebraic differs from '{first_name}'"
            )
        if dict(bo.derivatives) != dict(ref_bo.derivatives):
            errors.append(
                f"process '{name}' biological_ode.derivatives differs from "
                f"'{first_name}'"
            )
        if dict(bo.rates) != dict(ref_bo.rates):
            errors.append(
                f"process '{name}' biological_ode.rates differs from '{first_name}'"
            )

    if errors:
        return _check_result(
            "FAIL", "biological_ode_equivalence", _join_details(errors, bulleted=True)
        )
    return _check_result(
        "PASS", "biological_ode_equivalence", f"identical across {len(procs)} processes"
    )


def validate_bounds(process: BioProcess) -> Tuple[bool, str]:
    """Sanity-check all bounds tuples on the process (states, PVs, volume)."""
    errors: List[str] = []
    if process.reactor_medium:
        for cname, comp in process.reactor_medium.components.items():
            ok, detail = _check_bounds_tuple(comp.bounds, f"reactor component {cname!r}")
            if not ok:
                errors.append(detail)
    for pname, pv in process.process_variables.items():
        ok, detail = _check_bounds_tuple(pv.bounds, f"process variable {pname!r}")
        if not ok:
            errors.append(detail)
    ok, detail = _check_bounds_tuple(process.volume.bounds, "volume")
    if not ok:
        errors.append(detail)
    if errors:
        return _check_result("FAIL", "bounds", _join_details(errors))
    return _check_result(
        "PASS", "bounds", "lo <= hi holds for reactor components, process variables, volume"
    )


def _check_data_against_bounds(
    value: object, bounds: Bounds, label: str
) -> Tuple[bool, str]:
    """Compare a raw measured value (StaticVariable or TimeSeries) against its
    own declared Bounds tuple; report count/min/max of violations.

    Returns a bare detail fragment (empty string when there's nothing to
    report) for the caller to assemble into its own top-level check result.
    """
    if bounds is None or (bounds[0] is None and bounds[1] is None):
        return True, ""
    lo, hi = bounds

    if isinstance(value, StaticVariable):
        v = float(value.value)
        if lo is not None and v < lo:
            return False, f"{label} value {v:g} is below lower bound {lo:g}"
        if hi is not None and v > hi:
            return False, f"{label} value {v:g} is above upper bound {hi:g}"
        return True, ""

    if not _is_dynamic_series(value):
        return True, ""

    vals = jnp.asarray(value.values)
    fragments: List[str] = []
    if lo is not None:
        n_below = int(jnp.sum(vals < lo))
        if n_below:
            fragments.append(
                f"{n_below} datapoint(s) below lower bound {lo:g} "
                f"(min observed {float(jnp.min(vals)):g})"
            )
    if hi is not None:
        n_above = int(jnp.sum(vals > hi))
        if n_above:
            fragments.append(
                f"{n_above} datapoint(s) above upper bound {hi:g} "
                f"(max observed {float(jnp.max(vals)):g})"
            )
    if fragments:
        return False, f"{label} " + "; ".join(fragments)
    return True, ""


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
            ok, detail = _check_data_against_bounds(
                comp.concentration, comp.bounds, f"reactor component {cname!r}"
            )
            if not ok:
                errors.append(detail)

    for pname, pv in process.process_variables.items():
        ok, detail = _check_data_against_bounds(
            pv.values, pv.bounds, f"process variable {pname!r}"
        )
        if not ok:
            errors.append(detail)

    if process.volume.total_volume is not None:
        ok, detail = _check_data_against_bounds(
            process.volume.total_volume, process.volume.bounds, "volume"
        )
        if not ok:
            errors.append(detail)

    if errors:
        return _check_result("FAIL", "bounds_against_data", _join_details(errors))
    return _check_result(
        "PASS", "bounds_against_data", "all measured datapoints with declared bounds fall within them"
    )


def _is_dynamic_series(value: object) -> bool:
    """Return True when *value* looks like a TimeSeries-like dynamic object."""
    times = getattr(value, "times", None)
    values = getattr(value, "values", None)
    return times is not None and values is not None


def validate_timeseries_shape(
    ts: TimeSeries, name: str = "", *, allow_empty: bool = False
) -> Tuple[bool, str]:
    """
    Check that a TimeSeries has consistent shapes and ordered times.

    Verifies:
    - ``times`` and ``values`` are 1-D arrays.
    - Both arrays have the same length.
    - ``times`` are nonempty unless ``allow_empty`` is true.
    - ``times`` are strictly monotonically increasing (no duplicates).

    Args:
        ts: TimeSeries object to validate.
        name: Optional label used in the message (e.g. the variable name).
        allow_empty: Whether matching empty arrays represent a valid event sequence.

    Returns:
        A tuple ``(is_valid, message)``.
    """
    label = f"'{name}'" if name else "(unnamed)"
    errors: List[str] = []

    if not _is_dynamic_series(ts):
        return _check_result(
            "FAIL", "timeseries_shape",
            f"TimeSeries {label} — missing discrete times/values arrays",
        )

    tp = jnp.asarray(ts.times)
    vals = jnp.asarray(ts.values)

    if tp.ndim != 1:
        errors.append(f"times must be 1-D, got shape {tp.shape}")
    if vals.ndim != 1:
        errors.append(f"values must be 1-D, got shape {vals.shape}")

    if tp.ndim == 1 and vals.ndim == 1:
        if tp.shape[0] == 0 and not allow_empty:
            errors.append("times and values must not be empty")
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
        return _check_result(
            "FAIL", "timeseries_shape", f"TimeSeries {label} — " + _join_details(errors)
        )
    return _check_result(
        "PASS", "timeseries_shape",
        f"TimeSeries {label} — 1-D, equal-length, strictly increasing times",
    )


def validate_discrete_events(process: BioProcess) -> Tuple[bool, str]:
    """Check event time shape, ordering, bounds, and label count."""
    events = process.discrete_events
    if events is None:
        return _check_result(
            "SKIP", "discrete_events",
            "process.discrete_events is None — timing/label checks skipped",
        )

    times = jnp.asarray(events.times)
    errors = []
    if times.ndim != 1:
        errors.append(f"times must be 1-D, got shape {times.shape}")
    else:
        if times.shape[0] > 1 and not bool(jnp.all(jnp.diff(times) > 0)):
            errors.append("times are not strictly monotonically increasing")

        start = process.time_axis.start
        end = process.time_axis.end
        tolerance = _timestamp_bounds_tolerance(start, end)
        outside = ~((times >= start - tolerance) & (times <= end + tolerance))
        count = int(jnp.sum(outside))
        if count:
            errors.append(f"{count} timestamp(s) outside [{start}, {end}]")

        if events.labels is not None and len(events.labels) != times.shape[0]:
            errors.append(
                f"labels length ({len(events.labels)}) does not match "
                f"times length ({times.shape[0]})"
            )

    if errors:
        return _check_result("FAIL", "discrete_events", _join_details(errors))
    return _check_result(
        "PASS", "discrete_events",
        f"{times.shape[0]} event(s), strictly increasing, within time axis bounds",
    )


def validate_time_axis(process: BioProcess) -> Tuple[bool, str]:
    """Check that the process time axis starts no later than it ends."""
    axis = process.time_axis
    if axis.start > axis.end:
        return _check_result(
            "FAIL", "time_axis", f"start {axis.start} is after end {axis.end} {axis.unit}"
        )
    return _check_result("PASS", "time_axis", f"[{axis.start}, {axis.end}] {axis.unit}")


def validate_timestamp_bounds(process: BioProcess) -> Tuple[bool, str]:
    """Check timestamps against valid inclusive time-axis bounds.

    Skip this policy-dependent check when the time axis itself is inverted.
    """
    axis_ok, _ = validate_time_axis(process)
    if not axis_ok:
        return _check_result(
            "SKIP", "timestamp_bounds", "time_axis is invalid — bounds check skipped"
        )

    series = []
    if process.reactor_medium:
        for name, component in process.reactor_medium.components.items():
            if _is_dynamic_series(component.concentration):
                series.append((f"reactor component {name!r}", component.concentration))

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
    tolerance = _timestamp_bounds_tolerance(start, end)
    errors = []
    for label, time_series in series:
        times = jnp.asarray(time_series.times)
        invalid = ~((times >= start - tolerance) & (times <= end + tolerance))
        count = int(jnp.sum(invalid))
        if count:
            errors.append(f"{label}: {count} timestamp(s) outside [{start}, {end}]")

    if errors:
        return _check_result("FAIL", "timestamp_bounds", _join_details(errors))
    return _check_result(
        "PASS", "timestamp_bounds",
        f"all timestamps within [{start}, {end}] {process.time_axis.unit}",
    )


def validate_volume_change_sign(
    volume_change,
) -> Tuple[bool, str]:
    """
    Verify that a volume change has correct sign for its type.

    For a ``Inflow`` all values must be ≥ 0.
    For a ``Outflow`` all values must be ≤ 0.
    If the concrete type is unknown, fall back to verifying that the change is
    purely positive or purely negative (never mixed).

    Args:
        volume_change: Inflow or Outflow object.

    Returns:
        A tuple ``(is_valid, message)``.
    """
    vals = jnp.asarray(volume_change.values.values)

    if isinstance(volume_change, Inflow):
        if bool(jnp.all(vals >= -_VOLUME_SIGN_EPS)):
            return _check_result(
                "PASS", "volume_change_sign",
                f"'{volume_change.name}' (Inflow) has all non-negative values",
            )
        return _check_result(
            "FAIL", "volume_change_sign",
            f"'{volume_change.name}' (Inflow) contains negative values; "
            "Inflows must have all values >= 0",
        )
    elif isinstance(volume_change, Outflow):
        if bool(jnp.all(vals <= _VOLUME_SIGN_EPS)):
            return _check_result(
                "PASS", "volume_change_sign",
                f"'{volume_change.name}' (Outflow) has all non-positive values",
            )
        return _check_result(
            "FAIL", "volume_change_sign",
            f"'{volume_change.name}' (Outflow) contains positive values; "
            "Outflows must have all values <= 0",
        )
    else:
        # Fallback for unknown types
        all_non_negative = bool(jnp.all(vals >= 0))
        all_non_positive = bool(jnp.all(vals <= 0))

        if all_non_negative or all_non_positive:
            sign = "positive" if all_non_negative else "negative"
            return _check_result(
                "PASS", "volume_change_sign", f"'{volume_change.name}' is purely {sign}"
            )
        return _check_result(
            "FAIL", "volume_change_sign",
            f"'{volume_change.name}' contains mixed positive and negative values; "
            "each volume change must be purely positive or purely negative",
        )


def validate_volume_change_states(
    process: BioProcess,
) -> Tuple[bool, str]:
    """
    For every *positive* volume change, verify that explicitly declared feed
    components corresponding to dynamic reactor states use the same unit
    string. Omitted reactor components are valid and mean zero concentration.

    A "state variable" is a reactor-medium component whose concentration is
    a TimeSeries (i.e. it is measured dynamically over time), as opposed to a
    StaticVariable.

    Args:
        process: BioProcess object to validate.

    Returns:
        A tuple ``(is_valid, message)`` where ``is_valid`` is ``True`` when
        every declared dynamic feed component uses the matching unit string.
    """
    # Collect names of dynamic state variables in the reactor medium
    state_names: List[str] = []
    if process.reactor_medium and process.reactor_medium.components:
        for comp_name, comp in process.reactor_medium.components.items():
            if _is_dynamic_series(comp.concentration):
                state_names.append(comp_name)

    if not state_names:
        return _check_result(
            "SKIP", "volume_change_states",
            "no dynamic state variables found in reactor medium",
        )

    errors: List[str] = []

    for vc_name, vc in process.volume.volume_changes.items():
        vals = jnp.asarray(vc.values.values)
        all_non_negative = bool(jnp.all(vals >= -_VOLUME_SIGN_EPS))
        has_positive = bool(jnp.any(vals > 0))
        is_positive = all_non_negative and has_positive
        if not is_positive:
            continue  # only check positive (inflowing) volume changes

        # Check that the feed medium defines all state variables with matching units
        if not isinstance(vc, Inflow):
            continue  # Outflow has no feed medium
        feed = vc.feed_medium
        if feed is None:
            errors.append(f"'{vc_name}' is positive but has no feed medium defined")
            continue

        feed_component_names = set(feed.components.keys()) if feed.components else set()
        for state_name in state_names:
            if state_name not in feed_component_names:
                continue
            reactor_unit = process.reactor_medium.components[state_name].unit
            feed_unit = feed.components[state_name].unit
            if feed_unit != reactor_unit:
                errors.append(
                    f"'{vc_name}' (feed: '{feed.name}') component '{state_name}' uses "
                    f"unit {feed_unit!r}; reactor medium uses {reactor_unit!r}"
                )

    if errors:
        return _check_result("FAIL", "volume_change_states", _join_details(errors))
    return _check_result(
        "PASS", "volume_change_states",
        "all declared dynamic feed components use matching reactor units",
    )


def validate_outflow_retention(process: BioProcess) -> Tuple[bool, str]:
    """
    For every Outflow's ``retention``, every key must be a declared
    reactor-medium component name, every value must satisfy
    ``0.0 <= sigma <= 1.0``, and a non-empty ``retention`` is only allowed
    on a continuous Outflow — the RHS mechanistic model only ever consults
    retention for continuous flows (see ``_build_retention`` in
    ``mechanistic.py``), so retention set on a discrete Outflow would
    otherwise be silently ignored rather than doing anything.

    Args:
        process: BioProcess object to validate.

    Returns:
        A tuple ``(is_valid, message)``.
    """
    rmc_names = (
        set(process.reactor_medium.components.keys())
        if process.reactor_medium
        else set()
    )
    errors: List[str] = []

    if process.volume and process.volume.volume_changes:
        for vc_name, vc in process.volume.volume_changes.items():
            if isinstance(vc, Outflow):
                errors.extend(_check_outflow_retention(vc_name, vc, rmc_names))

    if errors:
        return _check_result("FAIL", "outflow_retention", _join_details(errors))
    return _check_result("PASS", "outflow_retention", "all Outflow retention values are valid")


def validate_biomass_in_reactor_medium(process: BioProcess) -> Tuple[bool, str]:
    """
    Check that the reactor medium contains a component whose name is
    ``'biomass'`` (case-insensitive).

    Args:
        process: BioProcess object to validate.

    Returns:
        A tuple ``(is_valid, message)``.
    """
    if not process.reactor_medium or not process.reactor_medium.components:
        return _check_result(
            "FAIL", "biomass_in_reactor_medium",
            "reactor medium has no components — cannot verify biomass presence",
        )

    biomass_keys = [
        k for k in process.reactor_medium.components if k.strip().lower() == "biomass"
    ]

    if biomass_keys:
        return _check_result(
            "PASS", "biomass_in_reactor_medium",
            f"biomass found in reactor medium as {biomass_keys[0]!r}",
        )
    return _check_result(
        "FAIL", "biomass_in_reactor_medium",
        "reactor medium does not contain a 'biomass' component; found components: "
        f"{list(process.reactor_medium.components.keys())}",
    )


def validate_initial_state_alignment(process: BioProcess) -> Tuple[bool, str]:
    """Every dynamic reactor-medium component and process variable must have a
    measurement at ``process.time_axis.start``, matching where
    ``volume.initial_volume`` is anchored — otherwise the ODE's initial state
    and initial volume come from different points in time.
    """
    t0 = process.time_axis.start
    tolerance = _timestamp_bounds_tolerance(process.time_axis.start, process.time_axis.end)
    missing: List[str] = []

    if process.reactor_medium and process.reactor_medium.components:
        for name, comp in process.reactor_medium.components.items():
            if _is_dynamic_series(comp.concentration):
                ts = jnp.asarray(comp.concentration.times)
                if ts.shape[0] == 0 or not bool(jnp.any(jnp.abs(ts - t0) <= tolerance)):
                    missing.append(f"reactor component {name!r}")

    for name, pv in process.process_variables.items():
        if _is_dynamic_series(pv.values):
            ts = jnp.asarray(pv.values.times)
            if ts.shape[0] == 0 or not bool(jnp.any(jnp.abs(ts - t0) <= tolerance)):
                missing.append(f"process variable {name!r}")

    if missing:
        return _check_result(
            "FAIL", "initial_state_alignment",
            f"no measurement at time_axis.start={t0:g} for: {_join_details(missing)} "
            "(volume.initial_volume is anchored at time_axis.start; every dynamic "
            "state needs a value there too, or the ODE's initial condition is "
            "undefined)",
        )
    return _check_result(
        "PASS", "initial_state_alignment",
        "all dynamic reactor-medium components and process variables have a "
        f"measurement at time_axis.start={t0:g}, consistent with volume.initial_volume",
    )


def validate_volume_units(process: BioProcess) -> Tuple[bool, str]:
    """Require every volume change to use the process volume unit."""
    mismatches = [
        f"{name!r} uses {change.unit!r}"
        for name, change in process.volume.volume_changes.items()
        if change.unit != process.volume.unit
    ]
    if mismatches:
        return _check_result(
            "FAIL", "volume_units",
            f"volume changes must use volume unit {process.volume.unit!r}: "
            + ", ".join(mismatches),
        )
    return _check_result(
        "PASS", "volume_units", f"volume changes use volume unit {process.volume.unit!r}"
    )


def validate_process(process: BioProcess) -> Tuple[bool, List[Tuple[bool, str]]]:
    """
    Run all available validation checks on a single BioProcess.

    Checks performed, in order:
    - Discrete-event time shape, ordering, bounds, and label count.
    - TimeSeries shape and ordering for every reactor-medium component,
      process variable, volume change, and measured total volume.
    - The process time axis starts no later than it ends.
    - Every timestamp falls within the process time-axis bounds.
    - Every volume change uses the process volume unit.
    - Sign consistency for every volume change.
    - State-variable / feed-medium coverage for positive volume changes.
    - Outflow ``retention`` keys/values (:func:`validate_outflow_retention`).
    - Presence of a ``biomass`` component in the reactor medium.
    - Every dynamic state has a measurement at ``time_axis.start``, matching
      where ``volume.initial_volume`` is anchored.
    - Measurement/sampling timestamp alignment.
    - Bounds tuple sanity.
    - Measured data falls within its own declared ``Bounds``.
    - Biological ODE structure.

    Args:
        process: BioProcess object to validate.

    Returns:
        A tuple ``(all_valid, results)`` where ``all_valid`` is ``True`` only
        when every individual check passes and ``results`` is a list of
        ``(check_ok, message)`` pairs, one per check.

    Raises:
        TypeError: If ``process`` is not a :class:`BioProcess` instance.
    """
    if not isinstance(process, BioProcess):
        raise TypeError(
            f"validate_process() expects a BioProcess instance, "
            f"got {type(process).__name__!r}"
        )
    all_valid = True
    results: List[Tuple[bool, str]] = []

    def _record(result: Tuple[bool, str]) -> None:
        nonlocal all_valid
        ok, _ = result
        results.append(result)
        all_valid = all_valid and ok

    _record(validate_discrete_events(process))

    # --- TimeSeries shape checks ---
    # Reactor medium components
    if process.reactor_medium:
        for comp_name, comp in process.reactor_medium.components.items():
            if _is_dynamic_series(comp.concentration):
                _record(validate_timeseries_shape(comp.concentration, name=comp_name))

    # Process variables
    for pv_name, pv in process.process_variables.items():
        if _is_dynamic_series(pv.values):
            _record(validate_timeseries_shape(pv.values, name=pv_name))

    # Volume changes
    for vc_name, vc in process.volume.volume_changes.items():
        if vc.values is not None:
            _record(
                validate_timeseries_shape(
                    vc.values, name=vc_name, allow_empty=not vc.is_continuous
                )
            )

    # Measured total volume
    if _is_dynamic_series(process.volume.total_volume):
        _record(
            validate_timeseries_shape(process.volume.total_volume, name="measured total volume")
        )

    # --- Time-axis checks ---
    _record(validate_time_axis(process))
    _record(validate_timestamp_bounds(process))
    _record(validate_volume_units(process))

    if process.volume.volume_changes:
        # --- Volume change sign checks ---
        for vc_name, vc in process.volume.volume_changes.items():
            if vc.values is not None:
                _record(validate_volume_change_sign(vc))

        # --- State-variable / feed-medium coverage ---
        _record(validate_volume_change_states(process))

        # --- Outflow component retention ---
        _record(validate_outflow_retention(process))

    # --- Biomass check ---
    _record(validate_biomass_in_reactor_medium(process))

    # --- Initial-state / initial-volume alignment check ---
    _record(validate_initial_state_alignment(process))

    # --- Measurement/sampling alignment check ---
    _record(validate_measurement_sampling_alignment(process))

    # --- Bounds sanity ---
    _record(validate_bounds(process))

    # --- Bounds vs. actual data ---
    _record(validate_bounds_against_data(process))

    # --- User-defined biological ODE ---
    _record(validate_biological_ode(process))

    return all_valid, results


def validate_measurement_sampling_alignment(
    process: BioProcess,
    rel_threshold: float = 1e-4,
) -> Tuple[bool, str]:
    """
    Check that reactor medium measurement times are not slightly offset from
    sampling times.

    When a concentration measurement is taken just *after* a sampling event
    (e.g. 0.0003 h later), it's ambiguous which side of the (discontinuous)
    event it belongs to — the accumulated dilution factor (ADF) in the
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
        A tuple ``(is_valid, message)``.
    """
    # Collect sampling times from Outflow objects
    sampling_times_list: List[float] = []
    if process.volume and process.volume.volume_changes:
        for vc in process.volume.volume_changes.values():
            if (
                isinstance(vc, Outflow)
                and not vc.is_continuous
                and _is_dynamic_series(vc.values)
            ):
                sampling_times_list.extend(
                    float(t) for t in jnp.asarray(vc.values.times)
                )

    if not sampling_times_list:
        return _check_result(
            "SKIP", "measurement_sampling_alignment", "no sampling events found"
        )

    sampling_times = jnp.array(sorted(sampling_times_list))
    proc_length = float(process.time_axis.end - process.time_axis.start)
    if proc_length <= 0:
        return _check_result(
            "SKIP", "measurement_sampling_alignment", "process length is zero"
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
                        f"'{comp_name}' measurement at t={mt_f:.6f} is {delta:.6f} "
                        f"{process.time_axis.unit} after sampling at t={nearest_st:.6f} "
                        f"({delta / proc_length * 100:.4f}% of process length)"
                    )

    if warnings:
        return _check_result(
            "FAIL", "measurement_sampling_alignment",
            "measurement times slightly offset from sampling times, which can cause "
            "incorrect ADF values in the pseudobatch normalisation and errors in "
            "the spline calculation: " + _join_details(warnings),
        )
    return _check_result(
        "PASS", "measurement_sampling_alignment",
        "no reactor-medium measurement times are suspiciously close to sampling events",
    )


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
    table: List[str] = []

    for name, change in volume.volume_changes.items():
        if change.is_continuous:
            # For continuous changes, data should be cumulative
            values = change.values.values
            # Cumulative volume: final - initial
            change_vol = float(values[-1] - values[0])
            total_change += change_vol
            table.append(
                f"  {name:15}: {change_vol:+8.2f} {volume.unit} (continuous)"
            )
        elif not change.is_continuous:
            # For discrete changes, sum all values from the timeseries
            values = change.values.values
            change_vol = float(jnp.sum(values))
            total_change += change_vol
            table.append(f"  {name:15}: {change_vol:+8.2f} {volume.unit} (discrete)")

    calculated_final = volume.initial_volume + total_change

    diff = abs(calculated_final - final_volume)
    delta = total_change
    rel_diff = diff / final_volume if final_volume > 0 else 0

    table.insert(0, f"Initial volume   : {volume.initial_volume:8.2f} {volume.unit}")
    table.append(f"Total change     : {total_change:8.2f} {volume.unit}")
    table.append(f"Calculated final : {calculated_final:8.2f} {volume.unit}")
    table.append(f"Expected final   : {final_volume:8.2f} {volume.unit}")
    table.append(
        f"Difference       : {diff:8.2f} {volume.unit} ({rel_diff * 100:.1f}%)"
    )

    detail = "\n" + "\n".join(table)
    verdict = "FAIL" if rel_diff > 0.05 else "PASS"  # More than 5% difference
    ok, msg = _check_result(verdict, "volume_consistency", detail)
    return ok, msg, delta


def validate_cross_process_consistency(
    collection: BioProcessCollection,
) -> Tuple[bool, List[Tuple[bool, str]]]:
    """Verify every process in *collection* shares identical structure.

    Compares, against the first process as reference:

    - The same reactor-medium component names, each with the same concentration
      type (``TimeSeries`` or ``StaticVariable``) and unit.
    - The same process-variable names, each with the same value type
      (``TimeSeries`` or ``StaticVariable``) and unit.
    - The same volume unit.
    - The same time-axis unit and reference. Start and end may differ.

    Volume-change names may differ because processes in one study can use
    different feed and sampling strategies.

    Collections with zero or one process trivially pass.

    Returns ``(is_valid, results)`` — ``results`` always contains at least
    one ``(check_ok, message)`` entry: one per mismatch found, or a single
    ``PASS`` entry when consistent.
    """
    processes = collection.processes
    if len(processes) <= 1:
        return True, [
            _check_result(
                "PASS", "cross_process_consistency",
                f"{len(processes)} process(es) — trivially consistent",
            )
        ]

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

    ref_reactor = _reactor_signature(first_process)
    ref_pv = _pv_signature(first_process)
    ref_volume_unit = first_process.volume.unit
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
                f"process '{proc_name}' time axis differs from '{first_name}': "
                f"expected {ref_time_axis}, got {time_axis}"
            )

        if process.volume.unit != ref_volume_unit:
            consistency_errors.append(
                f"process '{proc_name}' volume unit differs from '{first_name}': "
                f"expected {ref_volume_unit!r}, got {process.volume.unit!r}"
            )

        reactor_sig = _reactor_signature(process)
        if reactor_sig != ref_reactor:
            consistency_errors.append(
                f"process '{proc_name}' reactor medium components differ from "
                f"'{first_name}': expected {ref_reactor}, got {reactor_sig}"
            )

        pv_sig = _pv_signature(process)
        if pv_sig != ref_pv:
            consistency_errors.append(
                f"process '{proc_name}' process variables differ from "
                f"'{first_name}': expected {ref_pv}, got {pv_sig}"
            )

    if not consistency_errors:
        return True, [
            _check_result(
                "PASS", "cross_process_consistency",
                f"structure matches '{first_name}' across {len(processes)} processes",
            )
        ]
    return False, [
        _check_result("FAIL", "cross_process_consistency", e) for e in consistency_errors
    ]


def validate_for_publication(
    collection: BioProcessCollection,
) -> Tuple[bool, Dict[str, List[Tuple[bool, str]]]]:
    """
    Validate a collection for storage/publication as a coherent case study.

    This is bp-format's own concern — is this collection well-formed and
    internally coherent — distinct from bp-train's training-readiness
    concern (``bp_train.validate.validate_for_training``). Runs
    :func:`validate_process` for every process, then
    :func:`validate_cross_process_consistency` and
    :func:`validate_augmented_parent_refs`.

    Args:
        collection: :class:`BioProcessCollection` object to validate.

    Returns:
        A tuple ``(all_valid, report)`` where ``all_valid`` is ``True`` only
        when every per-process validation passes *and* the cross-process
        structure is consistent, and ``report`` is a dict mapping each
        process name to its list of ``(check_ok, message)`` results.
        Cross-process consistency results are stored under the key
        ``"__consistency__"``, augmented-parent findings under the key
        ``"__augmented__"`` — both always non-empty.

    Raises:
        TypeError: If ``collection`` is not a :class:`BioProcessCollection`.
    """
    if not isinstance(collection, BioProcessCollection):
        raise TypeError(
            f"validate_for_publication() expects a BioProcessCollection "
            f"instance, got {type(collection).__name__!r}"
        )

    all_valid = True
    report: Dict[str, List[Tuple[bool, str]]] = {}

    # --- Per-process validation ---
    for proc_name, process in collection.processes.items():
        ok, results = validate_process(process)
        report[proc_name] = results
        all_valid = all_valid and ok

    if not collection.processes:
        return all_valid, report

    # --- Cross-process consistency ---
    consistency_ok, consistency_results = validate_cross_process_consistency(collection)
    report["__consistency__"] = consistency_results
    all_valid = all_valid and consistency_ok

    # --- Augmented parent-reference validation ---
    aug_ok, aug_results = validate_augmented_parent_refs(collection)
    report["__augmented__"] = aug_results
    all_valid = all_valid and aug_ok

    return all_valid, report


def validate_augmented_parent_refs(
    container: "BioProcessCollection",
) -> Tuple[bool, List[Tuple[bool, str]]]:
    """Verify ``AugmentedBioProcess.parent_process`` references in a container.

    For every :class:`AugmentedBioProcess` in ``container.processes``, the
    referenced ``parent_process`` must be a key in the same ``processes``
    dict and must point to a non-augmented :class:`BioProcess` (chained
    augmentation is not supported in v1).

    Args:
        container: A :class:`BioProcessCollection`.

    Returns:
        ``(all_valid, results)``. ``results`` always contains at least one
        ``(check_ok, message)`` entry: one per problem found, or a single
        ``PASS`` entry when there are none.
    """
    if not isinstance(container, BioProcessCollection):
        raise TypeError(
            "validate_augmented_parent_refs() expects a "
            f"BioProcessCollection instance, got {type(container).__name__!r}"
        )

    processes = container.processes
    results: List[Tuple[bool, str]] = []
    all_valid = True

    for child_name, child in processes.items():
        if not isinstance(child, AugmentedBioProcess):
            continue
        parent_name = child.parent_process
        if parent_name not in processes:
            all_valid = False
            results.append(
                _check_result(
                    "FAIL", "augmented_parent_refs",
                    f"AugmentedBioProcess {child_name!r} references unknown "
                    f"parent_process {parent_name!r}",
                )
            )
            continue
        parent = processes[parent_name]
        if isinstance(parent, AugmentedBioProcess):
            all_valid = False
            results.append(
                _check_result(
                    "FAIL", "augmented_parent_refs",
                    f"AugmentedBioProcess {child_name!r} references parent "
                    f"{parent_name!r}, which is itself augmented; chained "
                    "augmentation is not supported",
                )
            )

    if not results:
        n_augmented = sum(isinstance(p, AugmentedBioProcess) for p in processes.values())
        results.append(
            _check_result(
                "PASS", "augmented_parent_refs",
                f"{n_augmented} augmented process(es) have valid parent references",
            )
        )
    return all_valid, results
