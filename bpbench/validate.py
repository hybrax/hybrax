"""
Validation utilities for bioprocess data
"""

import jax.numpy as jnp
from typing import Dict, List, Optional, Tuple
from .dataclasses import BioProcess, CaseStudy, TimeSeries, VolumeChange


def validate_timeseries_shape(ts: TimeSeries, name: str = "") -> Tuple[bool, str]:
    """
    Check that a TimeSeries has consistent shapes and ordered time points.

    Verifies:
    - ``timepoints`` and ``values`` are 1-D arrays.
    - Both arrays have the same length.
    - ``timepoints`` are strictly monotonically increasing (no duplicates).

    Args:
        ts: TimeSeries object to validate.
        name: Optional label used in error messages (e.g. the variable name).

    Returns:
        A tuple ``(is_valid, message)`` where ``is_valid`` is ``True`` when all
        checks pass and ``message`` contains a human-readable summary.
    """
    label = f"'{name}' " if name else ""
    errors: List[str] = []

    tp = jnp.asarray(ts.timepoints)
    vals = jnp.asarray(ts.values)

    if tp.ndim != 1:
        errors.append(f"timepoints must be 1-D, got shape {tp.shape}")
    if vals.ndim != 1:
        errors.append(f"values must be 1-D, got shape {vals.shape}")

    if tp.ndim == 1 and vals.ndim == 1:
        if tp.shape[0] != vals.shape[0]:
            errors.append(
                f"timepoints length ({tp.shape[0]}) does not match "
                f"values length ({vals.shape[0]})"
            )
        if tp.shape[0] > 1:
            diffs = jnp.diff(tp)
            if not bool(jnp.all(diffs > 0)):
                errors.append("timepoints are not strictly monotonically increasing")

    if errors:
        return False, f"TimeSeries {label}invalid:\n" + "\n".join(f"  - {e}" for e in errors)
    return True, f"TimeSeries {label}OK"


def validate_volume_change_sign(
    volume_change: VolumeChange,
) -> Tuple[bool, str]:
    """
    Verify that a volume change is purely positive or purely negative.

    For a *continuous* volume change the individual values (e.g. a feed rate)
    must all be ≥ 0 (purely positive/zero) or all be ≤ 0 (purely
    negative/zero).  For a *discrete* volume change the individual event
    values must satisfy the same constraint.

    Args:
        volume_change: VolumeChange object whose sign consistency is checked.

    Returns:
        A tuple ``(is_valid, message)`` where ``is_valid`` is ``True`` when the
        change is purely positive or purely negative and ``message`` contains a
        human-readable summary.
    """
    vals = jnp.asarray(volume_change.values.values)
    all_non_negative = bool(jnp.all(vals >= 0))
    all_non_positive = bool(jnp.all(vals <= 0))

    if all_non_negative or all_non_positive:
        sign = "positive" if all_non_negative else "negative"
        return True, (
            f"Volume change '{volume_change.name}' is purely {sign} — OK"
        )
    return False, (
        f"Volume change '{volume_change.name}' contains mixed positive and "
        "negative values. Each volume change must be purely positive or purely negative."
    )


def validate_volume_change_states(
    process: BioProcess,
) -> Tuple[bool, str]:
    """
    For every *positive* volume change, verify that all dynamic state
    variables defined in the reactor medium are also present as components
    in the referenced feed medium of that volume change.

    A "state variable" is a reactor-medium component whose concentration is
    a TimeSeries (i.e. it is measured dynamically over time), as opposed to a
    StaticVariable.

    Args:
        process: BioProcess object to validate.

    Returns:
        A tuple ``(is_valid, message)`` where ``is_valid`` is ``True`` when
        every positive volume change covers all dynamic state variables.
    """
    # Collect names of dynamic state variables in the reactor medium
    state_names: List[str] = []
    if process.reactor_medium and process.reactor_medium.components:
        for comp_name, comp in process.reactor_medium.components.items():
            if hasattr(comp.concentration, "timepoints"):
                state_names.append(comp_name)

    if not state_names:
        return True, "No dynamic state variables found in reactor medium — check skipped"

    errors: List[str] = []

    for vc_name, vc in process.volume.volume_changes.items():
        vals = jnp.asarray(vc.values.values)
        all_non_negative = bool(jnp.all(vals >= 0))
        has_positive = bool(jnp.any(vals > 0))
        is_positive = all_non_negative and has_positive
        if not is_positive:
            continue  # only check positive (inflowing) volume changes

        # Check that the feed medium defines all state variables
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

    if errors:
        return False, "State-variable/feed consistency errors:\n" + "\n".join(
            f"  - {e}" for e in errors
        )
    return True, "All positive volume changes cover all dynamic state variables — OK"


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
        k for k in process.reactor_medium.components
        if k.strip().lower() == "biomass"
    ]

    if biomass_keys:
        return True, f"Biomass found in reactor medium as '{biomass_keys[0]}' — OK"
    return False, (
        "Reactor medium does not contain a 'biomass' component. "
        f"Found components: {list(process.reactor_medium.components.keys())}"
    )


def validate_process(process: BioProcess) -> Tuple[bool, List[str]]:
    """
    Run all available validation checks on a single BioProcess.

    Checks performed:
    - TimeSeries shape and ordering for every reactor-medium component and
      process variable that carries a TimeSeries.
    - Sign consistency for every volume change.
    - State-variable / feed-medium coverage for positive volume changes.
    - Presence of a ``biomass`` component in the reactor medium.

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

    # --- TimeSeries shape checks ---
    # Reactor medium components
    if process.reactor_medium:
        for comp_name, comp in process.reactor_medium.components.items():
            if hasattr(comp.concentration, "timepoints"):
                ok, msg = validate_timeseries_shape(comp.concentration, name=comp_name)
                messages.append(msg)
                all_valid = all_valid and ok

    # Process variables
    for pv_name, pv in process.process_variables.items():
        if hasattr(pv.values, "timepoints"):
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

    return all_valid, messages


def validate_volume_consistency(process: BioProcess,
                                final_volume: Optional[float] = None) -> Tuple[bool, str, float]:
    """
    Validate that volume changes sum to expected final volume.
    
    This function checks whether the sum of all volume changes (feeds, sampling, etc.)
    is consistent with the expected final volume. It handles both continuous 
    (cumulative time series) and discrete volume changes.
    
    Note: as these values may be on different time-scale and this check is supposed to be
    run _before_ any modeling or spline interpolation happens, here only the last time points 
    are considered.

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
            messages.append(f"  {name:15}: {change_vol:+8.2f} {volume.unit} (continuous)")
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
    messages.append(f"Difference       : {diff:8.2f} {volume.unit} ({rel_diff*100:.1f}%)")
    
    if rel_diff > 0.05:  # More than 5% difference
        return (False, "Volume inconsistency detected:\n" + "\n".join(messages), delta)
    else:
        return (True, "Volume balance OK:\n" + "\n".join(messages), delta)


def validate_case_study(case_study: CaseStudy) -> Tuple[bool, Dict[str, List[str]]]:
    """
    Validate all processes in a case study and check cross-process consistency.

    Runs :func:`validate_process` for every process in the case study, then
    verifies that all processes share identical structure:

    - The same reactor-medium component names, each with the same concentration
      type (``TimeSeries`` or ``StaticVariable``).
    - The same process-variable names, each with the same value type
      (``TimeSeries`` or ``StaticVariable``).
    - The same volume-change names.

    Args:
        case_study: :class:`CaseStudy` object to validate.

    Returns:
        A tuple ``(all_valid, report)`` where ``all_valid`` is ``True`` only
        when every per-process validation passes *and* the cross-process
        structure is consistent, and ``report`` is a dict mapping each process
        name to its list of validation messages.  Cross-process consistency
        errors are stored under the key ``"__consistency__"``.

    Raises:
        TypeError: If ``case_study`` is not a :class:`CaseStudy` instance.
    """
    if not isinstance(case_study, CaseStudy):
        raise TypeError(
            f"validate_case_study() expects a CaseStudy instance, "
            f"got {type(case_study).__name__!r}"
        )

    all_valid = True
    report: Dict[str, List[str]] = {}

    # --- Per-process validation ---
    for proc_name, process in case_study.processes.items():
        ok, messages = validate_process(process)
        report[proc_name] = messages
        all_valid = all_valid and ok

    if not case_study.processes:
        return all_valid, report

    # --- Cross-process consistency ---
    consistency_errors: List[str] = []

    # Build a reference signature from the first process
    first_name, first_process = next(iter(case_study.processes.items()))

    def _reactor_signature(process: BioProcess) -> Dict[str, str]:
        """Map each reactor medium component name to its concentration type name."""
        if not process.reactor_medium or not process.reactor_medium.components:
            return {}
        return {
            name: type(comp.concentration).__name__
            for name, comp in process.reactor_medium.components.items()
        }

    def _pv_signature(process: BioProcess) -> Dict[str, str]:
        """Map each process variable name to its value type name."""
        return {
            name: type(pv.values).__name__
            for name, pv in process.process_variables.items()
        }

    def _vc_names(process: BioProcess) -> set:
        """Return the set of volume change names for a process."""
        if not process.volume or not process.volume.volume_changes:
            return set()
        return set(process.volume.volume_changes.keys())

    ref_reactor = _reactor_signature(first_process)
    ref_pv = _pv_signature(first_process)
    ref_vc = _vc_names(first_process)

    for proc_name, process in case_study.processes.items():
        if proc_name == first_name:
            continue

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

        vc_names = _vc_names(process)
        if vc_names != ref_vc:
            consistency_errors.append(
                f"Process '{proc_name}' volume change names differ from "
                f"'{first_name}': expected {sorted(ref_vc)}, got {sorted(vc_names)}"
            )

    if consistency_errors:
        all_valid = False
        report["__consistency__"] = consistency_errors
    else:
        report["__consistency__"] = ["Cross-process structure is consistent — OK"]

    return all_valid, report