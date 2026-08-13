from typing import Optional

import numpy as np
import jax.numpy as jnp
from .dataclasses import (
    BioProcess,
    BioProcessCollection,
    Inflow,
    Outflow,
    ProcessOrdering,
    StaticVariable,
    _format_biological_ode_lines,
    _format_bounds,
)
from .splines import _has_spline_state
from .time_series import TimeSeries
from .validate import _is_dynamic_series


def _is_spline_only_series(value: object) -> bool:
    """Return True for spline-backed series without discrete samples."""
    breaks = getattr(value, "breaks", None)
    coeffs = getattr(value, "coeffs", None)
    times = getattr(value, "times", None)
    values = getattr(value, "values", None)
    return (
        breaks is not None and coeffs is not None and times is None and values is None
    )


def _get_process_name(process: BioProcess) -> str:
    """Return a safe display name even when process metadata is absent."""
    metadata = getattr(process, "metadata", None)
    name = getattr(metadata, "name", None)
    return name if name else "<unnamed process>"


def _get_process_type(process: BioProcess) -> str:
    """Return a safe display type even when process metadata is absent."""
    metadata = getattr(process, "metadata", None)
    process_type = getattr(metadata, "process_type", None)
    return process_type if process_type else "<unknown type>"


def _get_process_notes(process: BioProcess) -> str | None:
    """Return notes when present; otherwise ``None``."""
    metadata = getattr(process, "metadata", None)
    notes = getattr(metadata, "notes", None)
    return notes if notes else None


def print_process_structure(process: BioProcess, verbosity: int = 3) -> None:
    """
    Print a hierarchical view of the BioProcess object structure.

    Args:
        process: BioProcess object to inspect
        verbosity: Detail level (1=minimal, 2=medium, 3=full)

            - Level 3 (most verbose): full details including units, value
              ranges, spline info
            - Level 2 (mid verbose): variable names and data type/size,
              no units or value ranges
            - Level 1 (least verbose): just which variables are saved,
              no other details

    Example:
        >>> print_process_structure(process, verbosity=1)
        >>> print_process_structure(process, verbosity=2)
        >>> print_process_structure(process)
    """
    print("=" * 80)
    print("BioProcess Structure")
    print("=" * 80)
    process_name = _get_process_name(process)
    process_type = _get_process_type(process)
    process_notes = _get_process_notes(process)

    if verbosity == 1:
        # Level 1: just list variable names
        print(f"Process: {process_name} ({process_type})")
        if process.reactor_medium and process.reactor_medium.components:
            print(
                "Reactor Medium Components: "
                f"{list(process.reactor_medium.components.keys())}"
            )
        if process.process_variables:
            print(f"Process Variables: {list(process.process_variables.keys())}")
        if process.volume and process.volume.volume_changes:
            print(f"Volume Changes: {list(process.volume.volume_changes.keys())}")

    elif verbosity == 2:
        # Level 2: names, controlled status, data type/size – no units or value ranges
        print(f"Process Name: {process_name}")
        print(f"Process Type: {process_type}")
        if process_notes:
            print(f"Notes: {process_notes}")

        if process.time_axis is not None:
            print(
                f"\nTime: {process.time_axis.start:.2f} to {process.time_axis.end:.2f}"
            )

        if process.reactor_medium and process.reactor_medium.components:
            print(f"\nReactor Medium: {process.reactor_medium.name}")
            print(f"  Components: ({len(process.reactor_medium.components)} total)")
            for comp in process.reactor_medium.components.values():
                if _is_dynamic_series(comp.concentration):
                    n = len(comp.concentration.times)
                    print(f"    - {comp.name}: TimeSeries ({n} points)")
                elif _is_spline_only_series(comp.concentration):
                    n = len(comp.concentration.breaks)
                    print(f"    - {comp.name}: Spline-only ({n} breakpoints)")
                else:
                    print(f"    - {comp.name}: Static")

        if process.process_variables:
            print(f"\nProcess Variables: ({len(process.process_variables)} total)")
            for pv in process.process_variables.values():
                if _is_dynamic_series(pv.values):
                    n = len(pv.values.times)
                    print(
                        f"  - {pv.name}: TimeSeries ({n} points),"
                        f" controlled={pv.is_controlled}"
                    )
                elif _is_spline_only_series(pv.values):
                    n = len(pv.values.breaks)
                    print(
                        f"  - {pv.name}: Spline-only ({n} breakpoints),"
                        f" controlled={pv.is_controlled}"
                    )
                else:
                    print(
                        f"  - {pv.name}: Static ({pv.values.value}),"
                        f" controlled={pv.is_controlled}"
                    )

        if process.volume is not None:
            print(f"\nVolume: {process.volume.initial_volume}")
            if process.volume.volume_changes:
                print(f"  Volume Changes: ({len(process.volume.volume_changes)} total)")
                for vc in process.volume.volume_changes.values():
                    if vc.values is None:
                        n = 0
                    elif _is_dynamic_series(vc.values):
                        n = len(vc.values.times)
                    elif _is_spline_only_series(vc.values):
                        n = len(vc.values.breaks)
                    else:
                        n = 0
                    print(
                        f"    - {vc.name}:"
                        f" {'Continuous' if vc.is_continuous else 'Discrete'}"
                        f" ({n} points)"
                    )

    else:
        # Level 3 (default): full details
        print(f"Process Name: {process_name}")
        print(f"Process Type: {process_type}")
        if process_notes:
            print(f"Notes: {process_notes}")

        if process.time_axis is not None:
            print("\nTime:")
            print(
                f"  Range: {process.time_axis.start:.2f} to"
                f" {process.time_axis.end:.2f} {process.time_axis.unit}"
            )
            print(f"  Reference: {process.time_axis.time_reference}")

        if process.reactor_medium:
            print("\nReactor Medium:")
            print(f"  Name: {process.reactor_medium.name}")
            print(
                f"  Density: {process.reactor_medium.density}"
                f" {process.reactor_medium.density_unit}"
            )
            if process.reactor_medium.components:
                print(f"  Components: ({len(process.reactor_medium.components)} total)")
                for comp in process.reactor_medium.components.values():
                    _print_reactor_component_info(comp, "    ")

        if process.process_variables:
            print(f"\nProcess Variables: ({len(process.process_variables)} total)")
            for pv in process.process_variables.values():
                _print_process_variable_info(pv, "  ")

        if process.volume is not None:
            print("\nVolume:")
            print(f"  Initial: {process.volume.initial_volume} {process.volume.unit}")
            print(f"  Bounds: {_format_bounds(process.volume.bounds)}")
            if process.volume.volume_changes:
                print(f"  Volume Changes: ({len(process.volume.volume_changes)} total)")
                for change in process.volume.volume_changes.values():
                    _print_volume_change_info(change, "    ")

        if process.biological_ode is not None:
            print("\nBiological ODE:")
            for line in _format_biological_ode_lines(process.biological_ode, prefix="  "):
                print(line)

    print("=" * 80)


def _print_process_variable_info(pv, prefix: str) -> None:
    """Helper function to print ProcessVariable information (verbosity=3)."""
    print(f"{prefix}{pv.name}")
    print(f"{prefix}  Unit: {pv.unit}")
    print(f"{prefix}  Controlled: {pv.is_controlled}")
    print(f"{prefix}  Bounds: {_format_bounds(pv.bounds)}")

    if _is_dynamic_series(pv.values):  # TimeSeries
        ts = pv.values
        n_points = len(ts.times)
        print(f"{prefix}  TimeSeries Data: {n_points} points")
        if n_points > 0:
            t_range = (float(ts.times[0]), float(ts.times[-1]))
            v_range = (float(jnp.min(ts.values)), float(jnp.max(ts.values)))
            print(f"{prefix}    Time range: {t_range[0]:.2f} to {t_range[1]:.2f}")
            print(f"{prefix}    Value range: {v_range[0]:.4f} to {v_range[1]:.4f}")
    elif _is_spline_only_series(pv.values):
        ts = pv.values
        n_breaks = len(ts.breaks)
        print(f"{prefix}  Spline-only Data: {n_breaks} breakpoints")
    elif hasattr(pv.values, "value"):  # StaticVariable
        print(f"{prefix}  Static Value: {pv.values.value}")

    if _has_spline_state(pv.values):
        print(f"{prefix}  Spline: available")


def _print_reactor_component_info(comp, prefix: str) -> None:
    """Helper function to print ReactorMediumComponent information (verbosity=3)."""
    print(f"{prefix}{comp.name}")
    print(f"{prefix}  Unit: {comp.unit}")
    print(f"{prefix}  Bounds: {_format_bounds(comp.bounds)}")

    if _is_dynamic_series(comp.concentration):  # TimeSeries
        ts = comp.concentration
        n_points = len(ts.times)
        print(f"{prefix}  TimeSeries Data: {n_points} points")
        if n_points > 0:
            t_range = (float(ts.times[0]), float(ts.times[-1]))
            v_range = (float(jnp.min(ts.values)), float(jnp.max(ts.values)))
            print(f"{prefix}    Time range: {t_range[0]:.2f} to {t_range[1]:.2f}")
            print(f"{prefix}    Value range: {v_range[0]:.4f} to {v_range[1]:.4f}")
    elif _is_spline_only_series(comp.concentration):
        ts = comp.concentration
        n_breaks = len(ts.breaks)
        print(f"{prefix}  Spline-only Data: {n_breaks} breakpoints")
    elif hasattr(comp.concentration, "value"):  # StaticVariable
        print(f"{prefix}  Static Concentration: {comp.concentration.value}")


def _print_volume_change_info(change, prefix: str) -> None:
    """Helper function to print VolumeChange information (verbosity=3)."""
    print(f"{prefix}{change.name}:")
    print(
        f"{prefix}  Type: {'Controlled' if change.is_controlled else 'Modeled'}, "
        f"{'Continuous' if change.is_continuous else 'Discrete'}"
    )
    print(f"{prefix}  Unit: {change.unit}")

    if isinstance(change, Inflow) and change.feed_medium:
        print(f"{prefix}  Feed Medium: {change.feed_medium.name}")

    if isinstance(change, Outflow) and change.retention:
        print(f"{prefix}  Retention: {change.retention}")

    if change.values is not None:
        if _is_dynamic_series(change.values):
            n_points = len(change.values.times)
            print(f"{prefix}  TimeSeries Points: {n_points}")
            domain_start = float(change.values.times[0]) if n_points > 0 else None
            domain_end = float(change.values.times[-1]) if n_points > 0 else None
        elif _is_spline_only_series(change.values):
            n_points = len(change.values.breaks)
            print(f"{prefix}  Spline-only Points: {n_points}")
            domain_start = float(change.values.breaks[0]) if n_points > 0 else None
            domain_end = float(change.values.breaks[-1]) if n_points > 0 else None
        else:
            n_points = 0
            domain_start = None
            domain_end = None

        if _is_dynamic_series(change.values) and n_points > 0:
            v_range = (
                float(jnp.min(change.values.values)),
                float(jnp.max(change.values.values)),
            )
            print(
                f"{prefix}    Value range: {v_range[0]:.4f} to"
                f" {v_range[1]:.4f} {change.unit}"
            )
            if change.is_continuous:
                total_change = float(change.values.values[-1] - change.values.values[0])
                print(f"{prefix}    Total change: {total_change:.2f} {change.unit}")
            else:
                total_change = float(jnp.sum(change.values.values))
                print(f"{prefix}    Total change: {total_change:.2f} {change.unit}")

        if (
            change.is_continuous
            and domain_start is not None
            and domain_end is not None
            and _has_spline_state(change.values)
            and hasattr(change.values, "integrate")
        ):
            integral = float(change.values.integrate(domain_start, domain_end))
            print(
                f"{prefix}    Series integral over span:"
                f" {integral:.4f} {change.unit}*time"
            )


def _count_datapoints_in_value(value) -> int:
    """
    Count datapoints in a value which may be a TimeSeries or StaticVariable.
    TimeSeries -> number of times
    StaticVariable -> count as 1
    """
    if _is_dynamic_series(value):
        return int(len(value.times))
    elif _is_spline_only_series(value):
        return int(len(value.breaks))
    elif hasattr(value, "value"):
        return 1
    return 0


def _count_datapoints_in_process(process: BioProcess) -> int:
    """
    Count all datapoints (time series lengths + static entries) contained
    in a BioProcess. This includes:
      - process_variables (TimeSeries / StaticVariable)
      - reactor_medium component concentrations (TimeSeries / StaticVariable)
      - volume changes (their TimeSeries)
      - feed medium components referenced in volume changes
        (TimeSeries / StaticVariable)
    """
    total = 0

    # Process variables
    for pv in process.process_variables.values():
        total += _count_datapoints_in_value(pv.values)

    # Reactor medium components
    if getattr(process, "reactor_medium", None) is not None:
        for comp in process.reactor_medium.components.values():
            total += _count_datapoints_in_value(comp.concentration)

    # Volume changes and their feed media components
    if getattr(process, "volume", None) is not None and process.volume.volume_changes:
        for vc in process.volume.volume_changes.values():
            if getattr(vc, "values", None) is not None:
                total += _count_datapoints_in_value(vc.values)
            # feed medium components
            if getattr(vc, "feed_medium", None) is not None and isinstance(
                vc, Inflow
            ):
                for fcomp in vc.feed_medium.components.values():
                    total += _count_datapoints_in_value(fcomp.concentration)

    return total


def print_collection_structure(collection: BioProcessCollection, verbosity: int = 3) -> None:
    """
    Print a hierarchical view of a BioProcessCollection.

    Args:
        collection: BioProcessCollection object to inspect
        verbosity: Detail level (1=minimal, 2=medium, 3=full)

            - Level 3 (most verbose): organism, citation (when set),
              per-process datapoint counts, and total datapoints
            - Level 2 (mid verbose): organism and process names listed
              (no citation, no datapoint counts)
            - Level 1 (least verbose): case_id/label and process count only
    """
    print("=" * 80)
    print("BioProcessCollection Structure")
    print("=" * 80)

    n_procs = len(collection.processes) if collection.processes else 0
    label = collection.case_id or "(untitled collection)"

    if verbosity == 1:
        print(f"{label}  ({n_procs} processes)")
        print("=" * 80)
        return

    if collection.case_id:
        print(f"Case ID:  {collection.case_id}")
        print(f"Organism: {collection.organism}")
        if verbosity == 3:
            print(f"Citation: {collection.citation}")
    else:
        print(f"Collection: {label}")
        if collection.metadata:
            print(f"Metadata keys: {sorted(collection.metadata.keys())}")
    print(f"Processes: {n_procs}")

    if not collection.processes:
        print("  (no processes)")
        print("=" * 80)
        return

    total_datapoints = 0
    for p_key, proc in collection.processes.items():
        name = _get_process_name(proc)
        if verbosity == 2:
            print(f"  * {p_key}: {name}")
        else:
            proc_dp = _count_datapoints_in_process(proc)
            total_datapoints += proc_dp
            print(f"  * {p_key}: {name}  (datapoints: {proc_dp})")

    if verbosity == 3:
        print("\n" + "-" * 80)
        print(f"Total datapoints in collection: {total_datapoints}")

    print("=" * 80)


# ---------------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------------


def _collect_process_panels(process: BioProcess):
    """
    Collect all plottable panels from a BioProcess.

    Returns a list of dicts, each with:
      - 'title': str
      - 'category': 'ReactorMedium' | 'ProcessVariable' | 'VolumeChange'
      - 'type': 'dynamic' | 'static'
      - for dynamic: 'x' (times array), 'y' (values array)
      - for static:  't_start' (float), 't_end' (float), 'value' (float)
      - optional: 'render': 'line' | 'bar'
      - optional: 'series': TimeSeries spline carrier (if available)
      - optional: 'series_type': 'direct'
    """
    t_start = float(process.time_axis.start) if process.time_axis else 0.0
    t_end = float(process.time_axis.end) if process.time_axis else 1.0

    panels = []

    # Reactor medium components
    if process.reactor_medium and process.reactor_medium.components:
        for comp in process.reactor_medium.components.values():
            unit_label = f" [{comp.unit}]" if comp.unit else ""
            if _is_dynamic_series(comp.concentration):
                panel = {
                    "title": f"{comp.name}{unit_label}",
                    "category": "ReactorMedium",
                    "type": "dynamic",
                    "x": comp.concentration.times,
                    "y": comp.concentration.values,
                    "render": "line",
                }
                if _has_spline_state(comp.concentration):
                    panel["series"] = comp.concentration
                    panel["series_type"] = "direct"
                panels.append(panel)
            elif _is_spline_only_series(comp.concentration):
                x = jnp.asarray(comp.concentration.breaks)
                y = comp.concentration.evaluate_many(x)
                panel = {
                    "title": f"{comp.name}{unit_label}",
                    "category": "ReactorMedium",
                    "type": "dynamic",
                    "x": x,
                    "y": y,
                    "render": "line",
                    "series": comp.concentration,
                    "series_type": "direct",
                }
                panels.append(panel)
            elif hasattr(comp.concentration, "value"):
                panels.append(
                    {
                        "title": f"{comp.name}{unit_label}",
                        "category": "ReactorMedium",
                        "type": "static",
                        "t_start": t_start,
                        "t_end": t_end,
                        "value": float(comp.concentration.value),
                    }
                )

    # Process variables
    if process.process_variables:
        for pv in process.process_variables.values():
            unit_label = f" [{pv.unit}]" if pv.unit else ""
            if _is_dynamic_series(pv.values):
                panel = {
                    "title": f"{pv.name}{unit_label}",
                    "category": "ProcessVariable",
                    "type": "dynamic",
                    "x": pv.values.times,
                    "y": pv.values.values,
                    "render": "line",
                }
                if _has_spline_state(pv.values):
                    panel["series"] = pv.values
                    panel["series_type"] = "direct"
                panels.append(panel)
            elif _is_spline_only_series(pv.values):
                x = jnp.asarray(pv.values.breaks)
                y = pv.values.evaluate_many(x)
                panels.append(
                    {
                        "title": f"{pv.name}{unit_label}",
                        "category": "ProcessVariable",
                        "type": "dynamic",
                        "x": x,
                        "y": y,
                        "render": "line",
                        "series": pv.values,
                        "series_type": "direct",
                    }
                )
            elif hasattr(pv.values, "value"):
                panels.append(
                    {
                        "title": f"{pv.name}{unit_label}",
                        "category": "ProcessVariable",
                        "type": "static",
                        "t_start": t_start,
                        "t_end": t_end,
                        "value": float(pv.values.value),
                    }
                )

    # Volume changes
    if process.volume and process.volume.volume_changes:
        for vc in process.volume.volume_changes.values():
            unit_label = f" [{vc.unit}]" if vc.unit else ""
            if vc.values is not None and _is_dynamic_series(vc.values):
                is_continuous = getattr(vc, "is_continuous", True)
                render = "line" if is_continuous else "bar"
                panel = {
                    "title": f"{vc.name}{unit_label}",
                    "category": "VolumeChange",
                    "type": "dynamic",
                    "x": vc.values.times,
                    "y": vc.values.values,
                    "render": render,
                }
                if _has_spline_state(vc.values):
                    panel["series"] = vc.values
                    panel["series_type"] = "direct"
                panels.append(panel)
            elif vc.values is not None and _is_spline_only_series(vc.values):
                x = jnp.asarray(vc.values.breaks)
                y = vc.values.evaluate_many(x)
                panels.append(
                    {
                        "title": f"{vc.name}{unit_label}",
                        "category": "VolumeChange",
                        "type": "dynamic",
                        "x": x,
                        "y": y,
                        "render": "line",
                        "series": vc.values,
                        "series_type": "direct",
                    }
                )

    total_volume_panel = _build_total_volume_panel(
        process, t_start=t_start, t_end=t_end
    )
    if total_volume_panel is not None:
        panels.append(total_volume_panel)

    return panels


def _build_total_volume_panel(process: BioProcess, t_start: float, t_end: float):
    """Construct total-volume trajectory panel from stored trace or changes."""
    volume = getattr(process, "volume", None)
    if volume is None:
        return None

    stored_total = getattr(volume, "total_volume", None)
    unit_label = f" [{volume.unit}]" if volume.unit else ""
    if _is_dynamic_series(stored_total):
        panel = {
            "title": f"total volume{unit_label}",
            "category": "Volume",
            "type": "dynamic",
            "x": stored_total.times,
            "y": stored_total.values,
            "render": "line",
        }
        if _has_spline_state(stored_total):
            panel["series"] = stored_total
            panel["series_type"] = "direct"
        return panel
    if _is_spline_only_series(stored_total):
        x = jnp.asarray(stored_total.breaks)
        y = stored_total.evaluate_many(x)
        return {
            "title": f"total volume{unit_label}",
            "category": "Volume",
            "type": "dynamic",
            "x": x,
            "y": y,
            "render": "line",
            "series": stored_total,
            "series_type": "direct",
        }

    time_grid = [t_start, t_end]
    continuous_changes = []
    discrete_events = []

    volume_changes = getattr(volume, "volume_changes", None) or {}
    for vc in volume_changes.values():
        if vc.values is None:
            continue

        if _is_dynamic_series(vc.values):
            x = np.asarray(vc.values.times, dtype=float)
            y = np.asarray(vc.values.values, dtype=float)
        elif _is_spline_only_series(vc.values):
            x = np.asarray(vc.values.breaks, dtype=float)
            y = np.asarray(vc.values.evaluate_many(vc.values.breaks), dtype=float)
        else:
            continue

        mask = np.isfinite(x) & np.isfinite(y)
        x = x[mask]
        y = y[mask]
        if x.size == 0:
            continue

        order = np.argsort(x)
        x = x[order]
        y = y[order]
        time_grid.extend(x.tolist())

        if getattr(vc, "is_continuous", True):
            continuous_changes.append((x, y))
        else:
            discrete_events.extend(zip(x.tolist(), y.tolist()))

    t_plot = np.unique(np.asarray(time_grid, dtype=float))
    total_volume = np.full_like(t_plot, float(volume.initial_volume), dtype=float)

    for x, y in continuous_changes:
        y0 = float(y[0])
        y_interp = np.interp(t_plot, x, y, left=y0, right=float(y[-1]))
        total_volume += y_interp - y0

    for event_time, delta_v in discrete_events:
        total_volume += np.where(t_plot >= event_time, float(delta_v), 0.0)

    return {
        "title": f"total volume{unit_label}",
        "category": "Volume",
        "type": "dynamic",
        "x": t_plot,
        "y": total_volume,
        "render": "line",
    }


def _pad_constant_ylim(ax, values):
    """If *values* are practically constant, pad the y-axis to avoid noisy scaling."""
    y = np.asarray(values, dtype=float)
    y = y[np.isfinite(y)]
    if len(y) == 0:
        return
    ymin, ymax = float(y.min()), float(y.max())
    span = ymax - ymin
    mean = (ymin + ymax) / 2.0
    # Consider "practically constant" when range < 1e-6 * |mean| (or absolute < 1e-12)
    if span < max(abs(mean) * 1e-6, 1e-12):
        pad = max(abs(mean) * 0.1, 1.0) if abs(mean) > 1e-12 else 1.0
        new_lo, new_hi = mean - pad, mean + pad
        cur_lo, cur_hi = ax.get_ylim()
        # Only expand, never shrink
        ax.set_ylim(min(cur_lo, new_lo), max(cur_hi, new_hi))


def _evaluate_series_curve(series, t_start, t_end, n_points=500):
    """Evaluate a spline-backed TimeSeries over [t_start, t_end]."""
    t_plot = np.linspace(t_start, t_end, n_points)
    y_plot = np.asarray(series.evaluate_many(jnp.asarray(t_plot, dtype=float)))
    return t_plot, y_plot


def _draw_panel(
    ax, panel, label=None, color=None, t_start=None, t_end=None, bar_transparent=False
):
    """Draw a single panel (dynamic or static) onto *ax*.

    If the panel has a ``'series'`` key, the spline curve is drawn and raw
    data is shown as scatter points (no connecting lines). Otherwise raw
    data is drawn with ``'o-'`` markers.
    """
    plot_kwargs = {}
    if color is not None:
        plot_kwargs["color"] = color
    else:
        plot_kwargs["color"] = "black"

    if label is None:
        label = "data"

    has_series = "series" in panel and panel["series"] is not None

    if panel["type"] == "dynamic":
        x = panel["x"]
        y = panel["y"]
        render = panel.get("render", "line")

        n = len(x)

        if n == 0:
            # Nothing to plot (e.g. empty volume change); add invisible
            # artist for legend
            ax.plot([], [], label=label, **plot_kwargs)
        elif render == "bar":
            # Bar plot for discrete (non-continuous) volume changes
            if t_start is not None and t_end is not None:
                plot_span = float(t_end) - float(t_start)
            else:
                plot_span = float(x[-1] - x[0])
            width = plot_span / 100
            bar_color = plot_kwargs["color"]
            if bar_transparent:
                ax.bar(
                    x,
                    y,
                    label=label,
                    width=width,
                    edgecolor=bar_color,
                    facecolor="none",
                )
            else:
                ax.bar(
                    x,
                    y,
                    label=label,
                    width=width,
                    edgecolor="k",
                    color=bar_color,
                )
        elif has_series and n <= 50:
            # Scatter raw observations when a spline-backed series is available.
            ax.scatter(x, y, s=16, zorder=5, label=label, **plot_kwargs)
        else:
            fmt = "o-" if n <= 50 else "-"
            ax.plot(x, y, fmt, markersize=4, label=label, **plot_kwargs)

        # Draw spline curve
        if has_series and t_start is not None and t_end is not None:
            try:
                t_plot, y_plot = _evaluate_series_curve(
                    panel["series"],
                    t_start,
                    t_end,
                )
                ax.plot(
                    t_plot,
                    y_plot,
                    "--",
                    color="red",
                    lw=1.5,
                    alpha=0.8,
                    label="spline",
                )
            except Exception as e:
                import warnings

                warnings.warn(
                    f"Could not evaluate spline for {panel.get('title', '?')}: {e}"
                )
    else:
        ax.hlines(
            panel["value"],
            panel["t_start"],
            panel["t_end"],
            linestyles="--",
            label=label,
            **plot_kwargs,
        )


def plot_collection(collection: BioProcessCollection, figsize_per_panel=(5, 3), save_path=None):
    """
    Plot all dynamic and static variables for every process in a BioProcessCollection.

    All unique variables are discovered across every process first.  Each
    variable gets its own subplot and all processes are overlaid using
    distinct colours.  TimeSeries with ≤ 30 points are drawn with markers;
    longer series are lines only.

    Args:
        collection: BioProcessCollection object to plot.
        figsize_per_panel: ``(width, height)`` in inches for each subplot.

    Returns:
        matplotlib.figure.Figure
    """
    import matplotlib.pyplot as plt

    variable_map = {}

    # Determine global x-axis range across all processes
    t_global_start = float("inf")
    t_global_end = float("-inf")
    time_unit = "time"

    for proc_key, process in collection.processes.items():
        if process.time_axis is not None:
            time_unit = process.time_axis.unit
            t_global_start = min(t_global_start, float(process.time_axis.start))
            t_global_end = max(t_global_end, float(process.time_axis.end))

        for panel in _collect_process_panels(process):
            key = panel["title"]
            if key not in variable_map:
                variable_map[key] = {
                    "title": panel["title"],
                    "time_unit": time_unit,
                    "data": [],
                }
            entry = dict(panel)
            entry["label"] = proc_key
            variable_map[key]["data"].append(entry)

    # some white space to the side
    delta_t_global = t_global_end - t_global_start
    t_global_start -= delta_t_global * 0.05
    t_global_end += delta_t_global * 0.05

    if not variable_map:
        fig, ax = plt.subplots()
        ax.text(
            0.5,
            0.5,
            "No variables to plot",
            ha="center",
            va="center",
            transform=ax.transAxes,
        )
        return fig

    panels = list(variable_map.values())
    fig, axes_flat = _make_figure(len(panels), figsize_per_panel)
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]

    # Stable label->color mapping so the same process always gets the same color
    # across all subplots, regardless of which variables each process has.
    seen_labels: dict[str, int] = {}
    for panel_meta in panels:
        for data in panel_meta["data"]:
            lab = data["label"]
            seen_labels.setdefault(lab, len(seen_labels))
    label_colors = {lab: colors[i % len(colors)] for lab, i in seen_labels.items()}

    # Collect legend entries once (process key -> (handle, label))
    legend_handles_by_label = {}

    for i, panel_meta in enumerate(panels):
        ax = axes_flat[i]
        for data in panel_meta["data"]:
            color = label_colors[data["label"]]
            _draw_panel(
                ax,
                data,
                label=data["label"],
                color=color,
                t_start=t_global_start,
                t_end=t_global_end,
            )

        category = (
            panel_meta["data"][0].get("category", "") if panel_meta["data"] else ""
        )
        ax.set_title(f"{panel_meta['title']} ({category})")
        ax.set_xlabel(f"time [{panel_meta['time_unit']}]")
        ax.set_xlim(t_global_start, t_global_end)
        ax.grid(True, alpha=0.3)

        # Pad y-axis if all overlaid data for this panel is practically constant
        # Skip bar-rendered panels (discrete events) — let matplotlib auto-scale
        has_bar = any(d.get("render") == "bar" for d in panel_meta["data"])
        if not has_bar:
            all_y = []
            for data in panel_meta["data"]:
                if data["type"] == "dynamic":
                    all_y.extend(np.asarray(data["y"], dtype=float).tolist())
                else:
                    all_y.append(data["value"])
            if all_y:
                _pad_constant_ylim(ax, all_y)

        # harvest handles/labels from this axis (without drawing a per-axis legend)
        handles, labels = ax.get_legend_handles_labels()
        for h, lab in zip(handles, labels):
            if lab and lab not in legend_handles_by_label:
                legend_handles_by_label[lab] = h

    for j in range(len(panels), len(axes_flat)):
        axes_flat[j].set_visible(False)

    label = collection.case_id or (collection.metadata or {}).get("name", "BioProcessCollection")
    fig.suptitle(f"BioProcessCollection: {label}", fontsize=12)

    if legend_handles_by_label:
        labels = list(legend_handles_by_label.keys())
        handles = [legend_handles_by_label[lab] for lab in labels]
        # Put one shared legend at bottom
        fig.legend(
            handles,
            labels,
            loc="lower center",
            ncol=min(len(labels), 5),
            fontsize="small",
            frameon=True,
            fancybox=False,
            edgecolor="k",
            bbox_to_anchor=(0.5, 0.0),
        )
        fig.tight_layout(rect=(0, 0.06, 1, 1))
    else:
        fig.tight_layout()

    if save_path is not None:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
    # fig.show()
    return fig


def _make_figure(n_panels, figsize_per_panel):
    """Create a two-column figure with the correct number of rows."""
    import matplotlib.pyplot as plt

    n_cols = 2
    n_rows = max(1, (n_panels + 1) // 2)
    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        squeeze=False,
        figsize=(figsize_per_panel[0] * n_cols, figsize_per_panel[1] * n_rows),
    )
    return fig, axes.flatten()


def plot_process(process: BioProcess, figsize_per_panel=(5, 3), save_path=None):
    """
    Plot all dynamic and static variables of a BioProcess in a two-column figure.

    Each variable gets its own subplot.  TimeSeries with ≤ 30 points are drawn
    with markers; longer series are drawn as lines only.  StaticVariable values
    are shown as horizontal dashed lines spanning the process time range.

    All subplots share the same x-axis range (``TimeAxis.start`` to
    ``TimeAxis.end``).

    Args:
        process: BioProcess object to plot.
        figsize_per_panel: ``(width, height)`` in inches for each subplot.

    Returns:
        matplotlib.figure.Figure
    """
    import matplotlib.pyplot as plt

    time_unit = process.time_axis.unit if process.time_axis else "time"
    t_start = float(process.time_axis.start) if process.time_axis else 0.0
    t_end = float(process.time_axis.end) if process.time_axis else 1.0
    panels = _collect_process_panels(process)

    if not panels:
        fig, ax = plt.subplots()
        ax.text(
            0.5,
            0.5,
            "No variables to plot",
            ha="center",
            va="center",
            transform=ax.transAxes,
        )
        return fig

    fig, axes_flat = _make_figure(len(panels), figsize_per_panel)

    for i, panel in enumerate(panels):
        ax = axes_flat[i]
        _draw_panel(ax, panel, t_start=t_start, t_end=t_end, bar_transparent=True)
        # Pad y-axis for constant-valued panels (skip bar-rendered discrete events)
        if panel.get("render") != "bar":
            if panel["type"] == "dynamic":
                _pad_constant_ylim(ax, panel["y"])
            else:
                _pad_constant_ylim(ax, [panel["value"]])
        category = panel.get("category", "")
        ax.set_title(f"{panel['title']} ({category})")
        ax.set_xlabel(f"time [{time_unit}]")
        ax.set_xlim(t_start, t_end)
        ax.legend(fontsize="small")
        ax.grid(True, alpha=0.3)

    for j in range(len(panels), len(axes_flat)):
        axes_flat[j].set_visible(False)

    fig.suptitle(
        f"{_get_process_name(process)} ({_get_process_type(process)})", fontsize=12
    )
    fig.tight_layout()
    if save_path is not None:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
    # fig.show()
    return fig


def plot_timeseries(ts: TimeSeries, figsize=(6, 4), save_path=None):
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=figsize)
    ax.plot(ts.times, ts.values, "-")

    if save_path is not None:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
    return fig


# ---------------------------------------------------------------------------
# RhsOde structure printer
# ---------------------------------------------------------------------------


def _cin_static_value(vc, rmc_name: str) -> float:
    """Static feed concentration of *rmc_name* in *vc*'s feed medium (0 if absent)."""
    if not isinstance(vc, Inflow) or vc.feed_medium is None:
        return 0.0
    comp = vc.feed_medium.components.get(rmc_name)
    if comp is None:
        return 0.0
    if isinstance(comp.concentration, StaticVariable):
        return float(comp.concentration.value)
    return 0.0  # TimeSeries Cin not supported (mirrors mechanistic._build_cin)


def _measure_subtable_width(headers: list, rows: list) -> int:
    """Width required by a sub-table with these headers and rows."""
    n = len(headers)
    col_w = [
        max(len(str(headers[i])), 3, *(len(str(r[i])) for r in rows)) for i in range(n)
    ]
    return sum(col_w) + 3 * n + 1


def _format_subtable_lines(
    headers: list, rows: list, aligns: list, target_width: int
) -> list:
    """Render header row + data rows, padding the last column so each row
    reaches *target_width*. Returns lines with ``| ... |`` borders included."""
    n = len(headers)
    col_w = [
        max(len(str(headers[i])), 3, *(len(str(r[i])) for r in rows)) for i in range(n)
    ]
    req = sum(col_w) + 3 * n + 1
    if target_width > req:
        col_w[-1] += target_width - req

    def _pad(s: str, w: int, align: str) -> str:
        return s.rjust(w) if align == "r" else s.ljust(w)

    def _row(cells) -> str:
        padded = [_pad(str(c), col_w[i], aligns[i]) for i, c in enumerate(cells)]
        return "| " + " | ".join(padded) + " |"

    return [_row(headers)] + [_row(r) for r in rows]


def _render_combined_box(title: str, sections: list) -> str:
    """Render a single ASCII box containing several sub-tables.

    *sections* is a list of ``(banner, headers, rows, aligns)`` tuples.
    """
    sub_widths = [_measure_subtable_width(s[1], s[2]) for s in sections]
    banner_widths = [4 + len(s[0]) for s in sections]
    title_str = f" {title} "
    title_width = len(title_str) + 2
    total_width = max([*sub_widths, *banner_widths, title_width])

    divider = "+" + "-" * (total_width - 2) + "+"
    pad = total_width - 2 - len(title_str)
    left = pad // 2
    right = pad - left
    title_line = "+" + "-" * left + title_str + "-" * right + "+"

    lines = [title_line]
    for banner, headers, rows, aligns in sections:
        banner_pad = total_width - 4 - len(banner)
        lines.append(f"| {banner}{' ' * banner_pad} |")
        lines.append(divider)
        lines.extend(_format_subtable_lines(headers, rows, aligns, total_width))
        lines.append(divider)
    return "\n".join(lines)


def _format_rmc_feed(
    rmc_name: str,
    inflow_names: list,
    process: BioProcess,
) -> str:
    """``+ feed(<Inflows supplying Cin for this RMC>)`` or empty when none."""
    feeders = [
        n
        for n in inflow_names
        if _cin_static_value(process.volume.volume_changes[n], rmc_name) != 0.0
    ]
    return "+ feed(" + ", ".join(feeders) + ")" if feeders else ""


def _format_rmc_dilution(inflow_names: list, outflow_names: list) -> str:
    """``− dilution(<all Inflow+Outflow>)`` or empty when there are no flows."""
    flows = list(inflow_names) + list(outflow_names)
    return "− dilution(" + ", ".join(flows) + ")" if flows else ""


def _discrete_volume_changes(process: BioProcess):
    """Return ``(discrete_Inflows, discrete_Outflows)`` — names of discrete (bolus
    / discrete-sample) volume changes. ``ProcessOrdering`` only enumerates
    continuous volume changes, so the discrete ones are recovered here.
    """
    disc_inflow = sorted(
        n
        for n, vc in process.volume.volume_changes.items()
        if not vc.is_continuous and isinstance(vc, Inflow)
    )
    disc_outflow = sorted(
        n
        for n, vc in process.volume.volume_changes.items()
        if not vc.is_continuous and isinstance(vc, Outflow)
    )
    return disc_inflow, disc_outflow


def _format_v_additions(
    cont_inflow: list,
    disc_inflow: list,
) -> str:
    """V's positive contributions: continuous Inflow flow rates and discrete
    bolus events. Returns ``0`` when neither is present."""
    parts: list = []
    if cont_inflow:
        parts.append(" + ".join(cont_inflow))
    if disc_inflow:
        parts.append("bolus(" + ", ".join(disc_inflow) + ")")
    return " + ".join(parts) if parts else "0"


def _format_v_removals(
    cont_outflow: list,
    disc_outflow: list,
) -> str:
    """V's negative contributions: continuous Outflow flow rates and discrete
    sampling events. Returns ``0`` when neither is present."""
    parts: list = []
    if cont_outflow:
        parts.append("− |" + " + ".join(cont_outflow) + "|")
    if disc_outflow:
        parts.append("− sample(" + ", ".join(disc_outflow) + ")")
    return " ".join(parts) if parts else "0"


def _resolve_target(target):
    """Return ``(process, title_label)`` from BioProcess / BioProcessCollection.

    For multi-process containers, equivalence of ``biological_ode`` is
    enforced before the first process is selected. The returned label is
    the collection's case-study id (or ``metadata["name"]`` / fallback),
    never a process name, so the printed title represents the whole
    container.
    """
    from .validate import validate_biological_ode_equivalence

    if isinstance(target, BioProcess):
        return target, _get_process_name(target)
    if isinstance(target, BioProcessCollection):
        if not target.processes:
            raise ValueError("print_rhs_ode: container has no processes.")
        ok, msg = validate_biological_ode_equivalence(target)
        if not ok:
            raise ValueError(f"Cannot print unified ODE structure: {msg}")
        process = next(iter(target.processes.values()))
        n = len(target.processes)
        label = target.case_id or (target.metadata or {}).get("name", "BioProcessCollection")
        if n > 1:
            label = f"{label} ({n} processes)"
        return process, label
    raise TypeError(
        "print_rhs_ode expects BioProcess or BioProcessCollection; "
        f"got {type(target).__name__!r}"
    )


def print_rhs_ode(
    target,
    ordering: Optional[ProcessOrdering] = None,
) -> None:
    """Print the mechanistic ODE structure as a single ASCII box with two
    sub-tables (Algebraic, Derivatives).

    Accepts a :class:`BioProcess` or a
    :class:`BioProcessCollection`. For multi-process containers,
    :func:`bp_format.validate.validate_biological_ode_equivalence` is
    invoked first and the title represents the whole container — the
    individual process picked to render is not exposed.

    The Derivatives sub-table separates the *Biological* expression
    (verbatim from ``biological_ode.derivatives``) from the *Feed* and
    *Dilution* contributions that bp-format adds on top: ``+ feed(<Inflows
    with Cin>)`` and ``− dilution(<all Inflow+Outflow>)`` per RMC. The Volume
    sub-table lists V separately with *Additions* (Inflow sum) and
    *Removals* (``− |<Outflow sum>|``) columns.

    Raises:
        ValueError: if a multi-process container's processes do not share
            equivalent ``biological_ode`` blocks.
    """
    from .mechanistic import get_process_ordering

    process, title_label = _resolve_target(target)

    bo = process.biological_ode
    if bo is None:
        raise ValueError("print_rhs_ode requires process.biological_ode to be set.")
    if ordering is None:
        ordering = get_process_ordering(process)

    sections: list = []

    if ordering.name_modeled_algebraic:
        alg_rows = [[n, bo.algebraic[n]] for n in ordering.name_modeled_algebraic]
        sections.append(
            (
                "Algebraic",
                ["Name", "Expression"],
                alg_rows,
                ["l", "l"],
            )
        )

    if ordering.name_modeled_rates:

        def _fmt_bound(b):
            return "—" if b is None else f"{b:g}"

        rate_rows = []
        for n in ordering.name_modeled_rates:
            lo, hi = bo.rates.get(n, (None, None))
            rate_rows.append([n, _fmt_bound(lo), _fmt_bound(hi)])
        sections.append(
            (
                "Rates (declaration order — this is `name_modeled_rates`)",
                ["Name", "Lower", "Upper"],
                rate_rows,
                ["l", "r", "r"],
            )
        )

    inflow_all = list(ordering.name_controlled_Inflows) + list(ordering.name_modeled_Inflows)
    outflow_all = list(ordering.name_controlled_Outflows) + list(ordering.name_modeled_Outflows)

    deriv_rows: list = []
    for n in ordering.name_modeled_RMCs:
        unit = process.reactor_medium.components[n].unit
        bio_str = bo.derivatives.get(n, "0")
        feed = _format_rmc_feed(n, inflow_all, process)
        dilution = _format_rmc_dilution(inflow_all, outflow_all)
        deriv_rows.append([n, f"[{unit}]", bio_str, feed, dilution])
    for n in ordering.name_modeled_PVs:
        unit = process.process_variables[n].unit
        bio_str = bo.derivatives.get(n, "0")
        deriv_rows.append([n, f"[{unit}]", bio_str, "", ""])

    sections.append(
        (
            "Derivatives",
            ["State", "Unit", "Biological", "Feed", "Dilution"],
            deriv_rows,
            ["l", "l", "l", "l", "l"],
        )
    )

    if process.volume is not None:
        disc_inflow, disc_outflow = _discrete_volume_changes(process)
        vol_rows = [
            [
                "V",
                f"[{process.volume.unit}]",
                _format_v_additions(inflow_all, disc_inflow),
                _format_v_removals(outflow_all, disc_outflow),
            ]
        ]
        sections.append(
            (
                "Volume",
                ["State", "Unit", "Additions", "Removals"],
                vol_rows,
                ["l", "l", "l", "l"],
            )
        )

    print(_render_combined_box(f"RhsOde Structure: {title_label}", sections))
