import numpy as np
import jax.numpy as jnp
from .dataclasses import BioProcess, CaseStudy, BenchmarkDataset, FeedVolumeChange


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

            - Level 3 (most verbose): full details including units, value ranges, spline info
            - Level 2 (mid verbose): variable names and data type/size, no units or value ranges
            - Level 1 (least verbose): just which variables are saved, no other details

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
            print(f"Reactor Medium Components: {list(process.reactor_medium.components.keys())}")
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
            print(f"\nTime: {process.time_axis.start:.2f} to {process.time_axis.end:.2f}")

        if process.reactor_medium and process.reactor_medium.components:
            print(f"\nReactor Medium: {process.reactor_medium.name}")
            print(f"  Components: ({len(process.reactor_medium.components)} total)")
            for comp in process.reactor_medium.components.values():
                if hasattr(comp.concentration, 'timepoints'):
                    n = len(comp.concentration.timepoints)
                    print(f"    - {comp.name}: TimeSeries ({n} points)")
                else:
                    print(f"    - {comp.name}: Static")

        if process.process_variables:
            print(f"\nProcess Variables: ({len(process.process_variables)} total)")
            for pv in process.process_variables.values():
                if hasattr(pv.values, 'timepoints'):
                    n = len(pv.values.timepoints)
                    print(f"  - {pv.name}: TimeSeries ({n} points), controlled={pv.is_controlled}")
                else:
                    print(f"  - {pv.name}: Static ({pv.values.value}), controlled={pv.is_controlled}")

        if process.volume is not None:
            print(f"\nVolume: {process.volume.initial_volume}")
            if process.volume.volume_changes:
                print(f"  Volume Changes: ({len(process.volume.volume_changes)} total)")
                for vc in process.volume.volume_changes.values():
                    n = len(vc.values.timepoints) if vc.values is not None else 0
                    print(f"    - {vc.name}: {'Continuous' if vc.is_continuous else 'Discrete'} ({n} points)")

    else:
        # Level 3 (default): full details
        print(f"Process Name: {process_name}")
        print(f"Process Type: {process_type}")
        if process_notes:
            print(f"Notes: {process_notes}")

        if process.time_axis is not None:
            print(f"\nTime:")
            print(f"  Range: {process.time_axis.start:.2f} to {process.time_axis.end:.2f} {process.time_axis.unit}")
            print(f"  Reference: {process.time_axis.time_reference}")

        if process.reactor_medium:
            print(f"\nReactor Medium:")
            print(f"  Name: {process.reactor_medium.name}")
            print(f"  Density: {process.reactor_medium.density} {process.reactor_medium.density_unit}")
            if process.reactor_medium.components:
                print(f"  Components: ({len(process.reactor_medium.components)} total)")
                for comp in process.reactor_medium.components.values():
                    _print_reactor_component_info(comp, "    ")

        if process.process_variables:
            print(f"\nProcess Variables: ({len(process.process_variables)} total)")
            for pv in process.process_variables.values():
                _print_process_variable_info(pv, "  ")

        if process.volume is not None:
            print(f"\nVolume:")
            print(f"  Initial: {process.volume.initial_volume} {process.volume.unit}")
            if process.volume.volume_changes:
                print(f"  Volume Changes: ({len(process.volume.volume_changes)} total)")
                for change in process.volume.volume_changes.values():
                    _print_volume_change_info(change, "    ")

    print("=" * 80)

def _print_process_variable_info(pv, prefix: str) -> None:
    """Helper function to print ProcessVariable information (verbosity=3)."""
    print(f"{prefix}{pv.name}")
    print(f"{prefix}  Unit: {pv.unit}")
    print(f"{prefix}  Controlled: {pv.is_controlled}")

    if hasattr(pv.values, 'timepoints'):  # TimeSeries
        ts = pv.values
        n_points = len(ts.timepoints)
        print(f"{prefix}  TimeSeries Data: {n_points} points")
        if n_points > 0:
            t_range = (float(ts.timepoints[0]), float(ts.timepoints[-1]))
            v_range = (float(jnp.min(ts.values)), float(jnp.max(ts.values)))
            print(f"{prefix}    Time range: {t_range[0]:.2f} to {t_range[1]:.2f}")
            print(f"{prefix}    Value range: {v_range[0]:.4f} to {v_range[1]:.4f}")
    elif hasattr(pv.values, 'value'):  # StaticVariable
        print(f"{prefix}  Static Value: {pv.values.value}")

    if pv.spline is not None:
        print(f"{prefix}  Spline: available")


def _print_reactor_component_info(comp, prefix: str) -> None:
    """Helper function to print ReactorMediumComponent information (verbosity=3)."""
    print(f"{prefix}{comp.name}")
    print(f"{prefix}  Unit: {comp.unit}")
    print(f"{prefix}  Intracellular: {comp.is_intracellular}")

    if hasattr(comp.concentration, 'timepoints'):  # TimeSeries
        ts = comp.concentration
        n_points = len(ts.timepoints)
        print(f"{prefix}  TimeSeries Data: {n_points} points")
        if n_points > 0:
            t_range = (float(ts.timepoints[0]), float(ts.timepoints[-1]))
            v_range = (float(jnp.min(ts.values)), float(jnp.max(ts.values)))
            print(f"{prefix}    Time range: {t_range[0]:.2f} to {t_range[1]:.2f}")
            print(f"{prefix}    Value range: {v_range[0]:.4f} to {v_range[1]:.4f}")
    elif hasattr(comp.concentration, 'value'):  # StaticVariable
        print(f"{prefix}  Static Concentration: {comp.concentration.value}")


def _print_volume_change_info(change, prefix: str) -> None:
    """Helper function to print VolumeChange information (verbosity=3)."""
    print(f"{prefix}{change.name}:")
    print(f"{prefix}  Type: {'Controlled' if change.is_controlled else 'Modeled'}, "
          f"{'Continuous' if change.is_continuous else 'Discrete'}")
    print(f"{prefix}  Unit: {change.unit}")

    if isinstance(change, FeedVolumeChange) and change.feed_medium:
        print(f"{prefix}  Feed Medium: {change.feed_medium.name}")

    if change.values is not None:
        n_points = len(change.values.timepoints)
        print(f"{prefix}  TimeSeries Points: {n_points}")
        if n_points > 0:
            v_range = (float(jnp.min(change.values.values)),
                       float(jnp.max(change.values.values)))
            print(f"{prefix}    Value range: {v_range[0]:.4f} to {v_range[1]:.4f} {change.unit}")
            if change.is_continuous:
                total_change = float(change.values.values[-1] - change.values.values[0])
                print(f"{prefix}    Total change: {total_change:.2f} {change.unit}")
            else:
                total_change = float(jnp.sum(change.values.values))
                print(f"{prefix}    Total change: {total_change:.2f} {change.unit}")


def _count_datapoints_in_value(value) -> int:
    """
    Count datapoints in a value which may be a TimeSeries or StaticVariable.
    TimeSeries -> number of timepoints
    StaticVariable -> count as 1
    """
    try:
        if hasattr(value, "timepoints"):
            return int(len(value.timepoints))
        elif hasattr(value, "value"):
            return 1
    except Exception:
        # If something unexpected (e.g., None or incompatible), treat as 0
        return 0
    return 0


def _count_datapoints_in_process(process: BioProcess) -> int:
    """
    Count all datapoints (time series lengths + static entries) contained in a BioProcess.
    This includes:
      - process_variables (TimeSeries / StaticVariable)
      - reactor_medium component concentrations (TimeSeries / StaticVariable)
      - volume changes (their TimeSeries)
      - feed medium components referenced in volume changes (TimeSeries / StaticVariable)
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
            if getattr(vc, "feed_medium", None) is not None and isinstance(vc, FeedVolumeChange):
                for fcomp in vc.feed_medium.components.values():
                    total += _count_datapoints_in_value(fcomp.concentration)

    return total


def print_dataset_structure(dataset: BenchmarkDataset, verbosity: int = 3) -> None:
    """
    Print a hierarchical view of the BenchmarkDataset.

    Args:
        dataset: BenchmarkDataset object to inspect
        verbosity: Detail level (1=minimal, 2=medium, 3=full)

            - Level 3 (most verbose): metadata, all case-study details (organism, citation),
              per-process datapoint counts, and total datapoints
            - Level 2 (mid verbose): metadata, case-study names with organism and process
              names listed (no citations, no datapoint counts)
            - Level 1 (least verbose): metadata, case-study names and process count only
    """
    print("=" * 80)
    print("Benchmark Dataset Structure")
    print("=" * 80)

    # Metadata – shown at all verbosity levels
    print("Metadata:")
    if dataset.metadata:
        for k, v in dataset.metadata.items():
            print(f"  {k}: {v}")
    else:
        print("  (no metadata)")

    print("\nCase Studies:")
    if not dataset.case_studies:
        print("  (no case studies)")
    elif verbosity == 1:
        for cs_key, cs in dataset.case_studies.items():
            n_procs = len(cs.processes) if cs.processes else 0
            print(f"  - {cs_key}  ({n_procs} processes)")
    elif verbosity == 2:
        for cs_key, cs in dataset.case_studies.items():
            n_procs = len(cs.processes) if cs.processes else 0
            print(f"  - {cs_key}  |  Organism: {cs.organism}  |  Processes: {n_procs}")
            if cs.processes:
                for p_key, proc in cs.processes.items():
                    name = _get_process_name(proc)
                    print(f"      * {p_key}: {name}")
    
    total_datapoints = 0
    for cs_key, cs in dataset.case_studies.items():
        cs_header = f"{cs_key}"
        try:
            cs_header += f"  (case_id: {cs.case_id})"
        except Exception:
            pass
        if verbosity == 3:
            print(f"  - {cs_header}")
            print(f"      Organism: {cs.organism}")
            print(f"      Citation: {cs.citation}")
        n_procs = len(cs.processes) if cs.processes else 0
        if verbosity == 3:
            print(f"      Processes: {n_procs}")
        if cs.processes:
            for p_key, proc in cs.processes.items():
                name = _get_process_name(proc)
                proc_dp = _count_datapoints_in_process(proc)
                total_datapoints += proc_dp
                if verbosity == 3:
                    print(f"        * {p_key}: {name}  (datapoints: {proc_dp})")
    print("\n" + "-" * 80)
    print(f"Total datapoints in dataset: {total_datapoints}")

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
      - for dynamic: 'x' (timepoints array), 'y' (values array)
      - for static:  't_start' (float), 't_end' (float), 'value' (float)
      - optional: 'render': 'line' | 'bar'
      - optional: 'spline': Interpolator (if available)
      - optional: 'spline_type': 'backtransform' | 'direct'
    """
    t_start = float(process.time_axis.start) if process.time_axis else 0.0
    t_end = float(process.time_axis.end) if process.time_axis else 1.0

    panels = []

    # Reactor medium components
    if process.reactor_medium and process.reactor_medium.components:
        for comp in process.reactor_medium.components.values():
            unit_label = f" [{comp.unit}]" if comp.unit else ""
            if hasattr(comp.concentration, 'timepoints'):
                panel = {
                    'title': f"{comp.name}{unit_label}",
                    'category': 'ReactorMedium',
                    'type': 'dynamic',
                    'x': comp.concentration.timepoints,
                    'y': comp.concentration.values,
                    'render': 'line',
                }
                if comp.spline is not None:
                    panel['spline'] = comp.spline
                    panel['spline_type'] = 'backtransform'
                panels.append(panel)
            else:
                panels.append({
                    'title': f"{comp.name}{unit_label}",
                    'category': 'ReactorMedium',
                    'type': 'static',
                    't_start': t_start, 't_end': t_end,
                    'value': float(comp.concentration.value),
                })

    # Process variables
    if process.process_variables:
        for pv in process.process_variables.values():
            unit_label = f" [{pv.unit}]" if pv.unit else ""
            if hasattr(pv.values, 'timepoints'):
                panel = {
                    'title': f"{pv.name}{unit_label}",
                    'category': 'ProcessVariable',
                    'type': 'dynamic',
                    'x': pv.values.timepoints,
                    'y': pv.values.values,
                    'render': 'line',
                }
                if pv.spline is not None:
                    panel['spline'] = pv.spline
                    panel['spline_type'] = 'direct'
                panels.append(panel)
            else:
                panels.append({
                    'title': f"{pv.name}{unit_label}",
                    'category': 'ProcessVariable',
                    'type': 'static',
                    't_start': t_start, 't_end': t_end,
                    'value': float(pv.values.value),
                })

    # Volume changes
    if process.volume and process.volume.volume_changes:
        for vc in process.volume.volume_changes.values():
            unit_label = f" [{vc.unit}]" if vc.unit else ""
            if vc.values is not None and hasattr(vc.values, 'timepoints'):
                is_continuous = getattr(vc, "is_continuous", True)
                render = 'line' if is_continuous else 'bar'
                panel = {
                    'title': f"{vc.name}{unit_label}",
                    'category': 'VolumeChange',
                    'type': 'dynamic',
                    'x': vc.values.timepoints,
                    'y': vc.values.values,
                    'render': render,
                }
                if getattr(vc, 'spline', None) is not None:
                    panel['spline'] = vc.spline
                    panel['spline_type'] = 'direct'
                panels.append(panel)

    return panels


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


def _evaluate_spline_curve(spline, spline_type, t_start, t_end, n_points=500):
    """Evaluate a spline over [t_start, t_end] and return (t_plot, y_plot)."""
    from .splines import build_backtransform_spline, evaluate_spline_at

    t_plot = np.linspace(t_start, t_end, n_points)
    if spline_type == 'backtransform':
        bt = build_backtransform_spline(spline)
        y_plot = np.array([float(bt(jnp.array(t))) for t in t_plot])
    else:
        y_plot = np.array([evaluate_spline_at(spline, t) for t in t_plot])
    return t_plot, y_plot


def _draw_panel(ax, panel, label=None, color=None, t_start=None, t_end=None):
    """Draw a single panel (dynamic or static) onto *ax*.

    If the panel has a ``'spline'`` key, the spline curve is drawn and raw
    data is shown as scatter points (no connecting lines).  Otherwise raw
    data is drawn with ``'o-'`` markers.
    """
    plot_kwargs = {}
    if color is not None:
        plot_kwargs['color'] = color
    else:
        plot_kwargs['color'] = 'black'

    if label is None:
        label = 'data'

    has_spline = 'spline' in panel and panel['spline'] is not None

    if panel['type'] == 'dynamic':
        x = panel['x']
        y = panel['y']
        render = panel.get('render', 'line')

        n = len(x)

        if render == 'bar':
            # Bar plot for discrete (non-continuous) volume changes
            delta = float(x[-1] - x[0])
            width = max(delta / 30, 0.1)
            ax.bar(x, y, label=label, width=width, edgecolor="k", **plot_kwargs)
        elif has_spline and n <= 50:
            # Scatter for raw data when spline is available and few points
            ax.scatter(x, y, s=16, zorder=5, label=label, **plot_kwargs)
        else:
            fmt = 'o-' if n <= 50 else '-'
            ax.plot(x, y, fmt, markersize=4, label=label, **plot_kwargs)

        if render != 'bar':
            _pad_constant_ylim(ax, y)

        # Draw spline curve
        if has_spline and t_start is not None and t_end is not None:
            try:
                t_plot, y_plot = _evaluate_spline_curve(
                    panel['spline'], panel.get('spline_type', 'direct'),
                    t_start, t_end,
                )
                ax.plot(t_plot, y_plot, '--', color='red', lw=1.5,
                        alpha=0.8, label='spline')
            except Exception as e:
                import warnings
                warnings.warn(f"Could not evaluate spline for {panel.get('title', '?')}: {e}")
    else:
        ax.hlines(
            panel['value'], panel['t_start'], panel['t_end'],
            linestyles='--', label=label, **plot_kwargs,
        )
        _pad_constant_ylim(ax, [panel['value']])


def plot_case_study(case_study: CaseStudy, figsize_per_panel=(5, 3), save_path=None):
    """
    Plot all dynamic and static variables for every process in a CaseStudy.

    All unique variables are discovered across every process first.  Each
    variable gets its own subplot and all processes are overlaid using
    distinct colours.  TimeSeries with ≤ 30 points are drawn with markers;
    longer series are lines only.

    Args:
        case_study: CaseStudy object to plot.
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

    for proc_key, process in case_study.processes.items():
        if process.time_axis is not None:
            time_unit = process.time_axis.unit
            t_global_start = min(t_global_start, float(process.time_axis.start))
            t_global_end = max(t_global_end, float(process.time_axis.end))

        for panel in _collect_process_panels(process):
            key = panel['title']
            if key not in variable_map:
                variable_map[key] = {
                    'title': panel['title'],
                    'time_unit': time_unit,
                    'data': [],
                }
            entry = dict(panel)
            entry['label'] = proc_key
            variable_map[key]['data'].append(entry)

    # some white space to the side
    delta_t_global = t_global_end - t_global_start
    t_global_start -= delta_t_global * 0.05
    t_global_end += delta_t_global * 0.05

    if not variable_map:
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, "No variables to plot", ha='center', va='center',
                transform=ax.transAxes)
        return fig

    panels = list(variable_map.values())
    fig, axes_flat = _make_figure(len(panels), figsize_per_panel)
    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

    # Collect legend entries once (process key -> (handle, label))
    legend_handles_by_label = {}

    for i, panel_meta in enumerate(panels):
        ax = axes_flat[i]
        for j, data in enumerate(panel_meta['data']):
            color = colors[j % len(colors)]
            _draw_panel(ax, data, label=data['label'], color=color,
                        t_start=t_global_start, t_end=t_global_end)

        category = panel_meta['data'][0].get('category', '') if panel_meta['data'] else ''
        ax.set_title(f"{panel_meta['title']} ({category})")
        ax.set_xlabel(f"time [{panel_meta['time_unit']}]")
        ax.set_xlim(t_global_start, t_global_end)
        ax.grid(True, alpha=0.3)

        # Pad y-axis if all overlaid data for this panel is practically constant
        # Skip bar-rendered panels (discrete events) — let matplotlib auto-scale
        has_bar = any(d.get('render') == 'bar' for d in panel_meta['data'])
        if not has_bar:
            all_y = []
            for data in panel_meta['data']:
                if data['type'] == 'dynamic':
                    all_y.extend(np.asarray(data['y'], dtype=float).tolist())
                else:
                    all_y.append(data['value'])
            if all_y:
                _pad_constant_ylim(ax, all_y)

        # harvest handles/labels from this axis (without drawing a per-axis legend)
        handles, labels = ax.get_legend_handles_labels()
        for h, lab in zip(handles, labels):
            if lab and lab not in legend_handles_by_label:
                legend_handles_by_label[lab] = h

    for j in range(len(panels), len(axes_flat)):
        axes_flat[j].set_visible(False)

    fig.suptitle(f"Case Study: {case_study.case_id}", fontsize=12)

    if legend_handles_by_label:
        labels = list(legend_handles_by_label.keys())
        handles = [legend_handles_by_label[l] for l in labels]
        # Put one shared legend at bottom
        fig.legend(
            handles, labels,
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
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
    # fig.show()
    return fig

def _make_figure(n_panels, figsize_per_panel):
    """Create a two-column figure with the correct number of rows."""
    import matplotlib.pyplot as plt

    n_cols = 2
    n_rows = max(1, (n_panels + 1) // 2)
    fig, axes = plt.subplots(
        n_rows, n_cols, squeeze=False,
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
        ax.text(0.5, 0.5, "No variables to plot", ha='center', va='center',
                transform=ax.transAxes)
        return fig

    fig, axes_flat = _make_figure(len(panels), figsize_per_panel)

    for i, panel in enumerate(panels):
        ax = axes_flat[i]
        _draw_panel(ax, panel, t_start=t_start, t_end=t_end)
        category = panel.get('category', '')
        ax.set_title(f"{panel['title']} ({category})")
        ax.set_xlabel(f"time [{time_unit}]")
        ax.set_xlim(t_start, t_end)
        ax.legend(fontsize='small')
        ax.grid(True, alpha=0.3)

    for j in range(len(panels), len(axes_flat)):
        axes_flat[j].set_visible(False)

    fig.suptitle(f"{_get_process_name(process)} ({_get_process_type(process)})", fontsize=12)
    fig.tight_layout()
    if save_path is not None:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
    # fig.show()
    return fig
