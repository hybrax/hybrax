import jax.numpy as jnp
from .dataclasses import BioProcess, CaseStudy, BenchmarkDataset, TimeSeries, FeedVolumeChange


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

    if verbosity == 1:
        # Level 1: just list variable names
        print(f"Process: {process.metadata.name} ({process.metadata.process_type})")
        if process.reactor_medium and process.reactor_medium.components:
            print(f"Reactor Medium Components: {list(process.reactor_medium.components.keys())}")
        if process.process_variables:
            print(f"Process Variables: {list(process.process_variables.keys())}")
        if process.volume and process.volume.volume_changes:
            print(f"Volume Changes: {list(process.volume.volume_changes.keys())}")

    elif verbosity == 2:
        # Level 2: names, controlled status, data type/size – no units or value ranges
        print(f"Process Name: {process.metadata.name}")
        print(f"Process Type: {process.metadata.process_type}")
        if process.metadata.notes:
            print(f"Notes: {process.metadata.notes}")

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
        print(f"Process Name: {process.metadata.name}")
        print(f"Process Type: {process.metadata.process_type}")
        if process.metadata.notes:
            print(f"Notes: {process.metadata.notes}")

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
                    try:
                        name = proc.metadata.name
                    except Exception:
                        name = "<unnamed process>"
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
                try:
                    name = proc.metadata.name
                except Exception:
                    name = "<unnamed process>"
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
      - 'type': 'dynamic' | 'static'
      - for dynamic: 'x' (timepoints array), 'y' (values array)
      - for static:  't_start' (float), 't_end' (float), 'value' (float)
      - optional: 'render': 'line' | 'bar'
    """
    t_start = float(process.time_axis.start) if process.time_axis else 0.0
    t_end = float(process.time_axis.end) if process.time_axis else 1.0

    panels = []

    # Reactor medium components
    if process.reactor_medium and process.reactor_medium.components:
        for comp in process.reactor_medium.components.values():
            unit_label = f" [{comp.unit}]" if comp.unit else ""
            if hasattr(comp.concentration, 'timepoints'):
                panels.append({
                    'title': f"{comp.name}{unit_label}",
                    'type': 'dynamic',
                    'x': comp.concentration.timepoints,
                    'y': comp.concentration.values,
                    'render': 'line',
                })
            else:
                panels.append({
                    'title': f"{comp.name}{unit_label}",
                    'type': 'static',
                    't_start': t_start, 't_end': t_end,
                    'value': float(comp.concentration.value),
                })

    # Process variables
    if process.process_variables:
        for pv in process.process_variables.values():
            unit_label = f" [{pv.unit}]" if pv.unit else ""
            if hasattr(pv.values, 'timepoints'):
                panels.append({
                    'title': f"{pv.name}{unit_label}",
                    'type': 'dynamic',
                    'x': pv.values.timepoints,
                    'y': pv.values.values,
                    'render': 'line',
                })
            else:
                panels.append({
                    'title': f"{pv.name}{unit_label}",
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
                panels.append({
                    'title': f"{vc.name}{unit_label}",
                    'type': 'dynamic',
                    'x': vc.values.timepoints,
                    'y': vc.values.values,
                    'render': render,
                })

    return panels


def _draw_panel(ax, panel, label=None, color=None):
    """Draw a single panel (dynamic or static) onto *ax*."""
    import matplotlib.pyplot as plt

    plot_kwargs = {}
    if color is not None:
        plot_kwargs['color'] = color

    if panel['type'] == 'dynamic':
        x = panel['x']
        y = panel['y']
        render = panel.get('render', 'line')

        if render == 'bar':
            # Bar plot for discrete (non-continuous) volume changes
            delta = x[-1]-x[0]
            width = delta/30
            ax.bar(x, y, label=label, width=width, edgecolor="k", **plot_kwargs)
        else:
            n = len(x)
            fmt = 'o-' if n <= 30 else '-'
            ax.plot(x, y, fmt, markersize=4, label=label, **plot_kwargs)
    else:
        ax.hlines(
            panel['value'], panel['t_start'], panel['t_end'],
            linestyles='--', label=label, **plot_kwargs,
        )
        ax.set_xlim(panel['t_start'], panel['t_end'])


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

    for proc_key, process in case_study.processes.items():
        time_unit = process.time_axis.unit if process.time_axis else "time"

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
            _draw_panel(ax, data, label=data['label'], color=color)

        ax.set_title(panel_meta['title'])
        ax.set_xlabel(f"time [{panel_meta['time_unit']}]")
        ax.grid(True, alpha=0.3)

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
    fig.show()

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

    Args:
        process: BioProcess object to plot.
        figsize_per_panel: ``(width, height)`` in inches for each subplot.

    Returns:
        matplotlib.figure.Figure
    """
    import matplotlib.pyplot as plt

    time_unit = process.time_axis.unit if process.time_axis else "time"
    panels = _collect_process_panels(process)

    if not panels:
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, "No variables to plot", ha='center', va='center',
                transform=ax.transAxes)
        return fig

    fig, axes_flat = _make_figure(len(panels), figsize_per_panel)

    for i, panel in enumerate(panels):
        ax = axes_flat[i]
        _draw_panel(ax, panel)
        ax.set_title(panel['title'])
        ax.set_xlabel(f"time [{time_unit}]")
        ax.grid(True, alpha=0.3)

    for j in range(len(panels), len(axes_flat)):
        axes_flat[j].set_visible(False)

    fig.suptitle(
        f"{process.metadata.name} ({process.metadata.process_type})", fontsize=12
    )
    fig.tight_layout()
    if save_path is not None:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
    fig.show()