import hashlib
import os
import shutil

os.environ.setdefault("JAX_ENABLE_X64", "true")

import diffrax  # noqa: E402
import equinox as eqx  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from bp_format.serialization import load_process_collection_json  # noqa: E402
from bp_format.serialization import save_process_collection_json  # noqa: E402

from .cstar_helpers import (  # noqa: E402
    CANONICAL_ARTIFACTS,
    DATA_DIR,
    EXPECTED_PROCESS_IDS,
    EXPECTED_REACTOR_COMPONENT_ORDER,
    dense_reactor_reference,
    event_aware_adf_value,
    event_aware_feed_correction_value,
    event_jump_times,
    fit_cstar_timeseries,
    populate_exact_pseudobatch_transform,
)
from .load_utils import parse_all_processes  # noqa: E402
from .simulation import REACTOR_STATE_UNITS, run_all_default  # noqa: E402
from .simulation import write_simulation_plots  # noqa: E402

DATA_JSON = DATA_DIR / "process_collection.json"
DIAGNOSTICS_DIR = DATA_DIR / "direct_cstar_diagnostics"
SOLVER_RTOL = 1e-10
SOLVER_ATOL = 1e-12
DIAGNOSTIC_DPI = 120
PLOT_POINTS = 241


class DirectCstarRHS(eqx.Module):
    breaks: jnp.ndarray
    coeffs: jnp.ndarray

    def __call__(self, t, y, args):
        del y, args
        t = jnp.asarray(t, dtype=self.breaks.dtype)
        idx = jnp.searchsorted(self.breaks, t, side="right") - 1
        idx = jnp.clip(idx, 0, self.breaks.shape[0] - 2)
        dt = t - self.breaks[idx]
        coeffs = self.coeffs[idx]
        a = coeffs[:, 0]
        b = coeffs[:, 1]
        c = coeffs[:, 2]
        d = coeffs[:, 3]
        return a + dt * (b + dt * (c + dt * d))


def _direct_rhs(process):
    derivative_series = [
        process.reactor_medium.components[name].c_star_concentration.deriv()
        for name in EXPECTED_REACTOR_COMPONENT_ORDER
    ]
    first_breaks = np.asarray(derivative_series[0].breaks, dtype=float)
    for series in derivative_series:
        np.testing.assert_allclose(np.asarray(series.breaks, dtype=float), first_breaks)
    coeffs = jnp.stack([series.coeffs for series in derivative_series], axis=1)
    return DirectCstarRHS(jnp.asarray(first_breaks), coeffs)


def _integrate_direct_cstar(process, rhs, times, y0):
    event_jumps = event_jump_times(process)
    jump_ts = event_jumps[(times[0] < event_jumps) & (event_jumps < times[-1])]
    solution = diffrax.diffeqsolve(
        diffrax.ODETerm(rhs),
        diffrax.Dopri8(),
        t0=float(times[0]),
        t1=float(times[-1]),
        dt0=0.05,
        y0=jnp.asarray(y0),
        saveat=diffrax.SaveAt(ts=jnp.asarray(times)),
        stepsize_controller=diffrax.PIDController(
            rtol=SOLVER_RTOL,
            atol=SOLVER_ATOL,
            jump_ts=jnp.asarray(jump_ts),
        ),
        max_steps=100_000,
    )
    return np.asarray(solution.ys, dtype=float)


def _backtransform_left(process, times, cstar_values):
    jump_times = event_jump_times(process)
    return np.column_stack(
        [
            np.asarray(
                [
                    (
                        cstar_values[row, column]
                        + event_aware_feed_correction_value(
                            process,
                            name,
                            float(time),
                            jump_times,
                            right_of_jump=False,
                        )
                    )
                    / event_aware_adf_value(
                        process,
                        float(time),
                        jump_times,
                        right_of_jump=False,
                    )
                    for row, time in enumerate(times)
                ],
                dtype=float,
            )
            for column, name in enumerate(EXPECTED_REACTOR_COMPONENT_ORDER)
        ]
    )


def _cumulative_discrete_change(volume_change, times):
    change_times = np.asarray(volume_change.values.times, dtype=float)
    change_values = np.asarray(volume_change.values.values, dtype=float)
    cumulative = np.zeros(len(times), dtype=float)
    for time, value in zip(change_times, change_values, strict=True):
        cumulative[times > time] += value
    return cumulative


def _mark_discrete_volume_events(process, axes):
    for ax in axes:
        for volume_change in process.volume.volume_changes.values():
            if volume_change.is_continuous:
                continue
            color = (
                "tab:red" if np.all(volume_change.values.values <= 0.0) else "tab:blue"
            )
            for time in np.asarray(volume_change.values.times, dtype=float):
                ax.axvline(time, color=color, linewidth=0.6, alpha=0.35)


def _plot_volume_panels(process, volume_axes):
    volume_times = np.asarray(process.volume.total_volume.times, dtype=float)
    total_volume = np.asarray(process.volume.total_volume.values, dtype=float)

    ax = volume_axes[0]
    ax.plot(volume_times, total_volume, color="black", label="total volume")
    ax.set_title("total volume [L]")
    ax.legend(loc="best", fontsize="x-small")

    ax = volume_axes[1]
    for name, volume_change in process.volume.volume_changes.items():
        if volume_change.is_continuous:
            ax.plot(
                np.asarray(volume_change.values.times, dtype=float),
                np.asarray(volume_change.values.values, dtype=float),
                label=name,
            )
            continue
        cumulative = _cumulative_discrete_change(volume_change, volume_times)
        if np.all(cumulative <= 0.0):
            ax.plot(volume_times, -cumulative, label=f"{name} removed")
        else:
            ax.plot(volume_times, cumulative, label=f"{name} added")
    ax.set_title("cumulative vol. changes [L]")
    ax.legend(loc="best", fontsize="x-small")


def _write_plots(
    diagnostics_dir,
    process,
    process_name,
    sparse_times,
    plot_times,
    sparse_cstar,
    fitted_plot_cstar,
    integrated_plot_cstar,
    observed_concentrations,
    recovered_plot_concentrations,
):
    diagnostics_dir.mkdir(parents=True, exist_ok=True)
    dense = dense_reactor_reference(process_name, float(plot_times[-1]))

    real_path = diagnostics_dir / f"{process_name}_real_space.png"
    fig, axes = plt.subplots(5, 2, figsize=(12, 12), sharex=True)
    component_axes = axes.ravel()[: len(EXPECTED_REACTOR_COMPONENT_ORDER)]
    for idx, (ax, name) in enumerate(
        zip(component_axes, EXPECTED_REACTOR_COMPONENT_ORDER, strict=True)
    ):
        ax.plot(dense["time"], dense[name], color="0.75", label="dense")
        ax.scatter(
            sparse_times,
            observed_concentrations[:, idx],
            s=18,
            color="tab:orange",
            label="sparse",
        )
        ax.plot(
            plot_times,
            recovered_plot_concentrations[:, idx],
            color="tab:blue",
            label="reintegrated",
        )
        ax.set_title(f"{name} [{REACTOR_STATE_UNITS[name]}]")
    _plot_volume_panels(process, axes[-1])
    _mark_discrete_volume_events(process, axes.ravel())
    axes.ravel()[0].legend(loc="best", fontsize="small")
    fig.tight_layout()
    fig.savefig(real_path, dpi=DIAGNOSTIC_DPI)
    plt.close(fig)

    cstar_path = diagnostics_dir / f"{process_name}_cstar.png"
    fig, axes = plt.subplots(5, 2, figsize=(12, 12), sharex=True)
    component_axes = axes.ravel()[: len(EXPECTED_REACTOR_COMPONENT_ORDER)]
    for idx, (ax, name) in enumerate(
        zip(component_axes, EXPECTED_REACTOR_COMPONENT_ORDER, strict=True)
    ):
        ax.plot(
            plot_times,
            fitted_plot_cstar[:, idx],
            color="tab:green",
            label="fitted",
        )
        ax.scatter(
            sparse_times,
            sparse_cstar[:, idx],
            s=18,
            color="tab:orange",
            label="sparse",
        )
        ax.plot(
            plot_times,
            integrated_plot_cstar[:, idx],
            "--",
            color="tab:blue",
            label="integrated",
        )
        ax.set_title(f"{name} c* [{REACTOR_STATE_UNITS[name]}]")
    _plot_volume_panels(process, axes[-1])
    _mark_discrete_volume_events(process, axes.ravel())
    axes.ravel()[0].legend(loc="best", fontsize="small")
    fig.tight_layout()
    fig.savefig(cstar_path, dpi=DIAGNOSTIC_DPI)
    plt.close(fig)

    for path in (real_path, cstar_path):
        assert path.exists()
        assert path.stat().st_size > 0


def _sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _assert_simulation_artifacts_match(new_root):
    for artifact in CANONICAL_ARTIFACTS:
        new_path = new_root / artifact
        old_path = DATA_DIR / artifact
        assert new_path.exists(), f"missing regenerated artifact: {new_path}"
        if not old_path.exists():
            old_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(new_path, old_path)
            continue
        assert _sha256(new_path) == _sha256(old_path), old_path


def test_sim_1_direct_cstar_reintegration(tmp_path):
    simulation_dir = tmp_path / "simulation"
    results = run_all_default(output_dir=simulation_dir)
    write_simulation_plots(simulation_dir / "simulation_plots", results)
    _assert_simulation_artifacts_match(simulation_dir)

    parsed_json = tmp_path / "process_collection.json"
    parsed_collection = parse_all_processes(
        dense_csv=simulation_dir / "simulation_dense_output.csv",
        events_csv=simulation_dir / "simulation_events.csv",
    )
    save_process_collection_json(parsed_collection, parsed_json)
    assert _sha256(parsed_json) == _sha256(DATA_JSON)

    collection = load_process_collection_json(DATA_JSON)
    assert set(collection.processes) == EXPECTED_PROCESS_IDS

    for process_name, process in collection.processes.items():
        populate_exact_pseudobatch_transform(process)
        for component in process.reactor_medium.components.values():
            component.c_star_concentration = fit_cstar_timeseries(process, component)

        sparse_times = np.asarray(
            process.reactor_medium.components[
                EXPECTED_REACTOR_COMPONENT_ORDER[0]
            ].concentration.times,
            dtype=float,
        )
        for name in EXPECTED_REACTOR_COMPONENT_ORDER:
            np.testing.assert_array_equal(
                np.asarray(
                    process.reactor_medium.components[name].concentration.times,
                    dtype=float,
                ),
                sparse_times,
            )

        fitted_cstar = np.column_stack(
            [
                np.asarray(
                    process.reactor_medium.components[
                        name
                    ].c_star_concentration.evaluate_many(jnp.asarray(sparse_times)),
                    dtype=float,
                )
                for name in EXPECTED_REACTOR_COMPONENT_ORDER
            ]
        )
        rhs = _direct_rhs(process)
        integrated_cstar = _integrate_direct_cstar(
            process,
            rhs,
            sparse_times,
            fitted_cstar[0],
        )
        np.testing.assert_allclose(
            integrated_cstar,
            fitted_cstar,
            rtol=1e-7,
            atol=1e-9,
        )

        recovered_concentrations = _backtransform_left(
            process,
            sparse_times,
            integrated_cstar,
        )
        observed_concentrations = np.column_stack(
            [
                np.asarray(
                    process.reactor_medium.components[name].concentration.values,
                    dtype=float,
                )
                for name in EXPECTED_REACTOR_COMPONENT_ORDER
            ]
        )
        np.testing.assert_allclose(
            recovered_concentrations,
            observed_concentrations,
            rtol=1e-7,
            atol=1e-9,
        )
        plot_times = np.linspace(sparse_times[0], sparse_times[-1], PLOT_POINTS)
        fitted_plot_cstar = np.column_stack(
            [
                np.asarray(
                    process.reactor_medium.components[
                        name
                    ].c_star_concentration.evaluate_many(jnp.asarray(plot_times)),
                    dtype=float,
                )
                for name in EXPECTED_REACTOR_COMPONENT_ORDER
            ]
        )
        integrated_plot_cstar = _integrate_direct_cstar(
            process,
            rhs,
            plot_times,
            fitted_cstar[0],
        )
        recovered_plot_concentrations = _backtransform_left(
            process,
            plot_times,
            integrated_plot_cstar,
        )
        for diagnostics_dir in (tmp_path / "direct_cstar_diagnostics", DIAGNOSTICS_DIR):
            _write_plots(
                diagnostics_dir,
                process,
                process_name,
                sparse_times,
                plot_times,
                fitted_cstar,
                fitted_plot_cstar,
                integrated_plot_cstar,
                observed_concentrations,
                recovered_plot_concentrations,
            )
