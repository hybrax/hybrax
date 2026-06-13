import hashlib
import os

os.environ.setdefault("JAX_ENABLE_X64", "true")

import bp_format as bp  # noqa: E402
import diffrax  # noqa: E402
import equinox as eqx  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from bp_format.serialization import load_process_collection_json  # noqa: E402
from bp_format.serialization import save_process_collection_json  # noqa: E402
from bp_format.splines import build_pseudobatch_transform  # noqa: E402
from bp_format.splines import evaluate_pseudobatch_transform  # noqa: E402

from .cstar_helpers import (  # noqa: E402
    CANONICAL_ARTIFACTS,
    DIAGNOSTIC_PLOTS_DIR,
    EVENT_TIME_ATOL,
    EXPECTED_PROCESS_IDS,
    EXPECTED_REACTOR_COMPONENT_ORDER,
    dense_event_pair_reference,
    dense_online_q_rate_reference,
    dense_online_reactor_reference,
    dense_reactor_reference,
    event_jump_times,
    fit_cstar_timeseries_from_values,
    SIM_1_DIR,
    SIM_RESULTS_DIR,
)
from .load_utils import parse_all_processes  # noqa: E402
from .simulation import REACTOR_STATE_UNITS, run_all_default  # noqa: E402
from .simulation import write_simulation_plots  # noqa: E402

DATA_JSON = SIM_RESULTS_DIR / "process_collection.json"
LOCAL_DIAGNOSTICS_DIR = SIM_1_DIR / "tmp" / "cstar_integration_sparse"
SOLVER_RTOL = 1e-10
SOLVER_ATOL = 1e-12
DENSE_SOLVER_RTOL = 1e-13
DENSE_SOLVER_ATOL = 1e-15
DENSE_FITTED_CSTAR_RTOL = 1e-10
DENSE_FITTED_CSTAR_ATOL = 1e-12
DENSE_CSTAR_RTOL = 1e-9
DENSE_CSTAR_ATOL = 1e-11
DENSE_CONCENTRATION_RTOL = 2e-9
DENSE_CONCENTRATION_ATOL = 1e-9
PUBLIC_TRANSFORM_RTOL = 1e-12
PUBLIC_TRANSFORM_ATOL = 1e-12
PUBLIC_BACKTRANSFORM_RTOL = 1e-7
PUBLIC_BACKTRANSFORM_ATOL = 1e-9
EVENT_CSTAR_CONTINUITY_RTOL = 1e-10
EVENT_CSTAR_CONTINUITY_ATOL = 1e-10
ADF_VOLUME_ORACLE_RTOL = 1e-9
ADF_VOLUME_ORACLE_ATOL = 1e-10
FEED_VOLUME_ORACLE_RTOL = 1e-9
FEED_VOLUME_ORACLE_ATOL = 1e-10
FEED_SPECIES_RATIO_RTOL = 1e-10
FEED_SPECIES_RATIO_ATOL = 1e-10
FEED_BOLUS_JUMP_RTOL = 1e-9
FEED_BOLUS_JUMP_ATOL = 1e-10
SPARSE_CSTAR_RTOL = 1e-7
SPARSE_CSTAR_ATOL = 1e-9
SPARSE_CONCENTRATION_RTOL = 1e-7
SPARSE_CONCENTRATION_ATOL = 1e-9
DIAGNOSTIC_DPI = 120
PLOT_POINTS = 241
PLOT_POST_EVENT_EPS = 1e-6


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


def _integrate_direct_cstar(
    process,
    rhs,
    times,
    y0,
    *,
    rtol=SOLVER_RTOL,
    atol=SOLVER_ATOL,
):
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
            rtol=rtol,
            atol=atol,
            jump_ts=jnp.asarray(jump_ts),
        ),
        max_steps=100_000,
    )
    return np.asarray(solution.ys, dtype=float)


def _build_public_pseudobatch_transform(process):
    process.pseudobatch_transform = build_pseudobatch_transform(
        process,
        list(EXPECTED_REACTOR_COMPONENT_ORDER),
    )


def _evaluate_left(series, times):
    return np.asarray(
        series.evaluate_many(jnp.asarray(times), side="left"), dtype=float
    )


def _transform_values(process, component_name, times, *, side):
    assert process.pseudobatch_transform is not None
    t_eval = jnp.asarray(times)
    adf = np.asarray(
        process.pseudobatch_transform.adf.evaluate_many(t_eval, side=side),
        dtype=float,
    )
    feed_correction = np.asarray(
        process.pseudobatch_transform.feed_corrections[component_name].evaluate_many(
            t_eval, side=side
        ),
        dtype=float,
    )
    return adf, feed_correction


def _transform_values_left(process, component_name, times):
    return _transform_values(process, component_name, times, side="left")


def _public_cstar_from_concentrations(process, dense):
    times = dense["time"]
    return np.column_stack(
        [
            dense[name] * adf - feed_correction
            for name in EXPECTED_REACTOR_COMPONENT_ORDER
            for adf, feed_correction in [_transform_values_left(process, name, times)]
        ]
    )


def _public_backtransform_from_cstar(process, times, cstar_values):
    return np.column_stack(
        [
            (cstar_values[:, column] + feed_correction) / adf
            for column, name in enumerate(EXPECTED_REACTOR_COMPONENT_ORDER)
            for adf, feed_correction in [_transform_values_left(process, name, times)]
        ]
    )


def _assert_public_volume_matches_dense_simulation(process, dense):
    assert process.volume.total_volume is not None
    public_volume = _evaluate_left(process.volume.total_volume, dense["time"])
    np.testing.assert_allclose(
        public_volume,
        dense["volume"],
        rtol=PUBLIC_TRANSFORM_RTOL,
        atol=PUBLIC_TRANSFORM_ATOL,
    )


def _assert_public_transform_preserves_event_cstar(process, event_pairs):
    for pre_event, post_event in event_pairs:
        time = pre_event["time"]
        assert post_event["time"] == time
        times = np.asarray([time], dtype=float)
        for name in EXPECTED_REACTOR_COMPONENT_ORDER:
            pre_adf, pre_feed_correction = _transform_values(
                process,
                name,
                times,
                side="left",
            )
            post_adf, post_feed_correction = _transform_values(
                process,
                name,
                times,
                side="right",
            )
            pre_cstar = pre_event[name] * pre_adf[0] - pre_feed_correction[0]
            post_cstar = post_event[name] * post_adf[0] - post_feed_correction[0]
            np.testing.assert_allclose(
                post_cstar,
                pre_cstar,
                rtol=EVENT_CSTAR_CONTINUITY_RTOL,
                atol=EVENT_CSTAR_CONTINUITY_ATOL,
            )


def _evaluate_public_backtransform(process, times):
    return np.column_stack(
        [
            np.asarray(
                evaluate_pseudobatch_transform(process, name, jnp.asarray(times)),
                dtype=float,
            )
            for name in EXPECTED_REACTOR_COMPONENT_ORDER
        ]
    )


# --- Orthogonal ground-truth checks for the public pseudobatch carriers ---
#
# The dense reintegration below builds c* from the public ADF/feed-correction
# carriers and inverts it with the same carriers, so a wrong carrier *magnitude*
# would cancel. The helpers below pin those magnitudes against simulator ground
# truth instead: expected values come from the integrated dense output (reactor
# volume, cumulative feed volumes) plus the known experimental protocol (sample
# and bolus volumes, feed concentrations), never from bp_format's carrier
# bookkeeping. See B-D-orthogonal-carrier-checks/TASK.md.


def _initial_volume(process):
    return float(process.volume.initial_volume)


def _feed_concentration(process, species_name):
    """Shared feed concentration of a species across all streams that carry it.

    Asserts every stream feeding this species agrees (true for sim_1, where the
    continuous nutrient feed and the bolus feed share one medium and the base feed
    carries nothing). Returns 0.0 for unfed species.
    """
    concentrations = set()
    for volume_change in process.volume.volume_changes.values():
        if not isinstance(volume_change, bp.FeedVolumeChange):
            continue
        component = volume_change.feed_medium.components[species_name]
        value = float(component.concentration.value)
        if value != 0.0:
            concentrations.add(value)
    assert len(concentrations) <= 1, (species_name, concentrations)
    return concentrations.pop() if concentrations else 0.0


def _sample_compensation_factors(process, event_pairs):
    """Independent per-sample compensation factors v_before / v_after.

    v_before is the simulator pre-event reactor volume; v_after applies only the
    protocol sample volume (sampling precedes any same-time bolus). Routes through
    the integrated volume trace, not the public sample-compensation series.
    """
    pre_event_volume = {pre["time"]: pre["volume"] for pre, _post in event_pairs}
    factors = []
    for volume_change in process.volume.volume_changes.values():
        if not isinstance(volume_change, bp.SampleVolumeChange):
            continue
        times = np.asarray(volume_change.values.times, dtype=float)
        deltas = np.asarray(volume_change.values.values, dtype=float)
        for time, delta in zip(times, deltas, strict=True):
            time = float(time)
            matches = [t for t in pre_event_volume if abs(t - time) <= EVENT_TIME_ATOL]
            assert len(matches) == 1, (time, matches)
            v_before = pre_event_volume[matches[0]]
            v_after = v_before + float(delta)
            factors.append((time, v_before / v_after))
    factors.sort()
    return factors


def _sample_compensation_at(factors, time, *, inclusive):
    """Product of sample factors active at `time`.

    inclusive=False gives the left/pre-event value (a sample exactly at `time` is
    excluded); inclusive=True includes a same-time sample, matching the fact that a
    bolus is applied after the sample that shares its timestamp.
    """
    compensation = 1.0
    for sample_time, factor in factors:
        if inclusive:
            active = sample_time <= time + EVENT_TIME_ATOL
        else:
            active = sample_time < time - EVENT_TIME_ATOL
        if active:
            compensation *= factor
    return compensation


def _assert_public_adf_matches_volume_oracle(process, dense, factors):
    """Check B: public ADF equals V_dense/V_init * sample_compensation."""
    times = dense["time"]
    v_init = _initial_volume(process)
    compensation = np.asarray(
        [_sample_compensation_at(factors, float(t), inclusive=False) for t in times],
        dtype=float,
    )
    expected_adf = dense["volume"] / v_init * compensation
    public_adf = _evaluate_left(process.pseudobatch_transform.adf, times)
    np.testing.assert_allclose(
        public_adf,
        expected_adf,
        rtol=ADF_VOLUME_ORACLE_RTOL,
        atol=ADF_VOLUME_ORACLE_ATOL,
    )


def _assert_public_feed_correction_pre_sample(process, dense, factors):
    """Check D1: pre-first-sample feed correction equals c_feed * fed_volume / V_init.

    The window before the first sample is chosen because sample compensation is
    identically 1 there, so the carrier is an unweighted mass balance over the
    recorded cumulative feed volumes. This has teeth only for processes that feed
    before their first sample (run_1, continuous feed); a process with no
    pre-sample feed (run_2) exercises only the trivial 0 == 0 case here, and its
    feed-correction magnitude is instead pinned by the bolus-jump check (D3).
    """
    times = dense["time"]
    v_init = _initial_volume(process)
    first_sample = min((t for t, _ in factors), default=np.inf)
    pre_sample = times < first_sample - EVENT_TIME_ATOL
    assert np.any(pre_sample)
    fed_volume = dense["cum_conti_feed"] + dense["cum_bolus_feed"]
    for name in EXPECTED_REACTOR_COMPONENT_ORDER:
        expected = _feed_concentration(process, name) * fed_volume / v_init
        public_fc = _evaluate_left(
            process.pseudobatch_transform.feed_corrections[name], times
        )
        np.testing.assert_allclose(
            public_fc[pre_sample],
            expected[pre_sample],
            rtol=FEED_VOLUME_ORACLE_RTOL,
            atol=FEED_VOLUME_ORACLE_ATOL,
        )


def _assert_public_feed_correction_species_ratio(process, dense):
    """Check D2: feed correction is linear in feed concentration across species."""
    times = dense["time"]
    fed = [
        (name, _feed_concentration(process, name))
        for name in EXPECTED_REACTOR_COMPONENT_ORDER
    ]
    fed = [(name, conc) for name, conc in fed if conc != 0.0]
    assert len(fed) >= 2
    ref_name, ref_conc = fed[0]
    ref_fc = _evaluate_left(
        process.pseudobatch_transform.feed_corrections[ref_name], times
    )
    for name, conc in fed[1:]:
        public_fc = _evaluate_left(
            process.pseudobatch_transform.feed_corrections[name], times
        )
        np.testing.assert_allclose(
            public_fc * ref_conc,
            ref_fc * conc,
            rtol=FEED_SPECIES_RATIO_RTOL,
            atol=FEED_SPECIES_RATIO_ATOL,
        )


def _assert_public_feed_correction_bolus_jumps(process, factors):
    """Check D3: each bolus jump equals sample_compensation * delta_V * c_feed / V_init.

    Feed correction is continuous across a pure sample, so the right-minus-left jump
    at a bolus time isolates the bolus contribution. The compensation is taken
    inclusive of any same-time sample.
    """
    v_init = _initial_volume(process)
    checked = 0
    for volume_change in process.volume.volume_changes.values():
        if not isinstance(volume_change, bp.FeedVolumeChange):
            continue
        if volume_change.is_continuous:
            continue
        times = np.asarray(volume_change.values.times, dtype=float)
        deltas = np.asarray(volume_change.values.values, dtype=float)
        for time, delta_v in zip(times, deltas, strict=True):
            time = float(time)
            t_eval = jnp.asarray([time])
            compensation = _sample_compensation_at(factors, time, inclusive=True)
            for name in EXPECTED_REACTOR_COMPONENT_ORDER:
                series = process.pseudobatch_transform.feed_corrections[name]
                left = float(series.evaluate_many(t_eval, side="left")[0])
                right = float(series.evaluate_many(t_eval, side="right")[0])
                c_feed = float(
                    volume_change.feed_medium.components[name].concentration.value
                )
                expected_jump = compensation * float(delta_v) * c_feed / v_init
                np.testing.assert_allclose(
                    right - left,
                    expected_jump,
                    rtol=FEED_BOLUS_JUMP_RTOL,
                    atol=FEED_BOLUS_JUMP_ATOL,
                )
            checked += 1
    assert checked > 0


def _plot_time_grid(process, start, end):
    base_times = np.linspace(start, end, PLOT_POINTS)
    jump_times = event_jump_times(process)
    jump_times = jump_times[(start < jump_times) & (jump_times < end)]
    post_jump_times = jump_times + PLOT_POST_EVENT_EPS
    post_jump_times = post_jump_times[post_jump_times < end]
    return np.unique(np.concatenate([base_times, jump_times, post_jump_times]))


def _print_component_errors(label, actual, desired):
    print(label)
    for column, name in enumerate(EXPECTED_REACTOR_COMPONENT_ORDER):
        abs_error = np.abs(actual[:, column] - desired[:, column])
        denominator = np.abs(desired[:, column])
        rel_error = np.full_like(abs_error, np.inf)
        nonzero = denominator > 0.0
        rel_error[nonzero] = abs_error[nonzero] / denominator[nonzero]
        rel_error[~nonzero & (abs_error == 0.0)] = 0.0
        print(
            f"  {name}: max_abs={np.max(abs_error):.6e}, "
            f"max_rel={np.max(rel_error):.6e}"
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


def _infer_q_rates(process, times, integrated_cstar):
    cstar_derivatives = np.column_stack(
        [
            np.asarray(
                process.reactor_medium.components[name]
                .c_star_concentration.deriv()
                .evaluate_many(jnp.asarray(times)),
                dtype=float,
            )
            for name in EXPECTED_REACTOR_COMPONENT_ORDER
        ]
    )
    biomass_index = EXPECTED_REACTOR_COMPONENT_ORDER.index("biomass")
    product_index = EXPECTED_REACTOR_COMPONENT_ORDER.index("product_intracellular")
    active_biomass_star = (
        integrated_cstar[:, biomass_index] - integrated_cstar[:, product_index]
    )
    return cstar_derivatives / active_biomass_star[:, np.newaxis]


def _assert_tracked_equivalent_matches(path):
    canonical_path = DIAGNOSTIC_PLOTS_DIR / "cstar_integration_sparse" / path.name
    if canonical_path.exists():
        assert _sha256(path) == _sha256(canonical_path), canonical_path


def _plot_real_axis(
    ax,
    dense,
    sparse_times,
    plot_times,
    observed_concentrations,
    recovered_plot_concentrations,
    idx,
    name,
):
    ax.plot(dense["time"], dense[name], color="0.75", label="dense")
    ax.scatter(
        sparse_times,
        observed_concentrations[:, idx],
        s=14,
        color="tab:orange",
        label="offline",
    )
    ax.plot(
        plot_times,
        recovered_plot_concentrations[:, idx],
        color="tab:blue",
        label="reintegrated",
    )


def _plot_cstar_axis(
    ax,
    sparse_times,
    plot_times,
    sparse_cstar,
    fitted_plot_cstar,
    idx,
):
    ax.plot(
        plot_times,
        fitted_plot_cstar[:, idx],
        color="tab:green",
        label="fitted c*",
    )
    ax.scatter(
        sparse_times,
        sparse_cstar[:, idx],
        s=14,
        color="tab:olive",
        label="sparse c*",
    )


def _mark_zero_lines(axes):
    for ax in axes:
        ymin, ymax = ax.get_ylim()
        if ymin <= 0.0 <= ymax:
            ax.axhline(0.0, color="0.3", linewidth=0.6, alpha=0.45)
            ax.set_ylim(ymin, ymax)


def _plot_q_axis(
    ax,
    truth,
    sparse_times,
    plot_times,
    inferred_q,
    sparse_inferred_q,
    idx,
    name,
):
    ax.plot(
        truth["time"],
        truth[name],
        color="0.65",
        label="simulator q",
    )
    ax.scatter(
        sparse_times,
        np.interp(sparse_times, truth["time"], truth[name]),
        s=14,
        color="0.45",
    )
    ax.plot(
        plot_times,
        inferred_q[:, idx],
        color="tab:purple",
        label="c* derivative / X_active*",
    )
    ax.scatter(
        sparse_times,
        sparse_inferred_q[:, idx],
        s=14,
        color="tab:purple",
    )
    ax.set_title(f"q_{name}")


def _clear_local_diagnostic_plots():
    if not LOCAL_DIAGNOSTICS_DIR.exists():
        return
    for path in LOCAL_DIAGNOSTICS_DIR.glob("*.png"):
        path.unlink()


def _write_plots(
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
    LOCAL_DIAGNOSTICS_DIR.mkdir(parents=True, exist_ok=True)
    dense = dense_reactor_reference(process_name, float(plot_times[-1]))
    truth = dense_online_q_rate_reference(process_name, float(plot_times[-1]))
    inferred_q = _infer_q_rates(process, plot_times, integrated_plot_cstar)
    sparse_inferred_q = _infer_q_rates(process, sparse_times, sparse_cstar)

    combined_path = LOCAL_DIAGNOSTICS_DIR / f"{process_name}_overview.png"
    fig, axes = plt.subplots(
        len(EXPECTED_REACTOR_COMPONENT_ORDER) + 1,
        2,
        figsize=(14, 22),
        sharex=True,
    )
    for idx, name in enumerate(EXPECTED_REACTOR_COMPONENT_ORDER):
        species_ax = axes[idx, 0]
        _plot_real_axis(
            species_ax,
            dense,
            sparse_times,
            plot_times,
            observed_concentrations,
            recovered_plot_concentrations,
            idx,
            name,
        )
        _plot_cstar_axis(
            species_ax,
            sparse_times,
            plot_times,
            sparse_cstar,
            fitted_plot_cstar,
            idx,
        )
        species_ax.set_title(f"{name}: real-space and c* [{REACTOR_STATE_UNITS[name]}]")

        _plot_q_axis(
            axes[idx, 1],
            truth,
            sparse_times,
            plot_times,
            inferred_q,
            sparse_inferred_q,
            idx,
            name,
        )

    volume_axes = axes[-1]
    _plot_volume_panels(process, volume_axes)
    _mark_discrete_volume_events(process, axes.ravel())
    _mark_zero_lines(axes.ravel())
    axes[0, 0].legend(loc="best", fontsize="small")
    axes[0, 1].legend(loc="best", fontsize="small")
    fig.tight_layout()
    fig.savefig(combined_path, dpi=DIAGNOSTIC_DPI)
    plt.close(fig)

    assert combined_path.exists()
    assert combined_path.stat().st_size > 0
    _assert_tracked_equivalent_matches(combined_path)


def _sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _assert_simulation_artifacts_match(new_root):
    for artifact, canonical_path in CANONICAL_ARTIFACTS.items():
        new_path = new_root / artifact
        assert new_path.exists(), f"missing regenerated artifact: {new_path}"
        assert canonical_path.exists(), f"missing canonical artifact: {canonical_path}"
        assert _sha256(new_path) == _sha256(canonical_path), canonical_path


def test_sim_1_direct_cstar_reintegration(tmp_path):
    simulation_dir = tmp_path / "simulation"
    results = run_all_default(output_dir=simulation_dir)
    write_simulation_plots(simulation_dir / "sim_plots", results)
    _assert_simulation_artifacts_match(simulation_dir)

    # First write the raw parsed collection and compare it to the canonical
    # parser artifact. This keeps c* fitting/pseudobatch enrichment out of the
    # canonical raw-data JSON contract.
    parsed_json = tmp_path / "process_collection.json"
    parsed_collection = parse_all_processes(
        dense_csv=simulation_dir / "simulation_dense_output.csv",
        events_csv=simulation_dir / "simulation_events.csv",
    )
    save_process_collection_json(parsed_collection, parsed_json)
    assert _sha256(parsed_json) == _sha256(DATA_JSON)

    collection = load_process_collection_json(DATA_JSON)
    assert set(collection.processes) == EXPECTED_PROCESS_IDS
    _clear_local_diagnostic_plots()

    for process in collection.processes.values():
        _build_public_pseudobatch_transform(process)

    # Then write a second, derived collection after adding the public
    # pseudobatch transform and fitted c* splines. Reloading it verifies that
    # the enrichment survives JSON serialization before reintegration uses it.
    enriched_json = tmp_path / "process_collection_with_cstar.json"
    save_process_collection_json(collection, enriched_json)
    collection = load_process_collection_json(enriched_json)
    assert set(collection.processes) == EXPECTED_PROCESS_IDS

    for process_name, process in collection.processes.items():
        assert process.pseudobatch_transform is not None
        for component in process.reactor_medium.components.values():
            assert component.c_star_concentration is not None

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
        recovered_concentrations = _public_backtransform_from_cstar(
            process,
            sparse_times,
            integrated_cstar,
        )
        public_backtransformed_concentrations = _evaluate_public_backtransform(
            process,
            sparse_times,
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
        plot_times = _plot_time_grid(process, sparse_times[0], sparse_times[-1])
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
        recovered_plot_concentrations = _public_backtransform_from_cstar(
            process,
            plot_times,
            integrated_plot_cstar,
        )
        _write_plots(
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
        np.testing.assert_allclose(
            integrated_cstar,
            fitted_cstar,
            rtol=SPARSE_CSTAR_RTOL,
            atol=SPARSE_CSTAR_ATOL,
        )
        np.testing.assert_allclose(
            public_backtransformed_concentrations,
            observed_concentrations,
            rtol=PUBLIC_BACKTRANSFORM_RTOL,
            atol=PUBLIC_BACKTRANSFORM_ATOL,
        )
        np.testing.assert_allclose(
            recovered_concentrations,
            observed_concentrations,
            rtol=SPARSE_CONCENTRATION_RTOL,
            atol=SPARSE_CONCENTRATION_ATOL,
        )

    assert {path.name for path in LOCAL_DIAGNOSTICS_DIR.glob("*.png")} == {
        f"{process_name}_overview.png" for process_name in EXPECTED_PROCESS_IDS
    }


def test_sim_1_dense_cstar_oracle_reintegration():
    collection = load_process_collection_json(DATA_JSON)
    assert set(collection.processes) == EXPECTED_PROCESS_IDS

    for process_name, process in collection.processes.items():
        _build_public_pseudobatch_transform(process)
        dense = dense_online_reactor_reference(process_name, process.time_axis.end)
        dense_times = dense["time"]
        assert len(dense_times) > 0
        assert dense_times[0] == process.time_axis.start
        assert dense_times[-1] == process.time_axis.end
        assert np.all(np.diff(dense_times) > 0.0)
        for jump_time in event_jump_times(process):
            assert np.any(
                np.isclose(dense_times, jump_time, rtol=0.0, atol=EVENT_TIME_ATOL)
            )

        # Dense online rows are left/pre-event states at exact event times:
        # Simulation.build_dense_rows emits online rows before pre/post-event rows,
        # and Sim1Simulation._integrate_left_continuous stores the event-time
        # state before discrete event application.
        _assert_public_volume_matches_dense_simulation(process, dense)
        event_pairs = dense_event_pair_reference(process_name, process.time_axis.end)
        _assert_public_transform_preserves_event_cstar(process, event_pairs)

        # Orthogonal carrier-magnitude checks (B + D): pin ADF and feed correction
        # against simulator ground truth so a wrong carrier magnitude cannot cancel
        # in the c* round-trip below.
        sample_factors = _sample_compensation_factors(process, event_pairs)
        _assert_public_adf_matches_volume_oracle(process, dense, sample_factors)
        _assert_public_feed_correction_pre_sample(process, dense, sample_factors)
        _assert_public_feed_correction_species_ratio(process, dense)
        _assert_public_feed_correction_bolus_jumps(process, sample_factors)

        dense_cstar = _public_cstar_from_concentrations(process, dense)
        for column, name in enumerate(EXPECTED_REACTOR_COMPONENT_ORDER):
            process.reactor_medium.components[
                name
            ].c_star_concentration = fit_cstar_timeseries_from_values(
                name,
                dense_times,
                dense_cstar[:, column],
                source="dense_online_oracle_left_event",
            )

        fitted_cstar = np.column_stack(
            [
                np.asarray(
                    process.reactor_medium.components[
                        name
                    ].c_star_concentration.evaluate_many(jnp.asarray(dense_times)),
                    dtype=float,
                )
                for name in EXPECTED_REACTOR_COMPONENT_ORDER
            ]
        )
        np.testing.assert_allclose(
            fitted_cstar,
            dense_cstar,
            rtol=DENSE_FITTED_CSTAR_RTOL,
            atol=DENSE_FITTED_CSTAR_ATOL,
        )

        rhs = _direct_rhs(process)
        integrated_cstar = _integrate_direct_cstar(
            process,
            rhs,
            dense_times,
            fitted_cstar[0],
            rtol=DENSE_SOLVER_RTOL,
            atol=DENSE_SOLVER_ATOL,
        )
        np.testing.assert_allclose(
            integrated_cstar,
            fitted_cstar,
            rtol=DENSE_CSTAR_RTOL,
            atol=DENSE_CSTAR_ATOL,
        )

        recovered_concentrations = _public_backtransform_from_cstar(
            process,
            dense_times,
            integrated_cstar,
        )
        dense_concentrations = np.column_stack(
            [dense[name] for name in EXPECTED_REACTOR_COMPONENT_ORDER]
        )
        _print_component_errors(
            "dense c* raw concentration reconstruction errors",
            recovered_concentrations,
            dense_concentrations,
        )
        # The real-space check includes ODE reintegration error propagated through
        # the ADF/feed backtransform. Keep it sub-nanomolar/sub-ng-per-L absolute.
        np.testing.assert_allclose(
            recovered_concentrations,
            dense_concentrations,
            rtol=DENSE_CONCENTRATION_RTOL,
            atol=DENSE_CONCENTRATION_ATOL,
        )
