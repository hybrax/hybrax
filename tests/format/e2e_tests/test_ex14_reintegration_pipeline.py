from collections import defaultdict
from collections.abc import Iterable
import json
import os
import sys
from contextlib import contextmanager
from pathlib import Path

os.environ.setdefault("JAX_ENABLE_X64", "true")

import bp_format as bp  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402
from scipy.integrate import solve_ivp  # noqa: E402
from bp_format.serialization import (  # noqa: E402
    load_process_collection_json,
    save_process_collection_json,
)

FIXTURE_ROOT = Path(__file__).resolve().parent / "ex14_fixture"
SIMULATION_DIR = FIXTURE_ROOT / "00_simulation"
SIMULATION_DENSE_OUTPUT = SIMULATION_DIR / "simulation_dense_output.csv"
EVENTS_OUTPUT = SIMULATION_DIR / "events.csv"
CANONICAL_ARTIFACTS = ("simulation_dense_output.csv", "events.csv")
EXPECTED_PROCESS_IDS = {"ex14_run_1", "ex14_run_2"}
EXPECTED_REACTOR_COMPONENT_ORDER = (
    "biomass",
    "product_extracellular",
    "product_intracellular",
    "dead_cells",
    "glucose",
    "glutamine",
    "lactate",
    "ammonia",
)
EXPECTED_REACTOR_COMPONENTS = set(EXPECTED_REACTOR_COMPONENT_ORDER)
EXPECTED_CONCENTRATION_TIMES = [0.0, 24.0, 48.0, 72.0, 96.0]
EXPECTED_SAMPLE_EVENT_TIMES = [24.0, 48.0, 72.0, 96.0]
EXPECTED_EVENT_JUMP_TIMES = [24.0, 36.0, 48.0, 72.0, 96.0, 108.0]
EXPECTED_NONZERO_FEED_CORRECTION_COMPONENTS = {"glucose", "glutamine"}
EXPECTED_VOLUME_CHANGES = {"conti_feed", "base_feed", "sampling", "bolus_feed"}
EXPECTED_RATE_NAMES = {
    "q_X_active",
    "q_product_extracellular",
    "q_product_intracellular",
    "q_dead_cells",
    "q_glucose",
    "q_glutamine",
    "q_lactate",
    "q_ammonia",
}
CSTAR_SPLINE_SMOOTHING_S = 0.0
EVENT_TIME_ATOL = 1e-12
PSEUDOBATCH_INTEGRATOR_METHOD = "DOP853"
PSEUDOBATCH_INTEGRATOR_RTOL = 1e-7
PSEUDOBATCH_INTEGRATOR_ATOL = 1e-9
RHS_STATE_PERTURBATION = 1.0
EXPECTED_BOLUS_RIGHT_LIMIT_TIME = 36.0
EXPECTED_BOLUS_RIGHT_LIMIT_COMPONENT = "glucose"
SPARSE_DIAGNOSTIC_PLOTS_PER_PROCESS = 2
SPARSE_CONCENTRATION_PLOT_FIGSIZE = (10, 11)
SPARSE_RATES_TRANSFORM_PLOT_FIGSIZE = (9, 8)
SPARSE_DIAGNOSTIC_PLOT_DPI = 120
# Loose first-stage sparse bounds. They cover five offline samples, spline-
# derivative rate inference, and linear rate interpolation. Limits are intentionally
# wider than current observed errors so this is a sanity check, not a golden
# trajectory check. Dense observations should replace these with tight recovery
# checks in a later milestone. Values are (absolute_error, relative_error).
SPARSE_REINTEGRATION_ERROR_LIMITS = {
    "biomass": (20.0, 0.10),
    "product_extracellular": (8.0, 0.25),
    "product_intracellular": (2.0, 0.25),
    "dead_cells": (2.0, 0.08),
    "glucose": (2.0, 0.10),
    "glutamine": (1.0, 1.00),
    "lactate": (1.5, 0.08),
    "ammonia": (0.6, 0.08),
}


def _clear_ex14_fixture_modules() -> None:
    sys.modules.pop("load_utils", None)
    sys.modules.pop("ex14_simulation", None)


@contextmanager
def _isolated_ex14_fixture_imports():
    sys_path = list(sys.path)
    dont_write_bytecode = sys.dont_write_bytecode
    _clear_ex14_fixture_modules()
    sys.dont_write_bytecode = True
    sys.path.insert(0, str(FIXTURE_ROOT))
    sys.path.insert(0, str(SIMULATION_DIR))
    try:
        yield
    finally:
        sys.path[:] = sys_path
        sys.dont_write_bytecode = dont_write_bytecode
        _clear_ex14_fixture_modules()


def _regenerate_simulation_artifacts(output_dir: Path) -> Path:
    with _isolated_ex14_fixture_imports():
        from ex14_simulation import run_all_default

        run_all_default(output_dir=output_dir)
    return output_dir


def _parse_lab_like_collection():
    with _isolated_ex14_fixture_imports():
        from load_utils import parse_all_processes

        return parse_all_processes(
            dense_csv=SIMULATION_DENSE_OUTPUT,
            events_csv=EVENTS_OUTPUT,
            collection_name="ex14_lab_like_e2e",
        )


def _array_values(values):
    return [float(value) for value in values]


def _series_values(series):
    return _array_values(series.values)


def _series_times(series):
    return _array_values(series.times)


def _assert_optional_array_equal(actual, expected) -> None:
    if expected is None:
        assert actual is None
        return
    assert actual is not None
    np.testing.assert_array_equal(np.asarray(actual), np.asarray(expected))


def _assert_series_equal(actual, expected) -> None:
    for attr in (
        "times",
        "values",
        "jump_times",
        "breaks",
        "coeffs",
        "segment_start_piece_idx",
    ):
        _assert_optional_array_equal(getattr(actual, attr), getattr(expected, attr))
    assert actual.derived == expected.derived
    assert actual.continuity_side == expected.continuity_side
    assert actual.metadata == expected.metadata
    assert np.dtype(actual.dtype) == np.dtype(expected.dtype)


def _time_grid_from_volume_changes(
    volume_changes: Iterable[bp.VolumeChange],
) -> np.ndarray:
    return np.unique(
        np.concatenate(
            [
                np.asarray(volume_change.values.times, dtype=float)
                for volume_change in volume_changes
            ]
        )
    )


def _value_at_time(times: np.ndarray, values: np.ndarray, time: float) -> float:
    matches = np.flatnonzero(times == time)
    if len(matches) != 1:
        raise ValueError(f"Expected exactly one grid entry at time {time}.")
    return float(values[int(matches[0])])


def _as_timeseries(times: np.ndarray, values: np.ndarray, **kwargs) -> bp.TimeSeries:
    return bp.TimeSeries(
        times=jnp.asarray(times),
        values=jnp.asarray(values),
        **kwargs,
    )


def _populate_exact_pseudobatch_transform(process: bp.BioProcess) -> None:
    volume_changes = process.volume.volume_changes
    times = _time_grid_from_volume_changes(volume_changes.values())
    total_volume = np.full(len(times), process.volume.initial_volume, dtype=float)
    sample_deltas: defaultdict[float, float] = defaultdict(float)
    discrete_deltas: defaultdict[float, float] = defaultdict(float)

    for volume_change in volume_changes.values():
        change_times = np.asarray(volume_change.values.times, dtype=float)
        change_values = np.asarray(volume_change.values.values, dtype=float)
        assert len(change_times) == len(change_values)

        if volume_change.is_continuous:
            total_volume += np.interp(times, change_times, change_values)
            continue

        for time, value in zip(change_times, change_values, strict=True):
            total_volume[times > time] += value
            discrete_deltas[float(time)] += float(value)
            if isinstance(volume_change, bp.SampleVolumeChange):
                sample_deltas[float(time)] += float(value)

    jump_times = np.asarray(sorted(discrete_deltas), dtype=float)
    jump_times = jump_times[(times[0] < jump_times) & (jump_times < times[-1])]
    process.volume.total_volume = _as_timeseries(
        times,
        total_volume,
        jump_times=jnp.asarray(jump_times),
    )

    sample_compensation = np.ones(len(times), dtype=float)
    for time, delta in sorted(sample_deltas.items()):
        pre_sample_volume = _value_at_time(times, total_volume, time)
        post_sample_volume = pre_sample_volume + delta
        sample_compensation[times > time] *= pre_sample_volume / post_sample_volume

    accumulated_feeds = {}
    for name, volume_change in volume_changes.items():
        if not isinstance(volume_change, bp.FeedVolumeChange):
            continue
        change_times = np.asarray(volume_change.values.times, dtype=float)
        change_values = np.asarray(volume_change.values.values, dtype=float)
        if volume_change.is_continuous:
            accumulated_feed = np.interp(times, change_times, change_values)
        else:
            accumulated_feed = np.zeros(len(times), dtype=float)
            for time, value in zip(change_times, change_values, strict=True):
                accumulated_feed[times > time] += value
        accumulated_feeds[name] = _as_timeseries(times, accumulated_feed)

    adf = total_volume / process.volume.initial_volume * sample_compensation
    feed_corrections = {}
    for species_name in process.reactor_medium.components:
        feed_correction = np.zeros(len(times), dtype=float)

        for name, volume_change in volume_changes.items():
            if not isinstance(volume_change, bp.FeedVolumeChange):
                continue

            feed_concentration = volume_change.feed_medium.components[
                species_name
            ].concentration.value
            if feed_concentration == 0.0:
                continue

            change_times = np.asarray(volume_change.values.times, dtype=float)
            change_values = np.asarray(volume_change.values.values, dtype=float)

            if volume_change.is_continuous:
                cumulative_feed = np.asarray(
                    accumulated_feeds[name].values, dtype=float
                )
                feed_volume_added = np.diff(cumulative_feed, prepend=cumulative_feed[0])
                feed_correction += np.cumsum(
                    sample_compensation
                    / process.volume.initial_volume
                    * feed_volume_added
                    * feed_concentration
                )
                continue

            for time, feed_volume_added in zip(
                change_times,
                change_values,
                strict=True,
            ):
                sample_compensation_at_bolus = _value_at_time(
                    times,
                    sample_compensation,
                    float(time),
                )
                sample_delta = sample_deltas[float(time)]
                if sample_delta != 0.0:
                    pre_sample_volume = _value_at_time(times, total_volume, float(time))
                    post_sample_volume = pre_sample_volume + sample_delta
                    sample_compensation_at_bolus *= (
                        pre_sample_volume / post_sample_volume
                    )
                feed_correction[times > time] += (
                    sample_compensation_at_bolus
                    * feed_volume_added
                    * feed_concentration
                    / process.volume.initial_volume
                )

        feed_corrections[species_name] = _as_timeseries(times, feed_correction)

    process.pseudobatch_transform = bp.PseudobatchTransform(
        adf=_as_timeseries(times, adf),
        feed_corrections=feed_corrections,
        sample_compensation=_as_timeseries(times, sample_compensation),
        accumulated_feeds=accumulated_feeds,
    )


def _assert_finite_positive(series: bp.TimeSeries) -> None:
    values = np.asarray(series.values, dtype=float)
    assert np.all(np.isfinite(values))
    assert np.all(values > 0.0)


def _assert_finite_nonnegative(series: bp.TimeSeries) -> None:
    values = np.asarray(series.values, dtype=float)
    assert np.all(np.isfinite(values))
    assert np.all(values >= 0.0)


def _series_values_at_times(series: bp.TimeSeries, times: np.ndarray) -> np.ndarray:
    if series.times is None or series.values is None:
        raise ValueError("Stored pseudobatch transform carriers need raw samples.")
    series_times = np.asarray(series.times, dtype=float)
    series_values = np.asarray(series.values, dtype=float)
    return np.asarray(
        [
            _value_at_time(series_times, series_values, float(time))
            for time in np.asarray(times, dtype=float)
        ],
        dtype=float,
    )


def _cstar_transform_metadata(component_name: str) -> dict[str, str]:
    return {
        "name": "pseudo_batch",
        "component": component_name,
        "source": "stored_pseudobatch_transform",
    }


def _cstar_values_for_component(
    process: bp.BioProcess,
    component: bp.ReactorMediumComponent,
) -> tuple[np.ndarray, np.ndarray]:
    transform = process.pseudobatch_transform
    if transform is None:
        raise ValueError("process.pseudobatch_transform is required.")
    if component.name not in transform.feed_corrections:
        raise KeyError(component.name)
    if not isinstance(component.concentration, bp.TimeSeries):
        raise TypeError(f"{component.name} concentration must be a TimeSeries.")

    times = np.asarray(component.concentration.times, dtype=float)
    raw_values = np.asarray(component.concentration.values, dtype=float)
    adf = _series_values_at_times(transform.adf, times)
    feed_correction = _series_values_at_times(
        transform.feed_corrections[component.name],
        times,
    )
    return times, raw_values * adf - feed_correction


def _fit_cstar_timeseries(
    process: bp.BioProcess,
    component: bp.ReactorMediumComponent,
) -> bp.TimeSeries:
    times, cstar_values = _cstar_values_for_component(process, component)
    fitted = bp.splines.fit_timeseries_spline(
        bp.TimeSeries(
            times=jnp.asarray(times),
            values=jnp.asarray(cstar_values),
        ),
        smoothing_s=CSTAR_SPLINE_SMOOTHING_S,
    )
    metadata = dict(fitted.metadata or {})
    metadata["transform"] = _cstar_transform_metadata(component.name)
    return bp.TimeSeries(
        times=fitted.times,
        values=fitted.values,
        jump_times=fitted.jump_times,
        breaks=fitted.breaks,
        coeffs=fitted.coeffs,
        segment_start_piece_idx=fitted.segment_start_piece_idx,
        continuity_side=fitted.continuity_side,
        metadata=metadata,
        dtype=fitted.dtype,
    )


def _populate_cstar_splines(process: bp.BioProcess) -> None:
    for component in process.reactor_medium.components.values():
        component.c_star_concentration = _fit_cstar_timeseries(process, component)


def _assert_cstar_splines_sane(process: bp.BioProcess) -> None:
    for component in process.reactor_medium.components.values():
        cstar = component.c_star_concentration
        assert isinstance(cstar, bp.TimeSeries)
        times, expected_values = _cstar_values_for_component(process, component)
        np.testing.assert_array_equal(np.asarray(cstar.times), times)
        np.testing.assert_array_equal(np.asarray(cstar.values), expected_values)
        assert cstar.breaks is not None
        assert cstar.coeffs is not None
        assert cstar.segment_start_piece_idx is not None
        assert cstar.metadata is not None
        assert cstar.metadata["smoothing_s"] == CSTAR_SPLINE_SMOOTHING_S
        assert cstar.metadata["transform"] == _cstar_transform_metadata(component.name)
        evaluated = np.asarray(cstar.evaluate_many(jnp.asarray(times)), dtype=float)
        assert np.all(np.isfinite(evaluated))
        np.testing.assert_allclose(evaluated, expected_values, rtol=1e-12, atol=1e-12)


def _assert_ex14_biological_ode_contract(process: bp.BioProcess) -> None:
    ode = process.biological_ode
    assert ode is not None
    assert ode.algebraic == {"X_active": "biomass - product_intracellular"}
    assert set(ode.rates) == EXPECTED_RATE_NAMES
    assert ode.derivatives["biomass"] == (
        "q_X_active * X_active + q_product_intracellular * X_active"
    )
    assert ode.derivatives["product_extracellular"] == (
        "q_product_extracellular * X_active"
    )
    assert ode.derivatives["product_intracellular"] == (
        "q_product_intracellular * X_active"
    )
    for name in ("dead_cells", "glucose", "glutamine", "lactate", "ammonia"):
        assert ode.derivatives[name] == f"q_{name} * X_active"


def _evaluate_cstar_states_and_derivatives(
    process: bp.BioProcess,
    times: np.ndarray,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    cstar_states = {}
    cstar_derivatives = {}
    for name, component in process.reactor_medium.components.items():
        cstar = component.c_star_concentration
        if not isinstance(cstar, bp.TimeSeries):
            raise TypeError(f"{name} c* concentration must be a TimeSeries.")
        cstar_states[name] = np.asarray(
            cstar.evaluate_many(jnp.asarray(times)),
            dtype=float,
        )
        cstar_derivatives[name] = np.asarray(
            cstar.deriv().evaluate_many(jnp.asarray(times)),
            dtype=float,
        )
    return cstar_states, cstar_derivatives


def _evaluate_reconstructed_reactor_states(
    process: bp.BioProcess,
    times: np.ndarray,
    cstar_states: dict[str, np.ndarray],
) -> dict[str, np.ndarray]:
    transform = process.pseudobatch_transform
    if transform is None:
        raise ValueError("process.pseudobatch_transform is required.")
    adf = _series_values_at_times(transform.adf, times)
    states = {}
    for name, cstar in cstar_states.items():
        feed_correction = _series_values_at_times(
            transform.feed_corrections[name],
            times,
        )
        states[name] = (cstar + feed_correction) / adf
    return states


def _evaluate_rate_inference_terms(
    process: bp.BioProcess,
    times: np.ndarray,
) -> dict[str, dict[str, np.ndarray]]:
    transform = process.pseudobatch_transform
    if transform is None:
        raise ValueError("process.pseudobatch_transform is required.")
    assert process.volume.total_volume is not None
    return {
        "volume": {
            "total_volume": _series_values_at_times(process.volume.total_volume, times)
        },
        "transform": {"adf": _series_values_at_times(transform.adf, times)},
        "feed_corrections": {
            name: _series_values_at_times(series, times)
            for name, series in transform.feed_corrections.items()
        },
    }


def _infer_ex14_rates_from_cstar_splines(
    process: bp.BioProcess,
    times: np.ndarray,
) -> tuple[dict[str, np.ndarray], dict[str, dict[str, np.ndarray]]]:
    _assert_ex14_biological_ode_contract(process)
    transform = process.pseudobatch_transform
    if transform is None:
        raise ValueError("process.pseudobatch_transform is required.")

    cstar_states, cstar_derivatives = _evaluate_cstar_states_and_derivatives(
        process,
        times,
    )
    states = _evaluate_reconstructed_reactor_states(process, times, cstar_states)
    adf = _series_values_at_times(transform.adf, times)
    biological_derivatives = {
        name: derivative / adf for name, derivative in cstar_derivatives.items()
    }
    active_biomass = states["biomass"] - states["product_intracellular"]
    q_product_intracellular = (
        biological_derivatives["product_intracellular"] / active_biomass
    )

    rates = {
        "q_product_extracellular": (
            biological_derivatives["product_extracellular"] / active_biomass
        ),
        "q_product_intracellular": q_product_intracellular,
        "q_dead_cells": biological_derivatives["dead_cells"] / active_biomass,
        "q_glucose": biological_derivatives["glucose"] / active_biomass,
        "q_glutamine": biological_derivatives["glutamine"] / active_biomass,
        "q_lactate": biological_derivatives["lactate"] / active_biomass,
        "q_ammonia": biological_derivatives["ammonia"] / active_biomass,
        "q_X_active": (
            biological_derivatives["biomass"] / active_biomass - q_product_intracellular
        ),
    }
    diagnostics = _evaluate_rate_inference_terms(process, times)
    diagnostics["cstar_states"] = cstar_states
    diagnostics["cstar_derivatives"] = cstar_derivatives
    diagnostics["reconstructed_states"] = states
    diagnostics["biological_derivatives"] = biological_derivatives
    diagnostics["algebraic"] = {"X_active": active_biomass}
    return rates, diagnostics


def _assert_inferred_rates_finite(
    rates: dict[str, np.ndarray],
    diagnostics: dict[str, dict[str, np.ndarray]],
) -> None:
    assert set(rates) == EXPECTED_RATE_NAMES
    for values in rates.values():
        assert values.shape == (len(EXPECTED_CONCENTRATION_TIMES),)
        assert np.all(np.isfinite(values))
    for group in diagnostics.values():
        for values in group.values():
            assert values.shape == (len(EXPECTED_CONCENTRATION_TIMES),)
            assert np.all(np.isfinite(values))
    assert np.all(diagnostics["volume"]["total_volume"] > 0.0)
    assert np.all(diagnostics["transform"]["adf"] > 0.0)
    assert np.all(diagnostics["algebraic"]["X_active"] > 0.0)


def _assert_rates_match_biological_derivatives(
    rates: dict[str, np.ndarray],
    diagnostics: dict[str, dict[str, np.ndarray]],
) -> None:
    adf = diagnostics["transform"]["adf"]
    active_biomass = diagnostics["algebraic"]["X_active"]
    biological_derivatives = diagnostics["biological_derivatives"]
    cstar_derivatives = diagnostics["cstar_derivatives"]

    for name in EXPECTED_REACTOR_COMPONENTS:
        np.testing.assert_allclose(
            biological_derivatives[name] * adf,
            cstar_derivatives[name],
            rtol=1e-12,
            atol=1e-12,
        )
    np.testing.assert_allclose(
        (rates["q_X_active"] + rates["q_product_intracellular"]) * active_biomass,
        biological_derivatives["biomass"],
        rtol=1e-12,
        atol=1e-12,
    )
    for name in EXPECTED_REACTOR_COMPONENTS - {"biomass"}:
        np.testing.assert_allclose(
            rates[f"q_{name}"] * active_biomass,
            biological_derivatives[name],
            rtol=1e-12,
            atol=1e-12,
        )


def _assert_state_transform_identities(
    diagnostics: dict[str, dict[str, np.ndarray]],
) -> None:
    adf = diagnostics["transform"]["adf"]
    cstar_states = diagnostics["cstar_states"]
    feed_corrections = diagnostics["feed_corrections"]
    reconstructed_states = diagnostics["reconstructed_states"]
    np.testing.assert_allclose(
        diagnostics["algebraic"]["X_active"],
        reconstructed_states["biomass"] - reconstructed_states["product_intracellular"],
        rtol=1e-12,
        atol=1e-12,
    )
    for name in EXPECTED_REACTOR_COMPONENTS:
        np.testing.assert_allclose(
            reconstructed_states[name] * adf - feed_corrections[name],
            cstar_states[name],
            rtol=1e-12,
            atol=1e-12,
        )


def _event_jump_times(process: bp.BioProcess) -> np.ndarray:
    assert process.volume.total_volume is not None
    return np.asarray(process.volume.total_volume.jump_times, dtype=float)


def _is_event_jump_time(time: float, jump_times: np.ndarray) -> bool:
    return bool(np.any(np.isclose(jump_times, time, rtol=0.0, atol=EVENT_TIME_ATOL)))


def _discrete_volume_delta_at(process: bp.BioProcess, time: float) -> float:
    delta = 0.0
    for volume_change in process.volume.volume_changes.values():
        if volume_change.is_continuous:
            continue
        change_times = np.asarray(volume_change.values.times, dtype=float)
        change_values = np.asarray(volume_change.values.values, dtype=float)
        matches = np.flatnonzero(
            np.isclose(change_times, time, rtol=0.0, atol=EVENT_TIME_ATOL)
        )
        if len(matches):
            delta += float(change_values[int(matches[0])])
    return delta


def _sample_volume_delta_at(process: bp.BioProcess, time: float) -> float:
    delta = 0.0
    for volume_change in process.volume.volume_changes.values():
        if not isinstance(volume_change, bp.SampleVolumeChange):
            continue
        change_times = np.asarray(volume_change.values.times, dtype=float)
        change_values = np.asarray(volume_change.values.values, dtype=float)
        matches = np.flatnonzero(
            np.isclose(change_times, time, rtol=0.0, atol=EVENT_TIME_ATOL)
        )
        if len(matches):
            delta += float(change_values[int(matches[0])])
    return delta


def _sample_compensation_after_sample_event(
    process: bp.BioProcess,
    time: float,
) -> float:
    transform = process.pseudobatch_transform
    assert transform is not None
    assert transform.sample_compensation is not None
    assert process.volume.total_volume is not None
    sample_compensation = _value_at_time(
        np.asarray(transform.sample_compensation.times, dtype=float),
        np.asarray(transform.sample_compensation.values, dtype=float),
        time,
    )
    sample_delta = _sample_volume_delta_at(process, time)
    if sample_delta == 0.0:
        return sample_compensation
    pre_sample_volume = _value_at_time(
        np.asarray(process.volume.total_volume.times, dtype=float),
        np.asarray(process.volume.total_volume.values, dtype=float),
        time,
    )
    post_sample_volume = pre_sample_volume + sample_delta
    return sample_compensation * pre_sample_volume / post_sample_volume


def _adf_jump_delta_at(process: bp.BioProcess, time: float) -> float:
    transform = process.pseudobatch_transform
    assert transform is not None
    assert process.volume.total_volume is not None
    total_volume_times = np.asarray(process.volume.total_volume.times, dtype=float)
    total_volume_values = np.asarray(process.volume.total_volume.values, dtype=float)
    adf_times = np.asarray(transform.adf.times, dtype=float)
    adf_values = np.asarray(transform.adf.values, dtype=float)
    pre_adf = _value_at_time(adf_times, adf_values, time)
    volume_delta = _discrete_volume_delta_at(process, time)
    sample_delta = _sample_volume_delta_at(process, time)
    if volume_delta == 0.0 and sample_delta == 0.0:
        return 0.0
    pre_total_volume = _value_at_time(total_volume_times, total_volume_values, time)
    post_total_volume = pre_total_volume + volume_delta
    sample_compensation = _sample_compensation_after_sample_event(process, time)
    post_adf = post_total_volume / process.volume.initial_volume * sample_compensation
    return post_adf - pre_adf


def _feed_correction_jump_delta_at(
    process: bp.BioProcess,
    component_name: str,
    time: float,
) -> float:
    sample_compensation = _sample_compensation_after_sample_event(process, time)
    delta = 0.0
    for volume_change in process.volume.volume_changes.values():
        if volume_change.is_continuous:
            continue
        if not isinstance(volume_change, bp.FeedVolumeChange):
            continue
        change_times = np.asarray(volume_change.values.times, dtype=float)
        change_values = np.asarray(volume_change.values.values, dtype=float)
        matches = np.flatnonzero(
            np.isclose(change_times, time, rtol=0.0, atol=EVENT_TIME_ATOL)
        )
        if not len(matches):
            continue
        feed_concentration = volume_change.feed_medium.components[
            component_name
        ].concentration.value
        delta += (
            sample_compensation
            * float(change_values[int(matches[0])])
            * feed_concentration
            / process.volume.initial_volume
        )
    return delta


def _event_aware_series_value(
    series: bp.TimeSeries,
    time: float,
    jump_times: np.ndarray,
    jump_delta_at,
    *,
    right_of_jump: bool = False,
) -> float:
    if series.times is None or series.values is None:
        raise ValueError("Stored pseudobatch transform carriers need raw samples.")
    times = np.asarray(series.times, dtype=float)
    values = np.asarray(series.values, dtype=float)
    if time < times[0] or time > times[-1]:
        raise ValueError(f"Time {time} is outside the transform carrier domain.")

    exact_matches = np.flatnonzero(
        np.isclose(times, time, rtol=0.0, atol=EVENT_TIME_ATOL)
    )
    if len(exact_matches):
        index = int(exact_matches[0])
        value = float(values[index])
        if right_of_jump and _is_event_jump_time(time, jump_times):
            value += jump_delta_at(time)
        return value

    left_index = int(np.searchsorted(times, time, side="right") - 1)
    right_index = left_index + 1
    left_time = float(times[left_index])
    left_value = float(values[left_index])
    if _is_event_jump_time(left_time, jump_times):
        left_value += jump_delta_at(left_time)
    return float(
        np.interp(
            time, [left_time, times[right_index]], [left_value, values[right_index]]
        )
    )


def _event_aware_adf_value(
    process: bp.BioProcess,
    time: float,
    jump_times: np.ndarray,
    *,
    right_of_jump: bool = False,
) -> float:
    transform = process.pseudobatch_transform
    assert transform is not None
    return _event_aware_series_value(
        transform.adf,
        time,
        jump_times,
        lambda jump_time: _adf_jump_delta_at(process, jump_time),
        right_of_jump=right_of_jump,
    )


def _event_aware_feed_correction_value(
    process: bp.BioProcess,
    component_name: str,
    time: float,
    jump_times: np.ndarray,
    *,
    right_of_jump: bool = False,
) -> float:
    transform = process.pseudobatch_transform
    assert transform is not None
    return _event_aware_series_value(
        transform.feed_corrections[component_name],
        time,
        jump_times,
        lambda jump_time: _feed_correction_jump_delta_at(
            process,
            component_name,
            jump_time,
        ),
        right_of_jump=right_of_jump,
    )


def _interpolated_rate_values(
    rates: dict[str, np.ndarray],
    rate_times: np.ndarray,
    time: float,
) -> dict[str, float]:
    if time < rate_times[0] or time > rate_times[-1]:
        raise ValueError(f"Time {time} is outside the inferred-rate domain.")
    return {
        name: float(np.interp(time, rate_times, values))
        for name, values in rates.items()
    }


def _ex14_biological_derivatives(
    states: dict[str, float],
    rates: dict[str, float],
) -> dict[str, float]:
    active_biomass = states["biomass"] - states["product_intracellular"]
    return {
        "biomass": (rates["q_X_active"] + rates["q_product_intracellular"])
        * active_biomass,
        "product_extracellular": rates["q_product_extracellular"] * active_biomass,
        "product_intracellular": rates["q_product_intracellular"] * active_biomass,
        "dead_cells": rates["q_dead_cells"] * active_biomass,
        "glucose": rates["q_glucose"] * active_biomass,
        "glutamine": rates["q_glutamine"] * active_biomass,
        "lactate": rates["q_lactate"] * active_biomass,
        "ammonia": rates["q_ammonia"] * active_biomass,
    }


def _pseudobatch_rhs_values(
    process: bp.BioProcess,
    rate_times: np.ndarray,
    rates: dict[str, np.ndarray],
    time: float,
    cstar_state: np.ndarray,
    *,
    right_of_jump: bool = False,
) -> np.ndarray:
    transform = process.pseudobatch_transform
    if transform is None:
        raise ValueError("process.pseudobatch_transform is required.")
    if set(process.reactor_medium.components) != EXPECTED_REACTOR_COMPONENTS:
        raise ValueError("Unexpected ex14 reactor component set.")

    jump_times = _event_jump_times(process)
    adf = _event_aware_adf_value(
        process,
        time,
        jump_times,
        right_of_jump=right_of_jump,
    )
    states = {}
    for index, name in enumerate(EXPECTED_REACTOR_COMPONENT_ORDER):
        feed_correction = _event_aware_feed_correction_value(
            process,
            name,
            time,
            jump_times,
            right_of_jump=right_of_jump,
        )
        states[name] = (float(cstar_state[index]) + feed_correction) / adf

    q_values = _interpolated_rate_values(rates, rate_times, time)
    biological_derivatives = _ex14_biological_derivatives(states, q_values)
    return np.asarray(
        [
            adf * biological_derivatives[name]
            for name in EXPECTED_REACTOR_COMPONENT_ORDER
        ],
        dtype=float,
    )


def _initial_cstar_state(diagnostics: dict[str, dict[str, np.ndarray]]) -> np.ndarray:
    return np.asarray(
        [
            diagnostics["cstar_states"][name][0]
            for name in EXPECTED_REACTOR_COMPONENT_ORDER
        ],
        dtype=float,
    )


def _assert_transform_right_limits_are_event_local(process: bp.BioProcess) -> None:
    transform = process.pseudobatch_transform
    assert transform is not None
    jump_times = _event_jump_times(process)
    bolus_time = EXPECTED_BOLUS_RIGHT_LIMIT_TIME
    component_name = EXPECTED_BOLUS_RIGHT_LIMIT_COMPONENT

    left_adf = _event_aware_adf_value(process, bolus_time, jump_times)
    right_adf = _event_aware_adf_value(
        process,
        bolus_time,
        jump_times,
        right_of_jump=True,
    )
    np.testing.assert_allclose(
        right_adf,
        left_adf + _adf_jump_delta_at(process, bolus_time),
        rtol=1e-12,
        atol=1e-12,
    )

    left_feed_correction = _event_aware_feed_correction_value(
        process,
        component_name,
        bolus_time,
        jump_times,
    )
    right_feed_correction = _event_aware_feed_correction_value(
        process,
        component_name,
        bolus_time,
        jump_times,
        right_of_jump=True,
    )
    np.testing.assert_allclose(
        right_feed_correction,
        left_feed_correction
        + _feed_correction_jump_delta_at(process, component_name, bolus_time),
        rtol=1e-12,
        atol=1e-12,
    )
    stored_times = np.asarray(
        transform.feed_corrections[component_name].times,
        dtype=float,
    )
    stored_values = np.asarray(
        transform.feed_corrections[component_name].values,
        dtype=float,
    )
    bolus_index = int(np.flatnonzero(stored_times == bolus_time)[0])
    next_stored_value = stored_values[bolus_index + 1]
    continuous_component_feed = any(
        isinstance(volume_change, bp.FeedVolumeChange)
        and volume_change.is_continuous
        and volume_change.feed_medium.components[component_name].concentration.value
        != 0.0
        and np.any(np.diff(np.asarray(volume_change.values.values, dtype=float)) > 0.0)
        for volume_change in process.volume.volume_changes.values()
    )
    assert continuous_component_feed
    assert not np.isclose(
        right_feed_correction,
        next_stored_value,
        rtol=1e-12,
        atol=1e-12,
    )


def _assert_pseudobatch_rhs_initial_algebra(
    process: bp.BioProcess,
    rate_times: np.ndarray,
    rates: dict[str, np.ndarray],
    diagnostics: dict[str, dict[str, np.ndarray]],
) -> None:
    time = float(rate_times[0])
    y0 = _initial_cstar_state(diagnostics)
    rhs0 = _pseudobatch_rhs_values(process, rate_times, rates, time, y0)
    expected = np.asarray(
        [
            diagnostics["cstar_derivatives"][name][0]
            for name in EXPECTED_REACTOR_COMPONENT_ORDER
        ],
        dtype=float,
    )
    np.testing.assert_allclose(rhs0, expected, rtol=1e-12, atol=1e-12)

    perturbed = y0.copy()
    perturbed[EXPECTED_REACTOR_COMPONENT_ORDER.index("biomass")] += (
        RHS_STATE_PERTURBATION
    )
    perturbed_rhs = _pseudobatch_rhs_values(
        process,
        rate_times,
        rates,
        time,
        perturbed,
    )
    assert not np.allclose(perturbed_rhs, rhs0, rtol=1e-12, atol=1e-12)


def _pseudobatch_integration_breakpoints(
    process: bp.BioProcess,
    output_times: np.ndarray,
) -> np.ndarray:
    jump_times = _event_jump_times(process)
    domain_start = float(output_times[0])
    domain_end = float(output_times[-1])
    jumps_in_domain = jump_times[
        (domain_start < jump_times) & (jump_times < domain_end)
    ]
    return np.unique(np.concatenate([output_times, jumps_in_domain]))


def _integrate_pseudobatch_cstar_states(
    process: bp.BioProcess,
    rate_times: np.ndarray,
    rates: dict[str, np.ndarray],
    diagnostics: dict[str, dict[str, np.ndarray]],
    output_times: np.ndarray,
) -> dict[str, np.ndarray]:
    output_time_set = set(float(time) for time in output_times)
    breakpoints = _pseudobatch_integration_breakpoints(process, output_times)
    jump_times = _event_jump_times(process)
    state = _initial_cstar_state(diagnostics)
    outputs = {float(output_times[0]): state.copy()}

    for start, end in zip(breakpoints[:-1], breakpoints[1:], strict=True):
        start = float(start)
        end = float(end)

        def rhs(time, cstar_state):
            return _pseudobatch_rhs_values(
                process,
                rate_times,
                rates,
                float(time),
                np.asarray(cstar_state, dtype=float),
                right_of_jump=(
                    _is_event_jump_time(start, jump_times)
                    and np.isclose(float(time), start, rtol=0.0, atol=EVENT_TIME_ATOL)
                ),
            )

        sol = solve_ivp(
            rhs,
            (start, end),
            state,
            method=PSEUDOBATCH_INTEGRATOR_METHOD,
            rtol=PSEUDOBATCH_INTEGRATOR_RTOL,
            atol=PSEUDOBATCH_INTEGRATOR_ATOL,
            t_eval=[end],
        )
        if not sol.success:
            raise RuntimeError(sol.message)
        state = np.asarray(sol.y[:, -1], dtype=float)
        if end in output_time_set:
            outputs[end] = state.copy()

    return {
        name: np.asarray(
            [outputs[float(time)][index] for time in output_times],
            dtype=float,
        )
        for index, name in enumerate(EXPECTED_REACTOR_COMPONENT_ORDER)
    }


def _assert_reintegrated_cstar_states_finite(
    reintegrated_cstar_states: dict[str, np.ndarray],
    diagnostics: dict[str, dict[str, np.ndarray]],
    output_times: np.ndarray,
) -> None:
    assert tuple(reintegrated_cstar_states) == EXPECTED_REACTOR_COMPONENT_ORDER
    for name, values in reintegrated_cstar_states.items():
        assert values.shape == (len(output_times),)
        assert np.all(np.isfinite(values))
        np.testing.assert_allclose(
            values[0],
            diagnostics["cstar_states"][name][0],
            rtol=1e-12,
            atol=1e-12,
        )


def _observed_sparse_reactor_concentrations(
    process: bp.BioProcess,
    times: np.ndarray,
) -> dict[str, np.ndarray]:
    observed = {}
    for name in EXPECTED_REACTOR_COMPONENT_ORDER:
        concentration = process.reactor_medium.components[name].concentration
        if not isinstance(concentration, bp.TimeSeries):
            raise TypeError(f"{name} concentration must be a TimeSeries.")
        observed[name] = _series_values_at_times(concentration, times)
    return observed


def _sparse_reintegration_errors(
    recovered_concentrations: dict[str, np.ndarray],
    observed_concentrations: dict[str, np.ndarray],
) -> dict[str, dict[str, np.ndarray]]:
    errors = {}
    for name in EXPECTED_REACTOR_COMPONENT_ORDER:
        absolute = np.abs(
            recovered_concentrations[name] - observed_concentrations[name]
        )
        errors[name] = {
            "absolute": absolute,
            "relative": absolute
            / np.maximum(np.abs(observed_concentrations[name]), 1e-12),
        }
    return errors


def _assert_sparse_real_space_reintegration_sane(
    recovered_concentrations: dict[str, np.ndarray],
    observed_concentrations: dict[str, np.ndarray],
    output_times: np.ndarray,
) -> None:
    assert tuple(recovered_concentrations) == EXPECTED_REACTOR_COMPONENT_ORDER
    assert tuple(observed_concentrations) == EXPECTED_REACTOR_COMPONENT_ORDER
    assert set(SPARSE_REINTEGRATION_ERROR_LIMITS) == EXPECTED_REACTOR_COMPONENTS

    errors = _sparse_reintegration_errors(
        recovered_concentrations,
        observed_concentrations,
    )
    for name in EXPECTED_REACTOR_COMPONENT_ORDER:
        recovered = recovered_concentrations[name]
        observed = observed_concentrations[name]
        assert recovered.shape == (len(output_times),)
        assert observed.shape == (len(output_times),)
        assert np.all(np.isfinite(recovered))
        assert np.all(np.isfinite(observed))
        np.testing.assert_allclose(recovered[0], observed[0], rtol=1e-12, atol=1e-12)

        max_abs_error, max_rel_error = SPARSE_REINTEGRATION_ERROR_LIMITS[name]
        assert float(np.max(errors[name]["absolute"])) <= max_abs_error
        assert float(np.max(errors[name]["relative"])) <= max_rel_error


def _float_list(values: np.ndarray) -> list[float]:
    return [float(value) for value in values]


def _sparse_reintegration_metrics(
    process: bp.BioProcess,
    times: np.ndarray,
    rates: dict[str, np.ndarray],
    diagnostics: dict[str, dict[str, np.ndarray]],
    recovered_concentrations: dict[str, np.ndarray],
    observed_concentrations: dict[str, np.ndarray],
) -> dict:
    errors = _sparse_reintegration_errors(
        recovered_concentrations,
        observed_concentrations,
    )
    return {
        "process": process.metadata.name,
        "times_h": _float_list(times),
        "components": {
            name: {
                "observed_concentration": _float_list(observed_concentrations[name]),
                "recovered_concentration": _float_list(recovered_concentrations[name]),
                "absolute_error": _float_list(errors[name]["absolute"]),
                "relative_error": _float_list(errors[name]["relative"]),
                "max_absolute_error": float(np.max(errors[name]["absolute"])),
                "max_relative_error": float(np.max(errors[name]["relative"])),
                "absolute_error_limit": SPARSE_REINTEGRATION_ERROR_LIMITS[name][0],
                "relative_error_limit": SPARSE_REINTEGRATION_ERROR_LIMITS[name][1],
            }
            for name in EXPECTED_REACTOR_COMPONENT_ORDER
        },
        "rates": {
            name: {
                "values": _float_list(values),
                "min": float(np.min(values)),
                "max": float(np.max(values)),
            }
            for name, values in sorted(rates.items())
        },
        "transform": {
            "total_volume": _float_list(diagnostics["volume"]["total_volume"]),
            "adf": _float_list(diagnostics["transform"]["adf"]),
            "feed_corrections": {
                name: _float_list(diagnostics["feed_corrections"][name])
                for name in EXPECTED_REACTOR_COMPONENT_ORDER
            },
        },
    }


def _write_sparse_reintegration_metrics_json(
    output_dir: Path,
    metrics: dict[str, dict],
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "sparse_reintegration_metrics.json"
    path.write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n")
    return path


def _plotting_pyplot(output_dir: Path):
    os.environ.setdefault("MPLCONFIGDIR", str(output_dir.parent / "matplotlib-cache"))
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def _write_sparse_reintegration_diagnostic_plots(
    output_dir: Path,
    process_name: str,
    times: np.ndarray,
    rates: dict[str, np.ndarray],
    diagnostics: dict[str, dict[str, np.ndarray]],
    recovered_concentrations: dict[str, np.ndarray],
    observed_concentrations: dict[str, np.ndarray],
) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    plt = _plotting_pyplot(output_dir)
    paths = []

    fig, axes = plt.subplots(
        4,
        2,
        figsize=SPARSE_CONCENTRATION_PLOT_FIGSIZE,
        sharex=True,
    )
    for ax, name in zip(axes.ravel(), EXPECTED_REACTOR_COMPONENT_ORDER, strict=True):
        ax.plot(times, recovered_concentrations[name], marker="o", label="recovered")
        ax.scatter(
            times,
            observed_concentrations[name],
            color="black",
            label="observed",
        )
        ax.set_title(name)
    axes[-1, 0].set_xlabel("time [h]")
    axes[-1, 1].set_xlabel("time [h]")
    axes[0, 0].legend(loc="best")
    fig.tight_layout()
    path = output_dir / f"{process_name}_sparse_reintegration.png"
    fig.savefig(path, dpi=SPARSE_DIAGNOSTIC_PLOT_DPI)
    plt.close(fig)
    paths.append(path)

    fig, axes = plt.subplots(
        3,
        1,
        figsize=SPARSE_RATES_TRANSFORM_PLOT_FIGSIZE,
        sharex=True,
    )
    for name, values in sorted(rates.items()):
        axes[0].plot(times, values, marker="o", label=name)
    axes[0].set_ylabel("rate")
    axes[0].legend(loc="center left", bbox_to_anchor=(1.0, 0.5), fontsize="small")
    axes[1].plot(times, diagnostics["volume"]["total_volume"], marker="o")
    axes[1].set_ylabel("total volume")
    axes[2].plot(times, diagnostics["transform"]["adf"], marker="o", label="ADF")
    for name in sorted(EXPECTED_NONZERO_FEED_CORRECTION_COMPONENTS):
        axes[2].plot(
            times,
            diagnostics["feed_corrections"][name],
            marker="o",
            label=f"{name} feed correction",
        )
    axes[2].set_xlabel("time [h]")
    axes[2].legend(loc="best", fontsize="small")
    fig.tight_layout()
    path = output_dir / f"{process_name}_rates_and_transform.png"
    fig.savefig(path, dpi=SPARSE_DIAGNOSTIC_PLOT_DPI)
    plt.close(fig)
    paths.append(path)

    return paths


def _assert_finite_metric_series(values: list[float], expected_length: int) -> None:
    assert len(values) == expected_length
    assert np.all(np.isfinite(np.asarray(values, dtype=float)))


def _assert_sparse_reintegration_outputs_written(
    metrics_path: Path,
    plot_paths: list[Path],
    process_count: int,
) -> None:
    assert metrics_path.exists()
    assert metrics_path.stat().st_size > 0
    metrics = json.loads(metrics_path.read_text())
    assert set(metrics) == EXPECTED_PROCESS_IDS

    for process_metrics in metrics.values():
        assert process_metrics["times_h"] == EXPECTED_CONCENTRATION_TIMES
        assert set(process_metrics["components"]) == EXPECTED_REACTOR_COMPONENTS
        for name, component_metrics in process_metrics["components"].items():
            for key in (
                "observed_concentration",
                "recovered_concentration",
                "absolute_error",
                "relative_error",
            ):
                _assert_finite_metric_series(
                    component_metrics[key],
                    len(EXPECTED_CONCENTRATION_TIMES),
                )
            assert component_metrics["max_absolute_error"] == max(
                component_metrics["absolute_error"]
            )
            assert component_metrics["max_relative_error"] == max(
                component_metrics["relative_error"]
            )
            assert (
                component_metrics["absolute_error_limit"]
                == (SPARSE_REINTEGRATION_ERROR_LIMITS[name][0])
            )
            assert (
                component_metrics["relative_error_limit"]
                == (SPARSE_REINTEGRATION_ERROR_LIMITS[name][1])
            )

        assert set(process_metrics["rates"]) == EXPECTED_RATE_NAMES
        for rate_metrics in process_metrics["rates"].values():
            _assert_finite_metric_series(
                rate_metrics["values"],
                len(EXPECTED_CONCENTRATION_TIMES),
            )
            assert rate_metrics["min"] == min(rate_metrics["values"])
            assert rate_metrics["max"] == max(rate_metrics["values"])

        transform_metrics = process_metrics["transform"]
        _assert_finite_metric_series(
            transform_metrics["total_volume"],
            len(EXPECTED_CONCENTRATION_TIMES),
        )
        _assert_finite_metric_series(
            transform_metrics["adf"],
            len(EXPECTED_CONCENTRATION_TIMES),
        )
        assert set(transform_metrics["feed_corrections"]) == EXPECTED_REACTOR_COMPONENTS
        for values in transform_metrics["feed_corrections"].values():
            _assert_finite_metric_series(values, len(EXPECTED_CONCENTRATION_TIMES))

    assert len(plot_paths) == SPARSE_DIAGNOSTIC_PLOTS_PER_PROCESS * process_count
    for path in plot_paths:
        assert path.exists()
        assert path.stat().st_size > 0


def _assert_exact_pseudobatch_transform_sane(process: bp.BioProcess) -> None:
    times = _time_grid_from_volume_changes(process.volume.volume_changes.values())
    total_volume = process.volume.total_volume
    transform = process.pseudobatch_transform
    assert total_volume is not None
    assert transform is not None
    assert _series_times(total_volume) == list(times)
    assert _series_times(transform.adf) == list(times)
    assert set(transform.feed_corrections) == set(process.reactor_medium.components)
    assert _series_times(total_volume) == _series_times(transform.adf)

    assert _array_values(total_volume.jump_times) == EXPECTED_EVENT_JUMP_TIMES
    _assert_finite_positive(total_volume)
    _assert_finite_positive(transform.adf)
    assert transform.sample_compensation is not None
    assert _series_times(transform.sample_compensation) == list(times)
    _assert_finite_positive(transform.sample_compensation)
    assert set(transform.accumulated_feeds) == {
        name
        for name, volume_change in process.volume.volume_changes.items()
        if isinstance(volume_change, bp.FeedVolumeChange)
    }
    for accumulated_feed in transform.accumulated_feeds.values():
        assert _series_times(accumulated_feed) == list(times)
        _assert_finite_nonnegative(accumulated_feed)

    final_volume = process.volume.initial_volume
    final_time = float(times[-1])
    for volume_change in process.volume.volume_changes.values():
        change_times = np.asarray(volume_change.values.times, dtype=float)
        change_values = np.asarray(volume_change.values.values, dtype=float)
        if volume_change.is_continuous:
            final_volume += float(np.interp(final_time, change_times, change_values))
        else:
            for time, value in zip(change_times, change_values, strict=True):
                if final_time > time:
                    final_volume += float(value)
    assert float(total_volume.values[-1]) == final_volume

    for species_name in EXPECTED_NONZERO_FEED_CORRECTION_COMPONENTS:
        feed_correction = transform.feed_corrections[species_name]
        _assert_finite_nonnegative(feed_correction)
        assert float(feed_correction.values[-1]) > 0.0
    for species_name, feed_correction in transform.feed_corrections.items():
        assert _series_times(feed_correction) == list(times)
        _assert_finite_nonnegative(feed_correction)
        if species_name not in EXPECTED_NONZERO_FEED_CORRECTION_COMPONENTS:
            assert _series_values(feed_correction) == [0.0] * len(times)


def _assert_pseudobatch_transform_equal(actual, expected) -> None:
    if expected is None:
        assert actual is None
        return
    assert actual is not None
    _assert_series_equal(actual.adf, expected.adf)
    assert set(actual.feed_corrections) == set(expected.feed_corrections)
    for name, series in expected.feed_corrections.items():
        _assert_series_equal(actual.feed_corrections[name], series)
    if expected.sample_compensation is None:
        assert actual.sample_compensation is None
    else:
        _assert_series_equal(actual.sample_compensation, expected.sample_compensation)
    assert set(actual.accumulated_feeds) == set(expected.accumulated_feeds)
    for name, series in expected.accumulated_feeds.items():
        _assert_series_equal(actual.accumulated_feeds[name], series)


def _assert_feed_media_equal(actual, expected) -> None:
    assert actual.name == expected.name
    assert actual.density == expected.density
    assert actual.density_unit == expected.density_unit
    assert set(actual.components) == set(expected.components)
    for name, component in expected.components.items():
        actual_component = actual.components[name]
        assert actual_component.name == component.name
        assert actual_component.unit == component.unit
        assert actual_component.is_controlled == component.is_controlled
        assert actual_component.concentration.value == component.concentration.value


def _assert_reloaded_collection_matches_parsed(parsed, reloaded) -> None:
    assert reloaded.metadata == parsed.metadata
    assert set(reloaded.processes) == set(parsed.processes)
    for process_id, process in parsed.processes.items():
        reloaded_process = reloaded.processes[process_id]
        assert reloaded_process.metadata == process.metadata
        assert reloaded_process.time_axis == process.time_axis
        assert reloaded_process.volume.initial_volume == process.volume.initial_volume
        assert reloaded_process.volume.unit == process.volume.unit
        if process.volume.total_volume is None:
            assert reloaded_process.volume.total_volume is None
        else:
            _assert_series_equal(
                reloaded_process.volume.total_volume,
                process.volume.total_volume,
            )
        assert reloaded_process.reactor_medium.name == process.reactor_medium.name
        assert reloaded_process.reactor_medium.density == process.reactor_medium.density
        assert (
            reloaded_process.reactor_medium.density_unit
            == process.reactor_medium.density_unit
        )
        assert set(reloaded_process.reactor_medium.components) == set(
            process.reactor_medium.components
        )
        assert set(reloaded_process.process_variables) == set(process.process_variables)
        assert set(reloaded_process.volume.volume_changes) == set(
            process.volume.volume_changes
        )
        assert reloaded_process.biological_ode == process.biological_ode

        for name, component in process.reactor_medium.components.items():
            reloaded_component = reloaded_process.reactor_medium.components[name]
            assert reloaded_component.name == component.name
            assert reloaded_component.unit == component.unit
            assert reloaded_component.bounds == component.bounds
            if component.c_star_concentration is None:
                assert reloaded_component.c_star_concentration is None
            else:
                assert isinstance(
                    reloaded_component.c_star_concentration, bp.TimeSeries
                )
                _assert_series_equal(
                    reloaded_component.c_star_concentration,
                    component.c_star_concentration,
                )
            _assert_series_equal(
                reloaded_component.concentration,
                component.concentration,
            )

        for name, variable in process.process_variables.items():
            reloaded_variable = reloaded_process.process_variables[name]
            assert reloaded_variable.name == variable.name
            assert reloaded_variable.unit == variable.unit
            assert reloaded_variable.is_controlled == variable.is_controlled
            assert reloaded_variable.bounds == variable.bounds
            _assert_series_equal(reloaded_variable.values, variable.values)

        for name, volume_change in process.volume.volume_changes.items():
            reloaded_volume_change = reloaded_process.volume.volume_changes[name]
            assert type(reloaded_volume_change) is type(volume_change)
            assert reloaded_volume_change.name == volume_change.name
            assert reloaded_volume_change.unit == volume_change.unit
            assert reloaded_volume_change.is_controlled == volume_change.is_controlled
            assert reloaded_volume_change.is_continuous == volume_change.is_continuous
            _assert_series_equal(reloaded_volume_change.values, volume_change.values)
            if hasattr(volume_change, "feed_medium"):
                _assert_feed_media_equal(
                    reloaded_volume_change.feed_medium,
                    volume_change.feed_medium,
                )

        _assert_pseudobatch_transform_equal(
            reloaded_process.pseudobatch_transform,
            process.pseudobatch_transform,
        )


def test_ex14_simulation_artifacts_match_tracked_canonical_files(tmp_path):
    regenerated_dir = _regenerate_simulation_artifacts(tmp_path / "simulation")

    for filename in CANONICAL_ARTIFACTS:
        tracked_path = SIMULATION_DIR / filename
        regenerated_path = regenerated_dir / filename
        assert regenerated_path.read_bytes() == tracked_path.read_bytes(), (
            f"ex14 simulation artifact drifted: {filename}\n"
            f"tracked: {tracked_path}\n"
            f"regenerated: {regenerated_path}\n"
            "Inspect both files. If the change is intentional, regenerate the "
            "tracked fixture artifact and commit it with the simulation change."
        )


def test_ex14_lab_like_csv_parse_roundtrips_basic_json(tmp_path):
    collection = _parse_lab_like_collection()

    assert collection.metadata["name"] == "ex14_lab_like_e2e"
    assert set(collection.processes) == EXPECTED_PROCESS_IDS
    for process in collection.processes.values():
        assert process.time_axis.start == 0.0
        assert process.time_axis.end == 120.0
        assert set(process.reactor_medium.components) == EXPECTED_REACTOR_COMPONENTS
        assert set(process.volume.volume_changes) == EXPECTED_VOLUME_CHANGES

        for component in process.reactor_medium.components.values():
            assert (
                _series_times(component.concentration) == EXPECTED_CONCENTRATION_TIMES
            )
            assert len(component.concentration.times) == 5

        ph = process.process_variables["pH"].values
        assert _series_times(ph)[0] == 0.0
        assert _series_times(ph)[-1] == 120.0
        assert len(ph.times) > len(EXPECTED_CONCENTRATION_TIMES)

        sampling = process.volume.volume_changes["sampling"].values
        assert _series_times(sampling) == EXPECTED_SAMPLE_EVENT_TIMES
        assert _series_values(sampling) == [-0.05, -0.05, -0.05, -0.05]

        bolus = process.volume.volume_changes["bolus_feed"].values
        assert _series_times(bolus) == [36.0, 72.0, 108.0]
        assert all(value > 0.0 for value in _series_values(bolus))

    output_path = tmp_path / "parsed_collection.json"
    save_process_collection_json(collection, output_path)
    reloaded = load_process_collection_json(output_path)
    _assert_reloaded_collection_matches_parsed(collection, reloaded)


def test_ex14_exact_pseudobatch_transform_is_sane_and_roundtrips(tmp_path):
    collection = _parse_lab_like_collection()

    for process in collection.processes.values():
        _populate_exact_pseudobatch_transform(process)
        _assert_exact_pseudobatch_transform_sane(process)
        for component in process.reactor_medium.components.values():
            assert component.c_star_concentration is None

    output_path = tmp_path / "transformed_collection.json"
    save_process_collection_json(collection, output_path)
    reloaded = load_process_collection_json(output_path)
    _assert_reloaded_collection_matches_parsed(collection, reloaded)


def test_ex14_cstar_splines_survive_json_roundtrip(tmp_path):
    collection = _parse_lab_like_collection()

    for process in collection.processes.values():
        _populate_exact_pseudobatch_transform(process)
        _populate_cstar_splines(process)
        _assert_exact_pseudobatch_transform_sane(process)
        _assert_cstar_splines_sane(process)

    output_path = tmp_path / "cstar_fitted_collection.json"
    save_process_collection_json(collection, output_path)
    reloaded = load_process_collection_json(output_path)
    _assert_reloaded_collection_matches_parsed(collection, reloaded)
    for process in reloaded.processes.values():
        _assert_cstar_splines_sane(process)


def test_ex14_infers_finite_rates_from_reloaded_cstar_splines(tmp_path):
    collection = _parse_lab_like_collection()

    for process in collection.processes.values():
        _populate_exact_pseudobatch_transform(process)
        _populate_cstar_splines(process)

    output_path = tmp_path / "cstar_fitted_collection.json"
    save_process_collection_json(collection, output_path)
    reloaded = load_process_collection_json(output_path)

    times = np.asarray(EXPECTED_CONCENTRATION_TIMES, dtype=float)
    for process in reloaded.processes.values():
        rates, diagnostics = _infer_ex14_rates_from_cstar_splines(process, times)
        _assert_inferred_rates_finite(rates, diagnostics)
        _assert_state_transform_identities(diagnostics)
        _assert_rates_match_biological_derivatives(rates, diagnostics)


def test_ex14_reintegrates_and_backtransforms_sparse_real_space_concentrations(
    tmp_path,
):
    collection = _parse_lab_like_collection()

    for process in collection.processes.values():
        _populate_exact_pseudobatch_transform(process)
        _populate_cstar_splines(process)

    output_path = tmp_path / "cstar_fitted_collection.json"
    save_process_collection_json(collection, output_path)
    reloaded = load_process_collection_json(output_path)

    times = np.asarray(EXPECTED_CONCENTRATION_TIMES, dtype=float)
    diagnostics_dir = tmp_path / "ex14_sparse_reintegration_diagnostics"
    metrics = {}
    plot_paths = []
    for process in reloaded.processes.values():
        rates, diagnostics = _infer_ex14_rates_from_cstar_splines(process, times)
        _assert_inferred_rates_finite(rates, diagnostics)
        if process.metadata.name == "ex14_run_1":
            _assert_transform_right_limits_are_event_local(process)
        _assert_pseudobatch_rhs_initial_algebra(process, times, rates, diagnostics)
        reintegrated_cstar_states = _integrate_pseudobatch_cstar_states(
            process,
            rate_times=times,
            rates=rates,
            diagnostics=diagnostics,
            output_times=times,
        )
        _assert_reintegrated_cstar_states_finite(
            reintegrated_cstar_states,
            diagnostics,
            times,
        )
        recovered_concentrations = _evaluate_reconstructed_reactor_states(
            process,
            times,
            reintegrated_cstar_states,
        )
        observed_concentrations = _observed_sparse_reactor_concentrations(
            process,
            times,
        )
        _assert_sparse_real_space_reintegration_sane(
            recovered_concentrations,
            observed_concentrations,
            times,
        )
        assert process.metadata.name is not None
        metrics[process.metadata.name] = _sparse_reintegration_metrics(
            process,
            times,
            rates,
            diagnostics,
            recovered_concentrations,
            observed_concentrations,
        )
        plot_paths.extend(
            _write_sparse_reintegration_diagnostic_plots(
                diagnostics_dir,
                process.metadata.name,
                times,
                rates,
                diagnostics,
                recovered_concentrations,
                observed_concentrations,
            )
        )

    metrics_path = _write_sparse_reintegration_metrics_json(diagnostics_dir, metrics)
    _assert_sparse_reintegration_outputs_written(
        metrics_path,
        plot_paths,
        len(reloaded.processes),
    )
