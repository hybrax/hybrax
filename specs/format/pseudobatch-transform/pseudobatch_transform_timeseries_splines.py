# %% ###################################################################################
"""Minimal pseudobatch-transform construction examples for ex14.

This script shows two ways to populate ``process.pseudobatch_transform`` from
volume-change information.

1. ``populate_pseudobatch_transform_TimeSeries_discrete_values``

   This is the exact/no-noise path. It assumes the cumulative continuous feeds
   and the discrete sampling/bolus events are already clean enough to use as
   tabulated values. Total reactor volume is reconstructed directly on the union
   of all volume-change timestamps:

   - continuous feed traces are linearly interpolated and added as cumulative
     volumes;
   - sampling and bolus entries are treated as trusted discrete jumps;
   - ADF is computed from the reconstructed total volume plus explicit
     sample-compensation factors;
   - feed corrections use cumulative feed-volume increments via ``diff()`` and
     exact bolus jumps.

   The resulting ``TimeSeries`` objects mainly store raw values. This is useful
   as a transparent reference calculation and as a compact explanation of the
   pseudobatch formulas.

2. ``populate_pseudobatch_transform_TimeSeries_splines_noisy``

   This is the noisy-scale path. It mimics the common lab situation where
   continuous feeds are measured by putting feed flasks on scales: the recorded
   trace is the cumulative amount of feed medium added up to each timestamp, but
   it contains scale noise. In contrast, sampling and bolus events are assumed
   to be trusted lab-recorded events with known times and volume deltas.

   The function therefore smooths only the continuous cumulative feed traces and
   stores those smoothing splines on the feed ``TimeSeries``. It then derives all
   downstream quantities from those smoothed feed splines plus trusted discrete
   events:

   - total volume values keep the raw noisy cumulative-feed measurements plus
     trusted event jumps for plotting/inspection;
   - total volume spline evaluation uses smoothed cumulative feeds plus trusted
     event jumps;
   - ADF is built from event-aware total volume, preserving left-continuous
     jump semantics so sample-only events do not jump ADF while bolus events do;
   - continuous feed corrections use ``cumulative_trapezoid`` on the derivative
     of the smoothed cumulative feed spline, matching the minimal
     backtransform-script formula ``integral ADF * c_feed * F / V dt``;
   - bolus feed corrections remain exact jumps.

   This second path shows how to use spline-backed ``TimeSeries`` objects when
   raw cumulative feed traces are noisy but event metadata is trustworthy.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping
from dataclasses import dataclass
import os
import sys
from pathlib import Path

os.environ.setdefault("JAX_ENABLE_X64", "true")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/bpbench-matplotlib")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.integrate import cumulative_trapezoid

REPO_ROOT = Path(__file__).resolve().parents[2]
EXAMPLE_ROOT = REPO_ROOT / "examples" / "14_simulation_intracellular"
COLLECTION_JSON = EXAMPLE_ROOT / "02_all_processes" / "output" / "data.json"

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import bp_format as bp  # noqa: E402

# from bp_format import TimeSeries  # noqa: E402
# from bp_format.dataclasses import FeedVolumeChange  # noqa: E402
# from bp_format.dataclasses import PseudobatchTransform  # noqa: E402
# from bp_format.dataclasses import SampleVolumeChange  # noqa: E402
from bp_format.serialization import load_process_collection_json  # noqa: E402


collection = load_process_collection_json(COLLECTION_JSON)
# %% ###################################################################################


def populate_pseudobatch_transform_TimeSeries_discrete_values(
    process: bp.Bioprocess,
) -> None:
    """
    Calculate total volume, ADF, and feed corrections using discrete volume changes
    (also for continuous feeds).
    """
    volume_changes = process.volume.volume_changes.values()
    times = np.unique(
        np.concatenate(
            [np.asarray(vc.values.times, dtype=float) for vc in volume_changes]
        )
    )

    V_cont = np.full(len(times), process.volume.initial_volume)
    total_volume = np.full(len(times), process.volume.initial_volume)
    sample_deltas = defaultdict(float)
    discrete_deltas = defaultdict(float)

    for vc in process.volume.volume_changes.values():
        vc_times = np.asarray(vc.values.times, dtype=float)
        vc_values = np.asarray(vc.values.values, dtype=float)
        assert len(vc_times) == len(vc_values)

        if vc.is_continuous:
            V_cont += np.interp(times, vc_times, vc_values)
            total_volume += np.interp(times, vc_times, vc_values)
        else:
            for time, value in zip(vc_times, vc_values, strict=True):
                total_volume[times > time] += value
                discrete_deltas[time] += value

                if isinstance(vc, bp.SampleVolumeChange):
                    sample_deltas[time] += value

    jump_times = np.asarray(sorted(discrete_deltas), dtype=float)
    jump_times = jump_times[(times[0] < jump_times) & (jump_times < times[-1])]

    # turn total volume into TimeSeries and add to process object; get pd.Series for
    # convenience
    process.volume.total_volume = bp.TimeSeries(
        times=times,
        values=total_volume,
        jump_times=jump_times,
    )

    # ADF(t) = V(t) / V0 times cumulative undo-sampling factors.
    # For a sample with volume delta < 0:
    #   sample_factor = V_pre_sample / V_post_sample

    sample_compensation = np.ones_like(times)
    for time, delta in sample_deltas.items():
        v_pre_sample = np.interp(time, times, total_volume)
        v_post_sample = v_pre_sample + delta
        sample_compensation[times > time] *= v_pre_sample / v_post_sample

    adf = total_volume / process.volume.initial_volume * sample_compensation
    adf = bp.TimeSeries(times=times, values=adf)

    # get pd.Series of a couple things for convenience
    volume_series = process.volume.total_volume.to_pd_series()
    sample_compensation = pd.Series(sample_compensation, index=times)

    # calculate feed corrections for each species
    feed_corrections = {}
    for species_name in process.reactor_medium.components:
        feed_corr = pd.Series(0.0, index=times)

        for vc in process.volume.volume_changes.values():
            if not isinstance(vc, bp.FeedVolumeChange):
                continue

            c_feed = vc.feed_medium.components[species_name].concentration.value
            if c_feed == 0.0:
                continue

            vc_times = np.asarray(vc.values.times, dtype=float)
            vc_values = np.asarray(vc.values.values, dtype=float)

            if vc.is_continuous:
                # The library stores cumulative feed volume V_feed(t), not feed
                # rate F(t). Starting from the usual feed correction
                #   feed_corr = integral ADF * c_feed * F / V dt
                # and using
                #   ADF = V * sample_compensation / V0
                # gives
                #   d(feed_corr) = sample_compensation * c_feed / V0 * dV_feed
                # so we use feed-volume increments, not cumulative_trapezoid.
                cumulative_feed = pd.Series(
                    np.interp(times, vc_times, vc_values),
                    index=times,
                )
                feed_volume_added = cumulative_feed.diff().fillna(0.0)
                feed_corr += (
                    sample_compensation
                    / process.volume.initial_volume
                    * feed_volume_added
                    * c_feed
                ).cumsum()
            else:
                for time, feed_volume_added in zip(vc_times, vc_values, strict=True):
                    # Bolus feed is the same formula as above, but dV_feed is a
                    # jump. If sample and bolus share a timestamp, ex14/library
                    # convention applies the sample first, then the bolus.
                    sample_comp_at_bolus = sample_compensation[time]
                    V_sample = sample_deltas[time]
                    if V_sample != 0.0:
                        v_pre_sample = volume_series[time]
                        v_post_sample = v_pre_sample + V_sample
                        sample_comp_at_bolus *= v_pre_sample / v_post_sample
                    feed_corr[times > time] += (
                        sample_comp_at_bolus
                        * feed_volume_added
                        * c_feed
                        / process.volume.initial_volume
                    )

        feed_corrections[species_name] = bp.TimeSeries(times=times, values=feed_corr)

    process.pseudobatch_transform = bp.PseudobatchTransform(
        adf=adf, feed_corrections=feed_corrections
    )


for process_id, process in collection.processes.items():
    populate_pseudobatch_transform_TimeSeries_discrete_values(process)
# %% ###################################################################################


CONTINUOUS_FEED_NOISE_L = 0.002


@dataclass
class _SplinePseudobatchInputs:
    times: np.ndarray
    boundaries: np.ndarray
    total_volume: bp.TimeSeries
    sample_deltas: Mapping[float, float]
    discrete_deltas: Mapping[float, float]
    bolus_feed_deltas: Mapping[str, Mapping[float, float]]
    continuous_feed_splines: Mapping[str, bp.TimeSeries]


def _fit_spline(
    values: np.ndarray,
    times: np.ndarray,
    boundaries: np.ndarray,
    smoothing_s: float,
) -> bp.TimeSeries:
    return bp.splines.fit_timeseries_spline(
        bp.TimeSeries(times=times, values=values),
        boundaries=boundaries,
        smoothing_s=smoothing_s,
    )


def _anchor_spline_at_zero(series: bp.TimeSeries) -> bp.TimeSeries:
    offset = float(series.evaluate(0.0))
    coeffs = np.asarray(series.coeffs, dtype=float).copy()
    coeffs[:, 0] -= offset
    return bp.TimeSeries(
        times=series.times,
        values=np.asarray(series.values, dtype=float) - offset,
        breaks=series.breaks,
        coeffs=coeffs,
        segment_start_piece_idx=series.segment_start_piece_idx,
        continuity_side=series.continuity_side,
        jump_times=series.jump_times,
        metadata=series.metadata,
    )


def _add_discrete_offsets_to_spline(
    base_spline: bp.TimeSeries,
    discrete_deltas: Mapping[float, float],
    times: np.ndarray,
    values: np.ndarray,
) -> bp.TimeSeries:
    breaks = np.asarray(base_spline.breaks, dtype=float)
    coeffs = np.asarray(base_spline.coeffs, dtype=float).copy()
    offset_after_break = np.zeros(len(breaks))
    for time, value in discrete_deltas.items():
        offset_after_break[breaks >= time] += value
    coeffs[:, 0] += offset_after_break[:-1]

    event_times = np.asarray(sorted(discrete_deltas), dtype=float)
    jump_times = event_times[(times[0] < event_times) & (event_times < times[-1])]
    return bp.TimeSeries(
        times=times,
        values=values,
        breaks=breaks,
        coeffs=coeffs,
        segment_start_piece_idx=base_spline.segment_start_piece_idx,
        continuity_side="left",
        jump_times=jump_times,
    )


def add_noise_to_continuous_feed_measurements(process: bp.Bioprocess) -> None:
    """Pretend cumulative feed traces came from noisy feed-flask scales."""
    for vc in process.volume.volume_changes.values():
        if not (vc.is_continuous and isinstance(vc, bp.FeedVolumeChange)):
            continue

        times = np.asarray(vc.values.times, dtype=float)
        original = np.asarray(vc.values.values, dtype=float)
        noisy = original + np.random.normal(
            scale=CONTINUOUS_FEED_NOISE_L,
            size=len(times),
        )
        noisy = np.clip(noisy, 0.0, None)
        noisy[0] = original[0]
        vc.values = bp.TimeSeries(times=times, values=noisy)


def _prepare_spline_pseudobatch_inputs(
    process: bp.Bioprocess,
) -> _SplinePseudobatchInputs:
    """Smooth feed scale traces once and collect trusted discrete events."""
    volume_changes = process.volume.volume_changes.values()
    times = np.unique(
        np.concatenate(
            [np.asarray(vc.values.times, dtype=float) for vc in volume_changes]
        )
    )

    smooth_volume_without_events = np.full(len(times), process.volume.initial_volume)
    raw_total_volume_values = np.full(len(times), process.volume.initial_volume)
    sample_deltas = defaultdict(float)
    discrete_deltas = defaultdict(float)
    bolus_feed_deltas = defaultdict(dict)
    continuous_feed_splines = {}

    for name, vc in process.volume.volume_changes.items():
        vc_times = np.asarray(vc.values.times, dtype=float)
        vc_values = np.asarray(vc.values.values, dtype=float)
        assert len(vc_times) == len(vc_values)

        if vc.is_continuous:
            # Continuous feed traces are cumulative volumes inferred from noisy
            # feed-flask scales. Smooth them once, anchor at 0 L, then use the
            # smoothed cumulative feed everywhere downstream.
            measured_feed = np.interp(times, vc_times, vc_values)
            feed_spline = _fit_spline(
                measured_feed,
                times,
                np.asarray([times[0], times[-1]], dtype=float),
                smoothing_s=1.0,
            )
            feed_spline = _anchor_spline_at_zero(feed_spline)
            smoothed_feed = np.asarray(feed_spline.evaluate_many(times), dtype=float)
            vc.values = bp.TimeSeries(
                times=times,
                values=measured_feed,
                breaks=feed_spline.breaks,
                coeffs=feed_spline.coeffs,
                segment_start_piece_idx=feed_spline.segment_start_piece_idx,
                continuity_side=feed_spline.continuity_side,
                metadata=feed_spline.metadata,
            )
            continuous_feed_splines[name] = vc.values
            smooth_volume_without_events += smoothed_feed
            raw_total_volume_values += measured_feed
            continue

        # Discrete bolus/sample entries are lab-recorded events, not scale traces.
        # Keep their times and volume deltas exact/trusted.
        for time, value in zip(vc_times, vc_values, strict=True):
            raw_total_volume_values[times > time] += value
            discrete_deltas[time] += value
            if isinstance(vc, bp.SampleVolumeChange):
                sample_deltas[time] += value
            if isinstance(vc, bp.FeedVolumeChange):
                bolus_feed_deltas[name][time] = value

    event_times = np.asarray(sorted(discrete_deltas), dtype=float)
    boundaries = np.unique(np.concatenate(([times[0]], event_times, [times[-1]])))

    # Fit the continuous volume exactly from the smoothed feeds and then add the
    # trusted discrete jumps to the PPoly constants. The raw TimeSeries values
    # remain raw noisy-feed-derived volume values for plotting/inspection.
    volume_spline = _fit_spline(
        smooth_volume_without_events,
        times,
        boundaries,
        smoothing_s=0.0,
    )
    total_volume = _add_discrete_offsets_to_spline(
        volume_spline,
        discrete_deltas,
        times,
        raw_total_volume_values,
    )

    return _SplinePseudobatchInputs(
        times=times,
        boundaries=boundaries,
        total_volume=total_volume,
        sample_deltas=sample_deltas,
        discrete_deltas=discrete_deltas,
        bolus_feed_deltas=bolus_feed_deltas,
        continuous_feed_splines=continuous_feed_splines,
    )


def _build_adf_from_total_volume(
    process: bp.Bioprocess,
    inputs: _SplinePseudobatchInputs,
) -> tuple[bp.TimeSeries, pd.Series]:
    """Build ADF and sample-compensation factors from event-aware volume."""
    times = inputs.times
    total_volume = inputs.total_volume
    total_volume_values = np.asarray(total_volume.evaluate_many(times, side="left"))

    # ADF(t) = V(t) / V0 times factors that undo sample removals. A sample-only
    # event should not jump ADF: the V drop and compensation-factor increase
    # cancel. Bolus events do jump ADF because they add feed volume.
    sample_compensation = pd.Series(1.0, index=times)
    adf_jump_deltas = {}
    sample_comp = 1.0
    for time in sorted(inputs.discrete_deltas):
        v_pre = float(total_volume.evaluate(time, side="left"))
        adf_pre = v_pre / process.volume.initial_volume * sample_comp
        sample_delta = inputs.sample_deltas[time]
        if sample_delta != 0.0:
            sample_comp *= v_pre / (v_pre + sample_delta)
            sample_compensation.loc[times > time] = sample_comp
        v_post = v_pre + inputs.discrete_deltas[time]
        adf_post = v_post / process.volume.initial_volume * sample_comp
        adf_jump = adf_post - adf_pre
        if abs(adf_jump) > 1e-12:
            adf_jump_deltas[time] = adf_jump

    adf_values = (
        total_volume_values
        / process.volume.initial_volume
        * sample_compensation.to_numpy()
    )

    # Fit the continuous part exactly after subtracting event jumps, then add the
    # jumps back as explicit PPoly offsets so ADF has the same event semantics as
    # total volume.
    adf_jump_offsets = np.zeros(len(times))
    for time, jump in adf_jump_deltas.items():
        adf_jump_offsets[times > time] += jump
    adf_cont_spline = _fit_spline(
        adf_values - adf_jump_offsets,
        times,
        inputs.boundaries,
        smoothing_s=0.0,
    )
    adf = _add_discrete_offsets_to_spline(
        adf_cont_spline,
        adf_jump_deltas,
        times,
        adf_values,
    )
    return adf, sample_compensation


def _build_feed_corrections_from_smoothed_feeds(
    process: bp.Bioprocess,
    inputs: _SplinePseudobatchInputs,
    sample_compensation: pd.Series,
) -> dict[str, bp.TimeSeries]:
    feed_corrections = {}
    times = inputs.times

    for species_name in process.reactor_medium.components:
        feed_corr_cont = pd.Series(0.0, index=times)
        feed_corr = pd.Series(0.0, index=times)
        feed_corr_jumps = defaultdict(float)

        for name, vc in process.volume.volume_changes.items():
            if not isinstance(vc, bp.FeedVolumeChange):
                continue

            c_feed = vc.feed_medium.components[species_name].concentration.value
            if c_feed == 0.0:
                continue

            if vc.is_continuous:
                # Same form as the minimal backtransform script:
                #   feed_corr = integral ADF * c_feed * F / V dt
                # Here F and V come from smoothing splines. Because
                # ADF = V / V0 * sample_compensation, this simplifies to
                #   integral sample_compensation / V0 * c_feed * F dt
                feed_rate = np.asarray(
                    inputs.continuous_feed_splines[name].deriv().evaluate_many(times),
                    dtype=float,
                )
                integrand = (
                    sample_compensation.to_numpy()
                    / process.volume.initial_volume
                    * c_feed
                    * feed_rate
                )
                contribution = pd.Series(
                    cumulative_trapezoid(integrand, times, initial=0.0),
                    index=times,
                )
                feed_corr_cont += contribution
                feed_corr += contribution
            else:
                # Bolus corrections are exact jumps. If a sample and bolus share
                # a timestamp, ex14 convention applies the sample first.
                for time, feed_volume_added in inputs.bolus_feed_deltas[name].items():
                    sample_comp_at_bolus = sample_compensation[time]
                    sample_delta = inputs.sample_deltas[time]
                    if sample_delta != 0.0:
                        v_pre_sample = float(
                            inputs.total_volume.evaluate(time, side="left")
                        )
                        v_post_sample = v_pre_sample + sample_delta
                        sample_comp_at_bolus *= v_pre_sample / v_post_sample
                    jump = (
                        sample_comp_at_bolus
                        * feed_volume_added
                        * c_feed
                        / process.volume.initial_volume
                    )
                    feed_corr_jumps[time] += jump
                    feed_corr[times > time] += jump

        feed_corr_cont_spline = _fit_spline(
            feed_corr_cont.to_numpy(),
            times,
            inputs.boundaries,
            smoothing_s=0.0,
        )
        feed_corrections[species_name] = _add_discrete_offsets_to_spline(
            feed_corr_cont_spline,
            feed_corr_jumps,
            times,
            feed_corr.to_numpy(),
        )

    return feed_corrections


def populate_pseudobatch_transform_TimeSeries_splines_noisy(
    process: bp.Bioprocess,
) -> None:
    """
    Smooth noisy cumulative feed measurements, combine them with trusted
    sample/bolus events to build total volume, then build ADF/feed corrections.
    """
    inputs = _prepare_spline_pseudobatch_inputs(process)
    adf, sample_compensation = _build_adf_from_total_volume(process, inputs)
    feed_corrections = _build_feed_corrections_from_smoothed_feeds(
        process,
        inputs,
        sample_compensation,
    )
    process.volume.total_volume = inputs.total_volume
    process.pseudobatch_transform = bp.PseudobatchTransform(
        adf=adf,
        feed_corrections=feed_corrections,
    )


noisy_collection = load_process_collection_json(COLLECTION_JSON)
np.random.seed(14)
for process_id, process in noisy_collection.processes.items():
    # This mutates the process in-place: continuous feed volume traces become noisy
    # raw samples first and spline-backed smoothed traces after transform creation.
    add_noise_to_continuous_feed_measurements(process)
    populate_pseudobatch_transform_TimeSeries_splines_noisy(process)
# %% ###################################################################################
fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True, sharey="row")

for i, process in enumerate(noisy_collection.processes.values()):
    continuous_feed_changes = [
        vc for vc in process.volume.volume_changes.values() if vc.is_continuous
    ]
    times = np.unique(
        np.concatenate(
            [np.asarray(vc.values.times, dtype=float) for vc in continuous_feed_changes]
        )
    )
    V_cont_values = np.full(len(times), 0.0)
    V_cont_spline_values = np.full(len(times), 0.0)
    for vc in continuous_feed_changes:
        V_cont_values += np.interp(
            times,
            np.asarray(vc.values.times, dtype=float),
            np.asarray(vc.values.values, dtype=float),
        )
        V_cont_spline_values += np.asarray(vc.values.evaluate_many(times), dtype=float)

    axes[0, i].plot(times, V_cont_values, label="V_cont raw TS values")
    axes[0, i].plot(times, V_cont_spline_values, "--", label="V_cont spline eval")
    axes[0, i].set_title(f"{process.metadata.name}: continuous volume")
    axes[1, i].plot(
        process.volume.total_volume.times,
        process.volume.total_volume.values,
        label="V_tot raw TS values",
    )
    axes[1, i].plot(
        process.volume.total_volume.times,
        process.volume.total_volume.evaluate_many(process.volume.total_volume.times),
        "--",
        label="V_tot spline eval",
    )
    axes[1, i].set_title(f"{process.metadata.name}: total volume")

for ax in axes.ravel():
    ax.set_ylabel("volume [L]")
    ax.grid(alpha=0.25)
    ax.legend()
axes[-1, -1].set_xlabel("time [h]")
fig.tight_layout()
