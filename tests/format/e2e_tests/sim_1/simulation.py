"""Intracellular-product simulation 1.

This is a compact Martens/ex12-inspired CHO fed-batch model for mechanistic
verification. It keeps the source examples' pH/temperature control,
glucose/glutamine limitation, nutrient feeds, base-feed dilution, and
sampling/bolus event semantics, but simplifies product biology to an
intracellular/extracellular split.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
import os
from pathlib import Path
import sys
from typing import Iterable, Sequence

os.environ.setdefault("JAX_ENABLE_X64", "true")

import jax.numpy as jnp
import numpy as np
from scipy.integrate import solve_ivp

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from bp_format.simulation import Simulation, SimulationEvent, SimulationResult  # noqa: E402
from bp_format.simulation import EVENT_TYPE_BOLUS, EVENT_TYPE_SAMPLE  # noqa: E402
from bp_format.simulation import EVENT_TYPE_FERMENTATION_END  # noqa: E402

PROCESS_ID = "sim_1_run_1"
T_END = 120.0
INITIAL_VOLUME = 1.0
CONTI_FLOW_L_PER_H = 0.0015
BASE_PER_GLUCOSE_UPTAKE_L_PER_MMOL = 0.001
# Ex12 uses these feed concentrations for nutrient continuous/bolus feed.
FEED_GLUCOSE = 500.0
FEED_GLUTAMINE = 50.0
SAMPLE_VOLUME_L = 0.050
ONLINE_POINTS_PER_H = 12
# Martens virtual_lab uses Gaussian-like pH/temperature growth factors.
# Sim 1 keeps that dependency but centers it on fixed nominal setpoints.
PH_NOMINAL = 7.05
PH_WIDTH = 0.35
TEMPERATURE_NOMINAL = 36.8
TEMPERATURE_WIDTH = 1.5
# Ex12 used a time-varying non-glycosylated split. Sim 1 replaces that with an
# intracellular retention ratio that relaxes from fully secreted toward 50/50.
RATIO_MIN = 0.0
RATIO_MAX = 0.55
RATIO_TARGET = 0.5
RATIO_RELAXATION_PER_H = 0.030
# These rates are deliberately small and simple. They are chosen to make a
# well-behaved verification fixture, not fitted to a specific Martens run.
MU_MAX_PER_H = 0.0080
DEATH_RATE_PER_H = 0.0015
PRODUCT_RATE_PER_H = 0.0040
# Martens growth is Monod-limited by glucose and glutamine. Sim 1 applies the
# same limitation to growth/product formation and to substrate uptake, so
# depletion cannot create biomass/product without substrate.
GLUCOSE_HALF_SATURATION = 1.0
GLUTAMINE_HALF_SATURATION = 0.2
GLUCOSE_UPTAKE_RATE = 0.0015
GLUTAMINE_UPTAKE_RATE = 0.00035
LACTATE_PRODUCTION_RATE = 0.0012
AMMONIA_PRODUCTION_RATE = 0.00045

KINETIC_REACTOR_NAMES = (
    "biomass",
    "product_extracellular",
    "product_intracellular",
    "dead_cells",
    "glucose",
    "glutamine",
    "lactate",
    "ammonia",
)
# Concentration units for reactor-medium states. Biomass and products are mass
# concentrations because intracellular product is part of measured biomass.
# Nutrients/byproducts use molar concentrations. Volume states/feeds use L.
REACTOR_STATE_UNITS = {
    "biomass": "mg/L",
    "product_extracellular": "mg/L",
    "product_intracellular": "mg/L",
    "dead_cells": "mg/L",
    "glucose": "mmol/L",
    "glutamine": "mmol/L",
    "lactate": "mmol/L",
    "ammonia": "mmol/L",
    # Inert pseudobatch tracers (see TRACER_* below). Dimensionless synthetic
    # witnesses, so unit "-"; only the parser consumes these entries (plotting
    # iterates the real KINETIC_REACTOR_NAMES only).
    "tracer_unfed": "-",
    "tracer_fed": "-",
}
PROCESS_VARIABLE_STATE_NAMES = ("intracellular_product_ratio",)
KINETIC_STATE_NAMES = (*KINETIC_REACTOR_NAMES, *PROCESS_VARIABLE_STATE_NAMES, "volume")
REACTOR_INDEX = {name: index for index, name in enumerate(KINETIC_REACTOR_NAMES)}
PROCESS_VARIABLE_INDEX = {
    name: index for index, name in enumerate(PROCESS_VARIABLE_STATE_NAMES)
}
STATE_INDEX = {name: index for index, name in enumerate(KINETIC_STATE_NAMES)}
CONTROL_NAMES = ("conti_feed", "pH", "temperature")
CONTROL_INDEX = {name: index for index, name in enumerate(CONTROL_NAMES)}

# Inert pseudobatch tracers: NON-PHYSICAL witnesses (an unfed and a fed inert species
# whose pseudobatch c* is constant by construction), not real biology. They are full
# reactor-medium species (see REACTOR_MEDIUM_NAMES below) but have no rate model, so
# they stay out of KINETIC_REACTOR_NAMES and get a "0" biological_ode derivative on the
# parser side. The dense c* oracle test uses them as a closed-form, integrator-sourced
# ground truth for the public ADF / feed-correction carriers (the inert-tracer oracle
# checks A and C in test_direct_cstar_reintegration.py).
TRACER_UNFED_NAME = "tracer_unfed"
TRACER_FED_NAME = "tracer_fed"
TRACER_REACTOR_NAMES = (TRACER_UNFED_NAME, TRACER_FED_NAME)
TRACER_UNFED_INITIAL = 100.0
TRACER_FED_INITIAL = 0.0
# Fed-tracer concentration in BOTH nutrient streams (continuous feed + bolus), the
# streams that carry glucose/glutamine. Distinct from FEED_GLUCOSE/FEED_GLUTAMINE
# so the per-species feed-correction ratio check is non-trivial.
TRACER_FED_FEED_CONCENTRATION = 300.0

# Rate columns included in `simulation_dense_output.csv`. The `r` array is
# laid out as KINETIC_REACTOR_NAMES then PROCESS_VARIABLE_STATE_NAMES.
RATE_COLUMNS = (
    *(f"q_{name}" for name in KINETIC_REACTOR_NAMES),
    "glucose_total_uptake_rate",
    "base_flow_l_per_h",
    *(f"r_{name}" for name in KINETIC_REACTOR_NAMES),
    *(f"r_{name}" for name in PROCESS_VARIABLE_STATE_NAMES),
)

INITIAL_STATE = np.asarray(
    [
        400.0,  # biomass, mg/L
        0.0,  # product_extracellular, mg/L
        0.0,  # product_intracellular, mg/L
        0.0,  # dead_cells, mg/L
        25.0,  # glucose, mmol/L
        4.0,  # glutamine, mmol/L
        1.0,  # lactate, mmol/L
        0.2,  # ammonia, mmol/L
        0.0,  # intracellular_product_ratio
        INITIAL_VOLUME,
    ],
    dtype=float,
)

# Glossary of the four state tuples:
#   KINETIC_REACTOR_NAMES (8)  - species the rate model acts on; drives REACTOR_INDEX,
#                                the `_rhs_numpy` reactor slice, and RATE_COLUMNS.
#   KINETIC_STATE_NAMES (10)   - kinetic core = reactor species + pv + volume; drives
#                                STATE_INDEX and the integrated RHS layout.
#   REACTOR_MEDIUM_NAMES (10)  - the reactor medium = real species + the 2 inert
#                                tracers; mixed at events, serialized, parsed as JSON
#                                components. This is the base-class reactor_state_names.
#   FULL_STATE_NAMES (12)      - kinetic core + tracers; the integrated + serialized
#                                state. Tracers go LAST so every kinetic index above
#                                stays unchanged and the real-species RHS is untouched.
FULL_STATE_NAMES = (*KINETIC_STATE_NAMES, *TRACER_REACTOR_NAMES)
REACTOR_MEDIUM_NAMES = (*KINETIC_REACTOR_NAMES, *TRACER_REACTOR_NAMES)
FULL_INITIAL_STATE = np.concatenate(
    [INITIAL_STATE, [TRACER_UNFED_INITIAL, TRACER_FED_INITIAL]]
)
# Column indices of the two tracers within the full state vector (appended after
# the 10 kinetic-core states), used by `_rhs_numpy` to read the tracer values.
TRACER_UNFED_INDEX = len(KINETIC_STATE_NAMES)
TRACER_FED_INDEX = len(KINETIC_STATE_NAMES) + 1


@dataclass(frozen=True)
class Sim1Schedule:
    online_times: tuple[float, ...]
    sample_times: tuple[float, ...]
    bolus_feeds: tuple[tuple[float, float], ...]


@dataclass(frozen=True)
class Sim1ProcessConfig:
    process_id: str
    conti_flow_l_per_h: float
    base_per_glucose_uptake_l_per_mmol: float
    schedule: Sim1Schedule


DEFAULT_ONLINE_TIMES = tuple(
    index / ONLINE_POINTS_PER_H for index in range(int(T_END * ONLINE_POINTS_PER_H) + 1)
)
DEFAULT_CONFIGS = (
    # First run mirrors ex12's combined setup: continuous nutrient feed,
    # uptake-proportional base-feed dilution, discrete bolus feeds, and regular samples.
    Sim1ProcessConfig(
        process_id="sim_1_run_1",
        conti_flow_l_per_h=CONTI_FLOW_L_PER_H,
        base_per_glucose_uptake_l_per_mmol=BASE_PER_GLUCOSE_UPTAKE_L_PER_MMOL,
        schedule=Sim1Schedule(
            online_times=DEFAULT_ONLINE_TIMES,
            sample_times=(24.0, 48.0, 72.0, 96.0),
            bolus_feeds=((36.0, 0.060), (72.0, 0.060), (108.0, 0.060)),
        ),
    ),
    # Second run removes continuous nutrient feed and compensates with larger
    # nutrient boluses. Base feed still follows glucose uptake as a pH-control proxy.
    Sim1ProcessConfig(
        process_id="sim_1_run_2",
        conti_flow_l_per_h=0.0,
        base_per_glucose_uptake_l_per_mmol=BASE_PER_GLUCOSE_UPTAKE_L_PER_MMOL,
        schedule=Sim1Schedule(
            online_times=DEFAULT_ONLINE_TIMES,
            sample_times=(24.0, 48.0, 72.0, 96.0),
            bolus_feeds=((36.0, 0.130), (72.0, 0.130), (108.0, 0.130)),
        ),
    ),
)


class Sim1Simulation(Simulation):
    """Small fed-batch simulation with one intracellular reactor component."""

    # Bound to the medium/full tuples (see the glossary above), NOT the kinetic ones:
    # the base class mixes/serializes reactor_state_names and serializes state_names.
    state_names = FULL_STATE_NAMES
    reactor_state_names = REACTOR_MEDIUM_NAMES
    process_variable_state_names = PROCESS_VARIABLE_STATE_NAMES
    initial_state = FULL_INITIAL_STATE

    def __init__(self, config: Sim1ProcessConfig | None = None):
        self.config = config or DEFAULT_CONFIGS[0]
        self.process_id = self.config.process_id
        self.schedule = self.config.schedule

    def pH(self, t):
        return np.asarray(t, dtype=float) * 0.0 + PH_NOMINAL

    def temperature(self, t):
        return np.asarray(t, dtype=float) * 0.0 + TEMPERATURE_NOMINAL

    def cum_conti_feed(self, t):
        return np.asarray(t, dtype=float) * self.config.conti_flow_l_per_h

    def glucose_total_uptake_rate(self, state, q):
        active_biomass = max(
            state[STATE_INDEX["biomass"]] - state[STATE_INDEX["product_intracellular"]],
            0.0,
        )
        volume = state[STATE_INDEX["volume"]]
        return max(-q[REACTOR_INDEX["glucose"]] * active_biomass * volume, 0.0)

    def base_flow_l_per_h(self, state, q):
        return (
            self.config.base_per_glucose_uptake_l_per_mmol
            * self.glucose_total_uptake_rate(state, q)
        )

    def cumulative_base_feed(self, times: np.ndarray, states: np.ndarray) -> np.ndarray:
        out = []
        for time, state in zip(times, states, strict=True):
            sample_volume = sum(
                SAMPLE_VOLUME_L
                for sample_time in self.schedule.sample_times
                if sample_time < time
            )
            bolus_volume = sum(
                volume
                for bolus_time, volume in self.schedule.bolus_feeds
                if bolus_time < time
            )
            out.append(
                state[STATE_INDEX["volume"]]
                - INITIAL_VOLUME
                - float(self.cum_conti_feed(time))
                - bolus_volume
                + sample_volume
            )
        return np.asarray(out, dtype=float)

    def _evaluate_rates(self, t, state, controls=None, *, xp):
        state = xp.asarray(state, dtype=float)
        if controls is None:
            ph = xp.asarray(t, dtype=float) * 0.0 + PH_NOMINAL
            temp = xp.asarray(t, dtype=float) * 0.0 + TEMPERATURE_NOMINAL
        else:
            controls = xp.asarray(controls, dtype=float)
            ph = controls[CONTROL_INDEX["pH"]]
            temp = controls[CONTROL_INDEX["temperature"]]

        ratio = xp.clip(
            state[STATE_INDEX["intracellular_product_ratio"]],
            RATIO_MIN,
            RATIO_MAX,
        )
        glucose = xp.maximum(state[STATE_INDEX["glucose"]], 0.0)
        glutamine = xp.maximum(state[STATE_INDEX["glutamine"]], 0.0)

        control_factor = xp.exp(-0.5 * ((ph - PH_NOMINAL) / PH_WIDTH) ** 2)
        control_factor *= xp.exp(
            -0.5 * ((temp - TEMPERATURE_NOMINAL) / TEMPERATURE_WIDTH) ** 2
        )
        glucose_limit = glucose / (GLUCOSE_HALF_SATURATION + glucose)
        glutamine_limit = glutamine / (GLUTAMINE_HALF_SATURATION + glutamine)
        substrate_factor = glucose_limit * glutamine_limit

        # pH/temperature modulation follows the Martens virtual_lab idea.
        # Substrate limitation is multiplied in too; otherwise uptake would
        # taper at depletion while growth/product formation stayed high.
        mu = MU_MAX_PER_H * control_factor * substrate_factor
        q_product = PRODUCT_RATE_PER_H * control_factor * substrate_factor

        q_product_intracellular = q_product * ratio
        q_values = {
            # `biomass` is measured viable biomass and includes intracellular
            # product. Active biomass excludes intracellular product only as the
            # rate basis.
            "biomass": mu - DEATH_RATE_PER_H + q_product_intracellular,
            "product_extracellular": q_product * (1.0 - ratio),
            "product_intracellular": q_product_intracellular,
            "dead_cells": DEATH_RATE_PER_H,
            "glucose": -GLUCOSE_UPTAKE_RATE * glucose_limit,
            "glutamine": -GLUTAMINE_UPTAKE_RATE * glutamine_limit,
            "lactate": LACTATE_PRODUCTION_RATE,
            "ammonia": AMMONIA_PRODUCTION_RATE,
        }
        q = xp.asarray([q_values[name] for name in KINETIC_REACTOR_NAMES], dtype=float)
        ratio_derivative = RATIO_RELAXATION_PER_H * (RATIO_TARGET - ratio)
        r = xp.zeros(len(KINETIC_REACTOR_NAMES) + len(PROCESS_VARIABLE_STATE_NAMES))
        ratio_r_index = (
            len(KINETIC_REACTOR_NAMES)
            + PROCESS_VARIABLE_INDEX["intracellular_product_ratio"]
        )
        if xp is jnp:
            r = r.at[ratio_r_index].set(ratio_derivative)
        else:
            r[ratio_r_index] = ratio_derivative
        return q, r

    def run(self, output_dir: str | Path | None = None) -> SimulationResult:
        """Integrate piecewise, apply events, and write deterministic CSVs."""
        events = self.events()
        event_times = sorted({event.time for event in events})
        state_times = np.asarray(
            sorted(set(self.schedule.online_times) | set(event_times)),
            dtype=float,
        )
        # One full state (kinetic core + tracers) and one event set drive both the
        # solve and the serialized result, so the tracers are mixed at events,
        # written as state columns, and parsed like any other reactor-medium species.
        states = self._integrate_left_continuous(state_times, events)
        extras = self._extra_columns(state_times, states)
        result = self.build_result(
            process=self.process_id,
            state_times=state_times,
            states=states,
            online_times=self.schedule.online_times,
            state_names=self.state_names,
            reactor_state_names=self.reactor_state_names,
            events=events,
            extra_columns=extras,
        )
        result = self._with_sim_1_output_schema(result)
        if output_dir is not None:
            write_simulation_csvs((result,), output_dir)
        return result

    def events(self) -> list[SimulationEvent]:
        # Bolus feed composition over the full reactor medium: nutrients plus the
        # fed inert tracer (the unfed tracer is never fed -> 0). The same events
        # drive the solve and the serialized result, so these tracer feed
        # concentrations are integrated, written to the events CSV, and parsed.
        feed = {name: 0.0 for name in REACTOR_MEDIUM_NAMES}
        feed["glucose"] = FEED_GLUCOSE
        feed["glutamine"] = FEED_GLUTAMINE
        feed[TRACER_FED_NAME] = TRACER_FED_FEED_CONCENTRATION
        events: list[SimulationEvent] = []
        for time in self.schedule.sample_times:
            events.append(
                SimulationEvent(
                    self.process_id,
                    time,
                    EVENT_TYPE_SAMPLE,
                    -SAMPLE_VOLUME_L,
                )
            )
        for time, volume in self.schedule.bolus_feeds:
            events.append(
                SimulationEvent(
                    self.process_id,
                    time,
                    EVENT_TYPE_BOLUS,
                    volume,
                    feed_id="bolus_feed_medium",
                    feed_concentrations=feed,
                )
            )
        events.append(
            SimulationEvent(
                self.process_id,
                T_END,
                EVENT_TYPE_FERMENTATION_END,
                0.0,
            )
        )
        return events

    def _integrate_left_continuous(
        self,
        state_times: np.ndarray,
        events: Sequence[SimulationEvent],
    ) -> np.ndarray:
        # `_rhs_numpy` always integrates the full state (real species + pv + volume +
        # the two inert tracers), so the layout is fixed: use self.initial_state /
        # self.state_names / self.reactor_state_names rather than accepting a layout.
        grouped = self.group_events(events)
        events_by_time = {time: group for (_, time), group in grouped.items()}
        out = []
        next_index = 0
        current_time = float(state_times[0])
        current_state = self.initial_state.copy()
        for event_time in sorted(events_by_time):
            stop = int(np.searchsorted(state_times, event_time, side="right"))
            segment_times = state_times[next_index:stop]
            if len(segment_times):
                segment_states = self._integrate_segment(
                    current_time,
                    float(segment_times[-1]),
                    current_state,
                    segment_times,
                )
                out.extend(segment_states)
                current_state = segment_states[-1].copy()
                current_time = float(segment_times[-1])
                next_index = stop
            if event_time > current_time:
                current_state = self._integrate_segment(
                    current_time,
                    event_time,
                    current_state,
                    [event_time],
                )[-1]
                current_time = event_time
            if event_time in events_by_time:
                current_state = self.apply_events(
                    current_state,
                    events_by_time[event_time],
                    state_names=self.state_names,
                    reactor_state_names=self.reactor_state_names,
                )
                current_time = event_time

        remaining_times = state_times[next_index:]
        if len(remaining_times):
            segment_states = self._integrate_segment(
                current_time,
                float(remaining_times[-1]),
                current_state,
                remaining_times,
            )
            out.extend(segment_states)
        return np.vstack(out)

    def _integrate_segment(
        self,
        t_start: float,
        t_end: float,
        state: np.ndarray,
        t_eval: Sequence[float],
    ) -> np.ndarray:
        t_eval_array = np.asarray(t_eval, dtype=float)
        if t_end == t_start:
            return np.repeat(state[None, :], len(t_eval_array), axis=0)

        leading_state = len(t_eval_array) > 0 and t_eval_array[0] == t_start
        solver_t_eval = t_eval_array[1:] if leading_state else t_eval_array
        sol = solve_ivp(
            self._rhs_numpy,
            (t_start, t_end),
            state,
            method="DOP853",
            rtol=1e-9,
            atol=1e-11,
            t_eval=solver_t_eval,
        )
        if not sol.success:
            raise RuntimeError(sol.message)
        states = sol.y.T
        if leading_state:
            states = np.vstack([state, states])
        return states

    def _rhs_numpy(self, t: float, state: np.ndarray) -> np.ndarray:
        q, r = self._evaluate_rates(t, state, xp=np)
        q = np.asarray(q, dtype=float)
        r = np.asarray(r, dtype=float)
        volume = state[STATE_INDEX["volume"]]
        reactor = state[: len(KINETIC_REACTOR_NAMES)]
        active_biomass = max(
            state[STATE_INDEX["biomass"]] - state[STATE_INDEX["product_intracellular"]],
            0.0,
        )

        biological = q * active_biomass + r[: len(KINETIC_REACTOR_NAMES)]

        # Continuous nutrient feed follows ex12's standard CSTR inflow term:
        # D * (feed_concentration - reactor_concentration). Base feed is pure
        # dilution with flow proportional to total glucose uptake, representing
        # volume added for pH control.
        conti_feed = np.zeros(len(KINETIC_REACTOR_NAMES), dtype=float)
        conti_feed[REACTOR_INDEX["glucose"]] = FEED_GLUCOSE
        conti_feed[REACTOR_INDEX["glutamine"]] = FEED_GLUTAMINE
        base_feed = np.zeros(len(KINETIC_REACTOR_NAMES), dtype=float)
        base_flow_l_per_h = self.base_flow_l_per_h(state, q)
        flow_term = (
            self.config.conti_flow_l_per_h * (conti_feed - reactor)
            + base_flow_l_per_h * (base_feed - reactor)
        ) / volume
        d_reactor = biological + flow_term
        d_pv = r[len(KINETIC_REACTOR_NAMES) :]
        d_volume = self.config.conti_flow_l_per_h + base_flow_l_per_h

        # Inert tracers, appended after `volume` in the full state vector. The
        # unfed tracer is pure dilution; the fed tracer adds the continuous nutrient
        # stream at TRACER_FED_FEED_CONCENTRATION (base feed carries no tracer). Both
        # use the SAME live total flow (== d_volume) and live volume, so d(c*V)/dt is
        # exactly 0 (unfed) / conti_flow*c_feed (fed) at every solver substep.
        c_unfed, c_fed = state[TRACER_UNFED_INDEX], state[TRACER_FED_INDEX]
        total_flow = d_volume
        d_tracer_unfed = -(total_flow / volume) * c_unfed
        d_tracer_fed = (
            self.config.conti_flow_l_per_h * TRACER_FED_FEED_CONCENTRATION
        ) / volume - (total_flow / volume) * c_fed
        return np.concatenate(
            [d_reactor, d_pv, [d_volume], [d_tracer_unfed, d_tracer_fed]]
        )

    def _with_sim_1_output_schema(self, result: SimulationResult) -> SimulationResult:
        dense_rows = []
        rate_rows = self._build_rate_rows(result)
        for row, rate_row in zip(result.dense_rows, rate_rows, strict=True):
            dense_row = dict(row)
            dense_row.update({name: rate_row[name] for name in RATE_COLUMNS})
            dense_rows.append(dense_row)

        return SimulationResult(
            process=result.process,
            times=result.times,
            states=result.states,
            state_names=result.state_names,
            reactor_state_names=result.reactor_state_names,
            dense_rows=dense_rows,
            event_rows=result.event_rows,
            row_columns=(*result.row_columns, *RATE_COLUMNS),
            event_columns=result.event_columns,
        )

    def _build_rate_rows(self, result: SimulationResult) -> list[dict]:
        n_reactor = len(KINETIC_REACTOR_NAMES)
        rows: list[dict] = []
        for row in result.dense_rows:
            state = np.asarray([row[name] for name in KINETIC_STATE_NAMES], dtype=float)
            controls = np.asarray(
                [0.0, row["pH"], row["temperature"]],
                dtype=float,
            )
            q, r = self._evaluate_rates(row["time"], state, controls=controls, xp=np)
            q = np.asarray(q, dtype=float)
            r = np.asarray(r, dtype=float)
            out: dict = {
                "glucose_total_uptake_rate": self.glucose_total_uptake_rate(state, q),
                "base_flow_l_per_h": self.base_flow_l_per_h(state, q),
            }
            for index, name in enumerate(KINETIC_REACTOR_NAMES):
                out[f"q_{name}"] = float(q[index])
            for index, name in enumerate(KINETIC_REACTOR_NAMES):
                out[f"r_{name}"] = float(r[index])
            for index, name in enumerate(PROCESS_VARIABLE_STATE_NAMES):
                out[f"r_{name}"] = float(r[n_reactor + index])
            rows.append(out)
        return rows

    def _extra_columns(
        self,
        times: np.ndarray,
        states: np.ndarray,
    ) -> dict[str, np.ndarray]:
        return {
            "pH": np.asarray([float(self.pH(t)) for t in times]),
            "temperature": np.asarray([float(self.temperature(t)) for t in times]),
            "cum_base_feed": self.cumulative_base_feed(times, states),
            "cum_conti_feed": np.asarray(
                [float(self.cum_conti_feed(t)) for t in times]
            ),
        }


def run_all_default(output_dir: str | Path | None = None) -> list[SimulationResult]:
    """Run all default sim 1 processes and optionally write combined CSVs."""
    results = [Sim1Simulation(config).run() for config in DEFAULT_CONFIGS]
    if output_dir is not None:
        write_simulation_csvs(results, output_dir)
    return results


def write_simulation_csvs(
    results: Sequence[SimulationResult],
    output_dir: str | Path,
) -> None:
    """Write sim 1 dense/event CSVs for one or more simulation results."""
    if not results:
        raise ValueError("cannot write CSVs for an empty result sequence")

    path = Path(output_dir)
    path.mkdir(parents=True, exist_ok=True)
    _write_csv_lf(
        path / "simulation_dense_output.csv",
        results[0].row_columns,
        (row for result in results for row in result.dense_rows),
    )
    _write_csv_lf(
        path / "simulation_events.csv",
        results[0].event_columns,
        (row for result in results for row in result.event_rows),
    )


def _write_csv_lf(path: Path, columns: Sequence[str], rows: Iterable[dict]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _plotting_pyplot():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def _result_label(result: SimulationResult) -> str:
    return str(result.dense_rows[0]["process_id"])


def _result_frame(result: SimulationResult) -> dict[str, np.ndarray]:
    rows = [row for row in result.dense_rows if row["row_type"] != "offline"]
    return {
        name: np.asarray([float(row[name]) for row in rows], dtype=float)
        for name in result.row_columns
        if name not in {"process_id", "row_type"}
    }


def _event_times(result: SimulationResult, event_type: str) -> list[float]:
    return [
        float(row["time"])
        for row in result.event_rows
        if row["event_type"] == event_type
    ]


def _add_event_markers(ax, result: SimulationResult) -> None:
    styles = {
        "sample": ("tab:pink", ":", 1.4, 0.8),
        "bolus": ("tab:blue", "--", 0.8, 0.5),
        "fermentation_end": ("black", "-", 0.8, 0.5),
    }
    for event_type, (color, linestyle, linewidth, alpha) in styles.items():
        for time in _event_times(result, event_type):
            ax.axvline(
                time,
                color=color,
                linestyle=linestyle,
                linewidth=linewidth,
                alpha=alpha,
            )


def _save_panel_plot(
    results: Sequence[SimulationResult],
    output_dir: str | Path,
    filename: str,
    panels: Sequence[tuple[str, str]],
    *,
    ylabel: str = "value",
    mark_events: bool = False,
    n_cols: int = 2,
) -> Path:
    plt = _plotting_pyplot()
    path = Path(output_dir) / filename
    path.parent.mkdir(parents=True, exist_ok=True)
    n_rows = max(1, (len(panels) + n_cols - 1) // n_cols)
    fig, axes = plt.subplots(
        n_rows, n_cols, sharex=True, figsize=(5 * n_cols, 2.7 * n_rows)
    )
    axes = np.asarray(axes).ravel()
    for ax, (column, title) in zip(axes, panels, strict=False):
        for result in results:
            frame = _result_frame(result)
            ax.plot(frame["time"], frame[column], label=_result_label(result))
            ymin, ymax = ax.get_ylim()
            if ymin <= 0.0 <= ymax:
                ax.axhline(0.0, color="black", linewidth=0.6, alpha=0.35)
                ax.set_ylim(ymin, ymax)
            if mark_events:
                _add_event_markers(ax, result)
        ax.set_title(title)
        ax.set_ylabel(ylabel)
        ax.grid(alpha=0.25)
    for ax in axes[len(panels) :]:
        ax.set_visible(False)
    for ax in axes[-n_cols:]:
        ax.set_xlabel("time [h]")
    handles, labels = axes[0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="lower center", ncol=len(handles))
        fig.tight_layout(rect=(0, 0.05, 1, 1))
    else:
        fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def plot_reactor_states_and_volumes(
    results: Sequence[SimulationResult],
    output_dir: str | Path,
) -> Path:
    panels = [
        *(
            (name, f"{name} [{REACTOR_STATE_UNITS[name]}]")
            for name in KINETIC_REACTOR_NAMES
        ),
        ("volume", "reactor volume [L]"),
        ("cum_conti_feed", "continuous feed [L]"),
        ("cum_base_feed", "base feed [L]"),
        ("cum_bolus_feed", "bolus feed [L]"),
    ]
    return _save_panel_plot(
        results,
        output_dir,
        "reactor_states_and_volumes.png",
        panels,
        ylabel="value",
        mark_events=True,
        n_cols=3,
    )


def plot_process_variables(
    results: Sequence[SimulationResult],
    output_dir: str | Path,
) -> Path:
    panels = [
        (name, name) for name in ("pH", "temperature", *PROCESS_VARIABLE_STATE_NAMES)
    ]
    return _save_panel_plot(
        results,
        output_dir,
        "process_variables.png",
        panels,
        mark_events=True,
    )


def plot_rates(
    results: Sequence[SimulationResult],
    output_dir: str | Path,
) -> Path:
    panels = [
        *((f"q_{name}", f"q_{name}") for name in KINETIC_REACTOR_NAMES),
        ("glucose_total_uptake_rate", "glucose_total_uptake_rate [mmol/h]"),
        ("base_flow_l_per_h", "base_flow_l_per_h [L/h]"),
        *(
            (f"r_{name}", f"r_{name}")
            for name in (*KINETIC_REACTOR_NAMES, *PROCESS_VARIABLE_STATE_NAMES)
        ),
    ]
    return _save_panel_plot(
        results,
        output_dir,
        "rates.png",
        panels,
        ylabel="rate",
        mark_events=True,
    )


def write_simulation_plots(
    output_dir: str | Path,
    results: Sequence[SimulationResult] | None = None,
) -> list[Path]:
    if results is None:
        results = run_all_default()
    output_dir = Path(output_dir)
    return [
        plot_reactor_states_and_volumes(results, output_dir),
        plot_process_variables(results, output_dir),
        plot_rates(results, output_dir),
    ]


def _arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parent,
        help="Directory for simulation_dense_output.csv and simulation_events.csv.",
    )
    parser.add_argument(
        "--plot-dir",
        type=Path,
        default=None,
        help="Directory for simulation plots.",
    )
    parser.add_argument("--skip-plots", action="store_true")
    return parser


def main() -> None:
    args = _arg_parser().parse_args()
    results = run_all_default(output_dir=args.output_dir)
    dense_count = sum(len(result.dense_rows) for result in results)
    event_count = sum(len(result.event_rows) for result in results)
    print(f"Wrote {dense_count} dense rows to {args.output_dir}")
    print(f"Wrote {event_count} event rows to {args.output_dir}")

    if args.skip_plots:
        return
    plot_dir = args.plot_dir or args.output_dir / "output" / "sim_plots"
    for path in write_simulation_plots(plot_dir, results):
        print(f"Wrote plot {path}")


if __name__ == "__main__":
    main()
