"""Semantic contract tests for the sim_1 dense/event fixture.

These pin the fixture's data contract directly (not via the carrier math or the
artifact SHA): that event mixing is applied to the inert tracers like any real
species, and that the parsed JSON carries the tracers as reactor-medium components.
Both are intentional consequences of modelling the tracers as real species; testing
them here makes a regression fail semantically instead of only as an opaque hash diff.
"""

import os

os.environ.setdefault("JAX_ENABLE_X64", "true")

import bp_format as bp  # noqa: E402
import numpy as np  # noqa: E402
from bp_format.serialization import load_process_collection  # noqa: E402

from .cstar_helpers import (  # noqa: E402
    EXPECTED_PROCESS_IDS,
    EXPECTED_REACTOR_COMPONENT_ORDER,
    SIM_RESULTS_DIR,
)
from .simulation import run_all_default  # noqa: E402

DATA_JSON = SIM_RESULTS_DIR / "process_collection.json"
TRACER_NAMES = ("tracer_unfed", "tracer_fed")
MIXING_RTOL = 1e-12
MIXING_ATOL = 1e-12


def test_sim_1_post_event_tracer_mixing():
    """Post-event tracer rows are event-mixed, not stale copies of the pre-event row.

    This is the property the "tracers as real species" change exists to guarantee.
    For each event timestamp it reconstructs the expected post-event tracer
    concentration independently from the pre-event row and the event rows
    (sample preserves concentration; bolus dilution-mixes with the feed value) and
    compares it to the serialized post-event row.
    """
    results = run_all_default()
    checked_bolus = 0
    for result in results:
        events_by_time: dict[float, list[dict]] = {}
        for event_row in result.event_rows:
            events_by_time.setdefault(float(event_row["time"]), []).append(event_row)
        pre_rows = {
            float(row["time"]): row
            for row in result.dense_rows
            if row["row_type"] == "pre-event"
        }
        post_rows = {
            float(row["time"]): row
            for row in result.dense_rows
            if row["row_type"] == "post-event"
        }
        assert post_rows, "fixture must emit post-event rows"

        for time, post_row in post_rows.items():
            pre_row = pre_rows[time]
            group = sorted(
                events_by_time[time],
                key=lambda event_row: int(event_row["event_order"]),
            )
            concentrations = {name: float(pre_row[name]) for name in TRACER_NAMES}
            volume = float(pre_row["volume"])
            for event_row in group:
                event_type = event_row["event_type"]
                if event_type == "fermentation_end":
                    continue
                delta_volume = float(event_row["delta_volume"])
                new_volume = volume + delta_volume
                if event_type == "sample":
                    volume = new_volume  # sampling preserves concentration
                    continue
                # bolus: concentration-dilution mixing, like every real species
                for name in TRACER_NAMES:
                    feed_concentration = float(event_row[f"feed_{name}"])
                    concentrations[name] = (
                        concentrations[name] * volume
                        + feed_concentration * delta_volume
                    ) / new_volume
                volume = new_volume
                checked_bolus += 1
            for name, expected in concentrations.items():
                np.testing.assert_allclose(
                    float(post_row[name]), expected, rtol=MIXING_RTOL, atol=MIXING_ATOL
                )
            np.testing.assert_allclose(
                float(post_row["volume"]), volume, rtol=MIXING_RTOL, atol=MIXING_ATOL
            )
    assert checked_bolus > 0, "expected at least one bolus event to exercise mixing"


def test_sim_1_json_carries_tracer_components():
    """The parsed JSON carries the tracers as reactor-medium components (by design).

    Makes the accepted contract change explicit instead of buried in the JSON diff:
    both tracers are reactor-medium components with a concentration TimeSeries on the
    same sparse measurement grid as the real species, and a zero biological derivative.
    """
    collection = load_process_collection(DATA_JSON)
    assert set(collection.processes) == EXPECTED_PROCESS_IDS
    for process in collection.processes.values():
        components = process.reactor_medium.components
        real_times = np.asarray(
            components[EXPECTED_REACTOR_COMPONENT_ORDER[0]].concentration.times,
            dtype=float,
        )
        for tracer in TRACER_NAMES:
            assert tracer in components
            concentration = components[tracer].concentration
            assert isinstance(concentration, bp.TimeSeries)
            np.testing.assert_array_equal(
                np.asarray(concentration.times, dtype=float), real_times
            )
            assert process.biological_ode.derivatives[tracer] == "0"
