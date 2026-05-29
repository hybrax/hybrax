import numpy as np
import pytest

from bp_format import Simulation, SimulationEvent, SimulationResult
from bp_format.simulation import (
    EVENT_TYPE_BOLUS,
    EVENT_TYPE_SAMPLE,
    ROW_TYPE_OFFLINE,
    ROW_TYPE_ONLINE,
    ROW_TYPE_POST_EVENT,
    ROW_TYPE_PRE_EVENT,
)


STATE_NAMES = ("biomass", "glucose", "pH", "volume")
REACTOR_STATE_NAMES = ("biomass", "glucose")


def _sim():
    return Simulation()


def _state(biomass=10.0, glucose=20.0, ph=7.0, volume=1.0):
    return np.asarray([biomass, glucose, ph, volume], dtype=float)


def test_sampling_changes_volume_only():
    sim = _sim()
    event = SimulationEvent("run_1", 1.0, EVENT_TYPE_SAMPLE, -0.25)

    post = sim.apply_events(
        _state(),
        [event],
        state_names=STATE_NAMES,
        reactor_state_names=REACTOR_STATE_NAMES,
    )

    np.testing.assert_allclose(post, [10.0, 20.0, 7.0, 0.75])


def test_bolus_mixes_reactor_states_and_preserves_process_variables():
    sim = _sim()
    event = SimulationEvent(
        "run_1",
        1.0,
        EVENT_TYPE_BOLUS,
        0.5,
        feed_id="feed_a",
        feed_concentrations={"biomass": 0.0, "glucose": 80.0},
    )

    post = sim.apply_events(
        _state(),
        [event],
        state_names=STATE_NAMES,
        reactor_state_names=REACTOR_STATE_NAMES,
    )

    np.testing.assert_allclose(post, [10.0 / 1.5, 40.0, 7.0, 1.5])


def test_simultaneous_sample_then_bolus_ordering():
    sim = _sim()
    events = [
        SimulationEvent(
            "run_1",
            1.0,
            EVENT_TYPE_BOLUS,
            0.5,
            feed_id="feed_a",
            feed_concentrations={"glucose": 80.0},
        ),
        SimulationEvent("run_1", 1.0, EVENT_TYPE_SAMPLE, -0.25),
    ]

    post = sim.apply_events(
        _state(),
        events,
        state_names=STATE_NAMES,
        reactor_state_names=REACTOR_STATE_NAMES,
    )

    # Sample removes volume without changing concentration, then bolus mixes.
    np.testing.assert_allclose(post, [6.0, 44.0, 7.0, 1.25])


def test_duplicate_same_kind_event_at_same_timestamp_is_rejected():
    sim = _sim()
    events = [
        SimulationEvent("run_1", 1.0, EVENT_TYPE_SAMPLE, -0.1),
        SimulationEvent("run_1", 1.0, EVENT_TYPE_SAMPLE, -0.2),
    ]

    with pytest.raises(ValueError, match="At most one sample and one bolus"):
        sim.group_events(events)


def test_dense_row_ordering_including_start_and_end_events():
    sim = _sim()
    result = sim.build_result(
        process="run_1",
        state_times=[0.0, 1.0, 2.0],
        states=[
            _state(volume=1.0),
            _state(biomass=12.0, glucose=18.0, volume=1.1),
            _state(biomass=14.0, glucose=16.0, volume=1.2),
        ],
        online_times=[0.0, 1.0, 2.0],
        state_names=STATE_NAMES,
        reactor_state_names=REACTOR_STATE_NAMES,
        events=[
            SimulationEvent("run_1", 0.0, EVENT_TYPE_SAMPLE, -0.1),
            SimulationEvent(
                "run_1",
                1.0,
                EVENT_TYPE_BOLUS,
                0.1,
                feed_id="feed_a",
                feed_concentrations={"glucose": 120.0},
            ),
            SimulationEvent("run_1", 2.0, EVENT_TYPE_SAMPLE, -0.2),
        ],
    )

    assert [row["row_type"] for row in result.dense_rows] == [
        ROW_TYPE_ONLINE,
        ROW_TYPE_OFFLINE,
        ROW_TYPE_PRE_EVENT,
        ROW_TYPE_POST_EVENT,
        ROW_TYPE_ONLINE,
        ROW_TYPE_PRE_EVENT,
        ROW_TYPE_POST_EVENT,
        ROW_TYPE_ONLINE,
        ROW_TYPE_OFFLINE,
        ROW_TYPE_PRE_EVENT,
        ROW_TYPE_POST_EVENT,
    ]
    assert result.dense_rows[1]["volume"] == pytest.approx(1.0)
    assert result.dense_rows[2]["volume"] == pytest.approx(1.0)
    assert result.dense_rows[3]["volume"] == pytest.approx(0.9)
    assert result.dense_rows[-2]["volume"] == pytest.approx(1.2)
    assert result.dense_rows[-1]["volume"] == pytest.approx(1.0)


def test_sampling_event_adds_offline_equal_to_pre_event_for_states():
    sim = _sim()
    result = sim.build_result(
        process="run_1",
        state_times=[0.0, 1.0],
        states=[_state(), _state(biomass=11.0, glucose=19.0, volume=0.95)],
        online_times=[0.0],
        state_names=STATE_NAMES,
        reactor_state_names=REACTOR_STATE_NAMES,
        events=[SimulationEvent("run_1", 1.0, EVENT_TYPE_SAMPLE, -0.1)],
    )

    offline = result.dense_rows[1]
    pre_event = result.dense_rows[2]
    assert offline["row_type"] == ROW_TYPE_OFFLINE
    assert pre_event["row_type"] == ROW_TYPE_PRE_EVENT
    for state_name in STATE_NAMES:
        assert offline[state_name] == pytest.approx(pre_event[state_name])


def test_events_table_schema_order():
    sim = _sim()
    result = sim.build_result(
        process="run_1",
        state_times=[0.0, 1.0],
        states=[_state(), _state(volume=0.9)],
        online_times=[0.0, 1.0],
        state_names=STATE_NAMES,
        reactor_state_names=REACTOR_STATE_NAMES,
        events=[
            SimulationEvent("run_1", 1.0, EVENT_TYPE_SAMPLE, -0.1),
            SimulationEvent(
                "run_1",
                1.0,
                EVENT_TYPE_BOLUS,
                0.2,
                feed_id="feed_a",
                feed_concentrations={"glucose": 100.0},
            ),
        ],
    )

    assert isinstance(result, SimulationResult)
    assert result.event_columns == (
        "process",
        "time",
        "event_order",
        "event_type",
        "delta_volume",
        "feed_id",
        "feed_biomass",
        "feed_glucose",
    )
    assert [row["event_type"] for row in result.event_rows] == [
        EVENT_TYPE_SAMPLE,
        EVENT_TYPE_BOLUS,
    ]
    assert [row["event_order"] for row in result.event_rows] == [0, 1]
    assert result.event_rows[1]["feed_glucose"] == pytest.approx(100.0)


def test_build_result_scopes_event_rows_to_process():
    sim = _sim()
    result = sim.build_result(
        process="run_1",
        state_times=[0.0, 1.0, 2.0],
        states=[_state(), _state(volume=0.9), _state(volume=0.8)],
        online_times=[0.0],
        state_names=STATE_NAMES,
        reactor_state_names=REACTOR_STATE_NAMES,
        events=[
            SimulationEvent("run_2", 1.0, EVENT_TYPE_SAMPLE, -0.1),
            SimulationEvent("run_1", 2.0, EVENT_TYPE_SAMPLE, -0.2),
        ],
    )

    assert {row["process"] for row in result.dense_rows} == {"run_1"}
    assert {row["process"] for row in result.event_rows} == {"run_1"}
    assert [row["time"] for row in result.event_rows] == [2.0]


def test_cum_bolus_feed_updates_after_post_event_and_persists():
    sim = _sim()
    result = sim.build_result(
        process="run_1",
        state_times=[0.0, 1.0, 2.0],
        states=[_state(), _state(volume=1.0), _state(volume=1.2)],
        online_times=[0.0, 1.0, 2.0],
        state_names=STATE_NAMES,
        reactor_state_names=REACTOR_STATE_NAMES,
        events=[
            SimulationEvent(
                "run_1",
                1.0,
                EVENT_TYPE_BOLUS,
                0.2,
                feed_id="feed_a",
                feed_concentrations={"glucose": 100.0},
            )
        ],
    )

    assert [
        (row["time"], row["row_type"], row["cum_bolus_feed"])
        for row in result.dense_rows
    ] == [
        (0.0, ROW_TYPE_ONLINE, 0.0),
        (1.0, ROW_TYPE_ONLINE, 0.0),
        (1.0, ROW_TYPE_PRE_EVENT, 0.0),
        (1.0, ROW_TYPE_POST_EVENT, 0.2),
        (2.0, ROW_TYPE_ONLINE, 0.2),
    ]


def test_extra_dense_columns_are_repeated_for_same_time_rows():
    sim = _sim()
    result = sim.build_result(
        process="run_1",
        state_times=[0.0, 1.0],
        states=[_state(), _state(volume=0.9)],
        online_times=[0.0, 1.0],
        state_names=STATE_NAMES,
        reactor_state_names=REACTOR_STATE_NAMES,
        events=[SimulationEvent("run_1", 1.0, EVENT_TYPE_SAMPLE, -0.1)],
        extra_columns={
            "temperature": [36.5, 37.0],
            "cum_base_feed": [0.0, 0.03],
        },
    )

    assert result.row_columns[-2:] == ("temperature", "cum_base_feed")
    time_one_rows = [row for row in result.dense_rows if row["time"] == 1.0]
    assert [row["row_type"] for row in time_one_rows] == [
        ROW_TYPE_ONLINE,
        ROW_TYPE_OFFLINE,
        ROW_TYPE_PRE_EVENT,
        ROW_TYPE_POST_EVENT,
    ]
    assert {row["temperature"] for row in time_one_rows} == {37.0}
    assert {row["cum_base_feed"] for row in time_one_rows} == {0.03}
