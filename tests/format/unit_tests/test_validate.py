"""
Tests for bp_format.validate validation functions
"""

from types import SimpleNamespace

import pytest
import jax.numpy as jnp

from bp_format import (
    BiologicalOde,
    DiscreteEvents,
    TimeSeries,
    StaticVariable,
    BioProcessMetadata,
    TimeAxis,
    ProcessVariable,
    ReactorMediumComponent,
    FeedMediumComponent,
    FeedMedium,
    ReactorMedium,
    Inflow,
    Outflow,
    Volume,
    BioProcess,
    AugmentedBioProcess,
    BioProcessCollection,
    validate_time_axis,
    validate_timeseries_shape,
    validate_discrete_events,
    validate_timestamp_bounds,
    validate_volume_change_sign,
    validate_volume_change_states,
    validate_volume_units,
    validate_outflow_retention,
    validate_biomass_in_reactor_medium,
    validate_initial_state_alignment,
    validate_measurement_sampling_alignment,
    validate_process,
    validate_volume_consistency,
    validate_for_publication,
    validate_cross_process_consistency,
    validate_bounds_against_data,
    validate_augmented_parent_refs,
    validate_biological_ode,
    validate_bounds,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _ts(timepoints, values):
    """Shorthand for building a TimeSeries."""
    return TimeSeries(times=jnp.array(timepoints), values=jnp.array(values))


def _make_process(
    reactor_components=None,
    volume_changes=None,
    process_variables=None,
):
    """Build a minimal BioProcess for testing."""
    reactor_components = reactor_components or {}
    volume_changes = volume_changes or {}
    process_variables = process_variables or {}

    return BioProcess(
        metadata=BioProcessMetadata(name="test", process_type="batch"),
        time_axis=TimeAxis(
            unit="hours", start=0.0, end=10.0, time_reference="inoculation"
        ),
        volume=Volume(initial_volume=1.0, unit="L", volume_changes=volume_changes),
        reactor_medium=ReactorMedium(
            name="medium",
            density=1.0,
            density_unit="kg/L",
            components=reactor_components,
        ),
        process_variables=process_variables,
    )


def _find(results, contains: str):
    """Return the single (ok, message) entry whose message contains *contains*."""
    matches = [r for r in results if contains in r[1]]
    assert len(matches) == 1, f"expected exactly one match for {contains!r}, got {matches}"
    return matches[0]


# ---------------------------------------------------------------------------
# validate_discrete_events
# ---------------------------------------------------------------------------


class TestValidateDiscreteEvents:
    def test_no_events(self):
        ok, msg = validate_discrete_events(_make_process())

        assert ok is True
        assert "skipped" in msg

    def test_valid_events_include_time_axis_boundaries(self):
        process = _make_process()
        process.discrete_events = DiscreteEvents(
            times=jnp.array([0.0, 5.0, 10.0]),
            labels=["start", "middle", "end"],
        )

        ok, msg = validate_discrete_events(process)

        assert ok is True
        assert msg.startswith("PASS discrete_events:")

    def test_times_must_be_one_dimensional(self):
        process = _make_process()
        process.discrete_events = DiscreteEvents(times=jnp.array([[1.0, 2.0]]))

        ok, msg = validate_discrete_events(process)

        assert ok is False
        assert "times must be 1-D" in msg

    @pytest.mark.parametrize("times", [[2.0, 1.0], [1.0, 1.0]])
    def test_times_must_be_strictly_increasing(self, times):
        process = _make_process()
        process.discrete_events = DiscreteEvents(times=jnp.array(times))

        ok, msg = validate_discrete_events(process)

        assert ok is False
        assert "strictly monotonically increasing" in msg

    def test_times_must_be_within_process_time_axis(self):
        process = _make_process()
        process.discrete_events = DiscreteEvents(times=jnp.array([-1.0, 11.0]))

        ok, msg = validate_discrete_events(process)

        assert ok is False
        assert "2 timestamp(s) outside [0.0, 10.0]" in msg

    def test_float32_rounding_at_boundary_is_allowed(self):
        process = _make_process()
        process.time_axis.end = 319.85985985985985
        process.discrete_events = DiscreteEvents(times=jnp.array([319.85986328125]))

        ok, _ = validate_discrete_events(process)

        assert ok is True

    @pytest.mark.parametrize("labels", [["only one"], ["a", "b", "c"]])
    def test_labels_must_match_times_length(self, labels):
        process = _make_process()
        process.discrete_events = DiscreteEvents(
            times=jnp.array([1.0, 2.0]), labels=labels
        )

        ok, msg = validate_discrete_events(process)

        assert ok is False
        assert f"labels length ({len(labels)}) does not match times length (2)" in msg


# ---------------------------------------------------------------------------
# validate_timeseries_shape
# ---------------------------------------------------------------------------


class TestValidateTimeSeriesShape:
    def test_valid_timeseries(self):
        ts = _ts([0.0, 1.0, 2.0], [1.0, 2.0, 3.0])
        ok, msg = validate_timeseries_shape(ts, name="glucose")
        assert ok is True
        assert msg.startswith("PASS timeseries_shape:")

    def test_single_point_is_valid(self):
        ts = _ts([0.0], [1.0])
        ok, msg = validate_timeseries_shape(ts)
        assert ok is True

    def test_empty_timeseries_is_invalid(self):
        ok, msg = validate_timeseries_shape(_ts([], []))

        assert ok is False
        assert "must not be empty" in msg

    def test_empty_timeseries_can_be_allowed(self):
        ok, msg = validate_timeseries_shape(_ts([], []), allow_empty=True)

        assert ok is True
        assert msg.startswith("PASS timeseries_shape:")

    def test_unordered_timepoints(self):
        ts = SimpleNamespace(
            times=jnp.array([0.0, 2.0, 1.0]),
            values=jnp.array([1.0, 2.0, 3.0]),
        )
        ok, msg = validate_timeseries_shape(ts)
        assert ok is False
        assert "strictly monotonically increasing" in msg

    def test_duplicate_timepoints(self):
        ts = SimpleNamespace(
            times=jnp.array([0.0, 1.0, 1.0]),
            values=jnp.array([1.0, 2.0, 3.0]),
        )
        ok, msg = validate_timeseries_shape(ts)
        assert ok is False
        assert "strictly monotonically increasing" in msg

    def test_length_mismatch(self):
        ts = SimpleNamespace(
            times=jnp.array([0.0, 1.0, 2.0]),
            values=jnp.array([1.0, 2.0]),
        )
        ok, msg = validate_timeseries_shape(ts)
        assert ok is False
        assert "times length (3) does not match values length (2)" in msg

    def test_name_appears_in_message(self):
        ts = _ts([0.0, 1.0], [1.0, 2.0])
        _, msg = validate_timeseries_shape(ts, name="myvar")
        assert "myvar" in msg


# ---------------------------------------------------------------------------
# validate_time_axis
# ---------------------------------------------------------------------------


class TestValidateTimeAxis:
    def test_equal_start_and_end_is_valid(self):
        process = _make_process()
        process.time_axis.end = process.time_axis.start

        ok, msg = validate_time_axis(process)

        assert ok is True
        assert msg.startswith("PASS time_axis:")

    def test_start_after_end_is_invalid(self):
        process = _make_process()
        process.time_axis.start = 11.0

        ok, msg = validate_time_axis(process)

        assert ok is False
        assert "start 11.0 is after end 10.0 hours" in msg


# ---------------------------------------------------------------------------
# validate_timestamp_bounds
# ---------------------------------------------------------------------------


class TestValidateTimestampBounds:
    def test_inclusive_bounds_are_valid(self):
        process = _make_process(
            reactor_components={
                "biomass": ReactorMediumComponent(
                    name="biomass",
                    unit="g/L",
                    concentration=_ts([0.0, 10.0], [1.0, 2.0]),
                )
            }
        )

        ok, msg = validate_timestamp_bounds(process)

        assert ok is True
        assert "[0.0, 10.0] hours" in msg

    def test_reactor_component_timestamp_outside_bounds(self):
        process = _make_process(
            reactor_components={
                "biomass": ReactorMediumComponent(
                    name="biomass",
                    unit="g/L",
                    concentration=_ts([-1.0, 11.0], [1.0, 2.0]),
                )
            }
        )

        ok, msg = validate_timestamp_bounds(process)

        assert ok is False
        assert "reactor component 'biomass': 2 timestamp(s) outside" in msg

    def test_float32_rounding_at_boundary_is_allowed(self):
        process = _make_process(
            reactor_components={
                "biomass": ReactorMediumComponent(
                    name="biomass",
                    unit="g/L",
                    concentration=_ts([319.85986328125], [1.0]),
                )
            }
        )
        process.time_axis.end = 319.85985985985985

        ok, _ = validate_timestamp_bounds(process)

        assert ok is True

    def test_process_variable_timestamp_outside_bounds(self):
        variable = ProcessVariable(
            name="temperature",
            unit="degC",
            is_controlled=True,
            values=_ts([1.0, 11.0], [30.0, 31.0]),
        )
        process = _make_process(process_variables={"temperature": variable})

        ok, msg = validate_timestamp_bounds(process)

        assert ok is False
        assert "process variable 'temperature'" in msg

    def test_volume_change_timestamp_outside_bounds(self):
        change = Outflow(
            name="sample",
            unit="L",
            is_controlled=True,
            is_continuous=False,
            values=_ts([-1.0, 1.0], [-0.1, -0.1]),
        )
        process = _make_process(volume_changes={"sample": change})

        ok, msg = validate_timestamp_bounds(process)

        assert ok is False
        assert "volume change 'sample'" in msg

    def test_measured_total_volume_timestamp_outside_bounds(self):
        process = _make_process()
        process.volume.total_volume = _ts([1.0, 11.0], [1.0, 1.1])

        ok, msg = validate_timestamp_bounds(process)

        assert ok is False
        assert "measured total volume" in msg

    def test_invalid_time_axis_skips_check(self):
        process = _make_process()
        process.time_axis.start = 11.0

        ok, msg = validate_timestamp_bounds(process)

        assert ok is True
        assert msg.startswith("SKIP timestamp_bounds:")
        assert "time_axis is invalid" in msg


# ---------------------------------------------------------------------------
# validate_volume_change_sign
# ---------------------------------------------------------------------------


class TestValidateVolumeChangeSign:
    def _feed_vc(self, values, name="feed"):
        return Inflow(
            name=name,
            unit="L",
            is_controlled=True,
            is_continuous=True,
            feed_medium=FeedMedium(name="f", density=1.0, density_unit="kg/L"),
            values=_ts([0.0, 1.0], values),
        )

    def _sample_vc(self, values, name="sample"):
        return Outflow(
            name=name,
            unit="L",
            is_controlled=True,
            is_continuous=True,
            values=_ts([0.0, 1.0], values),
        )

    def test_purely_positive(self):
        vc = self._feed_vc([0.1, 0.2])
        ok, msg = validate_volume_change_sign(vc)
        assert ok is True
        assert "non-negative" in msg

    def test_purely_negative(self):
        vc = self._sample_vc([-0.1, -0.2])
        ok, msg = validate_volume_change_sign(vc)
        assert ok is True
        assert "non-positive" in msg

    def test_zero_values_are_positive(self):
        vc = self._feed_vc([0.0, 0.0])
        ok, msg = validate_volume_change_sign(vc)
        assert ok is True

    @pytest.mark.parametrize("volume_change", ["feed", "sample"])
    def test_float_noise_within_sign_tolerance(self, volume_change):
        vc = (
            self._feed_vc([-1e-13, 1.0])
            if volume_change == "feed"
            else self._sample_vc([1e-13, -1.0])
        )

        ok, _ = validate_volume_change_sign(vc)

        assert ok is True

    @pytest.mark.parametrize("volume_change", ["feed", "sample"])
    def test_values_outside_sign_tolerance_fail(self, volume_change):
        vc = (
            self._feed_vc([-1e-6, 1.0])
            if volume_change == "feed"
            else self._sample_vc([1e-6, -1.0])
        )

        ok, _ = validate_volume_change_sign(vc)

        assert ok is False

    def test_mixed_signs_invalid(self):
        vc = self._feed_vc([0.1, -0.2])
        ok, msg = validate_volume_change_sign(vc)
        assert ok is False
        assert "negative" in msg.lower()


# ---------------------------------------------------------------------------
# validate_volume_units
# ---------------------------------------------------------------------------


class TestValidateVolumeUnits:
    def _sample(self, unit):
        return Outflow(
            name="sample",
            unit=unit,
            is_controlled=True,
            is_continuous=False,
            values=_ts([1.0], [-0.1]),
        )

    def test_matching_volume_change_unit(self):
        process = _make_process(volume_changes={"sample": self._sample("L")})
        ok, msg = validate_volume_units(process)
        assert ok is True
        assert msg.startswith("PASS volume_units:")

    def test_mismatched_volume_change_unit(self):
        process = _make_process(volume_changes={"sample": self._sample("mL")})
        ok, msg = validate_volume_units(process)
        assert ok is False
        assert "'sample' uses 'mL'" in msg
        assert "volume unit 'L'" in msg


# ---------------------------------------------------------------------------
# validate_volume_change_states
# ---------------------------------------------------------------------------


class TestValidateVolumeChangeStates:
    def _reactor_comp(self, name, dynamic=True):
        conc = _ts([0.0, 1.0], [1.0, 2.0]) if dynamic else StaticVariable(value=1.0)
        return ReactorMediumComponent(name=name, unit="g/L", concentration=conc)

    def _feed_medium(self, component_names):
        comps = {
            n: FeedMediumComponent(
                name=n,
                unit="g/L",
                concentration=StaticVariable(value=10.0),
                is_controlled=True,
            )
            for n in component_names
        }
        return FeedMedium(
            name="feed", density=1.0, density_unit="kg/L", components=comps
        )

    def _vc(self, feed_medium, positive=True):
        vals = [0.1, 0.2] if positive else [-0.1, -0.2]
        if positive:
            return Inflow(
                name="feed_vc",
                unit="L",
                is_controlled=True,
                is_continuous=True,
                feed_medium=feed_medium,
                values=_ts([0.0, 1.0], vals),
            )
        else:
            return Outflow(
                name="feed_vc",
                unit="L",
                is_controlled=True,
                is_continuous=True,
                values=_ts([0.0, 1.0], vals),
            )

    def test_all_states_covered(self):
        process = _make_process(
            reactor_components={
                "biomass": self._reactor_comp("biomass"),
                "glucose": self._reactor_comp("glucose"),
            },
            volume_changes={"f": self._vc(self._feed_medium(["biomass", "glucose"]))},
        )
        ok, msg = validate_volume_change_states(process)
        assert ok is True

    def test_missing_state_in_feed_means_zero(self):
        process = _make_process(
            reactor_components={
                "biomass": self._reactor_comp("biomass"),
                "glucose": self._reactor_comp("glucose"),
            },
            volume_changes={
                "f": self._vc(self._feed_medium(["glucose"]))  # biomass omitted
            },
        )

        ok, msg = validate_volume_change_states(process)

        assert ok is True

    def test_feed_component_unit_must_match_reactor_component(self):
        feed = self._feed_medium(["biomass"])
        feed.components["biomass"].unit = "mg/mL"
        process = _make_process(
            reactor_components={"biomass": self._reactor_comp("biomass")},
            volume_changes={"f": self._vc(feed)},
        )

        ok, msg = validate_volume_change_states(process)

        assert ok is False
        assert "'biomass' uses unit 'mg/mL'" in msg
        assert "reactor medium uses 'g/L'" in msg

    def test_feed_unit_checked_with_tolerated_negative_noise(self):
        feed = self._feed_medium(["biomass"])
        feed.components["biomass"].unit = "mg/mL"
        change = self._vc(feed)
        change.values = _ts([0.0, 1.0], [-1e-13, 1.0])
        process = _make_process(
            reactor_components={"biomass": self._reactor_comp("biomass")},
            volume_changes={"f": change},
        )

        ok, msg = validate_volume_change_states(process)
        all_valid, results = validate_process(process)

        assert ok is False
        assert "'biomass' uses unit 'mg/mL'" in msg
        assert all_valid is False
        assert any("'biomass' uses unit 'mg/mL'" in message for _, message in results)

    def test_negative_volume_change_not_checked(self):
        """Negative (outflow) volume changes should skip state coverage check."""
        process = _make_process(
            reactor_components={
                "biomass": self._reactor_comp("biomass"),
            },
            volume_changes={"sample": self._vc(self._feed_medium([]), positive=False)},
        )
        ok, msg = validate_volume_change_states(process)
        assert ok is True

    def test_no_dynamic_states_skips_check(self):
        """If all reactor components are static, the check is skipped."""
        process = _make_process(
            reactor_components={
                "biomass": self._reactor_comp("biomass", dynamic=False),
            },
            volume_changes={"f": self._vc(self._feed_medium([]))},
        )
        ok, msg = validate_volume_change_states(process)
        assert ok is True


class TestValidateOutflowRetention:
    def _reactor_comp(self, name):
        return ReactorMediumComponent(
            name=name, unit="g/L", concentration=_ts([0.0, 1.0], [1.0, 2.0])
        )

    def _outflow(self, retention=None, is_continuous=True):
        return Outflow(
            name="sample",
            unit="L",
            is_controlled=True,
            is_continuous=is_continuous,
            values=_ts([0.0, 1.0], [-0.1, -0.2]),
            retention=retention or {},
        )

    def test_empty_retention_is_valid(self):
        process = _make_process(
            reactor_components={"biomass": self._reactor_comp("biomass")},
            volume_changes={"sample": self._outflow()},
        )
        ok, msg = validate_outflow_retention(process)
        assert ok is True

    def test_in_range_retention_is_valid(self):
        process = _make_process(
            reactor_components={"biomass": self._reactor_comp("biomass")},
            volume_changes={"sample": self._outflow(retention={"biomass": 0.95})},
        )
        ok, msg = validate_outflow_retention(process)
        assert ok is True

    def test_out_of_range_retention_rejected(self):
        # BioProcess.__post_init__ now rejects invalid retention at
        # construction time too, so an invalid Outflow can only reach
        # validate_outflow_retention via a post-construction mutation
        # (e.g. a loaded process patched in place) — exercise that path.
        process = _make_process(
            reactor_components={"biomass": self._reactor_comp("biomass")},
            volume_changes={"sample": self._outflow()},
        )
        process.volume.volume_changes["sample"].retention = {"biomass": 1.5}
        ok, msg = validate_outflow_retention(process)
        assert ok is False
        assert "biomass" in msg

    def test_negative_retention_rejected(self):
        process = _make_process(
            reactor_components={"biomass": self._reactor_comp("biomass")},
            volume_changes={"sample": self._outflow()},
        )
        process.volume.volume_changes["sample"].retention = {"biomass": -0.1}
        ok, msg = validate_outflow_retention(process)
        assert ok is False

    def test_unknown_component_rejected(self):
        process = _make_process(
            reactor_components={"biomass": self._reactor_comp("biomass")},
            volume_changes={"sample": self._outflow()},
        )
        process.volume.volume_changes["sample"].retention = {"typo_name": 0.5}
        ok, msg = validate_outflow_retention(process)
        assert ok is False
        assert "typo_name" in msg

    def test_discrete_retention_rejected(self):
        """retention is only ever consulted for continuous Outflows — a
        non-empty value on a discrete Outflow would otherwise be silently
        ignored by the RHS ODE, so it must be rejected instead."""
        process = _make_process(
            reactor_components={"biomass": self._reactor_comp("biomass")},
            volume_changes={"sample": self._outflow(is_continuous=False)},
        )
        process.volume.volume_changes["sample"].retention = {"biomass": 0.5}
        ok, msg = validate_outflow_retention(process)
        assert ok is False
        assert "discrete" in msg.lower()

    def test_construction_rejects_out_of_range_retention(self):
        """The same rule set is enforced at BioProcess construction time
        (dataclasses._check_outflow_retention via __post_init__), not just
        by this opt-in validator."""
        with pytest.raises(ValueError, match="biomass"):
            _make_process(
                reactor_components={"biomass": self._reactor_comp("biomass")},
                volume_changes={
                    "sample": self._outflow(retention={"biomass": 1.5})
                },
            )

    def test_construction_rejects_unknown_component_retention(self):
        with pytest.raises(ValueError, match="typo_name"):
            _make_process(
                reactor_components={"biomass": self._reactor_comp("biomass")},
                volume_changes={
                    "sample": self._outflow(retention={"typo_name": 0.5})
                },
            )

    def test_construction_rejects_discrete_retention(self):
        with pytest.raises(ValueError, match="discrete"):
            _make_process(
                reactor_components={"biomass": self._reactor_comp("biomass")},
                volume_changes={
                    "sample": self._outflow(
                        retention={"biomass": 0.5}, is_continuous=False
                    )
                },
            )

    def test_inflow_is_ignored(self):
        """retention only exists on Outflow; an Inflow in the mix must not
        confuse the check."""
        process = _make_process(
            reactor_components={"biomass": self._reactor_comp("biomass")},
            volume_changes={
                "feed": Inflow(
                    name="feed",
                    unit="L",
                    is_controlled=True,
                    is_continuous=True,
                    feed_medium=FeedMedium(
                        name="f",
                        components={
                            "biomass": FeedMediumComponent(
                                name="biomass",
                                unit="g/L",
                                concentration=StaticVariable(value=10.0),
                            )
                        },
                    ),
                    values=_ts([0.0, 1.0], [0.0, 0.1]),
                ),
            },
        )
        ok, msg = validate_outflow_retention(process)
        assert ok is True


# ---------------------------------------------------------------------------
# validate_biomass_in_reactor_medium
# ---------------------------------------------------------------------------


class TestValidateBiomassInReactorMedium:
    def _comp(self, name):
        return ReactorMediumComponent(
            name=name,
            unit="g/L",
            concentration=StaticVariable(value=1.0),
        )

    def test_biomass_present(self):
        process = _make_process(reactor_components={"biomass": self._comp("biomass")})
        ok, msg = validate_biomass_in_reactor_medium(process)
        assert ok is True
        assert "biomass" in msg.lower()

    def test_biomass_case_insensitive(self):
        process = _make_process(reactor_components={"Biomass": self._comp("Biomass")})
        ok, msg = validate_biomass_in_reactor_medium(process)
        assert ok is True

    def test_biomass_missing(self):
        # Auto-generation in BioProcess.__post_init__ raises before
        # validate_biomass_in_reactor_medium ever runs.
        with pytest.raises(ValueError, match="biomass"):
            _make_process(reactor_components={"glucose": self._comp("glucose")})

    def test_no_components(self):
        process = _make_process(reactor_components={})
        ok, msg = validate_biomass_in_reactor_medium(process)
        assert ok is False


# ---------------------------------------------------------------------------
# validate_initial_state_alignment
# ---------------------------------------------------------------------------


class TestValidateInitialStateAlignment:
    def test_all_states_measured_at_time_axis_start_passes(self):
        process = _make_process(
            reactor_components={
                "biomass": ReactorMediumComponent(
                    name="biomass",
                    unit="g/L",
                    concentration=_ts([0.0, 1.0, 2.0], [0.1, 0.5, 1.0]),
                ),
            },
            process_variables={
                "agitation": ProcessVariable(
                    name="agitation",
                    unit="rpm",
                    is_controlled=True,
                    values=_ts([0.0, 5.0], [200.0, 250.0]),
                ),
            },
        )

        ok, msg = validate_initial_state_alignment(process)

        assert ok is True
        assert msg.startswith("PASS initial_state_alignment:")

    def test_reactor_component_missing_t0_measurement_fails(self):
        process = _make_process(
            reactor_components={
                "biomass": ReactorMediumComponent(
                    name="biomass",
                    unit="g/L",
                    concentration=_ts([1.0, 2.0], [0.1, 0.5]),  # no t=0.0 point
                ),
            },
        )

        ok, msg = validate_initial_state_alignment(process)

        assert ok is False
        assert "reactor component 'biomass'" in msg

    def test_process_variable_missing_t0_measurement_fails(self):
        process = _make_process(
            reactor_components={
                "biomass": ReactorMediumComponent(
                    name="biomass",
                    unit="g/L",
                    concentration=StaticVariable(value=1.0),
                )
            },
            process_variables={
                "temperature": ProcessVariable(
                    name="temperature",
                    unit="degC",
                    is_controlled=True,
                    values=_ts([0.5, 1.0], [30.0, 31.0]),  # no t=0.0 point
                ),
            },
        )

        ok, msg = validate_initial_state_alignment(process)

        assert ok is False
        assert "process variable 'temperature'" in msg

    def test_static_variable_states_are_exempt(self):
        process = _make_process(
            reactor_components={
                "biomass": ReactorMediumComponent(
                    name="biomass",
                    unit="g/L",
                    concentration=StaticVariable(value=1.0),
                ),
            },
        )

        ok, _ = validate_initial_state_alignment(process)

        assert ok is True

    def test_wired_into_validate_process(self):
        process = _make_process(
            reactor_components={
                "biomass": ReactorMediumComponent(
                    name="biomass",
                    unit="g/L",
                    concentration=_ts([1.0, 2.0], [0.1, 0.5]),
                ),
            },
        )

        all_valid, results = validate_process(process)

        assert all_valid is False
        assert any(
            not ok and "initial_state_alignment" in msg for ok, msg in results
        )


# ---------------------------------------------------------------------------
# validate_process (integration)
# ---------------------------------------------------------------------------


class TestValidateProcess:
    def test_checks_reactor_and_total_volume_timeseries_shapes(self):
        malformed = SimpleNamespace(
            times=jnp.array([0.0, 1.0]), values=jnp.array([1.0])
        )
        biomass = ReactorMediumComponent(
            name="biomass",
            unit="g/L",
            concentration=_ts([0.0, 1.0], [1.0, 2.0]),
        )
        process = _make_process(reactor_components={"biomass": biomass})
        process.volume.total_volume = malformed

        all_valid, results = validate_process(process)

        assert all_valid is False
        ok_biomass, _ = _find(results, "TimeSeries 'biomass' —")
        assert ok_biomass is True
        ok_vol, _ = _find(results, "TimeSeries 'measured total volume'")
        assert ok_vol is False

    def test_invalid_time_axis_fails_process(self):
        process = _make_process(
            reactor_components={
                "biomass": ReactorMediumComponent(
                    name="biomass",
                    unit="g/L",
                    concentration=StaticVariable(value=1.0),
                )
            }
        )
        process.time_axis.start = 11.0

        all_valid, results = validate_process(process)

        assert all_valid is False
        ok_axis, msg_axis = _find(results, "start 11.0 is after end 10.0")
        assert ok_axis is False
        ok_bounds, msg_bounds = _find(results, "SKIP timestamp_bounds")
        assert ok_bounds is True
        assert "time_axis is invalid" in msg_bounds

    def test_valid_process_returns_all_ok(self):
        biomass_ts = _ts([0.0, 1.0, 2.0], [0.1, 0.5, 1.0])
        feed_medium = FeedMedium(
            name="fm",
            density=1.0,
            density_unit="kg/L",
            components={
                "biomass": FeedMediumComponent(
                    name="biomass",
                    unit="g/L",
                    concentration=StaticVariable(value=0.0),
                    is_controlled=True,
                ),
            },
        )
        vc = Inflow(
            name="feed",
            unit="L",
            is_controlled=True,
            is_continuous=True,
            feed_medium=feed_medium,
            values=_ts([0.0, 1.0, 2.0], [0.0, 0.1, 0.2]),
        )
        process = _make_process(
            reactor_components={
                "biomass": ReactorMediumComponent(
                    name="biomass",
                    unit="g/L",
                    concentration=biomass_ts,
                ),
            },
            volume_changes={"feed": vc},
        )
        all_valid, results = validate_process(process)
        assert all_valid is True
        assert all(ok for ok, _ in results)
        assert all(msg.startswith(("PASS ", "SKIP ")) for _, msg in results)

    def test_wrong_type_raises_type_error(self):
        """validate_process() must raise TypeError for non-BioProcess arguments."""
        with pytest.raises(TypeError, match="BioProcess"):
            validate_process("not a process")

    def test_wrong_type_dict_raises_type_error(self):
        with pytest.raises(TypeError):
            validate_process({"metadata": "fake"})

    def test_wrong_type_none_raises_type_error(self):
        with pytest.raises(TypeError):
            validate_process(None)

    def test_process_with_static_only_components(self):
        """A process with only static components should pass (biomass check aside)."""
        process = _make_process(
            reactor_components={
                "biomass": ReactorMediumComponent(
                    name="biomass",
                    unit="g/L",
                    concentration=StaticVariable(value=1.0),
                ),
            },
        )
        all_valid, results = validate_process(process)
        # biomass is present, no dynamic TS to fail -> should be valid
        assert all_valid is True

    def test_invalid_discrete_events_fail_process(self):
        process = _make_process(
            reactor_components={
                "biomass": ReactorMediumComponent(
                    name="biomass",
                    unit="g/L",
                    concentration=StaticVariable(value=1.0),
                )
            }
        )
        process.discrete_events = DiscreteEvents(times=jnp.array([2.0, 1.0]))

        all_valid, results = validate_process(process)

        assert all_valid is False
        assert any(
            not ok and "strictly monotonically increasing" in msg
            for ok, msg in results
        )

    def test_empty_measured_total_volume_fails_process(self):
        process = _make_process(
            reactor_components={
                "biomass": ReactorMediumComponent(
                    name="biomass",
                    unit="g/L",
                    concentration=StaticVariable(value=1.0),
                )
            }
        )
        process.volume.total_volume = _ts([], [])

        all_valid, results = validate_process(process)

        assert all_valid is False
        assert any(
            "measured total volume" in message and "empty" in message
            for _, message in results
        )

    @pytest.mark.parametrize(
        ("is_continuous", "expect_empty_error"),
        [(False, False), (True, True)],
    )
    def test_only_empty_discrete_volume_changes_are_valid(
        self,
        is_continuous,
        expect_empty_error,
    ):
        process = _make_process(
            volume_changes={
                "sampling": Outflow(
                    name="sampling",
                    unit="L",
                    is_controlled=True,
                    is_continuous=is_continuous,
                    values=_ts([], []),
                )
            }
        )

        _, results = validate_process(process)

        assert any("must not be empty" in message for _, message in results) is (
            expect_empty_error
        )

    def test_timestamp_outside_bounds_fails_process(self):
        process = _make_process(
            reactor_components={
                "biomass": ReactorMediumComponent(
                    name="biomass",
                    unit="g/L",
                    concentration=_ts([0.0, 11.0], [1.0, 2.0]),
                )
            }
        )

        all_valid, results = validate_process(process)

        assert all_valid is False
        ok_bounds, msg_bounds = _find(results, "FAIL timestamp_bounds")
        assert ok_bounds is False
        assert "outside" in msg_bounds

    def test_mismatched_volume_change_unit_fails_process(self):
        process = _make_process(
            reactor_components={
                "biomass": ReactorMediumComponent(
                    name="biomass",
                    unit="g/L",
                    concentration=StaticVariable(value=1.0),
                )
            },
            volume_changes={
                "sample": Outflow(
                    name="sample",
                    unit="mL",
                    is_controlled=True,
                    is_continuous=False,
                    values=_ts([1.0], [-100.0]),
                )
            },
        )

        all_valid, results = validate_process(process)

        assert all_valid is False
        assert any("volume unit 'L'" in message for _, message in results)

    def test_every_message_follows_verdict_check_name_template(self):
        """Regression test: every check's message follows the shared
        '<VERDICT> <check_name>: <detail>' template, and ok always agrees
        with the verdict."""
        process = _make_process(
            reactor_components={
                "biomass": ReactorMediumComponent(
                    name="biomass",
                    unit="g/L",
                    concentration=_ts([0.0, 1.0], [0.1, 0.5]),
                ),
            },
        )
        known_checks = {
            "discrete_events",
            "timeseries_shape",
            "time_axis",
            "timestamp_bounds",
            "volume_units",
            "volume_change_sign",
            "volume_change_states",
            "outflow_retention",
            "biomass_in_reactor_medium",
            "initial_state_alignment",
            "measurement_sampling_alignment",
            "bounds",
            "bounds_against_data",
            "biological_ode",
        }

        _, results = validate_process(process)

        assert results  # sanity: the process exercises at least one check
        for ok, msg in results:
            verdict, rest = msg.split(" ", 1)
            assert verdict in ("PASS", "FAIL", "SKIP")
            check_name, sep, _detail = rest.partition(": ")
            assert sep == ": "
            assert check_name in known_checks, check_name
            assert ok == (verdict != "FAIL")


# ---------------------------------------------------------------------------
# validate_volume_consistency
# ---------------------------------------------------------------------------


class TestValidateVolumeConsistency:
    def _make_process_with_volume(self, initial_volume, changes):
        """Build a BioProcess with given volume changes for consistency tests."""
        volume_changes = {}
        for name, (is_continuous, timepoints, values, feed_medium) in changes.items():
            if any(v < 0 for v in values):
                volume_changes[name] = Outflow(
                    name=name,
                    unit="L",
                    is_controlled=True,
                    is_continuous=is_continuous,
                    values=_ts(timepoints, values),
                )
            else:
                volume_changes[name] = Inflow(
                    name=name,
                    unit="L",
                    is_controlled=True,
                    is_continuous=is_continuous,
                    feed_medium=feed_medium
                    or FeedMedium(name="f", density=1.0, density_unit="kg/L"),
                    values=_ts(timepoints, values),
                )
        return BioProcess(
            metadata=BioProcessMetadata(name="test", process_type="fed_batch"),
            time_axis=TimeAxis(
                unit="hours", start=0.0, end=10.0, time_reference="inoculation"
            ),
            volume=Volume(
                initial_volume=initial_volume, unit="L", volume_changes=volume_changes
            ),
            reactor_medium=ReactorMedium(
                name="medium",
                density=1.0,
                density_unit="kg/L",
            ),
        )

    def test_continuous_volume_balance_ok(self):
        # Cumulative feed: 0 -> 1 L => total change = 1 L; initial=1, expected final=2
        process = self._make_process_with_volume(
            initial_volume=1.0,
            changes={
                "feed": (True, [0.0, 10.0], [0.0, 1.0], None),
            },
        )
        ok, msg, delta = validate_volume_consistency(process, final_volume=2.0)
        assert ok is True
        assert msg.startswith("PASS volume_consistency:")
        assert abs(delta - 1.0) < 1e-6

    def test_discrete_volume_balance_ok(self):
        # Discrete boluses: 0.5 + 0.5 = 1.0 L; initial=1, expected final=2
        process = self._make_process_with_volume(
            initial_volume=1.0,
            changes={
                "bolus": (False, [2.0, 5.0], [0.5, 0.5], None),
            },
        )
        ok, msg, delta = validate_volume_consistency(process, final_volume=2.0)
        assert ok is True
        assert abs(delta - 1.0) < 1e-6

    def test_volume_inconsistency_detected(self):
        # Cumulative feed: 0 -> 0.1 L; initial=1, expected final=3 (large mismatch)
        process = self._make_process_with_volume(
            initial_volume=1.0,
            changes={
                "feed": (True, [0.0, 10.0], [0.0, 0.1], None),
            },
        )
        ok, msg, delta = validate_volume_consistency(process, final_volume=3.0)
        assert ok is False
        assert msg.startswith("FAIL volume_consistency:")

    def test_negative_volume_change(self):
        # Sampling: cumulative removal 0 -> -0.2 L; initial=2, expected final=1.8
        process = self._make_process_with_volume(
            initial_volume=2.0,
            changes={
                "sample": (True, [0.0, 10.0], [0.0, -0.2], None),
            },
        )
        ok, msg, delta = validate_volume_consistency(process, final_volume=1.8)
        assert ok is True


# ---------------------------------------------------------------------------
# validate_for_publication
# ---------------------------------------------------------------------------


def _make_biomass_process(
    biomass_ts,
    extra_components=None,
    process_variables=None,
    volume_changes=None,
):
    """Build a minimal valid BioProcess with a biomass component."""
    components = {
        "biomass": ReactorMediumComponent(
            name="biomass",
            unit="g/L",
            concentration=biomass_ts,
        )
    }
    if extra_components:
        components.update(extra_components)
    return _make_process(
        reactor_components=components,
        process_variables=process_variables or {},
        volume_changes=volume_changes or {},
    )


def _make_feed_medium(component_names):
    """Build a FeedMedium with StaticVariable concentrations for the given names."""
    return FeedMedium(
        name="f",
        density=1.0,
        density_unit="kg/L",
        components={
            name: FeedMediumComponent(
                name=name,
                unit="g/L",
                concentration=StaticVariable(value=0.0),
                is_controlled=True,
            )
            for name in component_names
        },
    )


class TestValidateForPublication:
    def _collection(self, processes):
        return BioProcessCollection(
            case_id="cs1",
            organism="E. coli",
            citation="Test et al.",
            processes=processes,
        )

    def test_valid_collection_all_ok(self):
        ts = _ts([0.0, 1.0, 2.0], [0.1, 0.5, 1.0])
        p1 = _make_biomass_process(ts)
        p2 = _make_biomass_process(_ts([0.0, 1.0, 2.0], [0.2, 0.6, 1.1]))
        cs = self._collection({"run1": p1, "run2": p2})
        all_valid, report = validate_for_publication(cs)
        assert all_valid is True
        assert "run1" in report
        assert "run2" in report
        ok0, msg0 = report["__consistency__"][0]
        assert ok0 is True
        assert msg0.startswith("PASS cross_process_consistency:")

    def test_empty_collection(self):
        cs = self._collection({})
        all_valid, report = validate_for_publication(cs)
        assert all_valid is True
        assert report == {}

    def test_single_process_collection(self):
        ts = _ts([0.0, 1.0], [0.1, 0.5])
        p = _make_biomass_process(ts)
        cs = self._collection({"run1": p})
        all_valid, report = validate_for_publication(cs)
        assert all_valid is True
        assert "run1" in report
        ok0, msg0 = report["__consistency__"][0]
        assert ok0 is True
        assert msg0.startswith("PASS cross_process_consistency:")

    def test_inconsistent_reactor_medium_components(self):
        """Processes with different reactor medium components fail consistency."""
        ts = _ts([0.0, 1.0], [0.1, 0.5])
        p1 = _make_biomass_process(ts)
        # p2 has an extra 'glucose' component
        p2 = _make_biomass_process(
            ts,
            extra_components={
                "glucose": ReactorMediumComponent(
                    name="glucose",
                    unit="g/L",
                    concentration=StaticVariable(value=10.0),
                )
            },
        )
        cs = self._collection({"run1": p1, "run2": p2})
        all_valid, report = validate_for_publication(cs)
        assert all_valid is False
        assert any("reactor medium" in msg for _, msg in report["__consistency__"])

    def test_inconsistent_process_variable_names(self):
        """Processes with different process variable names should fail consistency."""
        ts = _ts([0.0, 1.0], [0.1, 0.5])
        pv = ProcessVariable(
            name="temperature",
            unit="°C",
            is_controlled=True,
            values=_ts([0.0, 1.0], [37.0, 37.0]),
        )
        p1 = _make_biomass_process(ts, process_variables={"temperature": pv})
        p2 = _make_biomass_process(ts)  # no process variables
        cs = self._collection({"run1": p1, "run2": p2})
        all_valid, report = validate_for_publication(cs)
        assert all_valid is False
        assert any("process variables" in msg for _, msg in report["__consistency__"])

    def test_inconsistent_process_variable_types(self):
        """Same variable name but different type fails."""
        ts = _ts([0.0, 1.0], [0.1, 0.5])
        pv_ts = ProcessVariable(
            name="temperature",
            unit="°C",
            is_controlled=True,
            values=_ts([0.0, 1.0], [37.0, 37.0]),
        )
        pv_static = ProcessVariable(
            name="temperature",
            unit="°C",
            is_controlled=True,
            values=StaticVariable(value=37.0),
        )
        p1 = _make_biomass_process(ts, process_variables={"temperature": pv_ts})
        p2 = _make_biomass_process(ts, process_variables={"temperature": pv_static})
        cs = self._collection({"run1": p1, "run2": p2})
        all_valid, report = validate_for_publication(cs)
        assert all_valid is False
        assert any("process variables" in msg for _, msg in report["__consistency__"])

    def test_different_volume_change_names_pass(self):
        """Processes may use different feed or sampling strategies."""
        ts = _ts([0.0, 1.0], [0.1, 0.5])
        vc = Inflow(
            name="feed",
            unit="L",
            is_controlled=True,
            is_continuous=True,
            feed_medium=_make_feed_medium(["biomass"]),
            values=_ts([0.0, 1.0], [0.0, 0.1]),
        )
        p1 = _make_biomass_process(ts, volume_changes={"feed": vc})
        p2 = _make_biomass_process(ts)  # no volume changes
        cs = self._collection({"run1": p1, "run2": p2})
        all_valid, report = validate_for_publication(cs)
        assert all_valid is True
        assert len(report["__consistency__"]) == 1
        ok0, msg0 = report["__consistency__"][0]
        assert ok0 is True
        assert msg0.startswith("PASS cross_process_consistency:")

    def test_wrong_type_raises_type_error(self):
        with pytest.raises(TypeError, match="BioProcessCollection"):
            validate_for_publication("not a collection")

    def test_wrong_type_none_raises_type_error(self):
        with pytest.raises(TypeError):
            validate_for_publication(None)

    def test_inconsistent_reactor_medium_units(self):
        """Same component name and type but different units should fail consistency."""
        ts = _ts([0.0, 1.0], [0.1, 0.5])
        p1 = _make_biomass_process(ts)  # biomass unit is "g/L"
        p2 = _make_biomass_process(ts)
        # Override the biomass component unit in p2
        p2.reactor_medium.components["biomass"] = ReactorMediumComponent(
            name="biomass",
            unit="mmol/L",
            concentration=ts,
        )
        cs = self._collection({"run1": p1, "run2": p2})
        all_valid, report = validate_for_publication(cs)
        assert all_valid is False
        assert any("reactor medium" in msg for _, msg in report["__consistency__"])

    def test_inconsistent_process_variable_units(self):
        """Same process variable name and type but different units should fail."""
        ts = _ts([0.0, 1.0], [0.1, 0.5])
        pv1 = ProcessVariable(
            name="temperature",
            unit="°C",
            is_controlled=True,
            values=_ts([0.0, 1.0], [37.0, 37.0]),
        )
        pv2 = ProcessVariable(
            name="temperature",
            unit="K",
            is_controlled=True,
            values=_ts([0.0, 1.0], [310.0, 310.0]),
        )
        p1 = _make_biomass_process(ts, process_variables={"temperature": pv1})
        p2 = _make_biomass_process(ts, process_variables={"temperature": pv2})
        cs = self._collection({"run1": p1, "run2": p2})
        all_valid, report = validate_for_publication(cs)
        assert all_valid is False
        assert any("process variables" in msg for _, msg in report["__consistency__"])

    def test_volume_change_units_are_checked_per_process(self):
        """Cross-process consistency does not compare volume-change units."""
        ts = _ts([0.0, 1.0], [0.1, 0.5])
        vc1 = Inflow(
            name="feed",
            unit="L",
            is_controlled=True,
            is_continuous=True,
            feed_medium=_make_feed_medium(["biomass"]),
            values=_ts([0.0, 1.0], [0.0, 0.1]),
        )
        vc2 = Inflow(
            name="feed",
            unit="mL",
            is_controlled=True,
            is_continuous=True,
            feed_medium=_make_feed_medium(["biomass"]),
            values=_ts([0.0, 1.0], [0.0, 100.0]),
        )
        p1 = _make_biomass_process(ts, volume_changes={"feed": vc1})
        p2 = _make_biomass_process(ts, volume_changes={"feed": vc2})
        cs = self._collection({"run1": p1, "run2": p2})
        all_valid, report = validate_for_publication(cs)
        assert all_valid is False
        assert len(report["__consistency__"]) == 1
        ok0, msg0 = report["__consistency__"][0]
        assert ok0 is True
        assert msg0.startswith("PASS cross_process_consistency:")
        assert any(
            "volume changes must use volume unit" in message
            for _, message in report["run2"]
        )


class TestValidateCrossProcessConsistency:
    """Direct coverage for the extracted validate_cross_process_consistency
    helper, independent of validate_for_publication's per-process checks."""

    def test_trivial_pass_empty(self):
        collection = BioProcessCollection(processes={})
        ok, results = validate_cross_process_consistency(collection)
        assert ok is True
        assert len(results) == 1
        assert results[0][0] is True
        assert results[0][1].startswith("PASS cross_process_consistency:")

    def test_trivial_pass_single_process(self):
        ts = _ts([0.0, 1.0], [0.1, 0.5])
        collection = BioProcessCollection(processes={"run1": _make_biomass_process(ts)})
        ok, results = validate_cross_process_consistency(collection)
        assert ok is True
        assert len(results) == 1
        assert results[0][0] is True

    def test_consistent_multi_process(self):
        ts = _ts([0.0, 1.0, 2.0], [0.1, 0.5, 1.0])
        p1 = _make_biomass_process(ts)
        p2 = _make_biomass_process(_ts([0.0, 1.0, 2.0], [0.2, 0.6, 1.1]))
        collection = BioProcessCollection(processes={"run1": p1, "run2": p2})
        ok, results = validate_cross_process_consistency(collection)
        assert ok is True
        assert len(results) == 1
        assert results[0][0] is True

    @pytest.mark.parametrize(
        ("field", "value"),
        [("unit", "days"), ("time_reference", "first_feed")],
    )
    def test_inconsistent_time_axis(self, field, value):
        ts = _ts([0.0, 1.0], [0.1, 0.5])
        p1 = _make_biomass_process(ts)
        p2 = _make_biomass_process(ts)
        setattr(p2.time_axis, field, value)
        collection = BioProcessCollection(processes={"run1": p1, "run2": p2})

        ok, results = validate_cross_process_consistency(collection)

        assert ok is False
        assert any("time axis" in message for _, message in results)

    def test_different_time_axis_bounds_are_consistent(self):
        ts = _ts([0.0, 1.0], [0.1, 0.5])
        p1 = _make_biomass_process(ts)
        p2 = _make_biomass_process(ts)
        p2.time_axis.start = 1.0
        p2.time_axis.end = 20.0
        collection = BioProcessCollection(processes={"run1": p1, "run2": p2})

        ok, results = validate_cross_process_consistency(collection)

        assert ok is True
        assert len(results) == 1
        assert results[0][0] is True

    def test_inconsistent_volume_units(self):
        ts = _ts([0.0, 1.0], [0.1, 0.5])
        p1 = _make_biomass_process(ts)
        p2 = _make_biomass_process(ts)
        p2.volume.unit = "mL"
        collection = BioProcessCollection(processes={"run1": p1, "run2": p2})

        ok, results = validate_cross_process_consistency(collection)

        assert ok is False
        assert any("volume unit" in message for _, message in results)

    def test_inconsistent_reactor_medium_components(self):
        ts = _ts([0.0, 1.0], [0.1, 0.5])
        p1 = _make_biomass_process(ts)
        p2 = _make_biomass_process(
            ts,
            extra_components={
                "glucose": ReactorMediumComponent(
                    name="glucose",
                    unit="g/L",
                    concentration=StaticVariable(value=10.0),
                )
            },
        )
        collection = BioProcessCollection(processes={"run1": p1, "run2": p2})
        ok, results = validate_cross_process_consistency(collection)
        assert ok is False
        assert any("reactor medium" in msg for _, msg in results)


# ---------------------------------------------------------------------------
# validate_measurement_sampling_alignment
# ---------------------------------------------------------------------------


class TestValidateMeasurementSamplingAlignment:
    def _sample_vc(self, times, values):
        return Outflow(
            name="sample",
            unit="L",
            is_controlled=True,
            is_continuous=False,
            values=_ts(times, values),
        )

    def test_aligned_times_pass(self):
        """Measurement times exactly match sampling times — should pass."""
        sample_times = [2.0, 5.0, 8.0]
        sample_vals = [-0.01, -0.01, -0.01]
        process = _make_process(
            reactor_components={
                "biomass": ReactorMediumComponent(
                    name="biomass",
                    unit="g/L",
                    concentration=_ts([2.0, 5.0, 8.0], [0.1, 0.5, 1.0]),
                ),
            },
            volume_changes={"sample": self._sample_vc(sample_times, sample_vals)},
        )
        ok, msg = validate_measurement_sampling_alignment(process)
        assert ok is True
        assert msg.startswith("PASS measurement_sampling_alignment:")

    def test_small_delay_detected(self):
        """Measurement shortly after sampling should warn."""
        sample_times = [2.0, 5.0, 8.0]
        sample_vals = [-0.01, -0.01, -0.01]
        # Measurements are slightly after sampling
        meas_times = [2.0003, 5.0003, 8.0003]
        process = _make_process(
            reactor_components={
                "biomass": ReactorMediumComponent(
                    name="biomass",
                    unit="g/L",
                    concentration=_ts(meas_times, [0.1, 0.5, 1.0]),
                ),
            },
            volume_changes={"sample": self._sample_vc(sample_times, sample_vals)},
        )
        ok, msg = validate_measurement_sampling_alignment(process)
        assert ok is False
        assert "biomass" in msg
        assert "spline" in msg

    def test_large_gap_passes(self):
        """Measurements far from any sampling time — not a misalignment, should pass."""
        sample_times = [2.0, 8.0]
        sample_vals = [-0.01, -0.01]
        meas_times = [0.5, 5.0, 9.5]  # far from 2.0 and 8.0
        process = _make_process(
            reactor_components={
                "biomass": ReactorMediumComponent(
                    name="biomass",
                    unit="g/L",
                    concentration=_ts(meas_times, [0.1, 0.5, 1.0]),
                ),
            },
            volume_changes={"sample": self._sample_vc(sample_times, sample_vals)},
        )
        ok, msg = validate_measurement_sampling_alignment(process)
        assert ok is True

    def test_no_sampling_events_skipped(self):
        """Process with no Outflow — check should be skipped."""
        process = _make_process(
            reactor_components={
                "biomass": ReactorMediumComponent(
                    name="biomass",
                    unit="g/L",
                    concentration=_ts([0.0, 5.0, 10.0], [0.1, 0.5, 1.0]),
                ),
            },
        )
        ok, msg = validate_measurement_sampling_alignment(process)
        assert ok is True
        assert msg.startswith("SKIP measurement_sampling_alignment:")

    def test_continuous_outflow_is_not_a_sampling_event(self):
        process = _make_process(
            reactor_components={
                "biomass": ReactorMediumComponent(
                    name="biomass",
                    unit="g/L",
                    concentration=_ts([5.0005], [0.5]),
                ),
            },
            volume_changes={
                "harvest": Outflow(
                    name="harvest",
                    unit="L",
                    is_controlled=True,
                    is_continuous=True,
                    values=_ts([0.0, 5.0, 10.0], [0.0, -0.5, -1.0]),
                )
            },
        )
        ok, msg = validate_measurement_sampling_alignment(process)
        assert ok is True
        assert msg.startswith("SKIP measurement_sampling_alignment:")


# ---------------------------------------------------------------------------
# validate_augmented_parent_refs
# ---------------------------------------------------------------------------


class TestValidateAugmentedParentRefs:
    def _aug_child(self, *, parent_process: str, name: str = "aug"):
        return AugmentedBioProcess(
            metadata=BioProcessMetadata(name=name, process_type="batch"),
            time_axis=TimeAxis(
                unit="hours",
                start=0.0,
                end=10.0,
                time_reference="inoculation",
            ),
            volume=Volume(initial_volume=1.0, unit="L"),
            reactor_medium=ReactorMedium(
                name="medium",
                density=1.0,
                density_unit="kg/L",
            ),
            parent_process=parent_process,
        )

    def _collection(self, processes):
        return BioProcessCollection(
            case_id="cs1",
            organism="E. coli",
            citation="Test et al.",
            processes=processes,
        )

    def test_ok_when_parent_exists(self):
        ts = _ts([0.0, 1.0], [0.1, 0.5])
        parent = _make_biomass_process(ts)
        child = self._aug_child(parent_process="parent")
        cs = self._collection({"parent": parent, "child": child})
        ok, results = validate_augmented_parent_refs(cs)
        assert ok is True
        assert any(r_ok for r_ok, _ in results)

    def test_unknown_parent_fails(self):
        ts = _ts([0.0, 1.0], [0.1, 0.5])
        parent = _make_biomass_process(ts)
        child = self._aug_child(parent_process="ghost")
        cs = self._collection({"parent": parent, "child": child})
        ok, results = validate_augmented_parent_refs(cs)
        assert ok is False
        assert any("unknown parent_process" in msg for _, msg in results)

    def test_rejects_augmented_of_augmented(self):
        ts = _ts([0.0, 1.0], [0.1, 0.5])
        parent = _make_biomass_process(ts)
        first_aug = self._aug_child(parent_process="parent", name="first_aug")
        chained = self._aug_child(parent_process="first_aug", name="chained")
        cs = self._collection(
            {
                "parent": parent,
                "first_aug": first_aug,
                "chained": chained,
            }
        )
        ok, results = validate_augmented_parent_refs(cs)
        assert ok is False
        assert any("itself augmented" in msg for _, msg in results)

    def test_no_augmented_processes_ok(self):
        ts = _ts([0.0, 1.0], [0.1, 0.5])
        cs = self._collection({"p1": _make_biomass_process(ts)})
        ok, results = validate_augmented_parent_refs(cs)
        assert ok is True
        assert any(r_ok for r_ok, _ in results)

    def test_works_on_bioprocess_collection(self):
        ts = _ts([0.0, 1.0], [0.1, 0.5])
        parent = _make_biomass_process(ts)
        child = self._aug_child(parent_process="parent")
        collection = BioProcessCollection(processes={"parent": parent, "child": child})
        ok, _ = validate_augmented_parent_refs(collection)
        assert ok is True

    def test_wrong_type_raises(self):
        with pytest.raises(TypeError):
            validate_augmented_parent_refs("not a collection")

    def test_validate_for_publication_runs_augmented_parent_refs(self):
        ts = _ts([0.0, 1.0], [0.1, 0.5])
        parent = _make_biomass_process(ts)
        child = self._aug_child(parent_process="ghost")
        cs = self._collection({"parent": parent, "child": child})
        all_valid, report = validate_for_publication(cs)
        assert all_valid is False
        assert "__augmented__" in report
        assert any("unknown parent_process" in msg for _, msg in report["__augmented__"])


# ---------------------------------------------------------------------------
# validate_biological_ode + validate_bounds
# ---------------------------------------------------------------------------


def _make_intra_process():
    """Process with biomass + intracellular product + glucose. No volume changes."""
    return _make_process(
        reactor_components={
            "biomass": ReactorMediumComponent("biomass", "g/L", StaticVariable(1.0)),
            "product": ReactorMediumComponent("product", "g/L", StaticVariable(0.0)),
            "glucose": ReactorMediumComponent("glucose", "g/L", StaticVariable(10.0)),
        }
    )


class TestValidateBiologicalOde:
    def test_auto_generated_block_validates_clean(self):
        # BioProcess.__post_init__ populates biological_ode automatically;
        # the auto-generated block must always pass validation.
        p = _make_intra_process()
        assert p.biological_ode is not None
        ok, msg = validate_biological_ode(p)
        assert ok is True

    def test_well_formed_block_passes(self):
        p = _make_intra_process()
        p.biological_ode = BiologicalOde(
            algebraic={"X_active": "biomass - product"},
            rates={"q_X": (None, None), "q_P": (None, None), "q_S": (None, None)},
            derivatives={
                "biomass": "q_X * X_active + q_P * X_active",
                "product": "q_P * X_active",
                "glucose": "q_S * X_active",
            },
        )
        ok, _ = validate_biological_ode(p)
        assert ok is True

    def test_unknown_symbol_in_expression_is_rejected(self):
        p = _make_intra_process()
        p.biological_ode = BiologicalOde(
            algebraic={},
            rates={"q_X": (None, None)},
            derivatives={
                "biomass": "q_X * biomass + zzz",
                "product": "0",
                "glucose": "0",
            },
        )
        ok, msg = validate_biological_ode(p)
        assert ok is False
        assert "zzz" in msg

    def test_missing_derivative_for_state_is_rejected(self):
        p = _make_intra_process()
        p.biological_ode = BiologicalOde(
            algebraic={},
            rates={"q_X": (None, None)},
            derivatives={"biomass": "q_X * biomass"},
        )
        ok, msg = validate_biological_ode(p)
        assert ok is False
        assert "missing entries" in msg
        assert "product" in msg
        assert "glucose" in msg

    def test_extra_derivative_for_non_state_is_rejected(self):
        p = _make_intra_process()
        p.biological_ode = BiologicalOde(
            algebraic={},
            rates={"q_X": (None, None)},
            derivatives={
                "biomass": "0",
                "product": "0",
                "glucose": "0",
                "ghost": "q_X",
            },
        )
        ok, msg = validate_biological_ode(p)
        assert ok is False
        assert "ghost" in msg

    def test_algebraic_dependency_cycle_is_rejected(self):
        p = _make_intra_process()
        p.biological_ode = BiologicalOde(
            algebraic={"a": "b + 1", "b": "a * 2"},
            rates={"q_X": (None, None)},
            derivatives={"biomass": "0", "product": "0", "glucose": "0"},
        )
        ok, msg = validate_biological_ode(p)
        assert ok is False
        assert "cycle" in msg.lower()

    def test_rate_name_collides_with_state_is_rejected(self):
        p = _make_intra_process()
        p.biological_ode = BiologicalOde(
            algebraic={},
            rates={"biomass": (None, None)},
            derivatives={"biomass": "biomass", "product": "0", "glucose": "0"},
        )
        ok, msg = validate_biological_ode(p)
        assert ok is False
        assert "collide" in msg.lower()
        assert "biomass" in msg

    def test_rate_name_collides_with_controlled_pv_is_rejected(self):
        p = _make_intra_process()
        p.process_variables = {
            "feed_rate": ProcessVariable(
                "feed_rate", "L/h", is_controlled=True, values=StaticVariable(0.1)
            )
        }
        p.biological_ode = BiologicalOde(
            algebraic={},
            rates={"feed_rate": (None, None)},
            derivatives={"biomass": "0", "product": "0", "glucose": "0"},
        )
        ok, msg = validate_biological_ode(p)
        assert ok is False
        assert "feed_rate" in msg

    def test_invalid_rate_bounds_lo_greater_than_hi_is_rejected(self):
        p = _make_intra_process()
        p.biological_ode = BiologicalOde(
            algebraic={},
            rates={"q_X": (2.0, 1.0)},
            derivatives={"biomass": "q_X * biomass", "product": "0", "glucose": "0"},
        )
        ok, msg = validate_biological_ode(p)
        assert ok is False
        assert "lo=2.0" in msg
        assert "hi=1.0" in msg

    def test_unit_consistent_state_subtraction_passes(self):
        """X_active = biomass - product with both g/L: accepted."""
        p = _make_intra_process()  # biomass, product, glucose all g/L
        p.biological_ode = BiologicalOde(
            algebraic={"X_active": "biomass - product"},
            rates={"q_X": (None, None), "q_P": (None, None), "q_S": (None, None)},
            derivatives={
                "biomass": "q_X * X_active + q_P * X_active",
                "product": "q_P * X_active",
                "glucose": "q_S * X_active",
            },
        )
        ok, _ = validate_biological_ode(p)
        assert ok is True

    def test_unit_mismatched_state_subtraction_is_rejected(self):
        """X_active = biomass (g/L) - product (mg/L): rejected with both
        names and both units in the message."""
        p = _make_process(
            reactor_components={
                "biomass": ReactorMediumComponent(
                    "biomass", "g/L", StaticVariable(1.0)
                ),
                "product": ReactorMediumComponent(
                    "product", "mg/L", StaticVariable(0.0)
                ),
                "glucose": ReactorMediumComponent(
                    "glucose", "g/L", StaticVariable(10.0)
                ),
            }
        )
        p.biological_ode = BiologicalOde(
            algebraic={"X_active": "biomass - product"},
            rates={"q_X": (None, None), "q_P": (None, None), "q_S": (None, None)},
            derivatives={
                "biomass": "q_X * X_active",
                "product": "q_P * X_active",
                "glucose": "q_S * X_active",
            },
        )
        ok, msg = validate_biological_ode(p)
        assert ok is False
        assert "X_active" in msg
        assert "biomass" in msg
        assert "product" in msg
        assert "g/L" in msg and "mg/L" in msg

    def test_unit_mismatch_in_derivative_expression_is_rejected(self):
        """Mismatch can also occur inside a `derivatives` expression, not
        just `algebraic`. ``biomass + glucose`` directly is unit-nonsense
        when biomass is g/L and glucose is mmol/L."""
        p = _make_process(
            reactor_components={
                "biomass": ReactorMediumComponent(
                    "biomass", "g/L", StaticVariable(1.0)
                ),
                "glucose": ReactorMediumComponent(
                    "glucose", "mmol/L", StaticVariable(10.0)
                ),
            }
        )
        p.biological_ode = BiologicalOde(
            algebraic={},
            rates={"q_X": (None, None)},
            derivatives={
                "biomass": "q_X * (biomass + glucose)",
                "glucose": "0",
            },
        )
        ok, msg = validate_biological_ode(p)
        assert ok is False
        assert "biomass" in msg
        assert "glucose" in msg

    def test_unit_check_ignores_products_only_subexpressions(self):
        """``q_X * X_active + q_P * X_active`` is an Add of two Muls; each
        operand contains only one state symbol, so no unit-mismatch error
        should fire even though the sum technically has two state-symbol
        free symbols when collected at the Add level."""
        # We use components where biomass and product DO match on units so
        # the algebraic check passes, then check that the derivative — whose
        # Add-level free_symbols include {q_X, X_active, q_P} but NO state
        # symbols (X_active is algebraic, q_* are rates) — does not trigger.
        p = _make_intra_process()
        p.biological_ode = BiologicalOde(
            algebraic={"X_active": "biomass - product"},
            rates={"q_X": (None, None), "q_P": (None, None), "q_S": (None, None)},
            derivatives={
                "biomass": "q_X * X_active + q_P * X_active",
                "product": "q_P * X_active",
                "glucose": "q_S * X_active",
            },
        )
        ok, _ = validate_biological_ode(p)
        assert ok is True

    def test_unit_mismatch_with_uncontrolled_pv_is_rejected(self):
        """A sum of a reactor component and an uncontrolled PV state with
        differing units should also be flagged. Controlled PVs are inputs,
        not states, so they do not appear in the state-name set and are
        intentionally ignored by the unit check."""
        p = _make_intra_process()
        p.process_variables = {
            "viability": ProcessVariable(
                "viability",
                "%",
                is_controlled=False,
                values=StaticVariable(95.0),
            ),
        }
        p.biological_ode = BiologicalOde(
            algebraic={"weird": "biomass + viability"},
            rates={"q_X": (None, None), "q_P": (None, None), "q_S": (None, None)},
            derivatives={
                "biomass": "q_X * biomass",
                "product": "q_P * biomass",
                "glucose": "q_S * biomass",
                "viability": "0",
            },
        )
        ok, msg = validate_biological_ode(p)
        assert ok is False
        assert "viability" in msg
        assert "biomass" in msg


class TestValidateBounds:
    def test_unbounded_default_passes(self):
        p = _make_intra_process()
        ok, _ = validate_bounds(p)
        assert ok is True

    def test_invalid_reactor_component_bounds_rejected(self):
        p = _make_intra_process()
        p.reactor_medium.components["biomass"].bounds = (5.0, 1.0)
        ok, msg = validate_bounds(p)
        assert ok is False
        assert "biomass" in msg

    def test_invalid_volume_bounds_rejected(self):
        p = _make_intra_process()
        p.volume.bounds = (10.0, 1.0)
        ok, msg = validate_bounds(p)
        assert ok is False
        assert "volume" in msg.lower()

    def test_invalid_pv_bounds_rejected(self):
        p = _make_intra_process()
        p.process_variables = {
            "pH": ProcessVariable(
                "pH",
                "",
                is_controlled=False,
                values=StaticVariable(7.0),
                bounds=(14.0, 0.0),
            )
        }
        ok, msg = validate_bounds(p)
        assert ok is False
        assert "pH" in msg


class TestValidateBoundsAgainstData:
    def test_in_bounds_passes(self):
        ts = _ts([0.0, 1.0, 2.0], [0.1, 0.5, 1.0])
        p = _make_biomass_process(ts)
        ok, _ = validate_bounds_against_data(p)
        assert ok is True

    def test_below_lower_bound_fails_with_count(self):
        # default RMC bounds are (0.0, None) even when never set explicitly
        ts = _ts([0.0, 1.0, 2.0], [-0.5, 0.5, 1.0])
        p = _make_biomass_process(ts)
        ok, msg = validate_bounds_against_data(p)
        assert ok is False
        assert "1 datapoint" in msg
        assert "biomass" in msg

    def test_above_upper_bound_fails(self):
        ts = _ts([0.0, 1.0], [0.1, 0.5])
        p = _make_biomass_process(ts)
        p.reactor_medium.components["biomass"].bounds = (0.0, 0.3)
        ok, msg = validate_bounds_against_data(p)
        assert ok is False
        assert "above upper bound" in msg

    def test_unset_bounds_skips_check(self):
        ts = _ts([0.0, 1.0], [-5.0, -3.0])
        p = _make_biomass_process(ts)
        p.reactor_medium.components["biomass"].bounds = (None, None)
        ok, _ = validate_bounds_against_data(p)
        assert ok is True

    def test_static_variable_checked(self):
        p = _make_intra_process()
        p.reactor_medium.components["product"].bounds = (0.0, None)
        p.reactor_medium.components["product"].concentration = StaticVariable(-1.0)
        ok, msg = validate_bounds_against_data(p)
        assert ok is False
        assert "product" in msg

    def test_process_variable_checked(self):
        p = _make_intra_process()
        p.process_variables = {
            "pH": ProcessVariable(
                "pH",
                "",
                is_controlled=False,
                values=StaticVariable(20.0),
                bounds=(0.0, 14.0),
            )
        }
        ok, msg = validate_bounds_against_data(p)
        assert ok is False
        assert "pH" in msg

    def test_wired_into_validate_process(self):
        ts = _ts([0.0, 1.0], [-1.0, 0.5])
        p = _make_biomass_process(ts)
        ok, results = validate_process(p)
        assert ok is False
        assert any("bounds_against_data" in message for _, message in results)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
