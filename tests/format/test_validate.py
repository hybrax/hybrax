"""
Tests for bp_format.validate validation functions
"""

import pytest
import jax.numpy as jnp

from bp_format import (
    TimeSeries,
    StaticVariable,
    BioProcessMetadata,
    TimeAxis,
    ProcessVariable,
    ReactorMediumComponent,
    FeedMediumComponent,
    FeedMedium,
    ReactorMedium,
    FeedVolumeChange,
    SampleVolumeChange,
    Volume,
    BioProcess,
    AugmentedBioProcess,
    BioProcessCollection,
    CaseStudy,
    validate_timeseries_shape,
    validate_volume_change_sign,
    validate_volume_change_states,
    validate_biomass_in_reactor_medium,
    validate_measurement_sampling_alignment,
    validate_intracellular_units,
    validate_process,
    validate_volume_consistency,
    validate_case_study,
    validate_augmented_parent_refs,
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
        time_axis=TimeAxis(unit="hours", start=0.0, end=10.0,
                           time_reference="inoculation"),
        volume=Volume(initial_volume=1.0, unit="L",
                      volume_changes=volume_changes),
        reactor_medium=ReactorMedium(
            name="medium",
            density=1.0,
            density_unit="kg/L",
            components=reactor_components,
        ),
        process_variables=process_variables,
    )


# ---------------------------------------------------------------------------
# validate_timeseries_shape
# ---------------------------------------------------------------------------

class TestValidateTimeSeriesShape:
    def test_valid_timeseries(self):
        ts = _ts([0.0, 1.0, 2.0], [1.0, 2.0, 3.0])
        ok, msg = validate_timeseries_shape(ts, name="glucose")
        assert ok is True
        assert "OK" in msg

    def test_single_point_is_valid(self):
        ts = _ts([0.0], [1.0])
        ok, msg = validate_timeseries_shape(ts)
        assert ok is True

    def test_unordered_timepoints(self):
        with pytest.raises(ValueError, match="strictly increasing"):
            _ts([0.0, 2.0, 1.0], [1.0, 2.0, 3.0])

    def test_duplicate_timepoints(self):
        with pytest.raises(ValueError, match="strictly increasing"):
            _ts([0.0, 1.0, 1.0], [1.0, 2.0, 3.0])

    def test_length_mismatch(self):
        with pytest.raises(ValueError, match="same length"):
            TimeSeries(
                times=jnp.array([0.0, 1.0, 2.0]),
                values=jnp.array([1.0, 2.0]),
            )

    def test_name_appears_in_message(self):
        ts = _ts([0.0, 1.0], [1.0, 2.0])
        _, msg = validate_timeseries_shape(ts, name="myvar")
        assert "myvar" in msg


# ---------------------------------------------------------------------------
# validate_volume_change_sign
# ---------------------------------------------------------------------------

class TestValidateVolumeChangeSign:
    def _feed_vc(self, values, name="feed"):
        return FeedVolumeChange(
            name=name,
            unit="L",
            is_controlled=True,
            is_continuous=True,
            feed_medium=FeedMedium(name="f", density=1.0, density_unit="kg/L"),
            values=_ts([0.0, 1.0], values),
        )

    def _sample_vc(self, values, name="sample"):
        return SampleVolumeChange(
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

    def test_mixed_signs_invalid(self):
        vc = self._feed_vc([0.1, -0.2])
        ok, msg = validate_volume_change_sign(vc)
        assert ok is False
        assert "negative" in msg.lower()


# ---------------------------------------------------------------------------
# validate_volume_change_states
# ---------------------------------------------------------------------------

class TestValidateVolumeChangeStates:
    def _reactor_comp(self, name, dynamic=True):
        conc = _ts([0.0, 1.0], [1.0, 2.0]) if dynamic else StaticVariable(value=1.0)
        return ReactorMediumComponent(
            name=name, unit="g/L", concentration=conc, is_intracellular=False
        )

    def _feed_medium(self, component_names):
        comps = {
            n: FeedMediumComponent(name=n, unit="g/L",
                                   concentration=StaticVariable(value=10.0),
                                   is_controlled=True)
            for n in component_names
        }
        return FeedMedium(name="feed", density=1.0, density_unit="kg/L",
                          components=comps)

    def _vc(self, feed_medium, positive=True):
        vals = [0.1, 0.2] if positive else [-0.1, -0.2]
        if positive:
            return FeedVolumeChange(
                name="feed_vc",
                unit="L",
                is_controlled=True,
                is_continuous=True,
                feed_medium=feed_medium,
                values=_ts([0.0, 1.0], vals),
            )
        else:
            return SampleVolumeChange(
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
            volume_changes={
                "f": self._vc(self._feed_medium(["biomass", "glucose"]))
            },
        )
        ok, msg = validate_volume_change_states(process)
        assert ok is True

    def test_missing_state_in_feed(self):
        process = _make_process(
            reactor_components={
                "biomass": self._reactor_comp("biomass"),
                "glucose": self._reactor_comp("glucose"),
            },
            volume_changes={
                "f": self._vc(self._feed_medium(["glucose"]))  # biomass missing
            },
        )
        ok, msg = validate_volume_change_states(process)
        assert ok is False
        assert "biomass" in msg

    def test_negative_volume_change_not_checked(self):
        """Negative (outflow) volume changes should skip state coverage check."""
        process = _make_process(
            reactor_components={
                "biomass": self._reactor_comp("biomass"),
            },
            volume_changes={
                "sample": self._vc(self._feed_medium([]), positive=False)
            },
        )
        ok, msg = validate_volume_change_states(process)
        assert ok is True

    def test_no_dynamic_states_skips_check(self):
        """If all reactor components are static, the check is skipped."""
        process = _make_process(
            reactor_components={
                "biomass": self._reactor_comp("biomass", dynamic=False),
            },
            volume_changes={
                "f": self._vc(self._feed_medium([]))
            },
        )
        ok, msg = validate_volume_change_states(process)
        assert ok is True
        assert "skipped" in msg.lower()


# ---------------------------------------------------------------------------
# validate_biomass_in_reactor_medium
# ---------------------------------------------------------------------------

class TestValidateBiomassInReactorMedium:
    def _comp(self, name):
        return ReactorMediumComponent(
            name=name, unit="g/L",
            concentration=StaticVariable(value=1.0),
            is_intracellular=False,
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
        process = _make_process(reactor_components={"glucose": self._comp("glucose")})
        ok, msg = validate_biomass_in_reactor_medium(process)
        assert ok is False
        assert "biomass" in msg.lower()

    def test_no_components(self):
        process = _make_process(reactor_components={})
        ok, msg = validate_biomass_in_reactor_medium(process)
        assert ok is False


# ---------------------------------------------------------------------------
# validate_process (integration)
# ---------------------------------------------------------------------------

class TestValidateProcess:
    def test_valid_process_returns_all_ok(self):
        biomass_ts = _ts([0.0, 1.0, 2.0], [0.1, 0.5, 1.0])
        feed_medium = FeedMedium(
            name="fm",
            density=1.0,
            density_unit="kg/L",
            components={
                "biomass": FeedMediumComponent(
                    name="biomass", unit="g/L",
                    concentration=StaticVariable(value=0.0),
                    is_controlled=True,
                ),
            },
        )
        vc = FeedVolumeChange(
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
                    name="biomass", unit="g/L",
                    concentration=biomass_ts,
                    is_intracellular=False,
                ),
            },
            volume_changes={"feed": vc},
        )
        all_valid, messages = validate_process(process)
        assert all_valid is True
        assert all("invalid" not in m.lower() for m in messages)

    def test_invalid_process_returns_false(self):
        # Unordered time points -> TimeSeries now rejects at construction
        with pytest.raises(ValueError, match="strictly increasing"):
            _ts([0.0, 2.0, 1.0], [0.1, 0.5, 1.0])

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
                    name="biomass", unit="g/L",
                    concentration=StaticVariable(value=1.0),
                    is_intracellular=False,
                ),
            },
        )
        all_valid, messages = validate_process(process)
        # biomass is present, no dynamic TS to fail -> should be valid
        assert all_valid is True

    def test_process_with_process_variable_timeseries(self):
        """Invalid TimeSeries (non-monotonic) is now rejected at construction."""
        with pytest.raises(ValueError, match="strictly increasing"):
            _ts([0.0, 2.0, 1.0], [0.1, 0.5, 1.0])


# ---------------------------------------------------------------------------
# validate_volume_consistency
# ---------------------------------------------------------------------------

class TestValidateVolumeConsistency:
    def _make_process_with_volume(self, initial_volume, changes):
        """Build a BioProcess with given volume changes for consistency tests."""
        volume_changes = {}
        for name, (is_continuous, timepoints, values, feed_medium) in changes.items():
            if any(v < 0 for v in values):
                volume_changes[name] = SampleVolumeChange(
                    name=name,
                    unit="L",
                    is_controlled=True,
                    is_continuous=is_continuous,
                    values=_ts(timepoints, values),
                )
            else:
                volume_changes[name] = FeedVolumeChange(
                    name=name,
                    unit="L",
                    is_controlled=True,
                    is_continuous=is_continuous,
                    feed_medium=feed_medium or FeedMedium(name="f", density=1.0, density_unit="kg/L"),
                    values=_ts(timepoints, values),
                )
        return BioProcess(
            metadata=BioProcessMetadata(name="test", process_type="fed_batch"),
            time_axis=TimeAxis(unit="hours", start=0.0, end=10.0,
                               time_reference="inoculation"),
            volume=Volume(initial_volume=initial_volume, unit="L",
                          volume_changes=volume_changes),
            reactor_medium=ReactorMedium(
                name="medium", density=1.0, density_unit="kg/L",
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
        assert "OK" in msg
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
        assert "inconsistency" in msg.lower()

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
# validate_case_study
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
            name="biomass", unit="g/L",
            concentration=biomass_ts,
            is_intracellular=False,
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
                name=name, unit="g/L",
                concentration=StaticVariable(value=0.0),
                is_controlled=True,
            )
            for name in component_names
        },
    )


class TestValidateCaseStudy:
    def _case_study(self, processes):
        return CaseStudy(
            case_id="cs1",
            organism="E. coli",
            citation="Test et al.",
            processes=processes,
        )

    def test_valid_case_study_all_ok(self):
        ts = _ts([0.0, 1.0, 2.0], [0.1, 0.5, 1.0])
        p1 = _make_biomass_process(ts)
        p2 = _make_biomass_process(_ts([0.0, 1.0, 2.0], [0.2, 0.6, 1.1]))
        cs = self._case_study({"run1": p1, "run2": p2})
        all_valid, report = validate_case_study(cs)
        assert all_valid is True
        assert "run1" in report
        assert "run2" in report
        assert "OK" in report["__consistency__"][0]

    def test_empty_case_study(self):
        cs = self._case_study({})
        all_valid, report = validate_case_study(cs)
        assert all_valid is True
        assert report == {}

    def test_single_process_case_study(self):
        ts = _ts([0.0, 1.0], [0.1, 0.5])
        p = _make_biomass_process(ts)
        cs = self._case_study({"run1": p})
        all_valid, report = validate_case_study(cs)
        assert all_valid is True
        assert "run1" in report
        assert "OK" in report["__consistency__"][0]

    def test_inconsistent_reactor_medium_components(self):
        """Processes with different reactor medium components should fail consistency."""
        ts = _ts([0.0, 1.0], [0.1, 0.5])
        p1 = _make_biomass_process(ts)
        # p2 has an extra 'glucose' component
        p2 = _make_biomass_process(
            ts,
            extra_components={
                "glucose": ReactorMediumComponent(
                    name="glucose", unit="g/L",
                    concentration=StaticVariable(value=10.0),
                    is_intracellular=False,
                )
            },
        )
        cs = self._case_study({"run1": p1, "run2": p2})
        all_valid, report = validate_case_study(cs)
        assert all_valid is False
        assert any("reactor medium" in e for e in report["__consistency__"])

    def test_inconsistent_process_variable_names(self):
        """Processes with different process variable names should fail consistency."""
        ts = _ts([0.0, 1.0], [0.1, 0.5])
        pv = ProcessVariable(
            name="temperature", unit="°C", is_controlled=True,
            values=_ts([0.0, 1.0], [37.0, 37.0]),
        )
        p1 = _make_biomass_process(ts, process_variables={"temperature": pv})
        p2 = _make_biomass_process(ts)  # no process variables
        cs = self._case_study({"run1": p1, "run2": p2})
        all_valid, report = validate_case_study(cs)
        assert all_valid is False
        assert any("process variables" in e for e in report["__consistency__"])

    def test_inconsistent_process_variable_types(self):
        """Same variable name but different type (TimeSeries vs StaticVariable) fails."""
        ts = _ts([0.0, 1.0], [0.1, 0.5])
        pv_ts = ProcessVariable(
            name="temperature", unit="°C", is_controlled=True,
            values=_ts([0.0, 1.0], [37.0, 37.0]),
        )
        pv_static = ProcessVariable(
            name="temperature", unit="°C", is_controlled=True,
            values=StaticVariable(value=37.0),
        )
        p1 = _make_biomass_process(ts, process_variables={"temperature": pv_ts})
        p2 = _make_biomass_process(ts, process_variables={"temperature": pv_static})
        cs = self._case_study({"run1": p1, "run2": p2})
        all_valid, report = validate_case_study(cs)
        assert all_valid is False
        assert any("process variables" in e for e in report["__consistency__"])

    def test_inconsistent_volume_change_names(self):
        """Processes with different volume change names should fail consistency."""
        ts = _ts([0.0, 1.0], [0.1, 0.5])
        vc = FeedVolumeChange(
            name="feed", unit="L", is_controlled=True, is_continuous=True,
            feed_medium=_make_feed_medium(["biomass"]),
            values=_ts([0.0, 1.0], [0.0, 0.1]),
        )
        p1 = _make_biomass_process(ts, volume_changes={"feed": vc})
        p2 = _make_biomass_process(ts)  # no volume changes
        cs = self._case_study({"run1": p1, "run2": p2})
        all_valid, report = validate_case_study(cs)
        assert all_valid is False
        assert any("volume change" in e for e in report["__consistency__"])

    def test_invalid_process_propagates_failure(self):
        """Non-monotonic times are now rejected at TimeSeries construction."""
        with pytest.raises(ValueError, match="strictly increasing"):
            _ts([0.0, 2.0, 1.0], [0.1, 0.5, 1.0])

    def test_wrong_type_raises_type_error(self):
        with pytest.raises(TypeError, match="CaseStudy"):
            validate_case_study("not a case study")

    def test_wrong_type_none_raises_type_error(self):
        with pytest.raises(TypeError):
            validate_case_study(None)

    def test_inconsistent_reactor_medium_units(self):
        """Same component name and type but different units should fail consistency."""
        ts = _ts([0.0, 1.0], [0.1, 0.5])
        p1 = _make_biomass_process(ts)  # biomass unit is "g/L"
        p2 = _make_biomass_process(ts)
        # Override the biomass component unit in p2
        p2.reactor_medium.components["biomass"] = ReactorMediumComponent(
            name="biomass", unit="mmol/L",
            concentration=ts,
            is_intracellular=False,
        )
        cs = self._case_study({"run1": p1, "run2": p2})
        all_valid, report = validate_case_study(cs)
        assert all_valid is False
        assert any("reactor medium" in e for e in report["__consistency__"])

    def test_inconsistent_process_variable_units(self):
        """Same process variable name and type but different units should fail."""
        ts = _ts([0.0, 1.0], [0.1, 0.5])
        pv1 = ProcessVariable(
            name="temperature", unit="°C", is_controlled=True,
            values=_ts([0.0, 1.0], [37.0, 37.0]),
        )
        pv2 = ProcessVariable(
            name="temperature", unit="K", is_controlled=True,
            values=_ts([0.0, 1.0], [310.0, 310.0]),
        )
        p1 = _make_biomass_process(ts, process_variables={"temperature": pv1})
        p2 = _make_biomass_process(ts, process_variables={"temperature": pv2})
        cs = self._case_study({"run1": p1, "run2": p2})
        all_valid, report = validate_case_study(cs)
        assert all_valid is False
        assert any("process variables" in e for e in report["__consistency__"])

    def test_inconsistent_volume_change_units(self):
        """Same volume change name but different units should fail consistency."""
        ts = _ts([0.0, 1.0], [0.1, 0.5])
        vc1 = FeedVolumeChange(
            name="feed", unit="L", is_controlled=True, is_continuous=True,
            feed_medium=_make_feed_medium(["biomass"]),
            values=_ts([0.0, 1.0], [0.0, 0.1]),
        )
        vc2 = FeedVolumeChange(
            name="feed", unit="mL", is_controlled=True, is_continuous=True,
            feed_medium=_make_feed_medium(["biomass"]),
            values=_ts([0.0, 1.0], [0.0, 100.0]),
        )
        p1 = _make_biomass_process(ts, volume_changes={"feed": vc1})
        p2 = _make_biomass_process(ts, volume_changes={"feed": vc2})
        cs = self._case_study({"run1": p1, "run2": p2})
        all_valid, report = validate_case_study(cs)
        assert all_valid is False
        assert any("volume change" in e for e in report["__consistency__"])


# ---------------------------------------------------------------------------
# validate_measurement_sampling_alignment
# ---------------------------------------------------------------------------

class TestValidateMeasurementSamplingAlignment:
    def _sample_vc(self, times, values):
        return SampleVolumeChange(
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
                    name="biomass", unit="g/L",
                    concentration=_ts([2.0, 5.0, 8.0], [0.1, 0.5, 1.0]),
                    is_intracellular=False,
                ),
            },
            volume_changes={"sample": self._sample_vc(sample_times, sample_vals)},
        )
        ok, msg = validate_measurement_sampling_alignment(process)
        assert ok is True
        assert "OK" in msg

    def test_small_delay_detected(self):
        """Measurement 0.0003 h after sampling in a 10 h process (0.003%) — should warn."""
        sample_times = [2.0, 5.0, 8.0]
        sample_vals = [-0.01, -0.01, -0.01]
        # Measurements are slightly after sampling
        meas_times = [2.0003, 5.0003, 8.0003]
        process = _make_process(
            reactor_components={
                "biomass": ReactorMediumComponent(
                    name="biomass", unit="g/L",
                    concentration=_ts(meas_times, [0.1, 0.5, 1.0]),
                    is_intracellular=False,
                ),
            },
            volume_changes={"sample": self._sample_vc(sample_times, sample_vals)},
        )
        ok, msg = validate_measurement_sampling_alignment(process)
        assert ok is False
        assert "biomass" in msg
        assert "ADF" in msg

    def test_large_gap_passes(self):
        """Measurements far from any sampling time — not a misalignment, should pass."""
        sample_times = [2.0, 8.0]
        sample_vals = [-0.01, -0.01]
        meas_times = [0.5, 5.0, 9.5]  # far from 2.0 and 8.0
        process = _make_process(
            reactor_components={
                "biomass": ReactorMediumComponent(
                    name="biomass", unit="g/L",
                    concentration=_ts(meas_times, [0.1, 0.5, 1.0]),
                    is_intracellular=False,
                ),
            },
            volume_changes={"sample": self._sample_vc(sample_times, sample_vals)},
        )
        ok, msg = validate_measurement_sampling_alignment(process)
        assert ok is True

    def test_no_sampling_events_skipped(self):
        """Process with no SampleVolumeChange — check should be skipped."""
        process = _make_process(
            reactor_components={
                "biomass": ReactorMediumComponent(
                    name="biomass", unit="g/L",
                    concentration=_ts([0.0, 5.0, 10.0], [0.1, 0.5, 1.0]),
                    is_intracellular=False,
                ),
            },
        )
        ok, msg = validate_measurement_sampling_alignment(process)
        assert ok is True
        assert "skipped" in msg.lower()


# ---------------------------------------------------------------------------
# validate_intracellular_units
# ---------------------------------------------------------------------------

class TestValidateIntracellularUnits:
    def test_same_units_pass(self):
        """Intracellular component with same unit as biomass — should pass."""
        process = _make_process(
            reactor_components={
                "biomass": ReactorMediumComponent(
                    name="biomass", unit="g/L",
                    concentration=_ts([0.0, 1.0], [0.1, 1.0]),
                    is_intracellular=False,
                ),
                "product": ReactorMediumComponent(
                    name="product", unit="g/L",
                    concentration=_ts([0.0, 1.0], [0.0, 0.5]),
                    is_intracellular=True,
                ),
            },
        )
        ok, msg = validate_intracellular_units(process)
        assert ok is True
        assert "OK" in msg

    def test_different_units_fail(self):
        """Intracellular component with mg/L vs biomass g/L — should warn."""
        process = _make_process(
            reactor_components={
                "biomass": ReactorMediumComponent(
                    name="biomass", unit="g/L",
                    concentration=_ts([0.0, 1.0], [0.1, 1.0]),
                    is_intracellular=False,
                ),
                "plasmid": ReactorMediumComponent(
                    name="plasmid", unit="mg/L",
                    concentration=_ts([0.0, 1.0], [0.0, 50.0]),
                    is_intracellular=True,
                ),
            },
        )
        ok, msg = validate_intracellular_units(process)
        assert ok is False
        assert "plasmid" in msg
        assert "mg/L" in msg

    def test_no_intracellular_pass(self):
        """No intracellular components — should pass."""
        process = _make_process(
            reactor_components={
                "biomass": ReactorMediumComponent(
                    name="biomass", unit="g/L",
                    concentration=_ts([0.0, 1.0], [0.1, 1.0]),
                    is_intracellular=False,
                ),
            },
        )
        ok, msg = validate_intracellular_units(process)
        assert ok is True

    def test_no_biomass_skipped(self):
        """No biomass component — check should be skipped."""
        process = _make_process(
            reactor_components={
                "glucose": ReactorMediumComponent(
                    name="glucose", unit="g/L",
                    concentration=_ts([0.0, 1.0], [10.0, 5.0]),
                    is_intracellular=False,
                ),
            },
        )
        ok, msg = validate_intracellular_units(process)
        assert ok is True
        assert "skipped" in msg.lower()


# ---------------------------------------------------------------------------
# validate_augmented_parent_refs
# ---------------------------------------------------------------------------

class TestValidateAugmentedParentRefs:
    def _aug_child(self, *, parent_process: str, name: str = "aug"):
        return AugmentedBioProcess(
            metadata=BioProcessMetadata(name=name, process_type="batch"),
            time_axis=TimeAxis(
                unit="hours", start=0.0, end=10.0,
                time_reference="inoculation",
            ),
            volume=Volume(initial_volume=1.0, unit="L"),
            reactor_medium=ReactorMedium(
                name="medium", density=1.0, density_unit="kg/L",
            ),
            parent_process=parent_process,
        )

    def _case_study(self, processes):
        return CaseStudy(
            case_id="cs1",
            organism="E. coli",
            citation="Test et al.",
            processes=processes,
        )

    def test_ok_when_parent_exists(self):
        ts = _ts([0.0, 1.0], [0.1, 0.5])
        parent = _make_biomass_process(ts)
        child = self._aug_child(parent_process="parent")
        cs = self._case_study({"parent": parent, "child": child})
        ok, messages = validate_augmented_parent_refs(cs)
        assert ok is True
        assert any("OK" in m for m in messages)

    def test_unknown_parent_fails(self):
        ts = _ts([0.0, 1.0], [0.1, 0.5])
        parent = _make_biomass_process(ts)
        child = self._aug_child(parent_process="ghost")
        cs = self._case_study({"parent": parent, "child": child})
        ok, messages = validate_augmented_parent_refs(cs)
        assert ok is False
        assert any("unknown parent_process" in m for m in messages)

    def test_rejects_augmented_of_augmented(self):
        ts = _ts([0.0, 1.0], [0.1, 0.5])
        parent = _make_biomass_process(ts)
        first_aug = self._aug_child(parent_process="parent", name="first_aug")
        chained = self._aug_child(parent_process="first_aug", name="chained")
        cs = self._case_study({
            "parent": parent,
            "first_aug": first_aug,
            "chained": chained,
        })
        ok, messages = validate_augmented_parent_refs(cs)
        assert ok is False
        assert any("itself augmented" in m for m in messages)

    def test_no_augmented_processes_ok(self):
        ts = _ts([0.0, 1.0], [0.1, 0.5])
        cs = self._case_study({"p1": _make_biomass_process(ts)})
        ok, messages = validate_augmented_parent_refs(cs)
        assert ok is True
        assert any("OK" in m for m in messages)

    def test_works_on_bioprocess_collection(self):
        ts = _ts([0.0, 1.0], [0.1, 0.5])
        parent = _make_biomass_process(ts)
        child = self._aug_child(parent_process="parent")
        collection = BioProcessCollection(
            processes={"parent": parent, "child": child}
        )
        ok, _ = validate_augmented_parent_refs(collection)
        assert ok is True

    def test_wrong_type_raises(self):
        with pytest.raises(TypeError):
            validate_augmented_parent_refs("not a collection")

    def test_validate_case_study_runs_augmented_parent_refs(self):
        ts = _ts([0.0, 1.0], [0.1, 0.5])
        parent = _make_biomass_process(ts)
        child = self._aug_child(parent_process="ghost")
        cs = self._case_study({"parent": parent, "child": child})
        all_valid, report = validate_case_study(cs)
        assert all_valid is False
        assert "__augmented__" in report
        assert any(
            "unknown parent_process" in m for m in report["__augmented__"]
        )


# ---------------------------------------------------------------------------
# validate_biological_ode + validate_bounds
# ---------------------------------------------------------------------------

from bp_format import (
    BiologicalOde,
    RateDecl,
    validate_biological_ode,
    validate_bounds,
)


def _make_intra_process():
    """Process with biomass + intracellular product + glucose. No volume changes."""
    return _make_process(
        reactor_components={
            "biomass": ReactorMediumComponent(
                "biomass", "g/L", StaticVariable(1.0), is_intracellular=False
            ),
            "product": ReactorMediumComponent(
                "product", "g/L", StaticVariable(0.0), is_intracellular=True
            ),
            "glucose": ReactorMediumComponent(
                "glucose", "g/L", StaticVariable(10.0), is_intracellular=False
            ),
        }
    )


class TestValidateBiologicalOde:
    def test_no_block_is_ok(self):
        p = _make_intra_process()
        ok, msg = validate_biological_ode(p)
        assert ok is True
        assert "not set" in msg

    def test_well_formed_block_passes(self):
        p = _make_intra_process()
        p.biological_ode = BiologicalOde(
            derived={"X_active": "biomass - product"},
            rates={"q_X": RateDecl(), "q_P": RateDecl(), "q_S": RateDecl()},
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
            derived={},
            rates={"q_X": RateDecl()},
            derivatives={"biomass": "q_X * biomass + zzz", "product": "0", "glucose": "0"},
        )
        ok, msg = validate_biological_ode(p)
        assert ok is False
        assert "zzz" in msg

    def test_missing_derivative_for_state_is_rejected(self):
        p = _make_intra_process()
        p.biological_ode = BiologicalOde(
            derived={},
            rates={"q_X": RateDecl()},
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
            derived={},
            rates={"q_X": RateDecl()},
            derivatives={
                "biomass": "0", "product": "0", "glucose": "0",
                "ghost": "q_X",
            },
        )
        ok, msg = validate_biological_ode(p)
        assert ok is False
        assert "ghost" in msg

    def test_derived_dependency_cycle_is_rejected(self):
        p = _make_intra_process()
        p.biological_ode = BiologicalOde(
            derived={"a": "b + 1", "b": "a * 2"},
            rates={"q_X": RateDecl()},
            derivatives={"biomass": "0", "product": "0", "glucose": "0"},
        )
        ok, msg = validate_biological_ode(p)
        assert ok is False
        assert "cycle" in msg.lower()

    def test_rate_name_collides_with_state_is_rejected(self):
        p = _make_intra_process()
        p.biological_ode = BiologicalOde(
            derived={},
            rates={"biomass": RateDecl()},
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
            derived={},
            rates={"feed_rate": RateDecl()},
            derivatives={"biomass": "0", "product": "0", "glucose": "0"},
        )
        ok, msg = validate_biological_ode(p)
        assert ok is False
        assert "feed_rate" in msg

    def test_invalid_rate_bounds_lo_greater_than_hi_is_rejected(self):
        p = _make_intra_process()
        p.biological_ode = BiologicalOde(
            derived={},
            rates={"q_X": RateDecl(bounds=(2.0, 1.0))},
            derivatives={"biomass": "q_X * biomass", "product": "0", "glucose": "0"},
        )
        ok, msg = validate_biological_ode(p)
        assert ok is False
        assert "invalid" in msg.lower()


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
                "pH", "", is_controlled=False, values=StaticVariable(7.0),
                bounds=(14.0, 0.0),
            )
        }
        ok, msg = validate_bounds(p)
        assert ok is False
        assert "pH" in msg


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
