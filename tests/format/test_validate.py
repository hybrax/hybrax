"""
Tests for bpbench.validate validation functions
"""

import pytest
import jax.numpy as jnp

from bpbench import (
    TimeSeries,
    StaticVariable,
    BioProcessMetadata,
    TimeAxis,
    ProcessVariable,
    ReactorMediumComponent,
    FeedMediumComponent,
    FeedMedium,
    ReactorMedium,
    VolumeChange,
    Volume,
    BioProcess,
    validate_timeseries_shape,
    validate_volume_change_sign,
    validate_volume_change_states,
    validate_biomass_in_reactor_medium,
    validate_process,
    validate_volume_consistency,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _ts(timepoints, values):
    """Shorthand for building a TimeSeries."""
    return TimeSeries(timepoints=jnp.array(timepoints), values=jnp.array(values))


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
        ts = _ts([0.0, 2.0, 1.0], [1.0, 2.0, 3.0])
        ok, msg = validate_timeseries_shape(ts, name="glucose")
        assert ok is False
        assert "monotonically" in msg

    def test_duplicate_timepoints(self):
        ts = _ts([0.0, 1.0, 1.0], [1.0, 2.0, 3.0])
        ok, msg = validate_timeseries_shape(ts)
        assert ok is False
        assert "monotonically" in msg

    def test_length_mismatch(self):
        ts = TimeSeries(
            timepoints=jnp.array([0.0, 1.0, 2.0]),
            values=jnp.array([1.0, 2.0]),
        )
        ok, msg = validate_timeseries_shape(ts)
        assert ok is False
        assert "does not match" in msg

    def test_name_appears_in_message(self):
        ts = _ts([0.0, 1.0], [1.0, 2.0])
        _, msg = validate_timeseries_shape(ts, name="myvar")
        assert "myvar" in msg


# ---------------------------------------------------------------------------
# validate_volume_change_sign
# ---------------------------------------------------------------------------

class TestValidateVolumeChangeSign:
    def _vc(self, values, name="feed"):
        return VolumeChange(
            name=name,
            unit="L",
            is_controlled=True,
            is_continuous=True,
            feed_medium=FeedMedium(name="f", density=1.0, density_unit="kg/L"),
            values=_ts([0.0, 1.0], values),
        )

    def test_purely_positive(self):
        vc = self._vc([0.1, 0.2])
        ok, msg = validate_volume_change_sign(vc)
        assert ok is True
        assert "positive" in msg

    def test_purely_negative(self):
        vc = self._vc([-0.1, -0.2])
        ok, msg = validate_volume_change_sign(vc)
        assert ok is True
        assert "negative" in msg

    def test_zero_values_are_positive(self):
        vc = self._vc([0.0, 0.0])
        ok, msg = validate_volume_change_sign(vc)
        assert ok is True

    def test_mixed_signs_invalid(self):
        vc = self._vc([0.1, -0.2])
        ok, msg = validate_volume_change_sign(vc)
        assert ok is False
        assert "mixed" in msg.lower()


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
        return VolumeChange(
            name="feed_vc",
            unit="L",
            is_controlled=True,
            is_continuous=True,
            feed_medium=feed_medium,
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
        vc = VolumeChange(
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
        # Unordered time points -> invalid TimeSeries
        bad_ts = _ts([0.0, 2.0, 1.0], [0.1, 0.5, 1.0])
        process = _make_process(
            reactor_components={
                "biomass": ReactorMediumComponent(
                    name="biomass", unit="g/L",
                    concentration=bad_ts,
                    is_intracellular=False,
                ),
            },
        )
        all_valid, messages = validate_process(process)
        assert all_valid is False
        assert any("monotonically" in m for m in messages)

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
        """Invalid TimeSeries in a process variable must be caught."""
        bad_ts = _ts([0.0, 2.0, 1.0], [0.1, 0.5, 1.0])  # non-monotonic
        pv = ProcessVariable(
            name="temperature", unit="°C", is_controlled=True, values=bad_ts
        )
        process = _make_process(
            reactor_components={
                "biomass": ReactorMediumComponent(
                    name="biomass", unit="g/L",
                    concentration=StaticVariable(value=1.0),
                    is_intracellular=False,
                ),
            },
            process_variables={"temperature": pv},
        )
        all_valid, messages = validate_process(process)
        assert all_valid is False
        assert any("temperature" in m for m in messages)


# ---------------------------------------------------------------------------
# validate_volume_consistency
# ---------------------------------------------------------------------------

class TestValidateVolumeConsistency:
    def _make_process_with_volume(self, initial_volume, changes):
        """Build a BioProcess with given volume changes for consistency tests."""
        volume_changes = {}
        for name, (is_continuous, timepoints, values, feed_medium) in changes.items():
            volume_changes[name] = VolumeChange(
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


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
