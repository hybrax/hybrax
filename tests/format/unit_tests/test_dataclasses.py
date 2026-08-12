"""
Tests for bp-format data structures (current architecture).
"""

import pytest
import jax.numpy as jnp

from bp_format import (
    TimeAxis,
    TimeSeries,
    StaticVariable,
    BiologicalOde,
    BioProcessMetadata,
    ProcessVariable,
    FeedMediumComponent,
    ReactorMediumComponent,
    FeedMedium,
    ReactorMedium,
    Inflow,
    Volume,
    BioProcess,
    AugmentedBioProcess,
    BioProcessCollection,
    silence_assumptions,
)
from bp_format.dataclasses import _format_biological_ode_lines
from bp_format.serialization import (
    save_process_collection,
    load_process_collection,
)


# ---------------------------------------------------------------------------
# Low-level structures
# ---------------------------------------------------------------------------


def test_time_axis_creation():
    ta = TimeAxis(unit="hours", start=0.0, end=48.0, time_reference="inoculation")
    assert ta.unit == "hours"
    assert ta.start == 0.0
    assert ta.end == 48.0
    assert ta.time_reference == "inoculation"


def test_timeseries_creation():
    ts = TimeSeries(
        times=jnp.array([0.0, 12.0, 24.0, 36.0, 48.0]),
        values=jnp.array([0.1, 1.2, 3.5, 5.8, 6.0]),
    )
    assert ts.times.shape == (5,)
    assert ts.values.shape == (5,)


def test_static_variable_creation():
    sv = StaticVariable(value=0.5)
    assert sv.value == 0.5


def test_bioprocess_metadata_creation():
    meta = BioProcessMetadata(name="batch_001", process_type="batch")
    assert meta.name == "batch_001"
    assert meta.process_type == "batch"
    assert meta.notes is None


def test_bioprocess_metadata_with_notes():
    meta = BioProcessMetadata(
        name="fb_001", process_type="fed_batch", notes="Replicate A"
    )
    assert meta.notes == "Replicate A"


# ---------------------------------------------------------------------------
# Component structures
# ---------------------------------------------------------------------------


def test_process_variable_timeseries():
    ts = TimeSeries(
        times=jnp.array([0.0, 1.0, 2.0]),
        values=jnp.array([37.0, 37.0, 37.0]),
    )
    pv = ProcessVariable(name="temperature", unit="°C", is_controlled=True, values=ts)
    assert pv.name == "temperature"
    assert pv.is_controlled is True
    assert hasattr(pv.values, "times")


def test_process_variable_static():
    sv = StaticVariable(value=7.0)
    pv = ProcessVariable(name="pH", unit="", is_controlled=False, values=sv)
    assert pv.name == "pH"
    assert isinstance(pv.values, StaticVariable)


def test_feed_medium_component_static():
    fmc = FeedMediumComponent(
        name="glucose",
        unit="g/L",
        concentration=StaticVariable(value=500.0),
        is_controlled=True,
    )
    assert fmc.name == "glucose"
    assert fmc.concentration.value == 500.0


def test_feed_medium_component_timeseries():
    ts = TimeSeries(times=jnp.array([0.0, 1.0]), values=jnp.array([100.0, 200.0]))
    fmc = FeedMediumComponent(
        name="glucose", unit="g/L", concentration=ts, is_controlled=True
    )
    assert hasattr(fmc.concentration, "times")


def test_reactor_medium_component_timeseries():
    ts = TimeSeries(times=jnp.array([0.0, 1.0, 2.0]), values=jnp.array([0.1, 0.5, 1.0]))
    rc = ReactorMediumComponent(name="biomass", unit="g/L", concentration=ts)
    assert rc.name == "biomass"
    assert rc.bounds == (0.0, None)


def test_reactor_medium_component_static():
    rc = ReactorMediumComponent(
        name="biomass",
        unit="g/L",
        concentration=StaticVariable(value=1.0),
    )
    assert isinstance(rc.concentration, StaticVariable)


# ---------------------------------------------------------------------------
# Medium-level structures
# ---------------------------------------------------------------------------


def test_feed_medium_empty_components():
    fm = FeedMedium(name="glucose_feed", density=1.0, density_unit="kg/L")
    assert fm.name == "glucose_feed"
    assert fm.components == {}


def test_feed_medium_with_components():
    fmc = FeedMediumComponent(
        name="glucose",
        unit="g/L",
        concentration=StaticVariable(value=500.0),
        is_controlled=True,
    )
    fm = FeedMedium(
        name="glucose_feed",
        density=1.1,
        density_unit="kg/L",
        components={"glucose": fmc},
    )
    assert "glucose" in fm.components
    assert fm.density == 1.1


def test_reactor_medium_empty_components():
    rm = ReactorMedium(name="medium", density=1.0, density_unit="kg/L")
    assert rm.components == {}


def test_reactor_medium_with_components():
    ts = TimeSeries(times=jnp.array([0.0, 1.0]), values=jnp.array([0.1, 0.5]))
    rc = ReactorMediumComponent(name="biomass", unit="g/L", concentration=ts)
    rm = ReactorMedium(
        name="medium", density=1.0, density_unit="kg/L", components={"biomass": rc}
    )
    assert "biomass" in rm.components


def test_volume_change_continuous():
    ts = TimeSeries(
        times=jnp.array([0.0, 5.0, 10.0]), values=jnp.array([0.0, 0.5, 1.0])
    )
    fm = FeedMedium(name="f", density=1.0, density_unit="kg/L")
    vc = Inflow(
        name="feed",
        unit="L",
        is_controlled=True,
        is_continuous=True,
        feed_medium=fm,
        values=ts,
    )
    assert vc.name == "feed"
    assert vc.is_continuous is True
    assert vc.values.times.shape == (3,)


def test_volume_change_discrete():
    ts = TimeSeries(times=jnp.array([2.0, 5.0]), values=jnp.array([0.5, 0.5]))
    fm = FeedMedium(name="f", density=1.0, density_unit="kg/L")
    vc = Inflow(
        name="bolus",
        unit="L",
        is_controlled=True,
        is_continuous=False,
        feed_medium=fm,
        values=ts,
    )
    assert vc.is_continuous is False


def test_volume_default_volume_changes():
    vol = Volume(initial_volume=1.0, unit="L")
    assert vol.initial_volume == 1.0
    assert vol.volume_changes == {}


def test_volume_with_changes():
    ts = TimeSeries(times=jnp.array([0.0, 10.0]), values=jnp.array([0.0, 0.5]))
    fm = FeedMedium(name="f", density=1.0, density_unit="kg/L")
    vc = Inflow(
        name="feed",
        unit="L",
        is_controlled=True,
        is_continuous=True,
        feed_medium=fm,
        values=ts,
    )
    vol = Volume(initial_volume=1.0, unit="L", volume_changes={"feed": vc})
    assert "feed" in vol.volume_changes


# ---------------------------------------------------------------------------
# Process-level structures
# ---------------------------------------------------------------------------


def test_bioprocess_minimal():
    process = BioProcess(
        metadata=BioProcessMetadata(name="batch_001", process_type="batch"),
        time_axis=TimeAxis(
            unit="hours", start=0.0, end=24.0, time_reference="inoculation"
        ),
        volume=Volume(initial_volume=1.0, unit="L"),
        reactor_medium=ReactorMedium(name="medium", density=1.0, density_unit="kg/L"),
    )
    assert process.metadata.name == "batch_001"
    assert process.metadata.process_type == "batch"
    assert process.process_variables == {}


def test_bioprocess_autogen_biological_ode_requires_biomass():
    """When ``biological_ode`` is omitted and the reactor medium has
    components, ``__post_init__`` auto-generates a default block — and
    that auto-generation requires a 'biomass' reactor component."""
    rm = ReactorMedium(
        name="medium",
        density=1.0,
        density_unit="kg/L",
        components={
            "glucose": ReactorMediumComponent(
                name="glucose",
                unit="g/L",
                concentration=TimeSeries(
                    times=jnp.array([0.0, 1.0]),
                    values=jnp.array([10.0, 5.0]),
                ),
            ),
        },
    )
    with pytest.raises(ValueError, match="biomass"):
        BioProcess(
            metadata=BioProcessMetadata(name="p", process_type="batch"),
            time_axis=TimeAxis(
                unit="hours", start=0.0, end=2.0, time_reference="inoculation"
            ),
            volume=Volume(initial_volume=1.0, unit="L"),
            reactor_medium=rm,
        )


def test_bioprocess_user_defined_biological_ode_skips_biomass_check():
    """When the user supplies their own ``biological_ode``,
    ``__post_init__`` skips auto-generation and the biomass-component
    requirement does not apply — the user's block is the source of truth."""
    rm = ReactorMedium(
        name="medium",
        density=1.0,
        density_unit="kg/L",
        components={
            "glucose": ReactorMediumComponent(
                name="glucose",
                unit="g/L",
                concentration=TimeSeries(
                    times=jnp.array([0.0, 1.0]),
                    values=jnp.array([10.0, 5.0]),
                ),
            ),
        },
    )
    user_block = BiologicalOde(
        rates={"q_glucose": (None, None)},
        derivatives={"glucose": "q_glucose"},
    )
    process = BioProcess(
        metadata=BioProcessMetadata(name="p", process_type="batch"),
        time_axis=TimeAxis(
            unit="hours", start=0.0, end=2.0, time_reference="inoculation"
        ),
        volume=Volume(initial_volume=1.0, unit="L"),
        reactor_medium=rm,
        biological_ode=user_block,
    )
    assert process.biological_ode is user_block


# ---------------------------------------------------------------------------
# Smart defaults: assumption notices, silencing, Inflow-concentration fill
# ---------------------------------------------------------------------------


def _process_with_incomplete_feed(silence=False):
    rm = ReactorMedium(
        name="medium",
        components={
            "biomass": ReactorMediumComponent(
                name="biomass",
                unit="g/L",
                concentration=TimeSeries(
                    times=jnp.array([0.0, 1.0]), values=jnp.array([1.0, 2.0])
                ),
            ),
            "product": ReactorMediumComponent(
                name="product",
                unit="g/L",
                concentration=TimeSeries(
                    times=jnp.array([0.0, 1.0]), values=jnp.array([0.0, 1.0])
                ),
            ),
        },
    )
    fm = FeedMedium(
        name="glucose_feed",
        components={
            "glucose": FeedMediumComponent(
                name="glucose", unit="g/L", concentration=StaticVariable(400.0)
            )
        },
    )
    vc = Inflow(
        name="feed",
        unit="L",
        is_controlled=True,
        is_continuous=True,
        feed_medium=fm,
        values=TimeSeries(times=jnp.array([0.0, 1.0]), values=jnp.array([0.0, 0.1])),
    )
    kwargs = dict(
        metadata=BioProcessMetadata(name="p", process_type="fed_batch"),
        time_axis=TimeAxis(unit="hours", start=0.0, end=1.0, time_reference="x"),
        volume=Volume(initial_volume=1.0, unit="L", volume_changes={"feed": vc}),
        reactor_medium=rm,
    )
    if silence:
        with silence_assumptions():
            process = BioProcess(**kwargs)
    else:
        process = BioProcess(**kwargs)
    return process, fm


def test_missing_inflow_concentrations_filled_with_zero_and_announced(capsys):
    process, fm = _process_with_incomplete_feed()
    assert set(fm.components.keys()) == {"glucose", "biomass", "product"}
    assert fm.components["biomass"].concentration.value == pytest.approx(0.0)
    assert fm.components["product"].concentration.value == pytest.approx(0.0)
    out = capsys.readouterr().out
    assert "[bp-format] Assumption:" in out
    assert "biomass" in out and "product" in out


def test_fully_specified_feed_produces_no_assumption_print(capsys):
    rm = ReactorMedium(
        name="medium",
        components={
            "biomass": ReactorMediumComponent(
                name="biomass",
                unit="g/L",
                concentration=TimeSeries(
                    times=jnp.array([0.0, 1.0]), values=jnp.array([1.0, 2.0])
                ),
            ),
        },
    )
    fm = FeedMedium(
        name="glucose_feed",
        components={
            "biomass": FeedMediumComponent(
                name="biomass", unit="g/L", concentration=StaticVariable(0.0)
            )
        },
    )
    vc = Inflow(
        name="feed",
        unit="L",
        is_controlled=True,
        is_continuous=True,
        feed_medium=fm,
        values=TimeSeries(times=jnp.array([0.0, 1.0]), values=jnp.array([0.0, 0.1])),
    )
    BioProcess(
        metadata=BioProcessMetadata(name="p", process_type="fed_batch"),
        time_axis=TimeAxis(unit="hours", start=0.0, end=1.0, time_reference="x"),
        volume=Volume(initial_volume=1.0, unit="L", volume_changes={"feed": vc}),
        reactor_medium=rm,
    )
    out = capsys.readouterr().out
    assert "feed medium" not in out.lower() or "did not define" not in out.lower()


def test_missing_feed_medium_entirely_is_not_filled():
    """feed_medium=None stays untouched by the fill — there's no reasonable
    way to fabricate an entire medium's identity."""
    rm = ReactorMedium(
        name="medium",
        components={
            "biomass": ReactorMediumComponent(
                name="biomass",
                unit="g/L",
                concentration=TimeSeries(
                    times=jnp.array([0.0, 1.0]), values=jnp.array([1.0, 2.0])
                ),
            ),
        },
    )
    vc = Inflow(
        name="feed",
        unit="L",
        is_controlled=True,
        is_continuous=True,
        feed_medium=None,
        values=TimeSeries(times=jnp.array([0.0, 1.0]), values=jnp.array([0.0, 0.1])),
    )
    process = BioProcess(
        metadata=BioProcessMetadata(name="p", process_type="fed_batch"),
        time_axis=TimeAxis(unit="hours", start=0.0, end=1.0, time_reference="x"),
        volume=Volume(initial_volume=1.0, unit="L", volume_changes={"feed": vc}),
        reactor_medium=rm,
    )
    assert process.volume.volume_changes["feed"].feed_medium is None


def test_silence_assumptions_suppresses_inflow_fill_notice(capsys):
    process, fm = _process_with_incomplete_feed(silence=True)
    assert fm.components["biomass"].concentration.value == pytest.approx(0.0)
    out = capsys.readouterr().out
    assert out == ""


def test_silence_assumptions_suppresses_biological_ode_notice(capsys):
    rm = ReactorMedium(
        name="medium",
        components={
            "biomass": ReactorMediumComponent(
                name="biomass",
                unit="g/L",
                concentration=TimeSeries(
                    times=jnp.array([0.0, 1.0]), values=jnp.array([1.0, 2.0])
                ),
            ),
        },
    )
    with silence_assumptions():
        BioProcess(
            metadata=BioProcessMetadata(name="p", process_type="batch"),
            time_axis=TimeAxis(
                unit="hours", start=0.0, end=1.0, time_reference="inoculation"
            ),
            volume=Volume(initial_volume=1.0, unit="L"),
            reactor_medium=rm,
        )
    out = capsys.readouterr().out
    assert out == ""


def test_silence_assumptions_restores_state_after_exception():
    with pytest.raises(RuntimeError):
        with silence_assumptions():
            raise RuntimeError("boom")
    # Verify the flag was restored by checking a notice fires again.
    rm = ReactorMedium(
        name="medium",
        components={
            "biomass": ReactorMediumComponent(
                name="biomass",
                unit="g/L",
                concentration=TimeSeries(
                    times=jnp.array([0.0, 1.0]), values=jnp.array([1.0, 2.0])
                ),
            ),
        },
    )
    from bp_format.dataclasses import _ANNOUNCE_ASSUMPTIONS

    assert _ANNOUNCE_ASSUMPTIONS is True


def test_density_defaults_are_silent(capsys):
    """density/density_unit default to 1.0/kg/L without any print — unlike
    the Inflow-concentration fill, this default never affects computed
    results (mechanistic.py never reads it), so there's nothing for a
    notice to usefully warn about."""
    fm = FeedMedium(name="f")
    rm = ReactorMedium(name="medium")
    assert fm.density == 1.0
    assert fm.density_unit == "kg/L"
    assert rm.density == 1.0
    assert rm.density_unit == "kg/L"
    out = capsys.readouterr().out
    assert out == ""


def test_format_biological_ode_lines_direct():
    bo = BiologicalOde(
        rates={"q_biomass": (None, None)},
        derivatives={"biomass": "q_biomass * biomass"},
    )
    lines = _format_biological_ode_lines(bo, prefix="  ")
    joined = "\n".join(lines)
    assert "Rates (1):" in joined
    assert "q_biomass" in joined
    assert "Derivatives (1):" in joined
    assert "q_biomass * biomass" in joined


def test_bioprocess_with_process_variables():
    ts = TimeSeries(times=jnp.array([0.0, 1.0]), values=jnp.array([37.0, 37.0]))
    pv = ProcessVariable(name="temperature", unit="°C", is_controlled=True, values=ts)
    process = BioProcess(
        metadata=BioProcessMetadata(name="p1", process_type="batch"),
        time_axis=TimeAxis(
            unit="hours", start=0.0, end=10.0, time_reference="inoculation"
        ),
        volume=Volume(initial_volume=1.0, unit="L"),
        reactor_medium=ReactorMedium(name="medium", density=1.0, density_unit="kg/L"),
        process_variables={"temperature": pv},
    )
    assert "temperature" in process.process_variables


def test_collection_case_study_fields():
    process = BioProcess(
        metadata=BioProcessMetadata(name="p1", process_type="batch"),
        time_axis=TimeAxis(
            unit="hours", start=0.0, end=24.0, time_reference="inoculation"
        ),
        volume=Volume(initial_volume=1.0, unit="L"),
        reactor_medium=ReactorMedium(name="medium", density=1.0, density_unit="kg/L"),
    )
    cs = BioProcessCollection(
        case_id="ecoli_study",
        organism="Escherichia coli",
        citation="Doe et al. 2024",
        processes={"p1": process},
    )
    assert cs.case_id == "ecoli_study"
    assert cs.organism == "Escherichia coli"
    assert "p1" in cs.processes


def test_collection_case_study_fields_multiple_processes():
    process_a = BioProcess(
        metadata=BioProcessMetadata(name="p1", process_type="batch"),
        time_axis=TimeAxis(
            unit="hours", start=0.0, end=24.0, time_reference="inoculation"
        ),
        volume=Volume(initial_volume=1.0, unit="L"),
        reactor_medium=ReactorMedium(name="medium", density=1.0, density_unit="kg/L"),
    )
    process_b = BioProcess(
        metadata=BioProcessMetadata(name="p2", process_type="fed_batch"),
        time_axis=TimeAxis(
            unit="hours", start=0.0, end=24.0, time_reference="inoculation"
        ),
        volume=Volume(initial_volume=1.0, unit="L"),
        reactor_medium=ReactorMedium(name="medium", density=1.0, density_unit="kg/L"),
    )
    cs = BioProcessCollection(
        case_id="ecoli_study",
        organism="Escherichia coli",
        citation="Doe et al. 2024",
        processes={"p1": process_a, "p2": process_b},
    )
    assert cs.case_id == "ecoli_study"
    assert "p1" in cs.processes
    assert len(cs.processes) == 2


def test_collection_without_case_study_fields_is_loose():
    """BioProcessCollection with all three case fields unset is raw/intermediate
    data — analogous to the deleted CaseStudy vs. loose split."""
    collection = BioProcessCollection()
    assert collection.case_id is None
    assert collection.organism is None
    assert collection.citation is None
    assert collection.metadata is None
    assert collection.processes == {}


# ---------------------------------------------------------------------------
# AugmentedBioProcess
# ---------------------------------------------------------------------------


def _make_minimal_process(name="p1"):
    return BioProcess(
        metadata=BioProcessMetadata(name=name, process_type="batch"),
        time_axis=TimeAxis(
            unit="hours", start=0.0, end=24.0, time_reference="inoculation"
        ),
        volume=Volume(initial_volume=1.0, unit="L"),
        reactor_medium=ReactorMedium(name="medium", density=1.0, density_unit="kg/L"),
    )


def test_augmented_bioprocess_inherits_bioprocess_fields():
    parent = _make_minimal_process("P0")
    child = AugmentedBioProcess(
        metadata=parent.metadata,
        time_axis=parent.time_axis,
        volume=parent.volume,
        reactor_medium=parent.reactor_medium,
        parent_process="P0",
    )
    assert isinstance(child, BioProcess)
    assert isinstance(child, AugmentedBioProcess)
    assert child.parent_process == "P0"
    assert child.time_axis is parent.time_axis
    assert child.process_variables == {}


def test_augmented_bioprocess_parent_process_required():
    with pytest.raises(TypeError):
        AugmentedBioProcess(  # type: ignore[call-arg]
            metadata=BioProcessMetadata(name="c", process_type="batch"),
            time_axis=TimeAxis(
                unit="hours",
                start=0.0,
                end=24.0,
                time_reference="inoculation",
            ),
            volume=Volume(initial_volume=1.0, unit="L"),
            reactor_medium=ReactorMedium(
                name="medium", density=1.0, density_unit="kg/L"
            ),
        )


def test_augmented_bioprocess_serialization_roundtrip(tmp_path):
    parent = _make_minimal_process("P0")
    child = AugmentedBioProcess(
        metadata=BioProcessMetadata(name="P0_aug", process_type="batch"),
        time_axis=parent.time_axis,
        volume=parent.volume,
        reactor_medium=parent.reactor_medium,
        parent_process="P0",
    )
    collection = BioProcessCollection(processes={"P0": parent, "P0_aug": child})
    out = tmp_path / "data.json"
    save_process_collection(collection, out)
    loaded = load_process_collection(out)

    assert isinstance(loaded.processes["P0"], BioProcess)
    assert not isinstance(loaded.processes["P0"], AugmentedBioProcess)
    assert isinstance(loaded.processes["P0_aug"], AugmentedBioProcess)
    assert loaded.processes["P0_aug"].parent_process == "P0"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
