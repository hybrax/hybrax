"""
Tests for bpbench.inspect utility functions.
"""

import pytest
import jax.numpy as jnp

from bpbench import (
    BioProcess,
    BioProcessMetadata,
    TimeAxis,
    TimeSeries,
    StaticVariable,
    ProcessVariable,
    ReactorMediumComponent,
    ReactorMedium,
    FeedMedium,
    FeedMediumComponent,
    VolumeChange,
    Volume,
    CaseStudy,
    BenchmarkDataset,
    print_process_structure,
    print_dataset_structure,
    plot_process,
    plot_case_study,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def simple_process():
    ts = TimeSeries(
        timepoints=jnp.array([0., 12., 24., 36., 48.]),
        values=jnp.array([0.1, 1.2, 3.5, 5.8, 6.0]),
    )
    rc = ReactorMediumComponent(name="biomass", unit="g/L", concentration=ts, is_intracellular=False)
    rm = ReactorMedium(name="medium", density=1.0, density_unit="kg/L", components={"biomass": rc})
    return BioProcess(
        metadata=BioProcessMetadata(name="test_001", process_type="batch"),
        time_axis=TimeAxis(unit="hours", start=0.0, end=48.0, time_reference="inoculation"),
        volume=Volume(initial_volume=1.0, unit="L"),
        reactor_medium=rm,
    )


@pytest.fixture
def complex_process():
    biomass_ts = TimeSeries(
        timepoints=jnp.array([0., 12., 24., 36., 48.]),
        values=jnp.array([0.1, 1.2, 3.5, 5.8, 6.0]),
    )
    glucose_ts = TimeSeries(
        timepoints=jnp.array([0., 12., 24., 36., 48.]),
        values=jnp.array([20.0, 15.0, 8.0, 2.0, 0.5]),
    )
    biomass_rc = ReactorMediumComponent(
        name="biomass", unit="g/L", concentration=biomass_ts, is_intracellular=False
    )
    glucose_rc = ReactorMediumComponent(
        name="glucose", unit="g/L", concentration=glucose_ts, is_intracellular=False
    )
    rm = ReactorMedium(
        name="medium", density=1.0, density_unit="kg/L",
        components={"biomass": biomass_rc, "glucose": glucose_rc},
    )

    temp_pv = ProcessVariable(
        name="temperature", unit="°C", is_controlled=True,
        values=TimeSeries(
            timepoints=jnp.array([0., 12., 24.]),
            values=jnp.array([37.0, 37.0, 37.0]),
        ),
    )
    ph_pv = ProcessVariable(
        name="pH", unit="", is_controlled=False, values=StaticVariable(value=7.0)
    )

    fm = FeedMedium(
        name="glucose_feed", density=1.1, density_unit="kg/L",
        components={
            "glucose": FeedMediumComponent(
                name="glucose", unit="g/L",
                concentration=StaticVariable(value=500.0), is_controlled=True
            )
        },
    )
    feed_ts = TimeSeries(
        timepoints=jnp.array([0., 12., 24., 36., 48.]),
        values=jnp.array([0.0, 0.05, 0.10, 0.15, 0.20]),
    )
    vc = VolumeChange(
        name="glucose_feed", unit="L",
        is_controlled=True, is_continuous=True,
        feed_medium=fm, values=feed_ts,
    )
    vol = Volume(initial_volume=1.0, unit="L", volume_changes={"glucose_feed": vc})

    return BioProcess(
        metadata=BioProcessMetadata(name="test_fed_batch_001", process_type="fed_batch",
                                    notes="Replicate 1"),
        time_axis=TimeAxis(unit="hours", start=0.0, end=48.0, time_reference="inoculation"),
        volume=vol,
        reactor_medium=rm,
        process_variables={"temperature": temp_pv, "pH": ph_pv},
    )


# ---------------------------------------------------------------------------
# print_process_structure tests (verbosity=3 – full, default)
# ---------------------------------------------------------------------------

def test_print_process_structure_simple(simple_process, capsys):
    print_process_structure(simple_process)
    captured = capsys.readouterr()

    assert "BioProcess Structure" in captured.out
    assert "test_001" in captured.out
    assert "batch" in captured.out
    assert "0.00 to 48.00 hours" in captured.out
    assert "biomass" in captured.out
    assert "g/L" in captured.out
    assert "Initial: 1.0 L" in captured.out


def test_print_process_structure_shows_process_name(simple_process, capsys):
    print_process_structure(simple_process)
    captured = capsys.readouterr()
    assert "Process Name: test_001" in captured.out


def test_print_process_structure_shows_process_type(simple_process, capsys):
    print_process_structure(simple_process)
    captured = capsys.readouterr()
    assert "Process Type: batch" in captured.out


def test_print_process_structure_shows_time_range(simple_process, capsys):
    print_process_structure(simple_process)
    captured = capsys.readouterr()
    assert "Range: 0.00 to 48.00 hours" in captured.out
    assert "inoculation" in captured.out


def test_print_process_structure_shows_reactor_medium(simple_process, capsys):
    print_process_structure(simple_process)
    captured = capsys.readouterr()
    assert "Reactor Medium:" in captured.out
    assert "medium" in captured.out
    assert "Components: (1 total)" in captured.out


def test_print_process_structure_complex_shows_variables(complex_process, capsys):
    print_process_structure(complex_process)
    captured = capsys.readouterr()

    assert "test_fed_batch_001" in captured.out
    assert "fed_batch" in captured.out
    assert "Replicate 1" in captured.out

    assert "biomass" in captured.out
    assert "glucose" in captured.out

    assert "Process Variables:" in captured.out
    assert "temperature" in captured.out
    assert "pH" in captured.out

    assert "Volume:" in captured.out
    assert "Initial: 1.0 L" in captured.out
    assert "Volume Changes: (1 total)" in captured.out
    assert "glucose_feed" in captured.out


def test_print_process_structure_no_crash_minimal():
    process = BioProcess(
        metadata=BioProcessMetadata(name="minimal", process_type="batch"),
        time_axis=TimeAxis(unit="hours", start=0.0, end=10.0, time_reference="inoculation"),
        volume=Volume(initial_volume=1.0, unit="L"),
        reactor_medium=ReactorMedium(name="m", density=1.0, density_unit="kg/L"),
    )
    try:
        print_process_structure(process)
    except Exception as exc:
        pytest.fail(f"print_process_structure raised {exc} unexpectedly!")


def test_print_process_structure_static_concentration(capsys):
    rc = ReactorMediumComponent(
        name="biomass", unit="g/L",
        concentration=StaticVariable(value=2.0), is_intracellular=False
    )
    rm = ReactorMedium(name="m", density=1.0, density_unit="kg/L", components={"biomass": rc})
    process = BioProcess(
        metadata=BioProcessMetadata(name="static_test", process_type="batch"),
        time_axis=TimeAxis(unit="hours", start=0.0, end=5.0, time_reference="inoculation"),
        volume=Volume(initial_volume=1.0, unit="L"),
        reactor_medium=rm,
    )
    print_process_structure(process)
    captured = capsys.readouterr()
    assert "Static Concentration: 2.0" in captured.out


# ---------------------------------------------------------------------------
# Verbosity level 2 tests
# ---------------------------------------------------------------------------

def test_print_process_structure_verbosity2_shows_names(complex_process, capsys):
    print_process_structure(complex_process, verbosity=2)
    captured = capsys.readouterr()
    assert "test_fed_batch_001" in captured.out
    assert "temperature" in captured.out
    assert "biomass" in captured.out


def test_print_process_structure_verbosity2_no_units(complex_process, capsys):
    print_process_structure(complex_process, verbosity=2)
    captured = capsys.readouterr()
    # Value ranges should not appear at verbosity 2
    assert "Value range:" not in captured.out
    # Units not printed individually
    assert "Unit:" not in captured.out


# ---------------------------------------------------------------------------
# Verbosity level 1 tests
# ---------------------------------------------------------------------------

def test_print_process_structure_verbosity1_lists_variables(complex_process, capsys):
    print_process_structure(complex_process, verbosity=1)
    captured = capsys.readouterr()
    assert "Process:" in captured.out
    assert "temperature" in captured.out or "Process Variables:" in captured.out


def test_print_process_structure_verbosity1_minimal_output(complex_process, capsys):
    print_process_structure(complex_process, verbosity=1)
    captured = capsys.readouterr()
    # Should not show detailed fields
    assert "Value range:" not in captured.out
    assert "Unit:" not in captured.out
    assert "TimeSeries Data:" not in captured.out


# ---------------------------------------------------------------------------
# Backward-compat alias: print_structure
# ---------------------------------------------------------------------------

def test_print_structure_alias(simple_process, capsys):
    print_structure(simple_process)
    captured = capsys.readouterr()
    assert "BioProcess Structure" in captured.out
    assert "Process Name: test_001" in captured.out
    assert "Process Type: batch" in captured.out
    assert "Range: 0.00 to 48.00 hours" in captured.out


# ---------------------------------------------------------------------------
# print_structure tests kept for regression (use alias)
# ---------------------------------------------------------------------------

def test_print_structure_simple(simple_process, capsys):
    print_structure(simple_process)
    captured = capsys.readouterr()

    assert "BioProcess Structure" in captured.out
    assert "test_001" in captured.out
    assert "batch" in captured.out
    assert "0.00 to 48.00 hours" in captured.out
    assert "biomass" in captured.out
    assert "g/L" in captured.out
    assert "Initial: 1.0 L" in captured.out


def test_print_structure_shows_process_name(simple_process, capsys):
    print_structure(simple_process)
    captured = capsys.readouterr()
    assert "Process Name: test_001" in captured.out


def test_print_structure_shows_process_type(simple_process, capsys):
    print_structure(simple_process)
    captured = capsys.readouterr()
    assert "Process Type: batch" in captured.out


def test_print_structure_shows_time_range(simple_process, capsys):
    print_structure(simple_process)
    captured = capsys.readouterr()
    assert "Range: 0.00 to 48.00 hours" in captured.out
    assert "inoculation" in captured.out


def test_print_structure_shows_reactor_medium(simple_process, capsys):
    print_structure(simple_process)
    captured = capsys.readouterr()
    assert "Reactor Medium:" in captured.out
    assert "medium" in captured.out
    assert "Components: (1 total)" in captured.out


def test_print_structure_complex_shows_variables(complex_process, capsys):
    print_structure(complex_process)
    captured = capsys.readouterr()

    assert "test_fed_batch_001" in captured.out
    assert "fed_batch" in captured.out
    assert "Replicate 1" in captured.out

    assert "biomass" in captured.out
    assert "glucose" in captured.out

    assert "Process Variables:" in captured.out
    assert "temperature" in captured.out
    assert "pH" in captured.out

    assert "Volume:" in captured.out
    assert "Initial: 1.0 L" in captured.out
    assert "Volume Changes: (1 total)" in captured.out
    assert "glucose_feed" in captured.out


def test_print_structure_no_crash_minimal():
    process = BioProcess(
        metadata=BioProcessMetadata(name="minimal", process_type="batch"),
        time_axis=TimeAxis(unit="hours", start=0.0, end=10.0, time_reference="inoculation"),
        volume=Volume(initial_volume=1.0, unit="L"),
        reactor_medium=ReactorMedium(name="m", density=1.0, density_unit="kg/L"),
    )
    try:
        print_structure(process)
    except Exception as exc:
        pytest.fail(f"print_structure raised {exc} unexpectedly!")


def test_print_structure_static_concentration(capsys):
    """Static concentrations should display without crashing."""
    rc = ReactorMediumComponent(
        name="biomass", unit="g/L",
        concentration=StaticVariable(value=2.0), is_intracellular=False
    )
    rm = ReactorMedium(name="m", density=1.0, density_unit="kg/L", components={"biomass": rc})
    process = BioProcess(
        metadata=BioProcessMetadata(name="static_test", process_type="batch"),
        time_axis=TimeAxis(unit="hours", start=0.0, end=5.0, time_reference="inoculation"),
        volume=Volume(initial_volume=1.0, unit="L"),
        reactor_medium=rm,
    )
    print_structure(process)
    captured = capsys.readouterr()
    assert "Static Concentration: 2.0" in captured.out


# ---------------------------------------------------------------------------
# print_dataset_structure tests
# ---------------------------------------------------------------------------

def _make_minimal_process(name):
    ts = TimeSeries(timepoints=jnp.array([0., 1.]), values=jnp.array([0.1, 0.5]))
    rc = ReactorMediumComponent(name="biomass", unit="g/L", concentration=ts, is_intracellular=False)
    rm = ReactorMedium(name="m", density=1.0, density_unit="kg/L", components={"biomass": rc})
    return BioProcess(
        metadata=BioProcessMetadata(name=name, process_type="batch"),
        time_axis=TimeAxis(unit="hours", start=0.0, end=10.0, time_reference="inoculation"),
        volume=Volume(initial_volume=1.0, unit="L"),
        reactor_medium=rm,
    )


@pytest.fixture
def sample_dataset():
    p1 = _make_minimal_process("p1")
    p2 = _make_minimal_process("p2")
    cs = CaseStudy(
        case_id="ecoli_study", organism="Escherichia coli", citation="Doe 2024",
        processes={"p1": p1, "p2": p2},
    )
    return BenchmarkDataset(
        metadata={"name": "TestDataset", "version": "0.1.0"},
        case_studies={"ecoli": cs},
    )


def test_print_dataset_structure_header(sample_dataset, capsys):
    print_dataset_structure(sample_dataset)
    captured = capsys.readouterr()
    assert "Benchmark Dataset Structure" in captured.out


def test_print_dataset_structure_metadata(sample_dataset, capsys):
    print_dataset_structure(sample_dataset)
    captured = capsys.readouterr()
    assert "TestDataset" in captured.out
    assert "0.1.0" in captured.out


def test_print_dataset_structure_case_study(sample_dataset, capsys):
    print_dataset_structure(sample_dataset)
    captured = capsys.readouterr()
    assert "ecoli" in captured.out
    assert "Escherichia coli" in captured.out
    assert "Doe 2024" in captured.out


def test_print_dataset_structure_processes(sample_dataset, capsys):
    print_dataset_structure(sample_dataset)
    captured = capsys.readouterr()
    assert "p1" in captured.out
    assert "p2" in captured.out
    assert "Processes: 2" in captured.out


def test_print_dataset_structure_total_datapoints(sample_dataset, capsys):
    print_dataset_structure(sample_dataset)
    captured = capsys.readouterr()
    # Each process has 1 biomass TimeSeries with 2 points => 2 processes * 2 = 4 total
    assert "Total datapoints in dataset: 4" in captured.out


def test_print_dataset_structure_empty(capsys):
    dataset = BenchmarkDataset()
    print_dataset_structure(dataset)
    captured = capsys.readouterr()
    assert "Benchmark Dataset Structure" in captured.out
    assert "(no case studies)" in captured.out


def test_print_dataset_structure_verbosity2(sample_dataset, capsys):
    print_dataset_structure(sample_dataset, verbosity=2)
    captured = capsys.readouterr()
    assert "Benchmark Dataset Structure" in captured.out
    assert "ecoli" in captured.out
    assert "Escherichia coli" in captured.out
    assert "p1" in captured.out
    # Citation and datapoints should NOT appear
    assert "Doe 2024" not in captured.out
    assert "datapoints:" not in captured.out


def test_print_dataset_structure_verbosity1(sample_dataset, capsys):
    print_dataset_structure(sample_dataset, verbosity=1)
    captured = capsys.readouterr()
    assert "Benchmark Dataset Structure" in captured.out
    assert "ecoli" in captured.out
    # Organism, citation and process names should NOT appear at verbosity 1
    assert "Escherichia coli" not in captured.out
    assert "Doe 2024" not in captured.out
    assert "p1" not in captured.out


def test_print_dataset_structure_empty_verbosity1(capsys):
    dataset = BenchmarkDataset()
    print_dataset_structure(dataset, verbosity=1)
    captured = capsys.readouterr()
    assert "Benchmark Dataset Structure" in captured.out
    assert "(no case studies)" in captured.out


# ---------------------------------------------------------------------------
# plot_process smoke tests
# ---------------------------------------------------------------------------

def test_plot_process_returns_figure(complex_process):
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig = plot_process(complex_process)
    assert fig is not None
    plt.close(fig)


def test_plot_process_simple(simple_process):
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig = plot_process(simple_process)
    assert fig is not None
    plt.close(fig)


def test_plot_process_empty():
    """A process with no plottable variables should still return a figure."""
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    process = BioProcess(
        metadata=BioProcessMetadata(name="empty", process_type="batch"),
        time_axis=TimeAxis(unit="hours", start=0.0, end=10.0, time_reference="inoculation"),
        volume=Volume(initial_volume=1.0, unit="L"),
        reactor_medium=ReactorMedium(name="m", density=1.0, density_unit="kg/L"),
    )
    fig = plot_process(process)
    assert fig is not None
    plt.close(fig)


# ---------------------------------------------------------------------------
# plot_case_study smoke tests
# ---------------------------------------------------------------------------

def test_plot_case_study_returns_figure(sample_dataset):
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    cs = list(sample_dataset.case_studies.values())[0]
    fig = plot_case_study(cs)
    assert fig is not None
    plt.close(fig)


def test_plot_case_study_empty():
    """A case study with no processes should still return a figure."""
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    cs = CaseStudy(case_id="empty", organism="None", citation="None")
    fig = plot_case_study(cs)
    assert fig is not None
    plt.close(fig)

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
