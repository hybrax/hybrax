"""
Tests for bp_format.inspect utility functions.
"""

import pytest
import jax.numpy as jnp
import numpy as np

from bp_format import (
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
    FeedVolumeChange,
    SampleVolumeChange,
    Volume,
    CaseStudy,
    BenchmarkDataset,
    print_process_structure,
    print_dataset_structure,
    plot_process,
    plot_case_study,
)
from bp_format.splines import build_pseudobatch_transform


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def simple_process():
    ts = TimeSeries(
        times=jnp.array([0.0, 12.0, 24.0, 36.0, 48.0]),
        values=jnp.array([0.1, 1.2, 3.5, 5.8, 6.0]),
    )
    rc = ReactorMediumComponent(
        name="biomass", unit="g/L", concentration=ts
    )
    rm = ReactorMedium(
        name="medium", density=1.0, density_unit="kg/L", components={"biomass": rc}
    )
    return BioProcess(
        metadata=BioProcessMetadata(name="test_001", process_type="batch"),
        time_axis=TimeAxis(
            unit="hours", start=0.0, end=48.0, time_reference="inoculation"
        ),
        volume=Volume(initial_volume=1.0, unit="L"),
        reactor_medium=rm,
    )


@pytest.fixture
def complex_process():
    biomass_ts = TimeSeries(
        times=jnp.array([0.0, 12.0, 24.0, 36.0, 48.0]),
        values=jnp.array([0.1, 1.2, 3.5, 5.8, 6.0]),
    )
    glucose_ts = TimeSeries(
        times=jnp.array([0.0, 12.0, 24.0, 36.0, 48.0]),
        values=jnp.array([20.0, 15.0, 8.0, 2.0, 0.5]),
    )
    biomass_rc = ReactorMediumComponent(
        name="biomass", unit="g/L", concentration=biomass_ts
    )
    glucose_rc = ReactorMediumComponent(
        name="glucose", unit="g/L", concentration=glucose_ts
    )
    rm = ReactorMedium(
        name="medium",
        density=1.0,
        density_unit="kg/L",
        components={"biomass": biomass_rc, "glucose": glucose_rc},
    )

    temp_pv = ProcessVariable(
        name="temperature",
        unit="°C",
        is_controlled=True,
        values=TimeSeries(
            times=jnp.array([0.0, 12.0, 24.0]),
            values=jnp.array([37.0, 37.0, 37.0]),
        ),
    )
    ph_pv = ProcessVariable(
        name="pH", unit="", is_controlled=False, values=StaticVariable(value=7.0)
    )

    fm = FeedMedium(
        name="glucose_feed",
        density=1.1,
        density_unit="kg/L",
        components={
            "glucose": FeedMediumComponent(
                name="glucose",
                unit="g/L",
                concentration=StaticVariable(value=500.0),
                is_controlled=True,
            )
        },
    )
    feed_ts = TimeSeries(
        times=jnp.array([0.0, 12.0, 24.0, 36.0, 48.0]),
        values=jnp.array([0.0, 0.05, 0.10, 0.15, 0.20]),
    )
    vc = FeedVolumeChange(
        name="glucose_feed",
        unit="L",
        is_controlled=True,
        is_continuous=True,
        feed_medium=fm,
        values=feed_ts,
    )
    vol = Volume(initial_volume=1.0, unit="L", volume_changes={"glucose_feed": vc})

    return BioProcess(
        metadata=BioProcessMetadata(
            name="test_fed_batch_001", process_type="fed_batch", notes="Replicate 1"
        ),
        time_axis=TimeAxis(
            unit="hours", start=0.0, end=48.0, time_reference="inoculation"
        ),
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
        time_axis=TimeAxis(
            unit="hours", start=0.0, end=10.0, time_reference="inoculation"
        ),
        volume=Volume(initial_volume=1.0, unit="L"),
        reactor_medium=ReactorMedium(name="m", density=1.0, density_unit="kg/L"),
    )
    try:
        print_process_structure(process)
    except Exception as exc:
        pytest.fail(f"print_process_structure raised {exc} unexpectedly!")


def test_print_process_structure_no_metadata_uses_fallbacks(capsys):
    process = BioProcess(
        metadata=None,
        time_axis=TimeAxis(
            unit="hours", start=0.0, end=10.0, time_reference="inoculation"
        ),
        volume=Volume(initial_volume=1.0, unit="L"),
        reactor_medium=ReactorMedium(name="m", density=1.0, density_unit="kg/L"),
    )
    print_process_structure(process)
    captured = capsys.readouterr()
    assert "Process Name: <unnamed process>" in captured.out
    assert "Process Type: <unknown type>" in captured.out


def test_print_process_structure_static_concentration(capsys):
    rc = ReactorMediumComponent(
        name="biomass",
        unit="g/L",
        concentration=StaticVariable(value=2.0),
    )
    rm = ReactorMedium(
        name="m", density=1.0, density_unit="kg/L", components={"biomass": rc}
    )
    process = BioProcess(
        metadata=BioProcessMetadata(name="static_test", process_type="batch"),
        time_axis=TimeAxis(
            unit="hours", start=0.0, end=5.0, time_reference="inoculation"
        ),
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


def test_print_process_structure_verbosity2_no_metadata_uses_fallbacks(capsys):
    process = BioProcess(
        metadata=None,
        time_axis=TimeAxis(
            unit="hours", start=0.0, end=10.0, time_reference="inoculation"
        ),
        volume=Volume(initial_volume=1.0, unit="L"),
        reactor_medium=ReactorMedium(name="m", density=1.0, density_unit="kg/L"),
    )
    print_process_structure(process, verbosity=2)
    captured = capsys.readouterr()
    assert "Process Name: <unnamed process>" in captured.out
    assert "Process Type: <unknown type>" in captured.out


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


def test_print_process_structure_verbosity1_no_metadata_uses_fallbacks(capsys):
    process = BioProcess(
        metadata=None,
        time_axis=TimeAxis(
            unit="hours", start=0.0, end=10.0, time_reference="inoculation"
        ),
        volume=Volume(initial_volume=1.0, unit="L"),
        reactor_medium=ReactorMedium(name="m", density=1.0, density_unit="kg/L"),
    )
    print_process_structure(process, verbosity=1)
    captured = capsys.readouterr()
    assert "Process: <unnamed process> (<unknown type>)" in captured.out


# ---------------------------------------------------------------------------
# print_dataset_structure tests
# ---------------------------------------------------------------------------


def _make_minimal_process(name):
    ts = TimeSeries(times=jnp.array([0.0, 1.0]), values=jnp.array([0.1, 0.5]))
    rc = ReactorMediumComponent(
        name="biomass", unit="g/L", concentration=ts
    )
    rm = ReactorMedium(
        name="m", density=1.0, density_unit="kg/L", components={"biomass": rc}
    )
    return BioProcess(
        metadata=BioProcessMetadata(name=name, process_type="batch"),
        time_axis=TimeAxis(
            unit="hours", start=0.0, end=10.0, time_reference="inoculation"
        ),
        volume=Volume(initial_volume=1.0, unit="L"),
        reactor_medium=rm,
    )


@pytest.fixture
def sample_dataset():
    p1 = _make_minimal_process("p1")
    p2 = _make_minimal_process("p2")
    cs = CaseStudy(
        case_id="ecoli_study",
        organism="Escherichia coli",
        citation="Doe 2024",
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


def test_print_dataset_structure_handles_process_without_metadata(capsys):
    process = BioProcess(
        metadata=None,
        time_axis=TimeAxis(
            unit="hours", start=0.0, end=10.0, time_reference="inoculation"
        ),
        volume=Volume(initial_volume=1.0, unit="L"),
        reactor_medium=ReactorMedium(name="m", density=1.0, density_unit="kg/L"),
    )
    cs = CaseStudy(
        case_id="raw_data",
        organism="Unknown",
        citation="n/a",
        processes={"p1": process},
    )
    dataset = BenchmarkDataset(case_studies={"raw": cs})
    print_dataset_structure(dataset, verbosity=2)
    captured = capsys.readouterr()
    assert "p1: <unnamed process>" in captured.out


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
        time_axis=TimeAxis(
            unit="hours", start=0.0, end=10.0, time_reference="inoculation"
        ),
        volume=Volume(initial_volume=1.0, unit="L"),
        reactor_medium=ReactorMedium(name="m", density=1.0, density_unit="kg/L"),
    )
    fig = plot_process(process)
    assert fig is not None
    plt.close(fig)


def test_plot_process_no_metadata_uses_fallback_title():
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    process = BioProcess(
        metadata=None,
        time_axis=TimeAxis(
            unit="hours", start=0.0, end=10.0, time_reference="inoculation"
        ),
        volume=Volume(initial_volume=1.0, unit="L"),
        reactor_medium=ReactorMedium(name="m", density=1.0, density_unit="kg/L"),
        process_variables={
            "temperature": ProcessVariable(
                name="temperature",
                unit="C",
                is_controlled=True,
                values=TimeSeries(
                    times=jnp.array([0.0, 1.0]), values=jnp.array([37.0, 37.0])
                ),
            )
        },
    )
    fig = plot_process(process)
    assert fig._suptitle is not None
    assert fig._suptitle.get_text() == "<unnamed process> (<unknown type>)"
    plt.close(fig)


def test_plot_process_contains_total_volume_panel(simple_process):
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig = plot_process(simple_process)
    axes = [ax for ax in fig.axes if ax.get_visible()]
    titles = [ax.get_title() for ax in axes]
    assert "total volume [L] (Volume)" in titles

    total_ax = next(ax for ax in axes if ax.get_title() == "total volume [L] (Volume)")
    line = total_ax.get_lines()[0]
    np.testing.assert_allclose(line.get_ydata(), np.array([1.0, 1.0]), atol=1e-12)
    plt.close(fig)


def test_plot_process_total_volume_integrates_continuous_and_discrete_changes():
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rm = ReactorMedium(name="medium", density=1.0, density_unit="kg/L")
    fm = FeedMedium(name="feed", density=1.0, density_unit="kg/L", components={})

    feed = FeedVolumeChange(
        name="feed",
        unit="L",
        is_controlled=True,
        is_continuous=True,
        feed_medium=fm,
        values=TimeSeries(
            times=jnp.array([0.0, 5.0, 10.0]),
            values=jnp.array([0.0, 0.2, 0.5]),
        ),
    )
    sample = SampleVolumeChange(
        name="sample",
        unit="L",
        is_controlled=False,
        is_continuous=False,
        values=TimeSeries(times=jnp.array([3.0, 7.0]), values=jnp.array([-0.1, -0.2])),
    )

    process = BioProcess(
        metadata=BioProcessMetadata(name="mix", process_type="fed_batch"),
        time_axis=TimeAxis(
            unit="hours", start=0.0, end=10.0, time_reference="inoculation"
        ),
        volume=Volume(
            initial_volume=1.0,
            unit="L",
            volume_changes={"feed": feed, "sample": sample},
        ),
        reactor_medium=rm,
    )

    fig = plot_process(process)
    axes = [ax for ax in fig.axes if ax.get_visible()]
    total_ax = next(ax for ax in axes if ax.get_title() == "total volume [L] (Volume)")
    line = total_ax.get_lines()[0]

    np.testing.assert_allclose(line.get_xdata(), np.array([0.0, 3.0, 5.0, 7.0, 10.0]))
    np.testing.assert_allclose(line.get_ydata(), np.array([1.0, 1.02, 1.1, 1.02, 1.2]))
    plt.close(fig)


def test_plot_process_total_volume_supports_spline_only_volume_change():
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rm = ReactorMedium(name="medium", density=1.0, density_unit="kg/L")
    fm = FeedMedium(name="feed", density=1.0, density_unit="kg/L", components={})

    spline_cumulative = TimeSeries(
        breaks=jnp.array([0.0, 10.0]),
        coeffs=jnp.array([[0.0, 0.05, 0.0, 0.0]]),
        segment_start_piece_idx=jnp.array([0]),
    )
    feed = FeedVolumeChange(
        name="feed",
        unit="L",
        is_controlled=True,
        is_continuous=True,
        feed_medium=fm,
        values=spline_cumulative,
    )

    process = BioProcess(
        metadata=BioProcessMetadata(name="spline", process_type="fed_batch"),
        time_axis=TimeAxis(
            unit="hours", start=0.0, end=10.0, time_reference="inoculation"
        ),
        volume=Volume(initial_volume=1.0, unit="L", volume_changes={"feed": feed}),
        reactor_medium=rm,
    )

    fig = plot_process(process)
    axes = [ax for ax in fig.axes if ax.get_visible()]
    total_ax = next(ax for ax in axes if ax.get_title() == "total volume [L] (Volume)")
    line = total_ax.get_lines()[0]

    np.testing.assert_allclose(line.get_xdata(), np.array([0.0, 10.0]))
    np.testing.assert_allclose(line.get_ydata(), np.array([1.0, 1.5]))
    plt.close(fig)


def test_plot_process_draws_curve_for_spline_only_series():
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    spline_only = TimeSeries(
        breaks=jnp.array([0.0, 1.0, 2.0]),
        coeffs=jnp.array(
            [
                [0.0, 0.0, 1.0, 0.0],
                [1.0, 2.0, 1.0, 0.0],
            ]
        ),
        segment_start_piece_idx=jnp.array([0]),
    )
    biomass_ts = TimeSeries(
        times=jnp.array([0.0, 1.0, 2.0]),
        values=jnp.array([0.5, 1.0, 1.5]),
    )
    process = BioProcess(
        metadata=BioProcessMetadata(name="curve", process_type="batch"),
        time_axis=TimeAxis(
            unit="hours", start=0.0, end=2.0, time_reference="inoculation"
        ),
        volume=Volume(initial_volume=1.0, unit="L"),
        reactor_medium=ReactorMedium(
            name="medium",
            density=1.0,
            density_unit="kg/L",
            components={
                "biomass": ReactorMediumComponent(
                    name="biomass",
                    unit="g/L",
                    concentration=biomass_ts,
                ),
                "curve": ReactorMediumComponent(
                    name="curve",
                    unit="g/L",
                    concentration=spline_only,
                ),
            },
        ),
    )

    fig = plot_process(process)
    axes = [ax for ax in fig.axes if ax.get_visible()]
    curve_ax = next(
        ax for ax in axes if ax.get_title() == "curve [g/L] (ReactorMedium)"
    )
    lines = curve_ax.get_lines()

    assert len(lines) == 1
    assert len(lines[0].get_xdata()) == 500

    x_dense = lines[0].get_xdata()
    y_dense = lines[0].get_ydata()
    idx_mid = int(np.argmin(np.abs(x_dense - 0.5)))
    idx_right = int(np.argmin(np.abs(x_dense - 1.5)))
    assert y_dense[idx_mid] == pytest.approx(0.25, abs=5e-3)
    assert y_dense[idx_right] == pytest.approx(2.25, abs=5e-3)
    plt.close(fig)


def test_plot_process_draws_pseudobatch_bundle_backtransform_curve():
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    feed = FeedMedium(
        name="feed",
        density=1.0,
        density_unit="kg/L",
        components={
            "glucose": FeedMediumComponent(
                name="glucose",
                unit="g/L",
                concentration=StaticVariable(value=100.0),
                is_controlled=True,
            )
        },
    )
    process = BioProcess(
        metadata=BioProcessMetadata(name="pb_plot", process_type="fed_batch"),
        time_axis=TimeAxis(unit="h", start=0.0, end=6.0, time_reference="t0"),
        volume=Volume(
            initial_volume=1.0,
            unit="L",
            volume_changes={
                "bolus": FeedVolumeChange(
                    name="bolus",
                    unit="L",
                    is_controlled=True,
                    is_continuous=False,
                    feed_medium=feed,
                    values=TimeSeries(
                        times=jnp.array([3.0]),
                        values=jnp.array([0.2]),
                    ),
                )
            },
        ),
        reactor_medium=ReactorMedium(
            name="medium",
            density=1.0,
            density_unit="kg/L",
            components={
                "biomass": ReactorMediumComponent(
                    name="biomass",
                    unit="g/L",
                    concentration=TimeSeries(
                        times=jnp.array([0.0, 2.0, 4.0, 6.0]),
                        values=jnp.array([0.5, 1.0, 1.5, 2.0]),
                    ),
                ),
                "glucose": ReactorMediumComponent(
                    name="glucose",
                    unit="g/L",
                    concentration=TimeSeries(
                        times=jnp.array([0.0, 2.0, 4.0, 6.0]),
                        values=jnp.array([10.0, 9.0, 8.0, 7.0]),
                    ),
                ),
            },
        ),
    )
    transform = build_pseudobatch_transform(process, ["glucose"])
    process.pseudobatch_transform = transform
    process.reactor_medium.components["glucose"].concentration = transform.species[
        "glucose"
    ].c_star_ts

    fig = plot_process(process)
    axes = [ax for ax in fig.axes if ax.get_visible()]
    glucose_ax = next(
        ax for ax in axes if ax.get_title() == "glucose [g/L] (ReactorMedium)"
    )
    assert len(glucose_ax.get_lines()) >= 1
    assert len(glucose_ax.get_lines()[0].get_xdata()) == 500
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
