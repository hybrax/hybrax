from __future__ import annotations

from copy import deepcopy
from dataclasses import replace
import json
from pathlib import Path
import warnings

import numpy as np
import pytest
from hybrax.format import validate_measurement_sampling_alignment
from hybrax.format.dataclasses import (
    AugmentedBioProcess,
    BioProcess,
    BioProcessCollection,
    BioProcessMetadata,
    FeedMedium,
    Inflow,
    ProcessVariable,
    ReactorMedium,
    ReactorMediumComponent,
    Outflow,
    TimeAxis,
    TimeSeries,
    Volume,
)
from hybrax.format.splines import fit_timeseries_spline
from hybrax.format.serialization import load_process_collection, save_process_collection
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from pydantic import ValidationError

import hybrax.train.augmentation as augmentation_module
import hybrax.train.augmentation_plot as augmentation_plot_module
import hybrax.train.prepare as prepare_module
from hybrax.train.augmentation import augment_process_collection
from hybrax.train.loo import _build_fold_groups
from hybrax.train.prepare import prepare_artifact
from hybrax.train.run_config import (
    AugmentationConfig,
    PrepareConfig,
    RunConfig,
    load_prepare_config,
)
from hybrax.train.training_data import TrainingDataStore


def _spline(values: list[float], *, smoothing_s: float = 0.08) -> TimeSeries:
    return fit_timeseries_spline(
        TimeSeries(
            times=np.linspace(0.0, 4.0, len(values)),
            values=np.asarray(values),
        ),
        smoothing_s=smoothing_s,
    )


def _dipping_spline() -> TimeSeries:
    return fit_timeseries_spline(
        TimeSeries(
            times=np.arange(5.0),
            values=np.asarray([0.1, 0.11, 1.0, 0.11, 0.1]),
        ),
        smoothing_s=0.0,
    )


def _late_spline(values: list[float]) -> TimeSeries:
    return fit_timeseries_spline(
        TimeSeries(
            times=np.linspace(1.0, 4.0, len(values)),
            values=np.asarray(values),
        ),
        smoothing_s=0.08,
    )


def _collection(*, zero_trace: bool = False) -> BioProcessCollection:
    biomass_values = [0.0] * 7 if zero_trace else [0.4, 0.7, 0.8, 1.4, 1.3, 2.0, 2.2]
    process = BioProcess(
        metadata=BioProcessMetadata(name="p1", process_type="batch"),
        time_axis=TimeAxis(unit="h", start=0.0, end=4.0, time_reference="start"),
        volume=Volume(initial_volume=1.0, unit="L", volume_changes={}),
        reactor_medium=ReactorMedium(
            name="rm",
            density=1.0,
            density_unit="kg/L",
            components={
                "biomass": ReactorMediumComponent(
                    name="biomass",
                    unit="g/L",
                    concentration=_spline(biomass_values),
                ),
                "product": ReactorMediumComponent(
                    name="product",
                    unit="g/L",
                    concentration=TimeSeries(
                        times=np.asarray([0.0, 2.0, 4.0]),
                        values=np.asarray([0.1, 0.3, 0.8]),
                    ),
                ),
            },
        ),
        process_variables={
            "ratio": ProcessVariable(
                name="ratio",
                unit="-",
                is_controlled=False,
                values=_spline([0.8, 0.9, 0.85, 1.1, 1.0, 1.2, 1.3]),
            ),
            "temperature": ProcessVariable(
                name="temperature",
                unit="degC",
                is_controlled=True,
                values=_spline([30.0, 30.2, 30.1, 30.3, 30.2, 30.4, 30.5]),
            ),
        },
    )
    return BioProcessCollection(processes={"p1": process}, metadata={})


def _config(
    *,
    noise_std: dict[str, float] | None = None,
    n_children: int = 1,
    n_time_points: int = 6,
    min_spacing_fraction: float = 0.1,
    initial_value_source: str | dict[str, str] = "measured",
) -> RunConfig:
    augmentation = AugmentationConfig(
        seed=12,
        n_children_per_process=n_children,
        n_time_points=n_time_points,
        min_spacing_fraction=min_spacing_fraction,
        noise_std=noise_std if noise_std is not None else {"biomass": 0.7},
        initial_value_source=initial_value_source,
    )
    return RunConfig(
        prepare=PrepareConfig(raw_input=Path("unused.json"), augmentation=augmentation)
    )


def _state_series(process, name: str) -> TimeSeries:
    if name in process.reactor_medium.components:
        return process.reactor_medium.components[name].concentration
    return process.process_variables[name].values


def _prepare_collection(
    tmp_path: Path,
    output_name: str,
    *,
    collection: BioProcessCollection | None = None,
    augmentation: dict | None = None,
    process_rename_map: dict[str, str] | None = None,
    custom_py: Path | None = None,
) -> BioProcessCollection:
    raw_path = tmp_path / f"{output_name}-raw.json"
    config_path = tmp_path / f"{output_name}-config.json"
    save_process_collection(collection or _collection(), raw_path)
    prepare = {
        "raw_input": str(raw_path),
        "diagnostics": False,
    }
    if augmentation is not None:
        prepare["augmentation"] = augmentation
    if process_rename_map is not None:
        prepare["process_rename_map"] = process_rename_map
    raw_config = {"prepare": prepare}
    if custom_py is not None:
        raw_config["custom_py"] = str(custom_py)
    config_path.write_text(json.dumps(raw_config), encoding="utf-8")
    return prepare_artifact(
        load_prepare_config(config_path),
        tmp_path / output_name,
    )


def _write_custom_module(tmp_path: Path, name: str, *lines: str) -> Path:
    path = tmp_path / f"custom-{name}.py"
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def _augmentation_dict(**updates) -> dict:
    config = {
        "seed": 12,
        "n_children_per_process": 2,
        "n_time_points": 6,
        "noise_std": {"biomass": 0.7},
    }
    config.update(updates)
    return config


def test_no_config_leaves_collection_unchanged():
    collection = _collection()
    config = RunConfig(prepare=PrepareConfig(raw_input=Path("unused.json")))

    assert augment_process_collection(collection, config) is collection
    assert list(collection.processes) == ["p1"]


def test_prepare_augmentation_does_not_claim_parent_rename(tmp_path):
    prepared = _prepare_collection(
        tmp_path,
        "prepared-rename-augmented",
        augmentation=_augmentation_dict(n_children_per_process=1),
        process_rename_map={"p1": "renamed_p1"},
    )

    provenance = prepared.metadata["hybrax.train"]["semantics_provenance"]["processes"]
    assert list(prepared.processes) == ["renamed_p1", "renamed_p1__aug_000"]
    assert provenance["renamed_p1"]["changed_by_hooks"] == [
        "transform_process_collection"
    ]
    assert provenance["renamed_p1__aug_000"]["changed_by_hooks"] == ["augmentation"]


def test_prepare_with_augmentation_writes_plot(tmp_path, monkeypatch):
    rendered_state_names = []
    rendered_process_names = []
    requested_state_names = []
    rendered_ylabels = []
    rendered_band_bounds = []
    rendered_initial_band_bounds = []
    render_augmentation_plot = prepare_module.render_augmentation_plot
    state_series = augmentation_plot_module._state_series
    fill_between = Axes.fill_between
    save_figure = Figure.savefig

    def track_render(collection, augmentation, output_path):
        rendered_state_names.append(tuple(augmentation.noise_std))
        rendered_process_names.append(tuple(collection.processes))
        render_augmentation_plot(collection, augmentation, output_path)

    def track_state_series(process, state_name):
        requested_state_names.append(state_name)
        return state_series(process, state_name)

    def track_save_figure(figure, *args, **kwargs):
        rendered_ylabels.append(tuple(axis.get_ylabel() for axis in figure.axes))
        return save_figure(figure, *args, **kwargs)

    def track_fill_between(axis, x, y1, y2, *args, **kwargs):
        rendered_band_bounds.append((np.asarray(y1), np.asarray(y2)))
        rendered_initial_band_bounds.append((y1[0], y2[0]))
        return fill_between(axis, x, y1, y2, *args, **kwargs)

    monkeypatch.setattr(prepare_module, "render_augmentation_plot", track_render)
    monkeypatch.setattr(augmentation_plot_module, "_state_series", track_state_series)
    monkeypatch.setattr(Axes, "fill_between", track_fill_between)
    monkeypatch.setattr(Figure, "savefig", track_save_figure)
    _prepare_collection(
        tmp_path,
        "with-augmentation-plot",
        augmentation=_augmentation_dict(
            n_children_per_process=1,
            noise_std={"biomass": 0.7, "ratio": 0.4},
        ),
    )

    output_dir = tmp_path / "with-augmentation-plot"
    plot_path = output_dir / "augmented-data.png"
    assert plot_path.is_file()
    assert plot_path.stat().st_size > 0
    assert (output_dir / "prepared.json").is_file()
    assert (output_dir / "prepare_config.json").is_file()
    assert rendered_state_names == [("biomass", "ratio")]
    assert rendered_process_names == [("p1", "p1__aug_000")]
    assert set(requested_state_names) == {"biomass", "ratio"}
    assert rendered_ylabels == [("p1\n[g/L]", "[-]")]
    np.testing.assert_allclose(
        rendered_initial_band_bounds,
        [(0.4, 0.4), (0.8, 0.8)],
    )
    biomass_lower, biomass_upper = rendered_band_bounds[0]
    ratio_lower, ratio_upper = rendered_band_bounds[1]
    assert np.all(biomass_lower >= 0.0)
    assert np.all(biomass_upper >= 0.0)
    np.testing.assert_allclose(ratio_upper[1:] - ratio_lower[1:], 1.568)


def test_augmentation_plot_shares_y_axis_by_state_column(tmp_path, monkeypatch):
    collection = _collection()
    second = deepcopy(collection.processes["p1"])
    second.metadata.name = "p2"
    collection.processes["p2"] = second
    augmentation = _config(
        noise_std={"biomass": 0.7, "ratio": 0.4}
    ).prepare.augmentation
    shared = {}

    def inspect_axes(figure, *_args, **_kwargs):
        axes = figure.axes
        siblings = axes[0].get_shared_y_axes()
        shared["column"] = siblings.joined(axes[0], axes[2])
        shared["row"] = siblings.joined(axes[0], axes[1])

    monkeypatch.setattr(Figure, "savefig", inspect_axes)
    augmentation_plot_module.render_augmentation_plot(
        collection,
        augmentation,
        tmp_path / "unused.png",
    )

    assert shared == {"column": True, "row": False}


def test_prepare_plot_failure_does_not_write_json(tmp_path, monkeypatch):
    output_dir = tmp_path / "failed-augmentation-plot"

    def fail_plot(*_):
        raise RuntimeError("failed to render augmentation plot")

    monkeypatch.setattr(prepare_module, "render_augmentation_plot", fail_plot)

    with pytest.raises(RuntimeError, match="failed to render augmentation plot"):
        _prepare_collection(
            tmp_path,
            output_dir.name,
            augmentation=_augmentation_dict(),
        )

    assert not (output_dir / "prepared.json").exists()
    assert not (output_dir / "prepare_config.json").exists()


def test_prepare_without_augmentation_removes_stale_plot(tmp_path):
    output_name = "removed-augmentation-plot"
    plot_path = tmp_path / output_name / "augmented-data.png"
    plot_path.parent.mkdir()
    plot_path.write_bytes(b"stale plot")

    _prepare_collection(tmp_path, output_name)

    assert not plot_path.exists()


@pytest.mark.parametrize(
    "config",
    [
        {},
        {"n_children_per_process": 1},
        {
            "n_children_per_process": 0,
            "n_time_points": 2,
            "noise_std": {"biomass": 0.1},
        },
        {
            "n_children_per_process": 1,
            "n_time_points": 1,
            "noise_std": {"biomass": 0.1},
        },
        {
            "n_children_per_process": 1,
            "n_time_points": 2,
            "noise_std": {},
        },
        {
            "n_children_per_process": 1,
            "n_time_points": 2,
            "noise_std": {"biomass": 0.1},
            "noise_model": "add",
        },
        {
            "n_children_per_process": 1,
            "n_time_points": 2,
            "min_spacing_fraction": 0.0,
            "noise_std": {"biomass": 0.1},
        },
        {
            "n_children_per_process": 1,
            "n_time_points": 2,
            "min_spacing_fraction": True,
            "noise_std": {"biomass": 0.1},
        },
        {
            "n_children_per_process": 1,
            "n_time_points": 2,
            "min_spacing_fraction": 1.1,
            "noise_std": {"biomass": 0.1},
        },
        {
            "n_children_per_process": 1,
            "n_time_points": 2,
            "noise_std": {"biomass": -0.1},
        },
        {
            "n_children_per_process": 1,
            "n_time_points": 2,
            "noise_std": {"biomass": float("nan")},
        },
        {
            "n_children_per_process": 1,
            "n_time_points": 2,
            "noise_std": {"biomass": 0.1},
            "initial_value_source": "other",
        },
        {
            "n_children_per_process": 1,
            "n_time_points": 2,
            "noise_std": {"biomass": 0.1, "ratio": 0.2},
            "initial_value_source": {"biomass": "measured"},
        },
    ],
)
def test_invalid_config_fails_fast(config):
    with pytest.raises(ValidationError):
        AugmentationConfig.model_validate(config)


def test_augmentation_defaults_allow_zero_noise():
    config = AugmentationConfig.model_validate(
        {
            "n_children_per_process": 1,
            "n_time_points": 2,
            "noise_std": {"biomass": 0.0},
        }
    )

    assert config.initial_value_source == "measured"
    assert config.min_spacing_fraction == 0.1
    assert config.noise_std == {"biomass": 0.0}


def test_degenerate_parent_time_range_fails_fast():
    collection = _collection()
    collection.processes["p1"].time_axis.end = 0.0

    with pytest.raises(ValueError, match="p1: cannot augment a degenerate time range"):
        augment_process_collection(collection, _config())


@pytest.mark.parametrize(
    ("source", "expected"),
    [
        ("measured", "measured"),
        ("spline", "spline"),
        ("augmented", "augmented"),
    ],
)
def test_initial_value_source_controls_augmented_t0(source, expected):
    collection = _collection()
    parent_series = _state_series(collection.processes["p1"], "biomass")
    measured = float(parent_series.values[0])
    spline = float(parent_series.evaluate(0.0))

    child = augment_process_collection(
        collection,
        _config(initial_value_source=source),
    ).processes["p1__aug_000"]

    actual = _state_series(child, "biomass").values[0]
    if expected == "augmented":
        assert actual != pytest.approx(spline)
    else:
        expected_value = {"measured": measured, "spline": spline}[expected]
        assert actual == pytest.approx(expected_value)


def test_initial_value_source_mapping_controls_each_listed_state():
    collection = _collection()
    parent = collection.processes["p1"]
    biomass = _state_series(parent, "biomass")
    ratio = _state_series(parent, "ratio")

    child = augment_process_collection(
        collection,
        _config(
            noise_std={"biomass": 10.0, "ratio": 10.0},
            initial_value_source={"biomass": "measured", "ratio": "spline"},
        ),
    ).processes["p1__aug_000"]

    assert _state_series(child, "biomass").values[0] == pytest.approx(biomass.values[0])
    assert _state_series(child, "ratio").values[0] == pytest.approx(ratio.evaluate(0.0))


def test_measured_initial_value_requires_observation_at_process_start():
    collection = _collection()
    collection.processes["p1"].process_variables["ratio"].values = _late_spline(
        [0.8, 0.9, 0.85, 1.1, 1.0, 1.2, 1.3]
    )

    with pytest.raises(
        ValueError,
        match="initial_value_source='measured'.*ratio.*observation at process start",
    ):
        augment_process_collection(
            collection,
            _config(
                noise_std={"ratio": 0.7},
                initial_value_source="measured",
            ),
        )


def test_measured_initial_value_failure_does_not_add_children():
    collection = _collection()
    second = deepcopy(collection.processes["p1"])
    second.metadata.name = "p2"
    second.process_variables["ratio"].values = _late_spline(
        [0.8, 0.9, 0.85, 1.1, 1.0, 1.2, 1.3]
    )
    collection.processes["p2"] = second

    with pytest.raises(ValueError, match="p2.*observation at process start"):
        augment_process_collection(
            collection,
            _config(
                noise_std={"ratio": 0.7},
                initial_value_source="measured",
            ),
        )

    assert list(collection.processes) == ["p1", "p2"]


def test_measured_initial_value_match_has_no_relative_time_tolerance():
    collection = _collection()
    process = collection.processes["p1"]
    t0 = 1_000_000.0
    process.time_axis.start = t0
    process.time_axis.end = t0 + 4.0
    process.reactor_medium.components["biomass"].concentration = fit_timeseries_spline(
        TimeSeries(
            times=np.linspace(t0 + 1.0, t0 + 4.0, 7),
            values=np.asarray([0.4, 0.7, 0.8, 1.4, 1.3, 2.0, 2.2]),
        ),
        smoothing_s=0.08,
    )

    with pytest.raises(ValueError, match="requires an observation at process start"):
        augment_process_collection(collection, _config())


@pytest.mark.parametrize("source", ["spline", "augmented"])
def test_late_listed_trace_warns_when_initial_value_uses_spline(source):
    collection = _collection()
    collection.processes["p1"].process_variables["ratio"].values = _late_spline(
        [0.8, 0.9, 0.85, 1.1, 1.0, 1.2, 1.3]
    )

    with pytest.warns(
        UserWarning,
        match="spline for 'ratio' is extrapolated before its first observation",
    ) as caught:
        augment_process_collection(
            collection,
            _config(
                noise_std={"ratio": 0.7},
                initial_value_source=source,
                n_children=3,
            ),
        )

    assert (
        sum("spline for 'ratio' is extrapolated" in str(w.message) for w in caught) == 1
    )


def test_late_unlisted_spline_trace_warns_about_implicit_extrapolation():
    collection = _collection()
    collection.processes["p1"].process_variables["ratio"].values = _late_spline(
        [0.8, 0.9, 0.85, 1.1, 1.0, 1.2, 1.3]
    )

    with pytest.warns(
        UserWarning,
        match="spline for 'ratio' is extrapolated before its first observation",
    ) as caught:
        augment_process_collection(collection, _config(n_children=3))

    assert (
        sum("spline for 'ratio' is extrapolated" in str(w.message) for w in caught) == 1
    )


def test_augmentation_accepts_missing_process_metadata():
    collection = _collection()
    collection.processes["p1"].metadata = None

    augmented = augment_process_collection(collection, _config())

    assert augmented.processes["p1__aug_000"].metadata is None


def test_children_have_independent_deterministic_common_endpoint_grids():
    first = augment_process_collection(
        _collection(), _config(n_children=2, n_time_points=7)
    )
    second = augment_process_collection(
        _collection(), _config(n_children=2, n_time_points=7)
    )

    grids = []
    for child_index in range(2):
        name = f"p1__aug_{child_index:03d}"
        child = first.processes[name]
        other = second.processes[name]
        biomass = _state_series(child, "biomass")
        ratio = _state_series(child, "ratio")
        grids.append(np.asarray(biomass.times))

        assert isinstance(child, AugmentedBioProcess)
        assert child.parent_process == "p1"
        assert child.metadata.name == name
        assert len(biomass.times) == 7
        assert biomass.times[0] == 0.0
        assert biomass.times[-1] == 4.0
        assert np.all(np.diff(biomass.times) >= 0.1 * 4.0 / 6.0)
        np.testing.assert_array_equal(biomass.times, ratio.times)
        np.testing.assert_array_equal(
            biomass.times, _state_series(other, "biomass").times
        )
        np.testing.assert_array_equal(
            biomass.values, _state_series(other, "biomass").values
        )

    assert not np.array_equal(grids[0], grids[1])


@pytest.mark.parametrize(
    ("n_time_points", "t_end"),
    [(2, 4.0), (7, 0.007), (100, 4.0)],
)
def test_full_min_spacing_fraction_makes_child_grid_evenly_spaced(n_time_points, t_end):
    collection = _collection()
    collection.processes["p1"].time_axis.end = t_end
    child = augment_process_collection(
        collection,
        _config(n_time_points=n_time_points, min_spacing_fraction=1.0),
    ).processes["p1__aug_000"]

    np.testing.assert_array_equal(
        _state_series(child, "biomass").times,
        np.linspace(0.0, t_end, n_time_points),
    )


def test_child_grid_retries_sampling_event_near_misses():
    augmentation = _config(n_time_points=20).prepare.augmentation
    assert augmentation is not None
    first = augmentation_module._child_grid(augmentation, "p1", 0, 0.0, 4.0)
    sampling_time = first[10] - 2e-4

    retried = augmentation_module._child_grid(
        augmentation,
        "p1",
        0,
        0.0,
        4.0,
        (sampling_time,),
    )

    assert not np.array_equal(retried, first)
    np.testing.assert_array_equal(
        retried,
        augmentation_module._child_grid(
            augmentation,
            "p1",
            0,
            0.0,
            4.0,
            (sampling_time,),
        ),
    )
    assert np.all(np.diff(retried) >= 0.1 * 4.0 / 19.0)
    deltas = retried - sampling_time
    assert not np.any((deltas > 0.0) & (deltas <= 4e-4))

    for allowed_sampling_time in (first[10], first[10] + 2e-4):
        np.testing.assert_array_equal(
            augmentation_module._child_grid(
                augmentation,
                "p1",
                0,
                0.0,
                4.0,
                (allowed_sampling_time,),
            ),
            first,
        )


def test_child_grid_sampling_event_retries_are_bounded(monkeypatch):
    augmentation = _config(
        n_time_points=20, min_spacing_fraction=1.0
    ).prepare.augmentation
    assert augmentation is not None
    first = augmentation_module._child_grid(augmentation, "p1", 0, 0.0, 4.0)
    monkeypatch.setattr(augmentation_module, "_MAX_GRID_ATTEMPTS", 2)

    with pytest.raises(ValueError, match="away from sampling-event near-misses"):
        augmentation_module._child_grid(
            augmentation,
            "p1",
            0,
            0.0,
            4.0,
            (first[10] - 2e-4,),
        )


def test_augmentation_passes_sampling_times_to_child_grid_retry():
    collection = _collection()
    augmentation = _config(n_time_points=20).prepare.augmentation
    assert augmentation is not None
    first = augmentation_module._child_grid(augmentation, "p1", 0, 0.0, 4.0)
    sampling_time = first[10] - 2e-4
    collection.processes["p1"].volume.volume_changes["sample"] = Outflow(
        name="sample",
        unit="L",
        is_controlled=False,
        is_continuous=False,
        values=TimeSeries(times=[sampling_time], values=[-0.01]),
    )

    child = augment_process_collection(collection, _config(n_time_points=20)).processes[
        "p1__aug_000"
    ]
    deltas = np.asarray(_state_series(child, "biomass").times) - sampling_time

    assert not np.any((deltas > 0.0) & (deltas <= 4e-4))
    valid, message = validate_measurement_sampling_alignment(child)
    assert valid, message


def test_even_grid_allows_relative_rounding_at_nonzero_origin():
    augmentation = _config(
        n_time_points=20, min_spacing_fraction=1.0
    ).prepare.augmentation
    assert augmentation is not None

    grid = augmentation_module._child_grid(
        augmentation,
        parent_name="p1",
        child_index=0,
        t0=24.0,
        t_end=192.0,
    )

    np.testing.assert_array_equal(grid, np.linspace(24.0, 192.0, 20))


@pytest.mark.parametrize(
    ("n_time_points", "min_spacing_fraction", "t0", "t_end"),
    [
        # Points collapse onto identical timestamps (non-strict grid).
        (20, 0.1, 1e16, 1e16 + 16.0),
        (3, 0.1, 0.0, np.nextafter(0.0, 1.0)),
        # Points stay strictly increasing (11 distinct floats) yet the smallest
        # gap (2.0) falls below the requested minimum spacing (30/10 = 3.0): a
        # large origin cannot represent the even grid, so this must still fail.
        (11, 1.0, 1e16, 1e16 + 30.0),
        # Gaps are ~= the requested minimum (2.000001), but that minimum sits at
        # the timestamp resolution floor (~1 ULP at this origin), so the request
        # is not representable and must fail rather than emit ~1-ULP spacing.
        (11, 0.666667, 1e16, 1e16 + 30.0),
    ],
)
def test_unrepresentable_minimum_child_grid_spacing_fails_fast(
    n_time_points, min_spacing_fraction, t0, t_end
):
    augmentation = _config(
        n_time_points=n_time_points, min_spacing_fraction=min_spacing_fraction
    ).prepare.augmentation
    assert augmentation is not None

    with pytest.raises(
        ValueError,
        match="p1__aug_000: cannot represent the requested minimum child-grid spacing",
    ):
        augmentation_module._child_grid(
            augmentation,
            parent_name="p1",
            child_index=0,
            t0=t0,
            t_end=t_end,
        )


def test_child_grid_failure_leaves_collection_unchanged(monkeypatch):
    collection = _collection()

    def build_grid(augmentation, parent_name, child_index, t0, t_end, sampling_times):
        del parent_name, sampling_times
        if child_index == 2:
            raise ValueError(
                "cannot represent the requested minimum child-grid spacing"
            )
        return np.linspace(t0, t_end, augmentation.n_time_points)

    monkeypatch.setattr(augmentation_module, "_child_grid", build_grid)

    with pytest.raises(ValueError, match="cannot represent the requested"):
        augment_process_collection(collection, _config(n_children=5))

    assert list(collection.processes) == ["p1"]


def test_late_child_failure_leaves_collection_unchanged(monkeypatch):
    collection = _collection()
    set_state_series = augmentation_module._set_state_series

    def fail_second_child(process, state_name, series):
        if process.metadata.name == "p1__aug_001":
            raise ValueError("failed to augment child")
        set_state_series(process, state_name, series)

    monkeypatch.setattr(augmentation_module, "_set_state_series", fail_second_child)

    with pytest.raises(ValueError, match="failed to augment child"):
        augment_process_collection(collection, _config(n_children=2))

    assert list(collection.processes) == ["p1"]


def test_children_preserve_physical_structure_without_sharing_objects():
    collection = _collection()
    parent = collection.processes["p1"]
    parent.volume.volume_changes["sample"] = Outflow(
        name="sample",
        unit="L",
        is_controlled=False,
        is_continuous=False,
        values=TimeSeries(times=[2.0], values=[-0.1]),
    )
    parent.volume.volume_changes["harvest"] = Outflow(
        name="harvest",
        unit="L",
        is_controlled=True,
        is_continuous=True,
        values=TimeSeries(times=[0.0, 4.0], values=[0.0, -0.1]),
        retention={"biomass": 0.25},
    )
    parent.volume.volume_changes["feed"] = Inflow(
        name="feed",
        unit="L",
        is_controlled=True,
        is_continuous=True,
        values=TimeSeries(times=[0.0, 4.0], values=[0.0, 0.2]),
        feed_medium=FeedMedium(name="feed", density=1.0, density_unit="kg/L"),
    )
    parent_biomass = np.asarray(_state_series(parent, "biomass").values).copy()

    augmented = augment_process_collection(collection, _config(n_children=2))
    first = augmented.processes["p1__aug_000"]
    second = augmented.processes["p1__aug_001"]

    for child in (first, second):
        assert child.volume is not parent.volume
        assert child.volume.initial_volume == parent.volume.initial_volume
        np.testing.assert_array_equal(
            child.volume.volume_changes["sample"].values.values,
            parent.volume.volume_changes["sample"].values.values,
        )
        assert child.volume.volume_changes["harvest"].retention == {"biomass": 0.25}
        assert child.volume.volume_changes["feed"].feed_medium.name == "feed"
        assert (
            child.process_variables["temperature"]
            is not (parent.process_variables["temperature"])
        )
        np.testing.assert_array_equal(
            child.process_variables["temperature"].values.values,
            parent.process_variables["temperature"].values.values,
        )

    assert _state_series(first, "biomass") is not _state_series(second, "biomass")
    np.testing.assert_array_equal(
        _state_series(parent, "biomass").values,
        parent_biomass,
    )


def test_listed_noise_includes_t0_and_unlisted_states_follow_three_way_rule():
    collection = _collection()
    parent = collection.processes["p1"]
    parent_product = deepcopy(_state_series(parent, "product"))
    child = augment_process_collection(
        collection,
        _config(initial_value_source="augmented"),
    ).processes["p1__aug_000"]
    child_grid = np.asarray(_state_series(child, "biomass").times)

    biomass_base = np.asarray(
        _state_series(parent, "biomass").evaluate_many(child_grid)
    )
    child_biomass = np.asarray(_state_series(child, "biomass").values)
    assert child_biomass[0] != pytest.approx(biomass_base[0])

    ratio = _state_series(child, "ratio")
    np.testing.assert_array_equal(ratio.times, child_grid)
    np.testing.assert_allclose(
        ratio.values,
        _state_series(parent, "ratio").evaluate_many(child_grid),
    )
    np.testing.assert_array_equal(
        _state_series(child, "product").times, parent_product.times
    )
    np.testing.assert_array_equal(
        _state_series(child, "product").values,
        parent_product.values,
    )


def test_mixed_grid_child_round_trips_and_is_accepted_by_training_data_store(tmp_path):
    collection = augment_process_collection(_collection(), _config(n_time_points=6))
    prepared_json = tmp_path / "prepared.json"
    save_process_collection(collection, prepared_json)

    loaded = load_process_collection(prepared_json)
    child = loaded.processes["p1__aug_000"]
    assert isinstance(child, AugmentedBioProcess)
    assert child.parent_process == "p1"

    store = TrainingDataStore.from_json(
        prepared_json,
        target_source="combined",
    )

    child_index = store.process_order.index("p1__aug_000")
    assert int(store.n_measured[child_index]) > 6
    assert np.all(np.asarray(store.mask_measured[child_index, 0, :3]))


def test_resampled_modeled_states_drop_splines_but_controls_keep_them():
    collection = augment_process_collection(_collection(), _config())

    child = collection.processes["p1__aug_000"]
    assert len(_state_series(child, "biomass").times) == 6
    assert len(_state_series(child, "ratio").times) == 6
    assert _state_series(child, "biomass").poly is None
    assert _state_series(child, "ratio").poly is None
    assert _state_series(child, "temperature").poly is not None
    assert _state_series(collection.processes["p1"], "biomass").poly is not None


@pytest.mark.parametrize(
    ("state_name", "message"),
    [
        ("temperature", "controlled process variable"),
        ("unknown", "is not a modeled state"),
        ("product", "requires a spline"),
    ],
)
def test_invalid_requested_state_fails_fast(state_name, message):
    with pytest.raises(ValueError, match=message):
        augment_process_collection(
            _collection(),
            _config(noise_std={state_name: 0.7}),
        )


def test_absolute_noise_std_matches_formula():
    collection = _collection()
    parent_series = _state_series(collection.processes["p1"], "ratio")
    noise_std = 0.4
    child = augment_process_collection(
        collection,
        _config(
            noise_std={"ratio": noise_std},
            initial_value_source="augmented",
        ),
    ).processes["p1__aug_000"]
    child_series = _state_series(child, "ratio")
    base_values = np.asarray(parent_series.evaluate_many(child_series.times))
    standard_normal = augmentation_module._rng(
        12,
        "p1",
        0,
        "ratio",
        "values",
    ).standard_normal(np.asarray(child_series.times).shape)

    np.testing.assert_allclose(
        child_series.values,
        base_values + noise_std * standard_normal,
    )


def test_zero_noise_std_only_resamples_time():
    collection = _collection()
    parent_series = _state_series(collection.processes["p1"], "biomass")
    child = augment_process_collection(
        collection,
        _config(
            noise_std={"biomass": 0.0},
            initial_value_source="spline",
        ),
    ).processes["p1__aug_000"]
    child_series = _state_series(child, "biomass")

    np.testing.assert_allclose(
        child_series.values,
        parent_series.evaluate_many(child_series.times),
    )


def test_identically_zero_parent_trace_receives_noise():
    child = augment_process_collection(
        _collection(zero_trace=True),
        _config(
            noise_std={"biomass": 10.0},
            initial_value_source="augmented",
        ),
    ).processes["p1__aug_000"]

    assert np.any(np.asarray(_state_series(child, "biomass").values) > 0.0)


def test_spline_only_identically_zero_parent_trace_receives_noise():
    collection = _collection(zero_trace=True)
    component = collection.processes["p1"].reactor_medium.components["biomass"]
    component.concentration = replace(
        component.concentration,
        times=None,
        values=None,
    )

    child = augment_process_collection(
        collection,
        _config(
            noise_std={"biomass": 10.0},
            initial_value_source="augmented",
        ),
    ).processes["p1__aug_000"]

    assert np.any(np.asarray(_state_series(child, "biomass").values) > 0.0)


def test_custom_hook_can_preserve_zero_trace_after_builtin_clipping(tmp_path):
    custom_py = _write_custom_module(
        tmp_path,
        "preserve-zero-trace",
        "import numpy as np",
        "",
        "def augment_state_values(*, base_values, augmented_values, **_):",
        "    assert np.all(augmented_values >= 0.0)",
        "    if np.all(base_values == 0.0):",
        "        return base_values",
        "    return augmented_values",
    )
    prepared = _prepare_collection(
        tmp_path,
        "preserved-zero-trace",
        collection=_collection(zero_trace=True),
        augmentation=_augmentation_dict(n_children_per_process=1),
        custom_py=custom_py,
    )

    child = prepared.processes["p1__aug_000"]
    np.testing.assert_array_equal(_state_series(child, "biomass").values, 0.0)
    assert (
        prepared.metadata["hybrax.train"]["transform_hooks"]["augment_state_values"]
        == "augment_state_values"
    )


def test_custom_hook_receives_unmodified_spline_base_values():
    collection = _collection()
    parent_series = _state_series(collection.processes["p1"], "biomass")
    captured = {}

    def capture_values(*, base_values, augmented_values, **_):
        captured["base"] = base_values
        captured["augmented"] = augmented_values
        return augmented_values

    child = augment_process_collection(
        collection,
        _config(noise_std={"biomass": 0.0}),
        capture_values,
    ).processes["p1__aug_000"]
    child_times = _state_series(child, "biomass").times

    np.testing.assert_array_equal(
        captured["base"],
        parent_series.evaluate_many(child_times),
    )
    assert captured["augmented"][0] == parent_series.values[0]
    assert captured["base"][0] != pytest.approx(captured["augmented"][0])


def test_custom_hook_shape_failure_does_not_add_children():
    collection = _collection()

    def wrong_shape(**_):
        return np.zeros(1)

    with pytest.raises(ValueError, match="returned shape.*expected"):
        augment_process_collection(collection, _config(), wrong_shape)

    assert list(collection.processes) == ["p1"]


@pytest.mark.parametrize("nonfinite", [np.nan, np.inf])
def test_custom_hook_nonfinite_failure_does_not_add_children(nonfinite):
    collection = _collection()

    def return_nonfinite(*, augmented_values, **_):
        return np.full_like(augmented_values, nonfinite)

    with pytest.raises(ValueError, match="returned non-finite values"):
        augment_process_collection(collection, _config(), return_nonfinite)

    assert list(collection.processes) == ["p1"]


def test_additive_noise_preserves_signed_process_variables():
    collection = _collection()
    collection.processes["p1"].process_variables["ratio"].values = _spline(
        [-1.5, -1.2, -0.9, -0.6, -0.3, -0.1, -0.05]
    )
    child = augment_process_collection(
        collection,
        _config(
            noise_std={"ratio": 0.2},
            initial_value_source="augmented",
        ),
    ).processes["p1__aug_000"]

    assert np.any(np.asarray(_state_series(child, "ratio").values) < 0.0)


def test_additive_noise_clips_at_zero():
    collection = _collection()
    child = augment_process_collection(
        collection,
        _config(noise_std={"biomass": 1_000.0}),
    ).processes["p1__aug_000"]

    assert np.any(np.asarray(_state_series(child, "biomass").values) == 0.0)


def test_reactor_medium_spline_warns_and_augmented_values_are_clipped():
    collection = _collection()
    collection.processes["p1"].reactor_medium.components[
        "biomass"
    ].concentration = _dipping_spline()

    with pytest.warns(
        UserWarning,
        match="spline for reactor-medium component 'biomass' evaluated below zero",
    ):
        child = augment_process_collection(
            collection,
            _config(noise_std={"ratio": 0.7}, n_time_points=2),
        ).processes["p1__aug_000"]

    values = np.asarray(_state_series(child, "biomass").values)
    assert np.all(values >= 0.0)


def test_mostly_nonnegative_process_variable_spline_dip_warns():
    collection = _collection()
    collection.processes["p1"].process_variables["ratio"].values = _dipping_spline()

    with pytest.warns(
        UserWarning,
        match="spline for process variable 'ratio' evaluated below zero",
    ):
        augment_process_collection(collection, _config())


def test_process_variable_with_half_nonnegative_observations_does_not_warn():
    collection = _collection()
    collection.processes["p1"].process_variables["ratio"].values = _spline(
        [-1.0, -1.0, 1.0, 1.0],
        smoothing_s=0.0,
    )

    with warnings.catch_warnings(record=True) as caught:
        augment_process_collection(collection, _config())

    assert not caught


def test_generated_child_name_collision_fails_fast():
    collection = _collection()
    collection.processes["p1__aug_000"] = deepcopy(collection.processes["p1"])

    with pytest.raises(ValueError, match="generated augmented process already exists"):
        augment_process_collection(collection, _config())


def test_prepared_children_round_trip_with_stable_values_and_content_hash(tmp_path):
    first = _prepare_collection(
        tmp_path,
        "prepared-first",
        augmentation=_augmentation_dict(),
    )
    second = _prepare_collection(
        tmp_path,
        "prepared-second",
        augmentation=_augmentation_dict(),
    )
    reloaded = load_process_collection(tmp_path / "prepared-first" / "prepared.json")

    child = reloaded.processes["p1__aug_000"]
    assert isinstance(child, AugmentedBioProcess)
    assert child.parent_process == "p1"
    np.testing.assert_array_equal(
        _state_series(first.processes["p1__aug_001"], "biomass").values,
        _state_series(second.processes["p1__aug_001"], "biomass").values,
    )
    assert (
        first.metadata["hybrax.train"]["provenance"]["content_hash"]
        == second.metadata["hybrax.train"]["provenance"]["content_hash"]
    )


def test_prepare_records_augmented_provenance(tmp_path):
    prepared = _prepare_collection(
        tmp_path,
        "prepared-provenance",
        augmentation=_augmentation_dict(n_children_per_process=1),
    )
    metadata = prepared.metadata["hybrax.train"]
    child_provenance = metadata["semantics_provenance"]["processes"]["p1__aug_000"]

    assert child_provenance["raw"] is None
    assert child_provenance["changed_by_hooks"] == ["augmentation"]
    assert set(metadata["transform_hooks"]) == {"transform_process_collection"}
    assert _build_fold_groups(prepared) == (("p1", ("p1", "p1__aug_000")),)


def test_prepare_handles_transform_added_process_provenance(tmp_path):
    custom_py = _write_custom_module(
        tmp_path,
        "add-process",
        "from copy import deepcopy",
        "",
        "def transform_process_collection(collection, config):",
        "    added = deepcopy(collection.processes['p1'])",
        "    added.metadata.name = 'added'",
        "    collection.processes = {",
        "        'added': added, **collection.processes",
        "    }",
        "    return collection",
    )
    prepared = _prepare_collection(
        tmp_path,
        "prepared-added-process",
        custom_py=custom_py,
    )
    provenance = prepared.metadata["hybrax.train"]["semantics_provenance"]["processes"][
        "added"
    ]

    assert provenance["raw"] is None
    assert provenance["changed_by_hooks"] == ["transform_process_collection"]


def test_transform_created_augmented_process_is_attributed_to_transform(tmp_path):
    custom_py = _write_custom_module(
        tmp_path,
        "add-augmented-process",
        "from copy import deepcopy",
        "from hybrax.format.dataclasses import AugmentedBioProcess",
        "",
        "def transform_process_collection(collection, config):",
        "    parent = collection.processes['p1']",
        "    copied = deepcopy(parent)",
        "    copied.metadata.name = 'transform_child'",
        "    collection.processes['transform_child'] = AugmentedBioProcess(",
        "        **vars(copied), parent_process='p1'",
        "    )",
        "    return collection",
    )
    prepared = _prepare_collection(
        tmp_path,
        "prepared-transform-augmented-process",
        custom_py=custom_py,
    )
    provenance = prepared.metadata["hybrax.train"]["semantics_provenance"]["processes"][
        "transform_child"
    ]

    assert provenance["raw"] is None
    assert provenance["changed_by_hooks"] == ["transform_process_collection"]


def test_prepare_rejects_ambiguous_copied_rename_tags(tmp_path):
    custom_py = _write_custom_module(
        tmp_path,
        "ambiguous-tags",
        "from copy import deepcopy",
        "",
        "def transform_process_collection(collection, config):",
        "    process = collection.processes.pop('p1')",
        "    first = deepcopy(process)",
        "    first.metadata.name = 'first'",
        "    process.metadata.name = 'second'",
        "    collection.processes = {'first': first, 'second': process}",
        "    return collection",
    )

    with pytest.raises(ValueError, match="ambiguous pre-transform provenance tag"):
        _prepare_collection(
            tmp_path,
            "prepared-ambiguous-tags",
            custom_py=custom_py,
        )


def test_prepare_rejects_invalid_augmented_parent_reference(tmp_path):
    custom_py = _write_custom_module(
        tmp_path,
        "invalid-parent",
        "from hybrax.format.dataclasses import AugmentedBioProcess",
        "",
        "def transform_process_collection(collection, config):",
        "    process = collection.processes.pop('p1')",
        "    process.metadata.name = 'child'",
        "    collection.processes['child'] = AugmentedBioProcess(",
        "        **vars(process), parent_process='missing'",
        "    )",
        "    return collection",
    )

    with pytest.raises(ValueError, match="augmented parent validation failed"):
        _prepare_collection(
            tmp_path,
            "prepared-invalid-parent",
            custom_py=custom_py,
        )
