from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
import warnings

import numpy as np
import pytest
from bp_format.dataclasses import (
    AugmentedBioProcess,
    BioProcess,
    BioProcessCollection,
    BioProcessMetadata,
    ProcessVariable,
    ReactorMedium,
    ReactorMediumComponent,
    SampleVolumeChange,
    TimeAxis,
    TimeSeries,
    Volume,
)
from bp_format.splines import fit_timeseries_spline
from bp_format.serialization import load_process_collection, save_process_collection
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from pydantic import ValidationError

import bp_train.augmentation as augmentation_module
import bp_train.augmentation_plot as augmentation_plot_module
import bp_train.prepare as prepare_module
from bp_train.augmentation import augment_process_collection
from bp_train.loo import _build_fold_groups
from bp_train.prepare import prepare_artifact
from bp_train.run_config import (
    AugmentationConfig,
    PrepareConfig,
    RunConfig,
    load_prepare_config,
)
from bp_train.training_data import TrainingDataStore


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


def _two_process_collection() -> BioProcessCollection:
    collection = _collection()
    second = deepcopy(collection.processes["p1"])
    second.metadata.name = "p2"
    second.reactor_medium.components["biomass"].concentration = _spline(
        [0.2, 0.9, 1.0, 1.8],
        smoothing_s=0.2,
    )
    collection.processes["p2"] = second
    return collection


def _config(
    *,
    variable_names: tuple[str, ...] = ("biomass",),
    noise_model: str = "add",
    n_children: int = 1,
    n_time_points: int = 6,
    min_spacing_fraction: float = 0.1,
    noise_scale: dict[str, float] | None = None,
    residual_scope: str = "process",
    initial_value_source: str | dict[str, str] = "measured",
) -> RunConfig:
    augmentation = AugmentationConfig(
        seed=12,
        n_children_per_process=n_children,
        n_time_points=n_time_points,
        min_spacing_fraction=min_spacing_fraction,
        variable_names=variable_names,
        noise_scale=noise_scale
        if noise_scale is not None
        else {name: 0.7 for name in variable_names},
        noise_model=noise_model,
        residual_scope=residual_scope,
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
        "variable_names": ["biomass"],
        "noise_scale": {"biomass": 0.7},
        "noise_model": "add",
    }
    config.update(updates)
    return config


def test_no_config_leaves_collection_unchanged():
    collection = _collection()
    config = RunConfig(prepare=PrepareConfig(raw_input=Path("unused.json")))

    assert augment_process_collection(collection, config) is collection
    assert list(collection.processes) == ["p1"]


def test_prepare_with_augmentation_writes_plot(tmp_path, monkeypatch):
    rendered_variable_names = []
    rendered_process_names = []
    requested_state_names = []
    rendered_ylabels = []
    rendered_noise_stds = []
    render_augmentation_plot = prepare_module.render_augmentation_plot
    state_series = augmentation_plot_module._state_series
    fill_between = Axes.fill_between
    save_figure = Figure.savefig

    def track_render(collection, augmentation, output_path):
        rendered_variable_names.append(augmentation.variable_names)
        rendered_process_names.append(tuple(collection.processes))
        render_augmentation_plot(collection, augmentation, output_path)

    def track_state_series(process, state_name):
        requested_state_names.append(state_name)
        return state_series(process, state_name)

    def track_save_figure(figure, *args, **kwargs):
        rendered_ylabels.append(tuple(axis.get_ylabel() for axis in figure.axes))
        return save_figure(figure, *args, **kwargs)

    def track_fill_between(axis, x, y1, y2, *args, **kwargs):
        rendered_noise_stds.append(np.asarray(y2) - np.asarray(y1))
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
            variable_names=["biomass", "ratio"],
            noise_scale={"biomass": 0.7, "ratio": 0.7},
        ),
    )

    plot_path = tmp_path / "with-augmentation-plot" / "augmented-data.png"
    assert plot_path.is_file()
    assert plot_path.stat().st_size > 0
    assert rendered_variable_names == [("biomass", "ratio")]
    assert rendered_process_names == [("p1", "p1__aug_000")]
    assert set(requested_state_names) == {"biomass", "ratio"}
    assert rendered_ylabels == [("p1\n[g/L]", "[-]")]
    parent = _collection().processes["p1"]
    expected_noise_stds = [
        0.7
        * augmentation_module._residual_statistics(
            "p1",
            state_name,
            _state_series(parent, state_name),
        )[0]
        for state_name in ("biomass", "ratio")
    ]
    for band_width, expected_noise_std in zip(
        rendered_noise_stds,
        expected_noise_stds,
        strict=True,
    ):
        np.testing.assert_allclose(band_width[0], 0.0)
        np.testing.assert_allclose(band_width[1:], 2.0 * expected_noise_std)


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
            "variable_names": ["biomass"],
        },
        {
            "n_children_per_process": 1,
            "n_time_points": 1,
            "variable_names": ["biomass"],
        },
        {
            "n_children_per_process": 1,
            "n_time_points": 2,
            "variable_names": [],
        },
        {
            "n_children_per_process": 1,
            "n_time_points": 2,
            "variable_names": ["biomass"],
            "noise_model": "other",
        },
        {
            "n_children_per_process": 1,
            "n_time_points": 2,
            "variable_names": ["biomass"],
            "residual_scope": "other",
        },
        {
            "n_children_per_process": 1,
            "n_time_points": 2,
            "min_spacing_fraction": 0.0,
            "variable_names": ["biomass"],
        },
        {
            "n_children_per_process": 1,
            "n_time_points": 2,
            "min_spacing_fraction": True,
            "variable_names": ["biomass"],
        },
        {
            "n_children_per_process": 1,
            "n_time_points": 2,
            "min_spacing_fraction": 1.1,
            "variable_names": ["biomass"],
        },
        {
            "n_children_per_process": 1,
            "n_time_points": 2,
            "variable_names": ["biomass"],
            "noise_scale": {"biomass": 0.0},
        },
        {
            "n_children_per_process": 1,
            "n_time_points": 2,
            "variable_names": ["biomass"],
            "noise_scale": {"biomass": float("nan")},
        },
        {
            "n_children_per_process": 1,
            "n_time_points": 2,
            "variable_names": ["biomass"],
            "initial_value_source": "other",
        },
        {
            "n_children_per_process": 1,
            "n_time_points": 2,
            "variable_names": ["biomass", "ratio"],
            "initial_value_source": {"biomass": "measured"},
        },
    ],
)
def test_invalid_config_fails_fast(config):
    with pytest.raises(ValidationError):
        AugmentationConfig.model_validate(config)


def test_augmentation_defaults():
    config = AugmentationConfig.model_validate(
        {
            "n_children_per_process": 1,
            "n_time_points": 2,
            "variable_names": ["biomass"],
        }
    )

    assert config.initial_value_source == "measured"
    assert config.min_spacing_fraction == 0.1
    assert config.residual_scope == "process"


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
        lambda *, base_values, **_: base_values + 10.0,
    ).processes["p1__aug_000"]

    expected_value = {
        "measured": measured,
        "spline": spline,
        "augmented": spline + 10.0,
    }[expected]
    assert _state_series(child, "biomass").values[0] == pytest.approx(expected_value)


def test_initial_value_source_mapping_controls_each_listed_state():
    collection = _collection()
    parent = collection.processes["p1"]
    biomass = _state_series(parent, "biomass")
    ratio = _state_series(parent, "ratio")

    child = augment_process_collection(
        collection,
        _config(
            variable_names=("biomass", "ratio"),
            initial_value_source={"biomass": "measured", "ratio": "spline"},
        ),
        lambda *, base_values, **_: base_values + 10.0,
    ).processes["p1__aug_000"]

    assert _state_series(child, "biomass").values[0] == pytest.approx(biomass.values[0])
    assert _state_series(child, "ratio").values[0] == pytest.approx(ratio.evaluate(0.0))


@pytest.mark.parametrize("source", ["measured", "spline"])
def test_initial_value_overwrite_accepts_read_only_hook_values(source):
    collection = _collection()
    parent_series = _state_series(collection.processes["p1"], "biomass")

    child = augment_process_collection(
        collection,
        _config(initial_value_source=source),
        lambda *, base_values, **_: np.broadcast_to(base_values[0], base_values.shape),
    ).processes["p1__aug_000"]

    expected = (
        parent_series.values[0] if source == "measured" else parent_series.evaluate(0.0)
    )
    assert _state_series(child, "biomass").values[0] == pytest.approx(expected)


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
                variable_names=("ratio",),
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
                variable_names=("ratio",),
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
                variable_names=("ratio",),
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

    def build_grid(augmentation, parent_name, child_index, t0, t_end):
        del parent_name
        if child_index == 2:
            raise ValueError(
                "cannot represent the requested minimum child-grid spacing"
            )
        return np.linspace(t0, t_end, augmentation.n_time_points)

    monkeypatch.setattr(augmentation_module, "_child_grid", build_grid)

    with pytest.raises(ValueError, match="cannot represent the requested"):
        augment_process_collection(collection, _config(n_children=5))

    assert list(collection.processes) == ["p1"]


def test_children_preserve_physical_structure_without_sharing_objects():
    collection = _collection()
    parent = collection.processes["p1"]
    parent.volume.volume_changes["sample"] = SampleVolumeChange(
        name="sample",
        unit="L",
        is_controlled=False,
        is_continuous=False,
        values=TimeSeries(times=[2.0], values=[-0.1]),
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
    collection = augment_process_collection(
        _collection(),
        _config(variable_names=("biomass",)),
    )

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
        augment_process_collection(_collection(), _config(variable_names=(state_name,)))


@pytest.mark.parametrize("noise_model", ["add", "mult"])
def test_residual_scaled_noise_matches_formula(noise_model):
    captured = {}
    parent_series = _state_series(_collection().processes["p1"], "biomass")
    observed = np.asarray(parent_series.values)
    fitted = np.asarray(parent_series.evaluate_many(parent_series.times))
    residual_rms = np.sqrt(np.mean((observed - fitted) ** 2))

    def capture(**kwargs):
        captured.update(kwargs)
        return None

    scale = 1.4
    collection = augment_process_collection(
        _collection(),
        _config(
            noise_model=noise_model,
            noise_scale={"biomass": scale},
            initial_value_source="augmented",
        ),
        capture,
    )
    actual = np.asarray(
        _state_series(collection.processes["p1__aug_000"], "biomass").values
    )
    base = captured["base_values"]
    z = captured["standard_normal"]
    assert captured["residual_rms"] == pytest.approx(residual_rms)
    err_std = scale * residual_rms

    if noise_model == "add":
        expected = np.clip(base + z * err_std, 0.0, None)
    else:
        fitted = np.asarray(parent_series.evaluate_many(parent_series.times))
        rel_std = err_std / max(np.mean(np.abs(fitted[fitted != 0.0])), 1e-8)
        sigma = np.sqrt(np.log1p(rel_std**2))
        expected = base * np.exp(-0.5 * sigma**2 + sigma * z)
    np.testing.assert_allclose(actual, expected)


def test_variable_residual_scope_uses_observation_weighted_pooled_rms():
    collection = _two_process_collection()
    residuals = []
    for process in collection.processes.values():
        series = _state_series(process, "biomass")
        observed = np.asarray(series.values)
        fitted = np.asarray(series.evaluate_many(series.times))
        residuals.extend(observed - fitted)
    zero_process = deepcopy(collection.processes["p1"])
    zero_process.metadata.name = "p3"
    zero_process.reactor_medium.components["biomass"].concentration = _spline(
        [0.0] * 5,
        smoothing_s=0.0,
    )
    collection.processes["p3"] = zero_process
    pooled_residual_rms = np.sqrt(np.mean(np.asarray(residuals) ** 2))
    captured = {}

    def capture(**kwargs):
        captured[kwargs["parent_name"]] = kwargs
        return None

    scale = 1.4
    augmented = augment_process_collection(
        collection,
        _config(
            noise_scale={"biomass": scale},
            residual_scope="variable",
            initial_value_source="augmented",
        ),
        capture,
    )

    assert set(captured) == {"p1", "p2", "p3"}
    for parent_name, values in captured.items():
        assert values["residual_rms"] == pytest.approx(pooled_residual_rms)
        actual = np.asarray(
            _state_series(
                augmented.processes[f"{parent_name}__aug_000"],
                "biomass",
            ).values
        )
        expected = np.clip(
            values["base_values"]
            + scale * pooled_residual_rms * values["standard_normal"],
            0.0,
            None,
        )
        np.testing.assert_allclose(actual, expected)


def test_variable_scope_pools_observed_scale_rms_excluding_zero_traces():
    collection = _two_process_collection()
    zero_process = deepcopy(collection.processes["p1"])
    zero_process.metadata.name = "p3"
    zero_process.reactor_medium.components["biomass"].concentration = _spline(
        [0.0] * 5,
        smoothing_s=0.0,
    )
    collection.processes["p3"] = zero_process

    augmentation = _config(residual_scope="variable").prepare.augmentation
    parents = augmentation_module._parent_processes(collection)
    statistics = augmentation_module._effective_residual_statistics(
        parents,
        augmentation,
    )

    weighted_squares = 0.0
    observation_count = 0
    per_parent_observed = {}
    for parent_name, process in parents:
        series = _state_series(process, "biomass")
        # Recompute from the fixture observations rather than reusing
        # _residual_statistics, so a bug in that helper cannot cancel out.
        observed = np.asarray(series.values, dtype=float)
        observed_rms = float(np.sqrt(np.mean(observed**2)))
        per_parent_observed[parent_name] = observed_rms
        if observed_rms == 0.0:
            continue
        weighted_squares += len(series.values) * observed_rms**2
        observation_count += len(series.values)
    expected_pooled_observed = np.sqrt(weighted_squares / observation_count)

    # The zero trace is excluded from the pool but must still be assigned it.
    assert per_parent_observed["p3"] == 0.0
    assert expected_pooled_observed > 0.0
    assert expected_pooled_observed != pytest.approx(per_parent_observed["p1"])
    for parent_name, _ in parents:
        _, observed_rms, observed_scale_rms = statistics[parent_name, "biomass"]
        # Middle slot keeps each parent's own observed RMS for the hook contract.
        assert observed_rms == pytest.approx(per_parent_observed[parent_name])
        # Third slot is the shared observation-weighted pool.
        assert observed_scale_rms == pytest.approx(expected_pooled_observed)


def test_process_residual_scope_keeps_parent_specific_rms():
    collection = _two_process_collection()
    expected = {
        parent_name: augmentation_module._residual_statistics(
            parent_name,
            "biomass",
            _state_series(process, "biomass"),
        )[0]
        for parent_name, process in collection.processes.items()
    }
    captured = {}

    def capture(**kwargs):
        captured[kwargs["parent_name"]] = kwargs["residual_rms"]
        return kwargs["base_values"]

    augment_process_collection(collection, _config(), capture)

    assert captured == pytest.approx(expected)
    assert captured["p1"] != pytest.approx(captured["p2"])


def test_variable_residual_scope_plot_uses_one_configured_noise_std(
    tmp_path,
    monkeypatch,
):
    collection = _two_process_collection()
    residuals = []
    for process in collection.processes.values():
        series = _state_series(process, "biomass")
        residuals.extend(
            np.asarray(series.values) - np.asarray(series.evaluate_many(series.times))
        )
    pooled_residual_rms = np.sqrt(np.mean(np.asarray(residuals) ** 2))
    scale = 2.5
    config = _config(
        noise_scale={"biomass": scale},
        residual_scope="variable",
    )
    augment_process_collection(collection, config)
    band_noise_stds = []
    fill_between = Axes.fill_between

    def track_fill_between(axis, x, y1, y2, *args, **kwargs):
        band_noise_stds.append((np.asarray(y2) - np.asarray(y1)) / 2.0)
        return fill_between(axis, x, y1, y2, *args, **kwargs)

    monkeypatch.setattr(Axes, "fill_between", track_fill_between)
    augmentation_plot_module.render_augmentation_plot(
        collection,
        config.prepare.augmentation,
        tmp_path / "shared-rms.png",
    )

    assert len(band_noise_stds) == 2
    for noise_std in band_noise_stds:
        np.testing.assert_allclose(noise_std[0], 0.0)
        np.testing.assert_allclose(noise_std[1:], scale * pooled_residual_rms)


@pytest.mark.parametrize(
    ("initial_value_source", "fixed_initial_value"),
    [("measured", True), ("spline", True), ("augmented", False)],
)
def test_multiplicative_plot_band_matches_pointwise_noise_std(
    tmp_path,
    monkeypatch,
    initial_value_source,
    fixed_initial_value,
):
    collection = _collection()
    config = _config(
        noise_model="mult",
        initial_value_source=initial_value_source,
    )
    parent_series = _state_series(collection.processes["p1"], "biomass")
    residual_rms, _ = augmentation_module._residual_statistics(
        "p1",
        "biomass",
        parent_series,
    )
    fitted = np.asarray(parent_series.evaluate_many(parent_series.times))
    reference_magnitude = np.mean(np.abs(fitted[fitted != 0.0]))
    rel_std = 0.7 * residual_rms / reference_magnitude
    augment_process_collection(collection, config)
    bands = []
    fill_between = Axes.fill_between

    def track_fill_between(axis, x, y1, y2, *args, **kwargs):
        bands.append((np.asarray(x), np.asarray(y1), np.asarray(y2)))
        return fill_between(axis, x, y1, y2, *args, **kwargs)

    monkeypatch.setattr(Axes, "fill_between", track_fill_between)
    augmentation_plot_module.render_augmentation_plot(
        collection,
        config.prepare.augmentation,
        tmp_path / "multiplicative-noise.png",
    )

    assert len(bands) == 1
    times, lower, upper = bands[0]
    spline_values = np.asarray(parent_series.evaluate_many(times))
    expected_std = np.abs(spline_values) * rel_std
    if fixed_initial_value:
        expected_std[0] = 0.0
    np.testing.assert_allclose((upper - lower) / 2.0, expected_std)
    assert expected_std[np.argmax(spline_values)] > expected_std[0]


@pytest.mark.parametrize(
    "levels",
    [
        np.asarray([-1.5]),
        np.asarray([-2.0, -0.5, 0.0, 0.2, 1.0]),
    ],
)
def test_multiplicative_noise_is_well_scaled_for_signed_values(levels):
    config = _config(
        variable_names=("signed_pv",),
        noise_model="mult",
        noise_scale={"signed_pv": 1.0},
    ).prepare.augmentation
    draws_per_level = 25_000
    base = np.repeat(levels, draws_per_level)
    standard_normal = np.random.default_rng(123).standard_normal(base.shape)

    actual = augmentation_module._built_in_values(
        parent_name="p1",
        state_name="signed_pv",
        base_values=base,
        residual_rms=0.75,
        observed_scale_rms=1.5,
        multiplicative_reference_magnitude=1.5,
        standard_normal=standard_normal,
        augmentation=config,
    )

    np.testing.assert_array_equal(np.sign(actual), np.sign(base))
    for level in levels:
        np.testing.assert_allclose(
            np.mean(actual[base == level]),
            level,
            rtol=0.02,
            atol=1e-12,
        )
    factors = actual[base != 0.0] / base[base != 0.0]
    assert np.quantile(factors, 0.01) > 0.1
    assert np.quantile(factors, 0.99) < 10.0


def test_residual_statistics_are_computed_once_per_parent_state(monkeypatch):
    calls = []
    original = augmentation_module._residual_statistics

    def track_call(parent_name, state_name, series):
        calls.append((parent_name, state_name))
        return original(parent_name, state_name, series)

    monkeypatch.setattr(augmentation_module, "_residual_statistics", track_call)

    augment_process_collection(_collection(), _config(n_children=4))

    assert calls == [("p1", "biomass")]


def test_built_in_noise_rejects_effectively_zero_relative_residual():
    collection = _collection()
    collection.processes["p1"].reactor_medium.components[
        "biomass"
    ].concentration = _spline(
        [0.4, 0.7, 0.8, 1.4, 1.3, 2.0, 2.2],
        smoothing_s=0.0,
    )

    with pytest.raises(
        ValueError, match="p1.*biomass.*effectively zero spline residual"
    ):
        augment_process_collection(collection, _config())


def test_built_in_noise_rejects_zero_observed_trace():
    with pytest.raises(
        ValueError, match="p1.*biomass.*effectively zero spline residual"
    ):
        augment_process_collection(_collection(zero_trace=True), _config())


def test_built_in_noise_uses_fixed_relative_residual_boundary():
    augmentation = _config().prepare.augmentation
    arguments = {
        "parent_name": "p1",
        "state_name": "biomass",
        "base_values": np.ones(1),
        "observed_scale_rms": 1.0,
        "multiplicative_reference_magnitude": 1.0,
        "standard_normal": np.zeros(1),
        "augmentation": augmentation,
    }

    with pytest.raises(ValueError, match="effectively zero spline residual"):
        augmentation_module._built_in_values(residual_rms=1e-6, **arguments)

    augmentation_module._built_in_values(
        residual_rms=np.nextafter(1e-6, np.inf),
        **arguments,
    )


def test_hook_can_override_zero_trace_with_absolute_noise():
    hook_inputs = {}

    def absolute_noise(*, state_name, base_values, standard_normal, **_):
        hook_inputs[state_name] = (base_values, standard_normal)
        if state_name == "biomass":
            return np.clip(base_values + 0.2 * standard_normal, 0.0, None)
        return None

    collection = augment_process_collection(
        _collection(zero_trace=True),
        _config(
            variable_names=("biomass", "ratio"),
            noise_scale={"ratio": 0.7},
            initial_value_source="augmented",
        ),
        absolute_noise,
    )
    values = np.asarray(
        _state_series(collection.processes["p1__aug_000"], "biomass").values
    )

    base, standard_normal = hook_inputs["biomass"]
    np.testing.assert_allclose(
        values,
        np.clip(base + 0.2 * standard_normal, 0.0, None),
    )
    assert set(hook_inputs) == {"biomass", "ratio"}


@pytest.mark.parametrize(
    ("returned", "message"),
    [
        (np.ones(2), "returned shape"),
        (np.asarray([np.nan] * 6), "returned non-finite values"),
    ],
)
def test_hook_result_shape_and_finiteness_are_validated(returned, message):
    with pytest.raises(ValueError, match=message):
        augment_process_collection(_collection(), _config(), lambda **_: returned)


def test_additive_noise_clips_at_zero():
    collection = _collection()
    child = augment_process_collection(
        collection,
        _config(noise_model="add", noise_scale={"biomass": 1_000.0}),
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
            _config(variable_names=("ratio",), n_time_points=2),
        ).processes["p1__aug_000"]

    values = np.asarray(_state_series(child, "biomass").values)
    assert np.all(values >= 0.0)


def test_custom_negative_reactor_medium_values_are_clipped():
    child = augment_process_collection(
        _collection(),
        _config(initial_value_source="augmented"),
        lambda *, base_values, **_: np.full_like(base_values, -1.0),
    ).processes["p1__aug_000"]

    assert np.all(np.asarray(_state_series(child, "biomass").values) == 0.0)


def test_custom_negative_process_variable_values_are_not_clipped():
    child = augment_process_collection(
        _collection(),
        _config(variable_names=("ratio",), initial_value_source="augmented"),
        lambda *, base_values, **_: np.full_like(base_values, -1.0),
    ).processes["p1__aug_000"]

    assert np.all(np.asarray(_state_series(child, "ratio").values) == -1.0)


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
        first.metadata["bp-train"]["provenance"]["content_hash"]
        == second.metadata["bp-train"]["provenance"]["content_hash"]
    )


def test_prepare_records_augmented_provenance_and_hook_metadata(tmp_path):
    prepared = _prepare_collection(
        tmp_path,
        "prepared-provenance",
        augmentation=_augmentation_dict(n_children_per_process=1),
    )
    metadata = prepared.metadata["bp-train"]
    child_provenance = metadata["semantics_provenance"]["processes"]["p1__aug_000"]

    assert child_provenance["raw"] is None
    assert child_provenance["changed_by_hooks"] == ["augmentation"]
    assert metadata["transform_hooks"]["augment_state_values"] is None
    assert _build_fold_groups(prepared) == (("p1", ("p1", "p1__aug_000")),)


def test_prepare_accepts_custom_absolute_noise_hook_for_zero_trace(tmp_path):
    custom_py = _write_custom_module(
        tmp_path,
        "absolute-noise",
        "import numpy as np",
        "",
        "def augment_state_values(*, base_values, standard_normal, **_):",
        "    return np.clip(base_values + 0.2 * standard_normal, 0, None)",
    )
    prepared = _prepare_collection(
        tmp_path,
        "prepared-custom-noise",
        collection=_collection(zero_trace=True),
        augmentation=_augmentation_dict(
            n_children_per_process=1,
            noise_scale={},
        ),
        custom_py=custom_py,
    )

    values = np.asarray(
        _state_series(prepared.processes["p1__aug_000"], "biomass").values
    )
    assert np.any(values > 0.0)
    assert (
        prepared.metadata["bp-train"]["transform_hooks"]["augment_state_values"]
        == "augment_state_values"
    )


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
    provenance = prepared.metadata["bp-train"]["semantics_provenance"]["processes"][
        "added"
    ]

    assert provenance["raw"] is None
    assert provenance["changed_by_hooks"] == ["transform_process_collection"]


def test_transform_created_augmented_process_is_attributed_to_transform(tmp_path):
    custom_py = _write_custom_module(
        tmp_path,
        "add-augmented-process",
        "from copy import deepcopy",
        "from bp_format.dataclasses import AugmentedBioProcess",
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
    provenance = prepared.metadata["bp-train"]["semantics_provenance"]["processes"][
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
        "from bp_format.dataclasses import AugmentedBioProcess",
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
