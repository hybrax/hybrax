from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path

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
from pydantic import ValidationError

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
    variable_names: tuple[str, ...] = ("biomass",),
    noise_model: str = "add",
    n_children: int = 1,
    n_time_points: int = 6,
    noise_scale: dict[str, float] | None = None,
    min_relative_residual_rms: float = 1e-6,
) -> RunConfig:
    augmentation = AugmentationConfig(
        seed=12,
        n_children_per_process=n_children,
        n_time_points=n_time_points,
        variable_names=variable_names,
        noise_scale=noise_scale
        if noise_scale is not None
        else {name: 0.7 for name in variable_names},
        noise_model=noise_model,
        min_relative_residual_rms=min_relative_residual_rms,
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
            "noise_scale": {"biomass": 0.0},
        },
        {
            "n_children_per_process": 1,
            "n_time_points": 2,
            "variable_names": ["biomass"],
            "noise_scale": {"biomass": float("nan")},
        },
    ],
)
def test_invalid_config_fails_fast(config):
    with pytest.raises(ValidationError):
        AugmentationConfig.model_validate(config)


def test_degenerate_parent_time_range_fails_fast():
    collection = _collection()
    collection.processes["p1"].time_axis.end = 0.0

    with pytest.raises(ValueError, match="p1: cannot augment a degenerate time range"):
        augment_process_collection(collection, _config())


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
        assert np.all(np.diff(biomass.times) > 0.0)
        np.testing.assert_array_equal(biomass.times, ratio.times)
        np.testing.assert_array_equal(
            biomass.times, _state_series(other, "biomass").times
        )
        np.testing.assert_array_equal(
            biomass.values, _state_series(other, "biomass").values
        )

    assert not np.array_equal(grids[0], grids[1])


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
    child = augment_process_collection(collection, _config()).processes["p1__aug_000"]
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


def test_modeled_reactor_component_and_uncontrolled_pv_are_accepted():
    collection = augment_process_collection(
        _collection(),
        _config(variable_names=("biomass", "ratio")),
    )

    child = collection.processes["p1__aug_000"]
    assert len(_state_series(child, "biomass").times) == 6
    assert len(_state_series(child, "ratio").times) == 6


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
        _config(noise_model=noise_model, noise_scale={"biomass": scale}),
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
        rel_std = err_std / max(np.mean(base[base > 0.0]), 1e-8)
        sigma = np.sqrt(np.log1p(rel_std**2))
        expected = base * np.exp(-0.5 * sigma**2 + sigma * z)
    np.testing.assert_allclose(actual, expected)


def test_built_in_noise_rejects_effectively_zero_relative_residual():
    with pytest.raises(
        ValueError, match="p1.*biomass.*effectively zero spline residual"
    ):
        augment_process_collection(
            _collection(),
            _config(min_relative_residual_rms=0.9),
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
    custom_py = tmp_path / "custom-absolute-noise.py"
    custom_py.write_text(
        "\n".join(
            [
                "import numpy as np",
                "",
                "def augment_state_values(*, base_values, standard_normal, **_):",
                "    return np.clip(base_values + 0.2 * standard_normal, 0, None)",
            ]
        ),
        encoding="utf-8",
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
    custom_py = tmp_path / "custom-add-process.py"
    custom_py.write_text(
        "\n".join(
            [
                "from copy import deepcopy",
                "",
                "def transform_process_collection(collection, config):",
                "    added = deepcopy(collection.processes['p1'])",
                "    added.metadata.name = 'added'",
                "    collection.processes = {",
                "        'added': added, **collection.processes",
                "    }",
                "    return collection",
            ]
        ),
        encoding="utf-8",
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
    custom_py = tmp_path / "custom-add-augmented-process.py"
    custom_py.write_text(
        "\n".join(
            [
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
            ]
        ),
        encoding="utf-8",
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
    custom_py = tmp_path / "custom-ambiguous-tags.py"
    custom_py.write_text(
        "\n".join(
            [
                "from copy import deepcopy",
                "",
                "def transform_process_collection(collection, config):",
                "    process = collection.processes.pop('p1')",
                "    first = deepcopy(process)",
                "    first.metadata.name = 'first'",
                "    process.metadata.name = 'second'",
                "    collection.processes = {'first': first, 'second': process}",
                "    return collection",
            ]
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="ambiguous pre-transform provenance tag"):
        _prepare_collection(
            tmp_path,
            "prepared-ambiguous-tags",
            custom_py=custom_py,
        )


def test_prepare_rejects_invalid_augmented_parent_reference(tmp_path):
    custom_py = tmp_path / "custom-invalid-parent.py"
    custom_py.write_text(
        "\n".join(
            [
                "from bp_format.dataclasses import AugmentedBioProcess",
                "",
                "def transform_process_collection(collection, config):",
                "    process = collection.processes.pop('p1')",
                "    process.metadata.name = 'child'",
                "    collection.processes['child'] = AugmentedBioProcess(",
                "        **vars(process), parent_process='missing'",
                "    )",
                "    return collection",
            ]
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="augmented parent validation failed"):
        _prepare_collection(
            tmp_path,
            "prepared-invalid-parent",
            custom_py=custom_py,
        )
