from __future__ import annotations

from pathlib import Path

import jax.numpy as jnp
import numpy as np
import pytest
from bpbench.dataclasses import (
    BioProcess,
    BioProcessCollection,
    BioProcessMetadata,
    ProcessVariable,
    ReactorMedium,
    ReactorMediumComponent,
    SampleVolumeChange,
    StaticVariable,
    TimeAxis,
    TimeSeries,
    Volume,
)

from bp_train.prepare import prepare_artifact
from bp_train.training_data import TrainingDataStore


def _make_two_process_collection() -> BioProcessCollection:
    p1 = BioProcess(
        metadata=BioProcessMetadata(name="p1", process_type="fed_batch"),
        time_axis=TimeAxis(unit="h", start=0.0, end=4.0, time_reference="start"),
        volume=Volume(
            initial_volume=1.0,
            unit="L",
            volume_changes={
                "sample_1": SampleVolumeChange(
                    name="sample_1",
                    unit="L",
                    is_controlled=False,
                    is_continuous=False,
                    values=TimeSeries(
                        times=jnp.asarray([2.0]),
                        values=jnp.asarray([-0.1]),
                    ),
                )
            },
        ),
        reactor_medium=ReactorMedium(
            name="rm",
            density=1.0,
            density_unit="kg/L",
            components={
                "biomass": ReactorMediumComponent(
                    name="biomass",
                    unit="g/L",
                    concentration=StaticVariable(0.1),
                    is_intracellular=False,
                )
            },
        ),
        process_variables={
            "CF": ProcessVariable(
                name="CF",
                unit="g/L",
                is_controlled=False,
                values=TimeSeries(
                    times=jnp.asarray([0.0, 2.0, 4.0]),
                    values=jnp.asarray([1.0, 1.2, 1.4]),
                ),
            ),
            "X": ProcessVariable(
                name="X",
                unit="g/L",
                is_controlled=False,
                values=TimeSeries(
                    times=jnp.asarray([0.0, 2.0, 4.0]),
                    values=jnp.asarray([0.2, 0.3, 0.4]),
                ),
            ),
            "T": ProcessVariable(
                name="T",
                unit="K",
                is_controlled=False,
                values=TimeSeries(
                    times=jnp.asarray([0.0, 2.0, 4.0]),
                    values=jnp.asarray([300.0, 301.0, 302.0]),
                ),
            ),
        },
    )

    p2 = BioProcess(
        metadata=BioProcessMetadata(name="p2", process_type="fed_batch"),
        time_axis=TimeAxis(unit="h", start=0.0, end=4.0, time_reference="start"),
        volume=Volume(
            initial_volume=1.5,
            unit="L",
            volume_changes={
                "sample_1": SampleVolumeChange(
                    name="sample_1",
                    unit="L",
                    is_controlled=False,
                    is_continuous=False,
                    values=TimeSeries(
                        times=jnp.asarray([2.0]),
                        values=jnp.asarray([-0.2]),
                    ),
                )
            },
        ),
        reactor_medium=ReactorMedium(
            name="rm",
            density=1.0,
            density_unit="kg/L",
            components={
                "biomass": ReactorMediumComponent(
                    name="biomass",
                    unit="g/L",
                    concentration=StaticVariable(0.1),
                    is_intracellular=False,
                )
            },
        ),
        process_variables={
            "CF": ProcessVariable(
                name="CF",
                unit="g/L",
                is_controlled=False,
                values=TimeSeries(
                    times=jnp.asarray([0.0, 1.0]),
                    values=jnp.asarray([1.1, 1.3]),
                ),
            ),
            "X": ProcessVariable(
                name="X",
                unit="g/L",
                is_controlled=False,
                values=TimeSeries(
                    times=jnp.asarray([0.0, 1.0]),
                    values=jnp.asarray([0.25, 0.35]),
                ),
            ),
            "T": ProcessVariable(
                name="T",
                unit="K",
                is_controlled=False,
                values=TimeSeries(
                    times=jnp.asarray([0.0, 1.0]),
                    values=jnp.asarray([299.0, 300.0]),
                ),
            ),
        },
    )

    return BioProcessCollection(
        metadata={"case_study": {"case_id": "training-data"}},
        processes={"p1": p1, "p2": p2},
    )


def _write_custom(path: Path) -> None:
    path.write_text(
        "\n".join(
            [
                "CONFIG = {'control_order': ['CF', 'T']}",
                "",
                "def transform_process_collection(collection, config):",
                "    for process in collection.processes.values():",
                "        process.process_variables['CF'].is_controlled = True",
                "        process.process_variables['T'].is_controlled = True",
                "    return collection",
            ]
        ),
        encoding="utf-8",
    )


def _prepare_collection(tmp_path: Path) -> Path:
    custom_py = tmp_path / "custom.py"
    _write_custom(custom_py)
    output = tmp_path / "prepared.json"
    prepare_artifact(_make_two_process_collection(), output, custom_py=custom_py)
    return output


def _make_reactor_target_collection() -> BioProcessCollection:
    p1 = BioProcess(
        metadata=BioProcessMetadata(name="p1", process_type="fed_batch"),
        time_axis=TimeAxis(unit="h", start=0.0, end=4.0, time_reference="start"),
        volume=Volume(
            initial_volume=1.0,
            unit="L",
            volume_changes={
                "sample_1": SampleVolumeChange(
                    name="sample_1",
                    unit="L",
                    is_controlled=False,
                    is_continuous=False,
                    values=TimeSeries(
                        times=jnp.asarray([2.0]),
                        values=jnp.asarray([-0.1]),
                    ),
                )
            },
        ),
        reactor_medium=ReactorMedium(
            name="rm",
            density=1.0,
            density_unit="kg/L",
            components={
                "biomass": ReactorMediumComponent(
                    name="biomass",
                    unit="g/L",
                    concentration=TimeSeries(
                        times=jnp.asarray([0.0, 2.0, 4.0]),
                        values=jnp.asarray([0.2, 0.3, 0.4]),
                    ),
                    is_intracellular=False,
                ),
                "product": ReactorMediumComponent(
                    name="product",
                    unit="g/L",
                    concentration=TimeSeries(
                        times=jnp.asarray([0.0, 2.0, 4.0]),
                        values=jnp.asarray([0.0, 0.1, 0.2]),
                    ),
                    is_intracellular=False,
                ),
            },
        ),
        process_variables={
            "temperature": ProcessVariable(
                name="temperature",
                unit="K",
                is_controlled=True,
                values=TimeSeries(
                    times=jnp.asarray([0.0, 2.0, 4.0]),
                    values=jnp.asarray([300.0, 300.5, 301.0]),
                ),
            )
        },
    )
    p2 = BioProcess(
        metadata=BioProcessMetadata(name="p2", process_type="fed_batch"),
        time_axis=TimeAxis(unit="h", start=0.0, end=4.0, time_reference="start"),
        volume=Volume(
            initial_volume=1.2,
            unit="L",
            volume_changes={
                "sample_1": SampleVolumeChange(
                    name="sample_1",
                    unit="L",
                    is_controlled=False,
                    is_continuous=False,
                    values=TimeSeries(
                        times=jnp.asarray([2.0]),
                        values=jnp.asarray([-0.15]),
                    ),
                )
            },
        ),
        reactor_medium=ReactorMedium(
            name="rm",
            density=1.0,
            density_unit="kg/L",
            components={
                "biomass": ReactorMediumComponent(
                    name="biomass",
                    unit="g/L",
                    concentration=TimeSeries(
                        times=jnp.asarray([0.0, 1.0]),
                        values=jnp.asarray([0.25, 0.35]),
                    ),
                    is_intracellular=False,
                ),
                "product": ReactorMediumComponent(
                    name="product",
                    unit="g/L",
                    concentration=TimeSeries(
                        times=jnp.asarray([0.0, 1.0]),
                        values=jnp.asarray([0.02, 0.09]),
                    ),
                    is_intracellular=False,
                ),
            },
        ),
        process_variables={
            "temperature": ProcessVariable(
                name="temperature",
                unit="K",
                is_controlled=True,
                values=TimeSeries(
                    times=jnp.asarray([0.0, 1.0]),
                    values=jnp.asarray([299.0, 300.0]),
                ),
            )
        },
    )
    return BioProcessCollection(processes={"p1": p1, "p2": p2}, metadata={})


def test_training_data_store_builds_padded_measurement_arrays(tmp_path):
    prepared_json = _prepare_collection(tmp_path)
    store = TrainingDataStore.from_json(prepared_json, target_variable_order=["X"])

    assert store.process_order == ["p1", "p2"]
    assert store.target_names == ["X"]
    assert tuple(store.t_meas.shape) == (2, 3)
    assert tuple(store.y_meas.shape) == (2, 3, 1)
    assert tuple(store.meas_mask.shape) == (2, 3)
    assert tuple(store.y0.shape) == (2, 2)

    assert np.asarray(store.t_meas[0]).tolist() == pytest.approx([0.0, 2.0, 4.0])
    p2_t = np.asarray(store.t_meas[1])
    assert p2_t[:2].tolist() == pytest.approx([0.0, 1.0])
    assert p2_t[2] == pytest.approx(0.0)
    assert np.asarray(store.y_meas[1, 2, 0]) == pytest.approx(0.0)
    assert np.asarray(store.meas_mask[0]).tolist() == [True, True, True]
    assert np.asarray(store.meas_mask[1]).tolist() == [True, True, False]


def test_training_data_store_exposes_per_process_active_views(tmp_path):
    prepared_json = _prepare_collection(tmp_path)
    store = TrainingDataStore.from_json(prepared_json, target_variable_order=["X"])
    process_data = store.get_process("p2")

    assert process_data.process_name == "p2"
    assert process_data.n_meas == 2
    assert process_data.target_names == ["X"]
    assert np.asarray(process_data.active_t_meas).tolist() == pytest.approx([0.0, 1.0])
    assert np.asarray(process_data.active_y_meas[:, 0]).tolist() == pytest.approx(
        [0.25, 0.35]
    )
    assert np.asarray(process_data.active_meas_mask).tolist() == [True, True]
    assert process_data.controls.sample_acc_name == "V_sample_acc"


def test_training_data_store_builds_y0_with_vcont_last(tmp_path):
    prepared_json = _prepare_collection(tmp_path)
    store = TrainingDataStore.from_json(prepared_json, target_variable_order=["X"])

    assert np.asarray(store.y0[0]).tolist() == pytest.approx([0.2, 1.0])
    assert np.asarray(store.y0[1]).tolist() == pytest.approx([0.25, 1.5])


def test_training_data_store_rejects_inconsistent_target_set(tmp_path):
    collection = _make_two_process_collection()
    del collection.processes["p2"].process_variables["X"]

    custom_py = tmp_path / "custom.py"
    _write_custom(custom_py)
    prepared_json = tmp_path / "prepared.json"
    prepare_artifact(collection, prepared_json, custom_py=custom_py)

    with pytest.raises(
        ValueError,
        match="identical measured target names/order across processes",
    ):
        TrainingDataStore.from_json(prepared_json)


def test_training_data_store_rejects_inconsistent_target_order(tmp_path):
    collection = _make_two_process_collection()
    p2 = collection.processes["p2"]
    x = p2.process_variables.pop("X")
    p2.process_variables["X"] = x

    custom_py = tmp_path / "custom.py"
    custom_py.write_text(
        "\n".join(
            [
                "CONFIG = {'control_order': ['CF']}",
                "",
                "def transform_process_collection(collection, config):",
                "    for process in collection.processes.values():",
                "        process.process_variables['CF'].is_controlled = True",
                "    return collection",
            ]
        ),
        encoding="utf-8",
    )
    prepared_json = tmp_path / "prepared.json"
    prepare_artifact(collection, prepared_json, custom_py=custom_py)

    with pytest.raises(
        ValueError,
        match="identical measured target names/order across processes",
    ):
        TrainingDataStore.from_json(prepared_json)


def test_training_data_store_rejects_controlled_target_in_configured_order(tmp_path):
    prepared_json = _prepare_collection(tmp_path)
    with pytest.raises(
        ValueError,
        match="must be measured .* controlled targets",
    ):
        TrainingDataStore.from_json(prepared_json, target_variable_order=["CF"])


def test_training_data_store_supports_reactor_component_targets():
    store = TrainingDataStore.from_collection(
        _make_reactor_target_collection(),
        target_variable_order=["biomass", "product"],
        target_source="reactor_components",
    )

    assert store.target_source == "reactor_components"
    assert store.target_names == ["biomass", "product"]
    assert tuple(store.y_meas.shape) == (2, 3, 2)
    assert np.asarray(store.y0[0]).tolist() == pytest.approx([0.2, 0.0, 1.0])
    assert np.asarray(store.y0[1]).tolist() == pytest.approx([0.25, 0.02, 1.2])


def test_training_data_store_auto_falls_back_to_reactor_components():
    store = TrainingDataStore.from_collection(
        _make_reactor_target_collection(),
        target_variable_order=["biomass"],
        target_source="auto",
    )
    assert store.target_source == "reactor_components"
    assert store.target_names == ["biomass"]


def test_training_data_store_auto_prefers_process_variables_when_available(tmp_path):
    prepared_json = _prepare_collection(tmp_path)
    store = TrainingDataStore.from_json(
        prepared_json,
        target_source="auto",
    )
    assert store.target_source == "process_variables"
    assert store.target_names == ["X"]


def test_training_data_store_auto_reactor_fallback_requires_timeseries_compatible():
    collection = _make_reactor_target_collection()
    collection.processes["p1"].reactor_medium.components[
        "biomass"
    ].concentration = StaticVariable(0.2)

    with pytest.raises(
        ValueError,
        match="target_source='auto' could not resolve configured targets",
    ):
        TrainingDataStore.from_collection(
            collection,
            target_variable_order=["biomass"],
            target_source="auto",
        )


def test_training_data_store_gather_batch_by_process_indices(tmp_path):
    prepared_json = _prepare_collection(tmp_path)
    store = TrainingDataStore.from_json(prepared_json, target_variable_order=["X"])

    batch = store.gather_batch(jnp.asarray([1, 0, 1], dtype=jnp.int32))

    assert np.asarray(batch.process_indices).tolist() == [1, 0, 1]
    assert tuple(batch.t_meas.shape) == (3, 3)
    assert tuple(batch.y_meas.shape) == (3, 3, 1)
    assert tuple(batch.meas_mask.shape) == (3, 3)
    assert np.asarray(batch.n_meas).tolist() == [2, 3, 2]
    assert np.asarray(batch.y0[0]).tolist() == pytest.approx([0.25, 1.5])
    assert np.asarray(batch.y0[1]).tolist() == pytest.approx([0.2, 1.0])


def test_training_data_store_gather_batch_rejects_invalid_indices(tmp_path):
    prepared_json = _prepare_collection(tmp_path)
    store = TrainingDataStore.from_json(prepared_json, target_variable_order=["X"])

    with pytest.raises(ValueError, match="1D"):
        store.gather_batch(jnp.asarray([[0, 1]], dtype=jnp.int32))
    with pytest.raises(ValueError, match="non-empty"):
        store.gather_batch(jnp.asarray([], dtype=jnp.int32))
    with pytest.raises(IndexError, match="out of range"):
        store.gather_batch(jnp.asarray([0, 2], dtype=jnp.int32))
