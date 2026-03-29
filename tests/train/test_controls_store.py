from __future__ import annotations

import json

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

from pathlib import Path

from bp_train.controls_store import ControlsStore
from bp_train.prepare import prepare_artifact


def _make_two_process_collection() -> BioProcessCollection:
    processes = {}
    for name in ["p1", "p2"]:
        processes[name] = BioProcess(
            metadata=BioProcessMetadata(name=name, process_type="fed_batch"),
            time_axis=TimeAxis(unit="h", start=0.0, end=1.0, time_reference="start"),
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
                            times=jnp.asarray([0.5]),
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
                        times=jnp.asarray([0.0, 1.0]),
                        values=jnp.asarray([1.0, 1.1]),
                    ),
                ),
                "T": ProcessVariable(
                    name="T",
                    unit="C",
                    is_controlled=False,
                    values=TimeSeries(
                        times=jnp.asarray([0.0, 1.0]),
                        values=jnp.asarray([30.0, 31.0]),
                    ),
                ),
            },
        )
    return BioProcessCollection(
        metadata={"case_study": {"case_id": "two-process"}}, processes=processes
    )


def _write_control_custom_py(path: Path) -> None:
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


def _prepare_two_process(tmp_path: Path) -> Path:
    custom_py = tmp_path / "custom.py"
    _write_control_custom_py(custom_py)
    output = tmp_path / "prepared.json"
    prepare_artifact(_make_two_process_collection(), output, custom_py=custom_py)
    return output


def _prepare_two_process_inconsistent_controls(tmp_path: Path) -> Path:
    custom_py = tmp_path / "custom-inconsistent.py"
    custom_py.write_text(
        "\n".join(
            [
                "CONFIG = {'require_consistent_controls': False}",
                "",
                "def transform_process_collection(collection, config):",
                "    p1 = collection.processes['p1']",
                "    p2 = collection.processes['p2']",
                "    p1.process_variables['CF'].is_controlled = True",
                "    p2.process_variables['T'].is_controlled = True",
                "    return collection",
            ]
        ),
        encoding="utf-8",
    )
    output = tmp_path / "prepared-inconsistent.json"
    prepare_artifact(_make_two_process_collection(), output, custom_py=custom_py)
    return output


def test_controls_store_loads_by_process_name_and_index(tmp_path):
    prepared_json = _prepare_two_process(tmp_path)

    store = ControlsStore.from_json(prepared_json)
    by_name = store.get_controls("p1")
    by_index = store.get_controls(0)

    assert store.process_order == ["p1", "p2"]
    assert by_name.process_name == "p1"
    assert by_index.process_name == "p1"
    assert by_name.control_names == ["CF", "T", "V_sample_acc"]
    assert by_name.sample_acc_name == "V_sample_acc"
    assert by_name.sample_acc_global_index == 2
    assert np.array_equal(
        np.asarray(by_name.dense_grid), np.asarray(by_index.dense_grid)
    )
    assert by_name.control_name_to_index["CF"] == 0
    assert by_name.control_name_to_index["V_sample_acc"] == 2
    assert by_name.control_names == by_name.global_control_names
    assert by_name.control_name_to_index == by_name.global_control_name_to_index
    assert tuple(store.control_values.shape) == (
        2,
        store.shape_metadata["max_grid_length"],
        store.shape_metadata["max_controls"],
    )
    assert tuple(by_name.control_values.shape) == (
        store.shape_metadata["max_grid_length"],
        store.shape_metadata["max_controls"],
    )


def test_controls_store_eval_matches_prepared_linear_payload(tmp_path):
    prepared_json = _prepare_two_process(tmp_path)

    store = ControlsStore.from_json(prepared_json)
    controls = store.get_controls("p1")

    scalar = controls.eval(0.25)
    assert scalar.shape == (3,)
    assert scalar[controls.control_name_to_index["CF"]] == pytest.approx(1.025)
    assert scalar[controls.control_name_to_index["T"]] == pytest.approx(30.25)
    assert scalar[controls.sample_acc_global_index] == pytest.approx(0.0)

    ts = np.asarray([0.25, 0.5, 1.0], dtype=float)
    values = controls.eval(ts)
    derivatives = controls.eval_derivative(ts)

    assert values.shape == (3, 3)
    assert derivatives.shape == (3, 3)
    assert values[:, controls.control_name_to_index["CF"]] == pytest.approx(
        [1.025, 1.05, 1.1]
    )
    assert values[:, controls.control_name_to_index["T"]] == pytest.approx(
        [30.25, 30.5, 31.0]
    )
    assert values[0, controls.sample_acc_global_index] == pytest.approx(0.0)
    assert values[-1, controls.sample_acc_global_index] == pytest.approx(0.1)
    assert derivatives[:, controls.control_name_to_index["CF"]] == pytest.approx(
        [0.1, 0.1, 0.1]
    )
    assert derivatives[:, controls.control_name_to_index["T"]] == pytest.approx(
        [1.0, 1.0, 1.0]
    )


def test_controls_store_rejects_unknown_process(tmp_path):
    prepared_json = _prepare_two_process(tmp_path)
    store = ControlsStore.from_json(prepared_json)

    with pytest.raises(KeyError, match="unknown process name"):
        store.get_controls("missing")

    with pytest.raises(IndexError, match="out of range"):
        store.get_controls(10)


def test_controls_store_rejects_different_control_order(tmp_path):
    prepared_json = _prepare_two_process(tmp_path)
    payload = json.loads(prepared_json.read_text(encoding="utf-8"))
    process_md = payload["metadata"]["bp_train"]["processes"]["p2"]
    process_md["local_control_names"] = ["T", "CF", "V_sample_acc"]
    prepared_json.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(
        ValueError,
        match="identical control names/order across processes",
    ):
        ControlsStore.from_json(prepared_json)


def test_controls_store_eval_clamps_outside_dense_grid(tmp_path):
    prepared_json = _prepare_two_process(tmp_path)
    store = ControlsStore.from_json(prepared_json)
    controls = store.get_controls("p1")

    values = controls.eval(np.asarray([-1.0, 2.0], dtype=float))
    derivatives = controls.eval_derivative(np.asarray([-1.0, 2.0], dtype=float))

    assert values[:, controls.control_name_to_index["CF"]] == pytest.approx([1.0, 1.1])
    assert values[:, controls.control_name_to_index["T"]] == pytest.approx([30.0, 31.0])
    assert derivatives[:, controls.control_name_to_index["CF"]] == pytest.approx(
        [0.1, 0.1]
    )


def test_controls_store_uses_custom_sample_acc_from_prepared_metadata(tmp_path):
    custom_py = tmp_path / "custom-sample.py"
    custom_py.write_text(
        "\n".join(
            [
                "import numpy as np",
                "from bp_train.controls import SignalSource",
                "",
                "CONFIG = {'control_order': ['CF', 'T']}",
                "",
                "def transform_process_collection(collection, config):",
                "    for process in collection.processes.values():",
                "        process.process_variables['CF'].is_controlled = True",
                "        process.process_variables['T'].is_controlled = True",
                "    return collection",
                "",
                "def build_sample_acc_series(process, process_name, collection_metadata, config):",
                "    t0 = float(process.time_axis.start)",
                "    t1 = float(process.time_axis.end)",
                "    times = np.asarray([t0, t1], dtype=float)",
                "    values = np.asarray([0.0, 0.2], dtype=float)",
                "    return SignalSource(",
                "        name='V_sample_acc',",
                "        kind='derived_control',",
                "        times=times,",
                "        values=values,",
                "        evaluator=lambda ts: np.interp(np.asarray(ts, dtype=float), times, values, left=values[0], right=values[-1]),",
                "        derivative=lambda ts: np.full_like(np.asarray(ts, dtype=float), 0.2, dtype=float),",
                "        step_ts=[t0, t1],",
                "        metadata={'source': 'custom_test'},",
                "    )",
            ]
        ),
        encoding="utf-8",
    )
    output = tmp_path / "prepared-custom-sample.json"
    prepare_artifact(_make_two_process_collection(), output, custom_py=custom_py)

    store = ControlsStore.from_json(output)
    controls = store.get_controls("p1")
    end_value = controls.eval(1.0)[controls.sample_acc_global_index]
    assert end_value == pytest.approx(0.2)


def test_controls_store_rejects_not_consistent_controls_at_init(tmp_path):
    prepared_json = _prepare_two_process_inconsistent_controls(tmp_path)
    with pytest.raises(
        ValueError,
        match="identical control names/order across processes",
    ):
        ControlsStore.from_json(prepared_json)
