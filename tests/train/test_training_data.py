from __future__ import annotations

import json
from pathlib import Path

import jax.numpy as jnp
import numpy as np
import pytest
from hybrax.format.dataclasses import (
    BioProcess,
    BioProcessCollection,
    BioProcessMetadata,
    ProcessVariable,
    ReactorMedium,
    ReactorMediumComponent,
    Outflow,
    StaticVariable,
    TimeAxis,
    TimeSeries,
    Volume,
)
from hybrax.format.serialization import save_process_collection

from hybrax.train.prepare import prepare_artifact
from hybrax.train.run_config import load_prepare_config
from hybrax.train.training_data import TrainingDataStore


def _prepare_from_collection(
    collection: BioProcessCollection,
    tmp_path: Path,
    output_dir: Path,
    *,
    custom_py: Path | None = None,
) -> None:
    raw_json = tmp_path / f"{output_dir.name}-raw.json"
    config_json = tmp_path / f"{output_dir.name}-config.json"
    save_process_collection(collection, raw_json)
    config: dict[str, object] = {"prepare": {"raw_input": str(raw_json)}}
    if custom_py is not None:
        config["custom_py"] = str(custom_py)
    config_json.write_text(json.dumps(config), encoding="utf-8")
    prepare_artifact(load_prepare_config(config_json), output_dir)


def _make_two_process_collection() -> BioProcessCollection:
    p1 = BioProcess(
        metadata=BioProcessMetadata(name="p1", process_type="fed_batch"),
        time_axis=TimeAxis(unit="h", start=0.0, end=4.0, time_reference="start"),
        volume=Volume(
            initial_volume=1.0,
            unit="L",
            volume_changes={
                "sample_1": Outflow(
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
                "sample_1": Outflow(
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
                "",
                "def transform_process_collection(collection, config):",
                "    for process in collection.processes.values():",
                "        process.process_variables['CF'].is_controlled = True",
                "        process.process_variables['T'].is_controlled = True",
                "        process.reaction_ode = None",
                "        process.__post_init__()",
                "    return collection",
            ]
        ),
        encoding="utf-8",
    )


def _prepare_collection(tmp_path: Path) -> Path:
    custom_py = tmp_path / "custom.py"
    _write_custom(custom_py)
    output_dir = tmp_path / "prepared"
    _prepare_from_collection(
        _make_two_process_collection(), tmp_path, output_dir, custom_py=custom_py
    )
    return output_dir / "prepared.json"


def _make_reactor_target_collection() -> BioProcessCollection:
    p1 = BioProcess(
        metadata=BioProcessMetadata(name="p1", process_type="fed_batch"),
        time_axis=TimeAxis(unit="h", start=0.0, end=4.0, time_reference="start"),
        volume=Volume(
            initial_volume=1.0,
            unit="L",
            volume_changes={
                "sample_1": Outflow(
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
                ),
                "product": ReactorMediumComponent(
                    name="product",
                    unit="g/L",
                    concentration=TimeSeries(
                        times=jnp.asarray([0.0, 2.0, 4.0]),
                        values=jnp.asarray([0.0, 0.1, 0.2]),
                    ),
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
                "sample_1": Outflow(
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
                ),
                "product": ReactorMediumComponent(
                    name="product",
                    unit="g/L",
                    concentration=TimeSeries(
                        times=jnp.asarray([0.0, 1.0]),
                        values=jnp.asarray([0.02, 0.09]),
                    ),
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
    # X is a process variable (test fixture sets it), so name_measured_PVs
    # carries the names and name_measured_RMCs is empty.
    assert store.name_measured_RMCs == ()
    assert store.name_measured_PVs == ("X",)
    assert store.name_measured == ("X",)
    assert tuple(store.t_measured.shape) == (2, 3)
    # y_measured has n_targets + n_modeled_flows columns. The fixture has no
    # Inflow so n_modeled_flows == 0 and y_measured has just 1 column.
    assert tuple(store.y_measured.shape) == (2, 3, 1)
    # Per-cell mask: (n_processes, max_n_meas, n_y_cols).
    assert tuple(store.mask_measured.shape) == (2, 3, 1)
    # y0 is the full physical state: [biomass, X, V].
    assert tuple(store.y0_measured.shape) == (2, 3)

    assert np.asarray(store.t_measured[0]).tolist() == pytest.approx([0.0, 2.0, 4.0])
    p2_t = np.asarray(store.t_measured[1])
    assert p2_t[:2].tolist() == pytest.approx([0.0, 1.0])
    assert p2_t[2] == pytest.approx(0.0)
    assert np.asarray(store.y_measured[1, 2, 0]) == pytest.approx(0.0)
    # Each row contributes one (timestamp, target) pair; the third row of
    # process p2 is padding and so its single column entry is False.
    assert np.asarray(store.mask_measured[0, :, 0]).tolist() == [True, True, True]
    assert np.asarray(store.mask_measured[1, :, 0]).tolist() == [True, True, False]


def test_training_data_store_exposes_per_process_active_views(tmp_path):
    prepared_json = _prepare_collection(tmp_path)
    store = TrainingDataStore.from_json(prepared_json, target_variable_order=["X"])
    process_data = store.get_process("p2")

    assert process_data.process_name == "p2"
    assert process_data.n_measured == 2
    assert process_data.name_measured == ("X",)
    assert np.asarray(process_data.active_t_measured).tolist() == pytest.approx(
        [0.0, 1.0]
    )
    assert np.asarray(process_data.active_y_measured[:, 0]).tolist() == pytest.approx(
        [0.25, 0.35]
    )
    assert np.asarray(process_data.active_mask_measured).tolist() == [[True], [True]]


def test_training_data_store_builds_y0_with_vcont_last(tmp_path):
    prepared_json = _prepare_collection(tmp_path)
    store = TrainingDataStore.from_json(prepared_json, target_variable_order=["X"])

    assert np.asarray(store.y0_measured[0]).tolist() == pytest.approx([0.1, 0.2, 1.0])
    assert np.asarray(store.y0_measured[1]).tolist() == pytest.approx([0.1, 0.25, 1.5])


def test_prepare_rejects_inconsistent_process_variable_set(tmp_path):
    """A process-variable mismatch between processes is now caught by
    prepare_artifact's cross-process consistency check (hybrax.format's
    validate_cross_process_consistency, wired into validate_for_training)
    before TrainingDataStore is ever constructed — earlier and with more
    detail than TrainingDataStore's own narrower target-only check."""
    collection = _make_two_process_collection()
    del collection.processes["p2"].process_variables["X"]

    custom_py = tmp_path / "custom.py"
    _write_custom(custom_py)
    output_dir = tmp_path / "prepared"

    with (
        pytest.warns(UserWarning, match="non-OK status"),
        pytest.raises(ValueError, match="process variables differ"),
    ):
        _prepare_from_collection(collection, tmp_path, output_dir, custom_py=custom_py)


def test_training_data_store_rejects_inconsistent_target_order(tmp_path):
    collection = _make_two_process_collection()
    p2 = collection.processes["p2"]
    x = p2.process_variables.pop("X")
    p2.process_variables["X"] = x

    custom_py = tmp_path / "custom.py"
    custom_py.write_text(
        "\n".join(
            [
                "",
                "def transform_process_collection(collection, config):",
                "    for process in collection.processes.values():",
                "        process.process_variables['CF'].is_controlled = True",
                "        process.reaction_ode = None",
                "        process.__post_init__()",
                "    return collection",
            ]
        ),
        encoding="utf-8",
    )
    output_dir = tmp_path / "prepared"
    _prepare_from_collection(collection, tmp_path, output_dir, custom_py=custom_py)

    with pytest.raises(
        ValueError,
        match="identical measured target names/order across processes",
    ):
        TrainingDataStore.from_json(output_dir / "prepared.json")


def test_training_data_store_rejects_controlled_target_in_configured_order(tmp_path):
    prepared_json = _prepare_collection(tmp_path)
    with pytest.raises(
        ValueError,
        match="must be measured .* controlled targets",
    ):
        TrainingDataStore.from_json(prepared_json, target_variable_order=["CF"])


def test_training_data_errors_use_collection_key_without_process_metadata():
    collection = _make_two_process_collection()
    collection.processes["p1"].metadata = None

    with pytest.raises(ValueError, match="p1: configured target variables missing"):
        TrainingDataStore.from_collection(
            collection,
            target_variable_order=["missing"],
            target_source="process_variables",
        )


def test_training_data_store_supports_reactor_component_targets():
    store = TrainingDataStore.from_collection(
        _make_reactor_target_collection(),
        target_variable_order=["biomass", "product"],
        target_source="reactor_components",
    )

    # Reactor-component targets land in name_measured_RMCs.
    assert store.name_measured_RMCs == ("biomass", "product")
    assert store.name_measured_PVs == ()
    # y_measured has n_targets + n_modeled_flows columns. The fixture has no
    # Inflow so n_modeled_flows == 0.
    assert tuple(store.y_measured.shape) == (2, 3, 2)
    # y0_measured = [biomass(0), product(0), V_cont(0)]
    assert np.asarray(store.y0_measured[0]).tolist() == pytest.approx([0.2, 0.0, 1.0])
    assert np.asarray(store.y0_measured[1]).tolist() == pytest.approx([0.25, 0.02, 1.2])


def test_control_support_validation_only_checks_selected_processes():
    collection = _make_reactor_target_collection()
    collection.processes["p2"].process_variables["temperature"].values = TimeSeries(
        times=jnp.asarray([0.0, 0.5]),
        values=jnp.asarray([299.0, 300.0]),
    )
    store = TrainingDataStore.from_collection(
        collection,
        target_variable_order=["biomass", "product"],
        target_source="reactor_components",
    )

    store.validate_control_support(("p1",))
    with pytest.raises(ValueError, match=r"process='p2'.*control='temperature'"):
        store.validate_control_support(("p2",))


def test_control_support_validation_aggregates_process_failures():
    collection = _make_reactor_target_collection()
    for process in collection.processes.values():
        process.process_variables["temperature"].values = TimeSeries(
            times=jnp.asarray([0.0, 0.5]),
            values=jnp.asarray([299.0, 300.0]),
        )
    store = TrainingDataStore.from_collection(
        collection,
        target_variable_order=["biomass", "product"],
        target_source="reactor_components",
    )

    with pytest.raises(ValueError) as exc_info:
        store.validate_control_support(("p1", "p2"))

    message = str(exc_info.value)
    assert "process='p1'" in message
    assert "process='p2'" in message
    assert "control='temperature'" in message
    assert "representation='raw'" in message
    assert "violated_side='right'" in message


def test_training_data_store_auto_falls_back_to_reactor_components():
    store = TrainingDataStore.from_collection(
        _make_reactor_target_collection(),
        target_variable_order=["biomass"],
        target_source="auto",
    )
    assert store.name_measured_RMCs == ("biomass",)
    assert store.name_measured_PVs == ()


def test_training_data_store_auto_prefers_process_variables_when_available(tmp_path):
    prepared_json = _prepare_collection(tmp_path)
    store = TrainingDataStore.from_json(
        prepared_json,
        target_source="auto",
    )
    assert store.name_measured_PVs == ("X",)
    assert store.name_measured_RMCs == ()


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
    assert tuple(batch.t_measured.shape) == (3, 3)
    # y_measured has n_targets + n_modeled_flows = 1 columns (no Inflow)
    assert tuple(batch.y_measured.shape) == (3, 3, 1)
    assert tuple(batch.mask_measured.shape) == (3, 3, 1)
    assert np.asarray(batch.n_measured).tolist() == [2, 3, 2]
    assert np.asarray(batch.y0_measured[0]).tolist() == pytest.approx([0.1, 0.25, 1.5])
    assert np.asarray(batch.y0_measured[1]).tolist() == pytest.approx([0.1, 0.2, 1.0])


def test_training_data_store_gather_batch_rejects_invalid_indices(tmp_path):
    prepared_json = _prepare_collection(tmp_path)
    store = TrainingDataStore.from_json(prepared_json, target_variable_order=["X"])

    with pytest.raises(ValueError, match="1D"):
        store.gather_batch(jnp.asarray([[0, 1]], dtype=jnp.int32))
    with pytest.raises(ValueError, match="non-empty"):
        store.gather_batch(jnp.asarray([], dtype=jnp.int32))
    with pytest.raises(IndexError, match="out of range"):
        store.gather_batch(jnp.asarray([0, 2], dtype=jnp.int32))


def _make_sparse_target_collection() -> BioProcessCollection:
    """Two RMC targets with different measurement grids but a shared t=0."""
    proc = BioProcess(
        metadata=BioProcessMetadata(name="p1", process_type="fed_batch"),
        time_axis=TimeAxis(unit="h", start=0.0, end=4.0, time_reference="start"),
        volume=Volume(
            initial_volume=1.0,
            unit="L",
            volume_changes={
                "sample_1": Outflow(
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
                        # Dense: 4 timepoints
                        times=jnp.asarray([0.0, 1.0, 2.0, 4.0]),
                        values=jnp.asarray([0.2, 0.25, 0.3, 0.4]),
                    ),
                ),
                "product": ReactorMediumComponent(
                    name="product",
                    unit="g/L",
                    concentration=TimeSeries(
                        # Sparse: only at t=0 and t=4 (no measurement at 1, 2)
                        times=jnp.asarray([0.0, 4.0]),
                        values=jnp.asarray([0.0, 0.5]),
                    ),
                ),
            },
        ),
        process_variables={},
    )
    return BioProcessCollection(processes={"p1": proc})


def test_training_data_store_builds_union_grid_with_per_cell_mask():
    """Sparse per-target grids merge into a union grid; mask reflects which
    cells are real vs synthetic."""
    store = TrainingDataStore.from_collection(
        _make_sparse_target_collection(),
        target_variable_order=["biomass", "product"],
        target_source="reactor_components",
    )

    # Union of biomass times {0,1,2,4} and product times {0,4} = {0,1,2,4}.
    assert tuple(store.t_measured.shape) == (1, 4)
    assert np.asarray(store.t_measured[0]).tolist() == pytest.approx(
        [0.0, 1.0, 2.0, 4.0]
    )

    # 2 RMC targets, no modeled feeds → 2 columns.
    assert tuple(store.y_measured.shape) == (1, 4, 2)
    assert tuple(store.mask_measured.shape) == (1, 4, 2)

    # biomass column: every cell is a real measurement.
    assert np.asarray(store.mask_measured[0, :, 0]).tolist() == [True, True, True, True]
    # product column: only t=0 and t=4 are real.
    assert np.asarray(store.mask_measured[0, :, 1]).tolist() == [
        True,
        False,
        False,
        True,
    ]

    # Real biomass values match input.
    assert np.asarray(store.y_measured[0, :, 0]).tolist() == pytest.approx(
        [0.2, 0.25, 0.3, 0.4]
    )
    # Real product values land at the right grid rows; synthetic rows are 0.0.
    assert np.asarray(store.y_measured[0, :, 1]).tolist() == pytest.approx(
        [0.0, 0.0, 0.0, 0.5]
    )

    # y0 reads y_matrix[0, :n_targets] — both targets must be measured at t=0
    # (verified by the strict t[0] check at build time).
    assert np.asarray(store.y0_measured[0]).tolist() == pytest.approx([0.2, 0.0, 1.0])


def test_training_data_store_rejects_target_missing_t0():
    """If a target has no measurement at the union grid's first time, building
    the store must raise rather than silently zero-init y0."""
    proc = BioProcess(
        metadata=BioProcessMetadata(name="p1", process_type="fed_batch"),
        time_axis=TimeAxis(unit="h", start=0.0, end=4.0, time_reference="start"),
        volume=Volume(
            initial_volume=1.0,
            unit="L",
            volume_changes={
                "sample_1": Outflow(
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
                        times=jnp.asarray([0.0, 4.0]),
                        values=jnp.asarray([0.2, 0.4]),
                    ),
                ),
                # product first measured at t=2.0 — missing the union t=0.
                "product": ReactorMediumComponent(
                    name="product",
                    unit="g/L",
                    concentration=TimeSeries(
                        times=jnp.asarray([2.0, 4.0]),
                        values=jnp.asarray([0.05, 0.1]),
                    ),
                ),
            },
        ),
        process_variables={},
    )
    collection = BioProcessCollection(processes={"p1": proc})

    with pytest.raises(ValueError, match="no measurement at union_grid t\\[0\\]"):
        TrainingDataStore.from_collection(
            collection,
            target_variable_order=["biomass", "product"],
            target_source="reactor_components",
        )


def _make_combined_collection() -> BioProcessCollection:
    """biomass RMC + one uncontrolled (modeled) PV, both with time series."""

    def _proc(name: str, v0: float) -> BioProcess:
        return BioProcess(
            metadata=BioProcessMetadata(name=name, process_type="batch"),
            time_axis=TimeAxis(unit="h", start=0.0, end=4.0, time_reference="start"),
            volume=Volume(initial_volume=v0, unit="L", volume_changes={}),
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
                    ),
                },
            ),
            process_variables={
                "ratio": ProcessVariable(
                    name="ratio",
                    unit="-",
                    is_controlled=False,
                    values=TimeSeries(
                        times=jnp.asarray([0.0, 2.0, 4.0]),
                        values=jnp.asarray([0.5, 0.55, 0.6]),
                    ),
                ),
            },
        )

    return BioProcessCollection(
        processes={"p1": _proc("p1", 1.0), "p2": _proc("p2", 1.2)}
    )


def test_selected_rows_keep_modeled_outflow_cumulative_state_zero():
    collection = _make_combined_collection()
    for process in collection.processes.values():
        process.volume.volume_changes["modeled_bleed"] = Outflow(
            name="modeled_bleed",
            unit="L",
            is_controlled=False,
            is_continuous=True,
            values=TimeSeries(times=[0.0, 4.0], values=[-0.3, -0.7]),
            retention={"biomass": 0.5},
        )

    store = TrainingDataStore.from_collection(
        collection,
        target_source="combined",
    ).select_processes(
        ("p2", "p1"),
        BioProcessCollection(
            processes={
                "p2": collection.processes["p2"],
                "p1": collection.processes["p1"],
            }
        ),
    )

    # Modeled cumulative states are integration states and always start at zero,
    # even when the source cumulative trace has a nonzero selected-row offset.
    np.testing.assert_array_equal(np.asarray(store.y0_measured[:, -1]), [0.0, 0.0])


def test_combined_target_error_uses_key_without_process_metadata():
    collection = _make_combined_collection()
    process = collection.processes["p1"]
    process.metadata = None
    process.process_variables["ratio"].values = TimeSeries(
        times=jnp.asarray([]), values=jnp.asarray([])
    )

    with pytest.raises(
        ValueError,
        match="p1: target_source='combined' requires every modeled PV",
    ):
        TrainingDataStore.from_collection(collection, target_source="combined")


def test_training_data_combined_fits_rmcs_and_pvs(tmp_path):
    output_dir = tmp_path / "prepared"
    _prepare_from_collection(_make_combined_collection(), tmp_path, output_dir)

    store = TrainingDataStore.from_json(
        output_dir / "prepared.json", target_source="combined"
    )

    # Both source families populate their tuple; name_measured is the ordered
    # [RMCs | PVs] leading state block.
    assert store.name_measured_RMCs == ("biomass",)
    assert store.name_measured_PVs == ("ratio",)
    assert store.name_measured == ("biomass", "ratio")

    # y0 leading block = [biomass(0), ratio(0)] followed by V(0).
    y0 = np.asarray(store.y0_measured[0])
    assert y0[0] == pytest.approx(0.2)  # biomass(0)
    assert y0[1] == pytest.approx(0.5)  # ratio(0)
    assert y0[2] == pytest.approx(1.0)  # V(0)

    # y_measured carries the PV column in the leading block (col index 1).
    y_meas = np.asarray(store.y_measured[0])  # [n_meas, n_y_cols]
    mask = np.asarray(store.mask_measured[0])
    pv_col = y_meas[mask[:, 1], 1]
    assert np.allclose(np.sort(pv_col), [0.5, 0.55, 0.6])
