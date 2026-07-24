from __future__ import annotations

import json
from pathlib import Path

import jax.numpy as jnp
import numpy as np
import pytest
from bp_format.dataclasses import (
    BioProcess,
    BioProcessCollection,
    BioProcessMetadata,
    FeedMedium,
    FeedMediumComponent,
    FeedVolumeChange,
    ProcessVariable,
    ReactorMedium,
    ReactorMediumComponent,
    SampleVolumeChange,
    StaticVariable,
    TimeAxis,
    TimeSeries,
    Volume,
)
from bp_format.serialization import save_process_collection

from bp_train.controls import select_control_sources
from bp_train.controls_store import ControlsStore
from bp_train.prepare import prepare_artifact
from bp_train.run_config import load_prepare_config


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


def _column_index(controls, name: str) -> int:
    """Return the canonical column index of a named control."""
    canonical = (
        controls.name_controlled_FVCs
        + controls.name_controlled_SVCs
        + controls.name_controlled_PVs
    )
    return canonical.index(name)


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
                    )
                },
            ),
            process_variables={
                "CF": ProcessVariable(
                    name="CF",
                    unit="g/L",
                    is_controlled=True,
                    values=TimeSeries(
                        times=jnp.asarray([0.0, 1.0]),
                        values=jnp.asarray([1.0, 1.1]),
                    ),
                ),
                "T": ProcessVariable(
                    name="T",
                    unit="C",
                    is_controlled=True,
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


def _spline_control_values() -> TimeSeries:
    return TimeSeries(
        times=jnp.asarray([0.0, 1.0]),
        values=jnp.asarray([0.0, 1.0]),
        breaks=jnp.asarray([0.0, 1.0]),
        coeffs=jnp.asarray([[0.0, 1.0, 0.0, 0.0]]),
        segment_start_piece_idx=jnp.asarray([0], dtype=jnp.int32),
    )


def test_select_control_sources_consumes_spline_process_variable_control():
    # A spline-backed control is now USED directly (PPoly), not rejected.
    collection = _make_two_process_collection()
    process = collection.processes["p1"]
    process.process_variables["CF"].is_controlled = True
    process.process_variables["CF"].values = _spline_control_values()

    src = select_control_sources(process).sources_by_name["CF"]
    assert src.metadata.get("source") == "spline"
    # linear spline p(dt)=dt over [0, 1] -> value 0.5 at t=0.5
    assert float(src.evaluator(np.asarray([0.5]))[0]) == pytest.approx(0.5, abs=1e-4)


def test_select_control_sources_consumes_spline_feed_control():
    collection = _make_two_process_collection()
    process = collection.processes["p1"]
    process.volume.volume_changes["feed_A"] = FeedVolumeChange(
        name="feed_A",
        unit="L",
        is_controlled=True,
        is_continuous=True,
        values=_spline_control_values(),
        feed_medium=FeedMedium(name="feed", density=1.0, density_unit="kg/L"),
    )

    src = select_control_sources(process).sources_by_name["feed_A"]
    assert src.metadata.get("source") == "spline"
    assert float(src.evaluator(np.asarray([0.5]))[0]) == pytest.approx(0.5, abs=1e-4)


def _write_control_custom_py(path: Path) -> None:
    path.write_text(
        "\n".join(
            [
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
    raw = tmp_path / "raw.json"
    save_process_collection(_make_two_process_collection(), raw)
    config_path = tmp_path / "prepare-config.json"
    config_path.write_text(
        json.dumps({"custom_py": str(custom_py), "prepare": {"raw_input": str(raw)}}),
        encoding="utf-8",
    )
    output_dir = tmp_path / "prepared"
    prepare_artifact(load_prepare_config(config_path), output_dir)
    return output_dir / "prepared.json"


def _prepare_two_process_inconsistent_controls(tmp_path: Path) -> Path:
    custom_py = tmp_path / "custom-inconsistent.py"
    custom_py.write_text(
        "\n".join(
            [
                "def transform_process_collection(collection, config):",
                "    p1 = collection.processes['p1']",
                "    p2 = collection.processes['p2']",
                "    p1.process_variables['T'].is_controlled = False",
                "    p2.process_variables['CF'].is_controlled = False",
                "    return collection",
            ]
        ),
        encoding="utf-8",
    )
    raw = tmp_path / "raw-inconsistent.json"
    save_process_collection(_make_two_process_collection(), raw)
    config_path = tmp_path / "prepare-inconsistent-config.json"
    config_path.write_text(
        json.dumps(
            {
                "custom_py": str(custom_py),
                "prepare": {
                    "raw_input": str(raw),
                    "require_consistent_controls": False,
                },
            }
        ),
        encoding="utf-8",
    )
    output_dir = tmp_path / "prepared-inconsistent"
    prepare_artifact(load_prepare_config(config_path), output_dir)
    return output_dir / "prepared.json"


def test_controls_store_loads_by_process_name_and_index(tmp_path):
    prepared_json = _prepare_two_process(tmp_path)

    store = ControlsStore.from_json(prepared_json)
    by_name = store.get_controls("p1")
    by_index = store.get_controls(0)

    assert store.process_order == ["p1", "p2"]
    assert by_name.process_name == "p1"
    assert by_index.process_name == "p1"
    assert by_name.name_controlled_FVCs == ()
    assert by_name.name_controlled_SVCs == ()
    assert by_name.name_controlled_PVs == ("CF", "T")
    assert np.array_equal(
        np.asarray(by_name.dense_grid), np.asarray(by_index.dense_grid)
    )
    assert _column_index(by_name, "CF") == 0
    assert _column_index(by_name, "T") == 1
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

    # CF and T are the two controlled PVs (this fixture has no controlled FVCs).
    pvs0 = controls.eval_controlled_PVs(0.25, None)
    assert pvs0.shape == (2,)
    assert pvs0[0] == pytest.approx(1.025)  # CF
    assert pvs0[1] == pytest.approx(30.25)  # T

    ts = np.asarray([0.25, 0.5, 1.0], dtype=float)
    pvs = controls.eval_controlled_PVs(ts, None)

    assert pvs.shape == (3, 2)
    assert pvs[:, 0] == pytest.approx([1.025, 1.05, 1.1])  # CF
    assert pvs[:, 1] == pytest.approx([30.25, 30.5, 31.0])  # T
    # No controlled FVCs/SVCs in this fixture → the rate accessors are empty.
    assert controls.eval_controlled_FVCs_rates(ts, None).shape == (3, 0)
    assert controls.eval_controlled_SVCs_rates(ts, None).shape == (3, 0)


def test_controls_store_exposes_discrete_event_metadata():
    feed_medium = FeedMedium(
        name="feed",
        density=1.0,
        density_unit="kg/L",
        components={
            "biomass": FeedMediumComponent(
                name="biomass",
                unit="g/L",
                concentration=StaticVariable(5.0),
                is_controlled=False,
            )
        },
    )
    process = BioProcess(
        metadata=BioProcessMetadata(name="p1", process_type="fed_batch"),
        time_axis=TimeAxis(unit="h", start=0.0, end=5.0, time_reference="start"),
        volume=Volume(
            initial_volume=1.0,
            unit="L",
            volume_changes={
                "sample_a": SampleVolumeChange(
                    name="sample_a",
                    unit="L",
                    is_controlled=False,
                    is_continuous=False,
                    values=TimeSeries(
                        times=jnp.asarray([2.0]),
                        values=jnp.asarray([-0.1]),
                    ),
                ),
                "sample_b": SampleVolumeChange(
                    name="sample_b",
                    unit="L",
                    is_controlled=False,
                    is_continuous=False,
                    values=TimeSeries(
                        times=jnp.asarray([2.0]),
                        values=jnp.asarray([-0.2]),
                    ),
                ),
                "sample_after_end": SampleVolumeChange(
                    name="sample_after_end",
                    unit="L",
                    is_controlled=False,
                    is_continuous=False,
                    values=TimeSeries(
                        times=jnp.asarray([6.0]),
                        values=jnp.asarray([-0.1]),
                    ),
                ),
                "bolus_feed": FeedVolumeChange(
                    name="bolus_feed",
                    unit="L",
                    is_controlled=True,
                    is_continuous=False,
                    values=TimeSeries(
                        times=jnp.asarray([3.0]),
                        values=jnp.asarray([0.4]),
                    ),
                    feed_medium=feed_medium,
                ),
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
                    concentration=StaticVariable(1.0),
                )
            },
        ),
        process_variables={},
    )
    store = ControlsStore.from_collection(
        BioProcessCollection(processes={"p1": process})
    )
    controls = store.get_controls("p1")

    assert np.asarray(controls.sample_event_mask).tolist() == [True]
    assert np.asarray(controls.sample_event_times).tolist() == pytest.approx([2.0])
    assert np.asarray(controls.sample_event_volumes).tolist() == pytest.approx([0.3])
    assert np.asarray(controls.bolus_event_mask).tolist() == [True]
    assert np.asarray(controls.bolus_event_times).tolist() == pytest.approx([3.0])
    assert np.asarray(controls.bolus_event_volumes).tolist() == pytest.approx([0.4])
    assert np.asarray(controls.bolus_event_Cin).shape == (1, 1)
    assert float(controls.bolus_event_Cin[0, 0]) == pytest.approx(5.0)
    # jump_ts holds genuine vector-field discontinuity times from
    # ``discrete_events`` (None for this process) — NOT the bolus/sample
    # STATE-jump events, which live in the ``*_event_*`` arrays above and are
    # applied by the callbacks solve. With no discrete_events, jump_ts is empty.
    assert np.asarray(controls.active_jump_ts, dtype=float).tolist() == []
    gathered_controls = store.gather_batch(jnp.asarray([0, 0], dtype=jnp.int32))
    assert gathered_controls.control_values.shape == (
        2,
        store.shape_metadata["max_grid_length"],
        0,
    )
    assert gathered_controls.bolus_event_Cin.shape == (2, 1, 1)
    empty_pvs = gathered_controls.eval_controlled_PVs(0, jnp.asarray([0.0, 5.0]), None)
    assert empty_pvs.shape == (2, 0)


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
    process_md = payload["metadata"]["bp-train"]["processes"]["p2"]
    process_md["name_controlled_PVs"] = ["T", "CF"]
    prepared_json.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(
        ValueError,
        match="prepared metadata name_controlled_PVs",
    ):
        ControlsStore.from_json(prepared_json)


def test_controls_store_eval_clamps_outside_dense_grid(tmp_path):
    prepared_json = _prepare_two_process(tmp_path)
    store = ControlsStore.from_json(prepared_json)
    controls = store.get_controls("p1")

    ts = np.asarray([-1.0, 2.0], dtype=float)
    pvs = controls.eval_controlled_PVs(ts, None)

    assert pvs[:, 0] == pytest.approx([1.0, 1.1])  # CF clamped to grid ends
    assert pvs[:, 1] == pytest.approx([30.0, 31.0])  # T clamped to grid ends


def test_controls_store_rejects_not_consistent_controls_at_init():
    """ControlsStore must reject collections whose processes disagree on
    categorised control layouts."""
    p1 = _make_two_process_collection().processes["p1"]
    p2_collection = _make_two_process_collection()
    p2 = p2_collection.processes["p2"]
    # Add a controlled FVC to p1 only — disagrees with p2 on name_controlled_FVCs.
    p1.volume.volume_changes["feed_A"] = FeedVolumeChange(
        name="feed_A",
        unit="L",
        is_controlled=True,
        is_continuous=True,
        values=TimeSeries(
            times=jnp.asarray([0.0, 1.0]),
            values=jnp.asarray([0.0, 0.1]),
        ),
        feed_medium=FeedMedium(
            name="feed",
            density=1.0,
            density_unit="kg/L",
            components={
                "biomass": FeedMediumComponent(
                    name="biomass",
                    unit="g/L",
                    concentration=StaticVariable(0.0),
                    is_controlled=False,
                ),
            },
        ),
    )
    collection = BioProcessCollection(processes={"p1": p1, "p2": p2})

    with pytest.raises(
        ValueError,
        match="identical categorised control layouts across processes",
    ):
        ControlsStore.from_collection(collection)


def test_per_process_controls_roundtrip_across_processes(tmp_path):
    """Cross-process eqx serialise/deserialise must round-trip.

    LOO-CV saves the trained wrapper after training on N-1 processes (its
    ``controls`` field carries the first *training* process's metadata),
    then evaluates the holdout fold by deserialising into a template
    built from the full collection (whose first process — and therefore
    the template's controls — is typically the held-out process). If
    ``PerProcessControls`` exposes per-process metadata as dynamic pytree
    leaves, the leaf order shifts with each process and
    ``eqx.tree_deserialise_leaves`` reads the wrong bytes into the wrong
    slots — the failure surfaces as a ``TreePathError``.
    """
    import equinox as eqx

    prepared_json = _prepare_two_process(tmp_path)
    store = ControlsStore.from_json(prepared_json)
    saved = store.get_controls("p1")
    template = store.get_controls("p2")

    saved_path = tmp_path / "controls.eqx"
    eqx.tree_serialise_leaves(saved_path, saved)
    loaded = eqx.tree_deserialise_leaves(saved_path, like=template)

    # Static metadata follows the template (controls are swapped per-process
    # at evaluation time via eqx.tree_at, so the loaded wrapper's stored
    # static fields are placeholders).
    assert loaded.process_name == template.process_name
    assert loaded.process_index == template.process_index
    # Dynamic arrays carry the saved process's values.
    assert np.array_equal(np.asarray(loaded.dense_grid), np.asarray(saved.dense_grid))
    assert np.array_equal(
        np.asarray(loaded.control_values), np.asarray(saved.control_values)
    )


def test_controls_store_pads_control_rows_with_last_active_values():
    grid, values, derivatives, _, grid_length, _ = ControlsStore._pad_dense_payload(
        payload={
            "grid": [0.0, 1.0],
            "values": [[1.0], [2.0]],
            "derivatives": [[3.0], [4.0]],
            "jump_ts": [],
        },
        max_grid_length=4,
        max_jump_ts_length=0,
    )

    assert grid_length == 2
    assert grid == [0.0, 1.0, 1.0, 1.0]
    assert values == [[1.0], [2.0], [2.0], [2.0]]
    assert derivatives == [[3.0], [4.0], [4.0], [4.0]]


def test_controls_store_gather_batch_preserves_order_duplicates_and_events():
    collection = _make_two_process_collection()
    p2 = collection.processes["p2"]
    p2.process_variables["CF"].values = TimeSeries(
        times=jnp.asarray([0.0, 1.0]),
        values=jnp.asarray([2.0, 2.2]),
    )
    p2.volume.volume_changes["sample_1"].values = TimeSeries(
        times=jnp.asarray([0.75]),
        values=jnp.asarray([-0.2]),
    )
    for index, process in enumerate(collection.processes.values(), start=1):
        process.volume.volume_changes["bolus"] = FeedVolumeChange(
            name="bolus",
            unit="L",
            is_controlled=True,
            is_continuous=False,
            values=TimeSeries(
                times=jnp.asarray([0.2 * index]),
                values=jnp.asarray([0.1 * index]),
            ),
            feed_medium=FeedMedium(
                name="feed",
                density=1.0,
                density_unit="kg/L",
                components={
                    "biomass": FeedMediumComponent(
                        name="biomass",
                        unit="g/L",
                        concentration=StaticVariable(3.0 * index),
                        is_controlled=False,
                    )
                },
            ),
        )

    store = ControlsStore.from_collection(collection)
    indices = jnp.asarray([1, 0, 1], dtype=jnp.int32)
    gathered = store.gather_batch(indices)

    for field in (
        "spline_breaks",
        "spline_coeffs",
        "dense_grid",
        "control_values",
        "control_derivatives",
        "sample_event_times",
        "sample_event_volumes",
        "sample_event_mask",
        "bolus_event_times",
        "bolus_event_volumes",
        "bolus_event_Cin",
        "bolus_event_mask",
    ):
        np.testing.assert_array_equal(
            np.asarray(getattr(gathered, field)),
            np.asarray(getattr(store, field))[[1, 0, 1]],
        )

    ts = jnp.asarray([0.25, 1.0])
    np.testing.assert_allclose(
        gathered.eval_controlled_PVs(0, ts, None),
        store.get_controls(1).eval_controlled_PVs(ts, None),
    )


def test_controls_store_batch_controls_eval_by_index():
    collection = _make_two_process_collection()
    collection.processes["p2"].process_variables["CF"].values = TimeSeries(
        times=jnp.asarray([0.0, 0.3, 1.0]),
        values=jnp.asarray([1.0, 1.03, 1.1]),
    )
    store = ControlsStore.from_collection(collection)
    batch_controls = store.gather_batch(jnp.asarray([0, 1], dtype=jnp.int32))

    assert store.grid_lengths.tolist() == [16, 17]

    ts = jnp.asarray([0.25, 1.5])
    pvs = batch_controls.eval_controlled_PVs(0, ts, None)
    per_process_pvs = store.get_controls("p1").eval_controlled_PVs(ts, None)
    assert pvs == pytest.approx(per_process_pvs)
    assert pvs[:, 0] == pytest.approx([1.025, 1.1])
    assert pvs[:, 1] == pytest.approx([30.25, 31.0])


def test_controls_store_batch_controls_rejects_out_of_range_row(tmp_path):
    prepared_json = _prepare_two_process(tmp_path)
    store = ControlsStore.from_json(prepared_json)
    batch_controls = store.gather_batch(jnp.asarray([0, 1], dtype=jnp.int32))

    with pytest.raises(IndexError, match="out of range"):
        batch_controls.eval_controlled_FVCs_cumulative(2, jnp.asarray(0.25), None)
    with pytest.raises(IndexError, match="out of range"):
        batch_controls.eval_controlled_FVCs_cumulative(999, jnp.asarray(0.25), None)
