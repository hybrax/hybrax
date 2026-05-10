from __future__ import annotations

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
from bp_format.serialization import load_process_collection_json

from bp_train.controls import (
    BOLUS_MIN_DT_DURATION_DENOMINATOR,
    EVENT_RUN_MIN_DT_CONFIG_KEY,
    build_bolus_sources,
    select_control_sources,
)
from bp_train.controls_store import ControlsStore
from bp_train.prepare import load_raw_collection, prepare_artifact

INPUT_JSON = Path(__file__).resolve().parent.parent / "input.json"


def _make_feed_collection() -> BioProcessCollection:
    feed_medium = FeedMedium(
        name="feed",
        density=1.0,
        density_unit="kg/L",
        components={
            "glucose": FeedMediumComponent(
                name="glucose",
                unit="g/L",
                concentration=TimeSeries(
                    times=jnp.asarray([0.0, 1.0]),
                    values=jnp.asarray([20.0, 20.0]),
                ),
                is_controlled=False,
            )
        },
    )
    process = BioProcess(
        metadata=BioProcessMetadata(name="p1", process_type="fed_batch"),
        time_axis=TimeAxis(unit="h", start=0.0, end=1.0, time_reference="start"),
        volume=Volume(
            initial_volume=1.0,
            unit="L",
            volume_changes={
                "feed_A": FeedVolumeChange(
                    name="feed_A",
                    unit="L/h",
                    is_controlled=True,
                    is_continuous=True,
                    values=TimeSeries(
                        times=jnp.asarray([0.0, 1.0]),
                        values=jnp.asarray([0.1, 0.1]),
                    ),
                    feed_medium=feed_medium,
                )
            },
        ),
        reactor_medium=ReactorMedium(name="rm", density=1.0, density_unit="kg/L"),
        process_variables={
            "X": ProcessVariable(
                name="X",
                unit="g/L",
                is_controlled=False,
                values=TimeSeries(
                    times=jnp.asarray([0.0, 1.0]),
                    values=jnp.asarray([1.0, 1.2]),
                ),
            )
        },
    )
    return BioProcessCollection(
        metadata={"case_study": {"case_id": "synthetic"}}, processes={"p1": process}
    )


def _write_sample_semantics_custom_py(path: Path) -> None:
    path.write_text(
        "\n".join(
            [
                "from bp_format.dataclasses import ReactorMediumComponent, TimeSeries",
                "import jax.numpy as jnp",
                "",
                "def transform_process_collection(collection, config):",
                "    process = next(iter(collection.processes.values()))",
                "    process.reactor_medium.components['biomass'] = "
                "ReactorMediumComponent(",
                "        name='biomass',",
                "        unit='g/L',",
                "        concentration=TimeSeries(",
                "            times=jnp.asarray([0.0, 1.0]),",
                "            values=jnp.asarray([0.1, 0.2]),",
                "        ),",
                "    )",
                "    return collection",
            ]
        ),
        encoding="utf-8",
    )


def _write_feed_semantics_custom_py(path: Path) -> None:
    path.write_text(
        "\n".join(
            [
                "from bp_format.dataclasses import ("
                "FeedMediumComponent, ReactorMediumComponent, "
                "StaticVariable, TimeSeries)",
                "import jax.numpy as jnp",
                "",
                "def transform_process_collection(collection, config):",
                "    process = next(iter(collection.processes.values()))",
                "    process.reactor_medium.components['biomass'] = "
                "ReactorMediumComponent(",
                "        name='biomass',",
                "        unit='g/L',",
                "        concentration=StaticVariable(0.1),",
                "    )",
                "    process.reactor_medium.components['glucose'] = "
                "ReactorMediumComponent(",
                "        name='glucose',",
                "        unit='g/L',",
                "        concentration=TimeSeries(",
                "            times=jnp.asarray([0.0, 1.0]),",
                "            values=jnp.asarray([1.0, 1.2]),",
                "        ),",
                "    )",
                "    process.volume.volume_changes['feed_A']"
                ".feed_medium.components['biomass'] = FeedMediumComponent(",
                "        name='biomass',",
                "        unit='g/L',",
                "        concentration=StaticVariable(0.0),",
                "        is_controlled=False,",
                "    )",
                "    return collection",
            ]
        ),
        encoding="utf-8",
    )


def _write_feed_semantics_incomplete_custom_py(path: Path) -> None:
    path.write_text(
        "\n".join(
            [
                "from bp_format.dataclasses import ("
                "FeedMediumComponent, ReactorMediumComponent, "
                "StaticVariable, TimeSeries)",
                "import jax.numpy as jnp",
                "",
                "def transform_process_collection(collection, config):",
                "    process = next(iter(collection.processes.values()))",
                "    process.reactor_medium.components['biomass'] = "
                "ReactorMediumComponent(",
                "        name='biomass',",
                "        unit='g/L',",
                "        concentration=StaticVariable(0.1),",
                "    )",
                "    process.reactor_medium.components['glucose'] = "
                "ReactorMediumComponent(",
                "        name='glucose',",
                "        unit='g/L',",
                "        concentration=TimeSeries(",
                "            times=jnp.asarray([0.0, 1.0]),",
                "            values=jnp.asarray([1.0, 1.2]),",
                "        ),",
                "    )",
                "    process.volume.volume_changes['feed_A']"
                ".feed_medium.components = {}",
                "    process.volume.volume_changes['feed_A']"
                ".feed_medium.components['biomass'] = FeedMediumComponent(",
                "        name='biomass',",
                "        unit='g/L',",
                "        concentration=StaticVariable(0.0),",
                "        is_controlled=False,",
                "    )",
                "    return collection",
            ]
        ),
        encoding="utf-8",
    )


def _make_invalid_collection() -> BioProcessCollection:
    process = BioProcess(
        metadata=BioProcessMetadata(name="invalid", process_type="fed_batch"),
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
        reactor_medium=ReactorMedium(name="rm", density=1.0, density_unit="kg/L"),
        process_variables={
            "X": ProcessVariable(
                name="X",
                unit="g/L",
                is_controlled=False,
                values=TimeSeries(
                    times=jnp.asarray([0.0, 1.0]),
                    values=jnp.asarray([1.0, 1.2]),
                ),
            )
        },
    )
    return BioProcessCollection(
        metadata={"case_study": {"case_id": "invalid"}}, processes={"invalid": process}
    )


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


def _make_bolus_collection() -> BioProcessCollection:
    feed_medium = FeedMedium(
        name="feed",
        density=1.0,
        density_unit="kg/L",
        components={
            "X": FeedMediumComponent(
                name="X",
                unit="g/L",
                concentration=StaticVariable(0.0),
                is_controlled=False,
            ),
            "biomass": FeedMediumComponent(
                name="biomass",
                unit="g/L",
                concentration=StaticVariable(0.0),
                is_controlled=False,
            ),
        },
    )
    process = BioProcess(
        metadata=BioProcessMetadata(name="bolus", process_type="fed_batch"),
        time_axis=TimeAxis(unit="h", start=0.0, end=10.0, time_reference="start"),
        volume=Volume(
            initial_volume=1.0,
            unit="L",
            volume_changes={
                "feed_bolus": FeedVolumeChange(
                    name="feed_bolus",
                    unit="L",
                    is_controlled=True,
                    is_continuous=False,
                    values=TimeSeries(
                        times=jnp.asarray([5.0]),
                        values=jnp.asarray([1.0]),
                    ),
                    feed_medium=feed_medium,
                )
            },
        ),
        reactor_medium=ReactorMedium(
            name="rm",
            density=1.0,
            density_unit="kg/L",
            components={
                "X": ReactorMediumComponent(
                    name="X",
                    unit="g/L",
                    concentration=StaticVariable(0.0),
                ),
                "biomass": ReactorMediumComponent(
                    name="biomass",
                    unit="g/L",
                    concentration=TimeSeries(
                        times=jnp.asarray([0.0, 5.0, 10.0]),
                        values=jnp.asarray([0.1, 0.5, 1.0]),
                    ),
                ),
            },
        ),
        process_variables={
            "X": ProcessVariable(
                name="X",
                unit="g/L",
                is_controlled=False,
                values=TimeSeries(
                    times=jnp.asarray([0.0, 2.5, 5.0, 7.5, 10.0]),
                    values=jnp.asarray([1.0, 1.0, 1.0, 1.0, 1.0]),
                ),
            )
        },
    )
    return BioProcessCollection(
        metadata={"case_study": {"case_id": "bolus"}}, processes={"bolus": process}
    )


def test_load_raw_collection_reads_input():
    collection = load_raw_collection(INPUT_JSON)
    assert len(collection.processes) == 12
    assert "hybrax" in (collection.metadata or {})


def test_prepare_artifact_writes_bp_train_metadata(tmp_path):
    output = tmp_path / "prepared.json"
    custom_py = tmp_path / "custom.py"
    _write_sample_semantics_custom_py(custom_py)
    with pytest.warns(UserWarning, match="bp_format validation reported non-OK status"):
        prepare_artifact(_make_invalid_collection(), output, custom_py=custom_py)

    prepared = load_process_collection_json(output)
    metadata = prepared.metadata["bp_train"]

    assert metadata["process_order"] == list(prepared.processes.keys())
    assert metadata["runtime_controls_config"]["initial_grid_points"] >= 2

    first_name = metadata["process_order"][0]
    process_md = metadata["processes"][first_name]
    assert process_md["sample_acc_name"] == "V_sample_acc"
    assert process_md["name_extras"][-1] == "V_sample_acc"
    assert process_md["control_metadata"]["V_sample_acc"]["event_count"] >= 1
    assert any(
        not entry["ok"] for entry in metadata["bp_format_validation_raw"].values()
    )
    assert all(entry["ok"] for entry in metadata["bp_format_validation"].values())
    assert all(
        entry["ok"] for entry in metadata["bp_format_validation_prepared"].values()
    )
    assert metadata["prepared_semantics_validation"][first_name]["ok"] is True
    semantics = metadata["semantics_provenance"]["processes"][first_name]
    assert semantics["changed_by_hooks"] == ["transform_process_collection"]
    assert semantics["reactor_components_added"] == ["biomass"]


def test_prepare_artifact_does_not_persist_padded_control_arrays(tmp_path):
    output = tmp_path / "prepared-minimal.json"
    custom_py = tmp_path / "custom.py"
    _write_sample_semantics_custom_py(custom_py)
    with pytest.warns(UserWarning):
        prepare_artifact(_make_invalid_collection(), output, custom_py=custom_py)

    prepared = load_process_collection_json(output)
    process_md = prepared.metadata["bp_train"]["processes"]["invalid"]

    assert "dense_grid" not in process_md
    assert "control_values" not in process_md
    assert "control_derivatives" not in process_md
    assert "step_ts" not in process_md
    assert "step_ts_mask" not in process_md


def test_prepare_artifact_respects_custom_control_order(tmp_path):
    custom_py = tmp_path / "custom.py"
    custom_py.write_text(
        "\n".join(
            [
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

    output = tmp_path / "prepared-custom.json"
    prepare_artifact(_make_two_process_collection(), output, custom_py=custom_py)

    prepared = load_process_collection_json(output)
    metadata = prepared.metadata["bp_train"]
    first_name = metadata["process_order"][0]
    process_md = metadata["processes"][first_name]

    assert process_md["name_controlled_PVs"] == ["CF", "T"]
    assert process_md["name_extras"][-1] == "V_sample_acc"


def test_prepare_artifact_can_rename_processes(tmp_path):
    output = tmp_path / "prepared-renamed.json"
    prepare_artifact(
        _make_two_process_collection(),
        output,
        config={"process_rename_map": {"p1": "process=p1", "p2": "process=p2"}},
    )

    prepared = load_process_collection_json(output)
    assert list(prepared.processes.keys()) == ["process=p1", "process=p2"]
    assert prepared.processes["process=p1"].metadata.name == "process=p1"
    assert prepared.metadata["bp_train"]["process_order"] == [
        "process=p1",
        "process=p2",
    ]


def test_prepare_artifact_rename_provenance_tracks_changes(tmp_path):
    """Provenance must detect changes even when processes are renamed."""
    output = tmp_path / "prepared-renamed-provenance.json"
    prepare_artifact(
        _make_two_process_collection(),
        output,
        config={
            "process_rename_map": {
                "p1": "process=p1",
                "p2": "process=p2",
            }
        },
    )

    prepared = load_process_collection_json(output)
    prov = prepared.metadata["bp_train"]["semantics_provenance"]["processes"]
    for new_name in ["process=p1", "process=p2"]:
        entry = prov[new_name]
        assert "transform_process_collection" in entry["changed_by_hooks"], (
            f"provenance for {new_name!r} should record the transform hook as a changer"
        )
        assert entry["raw"] == entry["prepared"], (
            "pure rename should resolve the correct raw snapshot"
        )


def test_prepare_artifact_rejects_duplicate_process_renames(tmp_path):
    with pytest.raises(ValueError, match="duplicate renamed process key"):
        prepare_artifact(
            _make_two_process_collection(),
            tmp_path / "prepared-duplicate-renames.json",
            config={"process_rename_map": {"p1": "same", "p2": "same"}},
        )


def test_prepare_artifact_partial_process_rename_preserves_unmapped_metadata_name(
    tmp_path,
):
    collection = _make_two_process_collection()
    collection.processes = {
        "key_p1": collection.processes["p1"],
        "key_p2": collection.processes["p2"],
    }
    assert collection.processes["key_p2"].metadata.name == "p2"

    output = tmp_path / "prepared-partial-rename.json"
    prepare_artifact(
        collection,
        output,
        config={"process_rename_map": {"key_p1": "renamed_p1"}},
    )

    prepared = load_process_collection_json(output)
    assert list(prepared.processes.keys()) == ["renamed_p1", "key_p2"]
    assert prepared.processes["renamed_p1"].metadata.name == "renamed_p1"
    assert prepared.processes["key_p2"].metadata.name == "p2"


def test_prepare_artifact_supports_transform_process_collection_hook(tmp_path):
    custom_py = tmp_path / "custom-transform-collection.py"
    custom_py.write_text(
        "\n".join(
            [
                "def transform_process_collection(collection, config):",
                "    reordered = {}",
                "    for name, process in collection.processes.items():",
                "        new_name = f'proc::{name}'",
                "        process.metadata.name = new_name",
                "        reordered[new_name] = process",
                "    collection.processes = reordered",
                "    collection.metadata = dict(collection.metadata or {})",
                "    collection.metadata['collection_transform_marker'] = 'applied'",
                "    return collection",
            ]
        ),
        encoding="utf-8",
    )
    output = tmp_path / "prepared-transform-collection.json"
    prepare_artifact(_make_two_process_collection(), output, custom_py=custom_py)

    prepared = load_process_collection_json(output)
    assert list(prepared.processes.keys()) == ["proc::p1", "proc::p2"]
    assert prepared.metadata["collection_transform_marker"] == "applied"
    assert (
        prepared.metadata["bp_train"]["transform_hooks"]["transform_process_collection"]
        == "transform_process_collection"
    )


def test_prepare_artifact_builds_sample_acc_amount_correctly(tmp_path):
    output = tmp_path / "prepared-sample.json"
    custom_py = tmp_path / "custom.py"
    _write_sample_semantics_custom_py(custom_py)
    with pytest.warns(UserWarning):
        prepare_artifact(_make_invalid_collection(), output, custom_py=custom_py)

    store = ControlsStore.from_json(output)
    controls = store.get_controls("invalid")
    end_t = _make_invalid_collection().processes["invalid"].time_axis.end
    assert controls.eval(end_t)[controls.sample_acc_global_index] == pytest.approx(0.1)


def test_load_raw_collection_accepts_in_memory_collection():
    collection = _make_feed_collection()
    loaded = load_raw_collection(collection)
    assert loaded is collection


def test_prepare_artifact_persists_feed_metadata(tmp_path):
    output = tmp_path / "prepared-feed.json"
    custom_py = tmp_path / "custom-feed.py"
    _write_feed_semantics_custom_py(custom_py)
    with pytest.warns(UserWarning):
        prepare_artifact(_make_feed_collection(), output, custom_py=custom_py)

    prepared = load_process_collection_json(output)
    metadata = prepared.metadata["bp_train"]
    process_md = metadata["processes"]["p1"]
    feed_md = process_md["control_metadata"]["feed_A"]
    semantics = metadata["semantics_provenance"]["processes"]["p1"]

    assert process_md["name_controlled_FVCs"] == ["feed_A"]
    assert process_md["name_extras"] == ["V_sample_acc"]
    assert feed_md["signal_family"] == "feed"
    assert feed_md["source_kind"] == "control"
    assert feed_md["inlet_feed_medium"]["components"]["glucose"]["unit"] == "g/L"
    assert "biomass" in feed_md["inlet_feed_medium"]["components"]
    assert semantics["reactor_components_added"] == ["biomass", "glucose"]
    assert semantics["feed_components_added"] == {"feed_A": ["biomass"]}


def test_prepare_artifact_fails_without_required_medium_enrichment(tmp_path):
    with (
        pytest.warns(UserWarning),
        pytest.raises(
            ValueError,
            match="prepared semantics validation failed",
        ),
    ):
        prepare_artifact(
            _make_invalid_collection(), tmp_path / "prepared-missing-medium.json"
        )


def test_prepare_artifact_fails_strict_post_transform_bp_format_validation(tmp_path):
    custom_py = tmp_path / "custom-incomplete-feed.py"
    _write_feed_semantics_incomplete_custom_py(custom_py)

    with (
        pytest.warns(UserWarning),
        pytest.raises(
            ValueError,
            match="bp_format validation failed",
        ),
    ):
        prepare_artifact(
            _make_feed_collection(),
            tmp_path / "prepared-incomplete-feed.json",
            custom_py=custom_py,
        )


def test_prepare_artifact_rejects_zero_feed_without_component_metadata(tmp_path):
    collection = _make_feed_collection()
    process = collection.processes["p1"]
    process.reactor_medium.components["biomass"] = ReactorMediumComponent(
        name="biomass",
        unit="g/L",
        concentration=StaticVariable(0.1),
    )
    process.volume.volume_changes["feed_A"].values = TimeSeries(
        times=jnp.asarray([0.0, 1.0]),
        values=jnp.asarray([0.0, 0.0]),
    )
    process.volume.volume_changes["feed_A"].feed_medium.components = {}

    with pytest.raises(
        ValueError,
        match="feed 'feed_A' has no feed-medium component metadata after prep",
    ):
        prepare_artifact(collection, tmp_path / "prepared-zero-feed.json")


def test_prepare_artifact_rejects_inconsistent_control_sets(tmp_path):
    custom_py = tmp_path / "custom-inconsistent.py"
    custom_py.write_text(
        "\n".join(
            [
                "def transform_process_collection(collection, config):",
                "    for process in collection.processes.values():",
                "        if process.metadata.name == 'p1':",
                "            process.process_variables['CF'].is_controlled = True",
                "        else:",
                "            process.process_variables['T'].is_controlled = True",
                "    return collection",
            ]
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="categorised control layout differs"):
        prepare_artifact(
            _make_two_process_collection(),
            tmp_path / "prepared-bad.json",
            custom_py=custom_py,
        )


def test_prepare_artifact_fails_on_missing_required_control(tmp_path):
    with pytest.raises(ValueError, match="config-declared controls are missing"):
        prepare_artifact(
            _make_two_process_collection(),
            tmp_path / "prepared-missing.json",
            config={"required_control_names": ["CF"]},
        )


def test_build_bolus_sources_triangle_geometry_and_step_ts():
    collection = _make_bolus_collection()
    process = collection.processes["bolus"]
    source = build_bolus_sources(process)[0]
    min_dt = (10.0 - 0.0) / BOLUS_MIN_DT_DURATION_DENOMINATOR
    triangle_peak = 5.0 + 0.5 * min_dt
    triangle_end = 5.0 + min_dt

    assert source.evaluator(jnp.asarray([2.5]))[0] == pytest.approx(0.0)
    assert float(source.metadata["triangle_min_dt"]) == pytest.approx(min_dt)
    assert float(source.metadata["triangle_width"]) == pytest.approx(min_dt)
    assert source.step_ts == pytest.approx([5.0, triangle_peak, triangle_end])
    assert source.evaluator(jnp.asarray([5.0]))[0] == pytest.approx(0.0)
    assert source.evaluator(jnp.asarray([triangle_peak]))[0] == pytest.approx(
        2.0 / min_dt,
        rel=1e-3,
    )
    assert source.evaluator(jnp.asarray([triangle_end]))[0] == pytest.approx(
        0.0, abs=1e-3
    )

    t_grid = np.asarray([5.0, triangle_peak, triangle_end], dtype=float)
    rates = source.evaluator(t_grid)
    assert float(np.trapezoid(rates, t_grid)) == pytest.approx(1.0, abs=1e-9)


def _make_bolus_collection_with_events(
    event_times: list[float], event_values: list[float]
) -> BioProcessCollection:
    """Return a bolus collection whose single feed has explicit event schedule."""
    if len(event_times) != len(event_values):
        raise ValueError("event_times and event_values must have equal length")

    feed_medium = FeedMedium(
        name="feed",
        density=1.0,
        density_unit="kg/L",
        components={
            "X": FeedMediumComponent(
                name="X",
                unit="g/L",
                concentration=StaticVariable(0.0),
                is_controlled=False,
            )
        },
    )
    process = BioProcess(
        metadata=BioProcessMetadata(name="bolus", process_type="fed_batch"),
        time_axis=TimeAxis(unit="h", start=0.0, end=10.0, time_reference="start"),
        volume=Volume(
            initial_volume=1.0,
            unit="L",
            volume_changes={
                "feed_bolus": FeedVolumeChange(
                    name="feed_bolus",
                    unit="L",
                    is_controlled=True,
                    is_continuous=False,
                    values=TimeSeries(
                        times=jnp.asarray(event_times),
                        values=jnp.asarray(event_values),
                    ),
                    feed_medium=feed_medium,
                )
            },
        ),
        reactor_medium=ReactorMedium(
            name="rm",
            density=1.0,
            density_unit="kg/L",
            components={
                "X": ReactorMediumComponent(
                    name="X",
                    unit="g/L",
                    concentration=StaticVariable(0.0),
                ),
                "biomass": ReactorMediumComponent(
                    name="biomass",
                    unit="g/L",
                    concentration=StaticVariable(0.1),
                ),
            },
        ),
        process_variables={
            "X": ProcessVariable(
                name="X",
                unit="g/L",
                is_controlled=False,
                values=TimeSeries(
                    times=jnp.asarray([0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 10.0]),
                    values=jnp.asarray([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]),
                ),
            )
        },
    )
    return BioProcessCollection(
        metadata={"case_study": {"case_id": "bolus"}}, processes={"bolus": process}
    )


def test_build_bolus_sources_raises_when_triangle_cannot_fit_before_end():
    collection = _make_bolus_collection_with_events([9.995], [0.5])
    process = collection.processes["bolus"]
    with pytest.raises(ValueError, match="cannot fit triangle width"):
        build_bolus_sources(process, run_min_dt=0.01)


def test_build_bolus_sources_superposes_overlapping_events():
    collection = _make_bolus_collection_with_events([5.0, 5.01], [1.0, 1.0])
    process = collection.processes["bolus"]
    source = build_bolus_sources(process)[0]
    min_dt = (10.0 - 0.0) / BOLUS_MIN_DT_DURATION_DENOMINATOR
    assert float(source.metadata["triangle_min_dt"]) == pytest.approx(min_dt)
    step_ts = np.asarray(source.step_ts, dtype=float)
    assert np.any(np.isclose(step_ts, 5.0, atol=1e-6))
    assert np.any(np.isclose(step_ts, 5.005, atol=1e-4))
    assert np.any(np.isclose(step_ts, 5.01, atol=1e-4))
    assert np.any(np.isclose(step_ts, 5.015, atol=1e-4))
    assert np.any(np.isclose(step_ts, 5.02, atol=1e-4))
    assert source.evaluator(jnp.asarray([5.015]))[0] == pytest.approx(200.0, rel=1e-3)

    t_grid = np.asarray([5.0, 5.005, 5.01, 5.015, 5.02], dtype=float)
    rates = source.evaluator(t_grid)
    assert float(np.trapezoid(rates, t_grid)) == pytest.approx(2.0, abs=1e-4)


def test_build_bolus_sources_rejects_event_at_process_end():
    collection = _make_bolus_collection_with_events([5.0, 10.0], [0.5, 0.5])
    process = collection.processes["bolus"]
    with pytest.raises(ValueError, match="at/after process end"):
        build_bolus_sources(process)


def test_prepare_allows_custom_sample_hook_without_run_min_dt(tmp_path):
    process = BioProcess(
        metadata=BioProcessMetadata(name="p1", process_type="fed_batch"),
        time_axis=TimeAxis(unit="h", start=0.0, end=10.0, time_reference="start"),
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
                        times=jnp.asarray([5.0]),
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
        process_variables={},
    )
    collection = BioProcessCollection(
        metadata={"case_study": {"case_id": "custom-sample"}},
        processes={"p1": process},
    )

    custom_py = tmp_path / "custom_sample_override.py"
    custom_py.write_text(
        "\n".join(
            [
                "import numpy as np",
                "from bp_train.controls import SignalSource",
                "",
                "def build_sample_acc_series("
                "process, process_name, collection_metadata, config):",
                "    del process_name, collection_metadata, config",
                "    t0 = float(process.time_axis.start)",
                "    t1 = float(process.time_axis.end)",
                "    times = np.asarray([t0, t1], dtype=float)",
                "    values = np.asarray([0.0, 0.1], dtype=float)",
                "    return SignalSource(",
                "        name='V_sample_acc',",
                "        kind='derived_control',",
                "        times=times,",
                "        values=values,",
                "        evaluator=lambda ts: np.interp("
                "np.asarray(ts, dtype=float), times, values, "
                "left=values[0], right=values[-1]),",
                "        derivative=lambda ts: np.full_like("
                "np.asarray(ts, dtype=float), 0.01, dtype=float),",
                "        step_ts=[t0, t1],",
                "        metadata={'source': 'custom_test'},",
                "    )",
            ]
        ),
        encoding="utf-8",
    )

    output_json = tmp_path / "prepared_custom_sample.json"
    prepared = prepare_artifact(collection, output_json, custom_py=custom_py)
    process_md = prepared.metadata["bp_train"]["processes"]["p1"]
    assert process_md["sample_acc_source"]["metadata"]["source"] == "custom_test"
    assert process_md["sample_acc_source"]["values"][-1] == pytest.approx(0.1)


def test_select_control_sources_handles_null_feed_medium():
    """select_control_sources must not crash with AttributeError when
    feed_medium is None; semantic validation handles the clear error."""
    process = BioProcess(
        metadata=BioProcessMetadata(name="p1", process_type="fed_batch"),
        time_axis=TimeAxis(unit="h", start=0.0, end=1.0, time_reference="start"),
        volume=Volume(
            initial_volume=1.0,
            unit="L",
            volume_changes={
                "feed_A": FeedVolumeChange(
                    name="feed_A",
                    unit="L/h",
                    is_controlled=True,
                    is_continuous=True,
                    values=TimeSeries(
                        times=jnp.asarray([0.0, 1.0]),
                        values=jnp.asarray([0.1, 0.1]),
                    ),
                    # feed_medium=None is allowed at runtime even though the
                    # type annotation forbids it; dataclasses do not enforce
                    # field types, so None can reach this code path when data
                    # is incomplete before semantic validation runs.
                    feed_medium=None,  # type: ignore[arg-type]
                )
            },
        ),
        reactor_medium=ReactorMedium(name="rm", density=1.0, density_unit="kg/L"),
        process_variables={
            "X": ProcessVariable(
                name="X",
                unit="g/L",
                is_controlled=False,
                values=TimeSeries(
                    times=jnp.asarray([0.0, 2.0, 4.0, 6.0, 8.0, 10.0]),
                    values=jnp.asarray([1.0, 1.0, 1.0, 1.0, 1.0, 1.0]),
                ),
            ),
        },
    )
    bundle = select_control_sources("p1", process, {})
    sources = bundle.all_sources
    assert len(sources) == 1
    assert sources[0].name == "feed_A"
    assert sources[0].metadata["inlet_feed_medium"] is None
    assert bundle.name_controlled_FVCs == ("feed_A",)


def test_build_bolus_sources_handles_null_feed_medium():
    """build_bolus_sources must not crash with AttributeError when
    feed_medium is None (bolus / is_continuous=False path)."""
    process = BioProcess(
        metadata=BioProcessMetadata(name="p1", process_type="fed_batch"),
        time_axis=TimeAxis(unit="h", start=0.0, end=10.0, time_reference="start"),
        volume=Volume(
            initial_volume=1.0,
            unit="L",
            volume_changes={
                "feed_bolus": FeedVolumeChange(
                    name="feed_bolus",
                    unit="L",
                    is_controlled=True,
                    is_continuous=False,
                    values=TimeSeries(
                        times=jnp.asarray([5.0]),
                        values=jnp.asarray([1.0]),
                    ),
                    feed_medium=None,  # type: ignore[arg-type]
                )
            },
        ),
        reactor_medium=ReactorMedium(name="rm", density=1.0, density_unit="kg/L"),
        process_variables={
            "X": ProcessVariable(
                name="X",
                unit="g/L",
                is_controlled=False,
                values=TimeSeries(
                    times=jnp.asarray([0.0, 2.0, 4.0, 6.0, 8.0, 10.0]),
                    values=jnp.asarray([1.0, 1.0, 1.0, 1.0, 1.0, 1.0]),
                ),
            ),
        },
    )
    sources = build_bolus_sources(process)
    assert len(sources) == 1
    assert sources[0].name == "feed_bolus"
    assert sources[0].metadata["inlet_feed_medium"] is None


def _write_bolus_biomass_custom_py(path: Path) -> None:
    """Add a biomass component (reactor + feed) to the single-process bolus
    fixture so it passes prepare's bp_format and semantics validations."""
    path.write_text(
        "\n".join(
            [
                "from bp_format.dataclasses import (",
                "    FeedMediumComponent,",
                "    ReactorMediumComponent,",
                "    StaticVariable,",
                "    TimeSeries,",
                ")",
                "import jax.numpy as jnp",
                "",
                "def transform_process_collection(collection, config):",
                "    process = next(iter(collection.processes.values()))",
                "    process.reactor_medium.components['biomass'] = "
                "ReactorMediumComponent(",
                "        name='biomass',",
                "        unit='g/L',",
                "        concentration=TimeSeries(",
                "            times=jnp.asarray([0.0, 5.0, 10.0]),",
                "            values=jnp.asarray([0.1, 0.5, 1.0]),",
                "        ),",
                "    )",
                "    feed = process.volume.volume_changes['feed_bolus'].feed_medium",
                "    feed.components['biomass'] = FeedMediumComponent(",
                "        name='biomass',",
                "        unit='g/L',",
                "        concentration=StaticVariable(0.0),",
                "        is_controlled=False,",
                "    )",
                "    return collection",
            ]
        ),
        encoding="utf-8",
    )


def _write_bolus_run_min_dt_custom_py(path: Path, value: float) -> None:
    path.write_text(
        "\n".join(
            [
                "CONFIG = {",
                f"    'bolus_run_min_dt': {value!r},",
                "}",
            ]
        ),
        encoding="utf-8",
    )


def test_prepare_artifact_honors_user_bolus_run_min_dt(tmp_path):
    """User-supplied ``bolus_run_min_dt`` must not be overwritten by auto-detection.

    Regression for a bug where prepare unconditionally overwrote user config
    with the collection-wide minimum online-timestamp delta, making the
    documented override knob a no-op.
    """
    output = tmp_path / "prepared_bolus_min_dt.json"
    custom_py = tmp_path / "custom_bolus.py"
    _write_bolus_biomass_custom_py(custom_py)
    # Pick user_value strictly below the duration cap (10 / 1000 = 0.01) so
    # ``get_bolus_min_dt`` does not silently clamp it; otherwise we'd be
    # asserting against the cap rather than the user setting.
    user_value = 0.005
    auto_value = 10.0 / BOLUS_MIN_DT_DURATION_DENOMINATOR
    assert user_value != auto_value

    prepared = prepare_artifact(
        _make_bolus_collection(),
        output,
        custom_py=custom_py,
        config={EVENT_RUN_MIN_DT_CONFIG_KEY: user_value},
    )

    rcc = prepared.metadata["bp_train"]["runtime_controls_config"]
    assert rcc[EVENT_RUN_MIN_DT_CONFIG_KEY] == pytest.approx(user_value)

    on_disk = load_process_collection_json(output)
    on_disk_rcc = on_disk.metadata["bp_train"]["runtime_controls_config"]
    assert on_disk_rcc[EVENT_RUN_MIN_DT_CONFIG_KEY] == pytest.approx(user_value)


def test_prepare_artifact_honors_custom_py_bolus_run_min_dt(tmp_path):
    """custom.py CONFIG must override auto-detected collection min_dt."""
    from bp_train.controls import get_collection_bolus_min_dt

    output = tmp_path / "prepared_custom_bolus_min_dt.json"
    custom_py = tmp_path / "custom.py"
    user_value = 0.005
    _write_bolus_run_min_dt_custom_py(custom_py, user_value)

    raw = _make_bolus_collection()
    raw.processes["bolus"].volume.volume_changes["sample_1"] = SampleVolumeChange(
        name="sample_1",
        unit="L",
        is_controlled=False,
        is_continuous=False,
        values=TimeSeries(
            times=jnp.asarray([6.0]),
            values=jnp.asarray([-0.1]),
        ),
    )
    auto_value = get_collection_bolus_min_dt(raw)
    assert auto_value == pytest.approx(1.0)

    prepared = prepare_artifact(raw, output, custom_py=custom_py)
    rcc = prepared.metadata["bp_train"]["runtime_controls_config"]
    assert rcc[EVENT_RUN_MIN_DT_CONFIG_KEY] == pytest.approx(user_value)

    prepared_md = prepared.metadata["bp_train"]["processes"]["bolus"]
    feed_md = prepared_md["control_metadata"]["feed_bolus"]
    sample_md = prepared_md["control_metadata"]["V_sample_acc"]
    assert float(feed_md["triangle_min_dt"]) == pytest.approx(user_value)
    assert float(sample_md["ramp_duration"]) == pytest.approx(user_value)

    on_disk = load_process_collection_json(output)
    store = ControlsStore.from_collection(on_disk)
    triangle_md = store.get_controls("bolus").control_metadata["feed_bolus"]
    assert float(triangle_md["triangle_min_dt"]) == pytest.approx(user_value)


def test_prepare_artifact_auto_detects_bolus_run_min_dt_when_unset(tmp_path):
    """Auto-detection still fires when the user did not supply a value."""
    from bp_train.controls import get_collection_bolus_min_dt

    output = tmp_path / "prepared_bolus_min_dt_auto.json"
    custom_py = tmp_path / "custom_bolus.py"
    _write_bolus_biomass_custom_py(custom_py)
    raw = _make_bolus_collection()
    expected = get_collection_bolus_min_dt(raw)
    prepared = prepare_artifact(raw, output, custom_py=custom_py)

    rcc = prepared.metadata["bp_train"]["runtime_controls_config"]
    assert rcc[EVENT_RUN_MIN_DT_CONFIG_KEY] == pytest.approx(expected)
    assert rcc[EVENT_RUN_MIN_DT_CONFIG_KEY] > 0


def test_controls_store_honors_prepared_bolus_run_min_dt(tmp_path):
    """ControlsStore must reuse the prepared ``bolus_run_min_dt`` instead of
    recomputing it from the collection at training time.

    Regression for a second overwrite site in ``ControlsStore.from_collection``.
    """
    output = tmp_path / "prepared_bolus_for_store.json"
    custom_py = tmp_path / "custom_bolus.py"
    _write_bolus_biomass_custom_py(custom_py)
    user_value = 0.005
    prepare_artifact(
        _make_bolus_collection(),
        output,
        custom_py=custom_py,
        config={EVENT_RUN_MIN_DT_CONFIG_KEY: user_value},
    )

    prepared = load_process_collection_json(output)
    store = ControlsStore.from_collection(prepared)
    triangle_md = store.get_controls("bolus").control_metadata["feed_bolus"]
    assert float(triangle_md["triangle_min_dt"]) == pytest.approx(user_value)
