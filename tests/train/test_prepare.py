from __future__ import annotations

from pathlib import Path

import jax.numpy as jnp
import pytest
from bpbench.dataclasses import (
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
from bpbench.serialization import load_process_collection_json

from bp_train.controls import build_bolus_sources, get_shortest_time_diff
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
                    timepoints=jnp.asarray([0.0, 1.0]),
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
                        timepoints=jnp.asarray([0.0, 1.0]),
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
                    timepoints=jnp.asarray([0.0, 1.0]),
                    values=jnp.asarray([1.0, 1.2]),
                ),
            )
        },
    )
    return BioProcessCollection(metadata={"case_study": {"case_id": "synthetic"}}, processes={"p1": process})


def _write_sample_semantics_custom_py(path: Path) -> None:
    path.write_text(
        "\n".join(
            [
                "from bpbench.dataclasses import ReactorMediumComponent, TimeSeries",
                "import jax.numpy as jnp",
                "",
                "def transform_states(process, config):",
                "    process.reactor_medium.components['biomass'] = ReactorMediumComponent(",
                "        name='biomass',",
                "        unit='g/L',",
                "        concentration=TimeSeries(",
                "            timepoints=jnp.asarray([0.0, 1.0]),",
                "            values=jnp.asarray([0.1, 0.2]),",
                "        ),",
                "        is_intracellular=False,",
                "    )",
                "    return process",
            ]
        ),
        encoding="utf-8",
    )


def _write_feed_semantics_custom_py(path: Path) -> None:
    path.write_text(
        "\n".join(
            [
                "from bpbench.dataclasses import FeedMediumComponent, ReactorMediumComponent, StaticVariable, TimeSeries",
                "import jax.numpy as jnp",
                "",
                "def transform_states(process, config):",
                "    process.reactor_medium.components['biomass'] = ReactorMediumComponent(",
                "        name='biomass',",
                "        unit='g/L',",
                "        concentration=StaticVariable(0.1),",
                "        is_intracellular=False,",
                "    )",
                "    process.reactor_medium.components['glucose'] = ReactorMediumComponent(",
                "        name='glucose',",
                "        unit='g/L',",
                "        concentration=TimeSeries(",
                "            timepoints=jnp.asarray([0.0, 1.0]),",
                "            values=jnp.asarray([1.0, 1.2]),",
                "        ),",
                "        is_intracellular=False,",
                "    )",
                "    process.volume.volume_changes['feed_A'].feed_medium.components['biomass'] = FeedMediumComponent(",
                "        name='biomass',",
                "        unit='g/L',",
                "        concentration=StaticVariable(0.0),",
                "        is_controlled=False,",
                "    )",
                "    return process",
            ]
        ),
        encoding="utf-8",
    )


def _write_feed_semantics_incomplete_custom_py(path: Path) -> None:
    path.write_text(
        "\n".join(
            [
                "from bpbench.dataclasses import FeedMediumComponent, ReactorMediumComponent, StaticVariable, TimeSeries",
                "import jax.numpy as jnp",
                "",
                "def transform_states(process, config):",
                "    process.reactor_medium.components['biomass'] = ReactorMediumComponent(",
                "        name='biomass',",
                "        unit='g/L',",
                "        concentration=StaticVariable(0.1),",
                "        is_intracellular=False,",
                "    )",
                "    process.reactor_medium.components['glucose'] = ReactorMediumComponent(",
                "        name='glucose',",
                "        unit='g/L',",
                "        concentration=TimeSeries(",
                "            timepoints=jnp.asarray([0.0, 1.0]),",
                "            values=jnp.asarray([1.0, 1.2]),",
                "        ),",
                "        is_intracellular=False,",
                "    )",
                "    process.volume.volume_changes['feed_A'].feed_medium.components = {}",
                "    process.volume.volume_changes['feed_A'].feed_medium.components['biomass'] = FeedMediumComponent(",
                "        name='biomass',",
                "        unit='g/L',",
                "        concentration=StaticVariable(0.0),",
                "        is_controlled=False,",
                "    )",
                "    return process",
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
                        timepoints=jnp.asarray([0.5]),
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
                    timepoints=jnp.asarray([0.0, 1.0]),
                    values=jnp.asarray([1.0, 1.2]),
                ),
            )
        },
    )
    return BioProcessCollection(metadata={"case_study": {"case_id": "invalid"}}, processes={"invalid": process})


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
                            timepoints=jnp.asarray([0.5]),
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
                        timepoints=jnp.asarray([0.0, 1.0]),
                        values=jnp.asarray([1.0, 1.1]),
                    ),
                ),
                "T": ProcessVariable(
                    name="T",
                    unit="C",
                    is_controlled=False,
                    values=TimeSeries(
                        timepoints=jnp.asarray([0.0, 1.0]),
                        values=jnp.asarray([30.0, 31.0]),
                    ),
                ),
            },
        )
    return BioProcessCollection(metadata={"case_study": {"case_id": "two-process"}}, processes=processes)


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
                        timepoints=jnp.asarray([5.0]),
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
                    is_intracellular=False,
                )
            },
        ),
        process_variables={
            "X": ProcessVariable(
                name="X",
                unit="g/L",
                is_controlled=False,
                values=TimeSeries(
                    timepoints=jnp.asarray([0.0, 5.0, 10.0]),
                    values=jnp.asarray([1.0, 1.0, 1.0]),
                ),
            )
        },
    )
    return BioProcessCollection(metadata={"case_study": {"case_id": "bolus"}}, processes={"bolus": process})


def test_load_raw_collection_reads_input():
    collection = load_raw_collection(INPUT_JSON)
    assert len(collection.processes) == 12
    assert "hybrax" in (collection.metadata or {})


def test_prepare_artifact_writes_bp_train_metadata(tmp_path):
    output = tmp_path / "prepared.json"
    custom_py = tmp_path / "custom.py"
    _write_sample_semantics_custom_py(custom_py)
    with pytest.warns(UserWarning, match="bpbench validation reported non-OK status"):
        prepare_artifact(_make_invalid_collection(), output, custom_py=custom_py)

    prepared = load_process_collection_json(output)
    metadata = prepared.metadata["bp_train"]

    assert metadata["shape_metadata"]["n_processes"] == len(prepared.processes)
    assert metadata["shape_metadata"]["max_grid_length"] >= 2
    assert metadata["shape_metadata"]["max_controls"] >= 1

    first_name = metadata["process_order"][0]
    process_md = metadata["processes"][first_name]
    assert process_md["sample_acc_name"] == "V_sample_acc"
    assert metadata["global_control_names"][process_md["sample_acc_index"]] == "V_sample_acc"
    assert len(process_md["dense_grid"]) == metadata["shape_metadata"]["max_grid_length"]
    assert len(process_md["control_values"]) == metadata["shape_metadata"]["max_grid_length"]
    assert len(process_md["control_values"][0]) == metadata["shape_metadata"]["max_controls"]
    assert len(process_md["step_ts"]) == metadata["shape_metadata"]["max_step_ts_length"]
    assert len(process_md["step_ts_mask"]) == metadata["shape_metadata"]["max_step_ts_length"]
    assert any(v > 0 for row in process_md["control_values"] for v in row)
    assert any(not entry["ok"] for entry in metadata["bpbench_validation_raw"].values())
    assert all(entry["ok"] for entry in metadata["bpbench_validation"].values())
    assert all(entry["ok"] for entry in metadata["bpbench_validation_prepared"].values())
    assert metadata["prepared_semantics_validation"][first_name]["ok"] is True
    semantics = metadata["semantics_provenance"]["processes"][first_name]
    assert semantics["changed_by_hooks"] == ["transform_states"]
    assert semantics["reactor_components_added"] == ["biomass"]
    assert process_md["control_mask"] == [True]


def test_prepare_artifact_respects_custom_control_order(tmp_path):
    custom_py = tmp_path / "custom.py"
    custom_py.write_text(
        "\n".join(
            [
                "CONFIG = {'control_order': ['CF', 'T']}",
                "",
                "def transform_controls(process, config):",
                "    process.process_variables['CF'].is_controlled = True",
                "    process.process_variables['T'].is_controlled = True",
                "    return process",
            ]
        ),
        encoding="utf-8",
    )

    output = tmp_path / "prepared-custom.json"
    prepare_artifact(_make_two_process_collection(), output, custom_py=custom_py)

    prepared = load_process_collection_json(output)
    metadata = prepared.metadata["bp_train"]
    first_name = metadata["process_order"][0]
    control_names = metadata["processes"][first_name]["local_control_names"]

    assert control_names[:2] == ["CF", "T"]
    assert control_names[-1] == "V_sample_acc"


def test_prepare_artifact_builds_sample_acc_amount_correctly(tmp_path):
    output = tmp_path / "prepared-sample.json"
    custom_py = tmp_path / "custom.py"
    _write_sample_semantics_custom_py(custom_py)
    with pytest.warns(UserWarning):
        prepare_artifact(_make_invalid_collection(), output, custom_py=custom_py)

    prepared = load_process_collection_json(output)
    metadata = prepared.metadata["bp_train"]
    process_md = metadata["processes"]["invalid"]
    sample_idx = process_md["sample_acc_index"]
    last_true_idx = max(i for i, flag in enumerate(process_md["dense_grid_mask"]) if flag)

    assert process_md["control_values"][last_true_idx][sample_idx] == pytest.approx(0.1)


def test_load_raw_collection_accepts_in_memory_collection():
    collection = _make_feed_collection()
    loaded = load_raw_collection(collection)
    assert loaded is collection


def test_prepare_artifact_persists_feed_metadata_and_global_axis(tmp_path):
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

    assert metadata["global_control_names"] == ["feed_A", "V_sample_acc"]
    assert process_md["control_mask"] == [True, True]
    assert feed_md["signal_family"] == "feed"
    assert feed_md["source_kind"] == "control"
    assert feed_md["inlet_feed_medium"]["components"]["glucose"]["unit"] == "g/L"
    assert "biomass" in feed_md["inlet_feed_medium"]["components"]
    assert semantics["reactor_components_added"] == ["biomass", "glucose"]
    assert semantics["feed_components_added"] == {"feed_A": ["biomass"]}


def test_prepare_artifact_fails_without_required_medium_enrichment(tmp_path):
    with pytest.warns(UserWarning), pytest.raises(
        ValueError,
        match="prepared semantics validation failed",
    ):
        prepare_artifact(_make_invalid_collection(), tmp_path / "prepared-missing-medium.json")


def test_prepare_artifact_fails_strict_post_transform_bpbench_validation(tmp_path):
    custom_py = tmp_path / "custom-incomplete-feed.py"
    _write_feed_semantics_incomplete_custom_py(custom_py)

    with pytest.warns(UserWarning), pytest.raises(
        ValueError,
        match="bpbench validation failed",
    ):
        prepare_artifact(_make_feed_collection(), tmp_path / "prepared-incomplete-feed.json", custom_py=custom_py)


def test_prepare_artifact_rejects_zero_feed_without_component_metadata(tmp_path):
    collection = _make_feed_collection()
    process = collection.processes["p1"]
    process.reactor_medium.components["biomass"] = ReactorMediumComponent(
        name="biomass",
        unit="g/L",
        concentration=StaticVariable(0.1),
        is_intracellular=False,
    )
    process.volume.volume_changes["feed_A"].values = TimeSeries(
        timepoints=jnp.asarray([0.0, 1.0]),
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
                "def transform_controls(process, config):",
                "    if process.metadata.name == 'p1':",
                "        process.process_variables['CF'].is_controlled = True",
                "    else:",
                "        process.process_variables['T'].is_controlled = True",
                "    return process",
            ]
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="control names/order differ"):
        prepare_artifact(_make_two_process_collection(), tmp_path / "prepared-bad.json", custom_py=custom_py)


def test_prepare_artifact_fails_on_missing_required_control(tmp_path):
    with pytest.raises(ValueError, match="config-declared controls are missing"):
        prepare_artifact(
            _make_two_process_collection(),
            tmp_path / "prepared-missing.json",
            config={"required_control_names": ["CF"]},
        )


def test_build_bolus_sources_stay_zero_before_event():
    collection = _make_bolus_collection()
    process = collection.processes["bolus"]
    source = build_bolus_sources(process)[0]

    assert source.evaluator(jnp.asarray([2.5]))[0] == pytest.approx(0.0)
    assert source.evaluator(jnp.asarray([5.5]))[0] > 0.0


def test_get_shortest_time_diff_ignores_near_duplicate_boundaries():
    collection = load_raw_collection(INPUT_JSON)
    process = collection.processes[next(iter(collection.processes))]
    assert get_shortest_time_diff(process) > 1e-4
