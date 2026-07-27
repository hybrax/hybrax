from __future__ import annotations

import json
import math
from pathlib import Path

import jax.numpy as jnp
import numpy as np
import pytest
from bp_format.dataclasses import (
    BiologicalOde,
    BioProcess,
    BioProcessCollection,
    BioProcessMetadata,
    CaseStudy,
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
from bp_format.serialization import (
    load_process_collection,
    save_case_study,
    save_process_collection,
)

from bp_train.constants import METADATA_NAMESPACE
from bp_train.controls import select_control_sources
from bp_train.controls_store import ControlsStore
from bp_train.prepare import load_raw_collection, prepare_artifact
from bp_train.run_config import load_prepare_config, resolve_prepared_path

INPUT_JSON = Path(__file__).resolve().parent.parent / "input.json"


def _prepare_from_collection(
    collection: BioProcessCollection,
    tmp_path: Path,
    output_dir: Path,
    *,
    custom_py: Path | None = None,
    prepare_config: dict[str, object] | None = None,
) -> BioProcessCollection:
    raw_json = tmp_path / f"{output_dir.name}-raw.json"
    config_json = tmp_path / f"{output_dir.name}-config.json"
    save_process_collection(collection, raw_json)
    prepare: dict[str, object] = {"raw_input": str(raw_json)}
    if prepare_config is not None:
        prepare.update(prepare_config)
    config: dict[str, object] = {"prepare": prepare}
    if custom_py is not None:
        config["custom_py"] = str(custom_py)
    config_json.write_text(json.dumps(config), encoding="utf-8")
    return prepare_artifact(load_prepare_config(config_json), output_dir)


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


def test_load_raw_collection_accepts_commented_case_study(tmp_path: Path):
    collection = _make_feed_collection()
    case_study = CaseStudy(
        case_id="commented",
        organism="test",
        citation="test",
        processes=collection.processes,
    )
    path = tmp_path / "case-study.json"
    save_case_study(case_study, path)
    path.write_text(
        "  // raw case study\n" + path.read_text(encoding="utf-8"),
        encoding="utf-8",
    )

    loaded = load_raw_collection(path)

    assert loaded.metadata["case_study"]["case_id"] == "commented"
    assert set(loaded.processes) == set(collection.processes)


def _make_explicit_ode_collection() -> BioProcessCollection:
    feed_medium = FeedMedium(
        name="feed",
        density=1.0,
        density_unit="kg/L",
        components={
            "biomass": FeedMediumComponent(
                name="biomass",
                unit="g/L",
                concentration=StaticVariable(0.0),
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
                        values=jnp.asarray([0.1, 0.2]),
                    ),
                )
            },
        ),
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
    process.biological_ode = BiologicalOde(
        algebraic={"growth": "mu * biomass"},
        rates={"mu": (None, None), "r_X": (None, None)},
        derivatives={"biomass": "growth", "X": "r_X"},
    )
    return BioProcessCollection(metadata={}, processes={"p1": process})


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
                "    process.biological_ode = None",
                "    process.__post_init__()",
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
                "    process.biological_ode = None",
                "    process.__post_init__()",
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
                "    process.biological_ode = None",
                "    process.__post_init__()",
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
    process.biological_ode = BiologicalOde(
        rates={"r_X": (None, None)},
        derivatives={"X": "r_X"},
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


def test_load_raw_collection_reads_input():
    collection = load_raw_collection(INPUT_JSON)
    assert len(collection.processes) == 12
    assert "hybrax" in (collection.metadata or {})


def test_prepare_artifact_preserves_valid_user_biological_ode(tmp_path):
    collection = _make_explicit_ode_collection()
    expected_ode = collection.processes["p1"].biological_ode

    output_dir = tmp_path / "prepared-explicit-ode"
    prepared = _prepare_from_collection(
        collection,
        tmp_path,
        output_dir,
    )

    prepared_ode = prepared.processes["p1"].biological_ode
    assert prepared_ode == expected_ode
    reloaded = load_process_collection(output_dir / "prepared.json")
    assert reloaded.processes["p1"].biological_ode == expected_ode


def test_prepare_artifact_rejects_missing_biological_ode_after_transform(
    tmp_path,
):
    collection = _make_explicit_ode_collection()
    custom_py = tmp_path / "clear_ode.py"
    custom_py.write_text(
        "\n".join(
            [
                "def transform_process_collection(collection, config):",
                "    collection.processes['p1'].biological_ode = None",
                "    return collection",
            ]
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="biological_ode is missing"):
        _prepare_from_collection(
            collection,
            tmp_path,
            tmp_path / "prepared-missing-ode",
            custom_py=custom_py,
        )


def test_prepare_artifact_rejects_invalid_stale_biological_ode_after_transform(
    tmp_path,
):
    collection = _make_explicit_ode_collection()
    collection.processes["p1"].biological_ode = None
    collection.processes["p1"].__post_init__()
    stale_ode = collection.processes["p1"].biological_ode
    assert stale_ode.derivatives == {"biomass": "q_biomass * biomass", "X": "r_X"}

    custom_py = tmp_path / "control_x.py"
    custom_py.write_text(
        "\n".join(
            [
                "def transform_process_collection(collection, config):",
                "    process = collection.processes['p1']",
                "    process.process_variables['X'].is_controlled = True",
                "    return collection",
            ]
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="biological_ode invalid"):
        _prepare_from_collection(
            collection,
            tmp_path,
            tmp_path / "prepared-regenerated-ode",
            custom_py=custom_py,
        )


def test_prepare_artifact_allows_hook_to_explicitly_regenerate_biological_ode(
    tmp_path,
):
    collection = _make_explicit_ode_collection()
    collection.processes["p1"].biological_ode = None
    collection.processes["p1"].__post_init__()

    custom_py = tmp_path / "control_x_regenerate.py"
    custom_py.write_text(
        "\n".join(
            [
                "def transform_process_collection(collection, config):",
                "    process = collection.processes['p1']",
                "    process.process_variables['X'].is_controlled = True",
                "    process.biological_ode = None",
                "    process.__post_init__()",
                "    return collection",
            ]
        ),
        encoding="utf-8",
    )

    prepared = _prepare_from_collection(
        collection,
        tmp_path,
        tmp_path / "prepared-regenerated-ode",
        custom_py=custom_py,
    )

    prepared_ode = prepared.processes["p1"].biological_ode
    assert prepared_ode.derivatives == {"biomass": "q_biomass * biomass"}
    assert prepared_ode.rates == {"q_biomass": (None, None)}


def test_prepare_artifact_writes_bp_train_metadata(tmp_path):
    output_dir = tmp_path / "prepared"
    custom_py = tmp_path / "custom.py"
    _write_sample_semantics_custom_py(custom_py)
    with pytest.warns(UserWarning, match="bp_format validation reported non-OK status"):
        _prepare_from_collection(
            _make_invalid_collection(), tmp_path, output_dir, custom_py=custom_py
        )

    prepared = load_process_collection(output_dir / "prepared.json")
    metadata = prepared.metadata["bp-train"]

    assert metadata["process_order"] == list(prepared.processes.keys())
    assert metadata["runtime_controls_config"]["initial_grid_points"] >= 2
    # The raw input lives one level above the output dir (tmp_path/prepared-raw.json
    # vs tmp_path/prepared/), so the portable path is recorded relative to output_dir.
    assert metadata["source_input_path"] == "../prepared-raw.json"

    first_name = metadata["process_order"][0]
    process_md = metadata["processes"][first_name]
    assert "sample_acc_name" not in process_md
    assert "name_extras" not in process_md
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
    output_dir = tmp_path / "prepared-minimal"
    custom_py = tmp_path / "custom.py"
    _write_sample_semantics_custom_py(custom_py)
    with pytest.warns(UserWarning):
        _prepare_from_collection(
            _make_invalid_collection(), tmp_path, output_dir, custom_py=custom_py
        )

    prepared = load_process_collection(output_dir / "prepared.json")
    process_md = prepared.metadata["bp-train"]["processes"]["invalid"]

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
                "        process.biological_ode = None",
                "        process.__post_init__()",
                "    return collection",
            ]
        ),
        encoding="utf-8",
    )

    output_dir = tmp_path / "prepared-custom"
    _prepare_from_collection(
        _make_two_process_collection(), tmp_path, output_dir, custom_py=custom_py
    )

    prepared = load_process_collection(output_dir / "prepared.json")
    metadata = prepared.metadata["bp-train"]
    first_name = metadata["process_order"][0]
    process_md = metadata["processes"][first_name]

    assert process_md["name_controlled_PVs"] == ["CF", "T"]


def test_prepare_artifact_can_rename_processes(tmp_path):
    output_dir = tmp_path / "prepared-renamed"
    _prepare_from_collection(
        _make_two_process_collection(),
        tmp_path,
        output_dir,
        prepare_config={"process_rename_map": {"p1": "process=p1", "p2": "process=p2"}},
    )

    prepared = load_process_collection(output_dir / "prepared.json")
    assert list(prepared.processes.keys()) == ["process=p1", "process=p2"]
    assert prepared.processes["process=p1"].metadata.name == "process=p1"
    assert prepared.metadata["bp-train"]["process_order"] == [
        "process=p1",
        "process=p2",
    ]


def test_prepare_artifact_rename_provenance_tracks_changes(tmp_path):
    """Provenance must detect changes even when processes are renamed."""
    output_dir = tmp_path / "prepared-renamed-provenance"
    _prepare_from_collection(
        _make_two_process_collection(),
        tmp_path,
        output_dir,
        prepare_config={
            "process_rename_map": {
                "p1": "process=p1",
                "p2": "process=p2",
            }
        },
    )

    prepared = load_process_collection(output_dir / "prepared.json")
    prov = prepared.metadata["bp-train"]["semantics_provenance"]["processes"]
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
        _prepare_from_collection(
            _make_two_process_collection(),
            tmp_path,
            tmp_path / "prepared-duplicate-renames",
            prepare_config={"process_rename_map": {"p1": "same", "p2": "same"}},
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

    output_dir = tmp_path / "prepared-partial-rename"
    _prepare_from_collection(
        collection,
        tmp_path,
        output_dir,
        prepare_config={"process_rename_map": {"key_p1": "renamed_p1"}},
    )

    prepared = load_process_collection(output_dir / "prepared.json")
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
    output_dir = tmp_path / "prepared-transform-collection"
    _prepare_from_collection(
        _make_two_process_collection(), tmp_path, output_dir, custom_py=custom_py
    )

    prepared = load_process_collection(output_dir / "prepared.json")
    assert list(prepared.processes.keys()) == ["proc::p1", "proc::p2"]
    assert prepared.metadata["collection_transform_marker"] == "applied"
    assert (
        prepared.metadata["bp-train"]["transform_hooks"]["transform_process_collection"]
        == "transform_process_collection"
    )


def test_prepare_artifact_builds_sample_acc_amount_correctly(tmp_path):
    output_dir = tmp_path / "prepared-sample"
    custom_py = tmp_path / "custom.py"
    _write_sample_semantics_custom_py(custom_py)
    with pytest.warns(UserWarning):
        _prepare_from_collection(
            _make_invalid_collection(), tmp_path, output_dir, custom_py=custom_py
        )

    store = ControlsStore.from_json(output_dir / "prepared.json")
    controls = store.get_controls("invalid")
    end_t = _make_invalid_collection().processes["invalid"].time_axis.end
    # V_real cumulative sampled volume at end == sum of sample-event volumes
    # (absolute SampleVolumeChange deltas) at/under end_t.
    sample_times = np.asarray(controls.sample_event_times)[
        np.asarray(controls.sample_event_mask)
    ]
    sample_volumes = np.asarray(controls.sample_event_volumes)[
        np.asarray(controls.sample_event_mask)
    ]
    v_real_end = float(sample_volumes[sample_times <= float(end_t)].sum())
    assert v_real_end == pytest.approx(0.1)


def test_load_raw_collection_accepts_in_memory_collection():
    collection = _make_feed_collection()
    loaded = load_raw_collection(collection)
    assert loaded is collection


def test_prepare_artifact_persists_feed_metadata(tmp_path):
    output_dir = tmp_path / "prepared-feed"
    custom_py = tmp_path / "custom-feed.py"
    _write_feed_semantics_custom_py(custom_py)
    with pytest.warns(UserWarning):
        _prepare_from_collection(
            _make_feed_collection(), tmp_path, output_dir, custom_py=custom_py
        )

    prepared = load_process_collection(output_dir / "prepared.json")
    metadata = prepared.metadata["bp-train"]
    process_md = metadata["processes"]["p1"]
    feed_md = process_md["control_metadata"]["feed_A"]
    semantics = metadata["semantics_provenance"]["processes"]["p1"]

    assert process_md["name_controlled_FVCs"] == ["feed_A"]
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
        _prepare_from_collection(
            _make_invalid_collection(),
            tmp_path,
            tmp_path / "prepared-missing-medium",
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
        _prepare_from_collection(
            _make_feed_collection(),
            tmp_path,
            tmp_path / "prepared-incomplete-feed",
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
    process.biological_ode = None
    process.__post_init__()

    with pytest.raises(
        ValueError,
        match="feed 'feed_A' has no feed-medium component metadata after prep",
    ):
        _prepare_from_collection(collection, tmp_path, tmp_path / "prepared-zero-feed")


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
                "        process.biological_ode = None",
                "        process.__post_init__()",
                "    return collection",
            ]
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="categorised control layout differs"):
        _prepare_from_collection(
            _make_two_process_collection(),
            tmp_path,
            tmp_path / "prepared-bad",
            custom_py=custom_py,
        )


def test_prepare_artifact_fails_on_missing_required_control(tmp_path):
    with pytest.raises(ValueError, match="config-declared controls are missing"):
        _prepare_from_collection(
            _make_two_process_collection(),
            tmp_path,
            tmp_path / "prepared-missing",
            prepare_config={"required_control_names": ["CF"]},
        )


def _write_control_custom_py(path: Path) -> None:
    """Mark CF/T controlled so prepare emits a control-diagnostics plot."""
    path.write_text(
        "\n".join(
            [
                "def transform_process_collection(collection, config):",
                "    for process in collection.processes.values():",
                "        process.process_variables['CF'].is_controlled = True",
                "        process.process_variables['T'].is_controlled = True",
                "        process.biological_ode = None",
                "        process.__post_init__()",
                "    return collection",
            ]
        ),
        encoding="utf-8",
    )


def test_prepare_artifact_writes_output_dir_layout(tmp_path):
    output_dir = tmp_path / "prepared-layout"
    custom_py = tmp_path / "control.py"
    _write_control_custom_py(custom_py)
    _prepare_from_collection(
        _make_two_process_collection(), tmp_path, output_dir, custom_py=custom_py
    )

    assert (output_dir / "prepared.json").is_file()
    assert (output_dir / "prepare_config.json").is_file()
    assert (output_dir / "prepare_diagnostics").is_dir()


def test_prepare_provenance_preserves_nonfinite_values(tmp_path):
    output_dir = tmp_path / "prepared-nonfinite"
    prepared = _prepare_from_collection(
        _make_two_process_collection(),
        tmp_path,
        output_dir,
        prepare_config={"diagnostics": False, "max_rel_error": float("inf")},
    )

    provenance = prepared.metadata[METADATA_NAMESPACE]["provenance"]
    assert math.isinf(provenance["prepare_config"]["max_rel_error"])
    standalone = json.loads(
        (output_dir / "prepare_config.json").read_text(encoding="utf-8")
    )
    assert math.isinf(standalone["provenance"]["prepare_config"]["max_rel_error"])


def test_resolve_prepared_path_dir_vs_file(tmp_path):
    output_dir = tmp_path / "prepared-resolve"
    _prepare_from_collection(_make_two_process_collection(), tmp_path, output_dir)

    # A directory resolves to the bundled prepared.json inside it.
    assert resolve_prepared_path(output_dir) == output_dir / "prepared.json"
    # A plain prepared.json file path passes through unchanged.
    prepared_file = output_dir / "prepared.json"
    assert resolve_prepared_path(prepared_file) == prepared_file


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
    bundle = select_control_sources(process)
    sources = bundle.all_sources
    assert len(sources) == 1
    assert sources[0].name == "feed_A"
    assert sources[0].metadata["inlet_feed_medium"] is None
    assert bundle.name_controlled_FVCs == ("feed_A",)
