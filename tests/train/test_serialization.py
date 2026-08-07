from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path

import equinox as eqx
import jax
import jax.numpy as jnp
import pytest
from bp_format.dataclasses import (
    BioProcess,
    BioProcessCollection,
    BioProcessMetadata,
    ReactorMedium,
    ReactorMediumComponent,
    SampleVolumeChange,
    TimeAxis,
    TimeSeries,
    Volume,
)

import bp_train.harness as harness
import bp_train.serialization as serialization
from bp_train.harness import _build_optimizer, _build_template_wrapper
from bp_train.model_api import partition_trainable
from bp_train.run_config import (
    DataConfig,
    RunConfig,
    TrainConfig,
    load_train_config,
)
from bp_train.serialization import (
    content_hash,
    load_opt_state,
    load_trained_wrapper,
    reconstruct_run,
    save_model,
    save_opt_state,
)
from bp_train.training_data import TrainingDataStore

# Reuse the tiny single-process reaction module + collection fixtures.
from test_checkpointing import _LinearReactionModule


def _collection(
    biomass_values=(1.0, 0.8, 0.64), *, n_processes: int = 1
) -> BioProcessCollection:
    p1 = BioProcess(
        metadata=BioProcessMetadata(name="p1", process_type="fed_batch"),
        time_axis=TimeAxis(unit="h", start=0.0, end=2.0, time_reference="start"),
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
                        times=jnp.asarray([1.0]),
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
                        times=jnp.asarray([0.0, 1.0, 2.0]),
                        values=jnp.asarray(list(biomass_values)),
                    ),
                ),
            },
        ),
        process_variables={},
    )
    processes = {
        f"p{i}": replace(p1, metadata=replace(p1.metadata, name=f"p{i}"))
        for i in range(1, n_processes + 1)
    }
    return BioProcessCollection(processes=processes, metadata={})


def _build_wrapper(collection: BioProcessCollection):
    store = TrainingDataStore.from_collection(
        collection,
        target_variable_order=["biomass"],
        target_source="reactor_components",
    )
    reaction = _LinearReactionModule()
    wrapper = _build_template_wrapper(
        store,
        reaction_module=reaction,
        selected_processes=("p1",),
        loss_module=None,
    )
    return wrapper


def _grow_controls(wrapper):
    """Return a copy whose controls store has a STRUCTURALLY DIFFERENT (longer)
    dense_grid leaf — simulating a controls store "initialized differently"
    (e.g. trainable DoE values / different grid padding)."""
    grid = wrapper.controls.dense_grid
    grown = jnp.concatenate([grid, jnp.zeros((3,), dtype=grid.dtype)])
    return eqx.tree_at(lambda m: m.controls.dense_grid, wrapper, grown)


def _trainable_arrays(module):
    trainable, _ = partition_trainable(module)
    return [
        leaf
        for leaf in jax.tree_util.tree_leaves(trainable)
        if eqx.is_inexact_array(leaf)
    ]


def test_write_json_emits_valid_json_and_preserves_finite_values(tmp_path: Path):
    path = tmp_path / "config.json"

    serialization.write_json(
        path,
        {
            "status": "complete",
            "finite": 1.25,
            "values": [float("nan"), float("inf"), -float("inf")],
        },
    )

    text = path.read_text(encoding="utf-8")
    assert "NaN" not in text
    assert "Infinity" not in text
    assert json.loads(text) == {
        "status": "complete",
        "finite": 1.25,
        "values": [None, None, None],
    }


def test_read_json_accepts_whole_line_comments(tmp_path: Path):
    path = tmp_path / "config.json"
    path.write_text('// run state\n{"status": "complete"}', encoding="utf-8")

    assert serialization.read_json(path) == {"status": "complete"}


@pytest.mark.parametrize("token", ["NaN", "Infinity", "-Infinity"])
def test_run_config_rejects_bare_nonfinite_tokens(tmp_path: Path, token: str):
    source_path = tmp_path / "source.json"
    source_path.write_text(
        "// non-finite JSON tokens are invalid\n"
        "{\n"
        '  "data": {"prepared": "prepared.json"},\n'
        f'  "train": {{"learning_rate": {token}}}\n'
        "}\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match=str(source_path)):
        load_train_config(source_path)


def test_reconstruct_run_preserves_stateful_opt_in(monkeypatch, tmp_path: Path):
    collection = _collection()
    prepared = tmp_path / "prepared.json"
    prepared.write_text("{}", encoding="utf-8")
    config = RunConfig(
        data=DataConfig(prepared=prepared, targets=("biomass",)),
        train=TrainConfig(allow_stateful_models=True),
    )
    document = {}
    seen = {}

    monkeypatch.setattr(
        serialization, "load_process_collection", lambda _path: collection
    )

    def fake_build_reaction_module(**kwargs):
        seen["allow_stateful_models"] = kwargs["config"].allow_stateful_models
        return object()

    monkeypatch.setattr(harness, "_build_reaction_module", fake_build_reaction_module)
    monkeypatch.setattr(harness, "_build_loss_module", lambda **_kwargs: object())

    reconstruct_run(tmp_path, config, document)

    assert seen["allow_stateful_models"] is True


def test_save_model_excludes_controls_and_roundtrips(tmp_path: Path):
    """save_model writes the trainable partition only; controls are not in it."""
    wrapper = _build_wrapper(_collection())
    path = tmp_path / "params.eqx"
    save_model(wrapper, path)

    # A whole-wrapper dump has strictly more leaves (controls + scales + indices).
    whole_path = tmp_path / "whole.eqx"
    eqx.tree_serialise_leaves(whole_path, wrapper)
    assert whole_path.stat().st_size > path.stat().st_size

    reloaded = load_trained_wrapper(path, template=wrapper)
    src = _trainable_arrays(wrapper)
    dst = _trainable_arrays(reloaded)
    assert len(src) == len(dst) and len(src) > 0
    for a, b in zip(src, dst):
        assert jnp.allclose(a, b)


def test_load_into_structurally_different_controls_template(tmp_path: Path):
    """The bug fix: a controls store that was 'initialized differently' must not
    break loading. The old whole-wrapper approach raises a shape mismatch; the
    new partition approach succeeds because controls come from the template."""
    wrapper = _build_wrapper(_collection())
    different_template = _grow_controls(_build_wrapper(_collection()))

    # OLD behaviour (regression witness): whole-wrapper serialise + deserialise
    # into the differently-shaped template fails.
    whole_path = tmp_path / "whole.eqx"
    eqx.tree_serialise_leaves(whole_path, wrapper)
    with pytest.raises(Exception):
        eqx.tree_deserialise_leaves(whole_path, like=different_template)

    # NEW behaviour: partition save + load succeeds; controls follow the
    # template, trainable leaves follow the file.
    path = tmp_path / "params.eqx"
    save_model(wrapper, path)
    loaded = load_trained_wrapper(path, template=different_template)
    assert (
        loaded.controls.dense_grid.shape == different_template.controls.dense_grid.shape
    )
    src = _trainable_arrays(wrapper)
    dst = _trainable_arrays(loaded)
    for a, b in zip(src, dst):
        assert jnp.allclose(a, b)


def test_opt_state_roundtrip(tmp_path: Path):
    wrapper = _build_wrapper(_collection())
    optimizer = _build_optimizer("adam", 1e-2)
    trainable, _ = partition_trainable(wrapper)
    opt_state = optimizer.init(trainable)

    path = tmp_path / "opt_state.eqx"
    save_opt_state(opt_state, path)

    template_trainable, _ = partition_trainable(_build_wrapper(_collection()))
    opt_template = optimizer.init(template_trainable)
    loaded = load_opt_state(path, template=opt_template)
    # Compare numeric optimizer-state leaves (mu/nu/count); the wrapper-shaped
    # static aux (e.g. zero_nans found_nan treedef) correctly follows the
    # template, which is what resume relies on.
    src = [a for a in jax.tree_util.tree_leaves(opt_state) if eqx.is_array(a)]
    dst = [a for a in jax.tree_util.tree_leaves(loaded) if eqx.is_array(a)]
    assert len(src) == len(dst) and len(src) > 0
    for a, b in zip(src, dst):
        assert a.shape == b.shape and a.dtype == b.dtype
        assert jnp.array_equal(a, b)


def test_content_hash_stable_and_provenance_excluded():
    h1 = content_hash(_collection())
    # Re-built identical collection → identical hash.
    assert content_hash(_collection()) == h1
    # A provenance block under the bp-train namespace must NOT change the hash.
    with_prov = _collection()
    with_prov.metadata["bp-train"] = {
        "provenance": {"timestamp": "2026-06-22T00:00:00"}
    }
    assert content_hash(with_prov) == h1
    # A genuine content change DOES change the hash.
    assert content_hash(_collection(biomass_values=(1.0, 0.5, 0.25))) != h1
