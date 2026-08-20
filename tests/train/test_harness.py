from __future__ import annotations

from copy import deepcopy
import dataclasses
import gc
import json
import logging
import re
import warnings
import weakref
from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import optax
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
from bp_format.mechanistic import build_rhs_ode

import bp_train.harness as harness_module
from bp_train.controls_store import ControlsStore
from bp_train.harness import (
    TrainHarnessConfig,
    _build_batch_index_stream,
    _build_optimizer,
    _build_reaction_module,
    _ensure_process_names,
    _resolve_estimated_scales,
    _target_state_indices,
    prepare_training_from_runtime_artifact,
    _validate_batching_config,
    train_from_collection,
    train_collection,
)
from bp_train.defaults import DefaultLossModule, default_build_reaction_module
from bp_train.runtime_artifact import RuntimeArtifact, RuntimeArtifactFold
from bp_train.runtime_context import (
    ProducerCollectionData,
    RuntimeDataContext,
    select_parent_collection,
)
from bp_train.model_api import (
    AffineScaler,
    EstimatedScales,
    LinearScaler,
    ReactionOutputs,
    UserReactionModule,
    frozen_field,
    trainable_field,
)
from bp_train.training_data import TrainingDataStore


def _runtime_artifact(
    store,
    collection,
    parent_names,
    augmentation_parents=None,
) -> RuntimeArtifact:
    """A loaded artifact standing in for one produced by `write_runtime_artifact`."""
    return RuntimeArtifact(
        "sha256:" + "0" * 64,
        store,
        EstimatedScales(**_DEFAULT_LINEAR_SCALES),
        RuntimeArtifactFold(0, ("holdout",), tuple(store.process_order), "fold-0", 0),
        select_parent_collection(collection, parent_names),
        augmentation_parents
        if augmentation_parents is not None
        else (None,) * len(store.process_order),
    )


def _biomass_loss() -> DefaultLossModule:
    return DefaultLossModule(target_names=["biomass"])


_DEFAULT_LINEAR_SCALES: dict[str, jnp.ndarray] = {
    # Defaults sized for ``_make_collection`` (single biomass species, one sample
    # event, no feeds). Tests with different layouts pass explicit kwargs to
    # override these.
    "SCALE_modeled_RMCs": jnp.ones(1),
    "SCALE_V_in_cumulative": jnp.asarray(1.0),
    "SCALE_modeled_FVCs_cumulative": jnp.ones(0),
    "SCALE_controlled_FVCs_cumulative": jnp.ones(0),
    "SCALE_controlled_FVCs_rates": jnp.ones(0),
    "SCALE_controlled_FVCs_Cin": jnp.ones((0, 1)),
    "SCALE_controlled_PVs": jnp.ones(0),
    "SCALE_modeled_FVCs_Cin": jnp.ones((0, 1)),
    "SCALE_modeled_BiologicalOde_rates": jnp.ones(1),
    "SCALE_modeled_FVCs_rates": jnp.ones(0),
}


class _StatefulHarnessModule(UserReactionModule):
    def __init__(self, **scale_kwargs):
        merged = {
            **_DEFAULT_LINEAR_SCALES,
            "SCALE_latent": jnp.ones(1),
            **scale_kwargs,
        }
        super().__init__(**merged)

    def __call__(self, t, inputs):
        del t
        return ReactionOutputs(
            SCL_modeled_BiologicalOde_rates=jnp.zeros(1),
            SCL_modeled_FVCs_rates=jnp.zeros(0),
            SCL_latent_derivative=jnp.zeros_like(inputs.SCL_latent),
        )


class _LinearReactionModule(UserReactionModule):
    model: eqx.nn.Linear = trainable_field()
    non_model_bias: jax.Array = frozen_field()

    def __init__(self, **scale_kwargs):
        merged = {**_DEFAULT_LINEAR_SCALES, **scale_kwargs}
        super().__init__(**merged)
        self.model = eqx.nn.Linear(1, 1, key=jax.random.key(42))
        self.non_model_bias = jnp.asarray([0.05])

    def __call__(self, t, inputs):
        del t
        SCL_modeled_RMCs = inputs.SCL_modeled_RMCs
        rate = self.model(SCL_modeled_RMCs)[0] + self.non_model_bias[0]
        return ReactionOutputs(
            SCL_modeled_BiologicalOde_rates=jnp.asarray(
                [rate], dtype=SCL_modeled_RMCs.dtype
            ),
            SCL_modeled_FVCs_rates=jnp.zeros((0,), dtype=SCL_modeled_RMCs.dtype),
        )


def _harness_unit_scale_kwargs(collection, process_name: str) -> dict[str, jnp.ndarray]:
    """Build unit SCALE_* kwargs sized to a process / its controls."""
    rhs_ode = build_rhs_ode(collection.processes[process_name])
    controls = ControlsStore.from_collection(collection).get_controls(process_name)
    n_RMCs = len(rhs_ode.name_modeled_RMCs)
    n_FVCs = len(rhs_ode.name_modeled_FVCs)
    n_rates = len(rhs_ode.name_modeled_rates)
    n_FVC = len(controls.name_controlled_FVCs)
    n_PV = len(controls.name_controlled_PVs)
    return {
        "SCALE_modeled_RMCs": jnp.ones(n_RMCs),
        "SCALE_V_in_cumulative": jnp.asarray(1.0),
        "SCALE_modeled_FVCs_cumulative": jnp.ones(n_FVCs),
        "SCALE_controlled_FVCs_cumulative": jnp.ones(n_FVC),
        "SCALE_controlled_FVCs_rates": jnp.ones(n_FVC),
        "SCALE_controlled_FVCs_Cin": jnp.ones((n_FVC, n_RMCs)),
        "SCALE_controlled_PVs": jnp.ones(n_PV),
        "SCALE_modeled_FVCs_Cin": jnp.ones((n_FVCs, n_RMCs)),
        "SCALE_modeled_BiologicalOde_rates": jnp.ones(n_rates),
        "SCALE_modeled_FVCs_rates": jnp.ones(n_FVCs),
    }


class _StatefulCustomModule:
    @staticmethod
    def build_reaction_module(
        *,
        target_names,
        process_names,
        config,
        seed,
        training_parent_collection,
        **scale_kwargs,
    ):
        del target_names, process_names, config, seed, training_parent_collection
        return _StatefulHarnessModule(**scale_kwargs)


def test_build_reaction_module_rejects_stateful_without_opt_in():
    collection = _make_collection()
    store = TrainingDataStore.from_collection(
        collection,
        target_variable_order=["biomass"],
        target_source="reactor_components",
    )

    with pytest.raises(ValueError, match="allow_stateful_models"):
        _build_reaction_module(
            config=TrainHarnessConfig(),
            custom_module=_StatefulCustomModule,
            custom_config={},
            store=store,
            scales=EstimatedScales(**_DEFAULT_LINEAR_SCALES),
            training_parent_collection=collection,
        )


def test_build_reaction_module_accepts_stateful_with_opt_in():
    collection = _make_collection()
    store = TrainingDataStore.from_collection(
        collection,
        target_variable_order=["biomass"],
        target_source="reactor_components",
    )

    module = _build_reaction_module(
        config=TrainHarnessConfig(allow_stateful_models=True),
        custom_module=_StatefulCustomModule,
        custom_config={},
        store=store,
        scales=EstimatedScales(**_DEFAULT_LINEAR_SCALES),
        training_parent_collection=collection,
    )

    assert module.n_latent == 1


def test_train_collection_rejects_direct_stateful_without_opt_in():
    collection = _make_collection()
    store = TrainingDataStore.from_collection(
        collection,
        target_variable_order=["biomass"],
        target_source="reactor_components",
    )

    with pytest.raises(ValueError, match="allow_stateful_models"):
        train_collection(
            store,
            reaction_module=_StatefulHarnessModule(),
            loss_module=_biomass_loss(),
            config=TrainHarnessConfig(epochs=1),
        )


def test_train_collection_accepts_direct_stateful_with_opt_in():
    collection = _make_collection()
    store = TrainingDataStore.from_collection(
        collection,
        target_variable_order=["biomass"],
        target_source="reactor_components",
    )

    result = train_collection(
        store,
        reaction_module=_StatefulHarnessModule(),
        loss_module=_biomass_loss(),
        config=TrainHarnessConfig(
            process_names=("p1",),
            epochs=1,
            batch_size=1,
            allow_stateful_models=True,
        ),
    )

    assert len(result.mean_loss_by_step) == 1


def _make_collection() -> BioProcessCollection:
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
                        values=jnp.asarray([1.0, 0.8, 0.64]),
                    ),
                ),
            },
        ),
        process_variables={},
    )
    p2 = BioProcess(
        metadata=BioProcessMetadata(name="p2", process_type="fed_batch"),
        time_axis=TimeAxis(unit="h", start=0.0, end=2.0, time_reference="start"),
        volume=Volume(
            initial_volume=1.1,
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
                        values=jnp.asarray([0.9, 0.72, 0.58]),
                    ),
                ),
            },
        ),
        process_variables={},
    )
    return BioProcessCollection(processes={"p1": p1, "p2": p2}, metadata={})


def _make_multi_process_collection(n: int) -> BioProcessCollection:
    """``n`` processes sharing ``_make_collection``'s layout (single biomass
    species, one sample event, no feeds/PVs) but with distinct biomass curves.
    """
    processes = {}
    for i in range(1, n + 1):
        name = f"p{i}"
        base = 1.0 + 0.05 * i
        processes[name] = BioProcess(
            metadata=BioProcessMetadata(name=name, process_type="fed_batch"),
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
                            values=jnp.asarray([base, base * 0.8, base * 0.64]),
                        ),
                    ),
                },
            ),
            process_variables={},
        )
    return BioProcessCollection(processes=processes, metadata={})


def _make_feed_mismatch_collection() -> BioProcessCollection:
    """Two processes with different feed compositions (Cin values differ)."""

    def _make_process(name: str, feed_biomass_concentration: float) -> BioProcess:
        feed_medium = FeedMedium(
            name="feed",
            density=1.0,
            density_unit="kg/L",
            components={
                "biomass": FeedMediumComponent(
                    name="biomass",
                    unit="g/L",
                    concentration=StaticVariable(feed_biomass_concentration),
                    is_controlled=False,
                )
            },
        )
        return BioProcess(
            metadata=BioProcessMetadata(name=name, process_type="fed_batch"),
            time_axis=TimeAxis(
                unit="h",
                start=0.0,
                end=2.0,
                time_reference="start",
            ),
            volume=Volume(
                initial_volume=1.0,
                unit="L",
                volume_changes={
                    "feed_A": FeedVolumeChange(
                        name="feed_A",
                        unit="L",
                        is_controlled=True,
                        is_continuous=True,
                        values=TimeSeries(
                            times=jnp.asarray([0.0, 2.0]),
                            values=jnp.asarray([0.0, 0.4]),
                        ),
                        feed_medium=feed_medium,
                    ),
                    "sample_1": SampleVolumeChange(
                        name="sample_1",
                        unit="L",
                        is_controlled=False,
                        is_continuous=False,
                        values=TimeSeries(
                            times=jnp.asarray([1.0]),
                            values=jnp.asarray([-0.1]),
                        ),
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
                        concentration=TimeSeries(
                            times=jnp.asarray([0.0, 2.0]),
                            values=jnp.asarray([1.0, 1.0]),
                        ),
                    ),
                },
            ),
            process_variables={},
        )

    return BioProcessCollection(
        processes={
            "p1": _make_process("p1", 0.0),
            "p2": _make_process("p2", 1.0),
        },
        metadata={},
    )


def _make_combined_target_collection() -> BioProcessCollection:
    process = BioProcess(
        metadata=BioProcessMetadata(name="p1", process_type="batch"),
        time_axis=TimeAxis(unit="h", start=0.0, end=2.0, time_reference="start"),
        volume=Volume(initial_volume=1.0, unit="L", volume_changes={}),
        reactor_medium=ReactorMedium(
            name="rm",
            density=1.0,
            density_unit="kg/L",
            components={
                "biomass": ReactorMediumComponent(
                    name="biomass",
                    unit="g/L",
                    concentration=TimeSeries(
                        times=jnp.asarray([0.0, 2.0]),
                        values=jnp.asarray([1.0, 1.0]),
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
                    times=jnp.asarray([0.0, 2.0]),
                    values=jnp.asarray([0.0, 1.0]),
                ),
            ),
        },
    )
    return BioProcessCollection(processes={"p1": process}, metadata={})


def test_target_state_indices_map_pv_only_targets_to_pv_state_column():
    collection = _make_combined_target_collection()
    store = TrainingDataStore.from_collection(
        collection,
        target_source="process_variables",
    )
    rhs_ode = build_rhs_ode(collection.processes["p1"])

    indices = _target_state_indices(store, rhs_ode)

    assert jnp.array_equal(indices, jnp.asarray([1], dtype=jnp.int32))


def test_default_reaction_module_scales_rmc_axis_not_combined_targets():
    collection = _make_combined_target_collection()

    module = default_build_reaction_module(
        target_names=["biomass", "ratio"],
        process_names=["p1"],
        config=TrainHarnessConfig(),
        seed=0,
        training_parent_collection=collection,
    )

    assert module.SCALE_modeled_RMCs.shape == (1,)
    assert module.SCALE_modeled_PVs.shape == (1,)


def test_train_collection_process_variable_target_uses_full_initial_state():
    collection = _make_combined_target_collection()
    store = TrainingDataStore.from_collection(
        collection,
        target_source="process_variables",
    )
    module = default_build_reaction_module(
        target_names=list(store.name_measured),
        process_names=list(store.process_order),
        config=TrainHarnessConfig(),
        seed=0,
        training_parent_collection=collection,
    )

    result = train_collection(
        store,
        reaction_module=module,
        loss_module=DefaultLossModule(target_names=["ratio"]),
        config=TrainHarnessConfig(process_names=("p1",), epochs=1, batch_size=1),
    )

    assert len(result.mean_loss_by_step) == 1


def test_train_collection_single_process_loss_decreases():
    collection = _make_collection()
    store = TrainingDataStore.from_collection(
        collection,
        target_variable_order=["biomass"],
        target_source="reactor_components",
    )
    result = train_collection(
        store,
        reaction_module=_LinearReactionModule(),
        loss_module=_biomass_loss(),
        config=TrainHarnessConfig(
            process_names=("p1",),
            epochs=8,
            batch_size=1,
            optimizer_name="adam",
            learning_rate=5e-2,
        ),
    )

    assert len(result.mean_loss_by_step) == 8
    assert result.mean_loss_by_step[-1] < result.mean_loss_by_step[0]
    assert result.compile_warmup_seconds > 0.0
    assert len(result.step_time_seconds) == 8
    assert all(dt > 0.0 for dt in result.step_time_seconds)
    assert len(result.batch_process_names_by_step) == 8
    assert all(names == ("p1",) for names in result.batch_process_names_by_step)
    assert result.train_step_rebuild_count == 0
    assert len(result.train_step_input_signature) > 0


def test_train_collection_runs_with_nonzero_affine_state_offset():
    # End-to-end training arm for OP12. Offset=1 centers the initial biomass at
    # SCL=0; the solver, targets, loss, optimizer and trained wrapper all carry
    # the affine scaler through a real update loop.
    collection = _make_collection()
    store = TrainingDataStore.from_collection(
        collection,
        target_variable_order=["biomass"],
        target_source="reactor_components",
    )
    scaler = AffineScaler(
        jnp.asarray([1.0]),
        jnp.asarray([1.0]),
    )
    assert jnp.array_equal(jnp.asarray([1.0]) / scaler, jnp.asarray([0.0]))
    result = train_collection(
        store,
        reaction_module=_LinearReactionModule(SCALE_modeled_RMCs=scaler),
        loss_module=_biomass_loss(),
        config=TrainHarnessConfig(
            process_names=("p1",),
            epochs=3,
            batch_size=1,
            optimizer_name="adam",
            learning_rate=5e-2,
        ),
    )
    trained_scaler = result.trained_wrapper.reaction_module.SCALE_modeled_RMCs
    assert isinstance(trained_scaler, AffineScaler)
    assert jnp.array_equal(trained_scaler.offset, jnp.asarray([1.0]))
    assert result.updates_completed == 3
    assert np.all(np.isfinite(result.mean_loss_by_step))


def test_train_collection_multi_process_tracks_per_process_histories():
    collection = _make_collection()
    store = TrainingDataStore.from_collection(
        collection,
        target_variable_order=["biomass"],
        target_source="reactor_components",
    )
    result = train_collection(
        store,
        reaction_module=_LinearReactionModule(),
        loss_module=_biomass_loss(),
        config=TrainHarnessConfig(
            process_names=("p1", "p2"),
            epochs=5,
            optimizer_name="sgd",
            learning_rate=2e-2,
        ),
    )

    assert len(result.mean_loss_by_step) == 5
    assert all(jnp.isfinite(jnp.asarray(result.mean_loss_by_step)))
    assert len(result.batch_process_names_by_step) == 5
    assert result.compile_warmup_seconds > 0.0
    assert len(result.step_time_seconds) == 5
    assert all(
        set(names) == {"p1", "p2"} for names in result.batch_process_names_by_step
    )
    assert result.train_step_rebuild_count == 0


def test_store_rejects_different_biological_equations():
    collection = _make_collection()
    collection.processes["p2"].biological_ode.derivatives["biomass"] = (
        "2 * q_biomass * biomass"
    )
    collection.processes["p2"].__post_init__()

    with pytest.raises(ValueError, match="biological_ode.derivatives differs"):
        TrainingDataStore.from_collection(
            collection,
            target_variable_order=["biomass"],
            target_source="reactor_components",
        )


def test_store_retains_batched_cin_without_retaining_collection():
    collection = _make_feed_mismatch_collection()
    expected = jnp.stack(
        [
            build_rhs_ode(collection.processes[name]).Cin_controlled_FVCs
            for name in ("p1", "p2")
        ]
    )
    collection_ref = weakref.ref(collection)

    store = TrainingDataStore.from_collection(
        collection,
        target_variable_order=["biomass"],
        target_source="reactor_components",
    )
    del collection
    gc.collect()

    assert collection_ref() is None
    assert store.rhs_ode.name_modeled_RMCs == ("biomass",)
    assert jnp.array_equal(store.Cin_controlled_FVCs, expected)


def test_subset_wrapper_uses_selected_process_cin_and_controls():
    collection = _make_feed_mismatch_collection()
    store = TrainingDataStore.from_collection(
        collection,
        target_variable_order=["biomass"],
        target_source="reactor_components",
    )

    result = train_collection(
        store,
        reaction_module=_LinearReactionModule(
            **_harness_unit_scale_kwargs(collection, "p1")
        ),
        loss_module=_biomass_loss(),
        config=TrainHarnessConfig(
            process_names=("p2",),
            epochs=1,
            batch_size=1,
        ),
    )

    assert result.trained_wrapper.controls.process_name == "p2"
    assert jnp.array_equal(
        result.trained_wrapper.rhs_ode.Cin_controlled_FVCs,
        store.Cin_controlled_FVCs[1],
    )
    assert jnp.array_equal(
        result.trained_wrapper.rhs_ode.Cin_modeled_FVCs,
        store.Cin_modeled_FVCs[1],
    )


def test_train_collection_cycles_batches_across_multiple_batches_per_epoch():
    """5 selected processes with batch_size=2 gives batches_per_epoch=2 (one
    process dropped per epoch as the incomplete tail), exercising per-epoch
    batch cycling that batches_per_epoch == 1 fixtures never touch.
    """
    n_selected = 5
    batch_size = 2
    epochs = 3
    batches_per_epoch = n_selected // batch_size  # 2
    total_updates = epochs * batches_per_epoch  # 6

    collection = _make_multi_process_collection(n_selected)
    process_names = tuple(f"p{i}" for i in range(1, n_selected + 1))
    store = TrainingDataStore.from_collection(
        collection,
        target_variable_order=["biomass"],
        target_source="reactor_components",
    )
    result = train_collection(
        store,
        reaction_module=_LinearReactionModule(),
        loss_module=_biomass_loss(),
        config=TrainHarnessConfig(
            process_names=process_names,
            epochs=epochs,
            batch_size=batch_size,
            shuffle_batches=False,
            optimizer_name="adam",
            learning_rate=5e-2,
        ),
    )

    assert total_updates == epochs * batches_per_epoch
    assert result.updates_completed == total_updates
    assert len(result.mean_loss_by_step) == total_updates
    assert len(result.batch_process_names_by_step) == total_updates

    # Every batch is full (batch_size processes); the dropped tail process
    # never appears in any batch.
    used_names = {
        name for batch in result.batch_process_names_by_step for name in batch
    }
    assert all(len(batch) == batch_size for batch in result.batch_process_names_by_step)
    assert used_names == set(process_names[: batches_per_epoch * batch_size])
    assert process_names[-1] not in used_names

    # With shuffling disabled, batch_in_epoch cycles 1..batches_per_epoch and
    # the exact same batch grouping repeats identically every epoch.
    per_epoch_batches = [
        result.batch_process_names_by_step[
            epoch * batches_per_epoch : (epoch + 1) * batches_per_epoch
        ]
        for epoch in range(epochs)
    ]
    assert all(batches == per_epoch_batches[0] for batches in per_epoch_batches)


def test_train_collection_with_different_cin_per_process():
    """Processes with different feed compositions should train without error."""
    collection = _make_feed_mismatch_collection()
    store = TrainingDataStore.from_collection(
        collection,
        target_variable_order=["biomass"],
        target_source="reactor_components",
    )
    result = train_collection(
        store,
        reaction_module=_LinearReactionModule(
            **_harness_unit_scale_kwargs(collection, "p1")
        ),
        loss_module=_biomass_loss(),
        config=TrainHarnessConfig(
            process_names=("p1", "p2"),
            epochs=4,
            batch_size=2,
            optimizer_name="adam",
            learning_rate=2e-2,
        ),
    )

    assert len(result.mean_loss_by_step) == 4
    assert all(jnp.isfinite(jnp.asarray(result.mean_loss_by_step)))


def test_train_collection_rejects_unknown_process_selection():
    collection = _make_collection()
    store = TrainingDataStore.from_collection(
        collection,
        target_variable_order=["biomass"],
        target_source="reactor_components",
    )
    with pytest.raises(ValueError, match="unknown process names"):
        train_collection(
            store,
            reaction_module=_LinearReactionModule(),
            config=TrainHarnessConfig(process_names=("unknown",), epochs=2),
        )


def test_train_collection_rejects_nonpositive_solver_max_steps():
    collection = _make_collection()
    store = TrainingDataStore.from_collection(
        collection,
        target_variable_order=["biomass"],
        target_source="reactor_components",
    )
    with pytest.raises(ValueError, match="solver_max_steps must be positive"):
        train_collection(
            store,
            reaction_module=_LinearReactionModule(),
            config=TrainHarnessConfig(
                process_names=("p1",),
                epochs=2,
                solver_max_steps=0,
            ),
        )


def test_train_collection_rejects_nonpositive_solver_tolerances():
    collection = _make_collection()
    store = TrainingDataStore.from_collection(
        collection,
        target_variable_order=["biomass"],
        target_source="reactor_components",
    )
    with pytest.raises(ValueError, match="solver_rtol must be positive"):
        train_collection(
            store,
            reaction_module=_LinearReactionModule(),
            config=TrainHarnessConfig(
                process_names=("p1",),
                epochs=2,
                solver_rtol=0.0,
            ),
        )
    with pytest.raises(ValueError, match="solver_atol must be positive"):
        train_collection(
            store,
            reaction_module=_LinearReactionModule(),
            config=TrainHarnessConfig(
                process_names=("p1",),
                epochs=2,
                solver_atol=0.0,
            ),
        )


def test_train_collection_rejects_unsupported_optimizer_name():
    collection = _make_collection()
    store = TrainingDataStore.from_collection(
        collection,
        target_variable_order=["biomass"],
        target_source="reactor_components",
    )
    with pytest.raises(ValueError, match="optimizer_name"):
        train_collection(
            store,
            reaction_module=_LinearReactionModule(),
            config=TrainHarnessConfig(
                process_names=("p1",),
                epochs=2,
                optimizer_name="rmsprop",
            ),
        )


def test_harness_process_name_validation_rejects_duplicates_and_empty():
    collection = _make_collection()
    store = TrainingDataStore.from_collection(
        collection,
        target_variable_order=["biomass"],
        target_source="reactor_components",
    )
    with pytest.raises(ValueError, match="duplicate entries in process_names"):
        _ensure_process_names(store, ("p1", "p1"))
    with pytest.raises(ValueError, match="non-empty"):
        _ensure_process_names(store, ())


def test_build_optimizer_evaluates_schedule_with_int64_count():
    """Prevent Optax's int32 counter from reducing schedule arithmetic to float32.

    A no-decay schedule and scalar rate must produce bitwise-identical float64
    updates; equality fails if the schedule receives Optax's original int32 count.
    The wrapper in ``_build_optimizer`` prevents this (fixed in commit b8ddce1).
    """
    scalar = _build_optimizer("sgd", 0.01, grad_clip_norm=0)
    # decay_rate=1 makes the schedule mathematically identical to the scalar rate.
    schedule = _build_optimizer(
        "sgd",
        optax.exponential_decay(0.01, transition_steps=10, decay_rate=1.0),
        grad_clip_norm=0,
    )
    params = jnp.asarray(1.0, dtype=jnp.float64)
    gradient = jnp.asarray(1.0, dtype=jnp.float64)

    scalar_update, _ = scalar.update(gradient, scalar.init(params))
    schedule_update, _ = schedule.update(gradient, schedule.init(params))

    # Exact equality catches float32 rounding that is later widened to float64.
    assert jnp.array_equal(schedule_update, scalar_update)


def test_harness_phase1_batching_validation_checks_basics():
    cfg = TrainHarnessConfig(
        epochs=3,
        batch_size=None,
        optimizer_name="adam",
        learning_rate=1e-3,
    )
    assert _validate_batching_config(cfg, selected_process_count=2) == 2

    with pytest.raises(ValueError, match="epochs"):
        _validate_batching_config(
            TrainHarnessConfig(epochs=0),
            selected_process_count=2,
        )
    with pytest.raises(ValueError, match="learning_rate"):
        _validate_batching_config(
            TrainHarnessConfig(epochs=1, learning_rate=0.0),
            selected_process_count=2,
        )
    with pytest.raises(ValueError, match="cannot exceed"):
        _validate_batching_config(
            TrainHarnessConfig(epochs=1, batch_size=3),
            selected_process_count=2,
        )
    with pytest.raises(ValueError, match="optimizer_name"):
        _validate_batching_config(
            TrainHarnessConfig(epochs=1, learning_rate=1e-3, optimizer_name="rmsprop"),
            selected_process_count=2,
        )
    with pytest.raises(ValueError, match="effective batch_size"):
        _validate_batching_config(
            TrainHarnessConfig(epochs=1, batch_size=0),
            selected_process_count=2,
        )
    with pytest.raises(ValueError, match="effective batch_size"):
        _validate_batching_config(
            TrainHarnessConfig(epochs=1, batch_size=-1),
            selected_process_count=2,
        )


def test_harness_batch_stream_preserves_order_and_drops_each_epoch_tail():
    stream = _build_batch_index_stream(
        selected_process_indices=jnp.asarray([5, 7, 9, 11, 13], dtype=jnp.int32),
        epochs=2,
        batch_size=2,
        shuffle_batches=False,
        batch_seed=None,
        seed=123,
    )
    assert tuple(stream.shape) == (4, 2)
    assert stream.tolist() == [[5, 7], [9, 11], [5, 7], [9, 11]]


def test_harness_batch_stream_shuffle_is_deterministic_and_seeded():
    kwargs = dict(
        selected_process_indices=jnp.asarray([0, 1, 2], dtype=jnp.int32),
        epochs=4,
        batch_size=2,
        shuffle_batches=True,
    )
    s1 = _build_batch_index_stream(batch_seed=999, seed=123, **kwargs)
    s2 = _build_batch_index_stream(batch_seed=999, seed=777, **kwargs)
    s3 = _build_batch_index_stream(batch_seed=1000, seed=123, **kwargs)
    assert s1.tolist() == s2.tolist()
    assert s1.tolist() != s3.tolist()
    for epoch in np.asarray(s1).reshape(4, -1):
        assert len(set(epoch.tolist())) == 2


def test_harness_batch_stream_falls_back_to_seed_when_batch_seed_none():
    kwargs = dict(
        selected_process_indices=jnp.asarray([0, 1, 2], dtype=jnp.int32),
        epochs=3,
        batch_size=3,
        shuffle_batches=True,
        batch_seed=None,
    )
    s1 = _build_batch_index_stream(seed=42, **kwargs)
    s2 = _build_batch_index_stream(seed=42, **kwargs)
    s3 = _build_batch_index_stream(seed=43, **kwargs)
    assert s1.tolist() == s2.tolist()
    assert s1.tolist() != s3.tolist()


def test_harness_config_has_no_drop_last_batch_field():
    assert "drop_last_batch" not in TrainHarnessConfig.__dataclass_fields__


def test_train_collection_signature_is_stable_and_no_rebuilds():
    collection = _make_collection()
    store = TrainingDataStore.from_collection(
        collection,
        target_variable_order=["biomass"],
        target_source="reactor_components",
    )
    result = train_collection(
        store,
        reaction_module=_LinearReactionModule(),
        config=TrainHarnessConfig(
            process_names=("p1", "p2"),
            epochs=4,
            batch_size=2,
            optimizer_name="adam",
            learning_rate=2e-2,
        ),
    )

    assert len(result.train_step_input_signature) > 0
    assert result.train_step_rebuild_count == 0
    assert result.updates_completed == 4


def test_train_collection_logs_sampled_losses_only_at_log_steps(caplog):
    collection = _make_collection()
    store = TrainingDataStore.from_collection(
        collection,
        target_variable_order=["biomass"],
        target_source="reactor_components",
    )
    caplog.set_level(logging.INFO, logger="bp_train.harness")
    train_collection(
        store,
        reaction_module=_LinearReactionModule(),
        config=TrainHarnessConfig(
            process_names=("p1", "p2"),
            epochs=4,
            batch_size=2,
            optimizer_name="adam",
            learning_rate=2e-2,
        ),
    )

    # The tabular layout emits one row per update.
    # " HH:MM:SS | <step> | <loss> | <tgt> | <dt> ", surrounded by an
    # initial header (column-name row + separator row) that re-prints every
    # `header_every` steps. Filter to data rows by matching the leading
    # whitespace + clock pattern.
    row_re = re.compile(r"^\s\d{2}:\d{2}:\d{2}\s\|")
    step_rows = [
        record.message for record in caplog.records if row_re.match(record.message)
    ]
    assert len(step_rows) == 4
    sampled_logs = [
        record.message for record in caplog.records if "per-process:" in record.message
    ]
    assert sampled_logs == []


def test_holdout_runs_once_at_periodic_final_collision(tmp_path, monkeypatch):
    assert harness_module._BATCHED_LOSS_FN_JIT is not harness_module._BATCHED_LOSS_FN
    jit_calls = []
    batched_loss_jit = harness_module._BATCHED_LOSS_FN_JIT

    def record_jit_call(*args, **kwargs):
        jit_calls.append(kwargs)
        return batched_loss_jit(*args, **kwargs)

    monkeypatch.setattr(harness_module, "_BATCHED_LOSS_FN_JIT", record_jit_call)
    collection = _make_collection()
    store = TrainingDataStore.from_collection(
        collection,
        target_variable_order=["biomass"],
        target_source="reactor_components",
    )
    result = train_collection(
        store,
        reaction_module=_LinearReactionModule(),
        loss_module=_biomass_loss(),
        config=TrainHarnessConfig(
            process_names=("p1",),
            holdout_processes=("p2",),
            epochs=2,
            batch_size=1,
            checkpoint_dir=tmp_path / "checkpoints",
            checkpoint_every=2.0,
        ),
    )

    assert set(result.holdout_loss_by_step) == {2}
    holdout_steps = [
        kwargs["step"] for kwargs in jit_calls if kwargs["step"] is not None
    ]
    assert len(holdout_steps) == 1
    assert int(holdout_steps[0]) == 2
    assert [p.name for p in (tmp_path / "checkpoints").glob("step_*")] == ["step_00002"]
    state = json.loads(
        (tmp_path / "checkpoints" / "latest" / "train_state.json").read_text()
    )
    assert state["holdout_loss"] == pytest.approx(result.holdout_loss_by_step[2])


def test_holdout_batches_weight_valid_samples_and_ignore_padding(tmp_path, monkeypatch):
    collection = _make_collection()
    for name in ("p3", "p4"):
        process = deepcopy(collection.processes["p2"])
        process.metadata.name = name
        collection.processes[name] = process
    store = TrainingDataStore.from_collection(
        collection,
        target_variable_order=["biomass"],
        target_source="reactor_components",
    )
    holdout_names = ("p2", "p3", "p4")

    def fake_batches(wrapper, received_store, process_names, **kwargs):
        del wrapper, kwargs
        assert received_store is store
        assert process_names == holdout_names
        yield (
            process_names[:2],
            (
                jnp.asarray(0.0),
                jnp.asarray([[0.0], [0.0]]),
                jnp.asarray([0.0, 0.0]),
                None,
                None,
                jnp.asarray([jnp.inf, jnp.inf]),
            ),
        )
        yield (
            process_names[2:],
            (
                jnp.asarray(514.5),
                jnp.asarray([[30.0], [999.0]]),
                jnp.asarray([30.0, 999.0]),
                None,
                None,
                jnp.asarray([jnp.inf, jnp.inf]),
            ),
        )

    monkeypatch.setattr(harness_module, "_iter_batched_loss_outputs", fake_batches)
    monkeypatch.setattr(
        harness_module,
        "compute_dense_exports",
        lambda *args, **kwargs: (np.zeros(1), np.zeros((1, 1)), {}),
    )
    result = train_collection(
        store,
        reaction_module=_LinearReactionModule(),
        loss_module=_biomass_loss(),
        config=TrainHarnessConfig(
            process_names=("p1",),
            holdout_processes=holdout_names,
            epochs=1,
            batch_size=1,
            checkpoint_dir=tmp_path / "checkpoints",
            checkpoint_every=1.0,
        ),
    )

    assert result.holdout_loss_by_step[1] == pytest.approx(10.0)
    assert result.holdout_per_target_by_step[1] == pytest.approx((10.0,))


def test_train_from_collection_warns_and_logs_when_targets_default(monkeypatch, caplog):
    collection = _make_collection()

    class _DummyStore:
        name_measured_RMCs = ("biomass",)
        name_measured_PVs: tuple[str, ...] = ()
        name_measured = ("biomass",)
        process_order = ("p1", "p2")

    def fake_from_collection(collection, *, target_variable_order, target_source):
        del collection, target_variable_order, target_source
        return _DummyStore()

    monkeypatch.setattr(
        "bp_train.harness.TrainingDataStore.from_collection",
        fake_from_collection,
    )
    monkeypatch.setattr("bp_train.harness.load_custom_module", lambda _p: object())
    monkeypatch.setattr("bp_train.harness.resolve_config", lambda _m, _r: {})
    monkeypatch.setattr(
        "bp_train.harness._ensure_process_names", lambda _s, _n: ("p1",)
    )
    monkeypatch.setattr(
        "bp_train.harness.ProducerCollectionData.from_collection",
        lambda store, _collection: ProducerCollectionData(
            store, (None,) * len(store.process_order), (), (), (), ()
        ),
    )
    monkeypatch.setattr(
        "bp_train.harness.ProducerCollectionData.select_training_parents",
        lambda self, collection, *_args: RuntimeDataContext(
            self.training_data,
            select_parent_collection(collection, ("p1",)),
            (),
            (),
            (),
            (),
        ),
    )
    monkeypatch.setattr(
        "bp_train.harness._resolve_estimated_scales",
        lambda **_kw: EstimatedScales(**_DEFAULT_LINEAR_SCALES),
    )
    monkeypatch.setattr(
        "bp_train.harness._build_reaction_module", lambda **_kw: object()
    )
    monkeypatch.setattr("bp_train.harness._build_loss_module", lambda **_kw: object())
    monkeypatch.setattr(
        "bp_train.harness.train_collection",
        lambda *args, **kwargs: "train-result",
    )

    caplog.set_level(logging.INFO, logger="bp_train.harness")
    with pytest.warns(UserWarning, match="No training targets specified"):
        result = train_from_collection(
            collection,
            config=TrainHarnessConfig(target_variable_order=None, epochs=1),
            custom_py=None,
            runtime_config=None,
        )

    assert result == "train-result"
    assert "Training targets: ('biomass',)" in caplog.text
    assert "train hooks detected: none" in caplog.text
    assert (
        "train hooks default: estimate_all_scales, build_reaction_module, "
        "build_learning_rate, build_optimizer, build_loss_module" in caplog.text
    )


def test_train_from_collection_uses_custom_config_targets_without_warning(
    monkeypatch, caplog
):
    collection = _make_collection()
    captured: dict[str, object] = {}

    class _DummyStore:
        name_measured_RMCs = ("cfg_biomass",)
        name_measured_PVs: tuple[str, ...] = ()
        name_measured = ("cfg_biomass",)
        process_order = ("p1", "p2")

    def fake_from_collection(collection, *, target_variable_order, target_source):
        del collection, target_source
        captured["target_variable_order"] = target_variable_order
        return _DummyStore()

    monkeypatch.setattr(
        "bp_train.harness.TrainingDataStore.from_collection",
        fake_from_collection,
    )
    monkeypatch.setattr("bp_train.harness.load_custom_module", lambda _p: object())
    monkeypatch.setattr(
        "bp_train.harness.resolve_config",
        lambda _m, _r: {"target_variable_order": ["cfg_biomass"]},
    )
    monkeypatch.setattr(
        "bp_train.harness._ensure_process_names", lambda _s, _n: ("p1",)
    )
    monkeypatch.setattr(
        "bp_train.harness.ProducerCollectionData.from_collection",
        lambda store, _collection: ProducerCollectionData(
            store, (None,) * len(store.process_order), (), (), (), ()
        ),
    )
    monkeypatch.setattr(
        "bp_train.harness.ProducerCollectionData.select_training_parents",
        lambda self, collection, *_args: RuntimeDataContext(
            self.training_data,
            select_parent_collection(collection, ("p1",)),
            (),
            (),
            (),
            (),
        ),
    )
    monkeypatch.setattr(
        "bp_train.harness._resolve_estimated_scales",
        lambda **_kw: EstimatedScales(**_DEFAULT_LINEAR_SCALES),
    )
    monkeypatch.setattr(
        "bp_train.harness._build_reaction_module", lambda **_kw: object()
    )
    monkeypatch.setattr("bp_train.harness._build_loss_module", lambda **_kw: object())
    monkeypatch.setattr(
        "bp_train.harness.train_collection",
        lambda *args, **kwargs: "train-result",
    )

    caplog.set_level(logging.INFO, logger="bp_train.harness")
    with warnings.catch_warnings(record=True) as warns:
        warnings.simplefilter("always")
        result = train_from_collection(
            collection,
            config=TrainHarnessConfig(target_variable_order=None, epochs=1),
            custom_py="custom.py",
            runtime_config=None,
        )

    assert result == "train-result"
    assert len(warns) == 0
    assert captured["target_variable_order"] == ("cfg_biomass",)
    assert "Training targets: ('cfg_biomass',)" in caplog.text


def _patch_train_from_collection_deps(monkeypatch, custom_module, captured):
    """Shared monkeypatch setup for hook wiring tests."""

    class _DummyStore:
        name_measured_RMCs = ("biomass",)
        name_measured_PVs: tuple[str, ...] = ()
        name_measured = ("biomass",)
        process_order = ("p1", "p2")

    def fake_from_collection(collection, *, target_variable_order, target_source):
        del collection, target_variable_order, target_source
        return _DummyStore()

    monkeypatch.setattr(
        "bp_train.harness.TrainingDataStore.from_collection",
        fake_from_collection,
    )
    monkeypatch.setattr(
        "bp_train.harness.load_custom_module",
        lambda _p: custom_module,
    )
    monkeypatch.setattr("bp_train.harness.resolve_config", lambda _m, _r: {})
    monkeypatch.setattr(
        "bp_train.harness._ensure_process_names", lambda _s, _n: ("p1",)
    )
    monkeypatch.setattr(
        "bp_train.harness.ProducerCollectionData.from_collection",
        lambda store, _collection: ProducerCollectionData(
            store, (None,) * len(store.process_order), (), (), (), ()
        ),
    )

    def select_training_parents(runtime_data, collection, process_names):
        captured["scale_process_names"] = tuple(process_names)
        return RuntimeDataContext(
            runtime_data.training_data,
            select_parent_collection(collection, tuple(process_names)),
            (),
            (),
            (),
            (),
        )

    monkeypatch.setattr(
        "bp_train.harness.ProducerCollectionData.select_training_parents",
        select_training_parents,
    )
    monkeypatch.setattr(
        "bp_train.harness._resolve_estimated_scales",
        lambda **_kw: EstimatedScales(**_DEFAULT_LINEAR_SCALES),
    )
    monkeypatch.setattr(
        "bp_train.harness._build_reaction_module", lambda **_kw: object()
    )
    monkeypatch.setattr("bp_train.harness._build_loss_module", lambda **_kw: object())

    def fake_train_collection(*args, **kwargs):
        del args
        captured["optimizer"] = kwargs.get("optimizer")
        captured["loss_module"] = kwargs.get("loss_module")
        return "train-result"

    monkeypatch.setattr("bp_train.harness.train_collection", fake_train_collection)


def test_train_from_collection_wires_build_optimizer_hook(monkeypatch):
    collection = _make_collection()
    captured: dict[str, object] = {}
    sentinel = optax.adam(1e-3)

    class _CustomModule:
        @staticmethod
        def build_optimizer(config, train_cfg):
            del config, train_cfg
            return sentinel

    _patch_train_from_collection_deps(monkeypatch, _CustomModule(), captured)

    result = train_from_collection(
        collection,
        config=TrainHarnessConfig(target_variable_order=("biomass",), epochs=1),
        custom_py="custom.py",
        runtime_config=None,
    )

    assert result == "train-result"
    assert captured["scale_process_names"] == ("p1",)
    assert captured["optimizer"] is sentinel


def test_learning_rate_hook_receives_derived_update_budget(monkeypatch):
    collection = _make_collection()
    captured: dict[str, object] = {}
    seen: dict[str, int] = {}

    class _CustomModule:
        @staticmethod
        def build_learning_rate(config, train_cfg, total_updates):
            del config, train_cfg
            seen["total_updates"] = total_updates
            return 1e-3

    _patch_train_from_collection_deps(monkeypatch, _CustomModule(), captured)
    monkeypatch.setattr(
        "bp_train.harness._ensure_process_names", lambda _store, _names: ("p1", "p2")
    )

    train_from_collection(
        collection,
        config=TrainHarnessConfig(
            target_variable_order=("biomass",), epochs=3, batch_size=1
        ),
        custom_py="custom.py",
    )

    assert seen["total_updates"] == 6


def test_train_from_collection_uses_default_optimizer_when_no_hook(monkeypatch):
    collection = _make_collection()
    captured: dict[str, object] = {}

    class _CustomModule:
        """No build_optimizer hook -> default _build_optimizer path."""

    _patch_train_from_collection_deps(monkeypatch, _CustomModule(), captured)

    result = train_from_collection(
        collection,
        config=TrainHarnessConfig(target_variable_order=("biomass",), epochs=1),
        custom_py="custom.py",
        runtime_config=None,
    )

    assert result == "train-result"
    # Optimizer construction is centralized in build_optimizer_for_run (shared
    # with serialization.load_run for resume): with no build_optimizer hook the
    # default chain is built eagerly and passed concretely (no longer None).
    import optax

    assert isinstance(captured["optimizer"], optax.GradientTransformation)


def test_build_loss_module_discovers_custom_hook():
    from bp_train.harness import _build_loss_module

    collection = _make_collection()
    store = TrainingDataStore.from_collection(
        collection,
        target_variable_order=["biomass"],
        target_source="reactor_components",
    )
    sentinel = DefaultLossModule(target_names=["biomass"])
    seen: dict[str, object] = {}

    class _CustomModule:
        @staticmethod
        def build_loss_module(
            *,
            target_names,
            process_names,
            config,
            seed,
            training_parent_collection,
        ):
            del process_names, config, seed, training_parent_collection
            seen["target_names"] = tuple(target_names)
            return sentinel

    module = _build_loss_module(
        config=TrainHarnessConfig(epochs=1),
        custom_module=_CustomModule(),
        custom_config={},
        store=store,
        training_parent_collection=collection,
    )
    assert module is sentinel
    assert seen["target_names"] == ("biomass",)


def test_build_loss_module_defaults_when_no_hook():
    from bp_train.harness import _build_loss_module

    collection = _make_collection()
    store = TrainingDataStore.from_collection(
        collection,
        target_variable_order=["biomass"],
        target_source="reactor_components",
    )
    module = _build_loss_module(
        config=TrainHarnessConfig(epochs=1),
        custom_module=None,
        custom_config={},
        store=store,
        training_parent_collection=collection,
    )
    assert isinstance(module, DefaultLossModule)
    assert tuple(module.loss_names) == ("biomass",)


def test_prepare_training_from_runtime_artifact_never_constructs_or_scales(
    monkeypatch, caplog
):
    collection = _make_collection()
    store = TrainingDataStore.from_collection(
        collection,
        target_variable_order=["biomass"],
        target_source="reactor_components",
    )

    monkeypatch.setattr(
        "bp_train.harness.ProducerCollectionData.from_collection",
        lambda *_args, **_kwargs: pytest.fail("constructed runtime data"),
    )
    monkeypatch.setattr(
        "bp_train.harness._resolve_estimated_scales",
        lambda **_kwargs: pytest.fail("estimated scales"),
    )

    caplog.set_level(logging.INFO, logger="bp_train.harness")
    prepared = prepare_training_from_runtime_artifact(
        _runtime_artifact(store, collection, ("p1",)),
        config=TrainHarnessConfig(process_names=("p1",), epochs=1),
        custom_module=None,
        custom_cfg={},
    )

    assert prepared.store is store
    assert prepared.config.process_names == ("p1",)
    assert "train hooks detected: none" in caplog.text
    assert (
        "train hooks default: estimate_all_scales, build_reaction_module, "
        "build_learning_rate, build_optimizer, build_loss_module" in caplog.text
    )


def test_prepare_training_preserves_all_original_prediction_parents():
    collection = _make_multi_process_collection(3)
    store = TrainingDataStore.from_collection(
        collection,
        target_variable_order=["biomass"],
        target_source="reactor_components",
    )
    artifact = _runtime_artifact(
        store, collection, ("p1",), augmentation_parents=(None, None, "p1")
    )

    prepared = prepare_training_from_runtime_artifact(
        artifact,
        config=TrainHarnessConfig(process_names=("p3",), epochs=1),
        custom_module=None,
        custom_cfg={},
    )

    assert prepared.config.process_names == ("p3",)
    assert prepared.prediction_parent_process_names == ("p1", "p2")


def test_constructor_hooks_receive_selected_processes_and_represented_parents():
    collection = _make_multi_process_collection(3)
    store = TrainingDataStore.from_collection(
        collection,
        target_variable_order=["biomass"],
        target_source="reactor_components",
    )
    artifact = _runtime_artifact(
        store, collection, ("p1", "p2"), augmentation_parents=(None, None, "p1")
    )
    seen = []

    class CustomModule:
        @staticmethod
        def build_reaction_module(
            *,
            target_names,
            process_names,
            config,
            seed,
            training_parent_collection,
            **scale_kwargs,
        ):
            seen.append(
                (
                    "reaction",
                    tuple(process_names),
                    tuple(training_parent_collection.processes),
                )
            )
            return default_build_reaction_module(
                target_names=target_names,
                process_names=process_names,
                config=config,
                seed=seed,
                training_parent_collection=training_parent_collection,
                **scale_kwargs,
            )

        @staticmethod
        def build_loss_module(
            *,
            target_names,
            process_names,
            config,
            seed,
            training_parent_collection,
        ):
            del config, seed
            seen.append(
                (
                    "loss",
                    tuple(process_names),
                    tuple(training_parent_collection.processes),
                )
            )
            return DefaultLossModule(target_names=target_names)

    prepare_training_from_runtime_artifact(
        artifact,
        config=TrainHarnessConfig(process_names=("p3", "p2"), epochs=1),
        custom_module=CustomModule,
        custom_cfg={},
    )

    # process_names keeps the caller's order and its augmented child; the parents
    # are deduplicated into canonical order, so the two tuples differ in order as
    # well as in content.
    assert seen == [
        ("reaction", ("p3", "p2"), ("p1", "p2")),
        ("loss", ("p3", "p2"), ("p1", "p2")),
    ]


def test_parent_collection_mismatch_fails_before_constructor_hooks():
    collection = _make_multi_process_collection(2)
    store = TrainingDataStore.from_collection(
        collection,
        target_variable_order=["biomass"],
        target_source="reactor_components",
    )

    class CustomModule:
        @staticmethod
        def build_reaction_module(
            *,
            target_names,
            process_names,
            config,
            seed,
            training_parent_collection,
            **scale_kwargs,
        ):
            pytest.fail("reaction hook called")

        @staticmethod
        def build_loss_module(
            *,
            target_names,
            process_names,
            config,
            seed,
            training_parent_collection,
        ):
            pytest.fail("loss hook called")

    with pytest.raises(ValueError, match="keys differ from represented parents"):
        prepare_training_from_runtime_artifact(
            _runtime_artifact(store, collection, ("p2",)),
            config=TrainHarnessConfig(process_names=("p1",), epochs=1),
            custom_module=CustomModule,
            custom_cfg={},
        )


@pytest.mark.parametrize("construction_path", ["runtime", "artifact"])
def test_reaction_hook_mutating_parent_collection_fails_before_loss_hook(
    construction_path,
):
    collection = _make_multi_process_collection(2)
    store = TrainingDataStore.from_collection(
        collection,
        target_variable_order=["biomass"],
        target_source="reactor_components",
    )

    class CustomModule:
        @staticmethod
        def build_reaction_module(
            *,
            target_names,
            process_names,
            config,
            seed,
            training_parent_collection,
            **scale_kwargs,
        ):
            module = default_build_reaction_module(
                target_names=target_names,
                process_names=process_names,
                config=config,
                seed=seed,
                training_parent_collection=training_parent_collection,
                **scale_kwargs,
            )
            training_parent_collection.processes.pop("p2")
            return module

        @staticmethod
        def build_loss_module(**kwargs):
            pytest.fail("loss hook called")

    config = TrainHarnessConfig(process_names=("p1", "p2"), epochs=1)
    with pytest.raises(ValueError, match="keys differ from represented parents"):
        if construction_path == "runtime":
            harness_module._build_runtime_modules(
                store=store,
                collection=collection,
                config=config,
                custom_module=CustomModule,
                custom_config={},
            )
        else:
            prepare_training_from_runtime_artifact(
                _runtime_artifact(store, collection, ("p1", "p2")),
                config=config,
                custom_module=CustomModule,
                custom_cfg={},
            )


def test_build_runtime_modules_selects_scale_processes(monkeypatch):
    collection = _make_collection()
    store = TrainingDataStore.from_collection(
        collection,
        target_variable_order=["biomass"],
        target_source="reactor_components",
    )
    runtime_data = ProducerCollectionData.from_collection(store, collection)
    seen = {}
    select_training_parents = ProducerCollectionData.select_training_parents

    def select(self, received_collection, process_names):
        seen["selection"] = (received_collection, process_names)
        return select_training_parents(self, received_collection, process_names)

    monkeypatch.setattr(
        "bp_train.harness.ProducerCollectionData.from_collection",
        lambda *_args: runtime_data,
    )
    monkeypatch.setattr(ProducerCollectionData, "select_training_parents", select)
    scales = EstimatedScales(**_DEFAULT_LINEAR_SCALES)

    def resolve_scales(**kwargs):
        seen["scale_data"] = kwargs["runtime_data"]
        return scales

    monkeypatch.setattr("bp_train.harness._resolve_estimated_scales", resolve_scales)
    sentinel = object()

    def build_reaction_module(**kwargs):
        seen["scales"] = kwargs["scales"]
        seen["training_parent_collection"] = kwargs["training_parent_collection"]
        return sentinel

    monkeypatch.setattr(
        "bp_train.harness._build_reaction_module", build_reaction_module
    )

    reaction_module, loss_module = harness_module._build_runtime_modules(
        store=store,
        collection=collection,
        config=TrainHarnessConfig(process_names=("p1",), epochs=1),
        custom_module=None,
        custom_config={},
        build_loss=False,
    )

    assert reaction_module is sentinel
    assert loss_module is None
    assert seen["selection"] == (collection, ("p1",))
    assert seen["scale_data"].process_order == ("p1",)
    assert seen["scales"] is scales
    assert tuple(seen["training_parent_collection"].processes) == ("p1",)


def test_scale_hook_mutating_parent_collection_fails_before_constructor_hooks():
    """The guard in `_build_runtime_modules` exists for exactly this case.

    `estimate_all_scales` runs before the constructor hooks and is handed the same
    mutable parent collection, so it can desynchronize the collection from the
    process order the scales were selected for. `TrainingDataStore.select_processes`
    has already run by then, so nothing else catches it.
    """
    collection = _make_multi_process_collection(2)
    store = TrainingDataStore.from_collection(
        collection,
        target_variable_order=["biomass"],
        target_source="reactor_components",
    )

    class CustomModule:
        @staticmethod
        def estimate_all_scales(data, target_names, config):
            del target_names, config
            data.training_parent_collection.processes.pop("p2")
            return EstimatedScales(
                **{
                    field.name: jnp.zeros(())
                    for field in dataclasses.fields(EstimatedScales)
                }
            )

        @staticmethod
        def build_reaction_module(**kwargs):
            pytest.fail("reaction hook called")

        @staticmethod
        def build_loss_module(**kwargs):
            pytest.fail("loss hook called")

    with pytest.raises(ValueError, match="keys differ from represented parents"):
        harness_module._build_runtime_modules(
            store=store,
            collection=collection,
            config=TrainHarnessConfig(process_names=("p1", "p2"), epochs=1),
            custom_module=CustomModule,
            custom_config={},
        )


def test_resolve_estimated_scales_receives_runtime_data():
    collection = _make_collection()
    store = TrainingDataStore.from_collection(
        collection,
        target_variable_order=["biomass"],
        target_source="reactor_components",
    )
    runtime_data = RuntimeDataContext(
        store, select_parent_collection(collection, ("p1",)), (), (), (), ()
    )
    scales = EstimatedScales(
        **{field.name: jnp.zeros(()) for field in dataclasses.fields(EstimatedScales)}
    )
    seen: dict[str, Any] = {}

    class CustomModule:
        @staticmethod
        def estimate_all_scales(data, target_names, config):
            seen["data"] = data
            return scales

    estimated = _resolve_estimated_scales(
        custom_module=CustomModule,
        runtime_data=runtime_data,
        custom_cfg={},
    )
    assert seen == {"data": runtime_data}
    for field in dataclasses.fields(EstimatedScales):
        assert isinstance(getattr(estimated, field.name), LinearScaler)
