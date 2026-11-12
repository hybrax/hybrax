"""Training API with float64 enabled and lazy public re-exports."""

from importlib import import_module
from typing import Any

import jax as _jax

_jax.config.update("jax_enable_x64", True)


_EXPORTS = {
    "ControlsStore": "controls_store",
    "TrainingDataStore": "training_data",
    "PrepareConfig": "run_config",
    "PerProcessControls": "controls_store",
    "PerProcessTrainingData": "training_data",
    "Scaler": "model_api",
    "LinearScaler": "model_api",
    "AffineScaler": "model_api",
    "EstimatedScales": "model_api",
    "ReactionInputs": "model_api",
    "ReactionOutputs": "model_api",
    "LossInputs": "model_api",
    "LossOutputs": "model_api",
    "RateModule": "model_api",
    "UserLossModule": "model_api",
    "partition_trainable": "model_api",
    "trainable_field": "model_api",
    "frozen_field": "model_api",
    "format_trainable_structure": "inspect",
    "format_reaction_schema": "inspect",
    "print_trainable_structure": "inspect",
    "print_reaction_schema": "inspect",
    "HybridOdeWrapper": "wrapper",
    "validate_rhs_ode_compatibility": "wrapper",
    "build_union_time_grid": "dense",
    "dense_point_mask_away_from_jumps": "dense",
    "dense_triple_mask_away_from_jumps": "dense",
    "simulate_measurement_states": "trainer",
    "SingleSampleResult": "trainer",
    "evaluate_sample_with_loss_module": "trainer",
    "DefaultReactionModule": "defaults",
    "DefaultGruReactionModule": "defaults",
    "DefaultLstmReactionModule": "defaults",
    "DefaultStatefulReactionModule": "defaults",
    "DefaultLossModule": "defaults",
    "BoundsViolationLossModule": "bounds_loss",
    "bound_records_from_collection": "bounds_loss",
    "rhs_ode_from_training_parents": "runtime_context",
    "ForwardConfig": "harness",
    "ForwardResult": "harness",
    "TrainHarnessConfig": "harness",
    "TrainHarnessResult": "harness",
    "forward_from_collection": "harness",
    "compute_dense_exports": "harness",
    "train_collection": "harness",
    "train_from_collection": "harness",
    "train_from_prepared_json": "harness",
    "Fold": "loo",
    "FoldResult": "loo",
    "LOOResult": "loo",
    "resolve_folds": "loo",
    "compute_parallel_split": "loo",
    "run_loo_cv": "loo",
    "run_single_fold": "loo",
    "DEFAULT_METRICS": "loo_metrics",
    "compute_aggregated_metrics": "loo_metrics",
    "compute_per_process_metrics": "loo_metrics",
    "load_raw_collection": "prepare",
    "prepare_artifact": "prepare",
    "model_load": "serialization",
    "model_reload": "serialization",
    "model_predict": "harness",
    "reconstruct_training": "serialization",
    "save_model": "serialization",
    "load_trained_wrapper": "serialization",
    "save_opt_state": "serialization",
    "load_opt_state": "serialization",
    "content_hash": "serialization",
    "file_hash": "serialization",
}

__all__ = list(_EXPORTS)


def __getattr__(name: str) -> Any:
    """Lazily import and cache a public name from its owning submodule.

    Args:
        name: Attribute name looked up on this module.

    Raises:
        AttributeError: If ``name`` is not in ``_EXPORTS``.
    """
    if name not in _EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module = import_module(f".{_EXPORTS[name]}", __name__)
    value = getattr(module, name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    """Every already-imported name plus every lazily-exported name."""
    return sorted([*globals(), *_EXPORTS])
