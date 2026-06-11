from __future__ import annotations

from importlib import import_module
from typing import Any

_EXPORTS = {
    "ControlsStore": "controls_store",
    "TrainingDataStore": "training_data",
    "PrepareConfig": "run_config",
    "PerProcessControls": "controls_store",
    "PerProcessTrainingData": "training_data",
    "EstimatedScales": "model_api",
    "ReactionInputs": "model_api",
    "ReactionOutputs": "model_api",
    "LossInputs": "model_api",
    "LossOutputs": "model_api",
    "UserReactionModule": "model_api",
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
    "DefaultLossModule": "defaults",
    "ForwardConfig": "harness",
    "ForwardResult": "harness",
    "TrainHarnessConfig": "harness",
    "TrainHarnessResult": "harness",
    "forward_from_collection": "harness",
    "train_collection": "harness",
    "train_from_collection": "harness",
    "train_from_prepared_json": "harness",
    "FoldResult": "loo",
    "LOOConfig": "loo",
    "LOOResult": "loo",
    "run_loo_cv": "loo",
    "run_loo_fold": "loo",
    "run_loo_from_prepared_json": "loo",
    "DEFAULT_METRICS": "loo_metrics",
    "compute_aggregated_metrics": "loo_metrics",
    "compute_per_process_metrics": "loo_metrics",
    "load_raw_collection": "prepare",
    "prepare_artifact": "prepare",
}

__all__ = list(_EXPORTS)


def __getattr__(name: str) -> Any:
    if name not in _EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module = import_module(f".{_EXPORTS[name]}", __name__)
    value = getattr(module, name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted([*globals(), *_EXPORTS])
