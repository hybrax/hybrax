# --- multi-core bootstrap: MUST run before JAX initialises ---
# OPT-IN. Exposes N CPU devices so training can shard the process batch across
# cores via pmap (~N speedup). Default OFF (1 device, unchanged behaviour) so it
# never competes for cores with other work. Enable with EITHER:
#   * the CLI flag:  bp-train train --devices 8 ...   (pre-scanned from argv here,
#     since the device count must be fixed before JAX initialises)
#   * the env var:   BP_TRAIN_DEVICES=8
# Pick <= number of free cores. No effect on GPU (host-device flag is CPU-only).
import os as _os
import sys as _sys


def _bp_resolve_devices():
    _n = _os.environ.get("BP_TRAIN_DEVICES")
    if _n is not None:
        return _n
    _argv = _sys.argv
    for _i, _a in enumerate(_argv):
        if _a == "--devices" and _i + 1 < len(_argv):
            return _argv[_i + 1]
        if _a.startswith("--devices="):
            return _a.split("=", 1)[1]
    return None


if "xla_force_host_platform_device_count" not in _os.environ.get("XLA_FLAGS", ""):
    _bp_devices = _bp_resolve_devices()
    if _bp_devices is not None:
        if str(_bp_devices).strip().lower() in ("max", "all", "auto"):
            _bp_devices = _os.cpu_count() or 1            # use every CPU core
        else:
            try:
                _bp_devices = int(_bp_devices)
            except (TypeError, ValueError):
                _bp_devices = 1
        if _bp_devices > 1:
            _os.environ["XLA_FLAGS"] = (
                _os.environ.get("XLA_FLAGS", "")
                + f" --xla_force_host_platform_device_count={_bp_devices}"
            ).strip()

from .prepare import PrepareConfig, load_raw_collection, prepare_artifact
from .controls_store import ControlsStore, PerProcessControls
from .training_data import PerProcessTrainingData, TrainingDataStore
from .model_api import (
    EstimatedScales,
    LossInputs,
    LossOutputs,
    ReactionInputs,
    ReactionOutputs,
    UserLossModule,
    UserReactionModule,
    frozen_field,
    partition_trainable,
    trainable_field,
)
from .inspect import (
    format_reaction_schema,
    format_trainable_structure,
    print_reaction_schema,
    print_trainable_structure,
)
from .wrapper import HybridOdeWrapper, validate_rhs_ode_compatibility
from .dense import (
    build_union_time_grid,
    dense_point_mask_away_from_jumps,
    dense_triple_mask_away_from_jumps,
)
from .defaults import DefaultLossModule, DefaultReactionModule
from .trainer import (
    SingleSampleResult,
    evaluate_sample_with_loss_module,
    simulate_measurement_states,
)
from .harness import (
    ForwardConfig,
    ForwardResult,
    TrainHarnessConfig,
    TrainHarnessResult,
    forward_from_collection,
    train_collection,
    train_from_collection,
    train_from_prepared_json,
)
from .loo import (
    FoldResult,
    LOOConfig,
    LOOResult,
    run_loo_cv,
    run_loo_fold,
    run_loo_from_prepared_json,
)
from .loo_metrics import (
    DEFAULT_METRICS,
    compute_aggregated_metrics,
    compute_per_process_metrics,
)

__all__ = [
    "ControlsStore",
    "TrainingDataStore",
    "PrepareConfig",
    "PerProcessControls",
    "PerProcessTrainingData",
    "EstimatedScales",
    "ReactionInputs",
    "ReactionOutputs",
    "LossInputs",
    "LossOutputs",
    "UserReactionModule",
    "UserLossModule",
    "partition_trainable",
    "trainable_field",
    "frozen_field",
    "format_trainable_structure",
    "format_reaction_schema",
    "print_trainable_structure",
    "print_reaction_schema",
    "HybridOdeWrapper",
    "validate_rhs_ode_compatibility",
    "build_union_time_grid",
    "dense_point_mask_away_from_jumps",
    "dense_triple_mask_away_from_jumps",
    "simulate_measurement_states",
    "SingleSampleResult",
    "evaluate_sample_with_loss_module",
    "DefaultReactionModule",
    "DefaultLossModule",
    "ForwardConfig",
    "ForwardResult",
    "TrainHarnessConfig",
    "TrainHarnessResult",
    "forward_from_collection",
    "train_collection",
    "train_from_collection",
    "train_from_prepared_json",
    "FoldResult",
    "LOOConfig",
    "LOOResult",
    "run_loo_cv",
    "run_loo_fold",
    "run_loo_from_prepared_json",
    "DEFAULT_METRICS",
    "compute_aggregated_metrics",
    "compute_per_process_metrics",
    "load_raw_collection",
    "prepare_artifact",
]
