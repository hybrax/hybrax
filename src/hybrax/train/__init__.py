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


def _bp_count_processes():
    """Best-effort process count from the ``--input`` prepared JSON (pre-JAX).

    Used to resolve ``--devices max`` to ``min(n_processes, n_cpus)`` instead of every
    core: exposing more CPU devices than there are processes leaves them idle but still
    oversubscribes the XLA collective threadpool, which can starve the pmap rendezvous
    (~20 s) and deadlock mid-training. Returns ``None`` if it can't be determined (then
    the caller falls back to ``cpu_count``)."""
    _argv = _sys.argv
    _path = None
    for _i, _a in enumerate(_argv):
        if _a == "--input" and _i + 1 < len(_argv):
            _path = _argv[_i + 1]; break
        if _a.startswith("--input="):
            _path = _a.split("=", 1)[1]; break
    if not _path:
        return None
    try:
        import json as _json
        with open(_path) as _f:
            _d = _json.load(_f)
        for _k in ("processes", "process_order", "case_studies"):
            if isinstance(_d, dict) and isinstance(_d.get(_k), (dict, list)):
                return len(_d[_k]) or None
    except Exception:
        return None
    return None


if "xla_force_host_platform_device_count" not in _os.environ.get("XLA_FLAGS", ""):
    _bp_devices = _bp_resolve_devices()
    if _bp_devices is not None:
        if str(_bp_devices).strip().lower() in ("max", "all", "auto"):
            # "max" = as many devices as are *useful*: one per process, capped at cores.
            # Never every core — idle surplus devices oversubscribe the collective
            # threadpool and deadlock the pmap rendezvous (see _bp_count_processes).
            _cores = _os.cpu_count() or 1
            _nproc = _bp_count_processes()
            # LOO leaves >=1 process out per fold, so the largest fold batch is
            # n_processes - 1; don't expose an idle surplus device.
            if _nproc and len(_sys.argv) > 1 and _sys.argv[1] == "loo":
                _nproc = max(1, _nproc - 1)
            _bp_devices = min(_cores, _nproc) if _nproc else _cores
        else:
            try:
                _bp_devices = int(_bp_devices)
            except (TypeError, ValueError):
                _bp_devices = 1
        # Never expose more CPU devices than physical cores. Oversubscribed XLA
        # collective threads can starve past the AllReduce rendezvous timeout
        # (~20 s) and deadlock mid-training — and extra devices never speed up a
        # core-bound CPU run. Cap at cpu_count and warn if the user asked for more.
        _bp_cap = _os.cpu_count() or 1
        if _bp_devices > _bp_cap:
            _sys.stderr.write(
                f"[bp_train] requested --devices {_bp_devices} exceeds {_bp_cap} "
                f"CPU cores; capping to {_bp_cap} (more devices than cores can "
                f"only deadlock the pmap collective, never speed it up).\n"
            )
            _bp_devices = _bp_cap
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
