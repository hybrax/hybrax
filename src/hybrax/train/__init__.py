from __future__ import annotations

# --- multi-core bootstrap: MUST run before JAX initialises ---
# OPT-IN. Exposes N CPU devices so training can shard the process batch across
# cores via pmap (~N speedup). Default OFF (1 device, unchanged behaviour) so it
# never competes for cores with other work. Enable with EITHER:
#   * the config:  train.devices = 8 (or "max") in the --config JSON  (pre-scanned
#     from argv here, since the device count must be fixed before JAX initialises)
#   * the env var: BP_TRAIN_DEVICES=8  (always wins)
# Pick <= number of free cores. No effect on GPU (host-device flag is CPU-only).
import os as _os
import sys as _sys


def _bp_load_config():
    """Best-effort load of the ``--config`` JSON (pre-JAX). Returns a dict or None."""
    _argv = _sys.argv
    _path = None
    for _i, _a in enumerate(_argv):
        if _a == "--config" and _i + 1 < len(_argv):
            _path = _argv[_i + 1]
            break
        if _a.startswith("--config="):
            _path = _a.split("=", 1)[1]
            break
    if not _path:
        return None
    try:
        import json as _json

        with open(_path) as _f:
            return _json.load(_f), _path
    except Exception:
        return None


def _bp_resolve_devices():
    _n = _os.environ.get("BP_TRAIN_DEVICES")
    if _n is not None:
        return _n
    # legacy/ad-hoc --devices flag still honoured (e.g. the flag-based loo path)
    _argv = _sys.argv
    for _i, _a in enumerate(_argv):
        if _a == "--devices" and _i + 1 < len(_argv):
            return _argv[_i + 1]
        if _a.startswith("--devices="):
            return _a.split("=", 1)[1]
    # config-driven train: read train.devices from the --config JSON
    _loaded = _bp_load_config()
    if _loaded is not None:
        _cfg, _ = _loaded
        _train = _cfg.get("train") if isinstance(_cfg, dict) else None
        if isinstance(_train, dict) and _train.get("devices") is not None:
            return _train["devices"]
    return None


def _bp_count_processes():
    """Best-effort process count from the prepared JSON (pre-JAX).

    Used to resolve ``devices: "max"`` to ``min(n_processes, n_cpus)`` instead of every
    core: exposing more CPU devices than there are processes leaves them idle but still
    oversubscribes the XLA collective threadpool, which can starve the pmap rendezvous
    (~20 s) and deadlock mid-training. Returns ``None`` if it can't be determined (then
    the caller falls back to ``cpu_count``)."""
    _prepared = None
    _cfg_dir = None
    _loaded = _bp_load_config()
    if _loaded is not None:
        _cfg, _cfg_path = _loaded
        _cfg_dir = _os.path.dirname(_os.path.abspath(_cfg_path))
        _data = _cfg.get("data") if isinstance(_cfg, dict) else None
        if isinstance(_data, dict):
            _prepared = _data.get("prepared")
    if not _prepared:
        # legacy flag-based path (e.g. loo --input)
        _argv = _sys.argv
        for _i, _a in enumerate(_argv):
            if _a == "--input" and _i + 1 < len(_argv):
                _prepared = _argv[_i + 1]
                break
            if _a.startswith("--input="):
                _prepared = _a.split("=", 1)[1]
                break
    if not _prepared:
        return None
    # the prepared path may be relative to cwd or to the config-file directory
    _candidates = [_prepared]
    if _cfg_dir and not _os.path.isabs(_prepared):
        _candidates.append(_os.path.join(_cfg_dir, _prepared))
    for _path in _candidates:
        try:
            import json as _json

            with open(_path) as _f:
                _d = _json.load(_f)
            for _k in ("processes", "process_order", "case_studies"):
                if isinstance(_d, dict) and isinstance(_d.get(_k), (dict, list)):
                    return len(_d[_k]) or None
        except Exception:
            continue
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
                f"[bp_train] requested devices {_bp_devices} exceeds {_bp_cap} "
                f"CPU cores; capping to {_bp_cap} (more devices than cores can "
                f"only deadlock the pmap collective, never speed it up).\n"
            )
            _bp_devices = _bp_cap
        if _bp_devices > 1:
            _os.environ["XLA_FLAGS"] = (
                _os.environ.get("XLA_FLAGS", "")
                + f" --xla_force_host_platform_device_count={_bp_devices}"
            ).strip()


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
    "compute_dense_exports": "harness",
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
    "load_run": "serialization",
    "load_params": "serialization",
    "reconstruct_run": "serialization",
    "LoadedRun": "serialization",
    "save_model": "serialization",
    "load_trained_wrapper": "serialization",
    "save_opt_state": "serialization",
    "load_opt_state": "serialization",
    "content_hash": "serialization",
    "file_hash": "serialization",
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
