from __future__ import annotations

# --- multi-core bootstrap: MUST run before JAX initialises ---
# OPT-IN. Exposes N CPU devices so training can shard the process batch across
# cores via pmap (~N speedup). Default OFF (1 device, unchanged behaviour) so it
# never competes for cores with other work. Enable with EITHER:
#   * the config:  train.devices = 8 (or "max") in the --config JSON  (pre-scanned
#     from argv here, since the device count must be fixed before JAX initialises)
#   * the env var: HYBRAX_TRAIN_DEVICES=8  (always wins)
# Pick <= number of free cores. No effect on GPU (host-device flag is CPU-only).
import gzip as _gzip
import os as _os
import sys as _sys

from hybrax.format.json_io import load_json as _load_json
from hybrax.format.json_io import loads_json as _loads_json


def _load_config():
    """Best-effort flat RunConfig dict from the pre-JAX ``--config`` argument."""
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
        _doc = _load_json(_path)
        if isinstance(_doc, dict) and isinstance(_doc.get("config"), dict):
            return _doc["config"], _path
        return _doc, _path
    except Exception:
        return None


def _resolve_devices():
    _argv = _sys.argv
    # The loo *orchestrator* (no --fold) trains nothing — it only dispatches
    # per-fold worker subprocesses, each launched with its own HYBRAX_TRAIN_DEVICES.
    # Force it onto a single device FIRST (before the env-var branch) so an
    # exported HYBRAX_TRAIN_DEVICES — which is meant for the workers — does not make
    # the idle orchestrator reserve the whole CPU pool. Workers (--fold present)
    # fall through to the env-var branch and honour the count the orchestrator
    # set for them.
    _is_loo = len(_argv) > 1 and _argv[1] == "loo"
    _has_fold = any(_a == "--fold" or _a.startswith("--fold=") for _a in _argv)
    if _is_loo and not _has_fold:
        return None
    _n = _os.environ.get("HYBRAX_TRAIN_DEVICES")
    if _n is not None:
        return _n
    # config-driven train: read train.devices from the --config JSON
    _loaded = _load_config()
    if _loaded is not None:
        _cfg, _ = _loaded
        _train = _cfg.get("train") if isinstance(_cfg, dict) else None
        if isinstance(_train, dict) and _train.get("devices") is not None:
            return _train["devices"]
    return None


def _read_prepared_json(_path):
    """Read a prepared JSON file or prepare-output directory."""
    if _os.path.isdir(_path):
        for _name in ("prepared.json.gz", "prepared.json"):
            _candidate = _os.path.join(_path, _name)
            if _os.path.isfile(_candidate):
                _path = _candidate
                break
    _opener = _gzip.open if str(_path).endswith(".gz") else open
    with _opener(_path, "rt", encoding="utf-8") as _f:
        return _loads_json(_f.read())


def _count_processes():
    """Best-effort process count from the prepared JSON (pre-JAX).

    Used to resolve ``devices: "max"`` to ``min(n_processes, n_cpus)`` instead of every
    core: exposing more CPU devices than there are processes leaves them idle but still
    oversubscribes the XLA collective threadpool, which can starve the pmap rendezvous
    (~20 s) and deadlock mid-training. Returns ``None`` if it can't be determined (then
    the caller falls back to ``cpu_count``).

    Returns ``None`` when the prepared artifact cannot be resolved."""
    _prepared = None
    _prepared_from_config = False
    _cfg_dir = None
    _loaded = _load_config()
    if _loaded is not None:
        _cfg, _cfg_path = _loaded
        _cfg_dir = _os.path.dirname(_os.path.abspath(_cfg_path))
        _data = _cfg.get("data") if isinstance(_cfg, dict) else None
        if isinstance(_data, dict):
            _prepared = _data.get("prepared")
            _prepared_from_config = bool(_prepared)
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
    # Config paths resolve from the config directory. Legacy --input paths
    # retain their historical cwd-relative behavior.
    if _prepared_from_config and _cfg_dir and not _os.path.isabs(_prepared):
        _candidates = [_os.path.join(_cfg_dir, _prepared)]
    else:
        _candidates = [_prepared]
    for _path in _candidates:
        try:
            _d = _read_prepared_json(_path)
            for _k in ("processes", "process_order", "case_studies"):
                if isinstance(_d, dict) and isinstance(_d.get(_k), (dict, list)):
                    return len(_d[_k]) or None
        except Exception:
            continue
    return None


if "xla_force_host_platform_device_count" not in _os.environ.get("XLA_FLAGS", ""):
    _hybrax_devices = _resolve_devices()
    if _hybrax_devices is not None:
        if str(_hybrax_devices).strip().lower() in ("max", "all", "auto"):
            # "max" = as many devices as are *useful*: one per process, capped at cores.
            # Never every core — idle surplus devices oversubscribe the collective
            # threadpool and deadlock the pmap rendezvous (see _count_processes).
            _cores = _os.cpu_count() or 1
            _nproc = _count_processes()
            _hybrax_devices = min(_cores, _nproc) if _nproc else _cores
        else:
            try:
                _hybrax_devices = int(_hybrax_devices)
            except (TypeError, ValueError):
                _hybrax_devices = 1
        # Never expose more CPU devices than physical cores. Oversubscribed XLA
        # collective threads can starve past the AllReduce rendezvous timeout
        # (~20 s) and deadlock mid-training — and extra devices never speed up a
        # core-bound CPU run. Cap at cpu_count and warn if the user asked for more.
        _device_cap = _os.cpu_count() or 1
        if _hybrax_devices > _device_cap:
            _sys.stderr.write(
                f"[hybrax.train] requested devices {_hybrax_devices} "
                f"exceeds {_device_cap} CPU cores; capping to {_device_cap} "
                f"(more devices than cores can "
                f"only deadlock the pmap collective, never speed it up).\n"
            )
            _hybrax_devices = _device_cap
        if _hybrax_devices > 1:
            _os.environ["XLA_FLAGS"] = (
                _os.environ.get("XLA_FLAGS", "")
                + f" --xla_force_host_platform_device_count={_hybrax_devices}"
            ).strip()


# Enable float64 (JAX x64) for the whole hybrax.train pipeline. Set after the XLA
# device-count env above and before any array is created downstream.
import jax as _jax  # noqa: E402

_jax.config.update("jax_enable_x64", True)


from importlib import import_module  # noqa: E402
from typing import Any  # noqa: E402

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
    if name not in _EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module = import_module(f".{_EXPORTS[name]}", __name__)
    value = getattr(module, name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted([*globals(), *_EXPORTS])
