"""All model + run-state (de)serialisation in one place.

Mirrors ``bp_format/serialization.py``. This module owns:

- ``save_model`` / ``load_trained_wrapper`` — **trainable-partition-only** model
  serialisation. The frozen/derived half of the wrapper (controls store,
  ``SCALE_*``, ``rhs_ode``, indices) is rebuilt from ``prepared.json`` +
  ``custom.py`` at load time and is **never** read from the saved file. This
  fixes the controls-store shape-mismatch on load and is forward-compatible
  with trainable controls (DoE): such leaves simply ride in the trainable
  partition.
- ``save_opt_state`` / ``load_opt_state`` — optimizer state twins.
- ``content_hash`` / ``file_hash`` — stable data integrity hashing.
- ``write_run_config_json`` / ``read_run_config_json`` — the run-dir ``config.json``.
- ``reconstruct_run`` — the single model-reconstruction path shared by forward
  and notebooks.
- ``model_load`` / ``model_reload`` — the user-facing loaders. Both return
  ``(trained_wrapper, config)``; ``model_reload`` skips the collection and only
  swaps trainable leaves.
"""

from __future__ import annotations

from contextlib import suppress
import hashlib
import json
import logging
import math
import os
from numbers import Integral, Real
from pathlib import Path
import secrets
import stat
from typing import Any

import equinox as eqx
import optax
from bp_format.dataclasses import BioProcessCollection
from bp_format.json_io import load_json
from bp_format.serialization import (
    NumpyEncoder,
    _process_collection_to_dict,
    load_process_collection,
)

from .constants import METADATA_NAMESPACE
from .model_api import UserLossModule, UserReactionModule, partition_trainable
from .run_config import RunConfig
from .training_data import TrainingDataStore
from .wrapper import HybridOdeWrapper

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Model serialisation — trainable partition only (the controls-store fix)
# ---------------------------------------------------------------------------


def save_model(wrapper: HybridOdeWrapper, path: str | Path) -> None:
    """Serialise the **trainable** leaves of a wrapper to disk.

    Only the trainable partition is written; the frozen/static half (controls,
    scales, ``rhs_ode``, indices) is reconstructed from ``prepared.json`` +
    ``custom.py`` at load time, never deserialised.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    trainable, _static = partition_trainable(wrapper)
    eqx.tree_serialise_leaves(path, trainable)
    logger.info("trained model (trainable leaves) saved to %s", path)


def load_trained_wrapper(
    path: str | Path, *, template: HybridOdeWrapper
) -> HybridOdeWrapper:
    """Load trainable leaves from disk into ``template``'s structure.

    The static half is taken verbatim from ``template`` (rebuilt fresh), so a
    structurally-different controls store at save time can no longer cause a
    shape mismatch — controls are not part of the serialised bytes.
    """
    trainable_template, static = partition_trainable(template)
    trainable = eqx.tree_deserialise_leaves(Path(path), like=trainable_template)
    return eqx.combine(trainable, static)


def save_opt_state(opt_state: optax.OptState, path: str | Path) -> None:
    """Serialise optimizer state leaf-by-leaf."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    eqx.tree_serialise_leaves(path, opt_state)


def load_opt_state(path: str | Path, *, template: optax.OptState) -> optax.OptState:
    """Deserialise optimizer state into an optimizer-state ``template``."""
    return eqx.tree_deserialise_leaves(Path(path), like=template)


# ---------------------------------------------------------------------------
# Integrity hashing
# ---------------------------------------------------------------------------


def file_hash(path: str | Path) -> str:
    """Exact-bytes sha256 of a file, prefixed ``sha256:``."""
    digest = hashlib.sha256(Path(path).read_bytes()).hexdigest()
    return f"sha256:{digest}"


def environment_versions() -> dict[str, str]:
    """Best-effort package versions for a run / prepare provenance block."""
    import platform
    from importlib.metadata import PackageNotFoundError
    from importlib.metadata import version as _version

    versions: dict[str, str] = {"python": platform.python_version()}
    for pkg in (
        "bp_train",
        "bp_format",
        "jax",
        "optax",
        "equinox",
        "diffrax",
        "pydantic",
    ):
        try:
            versions[pkg] = _version(pkg)
        except PackageNotFoundError:
            continue
    return versions


# Keys under metadata[METADATA_NAMESPACE] that are run-specific provenance, not
# prepared *science* — excluded from content_hash so re-preparing identical data
# (different timestamp / hashes / validation reports) yields the same hash.
_VOLATILE_NS_KEYS = frozenset(
    {
        "provenance",
        "prepared_at",
        "source_input_path",
        "source_input_sha256",
        "custom_py_sha256",
        "bp_format_validation_raw",
        "bp_format_validation_prepared",
        "prepared_semantics_validation",
        "semantics_provenance",
        "transform_hooks",
    }
)


def _strip_provenance(collection_dict: dict[str, Any]) -> dict[str, Any]:
    """Return a copy of the collection dict with bp-train provenance removed.

    The provenance keys carry timestamps / hashes / validation reports and are
    excluded from ``content_hash`` so that re-preparing identical data yields an
    identical hash. The prepared *science* (process order and per-process
    control layouts) is retained.
    """
    metadata = collection_dict.get("metadata")
    if not isinstance(metadata, dict):
        return collection_dict
    ns = metadata.get(METADATA_NAMESPACE)
    if not isinstance(ns, dict):
        return collection_dict
    new_ns = {k: v for k, v in ns.items() if k not in _VOLATILE_NS_KEYS}
    new_metadata = {k: v for k, v in metadata.items() if k != METADATA_NAMESPACE}
    if new_ns:  # keep the namespace only if it still carries science
        new_metadata[METADATA_NAMESPACE] = new_ns
    return {**collection_dict, "metadata": new_metadata}


def content_hash(collection: BioProcessCollection) -> str:
    """Stable sha256 over the collection's *science*, provenance excluded.

    Hashes the canonical re-serialisation (sorted keys, normalised numbers via
    ``NumpyEncoder``) of the deserialised collection, so JSON key-order / float
    formatting / timestamp differences across machines and re-prepares do not
    change it.
    """
    as_dict = _strip_provenance(_process_collection_to_dict(collection))
    canonical = dumps_json(
        as_dict, sort_keys=True, separators=(",", ":"), cls=NumpyEncoder
    )
    digest = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    return f"sha256:{digest}"


# ---------------------------------------------------------------------------
# config.json read/write
# ---------------------------------------------------------------------------


def run_config_to_jsonable(config: RunConfig) -> dict[str, Any]:
    """RunConfig → JSON-native dict (Paths → str, tuples → lists)."""
    return config.model_dump(mode="json")


def _json_safe(value: Any) -> Any:
    """Return JSON-native data with non-finite real scalars replaced by None."""
    if isinstance(value, bool):
        return value
    if isinstance(value, Integral):
        return int(value)
    if isinstance(value, Real):
        value = float(value)
        return value if math.isfinite(value) else None
    if isinstance(value, dict):
        return {key: _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return value


def dumps_json(document: Any, **kwargs: Any) -> str:
    """Encode valid JSON, normalizing non-finite real scalars to null."""
    kwargs["allow_nan"] = False
    return json.dumps(_json_safe(document), **kwargs)


def write_json(
    path: str | Path, document: Any, *, indent: int = 2, **kwargs: Any
) -> None:
    """Atomically publish a normalized JSON document."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    kwargs["allow_nan"] = False
    try:
        try:
            destination_mode = stat.S_IMODE(path.stat().st_mode)
        except FileNotFoundError:
            destination_mode = None
        while True:
            candidate = path.with_name(f".{path.name}.{secrets.token_hex(8)}.tmp")
            try:
                descriptor = os.open(
                    candidate,
                    os.O_WRONLY | os.O_CREAT | os.O_EXCL,
                    0o666,
                )
            except FileExistsError:
                continue
            temporary = candidate
            break
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            if destination_mode is not None:
                if hasattr(os, "fchmod"):
                    os.fchmod(stream.fileno(), destination_mode)
                else:
                    os.chmod(candidate, destination_mode)
            json.dump(_json_safe(document), stream, indent=indent, **kwargs)
            stream.flush()
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            with suppress(OSError):
                temporary.unlink(missing_ok=True)


def read_json(path: str | Path) -> dict[str, Any]:
    return load_json(path)


def read_run_config_json(path: str | Path) -> tuple[RunConfig, dict[str, Any]]:
    """Parse a run-dir ``config.json`` → (RunConfig, full document)."""
    document = read_json(path)
    if "config" not in document:
        raise ValueError(f"{path}: config.json is missing the 'config' block")
    config = RunConfig.model_validate(document["config"])
    return config, document


def update_json(path: str | Path, **fields: Any) -> dict[str, Any]:
    """Shallow-merge fields into an existing JSON object and publish it."""
    document = read_json(path)
    if not isinstance(document, dict):
        raise ValueError(f"{path}: expected a JSON object")
    document.update(fields)
    write_json(path, document)
    return document


# ---------------------------------------------------------------------------
# Single model-reconstruction path
# ---------------------------------------------------------------------------


def _resolve_prepared(run_dir: Path, config: RunConfig) -> Path:
    """Locate ``prepared.json``: a bundled copy in the run dir, else the
    recorded ``data.prepared`` path."""
    bundled = run_dir / "prepared.json.gz"
    if bundled.is_file():
        return bundled
    bundled = run_dir / "prepared.json"
    if bundled.is_file():
        return bundled
    if config.data is not None and config.data.prepared is not None:
        recorded = Path(config.data.prepared)
        if recorded.is_file():
            return recorded
    raise FileNotFoundError(
        f"could not resolve prepared.json for run {run_dir}: no bundled copy and "
        "the recorded data.prepared path does not exist"
    )


def _check_content_hash(
    collection: BioProcessCollection, document: dict[str, Any], *, where: Path
) -> None:
    """Hard-error if the prepared collection's content_hash disagrees with the
    value recorded in ``config.json`` (skips silently if none was recorded)."""
    recorded = document.get("inputs", {}).get("prepared_input", {}).get("content_hash")
    if not recorded:
        return
    actual = content_hash(collection)
    if actual != recorded:
        raise ValueError(
            f"prepared.json data for run {where} differs from the run's record: "
            f"recorded {recorded}, got {actual}"
        )


def reconstruct_run(
    run_dir: str | Path,
    config: RunConfig,
    document: dict[str, Any] | None = None,
) -> tuple[UserReactionModule, UserLossModule, TrainingDataStore, BioProcessCollection]:
    """THE single reconstruction path — forward, resume, and model_load all use it.

    Verifies the prepared ``content_hash`` against ``config.json`` first, then
    rebuilds ``(reaction_module, loss_module, store, collection)`` exactly as
    training did. Returns those four; callers build the template wrapper.
    """
    # Lazy import to avoid an import cycle (harness imports this module's twins).
    from .harness import TrainHarnessConfig, _build_runtime_modules
    from .training_data import TrainingDataStore
    from .utils import load_custom_module

    run_dir = Path(run_dir)
    if document is None:
        _, document = read_run_config_json(run_dir / "config.json")

    prepared = _resolve_prepared(run_dir, config)
    collection = load_process_collection(prepared)
    _check_content_hash(collection, document, where=run_dir)

    bundled_custom = run_dir / "custom.py"
    custom_module = (
        load_custom_module(bundled_custom) if bundled_custom.is_file() else None
    )
    # config.custom comes back from config.json as a raw dict; re-wrap it in the
    # typed object the hooks expect (mirrors a fresh run's get_custom_config).
    from .run_config import reresolve_custom

    config = reresolve_custom(config, custom_module)

    targets = config.data.targets if config.data is not None else None
    target_source = config.data.target_source if config.data is not None else "auto"
    store = TrainingDataStore.from_collection(
        collection,
        target_variable_order=targets,
        target_source=target_source,
    )

    train_like_cfg = TrainHarnessConfig(
        process_names=(
            config.data.processes
            if config.data is not None and config.data.processes is not None
            else tuple(store.process_order)
        ),
        target_variable_order=targets,
        target_source=target_source,
        seed=int(config.train.seed),
        allow_stateful_models=config.train.allow_stateful_models,
    )
    reaction_module, loss_module = _build_runtime_modules(
        store=store,
        collection=collection,
        config=train_like_cfg,
        custom_module=custom_module,
        custom_config=config,
    )
    assert loss_module is not None
    return reaction_module, loss_module, store, collection


def _resolve_run_dir(path: Path, *, max_levels: int = 4) -> Path | None:
    """Nearest directory at/above ``path`` that holds a ``config.json``."""
    current = path if path.is_dir() else path.parent
    for _ in range(max_levels + 1):
        if (current / "config.json").is_file():
            return current
        if current.parent == current:
            break
        current = current.parent
    return None


def _resolve_model_path(path: str | Path) -> tuple[Path, Path]:
    """Resolve a model reference to ``(run_dir, params_path)``.

    ``path`` is a run directory, a checkpoint directory, or a ``params.eqx``. A
    directory resolves its weights in one ordered pass — ``<dir>/params.eqx``,
    ``<dir>/model/params.eqx``, ``<dir>/checkpoints/latest/params.eqx`` — so a run
    that has not finished (no ``model/`` yet) still loads from its latest
    checkpoint. A file must be named ``params.eqx``: a legacy
    ``trained_wrapper.eqx`` raises instead of silently falling through to the run's
    final weights.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"model path does not exist: {path}")

    if path.is_file():
        if path.name != "params.eqx":
            raise FileNotFoundError(f"model file must be a params.eqx, got {path}")
        params_path = path
    else:
        for candidate in (
            path / "params.eqx",
            path / "model" / "params.eqx",
            path / "checkpoints" / "latest" / "params.eqx",
        ):
            if candidate.is_file():
                params_path = candidate
                break
        else:
            raise FileNotFoundError(f"no params.eqx found for model {path}")

    run_dir = _resolve_run_dir(path)
    if run_dir is None:
        raise FileNotFoundError(
            f"no config.json at or above {path}; pass a trained run directory "
            "or a self-contained checkpoint dir."
        )
    return run_dir, params_path


def model_load(path: str | Path) -> tuple[HybridOdeWrapper, RunConfig]:
    """Load a trained model and the run config it was trained under.

    ``path`` is a run directory, a checkpoint directory, or a ``params.eqx`` —
    see :func:`_resolve_model_path` for the ordered rule. Address a specific
    checkpoint by its path, e.g. ``model_load(run_dir / "checkpoints" / "step_00300")``.

    The run's own prepared collection is loaded to rebuild the **static** half of
    the wrapper (``rhs_ode``, ``controls``, every ``SCALE_*``, the index arrays);
    only the trainable leaves come from ``params.eqx``. That reconstruction is the
    expensive part of the call — use :func:`model_reload` to swap in a newer
    checkpoint of the *same* run without paying it again.

    Returns ``(trained_wrapper, config)``. ``config.solver`` carries the solver
    settings the model was fitted under; pass it to :func:`~bp_train.model_predict`.
    """
    from .harness import _build_template_wrapper

    run_dir, params_path = _resolve_model_path(path)
    config, document = read_run_config_json(run_dir / "config.json")
    reaction_module, loss_module, store, _collection = reconstruct_run(
        run_dir, config, document
    )
    selected_processes = (
        config.data.processes
        if config.data is not None and config.data.processes is not None
        else tuple(store.process_order)
    )
    template = _build_template_wrapper(
        store,
        reaction_module=reaction_module,
        selected_processes=selected_processes,
        loss_module=loss_module,
    )
    return load_trained_wrapper(params_path, template=template), config


def model_reload(
    path: str | Path, trained_wrapper: HybridOdeWrapper
) -> tuple[HybridOdeWrapper, RunConfig]:
    """Refresh **only** the trainable leaves from ``path`` into an existing wrapper.

    Returns the same ``(trained_wrapper, config)`` pair as :func:`model_load`, so
    the two are interchangeable at the call site. Skips the prepared collection
    entirely, which is the whole point: on a 61-process run that is ~0.03 s against
    ~100 s, almost all of which is the ``estimate_all_scales`` hook.

    .. danger::
       The **static** half — every ``SCALE_*``, ``controls``, ``Cin``, ``rhs_ode``,
       ``target_state_indices`` — is kept from ``trained_wrapper`` and is **not**
       read from the checkpoint (it was never written there; see :func:`save_model`).
       Equinox only checks that the *trainable* pytree matches, which for a typical
       MLP head depends on layer shapes alone. So passing a wrapper that came from a
       different run, or one built against a different collection, loads the weights
       into a different scaled space and every prediction is silently wrong — no
       exception, no NaN. Only use this to move between checkpoints of the **same**
       run. When in doubt, pay for :func:`model_load`.
    """
    run_dir, params_path = _resolve_model_path(path)
    config, _document = read_run_config_json(run_dir / "config.json")
    logger.warning(
        "model_reload(%s): refreshing trainable leaves only. The static half "
        "(SCALE_*, controls, Cin, rhs_ode) is kept from the wrapper you passed in "
        "and is NOT read from the checkpoint. If that wrapper came from a different "
        "run or a different collection, predictions will be silently wrong.",
        params_path,
    )
    return load_trained_wrapper(params_path, template=trained_wrapper), config
