"""All model + run-state (de)serialisation in one place.

Mirrors ``hybrax.format/serialization.py``. This module owns:

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
- ``reconstruct_training`` — the single model-reconstruction path shared by
  model loading, forward, ensembles, and notebooks. It rebuilds a model from the
  data *it* trained on, with that input's recorded ``content_hash`` verified
  before any hook runs.
- ``model_load`` / ``model_reload`` — the user-facing loaders. Both return
  ``(trained_wrapper, config)``; ``model_reload`` skips the collection and only
  swaps trainable leaves.
"""

from __future__ import annotations

from contextlib import suppress
from dataclasses import dataclass
import hashlib
from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as _version
import json
import logging
import math
import os
from numbers import Integral, Real
from pathlib import Path
import platform
import secrets
import stat
from typing import Any

import equinox as eqx
import optax
from hybrax.format.dataclasses import BioProcessCollection
from hybrax.format.json_io import load_json
from hybrax.format.serialization import (
    NumpyEncoder,
    _process_collection_to_dict,
    load_process_collection,
)

from .constants import METADATA_NAMESPACE
from .model_api import UserLossModule, UserReactionModule, partition_trainable
from .run_config import RunConfig
from .training_data import TARGET_SOURCE_AUTO, TrainingDataStore
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
    versions: dict[str, str] = {"python": platform.python_version()}
    for pkg in (
        "hybrax",
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
        "format_validation_raw",
        "format_validation_prepared",
        "prepared_semantics_validation",
        "semantics_provenance",
        "transform_hooks",
    }
)


def _strip_provenance(collection_dict: dict[str, Any]) -> dict[str, Any]:
    """Return a copy of the collection dict with hybrax.train provenance removed.

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


def _recorded_prepared_content_hash(document: dict[str, Any]) -> str | None:
    """The ``inputs.prepared_input.content_hash`` a run recorded, if any."""
    inputs = document.get("inputs") if isinstance(document, dict) else None
    prepared_input = inputs.get("prepared_input") if isinstance(inputs, dict) else None
    if not isinstance(prepared_input, dict):
        return None
    recorded = prepared_input.get("content_hash")
    return recorded if isinstance(recorded, str) and recorded else None


def _require_content_hash(
    collection: BioProcessCollection, document: dict[str, Any], *, where: Path
) -> str:
    """Verify a run's recorded prepared ``content_hash`` — before any hook runs.

    On the shared reconstruction path a *missing* record is an error too: a model
    whose training input is not pinned cannot be shown to be rebuilt from the data
    it was trained on, and constructing hooks from unverified data is exactly the
    silent-mismatch failure this path exists to prevent.
    """
    recorded = _recorded_prepared_content_hash(document)
    if recorded is None:
        raise ValueError(
            f"run {where} records no inputs.prepared_input.content_hash; its model "
            "cannot be reconstructed from unverified training input"
        )
    actual = content_hash(collection)
    if actual != recorded:
        raise ValueError(
            f"prepared.json data for run {where} differs from the run's record: "
            f"recorded {recorded}, got {actual}"
        )
    return actual


@dataclass(frozen=True)
class ReconstructedTraining:
    """One model's training-time construction, rebuilt from its own input.

    ``collection`` / ``store`` are the model's **training** data — never
    evaluation or novel data. ``training_process_names`` is the selection the run
    recorded; the hooks and ``template_wrapper`` were built from exactly it.
    """

    config: RunConfig
    custom_module: Any | None
    collection: BioProcessCollection
    store: TrainingDataStore
    reaction_module: UserReactionModule
    loss_module: UserLossModule
    training_process_names: tuple[str, ...]
    template_wrapper: HybridOdeWrapper
    prepared_path: Path
    prepared_content_hash: str


def reconstruct_training(
    run_dir: str | Path,
    config: RunConfig | None = None,
    document: dict[str, Any] | None = None,
    *,
    custom_module: Any | None = None,
    custom_py: str | Path | None = None,
    training_process_names: tuple[str, ...] | None = None,
) -> ReconstructedTraining:
    """THE shared reconstruction path — model loading, forward, and ensembles.

    Loads the run's **own** prepared collection, requires and verifies its
    recorded ``inputs.prepared_input.content_hash`` *before* invoking any hook,
    restricts the hook-visible data to the recorded training process names, and
    rebuilds ``(reaction_module, loss_module, template_wrapper)`` exactly as
    training did. Evaluation data plays no part here: callers that evaluate novel
    data build a separate store for their solves.
    """
    # Lazy import to avoid an import cycle (harness imports this module's twins).
    from .harness import (
        TrainHarnessConfig,
        _build_runtime_modules,
        _build_template_wrapper,
    )
    from .run_config import reresolve_custom
    from .utils import load_custom_module

    run_dir = Path(run_dir)
    if config is None or document is None:
        parsed_config, parsed_document = read_run_config_json(run_dir / "config.json")
        config = parsed_config if config is None else config
        document = parsed_document if document is None else document

    prepared = _resolve_prepared(run_dir, config)
    collection = load_process_collection(prepared)
    verified_hash = _require_content_hash(collection, document, where=run_dir)

    if custom_module is None:
        bundled_custom = run_dir / "custom.py"
        if custom_py is not None:
            custom_module = load_custom_module(custom_py)
        elif bundled_custom.is_file():
            custom_module = load_custom_module(bundled_custom)
    # config.custom comes back from config.json as a raw dict; re-wrap it in the
    # typed object the hooks expect (mirrors a fresh run's get_custom_config).
    config = reresolve_custom(config, custom_module)

    targets = config.data.targets if config.data is not None else None
    target_source = (
        config.data.target_source if config.data is not None else TARGET_SOURCE_AUTO
    )
    store = TrainingDataStore.from_collection(
        collection,
        target_variable_order=targets,
        target_source=target_source,
    )

    if training_process_names is not None:
        selected = tuple(training_process_names)
    elif config.data is not None and config.data.processes is not None:
        selected = tuple(config.data.processes)
    else:
        selected = tuple(store.process_order)

    train_like_cfg = TrainHarnessConfig(
        process_names=selected,
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
    template_wrapper = _build_template_wrapper(
        store,
        reaction_module=reaction_module,
        selected_processes=selected,
        loss_module=loss_module,
    )
    return ReconstructedTraining(
        config=config,
        custom_module=custom_module,
        collection=collection,
        store=store,
        reaction_module=reaction_module,
        loss_module=loss_module,
        training_process_names=selected,
        template_wrapper=template_wrapper,
        prepared_path=prepared,
        prepared_content_hash=verified_hash,
    )


def resolve_run_dir(path: str | Path, *, max_levels: int = 4) -> Path | None:
    """Nearest directory at/above ``path`` that holds a ``config.json``."""
    path = Path(path)
    current = path if path.is_dir() else path.parent
    for _ in range(max_levels + 1):
        if (current / "config.json").is_file():
            return current
        if current.parent == current:
            break
        current = current.parent
    return None


def _resolve_directory_weights(path: Path) -> Path:
    """Resolve directory weights with the canonical precedence."""
    for candidate in (
        path / "params.eqx",
        path / "model" / "params.eqx",
        path / "checkpoints" / "latest" / "params.eqx",
    ):
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(
        f"no model weights in directory {path}; expected params.eqx, "
        "model/params.eqx or checkpoints/latest/params.eqx"
    )


def _resolve_model_reference(
    path: str | Path,
    *,
    require_params_filename: bool,
    fall_back_to_run_weights: bool,
) -> tuple[Path, Path]:
    """Resolve a model reference while preserving its caller's file contract."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"model path does not exist: {path}")

    run_dir = None
    if path.is_file():
        if require_params_filename and path.name != "params.eqx":
            raise FileNotFoundError(f"model file must be a params.eqx, got {path}")
        params_path = path
    else:
        try:
            params_path = _resolve_directory_weights(path)
        except FileNotFoundError:
            if not fall_back_to_run_weights:
                raise FileNotFoundError(
                    f"no params.eqx found for model {path}"
                ) from None
            run_dir = resolve_run_dir(path)
            if run_dir is None or path == run_dir:
                raise
            params_path = _resolve_directory_weights(run_dir)

    if run_dir is None:
        run_dir = resolve_run_dir(path)
    if run_dir is None:
        raise FileNotFoundError(
            f"no config.json at or above {path}; pass a trained run directory "
            "or a self-contained checkpoint dir."
        )
    return run_dir, params_path


def resolve_forward_model_path(path: str | Path) -> tuple[Path, Path]:
    """Resolve permissive forward weights and their owning run directory.

    Forward accepts any existing file, including current LOO fold
    ``trained_wrapper.eqx`` files and notebook checkpoints. Direct model loading
    uses :func:`resolve_model_path`, which deliberately accepts only files named
    ``params.eqx``. Directory references share one weight precedence in both
    paths.
    """
    return _resolve_model_reference(
        path,
        require_params_filename=False,
        fall_back_to_run_weights=True,
    )


def resolve_model_path(path: str | Path) -> tuple[Path, Path]:
    """Resolve a model reference to ``(run_dir, params_path)``.

    ``path`` is a run directory, a checkpoint directory, or a ``params.eqx``. A
    directory resolves its weights in one ordered pass — ``<dir>/params.eqx``,
    ``<dir>/model/params.eqx``, ``<dir>/checkpoints/latest/params.eqx`` — so a run
    that has not finished (no ``model/`` yet) still loads from its latest
    checkpoint. A file must be named ``params.eqx``: a current LOO output named
    ``trained_wrapper.eqx`` raises instead of silently falling through to the run's
    final weights.
    """
    return _resolve_model_reference(
        path,
        require_params_filename=True,
        fall_back_to_run_weights=False,
    )


def model_load(path: str | Path) -> tuple[HybridOdeWrapper, RunConfig]:
    """Load a trained model and the run config it was trained under.

    ``path`` is a run directory, a checkpoint directory, or a ``params.eqx`` —
    see :func:`resolve_model_path` for the ordered rule. Address a specific
    checkpoint by its path, e.g. ``model_load(run_dir / "checkpoints" / "step_00300")``.

    The run's own prepared collection is loaded — with its recorded
    ``inputs.prepared_input.content_hash`` verified before any hook runs — to
    rebuild the **static** half of the wrapper (``rhs_ode``, ``controls``, every
    ``SCALE_*``, the index arrays); only the trainable leaves come from
    ``params.eqx``. That reconstruction is the expensive part of the call — use
    :func:`model_reload` to swap in a newer checkpoint of the *same* run without
    paying it again.

    Returns ``(trained_wrapper, config)``. ``config.solver`` carries the solver
    settings the model was fitted under; pass it to :func:`~hybrax.train.model_predict`.
    """
    run_dir, params_path = resolve_model_path(path)
    config, document = read_run_config_json(run_dir / "config.json")
    rebuilt = reconstruct_training(run_dir, config, document)
    return load_trained_wrapper(params_path, template=rebuilt.template_wrapper), config


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
    run_dir, params_path = resolve_model_path(path)
    config, _document = read_run_config_json(run_dir / "config.json")
    logger.warning(
        "model_reload(%s): refreshing trainable leaves only. The static half "
        "(SCALE_*, controls, Cin, rhs_ode) is kept from the wrapper you passed in "
        "and is NOT read from the checkpoint. If that wrapper came from a different "
        "run or a different collection, predictions will be silently wrong.",
        params_path,
    )
    return load_trained_wrapper(params_path, template=trained_wrapper), config
