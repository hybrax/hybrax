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
- ``reconstruct_run`` / ``load_run`` / ``load_params`` / ``LoadedRun`` — the
  single model-reconstruction path shared by forward, resume, and notebooks.
"""

from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import equinox as eqx
from bp_format.dataclasses import BioProcessCollection
from bp_format.serialization import (
    NumpyEncoder,
    _process_collection_to_dict,
    load_process_collection_json,
)

from .constants import METADATA_NAMESPACE
from .model_api import partition_trainable
from .run_config import RunConfig

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Model serialisation — trainable partition only (the controls-store fix)
# ---------------------------------------------------------------------------


def save_model(wrapper: Any, path: str | Path) -> None:
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


def load_trained_wrapper(path: str | Path, *, template: Any) -> Any:
    """Load trainable leaves from disk into ``template``'s structure.

    The static half is taken verbatim from ``template`` (rebuilt fresh), so a
    structurally-different controls store at save time can no longer cause a
    shape mismatch — controls are not part of the serialised bytes.
    """
    trainable_template, static = partition_trainable(template)
    trainable = eqx.tree_deserialise_leaves(Path(path), like=trainable_template)
    return eqx.combine(trainable, static)


def save_opt_state(opt_state: Any, path: str | Path) -> None:
    """Serialise optimizer state leaf-by-leaf."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    eqx.tree_serialise_leaves(path, opt_state)


def load_opt_state(path: str | Path, *, template: Any) -> Any:
    """Deserialise optimizer state into ``template`` (= ``optimizer.init(trainable)``)."""
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
    identical hash. The prepared *science* (process_order, per-process control
    layouts, runtime_controls_config) is retained.
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
    canonical = json.dumps(
        as_dict, sort_keys=True, separators=(",", ":"), cls=NumpyEncoder
    )
    digest = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    return f"sha256:{digest}"


# ---------------------------------------------------------------------------
# config.json read/write
# ---------------------------------------------------------------------------


def run_config_to_jsonable(config: RunConfig) -> dict[str, Any]:
    """RunConfig → JSON-native dict (Paths → str, tuples → lists)."""
    return json.loads(config.model_dump_json())


def write_json(path: str | Path, document: dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(document, indent=2), encoding="utf-8")


def read_json(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def read_run_config_json(path: str | Path) -> tuple[RunConfig, dict[str, Any]]:
    """Parse a run-dir ``config.json`` → (RunConfig, full document)."""
    document = read_json(path)
    if "config" not in document:
        raise ValueError(f"{path}: config.json is missing the 'config' block")
    config = RunConfig.model_validate(document["config"])
    return config, document


def update_run_config_status(path: str | Path, **fields: Any) -> dict[str, Any]:
    """Merge ``fields`` into an existing ``config.json`` and rewrite it."""
    document = read_json(path)
    document.update(fields)
    write_json(path, document)
    return document


# ---------------------------------------------------------------------------
# Single model-reconstruction path
# ---------------------------------------------------------------------------


def _resolve_prepared(run_dir: Path, config: RunConfig) -> Path:
    """Locate ``prepared.json``: a bundled copy in the run dir, else the
    recorded ``data.prepared`` path."""
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
    recorded = (
        document.get("inputs", {}).get("prepared_input", {}).get("content_hash")
    )
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
) -> tuple[Any, Any, Any, BioProcessCollection]:
    """THE single reconstruction path — forward, resume, and load_run all use it.

    Verifies the prepared ``content_hash`` against ``config.json`` first, then
    rebuilds ``(reaction_module, loss_module, store, collection)`` exactly as
    training did. Returns those four; callers build the template wrapper.
    """
    # Lazy import to avoid an import cycle (harness imports this module's twins).
    from .harness import (
        TrainHarnessConfig,
        _build_loss_module,
        _build_reaction_module,
        _resolve_estimated_scales,
    )
    from .training_data import TrainingDataStore
    from .utils import load_custom_module

    run_dir = Path(run_dir)
    if document is None:
        _, document = read_run_config_json(run_dir / "config.json")

    prepared = _resolve_prepared(run_dir, config)
    collection = load_process_collection_json(prepared)
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
    target_source = (
        config.data.target_source if config.data is not None else "auto"
    )
    store = TrainingDataStore.from_collection(
        collection,
        target_variable_order=targets,
        target_source=target_source,
    )

    train_like_cfg = TrainHarnessConfig(
        process_names=tuple(store.process_order),
        target_variable_order=targets,
        target_source=target_source,
        seed=int(config.train.seed),
    )
    scale_kwargs = _resolve_estimated_scales(
        custom_module=custom_module,
        collection=collection,
        store=store,
        custom_cfg=config,
    )
    reaction_module = _build_reaction_module(
        store=store,
        config=train_like_cfg,
        custom_module=custom_module,
        custom_config=config,
        collection=collection,
        scale_kwargs=scale_kwargs,
    )
    loss_module = _build_loss_module(
        store=store,
        config=train_like_cfg,
        custom_module=custom_module,
        custom_config=config,
        collection=collection,
    )
    return reaction_module, loss_module, store, collection


@dataclass(frozen=True)
class LoadedRun:
    """A trained model reconstructed from a run directory."""

    wrapper: Any
    collection: BioProcessCollection
    store: Any
    config: RunConfig
    run_dir: Path
    opt_state: Any | None = None

    def reload(self, checkpoint: str = "latest") -> Any:
        """Refresh just the weights from another checkpoint into this wrapper."""
        return load_params(self.run_dir, into=self.wrapper, checkpoint=checkpoint)


def checkpoint_params_path(run_dir: str | Path, checkpoint: str = "latest") -> Path:
    """Resolve the ``params.eqx`` path for a checkpoint name.

    ``"final"`` → ``model/params.eqx`` (the copy of best); otherwise
    ``checkpoints/<checkpoint>/params.eqx`` (``"best"``/``"latest"`` are symlinks).
    """
    run_dir = Path(run_dir)
    if checkpoint == "final":
        return run_dir / "model" / "params.eqx"
    return run_dir / "checkpoints" / checkpoint / "params.eqx"


def load_run(
    run_dir: str | Path,
    *,
    checkpoint: str = "best",
    load_opt_state: bool = False,
) -> LoadedRun:
    """Reconstruct a trained model from a run directory **alone**.

    ``checkpoint``: ``"best"`` | ``"latest"`` | ``"step_00300"`` resolve under
    ``checkpoints/``; ``"final"`` uses the run-root ``model/`` copy.
    """
    from .harness import _build_template_wrapper, build_optimizer_for_run
    from .harness import TrainHarnessConfig

    run_dir = Path(run_dir)
    config, document = read_run_config_json(run_dir / "config.json")
    reaction_module, loss_module, store, collection = reconstruct_run(
        run_dir, config, document
    )
    template, _extras = _build_template_wrapper(
        store,
        reaction_module=reaction_module,
        collection=collection,
        selected_processes=tuple(store.process_order),
        loss_module=loss_module,
    )

    params_path = checkpoint_params_path(run_dir, checkpoint)
    wrapper = load_trained_wrapper(params_path, template=template)

    opt_state = None
    if load_opt_state:
        bundled_custom = run_dir / "custom.py"
        from .utils import load_custom_module

        custom_module = (
            load_custom_module(bundled_custom) if bundled_custom.is_file() else None
        )
        train_like_cfg = TrainHarnessConfig(
            process_names=tuple(store.process_order),
            target_variable_order=(
                config.data.targets if config.data is not None else None
            ),
            target_source=(
                config.data.target_source if config.data is not None else "auto"
            ),
            seed=int(config.train.seed),
            optimizer_name=config.train.optimizer,
            learning_rate=config.train.learning_rate,
            grad_clip_norm=config.train.grad_clip_norm,
        )
        optimizer, _train_cfg = build_optimizer_for_run(
            custom_module=custom_module,
            custom_cfg=config,
            train_cfg=train_like_cfg,
        )
        trainable_params, _ = partition_trainable(wrapper)
        opt_template = optimizer.init(trainable_params)
        opt_state = load_opt_state(
            params_path.with_name("opt_state.eqx"), template=opt_template
        )

    return LoadedRun(
        wrapper=wrapper,
        collection=collection,
        store=store,
        config=config,
        run_dir=run_dir,
        opt_state=opt_state,
    )


def load_params(
    run_dir: str | Path, *, into: Any, checkpoint: str = "latest"
) -> Any:
    """Refresh weights into an **already-built** wrapper (no dataset/custom.py reload).

    ``into`` must be structurally identical to the trained wrapper; eqx raises on
    a pytree mismatch.
    """
    return load_trained_wrapper(
        checkpoint_params_path(run_dir, checkpoint), template=into
    )
