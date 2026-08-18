from __future__ import annotations

import argparse
import logging
import os
import shutil
import sys
import time
from pathlib import Path
from typing import Any

from bp_format.json_io import load_json
from bp_format.serialization import load_process_collection
import pandas as pd

from .harness import (
    ForwardConfig,
    ForwardResult,
    evaluate_trained_wrapper,
    forward_from_collection,
    prepare_training,
    train_collection,
    train_harness_config_from_run_config,
)
from .loo import (
    _dispatch_producer,
    _runtime_metadata,
    _validated_runtime_metadata,
    execute_trained_fold,
    prepare_single_fold_from_runtime_artifact,
    produce_runtime_artifact,
    run_loo_cv,
    train_prepared_fold,
)
from .runtime_artifact import FORMAT_VERSION
from .runtime_context import original_parent_processes
from .postprocessing import aggregate_dense_exports, export_predictions_csv
from .prepare import prepare_artifact
from .run_config import (
    PredictionScope,
    RunConfig,
    load_forward_config,
    load_loo_config,
    load_prepare_config,
    load_train_config,
    resolve_prepared_path,
)
from .serialization import (
    content_hash,
    environment_versions as _environment_versions,
    read_run_config_json,
    run_config_to_jsonable,
    update_json,
    write_json,
)


def _now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S%z", time.localtime())


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="bp-train")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # ---- prepare ----
    prepare_parser = subparsers.add_parser(
        "prepare",
        help="Transform a raw bp_format process collection into a prepared artifact.",
    )
    prepare_parser.add_argument(
        "--config",
        required=True,
        help="Path to prepare run config JSON.",
    )
    prepare_parser.add_argument(
        "--output-dir",
        required=True,
        help="Output directory for the prepare artifacts "
        "(prepared.json, prepare_config.json, optional augmented-data.png, "
        "prepare_diagnostics/).",
    )
    prepare_parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite an existing prepared.json in --output-dir.",
    )
    prepare_parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Python logging level.",
    )
    prepare_parser.set_defaults(handler=_handle_prepare)

    # ---- train ----
    train_parser = subparsers.add_parser(
        "train",
        help="Run minimal one/multi-process training from a prepared artifact.",
    )
    train_parser.add_argument(
        "--config",
        default=None,
        help="Path to train run config JSON.",
    )
    train_parser.add_argument(
        "--output-dir",
        default=None,
        help="Override output.dir from the config (the FAIR run directory).",
    )
    train_parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow re-running into a run dir that already completed.",
    )
    train_parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="Override train.epochs.",
    )
    # Console numeric formatting is config-only (`logging.decimals`), and
    # metrics.csv is always written to the run dir. The old --log-every /
    # --metrics-csv / --metrics-jsonl /
    # --log-process-losses / --log-decimals / --log-header-every flags are gone.
    train_parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Python logging level.",
    )
    train_parser.set_defaults(handler=_handle_train)

    # ---- forward ----
    forward_parser = subparsers.add_parser(
        "forward",
        help=(
            "Load a trained model and run one forward ODE pass per selected "
            "process (no training). Writes configured predictions and a loss table."
        ),
    )
    forward_parser.add_argument(
        "--config",
        required=True,
        help=(
            "forward_config.json: a `models` list of self-contained run/checkpoint "
            "dirs (len 1 = single, >1 = ensemble), plus optional `data` "
            "(prepared / processes) and `output` (dir / predictions)."
        ),
    )
    forward_parser.add_argument(
        "--output-dir",
        default=None,
        help=(
            "Directory for forward outputs (overrides output.dir). Defaults to "
            "<first model>/forward."
        ),
    )
    forward_parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow re-running into a forward output dir that already has results.",
    )
    forward_parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Python logging level.",
    )
    forward_parser.set_defaults(handler=_handle_forward)

    # ---- loo ----
    loo_parser = subparsers.add_parser(
        "loo",
        help=(
            "Run leave-one/some-process-out cross-validation from a run config: "
            "resolve folds (loo.per_fold_holdout_sets, or classic leave-one-out "
            "when omitted), train each fold in its own subprocess, and aggregate "
            "holdout losses. The run dir is self-contained (bundled config + "
            "custom.py + prepared); --resume continues an interrupted run."
        ),
    )
    loo_parser.add_argument(
        "--config",
        default=None,
        help=(
            "Path to the run config JSON (same schema as `train`, plus an "
            "optional `loo` section: per_fold_holdout_sets, parallel_folds). "
            "Required unless --resume."
        ),
    )
    loo_parser.add_argument(
        "--resume",
        default=None,
        help=(
            "Resume an interrupted LOO run from its output directory. Reloads the "
            "bundled loo-config.json verbatim (no overrides) and re-runs only "
            "folds without identity-matched completion records. Mutually "
            "exclusive with --config."
        ),
    )
    loo_parser.add_argument(
        "--output-dir",
        default=None,
        help="Override output.dir from the config (the LOO run directory).",
    )
    loo_parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace an existing LOO output directory.",
    )
    internal_loo_mode = loo_parser.add_mutually_exclusive_group()
    internal_loo_mode.add_argument(
        "--produce-runtime", action="store_true", help=argparse.SUPPRESS
    )
    internal_loo_mode.add_argument(
        "--runtime-artifact", default=None, help=argparse.SUPPRESS
    )
    loo_parser.add_argument("--fold", type=int, default=None, help=argparse.SUPPRESS)
    loo_parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Python logging level.",
    )
    loo_parser.set_defaults(handler=_handle_loo)

    return parser


def _handle_prepare(args: argparse.Namespace) -> int:
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    output_dir = Path(args.output_dir)
    prepared = output_dir / "prepared.json"
    if prepared.exists() and not args.overwrite:
        logging.getLogger(__name__).error(
            "prepared artifact already exists at %s; pass --overwrite to replace it",
            prepared,
        )
        return 1
    prepare_artifact(
        load_prepare_config(args.config),
        output_dir=args.output_dir,
        overwrite=args.overwrite,
    )
    return 0


def _select_prediction_processes(
    scope: PredictionScope,
    evaluated_processes: tuple[str, ...],
    parent_processes: tuple[str, ...],
) -> tuple[str, ...]:
    """Select prediction exports while preserving evaluation order."""
    if scope == "none":
        return ()
    if scope == "all":
        return evaluated_processes
    if scope != "parents":
        raise ValueError(f"unknown prediction scope: {scope!r}")
    parents = set(parent_processes)
    return tuple(name for name in evaluated_processes if name in parents)


def _write_train_results(
    *,
    output_dir: Path,
    forward_result: ForwardResult,
    prediction_processes: tuple[str, ...],
) -> None:
    """Write final losses and selected dense predictions."""
    log = logging.getLogger(__name__)
    _table, csv_rows = _format_loss_table(forward_result)
    loss_csv_path = output_dir / "losses.csv"
    _write_loss_csv(csv_rows, loss_csv_path)
    log.info("loss table saved to %s", loss_csv_path)
    predictions_path = output_dir / "predictions.csv"
    if prediction_processes:
        export_predictions_csv(
            forward_result.trained_wrapper,
            forward_result.dense_exports,
            predictions_path,
            process_names=prediction_processes,
        )
    else:
        predictions_path.unlink(missing_ok=True)


def _apply_train_cli_overrides(
    cfg: RunConfig,
    args: argparse.Namespace,
    *,
    resolve_output_symlinks: bool = True,
) -> RunConfig:
    """Apply the few CLI flags that override the config file (CLI wins)."""
    updates: dict[str, Any] = {}
    if args.output_dir is not None:
        output_dir = Path(args.output_dir)
        if resolve_output_symlinks:
            output_dir = output_dir.resolve()
        else:
            output_dir = Path(os.path.abspath(output_dir))
        updates["output"] = cfg.output.model_copy(update={"dir": output_dir})
    if getattr(args, "epochs", None) is not None:
        updates["train"] = cfg.train.model_copy(update={"epochs": int(args.epochs)})
    return cfg.model_copy(update=updates) if updates else cfg


def _clear_output_dir_for_overwrite(
    output_dir: Path,
    *,
    input_paths: tuple[str | Path | None, ...],
) -> None:
    """Remove an old run without deleting inputs needed by the new run."""
    if output_dir.is_symlink():
        output_dir.unlink()
        return
    root = output_dir.resolve()
    nested_inputs = [
        Path(path)
        for path in input_paths
        if path is not None and Path(path).resolve().is_relative_to(root)
    ]
    if nested_inputs:
        paths = ", ".join(str(path) for path in nested_inputs)
        raise ValueError(
            f"cannot overwrite output directory {output_dir}: "
            f"it contains input file(s): {paths}"
        )
    if not output_dir.exists():
        return
    for child in output_dir.iterdir():
        if child.is_dir() and not child.is_symlink():
            shutil.rmtree(child)
        else:
            child.unlink()


def _finalize_run_dir(run_dir: Path, result: Any, config_json: Path) -> None:
    """Copy the mandatory latest checkpoint to model/, then mark complete."""
    model_dir = run_dir / "model"
    model_dir.mkdir(parents=True, exist_ok=True)
    latest = run_dir / "checkpoints" / "latest"
    if not (latest / "params.eqx").is_file():
        raise RuntimeError("training completed without a final checkpoint")
    shutil.copyfile(latest / "params.eqx", model_dir / "params.eqx")
    if (latest / "opt_state.eqx").is_file():
        shutil.copyfile(latest / "opt_state.eqx", model_dir / "opt_state.eqx")
    final_mean = (
        float(result.mean_loss_by_step[-1]) if result.mean_loss_by_step else None
    )
    update_json(
        config_json,
        status="complete",
        finished_at=_now_iso(),
        updates_completed=int(getattr(result, "updates_completed", 0)),
        final_mean_loss=final_mean,
    )


def _handle_train(args: argparse.Namespace) -> int:
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    log = logging.getLogger(__name__)

    # ---- Fresh run ----
    if args.config is None:
        log.error("train requires --config")
        return 1
    loaded = load_train_config(args.config)
    cfg = _apply_train_cli_overrides(loaded.config, args)
    if cfg.data is None:
        raise ValueError("train command requires a data config section")
    run_dir = Path(cfg.output.dir)
    config_json = run_dir / "config.json"

    # Re-run guard: block only on a completed run unless --overwrite.
    if config_json.is_file():
        try:
            _, prior = read_run_config_json(config_json)
        except Exception:  # noqa: BLE001 - treat unparsable as overwritable
            prior = {}
        if prior.get("status") == "complete" and not args.overwrite:
            log.error(
                "run dir %s already holds a completed run; pass --overwrite to re-run",
                run_dir,
            )
            return 1

    collection = load_process_collection(cfg.data.prepared)

    if args.overwrite:
        _clear_output_dir_for_overwrite(
            run_dir,
            input_paths=(args.config, cfg.data.prepared, cfg.custom_py),
        )

    # Assemble the FAIR run directory.
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "model").mkdir(exist_ok=True)
    (run_dir / "checkpoints").mkdir(exist_ok=True)
    bundled_custom = None
    if cfg.custom_py is not None:
        shutil.copyfile(cfg.custom_py, run_dir / "custom.py")
        bundled_custom = "custom.py"

    document = {
        "status": "running",
        "started_at": _now_iso(),
        "cli_argv": list(sys.argv),
        "config": run_config_to_jsonable(cfg),
        "inputs": {
            "prepared_input": {
                "path": str(cfg.data.prepared),
                "content_hash": content_hash(collection),
            },
            "custom_py": {
                "bundled": bundled_custom,
                "file_hash": (
                    f"sha256:{loaded.custom_py_sha256}"
                    if loaded.custom_py_sha256
                    else None
                ),
            },
        },
        "environment": _environment_versions(),
    }
    write_json(config_json, document)

    config = train_harness_config_from_run_config(cfg, run_dir=run_dir)
    try:
        prepared = prepare_training(
            collection,
            config=config,
            custom_module=loaded.custom_module,
            run_config=cfg,
        )
        del collection
        result = train_collection(
            prepared.store,
            reaction_module=prepared.reaction_module,
            loss_module=prepared.loss_module,
            config=prepared.config,
            optimizer=prepared.optimizer,
        )
        eval_processes = tuple(
            prepared.config.process_names or prepared.store.process_order
        )
        prediction_processes = _select_prediction_processes(
            cfg.output.predictions,
            eval_processes,
            prepared.prediction_parent_process_names,
        )
        forward_result = evaluate_trained_wrapper(
            result.trained_wrapper,
            prepared.store,
            config=ForwardConfig(
                process_names=eval_processes,
                target_variable_order=prepared.config.target_variable_order,
                target_source=prepared.config.target_source,
                solver_max_steps=prepared.config.solver_max_steps,
                solver_rtol=prepared.config.solver_rtol,
                solver_atol=prepared.config.solver_atol,
                solver_use_jump_ts=prepared.config.solver_use_jump_ts,
            ),
            target_names=tuple(prepared.loss_module.loss_names),
            training_process_names=eval_processes,
            prediction_process_names=prediction_processes,
        )
        _write_train_results(
            output_dir=run_dir,
            forward_result=forward_result,
            prediction_processes=prediction_processes,
        )
    except Exception as exc:  # noqa: BLE001 - record failure, then re-raise
        update_json(
            config_json,
            status="failed",
            error={"type": type(exc).__name__, "message": str(exc)},
            finished_at=_now_iso(),
        )
        raise

    first = result.mean_loss_by_step[0]
    last = result.mean_loss_by_step[-1]
    log.info(
        "training complete: first_mean_loss=%.6g last_mean_loss=%.6g delta=%.6g",
        first,
        last,
        last - first,
    )
    _finalize_run_dir(run_dir, result, config_json)
    log.info("run directory: %s", run_dir)
    return 0


def _format_loss_table(result: ForwardResult) -> tuple[str, list[list[str]]]:
    """Render a processes × (total + per-target) loss table.

    Returns ``(printable_string, csv_rows)`` where ``csv_rows[0]`` is the header.
    """
    training_set = set(result.training_process_names)
    headers = ["process", "total", *result.target_names, "split"]
    csv_rows: list[list[str]] = [headers]

    data_rows: list[list[str]] = []
    total_sum = 0.0
    per_target_sum = [0.0] * len(result.target_names)
    train_total_sum = 0.0
    train_per_target_sum = [0.0] * len(result.target_names)
    n_train = 0
    holdout_total_sum = 0.0
    holdout_per_target_sum = [0.0] * len(result.target_names)
    n_holdout = 0
    for name in result.process_names:
        total = result.per_process_total_loss[name]
        per_target = result.per_process_per_target_loss[name]
        split = "train" if name in training_set else "holdout"
        data_rows.append(
            [name, f"{total:.6g}"] + [f"{v:.6g}" for v in per_target] + [split]
        )
        csv_rows.append(
            [name, f"{total:.6g}"] + [f"{v:.6g}" for v in per_target] + [split]
        )
        total_sum += total
        for i, v in enumerate(per_target):
            per_target_sum[i] += v
        if split == "train":
            train_total_sum += total
            n_train += 1
            for i, v in enumerate(per_target):
                train_per_target_sum[i] += v
        else:
            holdout_total_sum += total
            n_holdout += 1
            for i, v in enumerate(per_target):
                holdout_per_target_sum[i] += v

    n = max(len(result.process_names), 1)
    mean_row = (
        ["total (mean)", f"{total_sum / n:.6g}"]
        + [f"{v / n:.6g}" for v in per_target_sum]
        + [""]
    )
    data_rows.append(mean_row)
    csv_rows.append(mean_row)

    if n_train:
        train_mean_row = (
            ["train (mean)", f"{train_total_sum / n_train:.6g}"]
            + [f"{v / n_train:.6g}" for v in train_per_target_sum]
            + ["train"]
        )
        data_rows.append(train_mean_row)
        csv_rows.append(train_mean_row)

    if n_holdout:
        holdout_mean_row = (
            ["holdout (mean)", f"{holdout_total_sum / n_holdout:.6g}"]
            + [f"{v / n_holdout:.6g}" for v in holdout_per_target_sum]
            + ["holdout"]
        )
        data_rows.append(holdout_mean_row)
        csv_rows.append(holdout_mean_row)

    col_widths = [len(h) for h in headers]
    for row in data_rows:
        for i, cell in enumerate(row):
            col_widths[i] = max(col_widths[i], len(cell))

    def _fmt_row(row: list[str]) -> str:
        return " | ".join(cell.ljust(col_widths[i]) for i, cell in enumerate(row))

    # Summary rows: total (mean) + optional train (mean) + optional holdout (mean).
    n_summary = 1 + (1 if n_train else 0) + (1 if n_holdout else 0)
    n_data = len(data_rows) - n_summary

    sep = "-+-".join("-" * w for w in col_widths)
    lines = ["LOSSES (forward evaluation)", _fmt_row(headers), sep]
    for row in data_rows[:n_data]:
        lines.append(_fmt_row(row))
    lines.append(sep)
    for row in data_rows[n_data:]:
        lines.append(_fmt_row(row))
    return "\n".join(lines), csv_rows


def _write_loss_csv(rows: list[list[str]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    headers = rows[0]
    data = rows[1:]
    pd.DataFrame(data, columns=headers).to_csv(path, index=False)


def _resolve_forward_run_dir(path: Path, *, max_levels: int = 4) -> Path | None:
    """Return the nearest directory at/above ``path`` that holds config.json."""
    cur = path if path.is_dir() else path.parent
    for _ in range(max_levels + 1):
        if (cur / "config.json").is_file():
            return cur
        if cur.parent == cur:
            break
        cur = cur.parent
    return None


def _resolve_model_bundle(
    path: Path,
) -> tuple[Path, Path, RunConfig, Path | None]:
    """Resolve a model reference to its config, parameters, and data."""
    path = Path(path)
    if not path.exists():
        raise SystemExit(f"forward: model path does not exist: {path}")
    run_dir = _resolve_forward_run_dir(path)
    if run_dir is None:
        raise SystemExit(
            f"forward: no config.json at or above {path}; pass a trained run "
            "directory or a self-contained checkpoint dir."
        )
    model_cfg, _document = read_run_config_json(run_dir / "config.json")
    if path.is_file() and path.name == "params.eqx":
        params = path
    elif path.is_dir() and (path / "params.eqx").is_file():
        params = path / "params.eqx"
    elif (run_dir / "model" / "params.eqx").is_file():
        params = run_dir / "model" / "params.eqx"
    elif (run_dir / "params.eqx").is_file():
        params = run_dir / "params.eqx"
    else:
        raise SystemExit(f"forward: no params.eqx found for model {path}")
    if (run_dir / "prepared.json.gz").is_file():
        own_prepared: Path | None = run_dir / "prepared.json.gz"
    elif (run_dir / "prepared.json").is_file():
        own_prepared = run_dir / "prepared.json"
    elif model_cfg.data is not None:
        own_prepared = Path(model_cfg.data.prepared)
    else:
        own_prepared = None
    return run_dir, params, model_cfg, own_prepared


def _resolve_model_names(models: tuple[Any, ...]) -> list[str]:
    """Per-model name: explicit ``name`` else basename of the bundle's
    ``config.output.dir`` (run identity); de-duplicated with ``#2``/``#3``."""
    raw: list[str] = []
    for ref in models:
        if ref.name:
            raw.append(str(ref.name))
            continue
        run_dir = _resolve_forward_run_dir(Path(ref.path))
        nm: str | None = None
        if run_dir is not None:
            try:
                cfg, _ = read_run_config_json(run_dir / "config.json")
                nm = Path(cfg.output.dir).name
            except Exception:  # noqa: BLE001 - fall back to the path basename
                nm = None
        raw.append(nm or Path(ref.path).name)
    seen: dict[str, int] = {}
    out: list[str] = []
    for nm in raw:
        seen[nm] = seen.get(nm, 0) + 1
        out.append(nm if seen[nm] == 1 else f"{nm}#{seen[nm]}")
    return out


def _handle_forward(args: argparse.Namespace) -> int:
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    log = logging.getLogger(__name__)

    # --- Build the forward config (models + data + output all from the file) ---
    fcfg = load_forward_config(args.config)

    models = fcfg.models
    shared_prepared = fcfg.data.prepared if fcfg.data is not None else None
    config_processes = fcfg.data.processes if fcfg.data is not None else None

    if len(models) > 1 and shared_prepared is None:
        log.error(
            "ensemble forward (>1 model) needs a shared `data.prepared`; add a "
            "`data` block with `prepared` so per-model predictions align."
        )
        return 1

    names = _resolve_model_names(models)

    # --- Output directory (resolved up-front so the re-run guard fails fast) ---
    if args.output_dir is not None:
        output_dir = Path(args.output_dir)
    elif fcfg.output.dir is not None:
        output_dir = Path(fcfg.output.dir)
    else:
        first_run_dir, *_ = _resolve_model_bundle(models[0].path)
        output_dir = first_run_dir / "forward"
    if (output_dir / "losses.csv").is_file() and not args.overwrite:
        log.error(
            "forward output dir %s already holds results; pass --overwrite to "
            "replace them",
            output_dir,
        )
        return 1
    if args.overwrite:
        _clear_output_dir_for_overwrite(
            output_dir,
            input_paths=(
                args.config,
                shared_prepared,
                *(ref.path for ref in models),
            ),
        )
    output_dir.mkdir(parents=True, exist_ok=True)

    # --- Forward each model on its data ---
    per_model: list[tuple[str, Any]] = []  # (name, ForwardResult)
    prediction_processes: tuple[str, ...] = ()
    for ref, name in zip(models, names):
        _run_dir, params_path, model_cfg, own_prepared = _resolve_model_bundle(ref.path)
        prepared = shared_prepared if shared_prepared is not None else own_prepared
        if prepared is None:
            log.error("forward: could not resolve a prepared.json for model %s", name)
            return 1
        # The custom.py rebuilds the model's reaction module / loss hooks; without
        # it forward_from_collection builds the default module (wrong pytree).
        # Prefer the original recorded path (its sibling helper modules are on
        # disk on this machine); fall back to the bundled copy for a checkpoint
        # that was sent elsewhere.
        if model_cfg.custom_py is not None and Path(model_cfg.custom_py).is_file():
            custom_py: str | None = str(model_cfg.custom_py)
        elif (_run_dir / "custom.py").is_file():
            custom_py = str(_run_dir / "custom.py")
        else:
            custom_py = None
        collection = load_process_collection(resolve_prepared_path(Path(prepared)))
        eval_processes = (
            tuple(config_processes)
            if config_processes
            else tuple(collection.processes.keys())
        )
        unknown_processes = tuple(
            name for name in eval_processes if name not in collection.processes
        )
        if unknown_processes:
            log.error(
                "forward: unknown process names %s; available=%s",
                list(unknown_processes),
                list(collection.processes),
            )
            return 1
        model_targets = model_cfg.data.targets if model_cfg.data is not None else None
        model_source = (
            model_cfg.data.target_source if model_cfg.data is not None else "auto"
        )
        recorded_processes = (
            tuple(model_cfg.data.processes)
            if model_cfg.data is not None and model_cfg.data.processes
            else ()
        )
        if recorded_processes:
            training_processes = recorded_processes
        else:
            # A run that recorded no data.processes trained on every process of
            # its own prepared input. This value becomes the constructor-hook
            # process_names, so the module shape it implies must match training;
            # the evaluation selection would imply a different shape whenever it
            # is a subset. Without a shared prepared input `collection` IS the
            # model's own prepared input, so the full order is exact. With one it
            # is a guess, because the CLI never reconciles the model's own
            # prepared input; verifying the recorded content hash is Phase 5.
            training_processes = tuple(collection.processes)
            if shared_prepared is not None:
                log.warning(
                    "forward: model %s recorded no data.processes; assuming it "
                    "trained on all %d processes of the shared prepared input. "
                    "Constructor-hook shapes may not match training.",
                    ref.path,
                    len(collection.processes),
                )
        fwd_cfg = ForwardConfig(
            process_names=eval_processes,
            target_variable_order=model_targets,
            target_source=model_source,
            solver_max_steps=int(model_cfg.solver.max_steps),
            solver_rtol=float(model_cfg.solver.rtol),
            solver_atol=float(model_cfg.solver.atol),
            solver_use_jump_ts=bool(model_cfg.solver.jump_ts),
        )
        parent_processes = original_parent_processes(
            tuple(collection.processes),
            tuple(
                getattr(process, "parent_process", None)
                for process in collection.processes.values()
            ),
        )
        prediction_processes = _select_prediction_processes(
            fcfg.output.predictions, eval_processes, parent_processes
        )
        result = forward_from_collection(
            collection,
            model_path=params_path,
            config=fwd_cfg,
            custom_py=custom_py,
            run_config=model_cfg,
            training_process_names=training_processes,
            prediction_process_names=prediction_processes,
        )
        per_model.append((name, result))

    wrapper0 = per_model[0][1].trained_wrapper

    # --- Per-model predictions + loss tables ---
    for name, result in per_model:
        mdir = output_dir / "models" / name
        mdir.mkdir(parents=True, exist_ok=True)
        model_predictions_path = mdir / "predictions.csv"
        if prediction_processes:
            export_predictions_csv(
                result.trained_wrapper,
                result.dense_exports,
                model_predictions_path,
                process_names=prediction_processes,
            )
        else:
            model_predictions_path.unlink(missing_ok=True)
        _table, model_rows = _format_loss_table(result)
        _write_loss_csv(model_rows, mdir / "losses.csv")

    # --- Aggregate (mean + std across models) ---
    if prediction_processes:
        per_model_dense = [r.dense_exports for _n, r in per_model]
        if len(per_model_dense) > 1:
            mean_exports, std_exports = aggregate_dense_exports(per_model_dense)
        else:
            mean_exports, std_exports = per_model_dense[0], None
        export_predictions_csv(
            wrapper0,
            mean_exports,
            output_dir / "predictions.csv",
            process_names=prediction_processes,
        )
        predictions_std_path = output_dir / "predictions_std.csv"
        if std_exports is not None:
            export_predictions_csv(
                wrapper0,
                std_exports,
                predictions_std_path,
                process_names=prediction_processes,
            )
        else:
            predictions_std_path.unlink(missing_ok=True)
    else:
        (output_dir / "predictions.csv").unlink(missing_ok=True)
        (output_dir / "predictions_std.csv").unlink(missing_ok=True)

    # --- Loss table (representative = first model; per-model in models/<name>/) ---
    table_str, csv_rows = _format_loss_table(per_model[0][1])
    log.info("\n%s", table_str)
    _write_loss_csv(csv_rows, output_dir / "losses.csv")

    return 0


def _handle_loo(args: argparse.Namespace) -> int:
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    log = logging.getLogger(__name__)

    if args.produce_runtime or args.runtime_artifact is not None:
        if args.resume is not None or args.config is None:
            log.error("internal LOO modes require --config and cannot resume")
            return 1
        loaded = load_loo_config(args.config)
        cfg = _apply_train_cli_overrides(
            loaded.config, args, resolve_output_symlinks=False
        )
        if cfg.data is None:
            raise ValueError("LOO command requires a data config section")
        output_dir = Path(cfg.output.dir)
        if args.produce_runtime:
            if args.fold is not None:
                log.error("--produce-runtime cannot select a fold")
                return 1
            produce_runtime_artifact(
                cfg=cfg,
                custom_module=loaded.custom_module,
                output_dir=output_dir,
                bundle_path=Path(args.config),
            )
            return 0
        if args.fold is None:
            log.error("--runtime-artifact requires --fold")
            return 1
        prepared_fold = prepare_single_fold_from_runtime_artifact(
            cfg=cfg,
            custom_module=loaded.custom_module,
            output_dir=output_dir,
            bundle_path=Path(args.config),
            artifact_path=Path(args.runtime_artifact),
            fold_idx=args.fold,
        )
        trained_fold = train_prepared_fold(prepared_fold)
        del prepared_fold
        execute_trained_fold(trained_fold)
        return 0

    # ---- resume: reload only the self-contained bundle and runtime state ----
    if args.resume is not None:
        if args.config is not None:
            log.error("loo: --resume and --config are mutually exclusive")
            return 1
        resume_dir = Path(args.resume).resolve()
        bundle = resume_dir / "loo-config.json"
        if not bundle.is_file():
            log.error(
                "loo --resume: %s has no loo-config.json (not a LOO run dir)",
                resume_dir,
            )
            return 1
        config_json = resume_dir / "config.json"
        if not config_json.is_file():
            log.error(
                "loo --resume: %s has no config.json; initialization did not "
                "complete, use --overwrite to replace it",
                resume_dir,
            )
            return 1
        loaded = load_loo_config(bundle)
        cfg = loaded.config
        if cfg.data is None:
            raise ValueError("LOO run dir config is missing a data section")
        # Validate before the first status write so a rejected resume leaves the
        # run record untouched; run_loo_cv revalidates from the same manifest.
        _runtime_metadata(resume_dir, bundle_path=bundle, custom_path=cfg.custom_py)
        update_json(config_json, status="running", resumed_at=_now_iso())
        try:
            result = run_loo_cv(
                cfg=cfg,
                config_path=bundle,
                output_dir=resume_dir,
                resume=True,
            )
        except Exception as exc:  # noqa: BLE001 - record failure, then re-raise
            update_json(
                config_json,
                status="failed",
                error={"type": type(exc).__name__, "message": str(exc)},
                finished_at=_now_iso(),
            )
            raise
        update_json(
            config_json,
            status="complete",
            finished_at=_now_iso(),
            n_folds=len(result.fold_dirs),
            parallel_folds=result.parallel_folds,
            devices_per_fold=result.devices_per_fold,
            aggregate=result.aggregate,
        )
        log.info(
            "LOO resume complete: %d fold(s); aggregate=%s",
            len(result.fold_dirs),
            result.aggregate,
        )
        return 0

    if args.config is None:
        log.error("loo requires --config (or --resume <run_dir> to continue)")
        return 1

    loaded = load_loo_config(args.config)
    cfg = _apply_train_cli_overrides(loaded.config, args, resolve_output_symlinks=False)
    if cfg.data is None:
        raise ValueError("loo command requires a data config section")
    output_dir = Path(cfg.output.dir)

    config_json = output_dir / "config.json"
    if args.overwrite:
        _clear_output_dir_for_overwrite(
            output_dir,
            input_paths=(args.config, cfg.data.prepared, cfg.custom_py),
        )
        if output_dir.exists():
            output_dir.rmdir()
    try:
        output_dir.mkdir(parents=True, exist_ok=False)
    except FileExistsError:
        log.error(
            "LOO output dir %s already exists; pass --overwrite to replace it "
            "or --resume to continue it",
            output_dir,
        )
        return 1

    bundle_path = _bundle_loo_run_dir(
        raw_config_path=args.config, cfg=cfg, output_dir=output_dir
    )
    # Reload the bundle so every child and the parent use the same relative,
    # movable config identity. This path deliberately does not deserialize data.
    bundled = load_loo_config(bundle_path)
    bundled_cfg = bundled.config
    document = {
        "status": "running",
        "started_at": _now_iso(),
        "cli_argv": list(sys.argv),
        "config": run_config_to_jsonable(cfg),
        "inputs": {
            "prepared_input": {"path": str(cfg.data.prepared)},
            "custom_py": {
                "bundled": "custom.py" if cfg.custom_py is not None else None,
                "file_hash": (
                    f"sha256:{loaded.custom_py_sha256}"
                    if loaded.custom_py_sha256
                    else None
                ),
            },
        },
        "environment": _environment_versions(),
    }
    try:
        producer_rc = _dispatch_producer(bundle_path, output_dir)
        if producer_rc != 0:
            raise RuntimeError(f"LOO runtime producer failed: exit {producer_rc}")
        metadata = _validated_runtime_metadata(
            output_dir / "runtime-artifact",
            bundle_path=bundle_path,
            custom_path=bundled_cfg.custom_py,
        )
        document["runtime_artifact"] = {
            "format_version": FORMAT_VERSION,
            "identity": metadata.identity,
        }
        write_json(config_json, document)
    except Exception:
        if not config_json.exists():
            shutil.rmtree(output_dir / "runtime-artifact", ignore_errors=True)
        raise

    try:
        result = run_loo_cv(
            cfg=bundled_cfg,
            config_path=bundle_path,
            output_dir=output_dir,
        )
    except Exception as exc:  # noqa: BLE001 - record failure, then re-raise
        update_json(
            config_json,
            status="failed",
            error={"type": type(exc).__name__, "message": str(exc)},
            finished_at=_now_iso(),
        )
        raise

    update_json(
        config_json,
        status="complete",
        finished_at=_now_iso(),
        n_folds=len(result.fold_dirs),
        parallel_folds=result.parallel_folds,
        devices_per_fold=result.devices_per_fold,
        aggregate=result.aggregate,
    )
    log.info(
        "LOO complete: %d fold(s) (%d parallel x %d device(s) each); aggregate=%s",
        len(result.fold_dirs),
        result.parallel_folds,
        result.devices_per_fold,
        result.aggregate,
    )
    return 0


def _bundle_loo_run_dir(
    *, raw_config_path: str, cfg: RunConfig, output_dir: Path
) -> Path:
    """Materialise a self-contained LOO run dir.

    Copies ``custom.py`` and the prepared artifact (true byte copies) into
    ``output_dir`` and writes a loadable ``loo-config.json`` whose ``custom_py``,
    ``data.prepared`` and ``output.dir`` are RELATIVE to the run dir (so it stays
    valid even if the whole dir is moved). Returns the bundled config path.
    """
    assert cfg.data is not None
    src_prepared = Path(cfg.data.prepared)
    prepared_name = (
        "prepared.json.gz" if src_prepared.name.endswith(".gz") else "prepared.json"
    )
    shutil.copyfile(src_prepared, output_dir / prepared_name)

    custom_name: str | None = None
    if cfg.custom_py is not None:
        shutil.copyfile(cfg.custom_py, output_dir / "custom.py")
        custom_name = "custom.py"

    raw = load_json(raw_config_path)
    raw.setdefault("data", {})["prepared"] = prepared_name
    if custom_name is not None:
        raw["custom_py"] = custom_name
    else:
        raw.pop("custom_py", None)
    raw.setdefault("output", {})["dir"] = "."

    bundle_path = output_dir / "loo-config.json"
    write_json(bundle_path, raw)
    return bundle_path


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    handler = getattr(args, "handler")
    return int(handler(args))


if __name__ == "__main__":
    raise SystemExit(main())
