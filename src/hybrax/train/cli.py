from __future__ import annotations

import argparse
import json
import logging
import os
import shutil
import sys
import time
from pathlib import Path
from typing import Any

from bp_format.serialization import load_process_collection_json
import pandas as pd

from .harness import (
    ForwardConfig,
    ForwardResult,
    TrainHarnessConfig,
    forward_from_collection,
    resume_run,
    train_from_collection,
    train_harness_config_from_run_config,
)
from .loo import LOOConfig, run_loo_cv
from .loo_metrics import compute_loo_metrics
from .postprocessing import (
    export_observations_csv,
    load_model_metadata,
    plot_process_simulations,
    plot_training_results,
    save_model,
    save_model_metadata,
)
from .prepare import prepare_artifact
from .run_config import LoadedRunConfig, RunConfig, load_prepare_config, load_train_config
from .serialization import (
    content_hash,
    environment_versions as _environment_versions,
    read_json,
    read_run_config_json,
    run_config_to_jsonable,
    save_model as save_params_model,
    save_opt_state,
    update_run_config_status,
    write_json,
)
from .training_data import TARGET_SOURCES, TrainingDataStore
from .utils import load_custom_module, resolve_config


def _now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S%z", time.localtime())


def _load_config(config_path: str | None) -> dict[str, Any] | None:
    if config_path is None:
        return None
    return json.loads(Path(config_path).read_text(encoding="utf-8"))


def _build_parser() -> argparse.ArgumentParser:
    train_cfg_defaults = TrainHarnessConfig()
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
    prepare_parser.add_argument("--output", required=True, help="Path to output JSON.")
    prepare_parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite an existing prepared.json at --output.",
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
        help="Path to train run config JSON (required unless --resume).",
    )
    train_parser.add_argument(
        "--output-dir",
        default=None,
        help="Override output.dir from the config (the FAIR run directory).",
    )
    train_parser.add_argument(
        "--resume",
        default=None,
        help=(
            "Resume training in place from an existing run directory "
            "(continues from checkpoints/latest, appending to metrics.csv). "
            "Combine with --steps to extend the original target."
        ),
    )
    train_parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow re-running into a run dir that already completed.",
    )
    train_parser.add_argument(
        "--steps",
        type=int,
        default=None,
        help="Override optim.steps (with --resume, may extend the target).",
    )
    plot_group = train_parser.add_mutually_exclusive_group()
    plot_group.add_argument(
        "--plot",
        dest="plot",
        action="store_true",
        help="Generate per-process result plots (default).",
    )
    plot_group.add_argument(
        "--no-plot",
        dest="plot",
        action="store_false",
        help="Skip plot generation.",
    )
    train_parser.set_defaults(plot=True)
    # Cadence / console-table formatting are config-only now (the `logging`
    # section: every, decimals, header_every; metrics.csv is always written to
    # the run dir). The old --log-every / --metrics-csv / --metrics-jsonl /
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
            "process (no training). Regenerates plots and prints a loss table."
        ),
    )
    forward_parser.add_argument(
        "--model",
        required=True,
        help="Path to trained_wrapper.eqx.",
    )
    forward_parser.add_argument(
        "--input",
        help=(
            "Path to prepared.json. If omitted, read from the sidecar "
            "`<model>.meta.json`."
        ),
    )
    forward_parser.add_argument(
        "--custom",
        help="Optional custom.py path. Defaults to the sidecar value.",
    )
    forward_parser.add_argument(
        "--config",
        help="Optional legacy JSON runtime config for forward hooks.",
    )
    forward_parser.add_argument(
        "--process",
        action="append",
        default=[],
        help=(
            "Process name to evaluate. May be repeated or comma-separated. "
            "Defaults to every process in the input collection."
        ),
    )
    forward_parser.add_argument(
        "--target",
        action="append",
        default=[],
        help=(
            "Target variable order override. Normally inferred from "
            "the sidecar."
        ),
    )
    forward_parser.add_argument(
        "--target-source",
        default=None,
        choices=sorted(TARGET_SOURCES),
        help="Override the target-source resolution (defaults to sidecar/train).",
    )
    forward_parser.add_argument(
        "--solver-max-steps",
        type=int,
        default=None,
        help="Override sidecar solver max_steps.",
    )
    forward_parser.add_argument(
        "--solver-rtol",
        type=float,
        default=None,
        help="Override sidecar solver rtol.",
    )
    forward_parser.add_argument(
        "--solver-atol",
        type=float,
        default=None,
        help="Override sidecar solver atol.",
    )
    forward_parser.add_argument(
        "--no-jump-ts",
        action="store_true",
        help="Disable passing control step boundaries as jump_ts to the solver.",
    )
    forward_parser.add_argument(
        "--output-dir",
        default=None,
        help="Directory for forward outputs. Defaults to <model_dir>/forward.",
    )
    fwd_plot_group = forward_parser.add_mutually_exclusive_group()
    fwd_plot_group.add_argument(
        "--plot",
        dest="plot",
        action="store_true",
        help="Generate per-process plots (default).",
    )
    fwd_plot_group.add_argument(
        "--no-plot",
        dest="plot",
        action="store_false",
        help="Skip plot generation.",
    )
    forward_parser.set_defaults(plot=True)
    forward_parser.add_argument(
        "--loss-csv",
        default=None,
        help="Write the loss table to this CSV. Default: <output-dir>/losses.csv.",
    )
    forward_parser.add_argument(
        "--timeseries-csv",
        default=None,
        help=(
            "Write a single merged CSV of dense simulated trajectories with a "
            "`process` column."
        ),
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
            "Run leave-one-process-out cross-validation: train one fold per "
            "parent process group, evaluate each fold's holdout, and "
            "aggregate results."
        ),
    )
    loo_parser.add_argument(
        "--input",
        required=True,
        help="Path to prepared JSON.",
    )
    loo_parser.add_argument(
        "--custom",
        help="Optional custom.py path exposing build_reaction_module hooks.",
    )
    loo_parser.add_argument(
        "--config",
        help="Optional JSON runtime config.",
    )
    loo_parser.add_argument(
        "--holdouts",
        action="append",
        default=[],
        help=(
            "Parent process names to use as holdouts. Repeatable or "
            "comma-separated. Defaults to all parents. Pass exactly one "
            "name to run a single fold (cluster-friendly)."
        ),
    )
    loo_parser.add_argument(
        "--target",
        action="append",
        default=[],
        help=(
            "Target variable name to train against. Repeatable or "
            "comma-separated."
        ),
    )
    loo_parser.add_argument(
        "--target-source",
        default=train_cfg_defaults.target_source,
        choices=sorted(TARGET_SOURCES),
        help="Source family for training targets.",
    )
    loo_parser.add_argument(
        "--steps",
        type=int,
        default=train_cfg_defaults.steps,
        help="Number of training steps per fold.",
    )
    loo_parser.add_argument(
        "--batch-size",
        type=int,
        help="Batch size. Defaults to the number of training processes per fold.",
    )
    loo_parser.add_argument(
        "--batch-seed",
        type=int,
        help="Seed used for batch index generation (per fold).",
    )
    loo_parser.add_argument(
        "--optimizer",
        default=train_cfg_defaults.optimizer_name,
        choices=["adam", "sgd"],
        help="Optimizer to use for batched updates.",
    )
    loo_shuffle_group = loo_parser.add_mutually_exclusive_group()
    loo_shuffle_group.add_argument(
        "--shuffle-batches",
        dest="shuffle_batches",
        action="store_true",
        help="Shuffle selected processes when building batches.",
    )
    loo_shuffle_group.add_argument(
        "--no-shuffle-batches",
        dest="shuffle_batches",
        action="store_false",
        help="Keep batch construction deterministic and round-robin.",
    )
    loo_parser.set_defaults(shuffle_batches=train_cfg_defaults.shuffle_batches)
    loo_parser.add_argument(
        "--learning-rate",
        type=float,
        default=train_cfg_defaults.learning_rate,
        help="Learning rate (overridden by build_learning_rate hook).",
    )
    loo_parser.add_argument(
        "--grad-clip-norm",
        type=float,
        default=train_cfg_defaults.grad_clip_norm,
        help="Global gradient-norm clipping threshold; 0 disables clipping.",
    )
    loo_parser.add_argument(
        "--seed",
        type=int,
        default=train_cfg_defaults.seed,
        help=(
            "Base seed. Each fold uses seed = base + fold_idx so different "
            "folds get distinct, deterministic initializations."
        ),
    )
    loo_parser.add_argument(
        "--log-every",
        type=int,
        default=train_cfg_defaults.log_every,
        help="Emit progress log every N steps.",
    )
    loo_parser.add_argument(
        "--solver-max-steps",
        type=int,
        default=train_cfg_defaults.solver_max_steps,
    )
    loo_parser.add_argument(
        "--solver-rtol",
        type=float,
        default=train_cfg_defaults.solver_rtol,
    )
    loo_parser.add_argument(
        "--solver-atol",
        type=float,
        default=train_cfg_defaults.solver_atol,
    )
    loo_parser.add_argument(
        "--no-jump-ts",
        action="store_true",
        help="Disable passing control step boundaries as jump_ts to the solver.",
    )
    loo_parser.add_argument(
        "--output-dir",
        default="output/loo",
        help="Directory for fold artifacts (default: ./output/loo).",
    )
    loo_parser.add_argument(
        "--checkpoint-dir",
        default=None,
        help=(
            "Per-fold checkpoint directory under each fold's output. "
            "Defaults to <fold_dir>/checkpoints. Empty string disables."
        ),
    )
    loo_plot_group = loo_parser.add_mutually_exclusive_group()
    loo_plot_group.add_argument(
        "--plot",
        dest="plot",
        action="store_true",
        help="Generate per-fold result plots (default).",
    )
    loo_plot_group.add_argument(
        "--no-plot",
        dest="plot",
        action="store_false",
        help="Skip plot generation.",
    )
    loo_parser.set_defaults(plot=True)
    loo_parser.add_argument(
        "--log-process-losses",
        action="store_true",
        default=train_cfg_defaults.log_process_losses,
    )
    loo_parser.add_argument(
        "--log-decimals",
        type=int,
        default=train_cfg_defaults.log_decimals,
    )
    loo_parser.add_argument(
        "--log-header-every",
        type=int,
        default=train_cfg_defaults.log_header_every,
    )
    loo_parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
    )
    loo_parser.set_defaults(handler=_handle_loo)

    return parser


def _handle_prepare(args: argparse.Namespace) -> int:
    output = Path(args.output)
    if output.exists() and not args.overwrite:
        logging.getLogger(__name__).error(
            "prepared artifact already exists at %s; pass --overwrite to replace it",
            output,
        )
        return 1
    prepare_artifact(
        load_prepare_config(args.config),
        output_json=args.output,
    )
    return 0


def _split_multi_values(raw_values: list[str]) -> tuple[str, ...]:
    values: list[str] = []
    for value in raw_values:
        values.extend([part.strip() for part in value.split(",") if part.strip() != ""])
    return tuple(values)


def _write_train_results(
    *,
    output_dir: Path,
    collection: Any,
    trained_wrapper: Any,
    train_result: Any,
    config: TrainHarnessConfig,
    runtime_config: dict[str, Any] | None,
    custom_py: str | Path | None,
    training_process_names: tuple[str, ...],
    render_plots: bool,
    eval_process_names: tuple[str, ...] | None = None,
    run_config: Any | None = None,
    custom_module: Any | None = None,
) -> ForwardResult:
    """Write per-run forward artifacts (losses.csv, predictions.csv, plots).

    By default forward runs on ``training_process_names``. Pass
    ``eval_process_names`` (e.g. for LOO, where holdout processes also need
    losses) to evaluate a different set; the train/holdout split label in
    ``losses.csv`` is always derived from ``training_process_names``.

    Returns the :class:`ForwardResult` so callers can reuse per-process
    losses without rerunning the forward pass.
    """
    log = logging.getLogger(__name__)
    eval_processes = (
        eval_process_names if eval_process_names is not None else training_process_names
    )

    fwd_cfg = ForwardConfig(
        process_names=eval_processes,
        target_variable_order=config.target_variable_order,
        target_source=config.target_source,
        solver_max_steps=config.solver_max_steps,
        solver_rtol=config.solver_rtol,
        solver_atol=config.solver_atol,
        solver_use_jump_ts=config.solver_use_jump_ts,
    )
    fwd_result = forward_from_collection(
        collection,
        model_path=output_dir / "trained_wrapper.eqx",
        config=fwd_cfg,
        custom_py=custom_py,
        runtime_config=runtime_config,
        training_process_names=training_process_names,
        run_config=run_config,
        custom_module=custom_module,
    )

    _table, csv_rows = _format_loss_table(fwd_result)
    loss_csv_path = output_dir / "losses.csv"
    _write_loss_csv(csv_rows, loss_csv_path)
    log.info("loss table saved to %s", loss_csv_path)

    predictions_csv_path = output_dir / "predictions.csv"
    if render_plots:
        plot_training_results(
            train_result,
            collection,
            fwd_result.store,
            output_dir,
            # Use eval_processes (full set including holdouts in LOO) so the
            # predictions.csv has rows for every evaluated process. Per-process
            # plots are rendered for the same set.
            process_names=eval_processes,
            solver_max_steps=config.solver_max_steps,
            solver_rtol=config.solver_rtol,
            solver_atol=config.solver_atol,
            solver_use_jump_ts=config.solver_use_jump_ts,
            timeseries_csv_path=predictions_csv_path,
        )
        return fwd_result

    plot_process_simulations(
        trained_wrapper,
        collection,
        fwd_result.store,
        output_dir,
        process_names=eval_processes,
        solver_max_steps=config.solver_max_steps,
        solver_rtol=config.solver_rtol,
        solver_atol=config.solver_atol,
        solver_use_jump_ts=config.solver_use_jump_ts,
        training_process_names=training_process_names,
        timeseries_csv_path=predictions_csv_path,
        render_plots=False,
    )
    return fwd_result


def _apply_train_cli_overrides(
    cfg: RunConfig, args: argparse.Namespace
) -> RunConfig:
    """Apply the few CLI flags that override the config file (CLI wins)."""
    updates: dict[str, Any] = {}
    if args.output_dir is not None:
        updates["output"] = cfg.output.model_copy(
            update={"dir": Path(args.output_dir).resolve()}
        )
    if not args.plot:  # --no-plot
        base = updates.get("output", cfg.output)
        updates["output"] = base.model_copy(update={"plots": False})
    if args.steps is not None:
        updates["train"] = cfg.train.model_copy(update={"steps": int(args.steps)})
    return cfg.model_copy(update=updates) if updates else cfg


def _finalize_run_dir(run_dir: Path, result: Any, config_json: Path) -> None:
    """Copy best→model/, then mark the run complete in config.json."""
    model_dir = run_dir / "model"
    model_dir.mkdir(parents=True, exist_ok=True)
    best = run_dir / "checkpoints" / "best"
    best_info: dict[str, Any] | None = None
    if best.is_symlink() and (best / "params.eqx").is_file():
        shutil.copyfile(best / "params.eqx", model_dir / "params.eqx")
        if (best / "opt_state.eqx").is_file():
            shutil.copyfile(best / "opt_state.eqx", model_dir / "opt_state.eqx")
        try:
            best_state = read_json(best / "train_state.json")
            best_info = {
                "step": best_state.get("step"),
                "mean_loss": best_state.get("mean_loss"),
            }
        except OSError:
            best_info = None
    else:
        # No checkpoints written — persist the final state directly.
        save_params_model(result.trained_wrapper, model_dir / "params.eqx")
        if result.optimizer_state is not None:
            save_opt_state(result.optimizer_state, model_dir / "opt_state.eqx")
    final_mean = (
        float(result.mean_loss_by_step[-1]) if result.mean_loss_by_step else None
    )
    update_run_config_status(
        config_json,
        status="complete",
        finished_at=_now_iso(),
        steps_completed=int(getattr(result, "steps_completed", 0)),
        best=best_info,
        final_mean_loss=final_mean,
    )


def _handle_train(args: argparse.Namespace) -> int:
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    log = logging.getLogger(__name__)

    # ---- Resume: continue an existing run dir in place ----
    if args.resume:
        resume_arg = Path(args.resume)
        # Be forgiving: accept the run dir itself OR a sub-path inside it
        # (e.g. checkpoints/latest, checkpoints/step_00500, model/) and resolve
        # up to the directory that actually holds config.json. Resume always
        # continues from checkpoints/latest regardless.
        run_dir = next(
            (
                cand
                for cand in (resume_arg, resume_arg.parent, resume_arg.parent.parent)
                if (cand / "config.json").is_file()
            ),
            None,
        )
        if run_dir is None:
            log.error(
                "--resume: no config.json found at %s or its parent run "
                "directory; pass the RUN directory (e.g. output_single), not a "
                "checkpoint sub-directory",
                resume_arg,
            )
            return 1
        if run_dir != resume_arg:
            log.info(
                "--resume: resolved run directory %s (continuing from latest "
                "checkpoint)",
                run_dir,
            )
        config_json = run_dir / "config.json"
        update_run_config_status(config_json, status="running", resumed_at=_now_iso())
        try:
            result = resume_run(run_dir, steps_override=args.steps)
        except Exception as exc:  # noqa: BLE001 - record failure, then re-raise
            update_run_config_status(
                config_json,
                status="failed",
                error={"type": type(exc).__name__, "message": str(exc)},
                finished_at=_now_iso(),
            )
            raise
        _finalize_run_dir(run_dir, result, config_json)
        log.info("resume complete: %s", run_dir)
        return 0

    # ---- Fresh run ----
    if args.config is None:
        log.error("train requires --config (or --resume <run_dir> to continue)")
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
                "run dir %s already holds a completed run; pass --overwrite to "
                "re-run or --resume to continue",
                run_dir,
            )
            return 1

    collection = load_process_collection_json(cfg.data.prepared)

    # Assemble the FAIR run directory.
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "model").mkdir(exist_ok=True)
    (run_dir / "checkpoints").mkdir(exist_ok=True)
    bundled_custom = None
    if cfg.custom_py is not None:
        shutil.copyfile(cfg.custom_py, run_dir / "custom.py")
        bundled_custom = "custom.py"

    store = TrainingDataStore.from_collection(
        collection,
        target_variable_order=cfg.data.targets,
        target_source=cfg.data.target_source,
    )
    export_observations_csv(
        collection, store, run_dir / "observations.csv", process_names=cfg.data.processes
    )

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
        result = train_from_collection(
            collection,
            config=config,
            custom_module=loaded.custom_module,
            run_config=cfg,
        )
    except Exception as exc:  # noqa: BLE001 - record failure, then re-raise
        update_run_config_status(
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

    # Number of summary rows: total (mean) + optional train (mean) + optional holdout (mean)
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


def _handle_forward(args: argparse.Namespace) -> int:
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    log = logging.getLogger(__name__)

    model_arg = Path(args.model)

    # Resolve the FAIR run directory for the model: --model may point at the run
    # dir itself, or at a checkpoint's params.eqx.
    run_dir: Path | None = None
    if model_arg.is_dir() and (model_arg / "config.json").is_file():
        run_dir = model_arg
    elif (
        model_arg.name == "params.eqx"
        and (model_arg.parent.parent / "config.json").is_file()
    ):
        run_dir = model_arg.parent.parent

    run_config_obj = None
    if run_dir is not None:
        # --- New run-dir layout: solver/prepared/custom are recorded in config.json ---
        run_config_obj, _doc = read_run_config_json(run_dir / "config.json")
        model_path = (
            model_arg
            if model_arg.name == "params.eqx"
            else run_dir / "model" / "params.eqx"
        )
        if not model_path.is_file():
            raise SystemExit(f"no model params at {model_path}")

        bundled_prepared = run_dir / "prepared.json"
        if args.input is not None:
            prepared = str(args.input)
        elif bundled_prepared.is_file():
            prepared = str(bundled_prepared)
        elif run_config_obj.data is not None:
            prepared = str(run_config_obj.data.prepared)
        else:
            raise SystemExit(f"could not resolve prepared.json for run {run_dir}")

        bundled_custom = run_dir / "custom.py"
        if args.custom is not None:
            custom_py = str(args.custom)
        elif bundled_custom.is_file():
            custom_py = str(bundled_custom)
        elif run_config_obj.custom_py is not None:
            custom_py = str(run_config_obj.custom_py)
        else:
            custom_py = None

        # Solver accuracy is read-only from the model's config (reproduces the
        # trajectory the model was fit under); only the safety cap is overridable.
        solver_max_steps = (
            args.solver_max_steps
            if args.solver_max_steps is not None
            else int(run_config_obj.solver.max_steps)
        )
        solver_rtol = float(run_config_obj.solver.rtol)
        solver_atol = float(run_config_obj.solver.atol)
        solver_use_jump_ts = bool(run_config_obj.solver.jump_ts)
        effective_targets = (
            run_config_obj.data.targets if run_config_obj.data is not None else None
        )
        target_source = (
            run_config_obj.data.target_source
            if run_config_obj.data is not None
            else "auto"
        )
        configured_processes = (
            run_config_obj.data.processes if run_config_obj.data is not None else None
        )
        runtime_config = None
    else:
        # --- Legacy layout fallback (no run-dir config.json) ---
        model_path = model_arg
        if not model_path.exists():
            raise SystemExit(f"--model path does not exist: {model_path}")
        if args.input is None:
            raise SystemExit("--input is required when --model is not a run dir")
        prepared = str(args.input)
        custom_py = str(args.custom) if args.custom else None
        solver_max_steps = args.solver_max_steps or 4096
        solver_rtol = args.solver_rtol or 1e-5
        solver_atol = args.solver_atol or 1e-7
        solver_use_jump_ts = not args.no_jump_ts
        cli_targets = _split_multi_values(args.target)
        effective_targets = cli_targets or None
        target_source = args.target_source or "auto"
        configured_processes = None
        runtime_config = _load_config(args.config)

    collection = load_process_collection_json(Path(prepared))

    cli_processes = _split_multi_values(args.process)
    eval_processes = cli_processes or tuple(collection.processes.keys())

    training_processes = (
        tuple(configured_processes)
        if configured_processes
        else tuple(collection.processes.keys())
    )

    fwd_cfg = ForwardConfig(
        process_names=eval_processes,
        target_variable_order=effective_targets,
        target_source=target_source,
        solver_max_steps=solver_max_steps,
        solver_rtol=solver_rtol,
        solver_atol=solver_atol,
        solver_use_jump_ts=solver_use_jump_ts,
    )

    result = forward_from_collection(
        collection,
        model_path=model_path,
        config=fwd_cfg,
        custom_py=custom_py,
        runtime_config=runtime_config,
        training_process_names=training_processes,
        run_config=run_config_obj,
    )

    table_str, csv_rows = _format_loss_table(result)
    log.info("\n%s", table_str)

    if args.output_dir is not None:
        output_dir = Path(args.output_dir)
    elif run_dir is not None:
        output_dir = run_dir / "forward"
    else:
        output_dir = model_path.parent / "forward"
    output_dir.mkdir(parents=True, exist_ok=True)

    loss_csv_path = Path(args.loss_csv) if args.loss_csv else output_dir / "losses.csv"
    _write_loss_csv(csv_rows, loss_csv_path)
    log.info("loss table saved to %s", loss_csv_path)

    if args.plot:
        plot_process_simulations(
            result.trained_wrapper,
            collection,
            result.store,
            output_dir,
            process_names=eval_processes,
            solver_max_steps=solver_max_steps,
            solver_rtol=solver_rtol,
            solver_atol=solver_atol,
            solver_use_jump_ts=solver_use_jump_ts,
            training_process_names=training_processes,
            timeseries_csv_path=args.timeseries_csv,
        )

    return 0


def _handle_loo(args: argparse.Namespace) -> int:
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    log = logging.getLogger(__name__)

    collection = load_process_collection_json(Path(args.input))
    runtime_config = _load_config(args.config)
    custom_module = load_custom_module(args.custom)
    selected_holdouts_raw = _split_multi_values(args.holdouts)
    selected_targets = _split_multi_values(args.target)
    user_config = resolve_config(custom_module, None)
    config_targets = user_config.get("target_variable_order")
    if selected_targets:
        effective_targets = selected_targets
    elif config_targets:
        effective_targets = tuple(config_targets)
    else:
        effective_targets = None

    output_dir = Path(args.output_dir)
    if args.checkpoint_dir is None:
        # default: per-fold <fold_dir>/checkpoints (resolved inside loo.py)
        base_checkpoint_dir: Path | None = output_dir
    elif str(args.checkpoint_dir) == "":
        base_checkpoint_dir = None
    else:
        base_checkpoint_dir = Path(args.checkpoint_dir)

    base_train_config = TrainHarnessConfig(
        process_names=None,  # set per-fold inside loo.py
        target_variable_order=effective_targets,
        target_source=args.target_source,
        steps=args.steps,
        batch_size=args.batch_size,
        shuffle_batches=args.shuffle_batches,
        batch_seed=args.batch_seed,
        optimizer_name=args.optimizer,
        learning_rate=args.learning_rate,
        grad_clip_norm=args.grad_clip_norm,
        seed=args.seed,
        log_every=args.log_every,
        solver_max_steps=args.solver_max_steps,
        solver_rtol=args.solver_rtol,
        solver_atol=args.solver_atol,
        solver_use_jump_ts=not args.no_jump_ts,
        log_process_losses=args.log_process_losses,
        log_decimals=args.log_decimals,
        log_header_every=args.log_header_every,
        # Sentinel: loo.py overrides per fold (uses None to mean "disabled").
        checkpoint_dir=base_checkpoint_dir,
    )
    loo_cfg = LOOConfig(
        base_train_config=base_train_config,
        output_dir=output_dir,
        selected_holdouts=selected_holdouts_raw if selected_holdouts_raw else None,
        render_plots=args.plot,
    )

    result = run_loo_cv(
        collection,
        config=loo_cfg,
        custom_py=args.custom,
        runtime_config=runtime_config,
    )

    log.info(
        "LOO complete: %d folds; aggregate=%s",
        len(result.folds),
        result.aggregate,
    )
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    handler = getattr(args, "handler")
    return int(handler(args))


if __name__ == "__main__":
    raise SystemExit(main())
