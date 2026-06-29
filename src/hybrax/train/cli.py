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
    forward_plot_losses,
    resume_run,
    train_from_collection,
    train_harness_config_from_run_config,
)
from .loo import run_loo_cv, run_single_fold
from .loo_metrics import compute_loo_metrics
from .postprocessing import (
    aggregate_dense_exports,
    export_predictions_csv,
    load_model_metadata,
    plot_process_simulations,
    plot_training_results,
    save_model,
    save_model_metadata,
)
from .prepare import prepare_artifact
from .run_config import (
    ForwardRunConfig,
    LoadedRunConfig,
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
    read_json,
    read_run_config_json,
    run_config_to_jsonable,
    save_model as save_params_model,
    save_opt_state,
    update_run_config_status,
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
        "(prepared.json, prepare_config.json, prepare_diagnostics/).",
    )
    prepare_parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite an existing prepared.json in --output-dir.",
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
        "--config",
        help=(
            "forward_config.json: a `models` list of self-contained run/checkpoint "
            "dirs (len 1 = single, >1 = ensemble) + optional `data`/`output`. "
            "Mutually exclusive with --model."
        ),
    )
    forward_parser.add_argument(
        "--model",
        help=(
            "Shorthand for a 1-model config: a trained run directory, or a "
            "checkpoint dir / params.eqx inside it (resolved up to the run's "
            "config.json like `train --resume`)."
        ),
    )
    forward_parser.add_argument(
        "--input",
        help=(
            "Optional prepared.json file OR a prepare output-dir to forward on (new "
            "data + controls); defaults to each model's own bundled prepared.json."
        ),
    )
    forward_parser.add_argument(
        "--process",
        action="append",
        default=[],
        help=(
            "Process name to evaluate (repeatable or comma-separated). "
            "Defaults to every process in the data."
        ),
    )
    forward_parser.add_argument(
        "--output-dir",
        default=None,
        help="Directory for forward outputs. Defaults to <first model>/forward.",
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
            "bundled loo-config.json verbatim (no overrides) and re-runs only the "
            "folds missing a losses.csv. Mutually exclusive with --config."
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
        help="Allow re-running into a LOO output dir that already completed.",
    )
    # Internal: dispatched by the orchestrator to run exactly one fold in-process
    # (worker mode). Each worker gets its own BP_TRAIN_DEVICES + core affinity.
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
    named_losses, total_losses = forward_plot_losses(fwd_result)
    if render_plots:
        plot_training_results(
            train_result,
            collection,
            fwd_result.store,
            output_dir,
            fwd_result.dense_exports,
            # Use eval_processes (full set including holdouts in LOO) so the
            # predictions.csv has rows for every evaluated process. Per-process
            # plots are rendered for the same set.
            process_names=eval_processes,
            per_process_named_losses=named_losses,
            per_process_total_loss=total_losses,
            timeseries_csv_path=predictions_csv_path,
        )
        return fwd_result

    plot_process_simulations(
        trained_wrapper,
        collection,
        fwd_result.store,
        output_dir,
        fwd_result.dense_exports,
        process_names=eval_processes,
        training_process_names=training_process_names,
        per_process_named_losses=named_losses,
        per_process_total_loss=total_losses,
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
        # Resume replays the saved run config verbatim; only --steps may be
        # overridden. Reject other overrides loudly instead of silently dropping
        # them (the pre-JAX device count and output paths come from the saved
        # config, so a passed --output-dir/--no-plot would have no effect).
        rejected = []
        if args.output_dir is not None:
            rejected.append("--output-dir")
        if not args.plot:  # --no-plot was given (dest="plot", default True)
            rejected.append("--no-plot")
        if rejected:
            log.error(
                "--resume replays the saved run config; only --steps may be "
                "overridden. Remove %s.",
                " and ".join(rejected),
            )
            return 1
        resume_arg = Path(args.resume)
        # Be forgiving: accept the run dir itself OR a sub-path inside it
        # (e.g. checkpoints/latest, checkpoints/step_00500, model/) and resolve
        # up to the directory that actually holds config.json. Resume always
        # continues from checkpoints/latest regardless.
        # The run dir holds config.json AND a checkpoints/ subdir. Require both so
        # we skip self-contained checkpoint dirs (which also carry a config.json)
        # and land on the actual run directory.
        run_dir = next(
            (
                cand
                for cand in (resume_arg, resume_arg.parent, resume_arg.parent.parent)
                if (cand / "config.json").is_file() and (cand / "checkpoints").is_dir()
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


def _resolve_model_bundle(path: Path) -> tuple[Path, Path, RunConfig, Path | None]:
    """Resolve a model reference (run dir, checkpoint dir, or params.eqx) to
    ``(run_dir_with_config, params_path, model_config, own_prepared)``."""
    path = Path(path)
    if not path.exists():
        raise SystemExit(f"forward: model path does not exist: {path}")
    run_dir = _resolve_forward_run_dir(path)
    if run_dir is None:
        raise SystemExit(
            f"forward: no config.json at or above {path}; pass a trained run "
            "directory or a self-contained checkpoint dir."
        )
    model_cfg, _doc = read_run_config_json(run_dir / "config.json")
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

    # --- Build the forward config (full file, or --model/--input shorthand) ---
    if args.config and args.model:
        log.error("forward: pass either --config or --model, not both")
        return 1
    if args.config:
        fcfg = load_forward_config(args.config)
    elif args.model:
        raw: dict[str, Any] = {"models": [args.model]}
        if args.input:
            raw["data"] = {"prepared": args.input}
        fcfg = ForwardRunConfig.model_validate(raw)  # paths relative to cwd
    else:
        log.error("forward requires --config <forward_config.json> or --model <dir>")
        return 1

    models = fcfg.models
    shared_prepared = fcfg.data.prepared if fcfg.data is not None else None
    config_processes = fcfg.data.processes if fcfg.data is not None else None
    cli_processes = _split_multi_values(args.process)

    if len(models) > 1 and shared_prepared is None:
        log.error(
            "ensemble forward (>1 model) needs a shared `data.prepared`; add a "
            "`data` block with `prepared` so per-model predictions align."
        )
        return 1

    names = _resolve_model_names(models)

    # --- Forward each model on its data ---
    per_model: list[tuple[str, Any]] = []  # (name, ForwardResult)
    overlay_collection = None
    overlay_store = None
    eval_processes: tuple[str, ...] = ()
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
        collection = load_process_collection_json(resolve_prepared_path(Path(prepared)))
        eval_processes = (
            tuple(cli_processes)
            if cli_processes
            else (
                tuple(config_processes)
                if config_processes
                else tuple(collection.processes.keys())
            )
        )
        model_targets = model_cfg.data.targets if model_cfg.data is not None else None
        model_source = (
            model_cfg.data.target_source if model_cfg.data is not None else "auto"
        )
        training_processes = (
            tuple(model_cfg.data.processes)
            if model_cfg.data is not None and model_cfg.data.processes
            else eval_processes
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
        result = forward_from_collection(
            collection,
            model_path=params_path,
            config=fwd_cfg,
            custom_py=custom_py,
            run_config=model_cfg,
            training_process_names=training_processes,
        )
        per_model.append((name, result))
        overlay_collection = collection
        overlay_store = result.store

    # --- Output directory ---
    if args.output_dir is not None:
        output_dir = Path(args.output_dir)
    elif fcfg.output.dir is not None:
        output_dir = Path(fcfg.output.dir)
    else:
        first_run_dir, *_ = _resolve_model_bundle(models[0].path)
        output_dir = first_run_dir / "forward"
    output_dir.mkdir(parents=True, exist_ok=True)

    wrapper0 = per_model[0][1].trained_wrapper

    # --- Per-model predictions + loss tables ---
    for name, result in per_model:
        mdir = output_dir / "models" / name
        mdir.mkdir(parents=True, exist_ok=True)
        export_predictions_csv(
            result.trained_wrapper,
            result.dense_exports,
            mdir / "predictions.csv",
            process_names=eval_processes,
        )
        _table, model_rows = _format_loss_table(result)
        _write_loss_csv(model_rows, mdir / "losses.csv")

    # --- Aggregate (mean + std across models) ---
    per_model_dense = [r.dense_exports for _n, r in per_model]
    if len(per_model_dense) > 1:
        mean_exports, std_exports = aggregate_dense_exports(per_model_dense)
    else:
        mean_exports, std_exports = per_model_dense[0], None

    export_predictions_csv(
        wrapper0,
        mean_exports,
        output_dir / "predictions.csv",
        process_names=eval_processes,
    )
    if std_exports is not None:
        export_predictions_csv(
            wrapper0,
            std_exports,
            output_dir / "predictions_std.csv",
            process_names=eval_processes,
        )

    # --- Loss table (representative = first model; per-model in models/<name>/) ---
    table_str, csv_rows = _format_loss_table(per_model[0][1])
    log.info("\n%s", table_str)
    loss_csv_path = Path(args.loss_csv) if args.loss_csv else output_dir / "losses.csv"
    _write_loss_csv(csv_rows, loss_csv_path)

    # --- Optional merged timeseries CSV (mean) ---
    ts_path = args.timeseries_csv or fcfg.output.timeseries_csv
    if ts_path is not None:
        export_predictions_csv(
            wrapper0, mean_exports, ts_path, process_names=eval_processes
        )

    # --- Plots: mean line + ±std band + measured overlay ---
    if args.plot and fcfg.output.plots:
        named_losses, total_losses = forward_plot_losses(per_model[0][1])
        plot_process_simulations(
            wrapper0,
            overlay_collection,
            overlay_store,
            output_dir,
            mean_exports,
            process_names=eval_processes,
            std_exports=std_exports,
            training_process_names=eval_processes,
            per_process_named_losses=named_losses,
            per_process_total_loss=total_losses,
        )

    return 0


def _handle_loo(args: argparse.Namespace) -> int:
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    log = logging.getLogger(__name__)

    # ---- resume: reload the self-contained run dir, re-run only missing folds ----
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
        loaded = load_loo_config(bundle)
        cfg = loaded.config
        if cfg.data is None:
            raise ValueError("LOO run dir config is missing a data section")
        collection = load_process_collection_json(cfg.data.prepared)
        config_json = resume_dir / "config.json"
        update_run_config_status(config_json, status="running", resumed_at=_now_iso())
        try:
            result = run_loo_cv(
                collection,
                cfg=cfg,
                config_path=bundle,
                output_dir=resume_dir,
                custom_py=cfg.custom_py,
                resume=True,
            )
        except Exception as exc:  # noqa: BLE001 - record failure, then re-raise
            update_run_config_status(
                config_json,
                status="failed",
                error={"type": type(exc).__name__, "message": str(exc)},
                finished_at=_now_iso(),
            )
            raise
        update_run_config_status(
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
    cfg = loaded.config
    if args.output_dir is not None:
        cfg = cfg.model_copy(
            update={
                "output": cfg.output.model_copy(
                    update={"dir": Path(args.output_dir).resolve()}
                )
            }
        )
    if cfg.data is None:
        raise ValueError("loo command requires a data config section")
    output_dir = Path(cfg.output.dir)
    collection = load_process_collection_json(cfg.data.prepared)

    # ---- worker mode: run exactly one fold, no top-level artifacts ----
    if args.fold is not None:
        run_single_fold(
            collection,
            cfg=cfg,
            custom_module=loaded.custom_module,
            output_dir=output_dir,
            fold_idx=args.fold,
            custom_py=cfg.custom_py,
        )
        return 0

    # ---- orchestrator mode ----
    config_json = output_dir / "config.json"
    if config_json.is_file():
        try:
            _, prior = read_run_config_json(config_json)
        except Exception:  # noqa: BLE001 - treat unparsable as overwritable
            prior = {}
        if prior.get("status") == "complete" and not args.overwrite:
            log.error(
                "LOO output dir %s already holds a completed run; pass "
                "--overwrite to re-run (or --resume to continue)",
                output_dir,
            )
            return 1

    output_dir.mkdir(parents=True, exist_ok=True)
    # Bundle a self-contained run dir: true copies of custom.py + prepared, and a
    # loadable loo-config.json that points at the local copies. Every worker (and
    # --resume) loads ONLY from the run dir, so editing/moving the source tree
    # mid-run can't desync folds.
    bundle_path = _bundle_loo_run_dir(
        raw_config_path=args.config, cfg=cfg, output_dir=output_dir
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
    write_json(config_json, document)

    try:
        result = run_loo_cv(
            collection,
            cfg=cfg,
            config_path=bundle_path,
            output_dir=output_dir,
            custom_py=cfg.custom_py,
        )
    except Exception as exc:  # noqa: BLE001 - record failure, then re-raise
        update_run_config_status(
            config_json,
            status="failed",
            error={"type": type(exc).__name__, "message": str(exc)},
            finished_at=_now_iso(),
        )
        raise

    update_run_config_status(
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

    raw = json.loads(Path(raw_config_path).read_text(encoding="utf-8"))
    raw.setdefault("data", {})["prepared"] = prepared_name
    if custom_name is not None:
        raw["custom_py"] = custom_name
    else:
        raw.pop("custom_py", None)
    raw.setdefault("output", {})["dir"] = "."

    bundle_path = output_dir / "loo-config.json"
    bundle_path.write_text(json.dumps(raw, indent=2) + "\n", encoding="utf-8")
    return bundle_path


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    handler = getattr(args, "handler")
    return int(handler(args))


if __name__ == "__main__":
    raise SystemExit(main())
