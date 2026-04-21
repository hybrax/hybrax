from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any

from bpbench.serialization import load_process_collection_json

from .harness import (
    ForwardConfig,
    ForwardResult,
    TrainHarnessConfig,
    forward_from_collection,
    train_from_collection,
)
from .postprocessing import (
    load_model_metadata,
    plot_process_simulations,
    plot_training_results,
    save_model,
    save_model_metadata,
)
from .prepare import prepare_artifact
from .training_data import TARGET_SOURCES, TrainingDataStore
from .utils import load_custom_module, resolve_config


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
        help="Transform a raw bpbench process collection into a prepared artifact.",
    )
    prepare_parser.add_argument("--input", required=True, help="Path to input JSON.")
    prepare_parser.add_argument("--output", required=True, help="Path to output JSON.")
    prepare_parser.add_argument(
        "--custom",
        help="Path to the case-study Python module with prep hooks.",
    )
    prepare_parser.add_argument(
        "--config",
        help="Optional JSON file with additional prepare config.",
    )
    prepare_parser.add_argument(
        "--case-study",
        help=(
            "Case study name to extract from a BenchmarkDataset. "
            "Defaults to the first case study."
        ),
    )
    prepare_parser.set_defaults(handler=_handle_prepare)

    # ---- train ----
    train_parser = subparsers.add_parser(
        "train",
        help="Run minimal one/multi-process training from a prepared artifact.",
    )
    train_parser.add_argument(
        "--input",
        required=True,
        help="Path to prepared JSON.",
    )
    train_parser.add_argument(
        "--custom",
        help="Optional custom.py path exposing build_reaction_module hooks.",
    )
    train_parser.add_argument(
        "--config",
        help="Optional JSON runtime config.",
    )
    train_parser.add_argument(
        "--process",
        action="append",
        default=[],
        help=(
            "Process name to include. Can be passed multiple times or as a "
            "comma-separated list."
        ),
    )
    train_parser.add_argument(
        "--target",
        action="append",
        default=[],
        help=(
            "Target variable name to train against. Can be passed multiple times "
            "or as a comma-separated list."
        ),
    )
    train_parser.add_argument(
        "--target-source",
        default=train_cfg_defaults.target_source,
        choices=sorted(TARGET_SOURCES),
        help=(
            "Source family for training targets: process_variables, "
            "reactor_components, or auto."
        ),
    )
    train_parser.add_argument(
        "--steps",
        type=int,
        default=train_cfg_defaults.steps,
        help="Number of training steps.",
    )
    train_parser.add_argument(
        "--batch-size",
        type=int,
        help="Batch size. Defaults to the number of selected processes.",
    )
    train_parser.add_argument(
        "--batch-seed",
        type=int,
        help="Seed used for batch index generation.",
    )
    train_parser.add_argument(
        "--optimizer",
        default=train_cfg_defaults.optimizer_name,
        choices=["adam", "sgd"],
        help="Optimizer to use for batched updates.",
    )
    shuffle_group = train_parser.add_mutually_exclusive_group()
    shuffle_group.add_argument(
        "--shuffle-batches",
        dest="shuffle_batches",
        action="store_true",
        help="Shuffle selected processes when building batches.",
    )
    shuffle_group.add_argument(
        "--no-shuffle-batches",
        dest="shuffle_batches",
        action="store_false",
        help="Keep batch construction deterministic and round-robin.",
    )
    train_parser.set_defaults(shuffle_batches=train_cfg_defaults.shuffle_batches)
    train_parser.add_argument(
        "--learning-rate",
        type=float,
        default=train_cfg_defaults.learning_rate,
        help="Learning rate (overridden by build_learning_rate hook in custom.py).",
    )
    train_parser.add_argument(
        "--grad-clip-norm",
        type=float,
        default=train_cfg_defaults.grad_clip_norm,
        help="Global gradient-norm clipping threshold; 0 disables clipping.",
    )
    train_parser.add_argument(
        "--seed",
        type=int,
        default=train_cfg_defaults.seed,
        help="Random seed for default model initialization.",
    )
    train_parser.add_argument(
        "--log-every",
        type=int,
        default=train_cfg_defaults.log_every,
        help="Emit progress log every N steps.",
    )
    train_parser.add_argument(
        "--solver-max-steps",
        type=int,
        default=train_cfg_defaults.solver_max_steps,
        help="Maximum diffrax solver steps per simulation call.",
    )
    train_parser.add_argument(
        "--solver-rtol",
        type=float,
        default=train_cfg_defaults.solver_rtol,
        help="Diffrax relative tolerance.",
    )
    train_parser.add_argument(
        "--solver-atol",
        type=float,
        default=train_cfg_defaults.solver_atol,
        help="Diffrax absolute tolerance.",
    )
    train_parser.add_argument(
        "--no-jump-ts",
        action="store_true",
        help="Disable passing control step boundaries as jump_ts to the solver.",
    )
    train_parser.add_argument(
        "--output-dir",
        default="output",
        help="Directory for trained model and plots (default: ./output).",
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
    train_parser.add_argument(
        "--log-process-losses",
        action="store_true",
        default=train_cfg_defaults.log_process_losses,
        help="Emit per-process losses on every step (otherwise only at log steps).",
    )
    train_parser.add_argument(
        "--metrics-csv",
        default=train_cfg_defaults.metrics_csv,
        help="If set, write per-step metrics to this CSV file.",
    )
    train_parser.add_argument(
        "--metrics-jsonl",
        default=train_cfg_defaults.metrics_jsonl,
        help="If set, write per-step metrics to this JSONL file.",
    )
    train_parser.add_argument(
        "--log-decimals",
        type=int,
        default=train_cfg_defaults.log_decimals,
        help="Decimal places for numeric columns in the per-step console table.",
    )
    train_parser.add_argument(
        "--log-header-every",
        type=int,
        default=train_cfg_defaults.log_header_every,
        help="Re-emit the table header every N rows (0 disables re-emission).",
    )
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
        help="Optional JSON runtime config (same as `train --config`).",
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
            "custom.CONFIG or the sidecar."
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

    return parser


def _handle_prepare(args: argparse.Namespace) -> int:
    prepare_artifact(
        input_json=args.input,
        output_json=args.output,
        custom_py=args.custom,
        config=_load_config(args.config),
        case_study=getattr(args, "case_study", None),
    )
    return 0


def _split_multi_values(raw_values: list[str]) -> tuple[str, ...]:
    values: list[str] = []
    for value in raw_values:
        values.extend([part.strip() for part in value.split(",") if part.strip() != ""])
    return tuple(values)


def _handle_train(args: argparse.Namespace) -> int:
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    collection = load_process_collection_json(Path(args.input))
    runtime_config = _load_config(args.config)
    custom_module = load_custom_module(args.custom)
    selected_processes = _split_multi_values(args.process)
    selected_targets = _split_multi_values(args.target)
    user_config = resolve_config(custom_module, None)
    config_targets = user_config.get("target_variable_order")
    if selected_targets:
        effective_targets = selected_targets
    elif config_targets:
        effective_targets = tuple(config_targets)
    else:
        effective_targets = None
    config = TrainHarnessConfig(
        process_names=selected_processes if selected_processes else None,
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
        metrics_csv=args.metrics_csv,
        metrics_jsonl=args.metrics_jsonl,
        log_decimals=args.log_decimals,
        log_header_every=args.log_header_every,
    )
    result = train_from_collection(
        collection,
        config=config,
        custom_py=args.custom,
        runtime_config=runtime_config,
    )
    first = result.mean_loss_by_step[0]
    last = result.mean_loss_by_step[-1]
    delta = last - first
    log = logging.getLogger(__name__)
    log.info(
        "training complete: first_mean_loss=%.6g last_mean_loss=%.6g delta=%.6g",
        first,
        last,
        delta,
    )

    # Post-training outputs

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    model_path = output_dir / "trained_wrapper.eqx"
    save_model(result.trained_wrapper, model_path)

    # Sidecar metadata so `bp-train forward` can auto-resolve training context.
    # training_processes=None means "every process in prepared.json was used"
    # — `bp-train forward` treats that as a wildcard when classifying splits.
    training_processes_list = (
        list(config.process_names) if config.process_names else None
    )
    meta = {
        "prepared_input": str(Path(args.input).resolve()),
        "custom_py": str(Path(args.custom).resolve()) if args.custom else None,
        "training_processes": training_processes_list,
        "targets": list(effective_targets) if effective_targets is not None else None,
        "target_source": config.target_source,
        "solver": {
            "max_steps": int(config.solver_max_steps),
            "rtol": float(config.solver_rtol),
            "atol": float(config.solver_atol),
            "use_jump_ts": bool(config.solver_use_jump_ts),
        },
        "training": {
            "steps": int(config.steps),
            "batch_size": config.batch_size,
            "seed": int(config.seed),
            "final_mean_loss": float(last),
        },
    }
    save_model_metadata(output_dir / "trained_wrapper.meta.json", meta)

    if args.plot:
        store = TrainingDataStore.from_collection(
            collection,
            target_variable_order=config.target_variable_order,
            target_source=config.target_source,
        )
        plot_training_results(
            result,
            collection,
            store,
            output_dir,
            process_names=config.process_names,
            solver_max_steps=config.solver_max_steps,
            solver_rtol=config.solver_rtol,
            solver_atol=config.solver_atol,
        )

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
    for name in result.process_names:
        total = result.per_process_total_loss[name]
        per_target = result.per_process_per_target_loss[name]
        split = "train" if name in training_set else "holdout"
        data_rows.append(
            [name, f"{total:.6g}"]
            + [f"{v:.6g}" for v in per_target]
            + [split]
        )
        csv_rows.append(
            [name, f"{total:.6g}"]
            + [f"{v:.6g}" for v in per_target]
            + [split]
        )
        total_sum += total
        for i, v in enumerate(per_target):
            per_target_sum[i] += v

    n = max(len(result.process_names), 1)
    mean_row = (
        ["total (mean)", f"{total_sum / n:.6g}"]
        + [f"{v / n:.6g}" for v in per_target_sum]
        + [""]
    )
    data_rows.append(mean_row)
    csv_rows.append(mean_row)

    col_widths = [len(h) for h in headers]
    for row in data_rows:
        for i, cell in enumerate(row):
            col_widths[i] = max(col_widths[i], len(cell))

    def _fmt_row(row: list[str]) -> str:
        return " | ".join(cell.ljust(col_widths[i]) for i, cell in enumerate(row))

    sep = "-+-".join("-" * w for w in col_widths)
    lines = ["LOSSES (forward evaluation)", _fmt_row(headers), sep]
    for row in data_rows[:-1]:
        lines.append(_fmt_row(row))
    lines.append(sep)
    lines.append(_fmt_row(data_rows[-1]))
    return "\n".join(lines), csv_rows


def _write_loss_csv(rows: list[list[str]], path: Path) -> None:
    import csv as _csv

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = _csv.writer(fh)
        for row in rows:
            writer.writerow(row)


def _handle_forward(args: argparse.Namespace) -> int:
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    log = logging.getLogger(__name__)

    model_path = Path(args.model)
    if not model_path.exists():
        raise SystemExit(f"--model path does not exist: {model_path}")

    meta = load_model_metadata(model_path.with_suffix(".meta.json"))
    if not meta:
        log.warning(
            "no sidecar found at %s; --input and solver flags fall back to CLI/defaults",
            model_path.with_suffix(".meta.json"),
        )

    prepared = args.input or meta.get("prepared_input")
    if prepared is None:
        raise SystemExit(
            "no --input provided and no sidecar to read `prepared_input` from"
        )
    custom_py = args.custom or meta.get("custom_py")

    meta_solver = meta.get("solver", {})
    solver_max_steps = (
        args.solver_max_steps
        if args.solver_max_steps is not None
        else int(meta_solver.get("max_steps", 4096))
    )
    solver_rtol = (
        args.solver_rtol
        if args.solver_rtol is not None
        else float(meta_solver.get("rtol", 1e-5))
    )
    solver_atol = (
        args.solver_atol
        if args.solver_atol is not None
        else float(meta_solver.get("atol", 1e-7))
    )
    if args.no_jump_ts:
        solver_use_jump_ts = False
    else:
        solver_use_jump_ts = bool(meta_solver.get("use_jump_ts", True))

    runtime_config = _load_config(args.config)
    collection = load_process_collection_json(Path(prepared))

    cli_processes = _split_multi_values(args.process)
    if cli_processes:
        eval_processes = cli_processes
    else:
        eval_processes = tuple(collection.processes.keys())

    cli_targets = _split_multi_values(args.target)
    effective_targets: tuple[str, ...] | None = None
    if cli_targets:
        effective_targets = cli_targets
    elif meta.get("targets"):
        effective_targets = tuple(meta["targets"])

    target_source = args.target_source or meta.get("target_source") or "auto"

    # training_processes in the sidecar:
    #   * list  → the explicit subset trained on
    #   * None  → default (trained on every process in the input file)
    #   * missing → unknown (pre-sidecar model); treat everything as holdout
    if "training_processes" in meta:
        tp_value = meta["training_processes"]
        if tp_value is None:
            training_processes = tuple(collection.processes.keys())
        else:
            training_processes = tuple(tp_value)
    else:
        training_processes = ()

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
    )

    table_str, csv_rows = _format_loss_table(result)
    log.info("\n%s", table_str)

    if args.output_dir is not None:
        output_dir = Path(args.output_dir)
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
            training_process_names=training_processes,
            timeseries_csv_path=args.timeseries_csv,
        )

    return 0


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    handler = getattr(args, "handler")
    return int(handler(args))


if __name__ == "__main__":
    raise SystemExit(main())
