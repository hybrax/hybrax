from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any

from .harness import TrainHarnessConfig, train_from_collection, train_from_prepared_json
from .prepare import prepare_artifact
from .training_data import TARGET_SOURCES


def _load_config(config_path: str | None) -> dict[str, Any] | None:
    if config_path is None:
        return None
    return json.loads(Path(config_path).read_text(encoding="utf-8"))


def _build_parser() -> argparse.ArgumentParser:
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
        default="auto",
        choices=sorted(TARGET_SOURCES),
        help=(
            "Source family for training targets: process_variables, "
            "reactor_components, or auto."
        ),
    )
    train_parser.add_argument(
        "--steps",
        type=int,
        default=50,
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
        default="adam",
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
    train_parser.set_defaults(shuffle_batches=True)
    train_parser.add_argument(
        "--learning-rate",
        type=float,
        default=1e-3,
        help="Learning rate (overridden by build_learning_rate hook in custom.py).",
    )
    train_parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed for default model initialization.",
    )
    train_parser.add_argument(
        "--log-every",
        type=int,
        default=10,
        help="Emit progress log every N steps.",
    )
    train_parser.add_argument(
        "--solver-max-steps",
        type=int,
        default=2048,
        help="Maximum diffrax solver steps per simulation call.",
    )
    train_parser.add_argument(
        "--solver-rtol",
        type=float,
        default=1e-5,
        help="Diffrax relative tolerance.",
    )
    train_parser.add_argument(
        "--solver-atol",
        type=float,
        default=1e-7,
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
        help="Emit per-process losses on every step (otherwise only at log steps).",
    )
    train_parser.add_argument(
        "--metrics-csv",
        default=None,
        help="If set, write per-step metrics to this CSV file.",
    )
    train_parser.add_argument(
        "--metrics-jsonl",
        default=None,
        help="If set, write per-step metrics to this JSONL file.",
    )
    train_parser.add_argument(
        "--log-decimals",
        type=int,
        default=4,
        help="Decimal places for numeric columns in the per-step console table.",
    )
    train_parser.add_argument(
        "--log-header-every",
        type=int,
        default=30,
        help="Re-emit the table header every N rows (0 disables re-emission).",
    )
    train_parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Python logging level.",
    )
    train_parser.set_defaults(handler=_handle_train)

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
        level=getattr(logging, str(args.log_level)),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    from bpbench.serialization import load_process_collection_json

    from .training_data import TrainingDataStore

    collection = load_process_collection_json(Path(args.input))
    selected_processes = _split_multi_values(args.process)
    selected_targets = _split_multi_values(args.target)
    config = TrainHarnessConfig(
        process_names=selected_processes if selected_processes else None,
        target_variable_order=selected_targets if selected_targets else None,
        target_source=str(args.target_source),
        steps=int(args.steps),
        batch_size=None if args.batch_size is None else int(args.batch_size),
        shuffle_batches=bool(args.shuffle_batches),
        batch_seed=None if args.batch_seed is None else int(args.batch_seed),
        optimizer_name=str(args.optimizer),
        learning_rate=float(args.learning_rate),
        seed=int(args.seed),
        log_every=int(args.log_every),
        solver_max_steps=int(args.solver_max_steps),
        solver_rtol=float(args.solver_rtol),
        solver_atol=float(args.solver_atol),
        solver_use_jump_ts=not bool(args.no_jump_ts),
        log_process_losses=bool(args.log_process_losses),
        metrics_csv=args.metrics_csv,
        metrics_jsonl=args.metrics_jsonl,
        log_decimals=int(args.log_decimals),
        log_header_every=int(args.log_header_every),
    )
    result = train_from_collection(
        collection,
        config=config,
        custom_py=args.custom,
        runtime_config=_load_config(args.config),
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
    from .postprocessing import plot_training_results, save_model

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    save_model(result.trained_wrapper, output_dir / "trained_wrapper.eqx")

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
            solver_max_steps=config.solver_max_steps,
            solver_rtol=config.solver_rtol,
            solver_atol=config.solver_atol,
        )

    return 0


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    handler = getattr(args, "handler")
    return int(handler(args))


if __name__ == "__main__":
    raise SystemExit(main())
