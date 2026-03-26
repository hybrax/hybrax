from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from .prepare import prepare_artifact


def _load_config(config_path: str | None) -> dict[str, Any] | None:
    if config_path is None:
        return None
    return json.loads(Path(config_path).read_text(encoding="utf-8"))


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="bp-train")
    subparsers = parser.add_subparsers(dest="command", required=True)

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
    prepare_parser.set_defaults(handler=_handle_prepare)

    return parser


def _handle_prepare(args: argparse.Namespace) -> int:
    prepare_artifact(
        input_json=args.input,
        output_json=args.output,
        custom_py=args.custom,
        config=_load_config(args.config),
    )
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    handler = getattr(args, "handler")
    return int(handler(args))


if __name__ == "__main__":
    raise SystemExit(main())
