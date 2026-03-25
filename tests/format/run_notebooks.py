#!/usr/bin/env python3
"""Execute project notebooks with a predictable kernel environment."""

from __future__ import annotations

import argparse
import os
import sys
import traceback
from pathlib import Path

import nbformat
from nbclient import NotebookClient


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Execute notebooks with nbclient while forcing this repo's .venv "
            "and repo root onto the kernel environment."
        )
    )
    parser.add_argument(
        "paths",
        nargs="*",
        help="Notebook files or directories to run. Defaults to examples/**/*.ipynb.",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=900,
        help="Per-cell timeout in seconds. Default: 900.",
    )
    parser.add_argument(
        "--write",
        action="store_true",
        help="Write executed notebook outputs back to disk on success.",
    )
    parser.add_argument(
        "--kernel-name",
        default="python3",
        help="Kernel name to use. Default: python3.",
    )
    return parser.parse_args()


def repo_root() -> Path:
    return Path(__file__).resolve().parent.parent


def iter_notebooks(inputs: list[str], root: Path) -> list[Path]:
    if not inputs:
        return sorted((root / "examples").rglob("*.ipynb"))

    notebooks: list[Path] = []
    for raw in inputs:
        path = Path(raw)
        if not path.is_absolute():
            path = (Path.cwd() / path).resolve()

        if path.is_dir():
            notebooks.extend(sorted(path.rglob("*.ipynb")))
        elif path.suffix == ".ipynb":
            notebooks.append(path)
        else:
            raise FileNotFoundError(f"Not a notebook or directory: {raw}")

    deduped: list[Path] = []
    seen: set[Path] = set()
    for notebook in notebooks:
        if notebook not in seen:
            deduped.append(notebook)
            seen.add(notebook)
    return deduped


def configure_kernel_environment(root: Path) -> None:
    venv_bin = root / ".venv" / "bin"
    path_parts = [str(venv_bin)]
    if old_path := os.environ.get("PATH"):
        path_parts.append(old_path)
    os.environ["PATH"] = os.pathsep.join(path_parts)

    pythonpath_parts = [str(root)]
    if old_pythonpath := os.environ.get("PYTHONPATH"):
        pythonpath_parts.append(old_pythonpath)
    os.environ["PYTHONPATH"] = os.pathsep.join(pythonpath_parts)


def execute_notebook(
    notebook_path: Path,
    kernel_name: str,
    timeout: int,
    write: bool,
) -> tuple[str, str]:
    notebook = nbformat.read(notebook_path, as_version=4)
    client = NotebookClient(
        notebook,
        kernel_name=kernel_name,
        timeout=timeout,
        allow_errors=False,
        resources={"metadata": {"path": str(notebook_path.parent)}},
    )
    client.execute()
    if write:
        nbformat.write(notebook, notebook_path)
    return ("PASS", "")


def main() -> int:
    args = parse_args()
    root = repo_root()
    configure_kernel_environment(root)
    notebooks = iter_notebooks(args.paths, root)

    if not notebooks:
        print("No notebooks found.", file=sys.stderr)
        return 1

    failures: list[tuple[Path, str]] = []
    print(f"Running {len(notebooks)} notebooks")
    for notebook in notebooks:
        rel = notebook.relative_to(root)
        print(f"RUN  {rel}", flush=True)
        try:
            execute_notebook(notebook, args.kernel_name, args.timeout, args.write)
            print(f"PASS {rel}", flush=True)
        except Exception as exc:  # noqa: BLE001
            summary = f"{type(exc).__name__}: {exc}"
            failures.append((notebook, summary))
            print(f"FAIL {rel}: {summary}", flush=True)
            print(traceback.format_exc(limit=5), flush=True)

    print("\nSummary")
    for notebook, summary in failures:
        print(f"FAIL {notebook.relative_to(root)} :: {summary}")

    passed = len(notebooks) - len(failures)
    print(f"Counts: pass={passed} fail={len(failures)} total={len(notebooks)}")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
