"""Run the example 14 intracellular-product simulation."""

from __future__ import annotations

import argparse
from pathlib import Path

from ex14_simulation import run_all_default, write_simulation_plots


def _arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parent,
        help="Directory for simulation_dense_output.csv and events.csv.",
    )
    return parser


def main() -> None:
    args = _arg_parser().parse_args()
    results = run_all_default(output_dir=args.output_dir)
    plot_paths = write_simulation_plots(
        args.output_dir / "output" / "simulation_plots",
        results,
    )
    dense_count = sum(len(result.dense_rows) for result in results)
    event_count = sum(len(result.event_rows) for result in results)
    print(f"Wrote {dense_count} dense rows to {args.output_dir}")
    print(f"Wrote {event_count} event rows to {args.output_dir}")
    for path in plot_paths:
        print(f"Wrote plot {path}")


if __name__ == "__main__":
    main()
