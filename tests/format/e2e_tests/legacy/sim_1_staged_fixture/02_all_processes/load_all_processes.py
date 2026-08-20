"""Generate the sim 1 all-process target-layout artifacts."""

from __future__ import annotations

import os
import sys
from pathlib import Path

os.environ.setdefault("JAX_ENABLE_X64", "true")

EXAMPLE_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = EXAMPLE_ROOT.parents[4]
SIM_DIR = EXAMPLE_ROOT / "00_simulation"

for path in (REPO_ROOT / "src", EXAMPLE_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import hybrax.format as bp  # noqa: E402
from load_utils import parse_all_processes  # noqa: E402

OUTPUT_DIR = Path(__file__).resolve().parent / "output"
SIMULATION_DENSE_CSV = SIM_DIR / "simulation_dense_output.csv"
SIMULATION_EVENTS_CSV = SIM_DIR / "events.csv"


def generate_all_processes_output(output_dir: Path) -> bp.BioProcessCollection:
    """Parse canonical simulation CSVs and write the all-process artifact."""
    output_dir.mkdir(parents=True, exist_ok=True)
    collection = parse_all_processes(
        dense_csv=SIMULATION_DENSE_CSV,
        events_csv=SIMULATION_EVENTS_CSV,
    )
    stale_duplicate_csv = output_dir / "dense_truth.csv"
    if stale_duplicate_csv.exists():
        stale_duplicate_csv.unlink()
    bp.serialization.save_process_collection(collection, output_dir / "data.json")
    return collection


def main() -> None:
    collection = generate_all_processes_output(OUTPUT_DIR)
    process_ids = ", ".join(collection.processes)
    print(f"Wrote all-process sim 1 collection: {process_ids}")


if __name__ == "__main__":
    main()
