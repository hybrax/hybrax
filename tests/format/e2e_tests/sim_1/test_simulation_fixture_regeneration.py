"""Determinism check: a fresh simulation run reproduces the checked-in fixture.

Extracted from the (now-deleted) direct-cstar reintegration test, which was
the only place exercising this: regenerate `simulation.py`/`load_utils.py`'s
output from scratch and diff it against the canonical `sim_results/` fixture
that every other sim_1 test loads directly. Nothing else in this suite
re-derives the fixture and compares it, so this guard is load-bearing on its
own, independent of anything pseudobatch/c*-related.
"""

import csv
import json
import os

os.environ.setdefault("JAX_ENABLE_X64", "true")

import numpy as np  # noqa: E402
from hybrax.format.serialization import save_process_collection  # noqa: E402

from .load_utils import parse_all_processes  # noqa: E402
from .real_space_segments import SIM_RESULTS_DIR  # noqa: E402
from .simulation import run_all_default  # noqa: E402
from .simulation import write_simulation_plots  # noqa: E402

DATA_JSON = SIM_RESULTS_DIR / "process_collection.json"
EVENTS_OUTPUT = SIM_RESULTS_DIR / "simulation_events.csv"
SIMULATION_DENSE_OUTPUT = SIM_RESULTS_DIR / "simulation_dense_output.csv"
# Canonical CSV artifacts compared numerically (with tolerance) against a fresh
# simulation run. Plot PNGs are intentionally NOT compared: their bytes are not
# reproducible across matplotlib/freetype versions, and `write_simulation_plots`
# already exercises the plotting path.
CANONICAL_ARTIFACTS = {
    "simulation_dense_output.csv": SIMULATION_DENSE_OUTPUT,
    "simulation_events.csv": EVENTS_OUTPUT,
}

# Numeric tolerance for comparing a fresh run against the canonical artifacts.
# The integrator/serializer write full float precision, whose last digits drift
# across JAX/numpy/scipy/platform builds (observed ~1e-11 abs / ~1e-12 rel), so
# byte-exact comparison is not portable. The tolerance leaves several orders of
# margin yet still catches any real numerical or structural regression.
_ARTIFACT_RTOL = 1e-6
_ARTIFACT_ATOL = 1e-9


def _assert_csv_close(new_path, canonical_path):
    with open(new_path, newline="") as handle:
        new_rows = list(csv.reader(handle))
    with open(canonical_path, newline="") as handle:
        old_rows = list(csv.reader(handle))
    assert new_rows[0] == old_rows[0], f"header mismatch: {canonical_path}"
    assert len(new_rows) == len(old_rows), f"row count mismatch: {canonical_path}"
    for line, (new_row, old_row) in enumerate(zip(new_rows[1:], old_rows[1:]), start=2):
        assert len(new_row) == len(old_row), f"{canonical_path}:{line} column count"
        for new_cell, old_cell in zip(new_row, old_row):
            try:
                new_val, old_val = float(new_cell), float(old_cell)
            except ValueError:  # labels: process_id, row_type, ...
                assert new_cell == old_cell, (
                    f"{canonical_path}:{line} {new_cell!r} != {old_cell!r}"
                )
            else:
                assert np.isclose(
                    new_val, old_val, rtol=_ARTIFACT_RTOL, atol=_ARTIFACT_ATOL
                ), f"{canonical_path}:{line} {new_val!r} !~ {old_val!r}"


def _assert_json_close(new_path, canonical_path):
    def walk(new, old, where):
        if isinstance(new, dict):
            assert isinstance(old, dict) and new.keys() == old.keys(), (
                f"keys differ at {where or '/'}"
            )
            for key in new:
                walk(new[key], old[key], f"{where}/{key}")
        elif isinstance(new, list):
            assert isinstance(old, list) and len(new) == len(old), (
                f"length differs at {where}"
            )
            for index, (new_item, old_item) in enumerate(zip(new, old)):
                walk(new_item, old_item, f"{where}[{index}]")
        elif isinstance(new, bool) or isinstance(old, bool):
            assert new == old, f"{where}: {new!r} != {old!r}"
        elif isinstance(new, (int, float)) and isinstance(old, (int, float)):
            assert np.isclose(new, old, rtol=_ARTIFACT_RTOL, atol=_ARTIFACT_ATOL), (
                f"{where}: {new!r} !~ {old!r}"
            )
        else:
            assert new == old, f"{where}: {new!r} != {old!r}"

    with open(new_path) as handle:
        new = json.load(handle)
    with open(canonical_path) as handle:
        old = json.load(handle)
    walk(new, old, "")


def _assert_simulation_artifacts_match(new_root):
    for artifact, canonical_path in CANONICAL_ARTIFACTS.items():
        new_path = new_root / artifact
        assert new_path.exists(), f"missing regenerated artifact: {new_path}"
        assert canonical_path.exists(), f"missing canonical artifact: {canonical_path}"
        _assert_csv_close(new_path, canonical_path)


def test_sim_1_simulation_fixture_matches_checked_in_artifacts(tmp_path):
    simulation_dir = tmp_path / "simulation"
    results = run_all_default(output_dir=simulation_dir)
    write_simulation_plots(simulation_dir / "sim_plots", results)
    _assert_simulation_artifacts_match(simulation_dir)

    # Raw parsed collection must match the canonical parser artifact too.
    parsed_json = tmp_path / "process_collection.json"
    parsed_collection = parse_all_processes(
        dense_csv=simulation_dir / "simulation_dense_output.csv",
        events_csv=simulation_dir / "simulation_events.csv",
    )
    save_process_collection(parsed_collection, parsed_json)
    _assert_json_close(parsed_json, DATA_JSON)
