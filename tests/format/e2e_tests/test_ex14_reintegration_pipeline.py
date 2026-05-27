import os
import sys
from contextlib import contextmanager
from pathlib import Path

os.environ.setdefault("JAX_ENABLE_X64", "true")

FIXTURE_ROOT = Path(__file__).resolve().parent / "ex14_fixture"
SIMULATION_DIR = FIXTURE_ROOT / "00_simulation"
CANONICAL_ARTIFACTS = ("simulation_dense_output.csv", "events.csv")


def _clear_ex14_simulation_module() -> None:
    sys.modules.pop("ex14_simulation", None)


@contextmanager
def _isolated_ex14_simulation_imports():
    sys_path = list(sys.path)
    dont_write_bytecode = sys.dont_write_bytecode
    _clear_ex14_simulation_module()
    sys.dont_write_bytecode = True
    sys.path.insert(0, str(SIMULATION_DIR))
    try:
        yield
    finally:
        sys.path[:] = sys_path
        sys.dont_write_bytecode = dont_write_bytecode
        _clear_ex14_simulation_module()


def _regenerate_simulation_artifacts(output_dir: Path) -> Path:
    with _isolated_ex14_simulation_imports():
        from ex14_simulation import run_all_default

        run_all_default(output_dir=output_dir)
    return output_dir


def test_ex14_simulation_artifacts_match_tracked_canonical_files(tmp_path):
    regenerated_dir = _regenerate_simulation_artifacts(tmp_path / "simulation")

    for filename in CANONICAL_ARTIFACTS:
        tracked_path = SIMULATION_DIR / filename
        regenerated_path = regenerated_dir / filename
        assert regenerated_path.read_bytes() == tracked_path.read_bytes(), (
            f"ex14 simulation artifact drifted: {filename}\n"
            f"tracked: {tracked_path}\n"
            f"regenerated: {regenerated_path}\n"
            "Inspect both files. If the change is intentional, regenerate the "
            "tracked fixture artifact and commit it with the simulation change."
        )
