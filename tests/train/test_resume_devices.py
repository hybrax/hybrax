"""The pre-JAX device bootstrap must honour ``train.devices`` on ``--resume``.

The device count is fixed at import time (before JAX initialises) by scanning
``sys.argv`` for the config, so we exercise it in fresh subprocesses: each child
sets ``sys.argv`` *before* ``import bp_train`` and reports ``jax.device_count()``.
The bootstrap only needs a run dir holding ``config.json`` + a ``checkpoints/``
dir to resolve devices on resume — no real checkpoints required.
"""
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

_CPU = os.cpu_count() or 1
_needs_2 = pytest.mark.skipif(_CPU < 2, reason="needs >= 2 CPU cores to expose 2 devices")

# Sets argv before importing bp_train so the bootstrap reads it pre-JAX.
_SCRIPT = """
import json, sys
sys.argv = {argv!r}
import bp_train  # runs the pre-JAX device bootstrap (reads --config OR --resume)
import jax
print("RESULT_JSON " + json.dumps({{"devices": int(jax.device_count())}}))
"""


def _device_count(argv: list[str], env_extra: dict | None = None) -> int:
    env = {**os.environ, "JAX_PLATFORMS": "cpu"}
    env.pop("XLA_FLAGS", None)  # let the bootstrap set the device count cleanly
    env.pop("BP_TRAIN_DEVICES", None)
    if env_extra:
        env.update(env_extra)
    proc = subprocess.run(
        [sys.executable, "-c", _SCRIPT.format(argv=list(argv))],
        env=env, capture_output=True, text=True, timeout=300,
    )
    assert proc.returncode == 0, f"subprocess failed (argv={argv}):\n{proc.stderr[-3000:]}"
    line = next(ln for ln in proc.stdout.splitlines() if ln.startswith("RESULT_JSON "))
    return json.loads(line[len("RESULT_JSON "):])["devices"]


def _make_run_dir(tmp_path: Path, *, devices, prepared: str = "prepared.json") -> Path:
    """A minimal FAIR run dir: config.json (devices wrapped under "config") + checkpoints/."""
    run_dir = tmp_path / "run"
    (run_dir / "checkpoints").mkdir(parents=True)
    doc = {
        "status": "running",
        "config": {"train": {"devices": devices, "steps": 4}, "data": {"prepared": prepared}},
    }
    (run_dir / "config.json").write_text(json.dumps(doc), encoding="utf-8")
    return run_dir


@_needs_2
def test_resume_reads_devices_from_run_dir(tmp_path: Path):
    run_dir = _make_run_dir(tmp_path, devices=2)
    assert _device_count(["bp-train", "train", "--resume", str(run_dir)]) == 2


@_needs_2
def test_resume_subpath_resolution(tmp_path: Path):
    """--resume pointed at checkpoints/latest (with trailing slash) still resolves
    up to the run dir, so the device count is read pre-JAX just the same."""
    run_dir = _make_run_dir(tmp_path, devices=2)
    (run_dir / "checkpoints" / "latest").mkdir()
    subpath = str(run_dir / "checkpoints" / "latest") + "/"
    assert _device_count(["bp-train", "train", "--resume", subpath]) == 2


@_needs_2
def test_bp_train_devices_wins_on_resume(tmp_path: Path):
    """BP_TRAIN_DEVICES is checked before the config, so it overrides the saved
    value on resume — both suppressing (config 2, env 1) and elevating (config 1,
    env 2)."""
    run_lo = _make_run_dir(tmp_path / "a", devices=2)
    assert _device_count(
        ["bp-train", "train", "--resume", str(run_lo)], {"BP_TRAIN_DEVICES": "1"}
    ) == 1
    run_hi = _make_run_dir(tmp_path / "b", devices=1)
    assert _device_count(
        ["bp-train", "train", "--resume", str(run_hi)], {"BP_TRAIN_DEVICES": "2"}
    ) == 2


@_needs_2
def test_fresh_config_unchanged(tmp_path: Path):
    """Regression: the "config"-unwrap must not break the flat --config path
    (a fresh config file has train/data at the top level, no "config" key)."""
    flat = tmp_path / "config.json"
    flat.write_text(
        json.dumps({"train": {"devices": 2, "steps": 4}, "data": {"prepared": "x.json"}}),
        encoding="utf-8",
    )
    assert _device_count(["bp-train", "train", "--config", str(flat)]) == 2


@_needs_2
def test_resume_devices_max_absolute_prepared(tmp_path: Path):
    """devices: "max" with an absolute, resolvable data.prepared resolves to
    min(n_processes, cpu_count) on resume."""
    prepared = tmp_path / "prepared.json"
    prepared.write_text(json.dumps({"process_order": ["p1", "p2"]}), encoding="utf-8")
    run_dir = _make_run_dir(tmp_path, devices="max", prepared=str(prepared))
    assert _device_count(["bp-train", "train", "--resume", str(run_dir)]) == min(2, _CPU)


def test_resume_devices_max_relative_prepared_degrades(tmp_path: Path):
    """devices: "max" with a data.prepared relative to the original cwd (and thus
    unresolvable from the run dir) degrades to cpu_count via the documented
    fallback — not a crash."""
    run_dir = _make_run_dir(tmp_path, devices="max", prepared="does_not_exist.json")
    assert _device_count(["bp-train", "train", "--resume", str(run_dir)]) == _CPU
