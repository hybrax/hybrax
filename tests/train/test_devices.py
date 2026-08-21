"""The pre-JAX device bootstrap must honour ``train.devices`` from a fresh
``--config`` file.

The device count is fixed at import time (before JAX initialises) by scanning
``sys.argv`` for ``--config`` (see ``_bp_load_config`` / ``_bp_resolve_devices``
in ``hybrax/train/__init__.py``), so we exercise it in fresh subprocesses: each
child sets ``sys.argv`` *before* ``import hybrax.train`` and reports
``jax.device_count()``.
"""

import gzip
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

_CPU = os.cpu_count() or 1
_needs_2 = pytest.mark.skipif(
    _CPU < 2, reason="needs >= 2 CPU cores to expose 2 devices"
)

# Sets argv before importing hybrax.train so the bootstrap reads it pre-JAX.
_SCRIPT = """
import json, sys
sys.argv = {argv!r}
import hybrax.train  # runs the pre-JAX device bootstrap (reads --config)
import jax
print("RESULT_JSON " + json.dumps({{"devices": int(jax.device_count())}}))
"""


def _device_count(
    argv: list[str], env_extra: dict | None = None, *, cwd: Path | None = None
) -> int:
    env = {**os.environ, "JAX_PLATFORMS": "cpu"}
    env.pop("XLA_FLAGS", None)  # let the bootstrap set the device count cleanly
    env.pop("BP_TRAIN_DEVICES", None)
    if env_extra:
        env.update(env_extra)
    proc = subprocess.run(
        [sys.executable, "-c", _SCRIPT.format(argv=list(argv))],
        env=env,
        capture_output=True,
        text=True,
        timeout=300,
        cwd=cwd,
    )
    assert proc.returncode == 0, (
        f"subprocess failed (argv={argv}):\n{proc.stderr[-3000:]}"
    )
    line = next(ln for ln in proc.stdout.splitlines() if ln.startswith("RESULT_JSON "))
    return json.loads(line[len("RESULT_JSON ") :])["devices"]


def _make_config(tmp_path: Path, *, devices, prepared: str = "prepared.json") -> Path:
    """A flat, fresh --config JSON: train/data at the top level (no "config"
    wrapper, unlike a run dir's config.json which the resume path unwraps)."""
    config_path = tmp_path / "config.json"
    doc = {"train": {"devices": devices, "epochs": 4}, "data": {"prepared": prepared}}
    config_path.write_text("// device config\n" + json.dumps(doc), encoding="utf-8")
    return config_path


@_needs_2
def test_config_devices_absolute_int(tmp_path: Path):
    """train.devices as a plain int is read straight from the --config JSON
    (_bp_load_config's flat, non-"config"-wrapped branch)."""
    config_path = _make_config(tmp_path, devices=2)
    assert _device_count(["bp-train", "train", "--config", str(config_path)]) == 2


@_needs_2
@pytest.mark.parametrize("artifact_form", ["plain", "gzip", "directory"])
def test_devices_max_resolves_commented_prepared_forms(
    tmp_path: Path, artifact_form: str
):
    """The pre-JAX count supports config-relative files and prepare dirs."""
    document = '// artifact\n{"process_order": ["p1", "p2"]}'
    if artifact_form == "plain":
        prepared = tmp_path / "prepared.json"
        prepared.write_text(document, encoding="utf-8")
    elif artifact_form == "gzip":
        prepared = tmp_path / "prepared.json.gz"
        with gzip.open(prepared, "wt", encoding="utf-8") as f:
            f.write(document)
    else:
        prepared = tmp_path / "prepared"
        prepared.mkdir()
        with gzip.open(prepared / "prepared.json.gz", "wt", encoding="utf-8") as f:
            f.write(document)

    config_path = _make_config(tmp_path, devices="max", prepared=prepared.name)
    argv = ["bp-train", "train", "--config", str(config_path)]
    assert _device_count(argv) == min(2, _CPU)


@_needs_2
def test_devices_config_relative_path_wins_over_cwd_collision(tmp_path: Path):
    config_dir = tmp_path / "config-dir"
    config_dir.mkdir()
    (tmp_path / "prepared.json").write_text(
        json.dumps({"process_order": ["wrong-1", "wrong-2"]}),
        encoding="utf-8",
    )
    (config_dir / "prepared.json").write_text(
        json.dumps({"process_order": ["p1"]}), encoding="utf-8"
    )
    config_path = _make_config(config_dir, devices="max", prepared="prepared.json")

    argv = ["bp-train", "train", "--config", str(config_path)]
    assert _device_count(argv, cwd=tmp_path) == 1


@pytest.mark.parametrize("alias", ["all", "auto"])
def test_devices_alias_absolute_prepared(tmp_path: Path, alias: str):
    """ "all" and "auto" are accepted aliases for "max"."""
    prepared = tmp_path / "prepared.json"
    prepared.write_text(json.dumps({"process_order": ["p1", "p2"]}), encoding="utf-8")
    config_path = _make_config(tmp_path, devices=alias, prepared=str(prepared))
    argv = ["bp-train", "train", "--config", str(config_path)]
    assert _device_count(argv) == min(2, _CPU)


def test_devices_max_relative_prepared_degrades(tmp_path: Path):
    """devices: "max" with a data.prepared that cannot be resolved (relative to
    neither cwd nor the config-file directory) degrades to cpu_count via the
    documented fallback — not a crash."""
    config_path = _make_config(tmp_path, devices="max", prepared="does_not_exist.json")
    assert _device_count(["bp-train", "train", "--config", str(config_path)]) == _CPU
