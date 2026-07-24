"""Regression test for the opt-in pmap multi-core training path.

The device count is fixed at JAX initialisation, so we exercise the sharded
(pmap) step vs the single-device (vmap) step in subprocesses: one with
``BP_TRAIN_DEVICES=1`` and one with ``=2``. A reordered three-process batch with
distinct sample events forces one padded sharded row. The runs must train
identically up to float32 cross-device reduction order.
"""

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

_TESTS_DIR = str(Path(__file__).resolve().parent)

# Runs in a fresh interpreter so BP_TRAIN_DEVICES is honoured before JAX inits.
_SCRIPT = """
import copy, dataclasses, json, sys
sys.path.insert(0, {tests_dir!r})
import jax
import jax.numpy as jnp
from test_harness import _make_collection, _LinearReactionModule, _biomass_loss
from bp_format.dataclasses import (
    FeedMedium, FeedMediumComponent, FeedVolumeChange, StaticVariable, TimeSeries
)
from bp_train.training_data import TrainingDataStore
from bp_train.harness import train_collection, TrainHarnessConfig

def with_events(process, name, sample_value, bolus_concentration):
    process = copy.deepcopy(process)
    sample = process.volume.volume_changes["sample_1"]
    sample = dataclasses.replace(
        sample,
        values=dataclasses.replace(sample.values, values=jnp.asarray([sample_value])),
    )
    bolus = FeedVolumeChange(
        name="bolus",
        unit="L",
        is_controlled=True,
        is_continuous=False,
        values=TimeSeries(times=jnp.asarray([0.5]), values=jnp.asarray([0.1])),
        feed_medium=FeedMedium(
            name="feed",
            density=1.0,
            density_unit="kg/L",
            components={{
                "biomass": FeedMediumComponent(
                    name="biomass",
                    unit="g/L",
                    concentration=StaticVariable(bolus_concentration),
                    is_controlled=False,
                )
            }},
        ),
    )
    volume = dataclasses.replace(
        process.volume,
        volume_changes={{
            **process.volume.volume_changes,
            "sample_1": sample,
            "bolus": bolus,
        }},
    )
    metadata = dataclasses.replace(process.metadata, name=name)
    return dataclasses.replace(process, metadata=metadata, volume=volume)

collection = _make_collection()
collection.processes["p1"] = with_events(collection.processes["p1"], "p1", -0.1, 2.0)
collection.processes["p2"] = with_events(collection.processes["p2"], "p2", -0.2, 4.0)
collection.processes["p3"] = with_events(collection.processes["p1"], "p3", -0.3, 6.0)
store = TrainingDataStore.from_collection(
    collection, target_variable_order=["biomass"], target_source="reactor_components"
)
result = train_collection(
    store,
    reaction_module=_LinearReactionModule(),
    loss_module=_biomass_loss(),
    config=TrainHarnessConfig(
        process_names=("p3", "p1", "p2"), epochs=4, batch_size=3,
        optimizer_name="adam", learning_rate=2e-2,
        shuffle_batches=False,
    ),
)
print("RESULT_JSON " + json.dumps(
    {{
        "devices": int(jax.device_count()),
        "losses": [float(x) for x in result.mean_loss_by_step],
    }}
))
"""

_TELEMETRY_SCRIPT = """
import json, sys, tempfile
from pathlib import Path
sys.path.insert(0, {tests_dir!r})
import jax
import jax.numpy as jnp
from test_harness import _make_collection, _LinearReactionModule, _biomass_loss
from bp_train.harness import train_collection, TrainHarnessConfig
from bp_train.model_api import ReactionOutputs
from bp_train.training_data import TrainingDataStore

class P2OnlyBlowUp(_LinearReactionModule):
    def __call__(self, t, inputs):
        state = inputs.SCL_modeled_RMCs[0]
        is_p2 = inputs.SCL_modeled_V > 0.95
        rate = jnp.where((t > 1.0) & is_p2, 1.0e4 * state, 0.0)
        return ReactionOutputs(
            SCL_modeled_BiologicalOde_rates=jnp.asarray([rate], dtype=state.dtype),
            SCL_modeled_FVCs_rates=jnp.zeros((0,), dtype=state.dtype),
        )

collection = _make_collection()
store = TrainingDataStore.from_collection(
    collection, target_variable_order=["biomass"], target_source="reactor_components"
)
with tempfile.TemporaryDirectory() as tmp:
    metrics = Path(tmp) / "metrics.jsonl"
    train_collection(
        store,
        reaction_module=P2OnlyBlowUp(),
        loss_module=_biomass_loss(),
        config=TrainHarnessConfig(
            process_names=("p2", "p1"), epochs=1, batch_size=2,
            shuffle_batches=False, solver_max_steps=512, metrics_jsonl=metrics,
        ),
    )
    row = json.loads(metrics.read_text())
print("RESULT_JSON " + json.dumps({{"devices": jax.device_count(), "row": row}}))
"""


def _run(devices: str, *, gspmd: bool = False) -> dict:
    env = {**os.environ, "JAX_PLATFORMS": "cpu", "BP_TRAIN_DEVICES": devices}
    env.pop("XLA_FLAGS", None)  # let the bootstrap set the device count cleanly
    if gspmd:
        env["BP_GSPMD"] = "1"
    else:
        env.pop("BP_GSPMD", None)
    proc = subprocess.run(
        [sys.executable, "-c", _SCRIPT.format(tests_dir=_TESTS_DIR)],
        env=env,
        capture_output=True,
        text=True,
        timeout=300,
    )
    assert proc.returncode == 0, (
        f"subprocess (devices={devices}) failed:\n{proc.stderr[-3000:]}"
    )
    line = next(ln for ln in proc.stdout.splitlines() if ln.startswith("RESULT_JSON "))
    return json.loads(line[len("RESULT_JSON ") :])


def _run_telemetry(mode: str) -> dict:
    assert mode in ("vmap", "pmap", "gspmd")
    devices = "1" if mode == "vmap" else "2"
    env = {**os.environ, "JAX_PLATFORMS": "cpu", "BP_TRAIN_DEVICES": devices}
    env.pop("XLA_FLAGS", None)
    if mode == "gspmd":
        env["BP_GSPMD"] = "1"
    else:
        env.pop("BP_GSPMD", None)
    proc = subprocess.run(
        [sys.executable, "-c", _TELEMETRY_SCRIPT.format(tests_dir=_TESTS_DIR)],
        env=env,
        capture_output=True,
        text=True,
        timeout=300,
    )
    assert proc.returncode == 0, proc.stderr[-3000:]
    line = next(ln for ln in proc.stdout.splitlines() if ln.startswith("RESULT_JSON "))
    return json.loads(line[len("RESULT_JSON ") :])


def test_sharded_training_matches_vmap():
    vmap = _run("1")
    assert vmap["devices"] == 1

    for gspmd in (False, True):
        sharded = _run("2", gspmd=gspmd)
        assert sharded["devices"] == 2
        assert len(vmap["losses"]) == len(sharded["losses"]) == 4
        assert sharded["losses"][-1] < sharded["losses"][0]
        for a, b in zip(vmap["losses"], sharded["losses"]):
            assert abs(a - b) <= 1e-4 + 1e-4 * abs(a), (
                vmap["losses"],
                sharded["losses"],
            )


@pytest.mark.parametrize("mode", ["vmap", "pmap", "gspmd"])
def test_failed_sample_telemetry_end_to_end(mode):
    result = _run_telemetry(mode)
    row = result["row"]

    assert result["devices"] == (1 if mode == "vmap" else 2)
    assert row["process_names"] == ["p2", "p1"]
    assert row["n_failed_samples"] == 1
    assert row["failed_processes"] == ["p2"]


def test_devices_exceeding_processes_shards_over_processes():
    """``--devices`` larger than the process count must shard across
    ``min(devices, batch)`` — not collapse to a single device. Here 4 devices on
    a 3-process batch shard over 3, and must match the vmap run."""
    vmap = _run("1")
    over = _run("4")
    assert over["devices"] == 4, "bootstrap should expose 4 CPU devices"
    assert len(over["losses"]) == len(vmap["losses"]) == 4
    assert over["losses"][-1] < over["losses"][0]
    for a, b in zip(vmap["losses"], over["losses"]):
        assert abs(a - b) <= 1e-4 + 1e-4 * abs(a), (vmap["losses"], over["losses"])


def test_device_count_capped_at_cpu_count():
    """The package bootstrap caps requested devices at ``cpu_count``:
    over-subscribed XLA CPU collectives can starve the AllReduce rendezvous and
    deadlock mid-training (and extra devices never speed up a core-bound run)."""
    env = {
        **os.environ,
        "JAX_PLATFORMS": "cpu",
        "BP_TRAIN_DEVICES": "9999",
    }
    env.pop("XLA_FLAGS", None)
    proc = subprocess.run(
        [
            sys.executable,
            "-c",
            "import bp_train, jax, os; "
            "print('CAP', jax.device_count(), os.cpu_count())",
        ],
        env=env,
        capture_output=True,
        text=True,
        timeout=180,
    )
    assert proc.returncode == 0, proc.stderr[-2000:]
    line = next(ln for ln in proc.stdout.splitlines() if ln.startswith("CAP"))
    _, devcount, cpu = line.split()
    assert int(devcount) == int(cpu), (
        f"device count {devcount} not capped to cpu_count {cpu}"
    )
    assert int(devcount) < 9999
