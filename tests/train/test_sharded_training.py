"""Regression test for the opt-in pmap multi-core training path.

The device count is fixed at JAX initialisation, so we exercise the sharded
(pmap) step vs the single-device (vmap) step in subprocesses: one with
``BP_TRAIN_DEVICES=1`` (vmap) and one with ``=2`` (sharded, one process per
device). The two must train identically up to float32 cross-device reduction
order. Reuses the ``p1``/``p2`` fixture from ``test_harness``.
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
import json, sys
sys.path.insert(0, {tests_dir!r})
import jax
from test_harness import _make_collection, _LinearReactionModule, _biomass_loss
from bp_train.training_data import TrainingDataStore
from bp_train.harness import train_collection, TrainHarnessConfig

collection = _make_collection()
store = TrainingDataStore.from_collection(
    collection, target_variable_order=["biomass"], target_source="reactor_components"
)
result = train_collection(
    store,
    reaction_module=_LinearReactionModule(),
    loss_module=_biomass_loss(),
    collection=collection,
    config=TrainHarnessConfig(
        process_names=("p1", "p2"), steps=4, batch_size=2,
        optimizer_name="adam", learning_rate=2e-2, log_every=1,
        shuffle_batches=False,
    ),
)
print("RESULT_JSON " + json.dumps(
    {{"devices": int(jax.device_count()), "losses": [float(x) for x in result.mean_loss_by_step]}}
))
"""


def _run(devices: str) -> dict:
    env = {**os.environ, "JAX_PLATFORMS": "cpu", "BP_TRAIN_DEVICES": devices}
    env.pop("XLA_FLAGS", None)  # let the bootstrap set the device count cleanly
    proc = subprocess.run(
        [sys.executable, "-c", _SCRIPT.format(tests_dir=_TESTS_DIR)],
        env=env, capture_output=True, text=True, timeout=300,
    )
    assert proc.returncode == 0, f"subprocess (devices={devices}) failed:\n{proc.stderr[-3000:]}"
    line = next(ln for ln in proc.stdout.splitlines() if ln.startswith("RESULT_JSON "))
    return json.loads(line[len("RESULT_JSON "):])


def test_sharded_training_matches_vmap():
    vmap = _run("1")
    sharded = _run("2")
    assert vmap["devices"] == 1
    assert sharded["devices"] == 2, "BP_TRAIN_DEVICES=2 should expose 2 CPU devices"
    assert len(vmap["losses"]) == len(sharded["losses"]) == 4
    # loss should decrease (sanity) ...
    assert sharded["losses"][-1] < sharded["losses"][0]
    # ... and match the single-device run up to float32 reduction order.
    for a, b in zip(vmap["losses"], sharded["losses"]):
        assert abs(a - b) <= 1e-4 + 1e-4 * abs(a), (vmap["losses"], sharded["losses"])
