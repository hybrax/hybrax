"""Regression test for the opt-in pmap multi-core training path.

The device count is fixed at JAX initialisation, so we exercise the sharded
(pmap) step vs the single-device (vmap) step in subprocesses: one with
``HYBRAX_TRAIN_DEVICES=1`` and one with ``=2``. A reordered three-process batch with
distinct sample events forces one padded sharded row. The runs must train
identically up to float32 cross-device reduction order.
"""

import dataclasses
import json
import os
import subprocess
import sys
from copy import deepcopy
from pathlib import Path

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest
from test_harness import _LinearReactionModule, _make_collection

import hybrax.train.harness as harness_module
from hybrax.format.dataclasses import (
    FeedMedium,
    FeedMediumComponent,
    Inflow,
    StaticVariable,
    TimeSeries,
)
from hybrax.train.harness import (
    TrainHarnessConfig,
    _build_optimizer,
    _build_template_wrapper,
    train_collection,
)
from hybrax.train.model_api import (
    LossOutputs,
    UserLossModule,
    partition_trainable,
    trainable_field,
)
from hybrax.train.serialization import load_opt_state
from hybrax.train.trainer import build_batched_loss_fn, clamp_padded_time_rows
from hybrax.train.training_data import TrainingDataStore


def _with_events(process, name, sample_value, bolus_concentration):
    process = deepcopy(process)
    sample = process.volume.volume_changes["sample_1"]
    sample = dataclasses.replace(
        sample,
        values=dataclasses.replace(sample.values, values=jnp.asarray([sample_value])),
    )
    bolus = Inflow(
        name="bolus",
        unit="L",
        is_controlled=True,
        is_continuous=False,
        values=TimeSeries(times=jnp.asarray([0.5]), values=jnp.asarray([0.1])),
        feed_medium=FeedMedium(
            name="feed",
            density=1.0,
            density_unit="kg/L",
            components={
                "biomass": FeedMediumComponent(
                    name="biomass",
                    unit="g/L",
                    concentration=StaticVariable(bolus_concentration),
                    is_controlled=False,
                )
            },
        ),
    )
    volume = dataclasses.replace(
        process.volume,
        volume_changes={
            **process.volume.volume_changes,
            "sample_1": sample,
            "bolus": bolus,
        },
    )
    metadata = dataclasses.replace(process.metadata, name=name)
    return dataclasses.replace(process, metadata=metadata, volume=volume)


def _make_event_collection():
    collection = _make_collection()
    collection.processes["p1"] = _with_events(
        collection.processes["p1"], "p1", -0.1, 2.0
    )
    collection.processes["p2"] = _with_events(
        collection.processes["p2"], "p2", -0.2, 4.0
    )
    collection.processes["p3"] = _with_events(
        collection.processes["p1"], "p3", -0.3, 6.0
    )
    return collection


_TESTS_DIR = str(Path(__file__).resolve().parent)

# Runs in a fresh interpreter so HYBRAX_TRAIN_DEVICES is honoured before JAX inits.
_SCRIPT = """
import json, sys
sys.path.insert(0, {tests_dir!r})
import jax
import jax.numpy as jnp
from test_harness import _make_collection, _LinearReactionModule, _biomass_loss
from test_sharded_training import _make_event_collection
from hybrax.train.training_data import TrainingDataStore
from hybrax.train.harness import train_collection, TrainHarnessConfig

collection = _make_event_collection()
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
import copy, dataclasses, json, sys, tempfile
from pathlib import Path
sys.path.insert(0, {tests_dir!r})
import jax
import jax.numpy as jnp
from test_harness import _make_collection, _LinearReactionModule, _biomass_loss
from hybrax.train.harness import train_collection, TrainHarnessConfig
from hybrax.train.model_api import ReactionOutputs
from hybrax.train.training_data import TrainingDataStore

class P2OnlyBlowUp(_LinearReactionModule):
    def __call__(self, t, inputs):
        state = inputs.SCL_modeled_RMCs[0]
        is_p2 = inputs.SCL_modeled_V > 0.95
        rate = jnp.where((t > 1.0) & is_p2, 1.0e4 * state, 0.0)
        return ReactionOutputs(
            SCL_modeled_ReactionOde_rates=jnp.asarray([rate], dtype=state.dtype),
            SCL_modeled_Inflows_rates=jnp.zeros((0,), dtype=state.dtype),
            SCL_modeled_Outflows_rates=jnp.zeros(0),
        )

collection = _make_collection()
collection.processes["p3"] = dataclasses.replace(
    copy.deepcopy(collection.processes["p1"]),
    metadata=dataclasses.replace(collection.processes["p1"].metadata, name="p3"),
)
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
            process_names={process_names!r}, epochs=1, batch_size={batch_size},
            microbatch_size={microbatch_size!r},
            shuffle_batches=False, solver_max_steps=512, metrics_jsonl=metrics,
        ),
    )
    row = json.loads(metrics.read_text())
print("RESULT_JSON " + json.dumps({{"devices": jax.device_count(), "row": row}}))
"""


def _run(devices: str, *, gspmd: bool = False) -> dict:
    env = {**os.environ, "JAX_PLATFORMS": "cpu", "HYBRAX_TRAIN_DEVICES": devices}
    env.pop("XLA_FLAGS", None)  # let the bootstrap set the device count cleanly
    if gspmd:
        env["HYBRAX_GSPMD"] = "1"
    else:
        env.pop("HYBRAX_GSPMD", None)
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
    assert mode in ("vmap", "pmap", "gspmd", "microbatch")
    devices = "1" if mode in ("vmap", "microbatch") else "2"
    env = {**os.environ, "JAX_PLATFORMS": "cpu", "HYBRAX_TRAIN_DEVICES": devices}
    env.pop("XLA_FLAGS", None)
    if mode == "gspmd":
        env["HYBRAX_GSPMD"] = "1"
    else:
        env.pop("HYBRAX_GSPMD", None)
    proc = subprocess.run(
        [
            sys.executable,
            "-c",
            _TELEMETRY_SCRIPT.format(
                tests_dir=_TESTS_DIR,
                process_names=("p1", "p3", "p2")
                if mode == "microbatch"
                else ("p2", "p1"),
                batch_size=3 if mode == "microbatch" else 2,
                microbatch_size=2 if mode == "microbatch" else None,
            ),
        ],
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
        "HYBRAX_TRAIN_DEVICES": "9999",
    }
    env.pop("XLA_FLAGS", None)
    proc = subprocess.run(
        [
            sys.executable,
            "-c",
            "import hybrax.train, jax, os; "
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


class _MicrobatchLoss(UserLossModule):
    weight: jax.Array = trainable_field(default_factory=lambda: jnp.asarray(0.4))

    @property
    def loss_names(self):
        return ("fit", "dense")

    @property
    def dense_grid_n(self):
        return 3

    def __call__(self, inputs):
        residual = inputs.SCL_target_pred - inputs.SCL_target_measured
        fit = jnp.mean(jnp.where(inputs.mask_measured, residual**2, 0))
        return LossOutputs(
            {
                "fit": self.weight * fit + 0.01 * inputs.step * self.weight**2,
                "dense": 0.02 * jnp.mean(inputs.dense_SCL_states[:, 0] ** 2),
            }
        )


@pytest.mark.parametrize("use_jump", [False, True])
def test_microbatch_ode_gradients_and_adam_state_match_full_batch(use_jump):
    """Check the complete solve outputs and raw gradients before active clipping."""
    collection = _make_event_collection()
    store = TrainingDataStore.from_collection(
        collection,
        target_variable_order=["biomass"],
        target_source="reactor_components",
    )
    names = ("p3", "p1", "p2")
    batch = store.gather_batch(jnp.asarray([2, 0, 1]))
    wrapper = _build_template_wrapper(
        store,
        reaction_module=_LinearReactionModule(),
        selected_processes=names,
        loss_module=_MicrobatchLoss(),
    )
    params, static = partition_trainable(wrapper)
    jump = (
        clamp_padded_time_rows(
            store.controls_store.jump_ts[batch.process_indices],
            store.controls_store.jump_ts_lengths[batch.process_indices],
        )
        if use_jump
        else None
    )
    loss_fn = build_batched_loss_fn()

    def objective(params, width):
        outputs = loss_fn(
            eqx.combine(params, static),
            batch,
            store.Cin_controlled_Inflows,
            store.Cin_modeled_Inflows,
            jump,
            max_solver_steps=2048,
            solver_rtol=1e-5,
            solver_atol=1e-7,
            step=7,
            microbatch_size=width,
        )
        return outputs[0], outputs

    evaluate = eqx.filter_jit(eqx.filter_value_and_grad(objective, has_aux=True))
    expected, full_grad = evaluate(params, None)
    assert float(optax.tree.norm(full_grad)) > 0.001
    assert abs(float(full_grad.loss_module.weight)) > 0
    optimizer = _build_optimizer(
        "adam",
        optax.linear_schedule(0.02, 0.001, 10),
        grad_clip_norm=0.001,
    )
    # Nonzero moments expose incorrect update counts and optimizer transforms.
    _, state = optimizer.update(full_grad, optimizer.init(params), params=params)
    expected_update, expected_state = optimizer.update(full_grad, state, params=params)
    for width in (1, 2):
        actual, grad = evaluate(params, width)
        update, next_state = optimizer.update(grad, state, params=params)
        for observed, reference in zip(
            jax.tree.leaves((actual, grad, update, next_state)),
            jax.tree.leaves((expected, full_grad, expected_update, expected_state)),
            strict=True,
        ):
            np.testing.assert_allclose(observed, reference, rtol=1e-7, atol=1e-8)
        np.testing.assert_allclose(
            optax.tree.norm(grad),
            optax.tree.norm(full_grad),
            rtol=1e-7,
            atol=1e-8,
        )


@pytest.mark.parametrize("batch_size,width", [(1, 4), (3, 1)])
def test_microbatch_rejects_actual_multiple_devices(batch_size, width):
    script = f"""
import jax
from hybrax.train.harness import TrainHarnessConfig, _validate_batching_config
assert jax.local_device_count() == 2
_validate_batching_config(
    TrainHarnessConfig(batch_size={batch_size}, microbatch_size={width}),
    selected_process_count=3,
)
"""
    env = {**os.environ, "HYBRAX_TRAIN_DEVICES": "2", "JAX_PLATFORMS": "cpu"}
    env.pop("XLA_FLAGS", None)
    proc = subprocess.run(
        [sys.executable, "-c", script],
        env=env,
        capture_output=True,
        text=True,
        timeout=300,
        check=False,
    )
    assert proc.returncode != 0
    assert "microbatching requires one local JAX device" in proc.stderr


def test_microbatch_failure_in_short_tail_keeps_process_identity():
    row = _run_telemetry("microbatch")["row"]
    assert row["process_names"] == ["p1", "p3", "p2"]
    assert row["failed_processes"] == ["p2"]
    assert row["n_failed_samples"] == 1


def test_microbatch_harness_preserves_updates_and_checkpoint_state(
    tmp_path, monkeypatch
):
    collection = _make_event_collection()
    store = TrainingDataStore.from_collection(
        collection,
        target_variable_order=["biomass"],
        target_source="reactor_components",
    )
    seen_widths = []
    original_loss = harness_module._BATCHED_LOSS_FN

    def record_loss(*args, **kwargs):
        seen_widths.append(kwargs.get("microbatch_size"))
        return original_loss(*args, **kwargs)

    monkeypatch.setattr(harness_module, "_BATCHED_LOSS_FN", record_loss)
    reference = None
    for width in (None, 2):
        run_dir = tmp_path / str(width)
        schedule = optax.linear_schedule(0.02, 0.001, 10)
        result = train_collection(
            store,
            reaction_module=_LinearReactionModule(),
            loss_module=_MicrobatchLoss(),
            config=TrainHarnessConfig(
                process_names=("p3", "p1", "p2"),
                epochs=3,
                batch_size=3,
                microbatch_size=width,
                shuffle_batches=False,
                learning_rate=schedule,
                grad_clip_norm=0.001,
                metrics_jsonl=run_dir / "metrics.jsonl",
                checkpoint_dir=run_dir / "checkpoints",
                checkpoint_every=0,
            ),
        )
        assert seen_widths[-1] == width
        params, _ = partition_trainable(result.trained_wrapper)
        optimizer = _build_optimizer("adam", schedule, grad_clip_norm=0.001)
        latest = run_dir / "checkpoints/latest"
        state = load_opt_state(
            latest / "opt_state.eqx", template=optimizer.init(params)
        )
        metadata = json.loads((latest / "train_state.json").read_text())
        assert metadata["step"] == result.updates_completed == 3
        assert metadata["samples_seen"] == 9
        counts = [
            int(x)
            for x in jax.tree.leaves(state)
            if jnp.issubdtype(x.dtype, jnp.integer)
        ]
        assert counts and all(count == 3 for count in counts)
        assert result.train_step_rebuild_count == 0
        assert result.batch_process_names_by_step == (("p3", "p1", "p2"),) * 3
        rows = [
            json.loads(line)
            for line in (run_dir / "metrics.jsonl").read_text().splitlines()
        ]
        assert [row["step"] for row in rows] == [1, 2, 3]
        assert [row["samples_seen"] for row in rows] == [3, 6, 9]
        assert all(norm > 0.001 for norm in result.grad_norm_by_step)
        observed = (
            params,
            state,
            result.mean_loss_by_step,
            result.per_process_loss_by_step,
            result.grad_norm_by_step,
        )
        if reference is None:
            reference = observed
        else:
            for actual, expected in zip(
                jax.tree.leaves(observed), jax.tree.leaves(reference), strict=True
            ):
                np.testing.assert_allclose(actual, expected, rtol=1e-7, atol=1e-8)
