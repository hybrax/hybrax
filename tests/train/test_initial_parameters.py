from __future__ import annotations

import csv
import json
import shutil
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from test_serialization import _collection

from hybrax.format.serialization import save_process_collection
from hybrax.train import cli, harness, serialization
from hybrax.train.model_api import partition_trainable
from hybrax.train.run_config import load_train_config

_CUSTOM_CODE = """\
import jax
import jax.numpy as jnp
import optax
from hybrax.train.defaults import DefaultLossModule
from hybrax.train.model_api import RateModule, ReactionOutputs, LossOutputs
from hybrax.train.model_api import trainable_field


class Reaction(RateModule):
    rate: jax.Array = trainable_field()

    def __init__(self, **scales):
        super().__init__(**scales)
        self.rate = jnp.array([0.1])

    def __call__(self, t, inputs):
        return ReactionOutputs(
            SCL_modeled_ReactionOde_rates=self.rate * inputs.SCL_modeled_RMCs,
            SCL_modeled_Inflows_rates=jnp.zeros(0),
            SCL_modeled_Outflows_rates=jnp.zeros(0),
        )


class Loss(DefaultLossModule):
    weight: jax.Array = trainable_field()

    def __init__(self, target_names):
        super().__init__(target_names=target_names)
        self.weight = jnp.array(1.0)

    @property
    def loss_names(self):
        return self.target_names + ("phase_step",)

    def __call__(self, inputs):
        losses = super().__call__(inputs).named_losses
        return LossOutputs(named_losses={
            **{name: self.weight ** 2 * value for name, value in losses.items()},
            "phase_step": inputs.step.astype(float),
        })


def build_reaction_module(**kwargs):
    return Reaction(**{
        name: value for name, value in kwargs.items() if name.startswith("SCALE_")
    })


def build_loss_module(*, target_names, **kwargs):
    return Loss(target_names)


def build_learning_rate(config, train_config, total_updates):
    return optax.exponential_decay(
        train_config.learning_rate * LR_FACTOR, 1, DECAY
    )
"""


def _write_phase_config(source, output: Path) -> Path:
    config = json.loads(source.phase_config.read_text())
    config["output"]["dir"] = str(output)
    source.phase_config.write_text(json.dumps(config))
    return source.phase_config


def _arrays(tree):
    return [leaf for leaf in jax.tree_util.tree_leaves(tree) if eqx.is_array(leaf)]


def _counts(state):
    return [int(value) for value in _arrays(state) if value.dtype.kind in "iu"]


def _assert_trainables_equal(actual, expected):
    actual_arrays = _arrays(partition_trainable(actual)[0])
    expected_arrays = _arrays(partition_trainable(expected)[0])
    assert len(actual_arrays) == len(expected_arrays) > 0
    for actual_array, expected_array in zip(actual_arrays, expected_arrays):
        np.testing.assert_array_equal(actual_array, expected_array)


@pytest.fixture
def initial_source(tmp_path: Path):
    """Real parameter/state files with source builders that cannot be imported."""
    collection = _collection(n_processes=2)
    prepared_path = tmp_path / "prepared.json"
    save_process_collection(collection, prepared_path)
    custom_path = tmp_path / "phase_custom.py"
    custom_path.write_text(_CUSTOM_CODE + "\nLR_FACTOR = 0.25\nDECAY = 0.5\n")
    phase_config = tmp_path / "phase.json"
    config = {
        "data": {
            "prepared": str(prepared_path),
            "targets": ["biomass"],
            "target_source": "reactor_components",
        },
        "custom_py": str(custom_path),
        "train": {"epochs": 2, "learning_rate": 0.08, "seed": 0},
        "checkpoint": {"every": 1},
        "output": {"dir": str(tmp_path / "phase"), "predictions": "none"},
    }
    phase_config.write_text(json.dumps(config))
    loaded = load_train_config(phase_config)
    prepared = harness.prepare_training(
        collection,
        config=harness.train_harness_config_from_run_config(
            loaded.config, run_dir=tmp_path / "phase"
        ),
        run_config=loaded.config,
        custom_module=loaded.custom_module,
    )
    wrapper = harness._build_template_wrapper(
        prepared.store,
        reaction_module=prepared.reaction_module,
        loss_module=prepared.loss_module,
        selected_processes=prepared.config.process_names,
    )
    wrapper = eqx.tree_at(
        lambda model: (model.reaction_module.rate, model.loss_module.weight),
        wrapper,
        (jnp.array([-0.15]), jnp.array(0.7)),
    )
    source_dir = tmp_path / "source"
    checkpoint = source_dir / "checkpoints" / "step_00007"
    checkpoint.mkdir(parents=True)
    model_dir = source_dir / "model"
    model_dir.mkdir()
    for directory in (checkpoint, model_dir):
        serialization.save_model(wrapper, directory / "params.eqx")
    trainable = partition_trainable(wrapper)[0]
    optimizer = harness._build_optimizer("adam", lambda step: 0.04 * 0.9**step)
    state = optimizer.init(trainable)
    grads = jax.tree_util.tree_map(jnp.ones_like, trainable)
    for _ in range(7):
        _, state = optimizer.update(grads, state, params=trainable)
    assert _counts(state) == [7, 7]
    serialization.save_opt_state(state, checkpoint / "opt_state.eqx")
    serialization.write_json(
        checkpoint / "train_state.json", {"step": 7, "samples_seen": 14}
    )
    document = {
        "status": "complete",
        "updates_completed": 7,
        "config": {
            **config,
            "custom_py": str(source_dir / "custom.py"),
            "data": {**config["data"], "prepared": str(source_dir / "missing.json")},
            "train": {**config["train"], "epochs": 7, "learning_rate": 0.04},
            "output": {**config["output"], "dir": str(source_dir)},
        },
        "inputs": {
            "prepared_input": {"content_hash": serialization.content_hash(collection)}
        },
    }
    for directory in (source_dir, checkpoint):
        serialization.write_json(directory / "config.json", document)
        (directory / "custom.py").write_text(
            _CUSTOM_CODE
            + "\nLR_FACTOR = 1.0\nDECAY = 0.9\n"
            + 'raise AssertionError("source builders must not be imported")\n'
        )
    latest = source_dir / "checkpoints" / "latest"
    latest.symlink_to(checkpoint.name)
    return SimpleNamespace(
        collection=collection,
        prepared_path=prepared_path,
        custom_path=custom_path,
        phase_config=phase_config,
        directory=source_dir,
        checkpoint=checkpoint,
        latest=latest,
        document=document,
        prepared=prepared,
        wrapper=wrapper,
    )


def test_initial_parameters_replace_both_modules_and_keep_phase_context(initial_source):
    source = initial_source
    prepared = source.prepared
    loaded = harness.with_initial_parameters(prepared, source.checkpoint / "params.eqx")

    np.testing.assert_array_equal(loaded.reaction_module.rate, [-0.15])
    np.testing.assert_array_equal(loaded.loss_module.weight, 0.7)
    assert loaded.store is prepared.store
    assert loaded.config is prepared.config
    assert loaded.optimizer is prepared.optimizer
    assert loaded.prediction_parent_process_names == ("p1", "p2")
    np.testing.assert_array_equal(prepared.reaction_module.rate, [0.1])
    np.testing.assert_array_equal(prepared.loss_module.weight, 1.0)


@pytest.mark.parametrize("reference", ["latest", "directory"])
def test_cli_initial_parameters_start_fresh_phase_and_reload_independently(
    initial_source, monkeypatch, tmp_path: Path, reference: str
):
    source = initial_source
    source_reference = getattr(source, reference)
    source_params = (
        source.checkpoint if reference == "latest" else source.directory / "model"
    ) / "params.eqx"
    output = tmp_path / "phase"
    config = _write_phase_config(source, output)
    real_train = cli.train_collection
    seen = {}

    def record_start(store, **kwargs):
        reaction, loss = kwargs["reaction_module"], kwargs["loss_module"]
        np.testing.assert_array_equal(
            reaction.rate, source.wrapper.reaction_module.rate
        )
        np.testing.assert_array_equal(loss.weight, source.wrapper.loss_module.weight)
        wrapper = harness._build_template_wrapper(
            store,
            reaction_module=reaction,
            loss_module=loss,
            selected_processes=kwargs["config"].process_names,
        )
        state = kwargs["optimizer"].init(partition_trainable(wrapper)[0])
        assert _counts(state) == [0, 0]
        for leaf in _arrays(state):
            np.testing.assert_array_equal(leaf, jnp.zeros_like(leaf))
        schedule = kwargs["config"].learning_rate
        assert float(schedule(0)) == pytest.approx(0.02)
        assert float(schedule(1)) == pytest.approx(0.01)
        seen["optimizer"] = kwargs["optimizer"]
        seen["result"] = real_train(store, **kwargs)
        return seen["result"]

    monkeypatch.setattr(cli, "train_collection", record_start)
    with monkeypatch.context() as patch:
        patch.setattr(
            serialization,
            "load_process_collection",
            lambda *_args: pytest.fail("initialization read source prepared data"),
        )
        patch.setattr(
            serialization,
            "load_opt_state",
            lambda *_args, **_kwargs: pytest.fail("loaded source optimizer state"),
        )
        assert (
            cli.main(
                ["train", "--config", str(config), "--init-from", str(source_reference)]
            )
            == 0
        )

    result = seen["result"]
    assert result.updates_completed == 2
    assert len(result.mean_loss_by_step) == 2
    np.testing.assert_array_equal(
        np.asarray(result.per_target_loss_by_step)[:, 1], [1.0, 2.0]
    )
    document = serialization.read_json(output / "config.json")
    assert "init_checkpoint" not in document["config"]
    assert document["inputs"]["init_checkpoint"] == {
        "path": str(source_params.parent.resolve()),
        "step": 7,
        "params_hash": serialization.file_hash(source_params),
        "prepared_hash": serialization.content_hash(source.collection),
        "optimizer_state": "fresh",
    }
    with (output / "metrics.csv").open() as stream:
        rows = list(csv.DictReader(stream))
    assert [int(row["step"]) for row in rows] == [1, 2]
    assert [int(row["samples_seen"]) for row in rows] == [2, 4]
    checkpoint = output / "checkpoints" / "latest"
    state_template = seen["optimizer"].init(
        partition_trainable(result.trained_wrapper)[0]
    )
    state = serialization.load_opt_state(
        checkpoint / "opt_state.eqx", template=state_template
    )
    assert _counts(state) == [2, 2]
    assert serialization.read_json(checkpoint / "train_state.json")["samples_seen"] == 4

    shutil.rmtree(source.directory)
    source.custom_path.unlink()
    reloaded, config = serialization.model_load(output)
    assert config.train.epochs == 2
    _assert_trainables_equal(reloaded, result.trained_wrapper)
    reloaded, _ = serialization.model_load(checkpoint)
    _assert_trainables_equal(reloaded, result.trained_wrapper)


@pytest.mark.parametrize(
    ("mismatch", "message"),
    [
        ("content", "prepared data differs"),
        ("missing_hash", "prepared data differs"),
        ("targets", "targets differ"),
        ("target_source", "targets differ"),
        ("selection", "training selection differs"),
        ("unresolved_holdout", "resolved training selection"),
    ],
)
def test_cli_initial_metadata_mismatch_rejected_before_output_writes(
    initial_source, tmp_path: Path, mismatch: str, message: str
):
    source = initial_source
    output = tmp_path / "phase"
    config = _write_phase_config(source, output)
    document = source.document
    data = document["config"]["data"]
    if mismatch == "content":
        save_process_collection(
            _collection((1.0, 0.7, 0.49), n_processes=2), source.prepared_path
        )
    elif mismatch == "missing_hash":
        document["inputs"].clear()
    elif mismatch == "targets":
        data["targets"] = ["other"]
    elif mismatch == "target_source":
        data["target_source"] = "combined"
    elif mismatch == "selection":
        data["processes"] = ["p2", "p1"]
    elif mismatch == "unresolved_holdout":
        document["config"]["train"]["holdout_processes"] = ["p2"]
    serialization.write_json(source.checkpoint / "config.json", document)
    source_bytes = (source.checkpoint / "params.eqx").read_bytes()

    with pytest.raises(ValueError, match=message):
        cli.main(["train", "--config", str(config), "--init-from", str(source.latest)])

    assert not output.exists()
    assert (source.checkpoint / "params.eqx").read_bytes() == source_bytes


def test_python_initial_parameters_use_loaded_modules(initial_source, monkeypatch):
    source = initial_source
    result = object()
    seen = []

    def execute(store, **kwargs):
        np.testing.assert_array_equal(kwargs["reaction_module"].rate, [-0.15])
        np.testing.assert_array_equal(kwargs["loss_module"].weight, 0.7)
        assert kwargs["config"].epochs == 2
        seen.append(store)
        return result

    monkeypatch.setattr(harness, "train_collection", execute)
    assert (
        harness.train_from_collection(
            source.collection,
            config=replace(source.prepared.config, learning_rate=0.08),
            custom_py=source.custom_path,
            init_from=source.checkpoint,
        )
        is result
    )
    assert len(seen) == 1


def test_parameter_shape_mismatch_preserves_existing_phase_output(
    initial_source, tmp_path: Path
):
    source = initial_source
    incompatible = eqx.tree_at(
        lambda wrapper: wrapper.reaction_module.rate,
        source.wrapper,
        jnp.array([-0.15, -0.2]),
    )
    serialization.save_model(incompatible, source.checkpoint / "params.eqx")
    output = tmp_path / "phase"
    output.mkdir()
    marker = output / "previous.txt"
    marker.write_text("keep previous phase")
    config = _write_phase_config(source, output)

    with pytest.raises(RuntimeError, match="changed shape"):
        cli.main(
            [
                "train",
                "--config",
                str(config),
                "--init-from",
                str(source.latest),
                "--overwrite",
            ]
        )
    assert list(output.iterdir()) == [marker]
    assert marker.read_text() == "keep previous phase"


def test_source_output_overlap_rejected_for_cli_and_python(initial_source, monkeypatch):
    source = initial_source
    original_files = {
        path: path.read_bytes()
        for path in source.directory.rglob("*")
        if path.is_file()
    }
    config = _write_phase_config(source, source.directory)
    monkeypatch.setattr(
        cli,
        "load_process_collection",
        lambda *_args: pytest.fail("overlap guard ran after data loading"),
    )
    with pytest.raises(ValueError, match="inside output directory"):
        cli.main(
            [
                "train",
                "--config",
                str(config),
                "--init-from",
                str(source.latest),
                "--overwrite",
            ]
        )
    monkeypatch.setattr(
        harness,
        "prepare_training",
        lambda *_args, **_kwargs: pytest.fail("overlap guard ran after preparation"),
    )
    with pytest.raises(ValueError, match="inside output directory"):
        harness.train_from_collection(
            source.collection,
            config=replace(
                source.prepared.config,
                checkpoint_dir=source.directory / "checkpoints",
            ),
            init_from=source.latest,
        )
    assert original_files == {
        path: path.read_bytes()
        for path in source.directory.rglob("*")
        if path.is_file()
    }
