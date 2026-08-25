"""The shared reconstruction path: every loader rebuilds a model from ITS OWN input.

``model_load``, ``forward_from_collection`` (standalone and per ensemble member),
and artifact-backed LOO folds all go through
:func:`hybrax.train.serialization.reconstruct_training`, which

- loads the prepared collection the model was trained on (never the evaluation
  collection),
- requires and verifies that input's recorded
  ``inputs.prepared_input.content_hash`` **before** any hook runs,
- restricts the hook-visible data to the recorded training process selection.

The fixtures give the two processes very different biomass levels and use a
``custom.py`` whose ``estimate_all_scales`` is max-abs over the training parents,
so a scale value is a fingerprint of *which* processes fed reconstruction.
"""

from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path

import jax.numpy as jnp
import numpy as np
import pytest
from hybrax.format.dataclasses import (
    BioProcessCollection,
    FeedMedium,
    FeedMediumComponent,
    Inflow,
    Outflow,
    ProcessVariable,
    ReactorMediumComponent,
    StaticVariable,
    TimeSeries,
)
from hybrax.format.serialization import load_process_collection, save_process_collection

import hybrax.train
import hybrax.train.harness as harness
from hybrax.train.cli import main
from hybrax.train.harness import ForwardConfig, forward_from_collection
from hybrax.train.serialization import content_hash, reconstruct_training
from hybrax.train.training_data import TARGET_SOURCE_AUTO
from stateful_helpers import make_process


def _collection(
    biomass_values=(1.0, 0.8, 0.64), *, n_processes: int = 1
) -> BioProcessCollection:
    p1 = make_process()
    p1.reactor_medium.components["biomass"].concentration = TimeSeries(
        times=jnp.asarray([0.0, 1.0, 2.0]),
        values=jnp.asarray(list(biomass_values)),
    )
    p1.volume.volume_changes["sample_1"] = Outflow(
        name="sample_1",
        unit="L",
        is_controlled=False,
        is_continuous=False,
        values=TimeSeries(
            times=jnp.asarray([1.0]),
            values=jnp.asarray([-0.1]),
        ),
    )
    return BioProcessCollection(
        processes={
            f"p{i}": replace(p1, metadata=replace(p1.metadata, name=f"p{i}"))
            for i in range(1, n_processes + 1)
        },
        metadata={},
    )


# `estimate_all_scales` = max-abs over the *training parents* only, so
# SCALE_modeled_RMCs is a readable fingerprint of the reconstruction inputs.
_CUSTOM_PY = """
import jax.numpy as jnp
import numpy as np

from hybrax.train.model_api import EstimatedScales


def estimate_all_scales(runtime_data, target_names, config):
    del config
    rhs_ode = runtime_data.rhs_ode
    scale = np.ones(len(rhs_ode.name_modeled_RMCs), dtype=float)
    for i, name in enumerate(rhs_ode.name_modeled_RMCs):
        peak = 0.0
        for index in range(len(runtime_data.process_order)):
            values = np.asarray(runtime_data.raw_state_trace(index, name)[1], float)
            peak = max(peak, float(np.max(np.abs(values))) if values.size else 0.0)
        scale[i] = max(peak, 1e-6)
    return EstimatedScales(
        SCALE_modeled_RMCs=jnp.asarray(scale),
        SCALE_modeled_PVs=jnp.ones(len(rhs_ode.name_modeled_PVs)),
        SCALE_V_in_cumulative=jnp.asarray(1.0),
        SCALE_modeled_Inflows_cumulative=jnp.ones(len(rhs_ode.name_modeled_Inflows)),
        SCALE_modeled_Outflows_cumulative=jnp.ones(len(rhs_ode.name_modeled_Outflows)),
        SCALE_controlled_Inflows_cumulative=jnp.ones(len(rhs_ode.name_controlled_Inflows)),
        SCALE_controlled_Outflows_cumulative=jnp.ones(len(rhs_ode.name_controlled_Outflows)),
        SCALE_controlled_Inflows_rates=jnp.ones(len(rhs_ode.name_controlled_Inflows)),
        SCALE_controlled_Outflows_rates=jnp.ones(len(rhs_ode.name_controlled_Outflows)),
        SCALE_controlled_Inflows_Cin=jnp.ones(
            (len(rhs_ode.name_controlled_Inflows), len(rhs_ode.name_modeled_RMCs))
        ),
        SCALE_controlled_PVs=jnp.ones(len(rhs_ode.name_controlled_PVs)),
        SCALE_modeled_Inflows_Cin=jnp.ones(
            (len(rhs_ode.name_modeled_Inflows), len(rhs_ode.name_modeled_RMCs))
        ),
        SCALE_modeled_BiologicalOde_rates=jnp.ones(len(rhs_ode.name_modeled_rates)),
        SCALE_modeled_Inflows_rates=jnp.ones(len(rhs_ode.name_modeled_Inflows)),
        SCALE_modeled_Outflows_rates=jnp.ones(len(rhs_ode.name_modeled_Outflows)),
    )
"""


# ---------------------------------------------------------------------------
# fixtures
# ---------------------------------------------------------------------------


def _controlled_feed(cin: float, final_amount: float = 0.1) -> Inflow:
    """A controlled continuous feed carrying biomass at ``cin`` g/L.

    A feed is what makes the reference-process choice observable: the
    deserialisation template bakes ONE process's ``Cin`` row and controls, and
    without a feed every reference is indistinguishable.
    """
    return Inflow(
        name="feed_A",
        unit="L",
        is_controlled=True,
        is_continuous=True,
        values=TimeSeries(
            times=jnp.asarray([0.0, 1.0, 2.0]),
            values=jnp.asarray([0.0, final_amount / 2, final_amount]),
        ),
        feed_medium=FeedMedium(
            name="feed",
            density=1.0,
            density_unit="kg/L",
            components={
                "biomass": FeedMediumComponent(
                    name="biomass",
                    unit="g/L",
                    concentration=StaticVariable(cin),
                    is_controlled=False,
                )
            },
        ),
    )


def _collection_with(
    levels: dict[str, tuple[float, ...]],
    *,
    feed_cin: dict[str, float] | None = None,
    feed_amount: dict[str, float] | None = None,
) -> BioProcessCollection:
    """A collection with one process per entry, each at its own biomass level.

    ``feed_cin`` optionally gives each process a controlled feed of its own
    composition; processes stay feed-free when it is omitted.
    """
    template = _collection().processes["p1"]
    processes = {}
    for name, values in levels.items():
        biomass = replace(
            template.reactor_medium.components["biomass"],
            concentration=TimeSeries(
                times=jnp.asarray([0.0, 1.0, 2.0]),
                values=jnp.asarray(list(values)),
            ),
        )
        volume = template.volume
        if feed_cin is not None:
            volume = replace(
                volume,
                volume_changes={
                    **volume.volume_changes,
                    "feed_A": _controlled_feed(
                        feed_cin[name],
                        0.1 if feed_amount is None else feed_amount[name],
                    ),
                },
            )
        processes[name] = replace(
            template,
            metadata=replace(template.metadata, name=name),
            volume=volume,
            reactor_medium=replace(
                template.reactor_medium, components={"biomass": biomass}
            ),
        )
    return BioProcessCollection(processes=processes, metadata={})


def _write_prepared(
    path: Path,
    levels: dict[str, tuple[float, ...]],
    *,
    feed_cin: dict[str, float] | None = None,
    feed_amount: dict[str, float] | None = None,
) -> Path:
    save_process_collection(
        _collection_with(levels, feed_cin=feed_cin, feed_amount=feed_amount), path
    )
    return path


def _train(
    tmp_path: Path,
    *,
    name: str,
    prepared: Path,
    processes: tuple[str, ...] | None = None,
    targets: tuple[str, ...] = ("biomass",),
    target_source: str = "reactor_components",
) -> Path:
    """Run the real ``train`` CLI and return its run dir."""
    run_dir = tmp_path / name
    custom_py = tmp_path / f"{name}-custom.py"
    custom_py.write_text(_CUSTOM_PY, encoding="utf-8")
    data: dict = {
        "prepared": str(prepared),
        "targets": list(targets),
        "target_source": target_source,
    }
    if processes is not None:
        data["processes"] = list(processes)
    config_path = tmp_path / f"{name}-config.json"
    config_path.write_text(
        json.dumps(
            {
                "data": data,
                "custom_py": str(custom_py),
                "train": {"epochs": 1, "learning_rate": 0.05, "seed": 0},
                "solver": {"max_steps": 2048, "rtol": 1e-4, "atol": 1e-6},
                "output": {"dir": str(run_dir), "predictions": "none"},
            }
        ),
        encoding="utf-8",
    )
    assert main(["train", "--config", str(config_path)]) == 0
    return run_dir


def _rmc_scale(wrapper) -> np.ndarray:
    return np.asarray(wrapper.reaction_module.SCALE_modeled_RMCs.scale)


@pytest.fixture(scope="module")
def two_level_run(tmp_path_factory):
    """A run trained on p1 (peak 1.0) only, out of a p1 + p2 (peak 100.0) input.

    Module-scoped: training it is the expensive part, and every test that shares
    it only reads it. Tests that break a run's record build their own throwaway.
    """
    tmp_path = tmp_path_factory.mktemp("two_level_run")
    prepared = _write_prepared(
        tmp_path / "prepared.json",
        {"p1": (1.0, 0.8, 0.64), "p2": (100.0, 80.0, 64.0)},
    )
    run_dir = _train(tmp_path, name="run", prepared=prepared, processes=("p1",))
    return run_dir, prepared


def _params(run_dir: Path) -> Path:
    return run_dir / "model" / "params.eqx"


def _mkdir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


# ---------------------------------------------------------------------------
# one shared path: model loading and forward agree, and both use the recorded
# training selection rather than the data in front of them
# ---------------------------------------------------------------------------


def test_forward_omission_inherits_recorded_target_source(tmp_path: Path):
    collection = _collection_with({"p1": (1.0, 0.8, 0.64)})
    process = collection.processes["p1"]
    collection.processes["p1"] = replace(
        process,
        process_variables={
            "ratio": ProcessVariable(
                name="ratio",
                unit="-",
                is_controlled=False,
                values=TimeSeries(
                    times=jnp.asarray([0.0, 1.0, 2.0]),
                    values=jnp.asarray([0.0, 0.5, 1.0]),
                ),
            )
        },
    )
    prepared = tmp_path / "prepared.json"
    save_process_collection(collection, prepared)
    run_dir = _train(
        tmp_path,
        name="combined",
        prepared=prepared,
        targets=("biomass", "ratio"),
        target_source="combined",
    )

    inherited = forward_from_collection(
        collection,
        model_path=_params(run_dir),
        prediction_process_names=(),
    )
    assert inherited.store.name_measured_RMCs == ("biomass",)
    assert inherited.store.name_measured_PVs == ("ratio",)

    with pytest.raises(ValueError, match="target_source='auto' could not resolve"):
        forward_from_collection(
            collection,
            model_path=_params(run_dir),
            config=ForwardConfig(target_source=TARGET_SOURCE_AUTO),
            prediction_process_names=(),
        )


def test_model_load_and_forward_share_selected_training_reconstruction(two_level_run):
    run_dir, prepared = two_level_run
    evaluation = load_process_collection(prepared)

    loaded, _config = hybrax.train.model_load(run_dir)
    result = forward_from_collection(
        evaluation,
        model_path=_params(run_dir),
        config=ForwardConfig(
            process_names=("p1", "p2"), target_source="reactor_components"
        ),
    )

    # The run recorded data.processes=["p1"], so only p1 fed the scale hook —
    # p2's peak of 100 must not appear even though it is in the store.
    np.testing.assert_allclose(_rmc_scale(loaded), [1.0])
    np.testing.assert_allclose(_rmc_scale(result.trained_wrapper), [1.0])
    # ... and the split labelling reports the recorded selection, not "nothing".
    assert result.training_process_names == ("p1",)
    assert set(result.per_process_total_loss) == {"p1", "p2"}

    # Teeth: the same reconstruction on the full selection DOES see p2, so the
    # assertions above would fail if either loader had widened the selection.
    widened = reconstruct_training(run_dir, training_process_names=("p1", "p2"))
    np.testing.assert_allclose(_rmc_scale(widened.template_wrapper), [100.0])


def test_template_bakes_the_first_recorded_training_processs_feed(tmp_path: Path):
    """The reference process is the recorded training selection's FIRST entry.

    That process supplies the deserialisation template's baked ``Cin`` (and its
    controls). Give the two training processes different feed compositions and
    the choice becomes a readable number, so reordering — or otherwise
    re-deriving — the selection can no longer pass unnoticed.
    """
    prepared = _write_prepared(
        tmp_path / "prepared.json",
        {"p1": (1.0, 0.8, 0.64), "p2": (2.0, 1.6, 1.28)},
        feed_cin={"p1": 3.0, "p2": 40.0},
        feed_amount={"p1": 0.1, "p2": 0.8},
    )
    run_dir = _train(tmp_path, name="feed", prepared=prepared, processes=("p1", "p2"))

    rebuilt = reconstruct_training(run_dir)
    # The store carries both rows; the template must bake p1's, not p2's.
    np.testing.assert_allclose(
        np.asarray(rebuilt.store.Cin_controlled_Inflows), [[[3.0]], [[40.0]]]
    )
    np.testing.assert_allclose(
        np.asarray(rebuilt.template_wrapper.rhs_ode.Cin_controlled_Inflows), [[3.0]]
    )
    np.testing.assert_allclose(
        rebuilt.template_wrapper.controls.eval_controlled_Inflows_cumulative(
            jnp.asarray(2.0), None
        ),
        [0.1],
    )
    assert rebuilt.training_process_names == ("p1", "p2")

    # Teeth: the same reconstruction with the selection ordered the other way
    # bakes the other feed, so the assertion above is about *which* process.
    reversed_selection = reconstruct_training(
        run_dir, training_process_names=("p2", "p1")
    )
    np.testing.assert_allclose(
        np.asarray(reversed_selection.template_wrapper.rhs_ode.Cin_controlled_Inflows),
        [[40.0]],
    )
    np.testing.assert_allclose(
        reversed_selection.template_wrapper.controls.eval_controlled_Inflows_cumulative(
            jnp.asarray(2.0), None
        ),
        [0.8],
    )

    # Both loaders resolve the same reference process from the same record.
    loaded, _config = hybrax.train.model_load(run_dir)
    np.testing.assert_allclose(
        np.asarray(loaded.rhs_ode.Cin_controlled_Inflows), [[3.0]]
    )
    np.testing.assert_allclose(
        loaded.controls.eval_controlled_Inflows_cumulative(jnp.asarray(2.0), None),
        [0.1],
    )
    result = forward_from_collection(
        load_process_collection(prepared),
        model_path=_params(run_dir),
        config=ForwardConfig(
            process_names=("p1", "p2"), target_source="reactor_components"
        ),
    )
    assert result.training_process_names == ("p1", "p2")
    np.testing.assert_allclose(
        np.asarray(result.trained_wrapper.rhs_ode.Cin_controlled_Inflows), [[3.0]]
    )
    np.testing.assert_allclose(
        result.trained_wrapper.controls.eval_controlled_Inflows_cumulative(
            jnp.asarray(2.0), None
        ),
        [0.1],
    )


def test_forward_never_rescales_against_the_evaluation_collection(
    two_level_run, tmp_path
):
    """Evaluation values cannot reach a constructor hook, however extreme."""
    run_dir, _prepared = two_level_run
    inflated = _collection_with({"p1": (1.0, 0.8, 0.64), "p2": (5.0e4, 4.0e4, 3.2e4)})

    result = forward_from_collection(
        inflated,
        model_path=_params(run_dir),
        config=ForwardConfig(
            process_names=("p1", "p2"), target_source="reactor_components"
        ),
    )

    np.testing.assert_allclose(_rmc_scale(result.trained_wrapper), [1.0])


def test_forward_evaluates_a_collection_the_model_never_saw(two_level_run):
    """Regression: a genuinely different evaluation collection must work.

    The model's recorded training names ("p1") do not exist in this collection.
    Narrowing now happens on the model's *own* prepared input, so the evaluation
    collection is free to carry entirely different processes.
    """
    run_dir, _prepared = two_level_run
    novel = _collection_with({"q1": (2.0, 1.6, 1.28), "q2": (3.0, 2.4, 1.92)})

    result = forward_from_collection(
        novel,
        model_path=_params(run_dir),
        config=ForwardConfig(
            process_names=("q1", "q2"), target_source="reactor_components"
        ),
    )

    assert result.process_names == ("q1", "q2")
    assert result.training_process_names == ("p1",)
    assert all(np.isfinite(v) for v in result.per_process_total_loss.values())
    np.testing.assert_allclose(_rmc_scale(result.trained_wrapper), [1.0])


def test_forward_rejects_evaluation_data_the_wrapper_cannot_score(two_level_run):
    """An evaluation collection with different modeled species fails explicitly."""
    run_dir, _prepared = two_level_run
    base = _collection_with({"p1": (1.0, 0.8, 0.64)})
    process = base.processes["p1"]
    product = ReactorMediumComponent(
        name="product",
        unit="g/L",
        concentration=TimeSeries(
            times=jnp.asarray([0.0, 1.0, 2.0]),
            values=jnp.asarray([0.1, 0.2, 0.3]),
        ),
    )
    incompatible = BioProcessCollection(
        processes={
            "p1": replace(
                process,
                reactor_medium=replace(
                    process.reactor_medium,
                    components={
                        **process.reactor_medium.components,
                        "product": product,
                    },
                ),
            )
        },
        metadata={},
    )

    with pytest.raises(ValueError, match="incompatible with the data this model"):
        forward_from_collection(
            incompatible,
            model_path=_params(run_dir),
            config=ForwardConfig(
                process_names=("p1",), target_source="reactor_components"
            ),
        )


# ---------------------------------------------------------------------------
# artifact-backed LOO folds are ordinary loadable models
# ---------------------------------------------------------------------------


def test_loo_fold_models_load_through_the_shared_path(tmp_path: Path):
    """A real LOO run: every fold pins its input and loads with its own scales.

    Fold ``p1`` holds out p1 and therefore trains on p2 (peak 100.0); fold ``p2``
    trains on p1 (peak 1.0). Loading a fold must reproduce *that* fold's selection,
    which is only possible because the fold config records the producer-validated
    prepared hash the shared path requires.
    """
    prepared = _write_prepared(
        tmp_path / "prepared.json",
        {"p1": (1.0, 0.8, 0.64), "p2": (100.0, 80.0, 64.0)},
    )
    custom_py = tmp_path / "custom.py"
    custom_py.write_text(_CUSTOM_PY, encoding="utf-8")
    output_dir = tmp_path / "out"
    config_path = tmp_path / "loo-config.json"
    config_path.write_text(
        json.dumps(
            {
                "data": {
                    "prepared": str(prepared),
                    "targets": ["biomass"],
                    "target_source": "reactor_components",
                },
                "custom_py": str(custom_py),
                "train": {"epochs": 1, "learning_rate": 0.05, "seed": 3},
                "solver": {"max_steps": 2048, "rtol": 1e-4, "atol": 1e-6},
                "output": {"dir": str(output_dir), "predictions": "none"},
                "loo": {"parallel_folds": 1},
            }
        ),
        encoding="utf-8",
    )

    assert main(["loo", "--config", str(config_path)]) == 0

    expected_hash = content_hash(load_process_collection(prepared))
    for slug, trained_on, scale in (("p1", ("p2",), 100.0), ("p2", ("p1",), 1.0)):
        fold_dir = output_dir / "folds" / slug
        document = json.loads((fold_dir / "config.json").read_text())
        assert document["inputs"]["prepared_input"]["content_hash"] == expected_hash
        for target in (fold_dir, fold_dir / "checkpoints" / "latest"):
            wrapper, config = hybrax.train.model_load(target)
            assert config.data.processes == trained_on
            np.testing.assert_allclose(_rmc_scale(wrapper), [scale])


# ---------------------------------------------------------------------------
# the model_path contract: resolve or reject before paying for a reconstruction
# ---------------------------------------------------------------------------


def test_forward_accepts_a_run_directory_as_model_path(two_level_run):
    """A run dir is what every other loader takes; forward resolves it too."""
    run_dir, prepared = two_level_run
    evaluation = load_process_collection(prepared)
    cfg = ForwardConfig(process_names=("p1", "p2"), target_source="reactor_components")

    from_dir = forward_from_collection(evaluation, model_path=run_dir, config=cfg)
    from_file = forward_from_collection(
        evaluation, model_path=_params(run_dir), config=cfg
    )

    assert from_dir.per_process_total_loss == from_file.per_process_total_loss


@pytest.mark.parametrize(
    ("make_path", "message"),
    [
        (
            lambda run_dir, tmp_path: run_dir / "model" / "paramz.eqx",
            "model path does not exist",
        ),
        (lambda run_dir, tmp_path: tmp_path / "nowhere", "model path does not exist"),
        (
            lambda run_dir, tmp_path: _mkdir(tmp_path / "empty"),
            "no model weights in directory",
        ),
    ],
)
def test_forward_rejects_an_unusable_model_path_before_reconstructing(
    two_level_run, tmp_path, monkeypatch, make_path, message
):
    """A typo (or a weight-less directory) must not cost a full reconstruction,
    and must name the path the caller gave rather than one equinox invented by
    appending ``.eqx``."""
    run_dir, _prepared = two_level_run
    from hybrax.train import serialization

    monkeypatch.setattr(
        serialization,
        "reconstruct_training",
        lambda *_a, **_k: pytest.fail("reconstruction ran for an unusable model path"),
    )
    bad_path = make_path(run_dir, tmp_path)

    with pytest.raises(FileNotFoundError, match=message) as error:
        forward_from_collection(
            _collection_with({"p1": (1.0, 0.8, 0.64)}),
            model_path=bad_path,
            config=ForwardConfig(
                process_names=("p1",), target_source="reactor_components"
            ),
        )
    assert str(bad_path) in str(error.value)


# ---------------------------------------------------------------------------
# the hash gate: missing, tampered, or stale input fails BEFORE any hook
# ---------------------------------------------------------------------------


def _record_only_run_dir(tmp_path: Path) -> tuple[Path, Path]:
    """A run *record* — config.json, prepared input, empty weights; never trained.

    The hash gate must fire before the hooks and before the weights are read, so a
    model that could not possibly deserialise is exactly the right subject.
    """
    from hybrax.train.run_config import RunConfig
    from hybrax.train.serialization import run_config_to_jsonable

    run_dir = tmp_path / "record"
    (run_dir / "model").mkdir(parents=True)
    (run_dir / "model" / "params.eqx").write_bytes(b"")
    prepared = _write_prepared(
        tmp_path / "prepared.json",
        {"p1": (1.0, 0.8, 0.64), "p2": (100.0, 80.0, 64.0)},
    )
    (run_dir / "custom.py").write_text(_CUSTOM_PY, encoding="utf-8")
    config = RunConfig.model_validate(
        {
            "data": {
                "prepared": str(prepared),
                "processes": ["p1"],
                "targets": ["biomass"],
                "target_source": "reactor_components",
            },
            "output": {"dir": str(run_dir)},
        }
    )
    (run_dir / "config.json").write_text(
        json.dumps(
            {
                "status": "complete",
                "config": run_config_to_jsonable(config),
                "inputs": {
                    "prepared_input": {
                        "path": str(prepared),
                        "content_hash": content_hash(load_process_collection(prepared)),
                    }
                },
            }
        ),
        encoding="utf-8",
    )
    return run_dir, prepared


def _omit_hash(run_dir: Path, prepared: Path) -> None:
    document = json.loads((run_dir / "config.json").read_text())
    document.pop("inputs")
    (run_dir / "config.json").write_text(json.dumps(document), encoding="utf-8")


def _tamper_hash(run_dir: Path, prepared: Path) -> None:
    document = json.loads((run_dir / "config.json").read_text())
    document["inputs"]["prepared_input"]["content_hash"] = "sha256:" + "0" * 64
    (run_dir / "config.json").write_text(json.dumps(document), encoding="utf-8")


def _tamper_data(run_dir: Path, prepared: Path) -> None:
    save_process_collection(
        _collection_with({"p1": (7.0, 6.0, 5.0), "p2": (100.0, 80.0, 64.0)}), prepared
    )


@pytest.mark.parametrize(
    ("break_it", "message"),
    [
        (_omit_hash, "records no inputs.prepared_input.content_hash"),
        (_tamper_hash, "differs from the run's record"),
        (_tamper_data, "differs from the run's record"),
    ],
)
def test_loading_a_model_without_verified_input_fails_before_any_hook(
    tmp_path: Path, monkeypatch, break_it, message
):
    run_dir, prepared = _record_only_run_dir(tmp_path)
    break_it(run_dir, prepared)
    monkeypatch.setattr(
        harness,
        "_build_runtime_modules",
        lambda **_kwargs: pytest.fail("a hook ran on unverified training input"),
    )

    with pytest.raises(ValueError, match=message):
        hybrax.train.model_load(run_dir)

    with pytest.raises(ValueError, match=message):
        forward_from_collection(
            _collection_with({"p1": (1.0, 0.8, 0.64)}),
            model_path=_params(run_dir),
            config=ForwardConfig(
                process_names=("p1",), target_source="reactor_components"
            ),
        )


# ---------------------------------------------------------------------------
# ensembles: distinct per-model inputs
# ---------------------------------------------------------------------------


def test_ensemble_forward_rebuilds_each_model_from_its_own_input(tmp_path: Path):
    """Two models with different training inputs, one shared evaluation input."""
    prepared_a = _write_prepared(
        tmp_path / "a.json", {"p1": (1.0, 0.8, 0.64), "p2": (100.0, 80.0, 64.0)}
    )
    prepared_b = _write_prepared(
        tmp_path / "b.json", {"p1": (7.0, 5.6, 4.5), "p2": (100.0, 80.0, 64.0)}
    )
    run_a = _train(tmp_path, name="a", prepared=prepared_a, processes=("p1",))
    run_b = _train(tmp_path, name="b", prepared=prepared_b, processes=("p1",))

    evaluation = load_process_collection(prepared_a)
    results = [
        forward_from_collection(
            evaluation,
            model_path=_params(run_dir),
            config=ForwardConfig(
                process_names=("p1", "p2"), target_source="reactor_components"
            ),
        )
        for run_dir in (run_a, run_b)
    ]

    # Each member kept its own scales; the shared evaluation input changed neither.
    np.testing.assert_allclose(_rmc_scale(results[0].trained_wrapper), [1.0])
    np.testing.assert_allclose(_rmc_scale(results[1].trained_wrapper), [7.0])

    # ... and the same two members run as an ensemble through the CLI.
    forward_config = tmp_path / "forward-config.json"
    forward_config.write_text(
        json.dumps(
            {
                "models": [str(run_a), str(run_b)],
                "data": {"prepared": str(prepared_a), "processes": ["p1", "p2"]},
                "output": {"dir": str(tmp_path / "fwd"), "predictions": "none"},
            }
        ),
        encoding="utf-8",
    )
    assert main(["forward", "--config", str(forward_config)]) == 0
    output_dir = tmp_path / "fwd"
    assert (output_dir / "losses.csv").is_file()
    assert len(list((output_dir / "models").glob("*/losses.csv"))) == 2
