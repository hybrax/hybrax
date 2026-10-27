"""HEK loss preserves MATLAB normalization and finite checkpoint selection."""

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd
import pytest

from hybrax.train import AffineScaler, LossInputs, RateModule

EXAMPLE = Path(__file__).resolve().parents[2] / "examples/gallery_hek293_lstm"


def load_example(name):
    spec = importlib.util.spec_from_file_location(f"hek_{name}", EXAMPLE / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def loss():
    return load_example("custom").build_loss_module(
        target_names=("A", "B"),
        config=SimpleNamespace(
            custom=SimpleNamespace(
                state_names=["A", "B"],
                feature_spread=[2.0, 4.0],
                loss_maxima=[10.0, 20.0],
            )
        ),
    )


def inputs_for(raw):
    raw = jnp.asarray(raw, dtype=float)
    offset = jnp.array([10.0, 20.0, 1.0])
    spread = jnp.array([2.0, 4.0, 1.0])
    # An initial mismatch must be excluded; later measured zeros remain active.
    measured = jnp.array([[20.0, 40.0], [0.0, 0.0], [0.0, 0.0]])
    empty = jnp.zeros((3, 0))
    return LossInputs(
        SCL_states=(raw - offset) / spread,
        RAW_states=raw,
        SCL_modeled_ReactionOde_rates=empty,
        RAW_modeled_ReactionOde_rates=empty,
        SCL_modeled_Inflows_rates=empty,
        RAW_modeled_Inflows_rates=empty,
        SCL_modeled_Outflows_rates=empty,
        RAW_modeled_Outflows_rates=empty,
        SCL_V=jnp.zeros(3),
        RAW_V=raw[:, 2],
        RAW_V_unclamped=raw[:, 2],
        auxiliary={},
        SCL_target_pred=(raw[:, :2] - offset[:2]) / spread[:2],
        SCL_target_measured=(measured - offset[:2]) / spread[:2],
        mask_measured=jnp.ones((3, 2), dtype=bool),
        mask_measured_any=jnp.ones(3),
        t_measured=jnp.arange(3, dtype=float),
        n_measured=jnp.array(3),
        reaction_module=RateModule(
            SCALE_modeled_RMCs=AffineScaler(spread[:2], offset[:2]),
            SCALE_V_in_cumulative=AffineScaler(spread[2], offset[2]),
        ),
        step=jnp.array(0),
    )


def test_hek_loss_max_normalizes_and_excludes_only_initial_time(loss):
    inputs = inputs_for([[2, 4, 1], [-2, 4, 1], [2, -4, 1]])
    result = eqx.filter_jit(loss)(inputs).named_losses
    # Post-initial physical errors are ±2 and ±4; maxima are 10 and 20.
    assert set(result) == {"A", "B"}
    assert float(result["A"]) == pytest.approx(0.04)
    assert float(result["B"]) == pytest.approx(0.04)


def test_hek_loss_masks_invalid_measurements_and_has_restoring_gradient(loss):
    inputs = inputs_for([[2, 4, 1], [-2, 4, 1], [200, -400, 1]])
    masked = eqx.tree_at(
        lambda x: x.mask_measured, inputs, inputs.mask_measured.at[2].set(False)
    )
    result = eqx.filter_jit(loss)(masked).named_losses
    assert float(result["A"]) == pytest.approx(0.04)
    assert float(result["B"]) == pytest.approx(0.04)

    def fit(value):
        changed = eqx.tree_at(
            lambda x: x.SCL_target_pred,
            masked,
            masked.SCL_target_pred.at[1, 0].set(value),
        )
        return loss(changed).named_losses["A"]

    # d[(2 * z + 10)^2 / 10^2]/dz at physical prediction -2.
    assert float(jax.grad(fit)(-6.0)) == pytest.approx(-0.08)


def test_hek_results_select_finite_validation_nmse(tmp_path):
    results = load_example("results")
    run = tmp_path / "run"
    (run / "checkpoints/step_00030").mkdir(parents=True)
    metrics = pd.DataFrame(
        {
            "epoch": [1, 10, 20, 30],
            "mean_loss": [0.4, 0.3, 0.2, 0.1],
            "holdout_loss": [np.nan, np.inf, 0.2, 0.1],
            "target_names": ["A;B"] * 4,
        }
    )
    metrics.to_csv(run / "metrics.csv", index=False)
    curves = results.evaluate_checkpoints(tmp_path)
    assert list(curves.columns) == ["Train NMSE", "Br7 NMSE"]
    assert results.select_checkpoint(tmp_path, curves) == 30
    metrics.holdout_loss = [np.nan, np.inf, np.inf, np.inf]
    metrics.to_csv(run / "metrics.csv", index=False)
    with pytest.raises(AssertionError, match="No successful Br7"):
        results.evaluate_checkpoints(tmp_path)


def test_hek_comparison_scores_pool_splits_and_reuse_saved_validation(
    tmp_path, monkeypatch
):
    results = load_example("results")
    truth = np.tile([2.0, 10.0], (3, 1))
    times = np.array([0.0, 12.0, 24.0])
    source = [SimpleNamespace(conc=truth.T, time=times) for _ in range(9)]
    reference = np.stack([truth + [2.0, 10.0]] * 9)
    monkeypatch.setattr(
        results, "load_benchmark", lambda _: (source, reference, np.array([2.0, 10.0]))
    )
    wrapper = SimpleNamespace(modeled_RMC_names=["B", "A"])
    config = SimpleNamespace(custom=SimpleNamespace(state_names=["A", "B"]))
    monkeypatch.setattr(results.hxt, "model_load", lambda _: (wrapper, config))
    monkeypatch.setattr(
        results.hxf.serialization, "load_process_collection", lambda _: source
    )
    predictions = {}
    test_errors = {"Br5": 10.0, "Br6": 20.0, "Br8": 40.0}
    for name in ("Br1", "Br2", "Br3", "Br4", "Br9", "Br5", "Br6", "Br8"):
        error = [0.0, test_errors[name]] if name in test_errors else [2.0, 0.0]
        values = truth + error
        values[0] += 10000  # The initial condition must not affect any score.
        # Include an extra event-time row; scoring must select measurement times.
        values = np.insert(values, 2, [1000.0, 1000.0], axis=0)
        predictions[name] = SimpleNamespace(
            t=np.array([0.0, 12.0, 12.0, 24.0]),
            c_species=values[:, ::-1],
            prediction_valid=np.ones(4, dtype=bool),
        )

    def predict(model, cfg, collection, *, process_names, grid_n):
        assert model is wrapper and cfg is config and collection is source
        assert process_names == tuple(predictions)
        assert grid_n == 21
        return predictions

    monkeypatch.setattr(results.hxt, "model_predict", predict)
    checkpoint = tmp_path / "run/checkpoints/step_00030"
    checkpoint.mkdir(parents=True)
    validation = truth + [2.0, 20.0]
    validation[0] += 10000
    pd.DataFrame(
        {
            "process": ["Br7"] * 3,
            "t": times,
            "prediction_valid": [True] * 3,
            "c_A": validation[:, 0],
            "c_B": validation[:, 1],
        }
    ).to_csv(checkpoint / "holdout_predictions.csv", index=False)
    scores = results.score_selected_checkpoint(tmp_path, 30)
    assert scores.loc["Training", "Hybrax"] == pytest.approx(np.sqrt(0.5))
    assert scores.loc["Validation (Br7)", "Hybrax"] == pytest.approx(np.sqrt(2.5))
    # Pool errors over cultures before taking the root: (1² + 2² + 4²) / (3 * 2).
    assert scores.loc["Test", "Hybrax"] == pytest.approx(np.sqrt(3.5))
    np.testing.assert_allclose(scores.MATLAB, 1.0)
    pd.testing.assert_frame_equal(
        pd.read_csv(tmp_path / "comparison-scores.csv", index_col=0), scores
    )
    predictions["Br5"].prediction_valid[-1] = False
    with pytest.raises(AssertionError, match="Br5"):
        results.score_selected_checkpoint(tmp_path, 30)
