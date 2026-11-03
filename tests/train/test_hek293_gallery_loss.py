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


def test_hek_nonneg_penalty_uses_the_fit_max_normalization():
    custom = load_example("custom_nonneg")
    # Bound records follow the ODE's species order; look them up by name.
    records = (("lwr_bnd/B", "state", 1, 1.0, 0.0), ("lwr_bnd/A", "state", 0, 1.0, 0.0))
    # Distinct SD / max ratios, so mixing up the bound order changes the result.
    ratio = jnp.array([2.0, 4.0]) / jnp.array([10.0, 80.0])
    loss = custom.MatlabLoss(("A", "B"), ratio, records, weight=10.0)
    inputs = inputs_for([[2, 4, 1], [-2, 4, 1], [2, -8, 1]])
    named = eqx.filter_jit(loss)(inputs).named_losses
    # One undershoot each, divided by the maxima 10 and 80, squared, averaged over
    # the three measurement times and weighted by 10.
    assert float(named["lwr_bnd/A"]) == pytest.approx(10 * (2 / 10) ** 2 / 3)
    assert float(named["lwr_bnd/B"]) == pytest.approx(10 * (8 / 80) ** 2 / 3)


def write_metrics(run, holdout):
    """Write a two-target metrics.csv with penalty terms and Br7 evaluations."""
    run.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        {
            "epoch": [1, 2, 3, 4],
            "mean_loss": [0.4, 0.3, 0.2, 0.1],
            "per_target_loss": ["0.4;0.2;0;0", "0.3;0.1;0;0", "0.2;0;0;1", "0;0;0;0"],
            "target_names": ["A;B;lwr_bnd/A;lwr_bnd/B"] * 4,
            "holdout_loss": [np.mean(row) if row else np.nan for row in holdout],
            "holdout_per_target_loss": [
                ";".join(map(str, row)) if row else "" for row in holdout
            ],
        }
    ).to_csv(run / "metrics.csv", index=False)


def test_hek_results_split_fit_and_penalty_and_select_by_full_loss(tmp_path):
    results = load_example("results")
    run = tmp_path / "run"
    (run / "checkpoints/step_00004").mkdir(parents=True)
    # Epoch 2 has the best fit, epoch 4 the best fit + penalty.
    write_metrics(run, [[], [0.1, 0.1, 0.4, 0.4], [np.inf] * 4, [0.2, 0.2, 0.0, 0.0]])
    curves = results.read_loss_curves(tmp_path)
    np.testing.assert_allclose(curves["Train NMSE"], [0.3, 0.2, 0.1, 0.0])
    np.testing.assert_allclose(curves["Br7 NMSE"], [np.nan, 0.1, np.inf, 0.2])
    np.testing.assert_allclose(curves["Br7 penalty"], [np.nan, 0.4, np.inf, 0.0])
    assert results.select_checkpoint(tmp_path) == 4
    write_metrics(run, [[], [np.inf] * 4, [np.inf] * 4, [np.inf] * 4])
    with pytest.raises(AssertionError, match="No successful Br7"):
        results.read_loss_curves(tmp_path)


def test_hek_scores_pool_splits_at_measurement_times(tmp_path, monkeypatch):
    results = load_example("results")
    truth = np.tile([2.0, 10.0], (3, 1))
    times = np.array([0.0, 12.0, 24.0])
    source = [SimpleNamespace(conc=truth.T, time=times) for _ in range(9)]
    reference = np.stack([truth + [2.0, 10.0]] * 9)
    monkeypatch.setattr(
        results, "load_benchmark", lambda _: (source, reference, np.array([2.0, 10.0]))
    )
    errors = {"Br5": 10.0, "Br6": 20.0, "Br8": 40.0, "Br7": 20.0}
    dense = {}
    for name in results.CULTURES:
        values = truth + ([0.0, errors[name]] if name in errors else [-2.0, 0.0])
        values[0] += 10000  # The initial condition must not affect any score.
        # An extra dense row between samples must not affect any score either.
        values = np.insert(values, 2, [-1000.0, 1000.0], axis=0)
        dense[name] = (np.array([0.0, 12.0, 18.0, 24.0]), values)
    scores = results.score(tmp_path, {"Hybrax": dense}, "scores.csv", True)
    hybrax = scores.loc["Hybrax"]
    assert hybrax["NRMSE", "Train"] == pytest.approx(np.sqrt(0.5))
    assert hybrax["NRMSE", "Validation (Br7)"] == pytest.approx(np.sqrt(2.0))
    # Pool errors over cultures before taking the root: (1² + 2² + 4²) / (3 * 2).
    assert hybrax["NRMSE", "Test"] == pytest.approx(np.sqrt(3.5))
    np.testing.assert_allclose(scores.loc["MATLAB", "NRMSE"], 1.0)
    # Training predicts A = 0 at both post-initial samples: no undershoot.
    assert hybrax["Undershoot [%]", "Train"] == 0.0
    dense["Br1"][1][1, 0] = -1.0  # 50% of A's maximum, one of four values.
    scores = results.score(tmp_path, {"Hybrax": dense}, "scores.csv", True)
    assert scores.loc["Hybrax", ("Undershoot [%]", "Train")] == pytest.approx(
        50 / (5 * 2 * 2)
    )
