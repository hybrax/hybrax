from types import SimpleNamespace

import equinox as eqx
import jax
import jax.numpy as jnp
import pytest
from bp_format.dataclasses import (
    BioProcess,
    BioProcessCollection,
    BioProcessMetadata,
    BiologicalOde,
    ProcessVariable,
    ReactorMedium,
    ReactorMediumComponent,
    StaticVariable,
    TimeAxis,
    TimeSeries,
    Volume,
)

from bp_train import BoundsViolationLossModule, DefaultLossModule
from bp_train.model_api import AffineScaler, LinearScaler, LossInputs
from bp_train.runtime_context import RuntimeDataContext
from bp_train.training_data import TrainingDataStore


NO_BOUNDS = (None, None)


def _series(values=(1.0, 1.0)):
    return TimeSeries(times=jnp.asarray([0.0, 1.0]), values=jnp.asarray(values))


def _process(
    name,
    *,
    rmc_name="biomass",
    rmc_bounds=NO_BOUNDS,
    oxygen_bounds=NO_BOUNDS,
    volume_bounds=NO_BOUNDS,
    rate_bounds=(NO_BOUNDS, NO_BOUNDS),
):
    return BioProcess(
        metadata=BioProcessMetadata(name=name, process_type="batch"),
        time_axis=TimeAxis(unit="h", start=0.0, end=1.0, time_reference="start"),
        reactor_medium=ReactorMedium(
            name="medium",
            density=1.0,
            density_unit="kg/L",
            components={
                rmc_name: ReactorMediumComponent(
                    name=rmc_name,
                    unit="g/L",
                    concentration=_series(),
                    bounds=rmc_bounds,
                )
            },
        ),
        process_variables={
            "oxygen": ProcessVariable(
                name="oxygen",
                unit="%",
                is_controlled=False,
                values=_series(),
                bounds=oxygen_bounds,
            ),
            "temperature": ProcessVariable(
                name="temperature",
                unit="K",
                is_controlled=True,
                values=StaticVariable(300.0),
                bounds=(290.0, 310.0),
            ),
        },
        volume=Volume(initial_volume=1.0, unit="L", bounds=volume_bounds),
        biological_ode=BiologicalOde(
            rates={f"q_{rmc_name}": rate_bounds[0], "r_oxygen": rate_bounds[1]},
            derivatives={
                rmc_name: f"q_{rmc_name} * {rmc_name}",
                "oxygen": "r_oxygen",
            },
        ),
    )


def _collection(**kwargs):
    return BioProcessCollection(processes={"p1": _process("p1", **kwargs)})


def _bound_snapshots(collection):
    if not collection.processes:
        raise ValueError("runtime context requires a non-empty collection")
    target_name = next(
        iter(next(iter(collection.processes.values())).reactor_medium.components)
    )
    store = TrainingDataStore.from_collection(
        collection,
        target_variable_order=[target_name],
        target_source="reactor_components",
    )
    return RuntimeDataContext.from_collection(store, collection).bound_snapshots


def _inputs(
    *,
    raw_states,
    raw_rates,
    mask_any,
    scl_pred=None,
    raw_v_unclamped=None,
    t_measured=None,
    dense_t=None,
    dense_raw_states=None,
    dense_raw_rates=None,
    dense_raw_v=None,
    dense_raw_v_unclamped=None,
    dense_valid_time=None,
):
    raw_states = jnp.asarray(raw_states)
    raw_rates = jnp.asarray(raw_rates)
    n_rows, n_states = raw_states.shape
    if scl_pred is None:
        scl_pred = jnp.zeros((n_rows, n_states))
    scl_pred = jnp.asarray(scl_pred)
    mask = jnp.broadcast_to(jnp.asarray(mask_any)[:, None].astype(bool), scl_pred.shape)
    zeros_states = jnp.zeros_like(raw_states)
    zeros_rates = jnp.zeros_like(raw_rates)
    if raw_v_unclamped is None:
        raw_v_unclamped = raw_states[:, 2]
    if t_measured is None:
        t_measured = jnp.arange(n_rows, dtype=float)
    dense_raw_states = (
        None if dense_raw_states is None else jnp.asarray(dense_raw_states)
    )
    dense_raw_rates = None if dense_raw_rates is None else jnp.asarray(dense_raw_rates)
    if dense_raw_v is None and dense_raw_states is not None:
        dense_raw_v = dense_raw_states[:, 2]
    if dense_raw_v_unclamped is None:
        dense_raw_v_unclamped = dense_raw_v
    if dense_raw_v_unclamped is not None:
        dense_raw_v_unclamped = jnp.asarray(dense_raw_v_unclamped)
    return LossInputs(
        SCL_states=zeros_states,
        RAW_states=raw_states,
        SCL_modeled_BiologicalOde_rates=zeros_rates,
        RAW_modeled_BiologicalOde_rates=raw_rates,
        SCL_modeled_FVCs_rates=jnp.zeros((n_rows, 0)),
        RAW_modeled_FVCs_rates=jnp.zeros((n_rows, 0)),
        SCL_V=zeros_states[:, 2],
        RAW_V=raw_states[:, 2],
        RAW_V_unclamped=jnp.asarray(raw_v_unclamped),
        auxiliary={},
        SCL_target_pred=scl_pred,
        SCL_target_measured=jnp.zeros_like(scl_pred),
        mask_measured=mask,
        mask_measured_any=jnp.asarray(mask_any),
        t_measured=jnp.asarray(t_measured),
        n_measured=jnp.sum(jnp.asarray(mask_any, dtype=jnp.int32)),
        dense_t=None if dense_t is None else jnp.asarray(dense_t),
        dense_RAW_states=dense_raw_states,
        dense_RAW_modeled_BiologicalOde_rates=dense_raw_rates,
        dense_RAW_V=dense_raw_v,
        dense_RAW_V_unclamped=dense_raw_v_unclamped,
        dense_valid_time=(
            None if dense_valid_time is None else jnp.asarray(dense_valid_time)
        ),
        reaction_module=SimpleNamespace(
            SCALE_state=AffineScaler(
                scale=jnp.asarray([2.0, 4.0, 8.0]),
                offset=jnp.asarray([100.0, -30.0, 20.0]),
            ),
            SCALE_modeled_BiologicalOde_rates=LinearScaler(jnp.asarray([0.5, 2.0])),
        ),
        step=jnp.asarray(0),
    )


def test_bounds_loss_is_raw_affine_safe_masked_and_named():
    module = BoundsViolationLossModule(
        target_names=("biomass", "oxygen", "V"),
        bound_snapshots=_bound_snapshots(
            _collection(
                rmc_bounds=(0.0, 10.0),
                oxygen_bounds=(1.0, None),
                volume_bounds=(0.5, 2.0),
                rate_bounds=((-1.0, 1.0), (None, 3.0)),
            )
        ),
        weight=0.5,
    )
    inputs = _inputs(
        raw_states=[[-2.0, 0.0, 3.0], [12.0, 2.0, 0.0], [-100.0, -100.0, 100.0]],
        raw_rates=[[-3.0, 5.0], [2.0, 1.0], [-100.0, 100.0]],
        mask_any=[1.0, 1.0, 0.0],
        scl_pred=[[1.0, 2.0, 3.0], [3.0, 4.0, 5.0], [100.0, 100.0, 100.0]],
    )

    assert module.loss_names == (
        "biomass",
        "oxygen",
        "V",
        "lwr_bnd/biomass",
        "upr_bnd/biomass",
        "lwr_bnd/oxygen",
        "lwr_bnd/V",
        "upr_bnd/V",
        "lwr_bnd/rate/q_biomass",
        "upr_bnd/rate/q_biomass",
        "upr_bnd/rate/r_oxygen",
    )
    losses = module(inputs).named_losses
    expected = {
        "biomass": 5.0,
        "oxygen": 10.0,
        "V": 17.0,
        "lwr_bnd/biomass": 0.25,
        "upr_bnd/biomass": 0.25,
        "lwr_bnd/oxygen": 0.015625,
        "lwr_bnd/V": 0.0009765625,
        "upr_bnd/V": 0.00390625,
        "lwr_bnd/rate/q_biomass": 4.0,
        "upr_bnd/rate/q_biomass": 1.0,
        "upr_bnd/rate/r_oxygen": 0.25,
    }
    assert losses.keys() == expected.keys()
    for name, value in expected.items():
        assert float(losses[name]) == pytest.approx(value)


def test_volume_bound_uses_unclamped_integrated_volume():
    module = BoundsViolationLossModule(
        target_names=("biomass", "oxygen", "V"),
        bound_snapshots=_bound_snapshots(_collection(volume_bounds=(0.5, None))),
        weight=1.0,
    )
    inputs = _inputs(
        raw_states=[[1.0, 2.0, 1e-8]],
        raw_rates=[[0.0, 0.0]],
        mask_any=[1.0],
        raw_v_unclamped=[-2.0],
    )

    loss = module(inputs).named_losses["lwr_bnd/V"]

    assert float(loss) == pytest.approx(((0.5 - -2.0) / 8.0) ** 2)


def test_rmc_named_v_does_not_collide_with_volume_bound():
    collection = _collection(
        rmc_name="V",
        rmc_bounds=(0.0, None),
        volume_bounds=(1.0, None),
    )
    module = BoundsViolationLossModule(
        target_names=("V", "oxygen", "reactor_volume"),
        bound_snapshots=_bound_snapshots(collection),
        weight=1.0,
    )
    inputs = _inputs(
        raw_states=[[-2.0, 2.0, 0.0]],
        raw_rates=[[0.0, 0.0]],
        mask_any=[1.0],
    )

    losses = module(inputs).named_losses

    assert "lwr_bnd/V" in losses
    assert "lwr_bnd/volume/V" in losses
    assert float(losses["lwr_bnd/V"]) == pytest.approx(1.0)
    assert float(losses["lwr_bnd/volume/V"]) == pytest.approx(0.015625)


def test_bound_name_collision_with_reconstruction_target_is_rejected():
    process = _process("p1", rmc_name="x", rmc_bounds=(0.0, None))
    process.reactor_medium.components["lwr_bnd/x"] = ReactorMediumComponent(
        name="lwr_bnd/x",
        unit="g/L",
        concentration=_series(),
    )
    process.biological_ode.derivatives["lwr_bnd/x"] = "0"

    with pytest.raises(ValueError, match="loss names must be unique"):
        BoundsViolationLossModule(
            target_names=("x", "lwr_bnd/x"),
            bound_snapshots=_bound_snapshots(
                BioProcessCollection(processes={"p1": process})
            ),
            weight=1.0,
        )


def test_unbounded_collection_is_default_loss_noop():
    target_names = ("biomass", "oxygen", "V")
    inputs = _inputs(
        raw_states=[[1.0, 2.0, 1.0]],
        raw_rates=[[0.0, 0.0]],
        mask_any=[1.0],
        scl_pred=[[1.0, 2.0, 3.0]],
    )
    module = BoundsViolationLossModule(
        target_names=target_names,
        bound_snapshots=_bound_snapshots(_collection()),
        weight=1.0,
    )

    losses = module(inputs).named_losses
    expected = DefaultLossModule(target_names=target_names)(inputs).named_losses

    assert module.dense_grid_n is None
    assert module.loss_names == target_names
    assert losses.keys() == expected.keys()
    for name in target_names:
        assert float(losses[name]) == pytest.approx(float(expected[name]))


def test_subclass_can_initialize_loss_name_fields_after_super():
    class ExtendedBoundsLoss(BoundsViolationLossModule):
        extra_names: tuple[str, ...] = eqx.field(static=True)

        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            self.extra_names = ("extra",)

        @property
        def loss_names(self):
            return super().loss_names + self.extra_names

    module = ExtendedBoundsLoss(
        target_names=("biomass",),
        bound_snapshots=_bound_snapshots(_collection()),
        weight=1.0,
    )
    assert module.loss_names == ("biomass", "extra")


def test_zero_weight_keeps_stable_zero_terms():
    target_names = ("biomass", "oxygen", "V")
    module = BoundsViolationLossModule(
        target_names=target_names,
        bound_snapshots=_bound_snapshots(_collection(rmc_bounds=(0.0, None))),
        weight=0.0,
    )
    inputs = _inputs(
        raw_states=[[-1.0, 2.0, 1.0]],
        raw_rates=[[0.0, 0.0]],
        mask_any=[1.0],
    )

    assert module.loss_names == target_names + ("lwr_bnd/biomass",)
    assert float(module(inputs).named_losses["lwr_bnd/biomass"]) == 0.0


def test_dense_bounds_use_deduplicated_union_and_derivative_scales():
    collection = _collection(
        rmc_bounds=(0.0, None),
        oxygen_bounds=(None, 10.0),
        volume_bounds=(1.0, None),
        rate_bounds=((0.0, None), (None, 2.0)),
    )
    module = BoundsViolationLossModule(
        target_names=("biomass",),
        bound_snapshots=_bound_snapshots(collection),
        weight=1.0,
        dense_grid_n=3,
    )
    measurement_module = BoundsViolationLossModule(
        target_names=("biomass",),
        bound_snapshots=_bound_snapshots(collection),
        weight=1.0,
    )
    inputs = _inputs(
        raw_states=[[0.0, 10.0, 1.0], [0.0, 10.0, 1.0]],
        raw_rates=[[0.0, 2.0], [0.0, 2.0]],
        mask_any=[True, True],
        scl_pred=[[2.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
        t_measured=[0.0, 1.0],
        dense_t=[0.0, 0.5, 1.0],
        dense_raw_states=[[0.0, 10.0, 1.0], [-4.0, 18.0, 0.0], [0.0, 10.0, 1.0]],
        dense_raw_rates=[[0.0, 2.0], [-1.0, 6.0], [0.0, 2.0]],
        dense_valid_time=[True, True, True],
    )

    losses = module(inputs).named_losses
    measurement_losses = measurement_module(inputs).named_losses

    assert module.dense_grid_n == 3
    assert module.loss_names == measurement_module.loss_names
    assert losses.keys() == measurement_losses.keys()
    assert losses["biomass"] == pytest.approx(measurement_losses["biomass"])
    assert losses["biomass"] == pytest.approx(2.0)
    for name in (
        "lwr_bnd/biomass",
        "upr_bnd/oxygen",
        "lwr_bnd/rate/q_biomass",
        "upr_bnd/rate/r_oxygen",
    ):
        assert losses[name] == pytest.approx(4.0 / 3.0)
    assert losses["lwr_bnd/V"] == pytest.approx(1.0 / 192.0)


def test_dense_bounds_mask_failed_rows_and_keep_gradients_finite():
    module = BoundsViolationLossModule(
        target_names=("biomass",),
        bound_snapshots=_bound_snapshots(_collection(rmc_bounds=(0.0, None))),
        weight=1.0,
        dense_grid_n=3,
    )

    def loss(dense_biomass):
        dense_states = jnp.column_stack((dense_biomass, jnp.zeros(3), jnp.ones(3)))
        inputs = _inputs(
            raw_states=[[0.0, 0.0, 1.0], [0.0, 0.0, 1.0]],
            raw_rates=jnp.zeros((2, 2)),
            mask_any=[True, True],
            t_measured=[0.0, 1.0],
            dense_t=[0.0, 0.5, 1.0],
            dense_raw_states=dense_states,
            dense_raw_rates=jnp.zeros((3, 2)),
            dense_valid_time=[True, True, False],
        )
        return module(inputs).named_losses["lwr_bnd/biomass"]

    dense_biomass = jnp.asarray([0.0, -4.0, -1e6])
    assert loss(dense_biomass) == pytest.approx(2.0)
    assert jnp.all(jnp.isfinite(jax.grad(loss)(dense_biomass)))


def test_dense_volume_bound_uses_unclamped_volume():
    module = BoundsViolationLossModule(
        target_names=("biomass",),
        bound_snapshots=_bound_snapshots(_collection(volume_bounds=(1.0, None))),
        weight=1.0,
        dense_grid_n=3,
    )
    inputs = _inputs(
        raw_states=[[0.0, 0.0, 1.0], [0.0, 0.0, 1.0]],
        raw_rates=jnp.zeros((2, 2)),
        mask_any=[True, True],
        t_measured=[0.0, 1.0],
        dense_t=[0.0, 0.5, 1.0],
        dense_raw_states=[[0.0, 0.0, 1.0]] * 3,
        dense_raw_rates=jnp.zeros((3, 2)),
        dense_raw_v_unclamped=[1.0, 0.0, 1.0],
        dense_valid_time=[True, True, True],
    )

    loss = module(inputs).named_losses["lwr_bnd/V"]

    assert loss == pytest.approx(1.0 / 192.0)


@pytest.mark.parametrize("dense_grid_n", [True, 1, 1.5])
def test_invalid_dense_grid_size_is_rejected(dense_grid_n):
    with pytest.raises(ValueError, match="dense_grid_n"):
        BoundsViolationLossModule(
            target_names=("biomass",),
            bound_snapshots=_bound_snapshots(_collection()),
            weight=1.0,
            dense_grid_n=dense_grid_n,
        )


@pytest.mark.parametrize("weight", [-1.0, float("inf"), float("nan")])
def test_invalid_weight_is_rejected(weight):
    with pytest.raises(ValueError, match="finite and nonnegative"):
        BoundsViolationLossModule(
            target_names=("biomass",),
            bound_snapshots=_bound_snapshots(_collection()),
            weight=weight,
        )


@pytest.mark.parametrize(
    ("bounds", "message"),
    [
        ((float("inf"), None), "finite or None"),
        ((1.0, 0.0), "must not exceed"),
    ],
)
def test_invalid_bounds_are_rejected(bounds, message):
    with pytest.raises(ValueError, match=message):
        BoundsViolationLossModule(
            target_names=("biomass",),
            bound_snapshots=_bound_snapshots(_collection(rmc_bounds=bounds)),
            weight=1.0,
        )


def test_empty_collection_is_rejected():
    with pytest.raises(ValueError, match="non-empty collection"):
        BoundsViolationLossModule(
            target_names=("biomass",),
            bound_snapshots=_bound_snapshots(BioProcessCollection(processes={})),
            weight=1.0,
        )


def test_missing_rate_in_later_process_is_rejected_clearly():
    p2 = _process("p2")
    p2.biological_ode.rates = {
        "q_other": NO_BOUNDS,
        "r_oxygen": NO_BOUNDS,
    }
    p2.biological_ode.derivatives["biomass"] = "q_other * biomass"
    collection = BioProcessCollection(processes={"p1": _process("p1"), "p2": p2})

    with pytest.raises(ValueError, match="biological_ode mismatch across processes"):
        BoundsViolationLossModule(
            target_names=("biomass",),
            bound_snapshots=_bound_snapshots(collection),
            weight=1.0,
        )


def test_inconsistent_process_bounds_are_rejected():
    collection = BioProcessCollection(
        processes={
            "p1": _process("p1", rmc_bounds=(0.0, None)),
            "p2": _process("p2", rmc_bounds=(-1.0, None)),
        }
    )
    with pytest.raises(ValueError, match="differ across processes"):
        BoundsViolationLossModule(
            target_names=("biomass",),
            bound_snapshots=_bound_snapshots(collection),
            weight=1.0,
        )
