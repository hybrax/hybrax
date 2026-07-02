"""Regression tests for :func:`bp_train.defaults.default_build_reaction_module`."""

from __future__ import annotations

from bp_format.mechanistic import build_rhs_ode
from bp_train.defaults import default_build_reaction_module
from test_harness import _make_collection


def test_default_reaction_module_scale_follows_modeled_state_not_targets():
    """``SCALE_modeled_RMCs`` is sized by the modeled RMC state slice, not by the targets.

    Under ``target_source="combined"`` / ``"process_variables"`` the measured-target set
    (reactor components *plus* modeled PVs) is a different length than the reactor-component
    state slice the module scales. Sizing the fallback RMC scale from ``len(target_names)``
    then produces a wrong-length axis that the wrapper rejects. This guards that regression:
    the RMC scale must always match ``len(rhs_ode.name_modeled_RMCs)``.
    """
    collection = _make_collection()
    rhs_ode = build_rhs_ode(collection.processes["p1"])
    n_rmc = len(rhs_ode.name_modeled_RMCs)

    # A measured-target set LONGER than the RMC state slice, mimicking
    # target_source="combined" (reactor components + process variables).
    target_names = ["biomass", "viability", "extra_pv"]
    assert len(target_names) != n_rmc  # precondition: counts differ, else nothing to catch

    module = default_build_reaction_module(
        target_names=target_names,
        process_names=list(collection.processes),
        config=None,
        seed=0,
        collection=collection,
    )

    assert module.SCALE_modeled_RMCs.shape[0] == n_rmc
    assert module.n_modeled_RMCs == n_rmc


def test_default_reaction_module_scale_independent_of_target_count():
    """The RMC scale length is invariant to how many targets are passed."""
    collection = _make_collection()
    n_rmc = len(build_rhs_ode(collection.processes["p1"]).name_modeled_RMCs)

    shapes = {
        default_build_reaction_module(
            target_names=targets,
            process_names=list(collection.processes),
            config=None,
            seed=0,
            collection=collection,
        ).SCALE_modeled_RMCs.shape[0]
        for targets in (
            ["biomass"],
            ["biomass", "viability"],
            ["biomass", "viability", "product", "lactate"],
        )
    }

    assert shapes == {n_rmc}
