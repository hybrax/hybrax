from __future__ import annotations

import jax.numpy as jnp
import pytest

from hybrax.train.model_api import ReactionInputs, ReactionOutputs, UserReactionModule
from stateful_helpers import build_stateful_wrapper, make_process


class _LatentAuxModule(UserReactionModule):
    h0: jnp.ndarray

    def __init__(self, h0):
        super().__init__()
        self.h0 = jnp.asarray(h0)

    @property
    def latent_observables(self) -> tuple[str, ...]:
        return ("mu",)

    def __call__(self, t, inputs: ReactionInputs) -> ReactionOutputs:
        del t
        mu = inputs.SCL_latent[0] + inputs.SCL_modeled_RMCs[0]
        return ReactionOutputs(
            SCL_modeled_BiologicalOde_rates=jnp.zeros(1, dtype=mu.dtype),
            SCL_modeled_Inflows_rates=jnp.zeros(0, dtype=mu.dtype),
            SCL_modeled_Outflows_rates=jnp.zeros(0),
            SCL_latent_derivative=jnp.zeros_like(inputs.SCL_latent),
            auxiliary={"mu": mu},
        )

    def initial_latent(self, RAW_phys_y0):
        del RAW_phys_y0
        return self.h0


class _RaisingLatentObserveModule(_LatentAuxModule):
    def observe(self, states):
        del states
        raise ValueError("latent observables are only available via auxiliary")


class _MissingAuxiliaryModule(_RaisingLatentObserveModule):
    def __call__(self, t, inputs: ReactionInputs) -> ReactionOutputs:
        del t
        return ReactionOutputs(
            SCL_modeled_BiologicalOde_rates=jnp.zeros(1, dtype=inputs.SCL_latent.dtype),
            SCL_modeled_Inflows_rates=jnp.zeros(0, dtype=inputs.SCL_latent.dtype),
            SCL_modeled_Outflows_rates=jnp.zeros(0),
            SCL_latent_derivative=jnp.zeros_like(inputs.SCL_latent),
        )


def test_validation_rejects_latent_observables_with_identity_posthoc_observe():
    with pytest.raises(ValueError, match="ReactionOutputs.auxiliary"):
        build_stateful_wrapper(
            make_process(),
            _LatentAuxModule(jnp.asarray([2.0])),
        )


def test_save_outputs_requires_declared_latent_observables_in_auxiliary():
    wrapper = build_stateful_wrapper(
        make_process(),
        _MissingAuxiliaryModule(jnp.asarray([2.0])),
    )

    with pytest.raises(ValueError, match="missing: \\['mu'\\]"):
        wrapper.physical_save_outputs(0.0, jnp.asarray([1.0, 1.0, 2.0]))


def test_latent_observable_uses_auxiliary_and_posthoc_observe_raises():
    wrapper = build_stateful_wrapper(
        make_process(),
        _RaisingLatentObserveModule(jnp.asarray([2.0])),
    )

    save_outputs = wrapper.physical_save_outputs(0.0, jnp.asarray([1.0, 1.0, 2.0]))

    assert save_outputs.auxiliary is not None
    assert jnp.allclose(save_outputs.auxiliary["mu"], 3.0)
    with pytest.raises(ValueError, match="auxiliary"):
        wrapper.reaction_module.observe(save_outputs.SCL_states)
