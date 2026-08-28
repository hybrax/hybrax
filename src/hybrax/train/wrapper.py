"""``HybridOdeWrapper``: RAW physical-state ODE wrapper around a user
reaction module.
"""

from __future__ import annotations

from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
from hybrax.format.dataclasses import BioProcess
from hybrax.format.mechanistic import RhsOde, build_rhs_ode

from .controls_store import PerProcessControls
from .model_api import (
    ReactionInputs,
    ReactionOutputs,
    UserLossModule,
    RateModule,
)


class SaveOutputs(eqx.Module):
    """Diffrax-saveable wrapper outputs.

    ``SCL_states`` are in the solver's scaled state space; rates are RAW. Use
    ``module.unscale_state`` to convert ``SCL_states`` to physical units when
    exporting to CSV/JSONL.
    """

    SCL_states: jax.Array
    RAW_V_export: jax.Array
    RAW_V: jax.Array
    RAW_modeled_ReactionOde_rates: jax.Array
    RAW_modeled_Inflows_rates: jax.Array
    RAW_modeled_Outflows_rates: jax.Array
    auxiliary: dict[str, jax.Array] | None = None


def _normalize_auxiliary_outputs(
    auxiliary: Any,
) -> dict[str, jax.Array] | None:
    """Validate and normalize reaction auxiliary outputs.

    Conservative contract for now:

    - ``None`` or ``dict[str, array]``
    - values must be scalar or 1D arrays at a single save time
    """
    if auxiliary is None:
        return None
    if not isinstance(auxiliary, dict):
        raise TypeError("ReactionOutputs.auxiliary must be None or dict[str, array]")

    normalized: dict[str, jax.Array] = {}
    for key, value in auxiliary.items():
        if not isinstance(key, str):
            raise TypeError("ReactionOutputs.auxiliary keys must be strings")
        arr = jnp.asarray(value)
        if arr.ndim not in (0, 1):
            raise ValueError(
                "ReactionOutputs.auxiliary values must be scalars or 1D arrays, "
                f"got key {key!r} with shape {tuple(arr.shape)}"
            )
        normalized[key] = arr
    return normalized


def _validate_reaction_output_shapes(
    module: RateModule, outputs: ReactionOutputs
) -> ReactionOutputs:
    expected_shapes = {
        "SCL_modeled_ReactionOde_rates": (module.n_modeled_ReactionOde_rates,),
        "SCL_modeled_Inflows_rates": (module.n_modeled_Inflows,),
        "SCL_modeled_Outflows_rates": (module.n_modeled_Outflows,),
        "SCL_latent_derivative": (module.n_latent,),
    }
    for name, expected_shape in expected_shapes.items():
        actual_shape = tuple(jnp.shape(getattr(outputs, name)))
        if actual_shape != expected_shape:
            raise ValueError(
                f"ReactionOutputs.{name} has shape {actual_shape}, "
                f"expected {expected_shape}"
            )
    return outputs


class HybridOdeWrapper(eqx.Module):
    """RAW physical-state wrapper around a rate module.

    ``physical_solve`` owns the scaled ODE reparameterization. This wrapper sees
    RAW integrated state ``[RMCs | PVs | V | modeled_cum | latent]``, builds
    scaled ``ReactionInputs`` for the module, and returns RAW derivatives/save
    outputs. Save-time ``SCL_states`` stay physical-only; latent-derived
    observables must use ``ReactionOutputs.auxiliary``.

    The wrapper holds **no scale fields of its own** — every scale comes from
    ``self.reaction_module.SCALE_*``.

    The wrapper applies one safety clip — ``RAW_RMC_rhs = max(RAW_RMCs, 0)`` —
    only on the path into ``RhsOde``. The module receives the **unclipped** SCL
    slice so MLP gradient flow survives transient negative excursions.
    """

    rhs_ode: RhsOde
    reaction_module: RateModule
    controls: PerProcessControls

    modeled_RMC_names: tuple[str, ...] = eqx.field(static=True)
    modeled_PV_names: tuple[str, ...] = eqx.field(static=True)
    modeled_Inflow_names: tuple[str, ...] = eqx.field(static=True)
    modeled_Outflow_names: tuple[str, ...] = eqx.field(static=True)

    # Cached slice sizes for the canonical controls vector; sourced from
    # PerProcessControls at construction time so the runtime path doesn't have
    # to introspect ``self.controls`` (which may be swapped to a
    # ``_BatchIndexedControls`` adapter that doesn't expose name tuples).
    n_controlled_Inflows: int = eqx.field(static=True)
    n_controlled_Outflows: int = eqx.field(static=True)
    n_controlled_PVs: int = eqx.field(static=True)

    target_state_indices: jax.Array  # which state columns are loss targets

    # User loss module (a UserLossModule). Untagged so the whole-wrapper
    # partition_trainable walk reads its own trainable_field()/frozen_field()
    # tags. Default None for direct constructors / forward-only paths; the
    # train harness always attaches a real module.
    loss_module: UserLossModule | None = None

    @classmethod
    def from_process(
        cls,
        *,
        reaction_module: RateModule,
        process: BioProcess,
        controls: PerProcessControls,
        target_state_indices: jax.Array | None = None,
        loss_module: UserLossModule | None = None,
    ) -> HybridOdeWrapper:
        """Build a wrapper from a BioProcess and per-process controls.

        Args:
            reaction_module: Trained/untrained reaction module; its
                ``SCALE_*`` fields are validated against ``process``.
            process: Process the ``RhsOde`` template is built from.
            controls: This process's controls; see :func:`from_rhs_ode`.
            target_state_indices: Forwarded to :func:`from_rhs_ode`.
            loss_module: Forwarded to :func:`from_rhs_ode`.

        Returns:
            The constructed wrapper.
        """
        return cls.from_rhs_ode(
            reaction_module=reaction_module,
            rhs_ode=build_rhs_ode(process),
            controls=controls,
            target_state_indices=target_state_indices,
            loss_module=loss_module,
        )

    @classmethod
    def from_rhs_ode(
        cls,
        *,
        reaction_module: RateModule,
        rhs_ode: RhsOde,
        controls: PerProcessControls,
        target_state_indices: jax.Array | None = None,
        loss_module: UserLossModule | None = None,
    ) -> HybridOdeWrapper:
        """Build a wrapper from an existing RhsOde runtime template.

        Scales are read from ``reaction_module`` (a ``RateModule``
        subclass with ``SCALE_*`` fields). The constructor validates each
        ``SCALE_*`` shape against the RhsOde / controls layout.

        Args:
            reaction_module: Reaction module supplying every ``SCALE_*``
                field and the model's ``__call__``.
            rhs_ode: Canonical ODE structure this process's controls must
                agree with.
            controls: This process's controls; its categorized name tuples
                must match ``rhs_ode``'s.
            target_state_indices: State columns that are loss targets, or
                ``None`` to default to ``[RMCs | PVs]`` plus the modeled
                cumulative-flow columns (excluding ``V``).
            loss_module: Loss module attached to the wrapper, or ``None`` for
                a forward-only wrapper.

        Returns:
            The constructed wrapper.

        Raises:
            ValueError: If ``rhs_ode`` and ``controls`` disagree on
                controlled-name categorization, a ``SCALE_*`` field has the
                wrong shape, a rate-axis scaler has a non-zero offset, or a
                stateful latent observable is not emitted via
                ``ReactionOutputs.auxiliary``.
            TypeError: If ``reaction_module`` is missing a required
                ``SCALE_*`` field.
        """
        for field_name in (
            "name_controlled_Inflows",
            "name_controlled_Outflows",
            "name_controlled_PVs",
        ):
            rhs_names = getattr(rhs_ode, field_name)
            control_names = getattr(controls, field_name)
            if rhs_names != control_names:
                raise ValueError(
                    f"RhsOde and controls disagree on {field_name}: "
                    f"{rhs_names} vs {control_names}"
                )

        n_RMCs = len(rhs_ode.name_modeled_RMCs)
        n_PVs = len(rhs_ode.name_modeled_PVs)
        n_Inflows = len(rhs_ode.name_modeled_Inflows)
        n_Outflows = len(rhs_ode.name_modeled_Outflows)
        n_rates = len(rhs_ode.name_modeled_rates)

        n_controlled_Inflows_count = len(controls.name_controlled_Inflows)
        n_controlled_Outflows_count = len(controls.name_controlled_Outflows)

        # Validate reaction_module SCALE_* shapes match the layout we'll feed.
        _expected_shapes: dict[str, tuple[int, ...]] = {
            "SCALE_modeled_RMCs": (n_RMCs,),
            "SCALE_modeled_PVs": (n_PVs,),
            "SCALE_modeled_Inflows_cumulative": (n_Inflows,),
            "SCALE_controlled_Inflows_cumulative": (n_controlled_Inflows_count,),
            "SCALE_controlled_Inflows_rates": (n_controlled_Inflows_count,),
            "SCALE_controlled_Inflows_Cin": (n_controlled_Inflows_count, n_RMCs),
            "SCALE_controlled_Outflows_cumulative": (n_controlled_Outflows_count,),
            "SCALE_controlled_Outflows_rates": (n_controlled_Outflows_count,),
            "SCALE_controlled_PVs": (len(controls.name_controlled_PVs),),
            "SCALE_modeled_Inflows_Cin": (n_Inflows, n_RMCs),
            "SCALE_modeled_ReactionOde_rates": (n_rates,),
            "SCALE_modeled_Inflows_rates": (n_Inflows,),
            "SCALE_modeled_Outflows_cumulative": (n_Outflows,),
            "SCALE_modeled_Outflows_rates": (n_Outflows,),
        }
        for field_name, expected in _expected_shapes.items():
            if not hasattr(reaction_module, field_name):
                raise TypeError(
                    f"reaction_module is missing SCALE field {field_name!r}; "
                    "subclass RateModule and pass all SCALE_* fields "
                    "to super().__init__(...)."
                )
            arr = getattr(reaction_module, field_name)
            if tuple(arr.shape) != expected:
                raise ValueError(
                    f"reaction_module.{field_name} has shape {tuple(arr.shape)}, "
                    f"expected {expected}"
                )
        # Rate derivatives are offset-free. This boundary catches the common
        # built-in AffineScaler mistake; custom scalers may omit offset metadata.
        for field_name in (
            "SCALE_controlled_Inflows_rates",
            "SCALE_controlled_Outflows_rates",
            "SCALE_modeled_ReactionOde_rates",
            "SCALE_modeled_Inflows_rates",
            "SCALE_modeled_Outflows_rates",
        ):
            scaler = getattr(reaction_module, field_name)
            offset = getattr(scaler, "offset", None)
            if offset is not None and bool(jnp.any(offset != 0)):
                raise ValueError(
                    f"{field_name} is a rate axis and cannot have a non-zero "
                    "offset; rate scaling is offset-free"
                )
        if (
            reaction_module.n_latent > 0
            and reaction_module.latent_observables
            and type(reaction_module).observe is RateModule.observe
        ):
            names = ", ".join(reaction_module.latent_observables)
            raise ValueError(
                "Stateful latent observables must be emitted via "
                f"ReactionOutputs.auxiliary during the solve: {names}"
            )

        # V_in_cumulative is a scalar — accept any ndim-0 array.
        if not hasattr(reaction_module, "SCALE_V_in_cumulative"):
            raise TypeError(
                "reaction_module is missing SCALE_V_in_cumulative; subclass "
                "RateModule and pass all SCALE_* fields to "
                "super().__init__(...)."
            )

        n_modeled = n_Inflows + len(rhs_ode.name_modeled_Outflows)

        # Default target_state_indices: the [RMCs | PVs] leading block + the
        # modeled cumulative-flow columns. V (at index
        # n_RMCs + n_PVs) is in the state but not a loss target.
        if target_state_indices is None:
            n_leading = n_RMCs + n_PVs
            default_indices = list(range(n_leading)) + list(
                range(n_leading + 1, n_leading + 1 + n_modeled)
            )
            _target_state_indices = jnp.asarray(default_indices, dtype=jnp.int32)
        else:
            _target_state_indices = jnp.asarray(target_state_indices, dtype=jnp.int32)

        return cls(
            rhs_ode=rhs_ode,
            reaction_module=reaction_module,
            controls=controls,
            modeled_RMC_names=rhs_ode.name_modeled_RMCs,
            modeled_PV_names=rhs_ode.name_modeled_PVs,
            modeled_Inflow_names=rhs_ode.name_modeled_Inflows,
            modeled_Outflow_names=rhs_ode.name_modeled_Outflows,
            n_controlled_Inflows=n_controlled_Inflows_count,
            n_controlled_Outflows=n_controlled_Outflows_count,
            n_controlled_PVs=len(controls.name_controlled_PVs),
            target_state_indices=_target_state_indices,
            loss_module=loss_module,
        )

    # ------ Physical-state RHS (continuous part of the diffrax_callbacks solve) ------
    #
    # Integrates the *physical* state
    # ``y = [RAW_RMCs | RAW_PVs | RAW_V | RAW_modeled_Inflows_cumulative |
    # RAW_modeled_Outflows_cumulative | RAW_latent]``. Save/loss state omits
    # only the latent suffix.
    # Bolus/sample events are applied as state jumps between
    # segments (``physical_solve.solve_physical_states``), not folded into the
    # vector field. Between events only continuous dynamics act
    # (biology, plus continuous feeds/dilution if any), so the integrated state stays
    # O(C) and the reverse-mode adjoint is well-conditioned. (Replaces the earlier
    # pseudobatch state, whose unbounded accumulator corrupted the gradient.)

    def physical_rhs(self, t: float | jax.Array, y_phys: jax.Array) -> jax.Array:
        """Return the derivative of the physical and latent state vector.

        Continuous part only — discrete boluses/samples are handled as jumps.
        """
        module = self.reaction_module
        n_RMCs = len(self.modeled_RMC_names)
        n_PVs = len(self.modeled_PV_names)
        n_Inflows = len(self.modeled_Inflow_names)
        n_Outflows = len(self.modeled_Outflow_names)
        n_phys = n_RMCs + n_PVs + 1 + n_Inflows + n_Outflows
        dtype = y_phys.dtype
        t_arr = jnp.asarray(t, dtype=dtype)

        RAW_phys = y_phys[:n_phys]
        RAW_RMCs = RAW_phys[:n_RMCs]
        RAW_PVs = RAW_phys[n_RMCs : n_RMCs + n_PVs]
        RAW_V = RAW_phys[n_RMCs + n_PVs]
        cumulative_start = n_RMCs + n_PVs + 1
        RAW_modeled_Inflows_cumulative = RAW_phys[
            cumulative_start : cumulative_start + n_Inflows
        ]
        RAW_modeled_Outflows_cumulative = RAW_phys[
            cumulative_start + n_Inflows : n_phys
        ]
        RAW_latent = y_phys[n_phys:]
        RAW_RMC_rhs = jnp.maximum(RAW_RMCs, 0.0)

        RAW_controlled_Inflows_cumulative = (
            self.controls.eval_controlled_Inflows_cumulative(t_arr, y_phys)
        )
        RAW_controlled_Inflows_rates = self.controls.eval_controlled_Inflows_rates(
            t_arr, y_phys
        )
        RAW_controlled_Outflows_cumulative = (
            self.controls.eval_controlled_Outflows_cumulative(t_arr, y_phys)
        )
        RAW_controlled_Outflows_rates = self.controls.eval_controlled_Outflows_rates(
            t_arr, y_phys
        )
        RAW_controlled_PVs = self.controls.eval_controlled_PVs(t_arr, y_phys)
        RAW_controlled_Inflows_Cin = self.rhs_ode.Cin_controlled_Inflows
        RAW_modeled_Inflows_Cin = self.rhs_ode.Cin_modeled_Inflows

        inputs = ReactionInputs(
            SCL_modeled_RMCs=module.scale_modeled_RMCs(RAW_RMCs),
            SCL_modeled_PVs=module.scale_modeled_PVs(RAW_PVs),
            SCL_modeled_V=module.scale_modeled_V(RAW_V),
            SCL_modeled_Inflows_cumulative=module.scale_modeled_Inflows_cumulative(
                RAW_modeled_Inflows_cumulative
            ),
            SCL_modeled_Outflows_cumulative=module.scale_modeled_Outflows_cumulative(
                RAW_modeled_Outflows_cumulative
            ),
            SCL_controlled_Inflows_cumulative=module.scale_controlled_Inflows_cumulative(
                RAW_controlled_Inflows_cumulative
            ),
            SCL_controlled_Inflows_rates=module.scale_controlled_Inflows_rates(
                RAW_controlled_Inflows_rates
            ),
            SCL_controlled_Inflows_Cin=module.scale_controlled_Inflows_Cin(
                RAW_controlled_Inflows_Cin
            ),
            SCL_controlled_Outflows_cumulative=(
                module.scale_controlled_Outflows_cumulative(
                    RAW_controlled_Outflows_cumulative
                )
            ),
            SCL_controlled_Outflows_rates=module.scale_controlled_Outflows_rates(
                RAW_controlled_Outflows_rates
            ),
            SCL_controlled_PVs=module.scale_controlled_PVs(RAW_controlled_PVs),
            SCL_modeled_Inflows_Cin=module.scale_modeled_Inflows_Cin(
                RAW_modeled_Inflows_Cin
            ),
            RAW_controlled_Outflows_retention=(
                self.rhs_ode.retention_controlled_Outflows
            ),
            RAW_modeled_Outflows_retention=self.rhs_ode.retention_modeled_Outflows,
            SCL_latent=module.scale_latent(RAW_latent),
        )
        outputs = _validate_reaction_output_shapes(module, module(t_arr, inputs))
        RAW_bio_rates = module.unscale_modeled_ReactionOde_rates(
            jnp.asarray(outputs.SCL_modeled_ReactionOde_rates, dtype=dtype)
        )
        RAW_modeled_Inflows_rates = module.unscale_modeled_Inflows_rates(
            jnp.asarray(outputs.SCL_modeled_Inflows_rates, dtype=dtype)
        )
        RAW_modeled_Outflows_rates = module.unscale_modeled_Outflows_rates(
            jnp.asarray(outputs.SCL_modeled_Outflows_rates, dtype=dtype)
        )

        # RhsOde state c = [RMCs | PVs | V]; PVs pass through unclipped (they need
        # not be non-negative). RhsOde owns continuous feed transport and dilution.
        RAW_RMCs_PVs_V = jnp.concatenate([RAW_RMC_rhs, RAW_PVs, RAW_V[None]])
        RAW_u = jnp.concatenate(
            [
                RAW_controlled_Inflows_rates,
                RAW_controlled_Outflows_rates,
                RAW_controlled_PVs,
            ]
        )
        RAW_d_dt = self.rhs_ode(
            RAW_RMCs_PVs_V,
            RAW_bio_rates,
            RAW_u,
            RAW_modeled_Inflows_rates,
            RAW_modeled_Outflows_rates,
            self.controls.min_V,
        )
        RAW_d_phys_dt = jnp.concatenate(
            [RAW_d_dt, RAW_modeled_Inflows_rates, RAW_modeled_Outflows_rates]
        )
        RAW_d_latent_dt = module.SCALE_latent.unscale_derivative(
            jnp.asarray(outputs.SCL_latent_derivative, dtype=dtype)
        )
        return jnp.concatenate([RAW_d_phys_dt, RAW_d_latent_dt])

    def initial_physical_state_from_raw(self, RAW_state: jax.Array) -> jax.Array:
        """Append the module's RAW latent initial state to the physical state."""
        n_RMCs = len(self.modeled_RMC_names)
        n_PVs = len(self.modeled_PV_names)
        n_Inflows = len(self.modeled_Inflow_names)
        n_Outflows = len(self.modeled_Outflow_names)
        RAW_phys = RAW_state[: n_RMCs + n_PVs + 1 + n_Inflows + n_Outflows]
        return jnp.concatenate(
            [RAW_phys, self.reaction_module.initial_latent(RAW_phys)]
        )

    def physical_save_outputs(
        self, t: float | jax.Array, y_phys: jax.Array
    ) -> "SaveOutputs":
        """``SaveOutputs`` computed from the physical state (for loss/exports)."""
        module = self.reaction_module
        n_RMCs = len(self.modeled_RMC_names)
        n_PVs = len(self.modeled_PV_names)
        n_Inflows = len(self.modeled_Inflow_names)
        n_Outflows = len(self.modeled_Outflow_names)
        n_phys = n_RMCs + n_PVs + 1 + n_Inflows + n_Outflows
        dtype = y_phys.dtype
        t_arr = jnp.asarray(t, dtype=dtype)
        RAW_phys = y_phys[:n_phys]
        RAW_RMCs = RAW_phys[:n_RMCs]
        RAW_PVs = RAW_phys[n_RMCs : n_RMCs + n_PVs]
        RAW_V = RAW_phys[n_RMCs + n_PVs]
        cumulative_start = n_RMCs + n_PVs + 1
        RAW_modeled_Inflows_cumulative = RAW_phys[
            cumulative_start : cumulative_start + n_Inflows
        ]
        RAW_modeled_Outflows_cumulative = RAW_phys[
            cumulative_start + n_Inflows : n_phys
        ]
        RAW_latent = y_phys[n_phys:]

        RAW_controlled_Inflows_cumulative = (
            self.controls.eval_controlled_Inflows_cumulative(t_arr, y_phys)
        )
        RAW_controlled_Inflows_rates = self.controls.eval_controlled_Inflows_rates(
            t_arr, y_phys
        )
        RAW_controlled_Outflows_cumulative = (
            self.controls.eval_controlled_Outflows_cumulative(t_arr, y_phys)
        )
        RAW_controlled_Outflows_rates = self.controls.eval_controlled_Outflows_rates(
            t_arr, y_phys
        )
        RAW_controlled_PVs = self.controls.eval_controlled_PVs(t_arr, y_phys)
        inputs = ReactionInputs(
            SCL_modeled_RMCs=module.scale_modeled_RMCs(RAW_RMCs),
            SCL_modeled_PVs=module.scale_modeled_PVs(RAW_PVs),
            SCL_modeled_V=module.scale_modeled_V(RAW_V),
            SCL_modeled_Inflows_cumulative=module.scale_modeled_Inflows_cumulative(
                RAW_modeled_Inflows_cumulative
            ),
            SCL_modeled_Outflows_cumulative=module.scale_modeled_Outflows_cumulative(
                RAW_modeled_Outflows_cumulative
            ),
            SCL_controlled_Inflows_cumulative=module.scale_controlled_Inflows_cumulative(
                RAW_controlled_Inflows_cumulative
            ),
            SCL_controlled_Inflows_rates=module.scale_controlled_Inflows_rates(
                RAW_controlled_Inflows_rates
            ),
            SCL_controlled_Inflows_Cin=module.scale_controlled_Inflows_Cin(
                self.rhs_ode.Cin_controlled_Inflows
            ),
            SCL_controlled_Outflows_cumulative=(
                module.scale_controlled_Outflows_cumulative(
                    RAW_controlled_Outflows_cumulative
                )
            ),
            SCL_controlled_Outflows_rates=module.scale_controlled_Outflows_rates(
                RAW_controlled_Outflows_rates
            ),
            SCL_controlled_PVs=module.scale_controlled_PVs(RAW_controlled_PVs),
            SCL_modeled_Inflows_Cin=module.scale_modeled_Inflows_Cin(
                self.rhs_ode.Cin_modeled_Inflows
            ),
            RAW_controlled_Outflows_retention=(
                self.rhs_ode.retention_controlled_Outflows
            ),
            RAW_modeled_Outflows_retention=self.rhs_ode.retention_modeled_Outflows,
            SCL_latent=module.scale_latent(RAW_latent),
        )
        outputs = _validate_reaction_output_shapes(module, module(t_arr, inputs))
        RAW_bio_rates = module.unscale_modeled_ReactionOde_rates(
            jnp.asarray(outputs.SCL_modeled_ReactionOde_rates, dtype=dtype)
        )
        RAW_modeled_Inflows_rates = module.unscale_modeled_Inflows_rates(
            jnp.asarray(outputs.SCL_modeled_Inflows_rates, dtype=dtype)
        )
        RAW_modeled_Outflows_rates = module.unscale_modeled_Outflows_rates(
            jnp.asarray(outputs.SCL_modeled_Outflows_rates, dtype=dtype)
        )
        RAW_state = jnp.concatenate(
            [
                RAW_RMCs,
                RAW_PVs,
                RAW_V[None],
                RAW_modeled_Inflows_cumulative,
                RAW_modeled_Outflows_cumulative,
            ]
        )
        auxiliary = _normalize_auxiliary_outputs(getattr(outputs, "auxiliary", None))
        if module.n_latent > 0 and module.latent_observables:
            missing = [
                name
                for name in module.latent_observables
                if auxiliary is None or name not in auxiliary
            ]
            if missing:
                raise ValueError(
                    "Latent observables must be emitted via "
                    "ReactionOutputs.auxiliary during the solve; missing: "
                    f"{missing}"
                )
        return SaveOutputs(
            SCL_states=module.scale_state(RAW_state),
            RAW_V_export=RAW_V,
            RAW_V=RAW_V,
            RAW_modeled_ReactionOde_rates=RAW_bio_rates,
            RAW_modeled_Inflows_rates=RAW_modeled_Inflows_rates,
            RAW_modeled_Outflows_rates=RAW_modeled_Outflows_rates,
            auxiliary=auxiliary,
        )


def validate_rhs_ode_compatibility(
    reference_name: str,
    reference_rhs_ode: RhsOde,
    candidate_name: str,
    candidate_rhs_ode: RhsOde,
) -> None:
    """Validate that two RhsOde instances have compatible runtime axes.

    Args:
        reference_name: Label for ``reference_rhs_ode`` in any error message.
        reference_rhs_ode: The RhsOde every name-field axis is compared against.
        candidate_name: Label for ``candidate_rhs_ode`` in any error message.
        candidate_rhs_ode: The RhsOde being checked for compatibility.

    Raises:
        ValueError: If any of the two RhsOdes' name-tuple fields differ.
    """
    name_fields = (
        "name_modeled_RMCs",
        "name_modeled_PVs",
        "name_modeled_Inflows",
        "name_modeled_Outflows",
        "name_controlled_PVs",
        "name_controlled_Inflows",
        "name_controlled_Outflows",
        "name_modeled_rates",
        "name_modeled_algebraic",
    )
    for field_name in name_fields:
        reference_value = getattr(reference_rhs_ode, field_name)
        candidate_value = getattr(candidate_rhs_ode, field_name)
        if reference_value != candidate_value:
            raise ValueError(
                f"RhsOde {field_name} differ between {reference_name!r} and "
                f"{candidate_name!r}: {reference_value} vs {candidate_value}"
            )
    for field_name in (
        "Cin_controlled_Inflows",
        "Cin_modeled_Inflows",
        "retention_controlled_Outflows",
        "retention_modeled_Outflows",
    ):
        reference_shape = getattr(reference_rhs_ode, field_name).shape
        candidate_shape = getattr(candidate_rhs_ode, field_name).shape
        if reference_shape != candidate_shape:
            raise ValueError(
                f"RhsOde {field_name} shapes differ between {reference_name!r} and "
                f"{candidate_name!r}: {reference_shape} vs {candidate_shape}"
            )
