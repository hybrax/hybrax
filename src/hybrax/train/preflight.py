"""Pre-training scalar checks of prescribed continuous-flow integration."""

from __future__ import annotations

import math

import diffrax
import equinox as eqx
import jax.numpy as jnp

from .constants import (
    DEFAULT_FLOW_INTEGRAL_RELATIVE_TOLERANCE,
    INITIAL_STEP_FRACTION,
)
from .controls_store import BatchControls
from .trainer import clamp_padded_time_rows
from .training_data import TrainingDataStore


# It would be intuitive to also run this check during prepare, but prepared data
# can be reused with different solver settings and process selections. Run it here
# with the actual training controls and settings; passing during prepare would not
# establish that the eventual training run integrates the flows correctly.
@eqx.filter_jit
def _integrate_flow(controls, flow_index, t0, t1, jump_ts, *, max_steps, rtol, atol):
    """Use the training rate evaluator without biology or state-jump callbacks."""
    solution = diffrax.diffeqsolve(
        diffrax.ODETerm(
            lambda t, y, args: controls._eval_derivatives(0, t)[flow_index]
        ),
        diffrax.Tsit5(),
        t0=t0,
        t1=t1,
        dt0=(t1 - t0) * INITIAL_STEP_FRACTION,
        y0=jnp.zeros((), dtype=t0.dtype),
        stepsize_controller=diffrax.PIDController(
            rtol=rtol, atol=atol, jump_ts=jump_ts
        ),
        max_steps=max_steps,
        saveat=diffrax.SaveAt(t1=True),
        throw=False,
    )
    expected = (
        controls._eval_values(0, t1)[flow_index]
        - controls._eval_values(0, t0)[flow_index]
    )
    return solution.ys[0], expected, solution.result


def check_continuous_flow_integration(
    store: TrainingDataStore,
    process_names: tuple[str, ...],
    *,
    max_steps: int,
    rtol: float,
    atol: float,
    use_jump_ts: bool,
    flow_integral_relative_tolerance: float = DEFAULT_FLOW_INTEGRAL_RELATIVE_TOLERANCE,
) -> None:
    """Reject scalar flow solves that miss their known cumulative increment.

    Each prescribed continuous inflow/outflow is checked independently over its
    training measurement span using Tsit5 and the configured PID settings. The
    endpoint allowance is the known increment's magnitude times
    `flow_integral_relative_tolerance`, independent of solver tolerances. A zero
    increment requires an exactly zero integral. No control knots are added.

    Passing is a preflight diagnostic, not proof of full-model integration accuracy;
    the coupled model takes different adaptive steps. Boluses, samples, modeled
    flows, and process-variable controls are not scalar continuous flows.
    """
    controls_store = store.controls_store
    flow_names = (
        controls_store.name_controlled_Inflows + controls_store.name_controlled_Outflows
    )
    if not flow_names:
        return
    process_indices = {name: i for i, name in enumerate(store.process_order)}
    for process_name in process_names:
        index = process_indices[process_name]
        t0 = store.t_measured[index, 0]
        t1 = store.t_measured[index, int(store.n_measured[index]) - 1]
        if bool(t0 == t1):
            continue
        # A one-row batch avoids recompiling for per-process static names/metadata.
        controls: BatchControls = controls_store.gather_batch(jnp.asarray([index]))
        jump_ts = (
            clamp_padded_time_rows(
                controls_store.jump_ts[index : index + 1],
                controls_store.jump_ts_lengths[index : index + 1],
            )[0]
            if use_jump_ts and controls_store.jump_ts.shape[1]
            else None
        )
        for flow_index, flow_name in enumerate(flow_names):
            integrated, expected, result = _integrate_flow(
                controls,
                jnp.asarray(flow_index),
                t0,
                t1,
                jump_ts,
                max_steps=max_steps,
                rtol=rtol,
                atol=atol,
            )
            integrated, expected = float(integrated), float(expected)
            tolerance = flow_integral_relative_tolerance * abs(expected)
            if result != diffrax.RESULTS.successful or not (
                math.isfinite(integrated)
                and math.isfinite(expected)
                and abs(integrated - expected) <= tolerance
            ):
                status = (
                    "successful" if result == diffrax.RESULTS.successful else result
                )
                summary = (
                    "The solver failed. Check the solver result above; if it "
                    "exhausted its step budget, consider increasing solver.max_steps."
                    if result != diffrax.RESULTS.successful
                    else "The integrated flow disagrees with its cumulative increment "
                    "or contains nonfinite values. The solver may have missed short "
                    "feed intervals or abrupt rate changes. Add their breakpoints via "
                    "discrete_events with solver.jump_ts enabled, or smooth noisy "
                    "cumulative data using a monotone spline that preserves the "
                    "intended feed total and timing."
                )
                raise ValueError(
                    f"Continuous-flow preflight failed for process {process_name!r}, "
                    f"flow {flow_name!r} over [{float(t0):g}, {float(t1):g}]: "
                    f"integrated={integrated:.9g}, expected={expected:.9g}, "
                    f"tolerance={tolerance:.9g}, solver_result={status}.\n\n"
                    "Training stopped because this flow could not be integrated "
                    f"reliably. {summary}\n\n"
                    "More details:\n"
                    "Before training, Hybrax integrates each prescribed continuous "
                    "flow independently and compares it with the cumulative "
                    "increment represented by that runtime control over the interval "
                    "above. The allowed absolute error is "
                    "train.flow_integral_relative_tolerance times the magnitude of "
                    "the expected increment. Failed solves and nonfinite values "
                    "also reject.\n\n"
                    "For raw cumulative data, linear interpolation produces a "
                    "constant rate between consecutive timestamps. An adaptive "
                    "solver can step across short feed intervals or abrupt rate "
                    "changes and report success despite integrating the wrong total. "
                    "This does not necessarily mean the cumulative data is invalid. "
                    "Possible remedies:\n\n"
                    "1. Supply relevant rate-change times through discrete_events "
                    "and enable solver.jump_ts. Include both ends of short feed "
                    "intervals so the solver respects those breakpoints.\n\n"
                    "2. If jagged rates reflect measurement noise or scale "
                    "quantization, smooth the cumulative data or fit a monotone "
                    "spline consistent with the flow direction. Check the "
                    "derivative's sign and preserve the intended total and timing. "
                    "Smoothing alone does not guarantee short pulses are resolved; "
                    "explicit breakpoints may still be needed.\n\n"
                    "3. Try tighter solver.rtol and solver.atol. If integration "
                    "exhausted the step budget, increase solver.max_steps as "
                    "appropriate. Tighter tolerances alone may still miss a pulse "
                    "if no solver evaluations encounter it.\n\n"
                    "4. If the discrepancy is physically acceptable, adjust "
                    "train.flow_integral_relative_tolerance (default: 1e-3, or "
                    "0.1%). This acceptance threshold is independent of solver "
                    "tolerances. Increasing it does not improve integration; do "
                    "not use it to conceal substantial missed flow.\n\n"
                    "The expected increment comes from the control used by "
                    "training. After changing or fitting that control, compare it "
                    "with the original cumulative measurements too: this check "
                    "cannot detect a fit that changes the intended total, and "
                    "passing does not certify coupled-model integration accuracy."
                )
