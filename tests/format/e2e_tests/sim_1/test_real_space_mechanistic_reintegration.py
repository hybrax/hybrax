import copy
import os

os.environ.setdefault("JAX_ENABLE_X64", "true")

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402
from bp_format.dataclasses import BioProcess  # noqa: E402
from bp_format.dataclasses import ProcessOrdering  # noqa: E402
from bp_format.mechanistic import ControlSplines  # noqa: E402
from bp_format.mechanistic import RhsOde  # noqa: E402
from bp_format.mechanistic import _timeseries_to_ppoly  # noqa: E402
from bp_format.mechanistic import build_rhs_ode  # noqa: E402
from bp_format.mechanistic import build_state_splines  # noqa: E402
from bp_format.mechanistic import get_control_splines  # noqa: E402
from bp_format.mechanistic import get_process_ordering  # noqa: E402
from bp_format.serialization import load_process_collection_json  # noqa: E402
from bp_format.splines import build_pseudobatch_transform  # noqa: E402
from bp_format.splines import make_cubic_ppoly  # noqa: E402
from bp_format.time_series import PPoly  # noqa: E402
from scipy.integrate import solve_ivp  # noqa: E402

from .cstar_helpers import EXPECTED_PROCESS_IDS  # noqa: E402
from .cstar_helpers import EXPECTED_REACTOR_COMPONENT_ORDER  # noqa: E402
from .cstar_helpers import SIMULATION_DENSE_OUTPUT  # noqa: E402
from .cstar_helpers import SIM_RESULTS_DIR  # noqa: E402
from .cstar_helpers import dense_online_reactor_reference  # noqa: E402
from .cstar_helpers import fit_cstar_timeseries_from_values  # noqa: E402
from .real_space_segments import PRE_EVENT_ROW  # noqa: E402
from .real_space_segments import RealSpaceSegment  # noqa: E402
from .real_space_segments import build_real_space_segments  # noqa: E402
from .real_space_segments import dense_rows_by_process  # noqa: E402
from .real_space_segments import segment_spline_times  # noqa: E402
from .real_space_segments import segment_state_matrix  # noqa: E402
from .real_space_segments import segment_times  # noqa: E402

DATA_JSON = SIM_RESULTS_DIR / "process_collection.json"
EXPECTED_MODELED_RMCS = (
    "ammonia",
    "biomass",
    "dead_cells",
    "glucose",
    "glutamine",
    "lactate",
    "product_extracellular",
    "product_intracellular",
    "tracer_fed",
    "tracer_unfed",
)
EXPECTED_MODELED_PVS = ("intracellular_product_ratio",)
EXPECTED_CONTROLLED_PVS = ("pH", "temperature")
EXPECTED_MODELED_RATES = (
    "q_biomass",
    "q_product_extracellular",
    "q_product_intracellular",
    "q_dead_cells",
    "q_glucose",
    "q_glutamine",
    "q_lactate",
    "q_ammonia",
)
EXPECTED_MODELED_FVCS = ("base_feed",)
EXPECTED_CONTROLLED_FVCS = ("conti_feed",)
EXPECTED_MODELED_SVCS = ()
EXPECTED_CONTROLLED_SVCS = ()
MODELED_FVC_FLOW_COLUMNS = {"base_feed": "base_flow_l_per_h"}
RHS_REINTEGRATION_RTOL = 1e-7
RHS_REINTEGRATION_ATOL = 5e-6
SOLVER_RTOL = 1e-10
SOLVER_ATOL = 1e-12
CONTROL_RTOL = 1e-12
CONTROL_ATOL = 1e-12
MODELED_FVC_FLOW_RTOL = 5e-2
MODELED_FVC_FLOW_ATOL = 2e-5
BACKTRANSFORM_RTOL = 1e-7
BACKTRANSFORM_ATOL = 5e-6
BACKTRANSFORM_QUADRATURE_ORDER = 5
# Pointwise derivative residuals compare spline-derived derivatives to dense-grid
# references. The guard skips cubic-spline boundary artifacts; the absolute
# tolerances sit just above the measured clean residual floor (~3.5e-4).
DERIVATIVE_RHS_RTOL = 0.0
DERIVATIVE_RHS_ATOL = 5e-4
DERIVATIVE_FD_RTOL = 0.0
DERIVATIVE_FD_ATOL = 5e-4
DERIVATIVE_BOUNDARY_GUARD_POINTS = 8
DERIVATIVE_GRID_RTOL = 1e-10
DERIVATIVE_GRID_ATOL = 1e-12


def _ppoly_from_segment_rows(segment: RealSpaceSegment, column: str) -> PPoly:
    return make_cubic_ppoly(
        jnp.asarray([float(row["time"]) for row in segment.rows]),
        jnp.asarray([float(row[column]) for row in segment.rows]),
    )


def _segment_rate_splines(
    segment: RealSpaceSegment,
    rate_names: tuple[str, ...],
) -> tuple[PPoly, ...]:
    return tuple(_ppoly_from_segment_rows(segment, name) for name in rate_names)


def _process_modeled_fvc_splines(
    process: BioProcess,
    fvc_names: tuple[str, ...],
) -> tuple[PPoly, ...]:
    return tuple(
        _timeseries_to_ppoly(process.volume.volume_changes[name].values)
        for name in fvc_names
    )


def _evaluate_left(series, times: np.ndarray) -> np.ndarray:
    return np.asarray(
        series.evaluate_many(jnp.asarray(times), side="left"),
        dtype=float,
    )


def _add_dense_pseudobatch_transform(process: BioProcess, process_id: str) -> None:
    process.pseudobatch_transform = build_pseudobatch_transform(
        process,
        list(EXPECTED_REACTOR_COMPONENT_ORDER),
    )
    dense = dense_online_reactor_reference(process_id, process.time_axis.end)
    dense_times = dense["time"]
    assert process.pseudobatch_transform is not None
    adf = _evaluate_left(process.pseudobatch_transform.adf, dense_times)
    for name in EXPECTED_REACTOR_COMPONENT_ORDER:
        feed_correction = _evaluate_left(
            process.pseudobatch_transform.feed_corrections[name],
            dense_times,
        )
        c_star = dense[name] * adf - feed_correction
        process.reactor_medium.components[
            name
        ].c_star_concentration = fit_cstar_timeseries_from_values(
            name,
            dense_times,
            c_star,
            source="dense_online_oracle_left_event",
        )


def _assert_expected_sim_1_ordering(ordering: ProcessOrdering) -> None:
    assert ordering.name_modeled_RMCs == EXPECTED_MODELED_RMCS
    assert ordering.name_modeled_PVs == EXPECTED_MODELED_PVS
    assert ordering.name_controlled_PVs == EXPECTED_CONTROLLED_PVS
    assert ordering.name_modeled_rates == EXPECTED_MODELED_RATES
    assert ordering.name_modeled_FVCs == EXPECTED_MODELED_FVCS
    assert ordering.name_controlled_FVCs == EXPECTED_CONTROLLED_FVCS
    assert ordering.name_modeled_SVCs == EXPECTED_MODELED_SVCS
    assert ordering.name_controlled_SVCs == EXPECTED_CONTROLLED_SVCS


def _integrate_backtransform_derivative_for_segment(
    segment: RealSpaceSegment,
    backtransform_derivative_fns: dict[str, object],
) -> tuple[np.ndarray, np.ndarray]:
    times = segment_times(segment)
    truth = segment_state_matrix(segment, EXPECTED_REACTOR_COMPONENT_ORDER)
    nodes, weights = np.polynomial.legendre.leggauss(BACKTRANSFORM_QUADRATURE_ORDER)

    starts = times[:-1]
    ends = times[1:]
    midpoints = 0.5 * (starts + ends)
    half_widths = 0.5 * (ends - starts)
    interval_eval_times = midpoints[:, np.newaxis] + half_widths[:, np.newaxis] * nodes
    flat_eval_times = jnp.asarray(interval_eval_times.ravel())
    derivative_blocks = np.stack(
        [
            np.asarray(
                backtransform_derivative_fns[name](flat_eval_times),
                dtype=float,
            ).reshape(len(half_widths), len(nodes))
            for name in EXPECTED_REACTOR_COMPONENT_ORDER
        ],
        axis=-1,
    )
    increments = half_widths[:, np.newaxis] * np.einsum(
        "n,ins->is", weights, derivative_blocks
    )

    recovered = truth.copy()
    recovered[1:] = recovered[0] + np.cumsum(increments, axis=0)
    return recovered, truth


def _backtransform_derivative_matrix(
    times: np.ndarray,
    backtransform_derivative_fns: dict[str, object],
) -> np.ndarray:
    return np.column_stack(
        [
            np.asarray(
                backtransform_derivative_fns[name](jnp.asarray(times)),
                dtype=float,
            )
            for name in EXPECTED_REACTOR_COMPONENT_ORDER
        ]
    )


def _physical_state_indices(state_names: tuple[str, ...]) -> list[int]:
    indices = [state_names.index(name) for name in EXPECTED_REACTOR_COMPONENT_ORDER]
    assert (
        tuple(state_names[index] for index in indices)
        == EXPECTED_REACTOR_COMPONENT_ORDER
    )
    return indices


def _pointwise_mask(segment: RealSpaceSegment) -> np.ndarray:
    mask = np.ones(len(segment.rows), dtype=bool)
    mask[:DERIVATIVE_BOUNDARY_GUARD_POINTS] = False
    mask[-DERIVATIVE_BOUNDARY_GUARD_POINTS:] = False
    assert all(row["row_type"] != PRE_EVENT_ROW for row in segment.rows[:-1])
    return mask


def _finite_difference_stencil_indices(
    times: np.ndarray, mask: np.ndarray
) -> np.ndarray:
    indices = []
    for i in range(2, len(times) - 2):
        if not mask[i]:
            continue
        local_mask = mask[i - 2 : i + 3]
        local_steps = np.diff(times[i - 2 : i + 3])
        if np.all(local_mask) and np.allclose(
            local_steps,
            local_steps[0],
            rtol=DERIVATIVE_GRID_RTOL,
            atol=DERIVATIVE_GRID_ATOL,
        ):
            indices.append(i)
    return np.asarray(indices, dtype=int)


def _truth_finite_difference_derivatives(
    segment: RealSpaceSegment,
) -> tuple[np.ndarray, np.ndarray]:
    times = segment_times(segment)
    truth = segment_state_matrix(segment, EXPECTED_REACTOR_COMPONENT_ORDER)
    indices = _finite_difference_stencil_indices(times, _pointwise_mask(segment))
    derivatives = []
    for i in indices:
        # The stencil selector verified this whole neighborhood is uniform.
        dt = times[i + 1] - times[i]
        derivatives.append(
            (-truth[i + 2] + 8.0 * truth[i + 1] - 8.0 * truth[i - 1] + truth[i - 2])
            / (12.0 * dt)
        )
    if not derivatives:
        return times[indices], np.empty((0, len(EXPECTED_REACTOR_COMPONENT_ORDER)))
    return times[indices], np.vstack(derivatives)


def _rhs_truth_derivatives(
    segment: RealSpaceSegment,
    ordering: ProcessOrdering,
    control_splines: ControlSplines,
    rhs_ode: RhsOde,
    modeled_fvc_splines: tuple[PPoly, ...],
) -> tuple[np.ndarray, np.ndarray]:
    times = segment_times(segment)
    indices = np.flatnonzero(_pointwise_mask(segment))
    eval_times = times[indices]
    state_names = ordering.name_modeled_RMCs + ordering.name_modeled_PVs + ("volume",)
    truth = segment_state_matrix(segment, state_names)[indices]
    rate_splines = _segment_rate_splines(segment, ordering.name_modeled_rates)
    rate_values = jnp.asarray(
        np.column_stack(
            [
                np.asarray(spline(jnp.asarray(eval_times)), dtype=float)
                for spline in rate_splines
            ]
        )
    )
    controls = control_splines(jnp.asarray(eval_times))
    modeled_fvc_flows = jnp.asarray(
        np.column_stack(
            [
                np.asarray(spline(jnp.asarray(eval_times), nu=1), dtype=float)
                for spline in modeled_fvc_splines
            ]
        )
    )
    rhs_values = jax.vmap(
        lambda y, rates, control, fvc_flows: rhs_ode(
            y,
            rates,
            control,
            fvc_flows,
            jnp.zeros(0),
        )
    )(
        jnp.asarray(truth),
        rate_values,
        controls,
        modeled_fvc_flows,
    )
    physical_indices = _physical_state_indices(state_names)
    return eval_times, np.asarray(rhs_values, dtype=float)[:, physical_indices]


def _assert_pointwise_derivatives_match_references(
    segment: RealSpaceSegment,
    ordering: ProcessOrdering,
    control_splines: ControlSplines,
    rhs_ode: RhsOde,
    modeled_fvc_splines: tuple[PPoly, ...],
    backtransform_derivative_fns: dict[str, object],
) -> tuple[int, int]:
    fd_times, fd_reference = _truth_finite_difference_derivatives(segment)
    rhs_times, rhs_reference = _rhs_truth_derivatives(
        segment,
        ordering,
        control_splines,
        rhs_ode,
        modeled_fvc_splines,
    )
    assertion_errors = []
    if len(fd_times):
        try:
            np.testing.assert_allclose(
                _backtransform_derivative_matrix(
                    fd_times,
                    backtransform_derivative_fns,
                ),
                fd_reference,
                rtol=DERIVATIVE_FD_RTOL,
                atol=DERIVATIVE_FD_ATOL,
            )
        except AssertionError as exc:
            assertion_errors.append(f"dense-truth finite-difference residual:\n{exc}")
    if len(rhs_times):
        try:
            np.testing.assert_allclose(
                _backtransform_derivative_matrix(
                    rhs_times,
                    backtransform_derivative_fns,
                ),
                rhs_reference,
                rtol=DERIVATIVE_RHS_RTOL,
                atol=DERIVATIVE_RHS_ATOL,
            )
        except AssertionError as exc:
            assertion_errors.append(f"truth-state rhs_ode residual:\n{exc}")
    if assertion_errors:
        raise AssertionError("\n\n".join(assertion_errors))
    return len(fd_times), len(rhs_times)


def _integrate_segment(
    segment: RealSpaceSegment,
    ordering: ProcessOrdering,
    control_splines: ControlSplines,
    rhs_ode: RhsOde,
    modeled_fvc_splines: tuple[PPoly, ...],
) -> tuple[np.ndarray, np.ndarray]:
    times = segment_times(segment)
    state_names = ordering.name_modeled_RMCs + ordering.name_modeled_PVs + ("volume",)
    truth = segment_state_matrix(segment, state_names)
    rate_splines = _segment_rate_splines(segment, ordering.name_modeled_rates)
    assert ordering.name_modeled_SVCs == ()

    def rhs(t, y):
        t_jax = jnp.asarray(t)
        return np.asarray(
            rhs_ode(
                jnp.asarray(y),
                jnp.asarray([spline(t_jax) for spline in rate_splines]),
                control_splines(t_jax),
                jnp.asarray([spline(t_jax, nu=1) for spline in modeled_fvc_splines]),
                jnp.zeros(0),
            ),
            dtype=float,
        )

    solution = solve_ivp(
        rhs,
        (float(times[0]), float(times[-1])),
        truth[0],
        method="DOP853",
        rtol=SOLVER_RTOL,
        atol=SOLVER_ATOL,
        t_eval=times[1:],
    )
    if not solution.success:
        raise RuntimeError(solution.message)
    return np.vstack([truth[0], solution.y.T]), truth


def _assert_control_splines_match_dense_rows(
    segment: RealSpaceSegment,
    ordering: ProcessOrdering,
    control_splines: ControlSplines,
) -> None:
    times = segment_spline_times(segment)
    actual = np.asarray(control_splines(jnp.asarray(times)), dtype=float)
    flow_columns = (
        *(f"cum_{name}" for name in ordering.name_controlled_FVCs),
        *(f"cum_{name}" for name in ordering.name_controlled_SVCs),
    )
    expected_blocks = []
    if flow_columns:
        expected_blocks.append(
            np.column_stack(
                [
                    np.asarray(
                        _ppoly_from_segment_rows(segment, column)(
                            jnp.asarray(times),
                            nu=1,
                        ),
                        dtype=float,
                    )
                    for column in flow_columns
                ]
            )
        )
    if ordering.name_controlled_PVs:
        expected_blocks.append(
            segment_state_matrix(segment, ordering.name_controlled_PVs)
        )
    expected = np.column_stack(expected_blocks)
    np.testing.assert_allclose(
        actual,
        expected,
        rtol=CONTROL_RTOL,
        atol=CONTROL_ATOL,
    )


def _assert_modeled_fvc_splines_match_dense_rows(
    segment: RealSpaceSegment,
    ordering: ProcessOrdering,
    modeled_fvc_splines: tuple[PPoly, ...],
) -> None:
    times = segment_spline_times(segment)
    actual = np.column_stack(
        [
            np.asarray(spline(jnp.asarray(times), nu=1), dtype=float)
            for spline in modeled_fvc_splines
        ]
    )
    expected = segment_state_matrix(
        segment,
        [MODELED_FVC_FLOW_COLUMNS[name] for name in ordering.name_modeled_FVCs],
    )
    mask = np.ones(len(segment.rows), dtype=bool)
    if segment.starts_after_event:
        mask[0] = False
    if segment.rows[-1]["row_type"] == PRE_EVENT_ROW:
        mask[-1] = False
    # `base_feed` is stored as a continuous cumulative carrier. Event-time rate
    # kinks are represented only by neighboring cumulative samples, so its cubic
    # derivative returns a smooth through-slope there. The resulting localized
    # endpoint mismatch stays below reintegration sensitivity; check carrier
    # alignment on open segment spans.
    np.testing.assert_allclose(
        actual[mask],
        expected[mask],
        rtol=MODELED_FVC_FLOW_RTOL,
        atol=MODELED_FVC_FLOW_ATOL,
    )


def test_sim_1_real_space_mechanistic_rhs_reintegration():
    collection = load_process_collection_json(DATA_JSON)
    assert set(collection.processes) == EXPECTED_PROCESS_IDS
    rows_by_process = dense_rows_by_process(
        SIMULATION_DENSE_OUTPUT,
        EXPECTED_PROCESS_IDS,
    )

    for process_id, process in collection.processes.items():
        ordering = get_process_ordering(process)
        _assert_expected_sim_1_ordering(ordering)
        control_splines = get_control_splines(process, ordering)
        rhs_ode = build_rhs_ode(process, ordering)
        modeled_fvc_splines = _process_modeled_fvc_splines(
            process,
            ordering.name_modeled_FVCs,
        )
        backtransform_process = copy.deepcopy(process)
        _add_dense_pseudobatch_transform(backtransform_process, process_id)
        backtransform_state_splines = build_state_splines(
            backtransform_process,
            ordering,
        )
        backtransform_derivative_fns = {
            name: backtransform_state_splines[name].derivative()
            for name in EXPECTED_REACTOR_COMPONENT_ORDER
        }
        segments = build_real_space_segments(rows_by_process[process_id])
        assert len(segments) > 1
        n_fd_points = 0
        n_rhs_points = 0

        for segment in segments:
            _assert_control_splines_match_dense_rows(
                segment,
                ordering,
                control_splines,
            )
            _assert_modeled_fvc_splines_match_dense_rows(
                segment,
                ordering,
                modeled_fvc_splines,
            )
            integrated, truth = _integrate_segment(
                segment,
                ordering,
                control_splines,
                rhs_ode,
                modeled_fvc_splines,
            )
            np.testing.assert_allclose(
                integrated,
                truth,
                rtol=RHS_REINTEGRATION_RTOL,
                atol=RHS_REINTEGRATION_ATOL,
            )
            segment_fd_points, segment_rhs_points = (
                _assert_pointwise_derivatives_match_references(
                    segment,
                    ordering,
                    control_splines,
                    rhs_ode,
                    modeled_fvc_splines,
                    backtransform_derivative_fns,
                )
            )
            n_fd_points += segment_fd_points
            n_rhs_points += segment_rhs_points
            recovered, physical_truth = _integrate_backtransform_derivative_for_segment(
                segment,
                backtransform_derivative_fns,
            )
            np.testing.assert_allclose(
                recovered,
                physical_truth,
                rtol=BACKTRANSFORM_RTOL,
                atol=BACKTRANSFORM_ATOL,
            )

        assert n_fd_points > 0
        assert n_rhs_points > 0
