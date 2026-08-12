import os

os.environ.setdefault("JAX_ENABLE_X64", "true")

import diffrax  # noqa: E402
import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402
from bp_format.dataclasses import BioProcess  # noqa: E402
from bp_format.dataclasses import ProcessOrdering  # noqa: E402
from bp_format.mechanistic import ControlSplines  # noqa: E402
from bp_format.mechanistic import RhsOde  # noqa: E402
from bp_format.mechanistic import _timeseries_to_ppoly  # noqa: E402
from bp_format.mechanistic import build_rhs_ode  # noqa: E402
from bp_format.mechanistic import get_control_splines  # noqa: E402
from bp_format.mechanistic import get_process_ordering  # noqa: E402
from bp_format.serialization import load_process_collection  # noqa: E402
from bp_format.splines import make_cubic_ppoly  # noqa: E402
from bp_format.time_series import PPoly  # noqa: E402

from .real_space_segments import EXPECTED_PROCESS_IDS  # noqa: E402
from .real_space_segments import SIMULATION_DENSE_OUTPUT  # noqa: E402
from .real_space_segments import SIM_RESULTS_DIR  # noqa: E402
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
EXPECTED_MODELED_INFLOWS = ("base_feed",)
EXPECTED_CONTROLLED_INFLOWS = ("conti_feed",)
EXPECTED_MODELED_OUTFLOWS = ()
EXPECTED_CONTROLLED_OUTFLOWS = ()
MODELED_Inflow_FLOW_COLUMNS = {"base_feed": "base_flow_l_per_h"}
RHS_REINTEGRATION_RTOL = 1e-7
RHS_REINTEGRATION_ATOL = 5e-6
SOLVER_RTOL = 1e-10
SOLVER_ATOL = 1e-12
SOLVER_MAX_STEPS = 1_000_000
CONTROL_RTOL = 1e-12
CONTROL_ATOL = 1e-12
MODELED_Inflow_FLOW_RTOL = 5e-2
MODELED_Inflow_FLOW_ATOL = 2e-5


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


def _process_modeled_inflow_splines(
    process: BioProcess,
    inflow_names: tuple[str, ...],
) -> tuple[PPoly, ...]:
    return tuple(
        _timeseries_to_ppoly(process.volume.volume_changes[name].values)
        for name in inflow_names
    )


def _assert_expected_sim_1_ordering(ordering: ProcessOrdering) -> None:
    assert ordering.name_modeled_RMCs == EXPECTED_MODELED_RMCS
    assert ordering.name_modeled_PVs == EXPECTED_MODELED_PVS
    assert ordering.name_controlled_PVs == EXPECTED_CONTROLLED_PVS
    assert ordering.name_modeled_rates == EXPECTED_MODELED_RATES
    assert ordering.name_modeled_Inflows == EXPECTED_MODELED_INFLOWS
    assert ordering.name_controlled_Inflows == EXPECTED_CONTROLLED_INFLOWS
    assert ordering.name_modeled_Outflows == EXPECTED_MODELED_OUTFLOWS
    assert ordering.name_controlled_Outflows == EXPECTED_CONTROLLED_OUTFLOWS


def _integrate_segment(
    segment: RealSpaceSegment,
    ordering: ProcessOrdering,
    control_splines: ControlSplines,
    rhs_ode: RhsOde,
    modeled_inflow_splines: tuple[PPoly, ...],
) -> tuple[np.ndarray, np.ndarray]:
    times = segment_times(segment)
    state_names = ordering.name_modeled_RMCs + ordering.name_modeled_PVs + ("volume",)
    truth = segment_state_matrix(segment, state_names)
    rate_splines = _segment_rate_splines(segment, ordering.name_modeled_rates)
    no_modeled_outflow_flows = jnp.zeros(0)
    solve_times = jnp.asarray(times)
    assert ordering.name_modeled_Outflows == ()

    def rhs(t, y, args):
        rate_values = jnp.asarray([spline(t) for spline in rate_splines])
        modeled_inflow_flows = jnp.asarray(
            [spline(t, nu=1) for spline in modeled_inflow_splines]
        )
        return rhs_ode(
            y,
            rate_values,
            control_splines(t),
            modeled_inflow_flows,
            no_modeled_outflow_flows,
        )

    @jax.jit
    def solve(y0):
        return diffrax.diffeqsolve(
            diffrax.ODETerm(rhs),
            diffrax.Dopri8(),
            t0=float(times[0]),
            t1=float(times[-1]),
            dt0=None,
            y0=y0,
            saveat=diffrax.SaveAt(ts=solve_times),
            stepsize_controller=diffrax.PIDController(
                rtol=SOLVER_RTOL,
                atol=SOLVER_ATOL,
            ),
            max_steps=SOLVER_MAX_STEPS,
        )

    solution = solve(jnp.asarray(truth[0]))
    return np.asarray(solution.ys, dtype=float), truth


def _assert_control_splines_match_dense_rows(
    segment: RealSpaceSegment,
    ordering: ProcessOrdering,
    control_splines: ControlSplines,
) -> None:
    times = segment_spline_times(segment)
    actual = np.asarray(control_splines(jnp.asarray(times)), dtype=float)
    flow_columns = (
        *(f"cum_{name}" for name in ordering.name_controlled_Inflows),
        *(f"cum_{name}" for name in ordering.name_controlled_Outflows),
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


def _assert_modeled_inflow_splines_match_dense_rows(
    segment: RealSpaceSegment,
    ordering: ProcessOrdering,
    modeled_inflow_splines: tuple[PPoly, ...],
) -> None:
    times = segment_spline_times(segment)
    actual = np.column_stack(
        [
            np.asarray(spline(jnp.asarray(times), nu=1), dtype=float)
            for spline in modeled_inflow_splines
        ]
    )
    expected = segment_state_matrix(
        segment,
        [MODELED_Inflow_FLOW_COLUMNS[name] for name in ordering.name_modeled_Inflows],
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
        rtol=MODELED_Inflow_FLOW_RTOL,
        atol=MODELED_Inflow_FLOW_ATOL,
    )


def test_sim_1_real_space_mechanistic_rhs_reintegration():
    collection = load_process_collection(DATA_JSON)
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
        modeled_inflow_splines = _process_modeled_inflow_splines(
            process,
            ordering.name_modeled_Inflows,
        )
        segments = build_real_space_segments(rows_by_process[process_id])
        assert len(segments) > 1

        for segment in segments:
            _assert_control_splines_match_dense_rows(
                segment,
                ordering,
                control_splines,
            )
            _assert_modeled_inflow_splines_match_dense_rows(
                segment,
                ordering,
                modeled_inflow_splines,
            )
            integrated, truth = _integrate_segment(
                segment,
                ordering,
                control_splines,
                rhs_ode,
                modeled_inflow_splines,
            )
            np.testing.assert_allclose(
                integrated,
                truth,
                rtol=RHS_REINTEGRATION_RTOL,
                atol=RHS_REINTEGRATION_ATOL,
            )
