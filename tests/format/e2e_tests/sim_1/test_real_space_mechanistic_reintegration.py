import os

os.environ.setdefault("JAX_ENABLE_X64", "true")

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
from bp_format.serialization import load_process_collection_json  # noqa: E402
from bp_format.splines import make_cubic_ppoly  # noqa: E402
from bp_format.time_series import PPoly  # noqa: E402
from scipy.integrate import solve_ivp  # noqa: E402

from .cstar_helpers import EXPECTED_PROCESS_IDS  # noqa: E402
from .cstar_helpers import SIMULATION_DENSE_OUTPUT  # noqa: E402
from .cstar_helpers import SIM_RESULTS_DIR  # noqa: E402
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


def _assert_expected_sim_1_ordering(ordering: ProcessOrdering) -> None:
    assert ordering.name_modeled_RMCs == EXPECTED_MODELED_RMCS
    assert ordering.name_modeled_PVs == EXPECTED_MODELED_PVS
    assert ordering.name_controlled_PVs == EXPECTED_CONTROLLED_PVS
    assert ordering.name_modeled_rates == EXPECTED_MODELED_RATES
    assert ordering.name_modeled_FVCs == EXPECTED_MODELED_FVCS
    assert ordering.name_controlled_FVCs == EXPECTED_CONTROLLED_FVCS
    assert ordering.name_modeled_SVCs == EXPECTED_MODELED_SVCS
    assert ordering.name_controlled_SVCs == EXPECTED_CONTROLLED_SVCS


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
        segments = build_real_space_segments(rows_by_process[process_id])
        assert len(segments) > 1

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
