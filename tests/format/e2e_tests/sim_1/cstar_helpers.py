import csv
from pathlib import Path

import bp_format as bp
import jax.numpy as jnp
import numpy as np

SIM_1_DIR = Path(__file__).resolve().parent
SIM_RESULTS_DIR = SIM_1_DIR / "sim_results"
DIAGNOSTIC_PLOTS_DIR = SIM_1_DIR / "diagnostic_plots"
SIMULATION_DENSE_OUTPUT = SIM_RESULTS_DIR / "simulation_dense_output.csv"
EVENTS_OUTPUT = SIM_RESULTS_DIR / "simulation_events.csv"
# Canonical CSV artifacts compared numerically (with tolerance) against a fresh
# simulation run. Plot PNGs are intentionally NOT compared: their bytes are not
# reproducible across matplotlib/freetype versions, and `write_simulation_plots`
# already exercises the plotting path.
CANONICAL_ARTIFACTS = {
    Path("simulation_dense_output.csv"): SIMULATION_DENSE_OUTPUT,
    Path("simulation_events.csv"): EVENTS_OUTPUT,
}
EXPECTED_PROCESS_IDS = {"sim_1_run_1", "sim_1_run_2"}
EXPECTED_REACTOR_COMPONENT_ORDER = (
    "biomass",
    "product_extracellular",
    "product_intracellular",
    "dead_cells",
    "glucose",
    "glutamine",
    "lactate",
    "ammonia",
)
CSTAR_SPLINE_SMOOTHING_S = 0.0
EVENT_TIME_ATOL = 1e-12
# Required columns of the dense simulation CSV: simulator-integrated cumulative
# feed volumes per stream, used as independent ground truth for feed-correction.
CUMULATIVE_FEED_COLUMNS = (
    "cum_conti_feed",
    "cum_bolus_feed",
    "cum_base_feed",
)
# Inert pseudobatch tracer state columns. The tracers are real reactor-medium
# species (also parsed into the JSON), but the A/C oracle reads them straight from
# the dense CSV: a closed-form, integrator-sourced ground truth for the public ADF
# / feed-correction carriers (checks A and C in the dense c* oracle test).
TRACER_COLUMNS = (
    "tracer_unfed",
    "tracer_fed",
)


def fit_cstar_timeseries_from_values(
    component_name: str,
    times: np.ndarray,
    cstar_values: np.ndarray,
    *,
    source: str,
) -> bp.TimeSeries:
    fitted = bp.splines.fit_timeseries_spline(
        bp.TimeSeries(times=jnp.asarray(times), values=jnp.asarray(cstar_values)),
        smoothing_s=CSTAR_SPLINE_SMOOTHING_S,
    )
    metadata = dict(fitted.metadata or {})
    metadata["transform"] = {
        "name": "pseudo_batch",
        "component": component_name,
        "source": source,
    }
    return bp.TimeSeries(
        times=fitted.times,
        values=fitted.values,
        jump_times=fitted.jump_times,
        breaks=fitted.breaks,
        coeffs=fitted.coeffs,
        segment_start_piece_idx=fitted.segment_start_piece_idx,
        continuity_side=fitted.continuity_side,
        metadata=metadata,
        dtype=fitted.dtype,
    )


def event_jump_times(process: bp.BioProcess) -> np.ndarray:
    assert process.volume.total_volume is not None
    return np.asarray(process.volume.total_volume.jump_times, dtype=float)


def _dense_reactor_rows(
    process_id: str,
    max_time: float,
    *,
    row_type: str | None,
) -> list[dict[str, str]]:
    with SIMULATION_DENSE_OUTPUT.open(newline="") as handle:
        return [
            row
            for row in csv.DictReader(handle)
            if row["process_id"] == process_id
            and (row_type is None or row["row_type"] == row_type)
            and float(row["time"]) <= max_time
        ]


def _reactor_rows_to_arrays(rows: list[dict[str, str]]) -> dict[str, np.ndarray]:
    return {
        "time": np.asarray([float(row["time"]) for row in rows], dtype=float),
        "volume": np.asarray([float(row["volume"]) for row in rows], dtype=float),
        **{
            name: np.asarray([float(row[name]) for row in rows], dtype=float)
            for name in EXPECTED_REACTOR_COMPONENT_ORDER
        },
        **{
            column: np.asarray([float(row[column]) for row in rows], dtype=float)
            for column in CUMULATIVE_FEED_COLUMNS
        },
        **{
            column: np.asarray([float(row[column]) for row in rows], dtype=float)
            for column in TRACER_COLUMNS
        },
    }


def _q_rate_rows_to_arrays(rows: list[dict[str, str]]) -> dict[str, np.ndarray]:
    return {
        "time": np.asarray([float(row["time"]) for row in rows], dtype=float),
        **{
            name: np.asarray([float(row[f"q_{name}"]) for row in rows], dtype=float)
            for name in EXPECTED_REACTOR_COMPONENT_ORDER
        },
    }


def dense_reactor_reference(process_id: str, max_time: float) -> dict[str, np.ndarray]:
    rows = _dense_reactor_rows(process_id, max_time, row_type=None)
    rows = [row for row in rows if row["row_type"] != "offline"]
    return _reactor_rows_to_arrays(rows)


def dense_online_reactor_reference(
    process_id: str,
    max_time: float,
) -> dict[str, np.ndarray]:
    rows = _dense_reactor_rows(process_id, max_time, row_type="online")
    return _reactor_rows_to_arrays(rows)


def dense_online_q_rate_reference(
    process_id: str,
    max_time: float,
) -> dict[str, np.ndarray]:
    rows = _dense_reactor_rows(process_id, max_time, row_type="online")
    return _q_rate_rows_to_arrays(rows)


def dense_event_pair_reference(
    process_id: str,
    max_time: float,
) -> list[tuple[dict[str, float], dict[str, float]]]:
    rows = _dense_reactor_rows(process_id, max_time, row_type=None)
    event_rows: dict[float, dict[str, dict[str, str]]] = {}
    for row in rows:
        if row["row_type"] not in {"pre-event", "post-event"}:
            continue
        event_rows.setdefault(float(row["time"]), {})[row["row_type"]] = row

    pairs = []
    for time in sorted(event_rows):
        pair = event_rows[time]
        assert set(pair) == {"pre-event", "post-event"}
        pairs.append(
            (
                _event_row_to_floats(pair["pre-event"]),
                _event_row_to_floats(pair["post-event"]),
            )
        )
    return pairs


def _event_row_to_floats(row: dict[str, str]) -> dict[str, float]:
    return {
        "time": float(row["time"]),
        "volume": float(row["volume"]),
        **{name: float(row[name]) for name in EXPECTED_REACTOR_COMPONENT_ORDER},
    }
