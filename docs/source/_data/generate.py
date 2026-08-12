"""Generate the demo datasets the documentation is written against.

Two datasets, deterministic, regenerated on every ``docs_rebuild.sh``. Together
they are the site's two-organism spine: one bacterial, one mammalian, so every
page can pick the shape (batch / fed-batch) and flavor (fast, simple / slow,
byproduct-forming) that fits what it needs to demonstrate.

``demo_batch``
    The beginner spine. Three *batch* E. coli runs on glucose — biomass,
    glucose, product, and **no volume changes at all**. Used by the quickstart
    and every tutorial. Also written out as raw CSVs, because Tutorial 1 starts
    from CSVs like the ones a user actually has.

``demo_fedbatch``
    A CHO-like mammalian fed-batch: slower growth, a days-not-hours timescale,
    a constant continuous feed, two boluses, sampling events that remove
    volume, one controlled process variable, and lactate as a byproduct
    alongside product (mAb-flavored) formation.

Both are simulated on **amounts** and converted to concentrations at the end, so
volume changes can never silently corrupt a mass balance. Substrate uptake is
gated by the same Monod term as growth, so it tapers at depletion rather than
being clipped afterwards.

Run directly to regenerate::

    python source/_data/generate.py
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

import bp_format as bp
from bp_format.time_series import TimeSeries

OUT = Path(__file__).parent / "out"

# --- bacterial kinetics (demo_batch only) -----------------------------------
MU_MAX = 0.45      # 1/h
KS = 0.05          # g/L   — E. coli glucose Ks is genuinely this small
Y_XS = 0.45        # g biomass / g glucose
M_S = 0.02         # g glucose / g biomass / h   (maintenance)
ALPHA = 0.08       # g product / g biomass       (growth-associated)
BETA = 0.006       # g product / g biomass / h   (non-growth-associated)

NOISE_REL = 0.03   # 3 % relative measurement noise


def _rates(x: float, s: float) -> tuple[float, float, float]:
    """Specific rates (1/h and g/g/h) at biomass ``x`` and glucose ``s``."""
    sigma = s / (KS + s) if s > 0.0 else 0.0
    mu = MU_MAX * sigma
    q_s = mu / Y_XS + M_S * sigma      # maintenance tapers with substrate too
    q_p = ALPHA * mu + BETA * sigma
    return mu, q_s, q_p


def _noisy(rng: np.random.Generator, values: np.ndarray, floor: float) -> np.ndarray:
    """Multiplicative measurement noise with a small absolute floor."""
    noise = rng.normal(1.0, NOISE_REL, size=values.shape)
    return np.maximum(values * noise, floor)


# ===========================================================================
# demo_batch
# ===========================================================================

BATCH_RUNS = {
    #  name     S0     X0
    "run_1": (10.0, 0.10),
    "run_2": (15.0, 0.08),
    "run_3": (20.0, 0.12),
}
BATCH_END = 14.0
BATCH_SAMPLE_TIMES = np.arange(0.0, BATCH_END + 0.5, 1.0)


def _simulate_batch(s0: float, x0: float) -> dict[str, np.ndarray]:
    """RK4 on concentrations — volume is constant, so amounts add nothing."""
    dt = 0.002
    n = int(round(BATCH_END / dt)) + 1
    t_grid = np.linspace(0.0, BATCH_END, n)
    y = np.array([x0, s0, 0.0])

    def deriv(state: np.ndarray) -> np.ndarray:
        x, s, _ = state
        mu, q_s, q_p = _rates(x, s)
        return np.array([mu * x, -q_s * x, q_p * x])

    traj = np.empty((n, 3))
    traj[0] = y
    for i in range(1, n):
        k1 = deriv(y)
        k2 = deriv(y + 0.5 * dt * k1)
        k3 = deriv(y + 0.5 * dt * k2)
        k4 = deriv(y + dt * k3)
        y = np.maximum(y + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4), 0.0)
        traj[i] = y

    return {
        "biomass": np.interp(BATCH_SAMPLE_TIMES, t_grid, traj[:, 0]),
        "glucose": np.interp(BATCH_SAMPLE_TIMES, t_grid, traj[:, 1]),
        "product": np.interp(BATCH_SAMPLE_TIMES, t_grid, traj[:, 2]),
    }


def _batch_process(name: str, meas: dict[str, np.ndarray]) -> bp.BioProcess:
    components = {
        species: bp.ReactorMediumComponent(
            name=species,
            unit="g/L",
            concentration=TimeSeries(
                times=np.asarray(BATCH_SAMPLE_TIMES, dtype=float),
                values=np.asarray(values, dtype=float),
            ),
            bounds=(0.0, None),
        )
        for species, values in meas.items()
    }
    return bp.BioProcess(
        metadata=bp.BioProcessMetadata(
            name=name,
            process_type="batch",
            notes="Simulated E. coli batch culture on glucose (documentation demo).",
        ),
        time_axis=bp.TimeAxis(
            unit="h", start=0.0, end=BATCH_END, time_reference="inoculation"
        ),
        # A true batch: no feeds, no boluses, no sampling volume.
        volume=bp.Volume(initial_volume=1.0, unit="L"),
        reactor_medium=bp.ReactorMedium(
            name="defined_medium", density=1.0, density_unit="kg/L",
            components=components,
        ),
        # biological_ode is left None on purpose -> bp-format auto-generates
        #   q_biomass, q_glucose, q_product  with  d<c>/dt = q_<c> * biomass
    )


def build_demo_batch() -> None:
    out = OUT / "demo_batch"
    (out / "raw").mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(20260806)
    processes, csv_rows = {}, []

    for run, (s0, x0) in BATCH_RUNS.items():
        truth = _simulate_batch(s0, x0)
        meas = {
            "biomass": _noisy(rng, truth["biomass"], 0.01),
            "glucose": _noisy(rng, truth["glucose"], 0.0),
            "product": _noisy(rng, truth["product"], 0.0),
        }
        # t=0 is measured exactly: bp-train requires a t0 value for every target.
        for species in meas:
            meas[species][0] = truth[species][0]

        processes[run] = _batch_process(run, meas)
        for i, t in enumerate(BATCH_SAMPLE_TIMES):
            csv_rows.append(
                f"{run},{t:.1f},{meas['biomass'][i]:.4f},"
                f"{meas['glucose'][i]:.4f},{meas['product'][i]:.4f}"
            )

    header = "run,time_h,biomass_gL,glucose_gL,product_gL"
    (out / "raw" / "offline.csv").write_text("\n".join([header, *csv_rows]) + "\n")

    collection = bp.BioProcessCollection(
        case_id="demo_batch",
        organism="Escherichia coli",
        citation="Simulated data — bp-docs demo, not a real experiment.",
        processes=processes,
    )
    bp.serialization.save_process_collection(collection, out / "data.json")

    (out / "ground_truth.json").write_text(json.dumps({
        "mu_max": MU_MAX, "Ks": KS, "Y_XS": Y_XS, "m_s": M_S,
        "alpha": ALPHA, "beta": BETA,
    }, indent=2) + "\n")


# --- mammalian kinetics (demo_fedbatch only, CHO-like) -----------------------
# Slower growth, glucose diverted mostly to lactate (Warburg-like overflow
# metabolism), product formation dominated by the non-growth-associated term,
# the usual pattern for mAb production that continues into stationary phase.
MAM_MU_MAX = 0.032   # 1/h   (~22 h doubling time)
MAM_KS = 0.4         # g/L   glucose Ks — mammalian cells are far less glucose-avid
MAM_Y_XS = 0.12      # g biomass / g glucose
MAM_M_S = 0.0015     # g glucose / g biomass / h   (maintenance)
MAM_Y_LAC = 0.28     # g lactate / g glucose consumed
MAM_ALPHA = 0.004    # g product / g biomass       (growth-associated)
MAM_BETA = 0.0009    # g product / g biomass / h   (non-growth-associated)


def _rates_mammalian(x: float, s: float) -> tuple[float, float, float, float]:
    """Specific rates (1/h and g/g/h) at biomass ``x`` and glucose ``s``."""
    sigma = s / (MAM_KS + s) if s > 0.0 else 0.0
    mu = MAM_MU_MAX * sigma
    q_s = mu / MAM_Y_XS + MAM_M_S * sigma
    q_lac = MAM_Y_LAC * q_s
    q_p = MAM_ALPHA * mu + MAM_BETA * sigma
    return mu, q_s, q_lac, q_p


# ===========================================================================
# demo_fedbatch
# ===========================================================================

FB_END = 240.0                  # 10 days: real CHO fed-batch runs are ~10-14 days
FB_SAMPLE_TIMES = np.arange(0.0, FB_END + 12.0, 24.0)   # one offline sample per day
FB_BOLUS_TIMES = np.array([120.0, 192.0])   # day 5, day 8
FB_BOLUS_VOLUME = 0.05          # L per bolus
FB_SAMPLE_VOLUME = 0.008        # L removed per offline sample
FB_FEED_START = 48.0            # day 2
FB_FEED_C_GLUCOSE = 300.0       # g/L in both the continuous feed and the bolus
FB_F0 = 0.00055                 # L/h, constant once feeding starts


def _feed_rate(t: float) -> float:
    return 0.0 if t < FB_FEED_START else FB_F0


def _simulate_fedbatch() -> dict[str, np.ndarray]:
    """RK4 on **amounts** (g) plus volume, with discrete events applied between
    segments. Sample first, then bolus, at a coincident timestamp."""
    dt = 0.01
    n = int(round(FB_END / dt)) + 1
    t_grid = np.linspace(0.0, FB_END, n)

    v = 1.0
    y = np.array([0.15 * v, 8.0 * v, 0.0, 0.0])    # mX, mS, mLac, mP  (g)
    v_in_cum = 0.0

    rec = {k: np.empty(n) for k in
           ("biomass", "glucose", "lactate", "product", "volume", "v_in")}

    def deriv(state: np.ndarray, vol: float, t: float) -> np.ndarray:
        m_x, m_s, _m_lac, _m_p = state
        x, s = m_x / vol, m_s / vol
        mu, q_s, q_lac, q_p = _rates_mammalian(x, s)
        f = _feed_rate(t)
        return np.array([mu * m_x, -q_s * m_x + f * FB_FEED_C_GLUCOSE,
                         q_lac * m_x, q_p * m_x])

    def record(i: int) -> None:
        rec["biomass"][i] = y[0] / v
        rec["glucose"][i] = y[1] / v
        rec["lactate"][i] = y[2] / v
        rec["product"][i] = y[3] / v
        rec["volume"][i] = v
        rec["v_in"][i] = v_in_cum

    record(0)
    for i in range(1, n):
        t = t_grid[i - 1]
        k1 = deriv(y, v, t)
        k2 = deriv(y + 0.5 * dt * k1, v, t + 0.5 * dt)
        k3 = deriv(y + 0.5 * dt * k2, v, t + 0.5 * dt)
        k4 = deriv(y + dt * k3, v, t + dt)
        y = np.maximum(y + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4), 0.0)

        f_mid = _feed_rate(t + 0.5 * dt)
        v += f_mid * dt
        v_in_cum += f_mid * dt

        t_now = t_grid[i]
        # 1. sampling: well-mixed removal — concentrations unchanged, V drops
        if np.any(np.isclose(t_now, FB_SAMPLE_TIMES, atol=dt / 2)) and t_now > 0.0:
            y *= (v - FB_SAMPLE_VOLUME) / v
            v -= FB_SAMPLE_VOLUME
        # 2. bolus: dilute from the post-sample volume, then add the fed mass
        if np.any(np.isclose(t_now, FB_BOLUS_TIMES, atol=dt / 2)):
            y[1] += FB_BOLUS_VOLUME * FB_FEED_C_GLUCOSE
            v += FB_BOLUS_VOLUME

        record(i)

    rec["t"] = t_grid
    return rec


def build_demo_fedbatch() -> None:
    out = OUT / "demo_fedbatch"
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(20260807)

    sim = _simulate_fedbatch()
    t_grid = sim["t"]

    def at_samples(key: str) -> np.ndarray:
        # sample the *pre-event* value: offline rows are drawn before feeding
        idx = [int(np.argmin(np.abs(t_grid - t))) for t in FB_SAMPLE_TIMES]
        idx = [max(i - 1, 0) if t > 0 else i for i, t in zip(idx, FB_SAMPLE_TIMES)]
        return sim[key][idx]

    meas = {
        "biomass": _noisy(rng, at_samples("biomass"), 0.01),
        "glucose": _noisy(rng, at_samples("glucose"), 0.0),
        "lactate": _noisy(rng, at_samples("lactate"), 0.0),
        "product": _noisy(rng, at_samples("product"), 0.0),
    }
    for species in meas:
        meas[species][0] = at_samples(species)[0]

    components = {
        species: bp.ReactorMediumComponent(
            name=species, unit="g/L",
            concentration=TimeSeries(times=FB_SAMPLE_TIMES.astype(float),
                                     values=values.astype(float)),
            bounds=(0.0, None),
        )
        for species, values in meas.items()
    }

    feed_medium = bp.FeedMedium(
        name="glucose_feed", density=1.0, density_unit="kg/L",
        components={
            # Every reactor species needs an explicit feed concentration —
            # "absent" and "unrecorded" must not be confusable.
            "glucose": bp.FeedMediumComponent(
                name="glucose", unit="g/L",
                concentration=bp.StaticVariable(FB_FEED_C_GLUCOSE)),
            "biomass": bp.FeedMediumComponent(
                name="biomass", unit="g/L", concentration=bp.StaticVariable(0.0)),
            "lactate": bp.FeedMediumComponent(
                name="lactate", unit="g/L", concentration=bp.StaticVariable(0.0)),
            "product": bp.FeedMediumComponent(
                name="product", unit="g/L", concentration=bp.StaticVariable(0.0)),
        },
    )

    # Continuous feed: stored as a CUMULATIVE volume trace, not a rate.
    online_t = np.arange(0.0, FB_END + 0.25, 0.5)
    v_in_online = np.interp(online_t, t_grid, sim["v_in"])

    volume_changes = {
        "glucose_feed": bp.FeedVolumeChange(
            name="glucose_feed", unit="L", is_controlled=True, is_continuous=True,
            values=TimeSeries(times=online_t, values=v_in_online),
            feed_medium=feed_medium,
        ),
        "glucose_bolus": bp.FeedVolumeChange(
            name="glucose_bolus", unit="L", is_controlled=True, is_continuous=False,
            values=TimeSeries(times=FB_BOLUS_TIMES.astype(float),
                              values=np.full(FB_BOLUS_TIMES.shape, FB_BOLUS_VOLUME)),
            feed_medium=feed_medium,
        ),
        "sampling": bp.SampleVolumeChange(
            name="sampling", unit="L", is_controlled=True, is_continuous=False,
            # samples are negative by convention; t=0 is not a draw
            values=TimeSeries(times=FB_SAMPLE_TIMES[1:].astype(float),
                              values=np.full(FB_SAMPLE_TIMES[1:].shape,
                                             -FB_SAMPLE_VOLUME)),
        ),
    }

    # One controlled process variable, measured online.
    do_pct = 40.0 + 8.0 * np.exp(-online_t / 6.0) + rng.normal(0.0, 0.4, online_t.shape)
    process_variables = {
        "dissolved_oxygen": bp.ProcessVariable(
            name="dissolved_oxygen", unit="%", is_controlled=True,
            values=TimeSeries(times=online_t, values=do_pct),
            bounds=(0.0, 100.0),
        )
    }

    process = bp.BioProcess(
        metadata=bp.BioProcessMetadata(
            name="fedbatch_1", process_type="fed_batch",
            notes="Simulated CHO-like mammalian fed-batch: constant feed + "
                  "2 boluses + sampling, lactate as a byproduct.",
        ),
        time_axis=bp.TimeAxis(unit="h", start=0.0, end=FB_END,
                              time_reference="inoculation"),
        volume=bp.Volume(initial_volume=1.0, unit="L",
                         volume_changes=volume_changes),
        reactor_medium=bp.ReactorMedium(name="defined_medium", density=1.0,
                                        density_unit="kg/L", components=components),
        process_variables=process_variables,
    )

    collection = bp.BioProcessCollection(
        case_id="demo_fedbatch",
        organism="Chinese hamster ovary (CHO) cell line",
        citation="Simulated data — bp-docs demo, not a real experiment.",
        processes={"fedbatch_1": process},
    )
    bp.serialization.save_process_collection(collection, out / "data.json")

    (out / "ground_truth.json").write_text(json.dumps({
        "mu_max": MAM_MU_MAX, "Ks": MAM_KS, "Y_XS": MAM_Y_XS, "m_s": MAM_M_S,
        "Y_lac": MAM_Y_LAC, "alpha": MAM_ALPHA, "beta": MAM_BETA,
        "final_volume": float(sim["volume"][-1]),
    }, indent=2) + "\n")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    build_demo_batch()
    build_demo_fedbatch()
    print(f"demo datasets written to {OUT}")


if __name__ == "__main__":
    main()
