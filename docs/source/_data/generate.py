"""Generate the demo datasets the documentation is written against.

Ten datasets, deterministic, regenerated on every ``docs_rebuild.sh``.
``demo_batch``/``demo_fedbatch`` are the site's two-organism spine: one
bacterial, one mammalian, so every page can pick the shape (batch / fed-batch)
and flavor (fast, simple / slow, byproduct-forming) that fits what it needs to
demonstrate. ``demo_products``, ``demo_ecoli_fba`` and ``demo_ecoli_blend`` are
separate, single-purpose families, each for one Gallery page whose teaching
point neither spine shape can carry.

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

``demo_products``
    Five batch products (four "historical," one "target"), same shape as
    ``demo_batch``, kinetics clustered around the target's own phenotype but
    each distinguishable. The target's two training runs sit in a narrow
    initial-condition slice; its two held-out runs sit at the extremes of a
    much wider design space the historical products actually cover.

``demo_ecoli_fba``
    Three batch E. coli runs (biomass, glucose, acetate) forward-simulated
    from a real, frozen surrogate-FBA model on ``e_coli_core.xml`` (Orth,
    Fleming & Palsson 2010), not from hand-written kinetics. For
    ``gallery/fba_hyb.md``.

``demo_ecoli_blend``
    Four batch E. coli runs (biomass, glucose, acetate, succinate) from the
    same surrogate, each under a different ``media_blend_fraction``. For
    ``gallery/pls_dfba.md``, which builds on ``fba_hyb.md``.

``demo_optfed``
    Six fed-batch runs (biomass, glucose, product) forward-simulated from a
    non-competitive-inhibition Michaelis-Menten rate law with an Eyring-equation
    temperature dependence (Schlögl et al. 2024, OptFed), under a central
    -composite-design-inspired spread of exponential feed rates and constant
    per-run ``temperature``. For ``gallery/optfed.md``.

``demo_glutamine_decay``
    Three CHO-like batch runs (biomass, Gln, NH4) over a 120 h window, where
    one rate constant (``r_Gln``, the real value from Ulonska et al. 2018)
    feeds two coupled derivatives at once: it drains Gln and, in the same
    equation, produces NH4. For ``gallery/glutamine_decay.md``.

``demo_spline_jump``
    One species, first-order decay, one feed bolus part-way through that jumps
    mass and volume together. Both phases (before and after the bolus) are
    closed-form exponential decay at constant volume, so this is the one
    dataset simulated analytically rather than by RK4: ``ground_truth.json``'s
    parameters reconstruct the exact dense curve, not an approximation of it.
    For ``gallery/pseudobatch_splines.md``, which takes 5 measurements
    straddling the jump and recovers the underlying curve from them.

``demo_continuous_overflow``
    One noiseless process that moves from batch through a one-hour pause and
    fed-batch fill into continuous culture. Equal prescribed feed and overflow
    rates hold the reactor at 1 L during the continuous phase. For
    ``gallery/continuous_overflow.md``.

``demo_modeled_pv``
    Three batch-with-one-bolus CHO-like runs, two independent first-order
    states: ``biomass`` (an ordinary modeled reactor component) and
    ``glyco_frac`` (a modeled, uncontrolled process variable standing in for a
    glycosylation-fraction quality attribute). One large dilution bolus lands
    midway through each run. Both states are simulated in closed form, so the
    dilution's effect on ``biomass`` and its absence on ``glyco_frac`` are
    exact, not RK4 approximations. For ``gallery/modeled_pv.md``.

All datasets but the last are simulated on **amounts** and converted to
concentrations at the end, so volume changes can never silently corrupt a mass
balance. ``demo_glutamine_decay`` is, like ``demo_batch``, a true batch with no
volume changes at all, so it is simulated directly on concentrations instead.
Substrate uptake is gated by the same Monod term as growth (``demo_batch``/
``demo_fedbatch``/``demo_products``) or the surrogate's own glucose-uptake
term (``demo_ecoli_fba``/``demo_ecoli_blend``), so it tapers at depletion
rather than being clipped afterwards. ``demo_spline_jump`` needs neither: its
whole point is a closed-form ground truth to fit against.

Run directly to regenerate::

    python source/_data/generate.py
"""

from __future__ import annotations

import json
from itertools import pairwise
from pathlib import Path

import numpy as np

import hybrax.format as hxf
from hybrax.format.time_series import PPoly, TimeSeries

OUT = Path(__file__).parent / "out"

# --- bacterial kinetics (demo_batch only) -----------------------------------
MU_MAX = 0.45  # 1/h
KS = 0.05  # g/L   — E. coli glucose Ks is genuinely this small
Y_XS = 0.45  # g biomass / g glucose
M_S = 0.02  # g glucose / g biomass / h   (maintenance)
ALPHA = 0.08  # g product / g biomass       (growth-associated)
BETA = 0.006  # g product / g biomass / h   (non-growth-associated)

NOISE_REL = 0.03  # 3 % relative measurement noise


def _rates(x: float, s: float) -> tuple[float, float, float]:
    """Specific rates (1/h and g/g/h) at biomass ``x`` and glucose ``s``."""
    sigma = s / (KS + s) if s > 0.0 else 0.0
    mu = MU_MAX * sigma
    q_s = mu / Y_XS + M_S * sigma  # maintenance tapers with substrate too
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


def _batch_process(name: str, meas: dict[str, np.ndarray]) -> hxf.BioProcess:
    components = {
        species: hxf.ReactorMediumComponent(
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
    return hxf.BioProcess(
        metadata=hxf.BioProcessMetadata(
            name=name,
            process_type="batch",
            notes="Simulated E. coli batch culture on glucose (documentation demo).",
        ),
        time_axis=hxf.TimeAxis(
            unit="h", start=0.0, end=BATCH_END, time_reference="inoculation"
        ),
        # A true batch: no feeds, no boluses, no sampling volume.
        volume=hxf.Volume(initial_volume=1.0, unit="L"),
        reactor_medium=hxf.ReactorMedium(
            name="defined_medium",
            density=1.0,
            density_unit="kg/L",
            components=components,
        ),
        # biological_ode is left None on purpose -> hybrax.format auto-generates
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
        # t=0 is measured exactly: hybrax.train requires a t0 value for every target.
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

    collection = hxf.BioProcessCollection(
        case_id="demo_batch",
        organism="Escherichia coli",
        citation="Simulated data — bp-docs demo, not a real experiment.",
        processes=processes,
    )
    hxf.serialization.save_process_collection(collection, out / "data.json")

    (out / "ground_truth.json").write_text(
        json.dumps(
            {
                "mu_max": MU_MAX,
                "Ks": KS,
                "Y_XS": Y_XS,
                "m_s": M_S,
                "alpha": ALPHA,
                "beta": BETA,
            },
            indent=2,
        )
        + "\n"
    )


# --- mammalian kinetics (demo_fedbatch only, CHO-like) -----------------------
# Slower growth, glucose diverted mostly to lactate (Warburg-like overflow
# metabolism), product formation dominated by the non-growth-associated term,
# the usual pattern for mAb production that continues into stationary phase.
MAM_MU_MAX = 0.032  # 1/h   (~22 h doubling time)
MAM_KS = 0.4  # g/L   glucose Ks — mammalian cells are far less glucose-avid
MAM_Y_XS = 0.12  # g biomass / g glucose
MAM_M_S = 0.0015  # g glucose / g biomass / h   (maintenance)
MAM_Y_LAC = 0.28  # g lactate / g glucose consumed
MAM_ALPHA = 0.004  # g product / g biomass       (growth-associated)
MAM_BETA = 0.0009  # g product / g biomass / h   (non-growth-associated)


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

FB_END = 240.0  # 10 days: real CHO fed-batch runs are ~10-14 days
FB_SAMPLE_TIMES = np.arange(0.0, FB_END + 12.0, 24.0)  # one offline sample per day
FB_BOLUS_TIMES = np.array([120.0, 192.0])  # day 5, day 8
FB_BOLUS_VOLUME = 0.05  # L per bolus
FB_SAMPLE_VOLUME = 0.008  # L removed per offline sample
FB_FEED_START = 48.0  # day 2
FB_FEED_C_GLUCOSE = 300.0  # g/L in both the continuous feed and the bolus
FB_F0 = 0.00055  # L/h, constant once feeding starts


def _feed_rate(t: float) -> float:
    return 0.0 if t < FB_FEED_START else FB_F0


def _simulate_fedbatch() -> dict[str, np.ndarray]:
    """RK4 on **amounts** (g) plus volume, with discrete events applied between
    segments. Sample first, then bolus, at a coincident timestamp."""
    dt = 0.01
    n = int(round(FB_END / dt)) + 1
    t_grid = np.linspace(0.0, FB_END, n)

    v = 1.0
    y = np.array([0.15 * v, 8.0 * v, 0.0, 0.0])  # mX, mS, mLac, mP  (g)
    v_in_cum = 0.0

    rec = {
        k: np.empty(n)
        for k in ("biomass", "glucose", "lactate", "product", "volume", "v_in")
    }

    def deriv(state: np.ndarray, vol: float, t: float) -> np.ndarray:
        m_x, m_s, _m_lac, _m_p = state
        x, s = m_x / vol, m_s / vol
        mu, q_s, q_lac, q_p = _rates_mammalian(x, s)
        f = _feed_rate(t)
        return np.array(
            [mu * m_x, -q_s * m_x + f * FB_FEED_C_GLUCOSE, q_lac * m_x, q_p * m_x]
        )

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
        species: hxf.ReactorMediumComponent(
            name=species,
            unit="g/L",
            concentration=TimeSeries(
                times=FB_SAMPLE_TIMES.astype(float), values=values.astype(float)
            ),
            bounds=(0.0, None),
        )
        for species, values in meas.items()
    }

    feed_medium = hxf.FeedMedium(
        name="glucose_feed",
        density=1.0,
        density_unit="kg/L",
        components={
            # Every reactor species needs an explicit feed concentration —
            # "absent" and "unrecorded" must not be confusable.
            "glucose": hxf.FeedMediumComponent(
                name="glucose",
                unit="g/L",
                concentration=hxf.StaticVariable(FB_FEED_C_GLUCOSE),
            ),
            "biomass": hxf.FeedMediumComponent(
                name="biomass", unit="g/L", concentration=hxf.StaticVariable(0.0)
            ),
            "lactate": hxf.FeedMediumComponent(
                name="lactate", unit="g/L", concentration=hxf.StaticVariable(0.0)
            ),
            "product": hxf.FeedMediumComponent(
                name="product", unit="g/L", concentration=hxf.StaticVariable(0.0)
            ),
        },
    )

    # Continuous feed: stored as a CUMULATIVE volume trace, not a rate.
    online_t = np.arange(0.0, FB_END + 0.25, 0.5)
    v_in_online = np.interp(online_t, t_grid, sim["v_in"])

    volume_changes = {
        "glucose_feed": hxf.Inflow(
            name="glucose_feed",
            unit="L",
            is_controlled=True,
            is_continuous=True,
            values=TimeSeries(times=online_t, values=v_in_online),
            feed_medium=feed_medium,
        ),
        "glucose_bolus": hxf.Inflow(
            name="glucose_bolus",
            unit="L",
            is_controlled=True,
            is_continuous=False,
            values=TimeSeries(
                times=FB_BOLUS_TIMES.astype(float),
                values=np.full(FB_BOLUS_TIMES.shape, FB_BOLUS_VOLUME),
            ),
            feed_medium=feed_medium,
        ),
        "sampling": hxf.Outflow(
            name="sampling",
            unit="L",
            is_controlled=True,
            is_continuous=False,
            # samples are negative by convention; t=0 is not a draw
            values=TimeSeries(
                times=FB_SAMPLE_TIMES[1:].astype(float),
                values=np.full(FB_SAMPLE_TIMES[1:].shape, -FB_SAMPLE_VOLUME),
            ),
        ),
    }

    # One controlled process variable, measured online.
    do_pct = 40.0 + 8.0 * np.exp(-online_t / 6.0) + rng.normal(0.0, 0.4, online_t.shape)
    process_variables = {
        "dissolved_oxygen": hxf.ProcessVariable(
            name="dissolved_oxygen",
            unit="%",
            is_controlled=True,
            values=TimeSeries(times=online_t, values=do_pct),
            bounds=(0.0, 100.0),
        )
    }

    process = hxf.BioProcess(
        metadata=hxf.BioProcessMetadata(
            name="fedbatch_1",
            process_type="fed_batch",
            notes="Simulated CHO-like mammalian fed-batch: constant feed + "
            "2 boluses + sampling, lactate as a byproduct.",
        ),
        time_axis=hxf.TimeAxis(
            unit="h", start=0.0, end=FB_END, time_reference="inoculation"
        ),
        volume=hxf.Volume(initial_volume=1.0, unit="L", volume_changes=volume_changes),
        reactor_medium=hxf.ReactorMedium(
            name="defined_medium",
            density=1.0,
            density_unit="kg/L",
            components=components,
        ),
        process_variables=process_variables,
    )

    collection = hxf.BioProcessCollection(
        case_id="demo_fedbatch",
        organism="Chinese hamster ovary (CHO) cell line",
        citation="Simulated data — bp-docs demo, not a real experiment.",
        processes={"fedbatch_1": process},
    )
    hxf.serialization.save_process_collection(collection, out / "data.json")

    (out / "ground_truth.json").write_text(
        json.dumps(
            {
                "mu_max": MAM_MU_MAX,
                "Ks": MAM_KS,
                "Y_XS": MAM_Y_XS,
                "m_s": MAM_M_S,
                "Y_lac": MAM_Y_LAC,
                "alpha": MAM_ALPHA,
                "beta": MAM_BETA,
                "final_volume": float(sim["volume"][-1]),
            },
            indent=2,
        )
        + "\n"
    )


# --- demo_products: five batch products, for gallery/knowledge_transfer.md ---
# Kinetics clustered around the target's own phenotype (slow growth, low
# glucose affinity, product-forming): distinguishable cell lines, not the
# fast/slow extremes demo_batch's own kinetics would give.
PRODUCTS_KINETICS = {
    #        mu_max  Ks    Y_XS  m_s    alpha  beta   end_h
    "H1": (0.18, 0.25, 0.28, 0.012, 0.25, 0.0035, 22.0),
    "H2": (0.12, 0.35, 0.22, 0.009, 0.34, 0.0025, 26.0),
    "H3": (0.20, 0.22, 0.30, 0.014, 0.22, 0.0040, 20.0),
    "H4": (0.13, 0.33, 0.23, 0.008, 0.32, 0.0028, 25.0),
    "T": (0.15, 0.30, 0.25, 0.010, 0.30, 0.0030, 24.0),
}
PRODUCTS_NOISE_REL = 0.06
PRODUCTS_N_RUNS_HISTORICAL = 6
# Historical products sample broadly, so pooled data covers regions a 2-run
# target set never sees on its own.
PRODUCTS_S0_RANGE = (5.0, 30.0)
PRODUCTS_X0_RANGE = (0.04, 0.20)
# The target's own runs: two training runs in a narrow mid-range slice, two
# held-out runs at the extremes of PRODUCTS_S0_RANGE/PRODUCTS_X0_RANGE, well
# outside what the two training runs cover.
PRODUCTS_TARGET_RUNS = {
    1: (12.0, 0.10),  # train: mid-range
    2: (14.0, 0.09),  # train: mid-range, deliberately close to run 1
    3: (26.0, 0.05),  # held out: high substrate, low inoculum
    4: (6.0, 0.18),  # held out: low substrate, high inoculum
}


def _noisy_products(
    rng: np.random.Generator, values: np.ndarray, floor: float
) -> np.ndarray:
    noise = rng.normal(1.0, PRODUCTS_NOISE_REL, size=values.shape)
    return np.maximum(values * noise, floor)


def _products_rates(mu_max, ks, y_xs, m_s, alpha, beta, x, s):
    sigma = s / (ks + s) if s > 0.0 else 0.0
    mu = mu_max * sigma
    q_s = mu / y_xs + m_s * sigma
    q_p = alpha * mu + beta * sigma
    return mu, q_s, q_p


def _simulate_products(mu_max, ks, y_xs, m_s, alpha, beta, end, s0, x0):
    dt = 0.002
    n = int(round(end / dt)) + 1
    t_grid = np.linspace(0.0, end, n)
    y = np.array([x0, s0, 0.0])

    def deriv(state):
        x, s, _ = state
        mu, q_s, q_p = _products_rates(mu_max, ks, y_xs, m_s, alpha, beta, x, s)
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
    return t_grid, traj


def _products_process(
    name: str, meas: dict[str, np.ndarray], sample_times, end: float
) -> hxf.BioProcess:
    components = {
        species: hxf.ReactorMediumComponent(
            name=species,
            unit="g/L",
            concentration=TimeSeries(
                times=np.asarray(sample_times, dtype=float),
                values=np.asarray(values, dtype=float),
            ),
            bounds=(0.0, None),
        )
        for species, values in meas.items()
    }
    return hxf.BioProcess(
        metadata=hxf.BioProcessMetadata(
            name=name,
            process_type="batch",
            notes="Simulated batch culture (documentation demo, knowledge transfer).",
        ),
        time_axis=hxf.TimeAxis(
            unit="h", start=0.0, end=end, time_reference="inoculation"
        ),
        volume=hxf.Volume(initial_volume=1.0, unit="L"),
        reactor_medium=hxf.ReactorMedium(
            name="defined_medium",
            density=1.0,
            density_unit="kg/L",
            components=components,
        ),
    )


def build_demo_products() -> None:
    out = OUT / "demo_products"
    out.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(20260812)
    processes = {}
    for key, (mu_max, ks, y_xs, m_s, alpha, beta, end) in PRODUCTS_KINETICS.items():
        n_runs = len(PRODUCTS_TARGET_RUNS) if key == "T" else PRODUCTS_N_RUNS_HISTORICAL
        sample_times = np.arange(0.0, end + 0.5, end / 16.0)
        for r in range(n_runs):
            if key == "T":
                s0, x0 = PRODUCTS_TARGET_RUNS[r + 1]
            else:
                s0 = float(rng.uniform(*PRODUCTS_S0_RANGE))
                x0 = float(rng.uniform(*PRODUCTS_X0_RANGE))
            t_grid, traj = _simulate_products(
                mu_max, ks, y_xs, m_s, alpha, beta, end, s0, x0
            )
            truth = {
                "biomass": np.interp(sample_times, t_grid, traj[:, 0]),
                "glucose": np.interp(sample_times, t_grid, traj[:, 1]),
                "product": np.interp(sample_times, t_grid, traj[:, 2]),
            }
            meas = {
                "biomass": _noisy_products(rng, truth["biomass"], 0.01),
                "glucose": _noisy_products(rng, truth["glucose"], 0.0),
                "product": _noisy_products(rng, truth["product"], 0.0),
            }
            for species in meas:
                meas[species][0] = truth[species][0]
            run_name = f"{key}_run_{r + 1}"
            processes[run_name] = _products_process(run_name, meas, sample_times, end)

    collection = hxf.BioProcessCollection(
        case_id="demo_products",
        organism="Five simulated E. coli-like product lines",
        citation="Simulated data — bp-docs demo, not a real experiment.",
        processes=processes,
    )
    hxf.serialization.save_process_collection(collection, out / "data.json")

    (out / "ground_truth.json").write_text(
        json.dumps(
            {
                k: dict(
                    zip(["mu_max", "Ks", "Y_XS", "m_s", "alpha", "beta", "end_h"], v)
                )
                for k, v in PRODUCTS_KINETICS.items()
            },
            indent=2,
        )
        + "\n"
    )


# --- demo_ecoli_fba / demo_ecoli_blend: forward-simulated from a real,
# frozen surrogate-FBA model, for gallery/fba_hyb.md and gallery/pls_dfba.md ---
#
# The surrogate itself (coefficients below) was fit offline against 10,000 real
# pFBA solves on e_coli_core.xml (Orth, Fleming & Palsson 2010), using the
# method from Gotsmy & Guillen-Gosalbez's FBA-Hyb
# (bioRxiv:10.64898/2026.04.22.720062v1): see
# gallery/_files/01_generate_fba_data.py and 02_fit_surrogate.py for the full,
# reproducible chain. Validation R^2 >= 0.999 on all four fitted fluxes;
# boundedness certificate passed (max overshoot 1.7x over the sampling box,
# min denominator 0.199 > 0 -- pole-free). Fixed here, not re-fit on every
# docs build: solving 10,000 LPs takes minutes and needs `cobra`, neither of
# which a doc build should pay for.
ECOLI_AVG_QG = 10.250012796042299
ECOLI_AVG_N = np.array([1.00000008, 1.00000023, 1.00000072, 0.99999969])
ECOLI_MW_GLC = 180.156
ECOLI_MW_ACE = 60.05
ECOLI_MW_SUC = 118.09
ECOLI_QG_MAX = 6.0
ECOLI_KM_GLC = 0.05


def _ecoli_pos(B: float) -> float:
    return 0.5 * (B + np.sqrt(B * B + 1.5))


def _ecoli_surrogate_fba(qG_raw: float, n_X: float, n_M: float, n_A: float, n_S: float):
    """RAW inputs -> RAW [q_glc, qX, qM, qA, qS], mmol/(gX.h) except q_glc.

    Identical math to the ``surrogate_fba`` embedded in fba_hyb_custom.py /
    pls_dfba_custom.py; duplicated here (not imported) because every
    Tutorial/Gallery ``custom.py`` is a standalone, self-contained file a
    reader can copy on its own, and generate.py is not distributed with them.
    """
    qG = qG_raw / ECOLI_AVG_QG
    n_X, n_M, n_A, n_S = np.array([n_X, n_M, n_A, n_S]) / ECOLI_AVG_N
    pos = _ecoli_pos
    q_glc = -ECOLI_AVG_QG * qG
    qX = (
        qG
        * (
            -40.05086 * n_X
            - 0.011743758 * n_M
            + 0.014346741 * n_A
            - 0.0052064408 * n_S
            + 0.0082783369
        )
        * (
            -49.319124 * n_X
            - 0.41473128 * n_M
            + 2.1157595 * n_A
            - 1.6528906 * n_S
            - 3.5412292
        )
        / (
            (
                pos(
                    24.688972 * n_X
                    + 0.20074873 * n_M
                    - 1.0261536 * n_A
                    + 0.79800013 * n_S
                    + 1.7869275
                )
                + 0.05
            )
            * (
                pos(
                    85.331771 * n_X
                    + 0.48121828 * n_M
                    + 1.7234505 * n_A
                    + 4.5642331 * n_S
                    - 0.46385535
                )
                + 0.05
            )
        )
    )
    qM = (
        qG
        * (
            -30.549273 * n_X
            - 0.26807454 * n_M
            + 0.60650343 * n_A
            - 1.143952 * n_S
            - 0.92775662
        )
        * (
            -0.037824693 * n_X
            - 24.825578 * n_M
            + 0.027984563 * n_A
            + 0.0020666315 * n_S
            - 0.007740588
        )
        / (
            (
                pos(
                    17.512874 * n_X
                    + 0.16739751 * n_M
                    - 0.27899872 * n_A
                    + 0.67477612 * n_S
                    + 0.76306517
                )
                + 0.05
            )
            * (
                pos(
                    46.415016 * n_X
                    + 0.22525384 * n_M
                    + 0.68795142 * n_A
                    + 2.3243431 * n_S
                    - 0.91656303
                )
                + 0.05
            )
        )
    )
    qA = (
        qG
        * (
            0.11560359 * n_X
            + 0.025412058 * n_M
            + 24.333516 * n_A
            + 0.037186754 * n_S
            - 0.094942148
        )
        * (
            20.777323 * n_X
            - 0.40676165 * n_M
            + 3.1964066 * n_A
            + 0.9569548 * n_S
            - 0.19868886
        )
        / (
            (
                pos(
                    41.246274 * n_X
                    - 0.34994074 * n_M
                    + 4.3166512 * n_A
                    + 2.1503221 * n_S
                    - 1.9194144
                )
                + 0.05
            )
            * (
                pos(
                    13.35614 * n_X
                    - 0.041404912 * n_M
                    + 0.72777261 * n_A
                    + 0.64795735 * n_S
                    + 0.29150121
                )
                + 0.05
            )
        )
    )
    qS = (
        qG
        * (
            -0.028994836 * n_X
            + 0.00034001442 * n_M
            + 0.018303022 * n_A
            - 17.239703 * n_S
            - 0.0048146897
        )
        * (
            -55.150953 * n_X
            - 0.50212877 * n_M
            + 1.331242 * n_A
            - 1.9245035 * n_S
            - 1.8896264
        )
        / (
            (
                pos(
                    44.312184 * n_X
                    + 0.20545498 * n_M
                    + 0.56363208 * n_A
                    + 2.2376853 * n_S
                    - 0.70094854
                )
                + 0.05
            )
            * (
                pos(
                    22.984164 * n_X
                    + 0.22825551 * n_M
                    - 0.41448157 * n_A
                    + 0.81603398 * n_S
                    + 1.0517311
                )
                + 0.05
            )
        )
    )
    return float(q_glc), float(qX), float(qM), float(qA), float(qS)


def _ecoli_obj_weights_fba(t: float, t_end: float) -> tuple[float, float, float, float]:
    """Growth-focused early, tapering; acetate weight rises over the batch
    (real overflow-metabolism trade-off); no deliberate product (n_S=0)."""
    frac = t / t_end
    return 1.6 - 1.0 * frac, 0.35, 0.1 + 0.8 * frac, 0.0


def _ecoli_obj_weights_blend(t: float, t_end: float, blend: float):
    """Same growth/maintenance/acetate shape as _ecoli_obj_weights_fba, lightly
    reduced by the blend fraction (competing for the same carbon), plus a
    succinate weight driven directly by the blend: this is the page's whole
    point, a controllable recipe choice shifting the kinetic corridor."""
    n_X, n_M, n_A, _ = _ecoli_obj_weights_fba(t, t_end)
    return n_X * (1.0 - 0.25 * blend), n_M, n_A * (1.0 - 0.4 * blend), blend * 1.5


def _ecoli_simulate(
    S0: float, X0: float, blend: float | None, t_end: float = 11.0, dt: float = 0.002
) -> tuple[np.ndarray, np.ndarray]:
    """Forward-Euler through the surrogate. blend=None -> demo_ecoli_fba (no
    succinate tracking); blend=0..1 -> demo_ecoli_blend."""
    n = int(round(t_end / dt)) + 1
    t_grid = np.linspace(0.0, t_end, n)
    X, G, A, S = X0, S0, 0.0, 0.0
    n_cols = 3 if blend is None else 4
    traj = np.empty((n, n_cols))
    traj[0] = [X, G, A] if blend is None else [X, G, A, S]
    for i in range(1, n):
        if blend is None:
            n_X, n_M, n_A, n_S = _ecoli_obj_weights_fba(t_grid[i - 1], t_end)
        else:
            n_X, n_M, n_A, n_S = _ecoli_obj_weights_blend(t_grid[i - 1], t_end, blend)
        qG_raw = ECOLI_QG_MAX * max(G, 0.0) / (ECOLI_KM_GLC + max(G, 0.0))
        q_glc, qX, qM, qA, qS = _ecoli_surrogate_fba(qG_raw, n_X, n_M, n_A, n_S)
        X = max(X + dt * (qX * X), 0.0)
        G = max(G + dt * (q_glc * ECOLI_MW_GLC / 1000.0) * X, 0.0)
        A = max(A + dt * (qA * ECOLI_MW_ACE / 1000.0) * X, 0.0)
        if blend is not None:
            S = max(S + dt * (qS * ECOLI_MW_SUC / 1000.0) * X, 0.0)
        traj[i] = [X, G, A] if blend is None else [X, G, A, S]
    return t_grid, traj


ECOLI_NOISE_REL = 0.04
ECOLI_T_END = 11.0
ECOLI_SAMPLE_TIMES = np.arange(0.0, ECOLI_T_END + 0.5, ECOLI_T_END / 16.0)


def _noisy_ecoli(
    rng: np.random.Generator, values: np.ndarray, floor: float
) -> np.ndarray:
    noise = rng.normal(1.0, ECOLI_NOISE_REL, size=values.shape)
    return np.maximum(values * noise, floor)


def build_demo_ecoli_fba() -> None:
    out = OUT / "demo_ecoli_fba"
    out.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(20260813)
    runs = {"run_1": (15.0, 0.05), "run_2": (12.0, 0.04), "run_3": (18.0, 0.06)}
    processes = {}
    for name, (s0, x0) in runs.items():
        t_grid, traj = _ecoli_simulate(s0, x0, blend=None, t_end=ECOLI_T_END)
        truth = {
            "biomass": np.interp(ECOLI_SAMPLE_TIMES, t_grid, traj[:, 0]),
            "glucose": np.interp(ECOLI_SAMPLE_TIMES, t_grid, traj[:, 1]),
            "acetate": np.interp(ECOLI_SAMPLE_TIMES, t_grid, traj[:, 2]),
        }
        meas = {
            "biomass": _noisy_ecoli(rng, truth["biomass"], 0.005),
            "glucose": _noisy_ecoli(rng, truth["glucose"], 0.0),
            "acetate": _noisy_ecoli(rng, truth["acetate"], 0.0),
        }
        for species in meas:
            meas[species][0] = truth[species][0]
        components = {
            species: hxf.ReactorMediumComponent(
                name=species,
                unit="g/L",
                concentration=TimeSeries(
                    times=ECOLI_SAMPLE_TIMES.astype(float), values=values.astype(float)
                ),
                bounds=(0.0, None),
            )
            for species, values in meas.items()
        }
        processes[name] = hxf.BioProcess(
            metadata=hxf.BioProcessMetadata(
                name=name,
                process_type="batch",
                notes=(
                    "Forward-simulated from a real surrogate-FBA model "
                    "(documentation demo)."
                ),
            ),
            time_axis=hxf.TimeAxis(
                unit="h", start=0.0, end=ECOLI_T_END, time_reference="inoculation"
            ),
            volume=hxf.Volume(initial_volume=1.0, unit="L"),
            reactor_medium=hxf.ReactorMedium(
                name="M9_glucose",
                density=1.0,
                density_unit="kg/L",
                components=components,
            ),
        )

    collection = hxf.BioProcessCollection(
        case_id="demo_ecoli_fba",
        organism="Escherichia coli (core metabolism, Orth/Fleming/Palsson 2010)",
        citation=(
            "Simulated via a surrogate-FBA forward model — bp-docs demo, "
            "not a real experiment."
        ),
        processes=processes,
    )
    hxf.serialization.save_process_collection(collection, out / "data.json")


ECOLI_BLEND_RUNS = {
    # name          S0     X0     blend
    "blend_00": (15.0, 0.05, 0.0),
    "blend_33": (14.0, 0.05, 0.33),
    "blend_67": (16.0, 0.05, 0.67),
    "blend_100": (15.0, 0.06, 1.0),
}


def build_demo_ecoli_blend() -> None:
    out = OUT / "demo_ecoli_blend"
    out.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(20260813)
    processes = {}
    for name, (s0, x0, blend) in ECOLI_BLEND_RUNS.items():
        t_grid, traj = _ecoli_simulate(s0, x0, blend=blend, t_end=ECOLI_T_END)
        truth = {
            "biomass": np.interp(ECOLI_SAMPLE_TIMES, t_grid, traj[:, 0]),
            "glucose": np.interp(ECOLI_SAMPLE_TIMES, t_grid, traj[:, 1]),
            "acetate": np.interp(ECOLI_SAMPLE_TIMES, t_grid, traj[:, 2]),
            "succinate": np.interp(ECOLI_SAMPLE_TIMES, t_grid, traj[:, 3]),
        }
        meas = {
            k: _noisy_ecoli(rng, v, 0.005 if k == "biomass" else 0.0)
            for k, v in truth.items()
        }
        for species in meas:
            meas[species][0] = truth[species][0]
        components = {
            species: hxf.ReactorMediumComponent(
                name=species,
                unit="g/L",
                concentration=TimeSeries(
                    times=ECOLI_SAMPLE_TIMES.astype(float), values=values.astype(float)
                ),
                bounds=(0.0, None),
            )
            for species, values in meas.items()
        }
        processes[name] = hxf.BioProcess(
            metadata=hxf.BioProcessMetadata(
                name=name,
                process_type="batch",
                notes=(
                    "Forward-simulated from a real surrogate-FBA model, "
                    "blend-dependent (documentation demo)."
                ),
            ),
            time_axis=hxf.TimeAxis(
                unit="h", start=0.0, end=ECOLI_T_END, time_reference="inoculation"
            ),
            volume=hxf.Volume(initial_volume=1.0, unit="L"),
            reactor_medium=hxf.ReactorMedium(
                name="M9_glucose_blend",
                density=1.0,
                density_unit="kg/L",
                components=components,
            ),
        )

    collection = hxf.BioProcessCollection(
        case_id="demo_ecoli_blend",
        organism="Escherichia coli (core metabolism, Orth/Fleming/Palsson 2010)",
        citation=(
            "Simulated via a surrogate-FBA forward model — bp-docs demo, "
            "not a real experiment."
        ),
        processes=processes,
    )
    hxf.serialization.save_process_collection(collection, out / "data.json")


# ===========================================================================
# demo_optfed
# ===========================================================================
# Reduced from OptFed's own equations (4a-4e): inhibition/activation over a
# smaller variable set ({P/X, X} inhibiting uptake and production, {q_glucose,
# X} activating maintenance), no "number of generations" term, yields frozen at
# their true value rather than fitted (matching the paper's own statement that
# yields come from a genome-scale model, not the fitted parameters).


def _optfed_eyring(
    T_K: float, A: float, Ea_R: float, Teq: float, dHeq_R: float
) -> float:
    return (
        A * T_K * np.exp(-Ea_R / T_K) / (1.0 + np.exp(dHeq_R * (1.0 / Teq - 1.0 / T_K)))
    )


OPTFED_E_DEG = dict(Ea_R=2200.0, Teq=309.0, dHeq_R=18000.0)
OPTFED_E_PI = dict(Ea_R=2000.0, Teq=306.0, dHeq_R=16000.0)
OPTFED_E_ALPHA = dict(Ea_R=1500.0, Teq=320.0, dHeq_R=15000.0)
# A is picked so each rate hits a target magnitude at a 305.15 K reference.
for _E, _target in ((OPTFED_E_DEG, 0.7), (OPTFED_E_PI, 0.18), (OPTFED_E_ALPHA, 0.05)):
    _E["A"] = _target / _optfed_eyring(305.15, A=1.0, **_E)

OPTFED_K_DEG_M = 0.5  # glucose half-saturation for gamma_deg, g/L
OPTFED_K_PI_M = 0.05  # (gamma_deg - gamma_alpha) half-saturation for gamma_pi
OPTFED_K_DEG_PX, OPTFED_K_DEG_X = 0.3, 60.0  # inhibition constants, gamma_deg
OPTFED_K_PI_PX, OPTFED_K_PI_X = 0.2, 40.0  # inhibition constants, gamma_pi
OPTFED_K_A_DEG, OPTFED_K_A_X = 2.0, 50.0  # activation constants, gamma_alpha
OPTFED_Y_XR_G = 0.45  # g active biomass / g glucose (frozen, not fitted)
OPTFED_Y_P_G = 0.25  # g product / g glucose (frozen, not fitted)

OPTFED_GF = 400.0  # g/L glucose in the feed
OPTFED_F0 = 0.02  # L/h feed-rate prefactor
OPTFED_END = 12.0
OPTFED_SAMPLE_TIMES = np.arange(0.0, OPTFED_END + 1.0, 2.0)  # every 2 h
OPTFED_NOISE_REL = 0.03

# name -> (X0 g/L, feed_mu 1/h, temperature degC), a small center + star +
# corner spread echoing OptFed's own central-composite design, not a literal
# reproduction of it.
OPTFED_RUNS = {
    "center": (20.0, 0.05, 32.0),
    "T_low": (20.0, 0.05, 28.0),
    "T_high": (20.0, 0.05, 40.0),
    "feed_low": (20.0, 0.01, 32.0),
    "feed_high": (20.0, 0.10, 32.0),
    "corner": (20.0, 0.09, 38.0),
}


def _optfed_rates(
    X: float, P: float, G: float, T_K: float
) -> tuple[float, float, float]:
    """(q_biomass, q_product, q_glucose), per unit active biomass except
    q_glucose (per unit total biomass, matching Eq. 1c's own X multiplier)."""
    px = P / max(X, 1e-9)

    gdeg_max = _optfed_eyring(T_K, **OPTFED_E_DEG)
    gdeg = (
        gdeg_max
        * (G / (OPTFED_K_DEG_M + G))
        * (1.0 / (1.0 + px / OPTFED_K_DEG_PX))
        * (1.0 / (1.0 + X / OPTFED_K_DEG_X))
    )

    galpha_min = _optfed_eyring(T_K, **OPTFED_E_ALPHA)
    galpha = galpha_min * (1.0 + gdeg / OPTFED_K_A_DEG) * (1.0 + X / OPTFED_K_A_X)

    gpi_max = _optfed_eyring(T_K, **OPTFED_E_PI)
    driver = max(gdeg - galpha, 0.0)
    gpi = (
        gpi_max
        * (driver / (OPTFED_K_PI_M + driver))
        * (1.0 / (1.0 + px / OPTFED_K_PI_PX))
        * (1.0 / (1.0 + X / OPTFED_K_PI_X))
    )

    gmu = gdeg - gpi - galpha
    return gmu * OPTFED_Y_XR_G, gpi * OPTFED_Y_P_G, gdeg


def _simulate_optfed(
    x0: float, feed_mu: float, temperature_c: float
) -> dict[str, np.ndarray]:
    """RK4 on amounts (g) plus volume, exponential feed rate."""
    t_k = temperature_c + 273.15
    dt = 0.01
    n = int(round(OPTFED_END / dt)) + 1
    t_grid = np.linspace(0.0, OPTFED_END, n)

    def feed_rate(t: float) -> float:
        return OPTFED_F0 * np.exp(feed_mu * t)

    v = 1.0
    y = np.array([x0 * v, 0.5 * v, 0.0])  # mX, mG, mP (g)
    rec = {k: np.empty(n) for k in ("biomass", "glucose", "product", "volume", "v_in")}
    v_in_cum = 0.0

    def deriv(state: np.ndarray, vol: float, t: float) -> np.ndarray:
        m_x, m_g, m_p = state
        X, G, P = m_x / vol, m_g / vol, m_p / vol
        q_b, q_p, q_g = _optfed_rates(X, P, G, t_k)
        m_xr = max(m_x - m_p, 0.0)
        f = feed_rate(t)
        return np.array(
            [q_b * m_xr + q_p * m_xr, -q_g * m_x + f * OPTFED_GF, q_p * m_xr]
        )

    def record(i: int) -> None:
        rec["biomass"][i] = y[0] / v
        rec["glucose"][i] = y[1] / v
        rec["product"][i] = y[2] / v
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

        f_mid = feed_rate(t + 0.5 * dt)
        v += f_mid * dt
        v_in_cum += f_mid * dt
        record(i)

    rec["t"] = t_grid
    return rec


def build_demo_optfed() -> None:
    out = OUT / "demo_optfed"
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(20260814)

    feed_medium = hxf.FeedMedium(
        name="optfed_feed",
        density=1.0,
        density_unit="kg/L",
        components={
            "glucose": hxf.FeedMediumComponent(
                name="glucose", unit="g/L", concentration=hxf.StaticVariable(OPTFED_GF)
            ),
            "biomass": hxf.FeedMediumComponent(
                name="biomass", unit="g/L", concentration=hxf.StaticVariable(0.0)
            ),
            "product": hxf.FeedMediumComponent(
                name="product", unit="g/L", concentration=hxf.StaticVariable(0.0)
            ),
        },
    )

    processes = {}
    for name, (x0, feed_mu, temperature_c) in OPTFED_RUNS.items():
        sim = _simulate_optfed(x0, feed_mu, temperature_c)
        t_grid = sim["t"]

        def at_samples(key: str, t_grid=t_grid, sim=sim) -> np.ndarray:
            idx = [int(np.argmin(np.abs(t_grid - t))) for t in OPTFED_SAMPLE_TIMES]
            return sim[key][idx]

        meas = {
            "biomass": _noisy(rng, at_samples("biomass"), 0.05),
            "glucose": _noisy(rng, at_samples("glucose"), 0.0),
            "product": _noisy(rng, at_samples("product"), 0.0),
        }
        for species in meas:
            meas[species][0] = at_samples(species)[0]

        components = {
            species: hxf.ReactorMediumComponent(
                name=species,
                unit="g/L",
                concentration=TimeSeries(
                    times=OPTFED_SAMPLE_TIMES.astype(float), values=values.astype(float)
                ),
                bounds=(0.0, None),
            )
            for species, values in meas.items()
        }

        v_in_online = np.interp(OPTFED_SAMPLE_TIMES, t_grid, sim["v_in"])
        volume_changes = {
            "glucose_feed": hxf.Inflow(
                name="glucose_feed",
                unit="L",
                is_controlled=True,
                is_continuous=True,
                values=TimeSeries(
                    times=OPTFED_SAMPLE_TIMES.astype(float),
                    values=v_in_online.astype(float),
                ),
                feed_medium=feed_medium,
            ),
        }

        process_variables = {
            "temperature": hxf.ProcessVariable(
                name="temperature",
                unit="degC",
                is_controlled=True,
                values=TimeSeries(
                    times=OPTFED_SAMPLE_TIMES.astype(float),
                    values=np.full(OPTFED_SAMPLE_TIMES.shape, temperature_c),
                ),
                bounds=(0.0, 60.0),
            ),
        }

        processes[name] = hxf.BioProcess(
            metadata=hxf.BioProcessMetadata(
                name=name,
                process_type="fed_batch",
                notes="Simulated from OptFed's non-competitive-inhibition rate "
                "law with Eyring temperature dependence (documentation demo).",
            ),
            time_axis=hxf.TimeAxis(
                unit="h", start=0.0, end=OPTFED_END, time_reference="inoculation"
            ),
            volume=hxf.Volume(
                initial_volume=1.0, unit="L", volume_changes=volume_changes
            ),
            reactor_medium=hxf.ReactorMedium(
                name="defined_medium",
                density=1.0,
                density_unit="kg/L",
                components=components,
            ),
            process_variables=process_variables,
        )
        processes[name].biological_ode = hxf.BiologicalOde(
            algebraic={"X_active": "biomass - product"},
            rates={
                "q_biomass": (None, None),
                "q_glucose": (None, None),
                "q_product": (None, None),
            },
            derivatives={
                "biomass": "q_biomass * X_active + q_product * X_active",
                "glucose": "-q_glucose * biomass",
                "product": "q_product * X_active",
            },
        )

    collection = hxf.BioProcessCollection(
        case_id="demo_optfed",
        organism="Escherichia coli (recombinant protein production, OptFed-inspired)",
        citation="Simulated data — bp-docs demo, not a real experiment.",
        processes=processes,
    )
    hxf.serialization.save_process_collection(collection, out / "data.json")

    (out / "ground_truth.json").write_text(
        json.dumps(
            {
                "eyring_deg": OPTFED_E_DEG,
                "eyring_pi": OPTFED_E_PI,
                "eyring_alpha": OPTFED_E_ALPHA,
                "K_deg_m": OPTFED_K_DEG_M,
                "K_pi_m": OPTFED_K_PI_M,
                "K_deg_px": OPTFED_K_DEG_PX,
                "K_deg_x": OPTFED_K_DEG_X,
                "K_pi_px": OPTFED_K_PI_PX,
                "K_pi_x": OPTFED_K_PI_X,
                "K_a_deg": OPTFED_K_A_DEG,
                "K_a_x": OPTFED_K_A_X,
                "Y_XrG": OPTFED_Y_XR_G,
                "Y_PG": OPTFED_Y_P_G,
            },
            indent=2,
        )
    )


# ===========================================================================
# demo_glutamine_decay
# ===========================================================================
# One physical rate, r_Gln, feeds two different derivatives at once: it
# drains Gln (a sink) and produces NH4 (a source), the same non-enzymatic
# decomposition Ulonska et al. 2018 report at a real, cited rate (Table 1,
# rNH4,gln). Reduced from the paper's own ammonia balance (Eq. 20), which
# also has a metabolic production term (qNH4 * VCC) and a feed-release term;
# this dataset keeps only the chemical-decomposition term, since that is the
# one rate this demo is actually about. Gln/NH4 are tracked in mol/L (unlike
# the rest of this file's g/L convention) so the coupling is exactly
# d(NH4)/dt = r_Gln * Gln, one shared rate, no separate yield constant — a
# deliberate, disclosed per-page unit choice, not a claim that mol/L is more
# correct than the paper's own g/L-plus-yield-constant approach.

GLN_Q_BIOMASS = 0.012  # 1/h
GLN_Q_GLN = 4.0e-5  # mol glutamine / (g biomass * h), uptake magnitude
GLN_R_GLN = 0.0036  # 1/h — Ulonska et al. 2018, Table 1 (rNH4,gln)

GLN_END = 120.0
GLN_SAMPLE_TIMES = np.arange(0.0, GLN_END + 0.5, 12.0)  # 11 samples

GLN_RUNS = {
    #  name      X0     Gln0     NH4_0
    "run_1": (0.30, 0.0060, 0.0005),
    "run_2": (0.25, 0.0070, 0.0004),
    "run_3": (0.35, 0.0050, 0.0006),
}


def _simulate_glutamine_decay(
    x0: float, gln0: float, nh4_0: float
) -> dict[str, np.ndarray]:
    dt = 0.005
    n = int(round(GLN_END / dt)) + 1
    t_grid = np.linspace(0.0, GLN_END, n)
    y = np.array([x0, gln0, nh4_0])

    def deriv(state: np.ndarray) -> np.ndarray:
        x, gln, _nh4 = state
        dX = GLN_Q_BIOMASS * x
        dGln = -GLN_Q_GLN * x - GLN_R_GLN * gln
        dNH4 = GLN_R_GLN * gln
        return np.array([dX, dGln, dNH4])

    traj = np.empty((n, 3))
    traj[0] = y
    for i in range(1, n):
        k1 = deriv(y)
        k2 = deriv(y + 0.5 * dt * k1)
        k3 = deriv(y + 0.5 * dt * k2)
        k4 = deriv(y + dt * k3)
        y = y + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
        y[1] = max(y[1], 0.0)
        traj[i] = y

    return {
        "biomass": np.interp(GLN_SAMPLE_TIMES, t_grid, traj[:, 0]),
        "Gln": np.interp(GLN_SAMPLE_TIMES, t_grid, traj[:, 1]),
        "NH4": np.interp(GLN_SAMPLE_TIMES, t_grid, traj[:, 2]),
    }


def build_demo_glutamine_decay() -> None:
    out = OUT / "demo_glutamine_decay"
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(20260817)

    processes = {}
    for name, (x0, gln0, nh4_0) in GLN_RUNS.items():
        sim = _simulate_glutamine_decay(x0, gln0, nh4_0)
        meas = {
            "biomass": _noisy(rng, sim["biomass"], 0.005),
            "Gln": _noisy(rng, sim["Gln"], 1e-5),
            "NH4": _noisy(rng, sim["NH4"], 1e-5),
        }
        for species in meas:
            meas[species][0] = sim[species][0]  # first sample noise-free

        components = {
            "biomass": hxf.ReactorMediumComponent(
                name="biomass",
                unit="g/L",
                concentration=TimeSeries(
                    times=GLN_SAMPLE_TIMES.astype(float),
                    values=meas["biomass"].astype(float),
                ),
                bounds=(0.0, None),
            ),
            "Gln": hxf.ReactorMediumComponent(
                name="Gln",
                unit="mol/L",
                concentration=TimeSeries(
                    times=GLN_SAMPLE_TIMES.astype(float),
                    values=meas["Gln"].astype(float),
                ),
                bounds=(0.0, None),
            ),
            "NH4": hxf.ReactorMediumComponent(
                name="NH4",
                unit="mol/L",
                concentration=TimeSeries(
                    times=GLN_SAMPLE_TIMES.astype(float),
                    values=meas["NH4"].astype(float),
                ),
                bounds=(0.0, None),
            ),
        }

        processes[name] = hxf.BioProcess(
            metadata=hxf.BioProcessMetadata(
                name=name,
                process_type="batch",
                notes="Simulated CHO-like batch culture; glutamine decomposes "
                "to NH4 at a first-order rate (documentation demo).",
            ),
            time_axis=hxf.TimeAxis(
                unit="h", start=0.0, end=GLN_END, time_reference="inoculation"
            ),
            volume=hxf.Volume(initial_volume=1.0, unit="L"),
            reactor_medium=hxf.ReactorMedium(
                name="defined_medium",
                density=1.0,
                density_unit="kg/L",
                components=components,
            ),
        )
        processes[name].biological_ode = hxf.BiologicalOde(
            rates={
                "q_biomass": (None, None),
                "q_Gln": (None, None),
                "r_Gln": (None, None),
            },
            derivatives={
                "biomass": "q_biomass * biomass",
                "Gln": "-q_Gln * biomass - r_Gln * Gln",
                "NH4": "r_Gln * Gln",
            },
        )

    collection = hxf.BioProcessCollection(
        case_id="demo_glutamine_decay",
        organism="CHO cell culture (Ulonska et al. 2018-inspired)",
        citation="Simulated data — bp-docs demo, not a real experiment.",
        processes=processes,
    )
    hxf.serialization.save_process_collection(collection, out / "data.json")
    (out / "ground_truth.json").write_text(
        json.dumps(
            {
                "q_biomass": GLN_Q_BIOMASS,
                "q_Gln": GLN_Q_GLN,
                "r_Gln": GLN_R_GLN,
            },
            indent=2,
        )
    )


# ===========================================================================
# demo_modeled_pv
# ===========================================================================
# Two independent first-order states, deliberately unrelated to each other so
# neither can explain the other's behavior: ``biomass`` (a modeled reactor
# component, growing exponentially) and ``glyco_frac`` (a modeled process
# variable, decaying exponentially, standing in for a glycosylation-fraction
# quality attribute). One large dilution bolus lands midway through each run.
# Both states are closed-form, so the dilution's effect on ``biomass`` and its
# absence on ``glyco_frac`` are exact: ``biomass`` is an amount growing
# unaffected by the bolus (the feed carries none), divided by a volume that
# steps up at the bolus, so its concentration steps down; ``glyco_frac`` has
# no relationship to volume at all, in the model or in the physics
# hybrax.format adds, so it is untouched.

MPV_Q_BIOMASS = 0.15  # 1/h
MPV_R_GLYCO_FRAC = 0.03  # 1/h

MPV_V0 = 1.0  # L
MPV_T_END = 20.0  # h
MPV_T_BOLUS = 10.0  # h, roughly mid-run
MPV_DELTA_V_BOLUS = 1.0  # L — doubles the volume, a big, unmistakable jump

MPV_SAMPLE_TIMES = np.array(
    [0.0, 2.0, 4.0, 6.0, 8.0, 9.9, 10.1, 12.0, 14.0, 16.0, 18.0, 20.0]
)

MPV_RUNS = {
    #  name      X0 [g/L]   glyco_frac0 [-]
    "run_1": (0.10, 0.95),
    "run_2": (0.15, 0.90),
    "run_3": (0.08, 0.97),
}


def _simulate_modeled_pv(
    x0: float, glyco0: float, t: np.ndarray
) -> dict[str, np.ndarray]:
    """Exact closed-form trajectories at times ``t``.

    ``biomass`` amount grows as ``X0*V0*exp(q*t)`` throughout (the bolus feed
    carries no biomass, so the amount itself never jumps); concentration is
    that amount divided by volume, which steps from ``V0`` to
    ``V0+delta_v_bolus`` at ``T_BOLUS``. ``glyco_frac`` decays as
    ``glyco0*exp(-r*t)`` with no dependence on volume at all.
    """
    amount = x0 * MPV_V0 * np.exp(MPV_Q_BIOMASS * t)
    volume = np.where(t < MPV_T_BOLUS, MPV_V0, MPV_V0 + MPV_DELTA_V_BOLUS)
    return {
        "biomass": amount / volume,
        "glyco_frac": glyco0 * np.exp(-MPV_R_GLYCO_FRAC * t),
    }


def build_demo_modeled_pv() -> None:
    out = OUT / "demo_modeled_pv"
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(20260824)

    processes = {}
    for name, (x0, glyco0) in MPV_RUNS.items():
        sim = _simulate_modeled_pv(x0, glyco0, MPV_SAMPLE_TIMES)
        meas = {
            "biomass": _noisy(rng, sim["biomass"], 0.005),
            "glyco_frac": np.clip(_noisy(rng, sim["glyco_frac"], 0.005), 0.0, 1.0),
        }
        for species in meas:
            meas[species][0] = sim[species][0]  # first sample noise-free

        feed_medium = hxf.FeedMedium(
            name="dilution_feed",
            density=1.0,
            density_unit="kg/L",
            components={
                "biomass": hxf.FeedMediumComponent(
                    name="biomass", unit="g/L", concentration=hxf.StaticVariable(0.0)
                ),
            },
        )

        process = hxf.BioProcess(
            metadata=hxf.BioProcessMetadata(
                name=name,
                process_type="fed_batch",
                notes="Simulated CHO-like batch culture with one large dilution "
                "bolus midway through the run: biomass concentration is "
                "diluted by the volume increase, glyco_frac (a modeled "
                "process variable) is not (documentation demo).",
            ),
            time_axis=hxf.TimeAxis(
                unit="h", start=0.0, end=MPV_T_END, time_reference="inoculation"
            ),
            volume=hxf.Volume(
                initial_volume=MPV_V0,
                unit="L",
                volume_changes={
                    "dilution_bolus": hxf.Inflow(
                        name="dilution_bolus",
                        unit="L",
                        is_controlled=True,
                        is_continuous=False,
                        values=TimeSeries(
                            times=np.array([MPV_T_BOLUS]),
                            values=np.array([MPV_DELTA_V_BOLUS]),
                        ),
                        feed_medium=feed_medium,
                    ),
                },
            ),
            reactor_medium=hxf.ReactorMedium(
                name="defined_medium",
                density=1.0,
                density_unit="kg/L",
                components={
                    "biomass": hxf.ReactorMediumComponent(
                        name="biomass",
                        unit="g/L",
                        concentration=TimeSeries(
                            times=MPV_SAMPLE_TIMES.astype(float),
                            values=meas["biomass"].astype(float),
                        ),
                        bounds=(0.0, None),
                    ),
                },
            ),
            process_variables={
                "glyco_frac": hxf.ProcessVariable(
                    name="glyco_frac",
                    unit="-",
                    is_controlled=False,
                    values=TimeSeries(
                        times=MPV_SAMPLE_TIMES.astype(float),
                        values=meas["glyco_frac"].astype(float),
                    ),
                    bounds=(0.0, 1.0),
                ),
            },
            biological_ode=hxf.BiologicalOde(
                rates={"q_biomass": (0.0, None), "r_glyco_frac": (0.0, None)},
                derivatives={
                    "biomass": "q_biomass * biomass",
                    "glyco_frac": "-r_glyco_frac * glyco_frac",
                },
            ),
        )
        processes[name] = process

    collection = hxf.BioProcessCollection(
        case_id="demo_modeled_pv",
        organism=(
            "None (synthetic biomass + a glycosylation-fraction quality attribute)"
        ),
        citation="Simulated data — bp-docs demo, not a real experiment.",
        processes=processes,
    )
    hxf.serialization.save_process_collection(collection, out / "data.json")
    (out / "ground_truth.json").write_text(
        json.dumps(
            {
                "q_biomass": MPV_Q_BIOMASS,
                "r_glyco_frac": MPV_R_GLYCO_FRAC,
            },
            indent=2,
        )
        + "\n"
    )


# ===========================================================================
# demo_spline_jump
# ===========================================================================
# One species, first-order decay (a physical/chemical rate, not growth-linked,
# so no biomass is needed: biological_ode is supplied explicitly). One feed
# bolus part-way through jumps mass and volume together. Both phases are
# closed-form exponential decay at constant volume, so the dense ground truth
# below is exact, not an RK4 approximation. It is the one dataset in this file
# built that way, since its whole point is a ground truth to fit against.

SJ_K = 0.15  # 1/h, first-order decay rate
SJ_V0 = 1.0  # L
SJ_C0 = 5.0  # g/L
SJ_T_END = 17.0  # h, matches the last measurement, no unobserved tail
SJ_T_JUMP = 10.0  # h, when the bolus lands
SJ_DELTA_V_BOLUS = 0.15  # L
SJ_C_FEED = 40.0  # g/L
SJ_SAMPLE_TIMES = np.array([0.0, 4.0, 9.0, 11.0, 17.0])


def spline_jump_truth(t) -> np.ndarray:
    """Exact concentration at time(s) ``t``: closed-form on both sides of the
    bolus at ``SJ_T_JUMP``. Shared with ``gallery/pseudobatch_splines.md``,
    which imports this function directly rather than re-deriving the formula.
    """
    t = np.asarray(t, dtype=float)
    m_at_jump = SJ_C0 * SJ_V0 * np.exp(-SJ_K * SJ_T_JUMP)
    v_after = SJ_V0 + SJ_DELTA_V_BOLUS
    m_after = m_at_jump + SJ_DELTA_V_BOLUS * SJ_C_FEED
    pre = SJ_C0 * np.exp(-SJ_K * t)
    post = (m_after / v_after) * np.exp(-SJ_K * (t - SJ_T_JUMP))
    return np.where(t < SJ_T_JUMP, pre, post)


def build_demo_spline_jump() -> None:
    out = OUT / "demo_spline_jump"
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(20260820)

    truth = spline_jump_truth(SJ_SAMPLE_TIMES)
    meas = _noisy(rng, truth, 0.0)
    meas[0] = truth[0]  # t=0 measured exactly

    feed_medium = hxf.FeedMedium(
        name="solute_feed",
        density=1.0,
        density_unit="kg/L",
        components={
            "solute": hxf.FeedMediumComponent(
                name="solute", unit="g/L", concentration=hxf.StaticVariable(SJ_C_FEED)
            ),
            "biomass": hxf.FeedMediumComponent(
                name="biomass", unit="g/L", concentration=hxf.StaticVariable(0.0)
            ),
        },
    )

    process = hxf.BioProcess(
        metadata=hxf.BioProcessMetadata(
            name="run_1",
            process_type="fed_batch",
            notes="Simulated single-species first-order decay with one feed "
            "bolus (documentation demo, pseudobatch splines gallery page).",
        ),
        time_axis=hxf.TimeAxis(
            unit="h", start=0.0, end=SJ_T_END, time_reference="inoculation"
        ),
        volume=hxf.Volume(
            initial_volume=SJ_V0,
            unit="L",
            volume_changes={
                "solute_bolus": hxf.Inflow(
                    name="solute_bolus",
                    unit="L",
                    is_controlled=True,
                    is_continuous=False,
                    values=TimeSeries(
                        times=np.array([SJ_T_JUMP]), values=np.array([SJ_DELTA_V_BOLUS])
                    ),
                    feed_medium=feed_medium,
                ),
            },
        ),
        reactor_medium=hxf.ReactorMedium(
            name="defined_medium",
            density=1.0,
            density_unit="kg/L",
            components={
                "solute": hxf.ReactorMediumComponent(
                    name="solute",
                    unit="g/L",
                    concentration=TimeSeries(
                        times=SJ_SAMPLE_TIMES.astype(float), values=meas.astype(float)
                    ),
                    bounds=(0.0, None),
                ),
                # A flat, non-dynamic placeholder: hybrax.format's biomass check
                # expects a 'biomass' component on every process, even one like
                # this with no growth at all. It plays no role in the demo.
                "biomass": hxf.ReactorMediumComponent(
                    name="biomass",
                    unit="g/L",
                    concentration=TimeSeries(
                        times=SJ_SAMPLE_TIMES.astype(float),
                        values=np.full(SJ_SAMPLE_TIMES.shape, 1.0),
                    ),
                    bounds=(0.0, None),
                ),
            },
        ),
        biological_ode=hxf.BiologicalOde(
            rates={"k_solute": (0.0, None)},
            derivatives={"solute": "-k_solute * solute", "biomass": "0"},
        ),
    )

    collection = hxf.BioProcessCollection(
        case_id="demo_spline_jump",
        organism="None (a physical decay process, not a cell culture)",
        citation="Simulated data, bp-docs demo, not a real experiment.",
        processes={"run_1": process},
    )
    hxf.serialization.save_process_collection(collection, out / "data.json")

    (out / "ground_truth.json").write_text(
        json.dumps(
            {
                "k_solute": SJ_K,
                "v0": SJ_V0,
                "c0": SJ_C0,
                "t_end": SJ_T_END,
                "t_jump": SJ_T_JUMP,
                "delta_v_bolus": SJ_DELTA_V_BOLUS,
                "c_feed": SJ_C_FEED,
            },
            indent=2,
        )
        + "\n"
    )


# ===========================================================================
# demo_continuous_overflow
# ===========================================================================

CO_INITIAL_VOLUME = 0.5  # L
CO_OVERFLOW_VOLUME = 1.0  # L
CO_INITIAL_BIOMASS = 0.5  # g/L
CO_INITIAL_GLUCOSE = 5.0  # g/L
CO_FEED_GLUCOSE = 50.0  # g/L
CO_FLOW_RATE = 0.1  # L/h
CO_YIELD_XS = 0.5  # g biomass / g glucose
CO_MU_MAX = 0.5  # 1/h
CO_KS = 0.5  # g/L
CO_END_TIME = 40.0  # h
CO_BATCH_END = (
    np.log((CO_INITIAL_BIOMASS + CO_YIELD_XS * CO_INITIAL_GLUCOSE) / CO_INITIAL_BIOMASS)
    / CO_MU_MAX
)
CO_FEED_START = CO_BATCH_END + 1.0
CO_OVERFLOW_START = (
    CO_FEED_START + (CO_OVERFLOW_VOLUME - CO_INITIAL_VOLUME) / CO_FLOW_RATE
)
CO_SAMPLE_TIMES = np.linspace(0.0, CO_END_TIME, 21)


def _co_cumulative_series(breaks, coeffs, values) -> TimeSeries:
    """An exact piecewise-linear cumulative-volume trace."""
    return TimeSeries(
        times=breaks,
        values=values,
        poly=PPoly(breaks, coeffs),
        segment_start_piece_idx=[0],
    )


def _simulate_continuous_overflow() -> dict[str, np.ndarray]:
    """RK4 on amounts, splitting the grid exactly at both flow changes."""
    dt = 0.002
    grid = np.unique(
        np.concatenate(
            [
                np.arange(0.0, CO_END_TIME, dt),
                CO_SAMPLE_TIMES,
                [CO_FEED_START, CO_OVERFLOW_START, CO_END_TIME],
            ]
        )
    )
    amounts = np.empty((grid.size, 3))
    amounts[0] = [
        CO_INITIAL_BIOMASS * CO_INITIAL_VOLUME,
        CO_INITIAL_GLUCOSE * CO_INITIAL_VOLUME,
        CO_INITIAL_VOLUME,
    ]

    def derivative(state, feed_rate, overflow_rate):
        biomass_amount, glucose_amount, volume = state
        biomass = biomass_amount / volume
        glucose = max(glucose_amount / volume, 0.0)
        mu = CO_MU_MAX * glucose / (CO_KS + glucose)
        return np.asarray(
            [
                mu * biomass_amount - overflow_rate * biomass,
                -mu * biomass_amount / CO_YIELD_XS
                + feed_rate * CO_FEED_GLUCOSE
                - overflow_rate * glucose,
                feed_rate - overflow_rate,
            ]
        )

    for i, (start, end) in enumerate(pairwise(grid)):
        step = end - start
        midpoint = (start + end) / 2
        feed_rate = CO_FLOW_RATE if midpoint >= CO_FEED_START else 0.0
        overflow_rate = CO_FLOW_RATE if midpoint >= CO_OVERFLOW_START else 0.0
        state = amounts[i]
        k1 = derivative(state, feed_rate, overflow_rate)
        k2 = derivative(state + step * k1 / 2, feed_rate, overflow_rate)
        k3 = derivative(state + step * k2 / 2, feed_rate, overflow_rate)
        k4 = derivative(state + step * k3, feed_rate, overflow_rate)
        amounts[i + 1] = state + step * (k1 + 2 * k2 + 2 * k3 + k4) / 6

    sample_indices = np.searchsorted(grid, CO_SAMPLE_TIMES)
    sampled = amounts[sample_indices]
    return {
        "biomass": sampled[:, 0] / sampled[:, 2],
        "glucose": sampled[:, 1] / sampled[:, 2],
    }


def build_demo_continuous_overflow() -> None:
    """Write the batch-to-continuous process used by the gallery example."""
    out = OUT / "demo_continuous_overflow"
    out.mkdir(parents=True, exist_ok=True)
    measurements = _simulate_continuous_overflow()

    feed = _co_cumulative_series(
        [0.0, CO_FEED_START, CO_END_TIME],
        [[0.0, 0.0, 0.0, 0.0], [0.0, CO_FLOW_RATE, 0.0, 0.0]],
        [0.0, 0.0, CO_FLOW_RATE * (CO_END_TIME - CO_FEED_START)],
    )
    overflow = _co_cumulative_series(
        [0.0, CO_OVERFLOW_START, CO_END_TIME],
        [[0.0, 0.0, 0.0, 0.0], [0.0, -CO_FLOW_RATE, 0.0, 0.0]],
        [0.0, 0.0, -CO_FLOW_RATE * (CO_END_TIME - CO_OVERFLOW_START)],
    )
    components = {
        name: hxf.ReactorMediumComponent(
            name=name,
            unit="g/L",
            concentration=TimeSeries(times=CO_SAMPLE_TIMES, values=values),
        )
        for name, values in measurements.items()
    }
    process = hxf.BioProcess(
        metadata=hxf.BioProcessMetadata(
            name="continuous_1",
            process_type="continuous",
            notes=(
                "Noiseless batch-to-continuous Monod simulation for the "
                "documentation gallery."
            ),
        ),
        time_axis=hxf.TimeAxis(
            unit="h", start=0.0, end=CO_END_TIME, time_reference="inoculation"
        ),
        volume=hxf.Volume(
            initial_volume=CO_INITIAL_VOLUME,
            unit="L",
            volume_changes={
                "feed": hxf.Inflow(
                    name="feed",
                    unit="L",
                    is_controlled=True,
                    is_continuous=True,
                    values=feed,
                    feed_medium=hxf.FeedMedium(
                        name="fresh_medium",
                        density=1.0,
                        density_unit="kg/L",
                        components={
                            "biomass": hxf.FeedMediumComponent(
                                name="biomass",
                                unit="g/L",
                                concentration=hxf.StaticVariable(0.0),
                                is_controlled=False,
                            ),
                            "glucose": hxf.FeedMediumComponent(
                                name="glucose",
                                unit="g/L",
                                concentration=hxf.StaticVariable(CO_FEED_GLUCOSE),
                                is_controlled=False,
                            ),
                        },
                    ),
                ),
                "overflow": hxf.Outflow(
                    name="overflow",
                    unit="L",
                    is_controlled=True,
                    is_continuous=True,
                    values=overflow,
                ),
            },
        ),
        reactor_medium=hxf.ReactorMedium(
            name="broth",
            density=1.0,
            density_unit="kg/L",
            components=components,
        ),
        biological_ode=hxf.BiologicalOde(
            rates={"mu": (None, None)},
            derivatives={
                "biomass": "mu * biomass",
                "glucose": f"-mu * biomass / {CO_YIELD_XS}",
            },
        ),
        discrete_events=hxf.DiscreteEvents(
            times=np.asarray([CO_FEED_START, CO_OVERFLOW_START]),
            labels=["feed starts", "overflow starts"],
        ),
    )
    collection = hxf.BioProcessCollection(
        case_id="demo_continuous_overflow",
        organism="None (synthetic Monod culture)",
        citation="Simulated data — Hybrax documentation demo, not an experiment.",
        processes={"continuous_1": process},
    )
    hxf.serialization.save_process_collection(collection, out / "data.json")
    (out / "ground_truth.json").write_text(
        json.dumps(
            {
                "initial_volume": CO_INITIAL_VOLUME,
                "overflow_volume": CO_OVERFLOW_VOLUME,
                "initial_biomass": CO_INITIAL_BIOMASS,
                "initial_glucose": CO_INITIAL_GLUCOSE,
                "feed_glucose": CO_FEED_GLUCOSE,
                "flow_rate": CO_FLOW_RATE,
                "yield_xs": CO_YIELD_XS,
                "mu_max": CO_MU_MAX,
                "Ks": CO_KS,
                "batch_end": CO_BATCH_END,
                "feed_start": CO_FEED_START,
                "overflow_start": CO_OVERFLOW_START,
                "end_time": CO_END_TIME,
            },
            indent=2,
        )
        + "\n"
    )


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    build_demo_batch()
    build_demo_fedbatch()
    build_demo_products()
    build_demo_ecoli_fba()
    build_demo_ecoli_blend()
    build_demo_optfed()
    build_demo_glutamine_decay()
    build_demo_continuous_overflow()
    build_demo_modeled_pv()
    build_demo_spline_jump()
    print(f"demo datasets written to {OUT}")


if __name__ == "__main__":
    main()
