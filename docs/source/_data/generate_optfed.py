"""Generate six synthetic fed-batch processes for the reduced OptFed model."""

import json
from itertools import pairwise
from pathlib import Path

import numpy as np
from scipy.integrate import solve_ivp

import hybrax.format as hxf
from hybrax.format.time_series import TimeSeries

# These synthetic parameters define the demo, not a fit to experimental data.
PARAMETERS = {
    "uptake_max": 0.4,
    "glycerol_inhibition": 80.0,
    "maintenance": 0.004,
    "uptake_maintenance": 2.0,
    "product_maintenance": 4.0,
    "production_max": 2e-8,
    "production_half_saturation": 0.08,
    "generation_inhibition": 8.0,
    "temperature": [54000.0, 800000.0, 309.0],
}
# name: temperature [K], carbon feed [L/h], initial glycerol [g/L],
# whether uptake is determined by feed supply.
RUNS = {
    "limited_cool": (301.15, 0.008, 0.0, True),
    "limited_center": (305.15, 0.012, 0.0, True),
    "limited_warm": (309.15, 0.016, 0.0, True),
    "kinetic_cool": (301.15, 0.04, 20.0, False),
    "kinetic_center": (305.15, 0.05, 30.0, False),
    "kinetic_warm": (309.15, 0.06, 40.0, False),
}
SAMPLE_TIMES = np.arange(0.0, 12.1, 2.0)
FEED_GLYCEROL = 500.0
SAMPLE_VOLUME = 0.002


def simulate(temperature, feed, glycerol, supply_limited):
    """Integrate amounts, volume and generations, with sampling volume losses."""
    p = PARAMETERS
    ea, enthalpy, teq = p["temperature"]
    ceiling = (
        p["production_max"]
        * 1.38e-23
        * temperature
        / 6.63e-34
        * np.exp(-ea / (8.314 * temperature))
        / (1 + np.exp(enthalpy * (1 / teq - 1 / temperature) / 8.314))
        * 3600
    )

    def rhs(t, state):
        mx, mg, mp, volume, generations = state
        x, g, product = mx / volume, max(mg / volume, 0), mp / volume
        active = max(x - product, 1e-8)
        if supply_limited:
            uptake = feed * FEED_GLYCEROL / (volume * active)
        else:
            uptake = p["uptake_max"] * g / (0.001 + g)
            uptake /= 1 + g / p["glycerol_inhibition"]
        maintenance = p["maintenance"] * (1 + uptake * p["uptake_maintenance"])
        maintenance *= 1 + product / x * p["product_maintenance"]
        maintenance = min(maintenance, uptake)
        available = uptake - maintenance
        production = ceiling * available / (p["production_half_saturation"] + available)
        production /= 1 + generations / p["generation_inhibition"]
        production = min(production, available)
        qx = (available - production) * 0.627
        qp = production * 0.652
        return [
            (qx + qp) * active * volume,
            feed * FEED_GLYCEROL - uptake * active * volume,
            qp * active * volume,
            feed,
            (qx + qp) * active / x / np.log(2),
        ]

    state = np.asarray([20.0, glycerol, 0.0, 1.0, 0.0])
    rows = [state[:3] / state[3]]
    for start, end in pairwise(SAMPLE_TIMES):
        solution = solve_ivp(
            rhs, (start, end), state, method="Radau", rtol=1e-9, atol=1e-11
        )
        assert solution.success
        state = solution.y[:, -1]
        rows.append(state[:3] / state[3])
        # Removing a homogeneous sample leaves concentrations and generations intact.
        state[:3] *= (state[3] - SAMPLE_VOLUME) / state[3]
        state[3] -= SAMPLE_VOLUME
    return np.asarray(rows)


def build_demo_optfed():
    out = Path(__file__).parent / "out/demo_optfed"
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(20260814)
    processes = {}
    feed_medium = hxf.FeedMedium(
        name="carbon_feed",
        density=1.0,
        density_unit="kg/L",
        components={
            name: hxf.FeedMediumComponent(
                name=name,
                unit="g/L",
                concentration=hxf.StaticVariable(
                    FEED_GLYCEROL if name == "glycerol" else 0
                ),
            )
            for name in ("biomass", "glycerol", "product")
        },
    )
    for name, (temperature, feed, glycerol, supply_limited) in RUNS.items():
        truth = simulate(temperature, feed, glycerol, supply_limited)
        observed = np.maximum(truth * (1 + rng.normal(0, 0.02, truth.shape)), 0)
        observed[0] = truth[0]
        components = {
            species: hxf.ReactorMediumComponent(
                name=species,
                unit="g/L",
                bounds=(0.0, None),
                concentration=TimeSeries(times=SAMPLE_TIMES, values=observed[:, i]),
            )
            for i, species in enumerate(("biomass", "glycerol", "product"))
        }
        processes[name] = hxf.BioProcess(
            metadata=hxf.BioProcessMetadata(
                name=name,
                process_type="fed_batch",
                notes="Synthetic reduced OptFed kinetics with 2% measurement noise.",
            ),
            time_axis=hxf.TimeAxis(
                unit="h", start=0.0, end=12.0, time_reference="inoculation"
            ),
            volume=hxf.Volume(
                initial_volume=1.0,
                unit="L",
                volume_changes={
                    "carbon_feed": hxf.Inflow(
                        name="carbon_feed",
                        unit="L",
                        is_controlled=True,
                        is_continuous=True,
                        feed_medium=feed_medium,
                        values=TimeSeries(
                            times=SAMPLE_TIMES, values=feed * SAMPLE_TIMES
                        ),
                    ),
                    "sampling": hxf.Outflow(
                        name="sampling",
                        unit="L",
                        is_controlled=True,
                        is_continuous=False,
                        values=TimeSeries(
                            times=SAMPLE_TIMES[1:],
                            values=np.full(SAMPLE_TIMES[1:].shape, -SAMPLE_VOLUME),
                        ),
                    ),
                },
            ),
            reactor_medium=hxf.ReactorMedium(
                name="defined_medium",
                density=1.0,
                density_unit="kg/L",
                components=components,
            ),
            process_variables={
                variable: hxf.ProcessVariable(
                    name=variable,
                    unit=unit,
                    is_controlled=True,
                    values=TimeSeries(
                        times=SAMPLE_TIMES, values=np.full(7, value, dtype=float)
                    ),
                )
                for variable, unit, value in [
                    ("temperature", "K", temperature),
                    ("supply_limited_uptake", "1", float(supply_limited)),
                ]
            },
            reaction_ode=hxf.ReactionOde(
                algebraic={"active_biomass": "biomass - product"},
                rates={
                    name: (None, None)
                    for name in ("q_biomass", "q_glycerol", "q_product")
                },
                derivatives={
                    "biomass": "(q_biomass + q_product) * active_biomass",
                    "glycerol": "-q_glycerol * active_biomass",
                    "product": "q_product * active_biomass",
                },
            ),
        )
    collection = hxf.BioProcessCollection(
        case_id="demo_optfed",
        organism="Synthetic recombinant E. coli fed-batch",
        citation="Synthetic documentation data, not experimental measurements.",
        processes=processes,
    )
    hxf.serialization.save_process_collection(collection, out / "data.json")
    (out / "ground_truth.json").write_text(json.dumps(PARAMETERS, indent=2) + "\n")


if __name__ == "__main__":
    build_demo_optfed()
