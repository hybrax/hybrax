"""Convert the published OptFed CSVs to a mass-based Hybrax collection."""

from pathlib import Path

import numpy as np
import pandas as pd

import hybrax.format as hxf
from hybrax.format.time_series import TimeSeries

REVISION = "7811f7b9ef76d624f53b852c03d06cb1b5cc96a3"
SOURCE = f"https://raw.githubusercontent.com/gschloegel/OptFed/{REVISION}/data"
KINETIC_UPTAKE = {"DoE1_R1", "DoE2_R1", "DoE3_R2"}
SPECIES = {"biomass": "X", "glycerol": "G", "product": "P"}
GLYCEROL_DENSITY = 1.261  # kg/L; use OptFed's additive-volume approximation


def control_series(frame, column, end, *, cumulative=False):
    """Clip an online trace to induction through the last retained measurement."""
    times = np.r_[0.0, frame.t[(frame.t > 0) & (frame.t < end)], end]
    values = np.interp(times, frame.t, frame[column])
    if cumulative:
        values = values - values[0]
    return TimeSeries(times=times, values=values)


def build_collection(source=SOURCE):
    """Read four upstream CSVs; source can also be a local directory."""
    measurements = pd.read_csv(f"{source}/sampling_points.csv", index_col=0)
    flows = pd.read_csv(f"{source}/volume_flow_rates.csv", index_col=0)
    temperatures = pd.read_csv(f"{source}/temperatures.csv", index_col=0)
    feed_concentrations = pd.read_csv(f"{source}/c_feed.csv", index_col="process")

    # The authors exclude this anomalous DoE1_R3 measurement in 01_fit.py.
    measurements = measurements.drop(index=40)
    processes = {}
    for name, measured in measurements.groupby("process", sort=True):
        flow = flows[flows.process == name]
        temperature = temperatures[temperatures.process == name]
        end = float(measured.t.iloc[-1])
        concentration = float(feed_concentrations.loc[name, "c_feed"])
        # The CSV reports feed concentration in g/L, but feeds and V in kg.
        density = 1 + concentration / 1000 * (1 - 1 / GLYCEROL_DENSITY)
        feed_glycerol = concentration / density
        feeds = {}
        for feed_name, column, glycerol, feed_density in (
            ("carbon_feed", "f_cum", feed_glycerol, density),
            ("base_feed", "f_base_cum", 0.0, 1.0),
        ):
            feeds[feed_name] = hxf.Inflow(
                name=feed_name,
                unit="kg",
                is_controlled=True,
                is_continuous=True,
                values=control_series(flow, column, end, cumulative=True),
                feed_medium=hxf.FeedMedium(
                    name=feed_name,
                    density=feed_density,
                    density_unit="kg/L",
                    components={
                        species: hxf.FeedMediumComponent(
                            name=species,
                            unit="g/kg",
                            is_controlled=species == "glycerol",
                            concentration=hxf.StaticVariable(
                                glycerol if species == "glycerol" else 0.0
                            ),
                        )
                        for species in SPECIES
                    },
                ),
            )

        # Downward reactor-mass jumps estimate individual withdrawals. Do not
        # cumulatively sum them: a discrete Outflow stores each event's delta.
        withdrawals = flow.V.diff()
        draws = (withdrawals < 0) & (flow.t > 0) & (flow.t <= end)
        feeds["sampling"] = hxf.Outflow(
            name="sampling",
            unit="kg",
            is_controlled=True,
            is_continuous=False,
            values=TimeSeries(times=flow.t[draws], values=withdrawals[draws]),
        )
        temp = control_series(temperature, "T", end)
        processes[name] = hxf.BioProcess(
            metadata=hxf.BioProcessMetadata(
                name=name,
                process_type="fedbatch",
                notes="Published E. coli protein L production measurements.",
            ),
            time_axis=hxf.TimeAxis(
                unit="h", start=0.0, end=end, time_reference="induction"
            ),
            volume=hxf.Volume(
                initial_volume=float(measured.V.iloc[0]),
                unit="kg",
                volume_changes=feeds,
            ),
            reactor_medium=hxf.ReactorMedium(
                name="reactor_medium",
                density=1.0,
                density_unit="kg/L",
                components={
                    species: hxf.ReactorMediumComponent(
                        name=species,
                        unit="g/kg",
                        concentration=TimeSeries(
                            times=measured.t, values=measured[column]
                        ),
                    )
                    for species, column in SPECIES.items()
                },
            ),
            process_variables={
                "temperature": hxf.ProcessVariable(
                    name="temperature", unit="K", is_controlled=True, values=temp
                ),
                "supply_limited_uptake": hxf.ProcessVariable(
                    name="supply_limited_uptake",
                    unit="1",
                    is_controlled=True,
                    values=TimeSeries(
                        times=[0.0, end],
                        values=[float(name not in KINETIC_UPTAKE)] * 2,
                    ),
                ),
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
    return hxf.BioProcessCollection(
        case_id="optfed",
        organism="Escherichia coli",
        citation="Schlögl et al. (2024), doi:10.1016/j.csbj.2024.09.024",
        metadata={"source": SOURCE, "excluded_measurement_row": 40},
        processes=processes,
    )


if __name__ == "__main__":
    hxf.serialization.save_process_collection(
        build_collection(), Path(__file__).with_name("data.json")
    )
