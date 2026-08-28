"""Tutorial 1: turn a CSV of experimental measurements into a
hybrax.format BioProcessCollection you can save, share and train on.

See docs/source/tutorials/01_your_first_dataset.md for the narrated version.
"""

from pathlib import Path

import numpy as np
import pandas as pd

import hybrax.format as hxf

HERE = Path(__file__).parent

# --- 1.1 Example data -------------------------------------------------------
df = pd.read_csv(HERE / "offline.csv")
run_1 = (
    df[df["run"] == "run_1"]
    .sort_values("time_h")
    .rename(
        columns={
            "time_h": "time",
            "biomass_gL": "biomass",
            "glucose_gL": "glucose",
            "product_gL": "product",
        }
    )
)
print(
    len(run_1), "samples from", run_1["time"].iloc[0], "to", run_1["time"].iloc[-1], "h"
)

# --- 1.3 Concentrations become ReactorMediumComponents ----------------------
components = {
    name: hxf.ReactorMediumComponent(
        name=name,
        unit="g/L",
        concentration=hxf.TimeSeries(
            times=run_1["time"].to_numpy(),
            values=run_1[name].to_numpy(),
        ),
        bounds=(0.0, None),
    )
    for name in ("biomass", "glucose", "product")
}

reactor_medium = hxf.ReactorMedium(
    name="defined_medium",
    density=1.0,
    density_unit="kg/L",
    components=components,
)

# --- 1.4 The clock and the vessel -------------------------------------------
time_axis = hxf.TimeAxis(
    unit="h",
    start=0.0,
    end=14.0,
    time_reference="inoculation",
)
volume = hxf.Volume(initial_volume=1.0, unit="L")

# --- 1.5 Assemble the BioProcess --------------------------------------------
process = hxf.BioProcess(
    metadata=hxf.BioProcessMetadata(
        name="run_1",
        process_type="batch",
        notes="Simulated E. coli batch culture on glucose.",
    ),
    time_axis=time_axis,
    volume=volume,
    reactor_medium=reactor_medium,
)
print("rates      :", list(process.reaction_ode.rates))
print("derivatives:", process.reaction_ode.derivatives)

# --- 1.6 Collect and save ----------------------------------------------------
collection = hxf.BioProcessCollection(
    case_id="my_first_dataset",
    organism="Escherichia coli",
    citation="Simulated data: tutorial only.",
    processes={"run_1": process},
)

out = HERE / "out"
out.mkdir(exist_ok=True)
hxf.serialization.save_process_collection(collection, out / "data.json")
print(f"wrote {out / 'data.json'}")

# --- 1.7 Check the round trip ------------------------------------------------
reloaded = hxf.serialization.load_process_collection(out / "data.json")
run = reloaded.processes["run_1"]
print("runs        :", list(reloaded.processes))
print("components  :", list(run.reactor_medium.components))
print(
    "first 3 t   :",
    np.asarray(run.reactor_medium.components["biomass"].concentration.times)[:3],
)
print(
    "first 3 X   :",
    np.asarray(run.reactor_medium.components["biomass"].concentration.values)[:3],
)
