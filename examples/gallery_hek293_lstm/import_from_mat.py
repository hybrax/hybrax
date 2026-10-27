"""Convert the pinned synthetic data into a Hybrax collection."""

import json
from pathlib import Path

import numpy as np
from scipy.io import loadmat

import hybrax.format as hxf

WORK = Path(__file__).resolve().parent
source = loadmat(WORK / "source-data.mat", squeeze_me=True, struct_as_record=False)[
    "data"
]

state_names = source[0].names.tolist()
# Hybrax validation expects this component name; keep all other source IDs.
state_names[0] = "biomass"
units = ["g/L", "mg/L"] + ["mM"] * (len(state_names) - 2)

train_names = ("Br1", "Br2", "Br3", "Br4", "Br9")

times = source[0].time
assert len(source) == 9
assert all(np.array_equal(batch.time, times) for batch in source)

N_RATES = 4
# Stack all cultures: rows are measurement times, columns are the 25 states.
# m_r already contains cumulative reacted amounts, not concentrations.
reacted = np.concatenate([batch.m_r.T for batch in source])

# Normalize each column so states with larger amounts do not dominate PCA.
# Reacted amounts can be negative when a component is consumed.
amount_scale = np.abs(reacted).max(axis=0)

# No mean subtraction: this matches MATLAB's uncentered PCA via SVD.
# Rows of vt are principal directions; singular values rank their importance.
_, singular_values, vt = np.linalg.svd(reacted / amount_scale, full_matrices=False)

# Keep four directions as columns, then restore each state's amount scale.
# The resulting 25-by-4 matrix maps learned mode rates to reaction terms.
coefficients = vt[:N_RATES].T
reaction_matrix = amount_scale[:, None] * coefficients

# For uncentered PCA this fraction measures retained squared magnitude,
# rather than variance about the column means.
explained = np.sum(singular_values[:N_RATES] ** 2) / np.sum(singular_values**2)
print(
    f"Four modes retain {explained:.2%} of normalized reacted-amount squared magnitude"
)

rate_names = [f"r_pc{i + 1}" for i in range(N_RATES)]

# Write one equation per state: biomass * sum(M[state, mode] * r_mode).
# The matrix is fixed; the reaction module learns only the four mode rates.
reaction_ode = hxf.ReactionOde(
    rates=dict.fromkeys(rate_names),
    derivatives={
        name: f"{state_names[0]} * ("
        + " + ".join(
            f"({coefficient:.16g}) * {rate}"
            for coefficient, rate in zip(row, rate_names, strict=True)
        )
        + ")"
        for name, row in zip(state_names, reaction_matrix, strict=True)
    },
)

# Feed formulation extracted from the pinned workbook; see provenance.json.
feed = json.loads((WORK / "feed.json").read_text())
assert feed["names"][1:] == source[0].names[1:].tolist()
feed_concentrations = feed["concentrations"]

feed_medium = hxf.FeedMedium(
    name="HEK293_feed",
    components={
        name: hxf.FeedMediumComponent(
            name=name, unit=unit, concentration=hxf.StaticVariable(float(value))
        )
        for name, unit, value in zip(
            state_names, units, feed_concentrations, strict=True
        )
    },
)

processes = {}
for i, batch in enumerate(source, 1):
    name = f"Br{i}"
    # Convert mL to L; concentration values are already in the assigned units.
    volume = batch.vol / 1000
    processes[name] = hxf.BioProcess(
        metadata=hxf.BioProcessMetadata(name=name, process_type="fed_batch"),
        time_axis=hxf.TimeAxis(
            unit="h",
            start=float(times[0]),
            end=float(times[-1]),
            time_reference="inoculation",
        ),
        reactor_medium=hxf.ReactorMedium(
            name="HEK293_medium",
            components={
                state: hxf.ReactorMediumComponent(
                    name=state,
                    unit=unit,
                    concentration=hxf.TimeSeries(times=times, values=values),
                )
                for state, unit, values in zip(
                    state_names, units, batch.conc, strict=True
                )
            },
        ),
        volume=hxf.Volume(
            initial_volume=float(volume[0]),
            unit="L",
            total_volume=hxf.TimeSeries(times=times, values=volume),
            volume_changes={
                "feed": hxf.Inflow(
                    name="feed",
                    unit="L",
                    is_controlled=True,
                    is_continuous=True,
                    feed_medium=feed_medium,
                    # With no removals, added feed equals the volume increase.
                    values=hxf.TimeSeries(times=times, values=volume - volume[0]),
                )
            },
        ),
        reaction_ode=reaction_ode,
        # Linear cumulative-feed interpolation gives interval-constant rates.
        # Tell the solver where those rates can jump.
        discrete_events=hxf.DiscreteEvents(times=times[1:-1]),
    )

collection = hxf.BioProcessCollection(
    case_id="ramos_2024_synthetic_HEK293",
    organism="HEK293 (synthetic)",
    citation="Ramos et al. (2024), doi:10.1002/bit.28668; synthetic repository data",
    processes=processes,
)

hxf.serialization.save_process_collection(collection, WORK / "data.json")

# Match MATLAB's use of all nine cultures for preprocessing. Concentrations
# and volume are kept in physical units here; time is in hours. The hooks use
# these means/sample SDs directly, without an intermediate max scaling.
# Loss normalization is separate from input z-scoring and reacted-amount PCA.
features = np.concatenate(
    [np.column_stack([batch.conc.T, batch.vol / 1000, batch.time]) for batch in source]
)

config = {
    "data": {"prepared": "prepared", "processes": list(train_names)},
    "custom_py": "custom.py",
    "train": {
        "epochs": 20000,
        "seed": 0,
        "allow_stateful_models": True,
        "learning_rate": 5e-5,
        "devices": 1,
        "holdout_processes": ["Br7"],
    },
    "checkpoint": {
        "every": 10,
        "keep_best": 20,
        "select_by": "holdout_loss",
        "bundle_prepared": False,
    },
    "output": {"dir": "run"},
    "custom": {
        "state_names": state_names,
        "loss_maxima": np.concatenate([b.conc.T for b in source]).max(axis=0).tolist(),
        "feature_center": features.mean(axis=0).tolist(),
        "feature_spread": features.std(axis=0, ddof=1).tolist(),
    },
}

(WORK / "train-config.json").write_text(json.dumps(config, indent=2) + "\n")
