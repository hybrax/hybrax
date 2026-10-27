# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "cobra==0.32.1",
#     "numpy==2.5.4",
#     "pandas==2.3.3",
#     "scipy==1.18.1",
# ]
# ///

"""Generate FBA training data for a fresh surrogate fit on e_coli_core.

Run from the repository root:

    uv run examples/gallery_fba_hyb/01_generate_fba_data.py

uv reads the dependencies above and manages an isolated environment, including
Cobra's GLPK bindings. The project's pandas dependency does not apply here.
The bundled XML input and output CSV are resolved relative to this script.
Solving 10,000 linear programs takes a couple of minutes. run.py also runs this
generator, but docs_rebuild.sh uses the frozen surrogate without regenerating it.

The output, 01_obj_fba_data.csv.gz, feeds 02_fit_surrogate.py. Its fitted
surrogate_fba body must be copied into custom.py in gallery_fba_hyb and
gallery_pls_dfba to update the frozen coefficients used during training.

Combined objective: max  n_X*biomass + n_M*ATPM + n_A*acetate_secretion
                         + n_S*succinate_secretion
at fixed glucose uptake qG. Samples (qG, n_X, n_M, n_A, n_S) via Latin Hypercube,
solves pFBA for each. n_S is included for both pages (page 1's reaction module
will simply never query it), avoiding duplicate offline LP-solving.

Mirrors the pattern in Gotsmy & Guillen-Gosalbez's FBA-Hyb method
(bioRxiv:10.64898/2026.04.22.720062v1), adapted to e_coli_core (Orth, Fleming
& Palsson 2010) -- a small, citeable, published teaching model, not a
genome-scale network -- with freshly-generated data and a freshly-fit
surrogate rather than reused coefficients.
"""

from pathlib import Path

import cobra
import numpy as np
import pandas as pd
from scipy.stats import qmc

GLC_RID = "EX_glc__D_e"
ATPM_RID = "ATPM"
BIOMASS_RID = "BIOMASS_Ecoli_core_w_GAM"
ACETATE_RID = "EX_ac_e"
SUCCINATE_RID = "EX_succ_e"
HERE = Path(__file__).resolve().parent


def load_model():
    model = cobra.io.read_sbml_model(str(HERE / "e_coli_core.xml"))
    model.solver = "glpk"

    # Glucose as the sole carbon source; uptake bound is set per-sample below.
    model.reactions.get_by_id(GLC_RID).bounds = (-1000, 0)
    # Relax ATPM's hard lower bound (8.39) so the maintenance objective weight
    # n_M actually has room to move the LP solution, matching
    # vibrio_slim's "remove the ATPM constraint" step.
    model.reactions.get_by_id(ATPM_RID).bounds = (0, 1000)

    xxx_equiv = cobra.Metabolite(
        "xxx_equiv", name="biomass equivalent", compartment="c"
    )
    atp_equiv = cobra.Metabolite(
        "atp_equiv", name="maintenance equivalent", compartment="c"
    )
    ace_equiv = cobra.Metabolite(
        "ace_equiv", name="acetate equivalent", compartment="c"
    )
    suc_equiv = cobra.Metabolite(
        "suc_equiv", name="succinate equivalent", compartment="c"
    )

    model.reactions.get_by_id(BIOMASS_RID).add_metabolites({xxx_equiv: 1})
    model.reactions.get_by_id(ATPM_RID).add_metabolites({atp_equiv: 1})
    model.reactions.get_by_id(ACETATE_RID).add_metabolites({ace_equiv: 1})
    model.reactions.get_by_id(SUCCINATE_RID).add_metabolites({suc_equiv: 1})

    return model, xxx_equiv, atp_equiv, ace_equiv, suc_equiv


def build_model(
    qG, n_X, n_M, n_A, n_S, tmp_model, xxx_equiv, atp_equiv, ace_equiv, suc_equiv
):
    reaction = cobra.Reaction("combined_obj")
    reaction.name = "combined_obj_function"
    reaction.lower_bound = 0.0
    reaction.upper_bound = 1000.0
    reaction.add_metabolites(
        {
            xxx_equiv: -n_X,
            atp_equiv: -n_M,
            ace_equiv: -n_A,
            suc_equiv: -n_S,
        }
    )
    tmp_model.add_reactions([reaction])
    tmp_model.objective = "combined_obj"
    tmp_model.objective_direction = "max"
    tmp_model.reactions.get_by_id(GLC_RID).bounds = (-qG, -qG)
    return tmp_model


if __name__ == "__main__":
    sampler = qmc.LatinHypercube(d=5, seed=20260813)
    n_samples = 10_000
    norm_sample = sampler.random(n_samples)
    # dims: qG, n_X, n_M, n_A, n_S
    scaled_sample = qmc.scale(
        norm_sample, [0.5, 0.0, 0.0, 0.0, 0.0], [20.0, 2.0, 2.0, 2.0, 2.0]
    )

    model, xxx_equiv, atp_equiv, ace_equiv, suc_equiv = load_model()

    rows = []
    n_fail = 0
    for qG, n_X, n_M, n_A, n_S in scaled_sample:
        with model as tmp:
            tmp = build_model(
                qG, n_X, n_M, n_A, n_S, tmp, xxx_equiv, atp_equiv, ace_equiv, suc_equiv
            )
            try:
                sol = cobra.flux_analysis.pfba(tmp)
            except Exception:  # noqa: BLE001 - skip failed solver samples
                n_fail += 1
                continue
            rows.append(
                np.concatenate(
                    [
                        np.array([qG, n_X, n_M, n_A, n_S]),
                        sol.fluxes.values.flatten(),
                    ]
                )
            )

    with model as tmp:
        tmp = build_model(
            1, 1, 1, 1, 1, tmp, xxx_equiv, atp_equiv, ace_equiv, suc_equiv
        )
        columns = ["qG", "n_X", "n_M", "n_A", "n_S"] + [r.id for r in tmp.reactions]

    df = pd.DataFrame(rows, columns=columns)
    print(f"solved {len(df)}/{n_samples} ({n_fail} failed)")
    output = HERE / "01_obj_fba_data.csv.gz"
    df.to_csv(output, compression={"method": "gzip", "mtime": 0})
    print(f"saved {output}")
