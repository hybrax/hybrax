"""Gallery: knowledge transfer.

Pools data from several products to help a data-poor new one, using a
constant-valued controlled process variable as a one-hot product-identity
feature, and an ensemble of GPs anchored to real training data. Compares a
model trained only on the new product's 2 runs ("local") against one trained
on those plus 24 historical runs ("pooled"), evaluated on 2 held-out runs at
the extremes of a design space the new product's own training runs never
covered.

See docs/source/gallery/knowledge_transfer.md for the narrated version.
"""

import os
import subprocess
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import hybrax.format as hxf
import hybrax.train as hxt

HERE = Path(__file__).parent
ENV = {
    **os.environ,
    "JAX_PLATFORMS": "cpu",
    "HYBRAX_TRAIN_DEVICES": "1",
    "MPLBACKEND": "Agg",
}


def hxt_cli(*args):
    proc = subprocess.run(
        [sys.executable, "-m", "hybrax.train.cli", *args],
        cwd=HERE,
        env=ENV,
        capture_output=True,
        text=True,
    )
    if proc.returncode != 0:
        raise RuntimeError(proc.stdout + proc.stderr)
    return proc.stdout + proc.stderr


_all = hxf.serialization.load_process_collection(HERE / "data.json")
_historical = {n: p for n, p in _all.processes.items() if not n.startswith("T_")}
_t_train = {n: p for n, p in _all.processes.items() if n in ("T_run_1", "T_run_2")}
_t_heldout = {n: p for n, p in _all.processes.items() if n in ("T_run_3", "T_run_4")}

hxf.serialization.save_process_collection(
    hxf.BioProcessCollection(processes={**_historical, **_t_train}),
    HERE / "pooled.json",
)
hxf.serialization.save_process_collection(
    hxf.BioProcessCollection(processes=dict(_t_train)), HERE / "local.json"
)
hxf.serialization.save_process_collection(
    hxf.BioProcessCollection(processes=dict(_t_heldout)), HERE / "heldout.json"
)

for variant in ("local", "pooled"):
    hxt_cli(
        "prepare",
        "--config",
        f"prepare-{variant}.json",
        "--output-dir",
        f"prepared_{variant}",
        "--overwrite",
    )
    hxt_cli("train", "--config", f"train-{variant}.json", "--overwrite")

_heldout = hxf.serialization.load_process_collection(HERE / "heldout.json")


def r2_by_target(run_dir):
    wrapper, cfg = hxt.model_load(str(HERE / run_dir))
    preds = hxt.model_predict(wrapper, cfg, _heldout, grid_n=200)
    species_order = list(wrapper.modeled_RMC_names)
    per_target = {s: [] for s in ("biomass", "glucose", "product")}
    for proc_name, export in preds.items():
        process = _heldout.processes[proc_name]
        for species in per_target:
            comp = process.reactor_medium.components[species].concentration
            t_meas, y_meas = np.asarray(comp.times), np.asarray(comp.values)
            idx = species_order.index(species)
            y_pred = np.interp(
                t_meas, np.asarray(export.t), np.asarray(export.c_species)[:, idx]
            )
            ss_res = np.sum((y_meas - y_pred) ** 2)
            ss_tot = np.sum((y_meas - y_meas.mean()) ** 2)
            per_target[species].append(1 - ss_res / ss_tot)
    return {k: float(np.mean(v)) for k, v in per_target.items()}


r2_local = r2_by_target("run_local")
r2_pooled = r2_by_target("run_pooled")
print(f"{'target':10s} {'local':>10s} {'pooled':>10s}")
for name in ("biomass", "glucose", "product"):
    print(f"{name:10s} {r2_local[name]:10.4f} {r2_pooled[name]:10.4f}")

_process_name = "T_run_4"
_process = _heldout.processes[_process_name]

fig, axes = plt.subplots(1, 3, figsize=(13, 3.5))
_species_list = ("biomass", "glucose", "product")

for run_dir, label, style in [
    ("run_local", "local", "--"),
    ("run_pooled", "pooled", "-"),
]:
    wrapper, cfg = hxt.model_load(str(HERE / run_dir))
    preds = hxt.model_predict(wrapper, cfg, _heldout, grid_n=200)
    export = preds[_process_name]
    species_order = list(wrapper.modeled_RMC_names)
    for ax, species in zip(axes, _species_list):
        idx = species_order.index(species)
        ax.plot(export.t, np.asarray(export.c_species)[:, idx], style, label=label)

for ax, species in zip(axes, _species_list):
    comp = _process.reactor_medium.components[species].concentration
    ax.plot(comp.times, comp.values, "ko", ms=4, label="measured")
    ax.set_title(species)
    ax.set_xlabel("t (h)")
axes[0].legend(fontsize=8)
fig.suptitle(f"Held-out {_process_name}: local vs. pooled")
fig.tight_layout()
out = HERE / "out"
out.mkdir(exist_ok=True)
fig.savefig(out / "local_vs_pooled.png")
print(f"wrote {out / 'local_vs_pooled.png'}")
