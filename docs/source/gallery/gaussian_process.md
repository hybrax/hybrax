---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
kernelspec:
  display_name: Python 3
  language: python
  name: python3
---

# A Gaussian-process model

> **Demonstrates.** A sparse Gaussian process, mean and variance both, occupying the
> reaction module's slot instead of a neural network, trained end to end by bp-train's
> own optimizer, with the predictive uncertainty read out through
> `ReactionOutputs.auxiliary`.

Inspired by Helleckes et al. 2024 <a href="#ref-helleckes2024">[1]</a>, who fit a GP-based
hybrid model for bioprocess rates. This page is not a replication:
their GP is fit by maximum-likelihood estimation on a precomputed rate target, ours is
trained by gradient descent through the continuous ODE solve, which is bp-train's own
training loop and nothing else. The mechanism (an SE kernel with automatic relevance
determination, predicting a rate from the current state) is the same; how it gets fit
is not. See [Knowledge transfer](knowledge_transfer.md) for the paper's other headline
result, pooling data across products, reproduced the same way.

The walkthrough below shows the file in pieces, next to the reasoning for each one. For
the whole thing at once: to copy, diff against your own, or just read top to bottom:

:::{dropdown} Full `custom.py`
```{literalinclude} _files/gaussian_process_custom.py
:language: python
:linenos:
```
:::

```{code-cell} ipython3
:tags: [remove-cell]

import os, shutil, subprocess, sys, textwrap
from pathlib import Path
%matplotlib inline

WORK = Path("../_data/out/runs/gallery_gaussian_process").resolve()
if WORK.exists():
    shutil.rmtree(WORK)
WORK.mkdir(parents=True)
shutil.copy(Path("../_data/out/demo_batch/data.json").resolve(), WORK / "data.json")
shutil.copy(Path("_files/gaussian_process_custom.py").resolve(), WORK / "custom.py")

ENV = {**os.environ, "JAX_PLATFORMS": "cpu", "BP_TRAIN_DEVICES": "1",
       "MPLBACKEND": "Agg"}

def bp_train_cli(*args):
    proc = subprocess.run([sys.executable, "-m", "bp_train.cli", *args],
                          cwd=WORK, env=ENV, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(proc.stdout + proc.stderr)
    return proc.stdout + proc.stderr

(WORK / "prepare-config.json").write_text(
    '{ "prepare": { "raw_input": "data.json" } }\n')
(WORK / "train-config.json").write_text(textwrap.dedent("""\
    {
      "data": { "prepared": "prepared" },
      "custom_py": "custom.py",
      "train": { "epochs": 800, "seed": 0, "learning_rate": 0.01 },
      "output": { "dir": "run" }
    }
    """))
(WORK / "forward-config.json").write_text('{ "models": ["run"] }\n')

import csv
import numpy as np
import matplotlib.pyplot as plt
import bp_format as bp
import bp_train

_collection = bp.serialization.load_process_collection(WORK / "data.json")

def r2_by_target(run_dir):
    rows_by_process = {}
    with (WORK / run_dir / "predictions.csv").open() as fh:
        for row in csv.DictReader(fh):
            rows_by_process.setdefault(row["process"], []).append(row)
    per_target = {}
    for name, process in _collection.processes.items():
        rows = rows_by_process[name]
        t_pred = np.array([float(r["t"]) for r in rows])
        for species in ("biomass", "glucose", "product"):
            comp = process.reactor_medium.components[species].concentration
            t_meas, y_meas = np.asarray(comp.times), np.asarray(comp.values)
            y_pred = np.interp(t_meas, t_pred,
                               np.array([float(r[f"c_{species}"]) for r in rows]))
            ss_res = np.sum((y_meas - y_pred) ** 2)
            ss_tot = np.sum((y_meas - y_meas.mean()) ** 2)
            per_target.setdefault(species, []).append(1 - ss_res / ss_tot)
    return {k: float(np.mean(v)) for k, v in per_target.items()}
```

## The kernel

A squared-exponential kernel with automatic relevance determination (ARD): one
lengthscale per input feature, so the module can learn that some state axes matter
more than others to "how similar are these two states."

```{literalinclude} _files/gaussian_process_custom.py
:language: python
:linenos:
:lines: 28-49
```

`centers` are the inducing points (`Z` in the usual GP notation): trainable locations
in state space, not a subsample of real data. `pseudo_targets` are their corresponding
outputs (`y`). Both start random and move under gradient descent, same as any other
`trainable_field()`.

## The posterior

```{literalinclude} _files/gaussian_process_custom.py
:language: python
:linenos:
:lines: 51-66
```

`jax.scipy.linalg.cho_factor`/`cho_solve` over the `n_inducing × n_inducing` kernel
matrix gives a real closed-form GP posterior: `mean` is the predictive mean (what gets
returned as the rate), `var` is the predictive variance at the current state. This is
not an RBF network only mimicking a GP's mean, it is the actual posterior computation,
just fit by a different procedure than a textbook GP.

The predictive std goes straight into `auxiliary`, which bp-train threads into
`predictions.csv` as extra columns: no new plumbing needed.

```{literalinclude} _files/gaussian_process_custom.py
:language: python
:linenos:
:lines: 69-71
```

`estimate_all_scales` is unchanged from [Tutorial 4](../tutorials/04_your_first_custom_py.md).

## Training

```{code-cell} ipython3
:tags: [remove-input]

bp_train_cli("prepare", "--config", "prepare-config.json",
         "--output-dir", "prepared", "--overwrite")
out = bp_train_cli("train", "--config", "train-config.json", "--overwrite", "--no-plot")
print([l for l in out.splitlines() if "training complete" in l][0])
print(f"run directory: ./{(WORK / 'run').relative_to(WORK.parents[4])}")
```

```{code-cell} ipython3
:tags: [remove-input]

r2 = r2_by_target("run")
for name, value in r2.items():
    print(f"{name:10s} R2 = {value:.4f}")
```

The GP's inductive bias, smoothness via the kernel, is a reasonable prior here: the
R² values above are what that prior achieves against this page's own data, which is
the question this page sets out to answer.

```{code-cell} ipython3
:tags: [remove-input]

bp_train_cli("forward", "--config", "forward-config.json",
         "--output-dir", "run/forward", "--overwrite")
from IPython.display import Image
Image(filename=str(WORK / "run/forward/run_1.png"))
```

Left column: measured points against the integrated trajectory. Right column: the
rates the GP actually predicted along the way, `q_biomass`/`q_glucose`/`q_product`,
which never enter the loss directly, only their integral does.

## What the GP actually learned

```{code-cell} ipython3
:tags: [remove-input]

import jax.numpy as jnp

wrapper, cfg = bp_train.model_load(str(WORK / "run"))
gp = wrapper.reaction_module
print(f"centers: spread (std) per feature = {jnp.std(gp.centers, axis=0)}")
print(f"lengthscale (exp)   = {jnp.exp(gp.log_lengthscale)}")
print(f"output_scale (exp)  = {float(jnp.exp(gp.log_output_scale)):.3f}")
print(f"noise (exp)         = {float(jnp.exp(gp.log_noise)):.3f}")
```

The inducing points spread across the trajectory's real state range rather than
collapsing to a point or exploding: a degenerate fit would show up here as a near-zero
or enormous spread. Lengthscales that stay the same order of magnitude as the state
itself (roughly 1, since states are in SCL space) mean the kernel is using genuine
spatial structure, not memorizing individual points (lengthscale near zero) or ignoring
the input entirely (lengthscale huge).

## Reading out the uncertainty

```{code-cell} ipython3
:tags: [remove-input]

rows = [r for r in csv.DictReader((WORK / "run" / "predictions.csv").open())
        if r["process"] == "run_1"]
t = np.array([float(r["t"]) for r in rows])

fig, axes = plt.subplots(1, 3, figsize=(11, 3), sharex=True)
for ax, (qcol, stdcol) in zip(axes, [("q_biomass", "aux_rate_std_0"),
                                      ("q_glucose", "aux_rate_std_1"),
                                      ("q_product", "aux_rate_std_2")]):
    q = np.array([float(r[qcol]) for r in rows])
    std = np.array([float(r[stdcol]) for r in rows])
    ax.plot(t, q, color="tab:blue")
    ax.fill_between(t, q - 2 * std, q + 2 * std, alpha=0.25, color="tab:blue")
    ax.set_title(qcol)
    ax.set_xlabel("t (h)")
fig.suptitle("run_1: predicted rate ± 2·rate_std")
fig.tight_layout()
```

`aux_rate_std_0/1/2` are `ReactionOutputs.auxiliary["rate_std"]`, one column per rate,
written out with no extra code on top of what every other page already does. All three
bands are identical width at a given timestep: the kernel is shared across every rate
output, so only `pseudo_targets` differs between them, meaning the predictive variance
does not either. That is a real simplification, not a bug: an independent kernel per
rate is a natural next step, not built here. The band narrows noticeably over the
trajectory here, roughly halving between `t=0` and the final measurement: a real
signal from the closed-form posterior, not a fixed constant dressed up as
uncertainty.

## Gotchas

- **The kernel is shared across all rate outputs.** `rate_std` is one number per
  timestep, not one per rate: read it as "how confident is the model here," not
  "how confident is the model about *this specific* rate."
- **`n_inducing` is a capacity knob**, exactly like a network's width. Too few and the
  GP cannot represent the trajectory; too many and every training step pays for an
  `n_inducing × n_inducing` Cholesky factorization it does not need.
- **No marginal-likelihood-fit hyperparameters here.** `log_lengthscale`,
  `log_output_scale`, `log_noise` are just more trainable leaves, moved by the same
  Adam step as everything else. A textbook GP fits these by maximizing the marginal
  likelihood directly; this one never computes that quantity.

## See also

Run the example yourself at `./source/_data/out/runs/gallery_gaussian_process/`.

- [Knowledge transfer](knowledge_transfer.md): the same GP, extended to pool data
  across products.
- [Mechanistic models](mechanistic_rates.md): a reaction module built from explicit
  kinetics instead.
- [The reaction module](../train/reaction_module.md): `auxiliary`, and everything else
  a `UserReactionModule` can return.

## References

1. <a id="ref-helleckes2024"></a>Helleckes, L. M., Wirnsperger, C., Polak, J.,
   Guillén-Gosálbez, G., Butté, A., & von Stosch, M. (2024). Novel calibration
   design improves knowledge transfer across products for the characterization of
   pharmaceutical bioprocesses. *Biotechnology Journal*, 19(7), e202400080.
   [https://doi.org/10.1002/biot.202400080](https://doi.org/10.1002/biot.202400080)
