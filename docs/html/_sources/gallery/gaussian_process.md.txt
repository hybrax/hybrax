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

# A Gaussian-Process Model

> This page replaces the neural network with a Gaussian process, a model that reports
> its own confidence alongside every prediction. It learns from real measured data, and
> hybrax trains it the same way it trains every other model in this gallery.

This page is inspired by two papers. Helleckes et al. 2024 <a href="#ref-helleckes2024">[1]</a>
fit a Gaussian process to real bioprocess measurements to predict reaction rates.
Cruz-Bournazou et al. 2022 <a href="#ref-cruz2022">[2]</a> showed that training a
Gaussian process on a smooth curve's derivative works better than using the raw
difference between two measurements. This page builds a Gaussian process reaction
module that combines both ideas. See [How This Compares to the Papers](#how-this-compares-to-the-papers)
at the end of this page for exactly where it matches each paper and where it departs.

The walkthrough below shows the file in pieces, next to the reasoning for each one. For
the whole thing at once: to copy, diff against your own, or just read top to bottom:

:::{dropdown} Full `custom.py`
```{literalinclude} ../../../examples/gallery_gaussian_process/custom.py
:language: python
:linenos:
```
:::

```{code-cell} ipython3
:tags: [remove-cell]

import os, shutil, subprocess, sys
from pathlib import Path
%matplotlib inline

WORK = Path("../_data/out/runs/gallery_gaussian_process").resolve()
if WORK.exists():
    shutil.rmtree(WORK)
WORK.mkdir(parents=True)
EXAMPLE = Path("../../../examples/gallery_gaussian_process").resolve()
shutil.copy(EXAMPLE / "data.json", WORK / "data.json")
shutil.copy(EXAMPLE / "custom.py", WORK / "custom.py")

ENV = {**os.environ, "JAX_PLATFORMS": "cpu", "HYBRAX_TRAIN_DEVICES": "1",
       "MPLBACKEND": "Agg"}

def hxt_cli(*args):
    proc = subprocess.run([sys.executable, "-m", "hybrax.train.cli", *args],
                          cwd=WORK, env=ENV, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(proc.stdout + proc.stderr)
    return proc.stdout + proc.stderr

shutil.copy(EXAMPLE / "prepare-config.json", WORK / "prepare-config.json")
shutil.copy(EXAMPLE / "train-config.json", WORK / "train-config.json")
shutil.copy(EXAMPLE / "forward-config.json", WORK / "forward-config.json")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import hybrax.format as hxf
import hybrax.train as hxt

_collection = hxf.serialization.load_process_collection(WORK / "data.json")

def r2_by_target(run_dir):
    df = pd.read_csv(WORK / run_dir / "predictions.csv")
    per_target = {}
    for name, process in _collection.processes.items():
        proc_df = df[df["process"] == name]
        t_pred = proc_df["t"].to_numpy()
        for species in ("biomass", "glucose", "product"):
            comp = process.reactor_medium.components[species].concentration
            t_meas, y_meas = np.asarray(comp.times), np.asarray(comp.values)
            y_pred = np.interp(t_meas, t_pred, proc_df[f"c_{species}"].to_numpy())
            ss_res = np.sum((y_meas - y_pred) ** 2)
            ss_tot = np.sum((y_meas - y_meas.mean()) ** 2)
            per_target.setdefault(species, []).append(1 - ss_res / ss_tot)
    return {k: float(np.mean(v)) for k, v in per_target.items()}
```

## The Kernel

A squared-exponential kernel with automatic relevance determination (ARD): one
lengthscale per input feature, so the module can learn that some state axes matter
more than others to "how similar are these two states."

```{literalinclude} ../../../examples/gallery_gaussian_process/custom.py
:language: python
:linenos:
:lines: 38-57
```

`centers` are real states (`Z` in the usual GP notation), one per real measurement,
built by `build_reaction_module` below. `targets` are the real rate estimates at those
same states (`y`). Both are `frozen_field()`: real data, held fixed through training.
`log_lengthscale`, `log_output_scale`, and `log_noise` stay `trainable_field()`: they
are what training actually fits.

## The Posterior

```{literalinclude} ../../../examples/gallery_gaussian_process/custom.py
:language: python
:linenos:
:lines: 59-78
```

`jax.scipy.linalg.cho_factor`/`cho_solve` over the real, full kernel matrix give a
genuine closed-form GP posterior, over the model's real training data: `mean` is the
predictive mean (what gets returned as the rate), `var` is the predictive variance at
the current state.

The predictive std goes straight into `auxiliary`, which `hybrax.train` threads into
`predictions.csv` as extra columns: no new plumbing needed.

`estimate_all_scales` is unchanged from [Tutorial 4](../tutorials/04_your_first_custom_py.md).

## The Training Data

```{literalinclude} ../../../examples/gallery_gaussian_process/custom.py
:language: python
:linenos:
:lines: 96-135
```

For every process, {py:func}`hybrax.format.splines.build_pseudobatch_transform` and
{py:func}`hybrax.format.splines.build_backtransform_spline` fit a smooth spline through
the real measured concentrations. Calling `.derivative()` on that spline gives the real
`dc/dt` at every real measurement time. Dividing by the real biomass value at that time
turns it into a specific rate, matching the declared ODE: `biomass' = q_biomass * biomass`.

This is Cruz-Bournazou et al. 2022's core idea <a href="#ref-cruz2022">[2]</a>: a
smooth curve's derivative is a better rate estimate than a raw finite difference
between two points. The pseudobatch spline is the general form of that curve: it fits
the dilution-corrected concentration first, so a bolus or sampling event's jump never
distorts the fit. This gallery's data has no such event, but the same code handles one
correctly either way.

`rmc_scaler.scale_value` converts a real state to SCL space. A rate needs the
derivative-specific method instead, `rate_scaler.scale_derivative`: subtracting a
value-space offset from a derivative would silently corrupt it.

## Fitting the Kernel by Marginal Likelihood

```{literalinclude} ../../../examples/gallery_gaussian_process/custom.py
:language: python
:linenos:
:lines: 80-93
```

`marginal_nll` is the standard GP negative log marginal likelihood, computed from the
module's own real `centers` and `targets`. It's the quantity a textbook GP fit
maximizes, reusing the same Cholesky factor `__call__` already computes.

```{literalinclude} ../../../examples/gallery_gaussian_process/custom.py
:language: python
:linenos:
:lines: 138-163
```

`GPLossModule` adds one more named loss, `gp_nll`, to the usual per-target trajectory
MSE terms `DefaultLossModule` already provides. `inputs.reaction_module` is the live
reaction-module instance, so `marginal_nll()` is reachable directly from inside the
loss. Both terms drive the same gradient step: the trajectory loss keeps the whole
model anchored to the real measured concentrations, and `gp_nll` fits the kernel
hyperparameters the way a real GP does. `nll_weight` scales `gp_nll` down to keep it
in the same rough range as the trajectory terms, since `hybrax.train` averages named
losses and a much larger term would dominate the average.

## Training

```{code-cell} ipython3
:tags: [remove-input]

hxt_cli("prepare", "--config", "prepare-config.json",
         "--output-dir", "prepared", "--overwrite")
out = hxt_cli("train", "--config", "train-config.json", "--overwrite")
lines = [l for l in out.splitlines() if "training complete" in l]
print(lines[0] if lines else "training complete")
print(f"run directory: ./{(WORK / 'run').relative_to(WORK.parents[4])}")
```

```{code-cell} ipython3
:tags: [remove-input]

r2 = r2_by_target("run")
for name, value in r2.items():
    print(f"{name:10s} R2 = {value:.4f}")
```

These R² values measure how well the trained GP's trajectory matches the real
measurements.

```{code-cell} ipython3
:tags: [remove-input]

hxt_cli("forward", "--config", "forward-config.json",
         "--output-dir", "run/forward", "--overwrite")
from IPython.display import Image
Image(filename=str(WORK / "run/forward/plots/run_1.png"))
```

Left column: measured points against the integrated trajectory. Right column: the
rates the GP actually predicted along the way, `q_biomass`/`q_glucose`/`q_product`,
which enter the trajectory loss only through their integral.

## What the GP Actually Learned

```{code-cell} ipython3
:tags: [remove-input]

import jax.numpy as jnp

wrapper, cfg = hxt.model_load(str(WORK / "run"))
gp = wrapper.reaction_module
print(f"centers: spread (std) per feature = {jnp.std(gp.centers, axis=0)}")
print(f"lengthscale (exp)   = {jnp.exp(gp.log_lengthscale)}")
print(f"output_scale (exp)  = {float(jnp.exp(gp.log_output_scale)):.3f}")
print(f"noise (exp)         = {float(jnp.exp(gp.log_noise)):.3f}")
```

`centers`' spread reflects the real training data: a `frozen_field()` stays fixed
through training. Lengthscales near the same order of magnitude as the state itself
(roughly 1, since states are in SCL space) mean the kernel uses genuine spatial
structure. A lengthscale near zero means the kernel is memorizing individual points; a
huge lengthscale means it is ignoring that input entirely. Those lengthscale,
output-scale, and noise values are what `gp_nll` actually fit.

```{code-cell} ipython3
:tags: [remove-input]

metrics = pd.read_csv(WORK / "run" / "metrics.csv")
loss_names = metrics["target_names"].iloc[0].split(";")
gp_nll_index = loss_names.index("gp_nll")
gp_nll = metrics["per_target_loss"].apply(lambda row: float(row.split(";")[gp_nll_index]))
print(f"gp_nll: first = {gp_nll.iloc[0]:.4f}, last = {gp_nll.iloc[-1]:.4f}")
```

`gp_nll` falling over training shows the marginal-likelihood term actually moving the
kernel hyperparameters toward a better fit to the real training pairs.

## Reading Out the Uncertainty

```{code-cell} ipython3
:tags: [remove-input]

df = pd.read_csv(WORK / "run" / "predictions.csv")
run_1 = df[df["process"] == "run_1"]
t = run_1["t"].to_numpy()

fig, axes = plt.subplots(1, 3, figsize=(11, 3), sharex=True)
for ax, (qcol, stdcol) in zip(axes, [("q_biomass", "aux_rate_std_0"),
                                      ("q_glucose", "aux_rate_std_1"),
                                      ("q_product", "aux_rate_std_2")]):
    q = run_1[qcol].to_numpy()
    std = run_1[stdcol].to_numpy()
    ax.plot(t, q, color="tab:blue")
    ax.fill_between(t, q - 2 * std, q + 2 * std, alpha=0.25, color="tab:blue")
    ax.set_title(qcol)
    ax.set_xlabel("t (h)")
fig.suptitle("run_1: predicted rate ± 2·rate_std")
fig.tight_layout()
```

`aux_rate_std_0/1/2` are `ReactionOutputs.auxiliary["rate_std"]`, one column per rate,
written out with no extra code on top of what every other page already does. All three
bands are identical width at a given timestep, since the kernel is shared across every
rate output and only `targets` differs between them. That is a real simplification: an
independent kernel per rate is a natural next step, left for later.

The band widens somewhat around the middle of the trajectory and narrows again toward
the end. That shape comes directly from the kernel: `rate_std` grows where `run_1`'s
own state is farther from the pooled real training states across all three runs, and
shrinks where it sits close to them.

## Gotchas

- **The kernel is shared across all rate outputs.** `rate_std` is one number per
  timestep, not one per rate: read it as "how confident is the model here," not
  "how confident is the model about *this specific* rate."
- **Two objectives fit the kernel hyperparameters together.** `gp_nll` and the
  trajectory loss both move `log_lengthscale`, `log_output_scale`, and `log_noise`
  every step. A textbook GP maximizes marginal likelihood alone; the fitted values here
  are a compromise between the two, weighted by `nll_weight`.
- **`nll_weight` needs tuning if you change the data or the epoch budget.**
  `hybrax.train` averages named losses, so an unscaled `gp_nll` can either get drowned
  out by the per-target MSE terms or dominate them. Watch `gp_nll` and the per-target
  losses together in `metrics.csv`: both should fall steadily, together, through
  training.

## How This Compares to the Papers

The kernel, the real training data, and the marginal-likelihood objective for the
kernel hyperparameters all match Helleckes et al. 2024's paper
<a href="#ref-helleckes2024">[1]</a>. Their kernel hyperparameters are fit by
likelihood alone, once, then held fixed for prediction. Here they are fit by
likelihood and trajectory error together, continuously, so the fitted values differ
from what a pure-likelihood fit would find. See
[Knowledge transfer](knowledge_transfer.md) for the paper's other headline result,
pooling data across products, reproduced the same way.

Cruz-Bournazou et al. 2022's core idea <a href="#ref-cruz2022">[2]</a>, training a GP
on a smooth curve's derivative instead of a raw difference between two points, builds
the real rate targets above. Their further step, iteratively refitting that curve
against the GP's own predictions, stays out of scope here: the trajectory loss plays a
comparable role, through real continuous integration.

## See Also

Run the example yourself at `./source/_data/out/runs/gallery_gaussian_process/`.

- [Knowledge transfer](knowledge_transfer.md): the same GP, extended to pool data
  across products.
- [Pseudobatch splines](pseudobatch_splines.md): `build_pseudobatch_transform` and
  `build_backtransform_spline`, on their own, without a reaction module.
- [Mechanistic models](mechanistic_rates.md): a reaction module built from explicit
  kinetics instead.
- [The Reaction Module](../train/reaction_module.md): `auxiliary`, and everything else
  a `UserReactionModule` can return.
- [The Loss Module](../train/loss_module.md): `LossInputs.reaction_module`, and
  everything else a custom `UserLossModule` can do.

## References

1. <a id="ref-helleckes2024"></a>Helleckes, L. M., Wirnsperger, C., Polak, J.,
   Guillén-Gosálbez, G., Butté, A., & von Stosch, M. (2024). Novel calibration
   design improves knowledge transfer across products for the characterization of
   pharmaceutical bioprocesses. *Biotechnology Journal*, 19(7), e202400080.
   [https://doi.org/10.1002/biot.202400080](https://doi.org/10.1002/biot.202400080)
2. <a id="ref-cruz2022"></a>Cruz-Bournazou, M. N., Narayanan, H., Fagnani, A., &
   Butté, A. (2022). Hybrid Gaussian process models for continuous time series in
   bolus fed-batch cultures. *IFAC-PapersOnLine*, 55(7), 204-209.
   [https://doi.org/10.1016/j.ifacol.2022.07.445](https://doi.org/10.1016/j.ifacol.2022.07.445)
