# Modelling comparison: `hyb-mod-fresh` and `bp-train`

Scope: supported model structures and fitting methods, not data or result formats.

## `reference/hyb-mod-fresh`

- The shipped model is a three-state (`S`, `X`, `P`) hybrid neural ODE. An FFNN
  or LSTM predicts substrate uptake and its allocation between growth, product,
  and maintenance; positivity and sum-to-one constraints are imposed before a
  mechanistic RHS computes state derivatives. Components can be replaced, but
  the shipped training path remains fixed to these states and case-specific
  controls.
- Training differentiates through Diffrax. It uses mini-batches of fixed-length,
  spline-derived noisy trajectories (some truncated), per-state MSE, AdamW, and
  optional training of one shared recurrent initial state. Independent
  leave-one-experiment-out folds use the held-out experiment for validation.

## `bp-train`

- A reaction module may be neural, mechanistic, or mixed and supplies arbitrary
  biological rates and optional modeled-feed rates to a generated physical mass
  balance. States may include any number of species, modeled process variables,
  volume, cumulative modeled feeds, and continuous latent variables. Continuous
  feeds and discrete bolus/sample events are modeled; continuous
  stream-volume-change terms are not yet supported.
- The automatic default is a stateless MLP rate model. The package also ships a
  hook-selected GRU latent-ODE with neural rate/feed heads; stateful models need
  explicit opt-in. The extension API permits other kinetic structures and
  mechanistic/neural combinations. Examples include structured
  uptake/allocation and surrogate-FBA hybrids.
- Training also differentiates through Diffrax, but fits process trajectories on
  sparse, per-cell-masked measurement grids; augmentation is optional. Tagged
  parameters in both reaction and loss modules train together. The default loss
  is scaled per-target MSE, while custom named losses can use states, rates,
  auxiliary outputs, trainable weights, or a dense time grid. Optimization is
  step-based with Adam by default (SGD or custom Optax are alternatives), with
  leave-one-or-several-process-out training supported.

## Bottom line

`hyb-mod-fresh` implements one narrow FFNN/LSTM-parameterised hybrid ODE and an
augmentation-heavy fitting scheme. `bp-train` generalises this into a broader
mechanistic-neural model and loss API; its present boundary is incomplete
transport coverage rather than neural architecture choice.
