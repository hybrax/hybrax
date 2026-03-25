# Follow-up Design Questions

Please answer each item inline. I included the current recommended answer so you can either confirm it or replace it.

## 1. `ControlsStore.get_controls(exp_idx)` contract

Question:
What exactly should `ControlsStore.get_controls(exp_idx)` return?

Recommended answer:
A per-experiment object with:
- `segment_times`
- `jump_times`
- `eval_segment(segment_idx, t)` returning all control values at scalar `t`
- `eval_segment_batch(segment_idx, ts)` returning shape `[n_t, n_controls]`

Why this matters:
This defines the core runtime API used by the solver and tests.

Your answer: as recommended; we can add this later, but perhaps we want a method to get an iterator over the segments so that we can use them in `lax.scan` (or similar) most ergonomically and efficiently.


## 2. Control values vs derivatives

Question:
Should the store expose raw continuous control values only, or also first derivatives?

Recommended answer:
Expose both value and derivative for every stored control trace, since the dense-grid linearization strategy explicitly precomputes both and future solver paths may need derivatives.

Why this matters:
This changes the stored payload, evaluation API, and test surface.

Your answer: The store should contain both; in the future we might want to optimize this (e.g. say in the config which controls the user wants derivatives for), but for now let's keep it simple.


## 3. Continuous feed / volume-change representation

Question:
For continuous feeds or other continuous volume changes, should the store represent cumulative quantity or instantaneous rate?

Recommended answer:
Instantaneous rate. `bpbench` may store cumulative quantities, but the solver should consume a normalized, unambiguous rate representation.

Why this matters:
This determines the semantics of the control vector and avoids duplicated conversion logic downstream.

Your answer: The store should contain the rates (as those are the actual controls). However, there is some nuance here because the `bp-prep` output won't contain the information whether a trace is a cumulative quantity or not. I'm starting to think that we need a way to inject user code (i.e. case study-specific preprocessing of the traces before creating the final Controls object) somewhere into the pipeline. I suppose we should have something similar for augmenting data. There should be a script for each of those tasks (with sensible defaults that can be customized). So the user (modelling researcher) can write a function (or eqx.Module) that is injected into the controls-generation pipeline (specifying which traces to take the derivative of, for example, or if traces should be combined to create a new one (like `D = (Cf + Bf) / V`)). Similarly, they should be able to affect the data augmentation pipeline (e.g. by changing the levels of additive or multiplicative noise to add for different traces etc.). Both scripts should add fields to the JSON (i.e. the JSON coming from bp-bench and later updated would be the main artifact).
I haven't decided yet how to best do this. We could follow a similar approach to hybrax-train where any non-defaults code is in `custom.py` and the other scripts / modules import from there (or from a default implementation; which is slightly different from what hybrax is doing but would make more sense)


## 4. Unified control interface

Question:
Should temperature and other controlled process variables be treated identically to feed rates in the store interface?

Recommended answer:
Yes. Internally they may originate from different `bpbench` fields, but the exported control vector should be a deterministic ordered list of controls.

Why this matters:
This decides whether downstream code sees one generic control vector or multiple special cases.

Your answer: They can be handled identically in the store, but volume and feed should be handled differently from other controls I think. Most processes will have volume and volume changes and we should have functionality to handle those properly in a standardized way (so that the user who sets up their model by writing down the ODE RHS doesn't have to worry about dilution etc). The store should keep track which control traces are volume (if used as control and not integrated as state variable) and feed rates (of which there can be multiple). The ODE wrapper should then take care of handling those and either expose dilution in a very easy-to-use way to the wrapped RHS (implemented by the user) or handle dilution internally (i guess we could have an argument for this).


## 5. Control ordering

Question:
How should control ordering be defined?

Recommended answer:
Config-defined order first. If absent, use a deterministic fallback:
1. continuous controlled volume changes
2. controlled process variables

This matches the existing `bpbench.mechanistic.get_control_splines` intent.

Why this matters:
The order affects model inputs, serialization, reproducibility, and tests.

Your answer: Use recommended answer.


## 6. Missing control spline / interpolator

Question:
What should happen if a config marks a variable as control but the dataset lacks a spline / `Interpolator`?

Recommended answer:
Fail fast during store build with a validation error naming the process and variable.

Why this matters:
Otherwise failures move downstream into training and become much harder to debug.

Your answer: Yes, we always wanna fail fast if something doesn't check out.


## 7. Dataset says controlled, config omits it

Question:
What should happen if the config omits a variable but `bpbench` marks it `is_controlled=True`?

Recommended answer:
Warn and ignore in v1 unless `strict_config=true`. Config remains authoritative.

Why this matters:
This defines conflict resolution between dataset metadata and training config.

Your answer: Like suggested the config should be authoritative, but since we're writing out a new JSON with the controls we can change `is_controlled` in that new version to match the config.


## 8. Minimum training data object

Question:
For state variables, what is the minimum training data object we should standardize in v1?

Recommended answer:
Per experiment:
- `process_name`
- `t_meas`
- `y_meas` with columns in config order
- `y0`
- `controls`
- optional `state_interpolators` for deterministic resampling

Why this matters:
This becomes the data boundary between parsing/prep and the trainer.

Your answer: Go with recommended answer. Later augmented data will be added as well (and we need to think about creating data loaders etc), but for now this is a good start.


## 9. Model-definition abstraction

Question:
Do you want one model-definition abstraction now, or just a thin function-based API first?

Recommended answer:
A thin abstract base class now with:
- `rhs(...)`
- `observe(...)` optional, default identity
- `partition_trainable(...)` optional, default “all neural-network params trainable”

Why this matters:
This controls how much code a researcher needs to write to plug in a new hybrid model.

Your answer: There should be a RHS wrapper object (library code) that takes care of calling the controls, handling dilution, etc. It should call the user-defined RHS eqx.Module whose job is to determine the rates based on the current state and controls. For now, let's ignore stateful models (like RNNs) and only support stateless ones (e.g. FFNN).
The user RHS has to implement `__call__` (and mid-term we want to add some inspection to double check the user supplied code really has the function signatures we expect etc.) and `partition_trainable()`.
`observe()` can be optional for now (default identity) as suggested, but we should think about whether we actually need this or if we should rather run the integration and then calculate the observed variables from the integrated state traces post-hoc.


## 10. Training grid in v1

Question:
Should the trainer optimize against measurement times only in v1, or support arbitrary training grids immediately?

Recommended answer:
Measurement times only. Deterministic spline resampling can exist in the data layer, but the first train-step path should evaluate loss exactly at measured timestamps.

Why this matters:
This strongly affects data prep, batching, solver API, and test scope.

Your answer: yes, keep things simple for now and only train on real measurements. Make sure to already pad those arrays though.


## 11. Initial conditions

Question:
How should initial conditions be chosen?

Recommended answer:
From the first measured state for observed states, with config overrides for modeled-but-unobserved states. If a required initial state is missing, fail during data prep.

Why this matters:
This determines the training sample contract and failure modes.

Your answer: Go with recommended answer.


## 12. `bpbench` API change log

Question:
Do you want the spec to include the concrete `bpbench` change-log file and proposed entries, or only mention that such a file should exist?

Recommended answer:
Include it concretely. The spec should contain a `bpbench-api-notes.md` section or companion file listing required adapters and upstream changes, including the `Interpolator` / `interpolator` compatibility seam already found.

Why this matters:
This turns vague upstream dependencies into an actionable interface contract.

Your answer: Yes, we definitely want to be concrete about what needs changing in bpbench and have a file that we keep updating as we discover more things (or things are changed in bpbench).
