# Final Blocking Design Questions

These are the remaining decisions that materially affect the detailed spec. Everything else is concrete enough to write down already.

## 1. Custom preprocessing injection point

Question:
Where should case-study-specific preprocessing live for v1?

Recommended answer:
Use a `custom.py` pattern similar to `hybrax-train`, but with explicit default hooks:
- `transform_controls(process, config) -> process`
- `transform_states(process, config) -> process`
- `augment_training_data(training_data, config) -> training_data`

If the user does not provide custom code, the defaults are identity functions.

Why this matters:
You identified a real need to derive rates from cumulative traces and possibly construct new traces like `D = (CF + BF) / V`. The spec needs one concrete extension mechanism.

Your answer: I guess we can put all the custom code into `custom.py` (including model definition and preprocessing for controls etc.). Let's go with the function signatures you suggested for now (but ignore augmentation)?


## 2. Main artifact after preprocessing

Question:
After these custom transforms run, what should be the main persisted artifact for v1?

Recommended answer:
A new JSON file in `bp_format` structure, derived from the original JSON, with updated `Interpolator` fields and metadata flags recording which transforms were applied. Training code should consume this transformed JSON, not re-run the custom transform path on every call.

Why this matters:
This decides whether preprocessing is part of runtime or an explicit build/prep stage.

Your answer: Yes, let's have a new JSON but it should contain all the fields from the original JSON and not just the Controls etc.


## 3. Feed and volume semantics in the runtime wrapper

Question:
What exactly should the library-managed RHS wrapper do with feed and volume information in v1?

Recommended answer:
The wrapper should:
- read feed-rate controls from the controls object,
- maintain volume as part of the integrated state when configured as dynamic,
- compute dilution/transport terms internally,
- call the user RHS only for reaction/source terms in concentration space.

The user RHS should not implement dilution logic manually in the default path.

Why this matters:
This is the core ergonomics promise of the package.

Your answer: Yes, the suggestion sounds good. The wrapper should check the Controls object for feed variables and handle dilution etc. internally. For now we only support the dynamic volume case (i.e. volume as state variable). Long-term we can expose a flag so that the user can handle dilution themselves in their RHS but for v1 we do it in library code.


## 4. Volume as control vs volume as state

Question:
Do you want v1 to support both:
- fixed/controlled volume supplied by controls
- dynamic volume integrated as a state

or only dynamic volume as a state?

Recommended answer:
Support both in the API, but prioritize and test only the dynamic-volume-as-state path in v1. Fixed-volume or externally supplied volume can be allowed as a simpler fallback mode.

Why this matters:
This affects the state vector contract and wrapper logic.

Your answer: For v1 only support dynamic volume (integrated as state variable). If only volume and no feed data is available in the input the user can derive the feed from the derivative of the volume in the Controls pre-processing hook. This should be good enough for now.


## 5. Multiple feed streams

Question:
How should multiple feed streams be represented when the wrapper computes dilution internally?

Recommended answer:
Represent them explicitly as separate controls with associated inlet composition metadata. The wrapper sums transport contributions over all feed streams; it should not collapse them into one scalar control.

Why this matters:
This determines whether the design stays general across bioprocess cases.

Your answer: Yes, handle as separate controls. Keep in mind that some feeds might also be a state variable and modeled (e.g. base feed for pH control could be modeled based on growth rate etc.). This might make the plumbing for dilution etc. more difficult, but as long as well know the indices of the feed streams in controls and states the wrapper should be able to handle this. However, if you think it complicates things more than is worth we can also consider making the user handle dilution in their RHS for v1.


## 6. Observation model boundary

Question:
Should `observe()` remain part of the model abstraction in v1, or should observed variables be computed only after integration as a post-processing step?

Recommended answer:
Keep `observe()` in the abstraction, but default it to identity and allow the trainer to call it post-integration. That preserves flexibility without forcing observation modeling into the first implementation.

Why this matters:
You explicitly raised doubt about whether `observe()` is needed, and the spec should not leave that ambiguous.

Your answer: One complication re. this will come once we introduce stateful models (LSTMs etc.) as they will give different outputs during the integration vs. if queried after the integration to recreate some latent variables (e.g. growth rate). For now we're going to ignore this but the spec should mention it. For now go with the recommended answer and keep `observe()` in the abstraction but default it to identity.


## 7. Controls object iteration API

Question:
You mentioned wanting an iterator over segments for `lax.scan`. Do you want that to be part of the public controls API in v1, or just an internal helper?

Recommended answer:
Keep public API minimal and make segment iteration an internal helper first. Public API can stay:
- `segment_times`
- `jump_times`
- `eval_segment(...)`
- `eval_segment_batch(...)`

The library can internally materialize scan-friendly arrays.

Why this matters:
This decides whether we freeze a low-level iteration protocol too early.

Your answer: I think I changed my mind re. this (at least for v1). For now, let's ignore segments (sampling and bolus) and do the following during Controls pre-processing:
- Create dense grid for each segment with enough points so that max. lin. interp. error is below threshold (later we can use adaptive point selection to use fewer points in regions with low 2nd derivative, but for now let's just use a simple fixed number of points per segment and increase it if error is too high)
- Combine segments into single linear interpolator.
- Keep track of start + end of each segment boundary and pass as `step_ts` to step size controller downstream.
- Model Bolus feed events as short (but not instantaneous) periods of high feed rate in the Controls object. Their duration should be the shortest time difference in the original online data (i.e. the highest sampling frequency, `np.diff(time_arr).min()`). Later we could replace this with something more sophisticated but for v1 this is good enough.
We should make sure to have good tests for this though to make sure the bolus events are properly represented (and exactly the right amount of feed is added etc) and that we don't get many rejected solver steps.

I realise that this is a fairly late change of hearts. LMK if you got any more questions about this.


## 8. Required metadata additions in transformed JSON

Question:
What minimum metadata should the transformed JSON record about the prep pipeline?

Recommended answer:
At minimum:
- source input path or source dataset id
- prep timestamp
- config hash or config path
- applied transform hook names
- control ordering
- whether volume is dynamic or controlled

Why this matters:
Without this, the transformed artifact will be hard to audit and debug.

Your answer: Sounds good. We should also change things like `is_controlled` of variables based on the config. I also like the idea of having hashes (of the config and the input JSON). Long-term we want some better way of tracking provenance and lineage of these artifacts but for now this should be good enough.
