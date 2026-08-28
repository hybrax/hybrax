# Silent failures

> The handful of ways to get a confident, plausible, wrong answer with no exception
> anywhere. Got an error instead? That's the better outcome: see [Errors](errors.md).

hybrax is built on "fail fast over silent fallbacks", and mostly it does. What
follows is the residue: the cases that cannot or do not raise. They are worth knowing
before you need them, because none of them announce themselves.

---

## 1. A misspelled hook name

**The failure.** `custom.py` defines `build_reaction_modul`. hybrax looks hooks up by
plain attribute lookup, finds nothing, and uses the default MLP. Nothing raises. Your
module never runs.

**Why it is hard to spot.** Training works. The loss goes down. It is just not your model.

**How to catch it.** Every `prepare`, `train` and `loo` run logs which hooks it found at
startup: `<stage> hooks detected: ...` and `<stage> hooks default: ...`. A hook you meant
to customise showing up in the `default` line is the tell. Failing that:

```python
import hybrax.train as hxt
wrapper, config = hxt.model_load("run")
hxt.print_trainable_structure(wrapper)
```

If you wrote a module with a field called `mu_max` and the structure shows
`reaction_module.mlp.layers[0].weight`, the default is running.

**Rule of thumb.** If an edit to `custom.py` appears to change nothing at all, it probably
changed nothing at all. Check the startup log line, then the spelling, every time.

---

## 2. No `estimate_all_scales`

**The failure.** The hook is optional. Omit it and every `SCALE_*` axis is 1.0: SCL space
is identical to RAW space, and the entire scaled-integration design is inert.

**Why it is hard to spot.** Nothing errors. On a small, well-conditioned dataset it even
converges. On real data it produces a model that trains badly for reasons that look like
stiffness, bad architecture or a bad learning rate: anything except the actual cause.

**How to catch it.** Look at the **initial** loss, before any learning has happened. A
well-scaled model starts within an order of magnitude or two of the data.
[Tutorial 4](../tutorials/04_your_first_custom_py.md#did-it-help) measures the gap on the
demo dataset: two orders of magnitude in initial loss, on data that is deliberately easy.

Also check `grad_norm_curve.png`. A raw gradient norm pinned at `grad_clip_norm` for the
whole run means your effective step size is not the learning rate.

**Fix.** [Scaling](../train/scaling.md).

---

## 3. `model_reload` across datasets

**The failure.** `model_reload` reuses the static half (including every `SCALE_*`) from
whatever you hand it, instead of rebuilding it from the run directory. Point it at a
different dataset and the trained weights are loaded into a **different scaled space**.

**Why it is hard to spot.** No exception, no shape mismatch, no `NaN`. The predictions are
smooth, plausible, and wrong by a scale factor per axis.

**Fix.** Use `model_load(run_dir)`, which rebuilds everything from the directory's own
bundled data. Reach for `model_reload` only when you specifically know why.

---

## 4. `forward` needs a working `custom.py`; `model_predict` does not

**The failure.** Two prediction paths, different dependencies:

| Path | Rebuilds the reaction module? | Needs `custom.py`? |
|---|---|---|
| `hybrax forward` / `forward_from_collection` | Yes, by re-running your hooks against the run's own recorded training input | Yes |
| `model_predict(trained_wrapper, ...)` | No, uses the wrapper you already loaded | No |

Both paths end up with the same `SCALE_*` values either way: `forward_from_collection`
never re-estimates scales against the collection you hand it for evaluation, only
against the model's own hash-verified training input. So pointing `forward`'s
`data.prepared` at a different campaign changes what gets predicted, not what scale it
is predicted in.

**Why it is hard to spot.** The dependency on `custom.py` only bites when the file has
moved, been deleted, or edited since training. A structurally different reconstruction
against the same trained weights usually raises at deserialisation rather than
predicting silently, so this is closer to a loud failure than the rest of this page.
The genuinely silent part is subtler: a `custom_py` that still resolves and still
produces a *structurally compatible* module, but a behaviourally different one (a hook
edited to compute something differently, without changing its return shapes), loads
without complaint and forward-predicts with the new code, not the code the model was
actually trained with.

**Fix.** For a checkpoint you intend to keep evaluating, do not edit `custom.py` after
training. If you must, keep a copy alongside the run directory rather than editing the
shared file other runs also point at.

---

## 5. Double-scaling in the reaction module

**The failure.** Your network reads `inputs.SCL_*`, so its output is already in SCL space.
If you also apply a `scale_*` helper to that output, it cancels against the wrapper's
unscale step and your rates are off by the scale factor.

**Why it is hard to spot.** Both conventions appear in the shipped examples, correctly,
because they depend on what the network is defined to emit:

- `examples/00_e2e_sim/custom.py` emits SCL directly and its comments warn against
  re-scaling;
- `tests/fixtures/martens_single/custom.py` computes in RAW and *does* call
  `scale_modeled_ReactionOde_rates`.

A beginner comparing the two concludes the code is inconsistent. It is not: they are
answering different questions.

**The rule.**

> **If the input was SCL, the output is SCL.**
> If you unscaled the inputs to compute in physical units, scale the output back.

Never mix. Ask "what space is this number in?" at every line of `__call__`. See
[The Reaction Module](../train/reaction_module.md#the-sclraw-convention).

---

## 6. `build_pseudobatch_transform` does not attach itself

**The failure.** It writes `c_star_concentration` onto every component *in place*, and
**returns** the transform bundle, but does not set `process.pseudobatch_transform`.
Ignore the return value and the components look transformed while the process has no
transform attached.

**Fix.**

```python
process.pseudobatch_transform = build_pseudobatch_transform(process)
```

**Related.** It also fills `volume.total_volume` only when that is currently `None`.

---

## 7. Feed composition with a species left out

**The failure.** A feed medium that omits a reactor species. hybrax.format will not guess
whether that means "absent" or "not recorded", so the dilution term for that species is
simply not generated.

**Why it is hard to spot.** The run integrates fine. One species is just never diluted by
that feed, and the model absorbs the discrepancy into its rates, which then look
physically wrong for reasons that are not obvious.

**How to catch it.** `validate_process` reports it. Run it.

**Fix.** Declare every reactor species in every feed medium, including the zeros.

---

## 8. Sample volume recorded as zero

**The failure.** Writing `0.0` for an unknown sample volume asserts that sampling removed
nothing. That is a claim, and usually a false one.

**Why it matters.** Volume is the denominator of every concentration and of every dilution
term. Ten 8 mL samples out of 1 L is nearly 8% of the vessel.

**Fix.** If sample volumes are unknown, that is under-specified metadata: treat it as
such rather than encoding a zero. If they are known, record them.

---

## 9. Bounds that do not bound

**The failure.** `bounds=(0.0, None)` on a concentration looks like a constraint. Nothing
in hybrax.format or the solver enforces it.

**Why it exists.** Bounds are *metadata*, so downstream consumers (`hybrax.train`'s loss module) can build soft penalties from a declaration you made once in the data.

**Fix.** If you want the constraint enforced, write the penalty. See
[The Loss Module](../train/loss_module.md#adding-a-physical-penalty).

---

## The general diagnostic

When a model fits the concentrations but something feels wrong, **plot the rates**.

A model can match every measurement with rates that are physically impossible: growth and
death both far too high, uptake compensating for a missing transport term, formation and
degradation cancelling. Compensating errors are invisible in a concentration plot and
obvious in a rate plot. That is why every `hybrax.train` process figure puts the rates in the
right-hand column, and it is the check most people skip.

The other strong one is a **transport-only run**: set the biological rates to zero and
integrate. Concentrations may then change only through feed composition, dilution and
sampling. Anything else that moves is a bookkeeping bug, found before you fit anything.

## See also

- [Errors](errors.md): the loud failures.
- [Limits and gotchas](../format/limits_and_gotchas.md): the hybrax.format equivalent.
- [Scaling](../train/scaling.md): the source of items 2, 3, 4 and 5.
