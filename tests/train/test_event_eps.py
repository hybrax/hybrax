"""Event matching in ``physical_solve`` must be exact, and must fire each event once.

``affect_fn`` identifies the firing bolus/sample from the preset **index** the
solver hands it (``preset_times[preset_index]``), then compares stored event times
to that value exactly. Two failure modes are pinned here, one on each side of that
choice:

**Double application.** The historical implementation matched
``|t - bt| < _EVENT_EPS`` against the solver's *realised* stop time, with
``_EVENT_EPS = 1e-4``. Node distinctness, however, is decided by a far tighter
tolerance in ``diffrax_callbacks._solve._find_next_preset_time``
(``2 * eps(dtype) * (1 + |t|)``). Any output node separated from a real event by
more than the step tolerance but less than
``1e-4`` was therefore accepted as its own segment boundary *and* still matched that
event's window, so the bolus/sample fired a SECOND time and the error persisted for the
rest of the trajectory. That needed a guard which parked such nodes; exact matching
removes the failure mode outright, so the guard is gone and these tests pass with no
guard present.

The window used to be, relative to an event at t=1.0:

===================  ==========  ==========
offset               float32     float64
===================  ==========  ==========
0                    safe        safe
1e-15 .. 1e-7        safe        DOUBLED
1e-6 .. 5e-5         DOUBLED     DOUBLED
>= 2e-4              safe        safe
===================  ==========  ==========

Exactly-coincident nodes were always safe (the strictly-future test rejects the
duplicate slot), which is why every measurement grid in the repo's fixtures and
datasets missed
this: it only ever bit on dense/prediction export grids, where a uniform grid point can
land just after a feed.

**Missed application** — the mirror risk that exact matching introduces. If a
preset entry were ever not bit-identical to its source event time, the match would
silently find nothing and the feed would simply never happen: volume and substrate
too low for the rest
of the run, no error, and a perfectly plausible-looking trajectory. Passing the index
(rather than a time) makes the lookup exact by construction, and
``test_every_active_event_fires_exactly_once`` pins it.
"""

from __future__ import annotations

from dataclasses import fields

import equinox as eqx
import jax.numpy as jnp
import numpy as np
import pytest

from stateful_helpers import (
    ZeroLatentDerivativeModule,
    build_stateful_wrapper,
    make_process,
    solve,
)

from bp_train.model_api import LinearScaler


# ``make_process(jump=True)``: sample -0.2 L AND bolus +0.3 L, both at
# t=1.0, V0 = 1.0 L.
# Applied once, V(1.5) = 1.0 - 0.2 + 0.3 = 1.1. Applied twice, V(1.5) = 1.2.
_EVENT_T = 1.0
_V_APPLIED_ONCE = 1.1


def _wrapper(dtype):
    process = make_process(jump=True)
    wrapper = build_stateful_wrapper(
        process, ZeroLatentDerivativeModule(jnp.zeros(2, dtype=dtype))
    )
    if dtype is jnp.float32:
        return wrapper
    # ``build_stateful_wrapper`` pins its scalers to float32; promote them so the solve
    # runs in the precision production uses (bp_train enables x64 at import).
    # ``tree_at`` selectors may only traverse stored PyTree fields, not properties.
    names = [
        field.name
        for field in fields(wrapper.reaction_module)
        if field.name.startswith("SCALE_")
        and isinstance(getattr(wrapper.reaction_module, field.name), LinearScaler)
    ]
    return eqx.tree_at(
        lambda w: tuple(getattr(w.reaction_module, n) for n in names),
        wrapper,
        tuple(
            LinearScaler(
                jnp.asarray(getattr(wrapper.reaction_module, n).scale, dtype=dtype)
            )
            for n in names
        ),
    )


def _volume_at_end(offset: float, dtype) -> float:
    """``V`` at the final output time when one node sits ``offset`` from the event."""
    wrapper = _wrapper(dtype)
    t_eval = jnp.asarray([0.0, _EVENT_T + offset, 1.5], dtype=dtype)
    states = solve(
        wrapper,
        t_eval,
        jnp.asarray([1.0, 1.0], dtype=dtype),
        rtol=1e-6,
        atol=1e-8,
    )
    return float(states[-1, 1])


@pytest.mark.parametrize("dtype", [jnp.float32, jnp.float64])
@pytest.mark.parametrize("offset", [1e-9, 1e-6, 1e-5, 5e-5])
def test_node_just_after_an_event_does_not_reapply_it(offset, dtype):
    """An output node in the old hazard window AFTER a bolus/sample must not re-fire
    it."""
    assert _volume_at_end(offset, dtype) == pytest.approx(_V_APPLIED_ONCE, abs=1e-4)


@pytest.mark.parametrize("dtype", [jnp.float32, jnp.float64])
@pytest.mark.parametrize("offset", [-1e-9, -1e-6, -1e-5, -5e-5])
def test_node_just_before_an_event_does_not_reapply_it(offset, dtype):
    """Symmetric case: the old window used ``abs()``, so a node just BEFORE the event
    fired it early and the real node then fired it again."""
    assert _volume_at_end(offset, dtype) == pytest.approx(_V_APPLIED_ONCE, abs=1e-4)


@pytest.mark.parametrize("dtype", [jnp.float32, jnp.float64])
def test_node_exactly_on_an_event_applies_it_once(dtype):
    """An output time exactly on an event: the duplicate slot is rejected as
    not-strictly-future, and the co-timed sample+bolus still group into one node."""
    assert _volume_at_end(0.0, dtype) == pytest.approx(_V_APPLIED_ONCE, abs=1e-4)


@pytest.mark.parametrize("dtype", [jnp.float32, jnp.float64])
def test_node_outside_the_event_window_is_unaffected(dtype):
    """Well clear of any event the node is an ordinary output time: its own segment
    boundary, matching no event, leaving the balance untouched."""
    assert _volume_at_end(2e-4, dtype) == pytest.approx(_V_APPLIED_ONCE, abs=1e-4)


def test_every_active_event_fires_exactly_once():
    """The mirror guarantee: exact matching must never SKIP an event.

    Re-integrates the mass balance independently of the solver -- final volume must be
    ``V0 + sum(boluses) - sum(samples)`` exactly once over -- on a dense output
    grid whose points deliberately straddle the events, which is precisely the
    configuration that
    used to double-count.
    """
    dtype = jnp.float64
    wrapper = _wrapper(dtype)
    controls = wrapper.controls
    n_bolus = int(jnp.sum(controls.bolus_event_mask))
    n_sample = int(jnp.sum(controls.sample_event_mask))
    assert (n_bolus, n_sample) == (1, 1), "fixture shape changed"

    expected = (
        1.0
        + float(
            jnp.sum(
                jnp.where(controls.bolus_event_mask, controls.bolus_event_volumes, 0.0)
            )
        )
        - float(
            jnp.sum(
                jnp.where(
                    controls.sample_event_mask, controls.sample_event_volumes, 0.0
                )
            )
        )
    )

    # 61 points over the process's full measurement window [0, 2]; several land within
    # 1e-4 of the t=1.0 events by design (1.0 is itself a grid point at this spacing).
    # The span must match the measurement window: the solver sizes its per-segment
    # output window from a RELATIVE inter-event gap fraction
    # (``_output_window_bounds``), so a deliberately narrower grid inflates the true
    # fraction past the bound and trips ``output_overflow``.
    t_eval = jnp.asarray(
        np.unique(
            np.concatenate([np.linspace(0.0, 2.0, 61), [1.0 - 1e-5, 1.0 + 1e-5]])
        ),
        dtype=dtype,
    )
    states = solve(
        wrapper, t_eval, jnp.asarray([1.0, 1.0], dtype=dtype), rtol=1e-8, atol=1e-10
    )
    assert float(states[-1, 1]) == pytest.approx(expected, abs=1e-8)

    # Pin the exact volume levels the grid should see. At the event node itself the
    # gather reports the POST-sample, PRE-bolus state (the offline-sample convention),
    # so the trace visits 1.0 -> 0.8 -> 1.1 and nothing else.
    #   applied twice  -> max 1.2, min 0.6
    #   skipped        -> max 1.0
    v = np.asarray(states[:, 1])
    assert float(v.max()) == pytest.approx(1.1, abs=1e-8)
    assert float(v.min()) == pytest.approx(0.8, abs=1e-8)
