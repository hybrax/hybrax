"""Augmenting a single fed-batch run: splines first, then a monotonicity fix.

Two things a plain "just turn on augmentation" attempt misses:

1. Augmentation resamples each modeled state onto new timepoints, which needs
   a fitted spline, not just the raw measured samples. `demo_fedbatch` (like
   any freshly loaded hybrax.format file) has none yet: `transform_process_collection`
   fits one per modeled reactor-medium component before augmentation runs.
2. Default augmentation adds independent Gaussian noise to every measured
   timepoint. That is fine for glucose or biomass, but ``product`` here is a
   cumulative quantity: it should never decrease, and independent noise can
   easily produce a synthetic child where it does. ``augment_state_values``
   fixes exactly that, only for the state that needs it.
"""

import numpy as np

from hybrax.format.splines import fit_timeseries_spline


def transform_process_collection(collection, config):
    del config
    for process in collection.processes.values():
        for component in process.reactor_medium.components.values():
            component.concentration = fit_timeseries_spline(
                component.concentration, smoothing_s=0.0)
    return collection


def augment_state_values(*, parent_name, child_name, state_name, times,
                         base_values, augmented_values, config):
    del parent_name, child_name, times, base_values, config
    if state_name != "product":
        return augmented_values
    # Default noise already computed; just repair monotonicity in-place.
    return np.maximum.accumulate(augmented_values)
