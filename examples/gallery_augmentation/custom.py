"""Augmenting a single fed-batch run: splines first, then a chosen shape constraint.

Two things a plain "just turn on augmentation" attempt misses:

1. Augmentation resamples each modeled state onto new timepoints, which needs
   a fitted spline, not just the raw measured samples. This example's raw
   `data.json` has none yet; files can also contain saved splines.
   `transform_process_collection` fits one per modeled reactor-medium component
   before augmentation runs.
2. Default augmentation adds independent Gaussian noise to resampled values of
   states named in `noise_std`. For this hook demonstration, we impose a
   nondecreasing `product` trajectory with `augment_state_values`. This is a
   chosen shape constraint: product concentration can decrease through dilution.
"""

import numpy as np

from hybrax.format.splines import fit_timeseries_spline


def transform_process_collection(collection, config):
    del config
    for process in collection.processes.values():
        for component in process.reactor_medium.components.values():
            component.concentration = fit_timeseries_spline(
                component.concentration, smoothing_s=0.0
            )
    return collection


def augment_state_values(
    *, parent_name, child_name, state_name, times, base_values, augmented_values, config
):
    del parent_name, child_name, times, base_values, config
    if state_name != "product":
        return augmented_values
    # Apply the demonstration's shape constraint after noise and clipping.
    return np.maximum.accumulate(augmented_values)
