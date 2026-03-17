# Preliminary spec and roadmap

## Overview

There will be 3-5 packages:
- `bp-form`:
    - contains:
        - data classes with hierarchical ontology to fully describe processes, variables, etc.
        - contains metadata to describe the process, measured variables, splines (optional), etc.
        - I/O: (de)serialization to/from multiple formats
        - simple / non-optimized implementation of process simulation / model evaluation
        - calculate ADF (from V splines or raw trace) & pseudobatch transform from parsed data
    - will be used by all other packages for parsing data and process APIs
- `bp-bench`:
    - "database" of case studies that were already transformed into `bp-form` format
- `bp-prep`:
    - web app for pre-processing raw experimental data; outputs `bp-form`-compliant files
- `bp-hyb`:
    - utilities for training hybrid models on process data in `bp-form` format
    - integration routines optimized for training (compile once, padded arrays, warm-start, etc.)
    - data augmentation
    - LOO-CV, checkpointing, hyperparameter sweep, model ensemble utils
- `bp-sim`:
    - simulate data and save in `bp-form` format
    - handles discontinuities etc.
    - user can select feed profile (continuous based on function or bolus with schedule) and sampling schedule 
    - parameters determined by user-defined functions
    - simple DoE utils (Latin hypercube sampling, etc.)