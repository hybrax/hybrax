# Migration Guide: Volume Structure Refactoring

## Overview

The BPbench data structure has been refactored to better handle volume tracking in bioprocess experiments. Volume is no longer treated as a simple "control" or "state" but has its own dedicated structure that can track multiple volume-changing operations (feeds, sampling, evaporation, etc.).

## Key Changes

### 1. Process Structure

**Before:**
```python
Process(
    process_id="batch_001",
    process_type="fed_batch",
    states={"biomass": ..., "product": ...},
    controls={"temperature": ..., "volume": ...},
    ...
)
```

**After:**
```python
Process(
    process_id="batch_001",
    process_type="fed_batch",
    dynamic_states={"biomass": ..., "product": ...},
    dynamic_controls={"temperature": ...},  # volume removed
    volume=Volume(...),  # New dedicated volume structure
    ...
)
```

### 2. New Volume Structure

The new `Volume` class contains:
- `initial_volume`: Starting volume of the reactor
- `volume_changes`: Dictionary of `VolumeChange` objects
- `volume_unit`: Unit of measurement (e.g., "L")
- `validate_volume_consistency()`: Method to check volume balance

### 3. VolumeChange Class

Each volume-changing operation is represented by a `VolumeChange`:

```python
VolumeChange(
    name="carbon_feed",
    controlled=True,  # Controlled vs. modeled
    continuous=True,  # Continuous vs. discrete
    unit="L/h",  # or "L" for discrete
    feed_medium="glucose_feed",  # Reference to feed
    timeseries=feed_rate_ts,  # For continuous
    # OR
    timepoints=jnp.array([...]),  # For discrete
    values=jnp.array([...])
)
```

## Migration Steps

### Step 1: Update Process Creation

Replace `states` and `controls` with the new field names:

```python
# Old
process = Process(
    process_id="exp_001",
    process_type="fed_batch",
    states={...},
    controls={...}
)

# New
process = Process(
    process_id="exp_001",
    process_type="fed_batch",
    dynamic_states={...},
    dynamic_controls={...}
)
```

### Step 2: Move Volume from Controls to Volume Structure

If you had volume in controls:

```python
# Old
controls = {
    "temperature": temp_ts,
    "volume": volume_ts
}

# New
dynamic_controls = {
    "temperature": temp_ts
}

# Create VolumeChange from the volume timeseries
feed_change = VolumeChange(
    name="main_feed",
    controlled=True,
    continuous=True,
    unit="L/h",
    timeseries=feed_rate_ts  # Need to convert volume to rate
)

volume = Volume(
    volume_changes={"main_feed": feed_change},
    initial_volume=1.0,
    volume_unit="L"
)
```

### Step 3: Update Access Patterns

```python
# Old
biomass = process.states["biomass"]
temp = process.controls["temperature"]

# New
biomass = process.dynamic_states["biomass"]
temp = process.dynamic_controls["temperature"]
```

## Volume Validation

The new structure includes built-in volume validation:

```python
# Validate that volume changes sum correctly
is_valid, message = process.volume.validate_volume_consistency(
    time_axis=process.time,
    final_volume=5.0  # Expected final volume
)

if not is_valid:
    print(f"Warning: {message}")
```

This will:
- Integrate continuous feed rates over time
- Sum discrete additions/removals
- Compare calculated final volume to expected
- Warn if difference exceeds 5%

## Examples

See `examples/volume_example.py` for complete examples showing:
1. Continuous fed-batch with volume validation
2. Discrete bolus additions
3. Multiple volume changes (feed + sampling)
4. Complete process with volume structure

## Benefits

1. **Clarity**: Volume is neither purely a state nor a control - it now has proper representation
2. **Validation**: Built-in consistency checking for volume balance
3. **Flexibility**: Can track multiple simultaneous volume-changing operations
4. **Feed Tracking**: Direct linkage between volume changes and feed media

## Serialization

The serialization format has been updated to support the new structure, but remains backward compatible for loading old datasets (they will be automatically converted on load).

