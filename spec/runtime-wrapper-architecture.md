# BP-Train Runtime Wrapper Architecture (Current)

This diagram reflects the current runtime stack in `bp_train` for single-process
training (Phase C, Steps 1-5).

```mermaid
flowchart TD
    A["prepared.json or BioProcessCollection"] --> B["ControlsStore.from_collection"]
    A --> C["TrainingDataStore.from_collection"]
    B --> D["PerProcessControls"]
    C --> E["PerProcessTrainingData"]
    E -->|controls| D

    F["UserReactionModule eqx.Module"] --> G["LibraryRhsWrapper.from_process_controls"]
    D --> G
    G --> H["LibraryRhsWrapper call(t, y)"]

    H --> I["controls.eval(t)"]
    H --> J["V_real = V_cont - V_sample_acc"]
    H --> K["reaction_module(t, c_species, controls_vector)"]
    K --> L["ReactionOutputs with reaction_terms and modeled_feed_rates"]
    H --> M["transport and dilution from controlled and modeled feeds"]
    L --> H
    M --> H
    H --> N["dy = concat(dc_species, dV_cont)"]

    E --> O["simulate_measurement_states"]
    G --> O
    O --> P["diffrax.diffeqsolve with jump_ts = controls.active_step_ts"]
    P --> Q["state trajectory at t_meas"]
    Q --> R["single_process_measurement_loss"]
    G --> R
    R --> S["reaction_module.observe(species_states)"]
    S --> T["masked MSE vs y_meas"]
    T --> U["single_process_train_step"]

    F --> V["partition_trainable()"]
    V --> U
    U --> W["updated wrapper with updated reaction_module"]
```

## Hierarchy (Objects and Ownership)

```mermaid
classDiagram
    class ControlsStore {
      +process_order
      +global_control_names
      +dense_grid
      +control_values
      +control_derivatives
      +step_ts
      +get_controls(process)
    }

    class PerProcessControls {
      +eval(ts)
      +eval_derivative(ts)
      +active_step_ts
      +control_metadata
    }

    class TrainingDataStore {
      +t_meas
      +y_meas
      +meas_mask
      +y0
      +controls_store
      +get_process(process)
    }

    class PerProcessTrainingData {
      +active_t_meas
      +active_y_meas
      +meas_mask
      +y0
      +controls
    }

    class UserReactionModule {
      +__call__(t, c_species, controls_vector)
      +observe(states)
      +partition_trainable()
    }

    class ReactionOutputs {
      +reaction_terms
      +modeled_feed_rates
    }

    class LibraryRhsWrapper {
      +reaction_module
      +controls
      +species_names
      +__call__(t, y)
    }

    class TrainerFns {
      +simulate_measurement_states()
      +single_process_measurement_loss()
      +single_process_train_step()
    }

    ControlsStore --> PerProcessControls : get_controls()
    TrainingDataStore --> PerProcessTrainingData : get_process()
    PerProcessTrainingData --> PerProcessControls : holds
    LibraryRhsWrapper --> UserReactionModule : wraps
    UserReactionModule --> ReactionOutputs : returns
    LibraryRhsWrapper --> PerProcessControls : uses
    TrainerFns --> LibraryRhsWrapper : integrates
    TrainerFns --> PerProcessTrainingData : consumes
```
