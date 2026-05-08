# Here is just a general TODO for the future.

* **NEXT steps:**
    * Spline fits are clearly broken right now. I have to think how to calculate the pseudo-batch transformation.
        * first I should make a continuos volume spline, then add the discrete jumps that I get my continuous volume spline.
        * from there I then should caclculate the pseudobatch transformation with t+- delta 
        * then calculate the splines from the other states.
    * Reimplement Splines:
        * also: the splines have to be fitted on the pseudo-concentrations from hesselberg, so that we have continuous rates (this should be only done in the backend.)
        ```
        Hesselberg-Thomsen, V., Groves, T., McCubbin, T., Martínez-Monge, I., de Mas, I. M., & Nielsen, L. K. (2024). Ps
        eudo batch transformation: A novel method to correct for mass removal through sample withdrawal of fed-batch fermentations. bioRxiv, 2024-05.
        ```
        * extend the `pseudo_batch_transform_timeseries` to mixed continuous + discrete volume streams.
    * Add mass balance functionality
        * especially 04 seems to run very slow, why? figure this out
        * NEXT: 
            1. use diffrax, not solve_ivp to solve the integration
            2. add discrete sampling events.
                * for that we need a function that calculates the delta for reactor concentrations + Volume at every event
                * the diffrax integration has to stop the deltas have to be calculated and then the diffrax integration has to start again.
    * Support time-varying feed concentrations in the mechanistic code
        * the schema already allows `FeedMediumComponent.concentration` to be either `TimeSeries` or `StaticVariable`
        * however, `get_rhs_ode()` currently raises when a feed concentration is a `TimeSeries`
        * this should be supported consistently for:
            * continuous feed terms in the RHS
            * discrete bolus-feed event composition (`Cin`)
            * any helper/spline/inversion code paths that assume static feed composition
    

* **Possible future compatabilities:**
    * How would I implement perfusion? - No idea.
    * How do we deal with initial concentrations and how do we indicate if they are controlled?

# What do acutally test in the benchmarking
    * Do we acutally need to predict base feed rates, or is it good enough to set them to 0 or a constant?
    * Different scaling methods
    * Different Augmentation methods
    * Different ML methods

# Modeling Choices

1. Volume is encouraged to be modeled indirectly with kg (there is an additional density tag one can use)
    * Most simulations assume a density of 1 kg/L within the bioreactor, so it does not matter there
    * feeds can have significantly differt densities, but they are usually tracked with mass flows, not volume.
    * densities are not constant in the bioreactor
    

# Final Structure

```
BioProcess
├── TimeAxis
├── ProcessMetadata
├── ReactorMedium
├── Volume
├── ProcessVariables[Dict] # here goes anything that is not a concentration and not a feed (e.g. pH, off-gas)
│   ├── name: str
│   ├── unit: str
│   ├── is_controlled: bool
│   └── values {TimeSeries, StaticVariable}
└── EventTimes


ReactorMedium(Medium): # here goes the classic biomass, product, substrate trio, etc.
├── name
├── density
├── density_unit
└── MediumComponents[Dict]
    ├── name: str
    ├── unit: str
    └── concentration: {TimeSeries, StaticVariable} # here most concentrations are going to be time-series
    # Intracellular accumulation (e.g. plasmid DNA, inclusion bodies) is no
    # longer expressed via a flag here; declare it as a BiologicalOde block
    # on the BioProcess: algebraic={"X_active": "biomass - product"}, etc.

Volume: # here go all the feed and sampling operations
├── initial_value: float
├── unit: str
└── VolumeChanges[Dict]
    ├── name: str
    ├── unit: str
    ├── is_controlled: bool
    ├── is_continuous: bool # False if discrete events (e.g, bolus, sampling)
    ├── values: TimeSeries # in L or kg, no rate (because rates are usually derived, i.e. out of scope)
    └── FeedMedium(Medium): # if the VolumeChange is due to sampling, I want to link the ReactorMedium here.
        ├── name
        ├── density: float
        ├── density_unit: str
        └── FeedComponents[Dict] # here we can check if all medium components in the reactor are also defined here, otherwise write out warning.
            ├── name: str
            ├── unit: str
            ├── is_controlled: bool
            └── concentration: {TimeSeries, StaticVariable} # here most concentrations are going to be static

StaticVariable:
└── value: float

TimeSeries:
├── values: jnp.Array
└── times: jnp.Array
```
