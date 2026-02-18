# Here is just a general TODO for the future.

* **NEXT steps:**
    * new notebook per case study where all processes are loaded & saved
        * this should check if there are any problems in the code when going from 1 to several
    * Add new case studies -> they will bring their own challenges.

* **Validation functions for `/bpbench/validate.py`**
    * Verify that a volume change is purely positive or purely negative: 
    * If positive, verify that all dynamic state variables (i.e., concentrations) that are part of the reactor are also defined here.
    * `TimeSeries` check if shapes are correct and if time points are ordered.
    * in the `00_combined/*` folder we could have `validate/*` subfolder where all validation tests are done for all proccesses of the dataset.
    * check if the reactor medium has a clearly defined ``biomass`` compound

* Maybe the `ReactorMedium` could be a subclass of the `FeedMedium`.
    * I think they could both be subclasses from Medium with both then pointing to `<Reactor,Feed>MediumComponent` which have both `is_intracellular` and `is_controlled` tags and then set `ReactorMediumComponent.is_controlled=False` and `FeedMediumComponent.is_intracellular=False` to default.
* Reimplement all the tests
    
* **Possible future compatabilities:**
    * How would I implement perfusion? - No idea.

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
    ├── is_intracellular: bool # indicate if, e.g., product is part of the biomass
    └── concentration: {TimeSeries, StaticVariable} # here most concentrations are going to be time-series

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

# Done TODO Points

* find a way to indicate if a product/ byproduct is WITHIN the biomass
    * something like an extracellular tag
    * this has to work well for my mass balance equation generator
    * to be honest, maybe just pre-calculating the correct terms is the easiest
        * but this only works if we have all the same time intervals measured.
    * ✅ solved by adding a tag to the ReactorMediumComponents
* I am not sure if `RawTimeSeries` is such a nice class concept. It is currently only used in the class `TimeSeries`
    * ✅ solved by integrating the fields of RawTimeSeries in TimeSeries
* delete one outlier measurement: DoE1_R3@t=10.345
    * this has the advantage that we can directly check what happens if the number of time points are not the same over different variables.
    * ✅ deleted.
* find a way to indicate that the feed_medium is the Reactor medium in a negative volume change. 
    * if we can link the current_reactor_medium to the feed_medium of a bleed and then change e.g., the biomass value to the retained value one could even try to save perfusion processes that way.
    * ✅ actually, the medium concentration does not matter in the case of sampling because the concentration does not change. this would only be a problem if the sampling is not exactly the reactor medium --> perfusion.
* How could gas measurements work in here?
    * ✅ I think they now would fit into the ProcessVariable class nicely, either controlled (DO) or modeled (offgas).
* ✅ fix the accidentally deleted `01_kittler/data_preprocessing`
* ✅ add correct density to the `01_kittler` data