# Here is just a general TODO for the future.

* find a way to indicate if a product/ byproduct is WITHIN the biomass
    * something like an extracellular tag
    * this has to work well for my mass balance equation generator
    * **to be honest, maybe just pre-calculating the correct terms is the easiest**
        * but this only works if we have all the same time intervals measured.
* find a way to indicate that the feed_medium is the Reactor medium in a negative volume change. 
    * if we can link the current_reactor_medium to the feed_medium of a bleed and then change e.g., the biomass value to the retained value one could even try to save perfusion processes that way.

* function for `/bpbench/validate.py`
    * Verify that a volume change is purely positive or purely negative: 
    * If positive, verify that all dynamic state variables (i.e., concentrations) that are part of the reactor are also defined here.
        * [!] Here is a problem currently [!] There is no straightforward way to automatically parse which of the dynamic variables are part of the concentrations. Current tags are: 
    * `TimeSeries` check if shapes are correct and if time points are ordered.
* just a general observation: how could gas measurements work in here?

* I am not sure if `RawTimeSeries` is such a nice class concept. It is currently only used in the class `TimeSeries`
* Maybe the `ReactorMedium` could be a subclass of the `FeedMedium`.

* delete one outlier measurement: 
    ```
    if process_name == "DoE1_R3":
        approx_val = onl.iloc[np.argmin(np.abs(onl.t-10.345555555555553)),:]
    ```
    * this has the advantage that we can directly check what happens if the number of time points are not the same over different variables.


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
