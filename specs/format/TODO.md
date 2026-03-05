# Here is just a general TODO for the future.

* **NEXT steps:**
    * BOLUS FEEDs
        * There are multiple layers to the problem
            * bolus feed (volume change) + sampling (no volume change) at the same time
            * bolus feed (volume change) + sampling (no volume change) at different times
            * bolus feed (volume change) + sampling (volume change) at same times
            * bolus feed (volume change) + sampling (volume change) at different times
        * and then in theory everything with "full data" and afterwards with "sparse data"
    * Reimplement Splines:
        * Check if everything works as intended.
        * Currently everything is a notebook in 00_combined, but I should do the spline fitting in the separate examples
        * the function `bpbench.splines.detect_discrete_events` should be renames to `detect_discrete_volume_changes` as it clearly does not detect all discrete events (e.g., in the control `TimeSeries`)
        * the `bpbench.plot_process` function should plot the splines if they are filled in the process field. Also they should again return the figure object.
        * after splines are re-implemented, re-combine the dataset and make the integration again 
            * also here, maybe integrating them in the examples is better than in combined.
        * also: the splines have to be fitted on the pseudo-concentrations from hesselberg, so that we have continuous rates (this should be only done in the backend.)
        ```
        Hesselberg-Thomsen, V., Groves, T., McCubbin, T., Martínez-Monge, I., de Mas, I. M., & Nielsen, L. K. (2024). Ps
        eudo batch transformation: A novel method to correct for mass removal through sample withdrawal of fed-batch fermentations. bioRxiv, 2024-05.
        ```
    * Add mass balance functionality
        * currently I asked the Agent to rewrite the mass balance equation to include modeled feed rates.
        * this changed the examples/{03*, 04*}.ipynb notebooks -> check if they are correct.
        * especially 04 seems to run very slow, why? figure this out
        * NEXT: 
            1. use diffrax, not solve_ivp to solve the integration
            2. add discrete sampling events.
                * for that we need a function that calculates the delta for reactor concentrations + Volume at every event
                * the diffrax integration has to stop the deltas have to be calculated and then the diffrax integration has to start again.
    * Add new case studies -> they will bring their own challenges.
        * Martens, et al. 2025
            * Synthetic mammalian case study
            * Simulated dataset needs multiple adaptations:
                * it is too diverse - they have 3 bioreactor scales and 30 different cell lines such a diverse dataset does only make sense in an iterative approach as they do it. I have to think how I could reduce the complexity - e.g. by randomly choosing one cell line & generate 3 datasets for the 3 scales 
                * They apply measurement error only on the last data point - i may need to rewrite parts of their code. 
                * The question: should we still include this dataset? - I think yes, to make the benchmarking more diverse. It is also good to have it because it is an mammalian process - albeit simulated.
            * BUT: I should include it, because it has bolus feeds
    

* **Validation functions for `/bpbench/validate.py`**
    * ...
    
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
* ✅ new notebook per case study where all processes are loaded & saved
        * this should check if there are any problems in the code when going from 1 to several
* **Validation functions for `/bpbench/validate.py`**
    * ✅ Verify that a volume change is purely positive or purely negative: 
    * ✅ If positive, verify that all dynamic state variables (i.e., concentrations) that are part of the reactor are also defined here.
    * ✅ `TimeSeries` check if shapes are correct and if time points are ordered.
    * ✅ in the `00_combined/*` folder we could have `validate/*` subfolder where all validation tests are done for all proccesses of the dataset.
    * ✅ check if the reactor medium has a clearly defined ``biomass`` compound
* ✅ Reimplement all the tests for the package