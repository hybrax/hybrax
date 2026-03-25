1. yes and please use `Interpolator` (this is new in `bpbench`)
2. yes, no pseudo-batch for now; we'll only implement this if we really need it
3. not sure I understood this question... please phrase differently if required (i.e. if the following answer is not sufficient). Yes, the controls should not be tied to [D, Cf_norm, T] but instead be generic. `bpbench` treats volume and feeds in a special way anyway, so we can use some of this, but generally you're right in that extra code might be needed to properly set up the controls.
4. One global padded shape
5. exact elevation
6. per-trace breakpoints have the disadvantage that we have to call `searchsorted` once per trace / control (which is bad for performance). I suggest to do the following for now: for each segment evaluate the spline on a dense grid (and its first derivative) and use linear interpolation (either `interpax` or `diffrax` lin. interp. implementation) on the evaluated values. This way we only use `searchsorted` once per experiment (or rather segment) and get all controls values. We might change this later, but I want to get a POC off the ground fast.
7. Our pre-processing tool focuses on visual pre-processing of the data (outlier removal, smoothing, jump detection, etc.) and has minimal options (so far) to give extra metadata. Therefore, we have to rely on configs for now (but we should check the flag if there's nothing in the config).
8. Your suggestions sound good. Mid-term we'll need some data augmentation params as well, but for now let's train simply with the raw data (or rather get a train step working with raw data).
9. I don't have a strong opinion on this, but I guess we can use a `partition` function like in hybrax (we can use an abstractmethod to force the user, i.e. the researcher setting up their hybrid model, to implement this method; or per default we can just take the neural network params if not implemented)
10. bpbench owns deserialization and bp-train preps the data for training
11. yeah let's keep it simple and stick to deterministic spline sampling first
12. temporary adapters as suggested