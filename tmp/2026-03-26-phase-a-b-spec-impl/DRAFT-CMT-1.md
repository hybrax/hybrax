Implement phase A/B artifact loading and preparation

- add bp_train package with raw collection loading and preparation pipeline
- generate padded dense control payloads and bp_train metadata in prepared.json
- support custom.py transforms and default V_sample_acc construction
- add focused tests for loading, preparation, and control ordering
