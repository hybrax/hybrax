# Utilities

Source: `bp_format/utils.py`

## Purpose

Cross-validation helpers for benchmarking workflows. Provides generators for leave-one-process-out (LOOCV) iteration, the standard evaluation protocol for bioprocess model benchmarking.

## Public API

### `leave_one_process_out(case_study) -> Generator`

Generator for leave-one-process-out cross-validation within a single case study.

**Yields:** `(train_process_ids: List[str], test_process_id: str)` tuples.

Each iteration holds out one process for testing and returns the remaining process IDs for training.

### `iter_loocv(dataset) -> Generator`

Generator for leave-one-process-out cross-validation across all case studies in a dataset.

**Yields:** `(case_id: str, train_process_ids: List[str], test_process_id: str)` tuples.

Iterates through every case study and every process within each case study.

## Examples

### LOO-CV Within a Single Case Study

```python
import bp_format as bp

dataset = bp.serialization.load_dataset("data.json")
case_study = dataset.case_studies["kittler_2022"]

for train_ids, test_id in bp.utils.leave_one_process_out(case_study):
    train_processes = {pid: case_study.processes[pid] for pid in train_ids}
    test_process = case_study.processes[test_id]

    print(f"Test: {test_id}, Train: {train_ids}")
    # ... train model on train_processes, evaluate on test_process
```

### Full Dataset LOO-CV

```python
import bp_format as bp

dataset = bp.serialization.load_dataset("data.json")

results = {}
for case_id, train_ids, test_id in bp.utils.iter_loocv(dataset):
    case_study = dataset.case_studies[case_id]
    train_processes = {pid: case_study.processes[pid] for pid in train_ids}
    test_process = case_study.processes[test_id]

    print(f"[{case_id}] Test: {test_id}, Train: {train_ids}")
    # ... train and evaluate
    # results[(case_id, test_id)] = metrics
```

## See Also

- [Data Model](02_data_model.md) -- `CaseStudy` and `BenchmarkDataset` structures
- [Mechanistic](08_mechanistic.md) -- building models to train/evaluate in the CV loop
