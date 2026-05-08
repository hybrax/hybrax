"""
Utility functions for bioprocess benchmarking
"""

from typing import Generator, List, Tuple

from .dataclasses import BenchmarkDataset, CaseStudy


def leave_one_process_out(
    case_study: CaseStudy,
) -> Generator[Tuple[List[str], str], None, None]:
    """Generator for leave-one-process-out cross-validation.

    Yields ``(train_process_ids, test_process_id)`` tuples.
    """
    process_ids = list(case_study.processes.keys())
    for i, test_id in enumerate(process_ids):
        train_ids = [pid for j, pid in enumerate(process_ids) if j != i]
        yield train_ids, test_id


def iter_loocv(
    dataset: BenchmarkDataset,
) -> Generator[Tuple[str, List[str], str], None, None]:
    """Iterator for leave-one-process-out cross-validation across all case studies.

    Yields ``(case_id, train_process_ids, test_process_id)`` tuples.
    """
    for case_id, case_study in dataset.case_studies.items():
        for train_ids, test_id in leave_one_process_out(case_study):
            yield case_id, train_ids, test_id
