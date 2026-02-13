"""
Utility functions for bioprocess benchmarking
"""

import jax.numpy as jnp
from typing import Tuple, List, Generator
from .dataclasses import Process, CaseStudy, BenchmarkDataset, TimeSeries, VolumeChange

# Import spline functions to re-export them
from .splines import fit_cubic_spline, compute_rate_from_cumulative



def get_event_times(process: Process) -> jnp.ndarray:
    """
    Extract event times for diffrax solver
    
    Args:
        process: Process object containing event times
        
    Returns:
        Array of event times, or empty array if None
    """
    return process.event_times if process.event_times is not None else jnp.array([])


def leave_one_process_out(case_study: CaseStudy) -> Generator[Tuple[List[str], str], None, None]:
    """
    Generator for leave-one-process-out cross-validation.
    
    Args:
        case_study: CaseStudy object containing multiple processes
        
    Yields:
        (train_process_ids, test_process_id) tuples
    """
    process_ids = list(case_study.processes.keys())
    for i, test_id in enumerate(process_ids):
        train_ids = [pid for j, pid in enumerate(process_ids) if j != i]
        yield train_ids, test_id


def iter_loocv(dataset: BenchmarkDataset) -> Generator[Tuple[str, List[str], str], None, None]:
    """
    Iterator for leave-one-process-out cross-validation across all case studies.
    
    Args:
        dataset: BenchmarkDataset containing multiple case studies
        
    Yields:
        (case_id, train_process_ids, test_process_id) tuples
    """
    for case_id, case_study in dataset.case_studies.items():
        for train_ids, test_id in leave_one_process_out(case_study):
            yield case_id, train_ids, test_id




