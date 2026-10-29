"""Lease disjoint CPU sets for fold subprocesses."""

import os
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from queue import Queue


def ordered_cpus(allowed: set[int], sysfs: Path) -> list[int]:
    """Prefer one logical CPU per physical core before using SMT siblings.

    Package IDs distinguish equal core IDs on different sockets. If topology
    is unavailable, use sorted allowed CPU IDs instead.
    """
    cores: dict[tuple[int, int], list[int]] = {}
    try:
        for cpu in sorted(allowed):
            topology = sysfs / f"cpu{cpu}" / "topology"
            key = (
                int((topology / "physical_package_id").read_text()),
                int((topology / "core_id").read_text()),
            )
            cores.setdefault(key, []).append(cpu)
    except (OSError, ValueError):
        return sorted(allowed)
    return [
        siblings[layer]
        for layer in range(max(map(len, cores.values()), default=0))
        for siblings in cores.values()
        if layer < len(siblings)
    ]


def check_cpu_budget(cpus_per_fold: int, parallel: int) -> set[int]:
    """Return the allowed CPUs, or raise if `parallel` leases cannot fit them.

    Cheap enough to call before the LOO producer, so a budget that cannot fit
    fails before any expensive setup.
    """
    if not hasattr(os, "sched_getaffinity") or not hasattr(os, "sched_setaffinity"):
        raise ValueError("loo.cpus_per_fold requires OS CPU affinity support")
    allowed = os.sched_getaffinity(0)
    required = cpus_per_fold * parallel
    if required > len(allowed):
        raise ValueError(
            f"loo.cpus_per_fold={cpus_per_fold} x {parallel} parallel folds "
            f"requires {required} CPUs, but parent affinity allows {len(allowed)}"
        )
    return allowed


class CpuLeases:
    """Own fixed CPU sets; return a worker's set even when launch fails."""

    def __init__(
        self,
        cpus_per_fold: int,
        parallel: int,
        *,
        sysfs: Path = Path("/sys/devices/system/cpu"),
    ) -> None:
        allowed = check_cpu_budget(cpus_per_fold, parallel)
        required = cpus_per_fold * parallel
        cpus = ordered_cpus(allowed, sysfs)
        self.available: Queue[tuple[int, ...]] = Queue()
        for start in range(0, required, cpus_per_fold):
            self.available.put(tuple(cpus[start : start + cpus_per_fold]))

    @contextmanager
    def lease(self) -> Iterator[tuple[int, ...]]:
        """Reserve a CPU set until the fold subprocess exits."""
        cpus = self.available.get()
        try:
            yield cpus
        finally:
            self.available.put(cpus)
