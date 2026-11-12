"""CPU leases and pre-import worker affinity."""

import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Barrier

import pytest
from pydantic import ValidationError

from hybrax.train import loo
from hybrax.train.cpu_affinity import CpuLeases, ordered_cpus
from hybrax.train.run_config import LooConfig


def test_topology_prefers_distinct_cores_and_packages(tmp_path):
    for cpu, package, core in [(1, 0, 0), (2, 0, 1), (3, 0, 0), (5, 1, 0), (7, 1, 0)]:
        topology = tmp_path / f"cpu{cpu}" / "topology"
        topology.mkdir(parents=True)
        (topology / "physical_package_id").write_text(str(package))
        (topology / "core_id").write_text(str(core))
    assert ordered_cpus({1, 2, 3, 5, 7}, tmp_path) == [1, 2, 5, 3, 7]
    assert ordered_cpus({3, 5}, tmp_path) == [3, 5]


def test_topology_fallback(tmp_path):
    assert ordered_cpus({9, 2, 6}, tmp_path) == [2, 6, 9]
    topology = tmp_path / "cpu2" / "topology"
    topology.mkdir(parents=True)
    (topology / "physical_package_id").write_text("unknown")
    (topology / "core_id").write_text("0")
    assert ordered_cpus({2}, tmp_path) == [2]


@pytest.mark.parametrize("value", [0, 1, -2, True, 2.5])
def test_invalid_config_cpu_count(value):
    with pytest.raises(ValidationError):
        LooConfig(cpus_per_fold=value)


def test_default_unrestricted():
    assert LooConfig().cpus_per_fold is None
    assert LooConfig(cpus_per_fold=2).cpus_per_fold == 2


def test_budget_and_platform_errors(monkeypatch, tmp_path):
    monkeypatch.setattr(os, "sched_getaffinity", lambda _: {2, 4, 6}, raising=False)
    monkeypatch.setattr(os, "sched_setaffinity", lambda *_: None, raising=False)
    with pytest.raises(ValueError, match="requires 4 CPUs.*allows 3"):
        CpuLeases(2, 2, sysfs=tmp_path)
    monkeypatch.delattr(os, "sched_setaffinity")
    with pytest.raises(ValueError, match="OS CPU affinity support"):
        CpuLeases(2, 1, sysfs=tmp_path)


def test_concurrent_leases_and_release_on_error(monkeypatch, tmp_path):
    monkeypatch.setattr(os, "sched_getaffinity", lambda _: {1, 3, 5, 7}, raising=False)
    monkeypatch.setattr(os, "sched_setaffinity", lambda *_: None, raising=False)
    leases = CpuLeases(2, 2, sysfs=tmp_path)
    barrier = Barrier(2)

    def reserve(_):
        with leases.lease() as cpus:
            barrier.wait(timeout=10)
            return cpus

    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(reserve, range(2)))
    assert set(results) == {(1, 3), (5, 7)}
    with pytest.raises(RuntimeError), leases.lease():
        raise RuntimeError("launch failed")
    with leases.lease() as first, leases.lease() as second:
        assert set(first).isdisjoint(second)
        assert set(first + second) == {1, 3, 5, 7}


@pytest.mark.skipif(
    not hasattr(os, "sched_setaffinity"), reason="requires CPU affinity support"
)
def test_worker_affinity_before_import(monkeypatch, tmp_path):
    allowed = os.sched_getaffinity(0)
    if len(allowed) < 3:
        pytest.skip("need >= 3 allowed CPUs to verify narrowing to two")
    cpus = tuple(sorted(allowed)[:2])
    output = tmp_path / "affinity.json"
    # Exercise the real worker launcher; replace training with an import probe.
    monkeypatch.setattr(
        loo,
        "_worker_cmd",
        lambda *_: [
            sys.executable,
            "-c",
            (
                "import os, json; "
                f"assert os.sched_getaffinity(0) == {set(cpus)!r}; "
                "import hybrax.train, jax; "
                "jax.devices(); "
                f"open({str(output)!r}, 'w').write(json.dumps("
                "[sorted(os.sched_getaffinity(int(t))) "
                "for t in os.listdir('/proc/self/task')]))"
            ),
        ],
    )
    before = os.sched_getaffinity(0)
    assert loo._dispatch_worker(Path(), Path(), Path(), 0, cpus) == 0
    affinities = json.loads(output.read_text())
    assert affinities and all(set(mask) == set(cpus) for mask in affinities)
    assert os.sched_getaffinity(0) == before


def test_dispatch_pool_reuses_leases_after_failure(monkeypatch, tmp_path, caplog):
    monkeypatch.setattr(os, "sched_getaffinity", lambda _: {2, 4}, raising=False)
    monkeypatch.setattr(os, "sched_setaffinity", lambda *_: None, raising=False)
    leases = CpuLeases(2, 1, sysfs=tmp_path)
    calls = []

    def dispatch(_config, _output, _artifact, idx, cpus):
        calls.append((idx, cpus))
        if idx == 0:
            raise OSError("launch failed")
        return 0

    monkeypatch.setattr(loo, "_dispatch_worker", dispatch)
    folds = [loo.Fold(i, (f"p{i}",), ("train",), f"p{i}", i) for i in range(3)]
    with (
        caplog.at_level("INFO"),
        pytest.raises(RuntimeError, match="p0.*launch failed"),
    ):
        loo._dispatch_pool(tmp_path, tmp_path, tmp_path, folds, 1, leases)
    assert calls == [(0, (2, 4)), (1, (2, 4)), (2, (2, 4))]
    assert "LOO fold 2 (p2): CPUs (2, 4)" in caplog.text
