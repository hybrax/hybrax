"""Off-critical-path plot rendering in a spawned worker process.

matplotlib import + render holds the GIL; doing it inline blocks the training
step. ``BackgroundPlotter`` submits **picklable** ``(fn, *args, **kwargs)`` plot
jobs to a single ``spawn``-ed worker so training never waits on matplotlib.

Submitted callables must be importable at module level and do **pure
numpy/pandas/matplotlib work — never JAX** (``spawn`` is chosen precisely so the
worker does not inherit the parent's initialised accelerator).
"""

from __future__ import annotations

import logging
import multiprocessing as mp
from collections import deque
from concurrent.futures import Future, ProcessPoolExecutor
from typing import Any, Callable

logger = logging.getLogger(__name__)


class BackgroundPlotter:
    """Submit picklable plot jobs to a single spawned worker.

    Non-lossy: **every** submitted job is queued and rendered. Jobs run in the
    spawned worker as it frees up, so training never blocks on matplotlib; any
    backlog is drained by ``close()`` before exit.
    """

    def __init__(self) -> None:
        ctx = mp.get_context("spawn")
        self._pool: ProcessPoolExecutor | None = ProcessPoolExecutor(
            max_workers=1, mp_context=ctx
        )
        self._pending: deque[Future] = deque()

    def submit(self, fn: Callable[..., Any], *args: Any, **kwargs: Any) -> None:
        if self._pool is None:
            return
        # Reap finished jobs (surfacing any worker exception in the log); the
        # rest stay queued — nothing is ever dropped.
        still_pending: deque[Future] = deque()
        for fut in self._pending:
            if fut.done():
                exc = fut.exception()
                if exc is not None:
                    logger.warning("background plot job failed: %r", exc)
            else:
                still_pending.append(fut)
        self._pending = still_pending
        self._pending.append(self._pool.submit(fn, *args, **kwargs))

    def close(self) -> None:
        """Drain the full plot backlog and shut the worker down."""
        if self._pool is None:
            return
        for fut in self._pending:
            try:
                fut.result()
            except Exception as exc:  # noqa: BLE001 - log and continue draining
                logger.warning("background plot job failed during close: %r", exc)
        self._pending.clear()
        self._pool.shutdown(wait=True)
        self._pool = None

    def __enter__(self) -> BackgroundPlotter:
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()
