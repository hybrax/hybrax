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

    Never blocks training: if the worker is backed up beyond ``max_pending``,
    the job is dropped — the next checkpoint re-renders the cumulative curve
    anyway. ``close()`` drains in-flight jobs before exit.
    """

    def __init__(self, max_pending: int = 2) -> None:
        ctx = mp.get_context("spawn")
        self._pool: ProcessPoolExecutor | None = ProcessPoolExecutor(
            max_workers=1, mp_context=ctx
        )
        self._pending: deque[Future] = deque()
        self._max_pending = int(max_pending)
        self._dropped = 0

    def submit(self, fn: Callable[..., Any], *args: Any, **kwargs: Any) -> None:
        if self._pool is None:
            return
        # Reap finished jobs (surfacing any worker exception in the log).
        still_pending: deque[Future] = deque()
        for fut in self._pending:
            if fut.done():
                exc = fut.exception()
                if exc is not None:
                    logger.warning("background plot job failed: %r", exc)
            else:
                still_pending.append(fut)
        self._pending = still_pending
        if len(self._pending) >= self._max_pending:
            self._dropped += 1
            return
        self._pending.append(self._pool.submit(fn, *args, **kwargs))

    def close(self) -> None:
        """Drain in-flight jobs and shut the worker down."""
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
        if self._dropped:
            logger.info(
                "background plotter dropped %d plot job(s) to keep training "
                "non-blocking",
                self._dropped,
            )

    def __enter__(self) -> BackgroundPlotter:
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()
