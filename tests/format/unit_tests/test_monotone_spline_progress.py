"""Progress notices for long-running individual spline fits."""

import logging
from threading import Event
from threading import Timer as ThreadTimer

import numpy as np
import pytest

from hybrax.format import TimeSeries, monotone_spline


@pytest.mark.parametrize("fail", [False, True])
def test_progress_timer_reports_once_and_is_cancelled(monkeypatch, caplog, fail):
    timers = []

    class ControlledTimer:
        def __init__(self, interval, function, args):
            self.interval = interval
            self.function = function
            self.args = args
            self.daemon = False
            self.started = False
            self.cancelled = False
            timers.append(self)

        def start(self):
            self.started = True

        def cancel(self):
            self.cancelled = True

    def fit(*args):
        timer = timers[0]
        assert timer.interval == 20.0
        assert timer.daemon and timer.started
        timer.function(*timer.args)
        if fail:
            raise RuntimeError("deliberate fitting failure")
        return np.array([0.0, 1.0]), np.array([[0.0, 1.0, 0.0, 0.0]]), {}

    monkeypatch.setattr(monotone_spline, "Timer", ControlledTimer)
    monkeypatch.setattr(monotone_spline, "_fit_nonconstant", fit)
    ts = TimeSeries(times=np.array([0.0, 1.0]), values=np.array([0.0, 1.0]))
    with caplog.at_level(logging.INFO, logger="hybrax.format"):
        if fail:
            with pytest.raises(RuntimeError, match="deliberate fitting failure"):
                monotone_spline.fit_monotone_timeseries_spline(
                    ts, rms_tol=0.01, increasing=True, max_knots=12
                )
        else:
            monotone_spline.fit_monotone_timeseries_spline(
                ts, rms_tol=0.01, increasing=True, max_knots=12
            )
    assert len(timers) == 1 and timers[0].cancelled
    records = [r for r in caplog.records if r.name == monotone_spline.__name__]
    assert len(records) == 1
    assert records[0].levelno == logging.INFO
    message = records[0].getMessage()
    for text in (
        "2 samples",
        "20 seconds",
        "rms_tol=0.01",
        "max_knots=12",
        "increase rms_tol",
        "physically acceptable",
        "reduce max_knots",
        "may then fail",
        "preserving short events",
        "retained samples",
        "Tightening rms_tol",
        "fit continues",
    ):
        assert text in message


def test_real_progress_timer_fires_while_a_fit_is_still_running(monkeypatch, caplog):
    logged = Event()
    timers = []
    log = monotone_spline._log_slow_fit

    def notify(*args):
        log(*args)
        logged.set()

    def timer(interval, function, args):
        instance = ThreadTimer(interval, function, args=args)
        timers.append(instance)
        return instance

    def fit(*args):
        assert logged.wait(timeout=5), "Progress timer did not fire during fitting"
        return np.array([0.0, 1.0]), np.array([[0.0, 1.0, 0.0, 0.0]]), {}

    monkeypatch.setattr(monotone_spline, "_SLOW_FIT_SECONDS", 0.01)
    monkeypatch.setattr(monotone_spline, "_log_slow_fit", notify)
    monkeypatch.setattr(monotone_spline, "Timer", timer)
    monkeypatch.setattr(monotone_spline, "_fit_nonconstant", fit)
    ts = TimeSeries(times=np.array([0.0, 1.0]), values=np.array([0.0, 1.0]))
    with caplog.at_level(logging.INFO, logger="hybrax.format"):
        monotone_spline.fit_monotone_timeseries_spline(
            ts, rms_tol=0.01, increasing=True
        )
    timers[0].join(timeout=5)
    assert not timers[0].is_alive()
    assert sum("fit continues" in r.getMessage() for r in caplog.records) == 1
