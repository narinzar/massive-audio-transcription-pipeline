"""Retry / exponential-backoff tests."""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.retry import compute_delays, retry_call, with_retry  # noqa: E402


class Flaky:
    """Fails the first `fail_times` calls, then returns `value`."""

    def __init__(self, fail_times, value="ok", exc=RuntimeError):
        self.fail_times = fail_times
        self.value = value
        self.exc = exc
        self.calls = 0

    def __call__(self):
        self.calls += 1
        if self.calls <= self.fail_times:
            raise self.exc(f"transient failure {self.calls}")
        return self.value


def test_compute_delays_are_geometric():
    delays = compute_delays(max_attempts=4, base_delay=0.5, factor=2.0, max_delay=100)
    # 3 waits for 4 attempts: 0.5, 1.0, 2.0
    assert delays == [0.5, 1.0, 2.0]


def test_compute_delays_clamped_by_max():
    delays = compute_delays(max_attempts=5, base_delay=1.0, factor=10.0, max_delay=5.0)
    assert delays == [1.0, 5.0, 5.0, 5.0]


def test_succeeds_after_transient_failures():
    slept = []
    flaky = Flaky(fail_times=2)
    result = retry_call(
        flaky,
        max_attempts=4,
        base_delay=0.5,
        factor=2.0,
        sleep=slept.append,
    )
    assert result == "ok"
    assert flaky.calls == 3  # 2 failures + 1 success
    # Two waits occurred, with increasing delays.
    assert slept == [0.5, 1.0]
    assert slept[1] > slept[0]


def test_raises_after_exhausting_attempts():
    slept = []
    flaky = Flaky(fail_times=99)  # always fails
    with pytest.raises(RuntimeError):
        retry_call(
            flaky,
            max_attempts=3,
            base_delay=0.1,
            factor=3.0,
            sleep=slept.append,
        )
    assert flaky.calls == 3  # exactly max_attempts tries
    assert slept == [0.1, pytest.approx(0.3)]  # 2 waits with growth


def test_delays_increase_monotonically():
    slept = []
    flaky = Flaky(fail_times=99)
    with pytest.raises(RuntimeError):
        retry_call(
            flaky,
            max_attempts=5,
            base_delay=0.5,
            factor=2.0,
            max_delay=100.0,
            sleep=slept.append,
        )
    assert slept == [0.5, 1.0, 2.0, 4.0]
    assert all(b > a for a, b in zip(slept, slept[1:]))


def test_only_configured_exceptions_are_retried():
    slept = []

    def raises_value_error():
        raise ValueError("not retried")

    with pytest.raises(ValueError):
        retry_call(
            raises_value_error,
            max_attempts=5,
            exceptions=(KeyError,),  # ValueError not covered -> immediate raise
            sleep=slept.append,
        )
    assert slept == []  # no retry happened


def test_on_retry_hook_receives_attempt_and_delay():
    events = []
    flaky = Flaky(fail_times=2)
    retry_call(
        flaky,
        max_attempts=4,
        base_delay=0.5,
        factor=2.0,
        sleep=lambda d: None,
        on_retry=lambda attempt, exc, delay: events.append((attempt, delay)),
    )
    assert events == [(0, 0.5), (1, 1.0)]


def test_decorator_form_retries():
    slept = []
    state = {"calls": 0}

    @with_retry(max_attempts=3, base_delay=0.2, factor=2.0, sleep=slept.append)
    def sometimes():
        state["calls"] += 1
        if state["calls"] < 2:
            raise RuntimeError("boom")
        return 42

    assert sometimes() == 42
    assert state["calls"] == 2
    assert slept == [0.2]


def test_max_attempts_one_means_no_retry():
    slept = []
    flaky = Flaky(fail_times=1)
    with pytest.raises(RuntimeError):
        retry_call(flaky, max_attempts=1, sleep=slept.append)
    assert flaky.calls == 1
    assert slept == []
