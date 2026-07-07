"""Exponential-backoff retry helper.

`retry_call` runs a callable and retries on exception with delays that grow
geometrically (base * factor**attempt), capped at `max_delay` and optionally
jittered. `with_retry` is the decorator form. Both stop after `max_attempts`
total tries and re-raise the last exception.

The sleep function is injected so tests can substitute a recorder instead of
`time.sleep`, keeping tests fast and deterministic.
"""

from __future__ import annotations

import functools
import random
import time
from typing import Callable, Iterable, Optional, Tuple, Type, TypeVar

T = TypeVar("T")

SleepFn = Callable[[float], None]


def compute_delays(
    max_attempts: int,
    base_delay: float,
    factor: float,
    max_delay: float,
) -> list[float]:
    """Return the sequence of delays used between attempts.

    For `max_attempts` tries there are `max_attempts - 1` waits. The i-th wait
    (0-indexed) is base_delay * factor**i, clamped to max_delay.
    """
    delays = []
    for i in range(max(0, max_attempts - 1)):
        delays.append(min(base_delay * (factor**i), max_delay))
    return delays


def retry_call(
    fn: Callable[..., T],
    *args,
    max_attempts: int = 4,
    base_delay: float = 0.5,
    factor: float = 2.0,
    max_delay: float = 30.0,
    jitter: float = 0.0,
    exceptions: Tuple[Type[BaseException], ...] = (Exception,),
    sleep: SleepFn = time.sleep,
    on_retry: Optional[Callable[[int, BaseException, float], None]] = None,
    **kwargs,
) -> T:
    """Call fn(*args, **kwargs), retrying on `exceptions` with backoff.

    max_attempts: total number of tries (>= 1).
    base_delay:   first backoff delay in seconds.
    factor:       geometric growth factor between delays.
    max_delay:    clamp for any single delay.
    jitter:       max random seconds added to each delay (0 disables).
    exceptions:   exception types that trigger a retry; others propagate.
    sleep:        injected sleep function (default time.sleep).
    on_retry:     optional hook(attempt_index, exception, delay) before sleeping.

    Raises the last caught exception if all attempts fail.
    """
    if max_attempts < 1:
        raise ValueError("max_attempts must be >= 1")

    last_exc: Optional[BaseException] = None
    for attempt in range(max_attempts):
        try:
            return fn(*args, **kwargs)
        except exceptions as exc:  # noqa: B902 - deliberately broad, configurable
            last_exc = exc
            is_last = attempt == max_attempts - 1
            if is_last:
                break
            delay = min(base_delay * (factor**attempt), max_delay)
            if jitter:
                delay += random.uniform(0.0, jitter)
            if on_retry is not None:
                on_retry(attempt, exc, delay)
            sleep(delay)
    assert last_exc is not None  # loop ran at least once
    raise last_exc


def with_retry(
    max_attempts: int = 4,
    base_delay: float = 0.5,
    factor: float = 2.0,
    max_delay: float = 30.0,
    jitter: float = 0.0,
    exceptions: Tuple[Type[BaseException], ...] = (Exception,),
    sleep: SleepFn = time.sleep,
    on_retry: Optional[Callable[[int, BaseException, float], None]] = None,
) -> Callable[[Callable[..., T]], Callable[..., T]]:
    """Decorator form of `retry_call`."""

    def decorator(fn: Callable[..., T]) -> Callable[..., T]:
        @functools.wraps(fn)
        def wrapper(*args, **kwargs) -> T:
            return retry_call(
                fn,
                *args,
                max_attempts=max_attempts,
                base_delay=base_delay,
                factor=factor,
                max_delay=max_delay,
                jitter=jitter,
                exceptions=exceptions,
                sleep=sleep,
                on_retry=on_retry,
                **kwargs,
            )

        return wrapper

    return decorator
