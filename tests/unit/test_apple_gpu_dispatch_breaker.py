"""APPLE-DISPATCH-WEDGE-1 — a device that stops answering must stop being asked.

`commit_mpsgraph_and_wait_with_timeout` waits 30 s (60 s at one site) and, on
expiry, reports the timeout and returns. Correctly — but with no memory that it
happened, so every later dispatch paid the full timeout again. An Apple sweep
was observed stalled for **70 minutes** where the healthy run takes 4 minutes.

Nothing accumulated by design: the runtime's `g_last_gpu_error_kind` is a
thread-local *last*-error for reporting, and the wait helper clears the dispatch
telemetry **on entry**, erasing the previous timeout before the next attempt.

Host-free: these drive the real `_apple_gpu_run_checked` with the error channel
faked, so the breaker's own logic is under test rather than a reimplementation
of it. Whether the 70 minutes was one uninterruptible wait or ~140 sequential
timeouts is still unresolved (see the Apple plan) — the breaker addresses the
second and bounds the damage either way.
"""
from __future__ import annotations

import pytest

from tessera import runtime as rt

TIMEOUT = rt._APPLE_GPU_ERROR_KIND_TIMEOUT
LIMIT = rt._APPLE_GPU_DISPATCH_TIMEOUT_LIMIT


@pytest.fixture(autouse=True)
def _closed_breaker():
    rt.reset_apple_gpu_dispatch_breaker()
    yield
    rt.reset_apple_gpu_dispatch_breaker()


class _Channel:
    """Stands in for the C last-error channel, and counts real dispatches."""

    def __init__(self, kind: int, detail: str | None = "simulated"):
        self.kind, self.detail, self.dispatches = kind, detail, 0

    def install(self, monkeypatch):
        monkeypatch.setattr(rt, "_apple_gpu_arm_gpu_error", lambda: None)
        monkeypatch.setattr(rt, "_apple_gpu_peek_gpu_error_kind", lambda: self.kind)
        monkeypatch.setattr(rt, "_apple_gpu_consume_gpu_error", lambda: self.detail)
        monkeypatch.setattr(rt, "_note_dispatch_fallback",
                            lambda *a, **k: None)
        return self

    def kernel(self):
        self.dispatches += 1
        return "gpu"


def _run(channel):
    return rt._apple_gpu_run_checked("tessera.add", channel.kernel, lambda: "host")


def test_repeated_timeouts_stop_reaching_the_device(monkeypatch):
    """The whole point: the (LIMIT+1)-th call must not dispatch at all."""
    channel = _Channel(TIMEOUT).install(monkeypatch)

    for _ in range(LIMIT):
        assert _run(channel) == "host"
    assert channel.dispatches == LIMIT
    assert rt.apple_gpu_dispatch_breaker_state()["open"] is True

    # Ten more calls, zero further dispatches -- ten timeouts not paid.
    for _ in range(10):
        assert _run(channel) == "host"
    assert channel.dispatches == LIMIT, (
        "breaker open but the device was still being asked")


def test_the_breaker_needs_a_streak_not_a_single_timeout(monkeypatch):
    """One timeout can be a slow dispatch under load; three in a row is a
    device that stopped answering. Tripping on the first would turn an
    ordinary hiccup into a process-wide GPU shutdown."""
    channel = _Channel(TIMEOUT).install(monkeypatch)
    for index in range(LIMIT - 1):
        _run(channel)
        assert rt.apple_gpu_dispatch_breaker_state()["open"] is False, index


def test_a_success_closes_the_breaker_and_forgets_the_streak(monkeypatch):
    channel = _Channel(TIMEOUT).install(monkeypatch)
    for _ in range(LIMIT - 1):
        _run(channel)
    assert rt.apple_gpu_dispatch_breaker_state()["consecutive_timeouts"] == LIMIT - 1

    channel.detail = None                      # the device answers again
    assert _run(channel) == "gpu"
    assert rt.apple_gpu_dispatch_breaker_state()["consecutive_timeouts"] == 0

    channel.detail = "simulated"
    for _ in range(LIMIT - 1):
        _run(channel)
    assert rt.apple_gpu_dispatch_breaker_state()["open"] is False, (
        "the streak survived a successful dispatch")


@pytest.mark.parametrize("kind", [2, 3, 4])
def test_an_ordinary_op_failure_never_trips_the_breaker(kind, monkeypatch):
    """Kinds 2-4 are per-op failures -- a bad buffer, an unsupported shape.

    They say nothing about whether the device is answering, and counting them
    would open the breaker on a workload that merely uses an unsupported op a
    few times in a row.
    """
    channel = _Channel(kind).install(monkeypatch)
    for _ in range(LIMIT * 3):
        assert _run(channel) == "host"
    assert channel.dispatches == LIMIT * 3
    assert rt.apple_gpu_dispatch_breaker_state()["open"] is False


def test_an_interleaved_op_failure_does_not_manufacture_a_streak(monkeypatch):
    """Timeout, op-failure, timeout is not two consecutive timeouts."""
    channel = _Channel(TIMEOUT).install(monkeypatch)
    _run(channel)
    channel.kind = 2
    _run(channel)
    channel.kind = TIMEOUT
    _run(channel)
    assert rt.apple_gpu_dispatch_breaker_state()["consecutive_timeouts"] == 1
    assert rt.apple_gpu_dispatch_breaker_state()["open"] is False


def test_reset_reopens_the_device_for_a_caller_that_knows_it_recovered(monkeypatch):
    channel = _Channel(TIMEOUT).install(monkeypatch)
    for _ in range(LIMIT):
        _run(channel)
    assert rt.apple_gpu_dispatch_breaker_state()["open"] is True

    rt.reset_apple_gpu_dispatch_breaker()
    assert rt.apple_gpu_dispatch_breaker_state()["open"] is False
    channel.detail = None
    assert _run(channel) == "gpu"
    assert channel.dispatches == LIMIT + 1


def test_the_escape_hatch_restores_the_old_behaviour(monkeypatch):
    """A breaker nobody can turn off is a new way to lose a working device."""
    monkeypatch.setenv("TESSERA_APPLE_GPU_NO_DISPATCH_BREAKER", "1")
    channel = _Channel(TIMEOUT).install(monkeypatch)
    for _ in range(LIMIT * 2):
        assert _run(channel) == "host"
    assert channel.dispatches == LIMIT * 2, "the breaker fired despite the opt-out"
    assert rt.apple_gpu_dispatch_breaker_state()["disabled"] is True


def test_the_open_breaker_still_returns_the_correct_host_result(monkeypatch):
    """Cheapness is not the contract; correctness is.

    Skipping the dispatch must not skip the computation -- an open breaker
    returns the host fallback's value, which is the same answer the GPU lane
    would have produced, just slower.
    """
    import numpy as np

    channel = _Channel(TIMEOUT).install(monkeypatch)
    for _ in range(LIMIT):
        _run(channel)

    a = np.arange(6, dtype=np.float32).reshape(2, 3)
    b = np.full((2, 3), 2.0, dtype=np.float32)
    out = rt._apple_gpu_run_checked(
        "tessera.add", lambda: pytest.fail("dispatched while open"),
        lambda: a + b)
    np.testing.assert_array_equal(out, a + b)
