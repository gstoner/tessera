"""The ROCm resident-launch timer must reject a lying clock.

HIP events on this fleet's WSL2 / `/dev/dxg` gfx1151 host were measured
(2026-08-04) returning `hipSuccess` while `hipEventElapsedTime` wrote garbage:
0.0 ms in one harness, -1.28e8 ms in a direct probe. A timer that cannot fail
is not a measurement.

Three ROCm timing sites had three different levels of validation before this:
the C++ `benchVariant` cross-checked a wall clock inside a two-sided band, the
Python attention-backward loop checked only `sample > 0.0`, and the flash-attn
forward and backward paths checked **nothing at all** -- the backward path
discarded the return codes of both `hipEventCreate` and `hipEventElapsedTime`.
They now share one validated helper, and these tests are what make the sharing
worth anything: they exercise the rejection, not the happy path.
"""
from __future__ import annotations

import ctypes
import time

import pytest

from tessera.runtime import _hip_resident_launch_latency


class _FakeHip:
    """A HIP stub whose event clock reports whatever `elapsed_ms` says."""

    def __init__(self, elapsed_ms: float | None, *, create_ok: bool = True,
                 elapsed_rc: int = 0, work_s: float = 0.002):
        self._elapsed = elapsed_ms
        self._create_ok = create_ok
        self._elapsed_rc = elapsed_rc
        self._work = work_s
        self.launches = 0

    def hipDeviceSynchronize(self):
        return 0

    def hipEventCreate(self, ref):
        if not self._create_ok:
            return 1
        ref._obj.value = 0xBEEF
        return 0

    def hipEventRecord(self, ev, stream):
        return 0

    def hipEventSynchronize(self, ev):
        return 0

    def hipEventDestroy(self, ev):
        return 0

    def hipEventElapsedTime(self, out, start, stop):
        if self._elapsed_rc != 0 or self._elapsed is None:
            return self._elapsed_rc or 1
        out._obj.value = ctypes.c_float(self._elapsed).value
        return 0

    def launch(self) -> int:
        self.launches += 1
        time.sleep(self._work)
        return 0


def _measure(hip, iters=5):
    return _hip_resident_launch_latency(
        hip, hip.launch, iters=iters, warmup=1, what="test")


def test_a_zero_event_value_is_rejected():
    """The exact garbage this host produced: 0.0 ms for real work."""
    hip = _FakeHip(0.0)
    latency, source = _measure(hip)
    assert source == "host_wall"
    assert latency > 0.0


def test_a_negative_event_value_is_rejected():
    """The other observed garbage: -1.28e8 ms."""
    _, source = _measure(_FakeHip(-1.28e8))
    assert source == "host_wall"


def test_an_absurdly_small_event_value_is_rejected():
    """The dangerous direction, and the one a one-sided bound misses.

    0.001 ms for a ~10 ms loop is finite, positive, and under any upper bound
    -- so it passes every check except the lower one, and yields a wildly
    INFLATED throughput. An over-estimate makes a kernel look slow and gets
    investigated; an under-estimate makes it look fast and gets published.
    """
    hip = _FakeHip(0.001)
    latency, source = _measure(hip)
    assert source == "host_wall", "a lower bound is what catches an inflated result"
    assert latency > 0.001


def test_an_absurdly_large_event_value_is_rejected():
    _, source = _measure(_FakeHip(1.0e6))
    assert source == "host_wall"


def test_a_plausible_event_value_is_accepted():
    """The guard must not reject a working clock, or it would silently demote
    every host to wall timing and quietly reintroduce launch overhead."""
    hip = _FakeHip(None, work_s=0.002)
    # ~5 iterations x 2 ms = ~10 ms of wall; report a value inside the band.
    hip._elapsed = 9.0
    latency, source = _measure(hip)
    assert source == "device_event"
    assert latency == pytest.approx(9.0 / 5, rel=1e-6)


def test_unusable_events_fall_back_instead_of_failing():
    """A host that honestly reports unsupported events must still measure.

    The wall path needs nothing from the event API, so a failed create used to
    be an unnecessary hard failure.
    """
    _, source = _measure(_FakeHip(5.0, create_ok=False))
    assert source == "host_wall"

    _, source = _measure(_FakeHip(5.0, elapsed_rc=1))
    assert source == "host_wall"


def test_the_fallback_can_only_make_a_kernel_look_slower():
    """Wall time includes launch overhead, so it never flatters the kernel."""
    hip = _FakeHip(0.0, work_s=0.003)
    latency, source = _measure(hip, iters=4)
    assert source == "host_wall"
    assert latency >= 0.003 * 1e3 * 0.5


def test_a_failed_launch_is_an_error_not_a_measurement():
    class _Failing(_FakeHip):
        def launch(self) -> int:
            return 1

    with pytest.raises(RuntimeError, match="launch failed"):
        _measure(_Failing(5.0))


def test_iteration_counts_are_validated():
    with pytest.raises(ValueError):
        _hip_resident_launch_latency(_FakeHip(5.0), lambda: 0, iters=0,
                                     warmup=0, what="test")
