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


# ── the device clock needs a lower bound too (PR #656 review) ────────────────
#
# The in-kernel stamps were taken without a block barrier. A 64-thread block on
# wave32 hardware spans two wavefronts, so the first could record the end time
# while the second was still computing -- an under-reading, which the original
# upper-bound-only check accepted. That would have made the lane look faster
# than it is and persisted the wrong autotune winner: the exact failure the
# event-clock band exists to prevent, reintroduced through a different door.


def test_a_device_clock_that_under_reads_is_caught_by_the_event_clock():
    """Wall cannot be the lower reference -- launch overhead makes a small
    kernel's device time legitimately far below it. The event clock brackets
    the same span on the same stream, so a large gap means one is lying."""
    from tessera.runtime import _select_rocm_latency_ms

    # A barrier bug: the device clock reports a fraction of the real span.
    chosen = _select_rocm_latency_ms(wall_ms=10.0, event_ms=9.8, device_ms=1.0)
    assert chosen == 9.8, (
        "an under-reading device clock must lose to the validated event clock")


def test_a_device_clock_agreeing_with_the_event_clock_is_preferred():
    """The healthy case measured on gfx1151: the three clocks agree to four
    significant figures and the kernel-only reading wins."""
    from tessera.runtime import _select_rocm_latency_ms

    assert _select_rocm_latency_ms(
        wall_ms=82.6946, event_ms=82.5909, device_ms=82.5600) == 82.5600


def test_the_device_clock_still_wins_when_no_event_clock_is_available():
    """With events unusable there is nothing to cross-check against, so the
    kernel-only reading is still the best of what is left -- it is bounded from
    above by the wall clock, which is all that can be checked."""
    from tessera.runtime import _select_rocm_latency_ms

    assert _select_rocm_latency_ms(
        wall_ms=10.0, event_ms=None, device_ms=4.0) == 4.0


def test_an_over_reading_device_clock_falls_back():
    from tessera.runtime import _select_rocm_latency_ms

    assert _select_rocm_latency_ms(
        wall_ms=10.0, event_ms=9.8, device_ms=999.0) == 9.8
    assert _select_rocm_latency_ms(
        wall_ms=10.0, event_ms=None, device_ms=999.0) == 10.0


def test_the_generated_kernel_barriers_bracket_the_whole_block():
    """The fix for the under-reading, asserted where it lives.

    `span` is block-uniform so both barriers are reached by every thread; a
    barrier inside a thread-0-only branch would hang the block.
    """
    from tessera.compiler import fusion_core as F
    from tessera.compiler.emit.rocm_hip import _synthesize_fused_hip

    kernel = _synthesize_fused_hip(
        F.FusedRegion(epilogue=("relu",))).split("__global__")[1].split(
            'extern "C"')[0]
    assert kernel.count("__syncthreads()") == 2, (
        "both the start and end stamps need a block barrier")
    start, end = kernel.index("atomicMin"), kernel.index("atomicMax")
    barriers = [i for i in range(len(kernel))
                if kernel.startswith("__syncthreads()", i)]
    assert any(start < b < end for b in barriers), (
        "a barrier must separate the start stamp from the end stamp, or one "
        "wavefront can stamp the end while another is still computing")
    assert any(b > start for b in barriers) and end > max(
        b for b in barriers if b < end), (
        "the end stamp must follow the barrier, not precede it")


def test_the_span_allocation_is_freed_even_when_its_init_fails():
    """Disabling the instrumentation must not discard the only pointer to the
    allocation -- cleanup's `if (dSpan) hipFree(dSpan)` would then never fire,
    leaking once per autotune call under a repeated copy failure."""
    from tessera.compiler import fusion_core as F
    from tessera.compiler.emit.rocm_hip import _synthesize_fused_hip

    bench = _synthesize_fused_hip(
        F.FusedRegion(epilogue=("relu",))).split("_bench(")[1]
    assert "useSpan=0" in bench, "the instrumentation is disabled by a flag"
    assert "hipFree(dSpan)" in bench
    # The only assignment of 0 to the pointer is its declaration initialiser,
    # which cleanup relies on.
    assigns = [ln.strip() for ln in bench.splitlines()
               if "dSpan=0" in ln.replace(" ", "")]
    assert assigns == ["unsigned long long *dSpan=0, hSpan[2];"], assigns
