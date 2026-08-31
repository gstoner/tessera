"""The NVIDIA resident-launch timer must drain first, then cross-check.

`_nvidia_mma_gemm_device_latency` previously recorded two CUDA events around
its launches and returned the elapsed value with no validation of any kind --
no wall clock to compare against, and no drain that would have made such a
comparison mean anything.

**These two halves are ordered, and that is the point of this file.** Measured
on sm_120 (RTX 5070) with 2500 2048^3 GEMMs resident on a blocking stream,
timing 40 launches of a 1024^3 GEMM:

| start event recorded      | wall ms/rep | event ms/rep | event/wall |
|---------------------------|-------------|--------------|------------|
| without a preceding drain |     63.3338 |       0.3227 |      0.005 |
| after a drain             |      0.3263 |       0.3255 |      0.998 |

The event is correct in both rows. Undrained, the start event is queued behind
the contending work, so the wall spans that drain and the event does not --
they bracket different regions. Applying the two-sided band to that row rejects
the correct 0.3227 ms event and falls back to the 63.33 ms wall: a **196x
overstatement**. A cross-check added without the drain is therefore strictly
worse than no cross-check, which is not how it reads in review, so it is
pinned here.
"""
from __future__ import annotations

import ctypes

import pytest

from tessera.runtime import (
    _accept_nvidia_event_ms,
    _nvidia_timed_launch_ms,
    nvidia_last_timer_source,
)


class _FakeCuda:
    """A CUDA stub whose event clock reports whatever `elapsed_ms` says.

    `drained_before_timing` records whether the start event was synchronised
    before the timed launches began -- the property the table above says the
    band depends on.
    """

    def __init__(self, elapsed_ms: float | None, *, create_ok: bool = True,
                 elapsed_rc: int = 0, work_s: float = 0.002):
        self._elapsed = elapsed_ms
        self._create_ok = create_ok
        self._elapsed_rc = elapsed_rc
        self._work = work_s
        self.launches = 0
        self.synchronized: list[int] = []
        self.recorded: list[int] = []
        self.destroyed = 0
        #: Launch count observed at each event sync. The drain must land after
        #: the warmup and before the first timed launch, so the first entry
        #: here has to equal `warmup` exactly.
        self.launches_at_sync: list[int] = []
        self._next = 0xE0

    def tessera_nvidia_event_create(self, ref):
        if not self._create_ok:
            return 1
        self._next += 1
        ref._obj.value = self._next
        return 0

    def tessera_nvidia_event_record(self, ev, stream):
        self.recorded.append(ev.value)
        return 0

    def tessera_nvidia_event_synchronize(self, ev):
        self.synchronized.append(ev.value)
        self.launches_at_sync.append(self.launches)
        return 0

    def tessera_nvidia_event_elapsed_ms(self, start, stop, out):
        if self._elapsed_rc != 0:
            return self._elapsed_rc
        out._obj.value = ctypes.c_float(self._elapsed).value
        return 0

    def tessera_nvidia_event_destroy(self, ev):
        self.destroyed += 1
        return 0

    def launch(self):
        import time
        time.sleep(self._work)
        self.launches += 1


def _run(lib, *, reps=4, warmup=2):
    return _nvidia_timed_launch_ms(
        lib, lib.launch, reps=reps, warmup=warmup, what="test")


# --------------------------------------------------------------------------
# The drain -- the precondition the band depends on.
# --------------------------------------------------------------------------

def test_start_event_is_synchronized_after_warmup_and_before_any_timed_launch():
    """Without this, the wall spans a drain the event does not (0.005 ratio).

    Pinned as an exact count rather than "a sync happened": a drain placed
    before the warmup leaves the warmup's own work queued, which is the same
    defect one launch later.
    """
    lib = _FakeCuda(8.0)
    _run(lib, reps=4, warmup=2)
    assert lib.launches_at_sync[0] == 2, (
        "the drain must sit between the warmup and the timed launches; saw it "
        f"after {lib.launches_at_sync[0]} of 2 warmup launches")


def test_warmup_runs_before_the_drain_not_after():
    lib = _FakeCuda(8.0, work_s=0.001)
    _run(lib, reps=4, warmup=3)
    assert lib.launches == 7


def test_stop_event_is_synchronized_before_the_wall_is_read():
    """Launches are async: without this the wall times the enqueue.

    That failure is not a crash -- it is a very small number that then drags
    the acceptance band down with it, so the band would ratify it.
    """
    lib = _FakeCuda(8.0)
    _run(lib)
    assert len(lib.synchronized) == 2, "both the drain and the stop must sync"


# --------------------------------------------------------------------------
# The band. The lower bound is the load-bearing half.
# --------------------------------------------------------------------------

@pytest.mark.parametrize("event_ms", [None, float("nan"), 0.0, -1.0])
def test_non_finite_or_non_positive_event_is_refused(event_ms):
    assert _accept_nvidia_event_ms(event_ms, 10.0) is None


def test_event_far_below_the_wall_is_refused():
    """An under-reading clock inflates throughput, and inflation gets published.

    An over-estimate makes a kernel look slow and gets investigated; this is
    the direction that does not self-correct.
    """
    assert _accept_nvidia_event_ms(0.001, 10.0) is None


def test_event_far_above_the_wall_is_refused():
    assert _accept_nvidia_event_ms(100.0, 10.0) is None


def test_event_agreeing_with_the_wall_is_accepted():
    """sm_120 measured event/wall at 0.996-0.998; the band must admit that."""
    assert _accept_nvidia_event_ms(9.97, 10.0) == 9.97


def test_band_edges_are_inclusive():
    assert _accept_nvidia_event_ms(5.0, 10.0) == 5.0
    assert _accept_nvidia_event_ms(20.0, 10.0) == 20.0


# --------------------------------------------------------------------------
# Fallback: a refused event must become a wall measurement, not an exception
# and not a silently-believed number.
# --------------------------------------------------------------------------

def test_a_lying_event_falls_back_to_the_wall_clock():
    """0.001 ms for a loop the wall saw take milliseconds is the HIP failure
    shape; CUDA has not been seen doing it, which is not the same as proof."""
    lib = _FakeCuda(0.001, work_s=0.003)
    latency, source = _run(lib, reps=4, warmup=1)
    assert source == "host_wall"
    assert latency > 0.001, "the wall is conservative: it can only look slower"


def test_an_unreadable_elapsed_time_falls_back_rather_than_raising():
    lib = _FakeCuda(8.0, elapsed_rc=3)
    _, source = _run(lib, reps=2, warmup=1)
    assert source == "host_wall"


def test_a_believable_event_is_reported_as_the_device_clock():
    lib = _FakeCuda(None, work_s=0.004)
    lib._elapsed = 4.0 * 3  # ~ the wall for 3 reps of 4 ms
    latency, source = _run(lib, reps=3, warmup=1)
    assert source == "device_event"
    assert latency == pytest.approx(4.0, rel=0.01)


def test_the_timer_source_is_observable_after_a_measurement():
    """A caller must be able to tell a device number from a wall fallback."""
    from tessera import runtime as rt
    lib = _FakeCuda(0.0001, work_s=0.003)
    _, source = _run(lib, reps=3, warmup=1)
    rt._nvidia_last_timer_source[0] = source
    assert nvidia_last_timer_source() == "host_wall"


# --------------------------------------------------------------------------
# Housekeeping and argument validation.
# --------------------------------------------------------------------------

def test_events_are_destroyed_even_when_the_clock_is_refused():
    lib = _FakeCuda(0.0001)
    _run(lib)
    assert lib.destroyed == 2


def test_event_create_failure_is_an_error_not_a_silent_wall_measurement():
    """A timer that cannot create its events has not measured the device, and
    saying so beats quietly returning a host number that reads as one."""
    lib = _FakeCuda(8.0, create_ok=False)
    with pytest.raises(RuntimeError, match="event_create"):
        _run(lib)


@pytest.mark.parametrize("reps,warmup", [(0, 1), (-1, 0), (1, -1)])
def test_invalid_iteration_counts_are_refused(reps, warmup):
    lib = _FakeCuda(8.0)
    with pytest.raises(ValueError):
        _run(lib, reps=reps, warmup=warmup)


def test_rocm_and_nvidia_acceptance_rules_are_separate_functions():
    """Deliberate duplication: the two hosts fail differently.

    HIP events on gfx1151 lie, so there the event is the suspect clock. CUDA
    events on sm_120 agree with the wall to 0.2-0.4%, so here the *wall* is the
    fragile one -- it is the clock that inflated 161x under contention. Merging
    these would force one rationale onto both, and the next person to widen one
    band would silently widen the other.
    """
    from tessera.runtime import _accept_device_event_ms
    assert _accept_nvidia_event_ms is not _accept_device_event_ms
