"""Apple's device timestamps must be checked against a host witness.

`ts_record_tile_gpu_elapsed` and `ts_record_dispatch_gpu_elapsed` took Metal's
self-reported interval and validated only `end >= start && end > 0.0` -- a
sanity check, which rejects an obviously broken clock and accepts any value
that is merely wrong. Every other backend cross-checks its device clock against
a host one; Apple had nothing to compare against, because the runtime captured
no wall time at all.

**The witness has to bracket the same region the device clock does, and getting
that wrong produces a wrong rule rather than an obvious failure.** Measured on
an M1 Max:

    against a Python-level wall (numpy marshalling included)   0.35 - 0.60
    against the runtime witness (starts at GPU submission)     0.568 - 0.937

The first argued for a one-sided band -- 0.35 fails a 0.5x floor -- and that
band was written before the second measurement existed. It was defending
against an artifact of its own denominator.

Host-free: the rule is pure arithmetic and is tested directly. The live M1 Max
numbers behind its constants are recorded in `accept_apple_device_ns`.
"""
from __future__ import annotations

import pytest

from tessera._apple_gpu_dispatch import (
    _SENTINEL_SYMBOLS,
    accept_apple_device_ns,
)

_WALL = 1_000_000  # 1 ms in ns


# --------------------------------------------------------------------------
# The measured envelope must be admitted end to end.
# --------------------------------------------------------------------------

@pytest.mark.parametrize("ratio", [0.568, 0.60, 0.714, 0.852, 0.937])
def test_every_measured_ratio_on_an_m1_max_is_accepted(ratio):
    """The observed range over 100 dispatches, 8^2 through 2048^2.

    A band that rejects real readings silently substitutes the wall clock,
    which carries the submission overhead the device measurement exists to
    exclude -- so a too-tight band degrades the number rather than failing.
    """
    assert accept_apple_device_ns(int(ratio * _WALL), _WALL) is not None


def test_the_python_level_ratio_that_produced_the_wrong_rule_is_below_the_floor():
    """0.35 is what a Python-level wall reported, and it fails the floor.

    Kept as a test because it is the evidence that the denominator mattered:
    had the witness been scoped that way, this band would reject roughly every
    dispatch and every device number would quietly become a wall number.
    """
    assert accept_apple_device_ns(int(0.35 * _WALL), _WALL) is None


# --------------------------------------------------------------------------
# Upper bound -- physical containment.
# --------------------------------------------------------------------------

def test_a_device_interval_longer_than_the_wall_is_refused():
    """GPU work is a strict subset of commit-and-wait, so this cannot happen;
    when it does, the timestamp is misattributed to the wrong region."""
    assert accept_apple_device_ns(2 * _WALL, _WALL) is None


def test_the_upper_bound_leaves_room_for_clock_skew():
    """Two independent clocks over nested regions measured up to 0.937, so a
    strict `device <= wall` has 6.3% of margin -- too little to hold."""
    assert accept_apple_device_ns(int(1.10 * _WALL), _WALL) is not None
    assert accept_apple_device_ns(int(1.40 * _WALL), _WALL) is None


# --------------------------------------------------------------------------
# Lower bound -- the direction that gets published.
# --------------------------------------------------------------------------

def test_an_under_reading_clock_is_refused():
    """An under-estimate inflates throughput and gets published; an
    over-estimate makes a kernel look slow and gets investigated."""
    assert accept_apple_device_ns(_WALL // 100, _WALL) is None


def test_zero_and_negative_device_intervals_are_refused():
    assert accept_apple_device_ns(0, _WALL) is None
    assert accept_apple_device_ns(-1, _WALL) is None
    assert accept_apple_device_ns(None, _WALL) is None


# --------------------------------------------------------------------------
# Absent witness -- the failure mode the sentinel exists to prevent.
# --------------------------------------------------------------------------

def test_a_missing_wall_passes_the_device_value_through():
    """Deliberate, and deliberately dangerous: with no witness there is
    nothing to check against, and refusing every dispatch on a runtime that
    predates the export would be worse than the status quo ante."""
    assert accept_apple_device_ns(500_000, None) == 500_000
    assert accept_apple_device_ns(500_000, 0) == 500_000


def test_the_wall_export_is_a_freshness_sentinel():
    """Because the line above means a stale dylib silently restores the
    unwitnessed behaviour -- with every test still passing. The staleness
    check is what turns that into a skip instead of a false green.
    """
    assert "tessera_apple_gpu_last_dispatch_wall_time_ns" in _SENTINEL_SYMBOLS


# --------------------------------------------------------------------------
# The rule is Apple's own, not a shared one.
# --------------------------------------------------------------------------

def test_apple_rocm_and_nvidia_keep_separate_acceptance_rules():
    """Three hosts, three failure modes: HIP events on gfx1151 lie outright,
    CUDA events on sm_120 track the wall to 0.2-0.4%, and Metal's interval
    legitimately sits at 0.57-0.94 of a witness that includes submission.
    One function would force a single rationale onto all three, and widening
    a band for one host would silently widen it for the others.
    """
    from tessera.runtime import _accept_device_event_ms, _accept_nvidia_event_ms
    assert len({id(accept_apple_device_ns), id(_accept_device_event_ms),
                id(_accept_nvidia_event_ms)}) == 3
