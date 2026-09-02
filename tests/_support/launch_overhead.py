"""One bound for `rt.launch` overhead, gated on the lane that actually ran.

**Why this is not a single constant** (rewritten 2026-09-02). The four
`*_perf_baseline_is_bounded` rows asserted `launch_ms < max(2.0, direct_ms*4)`
against a `direct` arm that is a numpy oracle. That 2.0 ms was calibrated when
the native ROCm lane was UNAVAILABLE on the measuring host, so `rt.launch` fell
straight through to the same oracle and cost 0.148 ms. The constant therefore
records the fallback path's cost.

When gfx1151 execution started working the launches began doing real device
work and the rows went red — without anything regressing. Measured on
Princess-Luna for the dspark draft block, steady state after the one-time
compile (642 ms):

    _rocm_sparse_launch                  1.63 ms/launch
    _rocm_dev_in  (13 host->device/call) 1.20 ms/launch
    native marshalling                   0.47 ms/launch

That is a HIP dispatch over WSL2's paravirtualised `/dev/dxg`, whose per-call
latency a numpy oracle will never be within 4x of. Asserting otherwise measures
the interconnect, not a regression.

So the bound is chosen by which lane ran:

* **fallback** — keep the original self-calibrating `max(2.0, direct_ms*4)`.
  This is the arm that caught the real defect the rows exist for: an uncached
  serialization failure that forked `tessera-opt` on every launch and cost
  70.6 ms.
* **native** — a flat ceiling, because there is no meaningful ratio between a
  device dispatch and a numpy oracle.

`NATIVE_LAUNCH_CEILING_MS` is 20.0, and the trade-off is stated rather than
hidden. Measured worst case on Princess-Luna: 4.703 ms idle, 5.541 ms under ten
busy cores. 20.0 leaves ~3.6x over that while still FAILING on the 70.6 ms
regression class with 3.5x to spare. The original row's "~12x margin over the
worst under load" criterion cannot be met here — 12 x 5.541 is 66 ms, close
enough to 70.6 that the bound would stop catching what it exists to catch. When
those two criteria conflict, catching the regression wins.
"""

from __future__ import annotations

from typing import Any, Mapping

#: Flat ceiling for a launch that dispatched to a GPU. See the module note.
#:
#: **GPU only, deliberately** (narrowed 2026-09-02, review on #686). This
#: number is 3.6x the worst gfx1151 dispatch measured over WSL2's
#: paravirtualised `/dev/dxg`. It says nothing about any other lane, and
#: applying it to one would be a hollow bound: `native_cpu` is reachable
#: (`runtime.py` picks it for `target == "x86"`), and an AVX-512 launch that
#: regressed from ~0.15 ms to 19 ms would sail past a 20 ms ceiling.
NATIVE_LAUNCH_CEILING_MS = 20.0

#: Kinds that get the flat ceiling: a device dispatch has no meaningful ratio
#: to a host-side oracle, because the two do not run on the same silicon.
#:
#: `native_cpu` is deliberately NOT here. A native CPU lane executes on the
#: same hardware as the `direct` oracle arm, so the self-calibrating
#: `max(2.0, direct_ms*4)` remains the meaningful comparison for it -- and is
#: far tighter than any flat ceiling derived from GPU dispatch would be. If a
#: CPU lane ever needs its own floor, measure it on an AVX-512 host and give it
#: a separate constant rather than widening this one.
_NATIVE_KINDS = frozenset({"native_gpu"})


def launch_execution_kind(result: Any) -> str:
    """The `execution_kind` a `rt.launch` result reports, or `"unknown"`.

    Executors with an internal fallback chain override this on the fallback
    path, which is exactly the distinction the bound needs.
    """
    if isinstance(result, Mapping):
        kind = result.get("execution_kind")
        if isinstance(kind, str) and kind:
            return kind
    return "unknown"


def assert_launch_overhead_bounded(
    *, launch_ms: float, direct_ms: float, execution_kind: str, what: str
) -> None:
    """Bound `rt.launch` overhead against the lane that actually executed."""
    if execution_kind in _NATIVE_KINDS:
        assert launch_ms < NATIVE_LAUNCH_CEILING_MS, (
            f"{what}: native launch overhead {launch_ms:.3f} ms exceeds the "
            f"{NATIVE_LAUNCH_CEILING_MS} ms ceiling (execution_kind="
            f"{execution_kind!r}). This bound is a device-dispatch ceiling, "
            f"not a ratio against the {direct_ms:.3f} ms oracle arm — compare "
            "against a previous native measurement, not against numpy."
        )
        return
    bound = max(2.0, direct_ms * 4.0)
    assert launch_ms < bound, (
        f"{what}: fallback launch overhead {launch_ms:.3f} ms exceeds "
        f"{bound:.3f} ms (execution_kind={execution_kind!r}). On the fallback "
        "arm the launch runs the same oracle as the direct arm, so anything "
        "beyond a small multiple is dispatch overhead — historically an "
        "uncached serialization failure re-forking tessera-opt per launch."
    )
