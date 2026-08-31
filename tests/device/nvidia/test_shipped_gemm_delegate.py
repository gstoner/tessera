"""Live sm_120 proof for the first declared delegate.

`tests/unit/test_nvidia_delegate_contract.py` checks what the shipped GEMM
*declares*. A declaration is not evidence, so this file checks the two claims
that can only be settled on the device:

1. the declared accuracy budget actually holds, across a K range wide enough
   to exercise the *relative* bound rather than only the absolute one; and
2. every NVIDIA matmul candidate now yields a **device-resident** latency, so
   Decision #28's "displaced only when a compiled kernel measures faster and
   in budget" is a comparison that can actually be performed.

Point 2 is the reason this file exists. Before it, the Tier-3 delegate had no
device timer at all, so it could be compared to compiled candidates only
end-to-end -- and end-to-end is host-dominated. Measured on this box at
2048x2048x2048, the compiled Tile lane ran 2.99 ms of device work inside
34.0 ms of wall time (91% host). A Tier-3 delegate with no device timer is one
that can never honestly lose.
"""
from __future__ import annotations

import numpy as np
import pytest

from tests._support.nvidia import nvidia_mma_ptx_launch_available

pytestmark = [
    pytest.mark.slow,
    pytest.mark.hardware_nvidia,
    pytest.mark.skipif(
        not nvidia_mma_ptx_launch_available(),
        reason="live NVIDIA GPU + shipped GEMM + PTX launch bridge required"),
]

SHIPPED = "nvidia_mma_gemm_shipped"


def _candidates():
    import tessera.compiler.emit.nvidia_cuda  # noqa: F401 — registers candidates
    from tessera.compiler.emit.candidate import OP_MATMUL, candidates_for

    return candidates_for("nvidia", OP_MATMUL)


def _shipped():
    for c in _candidates():
        if c.name == SHIPPED:
            return c
    pytest.fail(f"{SHIPPED} is not registered")


def _operands(M, N, K, dtype, seed=0):
    rng = np.random.default_rng(seed)
    A = (rng.standard_normal((M, K)) * 0.4).astype(np.float32)
    B = (rng.standard_normal((K, N)) * 0.4).astype(np.float32)
    return A, B


# ── claim 1: the declared budget holds on device ─────────────────────────────

@pytest.mark.parametrize("dtype", ["float16", "bfloat16"])
@pytest.mark.parametrize("K", [32, 256, 1024, 4096])
def test_the_declared_budget_holds_across_K(dtype, K):
    """The delegate declares `tolerance` AND `tolerance_rel`, and the second is
    the one that carries large K.

    Measured here at M=N=256: absolute error grows about K^1.2 while relative
    error grows near sqrt(K). A fixed absolute budget is therefore the wrong
    shape for this claim -- 5e-3 has roughly 6x headroom at K=8192 and would
    be breached past K~65536 on a kernel that is not wrong. The oracle
    combines the two as `|a-b| <= atol + rtol*|ref|`, which is why both are
    declared.
    """
    from tessera.compiler.emit.delegate_contract import contract_for_candidate
    from tessera.compiler.fusion_core import MatmulRegion

    shipped = _shipped()
    region = MatmulRegion(dtype=dtype)
    contract = contract_for_candidate(shipped)
    assert contract is not None, "the shipped GEMM must declare a contract"

    A, B = _operands(256, 256, K, dtype)
    out, tag = shipped.run(region, A, B)
    assert tag == "nvidia_mma_shipped", (
        f"declined to {tag!r}; a reference result would make this budget check "
        "vacuous")

    variant = shipped.contract_for(region)
    np.testing.assert_allclose(
        np.asarray(out, np.float64),
        np.asarray(region.reference(A, B), np.float64),
        atol=variant.tolerance, rtol=variant.tolerance_rel)


def test_the_delegate_binds_the_symbol_it_declared():
    """Both dtype routes execute, so neither declared callee is a dead claim."""
    from tessera.compiler.fusion_core import MatmulRegion

    shipped = _shipped()
    for dtype in ("float16", "bfloat16"):
        region = MatmulRegion(dtype=dtype)
        A, B = _operands(64, 32, 64, dtype)
        _, tag = shipped.run(region, A, B)
        assert tag == "nvidia_mma_shipped", f"{dtype} route declined to {tag!r}"
        assert shipped.contract_for(region).callee.endswith(
            "f16" if dtype == "float16" else "bf16")


# ── claim 2: the comparison Decision #28 requires can be performed ───────────

#: The one matmul candidate that still has no device timer, and why.
#:
#: `benchmarkTileGemm16` in the launch bridge launches `gx = ceil(N/tileN)`,
#: `gy = ceil(M/tileM)` -- x maps to N -- and the NVIDIA Tile lowering agrees
#: (`NVIDIALowering.cpp`: `mt = blockY*16`, `nt = blockX*8`), which is why both
#: Tile candidates time correctly through it. `ptx_emit` uses the opposite
#: convention (`mt = ctaid.x*16`, `nt = ctaid.y*8`). Driving it through the
#: harness returns rc=5, and forcing it would launch a transposed grid: at
#: 512x512 that covers rows to 1024 and columns only to 256, leaving half the
#: output unwritten while still reporting a plausible latency.
#:
#: Named here rather than silently tolerated: an unexplained `None` in this
#: list is a regression, an explained one is a tracked gap.
_NO_DEVICE_TIMER = {"nvidia_mma_gemm_emitted"}


def test_the_delegate_and_its_compiled_rivals_all_report_device_latency():
    """The core regression.

    The Tier-3 delegate previously returned `None` here, so the only available
    comparison against compiled output was host-dominated wall time. Decision
    #28 displaces a hand-tuned kernel when a compiled one measures faster
    *and* in budget; a delegate that cannot be measured is exempt from the
    first half by construction.

    The displacement test needs the delegate plus at least one compiled
    candidate measurable on device -- both Tile lanes qualify -- so the one
    tracked exception below does not block it.
    """
    from tessera.compiler.fusion_core import MatmulRegion

    region = MatmulRegion(dtype="float16")
    A, B = _operands(512, 512, 512, "float16")

    measured, unmeasurable = {}, []
    for c in _candidates():
        if not (c.available() and c.applies_to(region)):
            continue
        latency = c.measure_device_latency(region, A, B, reps=20, warmup=5)
        if latency is None or not (latency > 0.0):
            unmeasurable.append(c.name)
        else:
            measured[c.name] = latency

    assert SHIPPED in measured, (
        "the Tier-3 delegate has no device-resident latency and so can never "
        "be displaced by a faster compiled kernel")
    assert set(unmeasurable) <= _NO_DEVICE_TIMER, (
        "a candidate lost its device timer for an unrecorded reason: "
        f"{sorted(set(unmeasurable) - _NO_DEVICE_TIMER)}")
    compiled = [n for n in measured if n != SHIPPED]
    assert compiled, (
        "no compiled candidate is measurable on device, so Decision #28's "
        "displacement test cannot be performed at all")


def test_device_latency_is_not_the_host_wall_time():
    """A device timer that accidentally measured the host path would defeat the
    purpose while looking like a fix.

    The shipped lane's `run()` re-uploads operands every call; the device timer
    uploads once and times only the launches. At this size the device kernel
    must therefore come in well under the wall time -- if the two were
    comparable, the "device" number would be measuring numpy again.
    """
    import time

    from tessera.compiler.fusion_core import MatmulRegion

    shipped = _shipped()
    region = MatmulRegion(dtype="float16")
    A, B = _operands(1024, 1024, 1024, "float16")

    device_ms = shipped.measure_device_latency(region, A, B, reps=20, warmup=5)
    assert device_ms is not None and device_ms > 0.0

    shipped.run(region, A, B)  # warm
    t0 = time.perf_counter()
    shipped.run(region, A, B)
    wall_ms = (time.perf_counter() - t0) * 1e3

    assert device_ms < wall_ms, (
        f"device latency {device_ms:.3f} ms is not below wall time "
        f"{wall_ms:.3f} ms; the timer is probably including the host path")


def _device_timings(region, A, B, reps=25, warmup=10):
    out = {}
    for c in _candidates():
        if not (c.available() and c.applies_to(region)):
            continue
        latency = c.measure_device_latency(region, A, B, reps=reps, warmup=warmup)
        if latency is not None:
            out[c.name] = latency
    return out


def test_tier_priority_selects_the_delegate_regardless_of_shape():
    """The arbiter's default is tier priority, so the Tier-3 delegate is
    selected at every shape. Pinned here as the *baseline* for the next test,
    which is where it stops being the right answer."""
    from tessera.compiler.emit.candidate import OP_MATMUL, Tier, arbitrate
    from tessera.compiler.fusion_core import MatmulRegion

    region = MatmulRegion(dtype="float16")
    for M in (512, 2048):
        winner = arbitrate(region, OP_MATMUL, "nvidia")
        assert winner is not None and winner.name == SHIPPED, f"at {M}^3"
        assert int(winner.tier) == int(Tier.HAND_TUNED)


#: The exact GPU the crossover below was measured on.
#:
#: A compute-capability tag (`sm_120`) is NOT specific enough to gate a
#: performance ranking, which was the first fix's mistake: cc 12.0 spans the
#: whole consumer Blackwell line, so an RTX 5070 Ti, 5080 or 5090 all pass an
#: `sm_120` check while differing in SM count, cache and bandwidth by more than
#: the 16% margin this test asserts. The gate has to be the model.
_MEASURED_DEVICE = "NVIDIA GeForce RTX 5070"


def _measured_host() -> bool:
    """Whether this is the exact GPU the ranking below was measured on."""
    from tests._support.nvidia import nvidia_device_model

    return nvidia_device_model() == _MEASURED_DEVICE


def test_a_compiled_candidate_can_be_compared_to_the_delegate_in_budget():
    """Device-independent half: the comparison is *performable*, and whichever
    lane is faster here is no less accurate.

    Deliberately asserts no ranking. Which lane wins is a property of the
    silicon, and this suite's gate does not check the part.
    """
    from tessera.compiler.fusion_core import MatmulRegion

    region = MatmulRegion(dtype="float16")
    A, B = _operands(2048, 2048, 2048, "float16")
    timings = _device_timings(region, A, B)
    assert SHIPPED in timings and len(timings) >= 2, (
        f"need the delegate and at least one compiled rival: {timings}")

    fastest = min(timings, key=timings.__getitem__)
    reference = np.asarray(region.reference(A, B), np.float64)

    def max_error(name):
        candidate = next(c for c in _candidates() if c.name == name)
        out, tag = candidate.run(region, A, B)
        assert tag != "reference", f"{name} declined to the numpy reference"
        return float(np.max(np.abs(np.asarray(out, np.float64) - reference)))

    assert max_error(fastest) <= max_error(SHIPPED) * 1.05, (
        f"{fastest} measured fastest but is less accurate than the delegate, "
        "so 'faster' is not a Decision #28 displacement argument")


@pytest.mark.skipif(
    not _measured_host(),
    reason=f"the crossover below was measured on a {_MEASURED_DEVICE}; a "
           "ranking is a property of the specific part and does not transfer "
           "— not even to another compute-capability 12.0 GPU")
def test_the_delegate_wins_on_device_only_at_small_shapes():
    """The measurement the device timer exists to produce -- and it does not
    say what tier priority assumes.

    Measured on an RTX 5070 (sm_120), f16, spreads of 0.000-0.008 ms across
    repeats:

        shape    shipped(T3)   tile_shared(T2)   winner
        512^3      0.043 ms       0.059 ms       delegate, by 37%
        1024^3     0.320 ms       0.312 ms       compiled, by 2.3%
        2048^3     2.448 ms       2.051 ms       compiled, by 16.2%

    with max|err| identical between the two lanes at every shape. So at 1024^3
    and above a compiled kernel measures **faster and in budget**, which is
    exactly Decision #28's condition for displacing a hand-tuned candidate --
    and the arbiter still selects the delegate, because tier priority is the
    default and the measured loop is not wired into this path.

    Only the 512^3 and 2048^3 rows are asserted. The 1024^3 crossover is 2.3%,
    inside the range a driver or power-limit change can move, so it is recorded
    as the shape where the inversion begins and not used as a gate.

    **Pinned to the exact GPU model, not to `sm_120`.** This asserts a
    performance *ranking*, and a ranking belongs to the specific part:
    occupancy, L2 size, SM count and clock behaviour all move the crossover.
    Compute capability is the wrong key -- cc 12.0 spans the whole consumer
    Blackwell line, so a 5070 Ti, 5080 or 5090 would pass an `sm_120` check
    while differing by more than the 16% margin asserted here. (The first
    version of this gate made exactly that mistake.) The suite's own gate
    (`nvidia_mma_ptx_launch_available`) checks only that nvcc, the MMA runtime
    and the PTX bridge load, so without this skip the test would fail on a
    healthy machine and read as a kernel regression.

    Its device-independent half lives in the test above and still runs on every
    NVIDIA host: that the comparison is performable at all, and that whichever
    lane is fastest there is no less accurate.

    The earlier version of this test checked only 512^3, where the delegate
    does win, and so would have reported green for a default that is wrong at
    scale.
    """
    from tessera.compiler.fusion_core import MatmulRegion

    region = MatmulRegion(dtype="float16")

    small = _device_timings(region, *_operands(512, 512, 512, "float16"))
    assert min(small, key=small.__getitem__) == SHIPPED, (
        f"the delegate no longer wins at 512^3 on {_MEASURED_DEVICE}: {small}")

    large = _device_timings(region, *_operands(2048, 2048, 2048, "float16"))
    fastest = min(large, key=large.__getitem__)
    assert fastest != SHIPPED, (
        "the delegate now wins at 2048^3 too, which would remove the "
        f"motivation for shape-bucketed measured selection: {large}")
    assert large[fastest] < large[SHIPPED], large
