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

#: Matmul candidates allowed to lack a device timer, each with its reason.
#:
#: Empty since the emitted lane gained one: `ptx_emit`'s block-index
#: convention (`mt = ctaid.x*16`) is now carried to the launch bridge as a
#: `tileLaunchConfig` flag (`NvidiaMmaGemmEmittedCandidate.measure_device_latency`),
#: and on 2026-09-26 it timed 0.0285 ms at 512^3 on the RTX 5070. It used to be
#: listed here, which let a race that silently lacked the fastest kernel pass.
#: An unexplained `None` is a regression; add an entry only with its reason.
_NO_DEVICE_TIMER: frozenset[str] = frozenset()


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


def test_the_measured_arbiter_selects_the_fastest_in_budget_candidate():
    """What Decision #28 actually asks of this lane -- not which kernel wins.

    Replaces `test_the_delegate_wins_on_device_only_at_small_shapes`, which
    pinned a *ranking* (delegate fastest at 512^3, a compiled lane at 2048^3)
    to the RTX 5070. The ranking was never the contract, and it went stale the
    moment the field changed: once the emitted PTX lane gained a device timer
    it joined the race and measured fastest at 512^3 too (2026-09-26, this
    box, f16, device-resident CUDA events: emitted 0.0285 ms, shipped 0.0455,
    tile_direct 0.0524, tile_shared 0.0590). A test that fails when a
    compiled kernel starts beating the hand-tuned one is asserting against the
    arbiter's purpose.

    Decision #28's contract is: among candidates that pass the F4 oracle, the
    measured arbiter picks the fastest, and a hand-tuned delegate is displaced
    only by one that is faster *and* in budget. So, at a small and a large
    shape, on whatever NVIDIA part runs this:

    * the delegate is in the race and every live candidate was device-timed
      (an untimed candidate cannot win or lose honestly);
    * the verdict is the minimum of its own recorded measurements;
    * the winner executes a kernel (no reference fallback) and meets the
      delegate's own declared budget (`tolerance` + `tolerance_rel`) -- the
      budget it must match to displace the delegate.

    No ranking is asserted, so the model pin the old test needed is gone.
    """
    from tessera.compiler.emit import autotune as AT
    from tessera.compiler.emit.candidate import OP_MATMUL
    from tessera.compiler.fusion_core import MatmulRegion

    region = MatmulRegion(dtype="float16")
    budget = _shipped().contract_for(region)
    for size in (512, 2048):
        A, B = _operands(size, size, size, "float16")
        cache = AT.MeasureCache()
        winner = AT.measured_arbitrate(
            region, OP_MATMUL, "nvidia", A, B, dims=(size, size, size),
            dtype="float16", cache=cache, reps=25, warmup=10,
            timing=AT.TIMING_DEVICE)
        assert winner is not None, f"no in-budget candidate at {size}^3"
        [row] = cache.to_dict()["records"]
        assert SHIPPED in row["candidates"], (
            f"the delegate was not raced at {size}^3: {row}")
        assert set(row.get("unmeasured") or {}) <= _NO_DEVICE_TIMER, (
            f"an untimed candidate at {size}^3: {row.get('unmeasured')}")
        fastest = min(row["candidates"], key=row["candidates"].__getitem__)
        assert row["winner"] == winner.name == fastest, (
            f"at {size}^3 the verdict is not the measured minimum: {row}")
        out, tag = winner.run(region, A, B)
        assert tag != "reference", f"{winner.name} declined at {size}^3"
        np.testing.assert_allclose(
            np.asarray(out, np.float64),
            np.asarray(region.reference(A, B), np.float64),
            atol=budget.tolerance, rtol=budget.tolerance_rel,
            err_msg=f"{winner.name} measured fastest at {size}^3 but is out "
                    "of the delegate's declared budget")
