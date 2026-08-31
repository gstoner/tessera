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

def test_every_matmul_candidate_reports_a_device_latency():
    """The core regression.

    Two of the four NVIDIA matmul candidates -- including the Tier-3 delegate
    -- previously returned `None` here, so the only available comparison was
    host-dominated wall time. Decision #28 displaces a hand-tuned kernel when
    a compiled one measures faster *and* in budget; a candidate that cannot be
    measured is exempt from the first half by construction.
    """
    from tessera.compiler.fusion_core import MatmulRegion

    region = MatmulRegion(dtype="float16")
    A, B = _operands(512, 512, 512, "float16")

    unmeasurable = []
    for c in _candidates():
        if not (c.available() and c.applies_to(region)):
            continue
        latency = c.measure_device_latency(region, A, B, reps=20, warmup=5)
        if latency is None or not (latency > 0.0):
            unmeasurable.append((c.name, latency))
    assert not unmeasurable, (
        "candidates with no device-resident latency cannot be compared on "
        f"device and so can never be displaced: {unmeasurable}")


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


def test_the_delegate_is_selected_and_is_measurably_the_fastest():
    """Tier priority picks the delegate. This asserts that choice is also the
    right one *on device*, which is the only basis Decision #28 accepts.

    If a compiled Tier-2 lane ever measures faster here, this test failing is
    the correct outcome: it is the signal to let measurement override tier
    priority for this shape, not a reason to loosen the assertion.
    """
    from tessera.compiler.emit.candidate import OP_MATMUL, Tier, arbitrate
    from tessera.compiler.fusion_core import MatmulRegion

    region = MatmulRegion(dtype="float16")
    A, B = _operands(512, 512, 512, "float16")

    winner = arbitrate(region, OP_MATMUL, "nvidia")
    assert winner is not None and winner.name == SHIPPED
    assert int(winner.tier) == int(Tier.HAND_TUNED)

    timings = {}
    for c in _candidates():
        if not (c.available() and c.applies_to(region)):
            continue
        latency = c.measure_device_latency(region, A, B, reps=20, warmup=5)
        if latency is not None:
            timings[c.name] = latency
    assert SHIPPED in timings
    fastest = min(timings, key=timings.__getitem__)
    assert fastest == SHIPPED, (
        f"tier priority selected {SHIPPED} but {fastest} measured faster on "
        f"device: {timings}")
