"""Workstream E3 — the arbiter escape hatch: a hand-tuned kernel is never orphaned.

The three-tier / measured-arbiter model (Decision #28) is lead-safe only if the
crown-jewel hand-tuned lane (Tier 3) **wins by default** and can always be
**forced**. This asserts both — host-free with fake candidates, and live on the
NVIDIA shipped mma.sync GEMM (the hand-tuned Tier-3 matmul lane).
"""
from __future__ import annotations

import os
import shutil

import numpy as np
import pytest

from tessera.compiler.emit import candidate as C
from tessera.compiler.emit.candidate import (
    OP_MATMUL,
    ArbiterError,
    Candidate,
    Tier,
    arbitrate,
    register_candidate,
    run_arbitrated,
)

_TGT = "e3_faketarget"


class _FakeReg:
    dtype = "bfloat16"

    def reference(self, A, B):
        return np.asarray(A, np.float32) @ np.asarray(B, np.float32)


class _FakeCand(Candidate):
    op = OP_MATMUL
    target = _TGT

    def __init__(self, name, tier, tag):
        self.name = name
        self.tier = tier
        self._tag = tag

    def run(self, region, A, B, *a, **k):
        return region.reference(A, B), self._tag


# ── host-free: the force/tier contract with fake candidates ──────────────────

def test_e3_hand_tuned_wins_by_default_and_is_forceable():
    hand = _FakeCand("e3_handtuned", Tier.HAND_TUNED, "handtuned_real")
    synth = _FakeCand("e3_synth", Tier.SYNTHESIZED, "synth_real")
    register_candidate(synth)          # register the lower tier FIRST, so a default
    register_candidate(hand)           # win by tier (not registration order) is real
    region = _FakeReg()
    A = np.zeros((2, 2), np.float32)
    B = np.zeros((2, 2), np.float32)

    # 1) the hand-tuned lane wins by DEFAULT (tier priority) — never orphaned.
    win = arbitrate(region, OP_MATMUL, _TGT, verify=False)
    assert win is not None and win.name == "e3_handtuned"

    # 2) it CAN be forced, and runs.
    _, tag = run_arbitrated(region, OP_MATMUL, _TGT, A, B,
                            verify=False, force="e3_handtuned")
    assert tag == "handtuned_real"

    # 3) a LOWER tier is also forceable — tier priority is overridable (the D2/E3
    #    seam a measured cost model uses to displace the default).
    _, tag2 = run_arbitrated(region, OP_MATMUL, _TGT, A, B,
                             verify=False, force="e3_synth")
    assert tag2 == "synth_real"

    # 4) forcing an unknown candidate raises honestly (never silently picks another).
    with pytest.raises(ArbiterError, match="not available"):
        arbitrate(region, OP_MATMUL, _TGT, verify=False, force="e3_nonexistent")


# ── live: the NVIDIA shipped mma.sync GEMM is the hand-tuned Tier-3 lane ──────

def _nvidia_matmul_live() -> bool:
    if not (shutil.which("nvcc") or os.path.exists("/usr/local/cuda/bin/nvcc")):
        return False
    try:
        from tessera import runtime as rt
        return (rt._nvidia_mma_runtime_available()
                and rt._load_nvidia_ptx_launch() is not None)
    except Exception:
        return False


@pytest.mark.slow
@pytest.mark.hardware_nvidia
@pytest.mark.skipif(not _nvidia_matmul_live(),
                    reason="live NVIDIA GPU + shipped GEMM + PTX launch bridge required")
def test_e3_live_nvidia_hand_tuned_forced_and_wins():
    import tessera.compiler.emit.nvidia_cuda  # noqa: F401 — registers the candidates
    from tessera.compiler.fusion_core import MatmulRegion
    region = MatmulRegion(dtype="bfloat16")
    rng = np.random.default_rng(0)
    A = (rng.standard_normal((32, 32)) * 0.4).astype(np.float32)
    B = (rng.standard_normal((32, 16)) * 0.4).astype(np.float32)
    ref = region.reference(A, B)

    # the shipped hand-tuned GEMM (Tier 3) wins by default over the emitted Tier-2.
    win = arbitrate(region, OP_MATMUL, "nvidia")
    assert win is not None and win.name == "nvidia_mma_gemm_shipped"
    assert int(win.tier) == int(Tier.HAND_TUNED)

    # forceable + runs on-GPU.
    out, tag = run_arbitrated(region, OP_MATMUL, "nvidia", A, B,
                              force="nvidia_mma_gemm_shipped")
    assert tag == "nvidia_mma_shipped"
    np.testing.assert_allclose(out, ref, atol=5e-3, rtol=0)

    # the emitted Tier-2 lane is forceable too (never orphaned either way).
    out2, tag2 = run_arbitrated(region, OP_MATMUL, "nvidia", A, B,
                                force="nvidia_mma_gemm_emitted")
    assert tag2 == "nvidia_ptx_gemm"
    np.testing.assert_allclose(out2, ref, atol=5e-3, rtol=0)

    # forcing an unavailable candidate raises honestly.
    with pytest.raises(ArbiterError):
        arbitrate(region, OP_MATMUL, "nvidia", force="nvidia_does_not_exist")
