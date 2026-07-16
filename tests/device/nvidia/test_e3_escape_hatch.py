"""Native NVIDIA E3 arbiter escape-hatch proof.

The three-tier / measured-arbiter model (Decision #28) is lead-safe only if the
crown-jewel hand-tuned lane (Tier 3) **wins by default** and can always be
**forced**. This asserts both — host-free with fake candidates, and live on the
NVIDIA shipped mma.sync GEMM (the hand-tuned Tier-3 matmul lane).
"""
from __future__ import annotations

import numpy as np
import pytest

from tests._support.nvidia import nvidia_mma_ptx_launch_available

@pytest.mark.slow
@pytest.mark.hardware_nvidia
@pytest.mark.skipif(not nvidia_mma_ptx_launch_available(),
                    reason="live NVIDIA GPU + shipped GEMM + PTX launch bridge required")
def test_e3_live_nvidia_hand_tuned_forced_and_wins():
    import tessera.compiler.emit.nvidia_cuda  # noqa: F401 — registers the candidates
    from tessera.compiler.emit.candidate import OP_MATMUL, ArbiterError, Tier, arbitrate, run_arbitrated
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
