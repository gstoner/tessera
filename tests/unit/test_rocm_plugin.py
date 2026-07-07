"""Workstream C — ROCm gfx1151 plugin: wire shipped kernels into the F4 oracle.

Three layers:

1. **Registration + decline paths (host-free)** — a runner-only plugin registers
   for target "rocm" (no emitter/compiler); it declines matmul-epilogue / gated /
   pointwise regions to the numpy reference (ROCm has no single fused GPU kernel
   for those yet — that is C3).
2. **Accuracy-budget wiring (host-free)** — the F4 oracle widens its tolerance to
   a runner's declared ``accuracy_atol`` so an f16 lead kernel's rounding is not
   misread as a miscompile, while an O(1) miscompile still is.
3. **Live attention gate (needs a live gfx1151 + compiled flash lane)** — the
   shipped compiled flash-attn kernel is gated by the same universal oracle:
   `run_fused_attention` runs on-device ("rocm_hip") and matches the numpy
   reference within the f16 budget.
"""
from __future__ import annotations

import numpy as np
import pytest

import tessera.compiler.fusion as F
import tessera.compiler.emit.rocm_hip as rocm  # noqa: F401 — self-registers
from tessera.compiler.emit.kernel_emitter import KernelRunner, get_runner


def _rocm_flash_live() -> bool:
    try:
        from tessera import runtime as rt
        return rt._rocm_compiled_flash_attn_available()
    except Exception:
        return False


# ── 1. Registration + decline paths (host-free) ───────────────────────────────

def test_rocm_runner_registered_with_f16_budget():
    r = get_runner("rocm")
    assert r.target == "rocm"
    assert r.accuracy_atol == 5e-3          # f16 storage budget


def test_rocm_registers_no_emitter():
    # ROCm's kernels are shipped, not synthesized here — emit("rocm") still raises.
    from tessera.compiler.emit.kernel_emitter import EmitError, emit_kernel
    with pytest.raises(EmitError, match="no KernelEmitter registered"):
        emit_kernel(F.FusedRegion(epilogue=("relu",)), "rocm")


def test_rocm_declines_non_attention_regions():
    r = get_runner("rocm")
    A = np.zeros((8, 12), np.float32)
    B = np.zeros((12, 16), np.float32)
    _, ex = r.run_fused_region(F.FusedRegion(epilogue=("relu",)), A, B, None)
    assert ex == "reference"
    _, ex = r.run_gated_matmul_region(F.GatedMatmulRegion(),
                                      A, np.zeros((12, 16), np.float32),
                                      np.zeros((12, 16), np.float32))
    assert ex == "reference"


# ── 2. Accuracy-budget wiring (host-free) ─────────────────────────────────────

class _FakeF16Attn(KernelRunner):
    """Returns the exact reference plus a fixed 3e-3 perturbation under a real
    tag — models an f16 kernel whose rounding is within budget but past 1e-3."""
    target = "rocm_budget_test"
    accuracy_atol: float | None = 5e-3

    def run_fused_region(self, region, *a, **k): raise NotImplementedError
    def run_fused_attention(self, region, Q, K, V, *a, **k):
        return region.reference(Q, K, V) + np.float32(3e-3), "rocm_hip"
    def run_gated_matmul_region(self, region, *a, **k): raise NotImplementedError
    def run_pointwise_graph(self, region, *a, **k): raise NotImplementedError


def test_oracle_honors_accuracy_budget():
    F.clear_verification_cache()
    region = F.AttentionRegion(scale=0.25)
    within = _FakeF16Attn()
    assert F.verify_synthesized_attention(region, runner=within, force=True) is True


def test_oracle_without_budget_rejects_same_error():
    # Same 3e-3 error but NO declared budget (accuracy_atol=None) → the default
    # 1e-3 oracle rejects it. Proves the budget is what admits the f16 kernel.
    F.clear_verification_cache()

    class _NoBudget(_FakeF16Attn):
        target = "rocm_nobudget_test"
        accuracy_atol = None

    assert F.verify_synthesized_attention(
        F.AttentionRegion(scale=0.25), runner=_NoBudget(), force=True) is False


def test_oracle_budget_still_catches_gross_miscompile():
    # A budget must not blind the oracle to a real bug: an O(1) wrong result is
    # still rejected even under the f16 budget.
    F.clear_verification_cache()

    class _Wrong(_FakeF16Attn):
        target = "rocm_wrong_test"
        def run_fused_attention(self, region, Q, K, V, *a, **k):
            return np.full_like(region.reference(Q, K, V), 9.0), "rocm_hip"

    assert F.verify_synthesized_attention(
        F.AttentionRegion(scale=0.25), runner=_Wrong(), force=True) is False


# ── 3. Live attention gate (needs a live gfx1151 + compiled flash lane) ───────

@pytest.mark.slow
@pytest.mark.skipif(not _rocm_flash_live(),
                    reason="live gfx1151 + compiled flash-attn lane required")
@pytest.mark.parametrize("scale,causal", [(1.0, False), (0.25, False), (0.125, True)])
def test_live_rocm_attention_gated(scale, causal):
    F.clear_verification_cache()
    region = F.AttentionRegion(scale=scale, causal=causal)
    runner = get_runner("rocm")
    # It actually runs on the GPU...
    rng = np.random.default_rng(0)
    Q = rng.standard_normal((8, 16)).astype(np.float32)
    K = rng.standard_normal((8, 16)).astype(np.float32)
    V = rng.standard_normal((8, 16)).astype(np.float32)
    out, execution = runner.run_fused_attention(region, Q, K, V)
    assert execution == "rocm_hip"
    # ...and the universal F4 oracle gates it within the f16 budget.
    assert F.verify_synthesized_attention(region, runner=runner, force=True) is True
