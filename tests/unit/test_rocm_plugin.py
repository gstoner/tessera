"""Workstream C3 — ROCm gfx1151 plugin: generic synth → HIP + shipped-kernel gate.

Three layers:

1. **Registration + emit + decline paths (host-free)** — a full three-seam plugin
   for target "rocm": the emitter turns a FusedRegion into HIP source; gated /
   pointwise regions (no fused GPU kernel yet) decline to the numpy reference.
2. **Accuracy-budget wiring (host-free)** — the F4 oracle widens its tolerance to
   a runner's declared ``accuracy_atol`` so an f16 lead kernel's rounding is not
   misread as a miscompile, while an O(1) miscompile still is.
3. **Live gates (needs a live gfx1151)** — the generic HIP FusedRegion lane
   (`hipcc` compile + launch, "rocm_hip") and the shipped compiled flash-attn
   lane are both gated by the same universal oracle on-device.
"""
from __future__ import annotations

import os
import shutil

import numpy as np
import pytest

import tessera.compiler.fusion as F
import tessera.compiler.emit.rocm_hip as rocm  # noqa: F401 — self-registers
from tessera.compiler.emit import candidate as C
from tessera.compiler.emit.candidate import (
    OP_ATTENTION, OP_FUSED_REGION, Tier,
)
from tessera.compiler.emit.kernel_emitter import (
    EmitError, KernelRunner, SpecPolicy, get_emitter, get_runner,
)


def _rocm_flash_live() -> bool:
    try:
        from tessera import runtime as rt
        return rt._rocm_compiled_flash_attn_available()
    except Exception:
        return False


def _rocm_hip_live() -> bool:
    if not (shutil.which("hipcc") or os.path.exists("/opt/rocm/bin/hipcc")):
        return False
    try:
        from tessera import runtime as rt
        return rt._rocm_wmma_runtime_available()
    except Exception:
        return False


# ── 1. Registration + emit + decline paths (host-free) ────────────────────────

def test_rocm_runner_registered_with_f16_budget():
    r = get_runner("rocm")
    assert r.target == "rocm"
    assert r.accuracy_atol == 5e-3          # f16 storage budget


def test_rocm_emitter_registered_produces_hip():
    from tessera.compiler.emit.kernel_cache import get_compiler
    src = get_emitter("rocm").emit(F.FusedRegion(epilogue=("bias", "gelu")),
                                   dtype="f32")
    assert src.lang == "hip"
    assert src.entry == "tessera_rocm_fused"
    assert "__global__" in src.source and 'extern "C"' in src.source
    assert "hipLaunchKernelGGL" in src.source
    assert callable(get_compiler("rocm"))


def test_rocm_emitter_rejects_unsupported():
    e = get_emitter("rocm")
    with pytest.raises(EmitError, match="cannot emit"):
        e.emit(F.AttentionRegion())
    with pytest.raises(EmitError, match="DYNAMIC"):
        e.emit(F.FusedRegion(epilogue=("relu",)), spec=SpecPolicy.DYNAMIC)
    with pytest.raises(EmitError, match="f32"):
        e.emit(F.FusedRegion(epilogue=("relu",)), dtype="f16")


def test_rocm_declines_gated_and_pointwise():
    # No single fused GPU kernel for these yet — always the numpy reference.
    r = get_runner("rocm")
    A = np.zeros((8, 12), np.float32)
    _, ex = r.run_gated_matmul_region(F.GatedMatmulRegion(),
                                      A, np.zeros((12, 16), np.float32),
                                      np.zeros((12, 16), np.float32))
    assert ex == "reference"


def test_rocm_missing_required_buffer_declines_not_segfault():
    # Same NULL-deref guard as x86: a residual/bias region without the buffer must
    # not launch the HIP kernel (which would deref a null). Child process so a
    # regression is a failed assert, not a crashed session.
    import subprocess
    import sys
    import textwrap
    code = textwrap.dedent(
        """
        import numpy as np
        import tessera.compiler.fusion as F
        import tessera.compiler.emit.rocm_hip as rocm
        r = rocm.RocmHipRunner()
        A = np.zeros((8, 12), np.float32)
        B = np.zeros((12, 16), np.float32)
        for region in (F.FusedRegion(epilogue=("relu",), residual=True),
                       F.FusedRegion(epilogue=("bias", "relu"))):
            try:
                r.run_fused_region(region, A, B, None)
                raise SystemExit("expected ValueError, got a result")
            except ValueError:
                pass
        print("ok")
        """
    )
    p = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert p.returncode == 0, (
        f"missing-buffer guard failed (rc={p.returncode}, -11=SIGSEGV): "
        f"{p.stderr[-300:]}")
    assert "ok" in p.stdout


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


# ── 3. Live gates (need a live gfx1151) ───────────────────────────────────────

_C3_CHAINS = [
    F.FusedRegion(epilogue=("relu",)),
    F.FusedRegion(epilogue=("bias", "gelu")),
    F.FusedRegion(epilogue=("silu",)),
    F.FusedRegion(epilogue=("bias",), reduction="softmax"),
    F.FusedRegion(epilogue=(), reduction="rmsnorm"),
    F.FusedRegion(epilogue=("relu",), reduction="layer_norm"),
    F.FusedRegion(epilogue=("gelu",), prologue=("relu",)),
]


@pytest.mark.slow
@pytest.mark.skipif(not _rocm_hip_live(),
                    reason="live gfx1151 + hipcc required")
@pytest.mark.parametrize("region", _C3_CHAINS,
                         ids=lambda r: f"{r.epilogue}/{r.reduction}/{r.prologue}")
def test_live_rocm_generic_hip_gated(region):
    # C3: the generically-synthesized HIP FusedRegion kernel compiles with hipcc,
    # runs on gfx1151 ("rocm_hip"), matches numpy (f32), and passes the F4 oracle.
    F.clear_verification_cache()
    runner = get_runner("rocm")
    rng = np.random.default_rng(0)
    A = rng.standard_normal((8, 12)).astype(np.float32)
    B = rng.standard_normal((12, 16)).astype(np.float32)
    bias = rng.standard_normal((16,)).astype(np.float32) if region.has_bias else None
    out, execution = runner.run_fused_region(region, A, B, bias)
    assert execution == "rocm_hip"
    assert np.allclose(out, region.reference(A, B, bias), atol=1e-3)
    assert F.verify_synthesized_region(region, runner=runner, force=True) is True


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


# ── 4. D1 candidates + arbiter (C3 tail) ──────────────────────────────────────
#
# The crown-jewel WMMA GEMM `Generate*` kernel and the generic scalar HIP lane are
# both registered as arbiter candidates for target "rocm"; the flash lane is the
# attention candidate. Host-free: registration + tiers + applicability. Live: each
# candidate F4-gated on-device, the arbiter's tier-priority pick, and the E3
# escape hatch that forces the generic lane over the crown jewel.

# WMMA fuses bias-then-{relu,gelu,silu}; the arbiter must pick it for these and
# fall to the generic lane for a reduction it cannot fuse.
_WMMA_CHAINS = [
    F.FusedRegion(epilogue=("relu",)),
    F.FusedRegion(epilogue=("gelu",)),
    F.FusedRegion(epilogue=("silu",)),
    F.FusedRegion(epilogue=("bias",)),
    F.FusedRegion(epilogue=("bias", "gelu")),
]


def test_rocm_candidates_registered_with_tiers():
    fr = {c.name: c for c in C.candidates_for("rocm", OP_FUSED_REGION)}
    at = {c.name: c for c in C.candidates_for("rocm", OP_ATTENTION)}
    assert fr["rocm_generic_hip"].tier is Tier.SYNTHESIZED
    assert fr["rocm_wmma_gemm"].tier is Tier.HAND_TUNED
    assert fr["rocm_wmma_gemm"].accuracy_atol == 5e-3
    assert at["rocm_flash_attn"].tier is Tier.HAND_TUNED


def test_wmma_candidate_applicability():
    wmma = {c.name: c for c in
            C.candidates_for("rocm", OP_FUSED_REGION)}["rocm_wmma_gemm"]
    assert wmma.applies_to(F.FusedRegion(epilogue=("bias", "gelu")))
    assert wmma.applies_to(F.FusedRegion(epilogue=("relu",)))
    # act-before-bias, a non-fusable activation, and a reduction all decline:
    assert not wmma.applies_to(F.FusedRegion(epilogue=("gelu", "bias")))
    assert not wmma.applies_to(F.FusedRegion(epilogue=("sigmoid",)))
    assert not wmma.applies_to(F.FusedRegion(epilogue=("bias",),
                                             reduction="softmax"))
    assert not wmma.applies_to(F.FusedRegion(epilogue=("relu",), residual=True))


def test_wmma_candidate_declines_unrepresentable_host_free():
    # run() on a region it cannot fuse must return the honest reference, never a
    # mislabeled kernel — checkable without a GPU (it never reaches the device).
    wmma = {c.name: c for c in
            C.candidates_for("rocm", OP_FUSED_REGION)}["rocm_wmma_gemm"]
    region = F.FusedRegion(epilogue=("bias",), reduction="softmax")
    A = np.zeros((8, 12), np.float32)
    B = np.zeros((12, 16), np.float32)
    bias = np.zeros((16,), np.float32)
    _, tag = wmma.run(region, A, B, bias)
    assert tag == "reference"


@pytest.mark.slow
@pytest.mark.skipif(not _rocm_hip_live(),
                    reason="live gfx1151 + WMMA GEMM lane required")
@pytest.mark.parametrize("region", _WMMA_CHAINS,
                         ids=lambda r: f"{r.epilogue}")
def test_live_wmma_candidate_gated(region):
    # C3 tail: the hand-tuned WMMA GEMM Generate* kernel runs on gfx1151 with its
    # fused epilogue ("rocm_wmma"), matches numpy within the f16 budget, and passes
    # the same universal F4 oracle as the generic lane.
    F.clear_verification_cache()
    wmma = {c.name: c for c in
            C.candidates_for("rocm", OP_FUSED_REGION)}["rocm_wmma_gemm"]
    rng = np.random.default_rng(0)
    A = rng.standard_normal((8, 12)).astype(np.float32)
    B = rng.standard_normal((12, 16)).astype(np.float32)
    bias = rng.standard_normal((16,)).astype(np.float32) if region.has_bias else None
    out, tag = wmma.run(region, A, B, bias)
    assert tag == "rocm_wmma"
    assert np.allclose(out, region.reference(A, B, bias), atol=5e-3)
    assert C.verify_candidate(wmma, region) is True


@pytest.mark.slow
@pytest.mark.skipif(not _rocm_hip_live(),
                    reason="live gfx1151 + WMMA GEMM lane required")
def test_live_arbiter_prefers_wmma_but_falls_to_generic():
    # Default (tier-priority) arbitration: the crown-jewel WMMA wins where it
    # applies; a softmax region it cannot fuse falls to the generic HIP lane.
    F.clear_verification_cache()
    rng = np.random.default_rng(0)
    A = rng.standard_normal((8, 12)).astype(np.float32)
    B = rng.standard_normal((12, 16)).astype(np.float32)
    bias = rng.standard_normal((16,)).astype(np.float32)

    fusable = F.FusedRegion(epilogue=("bias", "gelu"))
    out, tag = C.run_arbitrated(fusable, OP_FUSED_REGION, "rocm", A, B, bias)
    assert tag == "rocm_wmma"
    assert np.allclose(out, fusable.reference(A, B, bias), atol=5e-3)

    reduce_region = F.FusedRegion(epilogue=("bias",), reduction="softmax")
    out2, tag2 = C.run_arbitrated(reduce_region, OP_FUSED_REGION, "rocm",
                                  A, B, bias)
    assert tag2 == "rocm_hip"            # generic lane; WMMA cannot fuse softmax
    assert np.allclose(out2, reduce_region.reference(A, B, bias), atol=1e-3)


@pytest.mark.slow
@pytest.mark.skipif(not _rocm_hip_live(),
                    reason="live gfx1151 + WMMA GEMM lane required")
def test_live_escape_hatch_forces_generic_over_crown_jewel():
    # E3: a hand-tuned candidate is never orphaned AND a lower tier can be forced.
    # Force the generic HIP lane on a region WMMA would otherwise win by tier.
    F.clear_verification_cache()
    rng = np.random.default_rng(0)
    A = rng.standard_normal((8, 12)).astype(np.float32)
    B = rng.standard_normal((12, 16)).astype(np.float32)
    region = F.FusedRegion(epilogue=("relu",))
    out, tag = C.run_arbitrated(region, OP_FUSED_REGION, "rocm", A, B, None,
                                force="rocm_generic_hip")
    assert tag == "rocm_hip"
    out_w, tag_w = C.run_arbitrated(region, OP_FUSED_REGION, "rocm", A, B, None,
                                    force="rocm_wmma_gemm")
    assert tag_w == "rocm_wmma"
