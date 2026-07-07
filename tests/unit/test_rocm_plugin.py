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
