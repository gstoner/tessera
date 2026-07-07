"""Workstream C2 — NVIDIA (sm_120) plugin: generic synth → CUDA.

Two layers:

1. **Registration + emit + decline paths (host-free)** — a full three-seam plugin
   for target "nvidia": the emitter turns a FusedRegion into CUDA source;
   attention / gated / pointwise regions (no fused CUDA kernel yet) decline to the
   numpy reference; unsupported regions/policies/dtypes raise EmitError.
2. **Live gate (needs a live NVIDIA GPU + nvcc)** — the generically-synthesized
   CUDA FusedRegion kernel compiles with nvcc, runs on-device ("nvidia_cuda"),
   matches numpy (f32), and passes the same universal F4 oracle as ROCm/x86.

The generic CUDA lane is the Tier-1 candidate; NVIDIA has no fused hand-tuned
FusedRegion kernel to register as Tier-3 yet (the shipped mma.sync GEMM is a pure
matmul, served by the jit nvidia_mma executor).
"""
from __future__ import annotations

import os
import shutil

import numpy as np
import pytest

import tessera.compiler.fusion as F
import tessera.compiler.emit.nvidia_cuda as nvidia  # noqa: F401 — self-registers
from tessera.compiler.emit import candidate as C
from tessera.compiler.emit.candidate import OP_FUSED_REGION, Tier
from tessera.compiler.emit.kernel_emitter import (
    EmitError, SpecPolicy, get_emitter, get_runner,
)


def _nvidia_cuda_live() -> bool:
    if not (shutil.which("nvcc") or os.path.exists("/usr/local/cuda/bin/nvcc")):
        return False
    try:
        from tessera import runtime as rt
        return rt._nvidia_mma_runtime_available()
    except Exception:
        return False


# ── 1. Registration + emit + decline paths (host-free) ────────────────────────

def test_nvidia_runner_registered():
    r = get_runner("nvidia")
    assert r.target == "nvidia"
    assert r.accuracy_atol is None          # f32 lane — exact, no budget widening


def test_nvidia_emitter_registered_produces_cuda():
    from tessera.compiler.emit.kernel_cache import get_compiler
    src = get_emitter("nvidia").emit(F.FusedRegion(epilogue=("bias", "gelu")),
                                     dtype="f32")
    assert src.lang == "cuda"
    assert src.entry == "tessera_nvidia_fused"
    assert "__global__" in src.source and 'extern "C"' in src.source
    assert "<<<" in src.source and "cudaMemcpy" in src.source
    assert callable(get_compiler("nvidia"))


def test_nvidia_emit_is_deterministic():
    # Byte-identical emit for the same region (golden-IR / cache-key discipline).
    e = get_emitter("nvidia")
    r = F.FusedRegion(epilogue=("relu",), reduction="softmax")
    assert e.emit(r).source == e.emit(r).source


def test_nvidia_emitter_rejects_unsupported():
    e = get_emitter("nvidia")
    with pytest.raises(EmitError, match="cannot emit"):
        e.emit(F.AttentionRegion())
    with pytest.raises(EmitError, match="DYNAMIC"):
        e.emit(F.FusedRegion(epilogue=("relu",)), spec=SpecPolicy.DYNAMIC)
    with pytest.raises(EmitError, match="f32"):
        e.emit(F.FusedRegion(epilogue=("relu",)), dtype="f16")


def test_nvidia_declines_attention_gated_pointwise():
    # No single fused CUDA kernel for these yet — always the numpy reference.
    r = get_runner("nvidia")
    A = np.zeros((8, 12), np.float32)
    _, ex = r.run_gated_matmul_region(F.GatedMatmulRegion(),
                                      A, np.zeros((12, 16), np.float32),
                                      np.zeros((12, 16), np.float32))
    assert ex == "reference"
    _, ex2 = r.run_fused_attention(F.AttentionRegion(scale=0.25),
                                   np.zeros((4, 8), np.float32),
                                   np.zeros((4, 8), np.float32),
                                   np.zeros((4, 8), np.float32))
    assert ex2 == "reference"


def test_nvidia_generic_candidate_registered():
    cands = C.candidates_for("nvidia", OP_FUSED_REGION)
    names = [c.name for c in cands]
    assert "nvidia_generic_cuda" in names
    gen = next(c for c in cands if c.name == "nvidia_generic_cuda")
    assert gen.tier == Tier.SYNTHESIZED and gen.target == "nvidia"


def test_nvidia_missing_required_buffer_declines_not_segfault():
    # Same NULL-deref guard as x86/ROCm: a residual/bias region without the buffer
    # must not launch the CUDA kernel (which would deref a null). Child process so a
    # regression is a failed assert, not a crashed session.
    import subprocess
    import sys
    import textwrap
    code = textwrap.dedent(
        """
        import numpy as np
        import tessera.compiler.fusion as F
        import tessera.compiler.emit.nvidia_cuda as nvidia
        r = nvidia.NvidiaCudaRunner()
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


# ── 2. Live gate (needs a live NVIDIA GPU + nvcc) ─────────────────────────────

_C2_CHAINS = [
    F.FusedRegion(epilogue=("relu",)),
    F.FusedRegion(epilogue=("bias", "gelu")),
    F.FusedRegion(epilogue=("silu",)),
    F.FusedRegion(epilogue=("bias",), reduction="softmax"),
    F.FusedRegion(epilogue=(), reduction="rmsnorm"),
    F.FusedRegion(epilogue=("relu",), reduction="layer_norm"),
    F.FusedRegion(epilogue=("gelu",), prologue=("relu",)),
]


@pytest.mark.slow
@pytest.mark.skipif(not _nvidia_cuda_live(),
                    reason="live NVIDIA GPU + nvcc required")
@pytest.mark.parametrize("region", _C2_CHAINS,
                         ids=lambda r: f"{r.epilogue}/{r.reduction}/{r.prologue}")
def test_live_nvidia_generic_cuda_gated(region):
    # C2: the generically-synthesized CUDA FusedRegion kernel compiles with nvcc,
    # runs on an NVIDIA GPU ("nvidia_cuda"), matches numpy (f32), passes F4.
    F.clear_verification_cache()
    runner = get_runner("nvidia")
    rng = np.random.default_rng(0)
    A = rng.standard_normal((8, 12)).astype(np.float32)
    B = rng.standard_normal((12, 16)).astype(np.float32)
    bias = rng.standard_normal((16,)).astype(np.float32) if region.has_bias else None
    out, execution = runner.run_fused_region(region, A, B, bias)
    assert execution == "nvidia_cuda"
    np.testing.assert_allclose(out, region.reference(A, B, bias), atol=1e-3)
    assert F.verify_synthesized_region(region, runner=runner, force=True) is True
