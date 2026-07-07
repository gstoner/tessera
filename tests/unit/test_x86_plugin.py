"""Workstream C1 — x86 (Zen 5) codegen plugin contract + F4 gating.

Mirrors the Apple emitter/runner contract tests for the new x86 backend
(`emit/x86_llvm.py`). Three layers, matching the handoff's definition of done:

1. **Registration + emit (host-free)** — the three seams register for target
   "x86"; `emit` produces C source and rejects unsupported regions/policies/dtypes
   (Decision #21).
2. **F4 gating (host-free-safe)** — the universal oracle gates the x86 runner:
   a wrong kernel is rejected, a correct one trusted. On a host without a C
   compiler the runner skip-cleans to the numpy reference (tag "reference"),
   which the oracle trusts — so the layer stays green everywhere and becomes a
   real silicon check on the Zen 5 box.
3. **Real execution (needs a C compiler)** — compile + `ctypes` launch on this
   box; assert the kernel ran ("x86_native") and matches numpy across the
   epilogue / reduction / prologue chains.
"""
from __future__ import annotations

import numpy as np
import pytest

import tessera.compiler.fusion as F
import tessera.compiler.emit.x86_llvm as x86  # noqa: F401 — self-registers
from tessera.compiler.emit.kernel_emitter import (
    EmitError, SpecPolicy, get_emitter, get_runner,
)

_HAVE_CC = x86._cc() is not None and __import__("shutil").which(x86._cc()) is not None


# ── 1. Registration + emit (host-free) ────────────────────────────────────────

def test_x86_seams_registered():
    from tessera.compiler.emit.kernel_cache import get_compiler
    assert get_emitter("x86").target == "x86"
    assert get_runner("x86").target == "x86"
    assert callable(get_compiler("x86"))


def test_x86_does_not_hijack_active_runner():
    # Registered default=False, so Apple stays the active default runner.
    from tessera.compiler.emit.kernel_emitter import active_runner
    ar = active_runner()
    assert ar is None or ar.target != "x86"


def test_emit_produces_c_source():
    src = get_emitter("x86").emit(F.FusedRegion(epilogue=("bias", "gelu")), dtype="f32")
    assert src.lang == "c"
    assert src.entry == "tessera_x86_fused"
    assert "int tessera_x86_fused(" in src.source
    assert "bias[n]" in src.source and "tanhf" in src.source  # bias + gelu emitted


def test_emit_rejects_non_fused_region():
    with pytest.raises(EmitError, match="cannot emit"):
        get_emitter("x86").emit(F.AttentionRegion())


def test_emit_rejects_dynamic_spec():
    with pytest.raises(EmitError, match="DYNAMIC"):
        get_emitter("x86").emit(F.FusedRegion(epilogue=("relu",)), spec=SpecPolicy.DYNAMIC)


def test_emit_rejects_non_f32_dtype():
    with pytest.raises(EmitError, match="f32"):
        get_emitter("x86").emit(F.FusedRegion(epilogue=("relu",)), dtype="f16")


# ── 2. F4 gating (host-free-safe) ─────────────────────────────────────────────

def test_oracle_trusts_correct_or_fallback_x86():
    # A correct kernel (or a reference fallback on a compiler-less host) is trusted.
    F.clear_verification_cache()
    for r in (F.FusedRegion(epilogue=("relu",)),
              F.FusedRegion(epilogue=("bias", "gelu")),
              F.FusedRegion(epilogue=(), reduction="softmax")):
        assert F.verify_synthesized_region(r, runner=get_runner("x86"), force=True) is True


def test_oracle_rejects_wrong_x86_kernel():
    # Prove the gate BITES for x86: a runner that returns a wrong result under the
    # real-execution tag must be rejected (not silently trusted).
    F.clear_verification_cache()

    class _WrongX86(x86.X86CRunner):
        def run_fused_region(self, region, A, B, bias=None, *a, **k):
            return np.full((A.shape[0], B.shape[1]), 999.0, np.float32), "x86_native"

    assert F.verify_synthesized_region(
        F.FusedRegion(epilogue=("relu",)), runner=_WrongX86(), force=True) is False


# ── 3. Real execution (needs a C compiler; runs on the Zen 5 box) ─────────────

_CHAINS = [
    F.FusedRegion(epilogue=("relu",)),
    F.FusedRegion(epilogue=("bias", "gelu")),
    F.FusedRegion(epilogue=("silu",)),
    F.FusedRegion(epilogue=("sigmoid",)),
    F.FusedRegion(epilogue=("tanh",)),
    F.FusedRegion(epilogue=("bias",), reduction="softmax"),
    F.FusedRegion(epilogue=(), reduction="rmsnorm"),
    F.FusedRegion(epilogue=("relu",), reduction="layer_norm"),
    F.FusedRegion(epilogue=("gelu",), prologue=("relu",)),
]


@pytest.mark.skipif(not _HAVE_CC, reason="no C compiler (clang/cc/gcc) on host")
@pytest.mark.parametrize("region", _CHAINS, ids=lambda r: f"{r.epilogue}/{r.reduction}/{r.prologue}")
def test_x86_kernel_runs_and_matches_numpy(region):
    runner = get_runner("x86")
    rng = np.random.default_rng(0)
    A = rng.standard_normal((8, 12)).astype(np.float32)
    B = rng.standard_normal((12, 16)).astype(np.float32)
    bias = rng.standard_normal((16,)).astype(np.float32) if region.has_bias else None
    out, execution = runner.run_fused_region(region, A, B, bias)
    assert execution == "x86_native"  # a real compiled kernel ran on this box
    assert np.allclose(out, region.reference(A, B, bias), atol=1e-3)


@pytest.mark.skipif(not _HAVE_CC, reason="no C compiler (clang/cc/gcc) on host")
def test_x86_residual_path_matches_numpy():
    region = F.FusedRegion(epilogue=("gelu",), residual=True)
    rng = np.random.default_rng(1)
    A = rng.standard_normal((8, 12)).astype(np.float32)
    B = rng.standard_normal((12, 16)).astype(np.float32)
    R = rng.standard_normal((8, 16)).astype(np.float32)
    out, execution = get_runner("x86").run_fused_region(region, A, B, None, residual=R)
    assert execution == "x86_native"
    assert np.allclose(out, region.reference(A, B, None, R), atol=1e-3)
