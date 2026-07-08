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


def test_emit_accepts_dynamic_spec():
    # DYNAMIC is supported (Workstream G / W2): the runtime-arg kernel is
    # dims-invariant, so DYNAMIC emits the same source as BUCKET — one compiled
    # kernel serves every shape (see test_dynamic_shape_emit.py for the full proof).
    e = get_emitter("x86")
    region = F.FusedRegion(epilogue=("relu",))
    dyn = e.emit(region, spec=SpecPolicy.DYNAMIC)
    assert dyn.spec is SpecPolicy.DYNAMIC
    assert dyn.source == e.emit(region, spec=SpecPolicy.BUCKET).source


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


def test_x86_missing_required_buffer_declines_not_segfault():
    # A residual/bias region invoked WITHOUT the required buffer must NOT launch
    # the kernel — the emitted C dereferences residual[...] / bias[n], so a null
    # would SIGSEGV past Python's except. The runner routes through the reference,
    # which raises a clean ValueError. Run in a CHILD process so a regression
    # (segfault) surfaces as a failed assert, not a crashed test session.
    import subprocess
    import sys
    import textwrap
    code = textwrap.dedent(
        """
        import numpy as np
        import tessera.compiler.fusion as F
        import tessera.compiler.emit.x86_llvm as x86
        r = x86.X86CRunner()
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


# ── D1 candidates + C1b AOCL-DLP (host-free) ──────────────────────────────────
#
# The generic C lane is the x86 Tier-1 candidate; AOCL-DLP (C1b) is the opt-in
# Tier-3 candidate the arbiter measures. On a host without a wired aocl-dlp
# install the Tier-3 lane is arbiter-visible but unavailable, so it never
# mis-selects — arbitration falls to the generic lane.

def test_x86_candidates_registered_with_tiers():
    from tessera.compiler.emit.candidate import (
        OP_FUSED_REGION, Tier, candidates_for,
    )
    cands = {c.name: c for c in candidates_for("x86", OP_FUSED_REGION)}
    assert cands["x86_generic_c"].tier is Tier.SYNTHESIZED
    assert cands["x86_aocl_dlp"].tier is Tier.HAND_TUNED


def test_aocl_dlp_unavailable_without_wired_install(monkeypatch):
    # No env → library absent → candidate declines out of arbitration.
    from tessera.compiler.emit.candidate import (
        OP_FUSED_REGION, candidates_for,
    )
    monkeypatch.delenv("TESSERA_AOCL_DLP_LIB", raising=False)
    monkeypatch.delenv("TESSERA_AOCL_DLP_SGEMM", raising=False)
    aocl = {c.name: c for c in
            candidates_for("x86", OP_FUSED_REGION)}["x86_aocl_dlp"]
    assert aocl.available() is False
    # applies_to is independent of availability (matmul-bound epilogue envelope):
    assert aocl.applies_to(F.FusedRegion(epilogue=("bias", "gelu")))
    assert not aocl.applies_to(F.FusedRegion(epilogue=("bias",),
                                             reduction="softmax"))


def test_aocl_dlp_declines_honestly_when_unwired():
    # Even if forced to run(), an unwired AOCL-DLP lane returns the reference —
    # never a mislabeled kernel (Decision #21).
    from tessera.compiler.emit.candidate import (
        OP_FUSED_REGION, candidates_for,
    )
    aocl = {c.name: c for c in
            candidates_for("x86", OP_FUSED_REGION)}["x86_aocl_dlp"]
    region = F.FusedRegion(epilogue=("bias", "relu"))
    A = np.zeros((8, 12), np.float32)
    B = np.zeros((12, 16), np.float32)
    bias = np.zeros((16,), np.float32)
    _, tag = aocl.run(region, A, B, bias)
    assert tag == "reference"


def test_force_aocl_dlp_raises_when_unavailable():
    from tessera.compiler.emit.candidate import ArbiterError, OP_FUSED_REGION, arbitrate
    with pytest.raises(ArbiterError, match="x86_aocl_dlp"):
        arbitrate(F.FusedRegion(epilogue=("bias", "gelu")),
                  OP_FUSED_REGION, "x86", force="x86_aocl_dlp")


@pytest.mark.skipif(not _HAVE_CC, reason="no C compiler (clang/cc/gcc) on host")
def test_x86_arbiter_falls_to_generic_when_aocl_absent():
    # With AOCL-DLP unavailable, the arbiter runs the generic C lane on Zen — the
    # crown-jewel slot is empty, the floor still executes and F4-gates.
    from tessera.compiler.emit.candidate import OP_FUSED_REGION, run_arbitrated
    region = F.FusedRegion(epilogue=("bias", "gelu"))
    rng = np.random.default_rng(0)
    A = rng.standard_normal((8, 12)).astype(np.float32)
    B = rng.standard_normal((12, 16)).astype(np.float32)
    bias = rng.standard_normal((16,)).astype(np.float32)
    out, tag = run_arbitrated(region, OP_FUSED_REGION, "x86", A, B, bias)
    assert tag == "x86_native"
    assert np.allclose(out, region.reference(A, B, bias), atol=1e-3)
