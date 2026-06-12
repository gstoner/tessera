"""Phase E1 — the execution-derived Evaluator (docs/audit/compiler/EVALUATOR_PLAN.md).

Two layers:
  * portable contract tests for ``verdict_for`` — the anti-silent-fallback
    decision logic over the runtime signal, runnable everywhere;
  * a Darwin-gated integration run that (a) proves genuinely-native ops
    (matmul/gelu) reach HARDWARE_VERIFIED through real Metal execution, and
    (b) demonstrates the Evaluator's core value: ops that *look* routed but
    actually fall back to eager numpy are honestly classified as fallbacks,
    never promoted — closing the "registry models reality" gap that
    ``accelerator_proof``'s static envelope-membership classification leaves open.
"""

from __future__ import annotations

import sys

import numpy as np
import pytest

import tessera as ts
from tessera.compiler.evaluator import (
    BackendVerdict,
    HorizontalVerdict,
    Rung,
    _cross_path_relation,
    _horizontal_relation,
    cross_path_equivalence,
    evaluate,
    horizontal_equivalence,
    nvidia_emission_verdict,
    run_native,
    verdict_for,
)


# Module-level so @jit can inspect their source (nested defs can't be lowered).
def _mm(a, b):
    return ts.ops.matmul(a, b)


def _gelu(a):
    return ts.ops.gelu(a)


def _sm_axis(a):
    return ts.ops.softmax(a, axis=-1)


def _mm_gelu(a, b):                      # fused chain: gelu(matmul(a, b))
    return ts.ops.gelu(ts.ops.matmul(a, b))


def _msm(a, b, c):                       # fused chain: softmax(matmul)·matmul
    return ts.ops.matmul(ts.ops.softmax(ts.ops.matmul(a, b), axis=-1), c)


def _mm_aa(a, b):                        # deliberately wrong: a@a, not a@b
    return ts.ops.matmul(a, a)


_MM = ts.jit(target="apple_gpu")(_mm)
_GELU = ts.jit(target="apple_gpu")(_gelu)
_SM = ts.jit(target="apple_gpu")(_sm_axis)
_MM_GELU = ts.jit(target="apple_gpu")(_mm_gelu)
_MSM = ts.jit(target="apple_gpu")(_msm)
_MM_CPU = ts.jit(target="apple_cpu")(_mm)
_MM_AA_CPU = ts.jit(target="apple_cpu")(_mm_aa)


# ── portable: the verdict contract over the runtime signal ───────────────────

def test_native_success_plus_match_is_hardware_verified():
    v = verdict_for("apple_gpu", "native_gpu", "success", oracle_match=True)
    assert v.rung is Rung.HARDWARE_VERIFIED
    assert v.provenance_ok and v.correctness == "pass"


def test_native_success_plus_mismatch_is_miscompile_not_pass():
    """Ran on the real backend but disagreed → EXECUTES + fail, never rung 7.
    This is the bug the Evaluator exists to surface."""
    v = verdict_for("apple_gpu", "native_gpu", "success", oracle_match=False)
    assert v.correctness == "fail"
    assert v.rung is Rung.EXECUTES and v.rung < Rung.HARDWARE_VERIFIED


def test_silent_fallback_cannot_earn_execution_rung_even_when_correct():
    """``runtime_status='unimplemented'`` + eager fallback must NOT be promoted,
    regardless of numerical agreement: the demanded backend never ran. The honest
    floor is rung 1 (artifact_only), not rung 2 (which needs verifier evidence)."""
    v = verdict_for("apple_gpu", "fallback_eager", "unimplemented", oracle_match=True)
    assert not v.provenance_ok and v.is_silent_fallback
    assert v.correctness == "unproven"
    assert v.rung is Rung.ARTIFACT_ONLY and v.rung < Rung.EXECUTES


def test_reference_cpu_success_is_not_native_execution():
    """A reference executor returning a correct number still isn't the demanded
    native backend — provenance must fail."""
    v = verdict_for("cpu", "reference_cpu", "success", oracle_match=True)
    assert not v.provenance_ok and v.rung is Rung.ARTIFACT_ONLY


def test_native_success_without_reference_is_executes_unproven():
    v = verdict_for("apple_gpu", "native_gpu", "success", oracle_match=None)
    assert v.provenance_ok and v.correctness == "unproven"
    assert v.rung is Rung.EXECUTES


def test_invalid_artifact_is_not_provenance():
    v = verdict_for("apple_gpu", "native_gpu", "invalid_artifact", oracle_match=True)
    assert not v.provenance_ok and v.rung is Rung.ARTIFACT_ONLY


def test_artifact_only_backend_reports_rung1_not_overstated():
    """NVIDIA/ROCm emit a Target IR artifact and do not execute here. The
    Evaluator must report ARTIFACT_ONLY (rung 1) and refuse to overstate to
    lowers_clean/executes — portable (no toolchain / GPU needed)."""
    rng = np.random.default_rng(0)
    a = rng.standard_normal((16, 16)).astype(np.float32)
    b = rng.standard_normal((16, 16)).astype(np.float32)
    v = evaluate("rocm", ts.jit(target="rocm")(_mm), (a, b), a @ b)
    assert v.rung is Rung.ARTIFACT_ONLY, v.detail
    assert not v.provenance_ok and v.rung < Rung.EXECUTES


def test_rung_order_is_meaningful():
    assert (
        Rung.ARTIFACT_ONLY < Rung.EMITS_ASM_TEXT < Rung.ASSEMBLES
        < Rung.EXECUTES < Rung.HARDWARE_VERIFIED
    )


def test_nvidia_matmul_emits_wgmma_ptx_and_reports_emission_rung():
    """Wiring check (portable, no toolchain): @jit(target="nvidia_sm90") on a
    matmul attaches structurally-valid WGMMA PTX to the artifact, and the
    Evaluator reports EMITS_ASM_TEXT (rung 2.5) — above artifact_only, but never
    claiming execution."""
    fn = ts.jit(target="nvidia_sm90")(_mm)
    meta = fn.runtime_artifact().metadata
    assert "nvidia_ptx" in meta, "lowering did not attach emitted PTX"
    assert meta.get("nvidia_ptx_valid") is True
    assert "wgmma.mma_async.sync.aligned.m64n64k16" in meta["nvidia_ptx"]

    v = nvidia_emission_verdict(fn)
    assert v.rung is Rung.EMITS_ASM_TEXT
    assert Rung.ARTIFACT_ONLY < v.rung < Rung.EXECUTES
    assert not v.provenance_ok  # emission ≠ execution


def test_nvidia_non_matmul_does_not_overclaim_emission():
    """A non-matmul NVIDIA program emits no WGMMA PTX → stays ARTIFACT_ONLY."""
    fn = ts.jit(target="nvidia_sm90")(_sm_axis)
    v = nvidia_emission_verdict(fn)
    assert v.rung is Rung.ARTIFACT_ONLY


# ── DESIL: cross-path differential oracle ────────────────────────────────────

def test_cross_path_relation_classifier():
    assert _cross_path_relation(1, 0.0, tol=1e-3)[0] == "inconclusive"   # <2 paths
    assert _cross_path_relation(2, None, tol=1e-3)[0] == "inconclusive"  # incomparable
    assert _cross_path_relation(2, 1e-6, tol=1e-3)[0] == "equivalent"
    assert _cross_path_relation(3, 1e-1, tol=1e-3)[0] == "divergent"


@pytest.mark.skipif(sys.platform != "darwin", reason="needs apple_gpu + apple_cpu execution.")
def test_matmul_agrees_across_metal_and_accelerate():
    """The same matmul lowered through two independent compilers (Metal MPS vs
    Accelerate) must agree — reference-free cross-path differential."""
    rng = np.random.default_rng(20260612)
    a = rng.standard_normal((64, 64)).astype(np.float32)
    b = rng.standard_normal((64, 64)).astype(np.float32)
    v = cross_path_equivalence([("apple_gpu", _MM), ("apple_cpu", _MM_CPU)], (a, b))
    assert v.relation == "equivalent", v.detail
    assert set(v.paths) == {"apple_gpu", "apple_cpu"}


@pytest.mark.skipif(sys.platform != "darwin", reason="needs apple_gpu + apple_cpu execution.")
def test_cross_path_catches_a_divergent_lowering():
    """If one path computes something different, cross-path must flag divergence
    (proving the oracle has teeth) — here the cpu path computes a@a not a@b."""
    rng = np.random.default_rng(3)
    a = rng.standard_normal((64, 64)).astype(np.float32)
    b = rng.standard_normal((64, 64)).astype(np.float32)
    v = cross_path_equivalence([("apple_gpu", _MM), ("apple_cpu", _MM_AA_CPU)], (a, b))
    assert v.is_divergent, v.detail


# ── Darwin-gated: derive verdicts from real Metal runs ───────────────────────

@pytest.mark.skipif(
    sys.platform != "darwin",
    reason="Apple GPU native execution is Darwin-only; the verdict contract "
    "above is exercised portably.",
)
def test_native_ops_reach_hardware_verified_on_darwin():
    """matmul and gelu genuinely execute on Metal (native_gpu/success) — the
    execution-derived rung-7 path, vs accelerator_proof's static `proven`."""
    rng = np.random.default_rng(20260611)
    a = rng.standard_normal((16, 16)).astype(np.float32)
    b = rng.standard_normal((16, 16)).astype(np.float32)

    vm = evaluate("apple_gpu", _MM, (a, b), a @ b, rtol=2e-3, atol=1e-4)
    assert vm.rung is Rung.HARDWARE_VERIFIED, vm.detail
    assert vm.execution_kind == "native_gpu" and vm.runtime_status == "success"

    gelu_ref = 0.5 * a * (1.0 + np.tanh(0.7978845608 * (a + 0.044715 * a**3)))
    vg = evaluate("apple_gpu", _GELU, (a,), gelu_ref, rtol=3e-3, atol=1e-3)
    assert vg.rung is Rung.HARDWARE_VERIFIED, vg.detail


@pytest.mark.skipif(
    sys.platform != "darwin",
    reason="Apple GPU native execution is Darwin-only.",
)
def test_fallback_lane_classified_honestly_no_silent_miscompile():
    """Run the existing differential lane through the Evaluator. Its ops fall
    back to eager numpy (the registry-vs-reality gap: envelope-adjacent ops that
    don't actually execute natively standalone). The Evaluator must (a) never
    report a native-execution-disagrees miscompile, and (b) classify every
    fallback honestly as LOWERS_CLEAN / silent-fallback — never a green rung."""
    from _diff_lane import numeric_cases  # noqa: E402 (sibling helper)

    nrng = np.random.default_rng(20260611)
    verdicts: list[BackendVerdict] = []
    for label, fn, args, oracle, exact in numeric_cases(nrng):
        v = evaluate("apple_gpu", fn, args, oracle, rtol=2e-3, atol=1e-4, exact=exact)
        verdicts.append(v)
        # (a) the load-bearing invariant: native execution implies correctness.
        assert not (v.provenance_ok and v.correctness == "fail"), (
            f"miscompile candidate on {label!r}: ran natively "
            f"(kind={v.execution_kind!r}) but disagreed with the numpy oracle "
            f"— {v.detail}"
        )
        # (b) any non-native verdict is an honest, un-promotable artifact.
        if not v.provenance_ok:
            assert v.rung is Rung.ARTIFACT_ONLY and v.is_silent_fallback

    # The Evaluator is genuinely distinguishing native from fallback — at least
    # one of these envelope-adjacent ops is shown to NOT execute natively (the
    # gap accelerator_proof's static `proven` would miss). If a future change
    # wires them all to Metal, update this to assert HARDWARE_VERIFIED instead.
    assert any(v.is_silent_fallback for v in verdicts), (
        "expected the differential lane to expose at least one envelope-adjacent "
        "op that falls back rather than executing natively"
    )


# ── E1b: horizontal-equivalence oracle (portable classifier) ─────────────────

def test_horizontal_equivalent_when_within_tol():
    rel, _ = _horizontal_relation(True, True, 1e-6, tol=1e-3)
    assert rel == "equivalent"


def test_horizontal_divergent_when_above_tol():
    rel, _ = _horizontal_relation(True, True, 1e-1, tol=1e-3)
    assert rel == "divergent"


def test_horizontal_inconclusive_when_a_side_not_native():
    assert _horizontal_relation(True, False, 0.0, tol=1e-3)[0] == "inconclusive"
    assert _horizontal_relation(False, True, 0.0, tol=1e-3)[0] == "inconclusive"


def test_horizontal_inconclusive_when_incomparable():
    assert _horizontal_relation(True, True, None, tol=1e-3)[0] == "inconclusive"


# ── E1b: horizontal oracle over real Metal fusion ────────────────────────────

@pytest.mark.skipif(sys.platform != "darwin", reason="Metal fusion is Darwin-only.")
def test_fused_chains_are_equivalent_to_their_unfused_composition():
    """The PolyJuice self-consistency check: a fused chain run as one kernel must
    equal the same math composed from separately-executed native ops, on the
    SAME backend. Isolates the fusion rewrite — no external reference needed."""
    rng = np.random.default_rng(20260611)
    a = rng.standard_normal((16, 16)).astype(np.float32)
    b = rng.standard_normal((16, 16)).astype(np.float32)
    c = rng.standard_normal((16, 16)).astype(np.float32)

    def unfused_mm_gelu(args):
        x, y = args
        mm_out, n1 = run_native("apple_gpu", _MM, (x, y))
        if not n1:
            return None, False
        g_out, n2 = run_native("apple_gpu", _GELU, (mm_out,))
        return g_out, (n1 and n2)

    v = horizontal_equivalence("apple_gpu", _MM_GELU, (a, b), unfused_mm_gelu, rtol=3e-3, atol=1e-3)
    assert v.relation == "equivalent", v.detail

    def unfused_msm(args):
        x, y, z = args
        s0, n1 = run_native("apple_gpu", _MM, (x, y))
        if not n1:
            return None, False
        sm, n2 = run_native("apple_gpu", _SM, (s0,))
        if not n2:
            return None, False
        out, n3 = run_native("apple_gpu", _MM, (sm, z))
        return out, (n1 and n2 and n3)

    vm = horizontal_equivalence("apple_gpu", _MSM, (a, b, c), unfused_msm, rtol=3e-3, atol=1e-3)
    assert vm.relation in ("equivalent", "inconclusive"), vm.detail


@pytest.mark.skipif(sys.platform != "darwin", reason="Metal fusion is Darwin-only.")
def test_horizontal_oracle_catches_a_divergent_unfused():
    """Prove the oracle actually flags divergence: a deliberately-perturbed (but
    'native') unfused operand must be reported divergent, not waved through."""
    rng = np.random.default_rng(7)
    a = rng.standard_normal((16, 16)).astype(np.float32)
    b = rng.standard_normal((16, 16)).astype(np.float32)

    def unfused_wrong(args):
        x, y = args
        mm_out, n1 = run_native("apple_gpu", _MM, (x, y))
        if not n1:
            return None, False
        g_out, n2 = run_native("apple_gpu", _GELU, (mm_out,))
        if not n2:
            return None, False
        return np.asarray(g_out) + 0.5, True   # deliberately wrong, still "native"

    v = horizontal_equivalence("apple_gpu", _MM_GELU, (a, b), unfused_wrong, rtol=3e-3, atol=1e-3)
    assert v.is_divergent, v.detail
    assert isinstance(v, HorizontalVerdict)
