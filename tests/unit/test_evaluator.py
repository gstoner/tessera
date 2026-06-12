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
    Rung,
    evaluate,
    verdict_for,
)


# Module-level so @jit can inspect their source (nested defs can't be lowered).
def _mm(a, b):
    return ts.ops.matmul(a, b)


def _gelu(a):
    return ts.ops.gelu(a)


_MM = ts.jit(target="apple_gpu")(_mm)
_GELU = ts.jit(target="apple_gpu")(_gelu)


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
    regardless of numerical agreement: the demanded backend never ran."""
    v = verdict_for("apple_gpu", "fallback_eager", "unimplemented", oracle_match=True)
    assert not v.provenance_ok and v.is_silent_fallback
    assert v.correctness == "unproven"
    assert v.rung is Rung.LOWERS_CLEAN and v.rung < Rung.EXECUTES


def test_reference_cpu_success_is_not_native_execution():
    """A reference executor returning a correct number still isn't the demanded
    native backend — provenance must fail."""
    v = verdict_for("cpu", "reference_cpu", "success", oracle_match=True)
    assert not v.provenance_ok and v.rung is Rung.LOWERS_CLEAN


def test_native_success_without_reference_is_executes_unproven():
    v = verdict_for("apple_gpu", "native_gpu", "success", oracle_match=None)
    assert v.provenance_ok and v.correctness == "unproven"
    assert v.rung is Rung.EXECUTES


def test_invalid_artifact_is_not_provenance():
    v = verdict_for("apple_gpu", "native_gpu", "invalid_artifact", oracle_match=True)
    assert not v.provenance_ok and v.rung is Rung.LOWERS_CLEAN


def test_rung_order_is_meaningful():
    assert Rung.ARTIFACT_ONLY < Rung.EXECUTES < Rung.HARDWARE_VERIFIED
    assert int(Rung.HARDWARE_VERIFIED) == 7


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
        # (b) any non-native verdict is an honest, un-promotable fallback.
        if not v.provenance_ok:
            assert v.rung is Rung.LOWERS_CLEAN and v.is_silent_fallback

    # The Evaluator is genuinely distinguishing native from fallback — at least
    # one of these envelope-adjacent ops is shown to NOT execute natively (the
    # gap accelerator_proof's static `proven` would miss). If a future change
    # wires them all to Metal, update this to assert HARDWARE_VERIFIED instead.
    assert any(v.is_silent_fallback for v in verdicts), (
        "expected the differential lane to expose at least one envelope-adjacent "
        "op that falls back rather than executing natively"
    )
