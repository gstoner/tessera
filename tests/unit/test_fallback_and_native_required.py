"""M3 — `native_required=True` + stable fallback reasons.

Locks the M3 acceptance criteria:

  1. ``FallbackReason`` is a closed, stable enum.
  2. Every enum member has a canonical, human-readable message.
  3. ``TesseraNativeRequiredError`` carries the enum code so callers
     can match without parsing free-form text.
  4. ``@clifford_jit(target="apple_gpu", native_required=True)``
     raises before runtime dispatch when the host can't actually
     run native MSL.
  5. The two canonical drivers accept ``native_required=True`` and
     raise the same error class on non-Darwin / runtime-unavailable
     hosts.
  6. ``CompileReport.fallback_reason`` accepts both the enum and
     legacy free-text strings, and serializes the enum to its
     ``.value`` for stable JSON round-trips.
"""

from __future__ import annotations

import json
import sys

import numpy as np
import pytest

from tessera import ga
from tessera.compiler import compile_report as cr
from tessera.compiler.canonical import (
    matmul_softmax_matmul,
    rotor_sandwich_norm,
)
from tessera.compiler.clifford_jit import (
    CliffordCompiledCallable,
    clifford_jit,
)
from tessera.compiler.fallback import (
    FallbackDecision,
    FallbackReason,
    TesseraNativeRequiredError,
    classify_host,
    message_for,
)


# ---------------------------------------------------------------------------
# Enum contract
# ---------------------------------------------------------------------------

def test_fallback_reason_is_closed_enum() -> None:
    """Adding a new fallback reason is a deliberate decision — this
    test catches accidental drift."""
    assert {r.value for r in FallbackReason} == {
        "non_darwin_host",
        "apple_gpu_runtime_unavailable",
        "manifest_miss",
        "dtype_not_covered",
        "shape_out_of_envelope",
        "capability_not_ready",
        "opt_out",
        "reference_forced",
    }


def test_every_reason_has_a_canonical_message() -> None:
    """Every enum member must have a message — otherwise
    ``message_for`` would KeyError at runtime."""
    for reason in FallbackReason:
        msg = message_for(reason)
        assert msg
        # Round-trip via the method form too.
        assert reason.message() == msg


def test_fallback_reason_is_string_compatible() -> None:
    """Members are ``str`` subclasses so JSON serialization is
    a no-op (``json.dumps(FallbackReason.OPT_OUT)`` works directly)."""
    assert FallbackReason.OPT_OUT == "opt_out"
    assert json.dumps(FallbackReason.OPT_OUT.value) == '"opt_out"'


# ---------------------------------------------------------------------------
# TesseraNativeRequiredError carries structured fields
# ---------------------------------------------------------------------------

def test_error_carries_reason_code() -> None:
    exc = TesseraNativeRequiredError(
        FallbackReason.NON_DARWIN_HOST,
        target="apple_gpu", op_name="demo",
    )
    assert exc.reason == FallbackReason.NON_DARWIN_HOST
    assert exc.target == "apple_gpu"
    assert exc.op_name == "demo"
    assert "code=non_darwin_host" in str(exc)


def test_error_caller_can_match_on_reason() -> None:
    """Callers should match on the enum, not on the text — this is
    the M3 motivation."""
    try:
        raise TesseraNativeRequiredError(FallbackReason.MANIFEST_MISS)
    except TesseraNativeRequiredError as e:
        assert e.reason == FallbackReason.MANIFEST_MISS


# ---------------------------------------------------------------------------
# classify_host helper
# ---------------------------------------------------------------------------

def test_classify_host_non_darwin() -> None:
    assert classify_host(
        is_darwin=False, runtime_available=False,
    ) == FallbackReason.NON_DARWIN_HOST
    # is_darwin trumps runtime_available — non-Darwin hosts never
    # have an Apple GPU runtime to be unavailable in.
    assert classify_host(
        is_darwin=False, runtime_available=True,
    ) == FallbackReason.NON_DARWIN_HOST


def test_classify_host_darwin_with_no_runtime() -> None:
    assert classify_host(
        is_darwin=True, runtime_available=False,
    ) == FallbackReason.APPLE_GPU_RUNTIME_UNAVAILABLE


def test_classify_host_darwin_with_runtime() -> None:
    assert classify_host(is_darwin=True, runtime_available=True) is None


def test_fallback_decision_raises_when_required() -> None:
    decision = FallbackDecision(
        reason=FallbackReason.NON_DARWIN_HOST, native_required=True,
    )
    with pytest.raises(TesseraNativeRequiredError) as exc:
        decision.raise_if_required(target="apple_gpu", op_name="demo")
    assert exc.value.reason == FallbackReason.NON_DARWIN_HOST


def test_fallback_decision_silent_when_not_required() -> None:
    decision = FallbackDecision(
        reason=FallbackReason.NON_DARWIN_HOST, native_required=False,
    )
    decision.raise_if_required()  # no raise


# ---------------------------------------------------------------------------
# CompileReport accepts both enum and legacy free-text reasons
# ---------------------------------------------------------------------------

def test_compile_report_accepts_enum_fallback_reason() -> None:
    r = cr.CompileReport(
        program_id="x", source="t.run", frontend="tessera.jit",
        value_kind="tensor", target="cpu",
        fallback_reason=FallbackReason.NON_DARWIN_HOST,
    )
    assert r.fallback_reason == FallbackReason.NON_DARWIN_HOST
    # The dict view serializes the enum to its `.value` so JSON
    # round-trips without leaking Python class names.
    d = r.as_dict()
    assert d["fallback_reason"] == "non_darwin_host"
    blob = r.as_json()
    reloaded = json.loads(blob)
    assert reloaded["fallback_reason"] == "non_darwin_host"


def test_compile_report_still_accepts_legacy_string_fallback() -> None:
    """Backwards-compatible: drivers that haven't migrated yet can
    still pass a free-form string and the report stays JSON-able."""
    r = cr.CompileReport(
        program_id="x", source="t.run", frontend="tessera.jit",
        value_kind="tensor", target="cpu",
        fallback_reason="custom free-text reason",
    )
    assert r.as_dict()["fallback_reason"] == "custom free-text reason"


# ---------------------------------------------------------------------------
# @clifford_jit native_required option
# ---------------------------------------------------------------------------

def test_clifford_jit_accepts_native_required() -> None:
    """The decorator surface gains a ``native_required`` keyword."""
    @clifford_jit(target="apple_gpu", native_required=False)
    def f(a, b):
        return ga.inner(a, b)
    assert isinstance(f, CliffordCompiledCallable)


@pytest.mark.skipif(
    sys.platform == "darwin",
    reason="Test exercises the non-Darwin failure path",
)
def test_clifford_jit_native_required_raises_on_non_darwin() -> None:
    """On a non-Darwin host, ``native_required=True`` must raise
    rather than fall back."""

    @clifford_jit(target="apple_gpu", native_required=True)
    def f(rotor, points):
        return ga.norm(ga.rotor_sandwich(rotor, points))

    a = ga.Cl(3, 0)
    rng = np.random.RandomState(0)
    R = rng.randn(4, 8).astype(np.float32) * 0.3
    V = rng.randn(4, 8).astype(np.float32)
    with pytest.raises(TesseraNativeRequiredError) as exc:
        f(ga.Multivector(R, a), ga.Multivector(V, a))
    assert exc.value.reason == FallbackReason.NON_DARWIN_HOST


@pytest.mark.skipif(
    sys.platform != "darwin",
    reason="Apple GPU runtime only on Darwin",
)
def test_clifford_jit_native_required_runs_on_darwin() -> None:
    """On Darwin with the runtime available, ``native_required=True``
    is a no-op — the function runs as normal."""

    @clifford_jit(target="apple_gpu", native_required=True)
    def f(rotor, points):
        return ga.norm(ga.rotor_sandwich(rotor, points))

    a = ga.Cl(3, 0)
    rng = np.random.RandomState(0)
    R = rng.randn(4, 8).astype(np.float32) * 0.3
    V = rng.randn(4, 8).astype(np.float32)
    out = f(ga.Multivector(R, a), ga.Multivector(V, a))
    assert out is not None


# ---------------------------------------------------------------------------
# Canonical drivers honor native_required
# ---------------------------------------------------------------------------

def test_rotor_canonical_driver_default_falls_back_on_non_darwin() -> None:
    """Default (``native_required=False``) keeps M1 behavior — even
    on non-Darwin the driver returns a report instead of raising."""
    report = rotor_sandwich_norm.run()
    if sys.platform != "darwin":
        assert report.fallback_reason == FallbackReason.NON_DARWIN_HOST
        # Reason serializes cleanly through JSON.
        assert report.as_dict()["fallback_reason"] == "non_darwin_host"


@pytest.mark.skipif(
    sys.platform == "darwin",
    reason="Test exercises the non-Darwin failure path",
)
def test_rotor_canonical_driver_native_required_raises_on_non_darwin() -> None:
    with pytest.raises(TesseraNativeRequiredError) as exc:
        rotor_sandwich_norm.run(native_required=True)
    assert exc.value.reason == FallbackReason.NON_DARWIN_HOST
    assert "rotor_sandwich_norm" in str(exc.value)


@pytest.mark.skipif(
    sys.platform == "darwin",
    reason="Test exercises the non-Darwin failure path",
)
def test_matmul_canonical_driver_native_required_raises_on_non_darwin() -> None:
    with pytest.raises(TesseraNativeRequiredError) as exc:
        matmul_softmax_matmul.run(native_required=True)
    assert exc.value.reason == FallbackReason.NON_DARWIN_HOST


@pytest.mark.skipif(
    sys.platform != "darwin",
    reason="Apple GPU runtime only on Darwin",
)
def test_canonical_drivers_native_required_run_on_darwin() -> None:
    """On Darwin both canonical drivers complete with
    ``native_required=True`` and emit reports with no fallback."""
    r1 = rotor_sandwich_norm.run(native_required=True)
    assert r1.fallback_reason is None
    r2 = matmul_softmax_matmul.run(native_required=True)
    assert r2.fallback_reason is None


def test_matmul_canonical_driver_default_falls_back_on_non_darwin() -> None:
    report = matmul_softmax_matmul.run()
    if sys.platform != "darwin":
        assert report.fallback_reason == FallbackReason.NON_DARWIN_HOST


# ---------------------------------------------------------------------------
# Diagnostic stability — the error message must include both the
# enum code and the scope so debugging tooling can grep for it.
# ---------------------------------------------------------------------------

def test_error_diagnostic_includes_code_and_scope() -> None:
    exc = TesseraNativeRequiredError(
        FallbackReason.SHAPE_OUT_OF_ENVELOPE,
        target="apple_gpu", op_name="matmul",
        detail="K=16384 exceeds tiled envelope (K ≤ 8192)",
    )
    text = str(exc)
    assert "[code=shape_out_of_envelope]" in text
    assert "[scope=apple_gpu/matmul]" in text
    assert "K=16384" in text
