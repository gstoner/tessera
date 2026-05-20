"""Tests for U3 — lane-specific next-action context in diagnostics.

Locks the contract that every documented lane-rejection code maps
to an actionable next-action hint with a stable code.  When
``@clifford_jit`` rejects a function for using ``np.dot``, the
developer sees a *fix*, not just a rejection.
"""

from __future__ import annotations

import pytest

from tessera.compiler import (
    ConstrainedDiagnosticCode,
    Diagnostic,
    FrontendDiagnosticCode,
    JitDiagnosticCode,
)
from tessera.compiler.explain import (
    NEXT_FIX_TEXTUAL_SYNTAX,
    NEXT_PROVIDE_SOURCE,
    NEXT_REWRITE_HOLOMORPHIC,
    NEXT_REWRITE_TO_GA_OPS,
    NEXT_USE_APPLE_GPU_TARGET,
    NEXT_USE_ENERGY_WHITELIST,
    NEXT_USE_FP32_FOR_ENERGY,
    NextAction,
    next_action_for_diagnostic,
)


@pytest.mark.parametrize(
    "code,expected_next_code",
    [
        (
            ConstrainedDiagnosticCode.CLIFFORD_OP_NOT_WHITELISTED,
            NEXT_REWRITE_TO_GA_OPS,
        ),
        (
            ConstrainedDiagnosticCode.CLIFFORD_EMPTY_OP_PLAN,
            NEXT_REWRITE_TO_GA_OPS,
        ),
        (
            ConstrainedDiagnosticCode.CLIFFORD_TARGET_MISMATCH,
            NEXT_USE_APPLE_GPU_TARGET,
        ),
        (
            ConstrainedDiagnosticCode.CLIFFORD_UNSUPPORTED_TARGET,
            NEXT_USE_APPLE_GPU_TARGET,
        ),
        (
            ConstrainedDiagnosticCode.COMPLEX_NON_HOLOMORPHIC,
            NEXT_REWRITE_HOLOMORPHIC,
        ),
        (
            ConstrainedDiagnosticCode.COMPLEX_CR_RESIDUAL_TOO_LARGE,
            NEXT_REWRITE_HOLOMORPHIC,
        ),
        (
            ConstrainedDiagnosticCode.ENERGY_FORBIDDEN_OP,
            NEXT_USE_ENERGY_WHITELIST,
        ),
        (
            ConstrainedDiagnosticCode.ENERGY_UNSUPPORTED_DTYPE,
            NEXT_USE_FP32_FOR_ENERGY,
        ),
        (
            FrontendDiagnosticCode.PARSE_UNEXPECTED_TOKEN,
            NEXT_FIX_TEXTUAL_SYNTAX,
        ),
        (
            FrontendDiagnosticCode.PARSE_UNEXPECTED_EOF,
            NEXT_FIX_TEXTUAL_SYNTAX,
        ),
        (
            JitDiagnosticCode.SOURCE_UNAVAILABLE,
            NEXT_PROVIDE_SOURCE,
        ),
    ],
)
def test_diagnostic_code_maps_to_targeted_next_action(
    code, expected_next_code: str,
) -> None:
    """Each documented lane-rejection code produces a stable
    next-action ID + non-empty prescription text."""

    d = Diagnostic(severity="error", code=code, message="m")
    hint = next_action_for_diagnostic(d)
    assert hint is not None, (
        f"diagnostic code {code.value!r} has no registered "
        f"next-action — add it to "
        f"explain._DIAGNOSTIC_CODE_NEXT_ACTIONS"
    )
    assert isinstance(hint, NextAction)
    assert hint.code == expected_next_code
    assert hint.message  # non-empty prescription


def test_info_level_codes_have_no_next_action() -> None:
    """``JIT_COMPILED_CPU`` is an info-level diagnostic; it
    doesn't need a fix, so the lookup returns ``None``."""

    d = Diagnostic(
        severity="info",
        code=JitDiagnosticCode.COMPILED_CPU,
        message="m",
    )
    assert next_action_for_diagnostic(d) is None


def test_unknown_code_returns_none() -> None:
    d = Diagnostic(severity="warning", code="SOME_UNKNOWN_CODE", message="m")
    assert next_action_for_diagnostic(d) is None


class TestExplainEndToEnd:
    """Wire a lane-rejection diagnostic into a fake JitFn and
    verify the produced ``.next_actions`` list includes the
    targeted hint."""

    def test_clifford_rejection_surfaces_rewrite_action(self) -> None:
        from tessera.compiler.explain import _build_next_actions

        class _FakeJitFn:
            execution_kind = "reference_cpu"
            target = "cpu"
            last_fallback_reason = None

        clifford_reject = Diagnostic.from_constrained(
            code=ConstrainedDiagnosticCode.CLIFFORD_OP_NOT_WHITELISTED,
            message="function used np.dot",
            lane="clifford_jit",
        )
        actions = _build_next_actions(
            _FakeJitFn(),  # type: ignore[arg-type]
            [clifford_reject],
        )
        codes = [a.code for a in actions]
        assert NEXT_REWRITE_TO_GA_OPS in codes

    def test_duplicate_codes_deduplicated(self) -> None:
        """Two diagnostics with the same code shouldn't produce two
        identical next-action hints."""

        from tessera.compiler.explain import _build_next_actions

        class _FakeJitFn:
            execution_kind = "reference_cpu"
            target = "cpu"
            last_fallback_reason = None

        d1 = Diagnostic.from_constrained(
            code=ConstrainedDiagnosticCode.COMPLEX_NON_HOLOMORPHIC,
            message="op A",
            lane="complex_jit",
        )
        d2 = Diagnostic.from_constrained(
            code=ConstrainedDiagnosticCode.COMPLEX_NON_HOLOMORPHIC,
            message="op B",
            lane="complex_jit",
        )
        actions = _build_next_actions(
            _FakeJitFn(),  # type: ignore[arg-type]
            [d1, d2],
        )
        # The same hint code appears at most once.
        codes = [a.code for a in actions]
        assert codes.count(NEXT_REWRITE_HOLOMORPHIC) == 1
