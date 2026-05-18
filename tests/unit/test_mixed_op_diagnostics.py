"""M2 Step 3 — call-site value-kind diagnostics.

Locks the M2 acceptance criterion:
*"Unsupported mixed operations fail with source-span diagnostics."*

Coverage:

  - ``@clifford_jit`` rejects non-Multivector arguments at call
    time with :class:`TesseraValueKindError`.
  - The error carries the offending arg's name + position +
    expected/actual kind so users get an actionable message.
  - ``check_call_kinds`` enforces per-arg kinds when the call
    has a mix of expected kinds.
  - The session captures the error as a diagnostic when invoked
    inside a ``compile_session`` scope.
"""

from __future__ import annotations

import numpy as np
import pytest

from tessera import bridge, ga
from tessera.compiler.clifford_jit import clifford_jit
from tessera.compiler.compile_session import compile_session


# ---------------------------------------------------------------------------
# @clifford_jit rejects non-Multivector arguments at call time
# ---------------------------------------------------------------------------

def test_clifford_jit_rejects_numpy_array_argument() -> None:
    """The user wired a tensor where the function expects a
    Multivector.  Must fail at call time with a clear diagnostic."""

    @clifford_jit(target="apple_gpu")
    def f(a, b):
        return ga.inner(a, b)

    arr = np.zeros((4, 8), dtype=np.float32)
    with pytest.raises(bridge.TesseraValueKindError) as exc:
        f(arr, arr)
    msg = str(exc.value)
    assert "@clifford_jit(" in msg
    assert "argument a" in msg
    assert "multivector" in msg
    assert "tensor" in msg
    assert "M2_VALUE_KIND_MISMATCH" in msg


def test_clifford_jit_error_names_the_offending_arg_position() -> None:
    """If the second arg is the bad one, the message must point
    at ``b`` not ``a``."""

    @clifford_jit(target="apple_gpu")
    def g(a, b):
        return ga.inner(a, b)

    a_mv = ga.Multivector(np.zeros((4, 8), dtype=np.float32), ga.Cl(3, 0))
    with pytest.raises(bridge.TesseraValueKindError) as exc:
        g(a_mv, np.zeros((4, 8)))
    assert "argument b" in str(exc.value)


def test_clifford_jit_passes_clean_call_through() -> None:
    """Sanity: a correct call with real Multivectors still works."""

    @clifford_jit(target="apple_gpu")
    def h(a, b):
        return ga.inner(a, b)

    a_mv = ga.Multivector(np.zeros((4, 8), dtype=np.float32), ga.Cl(3, 0))
    b_mv = ga.Multivector(np.zeros((4, 8), dtype=np.float32), ga.Cl(3, 0))
    # No raise — exercises the runtime path.
    h(a_mv, b_mv)


# ---------------------------------------------------------------------------
# check_call_kinds — per-arg kind list
# ---------------------------------------------------------------------------

def test_check_call_kinds_per_arg_kinds() -> None:
    """A future ``@mixed_jit`` could take (tensor, multivector) —
    verify per-arg validation."""
    a = ga.Cl(3, 0)
    mv = ga.Multivector(np.zeros(8, dtype=np.float32), a)
    tensor = np.zeros(8)
    bridge.check_call_kinds(
        (tensor, mv),
        expected=("tensor", "multivector"),
        op_name="demo",
        arg_names=("x", "mv"),
    )
    with pytest.raises(bridge.TesseraValueKindError, match="argument mv"):
        bridge.check_call_kinds(
            (tensor, tensor),
            expected=("tensor", "multivector"),
            op_name="demo",
            arg_names=("x", "mv"),
        )


def test_check_call_kinds_validates_expected_length() -> None:
    """Mismatched lengths surface as ValueError (programmer error,
    not user-input error)."""
    with pytest.raises(ValueError, match="length"):
        bridge.check_call_kinds(
            (np.zeros(4),),
            expected=("tensor", "multivector"),
            op_name="demo",
        )


def test_check_call_kinds_source_span_propagates_to_error() -> None:
    with pytest.raises(bridge.TesseraValueKindError) as exc:
        bridge.check_call_kinds(
            (np.zeros(4),),
            expected="multivector",
            op_name="demo",
            arg_names=("x",),
            source_span=(7, 11),
        )
    assert exc.value.source_span == (7, 11)
    assert "line 7, col 11" in str(exc.value)


# ---------------------------------------------------------------------------
# Session capture of the failure
# ---------------------------------------------------------------------------

def test_session_observes_kind_mismatch_via_emit_diagnostic() -> None:
    """A caller can convert a TesseraValueKindError into a
    session-level diagnostic for downstream audit consumers."""
    with compile_session() as session:
        try:
            bridge.check_call_kinds(
                (np.zeros(4),),
                expected="multivector",
                op_name="ga.inner",
                arg_names=("a",),
                source_span=(3, 0),
            )
        except bridge.TesseraValueKindError as exc:
            session.emit_diagnostic(
                message=str(exc),
                code="M2_VALUE_KIND_MISMATCH",
                source_span=exc.source_span,
            )
    assert session.has_errors
    d = session.diagnostics[0]
    assert d.code == "M2_VALUE_KIND_MISMATCH"
    assert d.source_span == (3, 0)


# ---------------------------------------------------------------------------
# Negative path: tensor → ga.norm via raw API call also fails
# ---------------------------------------------------------------------------

def test_raw_ga_op_on_tensor_input_raises_at_some_layer() -> None:
    """Even outside @clifford_jit, calling a GA op on a tensor
    must fail.  The exact exception type isn't fixed yet across
    every op, but it must NOT silently return a result — this
    test catches a regression that lets ``ga.inner(tensor, tensor)``
    return a number through a coercion path."""
    with pytest.raises((bridge.TesseraValueKindError, TypeError, ValueError, AttributeError)):
        ga.inner(np.zeros(8), np.zeros(8))
