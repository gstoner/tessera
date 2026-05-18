"""M2 Step 2 — ``tessera.bridge`` boundary ops + Decision #15a.

Locks:

  - The boundary ops accept only the right sibling kind and raise
    :class:`TesseraValueKindError` for anything else.
  - Round-trip identity: ``tensor_to_multivector(multivector_to_tensor(mv))
    ≈ mv``; same for complex.
  - ``value_kind_of`` is a **function** (Decision #15a) — not a
    method on the operand.  This is a structural property
    captured by a test, so a future refactor that adds
    ``x.value_kind`` would have to update this test too.
  - ``assert_value_kind`` produces actionable error text with
    op name + expected / actual kinds + (optional) source span.
"""

from __future__ import annotations

import numpy as np
import pytest

from tessera import bridge
from tessera import complex as tc
from tessera import ga


# ---------------------------------------------------------------------------
# Multivector ↔ tensor
# ---------------------------------------------------------------------------

def test_multivector_to_tensor_returns_independent_copy() -> None:
    """The op must return a copy, not a view — mutating the tensor
    must not corrupt the multivector."""
    a = ga.Cl(3, 0)
    mv = ga.Multivector(np.arange(8, dtype=np.float32), a)
    arr = bridge.multivector_to_tensor(mv)
    arr[0] = 99.0
    # The multivector is unchanged.
    assert float(mv._coefficients[0]) == 0.0


def test_multivector_to_tensor_round_trip_recovers_mv() -> None:
    a = ga.Cl(3, 0)
    mv = ga.Multivector(np.linspace(1, 8, 8, dtype=np.float32), a)
    coefs = bridge.multivector_to_tensor(mv)
    mv_back = bridge.tensor_to_multivector(coefs, a)
    np.testing.assert_array_equal(
        bridge.multivector_to_tensor(mv_back), coefs,
    )


def test_multivector_to_tensor_rejects_non_multivector() -> None:
    with pytest.raises(bridge.TesseraValueKindError, match="multivector_to_tensor"):
        bridge.multivector_to_tensor(np.zeros(8))


def test_tensor_to_multivector_rejects_wrong_trailing_axis() -> None:
    a = ga.Cl(3, 0)
    with pytest.raises(bridge.TesseraValueKindError, match="trailing axis"):
        bridge.tensor_to_multivector(np.zeros(7), a)


# ---------------------------------------------------------------------------
# Complex ↔ tensor
# ---------------------------------------------------------------------------

def test_complex_to_tensor_packs_re_im_on_last_axis() -> None:
    z = tc.from_pair(np.array([1.0, 2.0]), np.array([3.0, 4.0]))
    arr = bridge.complex_to_tensor(z)
    assert arr.shape == (2, 2)
    np.testing.assert_array_equal(arr[..., 0], [1.0, 2.0])
    np.testing.assert_array_equal(arr[..., 1], [3.0, 4.0])


def test_complex_to_tensor_round_trip_recovers_scalar() -> None:
    z = tc.from_pair(np.array([1.5, -2.5]), np.array([0.5, 7.5]))
    arr = bridge.complex_to_tensor(z)
    z_back = bridge.tensor_to_complex(arr)
    np.testing.assert_array_equal(z_back.re, z.re)
    np.testing.assert_array_equal(z_back.im, z.im)


def test_complex_to_tensor_rejects_non_complex_scalar() -> None:
    with pytest.raises(bridge.TesseraValueKindError, match="complex_to_tensor"):
        bridge.complex_to_tensor(np.array([1.0, 2.0]))


def test_tensor_to_complex_rejects_wrong_last_axis() -> None:
    with pytest.raises(bridge.TesseraValueKindError, match="trailing axis"):
        bridge.tensor_to_complex(np.zeros((4, 3)))


# ---------------------------------------------------------------------------
# value_kind_of — the Decision #15a classifier
# ---------------------------------------------------------------------------

def test_value_kind_of_classifies_each_sibling() -> None:
    a = ga.Cl(3, 0)
    mv = ga.Multivector(np.zeros(8, dtype=np.float32), a)
    z = tc.from_pair(0.0, 0.0)
    assert bridge.value_kind_of(mv) == "multivector"
    assert bridge.value_kind_of(z) == "complex"
    assert bridge.value_kind_of(np.zeros(4)) == "tensor"
    assert bridge.value_kind_of(3.14) == "tensor"
    assert bridge.value_kind_of(object()) == "unknown"


def test_value_kind_of_is_a_function_not_a_method() -> None:
    """Decision #15a structural lock: there is no ``x.value_kind``
    attribute on a tensor / multivector / complex scalar.  Code
    that branches on kind MUST call :func:`bridge.value_kind_of`.

    This test catches a regression that adds a ``value_kind``
    attribute to one of the sibling types — that would silently
    invite per-attribute reads instead of going through the
    bridge."""
    a = ga.Cl(3, 0)
    mv = ga.Multivector(np.zeros(8, dtype=np.float32), a)
    z = tc.from_pair(0.0, 0.0)
    arr = np.zeros(4)
    assert not hasattr(mv, "value_kind"), "Multivector grew a value_kind attr"
    assert not hasattr(z, "value_kind"), "ComplexScalar grew a value_kind attr"
    assert not hasattr(arr, "value_kind"), "ndarray grew a value_kind attr"


def test_value_kind_of_normative_set_matches_compile_report() -> None:
    """The bridge constants must match the values used by
    :class:`CompileReport.value_kind`'s normative set.  Drift
    would cause a session to record different kind names than
    the bridge classifies."""
    from tessera.compiler.compile_report import (
        VALID_VALUE_KINDS, VALUE_KIND_MULTIVECTOR, VALUE_KIND_TENSOR,
    )
    assert bridge.VALUE_KIND_TENSOR == VALUE_KIND_TENSOR
    assert bridge.VALUE_KIND_MULTIVECTOR == VALUE_KIND_MULTIVECTOR
    # `complex` and `mixed` are bridge-only / compile-report-only
    # respectively; the overlap is what matters here.
    assert bridge.VALUE_KIND_TENSOR in VALID_VALUE_KINDS
    assert bridge.VALUE_KIND_MULTIVECTOR in VALID_VALUE_KINDS


# ---------------------------------------------------------------------------
# assert_value_kind
# ---------------------------------------------------------------------------

def test_assert_value_kind_passes_when_kind_matches() -> None:
    bridge.assert_value_kind(np.zeros(4), "tensor")
    a = ga.Cl(3, 0)
    mv = ga.Multivector(np.zeros(8, dtype=np.float32), a)
    bridge.assert_value_kind(mv, "multivector")


def test_assert_value_kind_raises_with_actionable_message() -> None:
    with pytest.raises(bridge.TesseraValueKindError) as exc:
        bridge.assert_value_kind(
            np.zeros(4),
            "multivector",
            op_name="ga.norm",
            source_span=(12, 5),
        )
    text = str(exc.value)
    assert "ga.norm" in text
    assert "multivector" in text
    assert "tensor" in text
    assert "line 12, col 5" in text
    assert "M2_VALUE_KIND_MISMATCH" in text


def test_assert_value_kind_accepts_multiple_expected_kinds() -> None:
    """For ops that accept either tensor or complex (rare but
    possible), the assertion takes a varargs list."""
    bridge.assert_value_kind(np.zeros(2), "tensor", "complex")
    bridge.assert_value_kind(tc.from_pair(0.0, 0.0), "tensor", "complex")
    with pytest.raises(bridge.TesseraValueKindError):
        a = ga.Cl(3, 0)
        mv = ga.Multivector(np.zeros(8, dtype=np.float32), a)
        bridge.assert_value_kind(mv, "tensor", "complex")


# ---------------------------------------------------------------------------
# Error structure
# ---------------------------------------------------------------------------

def test_error_carries_structured_fields_for_session_capture() -> None:
    """Sessions need machine-readable provenance, not just text."""
    exc = bridge.TesseraValueKindError(
        "demo",
        expected_kind="tensor",
        actual_kind="multivector",
        source_span=(99, 4),
    )
    assert exc.expected_kind == "tensor"
    assert exc.actual_kind == "multivector"
    assert exc.source_span == (99, 4)
