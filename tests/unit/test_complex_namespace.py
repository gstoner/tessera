"""M7 Step 1 — ``tessera.complex`` namespace skeleton tests.

Locks the decision to use a non-GA :class:`ComplexScalar` sibling
kind (Decision #15a precedent — complex numbers are not tensors
and not multivectors) and verifies the encoder/decoder identities.
"""

from __future__ import annotations

import numpy as np
import pytest

from tessera import complex as tc


# ---------------------------------------------------------------------------
# Sibling-kind contract
# ---------------------------------------------------------------------------

def test_complex_scalar_is_frozen_sibling_value() -> None:
    """The dataclass is frozen — assignment must raise.
    Mirrors the Multivector sibling-kind invariant from Decision #15a."""
    z = tc.from_pair(3.0, 4.0)
    with pytest.raises((AttributeError, TypeError)):
        z.re = np.asarray(0.0)  # type: ignore[misc]


def test_re_and_im_must_have_matching_shapes() -> None:
    with pytest.raises(ValueError, match="re.shape="):
        tc.ComplexScalar(np.zeros(4), np.zeros(8))


def test_scalar_re_im_round_trip() -> None:
    z = tc.from_pair(3.0, 4.0)
    re, im = tc.to_pair(z)
    assert float(re) == 3.0
    assert float(im) == 4.0
    assert z.shape == ()
    assert z.ndim == 0


def test_batched_complex_scalar_preserves_shape_and_dtype() -> None:
    re = np.linspace(0, 1, 12, dtype=np.float32).reshape(3, 4)
    im = np.linspace(1, 2, 12, dtype=np.float32).reshape(3, 4)
    z = tc.from_pair(re, im)
    assert z.shape == (3, 4)
    assert z.dtype == np.float32
    np.testing.assert_array_equal(z.re, re)
    np.testing.assert_array_equal(z.im, im)


# ---------------------------------------------------------------------------
# numpy interop
# ---------------------------------------------------------------------------

def test_from_numpy_unwraps_complex_array() -> None:
    arr = np.array([1 + 2j, 3 - 4j], dtype=np.complex128)
    z = tc.from_numpy(arr)
    np.testing.assert_array_equal(z.re, [1.0, 3.0])
    np.testing.assert_array_equal(z.im, [2.0, -4.0])


def test_from_numpy_accepts_real_array_as_zero_imag() -> None:
    z = tc.from_numpy(np.array([1.0, 2.0, 3.0]))
    np.testing.assert_array_equal(z.im, [0.0, 0.0, 0.0])


def test_to_numpy_round_trip_is_exact_in_double_precision() -> None:
    arr = np.array([1 + 2j, 3 - 4j, 0.5 + 0.5j], dtype=np.complex128)
    reconstructed = tc.to_numpy(tc.from_numpy(arr))
    np.testing.assert_array_equal(reconstructed, arr)


def test_to_numpy_respects_requested_dtype() -> None:
    z = tc.from_pair(np.array([1.0]), np.array([2.0]))
    out = tc.to_numpy(z, dtype=np.complex64)
    assert out.dtype == np.complex64


# ---------------------------------------------------------------------------
# is_complex predicate (M7 verifier will gate on this)
# ---------------------------------------------------------------------------

def test_is_complex_recognizes_complex_scalar_and_numpy_complex() -> None:
    assert tc.is_complex(tc.from_pair(0.0, 0.0))
    assert tc.is_complex(np.array([1 + 2j]))
    assert not tc.is_complex(np.array([1.0, 2.0]))
    assert not tc.is_complex(3.14)


# ---------------------------------------------------------------------------
# Decision pin
# ---------------------------------------------------------------------------

def test_module_docstring_pins_the_decision() -> None:
    """The pivot from Cl(0,1) → ComplexScalar sibling kind must be
    documented at the top of the module so the decision is
    discoverable from the source."""
    from tessera import complex as tc_module
    assert tc_module.__doc__ is not None
    doc = tc_module.__doc__
    assert "Cl(0,1)" in doc
    assert "sibling kind" in doc.lower()
    assert "Decision #15a" in doc
