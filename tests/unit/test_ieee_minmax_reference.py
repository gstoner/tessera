"""Route-consistent IEEE-754-2019 min/max reference (host-independent).

Fleet contract (`IEEE-MINMAX-CONTRACT-2026-08-23`, rocm plan):
``tessera.maximum``/``minimum`` propagate NaN and order signed zeros
(max tie -> +0.0, min tie -> -0.0) on EVERY execution route. numpy is
not a valid tie-sign oracle — ``np.maximum(+0.0, -0.0)`` returns the
second operand on SSE hosts and the IEEE-ordered zero on NEON hosts —
so the eager op namespace and the numpy fallback lanes must not
delegate the tie to numpy. These tests run on any host (no device) and
pin the shared reference (`tessera/_ieee_minmax.py`) plus both routes
that consume it.
"""

from __future__ import annotations

import numpy as np
import pytest

import tessera as ts
from tessera._ieee_minmax import ieee_maximum, ieee_minimum


A = np.array([0.0, -0.0, 0.0, -0.0, np.nan, 1.0, -3.0], np.float32)
B = np.array([-0.0, 0.0, 0.0, -0.0, 1.0, np.nan, 4.0], np.float32)
# rows: (+0,-0) (-0,+0) (+0,+0) (-0,-0) (NaN,x) (x,NaN) ordered
_NAN = [False, False, False, False, True, True, False]


def _check(out, tie_sign, ordered_value):
    out = np.asarray(out).astype(np.float32)
    np.testing.assert_array_equal(np.isnan(out), _NAN)
    np.testing.assert_array_equal(
        np.signbit(out[:4]), [tie_sign, tie_sign, False, True])
    assert out[6] == ordered_value


@pytest.mark.parametrize("fn,tie_sign,ordered", [
    (ieee_maximum, False, 4.0),
    (ieee_minimum, True, -3.0),
])
def test_helper_orders_ties_and_propagates_nan(fn, tie_sign, ordered):
    _check(fn(A, B), tie_sign, ordered)


@pytest.mark.parametrize("name,tie_sign,ordered", [
    ("maximum", False, 4.0),
    ("minimum", True, -3.0),
])
def test_eager_ops_namespace_uses_ieee_ties(name, tie_sign, ordered):
    _check(getattr(ts.ops, name)(A, B), tie_sign, ordered)


@pytest.mark.parametrize("op,tie_sign,ordered", [
    ("tessera.maximum", False, 4.0),
    ("tessera.minimum", True, -3.0),
])
def test_apple_numpy_fallback_uses_ieee_ties(op, tie_sign, ordered):
    from tessera.runtime import _apple_gpu_binary_numpy

    _check(_apple_gpu_binary_numpy(op, A, B, np), tie_sign, ordered)


def test_helper_matches_numpy_away_from_ties():
    rng = np.random.default_rng(3)
    a = rng.standard_normal(512).astype(np.float32)
    b = rng.standard_normal(512).astype(np.float32)
    np.testing.assert_array_equal(ieee_maximum(a, b), np.maximum(a, b))
    np.testing.assert_array_equal(ieee_minimum(a, b), np.minimum(a, b))
