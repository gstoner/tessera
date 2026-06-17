"""Apple GPU softcap lane — the parameterized-unary follow-up.

softcap(x, cap) = cap * tanh(x / cap) (the Gemma logit soft-cap) was the one
genuinely numpy-only real-valued elementwise op the displacement worklist
flagged. It carries a scalar `cap`, so it doesn't fit the param-free
pointwise-vocab; instead it rides a GPU *compose* lane (div-by-scalar -> tanh
unary -> mul-by-scalar), the same pattern as clamp/where — no dedicated kernel,
no .mm change. `cap` is a config constant (a literal in the jitted source) in
practice; closure-captured scalars aren't folded into the apple_gpu metadata
path (a known frontend limit, not softcap-specific) and fail loudly.
"""

from __future__ import annotations

import sys

import numpy as np
import pytest

import tessera as ts
from tessera.compiler import apple_gpu_envelope as env
from tessera.runtime import _apple_gpu_dispatch_softcap

DARWIN = sys.platform == "darwin"
_RNG = np.random.default_rng(20260617)


def _ref(x, cap):
    return cap * np.tanh(x / cap)


def test_softcap_is_a_first_class_gpu_op():
    assert env.lane_for("tessera.softcap") == "softcap"
    assert "tessera.softcap" in env.runtime_ops()


@pytest.mark.parametrize("cap", [1.0, 20.0, 50.0])
def test_softcap_handler_matches_numpy(cap):
    """The GPU compose handler (div-scalar -> tanh -> mul-scalar) vs numpy."""
    x = (_RNG.standard_normal((8, 16)) * 15).astype(np.float32)
    out = _apple_gpu_dispatch_softcap("tessera.softcap", [x], {"cap": cap}, np)
    np.testing.assert_allclose(np.asarray(out), _ref(x, cap), rtol=1e-4, atol=1e-4)


def test_softcap_handler_nonpositive_cap_is_noop():
    x = np.array([[1.0, -2.0, 3.0]], np.float32)
    out = _apple_gpu_dispatch_softcap("tessera.softcap", [x], {"cap": 0.0}, np)
    np.testing.assert_allclose(np.asarray(out), x)


def test_softcap_handler_rejects_unresolved_scalar():
    with pytest.raises(ValueError, match="literal scalar"):
        _apple_gpu_dispatch_softcap(
            "tessera.softcap", [np.zeros((2, 2), np.float32)], {"cap": "%cap"}, np)


def test_jit_softcap_literal_cap_matches_numpy():
    """End-to-end @jit(apple_gpu) with a literal cap (the real Gemma usage)."""
    @ts.jit(target="apple_gpu")
    def f(x):
        return ts.ops.softcap(x, cap=20.0)

    x = (_RNG.standard_normal((8, 16)) * 15).astype(np.float32)
    np.testing.assert_allclose(np.asarray(f(x)), _ref(x, 20.0), rtol=1e-4, atol=1e-4)


@pytest.mark.skipif(not DARWIN, reason="GPU-residency check requires Metal (Darwin)")
def test_softcap_runs_on_metal_no_fallback():
    from tessera.compiler import apple_gpu_coverage as cov

    @ts.jit(target="apple_gpu")
    def f(x):
        return ts.ops.softcap(x, cap=20.0)

    x = (_RNG.standard_normal((8, 16)) * 15).astype(np.float32)
    assert cov.fallback_histogram(lambda: f(x)) == {}
