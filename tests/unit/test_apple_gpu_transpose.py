"""Apple GPU transpose — the first structural layout op on a real MPSGraph kernel.

tessera.transpose now runs on Metal via `transposeTensor:permutation:` (was the
numpy-reference fallback). Transpose is value-preserving data movement, so f32
goes native f32, and f16/bf16 share the 2-byte raw path. This is the first
"real_gap_structural" displacement (see apple_gpu_coverage disposition): a
structural op that previously demoted a mixed program off metal_runtime.
"""

from __future__ import annotations

import sys

import numpy as np
import pytest

import tessera as ts
from tessera.compiler import apple_gpu_envelope as env

DARWIN = sys.platform == "darwin"
_RNG = np.random.default_rng(20260617)


def test_transpose_is_a_first_class_gpu_op():
    assert env.lane_for("tessera.transpose") == "transpose"
    assert "tessera.transpose" in env.runtime_ops()


@pytest.mark.parametrize("shape,axes", [
    ((8, 16), None),            # 2D default reverse
    ((4, 6, 8), None),          # 3D reverse
    ((2, 3, 4, 5), (0, 2, 1, 3)),  # 4D explicit permute (attention-style)
])
def test_transpose_handler_matches_numpy(shape, axes):
    """The transpose dispatcher (Metal on Darwin, host fallback elsewhere) vs
    numpy. Exercises 2D/3D/4D + default-reverse and explicit-permute."""
    from tessera.runtime import _apple_gpu_dispatch_transpose
    x = _RNG.standard_normal(shape).astype(np.float32)
    out = _apple_gpu_dispatch_transpose(
        "tessera.transpose", [x], {"axes": axes} if axes is not None else {}, np)
    np.testing.assert_array_equal(np.asarray(out), np.transpose(x, axes))


def test_transpose_f16_matches_numpy():
    from tessera.runtime import _apple_gpu_dispatch_transpose
    x = _RNG.standard_normal((4, 8, 3)).astype(np.float16)
    out = _apple_gpu_dispatch_transpose("tessera.transpose", [x], {}, np)
    # Value-preserving: bit-exact permute, no arithmetic.
    np.testing.assert_array_equal(np.asarray(out), np.transpose(x))


def test_jit_transpose_matches_numpy():
    @ts.jit(target="apple_gpu")
    def f(x):
        return ts.ops.transpose(x)

    x = _RNG.standard_normal((8, 16)).astype(np.float32)
    np.testing.assert_array_equal(np.asarray(f(x)), x.T)


@pytest.mark.skipif(not DARWIN, reason="GPU-residency check requires Metal (Darwin)")
def test_transpose_runs_on_metal_no_fallback():
    from tessera.compiler import apple_gpu_coverage as cov

    @ts.jit(target="apple_gpu")
    def f(x):
        return ts.ops.transpose(x)

    x = _RNG.standard_normal((8, 16)).astype(np.float32)
    assert cov.fallback_histogram(lambda: f(x)) == {}
