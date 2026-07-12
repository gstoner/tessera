"""Direct numerical proofs for composed host CPU conformance rows."""

from __future__ import annotations

import numpy as np

import tessera as ts


def _inputs():
    rng = np.random.default_rng(20260712)
    return (rng.standard_normal((7, 5), dtype=np.float32),
            rng.standard_normal((5, 9), dtype=np.float32))


def test_cpu_matmul_relu_composition_matches_numpy():
    @ts.jit(target="cpu")
    def composed(a, b):
        return ts.ops.relu(ts.ops.matmul(a, b))

    a, b = _inputs()
    np.testing.assert_allclose(composed(a, b), np.maximum(a @ b, 0.0),
                               rtol=1e-5, atol=1e-5)


def test_cpu_matmul_softmax_composition_matches_numpy():
    @ts.jit(target="cpu")
    def composed(a, b):
        return ts.ops.softmax(ts.ops.matmul(a, b))

    a, b = _inputs()
    scores = a @ b
    exp = np.exp(scores - scores.max(axis=-1, keepdims=True))
    expected = exp / exp.sum(axis=-1, keepdims=True)
    np.testing.assert_allclose(composed(a, b), expected,
                               rtol=1e-5, atol=1e-5)
