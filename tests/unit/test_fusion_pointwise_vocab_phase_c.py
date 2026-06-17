"""Phase C/D — grow the pointwise-DAG fusion vocabulary, oracle-gated.

The Phase B worklist showed every elementwise op already has a single-op GPU
lane, so the displacement win is enlarging *fusable DAGs* (fewer dispatches).
This adds the real-valued ops that DAGs previously bailed at (sqrt/rsqrt/log/
log1p/expm1/reciprocal/softplus) to ``POINTWISE_OPS``. Every addition is
codegen-gated by ``verify_synthesized_pointwise`` (Phase A); these tests lock the
vocab growth and prove a DAG using each new op fuses and matches numpy.
"""

from __future__ import annotations

import sys

import numpy as np
import pytest

from tessera.compiler import fusion as F

DARWIN = sys.platform == "darwin"
_RNG = np.random.default_rng(20260617)

# new op → (numpy fn, domain: "all" reals or "pos" strictly-positive input)
_NEW_OPS = {
    "sqrt": (np.sqrt, "pos"),
    "rsqrt": (lambda a: 1.0 / np.sqrt(a), "pos"),
    "log": (np.log, "pos"),
    "log1p": (np.log1p, "pos"),
    "expm1": (np.expm1, "all"),
    "reciprocal": (lambda a: 1.0 / a, "pos"),
    "softplus": (lambda a: np.maximum(a, 0.0) + np.log1p(np.exp(-np.abs(a))), "all"),
}


def test_new_ops_are_in_pointwise_vocab():
    for name in _NEW_OPS:
        assert name in F.POINTWISE_OPS, f"{name} missing from POINTWISE_OPS"
        assert F.is_pointwise_op(f"tessera.{name}")


@pytest.mark.parametrize("name", sorted(_NEW_OPS))
def test_new_op_dag_fuses_and_matches_numpy(name):
    ref, domain = _NEW_OPS[name]
    # 2-op DAG: <new_op>(x) * a — forces the pointwise-DAG fusion path.
    region = F.PointwiseGraphRegion(
        ops=((name, ("x",), "u"), ("mul", ("u", "a"), "o")),
        inputs=("x", "a"), output="o")
    base = _RNG.standard_normal((4, 16)).astype(np.float32)
    x = (np.abs(base) + 0.5).astype(np.float32) if domain == "pos" else base
    a = _RNG.standard_normal((4, 16)).astype(np.float32)

    # The F4 oracle must accept the synthesized kernel.
    assert F.verify_synthesized_pointwise(region, force=True)

    out, ex = F.run_pointwise_graph(region, [x, a])
    expected = ref(x) * a
    np.testing.assert_allclose(np.asarray(out), expected,
                               rtol=1e-4, atol=1e-3, equal_nan=True)
    if DARWIN:
        assert ex == "metal_runtime", f"{name} DAG should fuse on Metal"


def test_pointwise_vocab_still_fully_gpu_covered():
    """Phase B invariant must still hold after the vocab growth: every new op
    also has a single-op GPU lane (so nothing regresses to numpy-only)."""
    from tessera.compiler import apple_gpu_coverage as cov
    assert cov.numpy_lane_worklist().pointwise_vocab_covered
