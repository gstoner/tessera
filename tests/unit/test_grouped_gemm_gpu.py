"""Ragged grouped_gemm — the MoE expert-FFN compute core — + Apple GPU execution.

Third pickup's keystone (the gap analysis flagged grouped-GEMM as genuinely
missing; `moe_dispatch`/`moe_combine` are identity-stub passthroughs, so the real
compute is grouped_gemm). Tokens are sorted by expert; each contiguous group is
multiplied by its own expert weight. Distinct from `batched_gemm` (equal-size
batches) — the groups are ragged.

Layers: op_catalog identity, registry contract status, numpy correctness +
guards, VJP/JVP vs finite differences, envelope agreement, and
`@jit(target="apple_gpu")` per-group MPS-matmul execution.
"""

import importlib

import numpy as np
import pytest

import tessera as ts
from tessera import _apple_gpu_backend as agb
from tessera import _jit_boundary as jb
from tessera import runtime as _runtime
from tessera.compiler import driver as _driver
from tessera.compiler import op_catalog as _cat
from tessera.compiler import primitive_coverage as _pc

_vjp = importlib.import_module("tessera.autodiff.vjp")
_jvp = importlib.import_module("tessera.autodiff.jvp")

_GPU = agb.is_available() and jb.is_available()
gpu = pytest.mark.skipif(
    not _GPU, reason="apple_gpu runtime / libtessera_jit unavailable")


def _ref(x, w, gs):
    out = np.zeros((x.shape[0], w.shape[2]), dtype=np.float64)
    off = 0
    for e in range(w.shape[0]):
        n = int(gs[e]); out[off:off + n] = x[off:off + n] @ w[e]; off += n
    return out


# ── catalog + registry ─────────────────────────────────────────────────────── #
def test_op_catalog_arity_three():
    spec = _cat.OP_SPECS.get("grouped_gemm")
    assert spec is not None and spec.graph_name == "tessera.grouped_gemm"
    assert (spec.min_arity, spec.max_arity) == (3, 3)   # x, weights, group_sizes


def test_registry_vjp_jvp_complete():
    cs = _pc.all_primitive_coverages()["grouped_gemm"].contract_status
    assert cs.get("vjp") == "complete" and cs.get("jvp") == "complete"
    assert "grouped_gemm" in _vjp._VJPS and "grouped_gemm" in _jvp._JVPS


# ── numpy correctness + guards ─────────────────────────────────────────────── #
@pytest.mark.parametrize("gs", [[5, 3, 4], [12, 0, 0], [0, 7, 5], [1, 1, 10]])
def test_grouped_gemm_correct(gs):
    rng = np.random.default_rng(sum(gs))
    gs = np.array(gs); T, K, N, E = int(gs.sum()), 4, 3, len(gs)
    x = rng.standard_normal((T, K)); w = rng.standard_normal((E, K, N))
    np.testing.assert_allclose(ts.ops.grouped_gemm(x, w, gs), _ref(x, w, gs))


def test_grouped_gemm_differs_from_batched_gemm():
    # ragged groups with distinct weights — a single matmul cannot reproduce it.
    rng = np.random.default_rng(0)
    gs = np.array([3, 3]); x = rng.standard_normal((6, 4)); w = rng.standard_normal((2, 4, 5))
    out = ts.ops.grouped_gemm(x, w, gs)
    assert not np.allclose(out[:3], x[:3] @ w[1])     # group 0 used w[0], not w[1]
    np.testing.assert_allclose(out[:3], x[:3] @ w[0])
    np.testing.assert_allclose(out[3:], x[3:] @ w[1])


def test_grouped_gemm_shape_guards():
    rng = np.random.default_rng(1)
    x = rng.standard_normal((6, 4)); w = rng.standard_normal((2, 4, 3))
    with pytest.raises(ValueError):
        ts.ops.grouped_gemm(x, w, np.array([3, 4]))            # sum != T
    with pytest.raises(ValueError):
        ts.ops.grouped_gemm(x, rng.standard_normal((2, 5, 3)), np.array([3, 3]))  # K mismatch
    with pytest.raises(ValueError):
        ts.ops.grouped_gemm(x, w, np.array([2, 2, 2]))         # E mismatch


# ── autodiff ───────────────────────────────────────────────────────────────── #
def test_vjp_matches_finite_diff():
    rng = np.random.default_rng(2)
    gs = np.array([5, 3, 4]); T, K, N, E = 12, 4, 3, 3
    x = rng.standard_normal((T, K)); w = rng.standard_normal((E, K, N)); dy = rng.standard_normal((T, N))
    dx, dw, dgs = _vjp._VJPS["grouped_gemm"](dy, x, w, gs)
    assert dgs is None                                # group_sizes is non-diff
    eps = 1e-6
    ndx = np.zeros_like(x)
    for i in range(x.size):
        xp = x.copy().ravel(); xm = x.copy().ravel(); xp[i] += eps; xm[i] -= eps
        ndx.ravel()[i] = np.sum((ts.ops.grouped_gemm(xp.reshape(x.shape), w, gs)
                                 - ts.ops.grouped_gemm(xm.reshape(x.shape), w, gs)) * dy) / (2 * eps)
    np.testing.assert_allclose(dx, ndx, atol=1e-7)
    ndw = np.zeros_like(w)
    for i in range(w.size):
        wp = w.copy().ravel(); wm = w.copy().ravel(); wp[i] += eps; wm[i] -= eps
        ndw.ravel()[i] = np.sum((ts.ops.grouped_gemm(x, wp.reshape(w.shape), gs)
                                 - ts.ops.grouped_gemm(x, wm.reshape(w.shape), gs)) * dy) / (2 * eps)
    np.testing.assert_allclose(dw, ndw, atol=1e-7)


def test_jvp_matches_directional():
    rng = np.random.default_rng(3)
    gs = np.array([4, 4]); T, K, N, E = 8, 3, 2, 2
    x = rng.standard_normal((T, K)); w = rng.standard_normal((E, K, N))
    dx = rng.standard_normal((T, K)); dw = rng.standard_normal((E, K, N))
    primal, tan = _jvp._JVPS["grouped_gemm"]((x, w, gs), (dx, dw, None))
    eps = 1e-6
    fd = (ts.ops.grouped_gemm(x + eps * dx, w + eps * dw, gs)
          - ts.ops.grouped_gemm(x - eps * dx, w - eps * dw, gs)) / (2 * eps)
    np.testing.assert_allclose(primal, _ref(x, w, gs))
    np.testing.assert_allclose(tan, fd, atol=1e-6)


# ── envelope ───────────────────────────────────────────────────────────────── #
def test_grouped_gemm_in_apple_gpu_envelope():
    assert "tessera.grouped_gemm" in _driver._APPLE_GPU_RUNTIME_OPS
    assert "tessera.grouped_gemm" in _runtime._APPLE_GPU_RUNTIME_OPS


# ── Apple GPU execution ────────────────────────────────────────────────────── #
@gpu
@pytest.mark.parametrize("gs", [[5, 3, 4], [0, 7, 5], [16, 16, 32]])
def test_grouped_gemm_apple_gpu(gs):
    rng = np.random.default_rng(sum(gs) + 1)
    gs = np.array(gs); T, K, N, E = int(gs.sum()), 8, 6, len(gs)
    x = rng.standard_normal((T, K)).astype(np.float32)
    w = rng.standard_normal((E, K, N)).astype(np.float32)
    ref = _ref(x, w, gs)

    @ts.jit(target="apple_gpu")
    def f(x, w, group_sizes):
        return ts.ops.grouped_gemm(x, w, group_sizes)

    out = np.asarray(f(x, w, gs))
    np.testing.assert_allclose(out, ref, rtol=1e-3, atol=1e-4)
    assert f.runtime_artifact().metadata["execution_mode"] == "metal_runtime"
