"""LDT / MoE-aux loss ops on Metal — z_loss + asymmetric_bce (MPSGraph subgraphs).

Promotes the router z-loss and asymmetric BCE from the numpy-fallback
(metal_artifact) lane to metal_runtime via MPSGraph subgraphs (mirroring the PPO
loss). Both reduce to a scalar; only ``reduction="mean"`` runs on GPU, other
reductions fall back to the numpy reference. Closes 2 of the lattice-reasoning
benchmark's apple_gpu artifact rows.
"""

import numpy as np
import pytest

import tessera as ts
import tessera.losses as L
from tessera import _apple_gpu_backend as agb
from tessera import _jit_boundary as jb
from tessera import runtime as _runtime
from tessera.compiler import driver as _driver

_GPU = agb.is_available() and jb.is_available()
gpu = pytest.mark.skipif(
    not _GPU, reason="apple_gpu runtime / libtessera_jit unavailable")


def test_sentinel_and_envelope():
    from tessera import _apple_gpu_dispatch as agd
    # The staleness sentinel moves as new C ABI symbols land (owned by the
    # newest-symbol test); here we only assert it is a real apple_gpu symbol.
    assert agd._SENTINEL_SYMBOL.startswith("tessera_apple_gpu_")
    for op in ("tessera.loss.z_loss", "tessera.loss.asymmetric_bce",
               "tessera.loss.load_balance_loss", "tessera.masked_categorical"):
        assert op in _driver._APPLE_GPU_RUNTIME_OPS
        assert op in _runtime._APPLE_GPU_RUNTIME_OPS


@gpu
@pytest.mark.parametrize("shape", [(6, 4), (1, 8), (32, 16), (4, 1)])
def test_gpu_z_loss_matches_reference(shape):
    rng = np.random.default_rng(sum(shape))
    logits = rng.standard_normal(shape).astype(np.float32)
    np.testing.assert_allclose(agb.gpu_z_loss(logits), float(L.z_loss(logits)),
                               rtol=1e-5, atol=1e-6)


@gpu
@pytest.mark.parametrize("pos_w,neg_w", [(1.0, 1.0), (2.0, 0.5), (3.0, 0.1)])
def test_gpu_asymmetric_bce_matches_reference(pos_w, neg_w):
    rng = np.random.default_rng(int(pos_w * 10 + neg_w * 100))
    z = rng.standard_normal((5, 5)).astype(np.float32)
    t = (rng.random((5, 5)) < 0.5).astype(np.float32)
    np.testing.assert_allclose(
        agb.gpu_asymmetric_bce(z, t, pos_w, neg_w),
        float(L.asymmetric_bce(z, t, pos_w, neg_w)), rtol=1e-5, atol=1e-6)


@gpu
def test_asymmetric_bce_large_logits_stable():
    z = np.array([-1e3, 1e3, -50.0, 50.0], np.float32)
    t = np.array([0.0, 1.0, 0.0, 1.0], np.float32)
    got = agb.gpu_asymmetric_bce(z, t, 3.0, 2.0)
    assert np.isfinite(got)
    np.testing.assert_allclose(got, float(L.asymmetric_bce(z, t, 3.0, 2.0)),
                               rtol=1e-4, atol=1e-5)


# ── @jit execution mode flips to metal_runtime ─────────────────────────────── #
@gpu
def test_z_loss_jit_metal_runtime():
    @ts.jit(target="apple_gpu")
    def f(x):
        return ts.ops.z_loss(x)
    logits = np.random.default_rng(1).standard_normal((6, 4)).astype(np.float32)
    art = f(logits)
    assert f.runtime_artifact().metadata["execution_mode"] == "metal_runtime"
    np.testing.assert_allclose(float(np.asarray(art)), float(L.z_loss(logits)),
                               rtol=1e-5, atol=1e-6)


@gpu
def test_asymmetric_bce_jit_metal_runtime():
    @ts.jit(target="apple_gpu")
    def f(z, t):
        return ts.ops.asymmetric_bce(z, t, pos_weight=2.0, neg_weight=0.5)
    rng = np.random.default_rng(2)
    z = rng.standard_normal((5, 5)).astype(np.float32)
    t = (rng.random((5, 5)) < 0.5).astype(np.float32)
    art = f(z, t)
    assert f.runtime_artifact().metadata["execution_mode"] == "metal_runtime"
    np.testing.assert_allclose(float(np.asarray(art)),
                               float(L.asymmetric_bce(z, t, 2.0, 0.5)),
                               rtol=1e-5, atol=1e-6)


@gpu
def test_non_mean_reduction_falls_back_correct():
    @ts.jit(target="apple_gpu")
    def f(x):
        return ts.ops.z_loss(x, reduction="sum")
    logits = np.random.default_rng(3).standard_normal((4, 3)).astype(np.float32)
    np.testing.assert_allclose(float(np.asarray(f(logits))),
                               float(L.z_loss(logits, reduction="sum")),
                               rtol=1e-5, atol=1e-5)


# ── load_balance_loss (argMax/oneHot subgraph) ─────────────────────────────── #
@gpu
@pytest.mark.parametrize("shape", [(64, 8), (128, 4), (32, 16)])
def test_gpu_load_balance_matches_reference(shape):
    rng = np.random.default_rng(sum(shape))
    p = np.exp(rng.standard_normal(shape)).astype(np.float32)
    p /= p.sum(-1, keepdims=True)
    np.testing.assert_allclose(agb.gpu_load_balance_loss(p),
                               float(L.load_balance_loss(p)), rtol=1e-5, atol=1e-6)


@gpu
def test_gpu_load_balance_bounds():
    E = 8
    uniform = np.full((512, E), 1.0 / E, np.float32)
    np.testing.assert_allclose(agb.gpu_load_balance_loss(uniform), 1.0, atol=1e-4)
    conc = np.zeros((512, E), np.float32); conc[:, 0] = 1.0
    np.testing.assert_allclose(agb.gpu_load_balance_loss(conc), float(E), atol=1e-3)


@gpu
def test_load_balance_jit_metal_runtime():
    @ts.jit(target="apple_gpu")
    def f(p):
        return ts.ops.load_balance_loss(p)
    rng = np.random.default_rng(7)
    p = np.exp(rng.standard_normal((64, 8))).astype(np.float32); p /= p.sum(-1, keepdims=True)
    assert f.runtime_artifact().metadata["execution_mode"] == "metal_runtime"
    np.testing.assert_allclose(float(np.asarray(f(p))),
                               float(L.load_balance_loss(p)), rtol=1e-5, atol=1e-6)


# ── masked_categorical (select/argMax subgraph) ────────────────────────────── #
@gpu
def test_gpu_masked_categorical_greedy():
    logits = np.array([[1.0, 5.0, 2.0], [3.0, 0.0, 4.0]], np.float32)
    mask = np.array([[1, 0, 1], [1, 1, 0]], np.float32)   # mask out each row's max
    np.testing.assert_array_equal(agb.gpu_masked_categorical(logits, mask), [2, 0])
    full = np.ones_like(mask)
    np.testing.assert_array_equal(
        agb.gpu_masked_categorical(logits, full), np.argmax(logits, axis=-1))


@gpu
@pytest.mark.parametrize("shape", [(8, 5), (16, 9), (4, 3)])
def test_gpu_masked_categorical_matches_reference(shape):
    rng = np.random.default_rng(sum(shape))
    logits = rng.standard_normal(shape).astype(np.float32)
    mask = (rng.random(shape) < 0.6); mask[:, 0] = True
    expect = ts.ops.masked_categorical(logits, mask.astype(np.int32))
    np.testing.assert_array_equal(
        agb.gpu_masked_categorical(logits, mask.astype(np.float32)), expect)


@gpu
def test_masked_categorical_jit_metal_runtime():
    @ts.jit(target="apple_gpu")
    def f(lo, m):
        return ts.ops.masked_categorical(lo, m)
    lo = np.array([[1.0, 5.0, 2.0], [3.0, 0.0, 4.0]], np.float32)
    m = np.array([[1, 0, 1], [1, 1, 0]], np.float32)
    np.testing.assert_array_equal(np.asarray(f(lo, m)), [2, 0])
    assert f.runtime_artifact().metadata["execution_mode"] == "metal_runtime"
