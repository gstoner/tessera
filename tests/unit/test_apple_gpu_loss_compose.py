"""Batch 3 — regression / CE losses on Apple GPU via opcode composition.

The 9 op_catalog-ready losses (mse/mae/huber/smooth_l1/log_cosh/vlb/ddpm/bce/
cross_entropy) execute on the GPU by chaining the batch-1/2 opcode lanes (a
per-element recipe + a reduce) — no dedicated loss kernel. metal_runtime.
"""

import numpy as np
import pytest

import tessera as ts
import tessera.losses as L
from tessera import _apple_gpu_backend as agb
from tessera import _jit_boundary as jb
from tessera import runtime as _runtime
from tessera.compiler import driver as _driver

R = _runtime
_GPU = agb.is_available() and jb.is_available()
gpu = pytest.mark.skipif(not _GPU, reason="apple_gpu runtime / libtessera_jit unavailable")

_LOSSES = ["mse", "mae", "huber", "smooth_l1", "log_cosh", "vlb",
           "ddpm_noise_pred", "binary_cross_entropy", "cross_entropy"]


def test_all_in_envelope():
    for n in _LOSSES:
        op = f"tessera.loss.{n}"
        assert op in _driver._APPLE_GPU_RUNTIME_OPS, op
        assert op in _runtime._APPLE_GPU_RUNTIME_OPS, op
        assert op in _driver._APPLE_GPU_LOSS_COMPOSE_OPS, op


def test_driver_runtime_loss_envelopes_agree():
    assert _driver._APPLE_GPU_LOSS_COMPOSE_OPS == _runtime._APPLE_GPU_LOSS_COMPOSE_OPS


def _ab(seed):
    rng = np.random.default_rng(seed)
    return (rng.standard_normal((4, 6)).astype(np.float32),
            rng.standard_normal((4, 6)).astype(np.float32))


@gpu
@pytest.mark.parametrize("name,ref", [
    ("mse", lambda a, b: L.mse_loss(a, b)),
    ("mae", lambda a, b: L.mae_loss(a, b)),
    ("huber", lambda a, b: L.huber_loss(a, b, delta=1.0)),
    ("smooth_l1", lambda a, b: L.smooth_l1_loss(a, b, beta=1.0)),
    ("log_cosh", lambda a, b: L.log_cosh_loss(a, b)),
    ("ddpm_noise_pred", lambda a, b: L.ddpm_noise_pred_loss(a, b)),
])
def test_regression_loss(name, ref):
    a, b = _ab(hash(name) % 99)
    out = R._apple_gpu_dispatch_loss(f"tessera.loss.{name}", [a, b], {}, np)
    np.testing.assert_allclose(float(np.asarray(out)), float(ref(a, b)), atol=1e-3)


@gpu
@pytest.mark.parametrize("reduction", ["mean", "sum", "none"])
def test_mse_reductions(reduction):
    a, b = _ab(5)
    out = R._apple_gpu_dispatch_loss("tessera.loss.mse", [a, b], {"reduction": reduction}, np)
    np.testing.assert_allclose(np.asarray(out), np.asarray(L.mse_loss(a, b, reduction=reduction)), atol=1e-3)


@gpu
def test_vlb_and_bce():
    a, b = _ab(6)
    np.testing.assert_allclose(
        float(np.asarray(R._apple_gpu_dispatch_loss("tessera.loss.vlb", [a], {}, np))),
        float(L.vlb_loss(a)), atol=1e-3)
    t = (b > 0).astype(np.float32)
    np.testing.assert_allclose(
        float(np.asarray(R._apple_gpu_dispatch_loss("tessera.loss.binary_cross_entropy", [a, t], {}, np))),
        float(L.binary_cross_entropy_loss(a, t)), atol=1e-3)


@gpu
def test_cross_entropy_onehot_gpu_and_integer_fallback():
    rng = np.random.default_rng(7)
    logits = rng.standard_normal((5, 4)).astype(np.float32)
    onehot = np.eye(4)[rng.integers(0, 4, 5)].astype(np.float32)
    out = R._apple_gpu_dispatch_loss("tessera.loss.cross_entropy", [logits, onehot], {}, np)
    np.testing.assert_allclose(float(np.asarray(out)), float(L.cross_entropy_loss(logits, onehot)), atol=1e-3)
    # integer targets fall back to the numpy reference (no GPU gather) — still correct.
    idx = rng.integers(0, 4, 5).astype(np.int64)
    out_i = R._apple_gpu_dispatch_loss("tessera.loss.cross_entropy", [logits, idx], {}, np)
    np.testing.assert_allclose(float(np.asarray(out_i)), float(L.cross_entropy_loss(logits, idx)), atol=1e-3)


@gpu
def test_mse_jit_metal_runtime():
    a, b = _ab(8)

    @ts.jit(target="apple_gpu")
    def f(a, b):
        return ts.ops.mse_loss(a, b)

    np.testing.assert_allclose(float(np.asarray(f(a, b))), float(L.mse_loss(a, b)), atol=1e-3)
    assert f.runtime_artifact().metadata["execution_mode"] == "metal_runtime"


@gpu
def test_huber_jit_metal_runtime():
    a, b = _ab(9)

    @ts.jit(target="apple_gpu")
    def f(a, b):
        return ts.ops.huber_loss(a, b)

    np.testing.assert_allclose(float(np.asarray(f(a, b))), float(L.huber_loss(a, b)), atol=1e-3)
    assert f.runtime_artifact().metadata["execution_mode"] == "metal_runtime"


def _probs(seed, shape=(4, 6)):
    rng = np.random.default_rng(seed)
    p = np.abs(rng.standard_normal(shape).astype(np.float32)) + 0.1
    q = np.abs(rng.standard_normal(shape).astype(np.float32)) + 0.1
    return p / p.sum(-1, keepdims=True), q / q.sum(-1, keepdims=True)


def test_kl_js_in_envelope():
    for op in ("tessera.loss.kl_divergence", "tessera.loss.js_divergence"):
        assert op in _driver._APPLE_GPU_RUNTIME_OPS, op
        assert op in _runtime._APPLE_GPU_RUNTIME_OPS, op
        assert op in _driver._APPLE_GPU_LOSS_COMPOSE_OPS, op


@gpu
@pytest.mark.parametrize("reduction", ["mean", "sum", "none"])
def test_kl_divergence(reduction):
    p, q = _probs(11)
    plog = np.log(p)  # kl_divergence takes log-probs as the first arg
    out = R._apple_gpu_dispatch_loss(
        "tessera.loss.kl_divergence", [plog, q], {"reduction": reduction}, np)
    np.testing.assert_allclose(
        np.asarray(out), np.asarray(L.kl_divergence(plog, q, reduction=reduction)), atol=1e-3)


@gpu
@pytest.mark.parametrize("reduction", ["mean", "sum", "none"])
def test_js_divergence(reduction):
    p, q = _probs(12)
    out = R._apple_gpu_dispatch_loss(
        "tessera.loss.js_divergence", [p, q], {"reduction": reduction}, np)
    np.testing.assert_allclose(
        np.asarray(out), np.asarray(L.js_divergence(p, q, reduction=reduction)), atol=1e-3)


@gpu
def test_kl_jit_metal_runtime():
    p, q = _probs(13)
    plog = np.log(p)

    @ts.jit(target="apple_gpu")
    def f(a, b):
        return ts.ops.kl_divergence(a, b)

    np.testing.assert_allclose(float(np.asarray(f(plog, q))), float(L.kl_divergence(plog, q)), atol=1e-3)
    assert f.runtime_artifact().metadata["execution_mode"] == "metal_runtime"
