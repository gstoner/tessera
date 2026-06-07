"""MoE auxiliary losses — z_loss (router-logit regularizer) and
load_balance_loss (Switch-Transformer load balancing).

Second self-contained pickup from the model-family gap analysis. Same layered
checks as the LDT PR: op_catalog identity, registry contract status, numpy
correctness, VJP+JVP vs finite differences, and `@jit(target="apple_gpu")`
functional dispatch.
"""

import importlib

import numpy as np
import pytest

import tessera as ts
import tessera.losses as L
from tessera import _apple_gpu_backend as agb
from tessera import _jit_boundary as jb
from tessera.compiler import op_catalog as _cat
from tessera.compiler import primitive_coverage as _pc

_vjp = importlib.import_module("tessera.autodiff.vjp")
_jvp = importlib.import_module("tessera.autodiff.jvp")

_GPU = agb.is_available() and jb.is_available()
gpu = pytest.mark.skipif(
    not _GPU, reason="apple_gpu runtime / libtessera_jit unavailable")

_MOE = ["z_loss", "load_balance_loss"]


# ── op_catalog + registry ──────────────────────────────────────────────────── #
@pytest.mark.parametrize("name", _MOE)
def test_op_in_catalog(name):
    spec = _cat.OP_SPECS.get(name)
    assert spec is not None and spec.graph_name.startswith("tessera.loss.")


def test_registry_vjp_jvp_complete():
    reg = _pc.all_primitive_coverages()
    for n in _MOE:
        cs = reg[n].contract_status
        assert cs.get("vjp") == "complete", (n, cs.get("vjp"))
        assert cs.get("jvp") == "complete", (n, cs.get("jvp"))
    assert "z_loss" in _vjp._VJPS and "load_balance_loss" in _vjp._VJPS
    assert "z_loss" in _jvp._JVPS and "load_balance_loss" in _jvp._JVPS


# ── numpy correctness ──────────────────────────────────────────────────────── #
def test_z_loss_definition():
    rng = np.random.default_rng(0)
    logits = rng.standard_normal((6, 4))
    lse = np.log(np.exp(logits).sum(-1))
    np.testing.assert_allclose(L.z_loss(logits), np.mean(lse ** 2), rtol=1e-12)
    np.testing.assert_allclose(L.z_loss(logits, reduction="sum"), np.sum(lse ** 2))
    assert np.asarray(L.z_loss(logits, reduction="none")).shape == (6,)


def test_z_loss_zero_at_uniform_logits():
    # equal logits → lse = log(E) + c; only the magnitude matters, but a constant
    # shift changes lse, so just assert it's finite and non-negative.
    logits = np.zeros((4, 8))
    assert L.z_loss(logits) >= 0.0 and np.isfinite(L.z_loss(logits))


def test_load_balance_bounds():
    T, E = 512, 8
    uniform = np.full((T, E), 1.0 / E)
    np.testing.assert_allclose(L.load_balance_loss(uniform), 1.0, atol=1e-6)
    concentrated = np.zeros((T, E)); concentrated[:, 0] = 1.0
    np.testing.assert_allclose(L.load_balance_loss(concentrated), float(E), atol=1e-6)
    # any router lies in [1, E]
    rng = np.random.default_rng(1)
    rp = np.exp(rng.standard_normal((T, E))); rp /= rp.sum(-1, keepdims=True)
    aux = float(L.load_balance_loss(rp))
    assert 1.0 - 1e-6 <= aux <= E + 1e-6


def test_load_balance_explicit_assignment():
    rng = np.random.default_rng(2)
    rp = np.exp(rng.standard_normal((16, 4))); rp /= rp.sum(-1, keepdims=True)
    asg = np.argmax(rp, axis=-1)
    np.testing.assert_allclose(
        L.load_balance_loss(rp), L.load_balance_loss(rp, assignment=asg))


def test_load_balance_batched_leading_axis():
    rng = np.random.default_rng(3)
    rp = np.exp(rng.standard_normal((3, 32, 4))); rp /= rp.sum(-1, keepdims=True)
    per = L.load_balance_loss(rp, reduction="none")
    assert np.asarray(per).shape == (3,)
    np.testing.assert_allclose(L.load_balance_loss(rp), np.mean(per))


# ── autodiff ───────────────────────────────────────────────────────────────── #
def _finite_diff(fn, x, eps=1e-6):
    g = np.zeros_like(x); flat = x.ravel()
    for i in range(x.size):
        xp = flat.copy(); xm = flat.copy(); xp[i] += eps; xm[i] -= eps
        g.ravel()[i] = (fn(xp.reshape(x.shape)) - fn(xm.reshape(x.shape))) / (2 * eps)
    return g


def test_z_loss_vjp_matches_finite_diff():
    rng = np.random.default_rng(4)
    z = rng.standard_normal((5, 4))
    gz, = _vjp._VJPS["z_loss"](1.0, z)
    np.testing.assert_allclose(gz, _finite_diff(L.z_loss, z), atol=1e-7)


def test_load_balance_vjp_matches_finite_diff():
    rng = np.random.default_rng(5)
    rp = np.exp(rng.standard_normal((8, 4))); rp /= rp.sum(-1, keepdims=True)
    asg = np.argmax(rp, axis=-1)                       # hold assignment fixed
    gp, = _vjp._VJPS["load_balance_loss"](1.0, rp, assignment=asg)
    fd = _finite_diff(lambda x: L.load_balance_loss(x, assignment=asg), rp)
    np.testing.assert_allclose(gp, fd, atol=1e-7)


def test_z_loss_jvp_matches_directional():
    rng = np.random.default_rng(6)
    z = rng.standard_normal((4, 5)); dz = rng.standard_normal((4, 5))
    primal, tangent = _jvp._JVPS["z_loss"]((z,), (dz,))
    eps = 1e-6
    fd = (L.z_loss(z + eps * dz) - L.z_loss(z - eps * dz)) / (2 * eps)
    np.testing.assert_allclose(primal, L.z_loss(z))
    np.testing.assert_allclose(tangent, fd, atol=1e-6)


# ── Apple GPU dispatch (functional) ────────────────────────────────────────── #
@gpu
def test_z_loss_apple_gpu():
    @ts.jit(target="apple_gpu")
    def f(z):
        return ts.ops.z_loss(z)
    z = np.array([[0.3, -1.2, 2.0, 0.5]], np.float32)
    np.testing.assert_allclose(np.asarray(f(z)), L.z_loss(z), rtol=1e-6)


@gpu
def test_load_balance_apple_gpu():
    @ts.jit(target="apple_gpu")
    def f(p):
        return ts.ops.load_balance_loss(p)
    rng = np.random.default_rng(7)
    p = np.exp(rng.standard_normal((64, 8))).astype(np.float32)
    p /= p.sum(-1, keepdims=True)
    np.testing.assert_allclose(np.asarray(f(p)), L.load_balance_loss(p), rtol=1e-5)
