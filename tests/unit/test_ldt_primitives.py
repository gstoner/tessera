"""LDT / lattice-reasoning primitives — count_nonzero, popcount, asymmetric_bce,
masked_categorical.

First PR of the LDT family (the gap analysis flagged these as the cheap,
self-contained adds). Each primitive is checked across the layers the new
compiler path cares about:

  1. **op_catalog** — the op has a Graph IR identity (so the frontend emits a
     stable name, not an opaque call).
  2. **primitive_coverage registry** — present with the right contract status
     (the three non-differentiable ops are vjp/jvp ``not_applicable``;
     ``asymmetric_bce`` is vjp+jvp ``complete``).
  3. **numpy reference** — `tessera.ops.*` math is correct.
  4. **autodiff** — `asymmetric_bce` VJP+JVP match finite differences.
  5. **Apple GPU dispatch** — each op executes correctly under
     `@jit(target="apple_gpu")` (Metal envelope ops run on Metal; these run via
     the numpy-fallback chain — *functional*, per the PR's scope).
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

# autodiff/__init__ rebinds the names `vjp`/`jvp` to the transform *functions*,
# shadowing the submodules — import the real modules (which carry the
# `_VJPS`/`_JVPS` registries) via importlib.
_vjp = importlib.import_module("tessera.autodiff.vjp")
_jvp = importlib.import_module("tessera.autodiff.jvp")

_GPU = agb.is_available() and jb.is_available()
gpu = pytest.mark.hardware_apple_gpu

_LDT = ["count_nonzero", "popcount", "asymmetric_bce", "masked_categorical"]


# ── 1. op_catalog identity ─────────────────────────────────────────────────── #
@pytest.mark.parametrize("name", _LDT)
def test_op_in_catalog(name):
    spec = _cat.OP_SPECS.get(name)
    assert spec is not None, f"{name} missing from op_catalog.OP_SPECS"
    assert spec.graph_name.startswith("tessera."), spec.graph_name


def test_catalog_effects():
    specs = _cat.OP_SPECS
    assert specs["masked_categorical"].effect == "random"  # sampling op
    assert specs["count_nonzero"].effect == "pure"
    assert specs["popcount"].effect == "pure"


# ── 2. registry contract status ────────────────────────────────────────────── #
def test_registry_contract_status():
    reg = _pc.all_primitive_coverages()
    for n in _LDT:
        assert n in reg, f"{n} not registered in primitive_coverage"
    # non-differentiable trio
    for n in ("count_nonzero", "popcount", "masked_categorical"):
        cs = reg[n].contract_status
        assert cs.get("vjp") == "non_differentiable", (n, cs.get("vjp"))
        assert cs.get("jvp") == "non_differentiable", (n, cs.get("jvp"))
    # differentiable loss
    cs = reg["asymmetric_bce"].contract_status
    assert cs.get("vjp") == "complete"
    assert cs.get("jvp") == "complete"


def test_autodiff_registered():
    assert "asymmetric_bce" in _vjp._VJPS
    assert "asymmetric_bce" in _jvp._JVPS


# ── 3. numpy reference correctness ─────────────────────────────────────────── #
def test_count_nonzero():
    x = np.array([[0.0, 1.0, 2.0], [0.0, 0.0, 3.0]], np.float32)
    np.testing.assert_array_equal(ts.ops.count_nonzero(x, axis=-1), [2, 1])
    assert int(ts.ops.count_nonzero(x)) == 3
    np.testing.assert_array_equal(
        ts.ops.count_nonzero(x, axis=-1, keepdims=True), [[2], [1]])


def test_popcount():
    b = np.array([0, 1, 2, 3, 7, 8, 255], np.int64)
    np.testing.assert_array_equal(ts.ops.popcount(b), [0, 1, 1, 2, 3, 1, 8])
    # matches Python int.bit_count for a random spread
    rng = np.random.default_rng(0)
    r = rng.integers(0, 2**31, size=64)
    expect = np.array([int(v).bit_count() for v in r])
    np.testing.assert_array_equal(ts.ops.popcount(r), expect)


def test_masked_categorical_greedy():
    logits = np.array([[1.0, 5.0, 2.0], [3.0, 0.0, 4.0]], np.float32)
    mask = np.array([[1, 0, 1], [1, 1, 0]], np.int32)   # mask out each row's max
    np.testing.assert_array_equal(ts.ops.masked_categorical(logits, mask), [2, 0])
    # all-candidates mask reduces to a plain argmax
    full = np.ones_like(mask)
    np.testing.assert_array_equal(
        ts.ops.masked_categorical(logits, full), np.argmax(logits, axis=-1))


def test_masked_categorical_sampling_respects_mask():
    logits = np.array([10.0, 0.0, 0.0, 0.0], np.float32)
    mask = np.array([0, 1, 1, 1], np.int32)             # the high-logit is masked
    # over many keys, the masked index 0 is never chosen.
    picks = {int(ts.ops.masked_categorical(logits, mask, key=k)) for k in range(50)}
    assert 0 not in picks and picks <= {1, 2, 3}


def test_asymmetric_bce_reduces_to_bce():
    rng = np.random.default_rng(1)
    z = rng.standard_normal((4, 5)); t = (rng.random((4, 5)) < 0.5).astype(float)
    np.testing.assert_allclose(
        L.asymmetric_bce(z, t, 1.0, 1.0), L.binary_cross_entropy_loss(z, t), rtol=1e-12)


def test_asymmetric_bce_weighting_direction():
    # one false-negative (t=1, z very negative) vs one false-positive (t=0, z+).
    z = np.array([-4.0, 4.0]); t = np.array([1.0, 0.0])
    base = L.asymmetric_bce(z, t, 1.0, 1.0)
    assert L.asymmetric_bce(z, t, 5.0, 1.0) > base   # up-weight false-negatives
    assert L.asymmetric_bce(z, t, 1.0, 5.0) > base   # up-weight false-positives
    # reductions
    assert L.asymmetric_bce(z, t, reduction="none").shape == (2,)
    assert np.isclose(L.asymmetric_bce(z, t, reduction="sum"),
                      L.asymmetric_bce(z, t, reduction="none").sum())


def test_asymmetric_bce_stable_large_logits():
    z = np.array([-1e3, 1e3]); t = np.array([0.0, 1.0])  # correct, large
    out = L.asymmetric_bce(z, t, 3.0, 2.0, reduction="none")
    assert np.all(np.isfinite(out)) and np.allclose(out, 0.0, atol=1e-6)


# ── 4. autodiff numeric checks ─────────────────────────────────────────────── #
def test_asymmetric_bce_vjp_matches_finite_diff():
    rng = np.random.default_rng(2)
    z = rng.standard_normal((4, 5)); t = (rng.random((4, 5)) < 0.5).astype(float)
    pw, nw = 2.0, 0.7
    gz, gt = _vjp._VJPS["asymmetric_bce"](1.0, z, t, pos_weight=pw, neg_weight=nw)
    eps = 1e-6
    num = np.zeros_like(z)
    flat = z.ravel()
    for i in range(z.size):
        zp = flat.copy(); zm = flat.copy(); zp[i] += eps; zm[i] -= eps
        num.ravel()[i] = (L.asymmetric_bce(zp.reshape(z.shape), t, pw, nw)
                          - L.asymmetric_bce(zm.reshape(z.shape), t, pw, nw)) / (2 * eps)
    np.testing.assert_allclose(gz, num, atol=1e-7)


def test_asymmetric_bce_jvp_matches_directional_derivative():
    rng = np.random.default_rng(3)
    z = rng.standard_normal((3, 4)); t = (rng.random((3, 4)) < 0.5).astype(float)
    dz = rng.standard_normal((3, 4)); dt = np.zeros_like(t)
    primal, tangent = _jvp._JVPS["asymmetric_bce"]((z, t), (dz, dt), pos_weight=1.5)
    eps = 1e-6
    fd = (L.asymmetric_bce(z + eps * dz, t, 1.5)
          - L.asymmetric_bce(z - eps * dz, t, 1.5)) / (2 * eps)
    np.testing.assert_allclose(tangent, fd, atol=1e-6)
    np.testing.assert_allclose(primal, L.asymmetric_bce(z, t, 1.5))


# ── 5. Apple GPU dispatch (functional) ─────────────────────────────────────── #
@gpu
def test_count_nonzero_apple_gpu():
    @ts.jit(target="apple_gpu")
    def f(x):
        return ts.ops.count_nonzero(x, axis=-1)
    x = np.array([[0.0, 1.0, 2.0], [0.0, 0.0, 3.0]], np.float32)
    np.testing.assert_array_equal(np.asarray(f(x)), [2, 1])


@gpu
def test_popcount_apple_gpu():
    @ts.jit(target="apple_gpu")
    def f(b):
        return ts.ops.popcount(b)
    b = np.array([0, 1, 3, 7, 255], np.int64)
    np.testing.assert_array_equal(np.asarray(f(b)), [0, 1, 2, 3, 8])


@gpu
def test_asymmetric_bce_apple_gpu():
    @ts.jit(target="apple_gpu")
    def f(z, t):
        return ts.ops.asymmetric_bce(z, t, pos_weight=2.0, neg_weight=0.5)
    z = np.array([0.3, -1.2, 2.0], np.float32); t = np.array([1.0, 0.0, 1.0], np.float32)
    np.testing.assert_allclose(np.asarray(f(z, t)), L.asymmetric_bce(z, t, 2.0, 0.5),
                               rtol=1e-6)


@gpu
def test_masked_categorical_apple_gpu():
    @ts.jit(target="apple_gpu")
    def f(lo, m):
        return ts.ops.masked_categorical(lo, m)
    lo = np.array([[1.0, 5.0, 2.0], [3.0, 0.0, 4.0]], np.float32)
    m = np.array([[1, 0, 1], [1, 1, 0]], np.int32)
    np.testing.assert_array_equal(np.asarray(f(lo, m)), [2, 0])
