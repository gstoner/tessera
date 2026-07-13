"""Apple GPU Philox base lane.

Uniform uses the bit-exact Metal Philox-4x32-10 stream; normal and training
dropout apply their device transforms to that stream. The same artifact has a
reference fallback for non-Darwin hosts; distribution samplers and RNG-key
transformations remain host-structured by design.
"""

from __future__ import annotations

import numpy as np
import pytest

from tessera import rng as keyed_rng
from tessera import rng_device as R
from tessera import runtime as rt


def _art(op, kwargs, operands=()):
    names = [f"a{i}" for i in range(len(operands))]
    return rt.RuntimeArtifact(metadata={
        "target": "apple_gpu", "compiler_path": "apple_gpu_rng_compiled",
        "executable": True, "execution_kind": "native_gpu",
        "arg_names": names, "output_name": "o",
        "ops": [{"op_name": op, "result": "o", "operands": names,
                 "kwargs": kwargs}]})


def _launch(op, kwargs, operands=()):
    res = rt.launch(_art(op, kwargs, operands), operands)
    assert res["ok"] is True, res.get("reason")
    assert res["compiler_path"] == "apple_gpu_rng_compiled"
    return res["output"]


def test_rng_execution_matrix_declares_native_base_lane():
    from tessera.compiler.execution_matrix import lookup
    row = lookup("apple_gpu", "apple_gpu_rng_compiled")
    assert row is not None
    assert row.execution_kind == "native_gpu"


def test_rng_uniform_bit_exact_and_shape():
    out = np.asarray(_launch("tessera.rng_uniform",
                             {"seed": 42, "shape": [4, 5], "low": -1.0, "high": 3.0}))
    assert out.shape == (4, 5)
    np.testing.assert_array_equal(out.ravel(), R.uniform(42, 20, -1.0, 3.0))


def test_rng_uniform_accepts_lo_hi_alias():
    a = np.asarray(_launch("tessera.rng_uniform",
                           {"seed": 5, "shape": [50], "lo": 2.0, "hi": 4.0}))
    np.testing.assert_array_equal(a, R.uniform(5, 50, 2.0, 4.0))
    assert a.min() >= 2.0 and a.max() < 4.0


def test_rng_normal_bit_exact_and_stats():
    out = np.asarray(_launch("tessera.rng_normal",
                             {"seed": 7, "shape": [100], "mean": 2.0, "std": 0.5}))
    # The Philox words/counter schedule are canonical.  Box--Muller uses device
    # transcendental functions, so native Metal is numerically equivalent rather
    # than bit-identical to NumPy's libm implementation.
    np.testing.assert_allclose(out, R.normal(7, 100, 2.0, 0.5), rtol=2e-6, atol=2e-6)
    big = np.asarray(_launch("tessera.rng_normal", {"seed": 1, "shape": [200000]}))
    assert abs(big.mean()) < 1e-2 and abs(big.std() - 1.0) < 1e-2


def test_dropout_rate_scale_and_eval():
    x = np.ones((20000,), np.float32)
    d = np.asarray(_launch("tessera.dropout",
                           {"seed": 3, "p": 0.3, "training": True}, (x,)))
    assert abs((d > 0).mean() - 0.7) < 0.02
    np.testing.assert_allclose(d[d > 0][0], 1.0 / 0.7, atol=1e-5)
    de = np.asarray(_launch("tessera.dropout",
                            {"seed": 3, "p": 0.3, "training": False}, (x,)))
    np.testing.assert_array_equal(de, x)


def test_rng_determinism():
    a = _launch("tessera.rng_uniform", {"seed": 9, "shape": [1000]})
    b = _launch("tessera.rng_uniform", {"seed": 9, "shape": [1000]})
    np.testing.assert_array_equal(np.asarray(a), np.asarray(b))


def test_rng_base_ops_report_native_gpu_on_darwin():
    if __import__("sys").platform != "darwin" or not rt.DeviceTensor.is_metal():
        pytest.skip("requires an available Apple Metal runtime")
    for op, kwargs, operands in (
        ("tessera.rng_uniform", {"seed": 11, "shape": [16]}, ()),
        ("tessera.rng_normal", {"seed": 12, "shape": [16]}, ()),
        ("tessera.dropout", {"seed": 13, "p": 0.25, "training": True},
         (np.ones(16, np.float32),)),
    ):
        res = rt.launch(_art(op, kwargs, operands), operands)
        assert res["ok"] is True, res.get("reason")
        assert res["execution_kind"] == "native_gpu"
        assert res["execution_mode"] == "metal_runtime"


def test_rng_distribution_tail_matches_keyed_reference():
    seed = 31
    key = keyed_rng.RNGKey.from_seed(seed)
    for op, kwargs, operands, expected in (
        ("tessera.rng_bernoulli", {"seed": seed, "shape": [8], "p": 0.35}, (),
         keyed_rng.bernoulli(key, (8,), p=0.35)),
        ("tessera.rng_randint", {"seed": seed, "shape": [8], "low": 2, "high": 9}, (),
         keyed_rng.randint(key, (8,), low=2, high=9)),
        ("tessera.rng_truncated_normal", {"seed": seed, "shape": [8]}, (),
         keyed_rng.truncated_normal(key, (8,))),
        ("tessera.rng_gamma", {"seed": seed, "shape": [8], "concentration": 2.0}, (),
         keyed_rng.gamma(key, (8,), concentration=2.0)),
        ("tessera.rng_beta", {"seed": seed, "shape": [8], "alpha": 2.0, "beta": 3.0}, (),
         keyed_rng.beta(key, (8,), alpha=2.0, beta_param=3.0)),
        ("tessera.rng_poisson", {"seed": seed, "shape": [8], "rate": 4.0}, (),
         keyed_rng.poisson(key, (8,), rate=4.0)),
    ):
        np.testing.assert_array_equal(np.asarray(_launch(op, kwargs, operands)),
                                      expected)


def test_rng_distribution_operands_and_key_state_match_reference():
    seed = 37
    key = keyed_rng.RNGKey.from_seed(seed)
    logits = np.array([[0.1, 2.0, -1.0], [3.0, 0.0, 0.5]], np.float32)
    np.testing.assert_array_equal(
        np.asarray(_launch("tessera.rng_categorical", {"seed": seed}, (logits,))),
        keyed_rng.categorical(key, logits))
    alpha = np.array([0.5, 1.5, 2.5], np.float32)
    np.testing.assert_allclose(
        np.asarray(_launch("tessera.rng_dirichlet", {"seed": seed, "shape": [2]},
                           (alpha,))),
        keyed_rng.dirichlet(key, alpha, shape=(2,)))
    states = _launch("tessera.rng_split", {"seed": seed, "num": 3})
    assert states == tuple(k.to_state() for k in key.split(3))
    folded = _launch("tessera.rng_fold_in", {"seed": seed, "data": "rank0"})
    assert folded == key.fold_in("rank0").to_state()
