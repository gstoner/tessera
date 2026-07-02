"""Compiler-generated device RNG on gfx1151 (P6 of S_SERIES_GAP_CLOSURE_PLAN) —
counter-based Philox-4x32-10 (generate-rocm-philox-kernel). The kernel produces
the uniform[0,1) bits; the host applies the distribution transform. A SEPARATE
deterministic stream from tessera.rng (host numpy-Generator) — validated
BIT-EXACTLY against the tessera.rng_device numpy reference (same algorithm; also
bit-identical to the x86 kernel) + statistics + determinism. Reachable via
`compiler_path="rocm_rng_compiled"`. Skip-clean: tessera-opt not built / no GPU.
"""

from __future__ import annotations

import numpy as np
import pytest

from tessera import rng_device as R
from tessera import rng as keyed_rng


def _rocm_or_skip():
    from tessera import runtime as rt
    if rt._tessera_opt_path() is None:
        pytest.skip("tessera-opt not built")
    if not rt._rocm_wmma_runtime_available():
        pytest.skip("no usable AMD GPU")
    return rt


def _art(rt, op, kwargs, operands=()):
    names = [f"a{i}" for i in range(len(operands))]
    return rt.RuntimeArtifact(metadata={
        "target": "rocm", "compiler_path": "rocm_rng_compiled",
        "executable": True, "execution_kind": "native_gpu",
        "arg_names": names, "output_name": "o",
        "ops": [{"op_name": op, "result": "o", "operands": names,
                 "kwargs": kwargs}]})


def test_rng_uniform_bit_exact():
    rt = _rocm_or_skip()
    res = rt.launch(_art(rt, "tessera.rng_uniform",
                         {"seed": 42, "shape": [4, 5], "low": -1.0, "high": 3.0}),
                    ())
    assert res["ok"] is True, res.get("reason")
    assert res["compiler_path"] == "rocm_rng_compiled"
    out = np.asarray(res["output"])
    assert out.shape == (4, 5)
    np.testing.assert_array_equal(out.ravel(), R.uniform(42, 20, -1.0, 3.0))


def test_rng_normal_bit_exact_and_stats():
    rt = _rocm_or_skip()
    res = rt.launch(_art(rt, "tessera.rng_normal",
                         {"seed": 7, "shape": [100], "mean": 2.0, "std": 0.5}), ())
    np.testing.assert_array_equal(np.asarray(res["output"]),
                                  R.normal(7, 100, 2.0, 0.5))
    big = np.asarray(rt.launch(_art(rt, "tessera.rng_normal",
                                    {"seed": 1, "shape": [200000]}), ())["output"])
    assert abs(big.mean()) < 1e-2 and abs(big.std() - 1.0) < 1e-2


def test_dropout_rate_scale_and_eval():
    rt = _rocm_or_skip()
    x = np.ones((20000,), np.float32)
    d = np.asarray(rt.launch(_art(rt, "tessera.dropout",
                                  {"seed": 3, "p": 0.3, "training": True},
                                  (x,)), (x,))["output"])
    assert abs((d > 0).mean() - 0.7) < 0.02
    np.testing.assert_allclose(d[d > 0][0], 1.0 / 0.7, atol=1e-5)
    de = np.asarray(rt.launch(_art(rt, "tessera.dropout",
                                   {"seed": 3, "p": 0.3, "training": False},
                                   (x,)), (x,))["output"])
    np.testing.assert_array_equal(de, x)


def test_rng_determinism():
    rt = _rocm_or_skip()
    a = rt.launch(_art(rt, "tessera.rng_uniform", {"seed": 9, "shape": [1000]}), ())
    b = rt.launch(_art(rt, "tessera.rng_uniform", {"seed": 9, "shape": [1000]}), ())
    np.testing.assert_array_equal(np.asarray(a["output"]), np.asarray(b["output"]))


def test_rng_distribution_tail_matches_keyed_reference_on_gpu():
    rt = _rocm_or_skip()
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
        out = rt.launch(_art(rt, op, kwargs, operands), operands)
        assert out["ok"] is True, out.get("reason")
        np.testing.assert_array_equal(np.asarray(out["output"]), expected)


def test_rng_distribution_operands_and_key_state_match_reference_on_gpu():
    rt = _rocm_or_skip()
    seed = 37
    key = keyed_rng.RNGKey.from_seed(seed)
    logits = np.array([[0.1, 2.0, -1.0], [3.0, 0.0, 0.5]], np.float32)
    out = rt.launch(_art(rt, "tessera.rng_categorical", {"seed": seed}, (logits,)),
                    (logits,))
    np.testing.assert_array_equal(np.asarray(out["output"]),
                                  keyed_rng.categorical(key, logits))

    alpha = np.array([0.5, 1.5, 2.5], np.float32)
    out = rt.launch(_art(rt, "tessera.rng_dirichlet",
                         {"seed": seed, "shape": [2]}, (alpha,)), (alpha,))
    np.testing.assert_allclose(np.asarray(out["output"]),
                               keyed_rng.dirichlet(key, alpha, shape=(2,)))

    states = rt.launch(_art(rt, "tessera.rng_split", {"seed": seed, "num": 3}), ())
    assert states["output"] == tuple(k.to_state() for k in key.split(3))
    folded = rt.launch(_art(rt, "tessera.rng_fold_in",
                            {"seed": seed, "data": "rank0"}), ())
    assert folded["output"] == key.fold_in("rank0").to_state()
