"""Compiler-generated device RNG on x86 AVX-512 (P6 of S_SERIES_GAP_CLOSURE_PLAN)
— counter-based Philox-4x32-10 (the JAX/cuRAND algorithm). The kernel produces
the uniform[0,1) bits; the host applies the distribution transform
(uniform-scale / Box-Muller normal / dropout mask). A SEPARATE deterministic
stream from tessera.rng (host numpy-Generator) — validated BIT-EXACTLY against
the tessera.rng_device numpy reference (same algorithm) + by statistics +
determinism. Reachable via `compiler_path="x86_rng_compiled"`. Skip-clean:
libtessera_x86_elementwise.so not built.
"""

from __future__ import annotations

import numpy as np
import pytest

from tessera import rng_device as R


def _rt_or_skip():
    from tessera import runtime as rt
    if not rt._x86_elementwise_available():
        pytest.skip("libtessera_x86_elementwise.so not built/loadable")
    return rt


def _art(rt, op, kwargs, operands=()):
    names = [f"a{i}" for i in range(len(operands))]
    return rt.RuntimeArtifact(metadata={
        "target": "x86", "compiler_path": "x86_rng_compiled",
        "executable": True, "execution_kind": "native_cpu",
        "arg_names": names, "output_name": "o",
        "ops": [{"op_name": op, "result": "o", "operands": names,
                 "kwargs": kwargs}]})


def test_rng_uniform_bit_exact_and_shape():
    rt = _rt_or_skip()
    res = rt.launch(_art(rt, "tessera.rng_uniform",
                         {"seed": 42, "shape": [4, 5], "low": -1.0, "high": 3.0}),
                    ())
    assert res["ok"] is True, res.get("reason")
    assert res["compiler_path"] == "x86_rng_compiled"
    out = np.asarray(res["output"])
    assert out.shape == (4, 5)
    np.testing.assert_array_equal(out.ravel(), R.uniform(42, 20, -1.0, 3.0))


def test_rng_uniform_statistics():
    rt = _rt_or_skip()
    out = np.asarray(rt.launch(_art(rt, "tessera.rng_uniform",
                                    {"seed": 7, "shape": [200000]}), ())["output"])
    assert abs(out.mean() - 0.5) < 5e-3
    assert abs(out.var() - 1.0 / 12.0) < 5e-3
    assert out.min() >= 0.0 and out.max() < 1.0


def test_rng_normal_bit_exact_and_stats():
    rt = _rt_or_skip()
    res = rt.launch(_art(rt, "tessera.rng_normal",
                         {"seed": 7, "shape": [100], "mean": 2.0, "std": 0.5}), ())
    np.testing.assert_array_equal(np.asarray(res["output"]),
                                  R.normal(7, 100, 2.0, 0.5))
    big = np.asarray(rt.launch(_art(rt, "tessera.rng_normal",
                                    {"seed": 1, "shape": [200000]}), ())["output"])
    assert abs(big.mean()) < 1e-2 and abs(big.std() - 1.0) < 1e-2


def test_dropout_rate_scale_and_eval():
    rt = _rt_or_skip()
    x = np.ones((20000,), np.float32)
    d = np.asarray(rt.launch(_art(rt, "tessera.dropout",
                                  {"seed": 3, "p": 0.3, "training": True},
                                  (x,)), (x,))["output"])
    assert abs((d > 0).mean() - 0.7) < 0.02            # ~keep prob
    np.testing.assert_allclose(d[d > 0][0], 1.0 / 0.7, atol=1e-5)  # inverted scale
    # eval mode -> identity
    de = np.asarray(rt.launch(_art(rt, "tessera.dropout",
                                   {"seed": 3, "p": 0.3, "training": False},
                                   (x,)), (x,))["output"])
    np.testing.assert_array_equal(de, x)


def test_rng_determinism():
    rt = _rt_or_skip()
    a = rt.launch(_art(rt, "tessera.rng_uniform", {"seed": 9, "shape": [1000]}), ())
    b = rt.launch(_art(rt, "tessera.rng_uniform", {"seed": 9, "shape": [1000]}), ())
    np.testing.assert_array_equal(np.asarray(a["output"]), np.asarray(b["output"]))
