"""Compiler-generated MoE compute on x86 AVX-512 (moe PR) — the routed per-token
expert matmuls (top-1). Routing (argmax / round-robin) is resolved on host; the
device kernel runs the expert GEMVs. Reachable via
`compiler_path="x86_moe_compiled"`. Validated vs tessera.ops.moe. Skip-clean: x86
lib not built.
"""

from __future__ import annotations

import numpy as np
import pytest

import tessera


def _rt_or_skip():
    from tessera import runtime as rt
    if not rt._x86_elementwise_available():
        pytest.skip("libtessera_x86_elementwise.so not built/loadable")
    return rt


def _art(rt, operands, extras):
    names = [f"a{i}" for i in range(len(operands))]
    return rt.RuntimeArtifact(metadata={
        "target": "x86", "compiler_path": "x86_moe_compiled",
        "executable": True, "execution_kind": "native_cpu",
        "arg_names": names, "output_name": "o",
        "ops": [{"op_name": "tessera.moe", "result": "o", "operands": names,
                 "kwargs": {"extras": extras}}],
    })


def test_moe_round_robin():
    rt = _rt_or_skip()
    rng = np.random.default_rng(1)
    x = rng.standard_normal((10, 5)).astype(np.float32)
    experts = rng.standard_normal((4, 5, 6)).astype(np.float32)
    res = rt.launch(_art(rt, [x, experts], []), (x, experts))
    assert res["ok"] is True, res.get("reason")
    assert res["compiler_path"] == "x86_moe_compiled"
    np.testing.assert_allclose(np.asarray(res["output"]),
                               np.asarray(tessera.ops.moe(x, experts)), atol=1e-4)


def test_moe_scores():
    rt = _rt_or_skip()
    rng = np.random.default_rng(2)
    x = rng.standard_normal((10, 5)).astype(np.float32)
    experts = rng.standard_normal((4, 5, 6)).astype(np.float32)
    scores = rng.standard_normal((10, 4)).astype(np.float32)
    res = rt.launch(_art(rt, [x, experts, scores], ["scores"]), (x, experts, scores))
    assert res["ok"] is True, res.get("reason")
    np.testing.assert_allclose(
        np.asarray(res["output"]),
        np.asarray(tessera.ops.moe(x, experts, scores=scores)), atol=1e-4)


def test_moe_route():
    rt = _rt_or_skip()
    rng = np.random.default_rng(3)
    x = rng.standard_normal((10, 5)).astype(np.float32)
    experts = rng.standard_normal((4, 5, 6)).astype(np.float32)
    route = rng.integers(0, 4, size=10).astype(np.int64)
    res = rt.launch(_art(rt, [x, experts, route], ["route"]), (x, experts, route))
    assert res["ok"] is True, res.get("reason")
    np.testing.assert_allclose(
        np.asarray(res["output"]),
        np.asarray(tessera.ops.moe(x, experts, route=route)), atol=1e-4)


def test_moe_single_expert_nd():
    rt = _rt_or_skip()
    rng = np.random.default_rng(4)
    x = rng.standard_normal((2, 3, 5)).astype(np.float32)
    w = rng.standard_normal((5, 6)).astype(np.float32)
    res = rt.launch(_art(rt, [x, w], []), (x, w))
    assert res["ok"] is True, res.get("reason")
    out = np.asarray(res["output"])
    ref = np.asarray(tessera.ops.moe(x, w))
    assert out.shape == ref.shape
    np.testing.assert_allclose(out, ref, atol=1e-4)
