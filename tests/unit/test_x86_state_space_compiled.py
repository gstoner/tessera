"""Compiler-generated state-space scan on x86 AVX-512 (state_space PR) —
selective_ssm (Mamba2). A single fused selective scan, vectorized over the state
dim N. Reachable via `compiler_path="x86_selective_ssm_compiled"`. Validated vs
the tessera.ops.selective_ssm numpy reference. Skip-clean: x86 lib not built.
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
        "target": "x86", "compiler_path": "x86_selective_ssm_compiled",
        "executable": True, "execution_kind": "native_cpu",
        "arg_names": names, "output_name": "o",
        "ops": [{"op_name": "tessera.selective_ssm", "result": "o",
                 "operands": names, "kwargs": {"extras": extras}}],
    })


def _inputs(rng, bsz, s, d, n, a_1d=False):
    x = rng.standard_normal((bsz, s, d)).astype(np.float32)
    a = (-rng.random((d,) if a_1d else (d, n))).astype(np.float32)
    b = rng.standard_normal((bsz, s, n)).astype(np.float32)
    c = rng.standard_normal((bsz, s, n)).astype(np.float32)
    delta = (rng.random((bsz, s, d)) * 0.5).astype(np.float32)
    return x, a, b, c, delta


@pytest.mark.parametrize("n", [16, 10])   # multiple-of-16 + scalar tail
def test_selective_ssm(n):
    rt = _rt_or_skip()
    rng = np.random.default_rng(1 + n)
    x, a, b, c, delta = _inputs(rng, 2, 7, 4, n)
    res = rt.launch(_art(rt, [x, a, b, c, delta], []), (x, a, b, c, delta))
    assert res["ok"] is True, res.get("reason")
    assert res["compiler_path"] == "x86_selective_ssm_compiled"
    ref = np.asarray(tessera.ops.selective_ssm(x, a, b, c, delta))
    np.testing.assert_allclose(np.asarray(res["output"]), ref, atol=2e-4)


def test_selective_ssm_a_1d():
    rt = _rt_or_skip()
    rng = np.random.default_rng(5)
    x, a, b, c, delta = _inputs(rng, 2, 6, 4, 16, a_1d=True)
    res = rt.launch(_art(rt, [x, a, b, c, delta], []), (x, a, b, c, delta))
    assert res["ok"] is True, res.get("reason")
    ref = np.asarray(tessera.ops.selective_ssm(x, a, b, c, delta))
    np.testing.assert_allclose(np.asarray(res["output"]), ref, atol=2e-4)


def test_selective_ssm_gate_state():
    rt = _rt_or_skip()
    rng = np.random.default_rng(9)
    x, a, b, c, delta = _inputs(rng, 2, 7, 4, 16)
    gate = rng.standard_normal((2, 7, 4)).astype(np.float32)
    state = rng.standard_normal((2, 4, 16)).astype(np.float32)
    res = rt.launch(_art(rt, [x, a, b, c, delta, gate, state],
                         ["gate", "state"]),
                    (x, a, b, c, delta, gate, state))
    assert res["ok"] is True, res.get("reason")
    ref = np.asarray(tessera.ops.selective_ssm(x, a, b, c, delta, gate=gate,
                                               state=state))
    np.testing.assert_allclose(np.asarray(res["output"]), ref, atol=2e-4)
