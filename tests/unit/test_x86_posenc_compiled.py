"""x86 position-encoding lanes — interleaved-pair rope and the ALiBi bias
generator, loaded from libtessera_x86_elementwise.so. The CPU analog of the
ROCm rope/alibi lanes, matching their op signatures.

Reachable through `runtime.launch()` via `compiler_path="x86_rope_compiled"`
(operands x, theta both [.., D], D even) and `"x86_alibi_compiled"` (num_heads /
seq_len kwargs, optional slopes operand). f32; validated vs numpy at 2e-5.

Skip-clean: libtessera_x86_elementwise.so absent.
"""

from __future__ import annotations

import numpy as np
import pytest


def _x86_or_skip():
    from tessera import runtime as rt
    if not rt._x86_elementwise_available():
        pytest.skip("libtessera_x86_elementwise.so not built/loadable")
    return rt


def _rope_artifact(rt):
    return rt.RuntimeArtifact(metadata={
        "target": "x86", "compiler_path": "x86_rope_compiled",
        "executable": True, "execution_kind": "native_cpu",
        "arg_names": ["x", "theta"], "output_name": "o",
        "ops": [{"op_name": "tessera.rope", "result": "o",
                 "operands": ["x", "theta"], "kwargs": {}}],
    })


def _alibi_artifact(rt, num_heads, seq_len, *, with_slopes=False):
    operands = ["slopes"] if with_slopes else []
    return rt.RuntimeArtifact(metadata={
        "target": "x86", "compiler_path": "x86_alibi_compiled",
        "executable": True, "execution_kind": "native_cpu",
        "arg_names": operands, "output_name": "o",
        "ops": [{"op_name": "tessera.alibi", "result": "o",
                 "operands": operands,
                 "kwargs": {"num_heads": num_heads, "seq_len": seq_len}}],
    })


def _rope_ref(x, theta):
    even, odd = x[..., 0::2], x[..., 1::2]
    a = theta[..., 0::2].astype(np.float32)
    out = np.empty_like(x)
    out[..., 0::2] = even * np.cos(a) - odd * np.sin(a)
    out[..., 1::2] = even * np.sin(a) + odd * np.cos(a)
    return out


def _make_theta(shape, base=10000.0):
    d = shape[-1]
    p = np.arange(d, dtype=np.float32)
    inv = 1.0 / (base ** ((p - (p % 2)) / d))   # theta[..,2p]=pos/base^(2p/D)
    pos = np.arange(int(np.prod(shape[:-1])) or 1, dtype=np.float32).reshape(
        *(shape[:-1] or (1,)), 1)
    return (pos * inv).astype(np.float32).reshape(shape)


@pytest.mark.parametrize("shape", [(4, 8), (2, 3, 16), (6, 64), (5, 34)])
def test_rope_matches_numpy(shape):
    rt = _x86_or_skip()
    rng = np.random.default_rng(13 + len(shape) + int(np.prod(shape)))
    x = rng.standard_normal(shape).astype(np.float32)
    theta = _make_theta(shape)
    res = rt.launch(_rope_artifact(rt), (x, theta))
    assert res["ok"] is True, res.get("reason")
    assert res["compiler_path"] == "x86_rope_compiled"
    np.testing.assert_allclose(np.asarray(res["output"]).astype(np.float32),
                               _rope_ref(x, theta), atol=2e-5, rtol=0)


def test_rope_zero_angle_identity():
    rt = _x86_or_skip()
    rng = np.random.default_rng(2)
    x = rng.standard_normal((4, 16)).astype(np.float32)
    theta = np.zeros((4, 16), np.float32)
    res = rt.launch(_rope_artifact(rt), (x, theta))
    assert res["ok"] is True, res.get("reason")
    np.testing.assert_allclose(np.asarray(res["output"]).astype(np.float32), x,
                               atol=1e-6)


def test_rope_odd_dim_rejected():
    rt = _x86_or_skip()
    x = np.zeros((4, 7), np.float32)
    with pytest.raises(ValueError, match="even innermost dim"):
        rt._execute_x86_compiled_rope(_rope_artifact(rt), (x, x))


def _alibi_ref(num_heads, seq_len, slopes=None):
    if slopes is None:
        slopes = 2.0 ** (-8.0 * np.arange(1, num_heads + 1, dtype=np.float32)
                         / num_heads)
    slopes = np.asarray(slopes, np.float32).reshape(num_heads, 1, 1)
    i = np.arange(seq_len).reshape(seq_len, 1)
    j = np.arange(seq_len).reshape(1, seq_len)
    return slopes * (j - i).astype(np.float32).reshape(1, seq_len, seq_len)


@pytest.mark.parametrize("h,s", [(3, 8), (4, 16), (2, 33), (1, 5)])
def test_alibi_default_ramp(h, s):
    rt = _x86_or_skip()
    res = rt.launch(_alibi_artifact(rt, h, s), ())
    assert res["ok"] is True, res.get("reason")
    assert res["compiler_path"] == "x86_alibi_compiled"
    np.testing.assert_allclose(np.asarray(res["output"]).astype(np.float32),
                               _alibi_ref(h, s), atol=2e-5, rtol=0)


def test_alibi_explicit_slopes():
    rt = _x86_or_skip()
    h, s = 3, 7
    slopes = np.array([0.5, 0.25, 0.125], np.float32)
    res = rt.launch(_alibi_artifact(rt, h, s, with_slopes=True), (slopes,))
    assert res["ok"] is True, res.get("reason")
    np.testing.assert_allclose(np.asarray(res["output"]).astype(np.float32),
                               _alibi_ref(h, s, slopes), atol=2e-5, rtol=0)


def test_alibi_diagonal_zero():
    rt = _x86_or_skip()
    res = rt.launch(_alibi_artifact(rt, 4, 6), ())
    assert res["ok"] is True, res.get("reason")
    out = np.asarray(res["output"]).astype(np.float32)
    for h in range(4):
        np.testing.assert_allclose(np.diag(out[h]), 0.0, atol=0)


def test_alibi_slopes_length_mismatch_rejected():
    rt = _x86_or_skip()
    bad = np.ones((5,), np.float32)
    with pytest.raises(ValueError, match="length num_heads"):
        rt._execute_x86_compiled_alibi(
            _alibi_artifact(rt, 3, 4, with_slopes=True), (bad,))
