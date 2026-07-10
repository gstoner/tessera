"""grad_clip_norm lane on x86 (§5.5) — global gradient-norm clipping
``g * min(1, max_norm/||g||)`` composed on the AVX-512 reduce kernel (the L2
norm's global sum-of-squares) + host sqrt + scale. Reachable via
``compiler_path="x86_grad_clip_compiled"``. Validated vs optim.clip_grad_norm.
Skip-clean when libtessera_x86_elementwise.so is not built/loadable.
"""

from __future__ import annotations

import numpy as np
import pytest

from tessera import optim


def _rt_or_skip():
    from tessera import runtime as rt
    if not rt._x86_elementwise_available():
        pytest.skip("libtessera_x86_elementwise.so not built/loadable")
    return rt


def _art(rt, max_norm, norm_type=2.0):
    return rt.RuntimeArtifact(metadata={
        "target": "x86", "compiler_path": "x86_grad_clip_compiled",
        "executable": True, "execution_kind": "native_cpu",
        "arg_names": ["g"], "output_name": "o",
        "ops": [{"op_name": "tessera.grad_clip_norm", "result": "o",
                 "operands": ["g"],
                 "kwargs": {"max_norm": max_norm, "norm_type": norm_type}}]})


def _run(rt, g, max_norm, norm_type=2.0):
    res = rt.launch(_art(rt, max_norm, norm_type), (g,))
    assert res["ok"] is True, res.get("reason")
    assert res["compiler_path"] == "x86_grad_clip_compiled"
    assert res["execution_kind"] == "native_cpu"
    return np.asarray(res["output"], np.float32)


_RNG = np.random.default_rng(23)


@pytest.mark.parametrize("shape,max_norm", [
    ((256,), 1.0),
    ((16, 32), 0.5),
    ((1024,), 2.0),
])
def test_grad_clip_l2_matches_reference(shape, max_norm):
    rt = _rt_or_skip()
    g = (_RNG.standard_normal(shape) * 2.0).astype(np.float32)
    out = _run(rt, g, max_norm)
    ref = np.asarray(optim.clip_grad_norm(g, max_norm)[0], np.float32)
    np.testing.assert_allclose(out, ref, rtol=1e-5, atol=1e-5)


def test_grad_clip_inf_norm():
    rt = _rt_or_skip()
    g = (_RNG.standard_normal((100,)) * 3.0).astype(np.float32)
    out = _run(rt, g, 1.0, norm_type=float("inf"))
    ref = np.asarray(
        optim.clip_grad_norm(g, 1.0, norm_type=float("inf"))[0], np.float32)
    np.testing.assert_allclose(out, ref, rtol=1e-5, atol=1e-5)
