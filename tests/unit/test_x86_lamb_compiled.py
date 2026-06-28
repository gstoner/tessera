"""Compiler-generated LAMB on x86 AVX-512 (P3 optimizer tail of
S_SERIES_GAP_CLOSURE_PLAN) — the AVX-512 adam kernel (lr=1/wd=0) followed by the
per-tensor trust ratio ‖p‖/‖update‖ applied on host (the layer-wise reduction
the fused elementwise lane can't do). Reachable via
`compiler_path="x86_lamb_compiled"`. Validated multi-step vs optim.lamb.
Skip-clean: libtessera_x86_elementwise.so not built.
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


def _art(rt, p, g, m, v, step, **kw):
    base = {"lr": 1e-2, "beta1": 0.9, "beta2": 0.999, "eps": 1e-6,
            "weight_decay": 0.01, "step": step, "extras": ["m", "v"]}
    base.update(kw)
    return rt.RuntimeArtifact(metadata={
        "target": "x86", "compiler_path": "x86_lamb_compiled",
        "executable": True, "execution_kind": "native_cpu",
        "arg_names": ["a0", "a1", "a2", "a3"], "output_name": "o",
        "ops": [{"op_name": "tessera.lamb", "result": "o",
                 "operands": ["a0", "a1", "a2", "a3"], "kwargs": base}],
    })


SHAPE = (4, 8)


def test_lamb_multistep():
    rt = _rt_or_skip()
    rng = np.random.default_rng(11)
    p = rng.standard_normal(SHAPE).astype(np.float32)
    m = np.zeros(SHAPE, np.float32)
    v = np.zeros(SHAPE, np.float32)
    state = None
    for step in range(1, 6):
        g = rng.standard_normal(SHAPE).astype(np.float32)
        res = rt.launch(_art(rt, p, g, m, v, step), (p, g, m, v))
        assert res["ok"] is True, res.get("reason")
        assert res["compiler_path"] == "x86_lamb_compiled"
        pn, m, v = (np.asarray(x) for x in res["output"])
        rp, state = optim.lamb(p, g, state, lr=1e-2, beta1=0.9, beta2=0.999,
                               eps=1e-6, weight_decay=0.01)
        np.testing.assert_allclose(pn, np.asarray(rp), atol=2e-6)
        p = pn


def test_lamb_zero_grad_trust_is_one():
    """With g=0 the update is 0, so the trust ratio falls back to 1 and the
    param only moves by the decoupled weight decay term."""
    rt = _rt_or_skip()
    p = np.ones(SHAPE, np.float32)
    z = np.zeros(SHAPE, np.float32)
    res = rt.launch(_art(rt, p, z, z, z, 1), (p, z, z, z))
    pn = np.asarray(res["output"][0])
    rp, _ = optim.lamb(p, z, None, lr=1e-2, beta1=0.9, beta2=0.999, eps=1e-6,
                       weight_decay=0.01)
    np.testing.assert_allclose(pn, np.asarray(rp), atol=2e-6)
