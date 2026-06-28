"""Compiler-generated LAMB on gfx1151 (P3 optimizer tail of
S_SERIES_GAP_CLOSURE_PLAN) — the COMPILER-GENERATED adam kernel (lr=1/wd=0)
followed by the per-tensor trust ratio ‖p‖/‖update‖ applied on host. Reachable
via `compiler_path="rocm_lamb_compiled"`. Validated multi-step vs optim.lamb on
gfx1151. Skip-clean: tessera-opt not built / no GPU.
"""

from __future__ import annotations

import numpy as np
import pytest

from tessera import optim


def _rocm_or_skip():
    from tessera import runtime as rt
    if rt._tessera_opt_path() is None:
        pytest.skip("tessera-opt not built")
    if not rt._rocm_wmma_runtime_available():
        pytest.skip("no usable AMD GPU")
    return rt


def _art(rt, p, g, m, v, step):
    return rt.RuntimeArtifact(metadata={
        "target": "rocm", "compiler_path": "rocm_lamb_compiled",
        "executable": True, "execution_kind": "native_gpu",
        "arg_names": ["a0", "a1", "a2", "a3"], "output_name": "o",
        "ops": [{"op_name": "tessera.lamb", "result": "o",
                 "operands": ["a0", "a1", "a2", "a3"],
                 "kwargs": {"lr": 1e-2, "beta1": 0.9, "beta2": 0.999,
                            "eps": 1e-6, "weight_decay": 0.01, "step": step,
                            "extras": ["m", "v"]}}],
    })


SHAPE = (4, 8)


def test_lamb_multistep():
    rt = _rocm_or_skip()
    rng = np.random.default_rng(11)
    p = rng.standard_normal(SHAPE).astype(np.float32)
    m = np.zeros(SHAPE, np.float32)
    v = np.zeros(SHAPE, np.float32)
    state = None
    for step in range(1, 6):
        g = rng.standard_normal(SHAPE).astype(np.float32)
        res = rt.launch(_art(rt, p, g, m, v, step), (p, g, m, v))
        assert res["ok"] is True, res.get("reason")
        assert res["compiler_path"] == "rocm_lamb_compiled"
        pn, m, v = (np.asarray(x) for x in res["output"])
        rp, state = optim.lamb(p, g, state, lr=1e-2, beta1=0.9, beta2=0.999,
                               eps=1e-6, weight_decay=0.01)
        np.testing.assert_allclose(pn, np.asarray(rp), atol=1e-5)
        p = pn
