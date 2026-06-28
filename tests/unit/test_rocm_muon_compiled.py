"""Compiler-generated Muon on gfx1151 (P3 optimizer tail of
S_SERIES_GAP_CLOSURE_PLAN) — momentum then the orthogonal polar factor U·Vh of
the momentum matrix from the gfx1151 SVD kernel; the small U@Vh + momentum/sgd
run on host. <2-D params normalize. Reachable via
`compiler_path="rocm_muon_compiled"`. Validated vs optim.muon on gfx1151.
Skip-clean: tessera-opt not built / no GPU.
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


def _art(rt, p, g, v):
    return rt.RuntimeArtifact(metadata={
        "target": "rocm", "compiler_path": "rocm_muon_compiled",
        "executable": True, "execution_kind": "native_gpu",
        "arg_names": ["a0", "a1", "a2"], "output_name": "o",
        "ops": [{"op_name": "tessera.muon", "result": "o",
                 "operands": ["a0", "a1", "a2"],
                 "kwargs": {"lr": 1e-2, "momentum": 0.95, "extras": ["v"]}}],
    })


@pytest.mark.parametrize("shape", [(8, 5), (5, 8), (6, 6)])
def test_muon_matrix_multistep(shape):
    rt = _rocm_or_skip()
    rng = np.random.default_rng(13)
    p = rng.standard_normal(shape).astype(np.float32)
    v = np.zeros(shape, np.float32)
    state = None
    for _ in range(4):
        g = rng.standard_normal(shape).astype(np.float32)
        res = rt.launch(_art(rt, p, g, v), (p, g, v))
        assert res["ok"] is True, res.get("reason")
        assert res["compiler_path"] == "rocm_muon_compiled"
        pn, v = (np.asarray(x) for x in res["output"])
        rp, state = optim.muon(p, g, state, lr=1e-2, momentum=0.95)
        np.testing.assert_allclose(pn, np.asarray(rp), atol=1e-4)
        p = pn


def test_muon_vector_normalizes():
    rt = _rocm_or_skip()
    rng = np.random.default_rng(5)
    p = rng.standard_normal((7,)).astype(np.float32)
    g = rng.standard_normal((7,)).astype(np.float32)
    v = np.zeros((7,), np.float32)
    res = rt.launch(_art(rt, p, g, v), (p, g, v))
    rp, _ = optim.muon(p, g, None, lr=1e-2, momentum=0.95)
    np.testing.assert_allclose(np.asarray(res["output"][0]), np.asarray(rp),
                               atol=1e-5)
