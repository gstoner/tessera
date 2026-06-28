"""Compiler-generated Muon on x86 AVX-512 (P3 optimizer tail of
S_SERIES_GAP_CLOSURE_PLAN) — momentum then the orthogonal polar factor U·Vh of
the momentum matrix, computed from the AVX-512 one-sided-Jacobi SVD kernel; the
small U@Vh + momentum/sgd run on host. <2-D params normalize. Reachable via
`compiler_path="x86_muon_compiled"`. Validated vs optim.muon. Skip-clean:
libtessera_x86_elementwise.so not built.
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


def _art(rt, p, g, v):
    return rt.RuntimeArtifact(metadata={
        "target": "x86", "compiler_path": "x86_muon_compiled",
        "executable": True, "execution_kind": "native_cpu",
        "arg_names": ["a0", "a1", "a2"], "output_name": "o",
        "ops": [{"op_name": "tessera.muon", "result": "o",
                 "operands": ["a0", "a1", "a2"],
                 "kwargs": {"lr": 1e-2, "momentum": 0.95, "extras": ["v"]}}],
    })


@pytest.mark.parametrize("shape", [(8, 5), (5, 8), (6, 6)])
def test_muon_matrix_multistep(shape):
    """Tall, wide, and square matrices exercise the SVD orthogonalization."""
    rt = _rt_or_skip()
    rng = np.random.default_rng(13)
    p = rng.standard_normal(shape).astype(np.float32)
    v = np.zeros(shape, np.float32)
    state = None
    for _ in range(4):
        g = rng.standard_normal(shape).astype(np.float32)
        res = rt.launch(_art(rt, p, g, v), (p, g, v))
        assert res["ok"] is True, res.get("reason")
        assert res["compiler_path"] == "x86_muon_compiled"
        pn, v = (np.asarray(x) for x in res["output"])
        rp, state = optim.muon(p, g, state, lr=1e-2, momentum=0.95)
        np.testing.assert_allclose(pn, np.asarray(rp), atol=2e-5)
        p = pn


def test_muon_vector_normalizes():
    """A <2-D param uses the normalize path, not SVD."""
    rt = _rt_or_skip()
    rng = np.random.default_rng(5)
    p = rng.standard_normal((7,)).astype(np.float32)
    g = rng.standard_normal((7,)).astype(np.float32)
    v = np.zeros((7,), np.float32)
    res = rt.launch(_art(rt, p, g, v), (p, g, v))
    rp, _ = optim.muon(p, g, None, lr=1e-2, momentum=0.95)
    np.testing.assert_allclose(np.asarray(res["output"][0]), np.asarray(rp),
                               atol=2e-6)
