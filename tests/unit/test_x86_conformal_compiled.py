"""Conformal-geometry lane on x86 AVX-512 (P5 of S_SERIES_GAP_CLOSURE_PLAN) —
mobius / stereographic. No new kernel: mobius f(z)=(az+b)/(cz+d) rides the
interleaved-f32 complex_mul/complex_div lane; stereographic projects a sphere
3-vector to ℂ via the AVX-512 binary div lane. Reachable via
`compiler_path="x86_conformal_compiled"`. Validated vs a numpy reference.
Skip-clean: libtessera_x86_elementwise.so not built.
"""

from __future__ import annotations

import numpy as np
import pytest


def _rt_or_skip():
    from tessera import runtime as rt
    if not rt._x86_elementwise_available():
        pytest.skip("libtessera_x86_elementwise.so not built/loadable")
    return rt


def _art(rt, op, kwargs, n_operands):
    names = [f"a{i}" for i in range(n_operands)]
    return rt.RuntimeArtifact(metadata={
        "target": "x86", "compiler_path": "x86_conformal_compiled",
        "executable": True, "execution_kind": "native_cpu",
        "arg_names": names, "output_name": "o",
        "ops": [{"op_name": op, "result": "o", "operands": names,
                 "kwargs": kwargs}]})


def _run(rt, op, kwargs, *arrs):
    res = rt.launch(_art(rt, op, kwargs, len(arrs)), arrs)
    assert res["ok"] is True, res.get("reason")
    assert res["compiler_path"] == "x86_conformal_compiled"
    return np.asarray(res["output"])


def _il(c):
    """complex array -> interleaved-f32 [...,2]."""
    return np.stack([c.real, c.imag], axis=-1).astype(np.float32)


def _cplx(il):
    """interleaved-f32 [...,2] -> complex array."""
    return (il[..., 0] + 1j * il[..., 1]).astype(np.complex64)


def test_mobius():
    rt = _rt_or_skip()
    rng = np.random.default_rng(0)
    z = rng.standard_normal((5, 2)).astype(np.float32)
    a = rng.standard_normal((5, 2)).astype(np.float32)
    b = rng.standard_normal((5, 2)).astype(np.float32)
    c = rng.standard_normal((5, 2)).astype(np.float32)
    d = rng.standard_normal((5, 2)).astype(np.float32)
    got = _run(rt, "tessera.mobius", {}, z, a, b, c, d)
    cz, ca, cb, cc, cd = (_cplx(v) for v in (z, a, b, c, d))
    ref = _il((ca * cz + cb) / (cc * cz + cd))
    np.testing.assert_allclose(got, ref, rtol=1e-4, atol=1e-5)


def test_stereographic():
    rt = _rt_or_skip()
    rng = np.random.default_rng(1)
    # points on the unit sphere, away from the north pole
    p = rng.standard_normal((6, 3)).astype(np.float32)
    p[:, 2] = np.clip(p[:, 2], -0.9, 0.9)
    p = (p / np.linalg.norm(p, axis=-1, keepdims=True)).astype(np.float32)
    got = _run(rt, "tessera.stereographic", {}, p)
    denom = (1.0 - p[:, 2]).astype(np.float32)
    ref = np.stack([p[:, 0] / denom, p[:, 1] / denom],
                   axis=-1).astype(np.float32)
    np.testing.assert_allclose(got, ref, rtol=1e-4, atol=1e-5)


def test_stereographic_north_pole_is_inf():
    rt = _rt_or_skip()
    p = np.array([[0.0, 0.0, 1.0]], np.float32)  # north pole -> ∞
    got = _run(rt, "tessera.stereographic", {}, p)
    assert np.isinf(got).all()
