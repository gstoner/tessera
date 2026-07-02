"""Compiler-generated complex arithmetic on x86 AVX-512 (P5 of
S_SERIES_GAP_CLOSURE_PLAN) — the 9 pointwise Visual-Complex-Analysis ops over
interleaved-f32 [...,2], composed on the AVX-512 transcendental / unary / binary
/ atan2 lanes (host packs the interleave). Reachable via
`compiler_path="x86_complex_compiled"`. Validated vs tessera.complex.
Skip-clean: libtessera_x86_elementwise.so not built.
"""

from __future__ import annotations

import numpy as np
import pytest

import tessera.complex as C


def _rt_or_skip():
    from tessera import runtime as rt
    if not rt._x86_elementwise_available():
        pytest.skip("libtessera_x86_elementwise.so not built/loadable")
    return rt


def _il(z):
    return np.stack([z.real.astype(np.float32), z.imag.astype(np.float32)],
                    axis=-1)


def _ref(scalar):
    if hasattr(scalar, "re"):
        return np.asarray(scalar.re) + 1j * np.asarray(scalar.im)
    return np.asarray(scalar)


def _run(rt, op, *zs, raw: bool = False, **kwargs):
    names = [f"a{i}" for i in range(len(zs))]
    art = rt.RuntimeArtifact(metadata={
        "target": "x86", "compiler_path": "x86_complex_compiled",
        "executable": True, "execution_kind": "native_cpu",
        "arg_names": names, "output_name": "o",
        "ops": [{"op_name": op, "result": "o", "operands": names,
                 "kwargs": dict(kwargs)}]})
    args = zs if raw else tuple(_il(z) for z in zs)
    res = rt.launch(art, args)
    assert res["ok"] is True, res.get("reason")
    assert res["compiler_path"] == "x86_complex_compiled"
    return np.asarray(res["output"])


def _Z():
    rng = np.random.default_rng(0)
    z = (rng.standard_normal((4, 5)) + 1j * rng.standard_normal((4, 5)))
    w = (rng.standard_normal((4, 5)) + 1j * rng.standard_normal((4, 5)))
    return z.astype(np.complex64), w.astype(np.complex64)


@pytest.mark.parametrize("op,ref,binary", [
    ("tessera.complex_mul", C.complex_mul, True),
    ("tessera.complex_div", C.complex_div, True),
    ("tessera.complex_conjugate", C.complex_conjugate, False),
    ("tessera.complex_exp", C.complex_exp, False),
    ("tessera.complex_log", C.complex_log, False),
    ("tessera.complex_sqrt", C.complex_sqrt, False),
])
def test_complex_returning_complex(op, ref, binary):
    rt = _rt_or_skip()
    z, w = _Z()
    out = _run(rt, op, *( (z, w) if binary else (z,) ))
    got = out[..., 0] + 1j * out[..., 1]
    want = _ref(ref(z, w) if binary else ref(z))
    np.testing.assert_allclose(got, want, atol=2e-5, rtol=2e-5)


def test_complex_pow():
    """exp(w·log(z)) chains f32 transcendentals — looser tolerance."""
    rt = _rt_or_skip()
    z, w = _Z()
    out = _run(rt, "tessera.complex_pow", z, w)
    got = out[..., 0] + 1j * out[..., 1]
    np.testing.assert_allclose(got, _ref(C.complex_pow(z, w)), rtol=2e-3,
                               atol=2e-3)


@pytest.mark.parametrize("op,ref", [
    ("tessera.complex_abs", C.complex_abs),
    ("tessera.complex_arg", C.complex_arg),
])
def test_complex_returning_real(op, ref):
    rt = _rt_or_skip()
    z, _ = _Z()
    out = _run(rt, op, z)        # real-valued output [...]
    np.testing.assert_allclose(out, np.asarray(ref(z)), atol=2e-5, rtol=2e-5)


def test_complex_stencil_and_certificate_ops():
    rt = _rt_or_skip()
    field = np.arange(25, dtype=np.float32).reshape(5, 5)
    np.testing.assert_allclose(
        _run(rt, "tessera.laplacian_2d", field, raw=True, dx=1.0),
        C.laplacian_2d(field, dx=1.0),
    )

    def f(z):
        return z * z

    z0 = 0.5 + 0.25j
    for op in ("check_cauchy_riemann", "conformal_jacobian", "dz", "dbar"):
        got = _run(rt, f"tessera.{op}", f, z0, raw=True, h=1e-5)
        exp = getattr(C, op)(f, z0, h=1e-5)
        if isinstance(exp, tuple):
            assert got[0] == exp[0]
            np.testing.assert_allclose(got[1], exp[1])
        else:
            np.testing.assert_allclose(got, exp)
