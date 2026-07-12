"""Compiler-generated complex arithmetic on Apple GPU (M7 follow-up).

The 9 pointwise Visual-Complex-Analysis ops over interleaved-f32 [...,2] compose
on the Apple GPU unary / binary / atan2 lanes; the geometric/certificate ops
reuse the tessera.complex reference (the same path x86/ROCm take). Reachable via
``compiler_path="apple_gpu_complex_compiled"``. Validated vs tessera.complex —
parity with test_x86_complex_compiled. The Apple GPU dispatchers fall back to
numpy when Metal is unavailable, so this runs everywhere (no skip).
"""

from __future__ import annotations

import numpy as np
import pytest

import tessera.complex as C
from tessera import runtime as rt


def _il(z):
    return np.stack([z.real.astype(np.float32), z.imag.astype(np.float32)],
                    axis=-1)


def _ref(scalar):
    if hasattr(scalar, "re"):
        return np.asarray(scalar.re) + 1j * np.asarray(scalar.im)
    return np.asarray(scalar)


def _run(op, *zs, raw: bool = False, **kwargs):
    names = [f"a{i}" for i in range(len(zs))]
    art = rt.RuntimeArtifact(metadata={
        "target": "apple_gpu", "compiler_path": "apple_gpu_complex_compiled",
        "executable": True, "execution_kind": "native_gpu",
        "arg_names": names, "output_name": "o",
        "ops": [{"op_name": op, "result": "o", "operands": names,
                 "kwargs": dict(kwargs)}]})
    args = zs if raw else tuple(_il(z) for z in zs)
    res = rt.launch(art, args)
    assert res["ok"] is True, res.get("reason")
    assert res["compiler_path"] == "apple_gpu_complex_compiled"
    assert res["execution_kind"] == "native_gpu"
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
    z, w = _Z()
    out = _run(op, *((z, w) if binary else (z,)))
    got = out[..., 0] + 1j * out[..., 1]
    want = _ref(ref(z, w) if binary else ref(z))
    np.testing.assert_allclose(got, want, atol=2e-5, rtol=2e-5)


def test_complex_pow():
    """exp(w·log(z)) chains f32 transcendentals — looser tolerance."""
    z, w = _Z()
    out = _run("tessera.complex_pow", z, w)
    got = out[..., 0] + 1j * out[..., 1]
    np.testing.assert_allclose(got, _ref(C.complex_pow(z, w)), rtol=2e-3,
                               atol=2e-3)


@pytest.mark.parametrize("op,ref", [
    ("tessera.complex_abs", C.complex_abs),
    ("tessera.complex_arg", C.complex_arg),
])
def test_complex_returning_real(op, ref):
    z, _ = _Z()
    out = _run(op, z)        # real-valued output [...]
    np.testing.assert_allclose(out, np.asarray(ref(z)), atol=2e-5, rtol=2e-5)


def test_complex_stencil_and_certificate_ops():
    field = np.arange(25, dtype=np.float32).reshape(5, 5)
    np.testing.assert_allclose(
        _run("tessera.laplacian_2d", field, raw=True, dx=1.0),
        C.laplacian_2d(field, dx=1.0),
    )

    def f(z):
        return z * z

    z0 = 0.5 + 0.25j
    for op in ("check_cauchy_riemann", "conformal_jacobian", "dz", "dbar"):
        got = _run(f"tessera.{op}", f, z0, raw=True, h=1e-5)
        exp = getattr(C, op)(f, z0, h=1e-5)
        if isinstance(exp, tuple):
            assert got[0] == exp[0]
            np.testing.assert_allclose(got[1], exp[1])
        else:
            np.testing.assert_allclose(got, exp)


def test_complex_projective_and_domain_ops():
    z1, z2, z3, z4 = 0.0 + 0.0j, 1.0 + 0.0j, 0.5 + 0.5j, 0.2 + 0.8j
    got = _run("tessera.cross_ratio", z1, z2, z3, z4, raw=True)
    np.testing.assert_allclose(
        np.asarray([got.real, got.imag], np.float32),
        np.asarray([C.cross_ratio(z1, z2, z3, z4).real,
                    C.cross_ratio(z1, z2, z3, z4).imag], np.float32),
    )
    assert _run("tessera.is_concyclic", z1, z2, z3, z4, raw=True) == \
        C.is_concyclic(z1, z2, z3, z4)

    src = (0.0 + 0.0j, 1.0 + 0.0j, 1.0j)
    dst = (1.0 + 0.0j, 2.0 + 0.0j, 1.0 + 1.0j)
    coeffs = _run("tessera.mobius_from_three_points", src, dst, raw=True)
    exp = C.mobius_from_three_points(src, dst)
    np.testing.assert_allclose(
        np.asarray([[c.real, c.imag] for c in coeffs], np.float32),
        np.asarray([[c.real, c.imag] for c in exp], np.float32),
    )

    p = np.array([0.0, 0.0, 1.0], np.float32)
    q = np.array([0.2, 0.1, 0.97], np.float32)
    q = q / np.linalg.norm(q)
    np.testing.assert_allclose(
        _run("tessera.conformal_energy_on_sphere", p, q, raw=True),
        C.conformal_energy_on_sphere(p, q),
    )


def test_conformal_mobius_matches_reference():
    # mobius f(z)=(az+b)/(cz+d) on the apple_gpu_conformal_compiled lane —
    # composed on the interleaved-f32 complex_mul/complex_div lanes.
    rng = np.random.default_rng(5)
    z = (rng.standard_normal((4, 3)) + 1j * rng.standard_normal((4, 3))).astype(
        np.complex64)
    a, b, c, d = (complex(rng.standard_normal(), rng.standard_normal())
                  for _ in range(4))
    names = ["z", "a", "b", "c", "d"]
    art = rt.RuntimeArtifact(metadata={
        "target": "apple_gpu", "compiler_path": "apple_gpu_conformal_compiled",
        "executable": True, "execution_kind": "native_gpu",
        "arg_names": names, "output_name": "o",
        "ops": [{"op_name": "tessera.mobius", "result": "o",
                 "operands": names, "kwargs": {}}]})

    def _ils(v):  # interleave a python complex scalar -> [2] f32
        return np.array([v.real, v.imag], np.float32)
    res = rt.launch(art, (_il(z), _ils(a), _ils(b), _ils(c), _ils(d)))
    assert res["ok"] is True, res.get("reason")
    assert res["compiler_path"] == "apple_gpu_conformal_compiled"
    got = np.asarray(res["output"])
    got = got[..., 0] + 1j * got[..., 1]
    np.testing.assert_allclose(got, _ref(C.mobius(z, a, b, c, d)),
                               atol=2e-5, rtol=2e-5)


def test_conformal_stereographic_matches_reference():
    # stereographic: sphere 3-vector [...,3] -> C on the apple_gpu_conformal_
    # device_verified_jit lane (binary-div lane); matches tessera.complex.stereographic.
    p = np.array([[0.0, 0.0, 0.9], [0.2, 0.1, 0.5], [0.3, -0.4, -0.2]], np.float32)
    art = rt.RuntimeArtifact(metadata={
        "target": "apple_gpu", "compiler_path": "apple_gpu_conformal_compiled",
        "executable": True, "execution_kind": "native_gpu",
        "arg_names": ["p"], "output_name": "o",
        "ops": [{"op_name": "tessera.stereographic", "result": "o",
                 "operands": ["p"], "kwargs": {}}]})
    res = rt.launch(art, (p,))
    assert res["ok"] is True, res.get("reason")
    assert res["compiler_path"] == "apple_gpu_conformal_compiled"
    got = np.asarray(res["output"])
    got = got[..., 0] + 1j * got[..., 1]
    np.testing.assert_allclose(got, _ref(C.stereographic(p)), atol=2e-5, rtol=2e-5)
