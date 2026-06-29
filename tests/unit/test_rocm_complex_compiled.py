"""Compiler-generated complex arithmetic on gfx1151 (P5 of
S_SERIES_GAP_CLOSURE_PLAN) — the 9 pointwise Visual-Complex-Analysis ops over
interleaved-f32 [...,2], composed on the COMPILER-GENERATED unary / binary /
atan2 kernels (host packs the interleave). Reachable via
`compiler_path="rocm_complex_compiled"`. Validated vs tessera.complex on
gfx1151. Skip-clean: tessera-opt not built / no GPU.
"""

from __future__ import annotations

import numpy as np
import pytest

import tessera.complex as C


def _rocm_or_skip():
    from tessera import runtime as rt
    if rt._tessera_opt_path() is None:
        pytest.skip("tessera-opt not built")
    if not rt._rocm_wmma_runtime_available():
        pytest.skip("no usable AMD GPU")
    return rt


def _il(z):
    return np.stack([z.real.astype(np.float32), z.imag.astype(np.float32)],
                    axis=-1)


def _ref(scalar):
    if hasattr(scalar, "re"):
        return np.asarray(scalar.re) + 1j * np.asarray(scalar.im)
    return np.asarray(scalar)


def _run(rt, op, *zs):
    names = [f"a{i}" for i in range(len(zs))]
    art = rt.RuntimeArtifact(metadata={
        "target": "rocm", "compiler_path": "rocm_complex_compiled",
        "executable": True, "execution_kind": "native_gpu",
        "arg_names": names, "output_name": "o",
        "ops": [{"op_name": op, "result": "o", "operands": names,
                 "kwargs": {}}]})
    res = rt.launch(art, tuple(_il(z) for z in zs))
    assert res["ok"] is True, res.get("reason")
    assert res["compiler_path"] == "rocm_complex_compiled"
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
    rt = _rocm_or_skip()
    z, w = _Z()
    out = _run(rt, op, *((z, w) if binary else (z,)))
    got = out[..., 0] + 1j * out[..., 1]
    want = _ref(ref(z, w) if binary else ref(z))
    np.testing.assert_allclose(got, want, atol=1e-4, rtol=1e-4)


def test_complex_pow():
    rt = _rocm_or_skip()
    z, w = _Z()
    out = _run(rt, "tessera.complex_pow", z, w)
    got = out[..., 0] + 1j * out[..., 1]
    np.testing.assert_allclose(got, _ref(C.complex_pow(z, w)), rtol=3e-3,
                               atol=3e-3)


@pytest.mark.parametrize("op,ref", [
    ("tessera.complex_abs", C.complex_abs),
    ("tessera.complex_arg", C.complex_arg),
])
def test_complex_returning_real(op, ref):
    rt = _rocm_or_skip()
    z, _ = _Z()
    out = _run(rt, op, z)
    np.testing.assert_allclose(out, np.asarray(ref(z)), atol=1e-4, rtol=1e-4)
