"""x86 spectral FFT lane (Spectral PR2) — fft / ifft / rfft / irfft over a
power-of-two axis, on the AVX-512 radix-2 C2C kernel + r2c/c2r pack-unpack.
Reachable via `compiler_path="x86_fft_compiled"`. Validated vs np.fft.

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


def _art(rt, op_name, kwargs):
    return rt.RuntimeArtifact(metadata={
        "target": "x86", "compiler_path": "x86_fft_compiled",
        "executable": True, "execution_kind": "native_cpu",
        "arg_names": ["x"], "output_name": "o",
        "ops": [{"op_name": op_name, "result": "o", "operands": ["x"],
                 "kwargs": kwargs}],
    })


_TOL = dict(atol=2e-4, rtol=2e-4)


@pytest.mark.parametrize("n", [2, 8, 64, 1024])
@pytest.mark.parametrize("shape_pre", [(), (3,), (2, 5)])
def test_fft_ifft(n, shape_pre):
    rt = _x86_or_skip()
    rng = np.random.default_rng(1 + n + len(shape_pre))
    x = (rng.standard_normal(shape_pre + (n,))
         + 1j * rng.standard_normal(shape_pre + (n,))).astype(np.complex64)
    rf = rt.launch(_art(rt, "tessera.fft", {"axis": -1}), (x,))
    assert rf["ok"] is True, rf.get("reason")
    assert rf["compiler_path"] == "x86_fft_compiled"
    np.testing.assert_allclose(np.asarray(rf["output"]).astype(np.complex64),
                               np.fft.fft(x, axis=-1).astype(np.complex64),
                               **_TOL)
    ri = rt.launch(_art(rt, "tessera.ifft", {"axis": -1}), (x,))
    assert ri["ok"] is True, ri.get("reason")
    np.testing.assert_allclose(np.asarray(ri["output"]).astype(np.complex64),
                               np.fft.ifft(x, axis=-1).astype(np.complex64),
                               **_TOL)


def test_fft_inner_axis():
    rt = _x86_or_skip()
    rng = np.random.default_rng(7)
    x = (rng.standard_normal((4, 16, 3))
         + 1j * rng.standard_normal((4, 16, 3))).astype(np.complex64)
    res = rt.launch(_art(rt, "tessera.fft", {"axis": 1}), (x,))
    assert res["ok"] is True, res.get("reason")
    np.testing.assert_allclose(np.asarray(res["output"]).astype(np.complex64),
                               np.fft.fft(x, axis=1).astype(np.complex64), **_TOL)


@pytest.mark.parametrize("n", [8, 64, 256])
def test_rfft_irfft(n):
    rt = _x86_or_skip()
    rng = np.random.default_rng(3 + n)
    x = (rng.standard_normal((2, n))).astype(np.float32)
    rf = rt.launch(_art(rt, "tessera.rfft", {"axis": -1}), (x,))
    assert rf["ok"] is True, rf.get("reason")
    ref = np.fft.rfft(x, axis=-1).astype(np.complex64)
    got = np.asarray(rf["output"]).astype(np.complex64)
    assert got.shape == ref.shape
    np.testing.assert_allclose(got, ref, **_TOL)
    # round-trip irfft(rfft(x)) == x
    ir = rt.launch(_art(rt, "tessera.irfft", {"axis": -1, "n": n}), (ref,))
    assert ir["ok"] is True, ir.get("reason")
    np.testing.assert_allclose(np.asarray(ir["output"]).astype(np.float32), x,
                               **_TOL)


def test_non_power_of_two_diagnoses():
    rt = _x86_or_skip()
    x = np.zeros((100,), np.complex64)
    res = rt.launch(_art(rt, "tessera.fft", {"axis": -1}), (x,))
    assert res["ok"] is False
    assert "power-of-two" in str(res.get("reason"))


def test_fft_unknown_op_rejected():
    from tessera import runtime as rt
    x = np.zeros((8,), np.complex64)
    with pytest.raises(ValueError, match="x86_fft_compiled executor"):
        rt._execute_x86_compiled_fft(_art(rt, "tessera.dct", {}), (x,))
