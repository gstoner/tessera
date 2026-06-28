"""Compiler-generated spectral FFT on gfx1151 (Spectral PR4) — fft / ifft /
rfft / irfft on the direct DFT kernel (generate-rocm-dft-kernel). Reachable via
`compiler_path="rocm_fft_compiled"`. Validated vs np.fft on gfx1151.

Skip-clean: tessera-opt not built / no GPU.
"""

from __future__ import annotations

import numpy as np
import pytest


def _rocm_or_skip():
    from tessera import runtime as rt
    if rt._tessera_opt_path() is None:
        pytest.skip("tessera-opt not built")
    if not rt._rocm_wmma_runtime_available():
        pytest.skip("no usable AMD GPU")
    return rt


def _art(rt, op_name, kwargs):
    return rt.RuntimeArtifact(metadata={
        "target": "rocm", "compiler_path": "rocm_fft_compiled",
        "executable": True, "execution_kind": "native_gpu",
        "arg_names": ["x"], "output_name": "o",
        "ops": [{"op_name": op_name, "result": "o", "operands": ["x"],
                 "kwargs": kwargs}],
    })


_TOL = dict(atol=1e-2, rtol=1e-2)


@pytest.mark.parametrize("n", [4, 8, 16, 100])   # pow2 + non-pow2 (one kernel)
@pytest.mark.parametrize("shape_pre", [(), (3,)])
def test_fft_ifft(n, shape_pre):
    rt = _rocm_or_skip()
    rng = np.random.default_rng(1 + n + len(shape_pre))
    x = (rng.standard_normal(shape_pre + (n,))
         + 1j * rng.standard_normal(shape_pre + (n,))).astype(np.complex64)
    rf = rt.launch(_art(rt, "tessera.fft", {"axis": -1}), (x,))
    assert rf["ok"] is True, rf.get("reason")
    assert rf["compiler_path"] == "rocm_fft_compiled"
    np.testing.assert_allclose(np.asarray(rf["output"]).astype(np.complex64),
                               np.fft.fft(x, axis=-1).astype(np.complex64), **_TOL)
    ri = rt.launch(_art(rt, "tessera.ifft", {"axis": -1}), (x,))
    assert ri["ok"] is True, ri.get("reason")
    np.testing.assert_allclose(np.asarray(ri["output"]).astype(np.complex64),
                               np.fft.ifft(x, axis=-1).astype(np.complex64), **_TOL)


def test_fft_inner_axis():
    rt = _rocm_or_skip()
    rng = np.random.default_rng(7)
    x = (rng.standard_normal((4, 16, 3))
         + 1j * rng.standard_normal((4, 16, 3))).astype(np.complex64)
    res = rt.launch(_art(rt, "tessera.fft", {"axis": 1}), (x,))
    assert res["ok"] is True, res.get("reason")
    np.testing.assert_allclose(np.asarray(res["output"]).astype(np.complex64),
                               np.fft.fft(x, axis=1).astype(np.complex64), **_TOL)


@pytest.mark.parametrize("n", [8, 16])
def test_rfft_irfft(n):
    rt = _rocm_or_skip()
    rng = np.random.default_rng(3 + n)
    x = rng.standard_normal((2, n)).astype(np.float32)
    rf = rt.launch(_art(rt, "tessera.rfft", {"axis": -1}), (x,))
    assert rf["ok"] is True, rf.get("reason")
    ref = np.fft.rfft(x, axis=-1).astype(np.complex64)
    np.testing.assert_allclose(np.asarray(rf["output"]).astype(np.complex64),
                               ref, **_TOL)
    ir = rt.launch(_art(rt, "tessera.irfft", {"axis": -1, "n": n}), (ref,))
    assert ir["ok"] is True, ir.get("reason")
    np.testing.assert_allclose(np.asarray(ir["output"]).astype(np.float32), x,
                               **_TOL)


def test_fft_codegen_lowers():
    import subprocess
    from pathlib import Path
    opt = Path(__file__).resolve().parents[2] / "build/tools/tessera-opt/tessera-opt"
    if not opt.is_file():
        pytest.skip("build tessera-opt")
    d = ('module {\n  "tessera_rocm.dft"() {name = "dt", inverse = false} '
         ': () -> ()\n}\n')
    low = subprocess.run(
        [str(opt), "-",
         "--pass-pipeline=builtin.module(generate-rocm-dft-kernel,"
         "gpu.module(convert-scf-to-cf,convert-gpu-to-rocdl,"
         "reconcile-unrealized-casts))"],
        input=d, capture_output=True, text=True)
    assert low.returncode == 0 and "llvm." in low.stdout
