"""Apple GPU spectral / FFT lane (the "special" kernel class).

fft / ifft / rfft / irfft execute on the GPU via MPSGraph's FourierTransform
ops (macOS 14+); dct / stft / istft / spectral_conv compose over them; the
9 spectral primitives move from `special` to `proven` in
s_series_accelerator_proof.md. Validated against the numpy reference at fp32 tol.
"""

import numpy as np
import pytest

import tessera as ts
from tessera import _apple_gpu_backend as agb
from tessera import _jit_boundary as jb
from tessera import runtime as R
from tessera.compiler import apple_gpu_envelope as env

_GPU = agb.is_available() and jb.is_available()
gpu = pytest.mark.skipif(not _GPU, reason="apple_gpu runtime unavailable")


def _D(op, operands, **kw):
    return np.asarray(R._apple_gpu_dispatch_spectral(op, operands, kw, np))


# ── envelope membership (drives the accelerator-proof flip) ──────────────────

def test_nine_spectral_ops_in_envelope_on_spectral_lane():
    ops = {"fft", "ifft", "rfft", "irfft", "dct", "stft", "istft",
           "spectral_conv", "spectral_filter"}
    for o in ops:
        assert env.APPLE_GPU_LANE_BY_OP.get(f"tessera.{o}") == "spectral", o


def test_accelerator_proof_marks_spectral_proven():
    from tessera.compiler.accelerator_proof import all_rows
    spectral = {r.name: r.accel_class for r in all_rows()
                if r.name in ("fft", "rfft", "irfft", "stft", "spectral_conv")}
    assert all(c == "proven" for c in spectral.values()), spectral


# ── numerical correctness vs numpy (direct dispatch) ─────────────────────────

@gpu
@pytest.mark.parametrize("n", [8, 16, 17, 32])
def test_fft_ifft_match_numpy(n):
    rng = np.random.default_rng(n)
    x = (rng.standard_normal((3, n)) + 1j * rng.standard_normal((3, n))).astype(np.complex64)
    np.testing.assert_allclose(_D("tessera.fft", [x]), np.fft.fft(x, axis=-1), rtol=1e-4, atol=1e-4)
    np.testing.assert_allclose(_D("tessera.ifft", [x]), np.fft.ifft(x, axis=-1), rtol=1e-4, atol=1e-4)


@gpu
@pytest.mark.parametrize("n", [8, 16, 15, 32])
def test_rfft_irfft_match_numpy(n):
    rng = np.random.default_rng(n + 1)
    r = rng.standard_normal((3, n)).astype(np.float32)
    np.testing.assert_allclose(_D("tessera.rfft", [r]), np.fft.rfft(r, axis=-1), rtol=1e-4, atol=1e-4)
    rc = np.fft.rfft(r, axis=-1).astype(np.complex64)
    np.testing.assert_allclose(_D("tessera.irfft", [rc], n=n), np.fft.irfft(rc, n=n, axis=-1), rtol=1e-4, atol=1e-4)


@gpu
def test_fft_off_last_axis():
    rng = np.random.default_rng(5)
    x = (rng.standard_normal((4, 8)) + 1j * rng.standard_normal((4, 8))).astype(np.complex64)
    np.testing.assert_allclose(_D("tessera.fft", [x], axis=0), np.fft.fft(x, axis=0), rtol=1e-4, atol=1e-4)


@gpu
def test_composites_match_host_reference():
    rng = np.random.default_rng(7)
    r = rng.standard_normal((3, 16)).astype(np.float32)
    np.testing.assert_allclose(_D("tessera.dct", [r]), np.asarray(ts.ops.dct(r)), rtol=1e-4, atol=1e-4)
    w = rng.standard_normal(5).astype(np.float32)
    np.testing.assert_allclose(_D("tessera.spectral_conv", [r, w]),
                               np.asarray(ts.ops.spectral_conv(r, w)), rtol=1e-4, atol=1e-4)
    win = np.hanning(8).astype(np.float32)
    sf = _D("tessera.stft", [r, win, 4])
    np.testing.assert_allclose(sf, np.asarray(ts.ops.stft(r, win, 4)), rtol=1e-4, atol=1e-4)
    np.testing.assert_allclose(_D("tessera.istft", [sf, win, 4]),
                               np.asarray(ts.ops.istft(sf, win, 4)), rtol=1e-3, atol=1e-3)


# ── ABI + @jit ───────────────────────────────────────────────────────────────

@gpu
def test_fft_abi_symbol_present():
    assert hasattr(agb._load(), "tessera_apple_gpu_fft_f32")


@gpu
def test_jit_fft_metal_runtime():
    rng = np.random.default_rng(0)
    x = (rng.standard_normal((2, 16)) + 1j * rng.standard_normal((2, 16))).astype(np.complex64)

    @ts.jit(target="apple_gpu")
    def f(x):
        return ts.ops.fft(x)

    got = np.asarray(f(x))
    np.testing.assert_allclose(got, np.fft.fft(x, axis=-1), rtol=1e-3, atol=1e-3)
    assert f.runtime_artifact().metadata["execution_mode"] == "metal_runtime"
