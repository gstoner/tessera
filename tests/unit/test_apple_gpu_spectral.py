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
gpu = pytest.mark.hardware_apple_gpu


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
    # Rank-matched kernel: `spectral_conv` requires equal ranks, stated by both
    # the host reference and the `conv_full` shape rule. This line used to pass
    # a rank-1 `w` against the rank-2 `r`, which the GPU dispatch accepted
    # (numpy broadcasting inside `xf * wf`) and the reference rejected -- so the
    # assertion never ran, it raised. The two forms compute bit-identical
    # results; the difference is only whether the contract admits the input.
    w = rng.standard_normal((1, 5)).astype(np.float32)
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


def test_spectral_conv_refuses_a_rank_mismatch_instead_of_broadcasting():
    """The GPU lane must not admit inputs the host reference raises on.

    `spectral_conv` declares equal ranks in the two places the contract is
    stated -- the host reference and the `conv_full` shape rule. The dispatch
    path stated it nowhere and inherited numpy broadcasting from `xf * wf`, so
    a rank-1 kernel against a rank-2 signal computed happily on GPU and raised
    on CPU.

    The divergence was in what the lanes *admit*, not what they compute: the
    two forms are bit-identical. That is what made it survive -- and it is how
    `test_composites_match_host_reference` came to be written against input the
    contract forbids, where the comparison never ran because the reference
    raised first.

    Shape inference was the quieter casualty: `_shape_conv_full` returns
    unknown on a rank mismatch rather than deriving `n + m - 1`.
    """
    rng = np.random.default_rng(11)
    x = rng.standard_normal((3, 16)).astype(np.float32)
    w1 = rng.standard_normal(5).astype(np.float32)

    with pytest.raises(ValueError, match="requires equal ranks"):
        _D("tessera.spectral_conv", [x, w1])

    # The rank-matched form is accepted and matches the reference.
    w2 = w1.reshape(1, 5)
    np.testing.assert_allclose(
        _D("tessera.spectral_conv", [x, w2]),
        np.asarray(ts.ops.spectral_conv(x, w2)), rtol=1e-4, atol=1e-4)


def test_spectral_conv_matches_numpy_full_convolution():
    """Ground truth outside Tessera, so 'both lanes agree' cannot mean 'both
    lanes are wrong the same way'."""
    rng = np.random.default_rng(3)
    x = rng.standard_normal((2, 12)).astype(np.float32)
    w = rng.standard_normal((1, 4)).astype(np.float32)
    got = _D("tessera.spectral_conv", [x, w])
    assert got.shape == (2, 12 + 4 - 1)
    for row in range(x.shape[0]):
        np.testing.assert_allclose(
            got[row], np.convolve(x[row], w[0], mode="full"),
            rtol=1e-4, atol=1e-4)
