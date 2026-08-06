"""Content-addressed TSOL composites in the prebuilt gfx1151 device package.

Framing, windowing, pointwise work, and overlap-add are native HIP work around
digest-bound persistent child FFT plans; no host NumPy composition is allowed.
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


def _art(rt, op_name, operands, kwargs):
    from tessera.compiler.scheduled_spectral import lower_scheduled_spectral

    names = [f"a{i}" for i in range(len(operands))]
    contract = lower_scheduled_spectral(
        target="rocm",
        op_name=op_name,
        input_shapes=tuple(tuple(int(dim) for dim in value.shape) for value in operands),
        axis=int(kwargs.get("axis", -1)),
        hop=kwargs.get("hop"),
    ).to_metadata()
    return rt.RuntimeArtifact(metadata={
        "target": "rocm", "compiler_path": "rocm_spectral_compiled",
        "executable": True, "execution_kind": "native_gpu",
        "arg_names": names, "output_name": "o",
        "scheduled_spectral": contract,
    })


def _stft_ref(x, win, hop):
    wl = win.shape[-1]
    return np.stack(
        [np.fft.rfft(x[..., s:s + wl] * win, axis=-1)
         for s in range(0, max(1, x.shape[-1] - wl + 1), hop)], axis=-2)


_TOL = dict(atol=1e-2, rtol=1e-2)


def test_dct():
    rt = _rocm_or_skip()
    rng = np.random.default_rng(1)
    x = rng.standard_normal((3, 8)).astype(np.float32)
    res = rt.launch(_art(rt, "tessera.dct", (x,), {"axis": -1}), (x,))
    assert res["ok"] is True, res.get("reason")
    assert res["compiler_path"] == "rocm_spectral_compiled"
    y = np.concatenate([x, np.flip(x, -1)], -1).astype(np.complex64)
    ref = np.real(np.fft.fft(y, axis=-1)[..., :8]).astype(np.float32)
    np.testing.assert_allclose(np.asarray(res["output"]), ref, **_TOL)


def test_spectral_conv():
    rt = _rocm_or_skip()
    rng = np.random.default_rng(2)
    x = rng.standard_normal((3, 12)).astype(np.float32)
    w = rng.standard_normal((3, 5)).astype(np.float32)
    res = rt.launch(_art(rt, "tessera.spectral_conv", (x, w), {}), (x, w))
    assert res["ok"] is True, res.get("reason")
    n = x.shape[-1] + w.shape[-1] - 1
    nfft = 1 << int(np.ceil(np.log2(n)))
    ref = np.fft.irfft(np.fft.rfft(x, nfft) * np.fft.rfft(w, nfft), nfft)[..., :n]
    np.testing.assert_allclose(np.asarray(res["output"]), ref, **_TOL)


def test_spectral_filter():
    rt = _rocm_or_skip()
    rng = np.random.default_rng(3)
    Xf = (rng.standard_normal((2, 5)) + 1j * rng.standard_normal((2, 5))
          ).astype(np.complex64)
    Hf = (rng.standard_normal((2, 5)) + 1j * rng.standard_normal((2, 5))
          ).astype(np.complex64)
    res = rt.launch(_art(rt, "tessera.spectral_filter", (Xf, Hf), {}), (Xf, Hf))
    assert res["ok"] is True, res.get("reason")
    np.testing.assert_allclose(np.asarray(res["output"]).astype(np.complex64),
                               (Xf * Hf).astype(np.complex64), atol=1e-4)


def test_stft_istft():
    rt = _rocm_or_skip()
    rng = np.random.default_rng(4)
    sig = rng.standard_normal((32,)).astype(np.float32)
    win = np.hanning(8).astype(np.float32)
    s = rt.launch(_art(rt, "tessera.stft", (sig, win), {"hop": 4}), (sig, win))
    assert s["ok"] is True, s.get("reason")
    sref = _stft_ref(sig, win, 4)
    out = np.asarray(s["output"])
    assert out.shape == sref.shape
    np.testing.assert_allclose(out.astype(np.complex64),
                               sref.astype(np.complex64), **_TOL)
    i = rt.launch(_art(rt, "tessera.istft", (out, win), {"hop": 4}), (out, win))
    assert i["ok"] is True, i.get("reason")
    rec = np.asarray(i["output"])
    assert rec.shape[-1] == (sref.shape[-2] - 1) * 4 + 8


def test_ragged_batched_bluestein_composites():
    """Prime transform lengths exercise child Bluestein plans, not radix-only fixtures."""
    rt = _rocm_or_skip()
    rng = np.random.default_rng(5)

    x = rng.standard_normal((2, 17)).astype(np.float32)
    d = rt.launch(_art(rt, "tessera.dct", (x,), {}), (x,))
    assert d["ok"] is True, d.get("reason")
    mirrored = np.concatenate([x, np.flip(x, -1)], -1).astype(np.complex64)
    dref = np.fft.fft(mirrored, axis=-1)[..., :17].real.astype(np.float32)
    np.testing.assert_allclose(np.asarray(d["output"]), dref, **_TOL)

    signal = rng.standard_normal((2, 43)).astype(np.float32)
    window = np.hanning(17).astype(np.float32)
    s = rt.launch(
        _art(rt, "tessera.stft", (signal, window), {"hop": 6}),
        (signal, window),
    )
    assert s["ok"] is True, s.get("reason")
    sref = _stft_ref(signal, window, 6)
    np.testing.assert_allclose(np.asarray(s["output"]), sref, **_TOL)

    i = rt.launch(
        _art(rt, "tessera.istft", (np.asarray(s["output"]), window), {"hop": 6}),
        (np.asarray(s["output"]), window),
    )
    assert i["ok"] is True, i.get("reason")
    assert np.asarray(i["output"]).shape == (2, 41)
