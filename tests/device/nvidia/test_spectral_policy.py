"""Exact-SM120 numerical certificates for the CUDA spectral policy ABI."""

from __future__ import annotations

import numpy as np
import pytest


def _runtime_or_skip():
    from tessera import runtime

    lib = runtime._load_nvidia_fft_runtime()
    if lib is None or not hasattr(lib, "tessera_nvidia_spectral_package_abi"):
        pytest.skip("NVIDIA spectral policy package is unavailable")
    if lib.tessera_nvidia_spectral_arch() != 120:
        pytest.skip("exact sm_120 device is unavailable")
    return runtime, lib


def _artifact(runtime, op_name, operands, kwargs):
    names = [f"a{index}" for index in range(len(operands))]
    return runtime.RuntimeArtifact(metadata={
        "target": "nvidia_sm120",
        "compiler_path": "nvidia_spectral_compiled",
        "executable": True,
        "execution_kind": "native_gpu",
        "arg_names": names,
        "output_name": "output",
        "ops": [{
            "op_name": op_name,
            "result": "output",
            "operands": names,
            "kwargs": dict(kwargs),
        }],
    })


def _launch(op_name, operands, kwargs):
    runtime, _ = _runtime_or_skip()
    result = runtime.launch(
        _artifact(runtime, op_name, operands, kwargs), tuple(operands)
    )
    assert result["ok"] is True, result.get("reason")
    assert result["compiler_path"] == "nvidia_spectral_compiled"
    return np.asarray(result["output"])


def _expanded_window(window, batch_shape, nfft):
    window = np.asarray(window, np.float32)
    broadcast = np.broadcast_to(window, tuple(batch_shape) + (window.shape[-1],))
    expanded = np.zeros(tuple(batch_shape) + (nfft,), np.float32)
    placement = (nfft - window.shape[-1]) // 2
    expanded[..., placement:placement + window.shape[-1]] = broadcast
    return expanded


def _stft_reference(x, window, *, axis, nfft, hop, center, pad_mode,
                    onesided, normalization):
    moved = np.moveaxis(np.asarray(x, np.float32), axis, -1)
    batch_shape = moved.shape[:-1]
    windows = _expanded_window(window, batch_shape, nfft)
    pad = nfft // 2 if center else 0
    if center:
        moved = np.pad(
            moved, ((0, 0),) * len(batch_shape) + ((pad, pad),), mode=pad_mode
        )
    if moved.shape[-1] < nfft:
        moved = np.pad(
            moved,
            ((0, 0),) * len(batch_shape) + ((0, nfft - moved.shape[-1]),),
        )
    transform = np.fft.rfft if onesided else np.fft.fft
    frames = np.stack([
        transform(moved[..., start:start + nfft] * windows, axis=-1,
                  norm=normalization)
        for start in range(0, moved.shape[-1] - nfft + 1, hop)
    ], axis=-2).astype(np.complex64)
    # The public STFT replaces one signal axis by adjacent frame/frequency axes.
    before = axis
    batch_rank = len(batch_shape)
    permutation = (tuple(range(before)) + (batch_rank, batch_rank + 1) +
                   tuple(range(before, batch_rank)))
    return np.transpose(frames, permutation)


def _istft_reference(spectrum, window, *, axis, nfft, hop, center, length,
                     onesided, normalization):
    values = np.asarray(spectrum, np.complex64)
    frame_axis = axis - 1
    batch_axes = tuple(
        index for index in range(values.ndim) if index not in {frame_axis, axis}
    )
    packed = np.transpose(values, batch_axes + (frame_axis, axis))
    batch_shape = packed.shape[:-2]
    windows = _expanded_window(window, batch_shape, nfft)
    transform = np.fft.irfft if onesided else np.fft.ifft
    frames = transform(packed, n=nfft, axis=-1, norm=normalization).real
    raw = (frames.shape[-2] - 1) * hop + nfft
    numerator = np.zeros(batch_shape + (raw,), np.float64)
    denominator = np.zeros_like(numerator)
    for frame in range(frames.shape[-2]):
        start = frame * hop
        numerator[..., start:start + nfft] += frames[..., frame, :] * windows
        denominator[..., start:start + nfft] += windows * windows
    output = numerator / np.maximum(denominator, 1.0e-12)
    trim = nfft // 2 if center else 0
    output = output[..., trim:trim + length].astype(np.float32)
    # Restore the output sample axis where the frame axis lived.
    before = frame_axis
    permutation = (tuple(range(before)) + (len(batch_shape),) +
                   tuple(range(before, len(batch_shape))))
    return np.transpose(output, permutation)


def _dct_reference(value, *, axis, dct_type):
    moved = np.moveaxis(np.asarray(value, np.float64), axis, -1)
    length = moved.shape[-1]
    output = np.empty_like(moved)
    for k in range(length):
        if dct_type == 1:
            result = moved[..., 0] + (-1.0 if k & 1 else 1.0) * moved[..., -1]
            for n in range(1, length - 1):
                result += 2.0 * moved[..., n] * np.cos(
                    np.pi * n * k / (length - 1)
                )
        elif dct_type == 2:
            result = sum(
                2.0 * moved[..., n] * np.cos(
                    np.pi * (2 * n + 1) * k / (2 * length)
                ) for n in range(length)
            )
        elif dct_type == 3:
            result = moved[..., 0].copy()
            for n in range(1, length):
                result += 2.0 * moved[..., n] * np.cos(
                    np.pi * n * (2 * k + 1) / (2 * length)
                )
        else:
            result = sum(
                2.0 * moved[..., n] * np.cos(
                    np.pi * (2 * n + 1) * (2 * k + 1) / (4 * length)
                ) for n in range(length)
            )
        output[..., k] = result
    return np.moveaxis(output.astype(np.float32), -1, axis)


def test_policy_abi_is_exact_sm120_and_host_composition_is_not_reentered(
    monkeypatch,
):
    runtime, lib = _runtime_or_skip()
    assert lib.tessera_nvidia_spectral_package_abi() == (
        b"tessera.nvidia.spectral_policy.v1")
    assert lib.tessera_nvidia_spectral_arch() == 120

    def forbidden(*_args, **_kwargs):
        raise AssertionError("STFT policy re-entered the Python FFT compositor")

    monkeypatch.setattr(runtime, "_nvidia_fftexec", forbidden)
    x = np.linspace(-1.0, 1.0, 32, dtype=np.float32)
    window = np.hanning(8).astype(np.float32)
    result = runtime.launch(
        _artifact(runtime, "tessera.stft", (x, window), {"hop": 4}),
        (x, window),
    )
    assert result["ok"] is True, result.get("reason")


@pytest.mark.parametrize("dct_type", (1, 2, 3, 4))
def test_dct_types_arbitrary_axis_and_true_stride(dct_type):
    rng = np.random.default_rng(1200 + dct_type)
    storage = rng.standard_normal((3, 18, 2)).astype(np.float32)
    value = storage[:, ::2, :]
    assert not value.flags.c_contiguous
    actual = _launch(
        "tessera.dct", (value,), {"axis": 1, "type": dct_type}
    )
    expected = _dct_reference(value, axis=1, dct_type=dct_type)
    np.testing.assert_allclose(actual, expected, rtol=3e-5, atol=3e-5)


@pytest.mark.parametrize("normalization", ("backward", "forward", "ortho"))
def test_onesided_forward_and_inverse_normalization(normalization):
    rng = np.random.default_rng(420 + len(normalization))
    x = rng.standard_normal((2, 37)).astype(np.float32)
    window = (0.25 + np.hanning(12)).astype(np.float32)
    kwargs = {
        "axis": -1,
        "n_fft": 16,
        "hop": 5,
        "center": True,
        "pad_mode": "constant",
        "onesided": True,
        "normalization": normalization,
    }
    actual_spectrum = _launch("tessera.stft", (x, window), kwargs)
    expected_spectrum = _stft_reference(
        x, window, axis=1, nfft=16, hop=5, center=True,
        pad_mode="constant", onesided=True, normalization=normalization,
    )
    np.testing.assert_allclose(
        actual_spectrum, expected_spectrum, rtol=4e-5, atol=4e-5
    )
    inverse_kwargs = {
        "axis": -1,
        "n_fft": 16,
        "hop": 5,
        "center": True,
        "length": 35,
        "onesided": True,
        "normalization": normalization,
    }
    actual_signal = _launch(
        "tessera.istft", (actual_spectrum, window), inverse_kwargs
    )
    expected_signal = _istft_reference(
        actual_spectrum, window, axis=2, nfft=16, hop=5, center=True,
        length=35, onesided=True, normalization=normalization,
    )
    np.testing.assert_allclose(
        actual_signal, expected_signal, rtol=5e-5, atol=5e-5
    )


def test_full_spectrum_arbitrary_axis_true_strides_reflect_and_broadcast():
    rng = np.random.default_rng(5070)
    signal_storage = rng.standard_normal((2, 92, 3)).astype(np.float32)
    signal = signal_storage[:, ::2, :]
    window_storage = (0.3 + rng.random((3, 30))).astype(np.float32)
    window = window_storage[:, ::2]
    assert not signal.flags.c_contiguous and not window.flags.c_contiguous
    kwargs = {
        "axis": 1,
        "n_fft": 20,
        "hop": 6,
        "center": True,
        "pad_mode": "reflect",
        "onesided": False,
        "normalization": "ortho",
    }
    actual_spectrum = _launch("tessera.stft", (signal, window), kwargs)
    expected_spectrum = _stft_reference(
        signal, window, axis=1, nfft=20, hop=6, center=True,
        pad_mode="reflect", onesided=False, normalization="ortho",
    )
    np.testing.assert_allclose(
        actual_spectrum, expected_spectrum, rtol=5e-5, atol=5e-5
    )

    spectrum_storage = np.empty(
        (2, actual_spectrum.shape[1], 40, 3), np.complex64
    )
    spectrum = spectrum_storage[:, :, ::2, :]
    spectrum[...] = actual_spectrum
    assert not spectrum.flags.c_contiguous
    inverse_kwargs = {
        "axis": 2,
        "n_fft": 20,
        "hop": 6,
        "center": True,
        "length": 38,
        "onesided": False,
        "normalization": "ortho",
    }
    actual_signal = _launch("tessera.istft", (spectrum, window), inverse_kwargs)
    expected_signal = _istft_reference(
        spectrum, window, axis=2, nfft=20, hop=6, center=True,
        length=38, onesided=False, normalization="ortho",
    )
    np.testing.assert_allclose(
        actual_signal, expected_signal, rtol=6e-5, atol=6e-5
    )
