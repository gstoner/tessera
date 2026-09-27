"""A stale HIP error left on the thread must not fail a spectral package call.

HIP's last-error slot is per host thread and sticky: only a failing call writes
it and only ``hipGetLastError`` resets it. The prebuilt spectral image checks
its launches with ``hipGetLastError``, so before SPECTRAL-STALE-HIP-ERROR-
2026-09-27 an unrelated refused HIP call earlier in the process (in the gfx1201
full sweep, an earlier test's HIP call) made the next correct launch report
failure: the streaming STFT returned rc=246 in a full ``tests/unit`` sweep and
passed alone. Each device-work entry point now discards errors older than the
call, once, on entry.

Every test here primes the slot deliberately (``hipSetDevice`` on an ordinal
that does not exist), proves the slot is primed without clearing it
(``hipPeekAtLastError``), and then requires the package call to succeed and
agree with the reference. On an image without the entry clear, each of these
fails with the package's own launch-check return code.
"""

from __future__ import annotations

import ctypes

import numpy as np
import pytest

from tests._support.rocm_build import runtime_for_host

_MISSING_ORDINAL = 97


def _rocm_composite_or_skip():
    from tessera import runtime as rt
    from tessera.compiler.emit import spectral_candidates

    if not rt._rocm_wmma_runtime_available():
        pytest.skip("no usable AMD GPU")
    arch = spectral_candidates._spectral_device_arch()
    if arch is None or spectral_candidates._amd_composite_lib() is None:
        pytest.skip(f"no ROCm composite spectral package for the live device ({arch})")
    return rt, arch


@pytest.fixture(autouse=True)
def _drain_hip_error_slot():
    """Never leak the primed error to a later test, whatever this one did."""
    yield
    from tessera import runtime as rt

    hip = rt._load_hip_for_launch()
    if hip is not None:
        hip.hipGetLastError.argtypes = []
        hip.hipGetLastError.restype = ctypes.c_int
        hip.hipGetLastError()


def _prime_stale_hip_error(rt) -> None:
    """Leave an unread error in this thread's HIP last-error slot."""
    hip = rt._load_hip_for_launch()
    assert hip is not None
    set_device = hip.hipSetDevice
    set_device.argtypes = [ctypes.c_int]
    set_device.restype = ctypes.c_int
    count = ctypes.c_int()
    hip.hipGetDeviceCount.argtypes = [ctypes.POINTER(ctypes.c_int)]
    hip.hipGetDeviceCount.restype = ctypes.c_int
    assert hip.hipGetDeviceCount(ctypes.byref(count)) == 0
    assert count.value <= _MISSING_ORDINAL, "the priming ordinal must not exist"
    assert set_device(_MISSING_ORDINAL) != 0
    # hipPeekAtLastError reads the slot without resetting it: the precondition
    # of every test below is a primed slot, not merely a failed call.
    peek = hip.hipPeekAtLastError
    peek.argtypes = []
    peek.restype = ctypes.c_int
    assert peek() != 0, "hipSetDevice failure did not reach the last-error slot"


def test_streaming_stft_survives_a_stale_hip_error():
    from tessera.compiler.spectral_streaming import (
        StreamingSTFTPolicy, stream_stft_chunk,
    )
    import tessera

    rt, arch = _rocm_composite_or_skip()
    rng = np.random.default_rng(813)
    signal = rng.standard_normal((2, 46, 3)).astype(np.float32)[:, ::2, :]
    window = np.stack((np.hanning(6), np.hamming(6)), axis=0).astype(
        np.float32
    )[:, None, :]
    policy = StreamingSTFTPolicy(
        axis=1, n_fft=8, window_length=6, hop=4, onesided=True,
        max_chunk_samples=9,
    )
    state = None
    outputs = []
    # Chunk 0 emits no frame and never reaches the FFT path; prime before
    # every chunk so the frame-producing chunks run against a stale slot.
    for piece in np.split(signal, [7, 16], axis=1):
        _prime_stale_hip_error(rt)
        output, state = stream_stft_chunk(piece, window, policy, state, target="rocm")
        outputs.append(output)
    assert sum(output.shape[1] for output in outputs) > 0
    expected = tessera.ops.stft(
        signal, window, axis=1, n_fft=8, hop=4, center=False, onesided=True,
    )
    np.testing.assert_allclose(
        np.concatenate(outputs, axis=1), expected, atol=3e-5, rtol=3e-5,
    )
    assert state is not None
    assert state.execution_certificate["architecture_identity"] == arch


def test_fft_plan_host_entries_survive_a_stale_hip_error():
    from tessera.compiler.emit import spectral_candidates

    rt, _ = _rocm_composite_or_skip()
    digest = "5" * 64
    rng = np.random.default_rng(814)
    # n=64 takes the fused-LDS path, whose launch check reads the slot.
    rows = (rng.standard_normal((4, 64)) + 1j * rng.standard_normal((4, 64))
            ).astype(np.complex64)
    _prime_stale_hip_error(rt)
    out = spectral_candidates.run_rocm_stockham_rows(
        rows, inverse=False, artifact_digest=digest
    )
    np.testing.assert_allclose(out, np.fft.fft(rows, axis=-1), atol=2e-4, rtol=2e-4)

    real = rng.standard_normal((3, 128)).astype(np.float32)
    _prime_stale_hip_error(rt)
    half = spectral_candidates.run_rocm_packed_real_rows(
        real, inverse=False, logical_n=128, artifact_digest=digest
    )
    np.testing.assert_allclose(half, np.fft.rfft(real, axis=-1), atol=2e-4, rtol=2e-4)


def test_broadcast_stft_and_istft_survive_a_stale_hip_error():
    from tests.unit.test_rocm_spectral_compiled import _art
    from tests.unit.test_x86_spectral_compiled import (
        _general_istft_ref, _general_stft_ref,
    )

    base, _ = _rocm_composite_or_skip()
    if base._tessera_opt_path() is None:
        pytest.skip("tessera-opt not built")
    rt = runtime_for_host(base)
    rng = np.random.default_rng(815)
    signal = rng.standard_normal((2, 48, 3)).astype(np.float32)[:, ::2, :]
    window = np.stack((np.hanning(8), np.hamming(8)), axis=0).astype(
        np.float32
    )[:, None, :]
    stft_artifact = _art(rt, "tessera.stft", (signal, window), {
        "axis": 1, "hop": 4, "n_fft": 10, "onesided": True,
    })
    _prime_stale_hip_error(base)
    stft = rt.launch(stft_artifact, (signal, window))
    assert stft["ok"] is True, stft.get("reason")
    actual = np.asarray(stft["output"])
    expected = np.empty_like(actual)
    for batch in range(2):
        for channel in range(3):
            expected[batch, :, :, channel] = _general_stft_ref(
                signal[batch, :, channel], window[batch, 0], 4,
                n_fft=10, center=False, onesided=True,
            )
    np.testing.assert_allclose(actual, expected, atol=8e-4, rtol=8e-4)

    istft_artifact = _art(rt, "tessera.istft", (actual, window), {
        "axis": 2, "hop": 4, "n_fft": 10, "onesided": True, "length": 22,
    })
    _prime_stale_hip_error(base)
    istft = rt.launch(istft_artifact, (actual, window))
    assert istft["ok"] is True, istft.get("reason")
    expected_inverse = np.empty((2, 22, 3), np.float32)
    for batch in range(2):
        for channel in range(3):
            expected_inverse[batch, :, channel] = _general_istft_ref(
                actual[batch, :, :, channel], window[batch, 0], 4,
                n_fft=10, center=False, length=22, onesided=True,
            )
    np.testing.assert_allclose(
        istft["output"], expected_inverse, atol=1e-3, rtol=1e-3,
    )
