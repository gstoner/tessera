"""A stale CUDA runtime error left on the thread must not fail an sm_120 hook.

The CUDA runtime keeps one last-error slot per host thread: any failing runtime
call writes it and only ``cudaGetLastError`` resets it. The hand-written sm_120
hooks (``libtessera_nvidia_fft.so`` = FFT + spectral policy, and
``libtessera_nvidia_rng.so``) check their launches with a post-launch
``cudaGetLastError``, so before SPECTRAL-STALE-HIP-ERROR-2026-09-27 an
unrelated refused runtime call earlier on the thread made the next correct
launch report failure -- the CUDA twin of the gfx1201 streaming STFT rc=246.
Each exported entry that does device work now discards errors older than the
call, once, on entry.

Every checked entry is wrapped here so that the slot is primed immediately
before each call (``cudaSetDevice`` on an ordinal that does not exist, proven
by ``cudaPeekAtLastError``, which does not reset the slot), and each test then
requires the public path to succeed, agree with its reference, and to have
actually reached the entries it names. On a library without the entry clear,
each test fails with that entry's launch-check status.
"""

from __future__ import annotations

import ctypes
import functools
import os
from collections import Counter

import numpy as np
import pytest

import tessera

# Declares the hardware this file needs (see tests/_support/device_accounting.py).
pytestmark = pytest.mark.hardware_nvidia

_MISSING_ORDINAL = 97

#: Entries whose device work the rule covers (the ones that clear on entry,
#: plus the storage/streaming wrappers that reach a clearing entry).
_FFT_WORK_ENTRIES = (
    "tessera_nvidia_fft_execute_c2c_f32",
    "tessera_nvidia_fft_execute_r2c_f32",
    "tessera_nvidia_fft_execute_c2r_f32",
    "tessera_nvidia_fft_execute_c2c_device_f32",
    "tessera_nvidia_fft_execute_r2c_device_f32",
    "tessera_nvidia_fft_execute_c2r_device_f32",
    "tessera_nvidia_dct_policy_layout_f32",
    "tessera_nvidia_dct_policy_layout_storage",
    "tessera_nvidia_stft_policy_broadcast_layout_f32",
    "tessera_nvidia_stft_policy_broadcast_layout_storage",
    "tessera_nvidia_stft_jvp_broadcast_layout_f32",
    "tessera_nvidia_stft_jvp_broadcast_layout_storage",
    "tessera_nvidia_istft_policy_broadcast_layout_f32",
    "tessera_nvidia_istft_policy_broadcast_layout_storage",
    "tessera_nvidia_istft_jvp_broadcast_layout_f32",
    "tessera_nvidia_istft_jvp_broadcast_layout_storage",
    "tessera_nvidia_stft_backward_broadcast_layout_f32",
    "tessera_nvidia_stft_backward_broadcast_layout_storage",
    "tessera_nvidia_istft_backward_broadcast_layout_f32",
    "tessera_nvidia_istft_backward_broadcast_layout_storage",
    "tessera_nvidia_streaming_stft_broadcast_layout_f32",
    "tessera_nvidia_spectral_conv_f32",
)
_RNG_WORK_ENTRIES = (
    "tessera_nvidia_philox_uniform_f32",
    "tessera_nvidia_philox_uniform_range_f32",
    "tessera_nvidia_philox_normal_f32",
    "tessera_nvidia_philox_dropout_f32",
)


@tessera.jit(target="nvidia_sm120", autodiff="jvp", wrt=("x", "window"))
def _stft_jvp(x, window):
    return tessera.ops.stft(
        x, window, axis=-1, n_fft=16, hop=8, center=False,
        onesided=True, norm="ortho",
    )


@tessera.jit(target="nvidia_sm120", autodiff="jvp", wrt=("spectrum", "window"))
def _istft_jvp(spectrum, window):
    return tessera.ops.istft(
        spectrum, window, axis=-1, n_fft=16, hop=8, center=False,
        onesided=True, length=56, norm="ortho",
    )


@tessera.jit(target="nvidia_sm120", autodiff="reverse", wrt=("x", "window"))
def _stft_vjp(x, window):
    return tessera.ops.stft(
        x, window, axis=-1, n_fft=16, hop=8, center=False,
        onesided=True, norm="backward",
    )


@tessera.jit(target="nvidia_sm120", autodiff="reverse", wrt=("spectrum", "window"))
def _istft_vjp(spectrum, window):
    return tessera.ops.istft(
        spectrum, window, axis=-1, n_fft=16, hop=8, center=False,
        onesided=True, length=56, norm="backward",
    )


def _loaded_cudart() -> ctypes.CDLL:
    """The one CUDA runtime instance the hook libraries resolve against.

    The hooks link the shared ``libcudart.so.13``; a second runtime instance in
    the process (a static cudart, another soname) would own a different slot,
    and priming it would prove nothing. So the priming handle is the mapped
    library itself, and there must be exactly one.
    """
    with open("/proc/self/maps", encoding="utf-8") as maps:
        paths = {os.path.realpath(line.split()[-1]) for line in maps
                 if "libcudart.so" in line and line.split()[-1].startswith("/")}
    assert len(paths) == 1, f"expected one CUDA runtime instance, found {sorted(paths)}"
    cudart = ctypes.CDLL(paths.pop())
    for name, argtypes in (("cudaSetDevice", [ctypes.c_int]),
                           ("cudaGetDeviceCount", [ctypes.POINTER(ctypes.c_int)]),
                           ("cudaPeekAtLastError", []),
                           ("cudaGetLastError", [])):
        function = getattr(cudart, name)
        function.argtypes = argtypes
        function.restype = ctypes.c_int
    return cudart


def _prime_stale_cuda_error(cudart: ctypes.CDLL) -> None:
    """Leave an unread, non-sticky error in this thread's last-error slot."""
    count = ctypes.c_int()
    assert cudart.cudaGetDeviceCount(ctypes.byref(count)) == 0
    assert count.value <= _MISSING_ORDINAL, "the priming ordinal must not exist"
    assert cudart.cudaSetDevice(_MISSING_ORDINAL) != 0
    # cudaPeekAtLastError reads without resetting: the precondition of every
    # call below is a primed slot, not merely a failed call.
    assert cudart.cudaPeekAtLastError() != 0, (
        "cudaSetDevice failure did not reach the last-error slot")


def _require_sm120_fft():
    from tessera import runtime

    lib = runtime._load_nvidia_fft_runtime()
    if lib is None or not hasattr(lib, "tessera_nvidia_spectral_package_abi"):
        pytest.skip("NVIDIA spectral/FFT package is unavailable")
    if lib.tessera_nvidia_spectral_arch() != 120:
        pytest.skip("exact sm_120 device is unavailable")
    return runtime, lib


def _require_rng():
    from tessera import runtime

    if runtime._load_nvidia_rng_runtime() is None:
        pytest.skip("libtessera_nvidia_rng.so not built")
    try:
        runtime._nvidia_philox_uniform(0, 0, 1)
    except RuntimeError as exc:
        pytest.skip(f"no usable NVIDIA GPU: {exc}")
    return runtime, runtime._load_nvidia_rng_runtime()


class _Primed:
    """Wraps checked entries so each call starts on a primed slot, and counts
    the calls so a test can prove it reached the entries it claims to cover (a
    path that fell back elsewhere would otherwise pass vacuously)."""

    def __init__(self, monkeypatch):
        self.calls: Counter[str] = Counter()
        self.cudart: ctypes.CDLL | None = None
        self._monkeypatch = monkeypatch

    def install(self, lib, names) -> None:
        if self.cudart is None:
            self.cudart = _loaded_cudart()
        cudart = self.cudart
        for name in names:
            if not hasattr(lib, name):
                continue
            original = getattr(lib, name)

            @functools.wraps(original)
            def wrapper(*args, _original=original, _name=name):
                _prime_stale_cuda_error(cudart)
                self.calls[_name] += 1
                return _original(*args)

            self._monkeypatch.setattr(lib, name, wrapper)

    def reached(self, *names) -> None:
        missing = [name for name in names if self.calls[name] == 0]
        assert not missing, (
            f"entries not reached (the test would be vacuous): {missing}; "
            f"reached {dict(self.calls)}")


@pytest.fixture
def primed(monkeypatch):
    """A :class:`_Primed`; drains the slot afterwards, whatever the test did."""
    state = _Primed(monkeypatch)
    yield state
    if state.cudart is not None:
        state.cudart.cudaGetLastError()


def _artifact(runtime, compiler_path, op_name, operands, kwargs):
    names = [f"a{index}" for index in range(len(operands))]
    return runtime.RuntimeArtifact(metadata={
        "target": "nvidia_sm120",
        "compiler_path": compiler_path,
        "executable": True,
        "execution_kind": "native_gpu",
        "arg_names": names,
        "output_name": "output",
        "ops": [{"op_name": op_name, "result": "output",
                 "operands": names, "kwargs": dict(kwargs)}],
    })


def _launch(runtime, compiler_path, op_name, operands, kwargs):
    result = runtime.launch(
        _artifact(runtime, compiler_path, op_name, operands, kwargs), tuple(operands))
    assert result["ok"] is True, result.get("reason")
    assert result["compiler_path"] == compiler_path
    return np.asarray(result["output"])


def test_fft_host_executes_survive_a_stale_cuda_error(primed):
    runtime, lib = _require_sm120_fft()
    primed.install(lib, _FFT_WORK_ENTRIES)
    rng = np.random.default_rng(2701)
    rows = (rng.standard_normal((3, 256)) +
            1j * rng.standard_normal((3, 256))).astype(np.complex64)
    # The inverse C2C runs the normalization kernel, whose check reads the slot.
    np.testing.assert_allclose(
        runtime._nvidia_fft_c2c_rows(rows, True, np), np.fft.ifft(rows, axis=-1),
        rtol=2e-5, atol=2e-5)
    real = rng.standard_normal((3, 128)).astype(np.float32)
    half = runtime._nvidia_fft_real_rows(real, False, None, np)
    np.testing.assert_allclose(half, np.fft.rfft(real, axis=-1), rtol=2e-4, atol=2e-4)
    back = runtime._nvidia_fft_real_rows(half, True, 128, np)
    np.testing.assert_allclose(back, real, rtol=2e-5, atol=2e-5)
    primed.reached("tessera_nvidia_fft_execute_c2c_f32",
             "tessera_nvidia_fft_execute_r2c_f32",
             "tessera_nvidia_fft_execute_c2r_f32")


@pytest.mark.parametrize("kind", ("c2c_inverse", "c2r"))
def test_fft_device_pointer_executes_survive_a_stale_cuda_error(primed, kind):
    _, lib = _require_sm120_fft()
    lib.tessera_nvidia_fft_execute_c2c_device_f32.argtypes = [
        ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
        ctypes.c_size_t, ctypes.c_int, ctypes.c_void_p]
    lib.tessera_nvidia_fft_execute_c2r_device_f32.argtypes = [
        ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
        ctypes.c_size_t, ctypes.c_void_p]
    primed.install(lib, _FFT_WORK_ENTRIES)
    cudart = _loaded_cudart()
    cudart.cudaMalloc.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_size_t]
    cudart.cudaFree.argtypes = [ctypes.c_void_p]
    cudart.cudaMemcpy.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t,
                                  ctypes.c_int]
    batch, length = 3, 512
    rng = np.random.default_rng(2702)
    if kind == "c2c_inverse":
        host_in = (rng.standard_normal((batch, length)) +
                   1j * rng.standard_normal((batch, length))).astype(np.complex64)
        host_out = np.empty_like(host_in)
        create = lib.tessera_nvidia_fft_plan_create_c2c_f32
        expected = np.fft.ifft(host_in, axis=-1)
        entry = "tessera_nvidia_fft_execute_c2c_device_f32"
    else:
        real = rng.standard_normal((batch, length)).astype(np.float32)
        host_in = np.fft.rfft(real, axis=-1).astype(np.complex64)
        host_out = np.empty((batch, length), np.float32)
        create = lib.tessera_nvidia_fft_plan_create_c2r_f32
        expected = real
        entry = "tessera_nvidia_fft_execute_c2r_device_f32"
    plan, size = ctypes.c_void_p(), ctypes.c_size_t()
    assert create(batch, length, ctypes.byref(plan), ctypes.byref(size)) == 0
    workspace = ctypes.c_void_p()
    assert lib.tessera_nvidia_fft_workspace_alloc(size.value, ctypes.byref(workspace)) == 0
    device_in, device_out = ctypes.c_void_p(), ctypes.c_void_p()
    try:
        assert cudart.cudaMalloc(ctypes.byref(device_in), host_in.nbytes) == 0
        assert cudart.cudaMalloc(ctypes.byref(device_out), host_out.nbytes) == 0
        assert cudart.cudaMemcpy(device_in, host_in.ctypes.data, host_in.nbytes, 1) == 0
        if kind == "c2c_inverse":
            rc = lib.tessera_nvidia_fft_execute_c2c_device_f32(
                plan, device_in, device_out, workspace, size.value, 1, None)
        else:
            rc = lib.tessera_nvidia_fft_execute_c2r_device_f32(
                plan, device_in, device_out, workspace, size.value, None)
        assert rc == 0, f"{entry} returned {rc} on a primed slot"
        # The synchronous copy orders after the (unsynchronized) transform.
        assert cudart.cudaMemcpy(host_out.ctypes.data, device_out, host_out.nbytes, 2) == 0
        np.testing.assert_allclose(host_out, expected, rtol=2e-4, atol=2e-4)
    finally:
        for pointer in (device_in, device_out):
            if pointer.value:
                cudart.cudaFree(pointer)
        lib.tessera_nvidia_fft_workspace_free(workspace)
        lib.tessera_nvidia_fft_plan_destroy(plan)
    primed.reached(entry)


@pytest.mark.parametrize("dct_type", (2, 4))
def test_dct_survives_a_stale_cuda_error(primed, dct_type):
    dct = pytest.importorskip("scipy.fft").dct

    runtime, lib = _require_sm120_fft()
    primed.install(lib, _FFT_WORK_ENTRIES)
    rng = np.random.default_rng(2703 + dct_type)
    value = rng.standard_normal((3, 18, 2)).astype(np.float32)[:, ::2, :]
    actual = _launch(runtime, "nvidia_spectral_compiled", "tessera.dct", (value,),
                     {"axis": 1, "type": dct_type})
    np.testing.assert_allclose(actual, dct(value, type=dct_type, axis=1),
                               rtol=3e-5, atol=3e-5)
    primed.reached("tessera_nvidia_dct_policy_layout_storage")


def test_stft_istft_and_convolution_survive_a_stale_cuda_error(primed):
    runtime, lib = _require_sm120_fft()
    primed.install(lib, _FFT_WORK_ENTRIES)
    rng = np.random.default_rng(2705)
    x = rng.standard_normal((2, 37)).astype(np.float32)
    window = (0.25 + np.hanning(12)).astype(np.float32)
    spectrum = _launch(runtime, "nvidia_spectral_compiled", "tessera.stft", (x, window), {
        "axis": -1, "n_fft": 16, "hop": 5, "center": True,
        "pad_mode": "constant", "onesided": True, "normalization": "backward",
    })
    expected = tessera.ops.stft(x, window, axis=-1, n_fft=16, hop=5, center=True,
                                pad_mode="constant", onesided=True)
    np.testing.assert_allclose(spectrum, expected, rtol=4e-5, atol=4e-5)
    signal = _launch(runtime, "nvidia_spectral_compiled", "tessera.istft",
                     (spectrum, window), {
                         "axis": -1, "n_fft": 16, "hop": 5, "center": True,
                         "length": 35, "onesided": True, "normalization": "backward",
                     })
    expected_signal = tessera.ops.istft(
        np.asarray(spectrum), window, axis=-1, n_fft=16, hop=5, center=True,
        length=35, onesided=True)
    np.testing.assert_allclose(signal, expected_signal, rtol=5e-5, atol=5e-5)

    a = rng.standard_normal((4, 257)).astype(np.float32)
    b = rng.standard_normal((4, 33)).astype(np.float32)
    conv = _launch(runtime, "nvidia_spectral_compiled", "tessera.spectral_conv",
                   (a, b), {"normalization": "backward"})
    expected_conv = np.stack([np.convolve(a[row], b[row]) for row in range(4)])
    np.testing.assert_allclose(conv, expected_conv, rtol=5e-5,
                               atol=5e-5 * max(1.0, float(np.max(np.abs(expected_conv)))))
    primed.reached("tessera_nvidia_stft_policy_broadcast_layout_storage",
             "tessera_nvidia_istft_policy_broadcast_layout_storage",
             "tessera_nvidia_spectral_conv_f32")


def test_streaming_stft_survives_a_stale_cuda_error(primed):
    from tessera.compiler.spectral_streaming import StreamingSTFTPolicy, stream_stft_chunk

    _, lib = _require_sm120_fft()
    primed.install(lib, _FFT_WORK_ENTRIES)
    rng = np.random.default_rng(2706)
    signal = rng.standard_normal((2, 46, 3)).astype(np.float32)[:, ::2, :]
    window = np.stack((np.hanning(6), np.hamming(6)), axis=0).astype(
        np.float32)[:, None, :]
    policy = StreamingSTFTPolicy(axis=1, n_fft=8, window_length=6, hop=4,
                                 onesided=True, max_chunk_samples=9)
    state = None
    outputs = []
    for piece in np.split(signal, [7, 16], axis=1):
        output, state = stream_stft_chunk(piece, window, policy, state,
                                          target="nvidia_sm120")
        outputs.append(output)
    expected = tessera.ops.stft(signal, window, axis=1, n_fft=8, hop=4,
                                center=False, onesided=True)
    np.testing.assert_allclose(np.concatenate(outputs, axis=1), expected,
                               atol=3e-5, rtol=3e-5)
    assert state is not None
    assert state.execution_certificate["architecture_identity"] == "sm_120"
    primed.reached("tessera_nvidia_streaming_stft_broadcast_layout_f32")


def test_stft_istft_jvp_and_vjp_survive_a_stale_cuda_error(primed):
    from tessera.autodiff import vjp

    _, lib = _require_sm120_fft()
    primed.install(lib, _FFT_WORK_ENTRIES)
    rng = np.random.default_rng(2707)
    x = rng.normal(size=(2, 56)).astype(np.float32)
    window = (0.25 + np.hanning(16)).astype(np.float32)
    dx = rng.normal(size=x.shape).astype(np.float32)
    dwindow = rng.normal(size=window.shape).astype(np.float32)
    primal, tangent = _stft_jvp.native_jvp(x, window, tangents=(dx, dwindow))
    frames = [(at, at + 16) for at in range(0, 41, 8)]
    np.testing.assert_allclose(primal, np.fft.rfft(np.stack(
        [x[:, a:b] * window for a, b in frames], axis=1), axis=-1) / 4.0,
        rtol=3e-5, atol=3e-5)
    np.testing.assert_allclose(tangent, np.fft.rfft(np.stack(
        [dx[:, a:b] * window + x[:, a:b] * dwindow for a, b in frames], axis=1),
        axis=-1) / 4.0, rtol=4e-5, atol=4e-5)

    spectrum = (rng.normal(size=(2, 6, 9)) +
                1j * rng.normal(size=(2, 6, 9))).astype(np.complex64)
    spectrum[..., (0, -1)] = spectrum[..., (0, -1)].real
    dspectrum = (rng.normal(size=spectrum.shape) +
                 1j * rng.normal(size=spectrum.shape)).astype(np.complex64)
    dspectrum[..., (0, -1)] = dspectrum[..., (0, -1)].real
    primal, _ = _istft_jvp.native_jvp(spectrum, window, tangents=(dspectrum, dwindow))
    np.testing.assert_allclose(primal, tessera.ops.istft(
        spectrum, window, n_fft=16, hop=8, length=56, norm="ortho"),
        rtol=4e-5, atol=4e-5)

    signal = rng.standard_normal(56).astype(np.float32)
    dy = (rng.standard_normal((6, 9)) +
          1j * rng.standard_normal((6, 9))).astype(np.complex64)
    actual = _stft_vjp.native_backward(signal, window, out_cotangents=dy)
    expected = vjp._VJPS["stft"](dy, signal, window, axis=-1, n_fft=16, hop=8,
                                 center=False, onesided=True, norm="backward")
    for got, want in zip(actual, expected):
        np.testing.assert_allclose(got, want, rtol=5e-5, atol=5e-5)
    one_spectrum = spectrum[0]
    dsignal = rng.standard_normal(56).astype(np.float32)
    actual = _istft_vjp.native_backward(one_spectrum, window, out_cotangents=dsignal)
    expected = vjp._VJPS["istft"](dsignal, one_spectrum, window, axis=-1, n_fft=16,
                                  hop=8, center=False, onesided=True, length=56,
                                  norm="backward")
    for got, want in zip(actual, expected):
        np.testing.assert_allclose(got, want, rtol=5e-5, atol=5e-5)
    primed.reached("tessera_nvidia_stft_jvp_broadcast_layout_storage",
             "tessera_nvidia_istft_jvp_broadcast_layout_storage")
    backward = [name for name in primed.calls if "backward" in name]
    assert backward, f"no backward entry reached: {dict(primed.calls)}"


def test_philox_entries_survive_a_stale_cuda_error(primed):
    from tessera import rng_device as reference

    runtime, lib = _require_rng()
    primed.install(lib, _RNG_WORK_ENTRIES)
    np.testing.assert_array_equal(runtime._nvidia_philox_uniform(9, 41, 1003),
                                  reference.philox_uniform(9, 41, 1003))
    key = np.array([0x1234, 0x55], dtype=np.uint64)
    counter = np.array([19], dtype=np.uint64)
    ranged = _launch(runtime, "nvidia_rng_compiled", "tessera.rng_philox_uniform",
                     (key, counter), {"shape": [17], "lo": -2.0, "hi": 3.0})
    np.testing.assert_array_equal(
        ranged, reference.uniform(int(key[0]) ^ int(key[1]), 17, -2.0, 3.0, 19))
    normal = _launch(runtime, "nvidia_rng_compiled", "tessera.rng_normal", (),
                     {"seed": 7, "counter_base": 11, "shape": [100],
                      "mean": 2.0, "std": 0.5})
    np.testing.assert_allclose(normal, reference.normal(7, 100, 2.0, 0.5, 11),
                               rtol=4e-6, atol=4e-6)
    x = np.linspace(-3.0, 3.0, 2003, dtype=np.float32)
    dropped = _launch(runtime, "nvidia_rng_compiled", "tessera.dropout", (x,),
                      {"seed": 3, "counter_base": 29, "p": 0.3, "training": True})
    np.testing.assert_array_equal(dropped, x * reference.dropout_mask(3, x.size, 0.3, 29))
    primed.reached(*_RNG_WORK_ENTRIES)
