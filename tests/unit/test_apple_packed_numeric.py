"""Owning-Mac packed-buffer proof; not matrix or block-scale admission."""
import ctypes as ct
import os

import numpy as np
import pytest

# Exact-device boundary via the shared marker (conftest -> require_apple_metal),
# not an inline capability skip; the env var only selects which dylib to load.
pytestmark = pytest.mark.hardware_apple_gpu


@pytest.mark.parametrize('code,name,bits', [
    (0, 'float8_e4m3fn', 8), (1, 'float8_e5m2', 8), (2, 'float4_e2m1fn', 4),
])
def test_native_packed_encoding_and_arithmetic(code, name, bits):
    path = os.environ.get('TESSERA_PACKED_NUMERIC_LIB')
    if not path:
        pytest.skip('requires TESSERA_PACKED_NUMERIC_LIB pointing at a fresh SDK27 runtime')
    ml = pytest.importorskip('ml_dtypes')
    dtype = getattr(ml, name)
    lib = ct.CDLL(path)
    run = lib.tessera_apple_gpu_packed_numeric_status
    run.argtypes = [ct.c_int32, ct.c_void_p, ct.c_void_p, ct.c_int32,
                    ct.c_void_p, ct.c_void_p]
    run.restype = ct.c_int32
    codes = (np.arange(256) % (1 << bits)).astype(np.uint8)
    decoded = codes.view(dtype).astype(np.float32)
    packed = codes if bits == 8 else (codes[::2] | (codes[1::2] << 4))
    # Every encoding, signed zeros, subnormals and exceptional inputs, plus
    # representable midpoint ties exercise conversion separately from decode.
    finite = decoded[np.isfinite(decoded)]
    ordered = np.unique(finite)
    midpoints = (ordered[:-1] + ordered[1:]) * np.float32(.5)
    samples = np.resize(np.concatenate((decoded, midpoints, [1e9, -1e9])), 1024).astype(np.float32)
    packed = np.tile(packed, 4)
    decoded = np.tile(decoded, 4)
    out = np.full((4, 1024), np.nan, np.float32)
    repacked = np.zeros(1024 if bits == 8 else 512, np.uint8)
    assert run(code, packed.ctypes.data, samples.ctypes.data, 1024,
               out.ctypes.data, repacked.ctypes.data) == 1, _error(lib)
    with np.errstate(invalid='ignore', over='ignore'):
        expected = np.stack((decoded, decoded + 1, decoded * 2, decoded / 2))
    np.testing.assert_array_equal(out, expected)
    # NaN sign/payload is not preserved by Metal unpack; signed zeros are.
    non_nan = ~np.isnan(decoded)
    np.testing.assert_array_equal(np.signbit(out[0, non_nan]), np.signbit(decoded[non_nan]))
    # Metal E4M3/FP4 saturate finite/inf overflow; E5M2 defaults to no saturation.
    quant_input = samples if code == 1 else np.clip(samples, -np.max(finite), np.max(finite))
    with np.errstate(invalid='ignore', over='ignore'):
        quant_expected = quant_input.astype(dtype).astype(np.float32)
    unpacked = repacked if bits == 8 else np.column_stack((repacked & 15, repacked >> 4)).reshape(-1)
    np.testing.assert_array_equal(unpacked.view(dtype).astype(np.float32), quant_expected)
    # Invalid arguments must neither submit nor overwrite caller output.
    before = out.copy()
    assert run(code, packed.ctypes.data, samples.ctypes.data, 7,
               out.ctypes.data, repacked.ctypes.data) == 0
    np.testing.assert_array_equal(out, before)


def _error(lib):
    lib.tessera_apple_gpu_last_error_message.restype = ct.c_char_p
    return lib.tessera_apple_gpu_last_error_message()


def test_public_packed_binding(monkeypatch):
    path = os.environ.get('TESSERA_PACKED_NUMERIC_LIB')
    if not path:
        pytest.skip('requires TESSERA_PACKED_NUMERIC_LIB pointing at a fresh SDK27 runtime')
    from tessera import _apple_gpu_dispatch
    from tessera.compiler.apple_packed_numeric import evaluate

    monkeypatch.setattr(_apple_gpu_dispatch, 'apple_gpu_runtime', lambda: ct.CDLL(path))
    out, packed = evaluate(np.full(8, 0x38, np.uint8), np.ones(8, np.float32), 'fp8_e4m3')
    np.testing.assert_array_equal(out, np.repeat([[1.0], [2.0], [2.0], [0.5]], 8, axis=1))
    np.testing.assert_array_equal(packed, np.full(8, 0x38, np.uint8))
    with pytest.raises(ValueError, match='multiple of eight'):
        evaluate(np.zeros(7, np.uint8), np.zeros(7, np.float32), 'fp8_e4m3')


def test_packed_binding_failure_has_no_reference_fallback(monkeypatch):
    from tessera import _apple_gpu_dispatch
    from tessera.compiler.apple_packed_numeric import evaluate

    class RefusingRuntime:
        @staticmethod
        def tessera_apple_gpu_packed_numeric_status(*args):
            return 0

    monkeypatch.setattr(_apple_gpu_dispatch, 'apple_gpu_runtime', lambda: RefusingRuntime())
    with pytest.raises(RuntimeError, match='no CPU fallback'):
        evaluate(np.zeros(8, np.uint8), np.zeros(8, np.float32), 'fp8_e4m3')


def test_packed_binding_validates_storage_before_loading(monkeypatch):
    from tessera import _apple_gpu_dispatch
    from tessera.compiler.apple_packed_numeric import evaluate

    def forbidden():
        raise AssertionError('invalid input reached runtime loader')

    monkeypatch.setattr(_apple_gpu_dispatch, 'apple_gpu_runtime', forbidden)
    with pytest.raises(ValueError, match='packed uint8'):
        evaluate(np.zeros(8, np.uint8), np.zeros(8, np.float32), 'fp4_e2m1')
    with pytest.raises(ValueError, match='rank-one fp32'):
        evaluate(np.zeros(8, np.uint8), np.zeros(8, np.float64), 'fp8_e4m3')
