"""Opt-in native SDK27 descriptor checks; these do not prove GPU execution."""
import ctypes as ct
import os

import pytest


@pytest.mark.parametrize('dtype', [0, 1, 2])
def test_sdk27_microscaled_descriptor(dtype):
    path = os.environ.get('TESSERA_SDK27_DESCRIPTOR_LIB')
    if not path:
        pytest.skip('requires a fresh SDK27 runtime on macOS27')
    lib = ct.CDLL(path)
    probe = lib.tessera_apple_gpu_microscaled_descriptor_probe
    probe.argtypes = [ct.c_int32, ct.c_int32, ct.c_int32,
                      ct.POINTER(ct.c_int64), ct.POINTER(ct.c_int64)]
    probe.restype = ct.c_int32
    dims = (ct.c_int64 * 2)(64, 16)  # Metal innermost-first order
    factors = (ct.c_int64 * 2)(32, 1)
    assert probe(dtype, 3, 2, dims, factors) == 1
    assert probe(dtype, -1, 2, dims, None) == 1
    assert probe(dtype, 0, 2, dims, factors) == 0  # E4M3 is not a scales-plane format
    assert probe(dtype, 3, 2, dims, None) == 0
    assert probe(dtype, 3, 2, (ct.c_int64 * 2)(63, 16), factors) == 0
    assert probe(dtype, 3, 2, dims, (ct.c_int64 * 2)(3, 1)) == 0
    assert probe(dtype, 3, 0, dims, factors) == 0
    assert probe(3, -1, 2, dims, None) == 0  # UE8M0 is scale-only
    assert lib.tessera_apple_gpu_supports_microscaling() == 0
