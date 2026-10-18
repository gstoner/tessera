"""Actual paired-reverse row SSA; raw member proof, not composed VJP closure."""
import ctypes as ct
import re

import numpy as np
import pytest

from tessera import runtime as rt
from tessera.compiler.emit.nvidia_cuda import NvidiaDeviceSession
from tessera.compiler.nvidia_native import _compile_tile_ir
from tessera.compiler.scheduled_matmul import find_tessera_opt
from tests.device.nvidia.test_scalar_adjoint_contraction import driver
from tests.unit.test_sm120_reverse_row_schedule import invoke, reverse_source

pytestmark = pytest.mark.hardware_nvidia


@pytest.mark.parametrize("storage", ["f16", "bf16"])
def test_reverse_row_numerical_and_store_bounds(storage):
    import ml_dtypes
    assert rt._nvidia_device_name() == "sm_120"
    tool = find_tessera_opt()
    assert tool is not None
    graph = reverse_source(tool, storage)
    schedule = invoke(tool, graph, "--tessera-graph-to-schedule")
    assert schedule.returncode == 0, schedule.stderr
    tile = invoke(tool, schedule.stdout, "--tessera-schedule-to-tile")
    assert tile.returncode == 0, tile.stderr
    entry = re.search(r'name = "([^"]+)"', tile.stdout)[1]
    _, ptx, *_ = _compile_tile_ir(tile.stdout, entry)
    dtype = np.float16 if storage == "f16" else ml_dtypes.bfloat16
    lib, module, function = driver(), ct.c_void_p(), ct.c_void_p()
    with NvidiaDeviceSession() as session:
        encoded = ct.create_string_buffer(ptx.encode())
        assert lib.cuModuleLoadData(ct.byref(module), encoded) == 0
        try:
            assert lib.cuModuleGetFunction(ct.byref(function), module, entry.encode()) == 0
            for seed in (193, 419):
                rng = np.random.default_rng(seed)
                x = rng.normal(0, .75, (17, 35)).astype(dtype)
                g = rng.normal(0, .25, (17, 35)).astype(dtype)
                xx = x.astype(np.float64)
                # Independent RMSNorm pullback with the preceding ReLU mask.
                gg = np.where(xx > 0, g.astype(np.float64), 0)
                inverse = 1 / np.sqrt(np.mean(xx * xx, axis=1, keepdims=True) + 1e-5)
                reference = gg * inverse - xx * inverse**3 * np.mean(gg * xx, axis=1, keepdims=True)
                rounded = reference.astype(dtype)
                ulp = np.abs(np.nextafter(rounded, dtype(np.inf)).astype(np.float64) - rounded.astype(np.float64))
                budget = 35 * 32 * np.finfo(np.float32).eps * (np.abs(reference) + .01) + ulp
                a, b = session.upload(x), session.upload(g)
                out = session.upload(np.full(17 * 35 + 16, 123, dtype=dtype))
                args = [ct.c_uint64(a.ptr), ct.c_uint64(b.ptr), ct.c_uint64(out.ptr),
                        ct.c_int64(17), ct.c_int64(35)]
                params = (ct.c_void_p * 5)(*(ct.cast(ct.pointer(v), ct.c_void_p) for v in args))
                assert lib.cuLaunchKernel(function, 1, 1, 1, 128, 1, 1, 0,
                                          ct.c_void_p(session.stream), params, None) == 0
                assert session.synchronize() == 0
                actual = session.download(out)
                np.testing.assert_array_equal(actual[17 * 35:], dtype(123))
                actual = actual[:17 * 35].reshape(17, 35).astype(np.float64)
                assert np.isfinite(actual).all()
                assert np.all(np.abs(actual - reference) <= budget)
        finally:
            assert session.synchronize() == 0
            if module.value:
                assert lib.cuModuleUnload(module) == 0
