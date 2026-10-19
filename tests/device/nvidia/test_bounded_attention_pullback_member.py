"""Actual attention-source normalization adjoint, as a bounded native member."""
import ctypes as ct
import math
import re

import ml_dtypes
import numpy as np
import pytest

from tessera import runtime as rt
from tessera.compiler.emit.nvidia_cuda import NvidiaDeviceSession
from tessera.compiler.nvidia_native import _compile_tile_ir
from tessera.compiler.scheduled_matmul import find_tessera_opt
from tests.device.nvidia.test_scalar_adjoint_contraction import driver
from tests.unit.test_sm120_bounded_attention_pullback import actual_bounded_pullback
from tests.unit.test_sm120_reverse_row_schedule import invoke

pytestmark = pytest.mark.hardware_nvidia


@pytest.mark.parametrize("storage", ["f16", "bf16"])
@pytest.mark.parametrize("capacity", [(2, 3, 7, 35), (1, 2, 17, 257)])
def test_actual_attention_pullback_member_replays_runtime_queries(storage, capacity):
    assert rt._nvidia_device_name() == "sm_120", "owning SM120 required"
    tool = find_tessera_opt()
    assert tool is not None
    graph = actual_bounded_pullback(tool, storage, capacity)
    schedule = invoke(tool, graph, "--tessera-graph-to-schedule")
    assert schedule.returncode == 0, schedule.stderr
    tile = invoke(tool, schedule.stdout, "--tessera-schedule-to-tile")
    assert tile.returncode == 0, tile.stderr
    entry = re.search(r'name = "([^"]+)"', tile.stdout)[1]
    _, ptx, *_ = _compile_tile_ir(tile.stdout, entry)
    dtype = np.float16 if storage == "f16" else ml_dtypes.bfloat16
    count = math.prod(capacity)
    lib, module, function = driver(), ct.c_void_p(), ct.c_void_p()
    with NvidiaDeviceSession() as session:
        encoded = ct.create_string_buffer(ptx.encode())
        assert lib.cuModuleLoadData(ct.byref(module), encoded) == 0
        try:
            assert lib.cuModuleGetFunction(ct.byref(function), module, entry.encode()) == 0
            for step, sq in enumerate((1, capacity[2], 3, 1)):
                shape = (*capacity[:2], sq, capacity[-1])
                rng = np.random.default_rng(941 + step)
                x = rng.normal(0, .75, shape).astype(dtype)
                g = rng.normal(0, .25, shape).astype(dtype)
                if step == 2:
                    x.fill(dtype(1e-7))
                elif step == 3:
                    x.fill(dtype(0))
                xx, gg = x.astype(np.float64), g.astype(np.float64)
                inv = 1 / np.sqrt(np.mean(xx * xx, axis=-1, keepdims=True) + 1e-5)
                expected = gg * inv - xx * inv**3 * np.mean(gg * xx, axis=-1, keepdims=True)
                rounded = expected.astype(dtype)
                ulp = np.abs(np.nextafter(rounded, dtype(np.inf)).astype(np.float64)
                             - rounded.astype(np.float64))
                budget = capacity[-1] * 32 * np.finfo(np.float32).eps * (np.abs(expected) + .01) + ulp
                a, b = session.upload(x), session.upload(g)
                out = session.upload(np.full(count + 16, 123, dtype=dtype))
                args = [ct.c_uint64(a.ptr), ct.c_uint64(b.ptr), ct.c_uint64(out.ptr),
                        ct.c_int64(math.prod(shape[:-1])), ct.c_int64(shape[-1])]
                params = (ct.c_void_p * 5)(*(ct.cast(ct.pointer(v), ct.c_void_p) for v in args))
                assert lib.cuLaunchKernel(function, (math.prod(capacity[:-1]) + 127) // 128,
                                          1, 1, 128, 1, 1, 0,
                                          ct.c_void_p(session.stream), params, None) == 0
                assert session.synchronize() == 0
                actual = session.download(out)
                values = actual[:expected.size].reshape(shape).astype(np.float64)
                assert np.isfinite(values).all()
                assert np.all(np.abs(values - expected) <= budget)
                np.testing.assert_array_equal(actual[expected.size:], dtype(123))
        finally:
            assert session.synchronize() == 0
            if module.value:
                assert lib.cuModuleUnload(module) == 0
