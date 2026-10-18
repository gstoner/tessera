"""Owning SM120 rank-preserving native pullback member proof.

This executes compiled native AD row SSA, not a composed public VJP program.
"""
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
from tests.unit.test_sm120_ranked_reverse_rows import ranked_source
from tests.unit.test_sm120_reverse_row_schedule import invoke

pytestmark = pytest.mark.hardware_nvidia


def reference(x, g):
    xx = x.astype(np.float64)
    gg = np.where(xx > 0, g.astype(np.float64), 0)
    inverse = 1 / np.sqrt(np.mean(xx * xx, axis=-1, keepdims=True) + 1e-5)
    return gg * inverse - xx * inverse**3 * np.mean(gg * xx, axis=-1, keepdims=True)


@pytest.mark.parametrize("storage", ["f16", "bf16"])
@pytest.mark.parametrize("shape", [(2, 3, 35), (2, 3, 5, 35), (2, 1, 3, 257), (2, 4, 17, 257)])
def test_ranked_reverse_row_numerical_geometry_and_store_bounds(storage, shape):
    assert rt._nvidia_device_name() == "sm_120"
    tool = find_tessera_opt()
    assert tool is not None
    graph = ranked_source(tool, storage, shape)
    schedule = invoke(tool, graph, "--tessera-graph-to-schedule")
    assert schedule.returncode == 0, schedule.stderr
    tile = invoke(tool, schedule.stdout, "--tessera-schedule-to-tile")
    assert tile.returncode == 0, tile.stderr
    entry = re.search(r'name = "([^"]+)"', tile.stdout)[1]
    _, ptx, *_ = _compile_tile_ir(tile.stdout, entry)
    dtype = np.float16 if storage == "f16" else ml_dtypes.bfloat16
    rows, columns = math.prod(shape[:-1]), shape[-1]
    count = rows * columns
    lib, module, function = driver(), ct.c_void_p(), ct.c_void_p()
    with NvidiaDeviceSession() as session:
        encoded = ct.create_string_buffer(ptx.encode())
        assert lib.cuModuleLoadData(ct.byref(module), encoded) == 0
        try:
            assert lib.cuModuleGetFunction(ct.byref(function), module, entry.encode()) == 0
            for pattern in ("random", "tiny", "zero"):
                rng = np.random.default_rng(20261018)
                x = rng.normal(0, .75, shape).astype(dtype)
                g = rng.normal(0, .25, shape).astype(dtype)
                if pattern == "tiny":
                    x.fill(dtype(1e-7))
                elif pattern == "zero":
                    x.fill(dtype(0))
                expected = reference(x, g)
                rounded = expected.astype(dtype)
                ulp = np.abs(np.nextafter(rounded, dtype(np.inf)).astype(np.float64)
                             - rounded.astype(np.float64))
                budget = columns * 32 * np.finfo(np.float32).eps * (np.abs(expected) + .01) + ulp
                a, b = session.upload(x), session.upload(g)
                out = session.upload(np.full(count + 16, 123, dtype=dtype))
                args = [ct.c_uint64(a.ptr), ct.c_uint64(b.ptr), ct.c_uint64(out.ptr),
                        ct.c_int64(rows), ct.c_int64(columns)]
                params = (ct.c_void_p * 5)(*(ct.cast(ct.pointer(v), ct.c_void_p) for v in args))

                def launch():
                    assert lib.cuLaunchKernel(function, (rows + 127) // 128, 1, 1,
                                              128, 1, 1, 0, ct.c_void_p(session.stream), params, None) == 0

                launch()
                assert session.synchronize() == 0
                actual = session.download(out)
                np.testing.assert_array_equal(actual[count:], dtype(123))
                values = actual[:count].reshape(shape).astype(np.float64)
                assert np.isfinite(values).all()
                assert np.all(np.abs(values - expected) <= budget)
                # Geometry guards must leave the entire output untouched.
                for axis in (3, 4):
                    guarded = session.upload(np.full(count + 16, 123, dtype=dtype))
                    args[2].value = guarded.ptr
                    args[axis].value += 1
                    launch()
                    assert session.synchronize() == 0
                    np.testing.assert_array_equal(session.download(guarded), dtype(123))
                    args[axis].value -= 1
        finally:
            assert session.synchronize() == 0
            if module.value:
                assert lib.cuModuleUnload(module) == 0
