"""Exact SM120 bounded-query RMSNorm member; no composed-public claim."""
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
from tests.unit.test_sm120_bounded_norm_schedule import source
from tests.unit.test_sm120_reverse_row_schedule import invoke

pytestmark = pytest.mark.hardware_nvidia


@pytest.mark.parametrize("storage", ["f16", "bf16"])
@pytest.mark.parametrize("capacity", [(2, 3, 7, 35), (1, 2, 17, 257)])
@pytest.mark.parametrize("mode", ["serial", "cooperative_128"])
def test_bounded_query_norm_reuses_image_and_guards_launch(storage, capacity, mode):
    assert rt._nvidia_device_name() == "sm_120", "owning SM120 required"
    tool = find_tessera_opt()
    assert tool is not None
    graph = source(storage, capacity).replace('numeric_policy = {accum = "fp32"}',
                                             f'numeric_policy = {{accum = "fp32"}}, schedule = "{mode}"')
    schedule = invoke(tool, graph, "--tessera-graph-to-schedule")
    assert schedule.returncode == 0, schedule.stderr
    tile = invoke(tool, schedule.stdout, "--tessera-schedule-to-tile")
    assert tile.returncode == 0, tile.stderr
    entry = re.search(r'\bllvm.func @(\w+)\(', tile.stdout)[1]
    _, ptx, *_ = _compile_tile_ir(tile.stdout, entry)
    dtype = np.float16 if storage == "f16" else ml_dtypes.bfloat16
    count, prefix, columns = math.prod(capacity), math.prod(capacity[:2]), capacity[-1]
    max_rows = prefix * capacity[2]
    grid = max_rows if mode == "cooperative_128" else (max_rows + 127) // 128
    lib, module, function = driver(), ct.c_void_p(), ct.c_void_p()
    with NvidiaDeviceSession() as session:
        encoded = ct.create_string_buffer(ptx.encode())
        assert lib.cuModuleLoadData(ct.byref(module), encoded) == 0
        try:
            assert lib.cuModuleGetFunction(ct.byref(function), module, entry.encode()) == 0
            for step, sq in enumerate((1, capacity[2], 3, capacity[2], 1)):
                shape = (*capacity[:2], sq, columns)
                rng = np.random.default_rng(19102026 + step)
                x = rng.normal(0, .5, shape).astype(dtype)
                if step == 2:
                    x.fill(dtype(1e-7))
                elif step == 3:
                    x.reshape(-1)[:2] = [dtype(np.inf), dtype(-np.inf)]
                elif step == 4:
                    x.fill(dtype(0))
                    x.reshape(-1)[0] = dtype(-0.0)
                xx = x.astype(np.float64)
                with np.errstate(invalid="ignore"):
                    expected = xx / np.sqrt(np.mean(xx * xx, axis=-1, keepdims=True)
                                            + float(np.float32(1e-5)))
                    rounded = expected.astype(dtype)
                    ulp = np.abs(np.nextafter(rounded, dtype(np.inf)).astype(np.float64)
                                 - rounded.astype(np.float64))
                budget = columns * 32 * np.finfo(np.float32).eps * (np.abs(expected) + .01) + ulp
                padded = np.zeros(count, dtype=dtype)
                padded[:x.size] = x.reshape(-1)
                a = session.upload(padded)
                out = session.upload(np.full(count + 16, 123, dtype=dtype))
                args = [ct.c_uint64(a.ptr), ct.c_uint64(out.ptr),
                        ct.c_int64(prefix * sq), ct.c_int64(columns)]
                params = (ct.c_void_p * 4)(*(ct.cast(ct.pointer(v), ct.c_void_p) for v in args))

                def launch():
                    assert lib.cuLaunchKernel(function, grid, 1, 1, 128, 1, 1, 0,
                                              ct.c_void_p(session.stream), params, None) == 0

                launch()
                assert session.synchronize() == 0
                actual = session.download(out)
                values = actual[:x.size].reshape(shape).astype(np.float64)
                finite = np.isfinite(expected)
                np.testing.assert_array_equal(np.isnan(values), np.isnan(expected))
                np.testing.assert_array_equal(np.isinf(values), np.isinf(expected))
                assert np.all(np.abs(values[finite] - expected[finite]) <= budget[finite])
                zeros = finite & (expected == 0)
                np.testing.assert_array_equal(np.signbit(values[zeros]), np.signbit(expected[zeros]))
                np.testing.assert_array_equal(actual[x.size:], dtype(123))
                np.testing.assert_array_equal(session.download(a).view(np.uint8), padded.view(np.uint8))

            for rows, cols in ((0, columns), (-1, columns), (max_rows + prefix, columns),
                               (1, columns), (prefix, 0), (prefix, columns + 1),
                               (2**63 - 1, columns)):
                guarded = session.upload(np.full(count + 16, 123, dtype=dtype))
                args[0].value, args[1].value = 0, guarded.ptr
                args[2].value, args[3].value = rows, cols
                launch()
                assert session.synchronize() == 0
                np.testing.assert_array_equal(session.download(guarded), dtype(123))
        finally:
            assert session.synchronize() == 0
            if module.value:
                assert lib.cuModuleUnload(module) == 0
