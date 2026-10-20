"""Exact SM120 bounded-query pullback member; composed VJP remains separate."""
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
from tests.device.nvidia.test_ranked_reverse_row_member import reference
from tests.device.nvidia.test_scalar_adjoint_contraction import driver
from tests.unit.test_sm120_bounded_source_rows import bounded_source
from tests.unit.test_sm120_reverse_row_schedule import invoke

pytestmark = pytest.mark.hardware_nvidia


@pytest.mark.parametrize("storage", ["f16", "bf16"])
@pytest.mark.parametrize("capacity", [(2, 3, 7, 35), (1, 2, 17, 257)])
def test_bounded_row_reuses_image_and_preserves_inactive_storage(storage, capacity):
    assert rt._nvidia_device_name() == "sm_120", "owning SM120 required"
    tool = find_tessera_opt()
    assert tool is not None
    graph = bounded_source(tool, storage, capacity)
    schedule = invoke(tool, graph, "--tessera-graph-to-schedule")
    assert schedule.returncode == 0, schedule.stderr
    tile = invoke(tool, schedule.stdout, "--tessera-schedule-to-tile")
    assert tile.returncode == 0, tile.stderr
    entry = re.search(r'name = "([^"]+)"', tile.stdout)[1]
    _, ptx, *_ = _compile_tile_ir(tile.stdout, entry)
    dtype = np.float16 if storage == "f16" else ml_dtypes.bfloat16
    prefix, columns = math.prod(capacity[:2]), capacity[-1]
    max_rows = prefix * capacity[2]
    count = max_rows * columns
    lib, module, function = driver(), ct.c_void_p(), ct.c_void_p()
    with NvidiaDeviceSession() as session:
        encoded = ct.create_string_buffer(ptx.encode())
        assert lib.cuModuleLoadData(ct.byref(module), encoded) == 0
        try:
            assert lib.cuModuleGetFunction(ct.byref(function), module, entry.encode()) == 0
            for step, sq in enumerate((1, capacity[2], 3, capacity[2], 1)):
                shape = (*capacity[:2], sq, columns)
                rng = np.random.default_rng(20261019 + step)
                x = rng.normal(0, .75, shape).astype(dtype)
                g = rng.normal(0, .25, shape).astype(dtype)
                if step == 2:
                    x.fill(dtype(1e-7))
                elif step == 4:
                    x.fill(dtype(0))
                expected = reference(x, g).reshape(-1)
                rounded = expected.astype(dtype)
                ulp = np.abs(np.nextafter(rounded, dtype(np.inf)).astype(np.float64)
                             - rounded.astype(np.float64))
                budget = columns * 32 * np.finfo(np.float32).eps * (np.abs(expected) + .01) + ulp
                active = expected.size
                # Source roots are compact runtime tensors in capacity allocations.
                xx, gg = np.zeros(count, dtype=dtype), np.zeros(count, dtype=dtype)
                xx[:active], gg[:active] = x.reshape(-1), g.reshape(-1)
                a, b = session.upload(xx), session.upload(gg)
                out = session.upload(np.full(count + 16, 123, dtype=dtype))
                args = [ct.c_uint64(a.ptr), ct.c_uint64(b.ptr), ct.c_uint64(out.ptr),
                        ct.c_int64(prefix * sq), ct.c_int64(columns)]
                params = (ct.c_void_p * 5)(*(ct.cast(ct.pointer(v), ct.c_void_p) for v in args))

                def launch():
                    assert lib.cuLaunchKernel(function, (max_rows + 127) // 128, 1, 1,
                                              128, 1, 1, 0, ct.c_void_p(session.stream), params, None) == 0

                launch()
                assert session.synchronize() == 0
                actual = session.download(out)
                np.testing.assert_array_equal(actual[active:], dtype(123))
                values = actual[:active].astype(np.float64)
                assert np.isfinite(values).all()
                assert np.all(np.abs(values - expected) <= budget)

            # Null inputs ensure rejected geometry cannot reach a read.
            for rows, cols in ((0, columns), (-1, columns),
                               (max_rows + prefix, columns), (1, columns),
                               (prefix, columns + 1), (2**63 - 1, columns)):
                guarded = session.upload(np.full(count + 16, 123, dtype=dtype))
                args[0].value = args[1].value = 0
                args[2].value = guarded.ptr
                args[3].value, args[4].value = rows, cols
                launch()
                assert session.synchronize() == 0
                np.testing.assert_array_equal(session.download(guarded), dtype(123))
        finally:
            assert session.synchronize() == 0
            if module.value:
                assert lib.cuModuleUnload(module) == 0
