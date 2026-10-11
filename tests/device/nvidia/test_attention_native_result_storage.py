"""Native half result store diagnostic, separate from public package proof."""
import ctypes as ct
import json
import re

import numpy as np
import pytest
import tessera as ts
from tessera import runtime as rt
from tessera.compiler.emit.nvidia_cuda import NvidiaDeviceSession
from tessera.compiler.nvidia_native import _compile_tile_ir
from tests.unit.test_attention_native_result_storage import attention, run_tool
from tests.device.nvidia.test_resident_attention_forward import oracle

pytestmark = pytest.mark.hardware_nvidia


@pytest.mark.parametrize("dtype", ["float16", "bfloat16"])
def test_native_half_final_store_matches_independent_quantized_output(dtype):
    if rt._nvidia_device_name() != "sm_120":
        pytest.skip("owning RTX5070 required")
    import os
    import ml_dtypes
    storage = ml_dtypes.bfloat16 if dtype == "bfloat16" else np.dtype(dtype)
    rng = np.random.default_rng(120094)
    values = tuple(rng.normal(0, .2, shape).astype(storage) for shape in
                   ((1, 2, 3, 4), (1, 1, 5, 4), (1, 1, 5, 3)))
    fn = ts.jit(target="nvidia_sm120")(attention)
    module, _ = fn._trace_frontend_capture(values, {})
    op = module.functions[0].body[0]
    op.kwargs.update(scale=.5, causal=True, window_left=-1, window_right=-1,
                     softcap=0., dropout_p=0., dropout_seed=0)
    module.module_attrs["tessera.target"] = '"nvidia_sm120"'
    module.module_attrs["tessera.arch"] = '"sm_120"'
    module.module_attrs["tessera.launch_bindings"] = json.dumps(
        [arg.name for arg in module.functions[0].args] + [op.result])
    graph = module.to_mlir(target="nvidia_sm120", canonical=True)
    schedule = run_tool(os.environ["TESSERA_OPT"], graph, "--tessera-graph-to-schedule")
    tile = run_tool(os.environ["TESSERA_OPT"], schedule, "--tessera-schedule-to-tile")
    entry = re.search(r"llvm.func @([^ (]+)", tile).group(1)
    _, ptx, *_ = _compile_tile_ir(tile, entry)
    driver = ct.CDLL("libcuda.so.1")
    P, U = ct.c_void_p, ct.c_uint
    driver.cuModuleLoadData.argtypes = [ct.POINTER(P), P]
    driver.cuModuleGetFunction.argtypes = [ct.POINTER(P), P, ct.c_char_p]
    driver.cuModuleUnload.argtypes = [P]
    driver.cuLaunchKernel.argtypes = [P] + [U] * 7 + [P, ct.POINTER(P), ct.POINTER(P)]
    with NvidiaDeviceSession() as session:
        inputs = tuple(session.upload(value) for value in values)
        output = session.empty((1, 2, 3, 3), storage)
        image = ct.create_string_buffer(ptx.encode())
        handle, kernel = P(), P()
        assert driver.cuModuleLoadData(ct.byref(handle), image) == 0
        try:
            assert driver.cuModuleGetFunction(ct.byref(kernel), handle, entry.encode()) == 0
            slots = [ct.c_uint64(value.ptr) for value in (*inputs, output)]
            slots += [ct.c_int64(value) for value in (1, 2, 1, 3, 5, 4, 3)]
            parameters = (P * len(slots))(*(ct.cast(ct.byref(slot), P) for slot in slots))
            assert driver.cuLaunchKernel(kernel, 1, 1, 1, 128, 1, 1, 0,
                                         P(session.stream), parameters, None) == 0
            assert session.synchronize() == 0
            actual = session.download(output)
            np.testing.assert_array_equal(actual, oracle(values)[0].astype(storage))
        finally:
            assert driver.cuModuleUnload(handle) == 0
