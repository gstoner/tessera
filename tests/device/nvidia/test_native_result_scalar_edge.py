"""Owned FP32 producer/JVP generations feed a compiled mixed-storage consumer."""
from contextlib import ExitStack
import subprocess

import numpy as np
import pytest
import tessera as ts

from tessera import runtime as rt
from tessera.compiler.canonical_compile import canonical_compile
from tessera.compiler.emit.nvidia_cuda import NvidiaDeviceSession
from tessera.compiler.native_tensor_result import NativeTensorResult
from benchmarks.nvidia.active_source_jvp_reference import inputs, oracle, product

pytestmark = pytest.mark.hardware_nvidia


@pytest.mark.parametrize("storage", ["f16", "bf16"])
@pytest.mark.parametrize("slot", [0, 1])
def test_owned_f32_result_reaches_compiled_scalar_consumer(storage, slot, monkeypatch):
    import ml_dtypes
    assert rt._nvidia_device_name() == "sm_120"
    a, b, da = inputs({"shape_mnk": (17, 19, 35), "storage": storage})
    upstream, upstream_budget = oracle(a, b, da)
    producer = ts.jit(target="nvidia_sm120", autodiff="forward", wrt=("a",))(product)
    pair = producer.native_jvp(a, b, tangents=(da,), result_placement="cuda")
    try:
        actual_source = pair[slot].to_host()
        assert np.all(np.abs(actual_source.astype(np.float64) - upstream[slot]) <= upstream_budget[slot])
        # Consumer storage is independent of the producer's original half storage.
        dtype = ml_dtypes.bfloat16 if storage == "f16" else np.float16
        weights = np.random.default_rng(181).normal(0, .25, (19, 13)).astype(dtype)
        frontend = ts.from_text(
            "def consumer(x, w):\n    return tessera.ops.matmul(x, w, output_dtype='fp32')\n",
            target="nvidia_sm120")
        graph, _ = frontend._trace_frontend_capture((actual_source, weights), {})
        compiled = canonical_compile(graph, target="nvidia_sm120", enable_tool_validation=False)
        assert compiled.executable
        artifact = rt.RuntimeArtifact.from_json(compiled.to_runtime_artifact().to_json())
        names = [arg.name for arg in graph.functions[0].args]
        producer.close_native_storage()
        with ExitStack() as stack:
            sessions = [stack.enter_context(NvidiaDeviceSession()) for _ in range(2)]
            rhs = sessions[0].upload(weights)
            output = sessions[1].empty((17, 13), np.float32)
            values = {names[0]: pair[slot], names[1]: rhs,
                      artifact.launch_descriptor.buffers[-1].name: output}
            arguments = {"buffers": values, "scalars": {"M": 17, "N": 13, "K": 19}}
            with monkeypatch.context() as replay:
                replay.setattr(subprocess, "run", lambda *a, **k: pytest.fail("resident edge compiled"))
                replay.setattr(NativeTensorResult, "to_host",
                               lambda *a, **k: pytest.fail("consumer downloaded its source"))
                result = rt.launch(artifact, arguments, stream=sessions[1].stream)
            assert result["ok"] and result["execution_kind"] == "native_gpu", result
            actual = sessions[1].download(output).astype(np.float64)
            rhs_wide = weights.astype(np.float64)
            local_expected = actual_source.astype(np.float64) @ rhs_wide
            gamma = 19 * np.finfo(np.float32).eps / (1 - 19 * np.finfo(np.float32).eps)
            local_bound = gamma * (np.abs(actual_source.astype(np.float64)) @ np.abs(rhs_wide)) + 1e-7
            assert np.all(np.abs(actual - local_expected) <= local_bound)
            propagated = upstream_budget[slot] @ np.abs(rhs_wide) + local_bound
            assert np.all(np.abs(actual - upstream[slot] @ rhs_wide) <= propagated)
    finally:
        for value in pair:
            value.close()
        producer.close_native_storage()
