"""Exact-device public/native and serialized standalone activation proof."""
import subprocess

import ml_dtypes
import numpy as np
import pytest
import tessera as ts
from tessera import runtime as rt
from tests.unit.test_sm120_public_relu import relu

pytestmark = pytest.mark.hardware_nvidia


def values(storage, shape):
    dtype = {"f16": np.float16, "bf16": ml_dtypes.bfloat16, "f32": np.float32}[storage]
    patterns = {
        "f16": [0, 0x8000, 1, 0x8001, 0x7bff, 0xfbff, 0x7c00, 0xfc00, 0x7e11, 0xfe55],
        "bf16": [0, 0x8000, 1, 0x8001, 0x7f7f, 0xff7f, 0x7f80, 0xff80, 0x7fc1, 0xffe5],
        "f32": [0, 0x80000000, 1, 0x80000001, 0x7f7fffff, 0xff7fffff,
                0x7f800000, 0xff800000, 0x7fc12345, 0xffc12345],
    }
    bits = np.uint32 if storage == "f32" else np.uint16
    return np.resize(np.array(patterns[storage], dtype=bits), shape).view(dtype)


@pytest.mark.parametrize("storage", ["f16", "bf16", "f32"])
@pytest.mark.parametrize("shape", [(7, 32), (129, 33)])
def test_ordinary_relu_jit_and_serialized_package_are_bitwise(storage, shape, monkeypatch):
    assert rt._nvidia_device_name() == "sm_120"
    inputs = values(storage, shape)
    bits = np.uint32 if storage == "f32" else np.uint16
    fn = ts.jit(target="nvidia_sm120")(relu)
    try:
        actual = fn(inputs)
        expected = np.maximum(0, inputs)
        np.testing.assert_array_equal(actual.view(bits), expected.view(bits))
        assert fn.execution_kind == "native_gpu"
        artifact = rt.RuntimeArtifact.from_json(fn.runtime_artifact().to_json())
        d = artifact.launch_descriptor
        assert d.abi_id == f"tessera.nvidia.relu.x_o_rows_columns.{storage}.v1"
        assert "schedule.activation" in artifact.schedule_ir
        assert "tile.activation_kernel" in artifact.tile_ir
        assert d.provenance["route"] == "canonical_scheduled_tile_consumer"
        saved = actual.copy()
        changed = inputs.copy()
        changed.view(bits)[:] ^= np.array(1 << (changed.dtype.itemsize * 8 - 1), dtype=bits)
        snapshot = changed.copy()

        def forbidden(*_, **__):
            pytest.fail("warm or serialized ReLU invoked external compiler tools")

        with monkeypatch.context() as guarded:
            guarded.setattr(subprocess, "run", forbidden)
            guarded.setattr(subprocess, "Popen", forbidden)
            output = fn(changed)
            np.testing.assert_array_equal(output.view(bits), np.maximum(0, changed).view(bits))
            np.testing.assert_array_equal(actual.view(bits), saved.view(bits))
            assert not np.shares_memory(actual, output)
            canary = np.full(changed.size + 16, -123, dtype=changed.dtype)
            view = canary[:changed.size].reshape(shape)
            result = rt.launch(artifact, {
                "buffers": {d.buffers[0].name: changed, d.buffers[1].name: view},
                "scalars": {"Rows": shape[0], "Columns": shape[1]},
            })
            assert result["ok"] and result["execution_kind"] == "native_gpu", result
            np.testing.assert_array_equal(view.view(bits), np.maximum(0, changed).view(bits))
            assert np.all(canary[changed.size:] == -123)
            np.testing.assert_array_equal(changed.view(bits), snapshot.view(bits))
    finally:
        fn.close_native_storage()


@pytest.mark.parametrize("schedule", ["serial", "flat_128"])
def test_standalone_f32_resident_package(schedule):
    from tessera.compiler.emit.nvidia_cuda import NvidiaDeviceSession
    from tessera.compiler import nvidia_native, scheduled_kernel
    from tests.unit.test_sm120_public_relu import module

    assert rt._nvidia_device_name() == "sm_120"
    artifact = scheduled_kernel.lower_scheduled_kernel(module(), target="nvidia_sm120", schedule=schedule)
    package = nvidia_native.package_scheduled_kernel(artifact, pipeline_name="tessera-nvidia-pipeline-sm120")
    runtime = rt.RuntimeArtifact(graph_ir=artifact.graph_ir, schedule_ir=artifact.schedule_ir,
        tile_ir=package.tile_ir, target_ir=package.target_ir,
        native_image=package.image, launch_descriptor=package.descriptor)
    source = values("f32", (7, 32))
    with NvidiaDeviceSession() as session:
        x = session.upload(source)
        y = session.empty(source.shape, source.dtype)
        result = rt.launch(runtime, {
            artifact.input_name: x, artifact.output_name: y, "Rows": 7, "Columns": 32,
        }, stream=session.stream)
        assert result["ok"] and result["execution_kind"] == "native_gpu", result
        np.testing.assert_array_equal(session.download(y).view(np.uint32),
                                      np.maximum(0, source).view(np.uint32))


def test_standalone_f32_package_replays_in_fresh_process_without_compiler():
    import json
    import os
    import sys

    assert rt._nvidia_device_name() == "sm_120"
    source = values("f32", (7, 32))
    fn = ts.jit(target="nvidia_sm120")(relu)
    try:
        expected = fn(source)
        payload = {"artifact": fn.runtime_artifact().to_json(),
                   "bits": source.view(np.uint32).tolist(),
                   "expected": expected.view(np.uint32).tolist()}
    finally:
        fn.close_native_storage()
    code = """
import json,sys,subprocess
import numpy as np
from tessera import runtime as rt
from tessera.compiler import scheduled_matmul,scheduled_kernel,nvidia_native
def forbidden(*_,**__):
    raise AssertionError("fresh native package replay invoked compiler or subprocess")
scheduled_matmul.find_tessera_opt=forbidden
scheduled_kernel.find_tessera_opt=forbidden
subprocess.run=forbidden
subprocess.Popen=forbidden
payload=json.load(sys.stdin)
artifact=rt.RuntimeArtifact.from_json(payload["artifact"])
source=np.array(payload["bits"],dtype=np.uint32).view(np.float32)
output=np.empty_like(source)
d=artifact.launch_descriptor
result=rt.launch(artifact,{"buffers":{d.buffers[0].name:source,d.buffers[1].name:output},
    "scalars":{"Rows":7,"Columns":32}})
assert result["ok"] and result["execution_kind"]=="native_gpu",result
np.testing.assert_array_equal(output.view(np.uint32),np.array(payload["expected"],dtype=np.uint32))
print("fresh compiler-free standalone FP32 ReLU passed")
"""
    env = {key: value for key, value in os.environ.items()
           if key not in {"TESSERA_OPT", "TESSERA_NVIDIA_OPT"}}
    result = subprocess.run([sys.executable, "-c", code], input=json.dumps(payload),
                            text=True, capture_output=True, env=env, check=False)
    assert result.returncode == 0, result.stdout + result.stderr
    assert "fresh compiler-free" in result.stdout
