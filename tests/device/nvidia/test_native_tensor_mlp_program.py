"""Exact SM120 proof for native three-root, two-matmul producer programs."""
import subprocess
import numpy as np
import pytest
import tessera as ts
from tessera import runtime as rt
from tests.unit.test_native_tensor_mlp_program import (
    mlp, reordered_mlp, default_mlp, reordered_default_mlp, mlp_roots, assert_mlp_numeric)

pytestmark = pytest.mark.hardware_nvidia


def reference(args, reordered, intermediate):
    down, up, x = args if reordered else (args[2], args[1], args[0])
    hidden = (x.astype(np.float64) @ up.astype(np.float64)).astype(
        np.float16 if intermediate == "f16" else np.float32)
    return np.maximum(0, hidden).astype(np.float64) @ down.astype(np.float64)


@pytest.mark.parametrize("intermediate", ["f16", "f32"])
@pytest.mark.parametrize("storage", ["f16", "bf16"])
@pytest.mark.parametrize("reordered", [False, True])
@pytest.mark.parametrize("shape", [(17, 32, 48, 11), (17, 35, 29, 19), (64, 256, 128, 64)])
def test_public_mlp_native_source_warm_reuse_and_rebind(storage, reordered, shape, intermediate, monkeypatch):
    assert rt._nvidia_device_name() == "sm_120"
    function = ((reordered_default_mlp if reordered else default_mlp)
                if intermediate == "f32" else (reordered_mlp if reordered else mlp))
    fn = ts.jit(target="nvidia_sm120")(function)
    args = mlp_roots(storage, reordered=reordered, shape=shape, intermediate=intermediate)
    try:
        first = fn(*args)
        assert_mlp_numeric(first, args, reordered, intermediate)
        assert fn.execution_kind == "native_gpu"
        program = fn._native_tensor_last_program
        assert program is not None and len(program.members) == 3
        assert program.frame["roots"] == [0, 1, 2]
        assert program.tile_ir.count("tile.program_invoke ") == 3
        # A/B each have a view in both matmuls; output storage is materialized
        # by fragment unpack/store rather than a third input-style view.
        tensorcore_stages = 2 if intermediate == "f16" else 1
        assert program.tile_ir.count("tile.view ") >= tensorcore_stages * 2
        assert program.tile_ir.count("tile.fragment_pack ") >= tensorcore_stages * 2
        assert program.tile_ir.count("tile.fragment_unpack ") >= tensorcore_stages
        assert program.tile_ir.count("tile.store ") >= tensorcore_stages
        assert [spec["storage"] for spec in program.frame["buffers"][3:]] == [
            intermediate, intermediate, "f32"]
        if intermediate == "f32":
            assert "tile.scalar_matmul_kernel" in program.tile_ir
        owner = fn._native_tensor_last_owner
        before = owner.stats()
        saved = first.copy()
        def forbidden(*args, **kwargs):
            pytest.fail("native MLP warm call or rebind invoked eager code, tracing or tools")
        with monkeypatch.context() as guarded:
            guarded.setattr(fn, "_fn", forbidden)
            guarded.setattr(fn, "_trace_frontend_capture", forbidden)
            guarded.setattr(subprocess, "run", forbidden)
            guarded.setattr(subprocess, "Popen", forbidden)
            for seed in (120322, 120323):
                inputs = mlp_roots(storage, reordered=reordered, shape=shape, seed=seed, intermediate=intermediate)
                snapshots = tuple(value.copy() for value in inputs)
                actual = fn(*inputs)
                assert_mlp_numeric(actual, inputs, reordered, intermediate)
                assert fn._native_tensor_last_owner is owner
                assert fn._native_tensor_last_program is program
                for value, snapshot in zip(inputs, snapshots, strict=True):
                    np.testing.assert_array_equal(value, snapshot)
                np.testing.assert_array_equal(first, saved)
                assert not np.shares_memory(first, actual)
            after = owner.stats()
            assert after["capacity"] == before["capacity"]
            assert after["allocations"] == before["allocations"] == 1
            fn.close_native_storage()
            assert owner.handle == 0
            actual = fn(*inputs)
            assert_mlp_numeric(actual, inputs, reordered, intermediate)
            assert fn._native_tensor_last_owner is not owner
    finally:
        fn.close_native_storage()


@pytest.mark.parametrize("intermediate", ["f16", "f32"])
@pytest.mark.parametrize("storage", ["f16", "bf16"])
@pytest.mark.parametrize("reordered", [False, True])
def test_mlp_runtime_artifact_replays_three_named_roots(storage, reordered, intermediate):
    assert rt._nvidia_device_name() == "sm_120"
    function = ((reordered_default_mlp if reordered else default_mlp)
                if intermediate == "f32" else (reordered_mlp if reordered else mlp))
    fn = ts.jit(target="nvidia_sm120")(function)
    inputs = mlp_roots(storage, reordered=reordered, intermediate=intermediate)
    try:
        first = fn(*inputs)
        artifact = rt.RuntimeArtifact.from_json(fn.runtime_artifact().to_json())
        assert len(artifact.metadata["arg_names"]) == 3
        args = dict(zip(artifact.metadata["arg_names"], inputs, strict=True))
        receipt = rt.launch(artifact, args)
        assert receipt["ok"] and receipt["execution_kind"] == "native_gpu", receipt
        assert receipt["compiler_path"] == "canonical_native_tensor_program"
        np.testing.assert_array_equal(receipt["output"], first)
        assert_mlp_numeric(receipt["output"], inputs, reordered, intermediate)
    finally:
        fn.close_native_storage()



@pytest.mark.parametrize("schedule", ["serial", "flat_128"])
def test_f32_relu_leaf_special_values_on_sm120(schedule):
    """Diagnostic leaf proof; public composition is proved independently above."""
    import ctypes as ct
    import os
    from tessera.compiler.nvidia_native import _compile_tile_ir
    from tessera.compiler.prepared_native_tensor_frame import PreparedNativeTensorFrame
    from tests.unit.test_native_tensor_mlp_program import f32_relu_tile

    assert rt._nvidia_device_name() == "sm_120"
    _, ptx, *_ = _compile_tile_ir(f32_relu_tile(schedule), "f32_relu")
    bits = np.array([0x80000000, 0, 0xbf800000, 0x7fc12345,
                     0x7f800000, 0xff800000, 1, 0x80000001], dtype=np.uint32)
    values = bits.view(np.float32).reshape(1, 8)
    expected = ts.ops.relu(values).view(np.uint32)
    frame = {
        "schema": "tessera.native.tensor_program_frame.v1",
        "target": "nvidia_sm120", "architecture": "sm_120", "function": "f32_relu",
        "buffers": [
            {"id": i, "role": role, "shape": [1, 8], "storage": "f32", "bytes": 32}
            for i, role in enumerate(("input", "output"))],
        "roots": [0], "output": 1,
        "stages": [{"member": "diagnostic_relu", "entry": "f32_relu", "inputs": [0],
                    "output": 1, "scalars": [1, 8], "geometry": [1, 1, 1, 128, 1, 1]}],
    }
    provider = ct.CDLL(os.environ["TESSERA_NATIVE_FRAME_TEST_LIB"])
    with PreparedNativeTensorFrame(frame, (ptx.encode(),), provider=provider) as owner:
        actual = owner((values,))
        np.testing.assert_array_equal(actual.view(np.uint32), expected)
        np.testing.assert_array_equal(values.view(np.uint32), bits.reshape(1, 8))


@pytest.mark.parametrize("storage", ["f16", "bf16"])
@pytest.mark.parametrize("intermediate", ["f16", "f32"])
@pytest.mark.parametrize("reordered", [False, True])
def test_three_resident_mlp_roots_order_independent_producer_streams(
    storage, intermediate, reordered, monkeypatch,
):
    from contextlib import ExitStack
    from tessera.compiler.emit.nvidia_cuda import NvidiaDeviceSession
    from tests.device.nvidia.test_ordered_resident_tensor_dag import Borrowed, queue_delayed_upload

    assert rt._nvidia_device_name() == "sm_120"
    function = ((reordered_default_mlp if reordered else default_mlp)
                if intermediate == "f32" else (reordered_mlp if reordered else mlp))
    fn = ts.jit(target="nvidia_sm120")(function)
    args = mlp_roots(storage, reordered=reordered, intermediate=intermediate)
    with ExitStack() as stack:
        sessions = [stack.enter_context(NvidiaDeviceSession()) for _ in args]
        buffers = [session.upload(np.zeros_like(value))
                   for session, value in zip(sessions, args, strict=True)]
        assert all(session.synchronize() == 0 for session in sessions)
        resident = [Borrowed(buffer, session.stream)
                    for buffer, session in zip(buffers, sessions, strict=True)]
        try:
            fn(*resident)
            owner = fn._native_tensor_last_owner
            assert owner is not None
            before = owner.stats()
            done = [[], [], []]
            callbacks = [
                queue_delayed_upload(session, buffer, value, session.stream, completed)
                for session, buffer, value, completed in zip(
                    sessions, buffers, args, done, strict=True)]
            assert not any(done)

            def forbidden(*_, **__):
                pytest.fail("resident MLP replay used eager code, host coercion, tracing or tools")

            with monkeypatch.context() as guarded:
                guarded.setattr(fn, "_fn", forbidden)
                guarded.setattr(fn, "_trace_frontend_capture", forbidden)
                guarded.setattr(subprocess, "run", forbidden)
                guarded.setattr(subprocess, "Popen", forbidden)
                actual = fn(*resident)
            assert all(done) and len(callbacks) == 3
            assert fn._native_tensor_last_owner is owner
            assert fn.execution_kind == "native_gpu"
            assert owner.stats()["allocations"] == before["allocations"] == 1
            assert owner.stats()["capacity"] == before["capacity"]
            assert_mlp_numeric(actual, args, reordered, intermediate)
            for session, buffer, expected in zip(sessions, buffers, args, strict=True):
                np.testing.assert_array_equal(session.download(buffer), expected)
            saved = actual.copy()
        finally:
            fn.close_native_storage()
    np.testing.assert_array_equal(actual, saved)
