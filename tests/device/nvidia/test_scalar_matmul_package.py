"""Native Graph/Schedule package and checked host/resident scalar ABI proof."""
from functools import lru_cache
import subprocess

import numpy as np
import pytest

from tessera import runtime as rt
from tessera.compiler import nvidia_native, scheduled_matmul
from tessera.compiler.emit.nvidia_cuda import NvidiaDeviceSession
from tessera.compiler.graph_ir import IRType
from tests.unit.test_scheduled_matmul_consumers import _module

pytestmark = pytest.mark.hardware_nvidia


@lru_cache(maxsize=64)
def packaged(half, side, output, ta, tb):
    a_dtype, b_dtype = ("fp32", half) if side == "a" else (half, "fp32")
    module = _module(target="nvidia_sm120", shape=(17, 35, 19),
                     dtype=a_dtype, b_dtype=b_dtype, output_dtype=output)
    function = module.functions[0]
    for arg, transpose in zip(function.args, (ta, tb), strict=True):
        if transpose:
            old = arg.ir_type
            shape = old.shape[::-1]
            element = {"fp32": "f32", "fp16": "f16", "bf16": "bf16"}[old.dtype]
            arg.ir_type = IRType(f"tensor<{shape[0]}x{shape[1]}x{element}>", shape, old.dtype)
    function.body[0].operand_types = [str(arg.ir_type) for arg in function.args]
    function.body[0].kwargs.update(transposeA=ta, transposeB=tb)
    before = module.to_mlir(canonical=True)
    scheduled = scheduled_matmul.lower_scheduled_matmul(module, target="nvidia_sm120")
    assert module.to_mlir(canonical=True) == before
    package = nvidia_native.package_scheduled_matmul(scheduled, pipeline_name="tessera-lower-to-nvidia-sm120")
    artifact = rt.RuntimeArtifact(
        graph_ir=scheduled.graph_ir, schedule_ir=scheduled.schedule_ir,
        tile_ir=package.tile_ir, target_ir=package.target_ir,
        native_image=package.image, launch_descriptor=package.descriptor)
    # Persist the complete ABI and image, then exercise that restored object.
    return rt.RuntimeArtifact.from_json(artifact.to_json())


@pytest.mark.parametrize("half", ["fp16", "bf16"])
@pytest.mark.parametrize("side", ["a", "b"])
@pytest.mark.parametrize("output", ["fp32", "fp16", "bf16"])
@pytest.mark.parametrize("orientation", [(False, False), (False, True), (True, False), (True, True)])
@pytest.mark.parametrize("resident", [False, True])
def test_checked_scalar_package_replays(half, side, output, orientation, resident, monkeypatch):
    import ml_dtypes
    assert rt._nvidia_device_name() == "sm_120"
    dtype = {"fp32": np.dtype("float32"), "fp16": np.dtype("float16"),
             "bf16": np.dtype(ml_dtypes.bfloat16)}
    ta, tb = orientation
    artifact = packaged(half, side, output, ta, tb)
    descriptor = artifact.launch_descriptor
    a_dtype, b_dtype = ("fp32", half) if side == "a" else (half, "fp32")
    for seed in (11, 17):
        rng = np.random.default_rng(seed)
        a = rng.normal(0, .25, (17, 35)).astype(dtype[a_dtype])
        b = rng.normal(0, .25, (35, 19)).astype(dtype[b_dtype])
        expected = (a.astype(np.float64) @ b.astype(np.float64)).astype(dtype[output])
        magnitude = np.abs(a.astype(np.float64)) @ np.abs(b.astype(np.float64))
        gamma = 35 * np.finfo(np.float32).eps / (1 - 35 * np.finfo(np.float32).eps)
        upper = np.nextafter(expected, dtype[output].type(np.inf)).astype(np.float64)
        budget = gamma * magnitude + np.abs(upper - expected.astype(np.float64)) + 1e-8
        host = (np.ascontiguousarray(a.T if ta else a),
                np.ascontiguousarray(b.T if tb else b),
                np.empty((17, 19), dtype=dtype[output]))
        with NvidiaDeviceSession() as session:
            values = tuple(session.upload(value) for value in host) if resident else host
            arguments = {"buffers": {binding.name: value for binding, value in
                                      zip(descriptor.buffers, values, strict=True)},
                         "scalars": {"M": 17, "N": 19, "K": 35}}
            with monkeypatch.context() as replay:
                replay.setattr(subprocess, "run", lambda *a, **k: pytest.fail("replay compiled"))
                result = rt.launch(artifact, arguments, stream=session.stream if resident else None)
            assert result["ok"] and result["execution_kind"] == "native_gpu", result
            actual = session.download(values[-1]) if resident else values[-1]
            assert actual.dtype == dtype[output] and np.isfinite(actual).all()
            assert np.all(np.abs(actual.astype(np.float64) - expected.astype(np.float64)) <= budget)


@pytest.mark.parametrize("half", ["fp16", "bf16"])
def test_scalar_owner_orders_pending_roots_without_allocation_fence(half):
    from contextlib import ExitStack
    import ml_dtypes
    from tessera.compiler.prepared_nvidia_scalar_matmul import PreparedScalarMatmulCall
    from tests.device.nvidia.test_ordered_resident_tensor_dag import queue_delayed_upload
    dtype = np.float16 if half == "fp16" else ml_dtypes.bfloat16
    artifact = packaged(half, "a", "fp32", False, True)
    owner = PreparedScalarMatmulCall(artifact.native_image, artifact.launch_descriptor)
    try:
        with ExitStack() as stack:
            sessions = [stack.enter_context(NvidiaDeviceSession()) for _ in range(3)]
            changed = (np.full((17, 35), .75, np.float32),
                       np.full((19, 35), .125, dtype))
            values = [sessions[i].upload(np.zeros_like(value)) for i, value in enumerate(changed)]
            values.append(sessions[2].empty((17, 19), np.float32))
            for session in sessions:
                assert session.synchronize() == 0
            owner.invoke(values)
            completed = []
            callbacks = [queue_delayed_upload(sessions[i], values[i], changed[i],
                                              sessions[i].stream, completed) for i in range(2)]
            assert not completed
            # Metadata projection and the native invocation allocate no GPU storage.
            # Observe completion before a download/free can accidentally order writes.
            owner.invoke(values)
            assert len(completed) == len(callbacks) == 2
            actual = sessions[2].download(values[2])
            expected = changed[0].astype(np.float64) @ changed[1].astype(np.float64).T
            np.testing.assert_array_equal(actual, expected)
    finally:
        owner.close()


@pytest.mark.parametrize("fault,reason", [
    ("dtype", "shape/stride/storage mismatch"),
    ("capacity", "allocation context/capacity mismatch"),
    ("alias", "output aliases an input"),
])
def test_native_scalar_owner_rejects_forged_views_and_recovers(fault, reason):
    import ctypes as ct
    from tessera.compiler.prepared_nvidia_scalar_matmul import PreparedScalarMatmulCall
    from tessera.compiler.resident_nvidia_tensor import ordered_resident_views
    artifact = packaged("fp16", "a", "fp32", False, True)
    owner = PreparedScalarMatmulCall(artifact.native_image, artifact.launch_descriptor)
    try:
        with NvidiaDeviceSession() as session:
            a = session.upload(np.full((17, 35), .25, np.float32))
            b = session.upload(np.full((19, 35), .125, np.float16))
            d = session.upload(np.full((17, 19), 123, np.float32))
            small = session.empty((1, 1), np.float32)
            assert session.synchronize() == 0
            views, _ = ordered_resident_views((a, b, d), session.stream, writable_from=2)
            if fault == "dtype":
                views[0].dtype = 2
            elif fault == "capacity":
                views[0].data = small.ptr
            else:
                views[2].data = a.ptr
            streams = (ct.c_uint64 * 3)(session.stream, session.stream, session.stream)
            status = owner.lib.tessera_nvidia_scalar_matmul_invoke_resident_v1(
                owner.handle, views, 3, streams, 3)
            assert status != 0
            assert reason in owner.lib.tessera_nvidia_matmul_last_error().decode()
            np.testing.assert_array_equal(session.download(a), .25)
            np.testing.assert_array_equal(session.download(d), 123)
            owner.invoke((a, b, d))
            np.testing.assert_array_equal(session.download(d), np.full((17, 19), 35 * .25 * .125))
    finally:
        owner.close()


@pytest.mark.parametrize("fault", ["endian", "readonly"])
def test_scalar_host_storage_contract_refuses_and_recovers(fault):
    from tessera.compiler.prepared_nvidia_scalar_matmul import PreparedScalarMatmulCall
    artifact = packaged("fp16", "a", "fp32", False, True)
    owner = PreparedScalarMatmulCall(artifact.native_image, artifact.launch_descriptor)
    try:
        a = np.full((17, 35), .25, np.float32)
        b = np.full((19, 35), .125, np.float16)
        out = np.full((17, 19), 123, np.float32)
        bad_a = a.astype(">f4") if fault == "endian" else a
        if fault == "readonly":
            out.flags.writeable = False
        with pytest.raises(ValueError, match="native-endian|writable storage"):
            owner.invoke((bad_a, b, out))
        np.testing.assert_array_equal(out, 123)
        out.flags.writeable = True
        owner.invoke((a, b, out))
        np.testing.assert_array_equal(out, np.full((17, 19), 35 * .25 * .125))
    finally:
        owner.close()


@pytest.mark.parametrize("half", ["fp16", "bf16"])
@pytest.mark.parametrize("output", ["fp32", "fp16", "bf16"])
@pytest.mark.parametrize("captured", [False, True])
def test_scalar_native_profile_uses_declared_output_storage(half, output, captured):
    import ml_dtypes
    from tessera.compiler.prepared_nvidia_scalar_matmul import PreparedScalarMatmulCall
    dtype = {"fp32": np.float32, "fp16": np.float16, "bf16": ml_dtypes.bfloat16}
    artifact = packaged(half, "a", output, False, True)
    owner = PreparedScalarMatmulCall(artifact.native_image, artifact.launch_descriptor)
    try:
        values = (np.full((17, 35), .25, np.float32),
                  np.full((19, 35), .125, dtype[half]),
                  np.empty((17, 19), dtype[output]))
        actual, timing = owner.profile_host(values, repeats=16, captured=captured)
        assert actual.dtype == np.dtype(dtype[output])
        np.testing.assert_array_equal(actual, np.full((17, 19), 35 * .25 * .125))
        assert timing["captured"] is captured
        assert np.isfinite(timing["kernel_stage_ms"]) and timing["kernel_stage_ms"] > 0
        assert np.isfinite(timing["program_ms"]) and timing["program_ms"] > 0
    finally:
        owner.close()
