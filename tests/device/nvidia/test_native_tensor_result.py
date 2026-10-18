"""Exact sm_120 public and replay ownership for paired native device results."""
import gc
from contextlib import ExitStack

import numpy as np
import pytest
import tessera as ts
from tessera import runtime as rt
from tessera.compiler.emit.nvidia_cuda import NvidiaDeviceSession
from tessera.compiler.native_tensor_result import NativeTensorResult
from benchmarks.nvidia.active_source_jvp_reference import inputs, oracle, product, product_reversed
from tests.device.nvidia.test_resident_active_source_jvp import check
from tests.device.nvidia.test_ordered_resident_tensor_dag import Borrowed

pytestmark = pytest.mark.hardware_nvidia


@pytest.mark.parametrize("storage", ["f16", "bf16"])
@pytest.mark.parametrize("shape", [(1, 1, 1), (17, 19, 35)])
@pytest.mark.parametrize("active", ["a", "b"])
@pytest.mark.parametrize("permuted", [False, True])
@pytest.mark.parametrize("resident", [False, True])
def test_public_native_results_own_generations_and_replay(
    storage, shape, active, permuted, resident, monkeypatch,
):
    assert rt._nvidia_device_name() == "sm_120"
    a, b, da = inputs({"shape_mnk": shape, "storage": storage})
    tangent = da if active == "a" else (b.astype(np.float32) * .5 + .125).astype(b.dtype)
    roots = (b, a) if permuted else (a, b)
    if active == "a":
        expected, budget = oracle(a, b, tangent)
    else:
        primal, pb = oracle(a, b, np.zeros_like(a))
        derivative, db = oracle(a, tangent, np.zeros_like(a))
        expected, budget = (primal[0], derivative[0]), (pb[0], db[0])
    fn = ts.jit(target="nvidia_sm120", autodiff="forward", wrt=(active,))(
        product_reversed if permuted else product)
    check(fn.native_jvp(*roots, tangents=(tangent,)), expected, budget)
    with ExitStack() as stack:
        if resident:
            sessions = [stack.enter_context(NvidiaDeviceSession()) for _ in range(3)]
            values = tuple(Borrowed(session.upload(value)) for session, value in
                           zip(sessions, (*roots, tangent), strict=True))
        else:
            values = (*roots, tangent)
        pair = fn.native_jvp(*values[:2], tangents=(values[2],), result_placement="cuda")
        assert all(isinstance(value, NativeTensorResult) for value in pair)
        assert all(value.__cuda_array_interface__["data"][1] for value in pair)
        check(tuple(value.to_host() for value in pair), expected, budget)
        first = tuple(value.to_host() for value in pair)
        artifact = rt.RuntimeArtifact.from_json(fn.native_jvp_runtime_artifact().to_json())
        assert artifact.metadata["result_placement"] == "cuda"
        assert fn.last_jvp_execution["artifact_hash"] == artifact.artifact_hash
        inspection = fn.native_jvp_runtime_artifact()
        inspection.metadata["result_placement"] = "host"
        assert fn.native_jvp_runtime_artifact().metadata["result_placement"] == "cuda"
        other = fn.native_jvp(*values[:2], tangents=(values[2],), result_placement="cuda")
        assert pair[0].__cuda_array_interface__["data"][0] != other[0].__cuda_array_interface__["data"][0]
        fn.close_native_storage()
        gc.collect()
        for value, saved in zip(pair, first, strict=True):
            np.testing.assert_array_equal(value.to_host(), saved)
        import subprocess
        with monkeypatch.context() as replay:
            replay.setattr(subprocess, "run", lambda *a, **k: pytest.fail("replay invoked compiler"))
            result = rt.launch(artifact, values)
        assert result["ok"], result
        replayed = result["output"]
        check(tuple(value.to_host() for value in replayed), expected, budget)
        for value in (*other, *replayed):
            value.close()
        pair[0].close()
        with pytest.raises(ValueError, match="closed"):
            pair[0].to_host()
        np.testing.assert_array_equal(pair[1].to_host(), first[1])
        with pytest.raises(TypeError, match="explicit"):
            np.asarray(pair[1])
        pair[1].close()
        pair[1].close()


@pytest.mark.parametrize("storage", ["f16", "bf16"])
@pytest.mark.parametrize("strategy", ["serial", "cooperative_128"])
def test_native_results_feed_compiled_fp32_consumer_after_owner_close(storage, strategy, monkeypatch):
    from tessera.compiler import nvidia_native, scheduled_kernel
    import subprocess
    a, b, da = inputs({"shape_mnk": (17, 19, 35), "storage": storage})
    fn = ts.jit(target="nvidia_sm120", autodiff="forward", wrt=("a",))(product)
    pair = fn.native_jvp(a, b, tangents=(da,), result_placement="cuda")
    saved = tuple(value.to_host() for value in pair)
    fn.close_native_storage()
    frontend = ts.from_text(
        "def consumer(x):\n    return tessera.ops.rmsnorm(x, eps=1e-5)\n",
        target="nvidia_sm120")
    graph, _ = frontend._trace_frontend_capture((saved[0],), {})
    scheduled = scheduled_kernel.lower_scheduled_kernel(
        graph, target="nvidia_sm120", schedule=strategy)
    native = nvidia_native.package_scheduled_kernel(
        scheduled, pipeline_name="tessera-nvidia-pipeline-sm120")
    assert "schedule.norm" in scheduled.schedule_ir
    assert "tile.norm_kernel" in scheduled.tile_ir
    assert "nvvm." in native.target_ir
    artifact = rt.RuntimeArtifact(
        graph_ir=scheduled.graph_ir, schedule_ir=scheduled.schedule_ir,
        tile_ir=native.tile_ir, target_ir=native.target_ir,
        metadata={"target": "nvidia_sm120"}, native_image=native.image,
        launch_descriptor=native.descriptor)
    artifact = rt.RuntimeArtifact.from_json(artifact.to_json())
    d = native.descriptor
    with NvidiaDeviceSession() as session:
        for value, source in zip(pair, saved, strict=True):
            output = session.empty(source.shape, np.float32)
            # Allocation alone writes no output. Its producer metadata names the
            # stream on which the compiled consumer will initialize every element.
            stream = value.__cuda_array_interface__["stream"]
            class Destination(Borrowed):
                @property
                def shape(self):
                    return self.buffer.shape

                @property
                def strides(self):
                    return self.buffer.__cuda_array_interface__["strides"]

                @property
                def nbytes(self):
                    return self.buffer.nbytes

                @property
                def __cuda_array_interface__(self):
                    interface = super().__cuda_array_interface__
                    interface["data"] = (interface["data"][0], False)
                    return interface
            destination = Destination(output, stream)
            args = {d.buffers[0].name: value, d.buffers[1].name: destination,
                    "Rows": source.shape[0], "Columns": source.shape[1]}
            with monkeypatch.context() as replay:
                replay.setattr(subprocess, "run", lambda *a, **k: pytest.fail("consumer replay compiled"))
                receipt = rt.launch(artifact, args, stream=stream)
            assert receipt["ok"] and receipt["execution_kind"] == "native_gpu", repr(receipt)
            x = source.astype(np.float64)
            expected = x / np.sqrt(np.mean(x * x, axis=-1, keepdims=True) + 1e-5)
            # The destination was written on the result's stream, not the
            # allocation session's stream used by numpy(). Complete that
            # writer explicitly before initiating the host copy.
            import ctypes as ct
            driver = ct.CDLL("libcuda.so.1")
            driver.cuStreamSynchronize.argtypes = [ct.c_void_p]
            driver.cuStreamSynchronize.restype = ct.c_int
            assert driver.cuStreamSynchronize(ct.c_void_p(stream)) == 0
            actual = output.numpy()
            np.testing.assert_allclose(actual, expected, rtol=3e-6, atol=3e-6)
            # Closing a consumed slot retains the other independently owned slot.
            value.close()
            refused = rt.launch(artifact, args, stream=stream)
            assert not refused["ok"] and refused["runtime_status"] == "invalid_artifact"
            assert refused["diagnostic_code"] == "E_LAUNCH_BINDING_MISMATCH"
            assert refused["artifact_hash"] == artifact.artifact_hash
    pair[0].close()
    pair[1].close()


def _result_pair():
    a, b, da = inputs({"shape_mnk": (17, 19, 35), "storage": "f16"})
    fn = ts.jit(target="nvidia_sm120", autodiff="forward", wrt=("a",))(product)
    return fn, fn.native_jvp(a, b, tangents=(da,), result_placement="cuda")


def test_result_context_refusal_restoration_and_foreign_context_cleanup():
    import ctypes as ct
    driver = ct.CDLL("libcuda.so.1")
    for symbol, signature in {
        "cuCtxCreate_v2": [ct.POINTER(ct.c_void_p), ct.c_uint, ct.c_int],
        "cuCtxGetCurrent": [ct.POINTER(ct.c_void_p)],
        "cuCtxSetCurrent": [ct.c_void_p],
        "cuCtxDestroy_v2": [ct.c_void_p],
    }.items():
        getattr(driver, symbol).argtypes = signature
        getattr(driver, symbol).restype = ct.c_int
    fn, pair = _result_pair()
    saved = pair[0].to_host()
    batch = pair[0]._batch
    original, foreign = ct.c_void_p(), ct.c_void_p()
    assert driver.cuCtxGetCurrent(ct.byref(original)) == 0
    assert driver.cuCtxSetCurrent(None) == 0
    np.testing.assert_array_equal(pair[0].to_host(), saved)
    current = ct.c_void_p()
    assert driver.cuCtxGetCurrent(ct.byref(current)) == 0 and current.value == original.value
    fn.close_native_storage()
    assert driver.cuCtxCreate_v2(ct.byref(foreign), 0, 0) == 0
    try:
        with pytest.raises(RuntimeError, match="context changed"):
            pair[0].to_host()
        with pytest.raises(RuntimeError, match="context changed"):
            _ = pair[0].__cuda_array_interface__
        # GC/final release restores its owning context internally and must
        # preserve the unrelated current context of the caller.
        pair[0].close()
        pair[1].close()
        del batch
        gc.collect()
        assert driver.cuCtxGetCurrent(ct.byref(current)) == 0
        assert current.value == foreign.value
    finally:
        assert driver.cuCtxSetCurrent(original) == 0
        assert driver.cuCtxDestroy_v2(foreign) == 0


def test_result_fork_refuses_before_touching_inherited_locks():
    import os
    import select
    fn, pair = _result_pair()
    read_fd, write_fd = os.pipe()
    # A child inherits this held lock with an owner that cannot execute there.
    # Every public operation must reject the PID before acquiring that lock.
    batch = pair[0]._batch
    batch.lock.acquire()
    child = os.fork()
    if child == 0:
        try:
            os.close(read_fd)
            checked = 0
            for action in (pair[0].to_host, pair[0].close,
                           lambda: pair[0].__cuda_array_interface__,
                           lambda: pair[0].record_consumer(1)):
                try:
                    action()
                except ValueError as exc:
                    if "fork" not in str(exc):
                        os._exit(2)
                    checked += 1
                else:
                    os._exit(3)
            os.write(write_fd, str(checked).encode())
            os._exit(0)
        except BaseException:
            os._exit(4)
    os.close(write_fd)
    try:
        ready, _, _ = select.select([read_fd], [], [], 10)
        if not ready:
            os.kill(child, 9)
        assert ready, "forked result operation blocked on an inherited lock"
        assert os.read(read_fd, 16) == b"4"
        _, status = os.waitpid(child, 0)
        assert os.waitstatus_to_exitcode(status) == 0
    finally:
        batch.lock.release()
        os.close(read_fd)
        fn.close_native_storage()
        for value in pair:
            value.close()


@pytest.mark.parametrize("storage", ["f16", "bf16"])
def test_registered_async_consumer_completes_before_last_result_release(storage):
    import ctypes as ct
    import time
    from tessera.compiler.prepared_nvidia_matmul import HostView
    a, b, da = inputs({"shape_mnk": (17, 19, 35), "storage": storage})
    fn = ts.jit(target="nvidia_sm120", autodiff="forward", wrt=("a",))(product)
    pair = fn.native_jvp(a, b, tangents=(da,), result_placement="cuda")
    saved = pair[0].to_host()
    interface = pair[0].__cuda_array_interface__
    lib, handle = pair[0]._batch.lib, pair[0]._batch.handle
    fn.close_native_storage()
    driver = ct.CDLL("libcuda.so.1")
    driver.cuLaunchHostFunc.argtypes = [ct.c_void_p, ct.c_void_p, ct.c_void_p]
    driver.cuLaunchHostFunc.restype = ct.c_int
    driver.cuMemcpyDtoDAsync_v2.argtypes = [ct.c_uint64, ct.c_uint64, ct.c_size_t, ct.c_void_p]
    driver.cuMemcpyDtoDAsync_v2.restype = ct.c_int
    completed = []
    @ct.CFUNCTYPE(None, ct.c_void_p)
    def delay(_):
        time.sleep(.1)
        completed.append(True)
    with NvidiaDeviceSession() as consumer:
        output = consumer.empty(saved.shape, saved.dtype)
        assert driver.cuLaunchHostFunc(ct.c_void_p(consumer.stream), ct.cast(delay, ct.c_void_p), None) == 0
        assert driver.cuMemcpyDtoDAsync_v2(
            output.ptr, interface["data"][0], saved.nbytes, ct.c_void_p(consumer.stream)) == 0
        pair[0].record_consumer(consumer.stream)
        assert not completed
        pair[0].close()
        pair[1].close()
        gc.collect()
        assert completed
        np.testing.assert_array_equal(output.numpy(), saved)
    view, stream, context = HostView(), ct.c_uint64(99), ct.c_uint64(99)
    assert lib.tessera_nvidia_tensor_result_query(handle, 0, ct.byref(view),
                                                ct.byref(stream), ct.byref(context)) != 0
    assert not view.data and stream.value == 0 and context.value == 0


def test_non_native_endian_roots_are_refused_without_poisoning_owner():
    fn, pair = _result_pair()
    for value in pair:
        value.close()
    a, b, da = inputs({"shape_mnk": (17, 19, 35), "storage": "f16"})
    swapped = a.astype(a.dtype.newbyteorder(">"))
    with pytest.raises(ValueError, match="shape/storage differs"):
        from tessera.compiler.native_tensor_result import execute
        execute(next(iter(fn._native_prepared_jvp_calls.values())), (swapped, b, da))
    result = fn.native_jvp(a, b, tangents=(da,), result_placement="cuda")
    assert all(np.isfinite(value.to_host()).all() for value in result)
    for value in result:
        value.close()
    fn.close_native_storage()


def test_result_survives_public_prepared_cache_eviction():
    fn, pair = _result_pair()
    saved = tuple(value.to_host() for value in pair)
    original = next(iter(fn._native_prepared_jvp_calls.values()))
    for m in range(1, 9):
        a, b, da = inputs({"shape_mnk": (m, 19, 35), "storage": "f16"})
        current = fn.native_jvp(a, b, tangents=(da,), result_placement="cuda")
        expected, budget = oracle(a, b, da)
        check(tuple(value.to_host() for value in current), expected, budget)
        for value in current:
            value.close()
    assert len(fn._native_prepared_jvp_calls) == 8
    assert original not in fn._native_prepared_jvp_calls.values()
    for value, expected in zip(pair, saved, strict=True):
        np.testing.assert_array_equal(value.to_host(), expected)
    fn.close_native_storage()
    for value in pair:
        value.close()


@pytest.mark.parametrize("storage", ["f16", "bf16"])
@pytest.mark.parametrize("active", ["a", "b"])
def test_fresh_process_portable_results_need_no_compiler(storage, active, tmp_path):
    import os
    import subprocess
    import sys
    a, b, da = inputs({"shape_mnk": (17, 19, 35), "storage": storage})
    tangent = da if active == "a" else (b.astype(np.float32) * .5 + .125).astype(b.dtype)
    fn = ts.jit(target="nvidia_sm120", autodiff="forward", wrt=(active,))(product)
    pair = fn.native_jvp(a, b, tangents=(tangent,), result_placement="cuda")
    expected = tuple(value.to_host() for value in pair)
    artifact = fn.native_jvp_runtime_artifact()
    (tmp_path / "artifact.json").write_text(artifact.to_json())
    # Store raw bytes so BF16 storage does not depend on NumPy's void-dtype
    # serialization spelling. The child reconstructs the declared storage.
    for name, value in zip(("a", "b", "da"), (a, b, tangent), strict=True):
        value.tofile(tmp_path / (name + ".bin"))
    for value in pair:
        value.close()
    fn.close_native_storage()
    code = """
import pathlib, subprocess, sys
import numpy as np
import ml_dtypes
from tessera import runtime as rt
from tessera.compiler.native_tensor_result import NativeTensorResult
root = pathlib.Path(sys.argv[1])
dtype = np.float16 if sys.argv[2] == "f16" else ml_dtypes.bfloat16
shapes = ((17,35),(35,19),(17,35) if sys.argv[3] == "a" else (35,19))
values = tuple(np.fromfile(root/(name+".bin"),dtype=dtype).reshape(shape)
               for name,shape in zip(("a","b","da"),shapes,strict=True))
artifact = rt.RuntimeArtifact.from_json((root/"artifact.json").read_text())
subprocess.run = lambda *a,**k: (_ for _ in ()).throw(AssertionError("portable replay invoked compiler"))
receipt = rt.launch(artifact, values)
assert receipt["ok"], receipt
pair = receipt["output"]
assert all(isinstance(value,NativeTensorResult) for value in pair)
np.savez(root/"result.npz", **{str(i):value.to_host() for i,value in enumerate(pair)})
for value in pair: value.close()
"""
    env = dict(os.environ, TESSERA_OPT="/compiler-must-not-run",
               TESSERA_NVIDIA_OPT="/compiler-must-not-run")
    result = subprocess.run([sys.executable, "-c", code, str(tmp_path), storage, active],
                            env=env, capture_output=True, text=True, timeout=60)
    assert result.returncode == 0, result.stdout + result.stderr
    with np.load(tmp_path / "result.npz") as loaded:
        for i, value in enumerate(expected):
            np.testing.assert_array_equal(loaded[str(i)], value)


@pytest.mark.parametrize("wrapped", [False, True])
def test_descriptor_cannot_overwrite_immutable_native_result(wrapped):
    from tests.device.nvidia.test_cooperative_norm import package
    fn, pair = _result_pair()
    before = tuple(value.to_host() for value in pair)
    fn.close_native_storage()
    _, native = package("rmsnorm", before[1], "serial")
    d = native.descriptor
    artifact = rt.RuntimeArtifact(metadata={"target": "nvidia_sm120"},
        tile_ir=native.tile_ir, target_ir=native.target_ir,
        native_image=native.image, launch_descriptor=d)
    source, destination = pair[1], pair[0]
    if wrapped:
        source = rt.NativeBufferValue(source, rt._native_buffer_value(source)[1])
        destination = rt.NativeBufferValue(destination, rt._native_buffer_value(destination)[1])
    try:
        receipt = rt.launch(artifact, {
            d.buffers[0].name: source, d.buffers[1].name: destination,
            "Rows": before[0].shape[0], "Columns": before[0].shape[1],
        }, stream=pair[0].__cuda_array_interface__["stream"])
        after = tuple(value.to_host() for value in pair)
        assert not receipt["ok"], (
            f"readonly result accepted as output; data changed={not np.array_equal(after[0],before[0])}")
        assert "writable CUDA storage" in receipt["reason"], receipt
        for actual, expected in zip(after, before, strict=True):
            np.testing.assert_array_equal(actual, expected)
    finally:
        for value in pair:
            value.close()
