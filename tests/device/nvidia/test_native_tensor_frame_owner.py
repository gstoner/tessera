"""Owning SM120 execution of the compiler-projected generic native frame."""
from contextlib import ExitStack
import copy
import ctypes as ct
import hashlib
import json
import os
from pathlib import Path
import select
import subprocess
from concurrent.futures import ThreadPoolExecutor

import ml_dtypes
import numpy as np
import pytest

from tessera import runtime as rt
from tessera.compiler.emit.nvidia_cuda import NvidiaDeviceSession
from tessera.compiler.prepared_native_tensor_frame import PreparedNativeTensorFrame
from tests.device.nvidia.test_ordered_resident_tensor_dag import Borrowed, queue_delayed_upload

pytestmark = pytest.mark.hardware_nvidia
ROOT = Path(__file__).resolve().parents[3]
CASES = [kind + "-" + storage for kind in ("relu_matmul", "norm_matmul", "two_sided")
         for storage in ("f16", "bf16")]


@pytest.fixture(scope="module")
def provider():
    if rt._nvidia_device_name() != "sm_120":
        pytest.skip("requires exact SM120 execution")
    path = os.environ.get("TESSERA_NATIVE_FRAME_TEST_LIB")
    if path is None:
        lib = rt._load_nvidia_ptx_launch()
        assert lib is not None and hasattr(lib, "tessera_nvidia_native_frame_prepare_v1"), (
            "owning SM120 validation requires the updated native frame provider")
        return lib
    return ct.CDLL(path)


def package(name):
    folder = Path(os.environ.get("TESSERA_NATIVE_FRAME_TEST_PACKET", str(
        ROOT / "benchmarks/baselines/native_tensor_program_pipeline_20261019/images")))
    record = json.loads((folder / (name + ".frame.json")).read_text())
    images = []
    for stage in record["stages"]:
        image = (folder / stage["ptx_file"]).read_bytes()
        assert hashlib.sha256(image).hexdigest() == stage["ptx_sha256"]
        images.append(image)
    frame = {key: copy.deepcopy(record[key]) for key in (
        "schema", "target", "architecture", "function", "buffers", "roots", "stages", "output")}
    for stage in frame["stages"]:
        del stage["ptx_file"], stage["ptx_sha256"]
    return frame, images


def values(name, seed=None):
    dtype = np.float16 if name.endswith("-f16") else ml_dtypes.bfloat16
    if seed is None:
        return np.full((17, 32), -.5, dtype), np.ones((32, 11), dtype)
    rng = np.random.default_rng(seed)
    return rng.normal(0, .2, (17, 32)).astype(dtype), rng.normal(0, .2, (32, 11)).astype(dtype)


def oracle(name, a, b):
    wide = a.astype(np.float64)
    lhs = np.maximum(wide, 0).astype(a.dtype)
    rhs = b
    if name.startswith("norm"):
        lhs = (wide / np.sqrt(np.mean(wide * wide, axis=-1, keepdims=True)
                             + float(np.float32(1e-5)))).astype(a.dtype)
    if name.startswith("two"):
        wide = b.astype(np.float64)
        wide -= wide.max(axis=-1, keepdims=True)
        rhs = (np.exp(wide) / np.exp(wide).sum(axis=-1, keepdims=True)).astype(b.dtype)
    return lhs.astype(np.float64) @ rhs.astype(np.float64)


@pytest.mark.parametrize("name", CASES)
def test_completed_host_replay_retains_allocations_and_snapshot(name, provider, monkeypatch):
    frame, images = package(name)
    with PreparedNativeTensorFrame(frame, images, provider=provider) as owner:
        initial = owner.stats()
        assert initial["allocations"] == 1 and initial["generation"] == 0
        assert initial["modules"] == len(images)
        with pytest.raises(RuntimeError, match="last completed frame"):
            owner.profile(2)
        old = []
        for seed in (None, 137, 138):
            a, b = values(name, seed)
            before = a.copy(), b.copy()
            expected = oracle(name, a, b)
            tail = np.full(17 * 11 + 16, 123, np.float32)
            output = tail[:17 * 11].reshape(17, 11)
            def forbidden(*_, **__):
                pytest.fail("warm native frame replay invoked a compiler")
            with monkeypatch.context() as guarded:
                guarded.setattr(subprocess, "run", forbidden)
                assert owner((a, b), output=output) is output
                timing = owner.profile(8, captured=True)
            np.testing.assert_allclose(output, expected, rtol=.004, atol=2e-5)
            np.testing.assert_array_equal(tail[17 * 11:], 123)
            np.testing.assert_array_equal(a, before[0])
            np.testing.assert_array_equal(b, before[1])
            assert timing["program_ms"] > 0
            assert all(value > 0 for value in timing["stage_ms"])
            old.append((output, output.copy()))
        after = owner.stats()
        assert after["allocations"] == initial["allocations"] == 1
        assert after["capacity"] == initial["capacity"]
        assert after["generation"] > initial["generation"]
        for output, saved in old:
            np.testing.assert_array_equal(output, saved)
    with pytest.raises(RuntimeError, match="closed"):
        owner.stats()


@pytest.mark.parametrize("name", CASES)
@pytest.mark.parametrize("mode", ["different", "shared", "legacy", "per_thread"])
def test_pending_resident_producers_complete_before_native_reads(name, mode, provider, monkeypatch):
    frame, images = package(name)
    a, b = values(name, 213)
    with ExitStack() as stack:
        first = stack.enter_context(NvidiaDeviceSession())
        second = first if mode != "different" else stack.enter_context(NvidiaDeviceSession())
        av, bv = first.upload(np.zeros_like(a)), second.upload(np.zeros_like(b))
        assert first.synchronize() == 0 and second.synchronize() == 0
        owner = stack.enter_context(PreparedNativeTensorFrame(frame, images, provider=provider))
        streams = (1, 1) if mode == "legacy" else (2, 2) if mode == "per_thread" else (
            first.stream, second.stream)
        roots = Borrowed(av, streams[0]), Borrowed(bv, streams[1])
        owner(roots)  # Prepare every allocation before delayed producer work.
        before = owner.stats()
        complete_a, complete_b = [], []
        callbacks = [
            queue_delayed_upload(first, av, a, streams[0], complete_a),
            queue_delayed_upload(second, bv, b, streams[1], complete_b),
        ]
        assert not complete_a and not complete_b
        def forbidden(*_, **__):
            pytest.fail("resident native frame replay invoked a compiler")
        with monkeypatch.context() as guarded:
            guarded.setattr(subprocess, "run", forbidden)
            result = owner(roots)
            timing = owner.profile(4, captured=True)
        assert complete_a and complete_b and len(callbacks) == 2
        np.testing.assert_allclose(result, oracle(name, a, b), rtol=.004, atol=2e-5)
        np.testing.assert_array_equal(first.download(av), a)
        np.testing.assert_array_equal(second.download(bv), b)
        assert owner.stats()["allocations"] == before["allocations"]
        assert timing["program_ms"] > 0
        saved = result.copy()
        # Completed host output survives retirement of all borrowed roots.
    np.testing.assert_array_equal(result, saved)


@pytest.mark.parametrize("edit", ["shape", "stride", "alias", "unknown", "uninitialized",
                                 "geometry", "entry", "target"])
def test_rejected_frames_preserve_caller_storage_and_previous_generation(edit, provider):
    frame, images = package("relu_matmul-f16")
    a, b = values("relu_matmul-f16", 552)
    with PreparedNativeTensorFrame(frame, images, provider=provider) as owner:
        correct = owner((a, b))
        before = owner.stats()
        if edit in {"shape", "stride", "alias"}:
            if edit == "shape":
                with pytest.raises(ValueError, match="shape"):
                    owner((a[:-1], b))
            elif edit == "stride":
                with pytest.raises(RuntimeError, match="stride"):
                    owner((np.asfortranarray(a), b))
            else:
                out = np.ndarray((17, 11), np.float32, buffer=a)
                saved = a.copy()
                with pytest.raises(RuntimeError, match="aliases"):
                    owner((a, b), output=out)
                np.testing.assert_array_equal(a, saved)
        else:
            broken = copy.deepcopy(frame)
            if edit == "unknown":
                broken["stages"][0]["numeric_policy"] = {"accum": "fp16"}
            elif edit == "uninitialized":
                broken["stages"][0]["inputs"][0] = broken["stages"][0]["output"]
            elif edit == "geometry":
                broken["stages"][0]["geometry"][3] = 2048
            elif edit == "entry":
                broken["stages"][0]["entry"] = "missing_native_program_entry"
            else:
                broken["target"] = "rocm"
            with pytest.raises((ValueError, RuntimeError)):
                PreparedNativeTensorFrame(broken, images, provider=provider)
        assert owner.stats() == before
        assert owner.profile(4)["program_ms"] > 0
        np.testing.assert_allclose(owner((a, b)), correct, rtol=0, atol=0)


def test_native_thread_and_fork_guards_precede_cuda_calls(provider):
    frame, images = package("relu_matmul-f16")
    with PreparedNativeTensorFrame(frame, images, provider=provider) as owner:
        owner(values("relu_matmul-f16"))
        def worker():
            fields = (ct.c_uint64 * 6)()
            status = provider.tessera_nvidia_native_frame_stats_v1(owner.handle, fields, len(fields))
            return status, provider.tessera_nvidia_matmul_last_error().decode()
        with ThreadPoolExecutor(max_workers=1) as pool:
            status, error = pool.submit(worker).result(timeout=5)
        assert status != 0 and "thread changed" in error
        read, write = os.pipe()
        pid = os.fork()
        if pid == 0:
            os.close(read)
            fields = (ct.c_uint64 * 6)()
            status = provider.tessera_nvidia_native_frame_stats_v1(owner.handle, fields, len(fields))
            error = provider.tessera_nvidia_matmul_last_error().decode()
            os.write(write, b"passed" if status and "fork" in error else b"failed")
            os._exit(0)
        os.close(write)
        try:
            ready, _, _ = select.select([read], [], [], 5)
            assert ready, "fork guard blocked inside a native/CUDA call"
            assert os.read(read, 32) == b"passed"
        finally:
            os.close(read)
            if not ready:
                os.kill(pid, 9)
            os.waitpid(pid, 0)
        assert owner.profile(4)["program_ms"] > 0



def test_independent_owners_do_not_invalidate_completed_frames(provider):
    frame, images = package("relu_matmul-f16")
    with PreparedNativeTensorFrame(frame, images, provider=provider) as first, (
            PreparedNativeTensorFrame(frame, images, provider=provider)) as second:
        first(values("relu_matmul-f16", 32))
        before = first.stats()
        second(values("relu_matmul-f16", 33))
        assert first.stats() == before
        assert first.profile(4)["program_ms"] > 0
        assert first.stats()["allocations"] == second.stats()["allocations"] == 1


def test_native_rejects_extreme_shape_before_capacity_arithmetic(provider):
    from tessera.compiler.prepared_native_tensor_frame import NativeFrameStep
    from tessera.compiler.prepared_nvidia_matmul import HostView

    # Construct/configure once, then bypass Python metadata guards deliberately.
    frame, images = package("relu_matmul-f16")
    with PreparedNativeTensorFrame(frame, images, provider=provider):
        buffers = (HostView * 2)()
        buffers[0].dtype, buffers[0].rank = 1, 2
        buffers[0].shape[:] = -(2**63), -(2**63)
        roles = (ct.c_uint8 * 2)(0, 2)
        steps = (NativeFrameStep * 1)()
        handle = ct.c_uint64(123)
        status = provider.tessera_nvidia_native_frame_prepare_v1(
            buffers, roles, 2, steps, 1, ct.byref(handle))
        assert status != 0 and handle.value == 0
        assert b"capacity/layout" in provider.tessera_nvidia_matmul_last_error()


def test_retired_resident_allocation_cannot_be_profiled(provider):
    frame, images = package("relu_matmul-f16")
    a, b = values("relu_matmul-f16", 72)
    with NvidiaDeviceSession() as session, PreparedNativeTensorFrame(frame, images, provider=provider) as owner:
        av, bv = session.upload(a), session.upload(b)
        output = owner((av, bv))
        saved = output.copy()
        av.close()
        replacement = session.empty(a.shape, a.dtype)
        times = (ct.c_float * len(frame["stages"]))(*([-123] * len(frame["stages"])))
        program = ct.c_float(-123)
        # Bypass the provider object's closed-property guard: the native frame
        # must validate both allocation extent and driver buffer identity itself.
        status = provider.tessera_nvidia_native_frame_profile_v1(
            owner.handle, 4, 0, times, len(times), ct.byref(program))
        assert status != 0
        assert program.value == -123 and list(times) == [-123] * len(times)
        assert replacement.ptr
        np.testing.assert_array_equal(output, saved)


@pytest.mark.parametrize("failure", ["capacity", "stream"])
def test_invalid_resident_roots_preserve_native_frame_and_output(failure, provider):
    frame, images = package("relu_matmul-f16")
    a, b = values("relu_matmul-f16", 94)
    with NvidiaDeviceSession() as session, PreparedNativeTensorFrame(frame, images, provider=provider) as owner:
        owner((a, b))
        before = owner.stats()
        av, bv = session.upload(a), session.upload(b)
        if failure == "capacity":
            driver = ct.CDLL("libcuda.so.1")
            query = driver.cuMemGetAddressRange_v2
            query.argtypes = [ct.POINTER(ct.c_uint64), ct.POINTER(ct.c_size_t), ct.c_uint64]
            query.restype = ct.c_int
            base, capacity = ct.c_uint64(), ct.c_size_t()
            assert query(ct.byref(base), ct.byref(capacity), av.ptr) == 0
            pointer = base.value + capacity.value - a.dtype.itemsize
        else:
            pointer = av.ptr
        class InvalidRoot(Borrowed):
            @property
            def __cuda_array_interface__(self):
                interface = super().__cuda_array_interface__
                interface["data"] = pointer, True
                if failure == "stream":
                    interface["stream"] = 3
                return interface
        roots = InvalidRoot(av), Borrowed(bv)
        output = np.full((17, 11), 123, np.float32)
        with pytest.raises(RuntimeError, match="capacity|stream token"):
            owner(roots, output=output)
        np.testing.assert_array_equal(output, 123)
        assert owner.stats() == before
        assert owner.profile(4)["program_ms"] > 0


def test_foreign_context_rejected_before_caller_output_commit(provider):
    frame, images = package("relu_matmul-f16")
    a, b = values("relu_matmul-f16", 294)
    cuda = ct.CDLL("libcuda.so.1")
    for name, signature in {
        "cuCtxGetCurrent": [ct.POINTER(ct.c_void_p)],
        "cuCtxCreate_v2": [ct.POINTER(ct.c_void_p), ct.c_uint, ct.c_int],
        "cuCtxSetCurrent": [ct.c_void_p], "cuCtxDestroy_v2": [ct.c_void_p],
    }.items():
        function = getattr(cuda, name)
        function.argtypes, function.restype = signature, ct.c_int
    with PreparedNativeTensorFrame(frame, images, provider=provider) as owner:
        expected = owner((a, b))
        before = owner.stats()
        original, foreign = ct.c_void_p(), ct.c_void_p()
        assert cuda.cuCtxGetCurrent(ct.byref(original)) == 0
        assert cuda.cuCtxCreate_v2(ct.byref(foreign), 0, 0) == 0
        output = np.full((17, 11), 123, np.float32)
        try:
            with pytest.raises(RuntimeError, match="context changed"):
                owner((a, b), output=output)
            np.testing.assert_array_equal(output, 123)
        finally:
            assert cuda.cuCtxSetCurrent(original) == 0
            assert cuda.cuCtxDestroy_v2(foreign) == 0
        assert owner.stats() == before
        assert owner.profile(4)["program_ms"] > 0
        np.testing.assert_array_equal(owner((a, b)), expected)
