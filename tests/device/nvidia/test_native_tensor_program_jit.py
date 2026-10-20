"""Ordinary frontend calls must consume the compiler-owned native program."""
import subprocess

import numpy as np
import pytest
import tessera as ts
from tessera import runtime as rt
from tests.device.nvidia.test_native_tensor_program_package_execution import (
    CASES, FUNCTIONS, oracle, roots, values,
)

pytestmark = pytest.mark.hardware_nvidia


@pytest.mark.parametrize("kind,storage", CASES)
def test_ordinary_jit_uses_native_program_and_replays_without_tools(kind, storage, monkeypatch):
    assert rt._nvidia_device_name() == "sm_120"
    fn = ts.jit(target="nvidia_sm120")(FUNCTIONS[kind])
    try:
        a, b = values(storage)
        first = fn(*roots(kind, a, b))
        np.testing.assert_allclose(first, oracle(kind, a, b), rtol=.004, atol=2e-5)
        assert fn.execution_kind == "native_gpu"
        artifact = fn.runtime_artifact()
        assert "tessera_nvidia.program_invoke " in artifact.target_ir
        assert "tessera.matmul" in artifact.graph_ir
        assert "schedule." in artifact.schedule_ir
        assert "tile.program_invoke " in artifact.tile_ir
        assert fn.schedule_ir == artifact.schedule_ir
        assert fn.tile_ir == artifact.tile_ir
        assert fn.target_ir == artifact.target_ir
        assert fn.ir_text() == artifact.graph_ir
        stages = fn.lowering_artifacts()
        assert all(b.input_digest == a.output_digest for a, b in zip(stages, stages[1:]))
        assert fn.compile_report().plan_hash == artifact.artifact_hash
        if kind == "norm":
            assert 'accum = "fp32"' in artifact.graph_ir
        saved = first.copy()

        def forbidden(*args, **kwargs):
            pytest.fail("ordinary native JIT replay invoked tracing or an external compiler")

        with monkeypatch.context() as guarded:
            guarded.setattr(subprocess, "run", forbidden)
            guarded.setattr(subprocess, "Popen", forbidden)
            guarded.setattr(fn, "_trace_frontend_capture", forbidden)
            for seed in (71, 72):
                a, b = values(storage, seed)
                before = a.copy(), b.copy()
                result = fn(*roots(kind, a, b))
                np.testing.assert_allclose(result, oracle(kind, a, b), rtol=.004, atol=2e-5)
                np.testing.assert_array_equal(a, before[0])
                np.testing.assert_array_equal(b, before[1])
                np.testing.assert_array_equal(first, saved)
                assert not np.shares_memory(first, result)
            fn.close_native_storage()
            # Closing resources preserves the verified compiler product, so
            # rebinding the same envelope does not retrace or recompile.
            result = fn(*roots(kind, a, b))
            np.testing.assert_allclose(result, oracle(kind, a, b), rtol=.004, atol=2e-5)
    finally:
        fn.close_native_storage()


@pytest.mark.parametrize("kind", ["relu", "norm"])
@pytest.mark.parametrize("storage", ["f16", "bf16"])
def test_ordinary_jit_runtime_artifact_executes_verified_source(kind, storage):
    assert rt._nvidia_device_name() == "sm_120"
    fn = ts.jit(target="nvidia_sm120")(FUNCTIONS[kind])
    a, b = values(storage, 73)
    try:
        fn(*roots(kind, a, b))
        artifact = rt.RuntimeArtifact.from_dict(fn.runtime_artifact().to_dict())
        receipt = rt.launch(artifact, roots(kind, a, b))
        assert receipt.get("ok") is True, receipt
        assert receipt["execution_kind"] == "native_gpu"
        assert receipt["compiler_path"] == "canonical_native_tensor_program"
        np.testing.assert_allclose(receipt["output"], oracle(kind, a, b), rtol=.004, atol=2e-5)
        refused = rt.launch(artifact, roots(kind, a, b), stream=1)
        assert refused.get("ok") is False
    finally:
        fn.close_native_storage()


@pytest.mark.timeout(300)
def test_full_frame_stage_capacity_executes_source_program():
    from tessera.compiler.native_tensor_program import compile_native_tensor_program
    from tests.unit.test_native_tensor_program_capacity import chain_source

    assert rt._nvidia_device_name() == "sm_120"
    program = compile_native_tensor_program(chain_source(64))
    assert program is not None and len(program.members) == 64
    a = np.linspace(-1, 1, 16, dtype=np.float16).reshape(1, 16)
    b = np.linspace(.1, .9, 16, dtype=np.float16).reshape(16, 1)
    expected = np.maximum(a.astype(np.float64), 0) @ b.astype(np.float64)
    with program.prepare() as owner:
        result = owner((a, b))
        np.testing.assert_allclose(result, expected, rtol=.004, atol=2e-5)
        assert owner.stats()["stages"] == 64


@pytest.mark.parametrize("kind,storage", CASES)
@pytest.mark.parametrize("mode", ["different", "shared", "legacy", "per_thread"])
def test_ordinary_jit_orders_resident_inputs_and_retains_completed_outputs(
    kind, storage, mode, monkeypatch,
):
    from contextlib import ExitStack
    from tessera.compiler.emit.nvidia_cuda import NvidiaDeviceSession
    from tests.device.nvidia.test_ordered_resident_tensor_dag import Borrowed, queue_delayed_upload

    assert rt._nvidia_device_name() == "sm_120"
    a, b = values(storage, 74)
    fn = ts.jit(target="nvidia_sm120")(FUNCTIONS[kind])
    with ExitStack() as stack:
        first = stack.enter_context(NvidiaDeviceSession())
        second = first if mode != "different" else stack.enter_context(NvidiaDeviceSession())
        av, bv = first.upload(np.zeros_like(a)), second.upload(np.zeros_like(b))
        assert first.synchronize() == 0 and second.synchronize() == 0
        streams = ((1, 1) if mode == "legacy" else (2, 2) if mode == "per_thread"
                   else (first.stream, second.stream))
        resident = Borrowed(av, streams[0]), Borrowed(bv, streams[1])
        try:
            fn(*roots(kind, *resident))
            owner = fn._native_tensor_last_owner
            assert owner is not None and fn._native_tensor_last_program is not None
            before = owner.stats()
            done_a, done_b = [], []
            callbacks = [
                queue_delayed_upload(first, av, a, streams[0], done_a),
                queue_delayed_upload(second, bv, b, streams[1], done_b),
            ]
            assert not done_a and not done_b

            def forbidden(*args, **kwargs):
                pytest.fail("resident JIT replay invoked tracing or external tools")

            with monkeypatch.context() as guarded:
                guarded.setattr(subprocess, "run", forbidden)
                guarded.setattr(subprocess, "Popen", forbidden)
                guarded.setattr(fn, "_trace_frontend_capture", forbidden)
                output = fn(*roots(kind, *resident))
                assert fn._native_tensor_last_owner is owner
                assert owner.stats()["allocations"] == before["allocations"] == 1
                assert owner.stats()["capacity"] == before["capacity"]
            assert done_a and done_b and len(callbacks) == 2
            np.testing.assert_allclose(output, oracle(kind, a, b), rtol=.004, atol=2e-5)
            assert fn.execution_kind == "native_gpu"
            np.testing.assert_array_equal(first.download(av), a)
            np.testing.assert_array_equal(second.download(bv), b)
            saved = output.copy()
        finally:
            fn.close_native_storage()
    np.testing.assert_array_equal(output, saved)


def layernorm_product(a, b):
    return ts.ops.matmul(ts.ops.layer_norm(a, eps=1e-5),
                         b, output_dtype="fp32")


def softmax_product(a, b):
    return ts.ops.matmul(ts.ops.softmax(a, axis=-1), b, output_dtype="fp32")


def relu_half_product(a, b):
    return ts.ops.matmul(ts.ops.relu(a), b, output_dtype="fp16")


def norm_half_product(a, b):
    return ts.ops.matmul(ts.ops.rmsnorm(a, eps=1e-5, numeric_policy={"accum": "fp32"}),
                         b, output_dtype="fp16")


@pytest.mark.parametrize("storage", ["f16", "bf16"])
@pytest.mark.parametrize("kind,function", [
    ("layernorm", layernorm_product), ("softmax", softmax_product),
])
def test_additional_unary_producers_follow_original_native_source(storage, kind, function):
    assert rt._nvidia_device_name() == "sm_120"
    a, b = values(storage, 75)
    wide = a.astype(np.float64)
    if kind == "layernorm":
        centered = wide - np.mean(wide, axis=-1, keepdims=True)
        produced = centered / np.sqrt(np.mean(centered * centered, axis=-1, keepdims=True)
                                     + float(np.float32(1e-5)))
    else:
        exponents = np.exp(wide - wide.max(axis=-1, keepdims=True))
        produced = exponents / exponents.sum(axis=-1, keepdims=True)
    expected = produced.astype(a.dtype).astype(np.float64) @ b.astype(np.float64)
    fn = ts.jit(target="nvidia_sm120")(function)
    try:
        output = fn(a, b)
        assert fn._native_tensor_last_program is not None
        assert fn.execution_kind == "native_gpu"
        np.testing.assert_allclose(output, expected, rtol=.004, atol=2e-5)
    finally:
        fn.close_native_storage()


@pytest.mark.parametrize("storage", ["f16", "bf16"])
@pytest.mark.parametrize("kind,function", [
    ("relu", relu_half_product), ("norm", norm_half_product),
])
def test_native_half_output_keeps_declared_store_storage(storage, kind, function):
    assert rt._nvidia_device_name() == "sm_120"
    a, b = values(storage, 76)
    fn = ts.jit(target="nvidia_sm120")(function)
    try:
        output = fn(a, b)
        program = fn._native_tensor_last_program
        assert program is not None
        assert output.dtype == np.dtype(np.float16)
        assert program.frame["buffers"][program.frame["output"]]["storage"] == "f16"
        expected = oracle(kind, a, b).astype(np.float16)
        np.testing.assert_allclose(output, expected, rtol=.004, atol=2e-5)
    finally:
        fn.close_native_storage()


@pytest.mark.parametrize("kind", ["relu", "norm"])
@pytest.mark.parametrize("storage", ["f16", "bf16"])
def test_native_jit_retains_and_closes_owners_on_each_calling_thread(kind, storage, monkeypatch):
    from concurrent.futures import ThreadPoolExecutor
    import threading

    assert rt._nvidia_device_name() == "sm_120"
    fn = ts.jit(target="nvidia_sm120")(FUNCTIONS[kind])
    a, b = values(storage, 810)
    fn(*roots(kind, a, b))
    primary = fn._native_tensor_last_owner
    ready = threading.Barrier(3, timeout=60)
    resume = threading.Event()

    def forbidden(*args, **kwargs):
        pytest.fail("thread-local owner reuse or rebind retraced or compiled")

    def worker(seed):
        current = None
        try:
            for offset in range(2):
                left, right = values(storage, seed + offset)
                actual = fn(*roots(kind, left, right))
                np.testing.assert_allclose(actual, oracle(kind, left, right), rtol=.004, atol=2e-5)
                assert fn.execution_kind == "native_gpu"
                if current is None:
                    current = fn._native_tensor_last_owner
                assert current is fn._native_tensor_last_owner
                assert current.thread is threading.current_thread()
                assert fn.runtime_artifact().graph_ir == fn._native_tensor_last_program.source_ir
            ready.wait()
            assert resume.wait(60)
            assert current.handle and current.stats()["allocations"] == 1
            fn.close_native_storage()
            assert current.handle == 0
            left, right = values(storage, seed + 2)
            actual = fn(*roots(kind, left, right))
            np.testing.assert_allclose(actual, oracle(kind, left, right), rtol=.004, atol=2e-5)
            rebound = fn._native_tensor_last_owner
            assert rebound is not current
            return current, rebound
        except BaseException:
            ready.abort()
            raise
        finally:
            fn.close_native_storage()

    try:
        with monkeypatch.context() as guarded:
            guarded.setattr(subprocess, "run", forbidden)
            guarded.setattr(subprocess, "Popen", forbidden)
            guarded.setattr(fn, "_trace_frontend_capture", forbidden)
            with ThreadPoolExecutor(max_workers=2) as pool:
                futures = [pool.submit(worker, seed) for seed in (811, 821)]
                try:
                    try:
                        ready.wait()
                    except threading.BrokenBarrierError:
                        for future in futures:
                            future.result(timeout=90)
                        raise
                    fn.close_native_storage()
                    assert primary.handle == 0
                finally:
                    resume.set()
                pairs = [future.result(timeout=90) for future in futures]
                assert pairs[0][0] is not pairs[1][0]
                assert all(not owner.handle for pair in pairs for owner in pair)
    finally:
        resume.set()
        fn.close_native_storage()
