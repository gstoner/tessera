"""Owning RTX5070 execution of semantic DAGs without a matrix terminal."""
import subprocess

import ml_dtypes
import numpy as np
import pytest
import tessera as ts
from tessera import runtime
from tessera.compiler.native_tensor_program import _frame
from tessera.compiler.nvidia_native import _compile_tile_ir
from tessera.compiler.prepared_native_tensor_frame import PreparedNativeTensorFrame
from tessera.compiler.scheduled_matmul import find_tessera_opt, run_tessera_opt
from tests.unit.test_native_semantic_dag_ownership import semantic_source

pytestmark = pytest.mark.hardware_nvidia
CASES = ("rmsnorm_then_relu", "softmax_then_rmsnorm", "layer_norm_then_softmax", "fork_join")


def inputs(storage, m, seed):
    dtype = {"f16": np.float16, "bf16": ml_dtypes.bfloat16, "f32": np.float32}[storage]
    return (np.random.default_rng(seed).normal(0, .2, (m, 32)).astype(dtype),)


def oracle(case, a):
    def apply(name, value):
        wide = value.astype(np.float64)
        if name == "relu":
            out = np.maximum(wide, 0)
        elif name == "softmax":
            exp = np.exp(wide - wide.max(axis=-1, keepdims=True))
            out = exp / exp.sum(axis=-1, keepdims=True)
        else:
            centered = wide - wide.mean(axis=-1, keepdims=True) if name == "layer_norm" else wide
            out = centered / np.sqrt(np.mean(centered * centered, axis=-1, keepdims=True) + 1e-5)
        return out.astype(a.dtype)
    if case == "fork_join":
        return (apply("relu", a).astype(np.float64) + a.astype(np.float64)).astype(a.dtype)
    first, second = case.split("_then_")
    return apply(second, apply(first, a))


def check(case, output, args):
    expected = oracle(case, args[0])
    assert np.isfinite(output).all()
    np.testing.assert_allclose(output.astype(np.float64), expected.astype(np.float64),
                               rtol=.012 if str(args[0].dtype) == "bfloat16" else .002,
                               atol=2e-5)


def compile_program(source):
    tool = find_tessera_opt()
    assert tool is not None
    target = run_tessera_opt(tool, source,
        "--pass-pipeline=builtin.module(tessera-nvidia-pipeline-sm120,tessera-lower-to-nvidia-sm120)")
    frame = _frame(target)
    assert len(frame["stages"]) == 2
    assert run_tessera_opt(tool, target, "--tessera-graph-to-schedule=tensor-program-replay") == target
    images = []
    for index, stage in enumerate(frame["stages"]):
        member = run_tessera_opt(tool, target, f"--tessera-schedule-to-tile=tensor-member={index}")
        _, ptx, *_ = _compile_tile_ir(member, stage["entry"], tessera_opt=tool)
        images.append(ptx.encode())
    return frame, tuple(images)


@pytest.mark.parametrize("case", CASES)
@pytest.mark.parametrize("storage", ["f16", "bf16", "f32"])
@pytest.mark.parametrize("form", ["static", "policy", "dynamic"])
def test_semantic_owned_execution(case, storage, form, monkeypatch):
    assert runtime._nvidia_device_name() == "sm_120"
    frame, images = compile_program(semantic_source(case, storage, form))
    kept = []
    with PreparedNativeTensorFrame(frame, images) as owner:
        before = owner.stats()
        for m in ((1,) if form == "static" else (1, 17, 65, 5)):
            args = inputs(storage, m, 19000 + m)
            snapshot = args[0].copy()
            tail = np.full(m * 32 + 16, 123, args[0].dtype)
            out = tail[:m * 32].reshape(m, 32)
            def forbidden(*args, **kwargs):
                pytest.fail("prepared execution invoked compiler")
            with monkeypatch.context() as guard:
                guard.setattr(subprocess, "run", forbidden)
                guard.setattr(subprocess, "Popen", forbidden)
                assert owner(args, output=out) is out
            check(case, out, args)
            np.testing.assert_array_equal(tail[m * 32:], 123)
            np.testing.assert_array_equal(args[0], snapshot)
            for previous, saved in kept:
                np.testing.assert_array_equal(previous, saved)
            kept.append((out, out.copy()))
        assert owner.stats()["allocations"] == before["allocations"] == 1
        assert owner.stats()["capacity"] == before["capacity"]
        if form != "static":
            with pytest.raises(ValueError):
                owner(inputs(storage, 66, 19100))


def rmsnorm_then_relu(a):
    return ts.ops.relu(ts.ops.rmsnorm(a, eps=1e-5))


def softmax_then_rmsnorm(a):
    return ts.ops.rmsnorm(ts.ops.softmax(a, axis=-1), eps=1e-5)


def layer_norm_then_softmax(a):
    return ts.ops.softmax(ts.ops.layer_norm(a, eps=1e-5), axis=-1)


def fork_join(a):
    return ts.ops.add(ts.ops.relu(a), a)


@pytest.mark.parametrize("case", CASES)
@pytest.mark.parametrize("storage", ["f16", "bf16", "f32"])
def test_public_semantic_jit_retains_native_owner(case, storage, monkeypatch):
    assert runtime._nvidia_device_name() == "sm_120"
    fn = ts.jit(target="nvidia_sm120", shape_bounds={"M": 65})(globals()[case])
    try:
        first_args = inputs(storage, 1, 19200)
        first = fn(*first_args)
        check(case, first, first_args)
        saved = first.copy()
        assert fn.execution_kind == "native_gpu"
        owner, program = fn._native_tensor_last_owner, fn._native_tensor_last_program
        assert program.frame["roots"] == [0]
        before = owner.stats()
        def forbidden(*args, **kwargs):
            pytest.fail("warm JIT invoked eager, trace or compiler")
        for m in (17, 65, 5):
            args = inputs(storage, m, 19200 + m)
            with monkeypatch.context() as guard:
                guard.setattr(fn, "_fn", forbidden)
                guard.setattr(fn, "_trace_frontend_capture", forbidden)
                guard.setattr(subprocess, "run", forbidden)
                output = fn(*args)
            check(case, output, args)
            np.testing.assert_array_equal(first, saved)
            assert fn._native_tensor_last_owner is owner
        assert owner.stats()["allocations"] == before["allocations"] == 1
    finally:
        owner = getattr(fn, "_native_tensor_last_owner", None)
        if owner is not None:
            owner.close()


@pytest.mark.parametrize("case", ["rmsnorm_then_relu", "fork_join"])
@pytest.mark.parametrize("storage", ["f16", "bf16", "f32"])
@pytest.mark.parametrize("mode", ["explicit", "legacy", "per_thread"])
def test_single_resident_root_orders_pending_producer(case, storage, mode, monkeypatch):
    from tessera.compiler.emit.nvidia_cuda import NvidiaDeviceSession
    from tests.device.nvidia.test_ordered_resident_tensor_dag import Borrowed, queue_delayed_upload

    assert runtime._nvidia_device_name() == "sm_120"
    args = inputs(storage, 17, 19400)
    fn = ts.jit(target="nvidia_sm120")(globals()[case])
    with NvidiaDeviceSession() as session:
        device = session.upload(np.zeros_like(args[0]))
        assert session.synchronize() == 0
        stream = 1 if mode == "legacy" else 2 if mode == "per_thread" else session.stream
        borrowed = Borrowed(device, stream)
        try:
            fn(borrowed)
            owner = fn._native_tensor_last_owner
            assert owner is not None
            before = owner.stats()
            completed = []
            callback = queue_delayed_upload(session, device, args[0], stream, completed)
            assert not completed
            def forbidden(*args, **kwargs):
                pytest.fail("resident warm JIT retraced or invoked compiler")
            with monkeypatch.context() as guard:
                guard.setattr(subprocess, "run", forbidden)
                guard.setattr(subprocess, "Popen", forbidden)
                guard.setattr(fn, "_trace_frontend_capture", forbidden)
                result = fn(borrowed)
            assert completed and callback is not None
            check(case, result, args)
            np.testing.assert_array_equal(session.download(device), args[0])
            assert fn._native_tensor_last_owner is owner
            assert owner.stats()["allocations"] == before["allocations"] == 1
            saved = result.copy()
        finally:
            fn.close_native_storage()
    np.testing.assert_array_equal(result, saved)
