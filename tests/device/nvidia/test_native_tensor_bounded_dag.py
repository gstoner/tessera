"""Exact SM120 changed-M execution of native SSA producer DAGs."""
import subprocess

import ml_dtypes
import numpy as np
import pytest
import tessera as ts
from tessera import runtime
from tessera.compiler.prepared_native_tensor_frame import PreparedNativeTensorFrame
from tests.device.nvidia.test_native_tensor_pipeline_layout import full_native_program, check_semantic
from tests.unit.test_native_tensor_bounded_dag import CASES, dag_source
from tests.unit.test_native_tensor_mlp_program import assert_mlp_numeric

pytestmark = pytest.mark.hardware_nvidia


def roots(case, storage, m, *, k=32, seed=16001):
    dtype = np.float16 if storage == "f16" else ml_dtypes.bfloat16
    rng = np.random.default_rng(seed)

    def tensor(shape, kind=dtype):
        return rng.normal(0, .2, shape).astype(kind)

    a = tensor((m, k))
    if case in ("two_matmuls", "fp32_middle"):
        return a, tensor((k, 16)), tensor((16, 11), np.float16 if case == "two_matmuls" else dtype)
    b = tensor((k, 11))
    if case == "static_rhs":
        return a, b, tensor((k, 11))
    if case == "chain":
        return a, b, tensor((m, k))
    return a, b


def check(case, actual, args):
    if case in ("two_matmuls", "fp32_middle"):
        assert_mlp_numeric(actual, args, False, "f32" if case == "fp32_middle" else "f16")
        return
    if case in ("rmsnorm", "softmax"):
        check_semantic(case, actual, args)
        return
    a, b, *extra = args
    dtype = a.dtype
    if case == "chain":
        x = np.maximum((a.astype(np.float32) + extra[0].astype(np.float32)).astype(dtype), 0).astype(dtype)
    elif case == "fork_join":
        u = np.maximum(a, 0).astype(dtype)
        y = (u.astype(np.float32) + a.astype(np.float32)).astype(dtype)
        x = (y.astype(np.float32) + u.astype(np.float32)).astype(dtype)
    else:
        x = np.maximum(a, 0).astype(dtype)
        b = (b.astype(np.float32) + extra[0].astype(np.float32)).astype(dtype)
    wide_a, wide_b = x.astype(np.float64), b.astype(np.float64)
    expected = wide_a @ wide_b
    gamma = (wide_a.shape[1] * 2**-24) / (1 - wide_a.shape[1] * 2**-24)
    bound = 2 * gamma * (np.abs(wide_a) @ np.abs(wide_b))
    bound += np.abs(np.spacing(expected.astype(np.float32))).astype(np.float64)
    assert np.isfinite(actual).all()
    assert np.all(np.abs(actual.astype(np.float64) - expected) <= bound)


@pytest.mark.parametrize("case", CASES)
@pytest.mark.parametrize("storage", ["f16", "bf16"])
@pytest.mark.parametrize("form", ["policy", "dynamic"])
def test_bounded_dag_executes_changed_rows_with_one_native_arena(case, storage, form, monkeypatch):
    assert runtime._nvidia_device_name() == "sm_120"
    frame, images = full_native_program(dag_source(case, storage, form=form))
    assert frame["row_bound"] == 65

    def forbidden(*args, **kwargs):
        pytest.fail("retained bounded DAG invoked a compiler")

    kept = []
    with PreparedNativeTensorFrame(frame, images) as owner:
        before = owner.stats()
        for seed, m in enumerate((17, 1, 65, 5), 16001):
            args = roots(case, storage, m, seed=seed)
            snapshots = tuple(value.copy() for value in args)
            tail = np.full(m * 11 + 16, 123, np.float32)
            output = tail[:m * 11].reshape(m, 11)
            with monkeypatch.context() as guard:
                guard.setattr(subprocess, "run", forbidden)
                guard.setattr(subprocess, "Popen", forbidden)
                assert owner(args, output=output) is output
            check(case, output, args)
            np.testing.assert_array_equal(tail[m * 11:], 123)
            for value, saved in zip(args, snapshots, strict=True):
                np.testing.assert_array_equal(value, saved)
            for value, saved in kept:
                np.testing.assert_array_equal(value, saved)
            kept.append((output, output.copy()))
        with pytest.raises(ValueError):
            owner(roots(case, storage, 66))
        for value, saved in kept:
            np.testing.assert_array_equal(value, saved)
        after = owner.stats()
        assert before["allocations"] == after["allocations"] == 1
        assert before["capacity"] == after["capacity"]


def chain(a, b, r):
    return ts.ops.matmul(ts.ops.relu(ts.ops.add(a, r)), b, output_dtype="fp32")


def static_rhs(a, b, r):
    return ts.ops.matmul(ts.ops.relu(a), ts.ops.add(b, r), output_dtype="fp32")


def fork_join(a, b):
    u = ts.ops.relu(a)
    return ts.ops.matmul(ts.ops.add(ts.ops.add(u, a), u), b, output_dtype="fp32")


@pytest.mark.parametrize("case", CASES)
@pytest.mark.parametrize("storage", ["f16", "bf16"])
@pytest.mark.parametrize("trace_m,k", [(17, 32), (1, 1), (17, 17)])
def test_public_bounded_dag_reuses_native_program_across_rows(case, storage, trace_m, k, monkeypatch):
    from tests.unit.test_native_tensor_mlp_program import mlp, default_mlp
    from tests.unit.test_native_tensor_program_pipeline import norm_product
    from tests.unit.test_native_tensor_pipeline_layout import softmax_product

    assert runtime._nvidia_device_name() == "sm_120"
    functions = {"chain": chain, "static_rhs": static_rhs, "fork_join": fork_join,
                 "two_matmuls": mlp, "fp32_middle": default_mlp,
                 "rmsnorm": norm_product, "softmax": softmax_product}
    fn = ts.jit(target="nvidia_sm120", shape_bounds={"M": 65})(functions[case])

    def forbidden(*args, **kwargs):
        pytest.fail("warm bounded frontend call traced, compiled or executed eager")

    try:
        initial = roots(case, storage, trace_m, k=k)
        first = fn(*initial)
        check(case, first, initial)
        saved = first.copy()
        program, owner = fn._native_tensor_last_program, fn._native_tensor_last_owner
        assert fn.execution_kind == "native_gpu"
        assert program.frame["schema"] == "tessera.native.tensor_program_frame.v2"
        before = owner.stats()
        for m in (1, 65, 5):
            args = roots(case, storage, m, k=k, seed=16100 + m)
            snapshots = tuple(value.copy() for value in args)
            with monkeypatch.context() as guard:
                guard.setattr(fn, "_fn", forbidden)
                guard.setattr(fn, "_trace_frontend_capture", forbidden)
                guard.setattr(subprocess, "run", forbidden)
                guard.setattr(subprocess, "Popen", forbidden)
                actual = fn(*args)
            check(case, actual, args)
            assert fn._native_tensor_last_program is program
            assert fn._native_tensor_last_owner is owner
            assert owner.stats()["allocations"] == before["allocations"] == 1
            assert owner.stats()["capacity"] == before["capacity"]
            np.testing.assert_array_equal(first, saved)
            for value, snapshot in zip(args, snapshots, strict=True):
                np.testing.assert_array_equal(value, snapshot)
        restored = runtime.RuntimeArtifact.from_json(fn.runtime_artifact().to_json())
        replay_args = roots(case, storage, 5, k=k, seed=16201)
        receipt = runtime.launch(restored, dict(zip(restored.metadata["arg_names"], replay_args, strict=True)))
        assert receipt["ok"] and receipt["execution_kind"] == "native_gpu", receipt
        check(case, receipt["output"], replay_args)
    finally:
        fn.close_native_storage()
