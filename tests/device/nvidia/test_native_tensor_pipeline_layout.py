"""Exact SM120 execution of complete registered Graph-to-NVVM programs."""
import subprocess

import numpy as np
import pytest
from tessera import runtime
from tessera.compiler.native_tensor_program import _frame
from tessera.compiler.nvidia_native import _compile_tile_ir
from tessera.compiler.prepared_native_tensor_frame import PreparedNativeTensorFrame
from tessera.compiler.scheduled_matmul import find_tessera_opt, run_tessera_opt
from tests.device.nvidia.test_native_tensor_binary_producers import check as check_binary, roots as binary_roots
from tests.device.nvidia.test_native_tensor_program_package_execution import oracle, values
from tests.unit.test_native_tensor_mlp_program import mlp_source, mlp_roots, assert_mlp_numeric
from tests.unit.test_native_tensor_pipeline_layout import annotated_source

pytestmark = pytest.mark.hardware_nvidia


def full_native_program(source):
    tool = find_tessera_opt()
    assert tool is not None
    target = run_tessera_opt(tool, source,
        "--pass-pipeline=builtin.module(tessera-nvidia-pipeline-sm120,tessera-lower-to-nvidia-sm120)")
    assert "tessera_nvidia.mma_sync " not in target
    assert "nvvm.mma.sync" in target
    assert run_tessera_opt(tool, target,
                          "--tessera-graph-to-schedule=tensor-program-replay") == target
    frame = _frame(target)
    images = []
    for index, stage in enumerate(frame["stages"]):
        member = run_tessera_opt(tool, target,
                                f"--tessera-schedule-to-tile=tensor-member={index}")
        _, ptx, *_ = _compile_tile_ir(member, stage["entry"], tessera_opt=tool)
        images.append(ptx.encode())
    return frame, tuple(images)


def forbidden(*args, **kwargs):
    pytest.fail("retained full-pipeline execution invoked a compiler")


def check_semantic(kind, output, args):
    if kind == "add":
        check_binary(output, "lhs", args)
        return
    a, b = args
    if kind == "softmax":
        wide = a.astype(np.float64)
        exp = np.exp(wide - wide.max(axis=-1, keepdims=True))
        stored = (exp / exp.sum(axis=-1, keepdims=True)).astype(a.dtype)
        expected = stored.astype(np.float64) @ b.astype(np.float64)
    else:
        expected = oracle("norm" if kind == "rmsnorm" else "relu", a, b)
    # Preserve the existing native producer/package numerical envelope.
    np.testing.assert_allclose(output, expected, rtol=.004, atol=2e-5)


@pytest.mark.parametrize("kind", ["add", "relu", "rmsnorm", "softmax"])
@pytest.mark.parametrize("storage", ["f16", "bf16"])
@pytest.mark.parametrize("key", ["layout", "tessera.layout"])
def test_full_pipeline_layout_executes_original_producers(kind, storage, key, monkeypatch):
    assert runtime._nvidia_device_name() == "sm_120"
    frame, images = full_native_program(annotated_source(kind, storage, key, '\"row_major\"'))
    assert len(frame["stages"]) == len(images) == 2
    kept = []
    with PreparedNativeTensorFrame(frame, images) as owner:
        before = owner.stats()
        assert before["allocations"] == 1
        for seed in (15001, 15002, 15003):
            args = (binary_roots("lhs", storage, (17, 32, 11), seed) if kind == "add"
                    else values(storage, seed))
            snapshots = tuple(value.copy() for value in args)
            tail = np.full(17 * 11 + 16, 123, np.float32)
            output = tail[:17 * 11].reshape(17, 11)
            with monkeypatch.context() as guard:
                guard.setattr(subprocess, "run", forbidden)
                guard.setattr(subprocess, "Popen", forbidden)
                assert owner(args, output=output) is output
            check_semantic(kind, output, args)
            np.testing.assert_array_equal(tail[17 * 11:], 123)
            for value, snapshot in zip(args, snapshots, strict=True):
                np.testing.assert_array_equal(value, snapshot)
            for previous, saved in kept:
                np.testing.assert_array_equal(previous, saved)
            kept.append((output, output.copy()))
        assert owner.stats()["allocations"] == before["allocations"]
        assert owner.stats()["capacity"] == before["capacity"]


@pytest.mark.parametrize("storage", ["f16", "bf16"])
@pytest.mark.parametrize("reordered", [False, True])
@pytest.mark.parametrize("intermediate", ["f16", "f32"])
def test_full_pipeline_inferred_mlp_layout_executes(storage, reordered, intermediate, monkeypatch):
    assert runtime._nvidia_device_name() == "sm_120"
    frame, images = full_native_program(mlp_source(storage, reordered, intermediate))
    assert len(frame["stages"]) == len(images) == 3
    with PreparedNativeTensorFrame(frame, images) as owner:
        before = owner.stats()
        for seed in (15004, 15005):
            args = mlp_roots(storage, reordered=reordered, seed=seed, intermediate=intermediate)
            snapshots = tuple(value.copy() for value in args)
            with monkeypatch.context() as guard:
                guard.setattr(subprocess, "run", forbidden)
                guard.setattr(subprocess, "Popen", forbidden)
                actual = owner(args)
            assert_mlp_numeric(actual, args, reordered, intermediate)
            for value, snapshot in zip(args, snapshots, strict=True):
                np.testing.assert_array_equal(value, snapshot)
        assert owner.stats()["allocations"] == before["allocations"] == 1
        assert owner.stats()["capacity"] == before["capacity"]
