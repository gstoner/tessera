from __future__ import annotations

import numpy as np

import tessera as ts


def test_cpu_matmul_reports_native_cpu_execution_kind():
    @ts.jit
    def mm(A, B):
        return ts.ops.matmul(A, B)

    A = np.arange(6, dtype=np.float32).reshape(2, 3)
    B = np.arange(12, dtype=np.float32).reshape(3, 4)

    np.testing.assert_allclose(mm(A, B), A @ B)
    assert mm.execution_kind == "native_cpu"
    assert mm.is_executable
    assert mm.is_native_execution
    assert not mm.is_reference_execution
    assert "uses_compiled_path" not in dir(mm)
    assert mm.runtime_artifact().metadata["execution_kind"] == "native_cpu"
    assert mm.runtime_artifact().metadata["compiler_path"] == "jit_cpu_numpy"


def test_cpu_non_native_plan_reports_reference_cpu():
    @ts.jit
    def relu(x):
        return ts.ops.relu(x)

    assert relu.execution_kind == "reference_cpu"
    assert relu.is_executable
    assert relu.is_reference_execution
    assert not relu.is_native_execution
    assert relu.runtime_artifact().metadata["execution_kind"] == "reference_cpu"


def test_uninspectable_function_reports_fallback_eager():
    ns = {}
    exec("def dynamic_relu(x):\n    return ts.ops.relu(x)\n", {"ts": ts}, ns)

    device_verified_jit = ts.jit(ns["dynamic_relu"])

    assert device_verified_jit.execution_kind == "fallback_eager"
    assert not device_verified_jit.is_executable
    assert device_verified_jit.runtime_artifact().metadata["execution_kind"] == "fallback_eager"


def test_target_execution_kinds_cover_native_and_artifact_modes():
    @ts.jit(target="apple_cpu")
    def apple_cpu_mm(A, B):
        return ts.ops.matmul(A, B)

    @ts.jit(target="apple_gpu")
    def apple_gpu_mm(A, B):
        return ts.ops.matmul(A, B)

    @ts.jit(target="nvidia")
    def nvidia_mm(A, B):
        return ts.ops.matmul(A, B)

    assert apple_cpu_mm.execution_kind == "native_cpu"
    assert apple_cpu_mm.is_native_execution
    assert apple_gpu_mm.execution_kind == "native_gpu"
    assert apple_gpu_mm.is_native_execution
    assert nvidia_mm.execution_kind == "artifact_only"
    assert not nvidia_mm.is_executable
