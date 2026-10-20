"""Owning source -> native Target package -> C++ owner execution on SM120."""
import ctypes as ct
import os
import subprocess

import ml_dtypes
import numpy as np
import pytest

import tessera as ts
from tessera import runtime as rt

pytestmark = pytest.mark.hardware_nvidia


def relu(a, b):
    return ts.ops.matmul(ts.ops.relu(a), b, output_dtype="fp32")


def norm(a, b):
    return ts.ops.matmul(ts.ops.rmsnorm(a, eps=1e-5, numeric_policy={"accum": "fp32"}),
                         b, output_dtype="fp32")


def two_sided(a, b):
    return ts.ops.matmul(ts.ops.relu(a), ts.ops.softmax(b, axis=-1), output_dtype="fp32")


def reverse(b, a):
    return ts.ops.matmul(ts.ops.relu(a), b, output_dtype="fp32")


FUNCTIONS = {"relu": relu, "norm": norm, "two_sided": two_sided, "reverse": reverse}
CASES = [(kind, storage) for kind in FUNCTIONS for storage in ("f16", "bf16")]


def values(storage, seed=None):
    dtype = np.float16 if storage == "f16" else ml_dtypes.bfloat16
    if seed is None:
        return np.full((17, 32), -.5, dtype), np.ones((32, 11), dtype)
    rng = np.random.default_rng(seed)
    return rng.normal(0, .2, (17, 32)).astype(dtype), rng.normal(0, .2, (32, 11)).astype(dtype)


def roots(kind, a, b):
    return (b, a) if kind == "reverse" else (a, b)


def oracle(kind, a, b):
    wide = a.astype(np.float64)
    lhs = np.maximum(wide, 0).astype(a.dtype)
    rhs = b
    if kind == "norm":
        lhs = (wide / np.sqrt(np.mean(wide * wide, axis=-1, keepdims=True)
                             + float(np.float32(1e-5)))).astype(a.dtype)
    if kind == "two_sided":
        wide = b.astype(np.float64)
        wide -= wide.max(axis=-1, keepdims=True)
        rhs = (np.exp(wide) / np.exp(wide).sum(axis=-1, keepdims=True)).astype(b.dtype)
    return lhs.astype(np.float64) @ rhs.astype(np.float64)


def compile_case(kind, storage):
    a, b = values(storage)
    fn = ts.jit(target="nvidia_sm120")(FUNCTIONS[kind])
    return fn.compile_native_tensor_program(*roots(kind, a, b))


@pytest.fixture(scope="module")
def provider():
    if rt._nvidia_device_name() != "sm_120":
        pytest.skip("requires owning RTX5070/SM120 proof")
    path = os.environ.get("TESSERA_NATIVE_FRAME_TEST_LIB")
    lib = ct.CDLL(path) if path else rt._load_nvidia_ptx_launch()
    assert lib is not None and hasattr(lib, "tessera_nvidia_native_frame_prepare_v1")
    return lib


@pytest.mark.parametrize("kind,storage", CASES)
def test_original_frontend_package_executes_with_tools_forbidden(kind, storage, provider, monkeypatch):
    program = compile_case(kind, storage)
    assert program is not None
    if kind == "norm":
        assert 'accum = "fp32"' in program.source_ir
    if kind == "reverse":
        assert program.frame["stages"][0]["inputs"] == [1]
        assert program.frame["stages"][-1]["inputs"] == [2, 0]
    before = program.frame
    def forbidden(*_, **__):
        pytest.fail("native package binding/replay started an external compiler")
    with monkeypatch.context() as guarded:
        guarded.setattr(subprocess, "run", forbidden)
        guarded.setattr(subprocess, "Popen", forbidden)
        with program.prepare(provider=provider) as owner:
            initial = owner.stats()
            assert initial["allocations"] == 1
            kept = []
            for seed in (None, 71, 72):
                a, b = values(storage, seed)
                snapshots = a.copy(), b.copy()
                tail = np.full(17 * 11 + 16, 123, np.float32)
                output = tail[:17 * 11].reshape(17, 11)
                assert owner(roots(kind, a, b), output=output) is output
                np.testing.assert_allclose(output, oracle(kind, a, b), rtol=.004, atol=2e-5)
                np.testing.assert_array_equal(tail[17 * 11:], 123)
                np.testing.assert_array_equal(a, snapshots[0])
                np.testing.assert_array_equal(b, snapshots[1])
                kept.append((output, output.copy()))
            timings = owner.profile(16, captured=True)
            assert timings["program_ms"] > 0
            assert len(timings["stage_ms"]) == len(program.members)
            assert all(value > 0 for value in timings["stage_ms"])
            assert owner.stats()["allocations"] == initial["allocations"] == 1
            assert owner.stats()["capacity"] == initial["capacity"]
            for output, saved in kept:
                np.testing.assert_array_equal(output, saved)
    assert program.frame == before
