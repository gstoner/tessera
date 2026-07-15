"""Canonical Tile accumulator ReLU epilogue on NVIDIA sm_120."""

from __future__ import annotations

import os
import shutil

import numpy as np
import pytest


def _cuda_or_skip():
    if not (shutil.which("nvcc") or os.path.exists("/usr/local/cuda/bin/nvcc")):
        pytest.skip("nvcc not installed")
    from tessera import runtime as rt
    if not rt._nvidia_mma_runtime_available():
        pytest.skip("no usable NVIDIA CUDA device")
    return rt


def _artifact(*, bias: bool = False, storage: str = "f16"):
    from tessera import runtime as rt
    operands = ["a", "b"] + (["bias"] if bias else [])
    return rt.RuntimeArtifact(metadata={
        "target": "nvidia_sm120",
        "compiler_path": "nvidia_matmul_relu_compiled",
        "executable": True,
        "execution_kind": "native_gpu",
        "arg_names": operands,
        "output_name": "o",
        "ops": [{
            "op_name": "tessera.fused_epilogue",
            "result": "o",
            "operands": operands,
            "kwargs": {"storage_dtype": storage, "activation": "relu"},
        }],
    })


@pytest.mark.slow
@pytest.mark.hardware_nvidia
@pytest.mark.parametrize("storage,atol", [("f16", 5e-3), ("bf16", 5e-2)])
@pytest.mark.parametrize("m,n,k", [(16, 8, 16), (37, 23, 29)])
@pytest.mark.parametrize("with_bias", [False, True])
def test_live_nvidia_matmul_relu_matches_numpy(
    storage, atol, m, n, k, with_bias
):
    rt = _cuda_or_skip()
    rng = np.random.default_rng(409 + m + n + k + with_bias)
    a = (rng.standard_normal((m, k)) * .25).astype(np.float32)
    b = (rng.standard_normal((k, n)) * .25).astype(np.float32)
    inputs: list[np.ndarray] = [a, b]
    bias = None
    if with_bias:
        bias = (rng.standard_normal(n) * .2).astype(np.float32)
        inputs.append(bias)

    result = rt.launch(_artifact(bias=with_bias, storage=storage), tuple(inputs))
    assert result["ok"] is True, result.get("reason")
    assert result["compiler_path"] == "nvidia_matmul_relu_compiled"
    ref_storage = np.float16
    if storage == "bf16":
        ml_dtypes = pytest.importorskip("ml_dtypes")
        ref_storage = ml_dtypes.bfloat16
    ref = (a.astype(ref_storage).astype(np.float32)
           @ b.astype(ref_storage).astype(np.float32))
    if bias is not None:
        ref += bias
    ref = np.maximum(ref, 0.0)
    np.testing.assert_allclose(result["output"], ref, rtol=0, atol=atol)


def test_nvidia_matmul_relu_rejects_bad_shapes_before_cuda():
    from tessera import runtime as rt
    a = np.zeros((4, 3), np.float32)
    b = np.zeros((2, 5), np.float32)
    with pytest.raises(ValueError, match="compatible rank-2"):
        rt._execute_nvidia_matmul_relu_compiled(_artifact(), (a, b))


def test_nvidia_matmul_relu_rejects_unsupported_storage_before_cuda():
    from tessera import runtime as rt
    a = np.zeros((4, 3), np.float32)
    b = np.zeros((3, 5), np.float32)
    with pytest.raises(ValueError, match="storage_dtype"):
        rt._execute_nvidia_matmul_relu_compiled(
            _artifact(storage="nvfp4"), (a, b))


def _round_tf32(x: np.ndarray) -> np.ndarray:
    u = x.astype(np.float32).view(np.uint32).astype(np.uint64)
    r = (u >> 13) & 1
    u = (u + 0xFFF + r) & ~np.uint64(0x1FFF)
    return u.astype(np.uint32).view(np.float32)


@pytest.mark.slow
@pytest.mark.hardware_nvidia
@pytest.mark.parametrize(
    "storage,atol", [("tf32", 1e-2), ("fp8_e4m3", 1.0),
                     ("fp8_e5m2", 2.0)])
def test_live_nvidia_matmul_relu_tf32_fp8_breadth(storage, atol):
    rt = _cuda_or_skip()
    rng = np.random.default_rng(1400 + len(storage))
    a = (rng.standard_normal((17, 31)) * .3).astype(np.float32)
    b = (rng.standard_normal((31, 9)) * .3).astype(np.float32)
    bias = (rng.standard_normal(9) * .1).astype(np.float32)
    result = rt.launch(_artifact(bias=True, storage=storage), (a, b, bias))
    assert result["ok"] is True, result.get("reason")
    if storage == "tf32":
        qa, qb = _round_tf32(a), _round_tf32(b)
    else:
        ml_dtypes = pytest.importorskip("ml_dtypes")
        dtype = (ml_dtypes.float8_e4m3fn if storage == "fp8_e4m3"
                 else ml_dtypes.float8_e5m2)
        qa, qb = a.astype(dtype), b.astype(dtype)
    expected = np.maximum(qa.astype(np.float32) @ qb.astype(np.float32) + bias, 0)
    np.testing.assert_allclose(result["output"], expected, rtol=0, atol=atol)
