"""Correct two-launch matmul→softmax composition on NVIDIA sm_120."""

from __future__ import annotations

import numpy as np
import pytest

from tests._support.nvidia import require_nvidia_mma_runtime


def _artifact():
    from tessera import runtime as rt
    return rt.RuntimeArtifact(metadata={
        "target": "nvidia_sm120",
        "compiler_path": "nvidia_matmul_softmax_compiled",
        "executable": True,
        "execution_kind": "native_gpu",
        "arg_names": ["a", "b"],
        "output_name": "probabilities",
        "ops": [{
            "op_name": "tessera.fused_epilogue",
            "result": "probabilities",
            "operands": ["a", "b"],
            "kwargs": {"activation": "softmax",
                       "composition": "matmul_softmax"},
        }],
    })


def _softmax(x: np.ndarray) -> np.ndarray:
    e = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return e / np.sum(e, axis=-1, keepdims=True)


@pytest.mark.slow
@pytest.mark.hardware_nvidia
@pytest.mark.parametrize("m,n,k", [(16, 8, 16), (39, 27, 31), (5, 300, 17)])
def test_live_nvidia_matmul_softmax_matches_numpy(m, n, k):
    rt = require_nvidia_mma_runtime()
    rng = np.random.default_rng(500 + m + n + k)
    a = (rng.standard_normal((m, k)) * .2).astype(np.float16)
    b = (rng.standard_normal((k, n)) * .2).astype(np.float16)
    result = rt.launch(_artifact(), (a, b))
    assert result["ok"] is True, result.get("reason")
    assert result["compiler_path"] == "nvidia_matmul_softmax_compiled"
    ref = _softmax(a.astype(np.float32) @ b.astype(np.float32))
    out = result["output"]
    np.testing.assert_allclose(out, ref, rtol=2e-5, atol=2e-6)
    np.testing.assert_allclose(out.sum(axis=-1), np.ones(m), atol=2e-6)


def _round_tf32(x: np.ndarray) -> np.ndarray:
    u = x.astype(np.float32).view(np.uint32).astype(np.uint64)
    r = (u >> 13) & 1
    u = (u + 0xFFF + r) & ~np.uint64(0x1FFF)
    return u.astype(np.uint32).view(np.float32)


@pytest.mark.slow
@pytest.mark.hardware_nvidia
@pytest.mark.parametrize(
    "storage,atol", [("tf32", 2e-4), ("e4m3", 2e-2), ("e5m2", 4e-2)])
def test_live_nvidia_matmul_softmax_tf32_fp8_breadth(storage, atol):
    rt = require_nvidia_mma_runtime()
    rng = np.random.default_rng(1500 + len(storage))
    a = (rng.standard_normal((17, 31)) * .15).astype(np.float32)
    b = (rng.standard_normal((31, 13)) * .15).astype(np.float32)
    if storage == "tf32":
        qa, qb = _round_tf32(a), _round_tf32(b)
    else:
        ml_dtypes = pytest.importorskip("ml_dtypes")
        dtype = (ml_dtypes.float8_e4m3fn if storage == "e4m3"
                 else ml_dtypes.float8_e5m2)
        qa, qb = a.astype(dtype), b.astype(dtype)
    result = rt.launch(_artifact(), (qa, qb))
    assert result["ok"] is True, result.get("reason")
    expected = _softmax(qa.astype(np.float32) @ qb.astype(np.float32))
    np.testing.assert_allclose(result["output"], expected, rtol=0, atol=atol)
