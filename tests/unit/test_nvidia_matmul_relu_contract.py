"""Host-free NVIDIA matmul-ReLU validation contracts."""
from __future__ import annotations

import numpy as np
import pytest


def _artifact(storage="f16"):
    from tessera import runtime as rt
    return rt.RuntimeArtifact(metadata={"target": "nvidia_sm120", "compiler_path": "nvidia_matmul_relu_compiled", "executable": True, "execution_kind": "native_gpu", "arg_names": ["a", "b"], "output_name": "o", "ops": [{"op_name": "tessera.fused_epilogue", "result": "o", "operands": ["a", "b"], "kwargs": {"storage_dtype": storage, "activation": "relu"}}]})


def test_nvidia_matmul_relu_rejects_bad_shapes_before_cuda():
    from tessera import runtime as rt
    with pytest.raises(ValueError, match="compatible rank-2"): rt._execute_nvidia_matmul_relu_compiled(_artifact(), (np.zeros((4, 3), np.float32), np.zeros((2, 5), np.float32)))


def test_nvidia_matmul_relu_rejects_unsupported_storage_before_cuda():
    from tessera import runtime as rt
    with pytest.raises(ValueError, match="storage_dtype"): rt._execute_nvidia_matmul_relu_compiled(_artifact("nvfp4"), (np.zeros((4, 3), np.float32), np.zeros((3, 5), np.float32)))
