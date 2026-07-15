"""Host-free NVIDIA matmul-softmax validation contract."""
from __future__ import annotations

import numpy as np
import pytest


def test_nvidia_matmul_softmax_requires_softmax_contract_before_cuda():
    from tessera import runtime as rt
    art = rt.RuntimeArtifact(metadata={"target": "nvidia_sm120", "compiler_path": "nvidia_matmul_softmax_compiled", "executable": True, "execution_kind": "native_gpu", "arg_names": ["a", "b"], "output_name": "o", "ops": [{"op_name": "tessera.fused_epilogue", "result": "o", "operands": ["a", "b"], "kwargs": {"activation": "relu", "composition": "matmul_softmax"}}]})
    with pytest.raises(ValueError, match="activation=softmax"):
        rt._execute_nvidia_matmul_softmax_compiled(art, (np.zeros((2, 3), np.float16), np.zeros((3, 4), np.float16)))
