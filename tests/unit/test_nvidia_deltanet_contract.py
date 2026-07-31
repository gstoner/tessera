"""Host-free NVIDIA DeltaNet rejection contract."""
from __future__ import annotations

import numpy as np
import pytest


def test_deltanet_rejects_noncausal_without_cuda():
    from tessera import runtime as rt

    z = np.zeros((1, 1, 2, 4), np.float32)
    artifact = rt.RuntimeArtifact(metadata={"target": "nvidia_sm120", "compiler_path": "nvidia_deltanet_compiled", "executable": True, "execution_kind": "native_gpu", "arg_names": ["x0", "x1", "x2"], "output_name": "o", "ops": [{"op_name": "tessera.gated_deltanet", "result": "o", "operands": ["x0", "x1", "x2"], "kwargs": {"causal": False}}]})
    with pytest.raises(ValueError, match="causal-only"):
        rt._execute_nvidia_deltanet_compiled(artifact, (z, z, z))


def test_deltanet_backward_rejects_unvalidated_head_dimensions_without_cuda():
    """The correctness-first v2 package must fail before a D>8 CUDA launch."""
    from tessera import runtime as rt

    q = np.zeros((1, 1, 2, 9), np.float32)
    v = np.zeros((1, 1, 2, 8), np.float32)
    artifact = rt.RuntimeArtifact(metadata={
        "target": "nvidia_sm120", "compiler_path": "nvidia_deltanet_bwd_compiled",
        "executable": True, "execution_kind": "native_gpu",
        "arg_names": ["q", "k", "v", "dy"], "out_cotangents": ["dy"],
        "ops": [{"op_name": "tessera.gated_deltanet", "result": "o",
                 "operands": ["q", "k", "v"], "kwargs": {"causal": True}}],
    })
    with pytest.raises(ValueError, match="Dqk and Dv <= 8"):
        rt._execute_nvidia_deltanet_backward(artifact, (q, q, v, v))
