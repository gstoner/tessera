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
