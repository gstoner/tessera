"""Host-free NVIDIA local-collective rejection contract."""
from __future__ import annotations

import numpy as np
import pytest


def _art(rt, world=1):
    return rt.RuntimeArtifact(metadata={"target": "nvidia_sm120", "compiler_path": "nvidia_local_collective_compiled", "executable": True, "execution_kind": "native_gpu", "arg_names": ["x"], "output_name": "o", "ops": [{"op_name": "tessera.all_reduce", "result": "o", "operands": ["x"], "kwargs": {"world_size": world}}]})


def test_multi_rank_collective_is_explicitly_deferred():
    from tessera import runtime as rt
    with pytest.raises(ValueError, match="requires world_size=1"):
        rt._execute_nvidia_local_collective(_art(rt, 2), (np.ones(4, np.float32),))
