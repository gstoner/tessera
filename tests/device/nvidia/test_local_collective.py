from __future__ import annotations

import numpy as np
import pytest

from tests._support.nvidia import nvidia_mma_runtime_available


def _art(rt, op):
    return rt.RuntimeArtifact(metadata={"target": "nvidia_sm120", "compiler_path": "nvidia_local_collective_compiled", "executable": True, "execution_kind": "native_gpu", "arg_names": ["x"], "output_name": "o", "ops": [{"op_name": op, "result": "o", "operands": ["x"], "kwargs": {"world_size": 1}}]})


@pytest.mark.skipif(not nvidia_mma_runtime_available(), reason="requires nvcc and NVIDIA GPU")
@pytest.mark.hardware_nvidia
@pytest.mark.parametrize("op", ["tessera.all_reduce", "tessera.reduce_scatter", "tessera.all_gather", "tessera.all_to_all"])
def test_single_device_collective_identity_runs_on_cuda(op):
    from tessera import runtime as rt
    x = np.arange(60, dtype=np.float32).reshape(3, 4, 5)
    result = rt.launch(_art(rt, op), (x,))
    assert result["execution_kind"] == "native_gpu"
    np.testing.assert_array_equal(result["output"], x)
