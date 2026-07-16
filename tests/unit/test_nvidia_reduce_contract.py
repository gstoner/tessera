"""Host-free NVIDIA reduction rejection contract."""
from __future__ import annotations

import numpy as np
import pytest


def _artifact():
    from tessera import runtime as rt
    return rt.RuntimeArtifact(metadata={"target": "nvidia_sm120", "compiler_path": "nvidia_reduce_compiled", "executable": True, "execution_kind": "native_gpu", "arg_names": ["x"], "output_name": "o", "ops": [{"op_name": "tessera.sum", "result": "o", "operands": ["x"], "kwargs": {"axis": -1}}]})


def test_nvidia_reduce_rejects_non_float_without_gpu():
    from tessera import runtime as rt
    with pytest.raises(ValueError, match="f32/f16"):
        rt._execute_nvidia_compiled_reduce(_artifact(), (np.zeros((2, 8), np.int32),))
