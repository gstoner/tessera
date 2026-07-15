"""Host-free NVIDIA FlashAttention rejection contract."""
from __future__ import annotations

import numpy as np
import pytest


def _artifact():
    from tessera import runtime as rt
    return rt.RuntimeArtifact(metadata={"target": "nvidia_sm120", "compiler_path": "nvidia_flash_attn_compiled", "executable": True, "execution_kind": "native_gpu", "arg_names": ["q", "k", "v"], "output_name": "o", "ops": [{"op_name": "tessera.flash_attn", "result": "o", "operands": ["q", "k", "v"], "kwargs": {}}]})


def test_nvidia_flash_attn_rejects_non_floating_storage_without_gpu():
    from tessera import runtime as rt
    q = np.zeros((1, 1, 2, 4), np.int32)
    with pytest.raises(ValueError, match="f32 or f16"):
        rt._execute_nvidia_flash_attn_compiled(_artifact(), (q, q, q))
