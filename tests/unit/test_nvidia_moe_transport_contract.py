"""Host-free contract retained from the NVIDIA MoE transport device family."""
from __future__ import annotations

import numpy as np
import pytest


def test_grouped_gemm_rejects_bad_partition_without_cuda():
    from tessera.compiler.emit.nvidia_cuda import run_grouped_gemm_f32

    with pytest.raises(ValueError, match="K/group_sizes"):
        run_grouped_gemm_f32(
            np.zeros((4, 3), np.float32),
            np.zeros((2, 3, 2), np.float32),
            np.array([1, 1]),
        )
