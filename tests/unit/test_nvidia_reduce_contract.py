"""Host-free malformed-contract coverage for the generated CUDA reduction ABI."""
from __future__ import annotations

import numpy as np
import pytest

from tessera.compiler.emit.nvidia_cuda import run_row_reduce


@pytest.mark.parametrize("x", [np.empty((0, 4), np.float32),
                             np.empty((2, 0), np.float16),
                             np.ones((2, 3, 4), np.float32),
                             np.ones((2, 3), np.int8)])
def test_row_reduce_rejects_invalid_storage_or_shape_before_launch(x):
    with pytest.raises(ValueError):
        run_row_reduce(x, "sum")


def test_row_reduce_rejects_unknown_operation_before_launch():
    with pytest.raises(ValueError, match="unknown NVIDIA reduction kind"):
        run_row_reduce(np.ones((2, 3), np.float32), "product")
