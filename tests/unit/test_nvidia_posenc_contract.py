"""Host-free NVIDIA positional-encoding rejection contract."""
from __future__ import annotations

import numpy as np
import pytest


def test_posenc_rejects_invalid_contract_without_cuda():
    from tessera.compiler.emit import nvidia_cuda as nv
    with pytest.raises(ValueError, match="even final"):
        nv.run_rope_f32(np.zeros((2, 7), np.float32), np.zeros((2, 7), np.float32))
    with pytest.raises(ValueError, match="length 3"):
        nv.run_alibi_f32(3, 4, np.ones(2, np.float32))
