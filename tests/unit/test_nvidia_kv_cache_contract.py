"""Host-free NVIDIA paged-KV validation contracts."""
from __future__ import annotations

import numpy as np
import pytest


def test_nvidia_paged_kv_read_rejects_invalid_page_table_before_cuda():
    from tessera.compiler.emit.nvidia_cuda import run_paged_kv_cache_read_f32
    pages = np.zeros((2, 4, 2, 8), np.float32)
    with pytest.raises(ValueError, match="invalid physical page"):
        run_paged_kv_cache_read_f32(pages, np.array([0, 2]), 0, 4)


def test_nvidia_paged_kv_read_rejects_bounds_before_cuda():
    from tessera.compiler.emit.nvidia_cuda import run_paged_kv_cache_read_f32
    pages = np.zeros((2, 4, 2, 8), np.float32)
    with pytest.raises(ValueError, match="bounds"):
        run_paged_kv_cache_read_f32(pages, np.array([0, 1]), 7, 9)
