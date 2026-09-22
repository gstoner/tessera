"""Typed frontend and route-receipt guards for folded gfx1201 matmul."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from tessera.compiler import rocm_mxfp4 as mx
from tessera.compiler.rocm_mxfp4_folded import prepare_folded_weights
from tessera.compiler.rocm_mxfp4_folded_frontend import (
    author_folded_scaled_matmul_graph,
    compile_folded_scaled_matmul,
)


def _inputs() -> tuple[np.ndarray, np.ndarray, object]:
    a = np.full((65, 64), 0x38, dtype=np.uint8)
    a_scale = np.ones(65, dtype=np.float32)
    codes = np.ones((48, 64), dtype=np.uint8)
    scales = np.full((2, 48), 127, dtype=np.uint8)
    folded = prepare_folded_weights(
        mx.pack_e2m1_codes(codes), scales, allow_approximate=True,
    )
    return a, a_scale, folded


def test_authored_graph_uses_shapes_and_explicit_folded_contract() -> None:
    graph = author_folded_scaled_matmul_graph(256, 80, 128)
    assert "tessera.scaled_matmul" in graph
    assert "tensor<256x128xui8>" in graph
    assert "tensor<80x128xui8>" in graph
    assert "block = [1, 128]" in graph
    assert "folded_row_reference_explicit_approximate" in graph
    with pytest.raises(ValueError, match="K divisible by 64"):
        author_folded_scaled_matmul_graph(256, 80, 96)


def test_frontend_refuses_implicit_policy_and_bad_physical_inputs() -> None:
    a, a_scale, folded = _inputs()
    tool = Path("/does/not/exist/tessera-opt")
    with pytest.raises(ValueError, match="explicit approximate"):
        compile_folded_scaled_matmul(a, a_scale, folded, tessera_opt=tool)
    with pytest.raises(ValueError, match="raw E4M3"):
        compile_folded_scaled_matmul(
            a.astype(np.float32), a_scale, folded, tessera_opt=tool,
            allow_approximate=True,
        )
    with pytest.raises(ValueError, match="fp32"):
        compile_folded_scaled_matmul(
            a, a_scale.astype(np.float16), folded, tessera_opt=tool,
            allow_approximate=True,
        )
    with pytest.raises(ValueError, match="weight K"):
        compile_folded_scaled_matmul(
            np.ascontiguousarray(a[:, :32]), a_scale, folded, tessera_opt=tool,
            allow_approximate=True,
        )
    with pytest.raises(FileNotFoundError, match="compiler not found"):
        compile_folded_scaled_matmul(
            a, a_scale, folded, tessera_opt=tool, allow_approximate=True,
        )
