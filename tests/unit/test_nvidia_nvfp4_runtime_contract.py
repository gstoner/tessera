"""Host-free contract checks for the general-shape SM120 NVFP4 runtime ABI."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from tessera import runtime as rt


ROOT = Path(__file__).resolve().parents[2]
SOURCE = (ROOT / "src/compiler/codegen/tessera_gpu_backend_NVIDIA/runtime/cuda"
          / "tessera_nvidia_gemm.cpp")


def test_nvfp4_shipped_source_has_general_shape_k64_dispatch():
    source = SOURCE.read_text()
    assert "tessera_nvidia_mma_gemm_nvfp4" in source
    assert "k0<K;k0+=64" in source
    assert "kind::mxf4nvf4.block_scale.scale_vec::4X" in source
    assert "SFa[M,ceil(K/16)]" in source
    assert "SFb[ceil(K/16),N]" in source
    assert "if(rr<M&&cc<N)D[rr*N+cc]=v" in source


@pytest.mark.parametrize(
    "field,value,match",
    [
        ("a", np.zeros((17, 32), np.uint8), "A_packed"),
        ("b", np.zeros((32, 9), np.uint8), "B_packed"),
        ("sa", np.zeros((17, 4), np.uint8), "scale_a"),
        ("sb", np.zeros((4, 9), np.uint8), "scale_b"),
    ],
)
def test_nvfp4_dispatch_rejects_malformed_views_before_loading_cuda(
        monkeypatch, field, value, match):
    values = {
        "a": np.zeros((17, 33), np.uint8),
        "b": np.zeros((33, 9), np.uint8),
        "sa": np.zeros((17, 5), np.uint8),
        "sb": np.zeros((5, 9), np.uint8),
    }
    values[field] = value
    monkeypatch.setattr(rt, "_load_nvidia_gemm_runtime",
                        lambda: pytest.fail("CUDA library loaded before validation"))
    with pytest.raises(ValueError, match=match):
        rt._nvidia_nvfp4_gemm_2d(
            values["a"], values["b"], values["sa"], values["sb"], 17, 9, 65)


@pytest.mark.parametrize("dims", [(0, 9, 65), (17, -1, 65), (17, 9, 0)])
def test_nvfp4_dispatch_rejects_nonpositive_dimensions(dims):
    with pytest.raises(ValueError, match="dimensions must be positive"):
        rt._nvidia_nvfp4_gemm_2d(None, None, None, None, *dims)


@pytest.mark.parametrize("policy", ["independent_rhs", "shared_rhs_rows", "shared_lhs"])
@pytest.mark.parametrize("field,value", [("M", 50), ("BatchRows", 16),
                                        ("BatchCount", 2), ("BatchCount", True)])
def test_native_batch_scalars_reject_shape_reinterpretation(policy, field, value):
    from types import SimpleNamespace
    descriptor = SimpleNamespace(provenance={
        "batching": policy, "batch_rows": [3, 17], "shape": [51, 19, 129]})
    scalars = {"M": 51, "N": 19, "K": 129, "BatchRows": 17, "BatchCount": 3}
    assert rt._validate_nvfp4_independent_batch_scalars(descriptor, scalars) == (3, 17)
    scalars[field] = value
    with pytest.raises(RuntimeError, match="scalars differ"):
        rt._validate_nvfp4_independent_batch_scalars(descriptor, scalars)
