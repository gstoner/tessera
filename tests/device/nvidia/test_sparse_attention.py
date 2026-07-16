"""Exact-device NVIDIA DSA sparse-attention execution proofs."""
from __future__ import annotations

import numpy as np
import pytest

from tests._support.nvidia import require_nvidia_mma_runtime


def _artifact(**kwargs):
    from tessera import runtime as rt

    return rt.RuntimeArtifact(metadata={
        "target": "nvidia_sm120", "compiler_path": "nvidia_sparse_attn_compiled",
        "executable": True, "execution_kind": "native_gpu",
        "arg_names": ["q", "k", "v"], "output_name": "o",
        "ops": [{"op_name": "tessera.dsa_block_sparse_attention", "result": "o",
                 "operands": ["q", "k", "v"], "kwargs": kwargs}],
    })


@pytest.mark.slow
@pytest.mark.hardware_nvidia
def test_live_nvidia_dsa_sparse_attention():
    from tessera.stdlib.attention import dsa_block_sparse_attention

    rng = np.random.default_rng(971)
    q = rng.standard_normal((1, 2, 5, 4), dtype=np.float32)
    k = rng.standard_normal((1, 1, 7, 4), dtype=np.float32)
    v = rng.standard_normal((1, 1, 7, 3), dtype=np.float32)
    kwargs = {"block_size": 2, "top_k_blocks": 2, "causal": True, "scale": .5}
    result = require_nvidia_mma_runtime().launch(_artifact(**kwargs), (q, k, v))
    assert result["ok"], result.get("reason")
    np.testing.assert_allclose(
        result["output"], dsa_block_sparse_attention(q, k, v, **kwargs), atol=8e-5, rtol=0,
    )


@pytest.mark.slow
@pytest.mark.hardware_nvidia
def test_live_nvidia_dsa_incremental_decode_uses_global_q_position():
    from tessera.stdlib.attention import dsa_block_sparse_attention

    rng = np.random.default_rng(977)
    q = rng.standard_normal((1, 2, 1, 4), dtype=np.float32)
    k = rng.standard_normal((1, 1, 7, 4), dtype=np.float32)
    v = rng.standard_normal((1, 1, 7, 3), dtype=np.float32)
    kwargs = {"block_size": 2, "top_k_blocks": 4, "causal": True,
              "scale": .5, "q_positions": [5]}
    result = require_nvidia_mma_runtime().launch(_artifact(**kwargs), (q, k, v))
    assert result["ok"], result.get("reason")
    np.testing.assert_allclose(
        result["output"], dsa_block_sparse_attention(q, k, v, **kwargs), atol=8e-5, rtol=0,
    )
