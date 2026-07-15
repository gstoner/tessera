"""Exact-device NVIDIA RoPE and ALiBi execute/compare proofs."""
from __future__ import annotations

import numpy as np
import pytest

from tests._support.nvidia import nvidia_mma_runtime_available


def _artifact(rt, op, operands, args, kwargs):
    return rt.RuntimeArtifact(metadata={"target": "nvidia_sm120", "compiler_path": "nvidia_posenc_compiled", "executable": True, "execution_kind": "native_gpu", "arg_names": args, "output_name": "o", "ops": [{"op_name": op, "result": "o", "operands": operands, "kwargs": kwargs}]})


@pytest.mark.skipif(not nvidia_mma_runtime_available(), reason="requires nvcc and a live NVIDIA GPU")
@pytest.mark.hardware_nvidia
def test_rope_runtime_matches_pairwise_oracle():
    from tessera import runtime as rt
    rng = np.random.default_rng(11); x = rng.standard_normal((3, 5, 32)).astype(np.float32); theta = rng.uniform(-2, 2, x.shape).astype(np.float32)
    output = rt.launch(_artifact(rt, "tessera.rope", ["x", "theta"], ["x", "theta"], {}), (x, theta))["output"]
    reference = np.empty_like(x); even, odd, angle = x[..., 0::2], x[..., 1::2], theta[..., 0::2]; reference[..., 0::2] = even * np.cos(angle) - odd * np.sin(angle); reference[..., 1::2] = even * np.sin(angle) + odd * np.cos(angle)
    np.testing.assert_allclose(output, reference, rtol=2e-6, atol=2e-6)


@pytest.mark.skipif(not nvidia_mma_runtime_available(), reason="requires nvcc and a live NVIDIA GPU")
@pytest.mark.hardware_nvidia
def test_alibi_default_and_explicit_slopes():
    from tessera import runtime as rt
    heads, seq = 4, 17
    for slopes in (None, np.array([.5, .25, .125, .0625], np.float32)):
        operands = ["slopes"] if slopes is not None else []; artifact = _artifact(rt, "tessera.alibi", operands, operands, {"num_heads": heads, "seq_len": seq}); output = rt.launch(artifact, (slopes,) if slopes is not None else tuple())["output"]
        scale = 2.0 ** (-8 * np.arange(1, heads + 1, dtype=np.float32) / heads) if slopes is None else slopes; position = np.arange(seq, dtype=np.float32); reference = scale[:, None, None] * (position[None, :] - position[:, None])[None]
        np.testing.assert_allclose(output, reference, rtol=1e-6, atol=1e-6)
