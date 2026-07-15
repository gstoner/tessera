"""Exact-device NVIDIA DeltaNet execute/compare proofs."""
from __future__ import annotations

import numpy as np
import pytest

from tests._support.nvidia import nvidia_mma_runtime_available


def _ref(q, k, v, gate=None, beta=None, decay=None, erase=False, modified=False):
    b, h, s, d = q.shape; dv = v.shape[-1]; state = np.zeros((b, h, d, dv), np.float64); output = np.zeros((b, h, s, dv), np.float64)
    q, k, v = q.astype(np.float64), k.astype(np.float64), v.astype(np.float64)
    for token in range(s):
        key, target = k[:, :, token], v[:, :, token]
        if erase: target = target - (decay[:, :, token, None] if decay is not None else 1) * np.einsum("bhd,bhde->bhe", key, state)
        if decay is not None: state = decay[:, :, token, None, None] * state
        delta = np.einsum("bhd,bhe->bhde", key, target)
        if modified: delta /= 1 + np.linalg.norm(delta, axis=(-2, -1), keepdims=True)
        state += (beta[:, :, token, None, None] if beta is not None else 1) * delta; output[:, :, token] = np.einsum("bhd,bhde->bhe", q[:, :, token], state)
    return (output * (1 / (1 + np.exp(-gate)) if gate is not None else 1)).astype(np.float32)


@pytest.mark.skipif(not nvidia_mma_runtime_available(), reason="requires nvcc and NVIDIA GPU")
@pytest.mark.hardware_nvidia
@pytest.mark.parametrize("variant", ["plain", "kimi", "gated", "erase", "modified"])
def test_deltanet_variants_match_oracle(variant):
    from tessera import runtime as rt
    rng = np.random.default_rng(70 + len(variant)); shape = (1, 2, 9, 16); q = (rng.standard_normal(shape) * .4).astype(np.float32); k = rng.standard_normal(shape); k = (k / np.maximum(np.linalg.norm(k, axis=-1, keepdims=True), 1e-6)).astype(np.float32); v = (rng.standard_normal(shape) * .4).astype(np.float32)
    values, kwargs, options = [q, k, v], {"causal": True}, {}
    name = "tessera.modified_delta_attention" if variant == "modified" else "tessera.kimi_delta_attention" if variant == "kimi" else "tessera.gated_deltanet"
    if variant == "gated":
        gate = rng.standard_normal(shape).astype(np.float32); beta = rng.uniform(.2, .9, shape[:3]).astype(np.float32); decay = rng.uniform(.85, .99, shape[:3]).astype(np.float32); values += [gate, beta, decay]; kwargs.update(has_gate=True, has_beta=True, has_decay=True); options.update(gate=gate, beta=beta, decay=decay)
    if variant == "erase":
        beta = rng.uniform(.2, .9, shape[:3]).astype(np.float32); decay = rng.uniform(.85, .99, shape[:3]).astype(np.float32); values += [beta, decay]; kwargs.update(erase=True, has_beta=True, has_decay=True); options.update(beta=beta, decay=decay, erase=True)
    if variant == "modified": options["modified"] = True
    names = [f"x{i}" for i in range(len(values))]; artifact = rt.RuntimeArtifact(metadata={"target": "nvidia_sm120", "compiler_path": "nvidia_deltanet_compiled", "executable": True, "execution_kind": "native_gpu", "arg_names": names, "output_name": "o", "ops": [{"op_name": name, "result": "o", "operands": names, "kwargs": kwargs}]})
    result = rt.launch(artifact, tuple(values)); assert result["ok"] is True, result.get("reason")
    np.testing.assert_allclose(result["output"], _ref(q, k, v, **options), rtol=3e-5, atol=3e-5)
