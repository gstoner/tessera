"""Exact-device NVIDIA supported fused-epilogue execution matrix."""

from __future__ import annotations

import itertools

import numpy as np
import pytest

from tessera.compiler.emit import candidate as candidates
from tessera.compiler.emit.candidate import OP_FUSED_REGION
from tessera.compiler.emit.nvidia_cuda import nvidia_epilogue_execution_contract
from tessera.compiler.fusion import FusedRegion
from tests._support.nvidia import require_nvidia_mma_runtime


STORAGES = ("f32", "f16", "bf16", "fp8_e4m3", "fp8_e5m2")
ACTIVATIONS = (None, "relu", "gelu", "silu")
CASES = [
    case for case in itertools.product(
        STORAGES, (False, True), ACTIVATIONS, (False, True))
    if (case[1] or case[2] is not None or case[3]) and
    (not case[3] or case[0] == "f32")
]


@pytest.mark.slow
@pytest.mark.hardware_nvidia
@pytest.mark.parametrize(
    "storage,bias,activation,residual", CASES,
    ids=lambda value: str(value),
)
def test_live_nvidia_supported_epilogue_matrix(
        storage, bias, activation, residual):
    require_nvidia_mma_runtime()
    epilogue = (("bias",) if bias else ()) + (
        (activation,) if activation is not None else ())
    region = FusedRegion(epilogue=epilogue, residual=residual,
                         storage_dtype=storage)
    contract = nvidia_epilogue_execution_contract(region)
    candidate = next(
        item for item in candidates.candidates_for("nvidia", OP_FUSED_REGION)
        if item.name == contract["candidate"])
    rng = np.random.default_rng(
        120 + STORAGES.index(storage) * 31 + int(bias) * 7 +
        ACTIVATIONS.index(activation) * 3 + int(residual))
    a = (rng.standard_normal((19, 29)) * 0.15).astype(np.float32)
    b = (rng.standard_normal((29, 23)) * 0.15).astype(np.float32)
    bias_value = ((rng.standard_normal(23) * 0.05).astype(np.float32)
                  if bias else None)
    residual_value = ((rng.standard_normal((19, 23)) * 0.05).astype(np.float32)
                      if residual else None)
    actual, tag = candidate.run(
        region, a, b, bias_value, residual=residual_value)
    assert tag in {"nvidia_cuda", "nvidia_cuda_composed"}
    expected = region.reference(a, b, bias_value, residual_value)
    atol = {
        "f32": 2e-5, "f16": 5e-3, "bf16": 5e-2,
        "fp8_e4m3": 2e-1, "fp8_e5m2": 4e-1,
    }[storage]
    np.testing.assert_allclose(actual, expected, atol=atol, rtol=atol)
