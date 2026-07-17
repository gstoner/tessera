"""Host-free totality checks for the NVIDIA fused-epilogue matrix."""

from __future__ import annotations

import itertools

import pytest

from tessera.compiler.emit.nvidia_cuda import nvidia_epilogue_execution_contract
from tessera.compiler.fusion import FusedRegion


STORAGES = ("f32", "f16", "bf16", "fp8_e4m3", "fp8_e5m2")
ACTIVATIONS = (None, "relu", "gelu", "silu")


def _region(storage, bias, activation, residual):
    epilogue = (("bias",) if bias else ()) + (
        (activation,) if activation is not None else ())
    return FusedRegion(epilogue=epilogue, residual=residual,
                       storage_dtype=storage)


@pytest.mark.parametrize(
    "storage,bias,activation,residual",
    [case for case in itertools.product(STORAGES, (False, True), ACTIVATIONS,
                                        (False, True))
     if (case[1] or case[2] is not None or case[3]) and
     (not case[3] or case[0] == "f32")],
)
def test_nvidia_supported_epilogue_matrix_is_total(
        storage, bias, activation, residual):
    contract = nvidia_epilogue_execution_contract(
        _region(storage, bias, activation, residual))
    assert contract["storage_dtype"] == storage
    assert contract["has_bias"] is bias
    assert contract["activation"] == activation
    assert contract["has_residual"] is residual
    assert contract["accumulation_dtype"] == "f32"
    assert contract["order"] == ("matmul", "bias", "activation", "residual")


@pytest.mark.parametrize("storage", STORAGES[1:])
def test_nvidia_low_precision_residual_rejects_with_registered_dtype_code(storage):
    with pytest.raises(ValueError, match="E_FUSED_EPILOGUE_BAD_DTYPE"):
        nvidia_epilogue_execution_contract(
            _region(storage, True, "relu", True))


@pytest.mark.parametrize("storage", STORAGES)
def test_nvidia_bias_after_activation_rejects_with_registered_order_code(storage):
    region = FusedRegion(epilogue=("relu", "bias"), storage_dtype=storage)
    with pytest.raises(ValueError, match="E_FUSED_EPILOGUE_BAD_ORDER"):
        nvidia_epilogue_execution_contract(region)


def test_nvidia_multiple_activation_chain_is_outside_fused_epilogue_contract():
    region = FusedRegion(epilogue=("relu", "silu"), storage_dtype="f32")
    with pytest.raises(ValueError, match="E_FUSED_EPILOGUE_BAD_ORDER"):
        nvidia_epilogue_execution_contract(region)
