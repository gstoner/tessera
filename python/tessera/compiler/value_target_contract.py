"""Backend-neutral Value Target IR contract helpers.

Stage 18 makes the Apple-proven value handoff a shared compiler contract:

    Graph IR op -> registered Tile op -> value Target IR call
      -> backend executor adapter -> numerical proof

Backends may choose different ABI names and runtime adapters, but the value-call
record keeps the same required fields.  This module is intentionally small and
hardware-free so NVIDIA/ROCm can adopt the shape before their executors exist.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping


VALUE_TARGET_HANDOFF_STEPS: tuple[str, ...] = (
    "graph_ir_op",
    "registered_tile_op",
    "value_target_ir_call",
    "backend_executor_adapter",
    "numerical_proof",
)

REQUIRED_VALUE_CALL_ATTRS: tuple[str, ...] = (
    "op_kind",
    "symbol",
    "status",
    "abi",
    "dtype",
    "framework",
)


@dataclass(frozen=True)
class BackendValueTargetContract:
    backend: str
    target_prefixes: tuple[str, ...]
    kernel_call_op: str
    package_call_op: str | None
    compiler_path: str
    executor_id: str | None
    required_attrs: tuple[str, ...] = REQUIRED_VALUE_CALL_ATTRS


BACKEND_VALUE_TARGET_CONTRACTS: dict[str, BackendValueTargetContract] = {
    "apple_gpu": BackendValueTargetContract(
        backend="apple_gpu",
        target_prefixes=("apple_gpu",),
        kernel_call_op="tessera_apple.gpu.kernel_call",
        package_call_op="tessera_apple.gpu.package_call",
        compiler_path="apple_value_target_ir",
        executor_id="apple_gpu_value_target_ir",
    ),
    "nvidia": BackendValueTargetContract(
        backend="nvidia",
        target_prefixes=("nvidia", "nvidia_sm80", "nvidia_sm90",
                         "nvidia_sm100", "nvidia_sm120"),
        kernel_call_op="tessera_nvidia.kernel_call",
        package_call_op=None,
        compiler_path="nvidia_value_target_ir",
        executor_id=None,
    ),
    "rocm": BackendValueTargetContract(
        backend="rocm",
        target_prefixes=("rocm", "rocm_gfx90a", "rocm_gfx940",
                         "rocm_gfx942", "rocm_gfx950", "rocm_gfx1100"),
        kernel_call_op="tessera_rocm.kernel_call",
        package_call_op=None,
        compiler_path="rocm_value_target_ir",
        executor_id=None,
    ),
}


def backend_family_for_target(target: str) -> str | None:
    """Return the value-contract backend family for a normalized target."""

    for family, contract in BACKEND_VALUE_TARGET_CONTRACTS.items():
        if target in contract.target_prefixes:
            return family
    return None


def value_contract_for_target(target: str) -> BackendValueTargetContract | None:
    family = backend_family_for_target(target)
    if family is None:
        return None
    return BACKEND_VALUE_TARGET_CONTRACTS[family]


def missing_value_call_attrs(call: Mapping[str, object], *,
                             target: str) -> tuple[str, ...]:
    """Return required value-call attrs missing for this backend target.

    The op mnemonic itself is validated as the first contract field because a
    CUDA/HIP/Metal executor adapter must not consume another backend's call op.
    """

    contract = value_contract_for_target(target)
    if contract is None:
        return ("backend_contract",)
    missing: list[str] = []
    if call.get("op") != contract.kernel_call_op:
        missing.append("op")
    for attr in contract.required_attrs:
        if call.get(attr) in (None, ""):
            missing.append(attr)
    return tuple(missing)


def is_complete_value_call_record(call: Mapping[str, object], *,
                                  target: str) -> bool:
    return not missing_value_call_attrs(call, target=target)


__all__ = [
    "BACKEND_VALUE_TARGET_CONTRACTS",
    "BackendValueTargetContract",
    "REQUIRED_VALUE_CALL_ATTRS",
    "VALUE_TARGET_HANDOFF_STEPS",
    "backend_family_for_target",
    "is_complete_value_call_record",
    "missing_value_call_attrs",
    "value_contract_for_target",
]
