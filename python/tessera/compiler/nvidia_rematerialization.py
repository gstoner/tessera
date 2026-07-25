"""CUDA device/model capacity policy for activation rematerialization.

This module is deliberately a policy/IR seam, not a CUDA schedule.  It queries
the same retained CUDA context used by the native PTX bridge, derives a
conservative activation budget, and stamps the shared Graph IR attributes
consumed by ``tessera-activation-rematerialization``.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import prod
from typing import Mapping, Optional, Set

from .graph_ir import GraphIRFunction, GraphIRModule, IRArg


_I64_MAX = (1 << 63) - 1
_STORAGE_BYTES = {
    "bool": 1,
    "i1": 1,
    "int8": 1,
    "uint8": 1,
    "i8": 1,
    "fp16": 2,
    "f16": 2,
    "bf16": 2,
    "int16": 2,
    "uint16": 2,
    "i16": 2,
    "fp32": 4,
    "f32": 4,
    "tf32": 4,
    "int32": 4,
    "uint32": 4,
    "i32": 4,
    "fp64": 8,
    "f64": 8,
    "int64": 8,
    "uint64": 8,
    "i64": 8,
}


@dataclass(frozen=True)
class NVIDIARematerializationPolicy:
    device_capacity_bytes: int
    device_free_bytes: int
    effective_capacity_bytes: int
    reserve_basis_points: int
    model_parameter_bytes: int
    gradient_copies: int
    optimizer_state_copies: int
    persistent_bytes: int
    model_state_bytes: int
    activation_budget_bytes: int


def _checked_nonnegative(name: str, value: int) -> int:
    value = int(value)
    if value < 0 or value > _I64_MAX:
        raise ValueError(f"{name} must be in signed-i64 non-negative range")
    return value


def _checked_product(name: str, *values: int) -> int:
    result = 1
    for value in values:
        result *= _checked_nonnegative(name, value)
        if result > _I64_MAX:
            raise ValueError(f"{name} overflows signed i64")
    return result


def _parameter_storage_bytes(arg: IRArg) -> Optional[int]:
    dtype = arg.ir_type.dtype
    width = _STORAGE_BYTES.get(str(dtype)) if dtype is not None else None
    shape = arg.ir_type.shape
    if width is None or not shape:
        return None
    dimensions: list[int] = []
    for dimension in shape:
        try:
            value = int(dimension)
        except (TypeError, ValueError):
            return None
        if value < 0:
            return None
        dimensions.append(value)
    return _checked_product("model parameter byte size", width, prod(dimensions))


def resolve_nvidia_rematerialization_policy(
    *,
    model_parameter_bytes: int,
    capacity_bytes: Optional[int] = None,
    free_bytes: Optional[int] = None,
    reserve_basis_points: int = 1000,
    gradient_copies: int = 1,
    optimizer_state_copies: int = 2,
    persistent_bytes: int = 0,
) -> NVIDIARematerializationPolicy:
    """Resolve the shared rematerialization envelope for one CUDA device.

    When capacity is not supplied, the query comes from the native NVIDIA PTX
    bridge.  Current free memory caps total capacity so allocations made by
    other processes cannot be promised to the rematerialization planner.
    """
    if capacity_bytes is None:
        from tessera.runtime import _nvidia_device_memory_envelope

        envelope = _nvidia_device_memory_envelope()
        capacity_bytes = envelope["capacity_bytes"]
        if free_bytes is None:
            free_bytes = envelope["free_bytes"]
    capacity = _checked_nonnegative("capacity_bytes", capacity_bytes)
    free = capacity if free_bytes is None else _checked_nonnegative(
        "free_bytes", free_bytes
    )
    effective = min(capacity, free)
    reserve = int(reserve_basis_points)
    if reserve < 0 or reserve > 10000:
        raise ValueError("reserve_basis_points must be in [0, 10000]")
    parameters = _checked_nonnegative(
        "model_parameter_bytes", model_parameter_bytes
    )
    gradients = _checked_nonnegative("gradient_copies", gradient_copies)
    optimizer = _checked_nonnegative(
        "optimizer_state_copies", optimizer_state_copies
    )
    persistent = _checked_nonnegative("persistent_bytes", persistent_bytes)
    state_copies = 1 + gradients + optimizer
    state = _checked_product("model state bytes", parameters, state_copies)
    state += persistent
    if state > _I64_MAX:
        raise ValueError("model state bytes overflows signed i64")
    usable = (effective * (10000 - reserve)) // 10000
    budget = max(0, usable - state)
    return NVIDIARematerializationPolicy(
        device_capacity_bytes=capacity,
        device_free_bytes=free,
        effective_capacity_bytes=effective,
        reserve_basis_points=reserve,
        model_parameter_bytes=parameters,
        gradient_copies=gradients,
        optimizer_state_copies=optimizer,
        persistent_bytes=persistent,
        model_state_bytes=state,
        activation_budget_bytes=budget,
    )


def apply_nvidia_rematerialization_policy(
    module: GraphIRModule,
    *,
    parameter_names: Set[str],
    parameter_bytes_bounds: Optional[Mapping[str, int]] = None,
    capacity_bytes: Optional[int] = None,
    free_bytes: Optional[int] = None,
    reserve_basis_points: int = 1000,
    gradient_copies: int = 1,
    optimizer_state_copies: int = 2,
    persistent_bytes: int = 0,
) -> dict[str, NVIDIARematerializationPolicy]:
    """Mark model parameters and stamp CUDA capacity policy on Graph IR.

    Dynamic parameters require an explicit conservative byte bound.  Functions
    without any named parameter are left untouched; at least one function must
    consume a requested parameter name.
    """
    bounds = dict(parameter_bytes_bounds or {})
    missing_bounds = set(bounds) - parameter_names
    if missing_bounds:
        raise ValueError(
            "parameter byte bounds supplied for unmarked parameters: "
            + ", ".join(sorted(missing_bounds))
        )
    device_envelope: Optional[dict[str, int]] = None
    if capacity_bytes is None:
        from tessera.runtime import _nvidia_device_memory_envelope

        device_envelope = _nvidia_device_memory_envelope()
        capacity_bytes = device_envelope["capacity_bytes"]
        if free_bytes is None:
            free_bytes = device_envelope["free_bytes"]

    policies: dict[str, NVIDIARematerializationPolicy] = {}
    consumed: set[str] = set()
    for function in module.functions:
        parameter_bytes = 0
        has_parameter = False
        for arg in function.args:
            if arg.name not in parameter_names:
                continue
            has_parameter = True
            consumed.add(arg.name)
            arg.model_parameter = True
            byte_bound = bounds.get(arg.name)
            static_bytes = _parameter_storage_bytes(arg)
            if static_bytes is None:
                if byte_bound is None:
                    raise ValueError(
                        f"dynamic or unsupported model parameter {arg.name!r} "
                        "requires parameter_bytes_bounds"
                    )
                arg.model_parameter_bytes_bound = _checked_nonnegative(
                    f"parameter bound for {arg.name}", byte_bound
                )
                size = arg.model_parameter_bytes_bound
            else:
                if byte_bound is not None and int(byte_bound) < static_bytes:
                    raise ValueError(
                        f"parameter bound for {arg.name!r} is smaller than its "
                        "static storage footprint"
                    )
                size = static_bytes
            parameter_bytes += size
            if parameter_bytes > _I64_MAX:
                raise ValueError("model parameter byte total overflows signed i64")
        if not has_parameter:
            continue
        policy = resolve_nvidia_rematerialization_policy(
            model_parameter_bytes=parameter_bytes,
            capacity_bytes=capacity_bytes,
            free_bytes=free_bytes,
            reserve_basis_points=reserve_basis_points,
            gradient_copies=gradient_copies,
            optimizer_state_copies=optimizer_state_copies,
            persistent_bytes=persistent_bytes,
        )
        _stamp_function_policy(function, policy)
        policies[function.name] = policy
    missing = parameter_names - consumed
    if missing:
        raise ValueError(
            "model parameters do not match Graph IR arguments: "
            + ", ".join(sorted(missing))
        )
    if not policies:
        raise ValueError("at least one model parameter name is required")
    return policies


def _stamp_function_policy(
    function: GraphIRFunction, policy: NVIDIARematerializationPolicy
) -> None:
    # The effective capacity is intentionally stamped: the shared pass must not
    # plan against bytes unavailable in the exact retained CUDA context.
    function.fn_attrs.update(
        {
            "tessera.device_memory_capacity_bytes": (
                f"{policy.effective_capacity_bytes} : i64"
            ),
            "tessera.device_memory_reserve_basis_points": (
                f"{policy.reserve_basis_points} : i32"
            ),
            "tessera.model_gradient_copies": f"{policy.gradient_copies} : i32",
            "tessera.model_optimizer_state_copies": (
                f"{policy.optimizer_state_copies} : i32"
            ),
            "tessera.model_persistent_bytes": (
                f"{policy.persistent_bytes} : i64"
            ),
        }
    )


__all__ = [
    "NVIDIARematerializationPolicy",
    "apply_nvidia_rematerialization_policy",
    "resolve_nvidia_rematerialization_policy",
]
