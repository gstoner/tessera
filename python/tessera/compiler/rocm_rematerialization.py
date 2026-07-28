"""ROCm device/model capacity injection for shared rematerialization.

The arithmetic and Graph-IR attribute contract are target-independent; this
module contributes only the exact retained-HIP-context capacity query.
"""

from __future__ import annotations

from typing import Mapping, Optional, Set

from .graph_ir import GraphIRModule
from .memory_capacity import (
    RematerializationCapacityPolicy as ROCmRematerializationPolicy,
    apply_rematerialization_capacity_policy,
    resolve_rematerialization_capacity_policy,
)


def resolve_rocm_rematerialization_policy(
    *,
    model_parameter_bytes: int,
    capacity_bytes: Optional[int] = None,
    free_bytes: Optional[int] = None,
    reserve_basis_points: int = 1000,
    gradient_copies: int = 1,
    optimizer_state_copies: int = 2,
    persistent_bytes: int = 0,
) -> ROCmRematerializationPolicy:
    if capacity_bytes is None:
        from tessera.runtime import _rocm_device_memory_envelope

        envelope = _rocm_device_memory_envelope()
        capacity_bytes = envelope["capacity_bytes"]
        if free_bytes is None:
            free_bytes = envelope["free_bytes"]
    return resolve_rematerialization_capacity_policy(
        model_parameter_bytes=model_parameter_bytes,
        capacity_bytes=capacity_bytes,
        free_bytes=free_bytes,
        reserve_basis_points=reserve_basis_points,
        gradient_copies=gradient_copies,
        optimizer_state_copies=optimizer_state_copies,
        persistent_bytes=persistent_bytes,
    )


def apply_rocm_rematerialization_policy(
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
) -> dict[str, ROCmRematerializationPolicy]:
    """Stamp model validation and the exact HIP capacity envelope on Graph IR."""
    if capacity_bytes is None:
        from tessera.runtime import _rocm_device_memory_envelope

        envelope = _rocm_device_memory_envelope()
        capacity_bytes = envelope["capacity_bytes"]
        if free_bytes is None:
            free_bytes = envelope["free_bytes"]
    return apply_rematerialization_capacity_policy(
        module,
        parameter_names=parameter_names,
        parameter_bytes_bounds=parameter_bytes_bounds,
        capacity_bytes=capacity_bytes,
        free_bytes=free_bytes,
        reserve_basis_points=reserve_basis_points,
        gradient_copies=gradient_copies,
        optimizer_state_copies=optimizer_state_copies,
        persistent_bytes=persistent_bytes,
    )


__all__ = [
    "ROCmRematerializationPolicy",
    "apply_rocm_rematerialization_policy",
    "resolve_rocm_rematerialization_policy",
]
