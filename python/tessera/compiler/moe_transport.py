"""Canonical MoE dispatch/combine metadata and ordering contracts.

This module owns the backend-neutral conversion from a capacity-aware
``DispatchPlan`` to the compact device ABI: stable expert-grouped slot order,
int32 token indices, group sizes/offsets, and combine weights in that same
order.  Backends own kernels and collective schedules; they must not reconstruct
or reinterpret this metadata independently.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from .native_artifact import OrderingSemantics


@dataclass(frozen=True)
class GroupedExpertMetadata:
    """Canonical contiguous/ragged expert partition for grouped compute."""

    group_sizes: np.ndarray
    group_offsets: np.ndarray
    num_tokens: int

    def __post_init__(self) -> None:
        sizes = np.asarray(self.group_sizes)
        offsets = np.asarray(self.group_offsets)
        if sizes.dtype != np.int32 or sizes.ndim != 1 or sizes.size <= 0:
            raise ValueError("grouped expert sizes must be canonical int32[E]")
        if offsets.dtype != np.int32 or offsets.shape != (sizes.size + 1,):
            raise ValueError("grouped expert offsets must be canonical int32[E+1]")
        if np.any(sizes < 0) or int(sizes.sum()) != self.num_tokens:
            raise ValueError("grouped expert sizes must partition all tokens")
        expected = np.concatenate((np.array([0], np.int32), np.cumsum(sizes)))
        if not np.array_equal(offsets, expected):
            raise ValueError("grouped expert offsets disagree with sizes")

    @property
    def ragged(self) -> bool:
        return bool(np.any(self.group_sizes != self.group_sizes[0]))

    def as_metadata_dict(self) -> dict[str, object]:
        return {
            "schema": "tessera.grouped_expert_metadata.v1",
            "num_tokens": self.num_tokens,
            "num_experts": int(self.group_sizes.size),
            "group_sizes": self.group_sizes.tolist(),
            "group_offsets": self.group_offsets.tolist(),
            "ragged": self.ragged,
            "index_dtype": "int32",
        }


def grouped_expert_metadata(group_sizes: Any, *, num_tokens: int) -> GroupedExpertMetadata:
    sizes64 = np.asarray(group_sizes, dtype=np.int64).reshape(-1)
    if sizes64.size <= 0 or np.any(sizes64 < 0) or int(sizes64.sum()) != num_tokens:
        raise ValueError("grouped expert sizes must partition all tokens")
    offsets64 = np.concatenate((np.array([0], np.int64), np.cumsum(sizes64)))
    if int(offsets64[-1]) > np.iinfo(np.int32).max:
        raise ValueError("grouped expert offsets exceed the int32 device ABI")
    return GroupedExpertMetadata(
        group_sizes=np.ascontiguousarray(sizes64, dtype=np.int32),
        group_offsets=np.ascontiguousarray(offsets64, dtype=np.int32),
        num_tokens=int(num_tokens),
    )


@dataclass(frozen=True)
class MoETransportDescriptor:
    num_tokens: int
    top_k: int
    num_experts: int
    capacity: int | None
    token_of_slot: np.ndarray
    expert_of_slot: np.ndarray
    combine_weights: np.ndarray
    group_sizes: np.ndarray
    group_offsets: np.ndarray
    ordering: OrderingSemantics
    collective_scope: str = "local_device"

    def __post_init__(self) -> None:
        if self.num_tokens < 0 or self.top_k <= 0 or self.num_experts <= 0:
            raise ValueError("MoE transport dimensions are invalid")
        if self.capacity is not None and self.capacity < 0:
            raise ValueError("MoE capacity must be nonnegative or None")
        token = np.asarray(self.token_of_slot)
        expert = np.asarray(self.expert_of_slot)
        weights = np.asarray(self.combine_weights)
        groups = np.asarray(self.group_sizes)
        offsets = np.asarray(self.group_offsets)
        slots = token.size
        if token.dtype != np.int32 or expert.dtype != np.int32:
            raise ValueError("MoE transport indices must use the canonical int32 ABI")
        if weights.dtype != np.float32:
            raise ValueError("MoE combine weights must use fp32")
        if token.shape != (slots,) or expert.shape != (slots,) or weights.shape != (slots,):
            raise ValueError("MoE slot metadata must be rank-1 with equal lengths")
        if groups.dtype != np.int32 or groups.shape != (self.num_experts,):
            raise ValueError("MoE group_sizes must be canonical int32[E]")
        if offsets.dtype != np.int32 or offsets.shape != (self.num_experts + 1,):
            raise ValueError("MoE group_offsets must be canonical int32[E+1]")
        if slots and (int(token.min()) < 0 or int(token.max()) >= self.num_tokens):
            raise ValueError("MoE token indices are out of range")
        if slots and (int(expert.min()) < 0 or int(expert.max()) >= self.num_experts):
            raise ValueError("MoE expert indices are out of range")
        if np.any(groups < 0) or int(groups.sum()) != slots:
            raise ValueError("MoE group sizes do not partition the kept slots")
        if not np.array_equal(offsets, np.concatenate((np.array([0], np.int32), np.cumsum(groups)))):
            raise ValueError("MoE group offsets disagree with group sizes")
        if slots > 1 and np.any(expert[1:] < expert[:-1]):
            raise ValueError("MoE slots must be stably grouped by expert")
        if self.capacity is not None and np.any(groups > self.capacity):
            raise ValueError("MoE expert group exceeds declared capacity")
        if self.collective_scope != "local_device":
            raise ValueError("canonical MoE transport v1 supports local_device scope only")

    @property
    def num_kept(self) -> int:
        return int(self.token_of_slot.size)

    @property
    def ragged(self) -> bool:
        return bool(self.group_sizes.size and np.any(self.group_sizes != self.group_sizes[0]))

    def as_metadata_dict(self) -> dict[str, object]:
        return {
            "schema": "tessera.moe_transport.v1",
            "num_tokens": self.num_tokens,
            "top_k": self.top_k,
            "num_experts": self.num_experts,
            "capacity": self.capacity,
            "num_kept": self.num_kept,
            "group_sizes": self.group_sizes.tolist(),
            "group_offsets": self.group_offsets.tolist(),
            "ragged": self.ragged,
            "index_dtype": "int32",
            "weight_dtype": "fp32",
            "collective_scope": self.collective_scope,
            "ordering": self.ordering.to_dict(),
        }


def descriptor_from_dispatch_plan(plan: Any) -> MoETransportDescriptor:
    """Validate and lower one stdlib ``DispatchPlan`` to the canonical ABI."""
    expert_ids = np.asarray(plan.expert_ids)
    weights = np.asarray(plan.weights)
    kept = np.asarray(plan.kept_mask)
    perm64 = np.asarray(plan.sort_perm, dtype=np.int64).reshape(-1)
    groups64 = np.asarray(plan.group_sizes, dtype=np.int64).reshape(-1)
    if expert_ids.shape != (int(plan.num_tokens), int(plan.top_k)):
        raise ValueError("MoE expert_ids shape disagrees with T/top_k")
    if weights.shape != expert_ids.shape or kept.shape != expert_ids.shape:
        raise ValueError("MoE weights/kept_mask must match expert_ids")
    total = int(plan.num_tokens) * int(plan.top_k)
    if perm64.size and (int(perm64.min()) < 0 or int(perm64.max()) >= total):
        raise ValueError("MoE sort permutation contains an invalid slot")
    if np.unique(perm64).size != perm64.size:
        raise ValueError("MoE sort permutation contains duplicate slots")
    expected_kept = np.flatnonzero(kept.reshape(-1))
    if not np.array_equal(np.sort(perm64), expected_kept):
        raise ValueError("MoE sort permutation does not cover exactly the kept slots")
    i32_max = np.iinfo(np.int32).max
    if total > i32_max or groups64.size > i32_max or np.any(groups64 > i32_max):
        raise ValueError("MoE metadata exceeds the int32 device ABI")
    flat_expert = expert_ids.reshape(-1)
    flat_weights = weights.astype(np.float32, copy=False).reshape(-1)
    token = (perm64 // int(plan.top_k)).astype(np.int32)
    expert = flat_expert[perm64].astype(np.int32)
    combine_weights = flat_weights[perm64].astype(np.float32, copy=False)
    groups = groups64.astype(np.int32)
    offsets = np.concatenate((np.array([0], np.int64), np.cumsum(groups64))).astype(np.int32)
    return MoETransportDescriptor(
        num_tokens=int(plan.num_tokens),
        top_k=int(plan.top_k),
        num_experts=int(groups.size),
        capacity=(None if plan.capacity is None else int(plan.capacity)),
        token_of_slot=np.ascontiguousarray(token),
        expert_of_slot=np.ascontiguousarray(expert),
        combine_weights=np.ascontiguousarray(combine_weights),
        group_sizes=np.ascontiguousarray(groups),
        group_offsets=np.ascontiguousarray(offsets),
        ordering=OrderingSemantics(
            ordered_submission=True,
            residency="inputs",
            synchronization=(
                "dispatch_before_expert_compute",
                "expert_compute_before_combine",
                "combine_completion",
            ),
        ),
    )


__all__ = [
    "GroupedExpertMetadata",
    "MoETransportDescriptor",
    "descriptor_from_dispatch_plan",
    "grouped_expert_metadata",
]
