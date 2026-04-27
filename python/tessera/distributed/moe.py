"""
tessera.distributed.moe — Mixture-of-Experts routing and all-to-all planning.

Provides:
  MoEConfig       — describes expert count, capacity, and top-k routing policy
  route_tokens()  — compute per-token expert assignments from gate logits
  plan_all_to_all() — derive the all-to-all bucket plan for a routing table

Design:
  Token routing is the critical scheduling primitive for MoE:
    1. The gate network produces a (num_tokens, num_experts) score matrix
    2. route_tokens() selects top-k experts per token and produces:
       - assignment: (num_tokens, top_k) expert indices
       - weights:    (num_tokens, top_k) softmax weights
       - load:       (num_experts,) token counts per expert
    3. plan_all_to_all() converts assignment into per-rank send/recv bucket
       lists for the collective all-to-all scatter.

The output of plan_all_to_all() is consumed by:
  - The Cyclic distribution's all_to_all rebalance in DistributedArray.parts()
  - The GPUCollectiveInsertionPass when it sees a tessera.moe.dispatch op

Reference: CLAUDE.md §Phase 4 — MoE helpers
           python/tessera/compiler/distributed_planner.py
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List, NamedTuple, Optional, Tuple
import numpy as np


# ─────────────────────────────────────────────────────────────────────────────
# MoEConfig
# ─────────────────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class MoEConfig:
    """
    Configuration for a Mixture-of-Experts layer.

    Attributes:
        num_experts     : total number of experts (must be divisible by num_ranks)
        top_k           : number of experts each token is routed to (typically 1 or 2)
        expert_capacity : maximum tokens per expert per batch
                          (capacity = capacity_factor × tokens_per_rank / num_experts)
        capacity_factor : multiplier for auto-computing expert_capacity from batch size
        jitter_noise    : small uniform noise added to gate logits for load balancing
        normalize_weights: if True, top-k weights sum to 1.0 (softmax renorm)

    Example:
        cfg = MoEConfig(num_experts=8, top_k=2, capacity_factor=1.25)
    """
    num_experts: int
    top_k: int = 1
    expert_capacity: Optional[int] = None
    capacity_factor: float = 1.25
    jitter_noise: float = 0.0
    normalize_weights: bool = True

    def __post_init__(self) -> None:
        if self.num_experts < 1:
            raise ValueError(f"num_experts must be >= 1, got {self.num_experts}")
        if self.top_k < 1 or self.top_k > self.num_experts:
            raise ValueError(
                f"top_k must be in [1, num_experts={self.num_experts}], got {self.top_k}"
            )
        if self.capacity_factor <= 0:
            raise ValueError(f"capacity_factor must be > 0, got {self.capacity_factor}")
        if not (0.0 <= self.jitter_noise < 1.0):
            raise ValueError(f"jitter_noise must be in [0, 1), got {self.jitter_noise}")

    def compute_capacity(self, num_tokens: int, num_ranks: int = 1) -> int:
        """
        Auto-compute expert_capacity from batch size if not explicitly set.

        capacity = ceil(capacity_factor × num_tokens / (num_experts / num_ranks))

        This ensures each expert slot can hold capacity_factor times the
        average token count if routing were perfectly balanced.
        """
        if self.expert_capacity is not None:
            return self.expert_capacity
        experts_per_rank = max(1, self.num_experts // num_ranks)
        tokens_per_expert = num_tokens / experts_per_rank
        return int(np.ceil(self.capacity_factor * tokens_per_expert))


# ─────────────────────────────────────────────────────────────────────────────
# Routing result
# ─────────────────────────────────────────────────────────────────────────────

class RoutingResult(NamedTuple):
    """
    Output of route_tokens().

    Attributes:
        assignment  : (num_tokens, top_k) int array — expert index for each slot
        weights     : (num_tokens, top_k) float32 array — routing weight per slot
        load        : (num_experts,) int array — token count routed to each expert
        overflow    : int — tokens dropped due to capacity overflow
    """
    assignment: np.ndarray   # shape (T, top_k), dtype int64
    weights:    np.ndarray   # shape (T, top_k), dtype float32
    load:       np.ndarray   # shape (E,),       dtype int64
    overflow:   int


# ─────────────────────────────────────────────────────────────────────────────
# route_tokens
# ─────────────────────────────────────────────────────────────────────────────

def route_tokens(
    scores: np.ndarray,
    config: MoEConfig,
    capacity: Optional[int] = None,
) -> RoutingResult:
    """
    Compute top-k expert routing from gate logit scores.

    Args:
        scores  : (num_tokens, num_experts) float32 gate logits
        config  : MoEConfig
        capacity: expert capacity override; if None, auto-computed

    Returns:
        RoutingResult with assignment, weights, load, overflow

    Example:
        scores = np.random.randn(128, 8).astype(np.float32)
        cfg = MoEConfig(num_experts=8, top_k=2)
        result = route_tokens(scores, cfg)
        assert result.assignment.shape == (128, 2)
        assert result.weights.shape == (128, 2)
        assert result.load.sum() == 128 * 2   # top_k tokens per expert slot
    """
    if scores.ndim != 2:
        raise ValueError(
            f"scores must be 2D (num_tokens, num_experts), got shape {scores.shape}"
        )
    num_tokens, num_experts = scores.shape
    if num_experts != config.num_experts:
        raise ValueError(
            f"scores has {num_experts} expert columns but config.num_experts={config.num_experts}"
        )

    # Optional jitter for load balancing
    if config.jitter_noise > 0.0:
        noise = np.random.uniform(
            1.0 - config.jitter_noise,
            1.0 + config.jitter_noise,
            size=scores.shape,
        ).astype(np.float32)
        scores = scores * noise

    # Softmax over experts
    exp_scores = np.exp(scores - scores.max(axis=1, keepdims=True))
    probs = exp_scores / exp_scores.sum(axis=1, keepdims=True)

    # Top-k selection
    top_k_indices = np.argsort(probs, axis=1)[:, -config.top_k:][:, ::-1]  # (T, k) descending
    top_k_weights = np.take_along_axis(probs, top_k_indices, axis=1)

    # Renormalize weights to sum to 1.0 per token
    if config.normalize_weights and config.top_k > 1:
        weight_sum = top_k_weights.sum(axis=1, keepdims=True)
        weight_sum = np.where(weight_sum > 0, weight_sum, 1.0)
        top_k_weights = top_k_weights / weight_sum

    # Compute load per expert
    load = np.zeros(num_experts, dtype=np.int64)
    for expert_idx in top_k_indices.flatten():
        load[expert_idx] += 1

    # Apply capacity constraint — overflow tracking
    if capacity is None:
        capacity = config.compute_capacity(num_tokens)

    overflow = 0
    expert_fill = np.zeros(num_experts, dtype=np.int64)
    assignment_out = top_k_indices.copy()
    weights_out = top_k_weights.copy()

    for token_i in range(num_tokens):
        for slot in range(config.top_k):
            expert = top_k_indices[token_i, slot]
            if expert_fill[expert] >= capacity:
                # Drop this assignment: mark as -1 (dropped)
                assignment_out[token_i, slot] = -1
                weights_out[token_i, slot] = 0.0
                overflow += 1
            else:
                expert_fill[expert] += 1

    return RoutingResult(
        assignment=assignment_out.astype(np.int64),
        weights=weights_out.astype(np.float32),
        load=load,
        overflow=overflow,
    )


# ─────────────────────────────────────────────────────────────────────────────
# AllToAll bucket plan
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class AllToAllPlan:
    """
    Per-rank send/receive bucket sizes for an all-to-all collective.

    send_counts[src_rank][dst_rank] = number of tokens src sends to dst
    recv_counts[dst_rank][src_rank] = number of tokens dst receives from src

    Used by the CollectiveScheduler to issue chunked sends and by
    GPUCollectiveInsertionPass to size buffers.
    """
    num_ranks: int
    num_experts: int
    send_counts: np.ndarray   # (num_ranks, num_ranks) int64
    recv_counts: np.ndarray   # (num_ranks, num_ranks) int64
    experts_per_rank: int

    @property
    def max_send_tokens(self) -> int:
        """Maximum tokens any single rank sends in one all-to-all step."""
        return int(self.send_counts.sum(axis=1).max())

    @property
    def max_recv_tokens(self) -> int:
        """Maximum tokens any single rank receives in one all-to-all step."""
        return int(self.recv_counts.sum(axis=1).max())

    def to_ir_attr(self) -> str:
        """Serialize for GPUCollectiveInsertionPass annotation."""
        sc = self.send_counts.flatten().tolist()
        return (
            f'{{tessera.moe_a2a = {{num_ranks = {self.num_ranks}, '
            f'experts_per_rank = {self.experts_per_rank}, '
            f'send_counts = {sc}}}}}'
        )


def plan_all_to_all(
    routing: RoutingResult,
    num_experts: int,
    num_ranks: int,
) -> AllToAllPlan:
    """
    Derive the all-to-all send/receive bucket plan from a routing table.

    Maps each expert to a rank (expert e → rank e // experts_per_rank) and
    computes how many tokens each rank must send to every other rank.

    Args:
        routing     : RoutingResult from route_tokens()
        num_experts : total expert count
        num_ranks   : number of devices in the collective group

    Returns:
        AllToAllPlan with send_counts and recv_counts matrices

    Example:
        plan = plan_all_to_all(result, num_experts=8, num_ranks=4)
        # With 4 ranks and 8 experts: experts 0-1 on rank 0, 2-3 on rank 1, …
        assert plan.send_counts.shape == (4, 4)
    """
    if num_experts % num_ranks != 0:
        raise ValueError(
            f"num_experts={num_experts} must be divisible by num_ranks={num_ranks} "
            f"for balanced all-to-all dispatch"
        )
    experts_per_rank = num_experts // num_ranks

    num_tokens = routing.assignment.shape[0]
    top_k = routing.assignment.shape[1]

    # expert e is owned by rank: e // experts_per_rank
    # For simplicity we assume tokens are evenly distributed across source ranks:
    # token i originates from rank i // (num_tokens // num_ranks)
    tokens_per_rank = max(1, num_tokens // num_ranks)

    send_counts = np.zeros((num_ranks, num_ranks), dtype=np.int64)

    for token_i in range(num_tokens):
        src_rank = min(token_i // tokens_per_rank, num_ranks - 1)
        for slot in range(top_k):
            expert = routing.assignment[token_i, slot]
            if expert < 0:
                continue  # dropped due to capacity overflow
            dst_rank = int(expert) // experts_per_rank
            send_counts[src_rank, dst_rank] += 1

    # recv_counts is the transpose of send_counts
    recv_counts = send_counts.T.copy()

    return AllToAllPlan(
        num_ranks=num_ranks,
        num_experts=num_experts,
        send_counts=send_counts,
        recv_counts=recv_counts,
        experts_per_rank=experts_per_rank,
    )
