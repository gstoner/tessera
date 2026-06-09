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
from dataclasses import dataclass
from typing import NamedTuple, Optional
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


# ─────────────────────────────────────────────────────────────────────────────
# Distributed MegaMoE — expert-parallel forward (token all-to-all dispatch/combine)
# ─────────────────────────────────────────────────────────────────────────────
#
# The deferred north star: a real expert-parallel MoE layer where experts are
# sharded across ranks and tokens are routed to the rank owning their expert via
# a 2× all-to-all (GShard / Switch-Transformer pattern), with the heavy expert
# FFN running through the fused GPU `moe_swiglu_block` kernel.
#
# Capacity-based dispatch keeps every exchange buffer a fixed size so the
# all-to-all is uniform (the only kind the mock thread group expresses): each
# expert receives exactly `capacity` token slots — overflow is dropped, underflow
# zero-padded. Per Decision #6, multi-rank tests run in-process via MockRankGroup
# (threads), so this is the production-shaped forward AND its own test harness.


class MegaMoEResult(NamedTuple):
    """Per-rank output of ``megamoe_forward``.

    y_local     : (T_local, Kout) combined expert outputs for this rank's tokens
    n_dropped   : tokens dropped to capacity overflow (load-imbalance telemetry)
    capacity    : per-expert capacity used for the dispatch buffers
    """
    y_local: np.ndarray
    n_dropped: int
    capacity: int


def expert_capacity(tokens_per_rank: int, num_experts: int, num_ranks: int,
                    top_k: int, capacity_factor: float) -> int:
    """Per-expert capacity for a uniform all-to-all dispatch buffer.

    Each expert reserves ``capacity_factor × (global token-slots / num_experts)``
    slots, so the dispatch tensor is fixed-size across ranks. Global token-slots
    = ``tokens_per_rank × num_ranks × top_k``.
    """
    global_slots = tokens_per_rank * num_ranks * top_k
    return max(1, int(np.ceil(capacity_factor * global_slots / num_experts)))


def megamoe_forward(
    rank,
    x_local: np.ndarray,
    W_router: np.ndarray,
    local_W_gate: np.ndarray,
    local_W_up: np.ndarray,
    local_W_down: np.ndarray,
    *,
    config: MoEConfig,
    capacity: Optional[int] = None,
    quant=None,
) -> MegaMoEResult:
    """Expert-parallel MoE forward for one rank — the distributed MegaMoE layer.

    Experts are sharded across ``rank.world_size`` ranks: rank ``r`` owns the
    contiguous expert block ``[r·Ep, (r+1)·Ep)`` (``Ep = num_experts/world_size``)
    and holds ONLY those experts' weights — ``local_W_*`` are shaped
    ``(Ep, …)``, not the full expert set (the memory win of expert parallelism).
    The router (``W_router``, shape ``(K, num_experts)``) is replicated.

    Forward (the GShard 2× all-to-all):

        1. route this rank's ``x_local`` tokens through the global router (top-k)
        2. scatter tokens into a capacity-padded dispatch buffer keyed by their
           destination expert's owner rank
        3. all-to-all DISPATCH — every rank receives the tokens bound for its
           local experts, gathered from all source ranks
        4. local expert FFN over the received tokens via the fused GPU
           ``moe_swiglu_block`` (Ep ragged groups, one dispatch)
        5. all-to-all COMBINE — send each result back to the rank that owns the
           originating token
        6. weighted scatter-combine of each token's top-k expert outputs

    Returns a :class:`MegaMoEResult`. Designed to be called inside
    ``MockRankGroup.run(worker)`` — see :func:`megamoe_layer` for the harness.
    """
    from .. import ops as _ops

    R = int(rank.world_size)
    E = int(config.num_experts)
    if E % R != 0:
        raise ValueError(
            f"megamoe_forward: num_experts={E} must be divisible by "
            f"world_size={R} for balanced expert parallelism")
    Ep = E // R

    x_local = np.ascontiguousarray(x_local, dtype=np.float32)
    Wr = np.asarray(W_router, dtype=np.float32)
    Tl, K = x_local.shape
    top_k = int(config.top_k)
    Kout = int(np.asarray(local_W_down).shape[2])

    if capacity is None:
        capacity = expert_capacity(Tl, E, R, top_k, config.capacity_factor)
    C = int(capacity)

    # 1. Local routing through the global (replicated) router. No capacity drop
    #    here — capacity is enforced by the fixed-size dispatch buffer below.
    scores = np.asarray(_ops.gemm(x_local, Wr), dtype=np.float32)
    route = route_tokens(scores, config, capacity=Tl * top_k + 1)
    assign = route.assignment                          # (Tl, top_k)
    weights = route.weights.astype(np.float32)         # (Tl, top_k)

    # 2. Scatter tokens into the dispatch buffer (R, Ep, C, K) keyed by the
    #    destination expert's owner rank. Track (expert, slot) per token slot so
    #    the combine can gather the matching result back.
    send = np.zeros((R, Ep, C, K), dtype=np.float32)
    disp_e = np.full((Tl, top_k), -1, dtype=np.int64)  # global expert per slot
    disp_c = np.full((Tl, top_k), -1, dtype=np.int64)  # capacity position
    fill = np.zeros(E, dtype=np.int64)
    n_dropped = 0
    for t in range(Tl):
        for k in range(top_k):
            e = int(assign[t, k])
            if e < 0:
                continue
            c = int(fill[e])
            if c >= C:                                 # capacity overflow → drop
                n_dropped += 1
                continue
            fill[e] += 1
            send[e // Ep, e % Ep, c] = x_local[t]
            disp_e[t, k] = e
            disp_c[t, k] = c

    # 3. All-to-all DISPATCH: row `dst` of `send` goes to rank `dst`; after the
    #    exchange row `s` of `recv` holds rank `s`'s tokens for our local experts.
    recv = rank.all_to_all(send.reshape(R, -1), scatter_axis=0, gather_axis=0)
    recv = recv.reshape(R, Ep, C, K)

    # 4. Local expert FFN over the (R·C) tokens per local expert — Ep ragged
    #    groups through the fused GPU MoE-SwiGLU kernel in a single dispatch.
    grouped_x = np.ascontiguousarray(
        np.transpose(recv, (1, 0, 2, 3)).reshape(Ep * R * C, K))  # ep-major
    group_sizes = np.full(Ep, R * C, dtype=np.int64)
    y = np.asarray(
        _ops.moe_swiglu_block(grouped_x, local_W_gate, local_W_up, local_W_down,
                              group_sizes, quant=quant),
        dtype=np.float32)
    out = np.transpose(y.reshape(Ep, R, C, Kout), (1, 0, 2, 3))   # (R, Ep, C, Kout)

    # 5. All-to-all COMBINE: send results back to the originating rank.
    back = rank.all_to_all(np.ascontiguousarray(out.reshape(R, -1)),
                           scatter_axis=0, gather_axis=0)
    back = back.reshape(R, Ep, C, Kout)

    # 6. Weighted scatter-combine of each token's top-k expert outputs.
    y_local = np.zeros((Tl, Kout), dtype=np.float32)
    for t in range(Tl):
        for k in range(top_k):
            e = int(disp_e[t, k])
            if e < 0:
                continue
            y_local[t] += weights[t, k] * back[e // Ep, e % Ep, int(disp_c[t, k])]

    return MegaMoEResult(y_local=y_local, n_dropped=n_dropped, capacity=C)


def megamoe_layer(
    x,
    W_router,
    W_gate,
    W_up,
    W_down,
    *,
    world_size: int,
    config: MoEConfig,
    capacity: Optional[int] = None,
    quant=None,
):
    """Run the distributed MegaMoE forward across a mock rank group and return
    ``(y, n_dropped)`` with ``y`` the gathered ``(T, Kout)`` output — the
    single-call harness over :func:`megamoe_forward`.

    ``W_gate`` / ``W_up`` are ``(E, K, H)`` and ``W_down`` is ``(E, H, Kout)`` —
    the FULL expert set; this helper shards them across ranks (rank ``r`` gets the
    ``Ep``-expert block it owns) and shards ``x`` ``(T, K)`` row-wise across ranks
    (``T`` must be divisible by ``world_size``). Tokens are gathered back into the
    original ``(T, Kout)`` order. Numerically equivalent to the single-device
    ``nn.functional.moe_layer`` when ``capacity`` is large enough to drop nothing.
    """
    from ..testing.mock_collective import MockRankGroup

    x = np.ascontiguousarray(np.asarray(x, dtype=np.float32))
    Wg = np.asarray(W_gate, dtype=np.float32)
    Wu = np.asarray(W_up, dtype=np.float32)
    Wd = np.asarray(W_down, dtype=np.float32)
    T = x.shape[0]
    E = config.num_experts
    if T % world_size != 0:
        raise ValueError(
            f"megamoe_layer: T={T} must be divisible by world_size={world_size}")
    if E % world_size != 0:
        raise ValueError(
            f"megamoe_layer: num_experts={E} must be divisible by "
            f"world_size={world_size}")
    Ep = E // world_size
    Tl = T // world_size

    def worker(rank):
        r = rank.rank
        x_local = x[r * Tl:(r + 1) * Tl]
        lg = Wg[r * Ep:(r + 1) * Ep]
        lu = Wu[r * Ep:(r + 1) * Ep]
        ld = Wd[r * Ep:(r + 1) * Ep]
        return megamoe_forward(
            rank, x_local, W_router, lg, lu, ld,
            config=config, capacity=capacity, quant=quant)

    results = MockRankGroup(n=world_size).run(worker)
    y = np.concatenate([res.y_local for res in results], axis=0)
    n_dropped = int(sum(res.n_dropped for res in results))
    return y, n_dropped


# ─────────────────────────────────────────────────────────────────────────────
# Comm/compute overlap — micro-batch pipelined expert-parallel forward
# ─────────────────────────────────────────────────────────────────────────────
#
# The production MoE comm/compute-overlap structure (Tutel / MegaBlocks /
# DeepSpeed-MoE): split a rank's tokens into micro-batches and pipeline them so
# micro-batch c+1's DISPATCH all-to-all overlaps micro-batch c's expert compute,
# and chunk c's COMBINE all-to-all overlaps chunk c+1's compute. On real hardware
# the comm runs on a copy/comm stream while the GEMM runs on the compute stream,
# synchronized by events (the "dependent launch" abstraction — Apple
# command-buffer events / CUDA streams / ROCm stream deps).
#
# HONEST SCOPE: the mock thread-group all_to_all is a cross-rank BARRIER, so this
# harness cannot reduce wall-clock — it validates the *pipeline restructuring*
# (micro-batch decomposition is numerically identical to the monolithic forward)
# and emits the OVERLAP SCHEDULE a stream-backed runtime would execute. The
# algorithmic restructuring is the portable part; the wall-clock win is gated on
# the async-stream lane.


class OverlapSchedule(NamedTuple):
    """The comm/compute pipeline a stream-backed runtime would execute.

    num_chunks    : number of token micro-batches
    stages        : ordered ``(chunk, kind)`` events actually issued, kind ∈
                    {"dispatch_a2a", "expert_compute", "combine_a2a"}
    overlap_pairs : ``(compute_chunk, comm_chunk)`` pairs a stream runtime runs
                    concurrently — compute of chunk c overlaps dispatch of c+1
    """
    num_chunks: int
    stages: list
    overlap_pairs: list


def megamoe_forward_overlapped(
    rank,
    x_local: np.ndarray,
    W_router: np.ndarray,
    local_W_gate: np.ndarray,
    local_W_up: np.ndarray,
    local_W_down: np.ndarray,
    *,
    config: MoEConfig,
    num_chunks: int = 2,
    capacity: Optional[int] = None,
    quant=None,
) -> "tuple[MegaMoEResult, OverlapSchedule]":
    """Micro-batch pipelined expert-parallel forward (comm/compute overlap).

    Splits this rank's tokens into ``num_chunks`` micro-batches and runs the full
    dispatch → expert FFN → combine per chunk, recording the :class:`OverlapSchedule`
    a stream-backed runtime would pipeline (chunk c+1's dispatch overlapping chunk
    c's compute). Numerically identical to :func:`megamoe_forward` when capacity
    drops nothing — micro-batching is a pure decomposition (tokens are processed
    independently; routing + combine are per-token).

    Each micro-batch sizes its own capacity buffer (so an over-subscribed expert
    in one chunk drops independently — the production per-chunk-capacity model).
    Returns ``(MegaMoEResult, OverlapSchedule)``.
    """
    x_local = np.ascontiguousarray(x_local, dtype=np.float32)
    Tl = int(x_local.shape[0])
    nc = max(1, min(int(num_chunks), Tl))
    chunks = [c for c in np.array_split(np.arange(Tl), nc) if c.size]
    nc = len(chunks)

    stages: list = []
    pieces: list = []
    n_dropped = 0
    last_cap = 0
    for c, idx in enumerate(chunks):
        stages.append((c, "dispatch_a2a"))
        stages.append((c, "expert_compute"))
        stages.append((c, "combine_a2a"))
        res = megamoe_forward(
            rank, x_local[idx], W_router, local_W_gate, local_W_up, local_W_down,
            config=config, capacity=capacity, quant=quant)
        pieces.append(res.y_local)
        n_dropped += res.n_dropped
        last_cap = res.capacity

    # A stream runtime overlaps compute of chunk c with dispatch of chunk c+1.
    overlap_pairs = [(c, c + 1) for c in range(nc - 1)]
    y_local = np.concatenate(pieces, axis=0) if pieces else x_local[:0]
    result = MegaMoEResult(y_local=y_local, n_dropped=n_dropped, capacity=last_cap)
    schedule = OverlapSchedule(num_chunks=nc, stages=stages, overlap_pairs=overlap_pairs)
    return result, schedule


def megamoe_layer_overlapped(
    x,
    W_router,
    W_gate,
    W_up,
    W_down,
    *,
    world_size: int,
    config: MoEConfig,
    num_chunks: int = 2,
    capacity: Optional[int] = None,
    quant=None,
):
    """Overlap-pipelined :func:`megamoe_layer` — returns ``(y, n_dropped, schedule)``.

    Same sharding/gather as :func:`megamoe_layer` but each rank runs the
    micro-batch pipelined :func:`megamoe_forward_overlapped`. ``schedule`` is
    rank 0's :class:`OverlapSchedule` (identical structure across ranks). When
    capacity drops nothing the gathered output equals the non-overlapped
    :func:`megamoe_layer` exactly — overlap is a scheduling transform, not a
    numeric one.
    """
    from ..testing.mock_collective import MockRankGroup

    x = np.ascontiguousarray(np.asarray(x, dtype=np.float32))
    Wg = np.asarray(W_gate, dtype=np.float32)
    Wu = np.asarray(W_up, dtype=np.float32)
    Wd = np.asarray(W_down, dtype=np.float32)
    T = x.shape[0]
    E = config.num_experts
    if T % world_size != 0:
        raise ValueError(
            f"megamoe_layer_overlapped: T={T} must be divisible by "
            f"world_size={world_size}")
    if E % world_size != 0:
        raise ValueError(
            f"megamoe_layer_overlapped: num_experts={E} must be divisible by "
            f"world_size={world_size}")
    Ep = E // world_size
    Tl = T // world_size

    def worker(rank):
        r = rank.rank
        return megamoe_forward_overlapped(
            rank, x[r * Tl:(r + 1) * Tl], W_router,
            Wg[r * Ep:(r + 1) * Ep], Wu[r * Ep:(r + 1) * Ep],
            Wd[r * Ep:(r + 1) * Ep],
            config=config, num_chunks=num_chunks, capacity=capacity, quant=quant)

    results = MockRankGroup(n=world_size).run(worker)
    y = np.concatenate([res.y_local for res, _ in results], axis=0)
    n_dropped = int(sum(res.n_dropped for res, _ in results))
    schedule = results[0][1]
    return y, n_dropped, schedule
