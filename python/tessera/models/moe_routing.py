"""DiffusionGemma Phase B — MoE routing & packing (reference semantics).

Production token routing built around the existing `grouped_gemm` /
`moe_swiglu_block` primitives: exact top-k routing over the experts, expert
sort/pack into contiguous groups (with group-size metadata), grouped SwiGLU over
the packed tokens, a shared-expert path, weighted combine, and scatter back to
the original token order.

These are the *exact* top-k + packing semantics the compiler IR uses; native
kernels may specialize later (fused routing, capacity-bounded packing). No
capacity dropping here — every routed slot is processed, so the reference is an
exact MoE forward to compare specialized kernels against.

Shapes (T tokens, H hidden, E experts, k = experts-per-token, F expert FFN):
  x            (T, H)
  w_router     (H, E)
  w_gate/w_up  (E, H, F)     w_down (E, F, H)        — routed experts
  w_sgate/w_sup(H, Fs)       w_sdown (Fs, H)         — shared expert
returns          (T, H)
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def _arr(x) -> np.ndarray:
    return np.asarray(x._data if hasattr(x, "_data") else x)


@dataclass(frozen=True)
class RoutingPlan:
    """Token→expert routing + the packing permutation that sorts the ``T*k``
    routed slots into contiguous per-expert groups (so `grouped_gemm` /
    `moe_swiglu_block` can run each group with its expert's weights)."""

    expert_ids: np.ndarray   # (T, k) int64 — selected experts per token
    weights: np.ndarray      # (T, k) float — normalized route weights
    sort_perm: np.ndarray    # (T*k,) int64 — slot order sorted by expert
    inverse_perm: np.ndarray # (T*k,) int64 — restores token-major slot order
    group_sizes: np.ndarray  # (E,) int64 — tokens routed to each expert (sums to T*k)
    num_tokens: int
    top_k: int

    def round_trips(self) -> bool:
        """True iff sort∘inverse is the identity (packing is order-preserving)."""
        n = self.sort_perm.shape[0]
        idx = np.arange(n)
        return bool(np.array_equal(self.sort_perm[self.inverse_perm], idx))


def route_top_k(router_logits, top_k: int, *, normalize: bool = True):
    """Exact top-k expert routing. Returns ``(expert_ids (T,k), weights (T,k))``.

    Stable selection: ties broken by lowest expert index (deterministic). When
    ``normalize`` the per-token weights are a softmax over the selected logits.
    """
    logits = _arr(router_logits).astype(np.float64)
    if logits.ndim != 2:
        raise ValueError("router_logits must be (T, num_experts)")
    T, E = logits.shape
    if not (1 <= top_k <= E):
        raise ValueError(f"top_k={top_k} out of [1, num_experts={E}]")
    # Stable descending sort → deterministic tie-break by lowest index.
    order = np.argsort(-logits, axis=1, kind="stable")[:, :top_k]   # (T, k)
    sel = np.take_along_axis(logits, order, axis=1)                 # (T, k)
    if normalize:
        sel = sel - sel.max(axis=1, keepdims=True)
        w = np.exp(sel)
        w = w / w.sum(axis=1, keepdims=True)
    else:
        w = np.ones_like(sel) / top_k
    return order.astype(np.int64), w.astype(np.float32)


def plan_packing(expert_ids, num_experts: int) -> RoutingPlan:
    """Build the per-expert packing plan from ``expert_ids`` (T, k)."""
    eids = _arr(expert_ids).astype(np.int64)
    if eids.ndim != 2:
        raise ValueError("expert_ids must be (T, k)")
    T, k = eids.shape
    if eids.size and (eids.min() < 0 or eids.max() >= num_experts):
        raise ValueError("expert id out of range")
    flat = eids.reshape(-1)                          # token-major slot order
    sort_perm = np.argsort(flat, kind="stable")      # group slots by expert
    inverse_perm = np.empty_like(sort_perm)
    inverse_perm[sort_perm] = np.arange(sort_perm.shape[0], dtype=np.int64)
    group_sizes = np.bincount(flat, minlength=num_experts).astype(np.int64)
    return RoutingPlan(
        expert_ids=eids, weights=np.zeros_like(eids, dtype=np.float32),
        sort_perm=sort_perm, inverse_perm=inverse_perm, group_sizes=group_sizes,
        num_tokens=T, top_k=k,
    )


def pack_tokens(x, plan: RoutingPlan) -> np.ndarray:
    """Gather ``x`` (T, H) into packed per-expert slot order (T*k, H)."""
    xa = _arr(x)
    token_of_slot = np.repeat(np.arange(plan.num_tokens), plan.top_k)  # (T*k,)
    return xa[token_of_slot][plan.sort_perm]


def unpack_combine(y_packed, plan: RoutingPlan, weights) -> np.ndarray:
    """Scatter packed expert outputs back to token order and combine the k
    per-token contributions with their route weights → (T, H)."""
    yp = _arr(y_packed)
    H = yp.shape[-1]
    y_slots = yp[plan.inverse_perm].reshape(plan.num_tokens, plan.top_k, H)
    w = _arr(weights).astype(yp.dtype)
    return (y_slots * w[:, :, None]).sum(axis=1)


def _swiglu(x, w_gate, w_up, w_down) -> np.ndarray:
    g = x @ w_gate
    u = x @ w_up
    h = (g * (1.0 / (1.0 + np.exp(-g)))) * u   # silu(g) * u
    return h @ w_down


def moe_forward(
    x, w_router, w_gate, w_up, w_down,
    w_shared_gate, w_shared_up, w_shared_down,
    *, top_k: int, normalize: bool = True,
):
    """Full MoE layer forward via the pack/grouped/scatter pipeline.

    Returns ``(y (T,H), plan)``. Uses ``tessera.ops.moe_swiglu_block`` for the
    grouped expert compute (the contiguous-grouped kernel surface).
    """
    from tessera import ops as _ops

    xa = _arr(x).astype(np.float64)
    wr = _arr(w_router).astype(np.float64)
    E = wr.shape[1]

    # 1. route + 2. plan packing
    expert_ids, weights = route_top_k(xa @ wr, top_k, normalize=normalize)
    plan = plan_packing(expert_ids, E)
    plan = RoutingPlan(  # attach the weights computed above
        expert_ids=plan.expert_ids, weights=weights, sort_perm=plan.sort_perm,
        inverse_perm=plan.inverse_perm, group_sizes=plan.group_sizes,
        num_tokens=plan.num_tokens, top_k=plan.top_k)

    # 3. pack → 4. grouped SwiGLU (per-expert contiguous groups)
    x_packed = pack_tokens(xa, plan)
    y_packed = np.asarray(_ops.moe_swiglu_block(
        x_packed, _arr(w_gate).astype(np.float64), _arr(w_up).astype(np.float64),
        _arr(w_down).astype(np.float64), plan.group_sizes, kind="contiguous"))

    # 5. scatter back + weighted combine → routed output
    y_routed = unpack_combine(y_packed, plan, weights)

    # 6. shared-expert path + combine
    y_shared = _swiglu(xa, _arr(w_shared_gate).astype(np.float64),
                       _arr(w_shared_up).astype(np.float64),
                       _arr(w_shared_down).astype(np.float64))
    return (y_routed + y_shared), plan


def moe_forward_naive(
    x, w_router, w_gate, w_up, w_down,
    w_shared_gate, w_shared_up, w_shared_down,
    *, top_k: int, normalize: bool = True,
) -> np.ndarray:
    """Naive per-token / per-expert reference (no packing) for parity checks."""
    xa = _arr(x).astype(np.float64)
    wr = _arr(w_router).astype(np.float64)
    wg, wu, wd = (_arr(w).astype(np.float64) for w in (w_gate, w_up, w_down))
    expert_ids, weights = route_top_k(xa @ wr, top_k, normalize=normalize)
    T, H = xa.shape
    y = np.zeros((T, H), dtype=np.float64)
    for t in range(T):
        for j in range(top_k):
            e = int(expert_ids[t, j])
            y[t] += weights[t, j] * _swiglu(xa[t:t + 1], wg[e], wu[e], wd[e])[0]
    y += _swiglu(xa, _arr(w_shared_gate).astype(np.float64),
                 _arr(w_shared_up).astype(np.float64),
                 _arr(w_shared_down).astype(np.float64))
    return y


def synthetic_moe_weights(config, *, seed: int = 0) -> dict:
    """Synthetic BF16-scale (stored fp32) weights for a DiffusionGemma MoE layer
    at the config's production dims. For tests/benchmarks only."""
    rng = np.random.default_rng(seed)
    H, E = config.hidden_size, config.num_experts
    F, Fs = config.moe_intermediate_size, config.shared_expert_intermediate_size
    s = 1.0 / np.sqrt(H)
    return {
        "w_router": (rng.standard_normal((H, E)) * s).astype(np.float32),
        "w_gate": (rng.standard_normal((E, H, F)) * s).astype(np.float32),
        "w_up": (rng.standard_normal((E, H, F)) * s).astype(np.float32),
        "w_down": (rng.standard_normal((E, F, H)) / np.sqrt(F)).astype(np.float32),
        "w_shared_gate": (rng.standard_normal((H, Fs)) * s).astype(np.float32),
        "w_shared_up": (rng.standard_normal((H, Fs)) * s).astype(np.float32),
        "w_shared_down": (rng.standard_normal((Fs, H)) / np.sqrt(Fs)).astype(np.float32),
    }


__all__ = [
    "RoutingPlan",
    "route_top_k",
    "plan_packing",
    "pack_tokens",
    "unpack_combine",
    "moe_forward",
    "moe_forward_naive",
    "synthetic_moe_weights",
]
