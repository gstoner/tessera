"""MoE engine: routing, auxiliary losses, and the feed-forward block.

Built directly on the existing, tested reference routing in
``tessera.models.moe_routing`` (``route_top_k`` + ``moe_forward_naive``) — no
new routing math, no registry indirection. A model file under ``models/``
instantiates ``MoEFeedForward`` directly and can read this entire engine in one
sitting.

The two auxiliary losses below (Switch-Transformer load balancing + router
z-loss) are the standard MoE training-stability terms; they are computed in
pure numpy here so the scaffold runs everywhere. The perf path swaps
``moe_forward_naive`` for the packed/grouped ``moe_forward`` (Apple-GPU capable
via ``ops.moe_swiglu_block``) without changing this module's interface.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from tessera import nn, ops
from tessera.models.moe_routing import moe_forward_naive, route_top_k


def _arr(x) -> np.ndarray:
    # Unwrap Parameter -> DistributedArray -> numpy (the buffer can be nested
    # one or two ``._data`` levels deep depending on the handle type).
    while hasattr(x, "_data"):
        x = x._data
    return np.asarray(x)


# ─────────────────────────────────────────────────────────────────────────────
# Auxiliary losses (standard MoE training-stability terms)
# ─────────────────────────────────────────────────────────────────────────────

def load_balancing_loss(router_logits, expert_ids, num_experts: int) -> float:
    """Switch-Transformer load-balancing auxiliary loss.

    ``loss = E * sum_i f_i * P_i`` where ``f_i`` is the fraction of routed
    tokens dispatched to expert ``i`` and ``P_i`` is the mean router
    probability assigned to expert ``i``. Minimized when load is uniform.

    Args:
        router_logits: ``(T, E)`` pre-softmax router scores.
        expert_ids:    ``(T, k)`` selected expert indices (from ``route_top_k``).
        num_experts:   ``E``.
    """
    logits = _arr(router_logits).astype(np.float64)
    eids = _arr(expert_ids).astype(np.int64)
    T = logits.shape[0]
    # P_i: mean softmax probability mass on each expert across all tokens.
    z = logits - logits.max(axis=1, keepdims=True)
    probs = np.exp(z)
    probs /= probs.sum(axis=1, keepdims=True)
    P = probs.mean(axis=0)  # (E,)
    # f_i: fraction of routed slots assigned to each expert.
    f = np.bincount(eids.reshape(-1), minlength=num_experts).astype(np.float64)
    f /= max(eids.size, 1)
    return float(num_experts * np.sum(f * P))


def router_z_loss(router_logits) -> float:
    """Router z-loss (ST-MoE): penalizes large router logits for stability.

    ``loss = mean_t (logsumexp_i logits[t, i])**2``.
    """
    logits = _arr(router_logits).astype(np.float64)
    m = logits.max(axis=1, keepdims=True)
    lse = (m[:, 0] + np.log(np.exp(logits - m).sum(axis=1)))
    return float(np.mean(lse ** 2))


# ─────────────────────────────────────────────────────────────────────────────
# Router
# ─────────────────────────────────────────────────────────────────────────────

class MoERouter(nn.Module):
    """Top-k token router: a single ``Linear`` over hidden → expert logits.

    ``forward(x)`` returns ``(expert_ids (T,k), weights (T,k), logits (T,E))``.
    No hidden state, no registry — the gate is one matmul plus ``route_top_k``.
    """

    def __init__(self, hidden_size: int, num_experts: int, top_k: int,
                 *, dtype: str = "fp32") -> None:
        super().__init__()
        self.num_experts = int(num_experts)
        self.top_k = int(top_k)
        self.gate = nn.Linear(hidden_size, num_experts, bias=False, dtype=dtype)

    def forward(self, x):
        logits = self.gate(_arr(x))            # (T, E)
        expert_ids, weights = route_top_k(logits, self.top_k, normalize=True)
        return expert_ids, weights, logits


# ─────────────────────────────────────────────────────────────────────────────
# MoE feed-forward block (routed experts + shared expert)
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class _ExpertWeights:
    """Holds the routed + shared expert parameters as plain arrays.

    Kept as a flat dataclass (not nested Modules) so the forward call to
    ``moe_forward_naive`` reads top-to-bottom with no indirection.
    """
    w_router: np.ndarray   # (H, E)
    w_gate: np.ndarray     # (E, H, F)
    w_up: np.ndarray       # (E, H, F)
    w_down: np.ndarray     # (E, F, H)
    w_sgate: np.ndarray    # (H, Fs)
    w_sup: np.ndarray      # (H, Fs)
    w_sdown: np.ndarray    # (Fs, H)


class MoEFeedForward(nn.Module):
    """Routed-expert MoE FFN with a shared expert, instantiated directly.

    ``forward(x)`` returns ``(y (T,H), aux)`` where ``aux`` carries the routing
    table and the two auxiliary losses, ready to add to the main loss in the
    training loop. Reference (numpy) forward by default; pass
    ``perf=True`` to route through the packed/grouped Apple-GPU-capable path.
    """

    def __init__(self, hidden_size: int, num_experts: int, top_k: int,
                 expert_intermediate: int, shared_intermediate: int,
                 *, dtype: str = "fp32", seed: int = 0) -> None:
        super().__init__()
        self.hidden_size = int(hidden_size)
        self.num_experts = int(num_experts)
        self.top_k = int(top_k)
        self.router = MoERouter(hidden_size, num_experts, top_k, dtype=dtype)

        rng = np.random.default_rng(seed)
        H, E = hidden_size, num_experts
        F, Fs = expert_intermediate, shared_intermediate
        s = 1.0 / np.sqrt(H)
        # Stored as Parameters so they appear in .parameters()/state_dict and
        # will be hooked by the Tier-2 autodiff tape when the training loop
        # wraps the forward (see loop/rl.py for the integration seam).
        self.w_gate = nn.Parameter((rng.standard_normal((E, H, F)) * s).astype(np.float32))
        self.w_up = nn.Parameter((rng.standard_normal((E, H, F)) * s).astype(np.float32))
        self.w_down = nn.Parameter((rng.standard_normal((E, F, H)) / np.sqrt(F)).astype(np.float32))
        self.w_sgate = nn.Parameter((rng.standard_normal((H, Fs)) * s).astype(np.float32))
        self.w_sup = nn.Parameter((rng.standard_normal((H, Fs)) * s).astype(np.float32))
        self.w_sdown = nn.Parameter((rng.standard_normal((Fs, H)) / np.sqrt(Fs)).astype(np.float32))

    def forward(self, x, *, perf: bool = False):
        xa = _arr(x)
        flat = xa.reshape(-1, self.hidden_size)            # (T, H)
        # Unwrap every Parameter to its numpy buffer up front — the reference
        # routing helpers only peel one ``._data`` level, so hand them arrays.
        w_router = _arr(self.router.gate.weight)            # (H, E)
        wg, wu, wd = _arr(self.w_gate), _arr(self.w_up), _arr(self.w_down)
        wsg, wsu, wsd = _arr(self.w_sgate), _arr(self.w_sup), _arr(self.w_sdown)

        logits = flat @ w_router
        expert_ids, _ = route_top_k(logits, self.top_k, normalize=True)

        if perf:
            from tessera.models.moe_routing import moe_forward
            y, _plan = moe_forward(
                flat, w_router, wg, wu, wd, wsg, wsu, wsd,
                top_k=self.top_k, normalize=True,
            )
            y = _arr(y)
        else:
            y = moe_forward_naive(
                flat, w_router, wg, wu, wd, wsg, wsu, wsd,
                top_k=self.top_k, normalize=True,
            )

        aux = {
            "expert_ids": expert_ids,
            "router_logits": logits,
            "load_balancing_loss": load_balancing_loss(logits, expert_ids, self.num_experts),
            "router_z_loss": router_z_loss(logits),
        }
        return y.reshape(xa.shape).astype(np.float32), aux


# ─────────────────────────────────────────────────────────────────────────────
# Compute-sparse MoE dispatch (real per-expert routing via gather/scatter)
# ─────────────────────────────────────────────────────────────────────────────

def top_k_selection(router_logits, top_k: int) -> np.ndarray:
    """Per-token top-k expert indices ``(N, k)`` from router logits (numpy).

    The discrete selection — the non-differentiable part of routing. Matches the
    stable-argsort tie-break used by ``ops.top_k_routing`` so the sparse weights
    and the dispatch agree on which experts are selected.
    """
    lz = _arr(router_logits)
    return np.argsort(-lz, axis=1, kind="stable")[:, :top_k].astype(np.int64)


def sparse_moe_dispatch(x, routing_weights, topk_idx, num_experts, eye, apply_expert):
    """Compute-sparse MoE combine: each expert runs **only on its routed tokens**.

    Unlike the dense soft-combine (which evaluates every expert on every token
    and zeroes the off-top-k contributions), this gathers each expert's assigned
    tokens, runs that expert's FFN on just those rows, and scatter-adds the
    weighted result back. Total expert work is ``N*k`` rows instead of ``N*E``,
    while the result is *numerically identical* to the dense path.

    Every step is tape-traceable (``ops.gather`` / ``ops.gemm`` / ``ops.mul`` /
    ``ops.scatter_add``), so gradients flow to the experts (through their tokens),
    the router (through ``routing_weights``), and any upstream params. The token
    index sets are data-dependent integer arrays (non-differentiable), exactly
    the part a production MoE computes at runtime.

    Args:
        x:               ``(N, H)`` traced token features.
        routing_weights: ``(N, E)`` traced sparse gate (e.g. ``ops.top_k_routing``).
        topk_idx:        ``(N, k)`` numpy selection (``top_k_selection``).
        num_experts:     ``E``.
        eye:             ``(E, E)`` constant identity (for traced column-select).
        apply_expert:    ``(xe, e) -> (n_e, H)`` traced per-expert FFN.
    Returns:
        ``(N, H)`` combined output.
    """
    xa = _arr(x)
    N, H = xa.shape
    zeros = np.zeros((N, H), dtype=np.float32)
    y = None
    for e in range(num_experts):
        tok = np.nonzero((topk_idx == e).any(axis=1))[0].astype(np.int64)
        if tok.size == 0:
            continue
        xe = ops.gather(x, tok, axis=0)                              # (n_e, H)
        we = ops.gather(ops.gemm(routing_weights, eye[:, e:e + 1]), tok, axis=0)  # (n_e, 1)
        ye = ops.mul(apply_expert(xe, e), we)                        # (n_e, H)
        contrib = ops.scatter_add(zeros, tok, ye, axis=0)           # (N, H)
        y = contrib if y is None else ops.add(y, contrib)
    if y is None:  # degenerate: nothing routed anywhere
        return ops.mul(x, np.zeros((N, 1), np.float32))
    return y
