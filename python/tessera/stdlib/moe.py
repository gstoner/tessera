"""``tessera.stdlib.moe`` — compiler-owned MoE dispatch (the M2 lowering surface).

``tessera.models.moe_routing`` proved the *exact* (no-capacity) route → pack →
grouped-SwiGLU → combine pipeline.  Production MoE for Kimi-K2 / DeepSeek-V3.2 /
GLM-5.2 / MiniMax-M3 adds the parts a real router/dispatcher must own:

* **capacity & bucketing** — each expert processes at most ``capacity`` tokens;
  overflow slots are dropped (their combine contribution is zero), so the launch
  buffer is a fixed ``E × capacity`` shape (:func:`compute_capacity`,
  :func:`plan_dispatch`).
* **token permutation / expert packing** — kept slots sorted into contiguous
  per-expert groups (:func:`dispatch`), inverse-scattered on combine
  (:func:`combine`).
* **shared-expert fusion + residual combine** — the always-on shared FFN added
  to the routed output (:func:`shared_expert_swiglu`, :func:`moe_forward`).
* **quantized expert compute** — the grouped SwiGLU run on packed INT4/FP8
  experts via ``stdlib.quant`` (:func:`moe_swiglu_quantized`), the M1↔M2 join.

These are reference semantics (numpy) with the *exact* contract a fused kernel
must implement; the load-balance / z-loss auxiliaries already live in
``tessera.losses`` / the Apple GPU lane.  This module reuses the proven routing
primitives from ``models.moe_routing`` rather than re-deriving them.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from ..models.moe_routing import route_top_k
from . import quant as _quant


def _arr(x) -> np.ndarray:
    return np.asarray(x._data if hasattr(x, "_data") else x)


# ── capacity ─────────────────────────────────────────────────────────────────
def compute_capacity(num_tokens: int, top_k: int, num_experts: int,
                     capacity_factor: float) -> int:
    """Per-expert token capacity = ``ceil(factor · T · k / E)`` (≥ 1).

    The Switch/GShard convention: with perfectly balanced routing each expert
    sees ``T·k/E`` tokens; ``capacity_factor`` (typically 1.0–1.5) gives slack
    for imbalance before tokens drop.
    """
    if num_tokens <= 0:
        return 0
    base = capacity_factor * num_tokens * top_k / max(1, num_experts)
    return max(1, int(math.ceil(base)))


@dataclass(frozen=True)
class DispatchPlan:
    """Capacity-aware dispatch: which routed slots survive, and the contiguous
    per-expert packing over the survivors."""

    expert_ids: np.ndarray      # (T, k) int64 selected experts
    weights: np.ndarray         # (T, k) float route weights
    kept_mask: np.ndarray       # (T, k) bool — False = dropped (over capacity)
    sort_perm: np.ndarray       # (S,) int64 — kept token-major slot ids, expert-sorted
    group_sizes: np.ndarray     # (E,) int64 — kept tokens per expert (≤ capacity)
    num_tokens: int
    top_k: int
    capacity: int | None

    @property
    def num_kept(self) -> int:
        return int(self.sort_perm.shape[0])

    @property
    def drop_fraction(self) -> float:
        total = self.num_tokens * self.top_k
        return 0.0 if total == 0 else 1.0 - self.num_kept / total


def plan_dispatch(expert_ids, weights, num_experts: int,
                  *, capacity: int | None = None) -> DispatchPlan:
    """Build a capacity-aware :class:`DispatchPlan` from routing decisions.

    Slots are processed in token-major order; once an expert reaches
    ``capacity`` the remaining tokens routed to it are dropped (``capacity=None``
    keeps everything — the exact pipeline).  Deterministic: earlier tokens win
    a contested expert slot.
    """
    eids = _arr(expert_ids).astype(np.int64)
    w = _arr(weights).astype(np.float32)
    if eids.ndim != 2:
        raise ValueError("expert_ids must be (T, k)")
    T, k = eids.shape
    if eids.size and (eids.min() < 0 or eids.max() >= num_experts):
        raise ValueError("expert id out of range")

    flat_e = eids.reshape(-1)                          # token-major slot → expert
    kept = np.ones(flat_e.shape[0], dtype=bool)
    if capacity is not None:
        seen = np.zeros(num_experts, dtype=np.int64)
        for s in range(flat_e.shape[0]):
            e = int(flat_e[s])
            if seen[e] >= capacity:
                kept[s] = False
            else:
                seen[e] += 1
    kept_slots = np.nonzero(kept)[0]
    # stable sort kept slots by expert → contiguous per-expert groups
    order = np.argsort(flat_e[kept_slots], kind="stable")
    sort_perm = kept_slots[order].astype(np.int64)
    group_sizes = np.bincount(flat_e[sort_perm], minlength=num_experts).astype(np.int64)
    return DispatchPlan(
        expert_ids=eids, weights=w, kept_mask=kept.reshape(T, k),
        sort_perm=sort_perm, group_sizes=group_sizes,
        num_tokens=T, top_k=k, capacity=capacity)


def dispatch(x, plan: DispatchPlan) -> np.ndarray:
    """Gather token rows for the kept, expert-sorted slots → ``(S, H)``."""
    xa = _arr(x)
    token_of_slot = (plan.sort_perm // plan.top_k)
    return xa[token_of_slot]


def combine(y_packed, plan: DispatchPlan) -> np.ndarray:
    """Scatter packed expert outputs back to ``(T, H)`` token order, weighting
    each kept slot by its route weight.  Dropped slots contribute zero."""
    yp = _arr(y_packed)
    H = yp.shape[-1]
    out = np.zeros((plan.num_tokens, H), dtype=np.float64)
    flat_w = plan.weights.reshape(-1)
    for i, slot in enumerate(plan.sort_perm):
        t = int(slot) // plan.top_k
        out[t] += float(flat_w[int(slot)]) * yp[i]
    return out


# ── expert compute (dense + quantized) ───────────────────────────────────────
def _swiglu(x, w_gate, w_up, w_down) -> np.ndarray:
    g = x @ w_gate
    u = x @ w_up
    h = (g * (1.0 / (1.0 + np.exp(-g)))) * u
    return h @ w_down


def shared_expert_swiglu(x, w_gate, w_up, w_down) -> np.ndarray:
    """The always-on shared expert: a single SwiGLU FFN over every token."""
    return _swiglu(_arr(x).astype(np.float64), _arr(w_gate).astype(np.float64),
                   _arr(w_up).astype(np.float64), _arr(w_down).astype(np.float64))


def grouped_swiglu(x_packed, w_gate, w_up, w_down, group_sizes) -> np.ndarray:
    """Dense grouped SwiGLU over contiguous per-expert token groups → ``(S, H)``."""
    xp = _arr(x_packed).astype(np.float64)
    wg, wu, wd = (_arr(w).astype(np.float64) for w in (w_gate, w_up, w_down))
    gs = _arr(group_sizes).astype(np.int64)
    out = np.zeros((xp.shape[0], wd.shape[2]), dtype=np.float64)
    off = 0
    for e in range(wg.shape[0]):
        n = int(gs[e])
        if n:
            out[off:off + n] = _swiglu(xp[off:off + n], wg[e], wu[e], wd[e])
        off += n
    return out


def moe_swiglu_quantized(x_packed, gate_experts, up_experts, down_experts,
                         group_sizes, *, backend: str = "reference") -> np.ndarray:
    """Grouped SwiGLU where each expert's gate/up/down are
    :class:`stdlib.quant.PackedQuantTensor`s — the M1↔M2 quantized MoE path.

    ``gate_experts`` / ``up_experts`` / ``down_experts`` are length-``E`` lists
    of packed weights ``(H, F)``, ``(H, F)``, ``(F, H)``.  Each grouped GEMM runs
    through :func:`stdlib.quant.dequant_grouped_gemm` (fp32 accumulate over the
    packed codes + per-group scales).
    """
    xp = np.asarray(_arr(x_packed), dtype=np.float32)
    gs = _arr(group_sizes).astype(np.int64)
    gate = _quant.dequant_grouped_gemm(xp, gate_experts, gs, backend=backend)
    up = _quant.dequant_grouped_gemm(xp, up_experts, gs, backend=backend)
    g = gate.astype(np.float32)
    hidden = (g * (1.0 / (1.0 + np.exp(-g))) * up).astype(np.float32)
    return _quant.dequant_grouped_gemm(hidden, down_experts, gs, backend=backend)


# ── full layer ───────────────────────────────────────────────────────────────
@dataclass(frozen=True)
class MoEResult:
    y: np.ndarray               # (T, H) layer output (routed + shared)
    plan: DispatchPlan
    routed: np.ndarray          # (T, H) routed-experts contribution
    shared: np.ndarray | None   # (T, H) shared-expert contribution (or None)


def moe_forward(
    x, w_router, w_gate, w_up, w_down,
    *, top_k: int,
    shared=None,                # (w_sgate, w_sup, w_sdown) or None
    capacity_factor: float | None = None,
    normalize: bool = True,
) -> MoEResult:
    """Capacity-aware MoE layer forward: route → dispatch(capacity) → grouped
    SwiGLU → combine → + shared expert.

    ``capacity_factor=None`` runs the exact (no-drop) pipeline; a finite factor
    caps each expert at :func:`compute_capacity` tokens.  Returns a
    :class:`MoEResult` carrying the dispatch plan (so drop stats / load balance
    are inspectable).
    """
    xa = _arr(x).astype(np.float64)
    wr = _arr(w_router).astype(np.float64)
    E = wr.shape[1]
    T = xa.shape[0]

    expert_ids, weights = route_top_k(xa @ wr, top_k, normalize=normalize)
    cap = (compute_capacity(T, top_k, E, capacity_factor)
           if capacity_factor is not None else None)
    plan = plan_dispatch(expert_ids, weights, E, capacity=cap)

    x_packed = dispatch(xa, plan)
    y_packed = grouped_swiglu(x_packed, _arr(w_gate), _arr(w_up), _arr(w_down),
                              plan.group_sizes)
    routed = combine(y_packed, plan)

    shared_out = None
    y = routed
    if shared is not None:
        shared_out = shared_expert_swiglu(xa, shared[0], shared[1], shared[2])
        y = routed + shared_out
    return MoEResult(y=y, plan=plan, routed=routed, shared=shared_out)


__all__ = [
    "compute_capacity",
    "DispatchPlan",
    "plan_dispatch",
    "dispatch",
    "combine",
    "shared_expert_swiglu",
    "grouped_swiglu",
    "moe_swiglu_quantized",
    "MoEResult",
    "moe_forward",
]
