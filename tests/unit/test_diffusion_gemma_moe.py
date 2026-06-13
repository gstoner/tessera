"""DiffusionGemma Phase B — MoE routing & packing.

Covers the work plan's MoE test group:
  * top-8 router with ties / empty experts / saturated experts / shared combine;
  * packing/scatter round-trip preserving token order;
  * native-vs-reference parity at production dims (small T, then canvas-size 256).
"""

from __future__ import annotations

import numpy as np
import pytest

from tessera.models.diffusion_gemma import DiffusionGemmaConfig
from tessera.models import moe_routing as mr


def _x(T, H, seed=0):
    rng = np.random.default_rng(seed)
    return (rng.standard_normal((T, H)) / np.sqrt(H)).astype(np.float32)


# ── Routing ──────────────────────────────────────────────────────────────────

def test_top_k_selects_k_distinct_experts():
    rng = np.random.default_rng(1)
    logits = rng.standard_normal((16, 128))
    ids, w = mr.route_top_k(logits, top_k=8)
    assert ids.shape == (16, 8) and w.shape == (16, 8)
    for row in ids:
        assert len(set(row.tolist())) == 8           # distinct experts
    np.testing.assert_allclose(w.sum(axis=1), 1.0, atol=1e-5)  # normalized


def test_top_k_tie_break_is_lowest_index():
    # All-equal logits → the first k expert indices win, deterministically.
    logits = np.zeros((4, 128))
    ids, _ = mr.route_top_k(logits, top_k=8)
    for row in ids:
        assert row.tolist() == [0, 1, 2, 3, 4, 5, 6, 7]


def test_router_weights_match_softmax_over_selected():
    rng = np.random.default_rng(2)
    logits = rng.standard_normal((5, 128))
    ids, w = mr.route_top_k(logits, top_k=8)
    sel = np.take_along_axis(logits, ids, axis=1)
    e = np.exp(sel - sel.max(axis=1, keepdims=True))
    np.testing.assert_allclose(w, (e / e.sum(axis=1, keepdims=True)), atol=1e-5)


# ── Packing / scatter ────────────────────────────────────────────────────────

def test_packing_group_sizes_sum_to_slots():
    rng = np.random.default_rng(3)
    ids, _ = mr.route_top_k(rng.standard_normal((20, 128)), top_k=8)
    plan = mr.plan_packing(ids, 128)
    assert int(plan.group_sizes.sum()) == 20 * 8
    assert plan.group_sizes.shape == (128,)


def test_packing_round_trip_preserves_token_order():
    rng = np.random.default_rng(4)
    ids, _ = mr.route_top_k(rng.standard_normal((20, 128)), top_k=8)
    plan = mr.plan_packing(ids, 128)
    assert plan.round_trips()
    # pack then scatter (identity expert) recovers the per-slot tokens.
    H = 32
    x = _x(20, H, seed=9)
    packed = mr.pack_tokens(x, plan)
    # unsorting the packed rows must restore token-major slot order.
    unsorted = packed[plan.inverse_perm]
    token_of_slot = np.repeat(np.arange(20), 8)
    np.testing.assert_array_equal(unsorted, x[token_of_slot])


def test_empty_and_saturated_experts():
    # Saturated: every token → expert 0. group 0 holds all slots, rest empty.
    T, k, E = 12, 8, 128
    ids = np.zeros((T, k), dtype=np.int64)
    plan = mr.plan_packing(ids, E)
    assert int(plan.group_sizes[0]) == T * k
    assert int((plan.group_sizes == 0).sum()) == E - 1
    assert plan.round_trips()


# ── Native-vs-reference parity at production dims ────────────────────────────

@pytest.mark.parametrize("T", [4, 32])
def test_packed_matches_naive_small(T):
    cfg = DiffusionGemmaConfig()
    w = mr.synthetic_moe_weights(cfg, seed=7)
    x = _x(T, cfg.hidden_size, seed=T)
    y, plan = mr.moe_forward(x, **w, top_k=cfg.num_experts_per_tok)
    y_naive = mr.moe_forward_naive(x, **w, top_k=cfg.num_experts_per_tok)
    assert y.shape == (T, cfg.hidden_size)
    assert int(plan.group_sizes.sum()) == T * cfg.num_experts_per_tok
    np.testing.assert_allclose(y, y_naive, atol=1e-9)


def test_packed_matches_naive_at_canvas_size():
    # Production dims at the 256-token canvas.
    cfg = DiffusionGemmaConfig()
    w = mr.synthetic_moe_weights(cfg, seed=11)
    x = _x(cfg.canvas_size, cfg.hidden_size, seed=123)
    y, plan = mr.moe_forward(x, **w, top_k=cfg.num_experts_per_tok)
    y_naive = mr.moe_forward_naive(x, **w, top_k=cfg.num_experts_per_tok)
    assert y.shape == (cfg.canvas_size, cfg.hidden_size)
    np.testing.assert_allclose(y, y_naive, atol=1e-8)


def test_shared_expert_contributes():
    # Zeroing the routed experts leaves only the shared-expert output → nonzero,
    # and equals a standalone shared SwiGLU. Confirms the combine adds shared.
    cfg = DiffusionGemmaConfig()
    w = mr.synthetic_moe_weights(cfg, seed=5)
    x = _x(8, cfg.hidden_size, seed=2)
    w_zero = dict(w)
    w_zero["w_down"] = np.zeros_like(w["w_down"])     # routed experts output 0
    y, _ = mr.moe_forward(x, **w_zero, top_k=cfg.num_experts_per_tok)
    shared = mr._swiglu(x.astype(np.float64), w["w_shared_gate"].astype(np.float64),
                        w["w_shared_up"].astype(np.float64),
                        w["w_shared_down"].astype(np.float64))
    np.testing.assert_allclose(y, shared, atol=1e-9)
    assert np.abs(y).sum() > 0
