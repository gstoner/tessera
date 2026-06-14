"""G1 + G2 tests: trainable embeddings (ops.embedding) and differentiable
hard top-k routing (ops.top_k_routing), proven end-to-end on a tape-traceable
hard-MoE language model.
"""

from __future__ import annotations

import numpy as np

import tessera as ts
from tessera import ops
from tessera.train.loop.optimizer import adamw_step
from tessera.train.models.traced_hard_moe_lm import (
    TracedHardMoEConfig,
    TracedHardMoELM,
    traced_ce_loss,
)


# ── G1: ops.embedding ────────────────────────────────────────────────────────

def test_embedding_grad_scatter_adds_into_table():
    table = ts.nn.Parameter(np.random.default_rng(0).standard_normal((8, 4)).astype(np.float32))
    table._grad = None
    with ts.autodiff.tape() as t:
        rows = ops.embedding(table, np.array([2, 2, 5], np.int64))
        t.backward(ops.reduce(rows, op="sum"))
    g = table.grad.numpy()
    assert np.allclose(g[2], 2.0)       # index 2 used twice → accumulates
    assert np.allclose(g[5], 1.0)
    assert np.allclose(g[0], 0.0)       # unused rows get zero grad


# ── G2: ops.top_k_routing ────────────────────────────────────────────────────

def test_top_k_routing_forward_is_sparse_softmax():
    z = np.array([[1.0, 3.0, 2.0, 0.5]], np.float32)
    w = np.asarray(ops.top_k_routing(z, k=2))
    sm = np.exp([3.0, 2.0]); sm /= sm.sum()
    np.testing.assert_allclose(w[0], [0.0, sm[0], sm[1], 0.0], rtol=1e-6)
    # weights sum to 1 over the selected set
    assert np.isclose(w.sum(), 1.0)


def test_top_k_routing_vjp_matches_central_difference():
    W = ts.nn.Parameter(np.random.default_rng(1).standard_normal((3, 5)).astype(np.float32))
    target = np.random.default_rng(2).standard_normal((3, 5)).astype(np.float32)

    def loss_val(z):
        order = np.argsort(-z, axis=-1, kind="stable")[:, :2]
        mask = np.zeros_like(z, bool); np.put_along_axis(mask, order, True, axis=-1)
        masked = np.where(mask, z, -np.inf); m = masked.max(-1, keepdims=True)
        e = np.where(mask, np.exp(masked - m), 0.0); w = e / e.sum(-1, keepdims=True)
        return float(np.sum(w * target))

    W._grad = None
    with ts.autodiff.tape() as t:
        w = ops.top_k_routing(W, k=2)
        t.backward(ops.reduce(ops.mul(w, target), op="sum"))
    g = W.grad.numpy()
    base = W.numpy().copy()
    eps = 1e-3
    for (i, j) in [(0, 0), (1, 3), (2, 4), (0, 2)]:
        wp = base.copy(); wp[i, j] += eps
        wm = base.copy(); wm[i, j] -= eps
        num = (loss_val(wp) - loss_val(wm)) / (2 * eps)
        assert abs(g[i, j] - num) < 1e-2


# ── G1 + G2: end-to-end hard-MoE LM trains ───────────────────────────────────

def test_hard_moe_lm_trains_end_to_end():
    cfg = TracedHardMoEConfig()
    model = TracedHardMoELM(cfg, seed=1)
    rng = np.random.default_rng(3)
    N = 32
    ids = rng.integers(0, cfg.vocab_size, size=N).astype(np.int64)
    targets = rng.integers(0, cfg.vocab_size, size=N)
    onehot = np.eye(cfg.vocab_size, dtype=np.float32)[targets]

    def loss_fn():
        return traced_ce_loss(model.logits(ids), onehot)

    # All param families must receive gradients (embedding, router, experts, head).
    for p in model.parameters():
        p._grad = None
    with ts.autodiff.tape() as t:
        t.backward(loss_fn())
    named = dict(model.named_parameters())
    assert named["embed"].grad is not None          # G1: embedding trains
    assert named["w_router"].grad is not None        # G2: router trains through hard top-k
    assert named["w_gate_0"].grad is not None
    assert named["w_out"].grad is not None

    opt = None
    losses = []
    for _ in range(25):
        loss, opt = adamw_step(model, loss_fn, opt, lr=0.05)
        losses.append(loss)
    assert all(np.isfinite(v) for v in losses)
    assert losses[-1] < losses[0] - 0.05            # CE clearly decreased


# ── Compute-sparse MoE dispatch (gather/scatter per-expert routing) ──────────

def test_sparse_dispatch_equals_dense_combine():
    cfg = TracedHardMoEConfig()
    model = TracedHardMoELM(cfg, seed=2)
    ids = np.random.default_rng(4).integers(0, cfg.vocab_size, size=24).astype(np.int64)
    dense = np.asarray(model.logits(ids, dispatch="dense"))
    sparse = np.asarray(model.logits(ids, dispatch="sparse"))
    # Sparse dispatch must be numerically identical to the dense soft-combine.
    np.testing.assert_allclose(sparse, dense, atol=1e-5)


def test_sparse_dispatch_grads_flow_to_all_params():
    cfg = TracedHardMoEConfig()
    model = TracedHardMoELM(cfg, seed=3)
    rng = np.random.default_rng(5)
    ids = rng.integers(0, cfg.vocab_size, size=24).astype(np.int64)
    onehot = np.eye(cfg.vocab_size, dtype=np.float32)[rng.integers(0, cfg.vocab_size, size=24)]
    for p in model.parameters():
        p._grad = None
    with ts.autodiff.tape() as t:
        t.backward(traced_ce_loss(model.logits(ids, dispatch="sparse"), onehot))
    named = dict(model.named_parameters())
    # Through gather/scatter_add: embedding, router, experts, head all get grads.
    for key in ["embed", "w_router", "w_gate_0", "w_down_0", "w_out"]:
        assert named[key].grad is not None, key


def test_sparse_dispatch_trains_end_to_end():
    cfg = TracedHardMoEConfig()
    model = TracedHardMoELM(cfg, seed=1)
    rng = np.random.default_rng(6)
    ids = rng.integers(0, cfg.vocab_size, size=32).astype(np.int64)
    onehot = np.eye(cfg.vocab_size, dtype=np.float32)[rng.integers(0, cfg.vocab_size, size=32)]

    def loss_fn():
        return traced_ce_loss(model.logits(ids, dispatch="sparse"), onehot)

    opt = None
    losses = []
    for _ in range(25):
        loss, opt = adamw_step(model, loss_fn, opt, lr=0.05)
        losses.append(loss)
    assert all(np.isfinite(v) for v in losses)
    assert losses[-1] < losses[0] - 0.05
