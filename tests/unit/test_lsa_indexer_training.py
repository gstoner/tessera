"""Gap 3 — differentiable indexer for LSA training.

The hard selector (`memory_index_select`) is non-differentiable. Indexer keys
are instead trained through the differentiable scoring head `memory_index_score`
(closed-form VJP+JVP, finite-difference verified) or the straight-through
`memory_index_select_ste` (hard forward, sigmoid backward). The training loop
itself stays in user code. See `docs/audit/domain/archive/lsa_scope.md`.
"""

from __future__ import annotations

import numpy as np

import tessera as ts
from tessera import lsa
from tessera.autodiff.jvp import get_jvp
from tessera.autodiff.vjp import get_vjp
from tessera.compiler.op_catalog import get_op_spec
from tessera.compiler.primitive_coverage import coverage_for


def _keys_query(seed=0, B=1, H=2, nb=4, Sq=6, Dk=8):
    rng = np.random.default_rng(seed)
    keys = rng.standard_normal((B, H, nb, Dk))
    query = rng.standard_normal((B, H, Sq, Dk))
    return keys, query


def _num_grad(fn, x, eps=1e-6):
    g = np.zeros_like(x)
    it = np.nditer(x, flags=["multi_index"], op_flags=["readwrite"])
    while not it.finished:
        idx = it.multi_index
        plus, minus = x.copy(), x.copy()
        plus[idx] += eps
        minus[idx] -= eps
        g[idx] = (fn(plus) - fn(minus)) / (2.0 * eps)
        it.iternext()
    return g


def test_in_catalog_and_coverage():
    for name in ("memory_index_score", "memory_index_select_ste"):
        assert get_op_spec(name) is not None
    score = coverage_for("memory_index_score").contract_status
    assert score["vjp"] == "complete" and score["jvp"] == "complete"
    ste = coverage_for("memory_index_select_ste").contract_status
    # STE: straight-through VJP exists; forward-mode JVP is not_applicable.
    assert ste["vjp"] == "complete" and ste["jvp"] == "not_applicable"


def test_score_is_sigmoid_of_scaled_qk():
    keys, query = _keys_query()
    probs = ts.ops.memory_index_score(keys, query)
    sc = 1.0 / np.sqrt(keys.shape[-1])
    s = np.matmul(query, np.swapaxes(keys, -1, -2)) * sc
    np.testing.assert_allclose(probs, 1.0 / (1.0 + np.exp(-s)), atol=1e-12)
    assert ((probs >= 0) & (probs <= 1)).all()


def test_score_vjp_matches_finite_difference():
    keys, query = _keys_query(seed=1)
    dout = np.ones((1, 2, 6, 4))  # (B,H,S_q,nb)
    dk, dq = get_vjp("memory_index_score")(dout, keys, query)
    exp_k = _num_grad(lambda k: float(lsa.memory_index_score(k, query).sum()), keys)
    exp_q = _num_grad(lambda q: float(lsa.memory_index_score(keys, q).sum()), query)
    np.testing.assert_allclose(dk, exp_k, atol=1e-5)
    np.testing.assert_allclose(dq, exp_q, atol=1e-5)


def test_score_jvp_matches_finite_difference():
    keys, query = _keys_query(seed=2)
    dk = np.ones_like(keys) * 0.1
    dq = np.ones_like(query) * 0.05
    _, tangent = get_jvp("memory_index_score")((keys, query), (dk, dq))
    eps = 1e-6
    plus = lsa.memory_index_score(keys + eps * dk, query + eps * dq)
    minus = lsa.memory_index_score(keys - eps * dk, query - eps * dq)
    np.testing.assert_allclose(tangent, (plus - minus) / (2.0 * eps), atol=1e-5)


def test_ste_forward_is_hard_mask_backward_is_straight_through():
    keys, query = _keys_query(seed=3)
    # Forward: hard 0/1 == threshold of the soft score.
    mask = ts.ops.memory_index_select_ste(keys, query, threshold=0.5)
    soft = lsa.memory_index_score(keys, query)
    np.testing.assert_array_equal(mask, (soft >= 0.5).astype(np.float64))
    assert set(np.unique(mask)).issubset({0.0, 1.0})
    # Backward: STE gradient == the smooth score's gradient (straight-through).
    dout = np.ones_like(mask)
    dk_ste, dq_ste = get_vjp("memory_index_select_ste")(dout, keys, query)
    dk_soft, dq_soft = get_vjp("memory_index_score")(dout, keys, query)
    np.testing.assert_allclose(dk_ste, dk_soft, atol=1e-12)
    np.testing.assert_allclose(dq_ste, dq_soft, atol=1e-12)


def test_indexer_keys_are_trainable_gradient_descent_reduces_loss():
    # A tiny standalone training loop (lives in user code): push the indexer to
    # *select* a target block by maximizing its probability. Loss decreases and
    # the target block's mean probability rises — proof the keys are learnable.
    rng = np.random.default_rng(7)
    B, H, nb, Sq, Dk = 1, 1, 4, 4, 8
    keys = rng.standard_normal((B, H, nb, Dk)) * 0.1
    query = rng.standard_normal((B, H, Sq, Dk))
    target = 2  # want every query to select block 2

    def loss_and_grad(keys):
        probs = lsa.memory_index_score(keys, query)          # (B,H,Sq,nb)
        # BCE pushing block `target` → 1, others → 0.
        tgt = np.zeros_like(probs)
        tgt[..., target] = 1.0
        eps = 1e-7
        loss = -np.mean(tgt * np.log(probs + eps) + (1 - tgt) * np.log(1 - probs + eps))
        dprobs = -(tgt / (probs + eps) - (1 - tgt) / (1 - probs + eps)) / probs.size
        dk, _ = get_vjp("memory_index_score")(dprobs, keys, query)
        return loss, dk

    loss0, _ = loss_and_grad(keys)
    lr = 5.0
    for _ in range(200):
        loss, dk = loss_and_grad(keys)
        keys = keys - lr * dk
    loss1, _ = loss_and_grad(keys)
    assert loss1 < loss0 * 0.5  # training reduced the loss substantially
    final = lsa.memory_index_score(keys, query)[..., target].mean()
    assert final > 0.9         # the indexer now selects the target block
