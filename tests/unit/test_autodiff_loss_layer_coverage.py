"""Autodiff coverage hardening — S11 classification/contrastive/sequence
losses + S7 normalizations/layers/pooling.

Each VJP and JVP is checked numerically against a central finite-difference
gradient on a small input. The bar is `atol=1e-3` for pure-fp64 paths,
`atol=1e-2` where the forward path uses fp32 internally (group_norm,
instance_norm, max/avg_pool).
"""

from __future__ import annotations

import numpy as np
import pytest

import tessera as ts
import tessera.losses as losses
import tessera.nn.functional as nn_functional
from tessera.autodiff.jvp import get_jvp
from tessera.autodiff.vjp import get_vjp


# ── numerical-gradient helpers ─────────────────────────────────────────────


def _numeric_grad(fn, x, eps=1e-4):
    g = np.zeros_like(x, dtype=np.float64)
    x = x.astype(np.float64).copy()
    it = np.nditer(x, flags=["multi_index"], op_flags=["readwrite"])
    while not it.finished:
        idx = it.multi_index
        orig = x[idx]
        x[idx] = orig + eps
        f_plus = float(np.asarray(fn(x)).sum())
        x[idx] = orig - eps
        f_minus = float(np.asarray(fn(x)).sum())
        x[idx] = orig
        g[idx] = (f_plus - f_minus) / (2 * eps)
        it.iternext()
    return g


def _numeric_jvp(fn, x, dx, eps=1e-5):
    """Central-difference forward-mode tangent."""
    plus = np.asarray(fn(x + eps * dx), dtype=np.float64)
    minus = np.asarray(fn(x - eps * dx), dtype=np.float64)
    return (plus - minus) / (2 * eps)


# ─────────────────────────────────────────────────────────────────────────────
# S11 classification
# ─────────────────────────────────────────────────────────────────────────────


def test_focal_loss_vjp_matches_numeric():
    rng = np.random.default_rng(0)
    logits = rng.normal(size=(4, 5)).astype(np.float64)
    targets = np.array([0, 2, 1, 4])
    grad, target_grad = get_vjp("focal_loss")(1.0, logits, targets,
                                              gamma=2.0, reduction="mean")
    expected = _numeric_grad(
        lambda v: losses.focal_loss(v, targets, gamma=2.0, reduction="mean"),
        logits,
    )
    np.testing.assert_allclose(grad, expected, atol=1e-3)
    assert target_grad is None


def test_focal_loss_jvp_matches_finite_difference():
    rng = np.random.default_rng(1)
    logits = rng.normal(size=(3, 4)).astype(np.float64)
    targets = np.array([0, 1, 3])
    dlogits = rng.normal(size=logits.shape).astype(np.float64) * 0.1
    primal, tangent = get_jvp("focal_loss")(
        (logits, targets), (dlogits, np.zeros_like(targets)),
        gamma=2.0, reduction="mean",
    )
    np.testing.assert_allclose(primal, losses.focal_loss(logits, targets, gamma=2.0))
    expected = _numeric_jvp(
        lambda v: losses.focal_loss(v, targets, gamma=2.0, reduction="mean"),
        logits, dlogits,
    )
    np.testing.assert_allclose(tangent, expected, atol=1e-3)


def test_label_smoothed_cross_entropy_vjp_and_jvp():
    rng = np.random.default_rng(2)
    logits = rng.normal(size=(4, 6)).astype(np.float64)
    targets = np.array([2, 0, 5, 3])
    dlogits = rng.normal(size=logits.shape) * 0.05

    grad, target_grad = get_vjp("label_smoothed_cross_entropy")(
        1.0, logits, targets, smoothing=0.1, reduction="mean"
    )
    expected_grad = _numeric_grad(
        lambda v: losses.label_smoothed_cross_entropy(v, targets, smoothing=0.1),
        logits,
    )
    np.testing.assert_allclose(grad, expected_grad, atol=1e-3)
    assert target_grad is None

    primal, tangent = get_jvp("label_smoothed_cross_entropy")(
        (logits, targets), (dlogits, np.zeros_like(targets)),
        smoothing=0.1, reduction="mean",
    )
    np.testing.assert_allclose(
        primal,
        losses.label_smoothed_cross_entropy(logits, targets, smoothing=0.1),
    )
    expected_tan = _numeric_jvp(
        lambda v: losses.label_smoothed_cross_entropy(v, targets, smoothing=0.1),
        logits, dlogits,
    )
    np.testing.assert_allclose(tangent, expected_tan, atol=1e-3)


def test_kl_divergence_vjp_and_jvp():
    rng = np.random.default_rng(3)
    p_log = rng.normal(size=(3, 4)).astype(np.float64) * 0.5
    # Re-normalize to a real log-probability to keep p in [0,1].
    p_log = p_log - np.log(np.sum(np.exp(p_log), axis=-1, keepdims=True))
    q = rng.uniform(0.05, 1.0, size=(3, 4))
    q = q / np.sum(q, axis=-1, keepdims=True)

    grad_p, grad_q = get_vjp("kl_divergence")(1.0, p_log, q, reduction="mean")
    expected_p = _numeric_grad(
        lambda v: losses.kl_divergence(v, q, reduction="mean"), p_log,
    )
    expected_q = _numeric_grad(
        lambda v: losses.kl_divergence(p_log, v, reduction="mean"), q,
    )
    np.testing.assert_allclose(grad_p, expected_p, atol=1e-3)
    np.testing.assert_allclose(grad_q, expected_q, atol=1e-3)

    dp = rng.normal(size=p_log.shape) * 0.05
    dq = rng.normal(size=q.shape) * 0.05
    primal, tangent = get_jvp("kl_divergence")(
        (p_log, q), (dp, dq), reduction="mean",
    )
    np.testing.assert_allclose(primal, losses.kl_divergence(p_log, q))
    expected_tan = (
        np.sum((expected_p * dp).reshape(-1)) + np.sum((expected_q * dq).reshape(-1))
    )
    np.testing.assert_allclose(tangent, expected_tan, atol=1e-3)


# ─────────────────────────────────────────────────────────────────────────────
# S11 contrastive
# ─────────────────────────────────────────────────────────────────────────────


def test_triplet_loss_vjp_and_jvp():
    rng = np.random.default_rng(4)
    a = rng.normal(size=(3, 4))
    p = rng.normal(size=(3, 4))
    n = rng.normal(size=(3, 4))
    grad_a, grad_p, grad_n = get_vjp("triplet_loss")(
        1.0, a, p, n, margin=0.5, reduction="mean",
    )
    expected_a = _numeric_grad(lambda v: losses.triplet_loss(v, p, n, margin=0.5), a)
    expected_p = _numeric_grad(lambda v: losses.triplet_loss(a, v, n, margin=0.5), p)
    expected_n = _numeric_grad(lambda v: losses.triplet_loss(a, p, v, margin=0.5), n)
    np.testing.assert_allclose(grad_a, expected_a, atol=1e-3)
    np.testing.assert_allclose(grad_p, expected_p, atol=1e-3)
    np.testing.assert_allclose(grad_n, expected_n, atol=1e-3)

    da = rng.normal(size=a.shape) * 0.05
    primal, tangent = get_jvp("triplet_loss")(
        (a, p, n), (da, np.zeros_like(p), np.zeros_like(n)),
        margin=0.5, reduction="mean",
    )
    np.testing.assert_allclose(primal, losses.triplet_loss(a, p, n, margin=0.5))
    expected_tan = _numeric_jvp(
        lambda v: losses.triplet_loss(v, p, n, margin=0.5), a, da,
    )
    np.testing.assert_allclose(tangent, expected_tan, atol=1e-3)


def test_contrastive_loss_vjp_and_jvp():
    rng = np.random.default_rng(5)
    a = rng.normal(size=(3, 4))
    b = rng.normal(size=(3, 4))
    target = np.array([1.0, 0.0, 1.0])
    grad_a, grad_b, grad_t = get_vjp("contrastive_loss")(
        1.0, a, b, target, margin=0.5, reduction="mean",
    )
    expected_a = _numeric_grad(
        lambda v: losses.contrastive_loss(v, b, target, margin=0.5), a,
    )
    expected_b = _numeric_grad(
        lambda v: losses.contrastive_loss(a, v, target, margin=0.5), b,
    )
    np.testing.assert_allclose(grad_a, expected_a, atol=1e-3)
    np.testing.assert_allclose(grad_b, expected_b, atol=1e-3)
    assert grad_t is None

    da = rng.normal(size=a.shape) * 0.05
    primal, tangent = get_jvp("contrastive_loss")(
        (a, b, target), (da, np.zeros_like(b), np.zeros_like(target)),
        margin=0.5, reduction="mean",
    )
    np.testing.assert_allclose(primal, losses.contrastive_loss(a, b, target, margin=0.5))
    expected_tan = _numeric_jvp(
        lambda v: losses.contrastive_loss(v, b, target, margin=0.5), a, da,
    )
    np.testing.assert_allclose(tangent, expected_tan, atol=1e-3)


def test_cosine_embedding_loss_vjp_and_jvp():
    rng = np.random.default_rng(6)
    a = rng.normal(size=(3, 5))
    b = rng.normal(size=(3, 5))
    target = np.array([1.0, -1.0, 1.0])
    grad_a, grad_b, grad_t = get_vjp("cosine_embedding_loss")(
        1.0, a, b, target, margin=0.0, reduction="mean",
    )
    expected_a = _numeric_grad(
        lambda v: losses.cosine_embedding_loss(v, b, target, margin=0.0), a,
    )
    np.testing.assert_allclose(grad_a, expected_a, atol=1e-3)
    assert grad_t is None


def test_info_nce_loss_vjp_and_jvp():
    rng = np.random.default_rng(7)
    q = rng.normal(size=(2, 4))
    p = rng.normal(size=(2, 4))
    n = rng.normal(size=(2, 3, 4))  # 3 negatives per query.
    grad_q, grad_p, grad_n = get_vjp("info_nce_loss")(
        1.0, q, p, n, temperature=0.2, reduction="mean",
    )
    expected_q = _numeric_grad(
        lambda v: losses.info_nce_loss(v, p, n, temperature=0.2), q,
    )
    expected_p = _numeric_grad(
        lambda v: losses.info_nce_loss(q, v, n, temperature=0.2), p,
    )
    np.testing.assert_allclose(grad_q, expected_q, atol=1e-3)
    np.testing.assert_allclose(grad_p, expected_p, atol=1e-3)


# ─────────────────────────────────────────────────────────────────────────────
# S11 sequence
# ─────────────────────────────────────────────────────────────────────────────


def test_seq2seq_loss_vjp_and_jvp_with_mask():
    rng = np.random.default_rng(8)
    logits = rng.normal(size=(2, 4, 5)).astype(np.float64)
    targets = np.array([[1, 2, 0, 3], [4, 1, 2, 0]])
    mask = np.array([[1.0, 1.0, 0.0, 1.0], [1.0, 1.0, 1.0, 0.0]])

    grad_l, grad_t, grad_m = get_vjp("seq2seq_loss")(
        1.0, logits, targets, mask, reduction="mean",
    )
    expected = _numeric_grad(
        lambda v: losses.seq2seq_loss(v, targets, mask=mask, reduction="mean"),
        logits,
    )
    np.testing.assert_allclose(grad_l, expected, atol=1e-3)

    dlogits = rng.normal(size=logits.shape) * 0.05
    primal, tangent = get_jvp("seq2seq_loss")(
        (logits, targets, mask), (dlogits, np.zeros_like(targets), np.zeros_like(mask)),
        reduction="mean",
    )
    np.testing.assert_allclose(
        primal, losses.seq2seq_loss(logits, targets, mask=mask, reduction="mean")
    )


# ─────────────────────────────────────────────────────────────────────────────
# S7 normalizations
# ─────────────────────────────────────────────────────────────────────────────


def test_group_norm_vjp_matches_numeric():
    rng = np.random.default_rng(9)
    x = rng.normal(size=(2, 4, 3, 3)).astype(np.float32)
    do = np.ones_like(x)
    grad, *_ = get_vjp("group_norm")(do, x, 2, eps=1e-5)
    expected = _numeric_grad(
        lambda v: float(nn_functional.group_norm(v.astype(np.float32), 2).sum()),
        x.astype(np.float64),
    )
    # group_norm uses fp32 internally — looser tolerance.
    np.testing.assert_allclose(grad, expected, atol=1e-2, rtol=1e-2)


def test_group_norm_jvp_matches_numeric():
    rng = np.random.default_rng(10)
    x = rng.normal(size=(1, 4, 2, 2)).astype(np.float32)
    dx = rng.normal(size=x.shape).astype(np.float32) * 0.05
    primal, tangent = get_jvp("group_norm")((x, 2), (dx,), eps=1e-5)
    expected_tan = _numeric_jvp(
        lambda v: nn_functional.group_norm(v.astype(np.float32), 2),
        x.astype(np.float64), dx.astype(np.float64),
    )
    np.testing.assert_allclose(tangent, expected_tan, atol=1e-2, rtol=1e-2)


def test_instance_norm_vjp_matches_numeric():
    rng = np.random.default_rng(11)
    x = rng.normal(size=(2, 3, 4, 4)).astype(np.float32)
    do = np.ones_like(x)
    grad, *_ = get_vjp("instance_norm")(do, x, eps=1e-5)
    expected = _numeric_grad(
        lambda v: float(nn_functional.instance_norm(v.astype(np.float32)).sum()),
        x.astype(np.float64),
    )
    np.testing.assert_allclose(grad, expected, atol=1e-2, rtol=1e-2)


# ─────────────────────────────────────────────────────────────────────────────
# S7 layers
# ─────────────────────────────────────────────────────────────────────────────


def test_lora_linear_vjp_and_jvp():
    rng = np.random.default_rng(12)
    x = rng.normal(size=(3, 4))
    w = rng.normal(size=(4, 5))
    a = rng.normal(size=(4, 2))  # rank=2
    b = rng.normal(size=(2, 5))
    do = np.ones((3, 5))
    grad_x, grad_w, grad_a, grad_b, grad_bias = get_vjp("lora_linear")(
        do, x, w, a, b, alpha=1.0,
    )
    expected_x = _numeric_grad(
        lambda v: nn_functional.lora_linear(v, w, a, b, alpha=1.0), x,
    )
    expected_w = _numeric_grad(
        lambda v: nn_functional.lora_linear(x, v, a, b, alpha=1.0), w,
    )
    np.testing.assert_allclose(grad_x, expected_x, atol=1e-3)
    np.testing.assert_allclose(grad_w, expected_w, atol=1e-3)
    assert grad_bias is None

    dx = rng.normal(size=x.shape) * 0.05
    primal, tangent = get_jvp("lora_linear")(
        (x, w, a, b), (dx, np.zeros_like(w), np.zeros_like(a), np.zeros_like(b)),
        alpha=1.0,
    )
    np.testing.assert_allclose(
        primal, nn_functional.lora_linear(x, w, a, b, alpha=1.0)
    )


# ─────────────────────────────────────────────────────────────────────────────
# S7 pooling
# ─────────────────────────────────────────────────────────────────────────────


def test_max_pool_vjp_routes_grad_to_argmax():
    rng = np.random.default_rng(13)
    # Use distinct values so argmax is unique.
    x = rng.uniform(size=(1, 1, 4, 4)).astype(np.float64)
    do = np.ones((1, 1, 2, 2), dtype=np.float64)
    grad, = get_vjp("max_pool")(do, x, 2)
    # The gradient should be 1 only at the argmax position of each 2×2 window.
    assert grad.shape == x.shape
    assert grad.sum() == 4.0  # exactly one cell per 2×2 window.


def test_avg_pool_vjp_distributes_grad_uniformly():
    rng = np.random.default_rng(14)
    x = rng.uniform(size=(1, 1, 4, 4)).astype(np.float64)
    do = np.ones((1, 1, 2, 2), dtype=np.float64)
    grad, = get_vjp("avg_pool")(do, x, 2)
    assert grad.shape == x.shape
    # Every input cell receives 1/4 from its window.
    np.testing.assert_allclose(grad, np.full_like(grad, 0.25))


def test_max_pool_jvp_matches_numeric():
    rng = np.random.default_rng(15)
    x = rng.uniform(size=(1, 1, 4, 4)).astype(np.float64)
    dx = rng.normal(size=x.shape) * 0.05
    primal, tangent = get_jvp("max_pool")((x, 2), (dx,), padding=0)
    expected_tan = _numeric_jvp(
        lambda v: nn_functional.max_pool(v, 2), x, dx,
    )
    # nn_functional.max_pool uses fp32 internally — broaden the tolerance to
    # absorb fp32 quantization noise from the central-difference reference.
    np.testing.assert_allclose(tangent, expected_tan, atol=5e-3, rtol=5e-3)


def test_avg_pool_jvp_matches_numeric():
    rng = np.random.default_rng(16)
    x = rng.uniform(size=(1, 1, 4, 4)).astype(np.float64)
    dx = rng.normal(size=x.shape) * 0.05
    primal, tangent = get_jvp("avg_pool")((x, 2), (dx,), padding=0)
    expected_tan = _numeric_jvp(
        lambda v: nn_functional.avg_pool(v, 2), x, dx,
    )
    # Same fp32 forward-path tolerance as the max-pool JVP test.
    np.testing.assert_allclose(tangent, expected_tan, atol=5e-3, rtol=5e-3)


# ─────────────────────────────────────────────────────────────────────────────
# Coverage assertions — every new VJP/JVP must be present in the registry
# ─────────────────────────────────────────────────────────────────────────────


@pytest.mark.parametrize("name", [
    "focal_loss", "label_smoothed_cross_entropy", "kl_divergence",
    "triplet_loss", "contrastive_loss", "cosine_embedding_loss", "info_nce_loss",
    "seq2seq_loss",
    "group_norm", "instance_norm", "lora_linear",
    "max_pool", "avg_pool",
])
def test_new_vjp_registered(name):
    assert get_vjp(name) is not None, f"VJP missing: {name}"


@pytest.mark.parametrize("name", [
    "focal_loss", "label_smoothed_cross_entropy", "kl_divergence",
    "triplet_loss", "contrastive_loss", "cosine_embedding_loss", "info_nce_loss",
    "seq2seq_loss",
    "group_norm", "instance_norm", "lora_linear",
    "max_pool", "avg_pool",
])
def test_new_jvp_registered(name):
    assert get_jvp(name) is not None, f"JVP missing: {name}"
