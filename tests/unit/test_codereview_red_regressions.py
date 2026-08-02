"""Host-free regressions from the full-source code review (2026-06-20).

Each test encodes the *correct* contract and therefore FAILS against the
current (buggy) code — that is the point of a regression test: it reproduces
the defect and will turn green once the fix lands.

The native R1 AMX regression is owned by
``tests/device/x86/test_amx_int8_gemm.py`` so it remains executable without
placing an exact-device proof in the hermetic unit lane.

Findings under test here:
  R2. python/tessera/autodiff/vjp.py:2357
      cross_entropy_loss soft-target gradient is returned WITHOUT the
      reduction-scale factor (off by 1/N for reduction="mean"; ignores dout).
  R3. python/tessera/autodiff/vjp.py:847
      dropout VJP recomputes the mask from a fresh RNG; for seed=None the
      backward mask is uncorrelated with the forward mask -> wrong gradient.
"""

from __future__ import annotations

import numpy as np
import pytest

import tessera
import tessera.losses as losses
from tessera.autodiff.vjp import get_vjp


# ─────────────────────────────────────────────────────────────────────────────
# helpers
# ─────────────────────────────────────────────────────────────────────────────
def _numeric_grad_wrt(fn, x, eps=1e-6):
    """Central finite-difference gradient of ``fn(x).sum()`` w.r.t. ``x``."""
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


# ─────────────────────────────────────────────────────────────────────────────
# R2 — cross_entropy_loss soft-target gradient must be reduction-scaled
# ─────────────────────────────────────────────────────────────────────────────
def test_cross_entropy_soft_target_grad_is_reduction_scaled_mean():
    """reduction='mean' must scale the target gradient by 1/N.

    Reproduces vjp.py:2357 — `target_grad` is returned unscaled, so it is a
    factor of N (number of rows) too large for reduction='mean'.
    """
    rng = np.random.default_rng(0)
    logits = rng.standard_normal((4, 5))
    raw = rng.random((4, 5))
    targets = raw / raw.sum(axis=-1, keepdims=True)  # soft targets

    numeric = _numeric_grad_wrt(
        lambda tt: losses.cross_entropy_loss(logits, tt, reduction="mean"),
        targets,
    )

    _, target_grad = get_vjp("cross_entropy_loss")(1.0, logits, targets, reduction="mean")
    assert target_grad is not None
    np.testing.assert_allclose(target_grad, numeric, rtol=1e-5, atol=1e-7)


def test_cross_entropy_soft_target_grad_respects_dout_reduction_none():
    """reduction='none' must scale the target gradient by the upstream dout.

    The analytic gradient of -sum(targets * log_softmax(logits)) w.r.t. the
    soft targets is `-log_softmax(logits)` per row, scaled by the per-row dout.
    vjp.py:2357 ignores dout entirely.
    """
    rng = np.random.default_rng(1)
    logits = rng.standard_normal((3, 6))
    raw = rng.random((3, 6))
    targets = raw / raw.sum(axis=-1, keepdims=True)
    dout = np.array([0.5, 2.0, -1.0])  # non-uniform upstream cotangent

    log_softmax = logits - (
        np.log(np.sum(np.exp(logits - logits.max(-1, keepdims=True)), axis=-1, keepdims=True))
        + logits.max(-1, keepdims=True)
    )
    expected = -log_softmax * dout[:, None]

    _, target_grad = get_vjp("cross_entropy_loss")(dout, logits, targets, reduction="none")
    assert target_grad is not None
    np.testing.assert_allclose(target_grad, expected, rtol=1e-6, atol=1e-8)


# ─────────────────────────────────────────────────────────────────────────────
# R3 — dropout VJP must use the same mask as the forward pass
# ─────────────────────────────────────────────────────────────────────────────
def test_dropout_vjp_fails_loudly_without_reproducible_seed():
    """An unseeded differentiable dropout must raise, not silently mis-grad.

    Reproduces vjp.py:847 — with seed=None the backward mask is drawn from a
    fresh RNG, uncorrelated with the forward mask. The fix converts that
    silently-wrong gradient into a loud error (the mask is unrecoverable).
    """
    x = np.ones((8, 8), dtype=np.float64)
    with pytest.raises(ValueError, match="reproducible seed"):
        get_vjp("dropout")(np.ones_like(x), x, p=0.5, training=True, seed=None)


def test_nn_dropout_materializes_seed_and_mask_is_reproducible():
    """nn.Dropout (training) must record a concrete seed that the VJP can reuse.

    Validates the high-level half of the R3 fix: nn.Dropout materializes a
    per-call seed so the differentiable path stays correct, and that seed
    reproduces the exact forward mask (grad support == forward keep-mask).
    """
    from tessera.autodiff.tape import tape

    layer = tessera.nn.Dropout(p=0.5).train()
    x = np.ones((16, 16), dtype=np.float64)
    with tape() as t:
        y = layer(x)

    drop = [e for e in t.entries if e.op == "dropout"]
    assert drop, "dropout was not recorded on the tape"
    seed = drop[-1].kwargs.get("seed")
    assert seed is not None, "nn.Dropout must materialize a concrete seed while training"

    (grad,) = get_vjp("dropout")(np.ones_like(x), x, p=0.5, training=True, seed=seed)
    np.testing.assert_array_equal(grad != 0.0, np.asarray(y) != 0.0)


def test_dropout_vjp_seeded_matches_forward_mask():
    """Sanity anchor: even the seeded path must reproduce the forward mask.

    This currently passes (same seed -> same RNG draw), but pins the
    forward/backward consistency contract so a fix to R3 cannot regress it.
    """
    x = np.ones((32, 32), dtype=np.float64)
    p = 0.3
    seed = 12345

    y = np.asarray(tessera.ops.dropout(x, p=p, training=True, seed=seed))
    (grad,) = get_vjp("dropout")(np.ones_like(x), x, p=p, training=True, seed=seed)

    # kept units scale by 1/(1-p); dropped units are zero in both.
    np.testing.assert_allclose(grad, y, rtol=0, atol=0)
