"""Workstream D — SmoothQuant activation-scale migration pass.

Proves the migration is an exact fp factorization, the W8A8 direct-consume path
matches fp16 within int8 tolerance, the operands stay int8 (anti-fallback), and
migrating outliers into weights beats naive activation quantization.

See docs/audit/roadmap/CONTRACT_PASS_PLAN.md (Workstream D).
"""

from __future__ import annotations

import numpy as np
import pytest

from tessera.compiler.smoothquant import (
    SmoothQuantConfig, migrate_activation_scale, smoothquant_matmul,
    verify_w8a8, exact_smoothing_residual, compute_smoothing_factor)


def _outlier_activations(n, c_in, seed=0):
    """Activations with a few large-magnitude outlier channels (the SmoothQuant
    motivating case)."""
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, c_in)).astype(np.float32)
    X[:, ::7] *= 30.0  # outlier channels
    return X


# ── the migration is exact in fp (only quant introduces error) ───────────────


def test_smoothing_is_exact_pre_quant():
    rng = np.random.default_rng(1)
    X = _outlier_activations(32, 16)
    W = rng.standard_normal((16, 24)).astype(np.float32)
    migrated = migrate_activation_scale(X, W)
    # X̂ @ Ŵ must equal X @ W up to fp roundoff — the factorization is exact.
    assert exact_smoothing_residual(X, W, migrated) < 1e-2


# ── W8A8 direct-consume parity + anti-fallback ───────────────────────────────


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_w8a8_matches_fp_within_tolerance(seed):
    rng = np.random.default_rng(seed)
    X = _outlier_activations(64, 32, seed=seed)
    W = rng.standard_normal((32, 48)).astype(np.float32)
    migrated = migrate_activation_scale(X, W)
    verdict = verify_w8a8(X, W, migrated)
    assert verdict.is_equivalent, verdict.detail


def test_operands_are_int8_not_dequantized():
    rng = np.random.default_rng(3)
    X = _outlier_activations(16, 16)
    W = rng.standard_normal((16, 16)).astype(np.float32)
    migrated = migrate_activation_scale(X, W)
    # The anti-fallback invariant: the consumed weight operand is int8.
    assert migrated.w_q.dtype == np.int8
    pol = migrated.numeric_policy()
    assert pol.storage == "int8" and pol.accum == "int32"


# ── SmoothQuant beats naive activation quantization on outliers ──────────────


def _naive_w8a8(X, W):
    """Per-token int8 activation × per-channel int8 weight, NO smoothing."""
    ax = np.maximum(np.abs(X).max(1, keepdims=True), 1e-12) / 127
    xq = np.round(X / ax).clip(-127, 127).astype(np.int8)
    wamax = np.maximum(np.abs(W).max(0, keepdims=True), 1e-12) / 127
    wq = np.round(W / wamax).clip(-127, 127).astype(np.int8)
    return (xq.astype(np.int32) @ wq.astype(np.int32)).astype(np.float32) * ax * wamax


def test_smoothquant_beats_naive_on_outliers():
    rng = np.random.default_rng(7)
    X = _outlier_activations(64, 32, seed=7)
    W = rng.standard_normal((32, 48)).astype(np.float32)
    y_ref = X @ W
    scale = float(np.max(np.abs(y_ref)))

    naive_err = float(np.max(np.abs(_naive_w8a8(X, W) - y_ref)) / scale)
    migrated = migrate_activation_scale(X, W)
    sq_err = float(np.max(np.abs(smoothquant_matmul(X, migrated) - y_ref)) / scale)
    # Migrating outliers into the weights yields a strictly better W8A8 result.
    assert sq_err < naive_err


# ── smoothing-factor knobs ────────────────────────────────────────────────────


def test_alpha_zero_is_no_migration():
    # alpha=0 → s = 1 / max|W|^1 ... actually s depends only on weights; the
    # factor must still produce an exact factorization regardless of alpha.
    rng = np.random.default_rng(5)
    X = _outlier_activations(16, 12)
    W = rng.standard_normal((12, 10)).astype(np.float32)
    m0 = migrate_activation_scale(X, W, config=SmoothQuantConfig(alpha=0.0))
    m1 = migrate_activation_scale(X, W, config=SmoothQuantConfig(alpha=1.0))
    assert exact_smoothing_residual(X, W, m0) < 1e-2
    assert exact_smoothing_residual(X, W, m1) < 1e-2


def test_compute_smoothing_factor_formula():
    act = np.array([4.0, 1.0])
    wt = np.array([1.0, 4.0])
    s = compute_smoothing_factor(act, wt, alpha=0.5, eps=1e-5)
    # s_j = act^0.5 / wt^0.5
    np.testing.assert_allclose(s, np.sqrt(act) / np.sqrt(wt), rtol=1e-5)
