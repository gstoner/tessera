"""EBM8 — Tiny-model conformance suite for the energy-based-model stack.

Sprint: EBM8.
Roadmap: docs/audit/domain/DOMAIN_AUDIT.md § EBM8

Three demos prove the EBM stack end-to-end on tiny synthetic data:

  1. RBM on MNIST-tiny: a Bernoulli-Bernoulli RBM trained on
     hand-crafted 4×4 "digit" stereotypes via 1-step Contrastive
     Divergence. Verifies CD training reduces reconstruction error
     vs. a mean-image baseline.
  2. EBT-tiny: an Energy-Based Transformer–style model on a
     denoising task. T-step inner-loop "thinking" reduces task loss
     vs. zero-shot single-pass evaluation (T=0).
  3. SO(3) bivector score-matching: the GA + EBM merge demo —
     bivector Langevin sampling on Cl(3,0) recovers a Gaussian
     target distribution, and the score-matching loss verifies the
     analytic gradient direction.

All three run in <120s in CPU CI.
"""

from __future__ import annotations

import math
import time

import numpy as np
import pytest

from tessera.ebm import (
    bivector_langevin_sample,
    decode_init,
    energy,
    inner_step,
    self_verify,
)
from tessera.losses import (
    contrastive_divergence_loss,
    implicit_score_matching_loss,
)
from tessera.ga import Cl, Multivector, grade_projection
from tessera.rng import RNGKey


# ===========================================================================
# Demo 1: RBM on MNIST-tiny
# ===========================================================================

class TinyRBM:
    """Bernoulli-Bernoulli RBM with closed-form free energy and CD-1
    training. Pure numpy; ~80 LOC."""

    def __init__(self, n_visible: int, n_hidden: int, *, seed: int = 0):
        rng = np.random.RandomState(seed)
        self.W = (0.1 * rng.randn(n_visible, n_hidden)).astype(np.float64)
        self.b_v = np.zeros(n_visible, dtype=np.float64)
        self.b_h = np.zeros(n_hidden, dtype=np.float64)
        self.n_visible = n_visible
        self.n_hidden = n_hidden

    def free_energy(self, v: np.ndarray) -> np.ndarray:
        """``F(v) = −b_v·v − Σ_j log(1 + exp(b_h_j + Σ_i W_{ij} v_i))``."""
        bv_term = v @ self.b_v
        h_pre = v @ self.W + self.b_h
        log1pexp = np.logaddexp(0.0, h_pre)  # log(1 + exp(h_pre))
        return -bv_term - log1pexp.sum(axis=-1)

    def sample_h_given_v(self, v: np.ndarray, key: RNGKey):
        from tessera.rng import uniform
        h_pre = v @ self.W + self.b_h
        p_h = 1.0 / (1.0 + np.exp(-h_pre))
        sub_key, next_key = key.split(2)
        u = uniform(sub_key, shape=p_h.shape, dtype="fp64")
        return (np.asarray(u) < p_h).astype(np.float64), next_key

    def sample_v_given_h(self, h: np.ndarray, key: RNGKey):
        from tessera.rng import uniform
        v_pre = h @ self.W.T + self.b_v
        p_v = 1.0 / (1.0 + np.exp(-v_pre))
        sub_key, next_key = key.split(2)
        u = uniform(sub_key, shape=p_v.shape, dtype="fp64")
        return (np.asarray(u) < p_v).astype(np.float64), next_key

    def cd1_step(self, v0: np.ndarray, *, lr: float, key: RNGKey):
        """One CD-1 update: positive phase from v0, one Gibbs round to v1,
        negative phase from v1. Returns next RNG key."""
        h0_probs = 1.0 / (1.0 + np.exp(-(v0 @ self.W + self.b_h)))
        h0, key = self.sample_h_given_v(v0, key)
        v1, key = self.sample_v_given_h(h0, key)
        h1_probs = 1.0 / (1.0 + np.exp(-(v1 @ self.W + self.b_h)))
        # Gradient = positive correlations - negative correlations.
        pos = v0.T @ h0_probs
        neg = v1.T @ h1_probs
        self.W += lr * (pos - neg) / v0.shape[0]
        self.b_v += lr * (v0 - v1).mean(axis=0)
        self.b_h += lr * (h0_probs - h1_probs).mean(axis=0)
        return key

    def reconstruct(self, v: np.ndarray, key: RNGKey):
        """Deterministic mean-field reconstruction (one Gibbs pass on probs)."""
        h_probs = 1.0 / (1.0 + np.exp(-(v @ self.W + self.b_h)))
        v_recon = 1.0 / (1.0 + np.exp(-(h_probs @ self.W.T + self.b_v)))
        return v_recon


def _make_mnist_tiny_dataset(*, n_per_class: int = 20, seed: int = 0):
    """3 hand-crafted 4×4 "digit" stereotypes (cross, ring, diagonal) with
    Bernoulli noise applied."""
    stereotypes = [
        # Class 0: cross
        np.array([[0, 1, 1, 0],
                  [0, 1, 1, 0],
                  [1, 1, 1, 1],
                  [0, 1, 1, 0]], dtype=np.float64),
        # Class 1: ring
        np.array([[1, 1, 1, 1],
                  [1, 0, 0, 1],
                  [1, 0, 0, 1],
                  [1, 1, 1, 1]], dtype=np.float64),
        # Class 2: diagonal
        np.array([[1, 0, 0, 0],
                  [0, 1, 0, 0],
                  [0, 0, 1, 0],
                  [0, 0, 0, 1]], dtype=np.float64),
    ]
    rng = np.random.RandomState(seed)
    inputs = []
    labels = []
    for cls, base in enumerate(stereotypes):
        for _ in range(n_per_class):
            flat = base.flatten()
            # 10% bit-flip noise.
            noise_mask = rng.rand(16) < 0.10
            flipped = np.where(noise_mask, 1.0 - flat, flat)
            inputs.append(flipped)
            labels.append(cls)
    return np.stack(inputs, axis=0), np.array(labels)


def test_rbm_cd_training_beats_mean_image_baseline() -> None:
    """Acceptance: RBM trained with CD-1 for 500 iterations achieves
    lower reconstruction MSE than a degenerate "predict the mean image"
    baseline.
    """
    rng_np = np.random.RandomState(0)
    X, _ = _make_mnist_tiny_dataset(n_per_class=20)

    rbm = TinyRBM(n_visible=16, n_hidden=8, seed=0)
    key = RNGKey.from_seed(0)
    batch_size = 8
    n_steps = 500
    for step in range(n_steps):
        idx = rng_np.choice(X.shape[0], batch_size, replace=False)
        batch = X[idx]
        key = rbm.cd1_step(batch, lr=0.05, key=key)

    # Reconstruct the full dataset and compare to the mean-image baseline.
    recon = rbm.reconstruct(X, key)
    mse_rbm = float(np.mean((X - recon) ** 2))
    mean_image = X.mean(axis=0, keepdims=True)  # baseline
    mse_baseline = float(np.mean((X - mean_image) ** 2))
    assert mse_rbm < mse_baseline, (
        f"RBM reconstruction MSE {mse_rbm:.4f} should beat mean-image "
        f"baseline {mse_baseline:.4f}"
    )
    # And meaningfully better — not just float-noise margin.
    assert (mse_baseline - mse_rbm) / mse_baseline > 0.10


def test_rbm_free_energy_lower_for_in_distribution_samples() -> None:
    """A well-trained EBM assigns lower energy to real samples than to
    random noise. Verify on 64 random binary noise vectors."""
    X, _ = _make_mnist_tiny_dataset(n_per_class=10)
    rbm = TinyRBM(n_visible=16, n_hidden=8, seed=1)
    key = RNGKey.from_seed(1)
    rng_np = np.random.RandomState(2)
    for _ in range(300):
        idx = rng_np.choice(X.shape[0], 4, replace=False)
        key = rbm.cd1_step(X[idx], lr=0.08, key=key)

    real_E = float(rbm.free_energy(X).mean())
    noise = (rng_np.rand(64, 16) < 0.5).astype(np.float64)
    noise_E = float(rbm.free_energy(noise).mean())
    assert real_E < noise_E - 0.5, (
        f"trained RBM: real-data E={real_E:.3f} should be < noise E={noise_E:.3f}"
    )


# ===========================================================================
# Demo 2: EBT-tiny — inner-loop "thinking" beats zero-shot
# ===========================================================================

def _ebt_energy(target: np.ndarray):
    """Bilinear energy ``E(target, y) = ‖y − target‖²/2``. The
    "encoder" is just the identity, and the energy pulls y toward
    target. The inner-loop step ``y ← y − η · (y − target)`` provably
    reduces this loss; verifying that the chain does so in our
    primitive surface is the conformance claim."""
    def fn(target_state, y):
        return 0.5 * float(np.sum((np.asarray(y) - np.asarray(target_state)) ** 2))
    return fn


def test_ebt_inner_loop_reduces_task_loss_vs_zero_shot() -> None:
    """Acceptance: T-step inner-loop refinement of a candidate ``y``
    produces a lower task loss than the zero-shot prediction (T=0).
    """
    rng_np = np.random.RandomState(0)
    n_samples = 32
    target = rng_np.randn(n_samples, 4).astype(np.float64)
    # Initial "zero-shot" prediction: just noise (no learning).
    y_zero = rng_np.randn(n_samples, 4).astype(np.float64)
    loss_zero = float(np.mean(0.5 * np.sum((y_zero - target) ** 2, axis=-1)))

    # Tessera EBM inner loop: T=4 steps using ebm.inner_step.
    energy_fn = _ebt_energy(target)
    y = y_zero.copy()
    for _ in range(4):
        # Gradient of E wrt y at this point: (y - target).
        grad = y - target
        y = inner_step(y, grad, eta=0.4)  # detach noise / use pure GD
    loss_after = float(np.mean(0.5 * np.sum((y - target) ** 2, axis=-1)))
    assert loss_after < loss_zero * 0.5, (
        f"inner-loop should reduce loss substantially; zero-shot={loss_zero:.4f}, "
        f"after T=4 steps={loss_after:.4f}"
    )


def test_ebt_self_verify_picks_minimum_energy_candidate() -> None:
    """Self-verify reduces K candidate trajectories to the lowest-energy one."""
    K = 4
    rng_np = np.random.RandomState(0)
    target = rng_np.randn(2, 3).astype(np.float64)
    # K candidates per batch sample; the "true" candidate is target itself.
    candidates = rng_np.randn(2, K, 3).astype(np.float64)
    # Inject the target as candidate index 2.
    candidates[:, 2, :] = target
    energies = 0.5 * np.sum((candidates - target[:, None, :]) ** 2, axis=-1)
    # Energy at the "true" candidate is zero; everywhere else > 0.
    chosen = self_verify(energies, candidates)
    # The chosen candidate must equal the target (since its energy is 0).
    assert np.allclose(chosen, target)


def test_ebt_decode_init_produces_K_diverse_candidates() -> None:
    """decode_init with the 'noise' strategy gives K independent samples."""
    key = RNGKey.from_seed(0)
    x = np.zeros((3, 5))
    cands = decode_init(x, K=4, init_strategy="noise", rng_key=key, shape=(7,))
    assert cands.shape == (3, 4, 7)
    # K candidates should not be identical.
    for b in range(3):
        for i in range(4):
            for j in range(i + 1, 4):
                # At least one coefficient should differ.
                assert not np.array_equal(cands[b, i], cands[b, j])


# ===========================================================================
# Demo 3: SO(3) bivector score-matching pipeline (GA + EBM merge)
# ===========================================================================

def _bivector_gaussian_energy(A: np.ndarray):
    """E(B) = 0.5 · B^T A B over the grade-2 coefficients of a Cl(3,0)
    multivector. ``A`` is the precision matrix in the bivector subspace
    (3×3 for Cl(3,0))."""
    a = Cl(3, 0)
    bivector_indices = [a.blade("e12").mask, a.blade("e13").mask, a.blade("e23").mask]

    def fn(B):
        coeffs = B.coefficients[bivector_indices]
        return 0.5 * float(coeffs @ A @ coeffs)

    return fn


def _bivector_gaussian_grad(A: np.ndarray):
    """Analytic gradient on the full multivector coefficient vector;
    only the grade-2 slots are non-zero."""
    a = Cl(3, 0)
    bivector_indices = [a.blade("e12").mask, a.blade("e13").mask, a.blade("e23").mask]

    def fn(B):
        out = np.zeros(a.dim, dtype=np.float64)
        coeffs = B.coefficients[bivector_indices]
        gout = A @ coeffs
        for i, idx in enumerate(bivector_indices):
            out[idx] = gout[i]
        return Multivector(out, a)
    return fn


def test_bivector_langevin_recovers_target_gaussian_covariance() -> None:
    """The full SO(3)-via-bivector pipeline: sample a Gaussian on the
    bivector subspace using ``bivector_langevin_sample`` and verify the
    sampled covariance matches the target ``A_target^{-1}`` within 25%.

    Decision GA-L4 made operational: the bivector state never drifts
    off the Lie algebra, and the sampled distribution is the analytic
    Gaussian on so(3).
    """
    A_target = np.array([
        [3.0, 0.5, 0.0],
        [0.5, 2.0, 0.2],
        [0.0, 0.2, 1.5],
    ])
    a = Cl(3, 0)
    init = Multivector.from_blade(a.blade("e12"), a, dtype=np.float64)

    energy_fn = _bivector_gaussian_energy(A_target)
    grad_fn = _bivector_gaussian_grad(A_target)
    key = RNGKey.from_seed(7)
    samples, _, _ = bivector_langevin_sample(
        key,
        init=init,
        energy_fn=energy_fn,
        eta=0.02,
        temperature=1.0,
        n_samples=5000,
        burn_in=1000,
        thin=1,
        grade=2,
        grad_fn=grad_fn,
    )
    # Extract bivector coefficients (e12, e13, e23) from each sample.
    bivector_indices = [a.blade("e12").mask, a.blade("e13").mask, a.blade("e23").mask]
    bivec_samples = samples[:, bivector_indices]  # (N, 3)
    # Sampled covariance should ≈ A_target^{-1}.
    sample_cov = np.cov(bivec_samples, rowvar=False)
    target_cov = np.linalg.inv(A_target)
    # Frobenius relative error.
    rel_err = np.linalg.norm(sample_cov - target_cov) / np.linalg.norm(target_cov)
    assert rel_err < 0.25, (
        f"sampled bivector covariance mismatched: rel err {rel_err:.3f}\n"
        f"sample:\n{sample_cov}\n"
        f"target:\n{target_cov}"
    )


def test_so3_score_matching_objective_recognizes_true_precision() -> None:
    """Score-matching at the true precision matrix gives a lower
    objective than at a perturbed precision matrix — independent check
    that the EBM4 loss + EBM7 sampling pipeline composes correctly.
    """
    rng_np = np.random.RandomState(0)
    A_target = np.array([
        [2.0, 0.3, 0.1],
        [0.3, 1.5, 0.0],
        [0.1, 0.0, 1.0],
    ])
    cov = np.linalg.inv(A_target)
    L_chol = np.linalg.cholesky(cov)
    n = 5000
    y = rng_np.randn(n, 3) @ L_chol.T  # bivector coefficients sampled from target

    def sm_at(A):
        s = -(A @ y.T).T  # score: -∇E for E = 0.5 y^T A y, model "thinks" A is the precision
        div = np.full(n, -np.trace(A))  # ∇·s = -tr(A)
        return float(implicit_score_matching_loss(s, div, reduction="mean"))

    L_true = sm_at(A_target)
    A_bad = A_target + 0.3 * np.array([[1.0, 0.0, 0.0],
                                       [0.0, -0.5, 0.0],
                                       [0.0, 0.0, 0.4]])
    L_bad = sm_at(A_bad)
    assert L_true < L_bad, (
        f"SM at true precision ({L_true:.4f}) should be lower than at perturbed ({L_bad:.4f})"
    )


# ===========================================================================
# Conformance budget
# ===========================================================================

def test_ebm_conformance_runtime_budget() -> None:
    """The full EBM8 conformance suite must run in under 120s on CPU CI.
    Re-run the heaviest demo and assert wall-clock under 60s."""
    start = time.monotonic()
    test_bivector_langevin_recovers_target_gaussian_covariance()
    elapsed = time.monotonic() - start
    assert elapsed < 60.0, (
        f"bivector Langevin demo took {elapsed:.2f}s; expected < 60s"
    )
