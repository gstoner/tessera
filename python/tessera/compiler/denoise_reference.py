"""
tessera.compiler.denoise_reference — pure-numpy classification-as-denoising
reference model + metamorphic conformance fixture.

Lifted from the DiffusionBlocks released code (arXiv:2506.14202), which reframes
classification as **denoising a class-label embedding**: instead of a softmax
head, the network denoises a noisy label embedding ``z = y + σε`` back to the
clean class embedding ``y`` and reads the class off the recovered embedding.
This module is a small, fully-specified, seed-reproducible reference for that
recipe — it exercises, end to end:

  * :func:`tessera.compiler.diffusion_schedule.equiprob_band_schedule`
    band dispatch (which block owns each noise level),
  * EDM preconditioning / the σ-aware denoiser, and
  * an EDM **Euler ODE sampler** over :func:`...karras_sigmas`.

The "network" here is the *ideal* denoiser for a mixture-of-point-masses prior
over the class embeddings — the Bayes-optimal ``D*(z, σ) = E[y | z]`` — which is
closed-form, exact, and needs no training.  That makes it an oracle: a backend
or compiled denoiser can be graded against it, and it admits crisp metamorphic
invariants (denoise→renoise→denoise self-consistency; the probability-flow ODE
contracts toward the data).

Pure ``numpy`` + stdlib (Decision #23): no Torch/JAX/SciPy.  This is a reference
oracle, not a training surface.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .diffusion_schedule import BandSchedule, equiprob_band_schedule, karras_sigmas

__all__ = [
    "DenoiseClassifier",
    "make_classifier",
    "SampleTrace",
    "euler_denoise_sample",
    "renoise_consistency_gap",
]


@dataclass(frozen=True)
class DenoiseClassifier:
    """Bayes-optimal label-embedding denoiser over a fixed class codebook.

    Attributes
    ----------
    embeddings : np.ndarray
        ``(num_classes, dim)`` clean class embeddings (the codebook ``y_k``).
    schedule : BandSchedule
        Equal-mass σ-band partition; routes each noise level to a block index.
    sigma_data : float
        Data standard deviation (RMS of the codebook), the EDM ``σ_data``.
    """

    embeddings: np.ndarray
    schedule: BandSchedule
    sigma_data: float

    @property
    def num_classes(self) -> int:
        return int(self.embeddings.shape[0])

    @property
    def dim(self) -> int:
        return int(self.embeddings.shape[1])

    def class_logits(self, z: np.ndarray, sigma: float) -> np.ndarray:
        """Posterior log-probabilities ``log p(k | z, σ)`` over classes.

        For the mixture-of-point-masses prior, ``p(k | z) ∝ exp(−‖z − y_k‖² /
        (2σ²))``.  Returns ``(N, num_classes)`` log-posteriors (un-normalized
        constants dropped via the log-sum-exp shift), one row per ``z`` row.
        """
        z2 = np.atleast_2d(np.asarray(z, dtype=np.float64))
        # ‖z − y_k‖² for every (row, class) pair → (N, K)
        diff = z2[:, None, :] - self.embeddings[None, :, :]
        sq = np.sum(diff * diff, axis=-1)
        logits = -sq / (2.0 * sigma * sigma)
        logits -= logits.max(axis=-1, keepdims=True)
        return logits

    def posterior(self, z: np.ndarray, sigma: float) -> np.ndarray:
        """Class posterior ``p(k | z, σ)``, shape ``(N, num_classes)``."""
        logits = self.class_logits(z, sigma)
        ex = np.exp(logits)
        return ex / ex.sum(axis=-1, keepdims=True)

    def denoise(self, z: np.ndarray, sigma: float) -> np.ndarray:
        """Bayes denoiser ``D*(z, σ) = Σ_k p(k | z, σ) · y_k``.

        The posterior-mean estimate of the clean embedding — the EDM ``x₀``
        prediction for this prior.  Shape matches ``z`` (``(N, dim)``).
        """
        post = self.posterior(z, sigma)            # (N, K)
        return post @ self.embeddings              # (N, dim)

    def classify(self, z: np.ndarray, sigma: float) -> np.ndarray:
        """Predicted class indices = ``argmax_k p(k | z, σ)``."""
        return np.argmax(self.class_logits(z, sigma), axis=-1).astype(np.int64)

    def block_for_sigma(self, sigma: float | np.ndarray) -> np.ndarray:
        """Which block (band index) owns this noise level."""
        return self.schedule.block_for_sigma(sigma)


def make_classifier(
    num_classes: int = 8,
    dim: int = 16,
    *,
    seed: int = 0,
    num_blocks: int = 4,
    gamma: float = 0.1,
    embedding_scale: float | None = None,
) -> DenoiseClassifier:
    """Build a reproducible :class:`DenoiseClassifier` with orthonormal-ish
    class embeddings.

    Embeddings are the (scaled) rows of a random orthogonal matrix, so classes
    are maximally separated for the given ``dim`` (requires ``dim >=
    num_classes``).  ``embedding_scale`` defaults to ``sqrt(dim)`` so the
    per-coordinate data scale is ``O(1)`` and the inter-class separation
    (``sqrt(2·dim)·scale``) stays well above dim-scaled isotropic noise — i.e.
    classes remain resolvable for σ up to ≈ ``σ_data``.  ``sigma_data`` is set
    to the codebook RMS so EDM preconditioning is correctly scaled.
    """
    if dim < num_classes:
        raise ValueError(
            f"dim ({dim}) must be >= num_classes ({num_classes}) for an "
            f"orthonormal codebook"
        )
    if embedding_scale is None:
        embedding_scale = float(np.sqrt(dim))
    rng = np.random.default_rng(seed)
    # Orthonormal rows via QR of a Gaussian matrix.
    q, _ = np.linalg.qr(rng.standard_normal((dim, dim)))
    embeddings = embedding_scale * q[:num_classes]
    sigma_data = float(np.sqrt(np.mean(embeddings * embeddings)))
    schedule = equiprob_band_schedule(num_blocks, gamma=gamma)
    return DenoiseClassifier(
        embeddings=embeddings, schedule=schedule, sigma_data=sigma_data
    )


@dataclass(frozen=True)
class SampleTrace:
    """Result of an Euler ODE denoising run.

    Attributes
    ----------
    z_final : np.ndarray
        The integrated clean-embedding estimate, ``(N, dim)``.
    pred_class : np.ndarray
        ``argmax`` class read off ``z_final`` at the smallest σ, ``(N,)``.
    block_path : np.ndarray
        Block index that handled each ODE step, ``(num_steps,)`` — the
        band-dispatch trace.
    sigmas : np.ndarray
        The σ schedule that was integrated (descending).
    """

    z_final: np.ndarray
    pred_class: np.ndarray
    block_path: np.ndarray
    sigmas: np.ndarray


def euler_denoise_sample(
    model: DenoiseClassifier,
    z_init: np.ndarray,
    sigmas: np.ndarray | None = None,
    *,
    num_steps: int = 24,
) -> SampleTrace:
    """Integrate the EDM probability-flow ODE with first-order Euler steps.

    At each σ the denoiser is dispatched to its block (recorded in
    ``block_path``) and the EDM derivative ``d = (z − D*(z, σ)) / σ`` advances
    ``z`` toward lower noise: ``z ← z + (σ_{i+1} − σ_i) · d``.  Routing happens
    by σ-band exactly as in the DiffusionBlocks sampler.

    ``z_init`` is ``(N, dim)`` (start it at ``σ_max`` noise).  ``sigmas``
    defaults to a Karras ρ=7 schedule matching the model's σ window.
    """
    if sigmas is None:
        sigmas = karras_sigmas(
            num_steps,
            sigma_min=model.schedule.sigma_min,
            sigma_max=model.schedule.sigma_max,
        )
    sigmas = np.asarray(sigmas, dtype=np.float64)
    z = np.atleast_2d(np.asarray(z_init, dtype=np.float64)).copy()

    block_path = np.empty(sigmas.shape[0] - 1, dtype=np.int64)
    for i in range(sigmas.shape[0] - 1):
        sigma, next_sigma = float(sigmas[i]), float(sigmas[i + 1])
        block_path[i] = int(model.block_for_sigma(sigma))
        denoised = model.denoise(z, sigma)
        d = (z - denoised) / sigma
        z = z + (next_sigma - sigma) * d

    final_sigma = float(sigmas[-1])
    pred = model.classify(z, final_sigma)
    return SampleTrace(
        z_final=z, pred_class=pred, block_path=block_path, sigmas=sigmas
    )


def renoise_consistency_gap(
    model: DenoiseClassifier,
    y: np.ndarray,
    sigma_a: float,
    sigma_b: float,
    *,
    seed: int = 0,
) -> float:
    """Metamorphic self-consistency: max ‖D*(z_a, σ_a) − D*(renoise, σ_b)‖.

    Denoise a noised sample at σ_a, re-noise the estimate to σ_b, denoise again.
    For a consistent score field over well-separated classes both estimates
    recover the same clean embedding, so the gap → 0.  This is the
    "denoise-then-renoise" invariant — evaluator metamorphic-oracle fodder,
    returned as a scalar discrepancy (lower is better).
    """
    rng = np.random.default_rng(seed)
    y2 = np.atleast_2d(np.asarray(y, dtype=np.float64))
    z_a = y2 + sigma_a * rng.standard_normal(y2.shape)
    d_a = model.denoise(z_a, sigma_a)
    z_b = d_a + sigma_b * rng.standard_normal(y2.shape)
    d_b = model.denoise(z_b, sigma_b)
    return float(np.max(np.abs(d_a - d_b)))
