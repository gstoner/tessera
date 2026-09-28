"""Canonical ``tessera.ops.ebm_*`` flat-array shim over the EBM lane.

Energy-based-model primitives live in the ``tessera.ebm.*`` lane (and several
already GPU-dispatch through ``tessera.ebm`` to dedicated MSL kernels —
``tessera_apple_gpu_ebm_{energy_quadratic,inner_step,self_verify,refinement}_f32``).
This module projects the **tensor-clean** subset onto the canonical
``tessera.ops`` surface so they:

  1. are reachable from the standard ``tessera.ops`` / ``@jit`` surface, and
  2. flow through the autodiff tape chokepoint, which makes the VJP/JVP rules in
     ``autodiff/{vjp,jvp}.py`` meaningful.

Scope: only the ops whose entire signature is flat arrays + static scalars —
``energy_quadratic``, ``self_verify``, ``refinement``, ``inner_step``. The
callable/RNG-taking EBM ops (``energy``, ``partition_function*``,
``langevin_step``, ``decode_init``) cannot be flat ``tessera.ops`` ops (they take
``energy_fn`` callables or ``RNGKey``) and stay on the ``tessera.ebm`` lane.

Static scalars (``eta``/``T``/``beta``/``noise_scale``) are keyword-only so the
autodiff tape — which records only array inputs + kwargs — captures them.
"""

from __future__ import annotations

from typing import Any

import numpy as np


def ebm_energy_quadratic(x: Any, y: Any) -> np.ndarray:
    """0.5·‖x−y‖² reduced over all but the batch axis (the EBT/diffusion
    reconstruction energy). Routes to the cl/EBM MSL kernel for f32 rank-2."""
    from tessera import ebm as E
    return np.asarray(E.energy_quadratic(x, y))


def ebm_self_verify(energies: Any, candidates: Any, *, beta: float | None = None) -> np.ndarray:
    """Reduce K candidates by energy: hard argmin (``beta=None``) or soft-min
    (``beta>0`` ⇒ softmax(−β·energies)-weighted sum, differentiable)."""
    from tessera import ebm as E
    return np.asarray(E.self_verify(energies, candidates, beta=beta))


def ebm_refinement(y0: Any, grad: Any, *, eta: float, T: int) -> np.ndarray:
    """``T`` fixed-gradient inner steps: ``y_T = y0 − T·eta·grad``."""
    from tessera import ebm as E
    return np.asarray(E.refinement(y0, grad, eta=float(eta), T=int(T)))


def ebm_inner_step(y: Any, grad: Any, *, eta: float, noise_scale: float = 0.0) -> np.ndarray:
    """One Langevin/SGD inner step: ``y − eta·grad`` (+ optional noise)."""
    from tessera import ebm as E
    return np.asarray(E.inner_step(y, grad, float(eta), noise_scale=noise_scale))


_U32 = 1 << 32


def philox_key_counter(seed: Any, counter: Any) -> tuple[np.ndarray, np.ndarray]:
    """Map the Graph op's Philox operands onto the kernels' (key, counter).

    ``tessera.ebm.langevin_step_philox`` carries ``seed : tensor<1xi64>`` and
    ``counter : tensor<4xi64>`` (TesseraOps.td). The device kernels and the
    numpy reference take a 2 x u32 key and a 4 x u32 counter. The seed's 64
    bits split low word first (``k0 = seed & 0xFFFFFFFF``, ``k1 = seed >> 32``,
    two's complement for a negative seed); every counter word must already be
    a u32 value -- a word outside ``[0, 2**32)`` is refused rather than
    truncated, because truncation would silently alias two streams."""
    s = np.asarray(seed).reshape(-1)
    c = np.asarray(counter).reshape(-1)
    if s.size != 1 or not np.issubdtype(s.dtype, np.integer):
        raise ValueError(f"langevin_step_philox seed must be one integer word; got {s.dtype}{s.shape}")
    if c.size != 4 or not np.issubdtype(c.dtype, np.integer):
        raise ValueError(f"langevin_step_philox counter must be four integer words; got {c.dtype}{c.shape}")
    words = [int(w) for w in c]
    if any(w < 0 or w >= _U32 for w in words):
        raise ValueError(f"langevin_step_philox counter words must lie in [0, 2**32); got {words}")
    seed64 = int(s[0]) & (_U32 * _U32 - 1)
    key = np.array([seed64 & (_U32 - 1), seed64 >> 32], np.uint32)
    return key, np.array(words, np.uint32)


def langevin_philox_noise_scale(eta: float, temperature: float, noise_scale: float | None) -> float:
    """The Philox step's noise amplitude: ``noise_scale`` when given, else the
    Langevin ``sqrt(2 * eta * temperature)`` (TesseraOps.td). ``eta`` and
    ``temperature`` are required by the op; they are validated here so the
    reference and the executors refuse the same inputs."""
    eta, temperature = float(eta), float(temperature)
    if not (np.isfinite(eta) and eta > 0.0):
        raise ValueError(f"langevin_step_philox requires eta > 0; got {eta}")
    if not (np.isfinite(temperature) and temperature > 0.0):
        raise ValueError(f"langevin_step_philox requires temperature > 0; got {temperature}")
    if noise_scale is None:
        return float(np.sqrt(2.0 * eta * temperature))
    ns = float(noise_scale)
    if not (np.isfinite(ns) and ns >= 0.0):
        raise ValueError(f"langevin_step_philox requires noise_scale >= 0; got {ns}")
    return ns


def ebm_langevin_step_philox(y: Any, grad: Any, seed: Any, counter: Any, *,
                             eta: float, temperature: float,
                             noise_scale: float | None = None) -> np.ndarray:
    """Graph op ``tessera.ebm.langevin_step_philox``:
    ``y - eta*grad + noise_scale*z`` with ``z`` drawn from Philox-4x32-10 over
    ``(seed, counter)`` (per-element counter ``(c0+i, c1, c2, c3)``, first
    Box-Muller lobe). ``noise_scale`` defaults to ``sqrt(2*eta*temperature)``.
    Delegates to :func:`tessera.ebm.langevin_step_philox`, the numpy reference
    the device kernels are validated against."""
    from tessera import ebm as E
    key, ctr = philox_key_counter(seed, counter)
    ns = langevin_philox_noise_scale(eta, temperature, noise_scale)
    return np.asarray(E.langevin_step_philox(y, grad, eta=float(eta), noise_scale=ns,
                                             key=key, counter=ctr))


# Names registered into the tessera.ops namespace (see __init__).
EBM_OPS = {
    "ebm_energy_quadratic": ebm_energy_quadratic,
    "ebm_self_verify": ebm_self_verify,
    "ebm_refinement": ebm_refinement,
    "ebm_inner_step": ebm_inner_step,
    "ebm_langevin_step_philox": ebm_langevin_step_philox,
}
