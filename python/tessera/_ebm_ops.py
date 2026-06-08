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


# Names registered into the tessera.ops namespace (see __init__).
EBM_OPS = {
    "ebm_energy_quadratic": ebm_energy_quadratic,
    "ebm_self_verify": ebm_self_verify,
    "ebm_refinement": ebm_refinement,
    "ebm_inner_step": ebm_inner_step,
}
