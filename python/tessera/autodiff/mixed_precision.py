"""Mixed-precision support — Phase F1 of the execution roadmap.

Two pieces:

* :func:`autocast` — context manager that casts op-input arrays to a target
  dtype before the underlying op runs. Implemented by hooking into the
  existing tape wrapper's pre-call path via a ``contextvar``. Reductions
  (``layer_norm``, ``rmsnorm``, ``softmax``, ``reduce``) are kept in fp32
  for numerical stability — matches the standard mixed-precision recipe.

* :class:`GradScaler` — scales the loss before backward to keep small
  gradients out of the fp16 underflow zone, then unscales the gradients
  before the optimizer step. Detects ``inf``/``NaN`` and skips the step
  on overflow, halving the scale factor.

Both surfaces are intentionally narrow in v1 — no Graph-IR autocast pass,
no per-op dtype-policy table override. Phase F4 (Graph IR adjoints) is the
right home for the production version.
"""

from __future__ import annotations

import contextvars
from contextlib import contextmanager
from typing import Iterable

import numpy as np


# Ops that should stay in fp32 even when an autocast is active.
# Matches torch's default mixed-precision policy: reductions and norms
# are fp32-stable; matmul / elementwise math is the cast target.
_AUTOCAST_FP32_ALLOWLIST: frozenset[str] = frozenset({
    "layer_norm",
    "rmsnorm",
    "rmsnorm_safe",
    "softmax",
    "softmax_safe",
    "reduce",
    "sum",
    "online_softmax",
    "online_softmax_state",
})


# (target_dtype | None, allowlist) currently active. None = autocast off.
_ACTIVE_AUTOCAST: contextvars.ContextVar[str | None] = contextvars.ContextVar(
    "_tessera_autocast", default=None,
)


def autocast_dtype() -> str | None:
    """Return the active autocast target dtype, or ``None``."""
    return _ACTIVE_AUTOCAST.get()


def autocast_keep_fp32(op_name: str) -> bool:
    """Whether ``op_name`` should stay in fp32 under the active autocast."""
    return op_name in _AUTOCAST_FP32_ALLOWLIST


_VALID_AUTOCAST_DTYPES = (
    "fp16", "bf16", "fp32", "fp64",
    # Theme 10 — fp8 autocast. Inputs are routed through
    # `ops.quantize_fp8(..., format=<fmt>)` on the boundary; the
    # reference path stores them as fp32 (numerically equal to their
    # fp8-rounded value). Per-backend lowering — Hopper tcgen05 fp8 mma,
    # ROCm OCP fp8 — is deferred to Phase G; the autocast surface is the
    # API unblock.
    "fp8_e4m3",
    "fp8_e5m2",
    # Deferred-items plan, Item 2 — fp6 / fp4 autocast. Same boundary-
    # quantization mechanic as fp8. fp6 has two formats; fp4 ships only
    # the Blackwell-supported E2M1; nvfp4 uses block-scaled e2m1 (default
    # 16 elements per block).
    "fp6_e2m3",
    "fp6_e3m2",
    "fp4_e2m1",
    "nvfp4",
)


@contextmanager
def autocast(dtype: str = "fp16"):
    """Enter a mixed-precision region.

    Inside the ``with`` block, op input arrays are cast to ``dtype`` before
    the underlying numpy reference runs (except for the
    ``_AUTOCAST_FP32_ALLOWLIST`` ops, which stay in fp32). Output dtype
    follows the input dtype as numpy normally would.

    Supported dtypes:
      * ``"fp16"`` — native ``np.float16`` cast.
      * ``"bf16"`` — stored as fp32 by the numpy reference; downstream
        lowering materializes true bf16.
      * ``"fp32"`` / ``"fp64"`` — pass-through / upcast.
      * ``"fp8_e4m3"`` / ``"fp8_e5m2"`` — Theme 10. Inputs are quantized
        to fp8 on the boundary via :func:`tessera.ops.quantize_fp8` (per-
        tensor symmetric, scale derived from amax). The cast is lossy;
        accumulators stay in fp32.
    """
    if dtype not in _VALID_AUTOCAST_DTYPES:
        raise ValueError(
            f"autocast dtype must be one of {_VALID_AUTOCAST_DTYPES}; got {dtype!r}"
        )
    token = _ACTIVE_AUTOCAST.set(dtype)
    try:
        yield
    finally:
        _ACTIVE_AUTOCAST.reset(token)


# ─────────────────────────────────────────────────────────────────────────────
# GradScaler
# ─────────────────────────────────────────────────────────────────────────────


class GradScaler:
    """Loss-scaling helper for mixed-precision training.

    Standard recipe:

    ::

        scaler = GradScaler()
        with autocast("fp16"):
            y = model(x)
            loss = compute_loss(y, target)
        with autodiff.tape() as t:
            ...
            t.backward(y, cotangent=scaler.scale_grad(dy_seed))
        if scaler.step(optimizer_fn, params=model.parameters()):
            print("step taken")  # else: skipped due to overflow
    """

    def __init__(
        self,
        init_scale: float = 2.0 ** 16,
        growth_factor: float = 2.0,
        backoff_factor: float = 0.5,
        growth_interval: int = 2000,
    ):
        if init_scale <= 0:
            raise ValueError("init_scale must be positive")
        self._scale = float(init_scale)
        self._growth = float(growth_factor)
        self._backoff = float(backoff_factor)
        self._growth_interval = int(growth_interval)
        self._steps_since_growth = 0

    @property
    def scale(self) -> float:
        return self._scale

    def scale_loss(self, loss: float | np.ndarray) -> float | np.ndarray:
        """Multiply ``loss`` by the current scale factor before backward."""
        return loss * self._scale

    def scale_grad(self, dy: np.ndarray) -> np.ndarray:
        """Scale a cotangent seed for ``Tape.backward(..., cotangent=...)``."""
        return np.asarray(dy) * self._scale

    def _has_inf_nan(self, params: Iterable) -> bool:
        for p in params:
            if p.grad is None:
                continue
            g = p.grad.numpy()
            if not np.isfinite(g).all():
                return True
        return False

    def _unscale_(self, params: Iterable) -> None:
        inv = 1.0 / self._scale
        for p in params:
            if p.grad is None:
                continue
            p.grad._data[...] = p.grad._data * inv

    def step(self, optimizer_fn, *, params: Iterable) -> bool:
        """Unscale grads, check for overflow, and call ``optimizer_fn`` if safe.

        ``optimizer_fn`` is called with no args — the caller closes over the
        params. Returns ``True`` if the step was taken, ``False`` if it was
        skipped due to overflow.
        """
        params = list(params)
        if self._has_inf_nan(params):
            # Overflow — drop the step, halve the scale, reset growth counter
            self._scale = max(self._scale * self._backoff, 1.0)
            self._steps_since_growth = 0
            for p in params:
                if p.grad is not None:
                    p.zero_grad()
            return False
        self._unscale_(params)
        optimizer_fn()
        self._steps_since_growth += 1
        if self._steps_since_growth >= self._growth_interval:
            self._scale *= self._growth
            self._steps_since_growth = 0
        return True

    def update(self) -> None:
        """No-op alias for torch parity — call after every step regardless.

        Tessera's ``GradScaler.step`` already updates internal state, so this
        is just a compatibility hook for code ported from torch.
        """
        return


__all__ = [
    "autocast",
    "autocast_dtype",
    "autocast_keep_fp32",
    "GradScaler",
]
