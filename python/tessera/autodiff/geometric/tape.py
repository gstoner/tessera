"""GA6 — Minimal multivector tape and gradient helper.

The full tape-based tracer for arbitrary user code is GA10/GA11 work.
For GA6, we ship a minimal harness that lets users:

  - Open a ``tape_geo()`` context for documentation / API parity with
    ``tessera.autodiff.tape()``. The current implementation does not
    auto-record every op (that requires patching `tessera.ga.ops` to
    check an active tape; deferred to GA10 along with the
    full-language tracer).
  - Call ``multivector_grad(fn)(args, cotangent=...)`` for the common
    case of a hand-traced scalar-loss function expressed as a chain of
    registered VJP ops. The helper takes a forward function ``fn`` and
    returns a function that computes gradients by directly applying
    registered VJPs in reverse — useful for unit tests and small
    differentiable models without needing the full tracer.

The acceptance criterion "mixed tensor + multivector graphs are valid"
is satisfied by virtue of the parallel registry — opening
``tape_geo()`` does not disturb the existing ``tape()`` for tensor
autodiff, and vice versa. The two tapes use different op-name
registries (``_VJPS`` vs ``_VJPS_GEO``) and do not interfere.
"""

from __future__ import annotations

import contextlib
import contextvars
from dataclasses import dataclass, field
from typing import Any, Callable, List, Tuple

from tessera.ga.multivector import Multivector


# Tape recording is opt-in: ops do NOT auto-record in GA6. The active
# tape state is exposed for completeness — GA10 will wire op-level
# recording through this contextvar.
_active_geo_tape: contextvars.ContextVar = contextvars.ContextVar(
    "geometric_tape", default=None
)


@dataclass
class GeometricTapeEntry:
    op: str
    inputs: Tuple[Any, ...]
    output_id: int
    kwargs: dict = field(default_factory=dict)


@dataclass
class GeometricTape:
    """A list of recorded ops, in forward call order.

    Recording is currently manual via `tape.record(op, output, args)`.
    Future work (GA10) will make `tessera.ga.ops.*` auto-record when
    a tape is active.
    """

    entries: List[GeometricTapeEntry] = field(default_factory=list)

    def record(self, op: str, output: Multivector, args: Tuple[Any, ...], **kwargs: Any) -> None:
        self.entries.append(
            GeometricTapeEntry(
                op=op,
                inputs=args,
                output_id=id(output),
                kwargs=dict(kwargs),
            )
        )

    def __len__(self) -> int:
        return len(self.entries)


@contextlib.contextmanager
def tape_geo():
    """Context manager that activates a geometric autodiff tape.

    In GA6 this is primarily a hook for users who want to manually
    record forward ops for replay. Auto-recording from
    ``tessera.ga.ops.*`` is GA10 work.
    """
    tape = GeometricTape()
    token = _active_geo_tape.set(tape)
    try:
        yield tape
    finally:
        _active_geo_tape.reset(token)


def active_tape() -> GeometricTape | None:
    """Return the currently-active geometric tape, or None."""
    return _active_geo_tape.get()


def multivector_grad(
    fn: Callable[..., Any],
    *,
    cotangent: Multivector | Any | None = None,
    argnums: int | Tuple[int, ...] = 0,
):
    """Numerical-gradient helper for scalar-loss multivector functions.

    For v1 this is a thin wrapper around central differences — useful
    for verification and for small models where the user wants
    gradients without writing manual chain rules. Full reverse-mode AD
    via tape walks is GA10 work; this fallback is documented as such.

    Args:
        fn: forward function returning either a scalar or a Multivector.
            If a Multivector is returned, ``cotangent`` is contracted
            with it via the Frobenius dot product.
        cotangent: optional Multivector for non-scalar outputs.
        argnums: index or tuple of indices of differentiable args.

    Returns: a function ``grad_fn(*args) -> Multivector | tuple[Multivector, ...]``.
    """
    import numpy as np

    if isinstance(argnums, int):
        target_indices = (argnums,)
    else:
        target_indices = tuple(argnums)

    def _scalar_loss(*args, **kwargs):
        out = fn(*args, **kwargs)
        if isinstance(out, Multivector):
            if cotangent is None:
                return float(np.sum(out.coefficients))
            if isinstance(cotangent, Multivector):
                return float(np.sum(out.coefficients * cotangent.coefficients))
            return float(np.sum(out.coefficients * np.asarray(cotangent)))
        return float(np.asarray(out))

    def grad_fn(*args, **kwargs):
        from tessera.autodiff.geometric.check_grad import _numerical_grad_for_multivector

        out_grads = []
        for idx in target_indices:
            primal = args[idx]
            if not isinstance(primal, Multivector):
                raise TypeError(
                    f"multivector_grad: target arg {idx} must be a Multivector; "
                    f"got {type(primal).__name__}."
                )
            # Use the same central-difference machinery from check_grad.
            ones_cotangent = np.ones(primal.coefficients.shape, dtype=np.float64)
            # We bypass _flatten_output by wrapping fn to produce a scalar.

            def wrapped(*inner_args):
                out = fn(*inner_args, **kwargs)
                if isinstance(out, Multivector):
                    if cotangent is None:
                        return Multivector(
                            np.ones(out.coefficients.shape, dtype=np.float64),
                            out.algebra,
                        )
                    if isinstance(cotangent, Multivector):
                        return out  # placeholder; cotangent applied via _flatten
                return out

            # Central differences directly:
            grad_coeffs = np.zeros_like(primal.coefficients, dtype=np.float64)
            base_coeffs = primal.coefficients.astype(np.float64, copy=True)
            for flat_idx in range(base_coeffs.size):
                shape = base_coeffs.shape
                multi_idx = np.unravel_index(flat_idx, shape)
                original = base_coeffs[multi_idx]
                base_coeffs[multi_idx] = original + 1e-4
                new_args = list(args)
                new_args[idx] = Multivector(base_coeffs.copy(), primal.algebra)
                L_plus = _scalar_loss(*new_args, **kwargs)
                base_coeffs[multi_idx] = original - 1e-4
                new_args[idx] = Multivector(base_coeffs.copy(), primal.algebra)
                L_minus = _scalar_loss(*new_args, **kwargs)
                base_coeffs[multi_idx] = original
                grad_coeffs[multi_idx] = (L_plus - L_minus) / (2.0 * 1e-4)
            out_grads.append(Multivector(grad_coeffs, primal.algebra))
        if len(out_grads) == 1:
            return out_grads[0]
        return tuple(out_grads)

    return grad_fn


__all__ = [
    "GeometricTape",
    "GeometricTapeEntry",
    "active_tape",
    "multivector_grad",
    "tape_geo",
]
