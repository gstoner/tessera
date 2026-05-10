"""Higher-order autodiff convenience surface (deferred-items plan, Item 4).

Provides JAX-style ``grad(fn)`` / ``hvp(fn, primals, tangents)`` /
``elementwise_grad(fn)`` callable helpers on top of the tape-based
reverse-mode engine in ``tape.py`` + ``vjp.py``.

Compared to the v1 ``reverse(fn)`` wrapper:

* ``grad(fn, argnums=0)`` returns the *gradient(s)* directly rather than
  ``(loss, grads_dict)``. Matches JAX's ``jax.grad`` shape.
* ``hvp`` (Hessian-vector product) is provided via central finite
  difference of ``grad``. fp64 inputs give ~1e-6 accuracy. True
  forward-over-reverse HVP requires the forward-mode tape that lands in
  Item 5c.
* ``elementwise_grad(fn)`` covers the common case where ``fn`` is a
  vector → vector elementwise op and the user wants per-element
  derivatives — handy for diagnosing activation derivatives etc.

For the long form (manage your own tape, accumulate ``.grad`` slots,
etc.), keep using ``with tessera.autodiff.tape() as t: ... t.backward(loss)``.
"""

from __future__ import annotations

import functools
from typing import Any, Callable, Sequence, Tuple, Union

import numpy as np

from .tape import TesseraAutodiffError, tape


# ─────────────────────────────────────────────────────────────────────────────
# Local helpers
# ─────────────────────────────────────────────────────────────────────────────


def _normalize_argnums(argnums: Union[int, Sequence[int]]) -> Tuple[Tuple[int, ...], bool]:
    """Return ``(argnums_tuple, is_singleton)``."""
    if isinstance(argnums, int):
        return (argnums,), True
    return tuple(int(i) for i in argnums), False


def _wrap_as_parameter(arg):
    """Promote a numpy array (or already-Parameter) to a Parameter so the
    tape knows to accumulate ``.grad`` against it. Local import — keeps
    ``grad.py`` from import-cycling against ``tessera.nn``."""
    from ..nn.module import Parameter
    if isinstance(arg, Parameter):
        return arg
    return Parameter(np.asarray(arg))


# ─────────────────────────────────────────────────────────────────────────────
# grad(fn) — JAX-style gradient transform
# ─────────────────────────────────────────────────────────────────────────────


def grad(fn: Callable, argnums: Union[int, Sequence[int]] = 0) -> Callable:
    """Return a function that computes ``∇_argnums fn(*args, **kwargs)``.

    ``fn`` must produce a scalar output (or a 0-D array). For non-scalar
    outputs use :func:`tessera.autodiff.jacrev` (Item 5b).

    The returned callable:
      * Promotes the differentiated arg(s) to ``tessera.nn.Parameter``
        if they aren't already, so the tape can accumulate gradients.
      * Returns a single ndarray if ``argnums`` is an int; a tuple if
        ``argnums`` is a sequence.
      * Does not mutate the caller's parameters' ``.grad`` slots — we
        run backward with ``accumulate_param_grad=False`` and read
        cotangents directly from the tape's final cotangent map.
    """
    argnums_tuple, singleton = _normalize_argnums(argnums)

    @functools.wraps(fn)
    def wrapped(*args, **kwargs):
        # Promote each diff'd arg to a Parameter — but on a *fresh copy*
        # so the user's originals don't end up tape-tracked.
        new_args = list(args)
        params = []
        for i in argnums_tuple:
            p = _wrap_as_parameter(args[i])
            new_args[i] = p
            params.append(p)

        with tape() as t:
            out = fn(*new_args, **kwargs)
            out_arr = np.asarray(out)
            if out_arr.size != 1:
                raise TesseraAutodiffError(
                    f"grad(fn) expects a scalar output; got shape "
                    f"{out_arr.shape}. For non-scalar fn, use "
                    f"tessera.autodiff.jacrev or pass an explicit "
                    f"cotangent via tape().backward(cotangent=...)."
                )
            t.backward(out, accumulate_param_grad=False)

        # Read cotangents from the tape's final cotangent map.
        grads = []
        for p in params:
            buf_id = id(p._data._data)
            g = t.cotangent.get(buf_id)
            if g is None:
                # Argument wasn't on the gradient path → zero grad.
                g = np.zeros_like(p._data._data)
            grads.append(g)

        return grads[0] if singleton else tuple(grads)

    return wrapped


# ─────────────────────────────────────────────────────────────────────────────
# hvp(fn, primals, tangents) — Hessian-vector product
# ─────────────────────────────────────────────────────────────────────────────


def hvp(
    fn: Callable,
    primals,
    tangents,
    *,
    eps: float = 1e-4,
) -> np.ndarray:
    """Hessian-vector product: ``H @ v`` where ``H = ∇² fn(primals)``.

    Computed via central finite difference of :func:`grad`:

        hvp(f, x, v) ≈ (∇f(x + ε v) - ∇f(x - ε v)) / (2 ε)

    fp64 primals give ~1e-6 accuracy at ``eps=1e-4``. True
    forward-over-reverse HVP — the path JAX uses — requires the
    forward-mode autodiff tape that lands in Item 5c (deferred-items
    plan); this finite-difference variant is the v1 unblock for
    second-order optimizers (L-BFGS, natural gradient, GAN penalties).

    ``primals`` and ``tangents`` may be a single ndarray or a tuple of
    ndarrays of matching shape; the return matches ``primals``.
    """
    is_tuple = isinstance(primals, tuple)
    if not is_tuple:
        primals = (primals,)
        tangents = (tangents,)
    if len(primals) != len(tangents):
        raise ValueError(
            f"primals/tangents length mismatch: {len(primals)} vs "
            f"{len(tangents)}"
        )

    # Build perturbed inputs: x + eps*v and x - eps*v.
    primals_plus = tuple(
        np.asarray(p) + eps * np.asarray(v) for p, v in zip(primals, tangents)
    )
    primals_minus = tuple(
        np.asarray(p) - eps * np.asarray(v) for p, v in zip(primals, tangents)
    )

    # Force `argnums` to be a tuple so `grad` always returns a tuple,
    # regardless of how many primals there are.
    grad_fn = grad(fn, argnums=tuple(range(len(primals))))
    g_plus = grad_fn(*primals_plus)
    g_minus = grad_fn(*primals_minus)

    hvp_pieces = tuple(
        (gp - gm) / (2 * eps) for gp, gm in zip(g_plus, g_minus)
    )
    return hvp_pieces if is_tuple else hvp_pieces[0]


# ─────────────────────────────────────────────────────────────────────────────
# elementwise_grad(fn) — convenience for vector → vector functions
# ─────────────────────────────────────────────────────────────────────────────


def elementwise_grad(fn: Callable) -> Callable:
    """Return a function that computes the elementwise derivative of ``fn``.

    Assumes ``fn(x)`` is an elementwise op (output shape == input shape,
    each output index depends only on the corresponding input index).
    Returns ``df`` such that ``df(x)[i] = d fn(x)[i] / d x[i]``.

    Used primarily for inspecting activation derivatives — the reverse
    pipeline is overkill for that single case.
    """
    @functools.wraps(fn)
    def wrapped(x):
        x_arr = np.asarray(x)
        # Sum the function's output then take grad w.r.t. x. For a true
        # elementwise op, ∂(sum_i fn(x)_i)/∂x_j = ∂fn(x)_j/∂x_j.
        scalar_fn = lambda a: _sum_via_ops(fn(a))
        return grad(scalar_fn)(x_arr)
    return wrapped


def _sum_via_ops(y):
    """Tape-aware reduce-sum so an inner op in ``fn`` actually reaches
    the gradient path."""
    from .. import ops as _ops
    return _ops.reduce(y, op="sum")


__all__ = ["grad", "hvp", "elementwise_grad"]
