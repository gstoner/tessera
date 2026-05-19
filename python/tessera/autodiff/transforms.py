"""JAX-style program transformations (deferred-items plan, Item 5).

Three transforms:

* ``vmap(fn, in_axes=0, out_axes=0)`` — map ``fn`` over a batch axis.
  Naive form (scan-then-stack); a tape-replay version is a perf
  follow-up.
* ``jacrev(fn, argnums=0)`` — Jacobian via reverse-mode autodiff. One
  ``backward`` pass per output dim; depends on the
  ``retain_graph=True`` re-runnable tape from Item 4.
* ``jacfwd(fn, argnums=0)`` — Jacobian via forward-mode autodiff. Uses
  the standalone JVP engine in :mod:`tessera.autodiff.jvp` (also Item 5
  — small parallel registry mirroring the existing reverse-mode
  registry).

All three follow the JAX shape contract: for ``fn: R^n → R^m``, a
Jacobian transform returns shape ``(m, n)`` for scalar input or the
appropriate tensor shape for higher-rank inputs.
"""

from __future__ import annotations

import functools
from typing import Callable, Sequence, Union

import numpy as np

from .grad import grad
from .jvp import jvp


# ─────────────────────────────────────────────────────────────────────────────
# vmap — naive per-element scan
# ─────────────────────────────────────────────────────────────────────────────


def vmap(
    fn: Callable,
    in_axes: Union[int, Sequence[int], None] = 0,
    out_axes: Union[int, None] = 0,
) -> Callable:
    """Return a function that maps ``fn`` over a batch axis.

    ``in_axes`` may be:
      * ``int`` — same axis applied to every positional argument.
      * sequence of ``int`` or ``None`` — per-argument; ``None`` means
        the argument is not batched (broadcast through).
      * ``None`` — none of the args are batched (almost-no-op).

    ``out_axes`` is the position to stack the per-element outputs into;
    only ``int`` (or ``None`` for "no stacking") supported today.

    Implementation: scan-then-stack. For perf-critical paths, a
    tape-replay version that reuses the lowered IR with a batched
    leading dim is a Phase G follow-up.
    """
    @functools.wraps(fn)
    def wrapped(*args, **kwargs):
        # Normalize in_axes to a per-argument tuple.
        if in_axes is None:
            axes = tuple(None for _ in args)
        elif isinstance(in_axes, int):
            axes = tuple(in_axes for _ in args)
        else:
            axes = tuple(in_axes)
        if len(axes) != len(args):
            raise ValueError(
                f"vmap in_axes length {len(axes)} doesn't match number of "
                f"args {len(args)}"
            )

        # Determine the batch size from the first batched arg.
        batch_size = None
        for a, ax in zip(args, axes):
            if ax is None:
                continue
            arr = np.asarray(a) if not hasattr(a, "_data") else np.asarray(a._data)
            if ax >= arr.ndim:
                raise ValueError(
                    f"vmap in_axes={ax} out of range for arg with ndim "
                    f"{arr.ndim}"
                )
            sz = arr.shape[ax]
            if batch_size is None:
                batch_size = sz
            elif batch_size != sz:
                raise ValueError(
                    f"vmap inputs have inconsistent batch sizes: "
                    f"{batch_size} vs {sz}"
                )

        if batch_size is None:
            # No batched args — call fn once, no stacking.
            return fn(*args, **kwargs)

        # Build per-element call args by indexing each batched arg along
        # its axis. Non-batched args pass through unchanged.
        outputs = []
        for i in range(batch_size):
            call_args = []
            for a, ax in zip(args, axes):
                if ax is None:
                    call_args.append(a)
                else:
                    arr = np.asarray(a) if not hasattr(a, "_data") else np.asarray(a._data)
                    sliced = np.take(arr, i, axis=ax)
                    call_args.append(sliced)
            outputs.append(fn(*call_args, **kwargs))

        if out_axes is None:
            # Caller wants the list of per-element outputs as-is.
            return outputs

        # Stack along the requested out_axes. If outputs are tuples, the
        # transform stacks each component independently.
        if isinstance(outputs[0], tuple):
            return tuple(
                np.stack([o[i] for o in outputs], axis=out_axes)
                for i in range(len(outputs[0]))
            )
        return np.stack([np.asarray(o) for o in outputs], axis=out_axes)

    return wrapped


# ─────────────────────────────────────────────────────────────────────────────
# jacrev — Jacobian via reverse-mode autodiff
# ─────────────────────────────────────────────────────────────────────────────


def jacrev(
    fn: Callable,
    argnums: Union[int, Sequence[int]] = 0,
) -> Callable:
    """Return a function that computes the Jacobian of ``fn`` via
    reverse-mode autodiff. One ``backward`` per output dim.

    Shape contract:
      * ``fn: ndarray(in_shape) → ndarray(out_shape)``
      * Jacobian shape = ``out_shape + in_shape``

    For scalar inputs / outputs reduces to ``grad``. For tuple
    ``argnums``, returns a tuple of Jacobians, one per arg.

    Implementation: re-runs ``fn`` once per output dim and seeds a
    one-hot cotangent. Uses ``retain_graph=True`` (Item 4) so the
    inner tape can be backward'd repeatedly.
    """
    argnums_tuple: tuple[int, ...]
    if isinstance(argnums, int):
        argnums_tuple = (argnums,)
        singleton = True
    else:
        argnums_tuple = tuple(int(i) for i in argnums)
        singleton = False

    @functools.wraps(fn)
    def wrapped(*args, **kwargs):
        # Sample run to get output shape.
        sample = np.asarray(fn(*args, **kwargs))
        out_shape = sample.shape

        # Pre-allocate Jacobian buffers, one per diff'd arg.
        jacobians = []
        for i in argnums_tuple:
            ai = np.asarray(args[i])
            jac = np.zeros(out_shape + ai.shape, dtype=np.float64)
            jacobians.append((i, jac, ai.shape))

        # For each output index, build the one-hot cotangent and run
        # reverse-mode grad.
        out_size = sample.size
        for k in range(out_size):
            cotan = np.zeros(out_shape, dtype=np.float64)
            cotan.flat[k] = 1.0

            # Wrap fn so it returns sum(out * cotan) — which has gradient
            # exactly the row of the Jacobian we want.
            def loss_fn(*inner_args):
                from .. import ops as _ops
                y = fn(*inner_args, **kwargs)
                return _ops.reduce(_ops.mul(y, cotan), op="sum")

            grad_fn = grad(loss_fn, argnums=argnums_tuple)
            grads = grad_fn(*args)
            if not isinstance(grads, tuple):
                grads = (grads,)

            # Place the gradient into the (k,) row of the Jacobian.
            out_idx = np.unravel_index(k, out_shape)
            for (arg_i, jac_buf, ai_shape), g in zip(jacobians, grads):
                jac_buf[out_idx] = np.asarray(g).reshape(ai_shape)

        result = tuple(jac for _, jac, _ in jacobians)
        return result[0] if singleton else result

    return wrapped


# ─────────────────────────────────────────────────────────────────────────────
# jacfwd — Jacobian via forward-mode autodiff
# ─────────────────────────────────────────────────────────────────────────────


def jacfwd(
    fn: Callable,
    argnums: Union[int, Sequence[int]] = 0,
) -> Callable:
    """Return a function that computes the Jacobian of ``fn`` via
    forward-mode (JVP-style) autodiff. One ``jvp`` call per *input*
    dim — efficient when ``in_dim < out_dim`` (the opposite regime
    from ``jacrev``).

    Uses the JVP engine in :mod:`tessera.autodiff.jvp`.
    """
    argnums_tuple: tuple[int, ...]
    if isinstance(argnums, int):
        argnums_tuple = (argnums,)
        singleton = True
    else:
        argnums_tuple = tuple(int(i) for i in argnums)
        singleton = False

    @functools.wraps(fn)
    def wrapped(*args, **kwargs):
        # Sample run to get output shape.
        sample = np.asarray(fn(*args, **kwargs))
        out_shape = sample.shape

        # Pre-allocate Jacobian buffers, one per diff'd arg.
        jacobians = []
        for i in argnums_tuple:
            ai = np.asarray(args[i])
            jac = np.zeros(out_shape + ai.shape, dtype=np.float64)
            jacobians.append((i, jac, ai))

        # For each diff'd arg, sweep one-hot tangents over its input
        # space and accumulate columns of the Jacobian.
        for slot_idx, (arg_i, jac_buf, ai) in enumerate(jacobians):
            in_size = ai.size
            for k in range(in_size):
                tangent = np.zeros_like(ai, dtype=np.float64)
                tangent.flat[k] = 1.0
                # Build a JVP wrapper: vary only the arg_i input.
                def fn_of_one(x_at_i, _arg_i=arg_i):
                    new_args = list(args)
                    new_args[_arg_i] = x_at_i
                    return fn(*new_args, **kwargs)
                _, dy = jvp(fn_of_one, ai, tangent)
                in_idx = np.unravel_index(k, ai.shape)
                # jac_buf[..., *in_idx] = dy[...]  — broadcasting via tuple slicing
                slicer = (slice(None),) * len(out_shape) + tuple(in_idx)
                jac_buf[slicer] = np.asarray(dy)

        result = tuple(jac for _, jac, _ in jacobians)
        return result[0] if singleton else result

    return wrapped


__all__ = ["vmap", "jacrev", "jacfwd"]
