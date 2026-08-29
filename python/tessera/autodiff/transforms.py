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
from typing import Any, Callable, Sequence, Union

import numpy as np

from .grad import _wrap_as_parameter
from .jvp import jvp
from .tape import TesseraAutodiffError, tape


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
    def wrapped(*args: Any, **kwargs: Any) -> Any:
        # Normalize in_axes to a per-argument tuple.
        axes: tuple[int | None, ...]
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


def _returns_this_parameter(out, param) -> bool:
    """True when `fn` handed back the very buffer it was given for `param`.

    Identity has to be decided by BUFFER identity, not by value: two arrays
    that merely compare equal have unrelated Jacobians.
    """
    if out is param:
        return True
    buf = getattr(getattr(param, "_data", None), "_data", None)
    if buf is None:
        return False
    if out is buf:
        return True
    out_arr = np.asarray(out)
    return (
        isinstance(out_arr, np.ndarray)
        and out_arr.shape == np.asarray(buf).shape
        and np.shares_memory(out_arr, buf)
    )


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

    Implementation: records the forward pass **once** on a single tape,
    then re-runs ``backward`` with ``retain_graph=True`` and a one-hot
    cotangent per output element. Cost is one forward plus
    ``out_size`` backward sweeps.

    (Before W0.4 this docstring described tape reuse while the code
    actually re-ran the full forward pass inside ``grad`` once per
    output element -- ``out_size`` forwards, not one. The
    ``retain_graph=True`` machinery in :meth:`Tape.backward` had been
    built for exactly this caller and was never wired to it.)
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
        # Promote each diff'd arg to a Parameter on a fresh copy, so the
        # tape can carry cotangents without mutating the caller's arrays.
        new_args = list(args)
        params = []
        for i in argnums_tuple:
            p = _wrap_as_parameter(args[i])
            new_args[i] = p
            params.append(p)

        # ── ONE forward pass, recorded on ONE tape ────────────────────
        with tape() as t:
            out = fn(*new_args, **kwargs)
            out_arr = np.asarray(out)
            out_shape = out_arr.shape

            # Pre-allocate Jacobian buffers, one per diff'd arg.
            jacobians = []
            for i, p in zip(argnums_tuple, params):
                ai_shape = np.asarray(args[i]).shape
                jacobians.append(
                    (p, np.zeros(out_shape + ai_shape, dtype=np.float64), ai_shape)
                )

            # An output the tape never produced has no gradient path to walk:
            # `fn` either returned one of its arguments unchanged, or returned a
            # constant. Both are legitimate — `jacrev(lambda x: x)` is the
            # identity Jacobian — and `backward` would raise on them, so resolve
            # them structurally instead.
            #
            # The pre-W0.4 implementation never hit this because it wrapped `fn`
            # in `sum(out * cotangent)` through `ops.*`, which made the target
            # tape-produced no matter what `fn` did. Reusing one tape removed
            # that accidental shield.
            if not any(entry.output_id == id(out) for entry in t.entries):
                # Only two situations make a zero/identity Jacobian provable:
                # `fn` returned one of its own arguments, or it recorded nothing
                # at all (a genuine constant). If the tape DID record ops and
                # the returned object is neither, then `fn`'s tail ran in raw
                # numpy on top of taped values — the gradient path is real but
                # unreachable from here, and returning zeros would report "this
                # function is constant" about a function that is not. Fail
                # closed instead.
                if t.entries and not any(
                    _returns_this_parameter(out, p) for p, _, _ in jacobians
                ):
                    raise TesseraAutodiffError(
                        "jacrev cannot resolve the Jacobian: the function "
                        "recorded tessera.ops.* calls but returned a value that "
                        "no recorded op produced, so its tail math ran outside "
                        "the tape (e.g. `np.tanh(ops.matmul(...))`). Returning "
                        "zeros here would claim the function is constant. "
                        "Express the tail through tessera.ops.* so it is "
                        "recorded, or differentiate the taped output directly."
                    )
                for p, jac_buf, ai_shape in jacobians:
                    if _returns_this_parameter(out, p):
                        # d(x)/d(x) = I, laid out as out_shape + in_shape.
                        size = int(np.prod(out_shape)) if out_shape else 1
                        jac_buf[...] = np.eye(size, dtype=np.float64).reshape(
                            jac_buf.shape
                        )
                    # Otherwise the output is constant in this argument and the
                    # pre-zeroed buffer is already correct.
                result = tuple(jac for _, jac, _ in jacobians)
                return result[0] if singleton else result

            # ── One backward sweep per output element, reusing the tape ──
            for k in range(out_arr.size):
                cotan = np.zeros(out_shape, dtype=np.float64)
                cotan.flat[k] = 1.0

                t.backward(
                    out,
                    cotangent=cotan,
                    retain_graph=True,
                    accumulate_param_grad=False,
                )

                out_idx = np.unravel_index(k, out_shape)
                for p, jac_buf, ai_shape in jacobians:
                    g = t.cotangent.get(id(p._data._data))
                    if g is None:
                        # This arg is not on the gradient path → zero row.
                        continue
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
