"""Activation checkpointing — Phase F2 of the execution roadmap.

``rematerialize(fn)`` wraps a callable so that, when called inside a
``tape()`` block, its intermediate ops are NOT recorded on the outer tape.
On backward, the function is re-run inside a nested tape and gradients are
extracted from there.

Trade-off: each backward pass re-runs ``fn`` (extra compute) in exchange for
not retaining ``fn``'s intermediate activations on the outer tape (memory
savings). For numpy reference paths the memory win is small; for fused
GPU kernels (Phase G+) the win is large.

Limitations of this v1 (numpy-tape) implementation:

* Inputs to ``fn`` must all be array-like (numpy / Parameter / DistributedArray).
  Non-array kwargs are forwarded verbatim and aren't differentiable.
* Only single-output functions are supported. ``fn`` must return a single
  numpy array — tuple outputs would require multi-output cotangents on the tape,
  which the v1 tape doesn't yet support.
* Parameter gradients flow correctly through the buffer-id registry.
* Non-Parameter input cotangents flow through the outer tape via the
  rematerialize entry's VJP.
"""

from __future__ import annotations

import functools
from typing import Callable

import numpy as np

from .tape import (
    TesseraAutodiffError,
    _ACTIVE_TAPE,
    _NON_ARRAY,
    _describe,
    _to_forward_arg,
)


def rematerialize(fn: Callable) -> Callable:
    """Wrap ``fn`` so its intermediate ops are not retained on the outer tape.

    On backward, re-runs ``fn`` inside a nested tape and propagates gradients.

    Returns a callable with the same forward semantics as ``fn``.
    """
    @functools.wraps(fn)
    def wrapped(*args, **kwargs):
        active = _ACTIVE_TAPE.get()
        if active is None:
            # No tape — just forward the call.
            return fn(*[_to_forward_arg(a) for a in args], **kwargs)

        # Forward without recording on the outer tape.
        token = _ACTIVE_TAPE.set(None)
        try:
            forward_args = tuple(_to_forward_arg(a) for a in args)
            out = fn(*forward_args, **kwargs)
        finally:
            _ACTIVE_TAPE.reset(token)

        if not isinstance(out, np.ndarray):
            raise TesseraAutodiffError(
                f"rematerialize requires fn to return a single ndarray; "
                f"got {type(out).__name__}"
            )

        descs_full = tuple(_describe(a) for a in args)
        array_descs = tuple(d for d in descs_full if d is not _NON_ARRAY)

        # Capture the original input objects (Parameter / arrays) so we can
        # re-pass them on backward — the buffer-id lookup depends on them.
        kept_args = args

        def _remat_vjp(dout, *fwd_arrays, **_kw):
            # Re-run fn under a nested tape; extract input cotangents from
            # the sub-tape's cotangent dict.
            from .tape import tape as _open_tape  # local to avoid cycles
            with _open_tape() as sub_t:
                # Pass the original args (Parameters etc.) so the sub-tape
                # describes them with their proper provenance.
                sub_forward = tuple(_to_forward_arg(a) for a in kept_args)
                sub_out = fn(*sub_forward, **kwargs)
                sub_t.backward(sub_out, cotangent=dout)
            # Map cotangents back to input order — the outer tape only sees
            # array-like inputs, so non-array positional args don't get a slot.
            #
            # Important: for inputs that are Parameters, their `.grad` was
            # already accumulated by the sub-tape's backward. Returning the
            # cotangent here would cause the outer tape to *also* write to
            # `.grad` (via `_accumulate_param_grad`), double-counting. So we
            # return None for Parameter slots — the outer tape skips them.
            # Non-Parameter inputs return their cotangent normally so it
            # propagates up the outer tape.
            d_in: list[np.ndarray | None] = []
            for desc in array_descs:
                if desc.param is not None:
                    d_in.append(None)
                else:
                    d_in.append(sub_t.cotangent.get(desc.array_id))
            return tuple(d_in)

        active.record("rematerialize", array_descs, dict(kwargs), out, _remat_vjp)
        return out

    return wrapped


# Convenient alias matching torch's spelling
checkpoint = rematerialize


__all__ = ["rematerialize", "checkpoint"]
