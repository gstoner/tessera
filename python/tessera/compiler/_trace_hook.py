"""Phase-F abstract-interp tracing — the neutral context hook.

Kept dependency-free (no graph_ir / ops imports) so the op wrapper in
``autodiff/tape.py`` can consult ``active_tracer()`` at op-call time without a
load-time import cycle. The actual ``Tracer`` / ``TraceBuilder`` live in
``compiler/trace.py``; an active builder is any object exposing
``record_op(name, args, kwargs)``.
"""

from __future__ import annotations

import contextvars
from typing import Any, Optional

# The active trace builder (a ``compiler.trace.TraceBuilder``), or None. Sibling
# to ``autodiff.tape._ACTIVE_TAPE`` — the two are mutually exclusive contexts.
_ACTIVE_TRACER: contextvars.ContextVar[Optional[Any]] = contextvars.ContextVar(
    "_tessera_active_tracer", default=None
)


def active_tracer() -> Optional[Any]:
    """The active trace builder, or ``None`` when not tracing."""
    return _ACTIVE_TRACER.get()


def set_active_tracer(builder: Optional[Any]):
    """Bind ``builder`` as the active tracer; returns a reset token."""
    return _ACTIVE_TRACER.set(builder)


def reset_active_tracer(token) -> None:
    _ACTIVE_TRACER.reset(token)
