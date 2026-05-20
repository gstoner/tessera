"""U4 (2026-05-19) — ``tessera.compiler.dry_run(fn)``.

Compile a function to IR + Explain shape **without launching it**.
Useful for:

  * CI ("does this compile?") — no need to fabricate inputs.
  * Static analysis ("what kernels does this reach? which ops?") —
    walk the returned :class:`~tessera.compiler.explain.Explain` to
    extract the kernel list.
  * Diagnostic preview — call ``dry_run`` on a candidate function
    before running it, surface ``.diagnostics`` to the user.

The function is decorated through ``@tessera.jit`` so the compile
path is *identical* to what ``f(...)`` would trigger.  We just
return the JitFn's Explain object without ever calling it with
concrete inputs.
"""

from __future__ import annotations

from typing import Any, Callable, Optional

from .explain import Explain
from .jit import jit as _jit


def dry_run(
    fn: Callable[..., Any],
    *,
    target: Any = None,
    deterministic: bool = False,
    seed: Optional[int] = None,
    source: Optional[str] = None,
    source_path: Optional[str] = None,
    native_required: bool = False,
) -> Explain:
    """Return an :class:`Explain` for ``fn`` without executing it.

    Compiles ``fn`` through ``@tessera.jit`` with the supplied
    options, then returns ``jitted.explain()``.  No runtime
    dispatch happens — the inputs to the function are never
    materialized.

    Useful in CI for compile-only smokes and in notebooks for
    "preview what would happen" workflows.

    Parameters
    ----------
    fn:
        The function to dry-compile.  Must be inspectable by
        ``inspect.getsource`` or accompanied by ``source=``.
    target, deterministic, seed, source, source_path, native_required:
        Forwarded to ``@tessera.jit``.

    Returns
    -------
    Explain
        The same shape ``fn.explain()`` would produce after a real
        call, minus the call.
    """

    jit_kwargs: dict[str, Any] = {
        "deterministic": deterministic,
        "native_required": native_required,
    }
    if target is not None:
        jit_kwargs["target"] = target
    if seed is not None:
        jit_kwargs["seed"] = seed
    if source is not None:
        jit_kwargs["source"] = source
    if source_path is not None:
        jit_kwargs["source_path"] = source_path

    jitted = _jit(fn, **jit_kwargs)
    return jitted.explain()


__all__ = ["dry_run"]
