"""Notebook-safe factory for source-driven JIT construction.

P1-6 of the 2026-05-19 compiler-surface consolidation: developers
working in REPLs / notebooks hit ``JIT_SOURCE_UNAVAILABLE`` because
``inspect.getsource(fn)`` returns nothing for dynamically-defined
functions.  The escape hatch was always ``ts.jit(fn, source=<str>)``
plus an ``exec(...)`` dance to materialize the function from the
string — fiddly, error-prone, and easy to forget the
``source=`` keyword.

:func:`from_text` replaces that pattern with a single blessed call:

    >>> import tessera as ts
    >>> f = ts.from_text(\"\"\"
    ...     def f(x):
    ...         return ts.ops.relu(x)
    ... \"\"\")
    >>> print(f.explain())

Behind the scenes it executes the source in a fresh namespace that
exposes ``ts`` / ``tessera`` / ``np``, picks the named function (or
the single ``def`` when ``name=None``), and routes through
``tessera.jit`` with ``source=`` populated so AST inspection
succeeds.

Errors are explicit:

  * Multiple top-level ``def``s + no ``name=`` → :class:`ValueError`.
  * Named function not found → :class:`KeyError`.
  * Source raises during exec → re-raised with a contextualizing message.
"""

from __future__ import annotations

import ast
import textwrap
from typing import Any, Callable, Optional

from .jit import jit as _jit


_NAMESPACE_DEFAULTS: dict[str, str] = {
    "ts": "tessera",
    "tessera": "tessera",
    "np": "numpy",
}


def _build_namespace(extras: Optional[dict[str, Any]] = None) -> dict[str, Any]:
    """Build the exec namespace.  Pre-populates with ``ts`` / ``tessera`` /
    ``np`` so notebook-style source compiles without explicit imports."""

    import importlib

    ns: dict[str, Any] = {}
    for alias, module_name in _NAMESPACE_DEFAULTS.items():
        try:
            ns[alias] = importlib.import_module(module_name)
        except ImportError:  # pragma: no cover — numpy is a hard dep
            continue
    if extras:
        ns.update(extras)
    return ns


def _enumerate_def_names(source: str) -> list[str]:
    """Return the names of top-level ``def`` statements in ``source``."""

    try:
        tree = ast.parse(source)
    except SyntaxError as exc:
        raise ValueError(
            f"tessera.from_text: source does not parse as Python: {exc}"
        ) from exc
    return [
        node.name
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    ]


def from_text(
    source: str,
    *,
    name: Optional[str] = None,
    namespace: Optional[dict[str, Any]] = None,
    target: Any = None,
    deterministic: bool = False,
    seed: Optional[int] = None,
    native_required: bool = False,
    **jit_kwargs: Any,
) -> Callable[..., Any]:
    """Compile a JIT function from a source string.

    Parameters
    ----------
    source:
        Python source containing one or more ``def`` statements.
        Trailing/leading whitespace is dedented before exec.
    name:
        Function name to extract.  When ``None`` and the source
        contains exactly one ``def``, that function is used
        automatically.  When ``None`` and multiple ``def``s are
        present, raises :class:`ValueError`.
    namespace:
        Optional extra names to expose during ``exec`` (in addition
        to the default ``ts``/``tessera``/``np`` aliases).
    target:
        Forwarded to ``@tessera.jit(target=...)`` — accepts a
        :class:`GPUTargetProfile` instance or one of the documented
        string aliases (``"rocm"`` / ``"apple_cpu"``
        / ``"apple_gpu"``).  Documented as a first-class kwarg in U5
        (2026-05-19) so notebook flows don't have to drill through
        ``**jit_kwargs``.
    deterministic:
        Forwarded to ``@tessera.jit(deterministic=...)``.  Refuses
        functions with non-seeded random effects when ``True``.
    seed:
        Forwarded to ``@tessera.jit(seed=...)`` — required when
        ``deterministic=True`` and the function uses random ops.
    native_required:
        Forwarded to ``@tessera.jit(native_required=...)``.  Raises
        :class:`~tessera.compiler.TesseraNativeRequiredError` on
        any fallback condition instead of silently dropping to the
        reference path.
    **jit_kwargs:
        Forwarded to :func:`tessera.compiler.jit.jit` for any
        remaining options.  ``source=`` is populated automatically
        from the dedented input — don't pass it manually.

    Returns
    -------
    JitFn
        A standard ``@tessera.jit``-decorated function suitable for
        immediate invocation or ``.explain()`` inspection.

    Examples
    --------

    Notebook-style construction with a target + native_required::

        f = tessera.from_text(
            \"\"\"
                def f(x, y):
                    return ts.ops.matmul(x, y)
            \"\"\",
            target="apple_gpu",
            native_required=True,
        )
    """

    if "source" in jit_kwargs:
        raise TypeError(
            "tessera.from_text: do not pass `source=` — the factory "
            "populates it automatically from the dedented input string"
        )

    dedented = textwrap.dedent(source).strip("\n") + "\n"

    def_names = _enumerate_def_names(dedented)
    if not def_names:
        raise ValueError(
            "tessera.from_text: source contains no top-level `def` "
            "statements; nothing to compile"
        )

    if name is None:
        if len(def_names) > 1:
            raise ValueError(
                f"tessera.from_text: source defines multiple top-level "
                f"functions ({def_names}); pass `name=` to disambiguate"
            )
        name = def_names[0]
    elif name not in def_names:
        raise KeyError(
            f"tessera.from_text: name={name!r} is not defined in the "
            f"source; available: {def_names}"
        )

    ns = _build_namespace(namespace)
    try:
        exec(dedented, ns)  # noqa: S102 — by construction, source is caller-supplied
    except Exception as exc:
        raise RuntimeError(
            f"tessera.from_text: failed to exec source: {exc}"
        ) from exc

    fn = ns.get(name)
    if not callable(fn):
        raise RuntimeError(
            f"tessera.from_text: exec did not produce a callable "
            f"named {name!r}; got {type(fn).__name__}"
        )

    # Route through @tessera.jit with source= populated so AST
    # inspection succeeds.  The decorator emits an info-level
    # ``JIT_SOURCE_PROVIDED`` diagnostic.  Documented kwargs
    # (target / deterministic / seed / native_required) override
    # any matching values that slipped through **jit_kwargs.
    if target is not None:
        jit_kwargs["target"] = target
    if seed is not None:
        jit_kwargs["seed"] = seed
    jit_kwargs["deterministic"] = deterministic
    jit_kwargs["native_required"] = native_required
    return _jit(fn, source=dedented, **jit_kwargs)


__all__ = ["from_text"]
