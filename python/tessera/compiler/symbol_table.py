"""Shared scoped symbol-table substrate for the three frontend lanes.

Before this module, each lane tracked identifier-to-SSA bindings in
its own ad-hoc dict:

  * Python ``@tessera.jit`` lane: ``self._value_types: dict[str, IRType]``
    in :mod:`tessera.compiler.graph_ir`.
  * Constrained math lanes: ``env: dict[str, str]`` in
    :mod:`tessera.compiler.ast_ir`.
  * Textual DSL parser: scattered identifier lookups in
    :mod:`tessera.compiler.frontend.parser`.

Three implementations of the same idea — and none of them supported
scoped semantics (``with`` blocks, comprehensions, nested defs) or
lookup with position info.

:class:`SymbolTable` is the substrate.  Each lane keeps its existing
visitor / parser logic but constructs a :class:`SymbolTable` per
function lowering.  When we later add scoped semantics or LSP
hover-info, both flow through one place.

Design notes
------------

* **Stacked scopes.**  ``enter_scope()`` / ``leave_scope()`` push
  and pop a fresh layer.  Lookups walk inside-out.  Define-rebind in
  an inner scope shadows the outer binding without mutating it.
* **Lightweight entries.**  :class:`SymbolEntry` carries the bound
  name, an opaque ``ir_ref`` (the lane fills this with whatever it
  uses — an :class:`IRType`, an SSA ref string, etc.), and an
  optional :class:`SourceLocation`.
* **Kind tag.**  ``kind`` distinguishes argument bindings from let
  bindings from loop-induction variables.  Lanes can ignore this
  initially; future passes (e.g., dead-binding elimination) will
  read it.
* **No magic methods.**  Plain :func:`define` / :func:`lookup` /
  :func:`undefined_lookup_error` — no ``__getitem__`` overload, no
  context manager.  Cheap to inline; easy to debug.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterator, Optional

from .diagnostics import (
    ConstrainedDiagnosticCode,
    Diagnostic,
    FrontendDiagnosticCode,
    JitDiagnosticCode,
    SourceLocation,
)


# Stable string codes for "name not defined" reported per lane.  These
# don't need their own enum members — they reuse existing codes from
# the lane-specific vocabularies.  Surfaced as a dict so consumers can
# map lane → code without a switch statement.
_UNDEFINED_NAME_CODES: dict[str, str] = {
    "tessera_jit": JitDiagnosticCode.EAGER_FALLBACK_UNSUPPORTED_BODY.value,
    "textual_dsl": FrontendDiagnosticCode.SEMANTIC_UNKNOWN_OP.value,
    "clifford_jit": ConstrainedDiagnosticCode.CLIFFORD_OP_NOT_WHITELISTED.value,
    "complex_jit": ConstrainedDiagnosticCode.COMPLEX_NON_HOLOMORPHIC.value,
    "energy_jit": ConstrainedDiagnosticCode.ENERGY_FORBIDDEN_OP.value,
}


@dataclass(frozen=True)
class SymbolEntry:
    """One row in a :class:`SymbolTable` scope."""

    name: str
    kind: str
    """``arg`` / ``let`` / ``loop_var`` / ``op_result``.  Lanes are
    free to extend this with their own kinds (e.g., ``mesh_axis``);
    the table doesn't validate the string."""

    ir_ref: Any
    """Opaque payload — whatever the lane wants to associate with the
    name.  Python JIT lane uses an :class:`IRType`; constrained
    lanes use an SSA-ref string; textual DSL uses a tuple."""

    source_position: Optional[SourceLocation] = None
    """Where this binding was introduced.  Optional — emission sites
    that can't easily derive a position pass ``None`` and lookups
    still work."""


@dataclass
class SymbolTable:
    """Lexically-scoped symbol table for one function lowering.

    Lanes call:

    1. :meth:`define(name, kind, ir_ref, source_position)` — bind a
       name in the current scope.
    2. :meth:`lookup(name)` — walk inside-out; returns ``None`` if not
       defined.
    3. :meth:`enter_scope` / :meth:`leave_scope` — push/pop a fresh
       layer.

    The table also exposes :meth:`names_in_scope` for testing and
    :meth:`bindings_introduced_in_current_scope` for dead-binding
    elimination passes (currently uncalled — wires up cleanly once
    a DBE pass lands).
    """

    lane: str
    """The lane that owns this table (``tessera_jit`` / ``textual_dsl``
    / ``clifford_jit`` / ``complex_jit`` / ``energy_jit``).  Used to
    pick the right diagnostic code on undefined-name errors."""

    _scopes: list[dict[str, SymbolEntry]] = field(
        default_factory=lambda: [{}]
    )

    def define(
        self,
        name: str,
        *,
        kind: str = "let",
        ir_ref: Any = None,
        source_position: Optional[SourceLocation] = None,
    ) -> SymbolEntry:
        """Bind ``name`` in the **innermost** scope.  Overwrites any
        existing binding at the same scope level (rebinding is legal
        in Python, so we don't reject it)."""

        entry = SymbolEntry(
            name=name,
            kind=kind,
            ir_ref=ir_ref,
            source_position=source_position,
        )
        self._scopes[-1][name] = entry
        return entry

    def lookup(self, name: str) -> Optional[SymbolEntry]:
        """Inside-out scope walk.  Returns ``None`` when ``name`` is
        not defined in any scope on the stack."""

        for scope in reversed(self._scopes):
            entry = scope.get(name)
            if entry is not None:
                return entry
        return None

    def __contains__(self, name: str) -> bool:
        return self.lookup(name) is not None

    def enter_scope(self) -> None:
        """Push a fresh scope layer."""

        self._scopes.append({})

    def leave_scope(self) -> None:
        """Pop the innermost scope.  Raises :class:`IndexError` if the
        caller tries to pop the function-level scope."""

        if len(self._scopes) <= 1:
            raise IndexError(
                "SymbolTable.leave_scope: cannot pop the "
                "function-level scope; mismatched enter/leave"
            )
        self._scopes.pop()

    def names_in_scope(self) -> tuple[str, ...]:
        """Every name visible from the current scope (inside-out)."""

        seen: set[str] = set()
        ordered: list[str] = []
        for scope in reversed(self._scopes):
            for name in scope:
                if name not in seen:
                    seen.add(name)
                    ordered.append(name)
        return tuple(ordered)

    def bindings_introduced_in_current_scope(self) -> tuple[SymbolEntry, ...]:
        """Bindings in the innermost scope only — used by passes that
        want to know what a ``with`` block / comprehension added."""

        return tuple(self._scopes[-1].values())

    def depth(self) -> int:
        """Current scope-stack depth (1 = function-level only)."""

        return len(self._scopes)

    def iter_all_entries(self) -> Iterator[SymbolEntry]:
        """Walk every entry in every scope.  Order: outermost → innermost."""

        for scope in self._scopes:
            yield from scope.values()

    # ─────────────────────────────────────────────────────────────────
    # Diagnostic helper — turns an undefined-name lookup into a typed
    # :class:`Diagnostic` so emission sites don't duplicate the
    # "lane → code" mapping logic.
    # ─────────────────────────────────────────────────────────────────

    def undefined_name_diagnostic(
        self,
        name: str,
        *,
        source_position: Optional[SourceLocation] = None,
        context: str = "",
    ) -> Diagnostic:
        """Build a typed Diagnostic for an undefined-name lookup.

        Uses the lane-appropriate code from
        :data:`_UNDEFINED_NAME_CODES`.  Falls back to the textual DSL
        code when ``self.lane`` is unrecognized.
        """

        code = _UNDEFINED_NAME_CODES.get(
            self.lane,
            FrontendDiagnosticCode.SEMANTIC_UNKNOWN_OP.value,
        )
        prefix = f"{context}: " if context else ""
        in_scope = list(self.names_in_scope())
        message = (
            f"{prefix}undefined name {name!r}; "
            f"names in scope: {in_scope}"
        )
        return Diagnostic(
            severity="error",
            code=code,
            message=message,
            source_position=source_position,
            lane=self.lane,
            detail={
                "undefined_name": name,
                "names_in_scope": in_scope,
            },
        )


__all__ = [
    "SymbolEntry",
    "SymbolTable",
]
