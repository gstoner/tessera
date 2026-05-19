"""Regression: every C runtime symbol that the implementation
exposes must also be declared in the public header.

Findings audit (2026-05-19) caught ``tsrIsInitialized`` shipping
in the ``.cpp`` without a corresponding declaration in
``src/runtime/include/tessera/tessera_runtime.h``.  That forced
C/C++ callers and tests to fall back to ad-hoc ``extern`` decls
— a real ABI-discipline failure.

This is a source-level guard (no compilation required): we parse
the two files and assert that every ``tsr*`` symbol the .cpp
exports under ``extern "C"`` is declared in the header.
"""

from __future__ import annotations

import re
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
HEADER_DIR = REPO_ROOT / "src" / "runtime" / "include" / "tessera"
HEADER = HEADER_DIR / "tessera_runtime.h"
IMPL = REPO_ROOT / "src" / "runtime" / "src" / "tessera_runtime.cpp"


def _public_header_surface() -> str:
    """The C ABI is split across ``tessera_runtime.h`` plus the
    sibling ``tsr_*.h`` headers it includes.  Return the union as
    a single text blob for symbol-presence checks."""
    parts: list[str] = []
    for path in sorted(HEADER_DIR.glob("*.h")):
        parts.append(path.read_text(encoding="utf-8"))
    return "\n".join(parts)


def _collect_tsr_symbols_from_cpp(text: str) -> set[str]:
    """Pull every ``tsr*`` symbol defined in an ``extern "C"`` block.

    A defining occurrence looks like::

        TsrStatus tsrSomething(...)

    or any return type followed by ``tsrSomething(...)``.  We collect
    bare function names; the test then asserts each appears in the
    header text.
    """
    # Match `<return type> tsrSomething(` at the start of a line
    # (allowing leading whitespace and optional `inline`/`static`).
    pattern = re.compile(
        r"^\s*(?:[A-Za-z_][A-Za-z0-9_]*\s+)+(tsr[A-Z][A-Za-z0-9_]*)\s*\(",
        re.MULTILINE,
    )
    return set(pattern.findall(text))


def _collect_tsr_symbols_from_header(text: str) -> set[str]:
    """Pull every ``tsr*`` symbol declared in the header."""
    pattern = re.compile(r"\b(tsr[A-Z][A-Za-z0-9_]*)\s*\(")
    return set(pattern.findall(text))


def test_every_implementation_symbol_has_a_header_declaration() -> None:
    impl_text = IMPL.read_text(encoding="utf-8")
    header_text = _public_header_surface()

    impl_symbols = _collect_tsr_symbols_from_cpp(impl_text)
    header_symbols = _collect_tsr_symbols_from_header(header_text)

    # Symbols implementation-only by design (internal helpers) can
    # be allowlisted here.  Today there are none.
    internal_only: set[str] = set()
    missing = (impl_symbols - header_symbols) - internal_only
    assert not missing, (
        f"The following C ABI symbols are defined in {IMPL.name} but "
        f"NOT declared in any public header under {HEADER_DIR.name}/:\n"
        f"  {sorted(missing)!r}\n"
        f"Add a declaration to the appropriate header so C/C++ "
        f"callers and tests don't rely on ad hoc extern decls."
    )


def test_tsrIsInitialized_is_declared() -> None:
    """Spot-check the specific symbol the audit flagged."""
    header_text = HEADER.read_text(encoding="utf-8")
    assert "tsrIsInitialized" in header_text, (
        "tsrIsInitialized is implemented in tessera_runtime.cpp but "
        "missing from the public header — that's the regression the "
        "audit caught."
    )
