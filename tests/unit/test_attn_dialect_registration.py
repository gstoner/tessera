"""Sprint V7 (2026-05-22) — structural guard for tessera.attn dialect
registration in tessera-opt.

V7 lands the dialect registration plumbing (public header,
`registerAttnDialect()` function, tessera-opt.cpp wiring, CMake
linking).  These tests pin the source-level content so a future
edit that accidentally removes the registration fails this fast
Python sweep rather than waiting for a manual `--show-dialects`
check.

What V7 closes:
  - Public header `Tessera/Dialect/Attn/AttnDialect.h` exposing
    `registerAttnDialect()` (mirrors the Apple backend pattern).
  - `registerAttnDialect(DialectRegistry&)` body in AttnOps.cpp.
  - tessera-opt.cpp includes the header + calls registerAttnDialect
    when TESSERA_HAVE_FA4_ATTN is defined.
  - tessera-opt CMake links TesseraAttnDialect library + sets
    TESSERA_HAVE_FA4_ATTN compile definition.

What V7 does NOT close (tracked as V7b):
  - End-to-end lit-exercise of `tessera.attn.*` ops.  The dialect
    appears in `--show-dialects` and its symbols are linked into
    the binary, but MLIR's parser does not lazy-load it for
    standalone IR fixtures (no pass references it).  Loading
    happens correctly when a pass that uses the dialect runs,
    which is why existing Phase 3 fixtures using
    `tessera.attn.scaled_dot_product` carry `XFAIL: *`.
"""
from __future__ import annotations

from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
ATTN_HEADER = (
    REPO_ROOT / "src" / "compiler" / "tile_opt_fa4" / "include"
    / "tessera" / "Dialect" / "Attn" / "AttnDialect.h"
)
ATTN_OPS_CPP = (
    REPO_ROOT / "src" / "compiler" / "tile_opt_fa4" / "lib" / "Dialect"
    / "Attn" / "AttnOps.cpp"
)
TESSERA_OPT_CPP = (
    REPO_ROOT / "tools" / "tessera-opt" / "tessera-opt.cpp"
)
TESSERA_OPT_CMAKE = (
    REPO_ROOT / "tools" / "tessera-opt" / "CMakeLists.txt"
)


def test_v7_public_header_exists() -> None:
    """V7 public header for the Attn dialect must exist + expose the
    canonical registration entry point."""
    assert ATTN_HEADER.exists(), (
        f"V7 header missing: {ATTN_HEADER.relative_to(REPO_ROOT)}"
    )
    text = ATTN_HEADER.read_text()
    assert "registerAttnDialect" in text, (
        "AttnDialect.h must declare registerAttnDialect()"
    )
    assert "TESSERA_DIALECT_ATTN_DIALECT_H" in text, (
        "AttnDialect.h must use the canonical include guard"
    )
    # The header must include the tablegen-generated dialect class.
    assert 'include "AttnDialect.h.inc"' in text


def test_v7_register_function_implemented() -> None:
    """`registerAttnDialect()` body must be present in AttnOps.cpp."""
    text = ATTN_OPS_CPP.read_text()
    assert "void registerAttnDialect" in text, (
        "AttnOps.cpp must implement registerAttnDialect()"
    )
    assert "registry.insert<TesseraAttnDialect>()" in text, (
        "registerAttnDialect must insert TesseraAttnDialect into the "
        "DialectRegistry — that's the contract Apple backend uses"
    )


def test_v7_tessera_opt_includes_header() -> None:
    """tessera-opt.cpp must include the Attn header under
    TESSERA_HAVE_FA4_ATTN."""
    text = TESSERA_OPT_CPP.read_text()
    assert "TESSERA_HAVE_FA4_ATTN" in text, (
        "tessera-opt.cpp must guard the Attn include behind "
        "TESSERA_HAVE_FA4_ATTN so non-FA4 builds keep working"
    )
    assert 'include "tessera/Dialect/Attn/AttnDialect.h"' in text


def test_v7_tessera_opt_calls_register() -> None:
    """tessera-opt.cpp main() must call registerAttnDialect when the
    feature is device_verified_jit in."""
    text = TESSERA_OPT_CPP.read_text()
    assert "tessera::attn::registerAttnDialect(registry)" in text, (
        "tessera-opt main() must call "
        "tessera::attn::registerAttnDialect(registry) inside the "
        "TESSERA_HAVE_FA4_ATTN ifdef block"
    )


def test_v7_cmake_links_library() -> None:
    """tools/tessera-opt/CMakeLists.txt must wire the
    TesseraAttnDialect library + set TESSERA_HAVE_FA4_ATTN."""
    text = TESSERA_OPT_CMAKE.read_text()
    assert "TesseraAttnDialect" in text, (
        "CMakeLists.txt must link TesseraAttnDialect into tessera-opt"
    )
    assert "TESSERA_HAVE_FA4_ATTN" in text, (
        "CMakeLists.txt must set TESSERA_HAVE_FA4_ATTN compile def "
        "when the Attn dialect target is present"
    )
    # Include path for the dialect's header tree.
    assert "tile_opt_fa4/include" in text, (
        "CMakeLists.txt must add the tile_opt_fa4 include directory"
    )


def test_v7_documented_partial_closure() -> None:
    """V7 documentation: the closure is partial — dialect registers
    cleanly but parser lazy-load doesn't fire for standalone IR
    fixtures.  Document this honestly so a future contributor knows
    what's left (V7b)."""
    text = ATTN_OPS_CPP.read_text()
    # The V7 closure comment must acknowledge the registration pattern
    # mirrors the Apple backend, so the next contributor understands
    # the architectural choice.
    assert "Sprint V7" in text and "Apple" in text, (
        "AttnOps.cpp registerAttnDialect comment must reference "
        "Sprint V7 + the Apple backend pattern (the canonical "
        "registration model)"
    )
