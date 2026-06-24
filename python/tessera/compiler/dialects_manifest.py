"""Arch-4 (2026-05-22) — Tessera MLIR dialect registration manifest.

V7 (tessera.attn, 2026-05-22) and V8 (tessera.queue, 2026-05-22) shipped
the same 5-step ritual:

  1. ODS .td file declaring the dialect.
  2. Public C++ header (``<Dialect>Dialect.h``) exposing a
     ``register<Name>Dialect(DialectRegistry&)`` function.
  3. Function body in the corresponding .cpp that ``insert``s the
     dialect class and (for dotted-name dialects) adds a
     :class:`DialectExtension` anchored on the parent ``tessera``
     dialect so the longest-prefix op-name parser routes correctly.
  4. CMake target compiled into a static library + linked into
     ``tessera-opt`` under a per-dialect ``TESSERA_HAVE_FA4_<NAME>``
     define.
  5. Call to the ``register<Name>Dialect(registry)`` function from
     ``tools/tessera-opt/tessera-opt.cpp``.

This manifest is the single Python-side source of truth that all
five touchpoints stay consistent.  The drift gate at
``tests/unit/test_dialects_manifest.py`` asserts:

  * Every entry's public header file exists on disk.
  * Every ``register_fn`` symbol appears in the corresponding .cpp.
  * Every ``cmake_flag`` is referenced in both
    ``tools/tessera-opt/CMakeLists.txt`` and
    ``tools/tessera-opt/tessera-opt.cpp``.
  * Every ``eager_load_parent`` is itself a registered dialect.
  * Adding a new dialect = adding one tuple entry + writing the 5
    files; the drift gate makes the missing touchpoint show up
    immediately.

Today's coverage: 7 dialects (tessera, tessera.attn, tessera.queue,
tessera.neighbors, tessera.solver, tessera_apple, tpp).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


_REPO_ROOT = Path(__file__).resolve().parents[3]


@dataclass(frozen=True)
class DialectSpec:
    """One row in the dialect registration manifest.

    Fields
    ------
    name
        MLIR dialect namespace string (the ``let name = ...`` value in
        the .td file).  Examples: ``"tessera"``, ``"tessera.attn"``,
        ``"tessera.queue"``, ``"tessera_apple"``, ``"tpp"``.
    target
        CMake target name (the ``add_mlir_library(...)`` first arg).
    header
        Repo-relative path to the public C++ header exposing the
        ``register<Name>Dialect`` function.
    register_fn
        Fully-qualified C++ function name (e.g.,
        ``"tessera::attn::registerAttnDialect"``).  Drift gate
        searches for this exact symbol in the matching .cpp file.
    cmake_flag
        The ``TESSERA_HAVE_*`` macro that gates the dialect's
        registration in ``tessera-opt``.  ``None`` for the root
        ``tessera`` dialect which uses ``TESSERA_HAVE_CORE_TESSERA_IR``.
    eager_load_parent
        For dotted-name dialects (``"tessera.attn"``,
        ``"tessera.queue"``): the parent dialect that anchors the
        :class:`DialectExtension` eager-load.  ``None`` for root
        dialects and for single-segment dialect names that don't need
        the trick.
    has_typedefs
        Informational — does the dialect declare any ``TypeDef``s?
        Useful context for whether standalone-lit IR fixtures need to
        spell types out (which today is blocked by an MLIR parser
        limitation for dotted names with types — see Queue note).
    standalone_lit_parseable
        Whether IR using this dialect's ops + types can parse from
        a standalone lit fixture.  ``False`` for ``tessera.queue``
        today because of the dotted-dialect-with-types MLIR parser
        limitation.  Documented in ``QueueVerifiers.cpp``.
    sprint
        Which sprint introduced the dialect registration, for
        archaeological context.
    """

    name: str
    target: str
    header: str            # repo-relative
    cpp_dir: str           # repo-relative directory containing the impl .cpp(s)
    register_fn: str
    cmake_flag: str | None
    eager_load_parent: str | None
    has_typedefs: bool
    standalone_lit_parseable: bool
    sprint: str


# ─────────────────────────────────────────────────────────────────────────
# Registry — alphabetised by dialect name.
# ─────────────────────────────────────────────────────────────────────────

REGISTERED_DIALECTS: tuple[DialectSpec, ...] = (
    DialectSpec(
        name="tessera",
        target="TesseraIR",
        header="src/compiler/ir/include/Tessera/IR/TesseraOps.h",
        cpp_dir="src/compiler/ir",
        register_fn="tessera::registerTesseraDialects",
        cmake_flag="TESSERA_HAVE_CORE_TESSERA_IR",
        eager_load_parent=None,
        has_typedefs=False,
        standalone_lit_parseable=True,
        sprint="Phase 1",
    ),
    DialectSpec(
        name="tessera.attn",
        target="TesseraAttnDialect",
        header="src/compiler/tile_opt_fa4/include/tessera/Dialect/Attn/AttnDialect.h",
        cpp_dir="src/compiler/tile_opt_fa4/lib/Dialect/Attn",
        register_fn="tessera::attn::registerAttnDialect",
        cmake_flag="TESSERA_HAVE_FA4_ATTN",
        eager_load_parent="tessera",
        has_typedefs=False,
        standalone_lit_parseable=True,
        sprint="V7",
    ),
    DialectSpec(
        name="tessera.queue",
        target="TesseraQueueDialect",
        header="src/compiler/tile_opt_fa4/include/tessera/Dialect/Queue/QueueDialect.h",
        cpp_dir="src/compiler/tile_opt_fa4/lib/Dialect/Queue",
        register_fn="tessera::queue::registerQueueDialect",
        cmake_flag="TESSERA_HAVE_FA4_QUEUE",
        eager_load_parent="tessera",
        has_typedefs=True,
        # NOTE: dotted dialect name + TypeDefs hits an MLIR parser
        # limitation — `!tessera.queue.tile_queue` parses as dialect
        # `tessera` + type `queue.tile_queue`.  Verifiers run when the
        # dialect is loaded programmatically (FA-4 lowering pipeline)
        # but standalone lit IR can't reference the types directly.
        # Documented in QueueVerifiers.cpp.
        standalone_lit_parseable=False,
        sprint="V8",
    ),
    DialectSpec(
        name="tessera_rocm",
        target="TesseraROCMDialect",
        header="src/compiler/codegen/Tessera_ROCM_Backend/include/TesseraROCM/IR/TesseraROCMDialect.td",
        cpp_dir="src/compiler/codegen/Tessera_ROCM_Backend/lib/Conversion",
        register_fn="mlir::tessera_rocm::registerTesseraROCMDialects",
        cmake_flag=None,
        eager_load_parent=None,
        has_typedefs=True,
        standalone_lit_parseable=True,
        sprint="ROCm Tile-IR convergence",
    ),
    # Sprint 9 value-lane Tile IR dialect (src/compiler/ir), grown in the TIRx
    # review (C1/C3/C5) with first-class attributes: #tile.layout / #tile.swizzle
    # / #tile.barrier / #tile.pipeline_state / #tile.pipeline_depths. Built into
    # core TesseraIR and registered unconditionally in tessera-opt.cpp (no
    # cmake_flag). Holds AttrDefs but no TypeDefs today.
    DialectSpec(
        name="tile",
        target="TesseraTileDialect",
        header="src/compiler/ir/include/Tessera/Dialect/Tile/TileDialect.h",
        cpp_dir="src/compiler/ir",
        register_fn="tessera::tile::registerTileDialect",
        cmake_flag=None,
        eager_load_parent="tessera",
        has_typedefs=False,
        standalone_lit_parseable=True,
        sprint="Sprint 9 / C1 (TIRx)",
    ),
)


# ─────────────────────────────────────────────────────────────────────────
# Public helpers
# ─────────────────────────────────────────────────────────────────────────


def all_dialect_names() -> tuple[str, ...]:
    """Return all registered dialect names, sorted."""
    return tuple(sorted(d.name for d in REGISTERED_DIALECTS))


def dialect_lookup(name: str) -> DialectSpec | None:
    """Look up a dialect by its MLIR namespace name."""
    for spec in REGISTERED_DIALECTS:
        if spec.name == name:
            return spec
    return None


def header_path(spec: DialectSpec) -> Path:
    """Return the absolute path to a dialect's public header."""
    return _REPO_ROOT / spec.header


def cpp_files_for(spec: DialectSpec) -> tuple[Path, ...]:
    """Return the .cpp files in the dialect's declared ``cpp_dir``.

    The directory is explicit on each DialectSpec (rather than
    derived from the header path) because the include/lib layouts
    vary across the codebase — the root ``tessera`` dialect lives in
    ``src/compiler/ir/`` while the FA-4 dialects use the canonical
    ``include/.../<Name>/`` + ``lib/Dialect/<Name>/`` split.
    """
    cpp_dir = _REPO_ROOT / spec.cpp_dir
    if not cpp_dir.is_dir():
        return ()
    return tuple(sorted(cpp_dir.glob("*.cpp")))


__all__ = [
    "DialectSpec",
    "REGISTERED_DIALECTS",
    "all_dialect_names",
    "dialect_lookup",
    "header_path",
    "cpp_files_for",
]
