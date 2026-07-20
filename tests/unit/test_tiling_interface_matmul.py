"""Structural guard for the matmul TilingInterface v2 implementation
(B3 v2, 2026-05-20).

The B3-v1 work landed a deferred-work doc + a precisely-documented
stub; B3-v2 ships the actual MLIR 23 ``TilingInterface`` impl for
``tessera.matmul`` and ``tessera.conv2d_nhwc``.  These guards lock the
v2 invariants:

  * ODS declares the explicit method list on
    ``DeclareOpInterfaceMethods<TilingInterface, [...]>`` for both
    ops (the only form MLIR 23's ODS generator picks up).
  * The C++ impl file carries the canonical annotation sentinel
    (``matmul_conservative_ranked_tensor``) plus all four method
    definitions matching the MLIR 23 signatures.
  * The build flag default is ON; the
    ``-DTESSERA_DISABLE_TILING_INTERFACE`` opt-out is documented in
    the source.
  * The generated ``TesseraOps.h.inc`` (built artifact) actually
    contains the per-Op TilingInterface method declarations for
    both ``MatmulOp`` and ``Conv2DNHWCOp`` — proof that the ODS
    switch we made actually emits the decls.

Skipped cleanly when the build artifacts aren't present.
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[2]
TESSERA_OPS_TD = ROOT / "src/compiler/ir/TesseraOps.td"
TESSERA_TILING_CPP = ROOT / "src/compiler/ir/TesseraTiling.cpp"
TILING_NOTES = ROOT / "src/compiler/ir/TilingInterface_NOTES.md"
GENERATED_HEADER_CANDIDATES = (
    ROOT / "build" / "src/compiler/ir/TesseraOps.h.inc",
    ROOT / "build-rocm" / "src/compiler/ir/TesseraOps.h.inc",
)


def test_ods_declares_explicit_method_list_for_matmul() -> None:
    """B3 v2 lock: MatmulOp's TilingInterface decl uses the explicit
    method-list form (the only form MLIR 23 picks up reliably)."""

    ods = TESSERA_OPS_TD.read_text(encoding="utf-8")
    # Find the MatmulOp block + extract its bracket bundle.
    matmul_block_re = re.compile(
        r"def Tessera_MatmulOp[\s\S]*?(?=^def Tessera_|\Z)", re.MULTILINE
    )
    block = matmul_block_re.search(ods)
    assert block, "could not locate Tessera_MatmulOp in TesseraOps.td"
    body = block.group(0)
    assert (
        "DeclareOpInterfaceMethods<TilingInterface, [" in body
    ), "MatmulOp must use the explicit method-list form of DeclareOpInterfaceMethods<TilingInterface>"
    for method in (
        "getLoopIteratorTypes",
        "getIterationDomain",
        "getTiledImplementation",
        "getResultTilePosition",
    ):
        assert (
            f'"{method}"' in body
        ), f"MatmulOp must list {method!r} in its TilingInterface method bundle"


def test_ods_declares_explicit_method_list_for_conv2d() -> None:
    """Same lock for Conv2DNHWCOp."""

    ods = TESSERA_OPS_TD.read_text(encoding="utf-8")
    block_re = re.compile(
        r"def Tessera_Conv2DNHWCOp[\s\S]*?(?=^def Tessera_|\Z)", re.MULTILINE
    )
    block = block_re.search(ods)
    assert block, "could not locate Tessera_Conv2DNHWCOp in TesseraOps.td"
    body = block.group(0)
    assert "DeclareOpInterfaceMethods<TilingInterface, [" in body, (
        "Conv2DNHWCOp must use the explicit method-list form"
    )
    for method in (
        "getLoopIteratorTypes",
        "getIterationDomain",
        "getTiledImplementation",
        "getResultTilePosition",
    ):
        assert f'"{method}"' in body, (
            f"Conv2DNHWCOp must list {method!r} in its TilingInterface bundle"
        )


def test_cpp_impl_defines_all_four_matmul_methods() -> None:
    """The C++ side actually defines the four MLIR 23 methods on
    MatmulOp with the correct return types."""

    cpp = TESSERA_TILING_CPP.read_text(encoding="utf-8")
    for sig_fragment in (
        "MatmulOp::getLoopIteratorTypes",
        "MatmulOp::getIterationDomain",
        "MatmulOp::getTiledImplementation",
        "MatmulOp::getResultTilePosition",
    ):
        assert sig_fragment in cpp, (
            f"missing {sig_fragment!r} definition in TesseraTiling.cpp"
        )
    # The annotation sentinel that tile drivers can FileCheck against.
    assert "matmul_conservative_ranked_tensor" in cpp, (
        "v1 conservative annotation sentinel missing — the tile "
        "interface lost its driver-observable hook."
    )
    # MLIR 23 return type for getTiledImplementation.
    assert "FailureOr<TilingResult>" in cpp, (
        "TilingInterface return type drifted from MLIR 23 signature"
    )


def test_cpp_impl_defines_all_four_conv2d_methods() -> None:
    cpp = TESSERA_TILING_CPP.read_text(encoding="utf-8")
    for sig_fragment in (
        "Conv2DNHWCOp::getLoopIteratorTypes",
        "Conv2DNHWCOp::getIterationDomain",
        "Conv2DNHWCOp::getTiledImplementation",
        "Conv2DNHWCOp::getResultTilePosition",
    ):
        assert sig_fragment in cpp, (
            f"missing {sig_fragment!r} definition in TesseraTiling.cpp"
        )
    # Conv2D still honest about deferring the stride/pad work.
    assert "stride/pad" in cpp, (
        "Conv2D impl must keep the explicit deferred-work note "
        "documenting why getTiledImplementation returns failure()"
    )


def test_build_flag_default_is_on() -> None:
    """``TESSERA_ENABLE_TILING_INTERFACE`` defaults to 1; the opt-out
    is via ``-DTESSERA_DISABLE_TILING_INTERFACE``."""

    cpp = TESSERA_TILING_CPP.read_text(encoding="utf-8")
    # Default-ON branch present.
    assert "#    define TESSERA_ENABLE_TILING_INTERFACE 1" in cpp, (
        "build-flag default must be ON in B3 v2"
    )
    # Opt-out branch present.
    assert "TESSERA_DISABLE_TILING_INTERFACE" in cpp, (
        "the -DTESSERA_DISABLE_TILING_INTERFACE opt-out should still "
        "be honored for downstream consumers"
    )


def test_generated_header_has_per_op_tiling_methods() -> None:
    """When the build artifacts are present, the generated header
    must carry the per-Op TilingInterface method declarations for
    both ops — proof the ODS-side switch actually emits the decls.

    Skips gracefully if no build dir exists.
    """

    for candidate in GENERATED_HEADER_CANDIDATES:
        if candidate.exists():
            generated = candidate.read_text(encoding="utf-8")
            break
    else:
        pytest.skip("TesseraOps.h.inc not built; skip generated-header guard")

    # Both ops should now have all four interface methods declared.
    # We don't pin exact byte offsets — just confirm presence
    # somewhere in the file, since the generator may re-order them.
    for op_class in ("MatmulOp", "Conv2DNHWCOp"):
        # Find the class scope.
        m = re.search(
            rf"^class {op_class} :[\s\S]*?^\}};$", generated, re.MULTILINE
        )
        assert m, f"could not locate class {op_class} in TesseraOps.h.inc"
        scope = m.group(0)
        for method in (
            "getLoopIteratorTypes",
            "getIterationDomain",
            "getTiledImplementation",
            "getResultTilePosition",
        ):
            assert method in scope, (
                f"{op_class} class scope missing decl for {method!r} "
                "— ODS DeclareOpInterfaceMethods<TilingInterface, "
                "[...]> didn't emit per-Op decls"
            )


def test_notes_doc_reflects_v2_status() -> None:
    """The notes doc should describe the matmul v1 ship + conv2d
    deferred state, not the old "scaffolding with TODOs" disclaimer.
    """

    notes = TILING_NOTES.read_text(encoding="utf-8")
    # Anti-pattern: the old "scaffolding with TODOs" warning must be
    # gone (was the original B3-v1 sentinel target).
    assert "scaffolding with TODOs" not in notes, (
        "the v1 scaffolding-warning text should have been replaced"
    )
    # Required signals for the v2 honest description.
    for term in (
        "matmul_conservative_ranked_tensor",
        "FailureOr<TilingResult>",
        "stride/pad",
    ):
        assert term in notes, (
            f"notes doc missing {term!r} — v2 description drifted"
        )
