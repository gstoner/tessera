"""LayoutLegalityPass is wired into the named lowering pipelines (2026-06-17).

The pass used to be registered only as standalone `--tessera-layout-legality`.
It now runs inside `tessera-lower-to-x86`, `tessera-lower-to-gpu`, and the
`buildCUDA13Pipeline` (nvidia-pipeline) builders, so layout violations surface
during real lowering. This source-level guard asserts the wiring in all three
builders (robust to downstream-pass noise that an end-to-end lit run on every
pipeline would hit); the x86 firing is proven end-to-end by
`tests/tessera-ir/phase2/layout_legality_in_pipeline.mlir`.
"""

from __future__ import annotations

from pathlib import Path

_PASSES = (Path(__file__).resolve().parents[2]
           / "src" / "transforms" / "lib" / "Passes.cpp")


def _pipeline_body(src: str, anchor: str) -> str:
    """Return the chunk of Passes.cpp from `anchor` to the next pass-manager
    builder boundary — enough to check a builder's pass list."""
    i = src.index(anchor)
    return src[i:i + 1200]


def test_layout_legality_in_all_three_lowering_pipelines():
    src = _PASSES.read_text(encoding="utf-8")
    assert "createLayoutLegalityPass" in src

    # x86 builder
    x86 = _pipeline_body(src, '"tessera-lower-to-x86"')
    assert "createLayoutLegalityPass()" in x86

    # gpu builder
    gpu = _pipeline_body(src, '"tessera-lower-to-gpu"')
    assert "createLayoutLegalityPass()" in gpu

    # cuda13 / nvidia-pipeline builder
    cuda13 = _pipeline_body(src, "buildCUDA13Pipeline = ")
    assert "createLayoutLegalityPass()" in cuda13


def test_layout_legality_runs_before_symbolic_dim_equality():
    """It must run early — before SymbolicDimEqualityPass — so layout errors
    surface with the other structural diagnostics."""
    src = _PASSES.read_text(encoding="utf-8")
    for anchor in ('"tessera-lower-to-x86"', '"tessera-lower-to-gpu"',
                   "buildCUDA13Pipeline = "):
        body = _pipeline_body(src, anchor)
        assert "createLayoutLegalityPass()" in body
        assert body.index("createLayoutLegalityPass()") < \
            body.index("createSymbolicDimEqualityPass()")
