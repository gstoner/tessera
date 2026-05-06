"""
Phase 6 - Error handling and diagnostics guide contract.
"""

from pathlib import Path

import pytest

from tessera.diagnostics import (
    DiagnosticLevel,
    DiagnosticWhere,
    ErrorReporter,
    TesseraError,
    TesseraErrorCode,
    TesseraShapeError,
)
from tessera import debug
from tessera.compiler.graph_ir import GraphIRDiagnostic, SourceSpan
from tessera.compiler.target_ir import TargetIRDiagnostic
from tessera.compiler.tile_ir import TileIRDiagnostic


ROOT = Path(__file__).resolve().parents[2]
GUIDE = ROOT / "docs" / "guides" / "Tessera_Error_Handling_And_Diagnostics_Guide.md"


def test_error_handling_guide_exists_and_covers_required_axes():
    text = GUIDE.read_text(encoding="utf-8")
    required = [
        "Tessera Error Handling And Diagnostics Guide",
        "Error Model",
        "Severity Levels",
        "Diagnostic Shape",
        "Python Error Surface",
        "C++ And Runtime Status Surface",
        "Compile-Time Errors",
        "Launch-Time Errors",
        "Runtime Execution Errors",
        "Distributed And Collective Errors",
        "Numerics And Determinism Errors",
        "Autotuner And Profiling Errors",
        "Stable Error Code Reference",
        "E_NONDETERMINISTIC",
    ]
    for term in required:
        assert term in text


def test_docs_map_and_related_guides_link_error_handling_guide():
    guide_path = "docs/guides/Tessera_Error_Handling_And_Diagnostics_Guide.md"
    assert guide_path in (ROOT / "docs" / "README.md").read_text(encoding="utf-8")
    assert guide_path in (
        ROOT / "docs" / "guides" / "Tessera_QA_Reliability_Guide.md"
    ).read_text(encoding="utf-8")
    assert guide_path in (
        ROOT / "docs" / "operations" / "Tessera_Standard_Operations.md"
    ).read_text(encoding="utf-8")


def test_structured_diagnostic_fields_render_code_where_and_hints():
    where = DiagnosticWhere(
        ir_level="tile-ir",
        pass_name="TileIRLoweringPass",
        device="GPU:0",
        stream="3",
        op_name="tessera.matmul",
    )
    reporter = ErrorReporter(capture_python_location=False)
    diag = reporter.error(
        "requested 3.2 GiB, free 1.1 GiB",
        op_name="tessera.matmul",
        code=TesseraErrorCode.OOM,
        where=where,
        hints=["reduce batch", "enable checkpointing"],
    )
    text = str(diag)
    assert "E_OOM" in text
    assert "tile-ir" in text
    assert "GPU:0" in text
    assert "reduce batch" in text
    assert diag.to_dict()["code"] == "E_OOM"


def test_raise_if_errors_preserves_structured_context():
    reporter = ErrorReporter(capture_python_location=False)
    reporter.error(
        "matmul shape mismatch",
        op_name="tessera.matmul",
        code=TesseraErrorCode.SHAPE_MISMATCH,
        where=DiagnosticWhere(ir_level="graph-ir", pass_name="shape-check"),
        hints=["print shapes"],
    )

    with pytest.raises(TesseraShapeError) as exc_info:
        reporter.raise_if_errors()

    err = exc_info.value
    assert err.code == TesseraErrorCode.SHAPE_MISMATCH
    assert err.where is not None
    assert err.where.ir_level == "graph-ir"
    assert err.hints == ["print shapes"]


def test_fatal_diagnostics_count_as_errors_and_info_does_not():
    reporter = ErrorReporter(capture_python_location=False)
    reporter.info("debug detail")
    assert not reporter.has_errors()
    reporter.fatal("driver reset", code=TesseraErrorCode.DRIVER)
    assert reporter.has_errors()
    assert reporter.error_count() == 1
    assert reporter.errors[0].level == DiagnosticLevel.FATAL


def test_compiler_ir_diagnostics_bridge_to_stable_error_codes_and_where():
    graph = GraphIRDiagnostic(
        "error",
        "matmul inner dimensions differ",
        span=SourceSpan(3, 10, source_name="model.py"),
        code="GRAPH_IR_MATMUL_SHAPE",
    )
    tile = TileIRDiagnostic("error", "wait_async stage must be >= 0", "TILE_IR_WAIT_STAGE")
    target = TargetIRDiagnostic("error", "unsupported target 'quantum'", "TARGET_IR_TARGET")

    graph_structured = graph.to_tessera_diagnostic()
    assert graph_structured.code == TesseraErrorCode.SHAPE_MISMATCH
    assert graph_structured.where is not None
    assert graph_structured.where.ir_level == "graph-ir"
    assert "model.py:3:10" in str(graph_structured)
    assert "GRAPH_IR_MATMUL_SHAPE" in graph.format()

    assert tile.to_tessera_diagnostic().code == TesseraErrorCode.TILE_LOWERING
    assert tile.to_tessera_diagnostic().where.ir_level == "tile-ir"
    assert target.to_tessera_diagnostic().code == TesseraErrorCode.TARGET_CODEGEN
    assert target.to_tessera_diagnostic().where.ir_level == "target-ir"


def test_debug_determinism_failures_surface_stable_nondeterminism_code():
    state = {"value": 0}

    def unstable():
        state["value"] += 1
        return state["value"]

    with pytest.raises(TesseraError) as exc_info:
        debug.check_determinism(unstable, runs=2)

    err = exc_info.value
    assert err.code == TesseraErrorCode.NONDETERMINISTIC
    assert err.where is not None
    assert err.where.pass_name == "check_determinism"
