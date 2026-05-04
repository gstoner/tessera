"""
Regression guards for compiler spec gap remediation.
"""

from pathlib import Path

from tessera.compiler.op_catalog import OP_SPECS


ROOT = Path(__file__).resolve().parents[2]
PYTHON_SPEC = ROOT / "docs" / "spec" / "PYTHON_API_SPEC.md"
GRAPH_SPEC = ROOT / "docs" / "spec" / "GRAPH_IR_SPEC.md"
RUNTIME_SPEC = ROOT / "docs" / "spec" / "RUNTIME_ABI_SPEC.md"
TARGET_SPEC = ROOT / "docs" / "spec" / "TARGET_IR_SPEC.md"
TILE_SPEC = ROOT / "docs" / "spec" / "TILE_IR.md"
MEMORY_SPEC = ROOT / "docs" / "spec" / "MEMORY_MODEL_SPEC.md"
SHAPE_SPEC = ROOT / "docs" / "spec" / "SHAPE_SYSTEM.md"
CONFORMANCE = ROOT / "docs" / "spec" / "CONFORMANCE.md"
COMPILER_REF = ROOT / "docs" / "spec" / "COMPILER_REFERENCE.md"


def test_python_api_spec_lists_current_runtime_op_catalog():
    text = PYTHON_SPEC.read_text(encoding="utf-8")
    missing = [
        name
        for name in sorted(OP_SPECS)
        if name not in text and OP_SPECS[name].graph_name not in text
    ]
    assert missing == []

    required_status_terms = [
        "NumPy NHWC/HWIO reference",
        "Single-rank/mock no-op",
        "Operator registry behavior",
        "Native hardware runtime support",
    ]
    for term in required_status_terms:
        assert term in text


def test_graph_ir_spec_documents_ods_extensions_and_string_based_ops():
    text = GRAPH_SPEC.read_text(encoding="utf-8")
    required = [
        "Additional ODS-backed semantic ops",
        "tessera.layer_norm",
        "tessera.dropout",
        "tessera.all_reduce",
        "tessera.fft",
        "tessera.kv_cache.create",
        "tessera.arch.parameter",
        "String-based ops referenced in canonicalization",
        "`tessera.add`",
    ]
    for term in required:
        assert term in text

    assert "not yet defined in `TesseraOps.td`" not in text


def test_runtime_abi_spec_uses_current_status_labels_for_runtime_surfaces():
    text = RUNTIME_SPEC.read_text(encoding="utf-8")
    required = [
        "src/runtime/src/backend/cpu_backend.cpp",
        "implemented / mock-runtime",
        "hardware-runtime when built with `TESSERA_ENABLE_CUDA`",
        "hardware-runtime when built with `TESSERA_ENABLE_HIP`",
        "TesseraRuntime` Python class in `python/tessera/runtime.py` is the current",
        "Artifact compile/load/get-kernel/launch",
    ]
    for term in required:
        assert term in text

    assert "to be implemented in `python/tessera/runtime.py`" not in text
    assert "Phase 6 planned" not in text


def test_target_ir_spec_splits_artifact_runtime_and_backend_status():
    text = TARGET_SPEC.read_text(encoding="utf-8")
    required = [
        "Backend Status Appendix",
        "Semantic compiler behavior",
        "Target artifact generation",
        "Mock/runtime fallback",
        "Native hardware runtime",
        "placeholder kernels are not native-runtime claims",
        "matmul lowering remains incomplete",
        "PJRT execute is scaffolded",
        "Cerebras",
        "Rubin CPX",
    ]
    for term in required:
        assert term in text


def test_tile_ir_spec_uses_canonical_alloc_shared_and_mbarrier_status():
    text = TILE_SPEC.read_text(encoding="utf-8")
    required = [
        "`tile.alloc_shared`",
        "active code uses `tile.mbarrier.*`",
        "planned alias only",
        "stubbed / lit-testable until real Blackwell PTX operands land",
    ]
    for term in required:
        assert term in text

    assert "### 3.1 `tshared.alloc`" not in text
    assert "| `tshared.alloc` |" not in text


def test_memory_shape_and_conformance_specs_record_deferred_work_truthfully():
    memory = MEMORY_SPEC.read_text(encoding="utf-8")
    for term in (
        "Current enforcement is intentionally narrower than the full memory model",
        "Scoped atomics | planned",
        "Complete happens-before race checking | planned",
        "Deterministic native mesh reductions | planned",
    ):
        assert term in memory

    shape = SHAPE_SPEC.read_text(encoding="utf-8")
    for term in (
        "Current Implementation Map",
        "Python shape objects and helpers",
        "MLIR verifier coverage is incremental",
        "downgrade that claim to `planned`",
    ):
        assert term in shape

    conformance = CONFORMANCE.read_text(encoding="utf-8")
    for term in (
        "implemented / scaffolded / lit-testable",
        "Native NCCL/RCCL/MPI cluster execution is",
        "Graduation criteria",
        "Phase 8",
    ):
        assert term in conformance


def test_compiler_reference_keeps_tiling_interface_distinct_from_tiling_pass():
    text = COMPILER_REF.read_text(encoding="utf-8")
    required = [
        "TilingInterface methods scaffolded",
        "`TilingPass` is implemented/lit-testable",
        "complete interface\ncoverage",
    ]
    for term in required:
        assert term in text
