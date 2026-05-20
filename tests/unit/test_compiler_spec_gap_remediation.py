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
LOWERING_SPEC = ROOT / "docs" / "spec" / "LOWERING_PIPELINE_SPEC.md"
METALIUM_LOWERING = ROOT / "src/compiler/codegen/Tessera_Metalium_Backend/lib/Target/Metalium/Lowering/TileToMetalium.cpp"
TMEM_PTX = ROOT / "src/compiler/tile_opt_fa4/lib/Conversion/TesseraTileToPTX/LowerTileToPTX.cpp"
TILING_INTERFACE = ROOT / "src/compiler/ir/TesseraTiling.cpp"
PM_VERIFY = ROOT / "src/compiler/programming_model/tools/tessera-opt/PassPipelinesPM11.cpp"


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
        "tessera.graph.debug_value",
        "tessera.graph.replay_capture",
        "tessera-mlir my_model.py --emit=all --artifacts-dir out",
        "tessera-autotune",
        "`method=\"on_device\"`",
        "wall-clock measurement",
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
        "matmul lowers to artifact op",
        "PJRT execute is scaffolded",
        "Cerebras",
        "Rubin CPX",
        "debug markers",
        "Target IR lowering elides them",
    ]
    for term in required:
        assert term in text


def test_tile_ir_spec_uses_canonical_alloc_shared_and_mbarrier_status():
    text = TILE_SPEC.read_text(encoding="utf-8")
    required = [
        "`tile.alloc_shared`",
        "active code uses `tile.mbarrier.*`",
        "planned alias only",
        "`tile.debug_artifact`",
        "`tile.debug_barrier`",
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
        "Scoped atomics | structural verifier",
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
        "Full T1 native hardware",
        "hardware-backed CUDA/HIP/device runtime tests",
    ):
        assert term in conformance
    assert "Phases 1–3 complete; Phases 4–6 planned" not in conformance
    assert "pytest tests/unit/ tests/unit/" not in conformance


def test_compiler_reference_keeps_tiling_interface_distinct_from_tiling_pass():
    text = COMPILER_REF.read_text(encoding="utf-8")
    required = [
        "Matmul TilingInterface conservative path implemented",
        "Conv2D interface scaffolded",
        "`TilingPass` is implemented/lit-testable",
        "complete interface",
        "coverage",
    ]
    for term in required:
        assert term in text


def test_lowering_spec_documents_python_object_model_path_and_debug_dumps():
    text = LOWERING_SPEC.read_text(encoding="utf-8")
    for term in (
        "Python Object-Model Lowering Path",
        "compile_graph_module",
        "GraphIRModule",
        "ScheduleIRModule",
        "TileIRModule",
        "TargetIRModule",
        "TESSERA_DEBUG_IR=1",
        "TESSERA_DUMP_STATE=1",
    ):
        assert term in text


def test_backend_mvp_source_contracts_are_not_placeholders():
    tiling = TILING_INTERFACE.read_text(encoding="utf-8")
    # B3 (2026-05-20): the file used to carry a pre-MLIR-17
    # ``getMixedSizes`` / ``matmul_conservative_ranked_tensor``
    # scaffold, which the original drift gate locked.  Under MLIR 21
    # the auto-emission behavior changed (see
    # ``TilingInterface_NOTES.md``), so the file is now a precisely
    # documented deferred-work artifact instead of a stale scaffold.
    # Lock the new evidence-of-real-work terms — they describe
    # precisely what was removed, why, and what has to land before
    # this file grows back its body.
    for term in (
        "TilingInterface::Trait",          # confirms the ODS-side wiring is intact
        "TESSERA_ENABLE_TILING_INTERFACE", # confirms the build guard is documented
        "DeclareOpInterfaceMethods",       # confirms the MLIR-21 migration plan is referenced
        "TilingInterface_NOTES.md",        # the deferred-work tracker
        "default-failure",                 # documents the safe-fallback behavior
    ):
        assert term in tiling, (
            f"expected sentinel {term!r} missing from TesseraTiling.cpp "
            "— see ``src/compiler/ir/TilingInterface_NOTES.md`` for the "
            "v2 plan."
        )
    # The classic placeholder phrasings the previous gate flagged
    # must stay out of the file.
    assert "return failure(); // TODO: implement with tensor.extract_slice" not in tiling
    # The notes file must spell out the deferred work in detail so
    # the next contributor doesn't re-create the stale scaffold.
    notes = TILING_INTERFACE.parent.joinpath("TilingInterface_NOTES.md").read_text(
        encoding="utf-8"
    )
    for term in (
        "external-model implementation",
        "FailureOr<TilingResult>",
        "out-parameter",
        "default-failure",
    ):
        assert term in notes, (
            f"expected sentinel {term!r} missing from "
            "TilingInterface_NOTES.md"
        )

    ptx = TMEM_PTX.read_text(encoding="utf-8")
    for term in (
        "isBlackwellTarget",
        "tcgen05.mma.cta_group::2",
        "tessera_acc_tmem",
        "requires target/arch containing sm100",
    ):
        assert term in ptx
    assert "schematic placeholder" not in ptx

    metalium = METALIUM_LOWERING.read_text(encoding="utf-8")
    for term in (
        'OperationState state(loc, "tessera_metalium.matmul")',
        "tile_shape",
        "artifact_only",
        "rewriter.replaceOp",
    ):
        assert term in metalium
    assert "Metalium matmul lowering is not implemented" not in metalium


def test_memory_model_verifier_source_covers_structural_negative_cases():
    text = PM_VERIFY.read_text(encoding="utf-8")
    for term in (
        "mbarrier requires target/arch",
        "'bytes' must be > 0",
        "'order' must be relaxed, acquire, release, acq_rel, or seq_cst",
        "barrier cannot be marked divergent",
        "expected exactly 2 operands (barrier, token)",
    ):
        assert term in text
