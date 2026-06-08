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
        # The conservative framing must remain — the memory model is
        # not fully enforced today, even after Sprint M5 (2026-05-22).
        "Current enforcement is intentionally narrower than the full memory model",
        # Sprint M5 promoted these 3 rows from `planned` → `structural
        # verifier`.  The test pins the post-promotion wording so a
        # silent regression on the verifier surface fails here.
        "Scoped atomic attribute validation (Sprint M5) | structural verifier",
        "Fence scope attribute validation (Sprint M5) | structural verifier",
        "Deterministic profile reduction enforcement (Sprint M5) | structural verifier",
        # These rows remain `planned` — they require dataflow analysis
        # or hardware-runtime evidence, not just attribute validation.
        "Complete happens-before race checking | planned",
        "Deterministic native mesh reductions (collective execution) | planned",
        # The Sprint M5 diagnostic codes must be referenced in the spec
        # so users searching for a diagnostic find the contract.
        "MEM_ATOMIC_INVALID_OP",
        "MEM_FENCE_INVALID_SCOPE",
        "MEM_DETERMINISTIC_NONDETERMINISTIC_REDUCTION",
    ):
        assert term in memory, (
            f"MEMORY_MODEL_SPEC.md missing required phrase {term!r} — "
            "either restore the §11 enforcement table or update this "
            "test if the spec has intentionally moved"
        )

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
    # B3 v2 (2026-05-20): the file ships the real MLIR 22
    # ``TilingInterface`` implementation for MatmulOp + Conv2DNHWCOp.
    # Lock the canonical v2 sentinels — they prove (a) all four
    # methods are defined on each op, (b) the v1 annotation
    # contract (``matmul_conservative_ranked_tensor``) survived, and
    # (c) the build flag defaults to ON.
    for term in (
        "matmul_conservative_ranked_tensor",   # v1 driver-observable annotation
        "MatmulOp::getTiledImplementation",    # MLIR-21 sig on MatmulOp
        "MatmulOp::getLoopIteratorTypes",
        "Conv2DNHWCOp::getTiledImplementation",
        "Conv2DNHWCOp::getLoopIteratorTypes",
        "FailureOr<TilingResult>",              # MLIR-21 return type signature
        "TESSERA_ENABLE_TILING_INTERFACE",      # build guard
    ):
        assert term in tiling, (
            f"expected sentinel {term!r} missing from TesseraTiling.cpp "
            "— see ``src/compiler/ir/TilingInterface_NOTES.md`` for the "
            "v2 contract."
        )
    # Anti-pattern placeholders that earlier scaffolds carried; the
    # v2 file should never grow them back.
    for anti_pattern in (
        "return failure(); // TODO: implement with tensor.extract_slice",
        "scaffolding with TODOs",
        "Real implementation",
    ):
        assert anti_pattern not in tiling, (
            f"placeholder phrase {anti_pattern!r} reappeared in "
            "TesseraTiling.cpp — was the v2 implementation reverted?"
        )
    # The notes file must spell out v2 status + deferred work so the
    # next contributor doesn't re-create the stale scaffold.
    notes = TILING_INTERFACE.parent.joinpath("TilingInterface_NOTES.md").read_text(
        encoding="utf-8"
    )
    for term in (
        "explicit method-list",                  # describes the MLIR-21 ODS fix
        "FailureOr<TilingResult>",               # locks the MLIR-21 signature
        "matmul_conservative_ranked_tensor",     # locks the v1 sentinel chain
        "stride/pad",                            # documents the conv2d v3 deferred work
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
