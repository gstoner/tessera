"""
Phase 6 - Language/IR, memory model, and layout/movement spec contracts.
"""

from pathlib import Path

from tessera.compiler.autotune_v2 import GEMMWorkload
from tessera.compiler.graph_ir import _dtype_to_ir_type
from tessera.compiler.gpu_target import GPUTargetProfile, ISA


ROOT = Path(__file__).resolve().parents[2]
LANG = ROOT / "docs" / "spec" / "LANGUAGE_AND_IR_SPEC.md"
MEMORY = ROOT / "docs" / "spec" / "MEMORY_MODEL_SPEC.md"
LAYOUT = ROOT / "docs" / "guides" / "Tessera_Tensor_Layout_And_Data_Movement_Guide.md"


def test_language_ir_spec_covers_grammar_ir_layers_rubin_dtypes_and_mbarriers():
    text = LANG.read_text(encoding="utf-8")
    required = [
        "Tessera Language And IR Specification",
        "Core BNF",
        "fp6_e2m3",
        "fp6_e3m2",
        "fp4_e2m1",
        "nvfp4",
        "Graph IR",
        "Schedule IR",
        "Tile IR",
        "Target IR",
        "tile.mbarrier.arrive_expect_tx",
        "Hopper forward rule",
    ]
    for term in required:
        assert term in text


def test_memory_model_spec_covers_scopes_happens_before_and_mbarriers():
    text = MEMORY.read_text(encoding="utf-8")
    required = [
        "Tessera Memory Model Specification",
        "Memory Spaces",
        "Synchronization Scopes",
        "mbarrier.arrive_expect_tx",
        "Happens-Before",
        "Data race",
        "Backend Mapping",
    ]
    for term in required:
        assert term in text


def test_layout_movement_guide_covers_layouts_movement_and_schedule_artifacts():
    text = LAYOUT.read_text(encoding="utf-8")
    required = [
        "Tessera Tensor Layout And Data Movement Guide",
        "paged(page_size, order)",
        "Hopper+ Mbarrier Movement Pattern",
        "tile.async_copy",
        "tile.mbarrier.try_wait",
        "Schedule Artifact Requirements",
        "nvfp4",
    ]
    for term in required:
        assert term in text


def test_docs_map_links_new_specs_and_guide():
    text = (ROOT / "docs" / "README.md").read_text(encoding="utf-8")
    for path in (
        "docs/spec/LANGUAGE_AND_IR_SPEC.md",
        "docs/spec/MEMORY_MODEL_SPEC.md",
        "docs/guides/Tessera_Tensor_Layout_And_Data_Movement_Guide.md",
    ):
        assert path in text


def test_rubin_dtype_names_are_accepted_by_compiler_surfaces():
    for dtype in ("nvfp4", "fp4_e2m1", "fp6_e2m3", "fp6_e3m2", "fp8_e4m3", "fp8_e5m2"):
        assert "tensor<*" in str(_dtype_to_ir_type(dtype))
        GEMMWorkload(M=128, N=128, K=128, dtype=dtype)

    profile = GPUTargetProfile(isa=ISA.SM_120)
    assert profile.supports_mbarrier
    assert profile.supports_tensor_core_dtype("nvfp4")
