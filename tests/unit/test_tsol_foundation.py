"""
Phase 6 - Tessera Standard Operator Library documentation and stub contract.
"""

from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
TSOL = ROOT / "docs" / "operations" / "Tessera_Standard_Operations.md"
OPS_STUB = ROOT / "python" / "tessera" / "ops.pyi"


def test_tsol_document_exists_and_covers_standard_library_axes():
    text = TSOL.read_text(encoding="utf-8")
    required = [
        "Tessera Standard Operator Library",
        "Numerics-first",
        "Determinism Contract",
        "Effect Mapping",
        "Linear Algebra",
        "Neural Network Primitives",
        "Spectral Operators",
        "Sparse, Segment, And Graph Operators",
        "RNG And Initialization",
        "Collectives",
        "Layout And Packing",
        "Schedule Artifact Contract",
        "TS_ERR_NONDETERMINISM",
    ]
    for term in required:
        assert term in text


def test_tsol_is_linked_from_docs_map_and_python_spec():
    guide_path = "docs/operations/Tessera_Standard_Operations.md"
    assert guide_path in (ROOT / "docs" / "README.md").read_text(encoding="utf-8")
    assert guide_path in (ROOT / "docs" / "spec" / "PYTHON_API_SPEC.md").read_text(
        encoding="utf-8"
    )


def test_ops_stub_defines_numeric_policy_and_core_operator_groups():
    text = OPS_STUB.read_text(encoding="utf-8")
    required = [
        "class NumericPolicy",
        "class Epilogue",
        "class Determinism",
        "class Transport",
        "class ScheduleArtifact",
        "def matmul",
        "def flash_attn",
        "cache: Optional[Any]",
        "def moe_dispatch",
        "def fft",
        "def bsmm",
        "def segment_reduce",
        "def rng_uniform",
        "def all_reduce",
        "def tile_view",
    ]
    for term in required:
        assert term in text


def test_ops_stub_keeps_current_runtime_names_available():
    text = OPS_STUB.read_text(encoding="utf-8")
    current_runtime_names = [
        "def gemm",
        "def matmul",
        "def layer_norm",
        "def softmax",
        "def gelu",
        "def relu",
        "def transpose",
        "def cast",
        "def dropout",
        "def conv2d",
        "def flash_attn",
        "def all_reduce",
        "def reduce_scatter",
        "def all_gather",
        "def fused_epilogue",
    ]
    for name in current_runtime_names:
        assert name in text
