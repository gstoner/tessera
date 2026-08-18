"""W1.1b semantic ``kind`` selectors are declaration-owned and fail closed."""

from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def test_graph_kind_sites_use_operation_specific_constraints() -> None:
    text = (ROOT / "src/compiler/ir/TesseraOps.td").read_text()
    assert "StrAttr:$kind" not in text
    for constraint in (
        "Tessera_DistributionBackwardKindAttr",
        "Tessera_RegressionBackwardKindAttr",
        "Tessera_ReductionKindAttr",
        "Tessera_SpectralCompoundKindAttr",
        "Tessera_TrainingLossKindAttr",
    ):
        assert constraint in text


def test_target_kind_sites_use_closed_constraints() -> None:
    apple = (ROOT / "src/compiler/codegen/Tessera_Apple_Backend/include/"
             "Tessera/Target/Apple/TesseraAppleOps.td").read_text()
    nvidia = (ROOT / "src/compiler/codegen/tessera_gpu_backend_NVIDIA/include/"
              "tessera/gpu/IR/TesseraNVIDIADialect.td").read_text()
    rocm = (ROOT / "src/compiler/codegen/Tessera_ROCM_Backend/include/"
            "TesseraROCM/IR/TesseraROCMOps.td").read_text()
    assert "Apple_KVCacheKindAttr:$kind" in apple
    assert "NVIDIA_CudaMathKindAttr:$kind" in nvidia
    assert "StrAttr:$kind" not in rocm
    assert "DefaultValuedAttr<StrAttr" not in "\n".join(
        line for line in rocm.splitlines() if ":$kind" in line
    )


def test_neighbors_topology_kind_is_exact_not_substring_classified() -> None:
    core = (ROOT / "src/compiler/ir/TesseraOps.td").read_text()
    ods = (ROOT / "src/compiler/tessera_neighbors/include/tessera/Dialect/"
           "Neighbors/IR/tessera_neighbors.td").read_text()
    dynamic = (ROOT / "src/compiler/tessera_neighbors/lib/Dialect/Neighbors/"
               "Transforms/DynamicTopologyPass.cpp").read_text()
    assert "Tessera_NeighborsTopologyKindAttr:$kind" in core
    assert "TopologyKindAttr:$kind" in ods
    assert 'kind.contains("dynamic")' not in dynamic
    assert 'kind == "dynamic"' in dynamic
    assert 'kind == "custom_graph"' in dynamic
