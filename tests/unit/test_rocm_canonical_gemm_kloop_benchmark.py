import json
from pathlib import Path

import pytest

from benchmarks.rocm.benchmark_rocm_canonical_gemm_kloop import CASES, ROUTE, run

ROOT = Path(__file__).resolve().parents[2]
#: Recorded on the retired Graph->Tile shortcut (Lane B). Kept as that route's
#: record; it is not a baseline for the scheduled route.
LANE_B_PACKET = ROOT / "benchmarks/baselines/rocm_gfx1151_canonical_gemm_kloop.json"


def test_canonical_gemm_packet_covers_dtype_accumulator_and_boundary_matrix():
    identities = {(case.family, case.dtype) for case in CASES}
    assert identities == {
        ("canonical_aligned", "f16"),
        ("canonical_ragged", "f16"),
        ("canonical_aligned", "bf16"),
        ("canonical_ragged", "bf16"),
        ("canonical_aligned", "int8"),
        ("canonical_ragged", "int8"),
    }
    assert ROUTE == "graph_schedule_tile_target"


@pytest.mark.parametrize("chip", [None, "gfx1200", "rocm"])
def test_benchmark_refuses_an_unnamed_or_unproved_chip(monkeypatch, chip):
    """A default chip would label one part's result as another's."""
    if chip is None:
        monkeypatch.delenv("TESSERA_ROCM_CHIP", raising=False)
    else:
        monkeypatch.setenv("TESSERA_ROCM_CHIP", chip)
    with pytest.raises(RuntimeError, match="exact device"):
        run(warmup=0, rounds=1, iterations=1)


def test_canonical_builder_goes_through_schedule_ir(monkeypatch):
    """The only Graph entry lowers Graph -> Schedule -> Tile, then packages."""
    from tessera import runtime as rt
    from tessera.compiler import rocm_native, scheduled_matmul

    calls = []

    def lower(module, *, target):
        calls.append(("lower_scheduled_matmul", target, module.functions[0].body[0].op_name))
        return "artifact"

    def package(artifact, *, pipeline_name, staging):
        calls.append(("package_scheduled_matmul", artifact, pipeline_name, staging))
        return "package"

    monkeypatch.setattr(scheduled_matmul, "lower_scheduled_matmul", lower)
    monkeypatch.setattr(rocm_native, "package_scheduled_matmul", package)
    monkeypatch.setattr(rt, "_rocm_scheduled_gemm_packages", {})
    assert rt.build_canonical_gemm_hsaco(16, 32, 64, "f16", chip="gfx1201") == "package"
    assert calls == [
        ("lower_scheduled_matmul", "rocm_gfx1201", "tessera.matmul"),
        ("package_scheduled_matmul", "artifact", "tessera-lower-to-rocm", "register"),
    ]
    assert not hasattr(rt, "_build_canonical_gemm_hsaco")


def test_graph_level_matmul_pipeline_is_refused():
    from tessera.compiler.rocm_pipeline import ROCMExecutablePipeline, ROCMInputLevel

    with pytest.raises(ValueError, match="no Graph-level pipeline entry"):
        ROCMExecutablePipeline(family="matmul", input_level=ROCMInputLevel.GRAPH)
    # Attention keeps its Graph entry; only matmul's shortcut was retired.
    ROCMExecutablePipeline(family="attention", input_level=ROCMInputLevel.GRAPH)


def test_retired_lane_b_packet_stays_as_its_record():
    packet = json.loads(LANE_B_PACKET.read_text())
    assert packet["semantic_route"]["ownership_proof"] == [
        "!tile.buffer",
        "!tile.async_token",
        "!tile.pipeline_state",
    ]
    assert len(packet["rows"]) == 6
    assert packet["decision"]["selected"] == "register"
    assert packet["decision"]["all_correct"]
