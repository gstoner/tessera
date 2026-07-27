import json
from pathlib import Path

from benchmarks.rocm.benchmark_rocm_canonical_gemm_kloop import (
    CASES,
    KERNEL_WALL_BASELINES_MS,
)

ROOT = Path(__file__).resolve().parents[2]
PACKET = ROOT / "benchmarks/baselines/rocm_gfx1151_canonical_gemm_kloop.json"


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
    assert set(KERNEL_WALL_BASELINES_MS) == identities


def test_canonical_gemm_exact_device_packet_keeps_register_incumbent():
    packet = json.loads(PACKET.read_text())
    assert packet["semantic_route"]["ownership_proof"] == [
        "!tile.buffer",
        "!tile.async_token",
        "!tile.pipeline_state",
    ]
    assert len(packet["rows"]) == 6
    assert packet["decision"]["selected"] == "register"
    assert packet["decision"]["all_correct"]
    assert packet["decision"]["all_register_ratchets_pass"]
    for row in packet["rows"]:
        assert row["lds_median_ms"] > row["register_median_ms"]
        assert row["register_resources"]["spill_count"] == 0
        assert row["lds_resources"]["spill_count"] == 0
        assert row["lds_resources"]["lds_bytes"] > 0
