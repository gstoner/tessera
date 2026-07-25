import json
from pathlib import Path

from benchmarks.nvidia.record_shared_arena_rematerialization import (
    OFFSET_RE,
    ARENA_RE,
    _llvm_contract,
)

ROOT = Path(__file__).parents[2]
BASELINE = ROOT / "benchmarks/baselines" / "nvidia_sm120_shared_arena_rematerialization.json"


def test_sm120_shared_arena_contract_preserves_size_offsets_and_address_space():
    lowered = """
    tile.smem_arena_bytes = 1024 : i64
    arith.constant 0 : index
    arith.constant 512 : index
    arith.constant 0 : index
    """
    assert {int(row.group("bytes")) for row in ARENA_RE.finditer(lowered)} == {1024}
    assert tuple(int(row.group("offset")) for row in OFFSET_RE.finditer(lowered)) == (
        0,
        512,
        0,
    )
    llvm_ir = _llvm_contract(1024)
    assert "addrspace(3) global" in llvm_ir
    assert "[1024 x i8]" in llvm_ir
    assert "i64 512" in llvm_ir


def test_sm120_shared_arena_exact_device_record_is_complete_and_non_promoting():
    payload = json.loads(BASELINE.read_text(encoding="utf-8"))
    assert payload["schema"] == "tessera.nvidia.shared-arena-rematerialization.v1"
    assert payload["work_item"] == "E2E-SPINE-3"
    assert payload["device"].endswith("12.0")
    assert payload["tile_plan"]["arena_bytes"] == 1024
    assert payload["tile_plan"]["bytes_before"] == 1536
    assert payload["tile_plan"]["reuse_offsets"] == [0, 512, 0]
    assert payload["llvm_nvptx"]["declared_shared_memory_bytes"] == 1024
    assert payload["llvm_nvptx"]["resources"]["spills_detected"] is False

    rows = {row["route"]: row for row in payload["rows"]}
    assert set(rows) == {
        "retained_shared_arena",
        "rematerialized_register_expression",
    }
    assert rows["retained_shared_arena"]["resources"]["static_shared_memory_bytes"] >= 1024
    assert rows["rematerialized_register_expression"]["resources"]["static_shared_memory_bytes"] == 0
    for row in rows.values():
        assert row["numerical_output"] == 42.0
        assert row["selector_eligible"] is False
        assert row["selector_changed"] is False
        assert row["resources"]["spill_evidence_complete"] is True
        assert row["resources"]["spills_detected"] is False
        assert set(row["timings"]) == {"device_event", "end_to_end"}
        assert all(timing["stable"] for timing in row["timings"].values())
