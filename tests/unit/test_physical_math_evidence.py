from __future__ import annotations

import json
from pathlib import Path


_ROOT = Path(__file__).resolve().parents[2]
_BASELINES = _ROOT / "benchmarks" / "baselines"


def _packet(name: str) -> dict:
    return json.loads((_BASELINES / name).read_text())


def test_zen5_math_packet_records_retained_scan_selector() -> None:
    packet = _packet("math_physical_zen5_2026_08_06.json")
    assert packet["schema"] == "tessera.physical_math_evidence.v1"
    assert packet["selector_eligible"] is True
    assert packet["storage_dtypes"] == ["f32"]
    assert len(packet["rows"]) == 7

    policies = {
        row["op_name"]: row for row in packet["scan_selector_evidence"]
    }
    assert set(policies) == {"cumsum", "cumprod", "cummax", "cummin"}
    for name in ("cumsum", "cumprod"):
        assert policies[name]["selected_policy"] == "avx512_hillis_steele_16"
        assert policies[name]["speedup"] > 1.05
    for name in ("cummax", "cummin"):
        assert policies[name]["selected_policy"] == "scalar_recurrence_retained"


def test_gfx1151_math_packet_covers_dtypes_and_cache_gain() -> None:
    packet = _packet("math_physical_gfx1151_2026_08_06.json")
    assert packet["schema"] == "tessera.physical_math_evidence.v1"
    assert packet["selector_eligible"] is False
    assert packet["device_event_follow_up"] == "bare_metal_required"
    assert packet["storage_dtypes"] == ["f32", "f16", "bf16"]
    assert len(packet["dtype_rows"]) == 21
    assert {row["dtype"] for row in packet["dtype_rows"]} == {
        "f32", "f16", "bf16"
    }
    assert all(
        row["max_abs_error"] <= row["error_limit"]
        for row in packet["dtype_rows"]
    )
    assert len(packet["f32_cache_comparison"]) == 7
    assert all(row["speedup"] > 1.4 for row in packet["f32_cache_comparison"])
