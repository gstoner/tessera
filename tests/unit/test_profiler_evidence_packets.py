from __future__ import annotations

import json
from pathlib import Path

from tessera.compiler.profiler_timing import validate_timing_sample
from tessera.compiler.profiler_rocm_native import validate_rocm_native_capture
from tessera.compiler.profiler_x86_evidence import validate_x86_profiler_packet
from tessera.compiler.profiler_x86_event_map import validate_x86_event_map


ROOT = Path(__file__).resolve().parents[2]


def test_committed_gfx1151_multiclock_packet_is_truthful_wsl_evidence() -> None:
    packet = json.loads(
        (ROOT / "benchmarks/baselines/rocm_gfx1151_multiclock_probe_2026_08_06.json").read_text(encoding="utf-8")
    )
    validate_timing_sample(packet)
    assert packet["target"] == "rocm_gfx1151"
    assert packet["execution_environment"] == "wsl2"
    assert packet["clocks"]["host_wall_ns"]["valid"] is True
    assert packet["clocks"]["device_wall_clock_ns"]["valid"] is True
    assert packet["clocks"]["hip_event_ns"]["valid"] is False
    assert packet["clocks"]["device_wall_clock_ns"]["eligible_for_promotion"] is False


def test_committed_zen5_profiler_packet_is_retain_only() -> None:
    packet = json.loads(
        (ROOT / "benchmarks/baselines/x86_zen5_profiler_packet_2026_08_06.json").read_text(encoding="utf-8")
    )
    validate_x86_profiler_packet(packet)
    assert packet["verdict"] == "retain"
    assert packet["eligible_for_promotion"] is False
    assert packet["environment"]["worktree_dirty"] is True
    assert "SYMBOL_SAMPLING_INVALID" in packet["ineligibility_reasons"]
    rows = packet["benchmark"]["rows"]
    assert {row["shape_class"] for row in rows} == {"aligned", "ragged"}
    assert all(row["correctness"]["route_parity_max_abs"] == 0.0 for row in rows)


def test_committed_gfx1151_native_capture_records_wsl_driver_blocker() -> None:
    capture = json.loads(
        (ROOT / "benchmarks/baselines/rocm_gfx1151_native_capture_2026_08_07.json").read_text(
            encoding="utf-8"
        )
    )
    validate_rocm_native_capture(capture)
    assert capture["status"] == "blocked"
    assert capture["reason"] == "ROCPROFILER_DEVICE_INTERFACE_UNAVAILABLE"
    assert capture["capabilities"]["dev_dxg"] is True
    assert capture["capabilities"]["dev_kfd"] is False
    assert capture["proof"]["dispatch_activity_seen"] is False
    assert capture["proof"]["counter_records_seen"] is False
    assert capture["proof"]["pc_samples_seen"] is False
    assert capture["eligible_for_promotion"] is False


def test_committed_zen5_event_map_records_wsl_pmu_blocker() -> None:
    event_map = json.loads(
        (ROOT / "benchmarks/baselines/x86_zen5_event_map_2026_08_07.json").read_text(encoding="utf-8")
    )
    validate_x86_event_map(event_map)
    assert event_map["exact_zen5_family"] is True
    assert event_map["environment"]["wsl"] is True
    assert event_map["perf"]["available"] is False
    assert event_map["eligible_for_collection"] is False
    assert event_map["eligible_for_promotion"] is False
