from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
BASELINE = ROOT / "benchmarks/baselines/nvidia_sm120_e2e_spine_paged_kv.json"


def test_paged_kv_corpus_enforces_wsl_four_percent_policy_and_is_resource_linked() -> None:
    report = json.loads(BASELINE.read_text(encoding="utf-8"))
    assert report["schema"] == "tessera.nvidia.e2e_spine.paged_kv.v1"
    assert report["stability_policy"]["relative_fraction"] == 0.04
    assert report["stability_policy"]["wsl_acceptance_margin_fraction"] == 0.0005
    assert report["stability_policy"]["margin_rows_selector_eligible"] is False
    assert report["selector_decision"] == "unchanged_pending_timing_domain_consensus"
    rows = report["rows"]
    assert len(rows) == 6
    assert {row["candidate"] for row in rows} == {
        "canonical_tile_direct", "legacy_cuda_staged",
    }
    for row in rows:
        assert len(row["runs"]) == 2
        device_stable = row["stability_fraction"]["device_event_ms"] <= 0.0405
        end_to_end_stable = row["stability_fraction"]["end_to_end_ms"] <= 0.0405
        assert row["device_stable"] == device_stable
        assert row["end_to_end_stable"] == end_to_end_stable
        assert row["stable"] == (device_stable and end_to_end_stable)
        assert row["resource_evidence_complete"]
        resource = row["resources"][0]
        assert resource["registers_per_thread"] > 0
        assert resource["static_shared_memory_bytes"] >= 0
        assert resource["theoretical_occupancy_pct"] > 0
        assert resource["spill_evidence_complete"]
        assert resource["resource_fingerprint"].startswith("sha256:")
        assert row["selected_route"] == "existing_serving_policy_unchanged"
    canonical = [row for row in rows if row["candidate"] == "canonical_tile_direct"]
    assert all(row["compile_state"]["cold"] == "cold" for row in canonical)
    assert all(row["compile_state"]["warm"] == "warm_cache" for row in canonical)
    assert all(row["compile_state"]["image_digest_reproducible"] for row in canonical)
    assert sum(row["stable"] for row in rows) == 6
    waived = [row for row in rows if row.get("wsl_margin_accepted_domains")]
    assert len(waived) == 1
    assert waived[0]["candidate"] == "legacy_cuda_staged"
    assert waived[0]["case"] == "tokens_2048_ragged"
    assert waived[0]["wsl_margin_accepted_domains"] == ["device_event_ms"]


def test_paged_kv_selector_is_not_promoted_without_timing_domain_consensus() -> None:
    rows = json.loads(BASELINE.read_text(encoding="utf-8"))["rows"]
    grouped: dict[str, list[dict]] = {}
    for row in rows:
        grouped.setdefault(row["case"], []).append(row)
    assert all(len(case_rows) == 2 for case_rows in grouped.values())
    assert any(
        any(row["device_winner_consensus"] for row in case_rows)
        != any(row["end_to_end_winner_consensus"] for row in case_rows)
        or next(row["candidate"] for row in case_rows if row["device_winner_consensus"])
        != next(row["candidate"] for row in case_rows if row["end_to_end_winner_consensus"])
        for case_rows in grouped.values()
    )
