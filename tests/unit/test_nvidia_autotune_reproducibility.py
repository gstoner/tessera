from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def test_committed_autotune_report_rejects_stale_evidence_and_replays_cache():
    data = json.loads((ROOT / "benchmarks/baselines/nvidia_sm120_autotune_reproducibility.json").read_text())
    assert data["schema"] == "tessera.nvidia.autotune-reproducibility.v1"
    assert data["kernel_cache_reproducible"]
    assert data["strict_records_considered"] == data["strict_records_admitted"]
    assert data["strict_records_admitted"] > 0
    assert all(data["stale_rejections"].values())
    assert data["selector_changed"] is False
    assert len({row["kernel_cache_key"] for row in data["builds"]}) == 1
    for row in data["builds"]:
        assert row["cold_compile_ms"] > row["warm_cache_lookup_ms"] > 0
        assert row["cache_hits"] == row["cache_misses"] == 1
        assert row["artifact_present"]
