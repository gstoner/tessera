"""Drift checks for the exact-device gfx1201 MXFP4 evidence packet."""
from __future__ import annotations

import json
from pathlib import Path
import re


ROOT = Path(__file__).resolve().parents[2]
PACKET = (
    ROOT
    / "benchmarks/baselines/gfx1201_mxfp4_w4a8_20260921/evidence.json"
)


def test_gfx1201_mxfp4_packet_covers_generic_materialization() -> None:
    packet = json.loads(PACKET.read_text())
    assert packet["schema"] == "tessera.rocm.gfx1201_mxfp4_evidence.v1"
    assert packet["live_architecture"] == "gfx1201"
    assert re.fullmatch(r"[0-9a-f]{40}", packet["source_revision"])
    assert packet["proof"]["result"].startswith("5 passed")

    rows = {row["route"]: row for row in packet["rows"]}
    assert set(rows) == {
        "scalar_exact",
        "wmma_exact",
        "generic_materialized_exact",
    }
    generic = rows["generic_materialized_exact"]
    direct = rows["wmma_exact"]
    assert generic["materializer"] == "tessera_rocm.scaled_wmma_gemm"
    assert re.fullmatch(r"[0-9a-f]{64}", generic["schedule_hash"])
    assert generic["image_sha256"] == direct["image_sha256"]
    assert generic["target_ir_sha256"] != direct["target_ir_sha256"]
    assert generic["isa"]["selected_matrix_instructions"] == [
        "v_wmma_f32_16x16x16_fp8_fp8"
    ]
