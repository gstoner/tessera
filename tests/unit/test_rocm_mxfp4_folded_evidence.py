"""Drift gate for the exact-device folded MXFP4 prefill packet."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
PACKET = (
    ROOT / "benchmarks/baselines/gfx1201_mxfp4_folded_prefill_20260922/evidence.json"
)


def test_historical_folded_packet_refuses_unbound_radiance_layout() -> None:
    packet = json.loads(PACKET.read_text())
    assert packet["schema"] == "tessera.rocm.gfx1201_mxfp4_folded_benchmark.v1"
    assert packet["architecture"] == "gfx1201"
    assert packet["device"] == "AMD Radeon RX 9070 XT"
    assert packet["source_revision"] == "4368c825b866c3eb7b40761c5c17bbcb817e2665"
    source = ROOT / "python/tessera/compiler/rocm_mxfp4_folded.py"
    assert packet["folded_generator_sha256"] == hashlib.sha256(
        source.read_bytes()
    ).hexdigest()
    assert packet["benchmark_sha256"] == (
        "6ea6d21d398e6623af0ca1d638eadaef25a47246e9a2c7a7eddbdd76c8fe8bbd"
    )
    assert packet["radiance_layout_verified"] is False
    assert packet["promotion_eligible"] is False
    assert packet["radiance"]["revision"] == (
        "dfdfa3832922c9a4253133f09c1f5c0d39748fc7"
    )
    assert len(packet["radiance"]["binary_sha256"]) == 64
    for case in {row["case"] for row in packet["rows"]}:
        rows = {row["engine"]: row for row in packet["rows"] if row["case"] == case}
        assert set(rows) == {"tessera", "tessera_folded", "radiance"}
        assert len({row["output_sha256"] for row in rows.values()}) == 1
        folded = rows["tessera_folded"]
        assert folded["metadata"]["route"]["numeric_policy"] == (
            "folded_row_reference_explicit_approximate"
        )
        assert folded["metadata"]["route"]["fold_lossless"] is True
        assert folded["metadata"]["isa"]["wmma_fp8_fp8"] == 32
        assert folded["metadata"]["resources"]["scratch_bytes"] == 0
        assert folded["metadata"]["resources"]["spills"] is False
        assert len(folded["metadata"]["compiler_fingerprint"]) == 64
        assert len(folded["metadata"]["toolchain_fingerprint"]) == 64
        assert folded["median_ms"] < rows["tessera"]["median_ms"]
