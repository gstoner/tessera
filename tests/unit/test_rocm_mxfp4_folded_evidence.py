"""Drift gate for the exact-device folded MXFP4 prefill packet."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
PACKET = (
    ROOT / "benchmarks/baselines/gfx1201_mxfp4_folded_prefill_20260922/evidence.json"
)


def test_folded_prefill_packet_binds_source_and_compares_matched_outputs() -> None:
    packet = json.loads(PACKET.read_text())
    assert packet["schema"] == "tessera.rocm.gfx1201_mxfp4_folded_benchmark.v1"
    assert packet["architecture"] == "gfx1201"
    assert packet["device"] == "AMD Radeon RX 9070 XT"
    assert packet["source_revision"] == "817cde29fda75c0323dba7bdbff3f673da32c2f4"
    source = ROOT / "python/tessera/compiler/rocm_mxfp4_folded.py"
    assert packet["folded_generator_sha256"] == hashlib.sha256(
        source.read_bytes()
    ).hexdigest()
    assert packet["benchmark_sha256"] == hashlib.sha256(
        (ROOT / "benchmarks/rocm/benchmark_gfx1201_mxfp4_folded.py").read_bytes()
    ).hexdigest()
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
