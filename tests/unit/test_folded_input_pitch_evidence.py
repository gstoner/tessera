"""Frozen same-image row-pitch evidence and legacy-image conservation."""
from pathlib import Path
import gzip
import hashlib
import json

import pytest

ROOT = Path(__file__).resolve().parents[2]
PACKET = ROOT / "benchmarks/baselines/gfx1201_folded_input_pitches_20261016"
SOURCE = "920bdadc4212fb89a22bb65dc05cffd169af6bf1"
IMAGE = "f80b91072c88878e2be6193b83e5ca72a6d59b033da0859a3b05ad31aba1b1fc"


def load(name):
    path = name + ".json.gz"
    manifest = json.loads((PACKET / "compression-manifest.json").read_text())[path]
    data = (PACKET / path).read_bytes()
    assert hashlib.sha256(data).hexdigest() == manifest["compressed_sha256"]
    raw = gzip.decompress(data)
    assert hashlib.sha256(raw).hexdigest() == manifest["raw_sha256"]
    return json.loads(raw)


@pytest.mark.parametrize("name", ["forward", "reverse", "long-forward", "long-reverse"])
def test_same_image_packet_retains_source_device_oracles_and_raw_windows(name):
    packet = load(name)
    assert packet["source_before"] == packet["source_after"]
    assert packet["source_before"]["commit"] == SOURCE
    assert packet["architecture"] == "gfx1201" and "9070" in packet["device"]
    assert packet["trials"] == 7
    ks = (5120, 8192) if name.startswith("long") else (4032, 4096, 4160)
    assert {tuple(c["shape_mnk"]) for c in packet["cases"]} == {
        (256, n, k) for n in (8192, 16384) for k in ks}
    for case in packet["cases"]:
        assert case["same_pitched_image_sha256"] == IMAGE
        assert case["frozen_contiguous_image_identical"]
        assert len({row["output_sha256"] for row in case["rows"]}) == 1
        assert "independent_oracle" in case["correctness"]
        rows = {row["name"]: row for row in case["rows"]}
        assert set(rows) == {"contiguous_control", "dense", "a_only", "b_only", "both"}
        m, n, k = case["shape_mnk"]
        for name, row in rows.items():
            assert len(row["windows"]) == 7 and len(row["public_wall_samples_ms"]) == 3
            assert not row["image_evidence"]["resources"]["spills"]
            if name != "contiguous_control":
                lda = k + (16 if name in {"a_only", "both"} else 0)
                ldb = k + (16 if name in {"b_only", "both"} else 0)
                assert row["physical_shapes"][:2] == [[m, lda], [n, ldb]]
                assert row["receipt"]["input_row_strides"] == [lda, ldb]
                assert row["receipt"]["hsaco_sha256"] == IMAGE
                assert row["image_evidence"]["selected_isa"]["payload_sha256"] == IMAGE
            for window in row["windows"]:
                clock, plain = window["clock"], window["plain"]
                assert clock["launches"] == plain["launches"] == 192
                assert clock["bracketed"] and not plain["bracketed"]
                assert clock["device_window_ms"] >= 5
                assert clock["device_event_disagreement"] <= 0.05


@pytest.mark.parametrize("prefix", ["", "long-"])
def test_independent_orders_use_the_same_compiler_and_reverse_cases(prefix):
    forward, reverse = load(prefix + "forward"), load(prefix + "reverse")
    assert forward["source_before"] == reverse["source_before"]
    assert [c["shape_mnk"] for c in forward["cases"]] == [
        c["shape_mnk"] for c in reversed(reverse["cases"])]


def test_legacy_cu_and_fp8_mxfp8_controls_retain_all_native_artifacts():
    packet = json.loads((PACKET / "legacy-image-parity.json").read_text())
    assert packet["architecture"] == "gfx1201"
    assert packet["source_before"] == packet["source_after"]
    assert packet["source_before"]["commit"] == SOURCE
    assert len(packet["cu_rows"]) == 22 and len(packet["format_rows"]) == 12
    for row in packet["cu_rows"] + packet["format_rows"]:
        assert row["identical"] and row["current"] == row["frozen"]
