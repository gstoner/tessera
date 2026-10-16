"""Frozen same-image row-pitch evidence and legacy-image conservation."""
from pathlib import Path
import gzip
import hashlib
import json
import subprocess
import sys

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


def test_canonicalization_hardening_preserves_all_measured_image_variants():
    packet = json.loads((PACKET / "post-canonicalization-image-parity.json").read_text())
    assert packet["measured_source_revision"] == SOURCE
    assert packet["source_before"] == packet["source_after"]
    assert packet["source_before"]["commit"] == "b50c2cc1349bc7406b7d7ef0e6c1fa40ec2183fa"
    assert len(packet["cu_rows"]) == 22 and len(packet["format_rows"]) == 12
    for row in packet["cu_rows"] + packet["format_rows"]:
        assert row["identical"] and row["current"] == row["frozen"]
    assert len(packet["pitched_rows"]) == 80
    actual = {(tuple(row["shape_mnk"]), tuple(row["pitches"]), row["canonicalized"])
              for row in packet["pitched_rows"]}
    expected = {((256, n, k), pitches, canonical)
                for n in (8192, 16384) for k in (4032, 4096, 4160, 5120, 8192)
                for pitches in ((k, k), (k + 16, k), (k, k + 16), (k + 16, k + 16))
                for canonical in (False, True)}
    assert actual == expected
    assert {row["payload_sha256"] for row in packet["pitched_rows"]} == {IMAGE}


def test_encoding_guard_retains_the_original_image_rows_with_final_source():
    observed = load("encoding-guard-image-parity")
    earlier = json.loads((PACKET / "post-canonicalization-image-parity.json").read_text())
    certificate = json.loads((PACKET / "encoding-guard-equivalence.json").read_text())
    assert observed["source_before"] == observed["source_after"]
    assert observed["source_before"]["commit"] == "f0d6fb7e5f1d3547575d5219650415e4d1c45fbd"
    keys = ("cu_rows", "format_rows", "pitched_rows")
    assert all(observed[key] == earlier[key] for key in keys)
    rows = {key: observed[key] for key in keys}
    digest = hashlib.sha256(json.dumps(rows, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    assert certificate["reference_rows_sha256"] == certificate["observed_rows_sha256"] == digest
    assert certificate["counts"] == {"cu_rows": 22, "format_rows": 12, "pitched_rows": 80}
    assert certificate["source_before"] == observed["source_before"]
    assert certificate["source_after"] == observed["source_after"]
    manifest = json.loads((PACKET / "compression-manifest.json").read_text())
    assert certificate["observed_raw_sha256"] == manifest[
        "encoding-guard-image-parity.json.gz"]["raw_sha256"]
    assert certificate["reference_sha256"] == hashlib.sha256(
        (PACKET / certificate["reference"]).read_bytes()).hexdigest()


def test_incomplete_and_red_unit_history_is_retained_without_relabeling():
    root = PACKET / "validation-history"
    for phase, exit_code in (("before-encoding-guard", -15), ("before-doc-date-fix", 1)):
        directory = root / phase
        receipt = json.loads((directory / "full-unit-validation.json").read_text())
        raw = gzip.decompress((directory / "full-unit.log.gz").read_bytes())
        assert receipt["state"] == "failed" and receipt["exit_code"] == exit_code
        assert receipt["source_before"] == receipt["source_after"]
        assert hashlib.sha256(raw).hexdigest() == receipt["raw_log_sha256"]
    reason = json.loads((root / "before-encoding-guard/full-unit-interruption.json").read_text())
    assert "Intentionally interrupted" in reason["reason"]


def test_final_local_gate_is_green_and_bound_to_the_observed_compiler():
    receipt = json.loads((PACKET / "full-unit-validation.json").read_text())
    raw = gzip.decompress((PACKET / "full-unit.log.gz").read_bytes())
    assert receipt["state"] == "passed" and receipt["exit_code"] == 0
    assert receipt["source_before"] == receipt["source_after"]
    assert not receipt["source_before"]["status"]
    assert receipt["revision"] == "6a9610363c2e70632b905deb0d57eb96ff143843"
    assert receipt["summary"]["passed"] == 28603
    assert receipt["summary"]["skipped"] == 5873
    assert hashlib.sha256(raw).hexdigest() == receipt["raw_log_sha256"]
    observed = load("final-admission-image-parity")
    assert observed["source_before"] == observed["source_after"]
    assert observed["source_before"]["commit"] == receipt["revision"]
    assert observed["source_before"]["compiler_sha256"] == receipt["source_before"]["compiler_sha256"]
    previous = load("encoding-guard-image-parity")
    for key in ("cu_rows", "format_rows", "pitched_rows"):
        assert observed[key] == previous[key]


def test_required_compiler_route_and_owning_device_results_are_distinct(tmp_path):
    validation = json.loads((PACKET / "release-validation.json").read_text())
    junit = gzip.decompress((PACKET / validation["compiler_route"]["junit"]).read_bytes())
    junit_path = tmp_path / "retained-compiler-route.xml"
    junit_path.write_bytes(junit)
    checked = subprocess.run(
        [sys.executable, str(ROOT / "scripts/ci_require_executed.py"), str(junit_path)],
        capture_output=True, text=True, check=False,
    )
    assert checked.returncode == 0, checked.stdout + checked.stderr
    route_log = gzip.decompress((PACKET / "compiler-route.log.gz").read_bytes())
    assert b"856 passed" in route_log
    assert validation["compiler_route"]["passed"] == 856
    assert validation["compiler_route"]["skipped"] == 0
    raw = gzip.decompress((PACKET / validation["gfx1201"]["log"]).read_bytes())
    assert b"49 passed" in raw
    assert validation["gfx1201"]["passed"] == 49
    assert validation["generated_documents"] == 33
