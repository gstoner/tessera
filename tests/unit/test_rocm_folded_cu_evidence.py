"""Source and protocol checks for the frozen gfx1201 CU calibration packet."""
from pathlib import Path
import gzip
import hashlib
import json

import pytest

ROOT = Path(__file__).resolve().parents[2]
PACKET = ROOT / "benchmarks/baselines/gfx1201_folded_cu_policy_20261016"
REVISION = "68fb444434bbb20f396399e8a1df186c67e00f94"
FULL_UNIT_REVISION = "8e53d33556d04066c4a6cd0335b7be1b164b1f80"


def load(name):
    raw = (PACKET / name).read_bytes()
    manifest = json.loads((PACKET / "compression-manifest.json").read_text())
    entry = manifest[name]
    assert hashlib.sha256(raw).hexdigest() == entry["compressed_sha256"]
    expanded = gzip.decompress(raw)
    assert hashlib.sha256(expanded).hexdigest() == entry["raw_sha256"]
    return json.loads(expanded)


@pytest.mark.parametrize("order", ["forward", "reverse"])
def test_cu_grid_binds_frozen_lowering_and_benchmark_sources(order):
    packet = load(f"cu-grid-{order}.json.gz")
    assert packet["source_commit"] == REVISION
    assert packet["dirty_worktree"] is False
    assert packet["architecture"] == "gfx1201"
    assert "9070" in packet["device"]
    assert packet["profiler_counters"] is False
    assert packet["native_reference_schedule_match"] == "all fields except explicit workgroup mode"
    assert packet["shared_native_resident_frames"] is True
    assert packet["native_reference_fragment_read_policy_comparison"] is False
    # This packet is a historical measurement, not a prohibition on future
    # compiler development. Retain its actual Git-revision source bytes and
    # verify every original fingerprint; current-image parity is recorded in
    # the owning follow-on packet.
    snapshot = load("source-snapshot-68fb44443.json.gz")
    assert snapshot["revision"] == REVISION
    assert set(snapshot["files"]) == set(packet["source_sha256"])
    for relative, digest in packet["source_sha256"].items():
        source = snapshot["files"][relative]
        assert source["sha256"] == digest
        assert hashlib.sha256(source["text"].encode()).hexdigest() == digest, relative


@pytest.mark.parametrize("order", ["forward", "reverse"])
def test_cu_grid_retains_oracles_leases_samples_and_one_image_per_mode(order):
    packet = load(f"cu-grid-{order}.json.gz")
    shapes = {(256, n, k) for k in (1472, 1536, 1600)
              for n in range(8192, 32769, 4096)} | {(256, 8192, 2048)}
    assert {tuple(r["shape_mnk"]) for r in packet["rows"]} == shapes
    images = {"native_mlir": set(), "native_reference_control": set()}
    for row in packet["rows"]:
        assert row["correctness"] == "bitwise_matched_and_sampled_independent_oracle"
        frame = row["shared_native_resident_frames"]
        assert frame["closed"] and frame["leases"] == 0 and frame["copies"] == 3
        assert row["frontend_receipt"]["selected_schedule"]["workgroup_mode"] == (
            "cu" if order == "forward" else "wgp")
        for arm in row["rows"]:
            if arm["engine"] not in images:
                continue
            metadata = arm["image_evidence"]
            assert metadata["shared_resident_frames"]
            assert metadata["resident_frame_identity"] == frame["pointer_identity_sha256"]
            assert not metadata["resources"]["spills"]
            images[arm["engine"]].add(metadata["image_sha256"])
        assert len({arm["output_sha256"] for arm in row["rows"]}) == 1
        for arm in row["graph_rows"]:
            assert len(arm["device_window_per_launch_samples_ms"]) == 7
            assert arm["correctness"] == (
                "bracketed_and_plain_graphs_all_three_poisoned_outputs_bitwise_matched_before_timing")
    assert all(len(values) == 1 for values in images.values())
    assert images["native_mlir"] != images["native_reference_control"]


def test_cu_grid_reverses_compilers_and_preserves_both_source_witnesses():
    forward = load("cu-grid-forward.json.gz")
    reverse = load("cu-grid-reverse.json.gz")
    assert forward["compiler_sha256"] == reverse["native_reference_compiler_sha256"]
    assert reverse["compiler_sha256"] == forward["native_reference_compiler_sha256"]
    before = json.loads((PACKET / "source-before.json").read_text())
    after = json.loads((PACKET / "source-after.json").read_text())
    assert before == after and before["head"] == REVISION and not before["status"]
    assert before["compiler_sha256"] == forward["compiler_sha256"]
    assert [r["shape_mnk"] for r in forward["rows"]] == [
        r["shape_mnk"] for r in reversed(reverse["rows"])]


def test_cu_exclusions_and_fp8_controls_preserve_original_images():
    excluded = json.loads((PACKET / "excluded-image-equality.json").read_text())
    assert excluded["source_before"] == excluded["source_after"]
    assert excluded["source_before"]["head"] == FULL_UNIT_REVISION
    assert len(excluded["rows"]) == 9
    for row in excluded["rows"]:
        assert row["identical"] and row["arms"]["current"] == row["arms"]["control"]
    formats = json.loads((PACKET / "format-image-equality.json").read_text())
    assert formats["source_commit"] == FULL_UNIT_REVISION and formats["source_clean"]
    assert len(formats["rows"]) == 12
    for row in formats["rows"]:
        assert row["identical_native_package"]
        assert row["arms"]["baseline"] == row["arms"]["candidate"]


def test_frozen_full_unit_receipt_binds_the_same_compiler_and_raw_log():
    receipt = json.loads((PACKET / "post-grid-validation.json").read_text())
    assert receipt["revision"] == FULL_UNIT_REVISION and receipt["state"] == "passed"
    assert receipt["source_before"] == receipt["source_after"]
    assert not receipt["source_before"]["status"]
    assert receipt["source_before"]["compiler_sha256"] == load(
        "cu-grid-forward.json.gz")["compiler_sha256"]
    phase = next(p for p in receipt["phases"] if p["name"] == "full-unit")
    assert phase["state"] == "passed" and phase["returncode"] == 0
    assert phase["command"][-3:] == ["tests/unit", "-m", "not slow"]
    raw = (PACKET / "full-unit.log.gz").read_bytes()
    manifest = json.loads((PACKET / "compression-manifest.json").read_text())
    assert hashlib.sha256(raw).hexdigest() == manifest["full-unit.log.gz"]["compressed_sha256"]
    expanded = gzip.decompress(raw)
    assert hashlib.sha256(expanded).hexdigest() == manifest["full-unit.log.gz"]["raw_sha256"]
    assert phase["tail"] in expanded.decode()


@pytest.mark.parametrize("order", ["forward", "reverse"])
def test_radiance_packets_bind_independent_binary_layout_and_timing_scope(order):
    packet = load(f"radiance-{order}.json.gz")
    assert packet["source_commit"] == REVISION and packet["dirty_worktree"] is False
    assert packet["architecture"] == "gfx1201" and packet["radiance_comparison"]
    assert packet["native_reference_fragment_read_policy_comparison"] is False
    reference = packet["radiance"]
    assert reference["binary_sha256"] == (
        "c9f91bc882ad39f331769043b4d51f9bd3813af48a37b4f3ee5f640598eac3a6")
    assert reference["declared_revision"] == "dfdfa3832922c9a4253133f09c1f5c0d39748fc7"
    assert reference["revision_binding"] == "caller_declared; binary hash is authoritative"
    assert reference["tuning_environment"]["RADIANCE_MXFP4_WPERM"] == "1"
    assert reference["tuning_environment"]["RADIANCE_MXFP4_EPIFAST"] == "1"
    assert {tuple(r["shape_mnk"]) for r in packet["rows"]} == {
        (256, n, k) for n in (8192, 16384) for k in (1536, 2048, 5120)}
    for row in packet["rows"]:
        assert row["correctness"] == "bitwise_matched_and_sampled_independent_oracle"
        frame = row["shared_native_resident_frames"]
        assert frame["closed"] and frame["leases"] == 0
        arm = next(r for r in row["rows"] if r["engine"] == "radiance")
        assert arm["public_launch_wall_samples_ms"] == []
        assert arm["public_launch_wall_median_ms"] is None
        graph = next(r for r in row["graph_rows"] if r["engine"] == "radiance")
        assert len(graph["device_window_per_launch_samples_ms"]) == 7
