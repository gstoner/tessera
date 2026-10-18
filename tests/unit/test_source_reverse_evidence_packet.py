"""Frozen source-VJP timing/ancestry evidence; no current-source hash pin."""
import gzip
import hashlib
import json
from pathlib import Path
import statistics

from tessera import runtime as rt
from tessera.compiler.portable_nvidia_source_reverse import restore

ROOT = Path(__file__).resolve().parents[2] / "benchmarks/baselines/sm120_source_reverse_20261018"


def test_compressed_source_reverse_records_have_complete_integrity():
    manifest = json.loads((ROOT / "compression-manifest.json").read_text())
    assert set(manifest) == {str(p.relative_to(ROOT)) for p in ROOT.rglob("*.gz")}
    for name, record in manifest.items():
        compressed = (ROOT / name).read_bytes()
        raw = gzip.decompress(compressed)
        assert len(compressed) == record["compressed_bytes"] and len(raw) == record["raw_bytes"]
        assert hashlib.sha256(compressed).hexdigest() == record["compressed_sha256"]
        assert hashlib.sha256(raw).hexdigest() == record["raw_sha256"]


def test_standalone_public_windows_and_native_programs_are_bound():
    packet = json.loads((ROOT / "timings.json").read_text())
    assert packet["source_before"] == packet["source_after"]
    assert packet["device"]["name"] == "NVIDIA GeForce RTX 5070"
    assert packet["device"]["compute_capability"] == "12.0"
    assert packet["device"]["uuid"] == "GPU-cba12639-821a-7a10-4cd3-f918f9c0a545"
    expected = {(shape, storage, order) for shape in ((17, 19, 35), (64, 64, 128), (128, 128, 512))
                for storage in ("f16", "bf16") for order in (False, True)}
    assert {(tuple(row["shape_mnk"]), row["storage"], row["permuted"]) for row in packet["rows"]} == expected
    snapshot = json.loads(gzip.decompress((ROOT / "source-snapshot.json.gz").read_bytes()))
    for name, digest in packet["source_before"]["modified_source_sha256"].items():
        assert hashlib.sha256(snapshot[name].encode()).hexdigest() == digest
    for row in packet["rows"]:
        assert row["correctness"] == "passed_before_and_after_timing"
        for mode, values in row["completed_host_samples_ms"].items():
            assert len(values) == 7 and all(value > 0 for value in values)
            assert statistics.median(values) == row["completed_host_median_ms"][mode]
        for domain in ("native_enqueue", "native_captured"):
            assert len(row[domain]) == 7
            assert all(len(t["stage_ms"]) == 2 and min(t["stage_ms"]) > 0 and t["program_ms"] > 0
                       for t in row[domain])
        raw = gzip.decompress((ROOT / "programs" / row["artifact"]["file"]).read_bytes())
        assert hashlib.sha256(raw).hexdigest() == row["artifact"]["raw_sha256"]
        artifact = rt.RuntimeArtifact.from_json(raw.decode())
        native, members = restore(artifact.metadata["native_program"])
        assert tuple(record["image_digest"] for record in members) == tuple(row["image_digests"])
        assert hashlib.sha256(native.plan_json.encode()).hexdigest() == row["plan_digest"]
        assert row["execution_certificate"]["physical_attestation"]["artifact_hash"] == artifact.artifact_hash


def test_metadata_ab_keeps_native_identity_and_raw_paired_windows():
    packet = json.loads(gzip.decompress((ROOT / "metadata-ab.json.gz").read_bytes()))
    assert packet["source_before"] == packet["source_after"]
    assert len(packet["rows"]) == 12
    for row in packet["rows"]:
        assert row["same_native_owner_and_images"]
        assert row["correctness"] == "both_arms_passed_independent_oracle_before_timing"
        assert len(row["image_digests"]) == 2 and len(row["execution_certificate_digest"]) == 64
        for mode in ("control", "candidate"):
            values = row["samples_ms"][mode]
            assert len(values) == 14 and all(value > 0 for value in values)
            assert statistics.median(values) == row["median_ms"][mode]
            assert len(row["native_captured"][mode]) == 14
        assert row["control_over_candidate"] == row["median_ms"]["control"] / row["median_ms"]["candidate"]
