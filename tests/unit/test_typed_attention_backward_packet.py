"""Frozen public typed reverse evidence stays valid as the compiler evolves."""
import gzip
import hashlib
import json
import math
from pathlib import Path
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[2]
PACKET = ROOT / "benchmarks/baselines/sm120_saved_lse_half_backward_20261016/public-stage"


def load(name):
    path = PACKET / name
    data = path.read_bytes()
    return json.loads(gzip.decompress(data) if path.suffix == ".gz" else data)


def test_public_typed_packet_retains_exact_declared_bytes():
    manifest = load("compression-manifest.json")
    assert manifest
    for name, record in manifest.items():
        path = PACKET / name
        assert path.resolve().is_relative_to(PACKET.resolve())
        raw = path.read_bytes()
        assert len(raw) == record["bytes"]
        assert hashlib.sha256(raw).hexdigest() == record["sha256"]
        if name.endswith(".gz"):
            decoded = gzip.decompress(raw)
            assert len(decoded) == record["decoded_bytes"]
            assert hashlib.sha256(decoded).hexdigest() == record["decoded_sha256"]
    archive = load("source-archive.json")
    assert len(archive["files"]) >= 20
    for name, record in archive["files"].items():
        assert name.startswith(("src/", "python/", "tests/", "benchmarks/"))
        raw = gzip.decompress((PACKET / record["archive"]).read_bytes())
        assert len(raw) == record["bytes"]
        assert hashlib.sha256(raw).hexdigest() == record["sha256"]


def test_public_typed_packet_has_clean_owning_execution_and_route_proof():
    witness = load("public-stage-witness.json.gz")
    assert witness["unchanged"] and witness["source_before"] == witness["source_after"]
    assert not witness["source_before"]["status"]
    assert "RTX 5070" in witness["device"] and ", 12.0," in witness["device"]
    assert load("source-archive.json")["commit"] == witness["source_before"]["head"]
    phase = load("public-stage-phase1-witness.json.gz")
    assert phase["unchanged"] and phase["source_before"] == phase["source_after"] == witness["source_before"]
    assert len(phase["lanes"]) == 3 and all(row["returncode"] == 0 for row in phase["lanes"])
    continuation = witness["continuations"][0]
    assert continuation["source_before"] == continuation["source_after"] == witness["source_before"]
    assert continuation["prior_witness_sha256"] == hashlib.sha256(gzip.decompress((PACKET / "public-stage-phase1-witness.json.gz").read_bytes())).hexdigest()
    assert continuation["helper_sha256"] == hashlib.sha256(gzip.decompress((PACKET / "resume_public_stage.py.gz").read_bytes())).hexdigest()
    rows = {r["lane"]: r for r in witness["lanes"]}
    assert len(rows) == 6 and all(r["returncode"] == 0 for r in rows.values())
    assert rows["public-final-device"]["cases"] == 339
    for name in ("public-final-device", "public-final-compiler-route"):
        assert rows[name]["skips"] == rows[name]["failures"] == 0
    xml_name = "public-final-compiler-route.xml.gz"
    frozen = (PACKET / xml_name).read_bytes()
    record = load("compression-manifest.json")[xml_name]
    assert hashlib.sha256(frozen).hexdigest() == record["sha256"]
    decoded = gzip.decompress(frozen)
    assert hashlib.sha256(decoded).hexdigest() == record["decoded_sha256"]
    assert b"<!DOCTYPE" not in decoded.upper() and b"<!ENTITY" not in decoded.upper()
    route = ET.fromstring(decoded)  # noqa: S314 - hash-checked repository receipt; DTD/entities forbidden
    for suffix, count in (("test_saved_lse_half_backward", 27), ("test_typed_attention_backward_package", 56)):
        assert sum(c.attrib.get("classname", "").endswith(suffix) for c in route.iter("testcase")) == count
    audit = load("public-audit-witness.json.gz")
    assert audit["head"] == witness["source_before"]["head"]
    assert len(audit["lanes"]) == 12 and all(r["returncode"] == 0 for r in audit["lanes"])


def test_public_typed_timing_keeps_storage_and_timing_domains_explicit():
    packet = load("benchmark/index.json")
    assert not packet["diagnostic"] and packet["rounds"] == 7 and packet["repetitions"] == 3
    assert packet["source_before"] == packet["source_after"]
    assert packet["source_before"]["commit"] == load("source-archive.json")["commit"]
    rows = packet["rows"]
    assert len(rows) == 32 and len({r["input_file"] for r in rows}) == 16
    assert {r["execution_storage"] for r in rows} == {"f16", "bf16", "fp32_matched_quantized_control"}
    assert all(r["host_resident_program_identical"] for r in rows)
    for row in rows:
        assert hashlib.sha256((PACKET / "benchmark" / row["input_file"]).read_bytes()).hexdigest() == row["input_sha256"]
        for lane in row["lanes"].values():
            assert lane["max_abs_error"] and all(math.isfinite(x) for x in lane["max_abs_error"].values())
            for kind in ("public_completed_call", "native_forward_event", "native_backward_event"):
                samples = lane[kind+"_samples_ms"]
                assert len(samples) == 7 and all(math.isfinite(x) and x > 0 for x in samples)
    replay = load("benchmark/fresh-replay.json")
    assert replay["compiler_subprocesses_forbidden"] and replay["first_package_preparation_inside_subprocess_guard"]
    assert {r["case"] for r in replay["rows"]} == {r["case"] for r in rows}
    assert all(r["execution_kind"] == "native_gpu" for r in replay["rows"])
    assert {r["artifact_hash"] for r in replay["rows"]} == {r["lanes"]["host"]["artifact_hash"] for r in rows}


def test_recorded_runtime_and_program_restore_without_subprocesses(monkeypatch):
    from tessera.runtime import RuntimeArtifact
    from tessera.compiler.native_attention_program import NativeAttentionVJPProgram
    def forbidden(*args, **kwargs):
        raise AssertionError("offline artifact restoration invoked a subprocess")
    for name in ("run", "Popen", "check_output"):
        monkeypatch.setattr("subprocess."+name, forbidden)
    for row in load("benchmark/index.json")["rows"]:
        for lane in row["lanes"].values():
            raw = gzip.decompress((PACKET / "benchmark" / lane["artifact"]).read_bytes()).decode()
            artifact = RuntimeArtifact.from_json(raw)
            assert artifact.artifact_hash == lane["artifact_hash"]
            program = NativeAttentionVJPProgram.from_json(
                artifact.metadata["program_json"], expected_digest=lane["program_digest"])
            assert program.program_digest == lane["program_digest"]
            schema = json.loads(program.to_json())["program"]["schema"]
            assert schema == ("tessera.native_attention_vjp_program.v1"
                if row["execution_storage"] == "fp32_matched_quantized_control"
                else "tessera.native_attention_vjp_program.v2")
