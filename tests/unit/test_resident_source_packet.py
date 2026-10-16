"""Frozen native source-active resident proof; no current-source hash pin."""
from pathlib import Path, PurePosixPath
import gzip
import hashlib
import json
import math
import statistics
import subprocess
import sys
import xml.etree.ElementTree as ET

import pytest

ROOT = Path(__file__).resolve().parents[2]
PACKET = ROOT / "benchmarks/baselines/sm120_resident_source_jvp_20261016"
SOURCE = "d1ba6e98df65483f921ec75ee34b08801cbfca2a"
COMPILER = "7ce3ed062676d502a267204bc104d3fbcb4d824be0ff32506f026b8188a39614"
PROVIDER = "2800b55abad107a7af51df763d9b7b53767ac5f37a515f9841dfe6e4ae55e2e5"


def raw(name):
    relative = PurePosixPath(name)
    assert not relative.is_absolute() and ".." not in relative.parts
    manifest = json.loads((PACKET / "compression-manifest.json").read_text())[name]
    compressed = (PACKET / name).read_bytes()
    assert hashlib.sha256(compressed).hexdigest() == manifest["compressed_sha256"]
    value = gzip.decompress(compressed)
    assert hashlib.sha256(value).hexdigest() == manifest["raw_sha256"]
    assert len(compressed) == manifest["compressed_bytes"]
    assert len(value) == manifest["raw_bytes"]
    return value


def data(name):
    return json.loads(raw(name))


def positive(value):
    assert type(value) in (int, float) and math.isfinite(value) and value > 0


def test_source_device_and_tool_witnesses_are_frozen_and_consistent():
    packet = data("timings.json.gz")
    source = packet["source_before"]
    assert source == packet["source_after"] and not source["status"]
    assert source["commit"] == SOURCE
    assert source["compiler_sha256"] == packet["compiler_sha256"] == COMPILER
    assert source["runtime_sha256"] == packet["runtime_sha256"] == PROVIDER
    assert packet["device"] == "NVIDIA GeForce RTX 5070"
    assert packet["compute_capability"] == "12.0"
    assert packet["device_uuid"] == "GPU-cba12639-821a-7a10-4cd3-f918f9c0a545"
    assert packet["resident_inputs"] and packet["include_synthetic_projection_shapes"]
    snapshot = data("source-snapshot-d1ba6e98d.json.gz")
    assert snapshot["commit"] == SOURCE
    for path, digest in source["files"].items():
        assert hashlib.sha256(snapshot["files"][path].encode()).hexdigest() == digest
    index = json.loads((PACKET / "programs/index.json").read_text())
    assert index["source_before"] == index["source_after"] == source
    assert index["device_uuid"] == packet["device_uuid"]


def test_every_compressed_record_is_listed_and_byte_checked():
    manifest = json.loads((PACKET / "compression-manifest.json").read_text())
    assert set(manifest) == {str(path.relative_to(PACKET)) for path in PACKET.rglob("*.gz")}
    assert {"timings.json.gz", "geometry-observations.json.gz",
            "validation/corrected-validation.json.gz",
            "validation/corrected-unit.log.gz",
            "validation/corrected-compiler-route.xml.gz",
            "validation/corrected-device.xml.gz"} <= set(manifest)
    for name in manifest:
        raw(name)


def test_raw_latency_windows_keep_host_and_device_costs_separate():
    packet = data("timings.json.gz")
    expected = {(storage, shape) for storage in ("f16", "bf16") for shape in (
        (17, 19, 35), (129, 33, 257), (256, 1024, 1536), (256, 1024, 2048))}
    assert {(row["storage"], tuple(row["shape_mnk"])) for row in packet["rows"]} == expected
    modes = {"public_jvp", "portable_cached_certificate", "portable_full_certificate",
             "resident_public_jvp", "resident_portable_cached_certificate"}
    for row in packet["rows"]:
        assert set(row["host_completed_samples_us"]) == modes
        for mode, samples in row["host_completed_samples_us"].items():
            assert len(samples) == 7
            for value in samples:
                positive(value)
            assert statistics.median(samples) == pytest.approx(
                row["host_completed_medians_us"][mode], rel=1e-12)
        for name in ("device_captured_stage_ms", "resident_input_device_captured_stage_ms"):
            assert len(row[name]) == 3
            for value in row[name]:
                positive(value)
        positive(row["device_captured_program_ms"])
        positive(row["resident_input_device_captured_program_ms"])
        assert row["same_native_owner_and_images"]
        assert row["root_uploads_per_call"] == {"host": 3, "resident": 0}
        assert row["result_ownership"] == "two independent completed host f32 arrays"


def test_measured_programs_restore_without_compiler_or_provider(monkeypatch):
    from tessera import runtime as rt
    from tessera.compiler.portable_nvidia_active_source import restore

    monkeypatch.setattr(subprocess, "run", lambda *a, **k: pytest.fail("restore invoked a subprocess"))
    monkeypatch.setattr(rt, "_load_nvidia_ptx_launch", lambda: pytest.fail("restore accessed provider"))
    packet = data("timings.json.gz")
    index = json.loads((PACKET / "programs/index.json").read_text())
    assert len(index["rows"]) == len(packet["rows"]) == 8
    for measured, captured in zip(packet["rows"], index["rows"], strict=True):
        artifact = rt.RuntimeArtifact.from_json(raw("programs/" + captured["artifact"]).decode())
        assert artifact.artifact_hash == captured["artifact_hash"] == measured["artifact_hash"]
        native, _, members = restore(artifact.metadata["native_program"])
        plan = native.manifest
        assert plan["execution_state"] == "artifact_only"
        assert plan["outputs"] == [5, 6]
        assert artifact.metadata["target"] == "nvidia_sm120"
        assert [member["image_digest"] for member in members] == measured["image_digests"]
        assert [member["abi"]["block_size"] for member in members] == [128, 32, 32]
        assert "tile.buffer_view" in members[0]["tile_ir"]
        assert "tile.row_region_kernel" in members[0]["tile_ir"]
        for member in members[1:]:
            assert all(op in member["tile_ir"] for op in ("tile.view", "tile.fragment_pack", "tile.mma "))
            opcode = f"mma.sync.aligned.m16n8k16.row.col.f32.{measured['storage']}.{measured['storage']}.f32"
            assert opcode in member["ptx"]
        assert captured["bitwise_host_resident_equal"]
        assert captured["compiler_subprocesses_during_replay"] == 0
        for name in ("host_errors", "standalone_resident_errors"):
            assert len(captured[name]) == 2
            for error in captured[name]:
                assert 0 <= error["max_error_over_budget"] <= 1
                assert math.isfinite(error["max_abs_error"]) and error["max_abs_error"] >= 0


def test_driver_geometry_is_bound_to_captured_native_packages():
    from tessera import runtime as rt
    from tessera.compiler.portable_nvidia_active_source import restore

    observations = data("geometry-observations.json.gz")
    validation = data("validation/corrected-validation.json.gz")
    device = next(phase for phase in validation["phases"] if phase["phase"] == "device")
    assert observations["device_gate_log_sha256"] == device["raw_log_sha256"]
    assert observations["source"] == device["source_after"]
    expected = {(storage, shape, role) for storage in ("f16", "bf16") for role in (0, 1)
                for shape in ((1, 1, 1), (17, 19, 35), (129, 33, 257))}
    assert {(row["storage"], tuple(row["shape_mnk"]), row["source_role"])
            for row in observations["rows"]} == expected
    for row in observations["rows"]:
        artifact = rt.RuntimeArtifact.from_json(raw(row["artifact_ref"]).decode())
        assert artifact.artifact_hash == row["artifact_hash"]
        native, _, members = restore(artifact.metadata["native_program"])
        assert native.manifest["role_indices"][0] == row["source_role"]
        assert [member["image_digest"] for member in members] == row["image_digests"]
        m, n, k = row["shape_mnk"]
        records = [line.split() for line in row["driver_trace"].splitlines()]
        assert len(row["launches"]) == len(records) == len(members) == 3
        for index, (observed, record, member) in enumerate(zip(row["launches"], records, members, strict=True)):
            assert observed["entry"] == record[0] == member["entry"]
            assert observed["grid"] == list(map(int, record[1:4]))
            assert observed["block"] == list(map(int, record[4:7]))
            assert observed["grid"] == ([(m+127)//128, 1, 1] if index == 0 else [(n+7)//8, (m+15)//16, 1])
            assert observed["block"] == [member["abi"]["block_size"], 1, 1]
            assert observed["block"] == ([128, 1, 1] if index == 0 else [32, 1, 1])
            assert member["abi"]["dimensions"] == ([m, k, 0] if index == 0 else [m, n, k])
        assert len(row["errors"]) == 2
        assert all(0 <= error["max_error_over_budget"] <= 1 for error in row["errors"])


def test_required_proof_cases_executed_without_skips(tmp_path):
    for name, expected in (("compiler-route", 880), ("device", 348)):
        value = raw(f"validation/corrected-{name}.xml.gz")
        path = tmp_path / (name + ".xml")
        path.write_bytes(value)
        completed = subprocess.run([sys.executable, str(ROOT / "scripts/ci_require_executed.py"), str(path)],
                                   capture_output=True, text=True, check=False)
        assert completed.returncode == 0, completed.stdout + completed.stderr
        # Only immutable, hash-verified pytest JUnit from this packet is parsed.
        cases = ET.fromstring(value).findall(".//testcase")  # noqa: S314
        assert len(cases) == expected
        if name == "compiler-route":
            normalization = [case for case in cases if case.get("classname", "").endswith(
                "test_native_normalization_accumulator_policy")]
            assert len(normalization) == 121
        else:
            assert sum("test_portable_geometry_matches_actual_driver_launches" in case.get("name", "")
                       for case in cases) == 12


def test_full_validation_and_helper_source_are_bound_to_the_frozen_epoch():
    validation = data("validation/corrected-validation.json.gz")
    assert validation["state"] == "passed" and validation["exit_code"] == 0
    source = validation["source_before"]
    assert source == validation["source_after"] and not source["status"]
    assert source["commit"] == SOURCE
    assert source["tools"]["TESSERA_OPT"]["sha256"] == COMPILER
    assert source["tools"]["TESSERA_NVIDIA_PTX_LAUNCH_LIB"]["sha256"] == PROVIDER
    assert all(phase["exit_code"] == 0 and phase["source_before"] == phase["source_after"] == source
               for phase in validation["phases"])
    unit = next(phase for phase in validation["phases"] if phase["phase"] == "unit")
    log = raw("validation/corrected-unit.log.gz")
    assert hashlib.sha256(log).hexdigest() == unit["raw_log_sha256"]
    assert unit["passed"] == 26836 and unit["skipped"] == 7678
    assert hashlib.sha256((PACKET / "capture_programs.py").read_bytes()).hexdigest() == source["helpers"]["capture_programs.py"]
    assert hashlib.sha256(raw("validation/validate_corrected.py.txt.gz")).hexdigest() == source["helpers"]["validate_corrected.py"]


def test_pre_geometry_measurements_remain_historical_and_images_are_conserved():
    old = data("history/pre_geometry/timings.json.gz")
    current = data("timings.json.gz")
    assert old["source_before"] == old["source_after"]
    assert old["source_before"]["commit"] == "c8a1f89a28578b6bf50898872969da19e92bb6bd"
    assert len(old["rows"]) == len(current["rows"]) == 8
    for before, after in zip(old["rows"], current["rows"], strict=True):
        assert (before["storage"], before["shape_mnk"], before["permuted"]) == (
            after["storage"], after["shape_mnk"], after["permuted"])
        assert before["image_digests"] == after["image_digests"]
        assert before["artifact_hash"] != after["artifact_hash"]
    equivalence = json.loads((PACKET / "history/fixture-geometry-equivalence.json").read_text())
    assert equivalence["old_blocks"] == [128, 128, 128]
    assert equivalence["new_blocks"] == [128, 32, 32]


def test_optional_strides_review_repair_has_separate_executed_device_evidence(tmp_path):
    receipt = data("validation/optional-strides-device-validation.json.gz")
    source = receipt["source_before"]
    assert source == receipt["source_after"] and not source["status"]
    assert receipt["exit_code"] == 0
    assert source["commit"] == "e164fd4d80373f326168d050497ef0663fee7440"
    assert source["src_tree"] == "b1dac085ee213deed38ad5eaf82707bdca0006b4"
    assert hashlib.sha256(receipt["source"].encode()).hexdigest() == source["source_sha256"]
    log = raw("validation/optional-strides-device.log.gz")
    assert hashlib.sha256(log).hexdigest() == receipt["log_sha256"]
    assert b"396 passed" in log
    assert b"12 failed" in raw("validation/optional-strides-before.log.gz")
    assert b"97 passed" in raw("validation/optional-strides-after.log.gz")
    junit = raw("validation/optional-strides-device.xml.gz")
    path = tmp_path / "review.xml"
    path.write_bytes(junit)
    result = subprocess.run([sys.executable, str(ROOT / "scripts/ci_require_executed.py"), str(path)],
                            capture_output=True, text=True, check=False)
    assert result.returncode == 0, result.stdout + result.stderr
    cases = ET.fromstring(junit).findall(".//testcase")  # noqa: S314
    assert len(cases) == 396
    omitted = [case for case in cases if "test_optional_strides_resident_jvp" in case.get("name", "")]
    assert len(omitted) == 48

def test_rms_review_epoch_binds_native_numerics_packages_and_full_validation(tmp_path, monkeypatch):
    from tessera import runtime as rt
    from tessera.compiler.portable_nvidia_active_source import restore

    prefix = "rms-review/"
    validation = data(prefix + "corrected-validation.json.gz")
    source = validation["source_before"]
    assert validation["state"] == "passed" and validation["exit_code"] == 0
    assert source == validation["source_after"] and not source["status"]
    assert source["commit"] == "46a5dd8da721055a51b672adde1eb6e534789c3b"
    assert source["tools"]["TESSERA_OPT"]["sha256"] != COMPILER
    assert source["tools"]["TESSERA_NVIDIA_PTX_LAUNCH_LIB"]["sha256"] == PROVIDER
    assert all(phase["exit_code"] == 0 and phase["source_before"] == phase["source_after"] == source
               for phase in validation["phases"])
    phases = {phase["phase"]: phase for phase in validation["phases"]}
    assert phases["unit"]["passed"] == 26889
    assert phases["unit"]["skipped"] == 7678
    for phase in validation["phases"]:
        assert hashlib.sha256(raw(prefix + phase["log"] + ".gz")).hexdigest() == phase["raw_log_sha256"]
    assert hashlib.sha256(raw(prefix + "validate_corrected.py.txt.gz")).hexdigest() == source["helpers"]["validate_corrected.py"]
    assert hashlib.sha256((PACKET / "capture_programs.py").read_bytes()).hexdigest() == source["helpers"]["capture_programs.py"]
    assert b"32 failed" in raw(prefix + "tiny-epsilon-before.log.gz")
    for lane, count in (("compiler-route", 912), ("device", 416)):
        junit = raw(prefix + f"corrected-{lane}.xml.gz")
        path = tmp_path / (lane + ".xml")
        path.write_bytes(junit)
        result = subprocess.run([sys.executable, str(ROOT / "scripts/ci_require_executed.py"), str(path)],
                                capture_output=True, text=True, check=False)
        assert result.returncode == 0, result.stdout + result.stderr
        cases = ET.fromstring(junit).findall(".//testcase")  # noqa: S314
        assert len(cases) == phases[lane]["passed"] == count
        if lane == "compiler-route":
            normalization = [case for case in cases if case.get("classname", "").endswith(
                "test_native_normalization_accumulator_policy")]
            assert len(normalization) == 153
            assert sum("inverse_square_overflow" in case.get("name", "") for case in normalization) == 32
        else:
            assert sum("test_tiny_epsilon_nonzero_bf16" in case.get("name", "") for case in cases) == 20

    timings = data(prefix + "corrected-timings.json.gz")
    measured = timings["source_before"]
    assert measured == timings["source_after"] and measured["commit"] == source["commit"]
    assert measured["compiler_sha256"] == source["tools"]["TESSERA_OPT"]["sha256"]
    assert measured["runtime_sha256"] == PROVIDER
    assert timings["device_uuid"] == "GPU-cba12639-821a-7a10-4cd3-f918f9c0a545"
    snapshot = data(prefix + "source-snapshot.json.gz")
    base = data(snapshot["base_snapshot"])
    assert snapshot["commit"] == source["commit"]
    assert not (set(snapshot["inherited"]) & set(snapshot["files"]))
    files = {path: base["files"][path] for path in snapshot["inherited"]}
    files.update(snapshot["files"])
    assert set(files) == set(snapshot["sha256"])
    for path, text in files.items():
        assert hashlib.sha256(text.encode()).hexdigest() == snapshot["sha256"][path]
    for path, digest in measured["files"].items():
        assert snapshot["sha256"][path] == digest

    index = json.loads((PACKET / prefix / "programs/index.json").read_text())
    assert index["source_before"] == index["source_after"] == measured
    assert len(index["rows"]) == len(timings["rows"]) == 8
    monkeypatch.setattr(subprocess, "run", lambda *a, **k: pytest.fail("replay restoration invoked compiler"))
    monkeypatch.setattr(rt, "_load_nvidia_ptx_launch", lambda: pytest.fail("restoration accessed provider"))
    for row, captured in zip(timings["rows"], index["rows"], strict=True):
        artifact = rt.RuntimeArtifact.from_json(raw(prefix + "programs/" + captured["artifact"]).decode())
        assert artifact.artifact_hash == row["artifact_hash"] == captured["artifact_hash"]
        _, _, members = restore(artifact.metadata["native_program"])
        assert [member["image_digest"] for member in members] == row["image_digests"]
        assert [member["abi"]["block_size"] for member in members] == [128, 32, 32]
        assert captured["compiler_subprocesses_during_replay"] == 0 and captured["bitwise_host_resident_equal"]
        for mode, samples in row["host_completed_samples_us"].items():
            assert len(samples) == 7
            assert all(math.isfinite(value) and value > 0 for value in samples)
            assert statistics.median(samples) == pytest.approx(row["host_completed_medians_us"][mode])
        for name in ("host_errors", "standalone_resident_errors"):
            assert all(0 <= error["max_error_over_budget"] <= 1 for error in captured[name])

