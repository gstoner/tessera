from __future__ import annotations

import hashlib
import json
from copy import deepcopy
from pathlib import Path

import pytest

from tessera.compiler.e2e_fleet import (
    FIXTURE_SCHEMA,
    MANIFEST_SCHEMA,
    REPORT_SCHEMA,
    FleetEvidenceError,
    compare_backend_reports,
    fleet_dashboard_rows,
    load_fixture_corpus,
    render_fleet_csv,
    seal_packet,
    validate_backend_report,
    validate_packet,
)


def _digest(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


def _fixtures() -> dict[str, dict]:
    return {
        "softmax": {
            "fixture_id": "softmax",
            "family": "softmax",
            "dtype": "fp32",
            "shape": [2, 2],
            "semantic_contract": "last-axis stable softmax",
            "oracle": [[0.5, 0.5], [0.5, 0.5]],
            "tolerance": {"atol": 1e-6, "rtol": 1e-6, "equal_nan": False},
        }
    }


def _report(target: str = "nvidia_sm120", architecture: str = "sm_120a") -> dict:
    image = _digest(f"{target}:image")
    descriptor = _digest(f"{target}:descriptor")
    cache_key = _digest(f"{target}:cache")
    benchmarks = []
    for domain in ("device_event", "end_to_end"):
        run_medians = [990.0, 1010.0] if domain == "device_event" else [1980.0, 2020.0]
        benchmarks.append({
            "family": "softmax",
            "route": "canonical_descriptor",
            "timing_domain": domain,
            "median_ns": 1000.0 if domain == "device_event" else 2000.0,
            "run_medians_ns": run_medians,
            "stability_limit_pct": 5.0,
            "stable": True,
            "selected": True,
            "repetitions": 20,
            "warmups": 2,
            "discard_first": True,
            "resource_fingerprint": _digest(f"{target}:resources"),
        })
    return {
        "schema": REPORT_SCHEMA,
        "target": target,
        "architecture": architecture,
        "device": {"exact": True, "identity": f"test-device:{target}"},
        "source_commit": "a" * 40,
        "toolchain_fingerprint": f"toolchain:{target}",
        "scope": ["softmax"],
        "required_timing_domains": ["device_event", "end_to_end"],
        "fixtures": [{
            "fixture_id": "softmax",
            "levels": {"a": "proven", "b": "proven", "c": "proven"},
            "actual": [[0.5, 0.5], [0.5, 0.5]],
            "image_digest": image,
            "descriptor_digest": descriptor,
        }],
        "cache_proofs": [{
            "fixture_id": "softmax",
            "cold": {
                "compile_state": "cold", "cache_key": cache_key,
                "image_digest": image, "descriptor_digest": descriptor,
            },
            "warm": {
                "compile_state": "warm_cache", "cache_key": cache_key,
                "image_digest": image, "descriptor_digest": descriptor,
            },
        }],
        "benchmarks": benchmarks,
    }


def test_checked_in_differential_fixture_corpus_is_valid() -> None:
    corpus = load_fixture_corpus()
    assert len(corpus) >= 5
    assert {row["family"] for row in corpus.values()} >= {
        "matmul", "softmax", "reduction", "paged_kv", "moe",
    }


def test_report_requires_numerical_level_c_cache_and_both_timing_domains() -> None:
    report = _report()
    summary = validate_backend_report(report, fixtures=_fixtures())
    assert summary["level_c_fixtures"] == 1
    assert summary["benchmark_rows"] == 2

    bad = deepcopy(report)
    bad["cache_proofs"][0]["warm"]["image_digest"] = _digest("stale")
    with pytest.raises(FleetEvidenceError, match="does not reproduce image_digest"):
        validate_backend_report(bad, fixtures=_fixtures())

    bad = deepcopy(report)
    bad["benchmarks"] = bad["benchmarks"][:1]
    with pytest.raises(FleetEvidenceError, match="lacks its required timing domains"):
        validate_backend_report(bad, fixtures=_fixtures())

    cpu = _report("x86", "x86_64_base")
    cpu["required_timing_domains"] = ["kernel_wall", "end_to_end"]
    cpu["benchmarks"][0]["timing_domain"] = "kernel_wall"
    assert validate_backend_report(cpu, fixtures=_fixtures())["target"] == "x86"

    packaged = deepcopy(cpu)
    packaged["cache_proofs"][0]["cold"]["compile_state"] = "prepackaged"
    packaged["cache_proofs"][0]["warm"]["compile_state"] = "prepackaged"
    assert validate_backend_report(packaged, fixtures=_fixtures())["architecture"] == "x86_64_base"

    bad = deepcopy(report)
    bad["benchmarks"][0]["run_medians_ns"] = [900.0, 1100.0]
    bad["benchmarks"][0]["median_ns"] = 1000.0
    with pytest.raises(FleetEvidenceError, match="is not stable"):
        validate_backend_report(bad, fixtures=_fixtures())

    bad = deepcopy(report)
    bad["fixtures"][0]["actual"][0][0] = 0.6
    with pytest.raises(FleetEvidenceError, match="fails its numerical policy"):
        validate_backend_report(bad, fixtures=_fixtures())


def test_cross_backend_differential_compares_common_actual_values() -> None:
    left = _report("nvidia_sm120")
    right = _report("rocm_gfx1151", "gfx1151")
    summary = compare_backend_reports(left, right, fixtures=_fixtures())
    assert summary == {
        "left_target": "nvidia_sm120",
        "left_architecture": "sm_120a",
        "right_target": "rocm_gfx1151",
        "right_architecture": "gfx1151",
        "common_fixtures": 1,
        "maximum_absolute_error": 0.0,
    }
    same_target = _report("x86", "x86_64_base")
    avx512 = _report("x86", "x86_64_avx512")
    assert compare_backend_reports(
        same_target, avx512, fixtures=_fixtures(),
    )["common_fixtures"] == 1
    right["fixtures"][0]["actual"][0][0] = 0.50001
    with pytest.raises(FleetEvidenceError, match="fails its numerical policy"):
        compare_backend_reports(left, right, fixtures=_fixtures())


def test_packet_seal_is_deterministic_and_tamper_evident(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture_path = tmp_path / "fixtures.json"
    fixture_path.write_text(json.dumps({
        "schema": FIXTURE_SCHEMA, "fixtures": list(_fixtures().values()),
    }))
    monkeypatch.setattr("tessera.compiler.e2e_fleet.FIXTURE_PATH", fixture_path)
    packet = tmp_path / "packet"
    packet.mkdir()
    (packet / "report.json").write_text(json.dumps(_report()))
    (packet / "resources.txt").write_text("registers=32 spills=0\n")
    manifest = seal_packet(packet)
    assert manifest["schema"] == MANIFEST_SCHEMA
    assert validate_packet(packet)["target"] == "nvidia_sm120"
    first = (packet / "manifest.json").read_text()
    seal_packet(packet)
    assert (packet / "manifest.json").read_text() == first
    (packet / "resources.txt").write_text("tampered\n")
    with pytest.raises(FleetEvidenceError, match="hash mismatch"):
        validate_packet(packet)


def test_packet_rejects_symlinked_attachments(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture_path = tmp_path / "fixtures.json"
    fixture_path.write_text(json.dumps({
        "schema": FIXTURE_SCHEMA, "fixtures": list(_fixtures().values()),
    }))
    monkeypatch.setattr("tessera.compiler.e2e_fleet.FIXTURE_PATH", fixture_path)
    packet = tmp_path / "packet"
    packet.mkdir()
    (packet / "report.json").write_text(json.dumps(_report()))
    (packet / "outside.txt").symlink_to(tmp_path / "outside.txt")
    with pytest.raises(FleetEvidenceError, match="cannot contain symlink"):
        seal_packet(packet)


def test_dashboard_keeps_missing_packets_and_hardware_terminals_explicit(
    tmp_path: Path,
) -> None:
    rows = fleet_dashboard_rows(tmp_path)
    states = {(row.target, row.state) for row in rows}
    assert ("nvidia_sm120", "packet_pending") in states
    assert ("nvidia_sm90", "hardware_deferred") in states
    assert "release_ready" not in {row.state for row in rows}
    csv_text = render_fleet_csv(rows)
    assert csv_text.startswith("schema,target,architecture,backend,family,state")
    assert "tessera.e2e-release-packet.v1,nvidia_sm90,sm_90a" in csv_text
