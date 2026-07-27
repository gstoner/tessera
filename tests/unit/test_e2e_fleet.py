from __future__ import annotations

import hashlib
import json
from copy import deepcopy
from pathlib import Path

import pytest

from tessera.compiler.e2e_fleet import (
    FIXTURE_SCHEMA,
    FLEET_REGISTRATIONS,
    MANIFEST_SCHEMA,
    REPORT_SCHEMA,
    FleetEvidenceError,
    compare_backend_reports,
    discover_packets,
    fleet_dashboard_rows,
    load_fixture_corpus,
    render_fleet_csv,
    seal_packet,
    validate_backend_report,
    validate_packet,
)


REPO_ROOT = Path(__file__).resolve().parents[2]


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
        benchmarks.append(
            {
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
            }
        )
    return {
        "schema": REPORT_SCHEMA,
        "target": target,
        "architecture": architecture,
        "device": {"exact": True, "identity": f"test-device:{target}"},
        "source_commit": "a" * 40,
        "toolchain_fingerprint": f"toolchain:{target}",
        "scope": ["softmax"],
        "required_timing_domains": ["device_event", "end_to_end"],
        "fixtures": [
            {
                "fixture_id": "softmax",
                "levels": {"a": "proven", "b": "proven", "c": "proven"},
                "actual": [[0.5, 0.5], [0.5, 0.5]],
                "image_digest": image,
                "descriptor_digest": descriptor,
            }
        ],
        "cache_proofs": [
            {
                "fixture_id": "softmax",
                "cold": {
                    "compile_state": "cold",
                    "cache_key": cache_key,
                    "image_digest": image,
                    "descriptor_digest": descriptor,
                },
                "warm": {
                    "compile_state": "warm_cache",
                    "cache_key": cache_key,
                    "image_digest": image,
                    "descriptor_digest": descriptor,
                },
            }
        ],
        "benchmarks": benchmarks,
    }


def test_checked_in_differential_fixture_corpus_is_valid() -> None:
    corpus = load_fixture_corpus()
    assert len(corpus) >= 8
    assert {row["family"] for row in corpus.values()} >= {
        "matmul",
        "softmax",
        "reduction",
        "epilogue",
        "attention",
        "paged_kv",
        "replay_ssm",
        "moe",
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
    assert (
        compare_backend_reports(
            same_target,
            avx512,
            fixtures=_fixtures(),
        )["common_fixtures"]
        == 1
    )
    right["fixtures"][0]["actual"][0][0] = 0.50001
    with pytest.raises(FleetEvidenceError, match="fails its numerical policy"):
        compare_backend_reports(left, right, fixtures=_fixtures())


def test_packet_seal_is_deterministic_and_tamper_evident(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture_path = tmp_path / "fixtures.json"
    fixture_path.write_text(
        json.dumps(
            {
                "schema": FIXTURE_SCHEMA,
                "fixtures": list(_fixtures().values()),
            }
        )
    )
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
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture_path = tmp_path / "fixtures.json"
    fixture_path.write_text(
        json.dumps(
            {
                "schema": FIXTURE_SCHEMA,
                "fixtures": list(_fixtures().values()),
            }
        )
    )
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


def test_every_checked_in_packet_still_validates_without_its_device() -> None:
    """A sealed packet is portable evidence: no device, no rebuild, no network.

    This is the property that makes fleet truth reviewable off-host. It runs
    on any machine, including one with no Apple, NVIDIA, or AMD hardware — the
    seal is over bytes, not over the ability to re-execute them.
    """
    packets = discover_packets()
    assert packets, "no sealed fleet packets are checked in"
    for (target, architecture), (packet_dir, summary) in packets.items():
        assert summary["target"] == target
        assert summary["architecture"] == architecture
        # An oracle-matching packet is the point; a packet that merely parses
        # is not evidence.
        assert summary["level_c_fixtures"] >= 1, f"{target} proves no Level-C fixture"
        manifest = json.loads((packet_dir / "manifest.json").read_text(encoding="utf-8"))
        assert "report.json" in manifest["files"]


def test_both_apple_identities_are_sealed_independently() -> None:
    """Apple GPU and Apple CPU are separate registrations, not one Apple lane.

    Recording one must never appear to close the other, so this asserts two
    distinct packets, two distinct directories, and two distinct device
    identities rather than a single shared "Apple" row.
    """
    packets = discover_packets()
    gpu = packets.get(("apple_gpu", "apple7"))
    cpu = packets.get(("apple_cpu", "apple_m1_max"))
    assert gpu is not None, "apple_gpu/apple7 has no sealed packet"
    assert cpu is not None, "apple_cpu/apple_m1_max has no sealed packet"
    assert gpu[0] != cpu[0], "the two Apple identities share a packet directory"

    reports = {}
    for name, (packet_dir, _) in (("gpu", gpu), ("cpu", cpu)):
        reports[name] = json.loads(
            (packet_dir / "report.json").read_text(encoding="utf-8"))
    # Each report speaks only for its own identity.
    assert reports["gpu"]["target"] == "apple_gpu"
    assert reports["cpu"]["target"] == "apple_cpu"
    # Neither lane can report device_event today: the value-descriptor ABI does
    # not feed the Metal command-buffer timer. If a future packet claims it,
    # that claim must come with a real device timer, not a host clock.
    for report in reports.values():
        assert set(report["required_timing_domains"]) == {"kernel_wall", "end_to_end"}


def test_apple_packets_only_claim_families_their_registration_declares() -> None:
    """The dashboard rejects undeclared families; catch it here with a clearer message."""
    registrations = {
        (row.target, row.architecture): row.families
        for row in FLEET_REGISTRATIONS
    }
    packets = discover_packets()
    for identity in (("apple_gpu", "apple7"), ("apple_cpu", "apple_m1_max")):
        packet_dir, _ = packets[identity]
        report = json.loads((packet_dir / "report.json").read_text(encoding="utf-8"))
        undeclared = set(report["scope"]) - set(registrations[identity])
        assert not undeclared, (
            f"{identity[0]} packet claims undeclared families: "
            f"{', '.join(sorted(undeclared))}")


def test_apple_gpu_packet_proves_metal_placement_not_just_numerics() -> None:
    """A matching oracle does not prove *where* the work ran.

    `tessera_apple_gpu_softmax_f32` and `tessera_apple_gpu_bmm_f32` are void
    ABIs that fall through to a numerically-identical CPU reference. Every
    numerical check in the packet would still pass on that path, so the packet
    must carry separate positive placement evidence from the status-bearing
    twins — at the fixture shape *and* at the larger timing shape.
    """
    packets = discover_packets()
    packet_dir, _ = packets[("apple_gpu", "apple7")]
    resources = json.loads(
        (packet_dir / "resources.json").read_text(encoding="utf-8"))
    rows = resources["rows"]
    assert rows, "apple_gpu packet records no resource rows"
    for row in rows:
        placement = row["placement"]
        for shape in ("fixture", "timing"):
            assert placement[shape]["gpu_placement_proven"] is True, (
                f"{row['family']} has no proven Metal placement at the {shape} "
                "shape; it may have run on the CPU reference")
            assert placement[shape]["status_symbol"].endswith("_status"), (
                "placement must come from a status-bearing ABI, not the void one")
    # The MSL dispatch record is route-dependent: the MSL softmax populates it,
    # the MPSGraph matmul does not. Absence is reported, never inferred as CPU.
    by_family = {row["family"]: row["placement"]["timing"] for row in rows}
    assert by_family["softmax"]["msl_dispatch_record"] is True
    assert by_family["softmax"]["execution_width"] > 0
    assert by_family["matmul"]["msl_dispatch_record"] is False


def test_apple_gpu_packet_fingerprints_kernel_source_not_the_toolchain() -> None:
    """`source_fingerprint` must identify the runtime source that produced the MSL.

    Recording the aggregate toolchain digest here would make the field useless
    for telling which revision of `apple_gpu_runtime.mm` ran.
    """
    packets = discover_packets()
    packet_dir, _ = packets[("apple_gpu", "apple7")]
    resources = json.loads(
        (packet_dir / "resources.json").read_text(encoding="utf-8"))
    report = json.loads((packet_dir / "report.json").read_text(encoding="utf-8"))
    fingerprint = resources["metal"]["evidence"]["source_fingerprint"]["value"]
    assert fingerprint.startswith("sha256:")
    assert fingerprint != f"sha256:{report['toolchain_fingerprint']}"
    assert fingerprint.removeprefix("sha256:") != report["toolchain_fingerprint"]

    runtime_source = (
        REPO_ROOT
        / "src/compiler/codegen/Tessera_Apple_Backend/runtime/apple_gpu_runtime.mm"
    )
    expected = hashlib.sha256(runtime_source.read_bytes()).hexdigest()
    assert fingerprint == f"sha256:{expected}", (
        "the sealed source fingerprint no longer matches apple_gpu_runtime.mm; "
        "re-record the packet after changing the runtime source")


def test_status_bearing_abis_exist_for_every_sealed_apple_gpu_route() -> None:
    """The void ABIs cannot prove placement, so their status twins must exist."""
    runtime_source = (
        REPO_ROOT
        / "src/compiler/codegen/Tessera_Apple_Backend/runtime/apple_gpu_runtime.mm"
    ).read_text(encoding="utf-8", errors="replace")
    for symbol in ("tessera_apple_gpu_softmax_f32_status",
                   "tessera_apple_gpu_bmm_f32_status"):
        assert f'extern "C" int32_t {symbol}(' in runtime_source, (
            f"{symbol} was removed; the apple_gpu packet cannot prove Metal "
            "placement without it")
