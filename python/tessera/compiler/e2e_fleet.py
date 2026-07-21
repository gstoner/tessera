"""Family-granular fleet evidence for E2E-SPINE-3.

This module is intentionally backend-neutral.  It validates evidence emitted
by an exact-device runner; it never launches a backend, chooses a schedule, or
turns missing hardware into a pass.  A packet joins four independently useful
proofs:

* shared differential fixtures and their numerical oracle;
* bounded Level-A/B/C provenance for one exact target;
* cold/warm native-image and launch-descriptor cache identity;
* device-event and end-to-end benchmark evidence.

The packet manifest hash-seals the report and any backend-owned attachments.
Portable CI can therefore validate and render fleet truth without possessing
the device that produced the evidence.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


REPORT_SCHEMA = "tessera.e2e-backend-report.v1"
MANIFEST_SCHEMA = "tessera.e2e-release-packet.v1"
FIXTURE_SCHEMA = "tessera.e2e-differential-fixtures.v1"

_REPO_ROOT = Path(__file__).resolve().parents[3]
EVIDENCE_ROOT = _REPO_ROOT / "docs" / "audit" / "evidence" / "e2e_spine"
FIXTURE_PATH = _REPO_ROOT / "benchmarks" / "e2e_spine" / "fixtures.json"
FLEET_MD_PATH = _REPO_ROOT / "docs" / "audit" / "generated" / "e2e_fleet.md"
FLEET_CSV_PATH = _REPO_ROOT / "docs" / "audit" / "generated" / "e2e_fleet.csv"


class FleetEvidenceError(ValueError):
    """A release packet is malformed, incomplete, or internally inconsistent."""


@dataclass(frozen=True)
class FleetRegistration:
    target: str
    architecture: str
    backend: str
    families: tuple[str, ...]
    terminal: str
    reason: str


# This registry describes E2E-SPINE-3 packet obligations, not backend support.
# Bounded family support continues to live in the backend registries/plans.
FLEET_REGISTRATIONS: tuple[FleetRegistration, ...] = (
    FleetRegistration(
        "apple_gpu", "apple7", "apple", ("matmul", "softmax", "linalg", "ppo", "ebm", "clifford"),
        "packet_pending", "Bounded Apple GPU Level-C scope awaits a v1 fleet packet.",
    ),
    FleetRegistration(
        "apple_cpu", "apple_m1_max", "apple", ("matmul", "linalg"), "packet_pending",
        "Bounded Apple CPU Level-C scope awaits exact-host release evidence.",
    ),
    FleetRegistration(
        "x86", "x86_64_base", "x86", ("softmax", "reduction"),
        "packet_pending", "Portable non-AVX512 x86 scope awaits an NR2-host packet.",
    ),
    FleetRegistration(
        "x86", "x86_64_avx512", "x86",
        ("matmul", "softmax", "reduction", "attention", "linalg"),
        "packet_pending", "AVX512 x86 scope awaits a Strix Halo host packet.",
    ),
    FleetRegistration(
        "nvidia_sm120", "sm_120a", "nvidia",
        ("matmul", "softmax", "reduction", "epilogue", "attention", "paged_kv", "replay_ssm", "moe"),
        "packet_pending", "Closed SM120 scope awaits a hash-sealed family packet.",
    ),
    FleetRegistration(
        "nvidia_sm90", "sm_90a", "nvidia", (), "hardware_deferred",
        "Exact Hopper device unavailable; SM120 evidence does not transfer.",
    ),
    FleetRegistration(
        "nvidia_sm100", "sm_100a", "nvidia", (), "hardware_deferred",
        "Exact datacenter Blackwell device unavailable; SM120 evidence does not transfer.",
    ),
    FleetRegistration(
        "rocm_gfx1151", "gfx1151", "rocm", ("softmax", "reduction", "paged_kv", "moe"),
        "packet_pending", "Closed gfx1151 E2E scope awaits a v1 fleet packet.",
    ),
)


@dataclass(frozen=True)
class FleetDashboardRow:
    target: str
    architecture: str
    backend: str
    family: str
    state: str
    packet: str
    source_commit: str
    reason: str


def _load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise FleetEvidenceError(f"cannot read JSON {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise FleetEvidenceError(f"{path} must contain a JSON object")
    return value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _is_digest(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(char in "0123456789abcdef" for char in value)
    )


def _require_text(row: Mapping[str, Any], key: str, where: str) -> str:
    value = row.get(key)
    if not isinstance(value, str) or not value.strip():
        raise FleetEvidenceError(f"{where}.{key} must be non-empty text")
    return value


def _flatten_numbers(value: object, where: str) -> list[float]:
    if isinstance(value, bool):
        raise FleetEvidenceError(f"{where} contains a boolean, not a number")
    if isinstance(value, (int, float)):
        return [float(value)]
    if isinstance(value, list):
        out: list[float] = []
        for index, item in enumerate(value):
            out.extend(_flatten_numbers(item, f"{where}[{index}]"))
        return out
    raise FleetEvidenceError(f"{where} must be a nested numeric list")


def _numerically_equal(
    actual: Sequence[float], oracle: Sequence[float], *, atol: float, rtol: float,
    equal_nan: bool,
) -> tuple[bool, float, float]:
    if len(actual) != len(oracle):
        return False, math.inf, math.inf
    max_abs = 0.0
    max_rel = 0.0
    for got, want in zip(actual, oracle):
        if math.isnan(got) or math.isnan(want):
            if equal_nan and math.isnan(got) and math.isnan(want):
                continue
            return False, math.inf, math.inf
        if math.isinf(got) or math.isinf(want):
            if got == want:
                continue
            return False, math.inf, math.inf
        delta = abs(got - want)
        relative = delta / abs(want) if want else (0.0 if delta == 0.0 else math.inf)
        max_abs = max(max_abs, delta)
        max_rel = max(max_rel, relative)
        if delta > atol + rtol * abs(want):
            return False, max_abs, max_rel
    return True, max_abs, max_rel


def load_fixture_corpus(path: Path | None = None) -> dict[str, dict[str, Any]]:
    path = FIXTURE_PATH if path is None else path
    payload = _load_json(path)
    if payload.get("schema") != FIXTURE_SCHEMA:
        raise FleetEvidenceError(f"{path} has unsupported fixture schema")
    fixtures = payload.get("fixtures")
    if not isinstance(fixtures, list) or not fixtures:
        raise FleetEvidenceError(f"{path} must declare at least one fixture")
    result: dict[str, dict[str, Any]] = {}
    for index, fixture in enumerate(fixtures):
        where = f"fixtures[{index}]"
        if not isinstance(fixture, dict):
            raise FleetEvidenceError(f"{where} must be an object")
        fixture_id = _require_text(fixture, "fixture_id", where)
        if fixture_id in result:
            raise FleetEvidenceError(f"duplicate fixture_id {fixture_id}")
        for key in ("family", "dtype", "semantic_contract"):
            _require_text(fixture, key, where)
        shape = fixture.get("shape")
        if not isinstance(shape, list) or not shape or not all(
            isinstance(dim, int) and dim > 0 for dim in shape
        ):
            raise FleetEvidenceError(f"{where}.shape must contain positive dimensions")
        policy = fixture.get("tolerance")
        if not isinstance(policy, dict):
            raise FleetEvidenceError(f"{where}.tolerance must be an object")
        for key in ("atol", "rtol"):
            value = policy.get(key)
            if not isinstance(value, (int, float)) or value < 0:
                raise FleetEvidenceError(f"{where}.tolerance.{key} must be non-negative")
        _flatten_numbers(fixture.get("oracle"), f"{where}.oracle")
        result[fixture_id] = fixture
    return result


def validate_backend_report(
    report: Mapping[str, Any], *,
    fixtures: Mapping[str, Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    """Validate one exact-target report and derive its proof summary."""
    if report.get("schema") != REPORT_SCHEMA:
        raise FleetEvidenceError("unsupported backend report schema")
    target = _require_text(report, "target", "report")
    architecture = _require_text(report, "architecture", "report")
    source_commit = _require_text(report, "source_commit", "report")
    if len(source_commit) != 40 or any(c not in "0123456789abcdef" for c in source_commit):
        raise FleetEvidenceError("report.source_commit must be a lowercase 40-hex commit")
    _require_text(report, "toolchain_fingerprint", "report")
    device = report.get("device")
    if not isinstance(device, dict) or device.get("exact") is not True:
        raise FleetEvidenceError("report.device must identify an exact device")
    _require_text(device, "identity", "report.device")
    scope = report.get("scope")
    if not isinstance(scope, list) or not scope or not all(
        isinstance(family, str) and family for family in scope
    ):
        raise FleetEvidenceError("report.scope must contain operation families")
    if len(set(scope)) != len(scope):
        raise FleetEvidenceError("report.scope contains duplicate families")
    required_domains = report.get("required_timing_domains")
    allowed_domains = {"device_event", "kernel_wall", "end_to_end"}
    if (
        not isinstance(required_domains, list)
        or len(required_domains) != 2
        or len(set(required_domains)) != 2
        or "end_to_end" not in required_domains
        or any(domain not in allowed_domains for domain in required_domains)
    ):
        raise FleetEvidenceError(
            "report.required_timing_domains must pair end_to_end with "
            "device_event or kernel_wall"
        )

    corpus = dict(fixtures) if fixtures is not None else load_fixture_corpus()
    results = report.get("fixtures")
    if not isinstance(results, list) or not results:
        raise FleetEvidenceError("report.fixtures must contain results")
    fixture_ids: set[str] = set()
    level_c_ids: set[str] = set()
    result_by_fixture: dict[str, Mapping[str, Any]] = {}
    families_seen: set[str] = set()
    maximum_error = 0.0
    for index, result in enumerate(results):
        where = f"report.fixtures[{index}]"
        if not isinstance(result, dict):
            raise FleetEvidenceError(f"{where} must be an object")
        fixture_id = _require_text(result, "fixture_id", where)
        if fixture_id in fixture_ids:
            raise FleetEvidenceError(f"duplicate fixture result {fixture_id}")
        fixture_ids.add(fixture_id)
        result_by_fixture[fixture_id] = result
        fixture = corpus.get(fixture_id)
        if fixture is None:
            raise FleetEvidenceError(f"unknown differential fixture {fixture_id}")
        family = str(fixture["family"])
        if family not in scope:
            raise FleetEvidenceError(f"fixture {fixture_id} is outside report scope")
        families_seen.add(family)
        levels = result.get("levels")
        if not isinstance(levels, dict) or any(
            levels.get(level) not in {"proven", "not_applicable"}
            for level in ("a", "b", "c")
        ):
            raise FleetEvidenceError(f"{where}.levels must explicitly classify A/B/C")
        actual = _flatten_numbers(result.get("actual"), f"{where}.actual")
        oracle = _flatten_numbers(fixture["oracle"], f"fixture {fixture_id}.oracle")
        tolerance = fixture["tolerance"]
        passed, max_abs, _ = _numerically_equal(
            actual, oracle, atol=float(tolerance["atol"]),
            rtol=float(tolerance["rtol"]),
            equal_nan=bool(tolerance.get("equal_nan", False)),
        )
        if not passed:
            raise FleetEvidenceError(
                f"fixture {fixture_id} fails its numerical policy (max_abs={max_abs})"
            )
        maximum_error = max(maximum_error, max_abs)
        if levels["c"] == "proven":
            level_c_ids.add(fixture_id)
            for key in ("image_digest", "descriptor_digest"):
                if not _is_digest(result.get(key)):
                    raise FleetEvidenceError(f"{where}.{key} must be a sha256 digest")
            if levels["a"] != "proven" or levels["b"] != "proven":
                raise FleetEvidenceError(f"fixture {fixture_id} cannot prove C without A and B")
    if families_seen != set(scope):
        missing = ", ".join(sorted(set(scope) - families_seen))
        raise FleetEvidenceError(f"report scope has no fixture evidence for: {missing}")

    cache_rows = report.get("cache_proofs")
    if not isinstance(cache_rows, list):
        raise FleetEvidenceError("report.cache_proofs must be a list")
    cache_by_fixture: dict[str, Mapping[str, Any]] = {}
    for index, proof in enumerate(cache_rows):
        where = f"report.cache_proofs[{index}]"
        if not isinstance(proof, dict):
            raise FleetEvidenceError(f"{where} must be an object")
        fixture_id = _require_text(proof, "fixture_id", where)
        if fixture_id in cache_by_fixture:
            raise FleetEvidenceError(f"duplicate cache proof {fixture_id}")
        cache_by_fixture[fixture_id] = proof
        cold, warm = proof.get("cold"), proof.get("warm")
        if not isinstance(cold, dict) or not isinstance(warm, dict):
            raise FleetEvidenceError(f"{where} requires cold and warm records")
        compile_states = (cold.get("compile_state"), warm.get("compile_state"))
        if compile_states not in {
            ("cold", "warm_cache"),
            ("prepackaged", "prepackaged"),
        }:
            raise FleetEvidenceError(f"{where} has invalid compile states")
        for key in ("cache_key", "image_digest", "descriptor_digest"):
            if not _is_digest(cold.get(key)) or cold.get(key) != warm.get(key):
                raise FleetEvidenceError(f"{where} does not reproduce {key}")
        result = result_by_fixture.get(fixture_id)
        if result is None:
            raise FleetEvidenceError(f"{where} refers to unknown result {fixture_id}")
        if fixture_id in level_c_ids:
            for key in ("image_digest", "descriptor_digest"):
                if cold.get(key) != result.get(key):
                    raise FleetEvidenceError(f"{where}.{key} disagrees with fixture provenance")
    missing_cache = level_c_ids - set(cache_by_fixture)
    if missing_cache:
        raise FleetEvidenceError(
            "missing cold/warm cache proof for: " + ", ".join(sorted(missing_cache))
        )

    benchmarks = report.get("benchmarks")
    if not isinstance(benchmarks, list):
        raise FleetEvidenceError("report.benchmarks must be a list")
    domains_by_family: dict[str, set[str]] = {family: set() for family in scope}
    selected_by_domain: dict[tuple[str, str], int] = {}
    for index, row in enumerate(benchmarks):
        where = f"report.benchmarks[{index}]"
        if not isinstance(row, dict):
            raise FleetEvidenceError(f"{where} must be an object")
        family = _require_text(row, "family", where)
        if family not in domains_by_family:
            raise FleetEvidenceError(f"{where}.family is outside report scope")
        domain = row.get("timing_domain")
        if domain not in required_domains:
            raise FleetEvidenceError(f"{where}.timing_domain is invalid")
        domains_by_family[family].add(str(domain))
        samples = row.get("run_medians_ns")
        if not isinstance(samples, list) or len(samples) < 2 or not all(
            isinstance(value, (int, float)) and value > 0 for value in samples
        ):
            raise FleetEvidenceError(f"{where}.run_medians_ns requires two positive runs")
        expected_median = float(statistics.median(samples))
        if not isinstance(row.get("median_ns"), (int, float)) or not math.isclose(
            float(row["median_ns"]), expected_median, rel_tol=1e-12, abs_tol=0.0,
        ):
            raise FleetEvidenceError(f"{where}.median_ns disagrees with run medians")
        stability_limit = row.get("stability_limit_pct")
        if not isinstance(stability_limit, (int, float)) or stability_limit <= 0:
            raise FleetEvidenceError(f"{where}.stability_limit_pct must be positive")
        stability_delta = (max(samples) - min(samples)) / min(samples) * 100.0
        if row.get("stable") is not True or stability_delta > float(stability_limit):
            raise FleetEvidenceError(
                f"{where} is not stable ({stability_delta:.3f}% > {stability_limit}%)"
            )
        if row.get("selected") not in {True, False}:
            raise FleetEvidenceError(f"{where}.selected must be boolean")
        if row["selected"]:
            selection_key = (family, str(domain))
            selected_by_domain[selection_key] = selected_by_domain.get(selection_key, 0) + 1
        if not isinstance(row.get("repetitions"), int) or row["repetitions"] <= 0:
            raise FleetEvidenceError(f"{where}.repetitions must be a positive integer")
        if not isinstance(row.get("warmups"), int) or row["warmups"] <= 0:
            raise FleetEvidenceError(f"{where}.warmups must be a positive integer")
        if row.get("discard_first") is not True:
            raise FleetEvidenceError(f"{where} must discard first-use compilation")
        _require_text(row, "route", where)
        _require_text(row, "resource_fingerprint", where)
    for family, domains in domains_by_family.items():
        if domains != set(required_domains):
            raise FleetEvidenceError(f"family {family} lacks its required timing domains")
        for domain in domains:
            if selected_by_domain.get((family, domain)) != 1:
                raise FleetEvidenceError(
                    f"family {family} requires one selected route in {domain}"
                )

    return {
        "target": target,
        "architecture": architecture,
        "source_commit": source_commit,
        "families": len(scope),
        "fixtures": len(results),
        "level_c_fixtures": len(level_c_ids),
        "cache_proofs": len(cache_rows),
        "benchmark_rows": len(benchmarks),
        "maximum_absolute_error": maximum_error,
    }


def seal_packet(packet_dir: Path) -> dict[str, Any]:
    """Validate ``report.json`` and hash-seal every other packet attachment."""
    packet_dir = packet_dir.resolve()
    report_path = packet_dir / "report.json"
    report = _load_json(report_path)
    summary = validate_backend_report(report)
    files: dict[str, dict[str, Any]] = {}
    for path in sorted(packet_dir.rglob("*")):
        if path.is_symlink():
            raise FleetEvidenceError(f"release packet cannot contain symlink {path}")
        if not path.is_file() or path.name == "manifest.json":
            continue
        relative = path.relative_to(packet_dir).as_posix()
        files[relative] = {"bytes": path.stat().st_size, "sha256": _sha256(path)}
    manifest = {
        "schema": MANIFEST_SCHEMA,
        "target": summary["target"],
        "architecture": summary["architecture"],
        "tested_commit": summary["source_commit"],
        "files": files,
        "validation": summary,
    }
    (packet_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return manifest


def validate_packet(packet_dir: Path) -> dict[str, Any]:
    manifest = _load_json(packet_dir / "manifest.json")
    if manifest.get("schema") != MANIFEST_SCHEMA:
        raise FleetEvidenceError(f"{packet_dir} has unsupported packet schema")
    files = manifest.get("files")
    if not isinstance(files, dict) or "report.json" not in files:
        raise FleetEvidenceError(f"{packet_dir} manifest must seal report.json")
    actual_files = {
        path.relative_to(packet_dir).as_posix()
        for path in packet_dir.rglob("*")
        if path.is_file() and path.name != "manifest.json"
    }
    if any(path.is_symlink() for path in packet_dir.rglob("*")):
        raise FleetEvidenceError(f"{packet_dir} cannot contain symlinks")
    if actual_files != set(files):
        raise FleetEvidenceError(f"{packet_dir} sealed file set does not match packet")
    for relative, record in files.items():
        path = packet_dir / relative
        if not isinstance(record, dict):
            raise FleetEvidenceError(f"invalid manifest record for {relative}")
        if record.get("bytes") != path.stat().st_size or record.get("sha256") != _sha256(path):
            raise FleetEvidenceError(f"hash mismatch for {relative}")
    report = _load_json(packet_dir / "report.json")
    summary = validate_backend_report(report)
    if manifest.get("target") != summary["target"]:
        raise FleetEvidenceError("packet target disagrees with report")
    if manifest.get("architecture") != summary["architecture"]:
        raise FleetEvidenceError("packet architecture disagrees with report")
    if manifest.get("tested_commit") != summary["source_commit"]:
        raise FleetEvidenceError("packet commit disagrees with report")
    if manifest.get("validation") != summary:
        raise FleetEvidenceError("packet validation summary is stale")
    return summary


def compare_backend_reports(
    left: Mapping[str, Any], right: Mapping[str, Any], *,
    fixtures: Mapping[str, Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    """Directly compare common fixture outputs from two exact targets."""
    corpus = dict(fixtures) if fixtures is not None else load_fixture_corpus()
    left_summary = validate_backend_report(left, fixtures=corpus)
    right_summary = validate_backend_report(right, fixtures=corpus)
    left_identity = (left_summary["target"], left_summary["architecture"])
    right_identity = (right_summary["target"], right_summary["architecture"])
    if left_identity == right_identity:
        raise FleetEvidenceError(
            "differential comparison requires distinct target/architecture identities"
        )
    left_rows = {row["fixture_id"]: row for row in left["fixtures"]}
    right_rows = {row["fixture_id"]: row for row in right["fixtures"]}
    common = sorted(set(left_rows) & set(right_rows))
    if not common:
        raise FleetEvidenceError("reports have no common differential fixture")
    maximum_error = 0.0
    for fixture_id in common:
        fixture = corpus[fixture_id]
        policy = fixture["tolerance"]
        lhs = _flatten_numbers(left_rows[fixture_id]["actual"], f"left.{fixture_id}")
        rhs = _flatten_numbers(right_rows[fixture_id]["actual"], f"right.{fixture_id}")
        passed, max_abs, _ = _numerically_equal(
            lhs, rhs, atol=float(policy["atol"]), rtol=float(policy["rtol"]),
            equal_nan=bool(policy.get("equal_nan", False)),
        )
        if not passed:
            raise FleetEvidenceError(
                f"cross-backend fixture {fixture_id} disagrees (max_abs={max_abs})"
            )
        maximum_error = max(maximum_error, max_abs)
    return {
        "left_target": left_summary["target"],
        "left_architecture": left_summary["architecture"],
        "right_target": right_summary["target"],
        "right_architecture": right_summary["architecture"],
        "common_fixtures": len(common),
        "maximum_absolute_error": maximum_error,
    }


def discover_packets(
    root: Path = EVIDENCE_ROOT,
) -> dict[tuple[str, str], tuple[Path, dict[str, Any]]]:
    packets: dict[tuple[str, str], tuple[Path, dict[str, Any]]] = {}
    if not root.exists():
        return packets
    for manifest in sorted(root.glob("*/*/manifest.json")):
        packet_dir = manifest.parent
        summary = validate_packet(packet_dir)
        key = (str(summary["target"]), str(summary["architecture"]))
        if key in packets:
            raise FleetEvidenceError(
                f"multiple active fleet packets for {key[0]} architecture {key[1]}"
            )
        packets[key] = (packet_dir, summary)
    return packets


def fleet_dashboard_rows(root: Path = EVIDENCE_ROOT) -> tuple[FleetDashboardRow, ...]:
    packets = discover_packets(root)
    rows: list[FleetDashboardRow] = []
    for registration in FLEET_REGISTRATIONS:
        packet = packets.get((registration.target, registration.architecture))
        if packet is None:
            families = registration.families or ("-",)
            for family in families:
                rows.append(FleetDashboardRow(
                    registration.target, registration.architecture,
                    registration.backend, family,
                    registration.terminal, "", "", registration.reason,
                ))
            continue
        packet_dir, summary = packet
        report = _load_json(packet_dir / "report.json")
        packet_families = set(report["scope"])
        undeclared = packet_families - set(registration.families)
        if undeclared:
            raise FleetEvidenceError(
                f"{registration.target} packet claims undeclared families: "
                + ", ".join(sorted(undeclared))
            )
        for family in registration.families:
            present = family in packet_families
            rows.append(FleetDashboardRow(
                registration.target, registration.architecture,
                registration.backend, family,
                "release_ready" if present else "packet_pending",
                packet_dir.relative_to(_REPO_ROOT).as_posix() if present else "",
                str(summary["source_commit"]) if present else "",
                "Validated hash-sealed packet." if present else "Family absent from active packet.",
            ))
    return tuple(rows)


FLEET_CSV_COLUMNS = (
    "schema", "target", "architecture", "backend", "family", "state", "packet",
    "source_commit", "reason",
)


def render_fleet_csv(rows: Iterable[FleetDashboardRow] | None = None) -> str:
    import csv
    import io

    buffer = io.StringIO()
    writer = csv.writer(buffer, lineterminator="\n")
    writer.writerow(FLEET_CSV_COLUMNS)
    for row in rows if rows is not None else fleet_dashboard_rows():
        writer.writerow((
            MANIFEST_SCHEMA, row.target, row.architecture, row.backend,
            row.family, row.state,
            row.packet, row.source_commit, row.reason,
        ))
    return buffer.getvalue()


def render_fleet_markdown(rows: Iterable[FleetDashboardRow] | None = None) -> str:
    values = tuple(rows if rows is not None else fleet_dashboard_rows())
    lines = [
        "# E2E fleet release evidence",
        "",
        "**Generated from the E2E-SPINE-3 packet registry and validated sealed packets. Do not hand-edit.**",
        "",
        "`release_ready` is family-granular. It does not promote a whole target or transfer exact-device evidence.",
        "",
        "| Target | Architecture | Backend | Family | State | Tested commit | Packet |",
        "|---|---|---|---|---|---|---|",
    ]
    for row in values:
        packet = f"`{row.packet}`" if row.packet else "-"
        commit = f"`{row.source_commit[:12]}`" if row.source_commit else "-"
        lines.append(
            f"| `{row.target}` | `{row.architecture}` | `{row.backend}` | `{row.family}` | "
            f"`{row.state}` | {commit} | {packet} |"
        )
    lines.extend(("", "The CSV companion retains the full reason and commit.", ""))
    return "\n".join(lines)


def _main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Validate and seal E2E fleet packets")
    subparsers = parser.add_subparsers(dest="command", required=True)
    for command in ("validate", "seal"):
        sub = subparsers.add_parser(command)
        sub.add_argument("packet", type=Path)
    subparsers.add_parser("check-fixtures")
    args = parser.parse_args(argv)
    if args.command == "validate":
        print(json.dumps(validate_packet(args.packet), indent=2, sort_keys=True))
    elif args.command == "seal":
        print(json.dumps(seal_packet(args.packet), indent=2, sort_keys=True))
    else:
        print(json.dumps({"fixtures": len(load_fixture_corpus())}, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(_main())
