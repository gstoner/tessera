"""Evidence-gated selection between Apple GPU execution routes.

The Apple backend has several valid execution lanes (MPSGraph, handwritten
MSL, Metal 4 cooperative tensors, and packaged MTL4 ML subgraphs).  A route is
not promoted merely because it is available: a characterization record must
show that it ran natively, matched its oracle, and beats the incumbent for the
same operation, shape, dtype, and device.

This module is deliberately runtime-free.  Benchmark drivers write its small
JSON schema; the JIT may read a selected report without importing Metal or
authoring packages during decoration.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from functools import lru_cache
import hashlib
import json
import os
import platform
from pathlib import Path
import re
import statistics
import subprocess
from typing import Any, Iterable, Mapping, Sequence


ROUTE_REPORT_SCHEMA_VERSION = 1
STABLE_ROUTE_LEDGER_SCHEMA_VERSION = 1
STRICT_ROUTE_LEDGER_SCHEMA = "tessera.apple.route-ledger.v2"
PACKAGE_ROUTE = "package"

_DEFAULT_STRICT_LEDGER = (
    Path(__file__).resolve().parents[3]
    / "benchmarks/baselines/apple_strict_route_ledger.json"
)

@lru_cache(maxsize=1)
def live_apple_device_tag() -> str:
    try:
        from .apple_target import probe_apple_runtime_limits
        limits = probe_apple_runtime_limits()
    except Exception:
        limits = None
    family = limits.apple_gpu_family if limits is not None else -1
    return (f"apple{family - 1000}" if 1001 <= family <= 1099
            else "apple_silicon_metal_unknown_family")


def _command_text(*args: str) -> str:
    try:
        return subprocess.run(
            args, check=True, capture_output=True, text=True, timeout=3,
        ).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return ""


def _runtime_source_fingerprint() -> str:
    source = (
        Path(__file__).resolve().parents[3]
        / "src/compiler/codegen/Tessera_Apple_Backend/runtime/apple_gpu_runtime.mm"
    )
    try:
        content = source.read_bytes()
    except OSError:
        return "unavailable"
    return f"sha256:{hashlib.sha256(content).hexdigest()}"


def _configured_llvm_dir() -> str:
    configured = os.environ.get("LLVM_DIR")
    if configured:
        return configured
    cache = Path(__file__).resolve().parents[3] / "build-apple/CMakeCache.txt"
    try:
        text = cache.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return ""
    match = re.search(r"^LLVM_DIR:[^=]+=(.+)$", text, re.MULTILINE)
    return match.group(1).strip() if match else ""


@dataclass(frozen=True)
class AppleRouteContext:
    """Live identity required before retained Apple evidence may select."""

    device: str
    physical_device: str
    os_version: str
    sdk_version: str
    compiler_fingerprint: str
    runtime_fingerprint: str

    def as_mapping(self) -> dict[str, str]:
        return {
            "device": self.device,
            "physical_device": self.physical_device,
            "os_version": self.os_version,
            "sdk_version": self.sdk_version,
            "compiler_fingerprint": self.compiler_fingerprint,
            "runtime_fingerprint": self.runtime_fingerprint,
        }


@lru_cache(maxsize=1)
def live_apple_route_context() -> AppleRouteContext:
    """Fingerprint the live Apple device, OS/SDK, compiler, and runtime source.

    Environment overrides are intentional release-lane inputs: a runner may
    supply a more exact physical-device identifier or compiler artifact digest
    than the portable probes can discover.
    """
    physical = os.environ.get("TESSERA_APPLE_PHYSICAL_DEVICE")
    if not physical:
        physical = _command_text("sysctl", "-n", "machdep.cpu.brand_string")
    sdk = os.environ.get("TESSERA_APPLE_SDK_VERSION") or _command_text(
        "xcrun", "--sdk", "macosx", "--show-sdk-version")
    compiler = os.environ.get("TESSERA_APPLE_COMPILER_FINGERPRINT")
    if not compiler:
        compiler_text = "\n".join(filter(None, (
            _command_text("clang", "--version"),
            _configured_llvm_dir(),
        )))
        compiler = (
            f"sha256:{hashlib.sha256(compiler_text.encode()).hexdigest()}"
            if compiler_text else "unavailable"
        )
    return AppleRouteContext(
        device=live_apple_device_tag(),
        physical_device=physical or "unavailable",
        os_version=platform.mac_ver()[0] or platform.platform(),
        sdk_version=sdk or "unavailable",
        compiler_fingerprint=compiler,
        runtime_fingerprint=(
            os.environ.get("TESSERA_APPLE_RUNTIME_FINGERPRINT")
            or _runtime_source_fingerprint()
        ),
    )


@dataclass(frozen=True)
class StrictRouteLedger:
    routes: Mapping[tuple[str, str, str, str, str], str]
    citations: Mapping[tuple[str, str, str, str, str], str]
    rejected: tuple[str, ...]


@dataclass(frozen=True)
class ProductionRouteDecision:
    route: str
    incumbent_route: str
    selected_from_ledger: bool
    citation: str | None
    rejected_evidence: tuple[str, ...]


def _parse_utc(value: Any) -> datetime | None:
    if not isinstance(value, str):
        return None
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
    return parsed if parsed.tzinfo is not None else parsed.replace(tzinfo=timezone.utc)


def load_strict_route_ledger(
    path: str | Path,
    *,
    context: AppleRouteContext | None = None,
    now: datetime | None = None,
) -> StrictRouteLedger:
    """Admit only fresh, exact-context, native, domain-specific decisions."""
    rejected: list[str] = []
    try:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        return StrictRouteLedger({}, {}, (f"ledger_unreadable:{type(exc).__name__}",))
    if payload.get("schema") != STRICT_ROUTE_LEDGER_SCHEMA:
        return StrictRouteLedger({}, {}, ("schema_mismatch",))
    ctx = context or live_apple_route_context()
    retained = payload.get("context")
    if not isinstance(retained, Mapping):
        return StrictRouteLedger({}, {}, ("missing_context",))
    for field, expected in ctx.as_mapping().items():
        actual = retained.get(field)
        if actual != expected:
            rejected.append(f"context_mismatch:{field}")
    measured_at = _parse_utc(payload.get("measured_at"))
    expires_at = _parse_utc(payload.get("expires_at"))
    current = now or datetime.now(timezone.utc)
    if measured_at is None or measured_at > current:
        rejected.append("invalid_measured_at")
    if expires_at is None or current > expires_at:
        rejected.append("stale_evidence")
    if rejected:
        return StrictRouteLedger({}, {}, tuple(rejected))
    decisions = payload.get("decisions")
    if not isinstance(decisions, list):
        return StrictRouteLedger({}, {}, ("missing_decisions",))
    routes: dict[tuple[str, str, str, str, str], str] = {}
    citations: dict[tuple[str, str, str, str, str], str] = {}
    for index, row in enumerate(decisions):
        prefix = f"decision[{index}]"
        if not isinstance(row, Mapping):
            rejected.append(f"{prefix}:not_mapping")
            continue
        try:
            domain = str(row["timing_domain"])
            key = (
                str(row["device"]), str(row["op"]), str(row["shape"]),
                str(row["dtype"]), domain,
            )
            selected = str(row["selected_route"])
        except KeyError as exc:
            rejected.append(f"{prefix}:missing:{exc.args[0]}")
            continue
        if domain not in {"device", "end_to_end"}:
            rejected.append(f"{prefix}:wrong_timing_domain")
            continue
        if key[0] != ctx.device:
            rejected.append(f"{prefix}:wrong_device")
            continue
        if row.get("status") not in {"promote_candidate", "retain_incumbent"}:
            rejected.append(f"{prefix}:ineligible_status")
            continue
        evidence = row.get("selected_evidence")
        if not isinstance(evidence, Mapping):
            rejected.append(f"{prefix}:missing_selected_evidence")
            continue
        if evidence.get("provenance") != "native_gpu":
            rejected.append(f"{prefix}:reference_provenance")
            continue
        if evidence.get("correctness") is not True:
            rejected.append(f"{prefix}:correctness_unproven")
            continue
        if evidence.get("timing_domain") != domain:
            rejected.append(f"{prefix}:wrong_evidence_domain")
            continue
        if evidence.get("device") != ctx.device:
            rejected.append(f"{prefix}:wrong_evidence_device")
            continue
        if key in routes:
            rejected.append(f"{prefix}:duplicate_key")
            continue
        routes[key] = selected
        citations[key] = f"{Path(path)}#decision[{index}]"
    return StrictRouteLedger(routes, citations, tuple(rejected))


@lru_cache(maxsize=16)
def _cached_strict_route_ledger(
    path: str, context: AppleRouteContext, mtime_ns: int, utc_hour: int,
) -> StrictRouteLedger:
    del mtime_ns, utc_hour  # cache-key invalidators, intentionally not payload fields
    return load_strict_route_ledger(path, context=context)


def production_route_decision(
    *, op: str, shape: str, dtype: str, incumbent_route: str,
    device: str | None = None, timing_domain: str = "end_to_end",
    ledger_path: str | Path | None = None,
    context: AppleRouteContext | None = None,
    now: datetime | None = None,
) -> ProductionRouteDecision:
    """Resolve one route and retain an auditable ledger-row citation."""
    if timing_domain not in {"device", "end_to_end"}:
        raise ValueError(f"unsupported timing domain: {timing_domain!r}")
    ctx = context or live_apple_route_context()
    tag = device or ctx.device
    path = Path(ledger_path or os.environ.get("TESSERA_APPLE_ROUTE_LEDGER")
                or _DEFAULT_STRICT_LEDGER)
    try:
        mtime_ns = path.stat().st_mtime_ns
    except OSError:
        mtime_ns = -1
    if now is None:
        utc_hour = int(datetime.now(timezone.utc).timestamp() // 3600)
        ledger = _cached_strict_route_ledger(
            str(path), ctx, mtime_ns, utc_hour)
    else:
        ledger = load_strict_route_ledger(path, context=ctx, now=now)
    key = (tag, op, shape, dtype, timing_domain)
    route = ledger.routes.get(key, incumbent_route)
    return ProductionRouteDecision(
        route=route,
        incumbent_route=incumbent_route,
        selected_from_ledger=key in ledger.routes,
        citation=ledger.citations.get(key),
        rejected_evidence=ledger.rejected,
    )


def production_route_for(*, op: str, shape: str, dtype: str,
                         incumbent_route: str, device: str | None = None,
                         timing_domain: str = "end_to_end",
                         ledger_path: str | Path | None = None,
                         context: AppleRouteContext | None = None,
                         now: datetime | None = None) -> str:
    """Return an admitted exact-device ledger decision or the incumbent."""
    return production_route_decision(
        op=op, shape=shape, dtype=dtype, incumbent_route=incumbent_route,
        device=device, timing_domain=timing_domain, ledger_path=ledger_path,
        context=context, now=now,
    ).route


@dataclass(frozen=True)
class AppleRouteMeasurement:
    """One warm, numerically checked route measurement."""

    op: str
    shape: str
    dtype: str
    device: str
    route: str
    latency_ms: float
    native_dispatched: bool
    numerically_validated: bool

    @classmethod
    def from_mapping(cls, row: Mapping[str, Any]) -> "AppleRouteMeasurement | None":
        try:
            latency = float(row["latency_ms"])
            measurement = cls(
                op=str(row["op"]),
                shape=str(row["shape"]),
                dtype=str(row["dtype"]),
                device=str(row["device"]),
                route=str(row.get("route", row.get("mode", ""))),
                latency_ms=latency,
                native_dispatched=bool(row["native_dispatched"]),
                numerically_validated=bool(row["numerically_validated"]),
            )
        except (KeyError, TypeError, ValueError):
            return None
        if not measurement.route or measurement.latency_ms <= 0:
            return None
        return measurement


def load_route_measurements(path: str | Path) -> tuple[AppleRouteMeasurement, ...]:
    """Load only complete, current-schema measurements from ``path``.

    Old benchmark snapshots intentionally do not drive compilation: they lack
    the native-dispatch and numerical-proof fields needed to make a promotion
    decision honestly.
    """
    try:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return ()
    if payload.get("schema_version") != ROUTE_REPORT_SCHEMA_VERSION:
        return ()
    runs = payload.get("runs")
    if not isinstance(runs, list):
        return ()
    return tuple(
        measurement
        for row in runs
        if isinstance(row, Mapping)
        for measurement in (AppleRouteMeasurement.from_mapping(row),)
        if measurement is not None
    )


def select_route(
    measurements: Iterable[AppleRouteMeasurement],
    *,
    op: str,
    shape: str,
    dtype: str,
    device: str,
    incumbent_route: str,
) -> str | None:
    """Return the fastest proven route or ``None`` when evidence is incomplete.

    Both the incumbent and winner must be native and numerically validated.
    This avoids promoting a route on a host fallback or comparing a package
    against an unrelated shape/device result.
    """
    matching = [
        row for row in measurements
        if (row.op, row.shape, row.dtype, row.device) == (op, shape, dtype, device)
        and row.native_dispatched and row.numerically_validated
    ]
    if not any(row.route == incumbent_route for row in matching):
        return None
    if not matching:
        return None
    return min(matching, key=lambda row: row.latency_ms).route


def package_route_selected(
    report_path: str | Path | None,
    *,
    op: str,
    shape: str,
    dtype: str = "f32",
    device: str = "apple_silicon_metal",
    incumbent_route: str = "live",
) -> bool:
    """Whether a report promotes a package route for this exact invocation."""
    if not report_path:
        return False
    return select_route(
        load_route_measurements(report_path), op=op, shape=shape, dtype=dtype,
        device=device, incumbent_route=incumbent_route,
    ) == PACKAGE_ROUTE


def _row_time_ns(row: Mapping[str, Any], domain: str) -> int | None:
    telemetry = row.get("telemetry")
    if not isinstance(telemetry, Mapping):
        return None
    field = {
        "end_to_end": "end_to_end_median_ns",
        "device": "device_time_median_ns",
    }.get(domain)
    if field is None:
        raise ValueError(f"unsupported timing domain: {domain!r}")
    value = telemetry.get(field)
    if value is None:
        return None
    try:
        value = int(value)
    except (TypeError, ValueError):
        return None
    return value if value > 0 else None


def _row_trial_times_ns(row: Mapping[str, Any], domain: str) -> list[int] | None:
    telemetry = row.get("telemetry")
    if not isinstance(telemetry, Mapping):
        return None
    field = {
        "end_to_end": "paired_trial_end_to_end_medians_ns",
        "device": "paired_trial_device_medians_ns",
    }.get(domain)
    values = telemetry.get(field) if field else None
    if not isinstance(values, list) or len(values) < 3:
        return None
    try:
        parsed = [int(value) for value in values]
    except (TypeError, ValueError):
        return None
    return parsed if all(value > 0 for value in parsed) else None


def _route_evidence(rows: Sequence[Mapping[str, Any] | None],
                    domain: str, max_run_drift: float) -> dict[str, Any]:
    complete = all(row is not None for row in rows)
    proof = complete and all(
        bool(row.get("native_dispatched")) and bool(row.get("numerically_validated"))
        for row in rows if row is not None)
    repeated = complete and all(
        isinstance(row.get("reps"), int) and int(row["reps"]) >= 2
        for row in rows if row is not None)
    resource_records = [
        row["telemetry"].get("resources")
        if row is not None and isinstance(row.get("telemetry"), Mapping)
        else None
        for row in rows
    ]
    resources = complete and all(
        isinstance(record, Mapping) for record in resource_records)
    times = [_row_time_ns(row, domain) if row is not None else None for row in rows]
    timed = all(value is not None for value in times)
    numeric_times = [int(value) for value in times if value is not None]
    drift = ((max(numeric_times) - min(numeric_times)) / min(numeric_times)
             if len(numeric_times) == len(rows) and numeric_times else None)
    stable = drift is not None and drift <= max_run_drift
    trial_times = [
        _row_trial_times_ns(row, domain) if row is not None else None
        for row in rows]
    paired = all(values is not None for values in trial_times)
    timing_coverage = []
    for row in rows:
        if domain == "end_to_end":
            timing_coverage.append(1.0)
        elif row is not None and isinstance(row.get("telemetry"), Mapping):
            coverage = row["telemetry"].get("device_time_coverage")
            if coverage is None:
                samples = row["telemetry"].get("device_time_samples")
                reps = row.get("reps")
                if isinstance(samples, int) and isinstance(reps, int) and reps > 0:
                    coverage = float(samples) / float(reps)
                elif (_row_time_ns(row, domain) is not None
                      and _row_trial_times_ns(row, domain) is not None):
                    # Schema-v1 reports written before the explicit coverage
                    # field retained only complete paired device medians.
                    coverage = 1.0
                else:
                    coverage = 0.0
            timing_coverage.append(float(coverage))
        else:
            timing_coverage.append(0.0)
    coverage_complete = all(value >= 0.9 for value in timing_coverage)
    return {
        "present_in_all_runs": complete,
        "placement_and_numerical_proof": proof,
        "repeated_measurement": repeated,
        "resource_evidence_retained": resources,
        "resource_records": resource_records,
        "timing_sources": [
            row["telemetry"].get("timing_source")
            if row is not None and isinstance(row.get("telemetry"), Mapping)
            else None for row in rows],
        "counter_sampling_supported": [
            row["telemetry"].get("counter_sampling_supported")
            if row is not None and isinstance(row.get("telemetry"), Mapping)
            else None for row in rows],
        "counter_timestamp_deltas": [
            row["telemetry"].get("counter_timestamp_delta_median")
            if row is not None and isinstance(row.get("telemetry"), Mapping)
            else None for row in rows],
        "times_ns": times,
        "run_drift_fraction": drift,
        "absolute_time_stable": stable,
        "paired_trial_times_ns": trial_times,
        "paired_measurement": paired,
        "timing_coverage": timing_coverage,
        "eligible": (proof and repeated and resources and timed and paired
                     and coverage_complete),
    }


def aggregate_stable_route_reports(
    reports: Sequence[Mapping[str, Any]], *,
    incumbent_routes: Mapping[str, str] | None = None,
    min_speedup: float = 0.05,
    max_run_drift: float = 0.15,
    min_paired_win_fraction: float = 0.75,
    max_speedup_spread: float = 0.05,
) -> dict[str, Any]:
    """Build an evidence ledger from two or more independent warm reports.

    A candidate is promoted only when it and the incumbent have exact matching
    rows in every report, retain placement/numerical/resource evidence, are
    collected in paired interleaved trials. The candidate must win at least
    ``min_paired_win_fraction`` of trials, clear ``min_speedup`` in every
    independent run, and keep its cross-run median speedup within
    ``max_speedup_spread``. Absolute clock drift is retained diagnostically but
    cannot invalidate an otherwise stable paired comparison.
    """
    if len(reports) < 2:
        raise ValueError("stable route selection requires at least two reports")
    if not 0.0 <= min_speedup < 1.0:
        raise ValueError("min_speedup must be in [0, 1)")
    if max_run_drift < 0.0:
        raise ValueError("max_run_drift must be non-negative")
    if not 0.5 <= min_paired_win_fraction <= 1.0:
        raise ValueError("min_paired_win_fraction must be in [0.5, 1]")
    if max_speedup_spread < 0.0:
        raise ValueError("max_speedup_spread must be non-negative")
    for report in reports:
        if report.get("schema_version") != ROUTE_REPORT_SCHEMA_VERSION:
            raise ValueError("all reports must use the current route schema")
        if not isinstance(report.get("runs"), list):
            raise ValueError("each report must contain a runs list")

    # These are the current production routes, not benchmark-preferred labels.
    incumbents = {"matmul": "mps", "softmax": "msl"}
    if incumbent_routes:
        incumbents.update(incumbent_routes)

    indexes: list[dict[tuple[str, str, str, str, str], Mapping[str, Any]]] = []
    comparison_keys: set[tuple[str, str, str, str]] = set()
    routes_by_key: dict[tuple[str, str, str, str], set[str]] = {}
    for report in reports:
        index: dict[tuple[str, str, str, str, str], Mapping[str, Any]] = {}
        for row in report["runs"]:
            if not isinstance(row, Mapping):
                continue
            try:
                base: tuple[str, str, str, str] = (
                    str(row["op"]),
                    str(row["shape"]),
                    str(row["dtype"]),
                    str(row["device"]),
                )
                route = str(row["route"])
            except KeyError:
                continue
            full: tuple[str, str, str, str, str] = (*base, route)
            if full in index:
                raise ValueError(f"duplicate route row in one report: {full!r}")
            index[full] = row
            comparison_keys.add(base)
            routes_by_key.setdefault(base, set()).add(route)
        indexes.append(index)

    decisions: list[dict[str, Any]] = []
    for base in sorted(comparison_keys):
        op, shape, dtype, device = base
        incumbent = incumbents.get(op)
        if incumbent is None:
            continue
        route_rows = {
            route: [index.get((*base, route)) for index in indexes]
            for route in sorted(routes_by_key[base])
        }
        for domain in ("end_to_end", "device"):
            evidence = {
                route: _route_evidence(rows, domain, max_run_drift)
                for route, rows in route_rows.items()
            }
            incumbent_evidence = evidence.get(incumbent)
            status = "insufficient_evidence"
            selected: str | None = None
            reason = "incumbent paired evidence is incomplete"
            winners: list[tuple[float, str]] = []
            if incumbent_evidence and incumbent_evidence["eligible"]:
                selected = incumbent
                status = "retain_incumbent"
                reason = "no candidate met the per-run stable-win threshold"
                incumbent_times = incumbent_evidence["times_ns"]
                for route, route_evidence in evidence.items():
                    if route == incumbent or not route_evidence["eligible"]:
                        continue
                    speedups = [
                        (inc_ns - candidate_ns) / inc_ns
                        for inc_ns, candidate_ns in zip(
                            incumbent_times, route_evidence["times_ns"])
                    ]
                    route_evidence["speedups_vs_incumbent"] = speedups
                    paired_speedups: list[list[float]] = []
                    for incumbent_trials, candidate_trials in zip(
                            incumbent_evidence["paired_trial_times_ns"],
                            route_evidence["paired_trial_times_ns"]):
                        if len(incumbent_trials) != len(candidate_trials):
                            paired_speedups = []
                            break
                        paired_speedups.append([
                            (inc_ns - candidate_ns) / inc_ns
                            for inc_ns, candidate_ns in zip(
                                incumbent_trials, candidate_trials)])
                    median_speedups = [
                        statistics.median(values) for values in paired_speedups]
                    win_fractions = [
                        sum(value > 0.0 for value in values) / len(values)
                        for values in paired_speedups]
                    spread = ((max(median_speedups) - min(median_speedups))
                              if median_speedups else None)
                    route_evidence["paired_speedups_vs_incumbent"] = paired_speedups
                    route_evidence["paired_median_speedups"] = median_speedups
                    route_evidence["paired_win_fractions"] = win_fractions
                    route_evidence["cross_run_speedup_spread"] = spread
                    if (median_speedups
                            and all(speedup >= min_speedup
                                    for speedup in median_speedups)
                            and all(fraction >= min_paired_win_fraction
                                    for fraction in win_fractions)
                            and spread is not None
                            and spread <= max_speedup_spread):
                        winners.append((min(median_speedups), route))
                if winners:
                    _, selected = max(winners)
                    status = "promote_candidate"
                    reason = "candidate met paired stable-win gates in every run"
            decisions.append({
                "op": op,
                "shape": shape,
                "dtype": dtype,
                "device": device,
                "timing_domain": domain,
                "incumbent_route": incumbent,
                "selected_route": selected,
                "status": status,
                "reason": reason,
                "route_evidence": evidence,
            })
    return {
        "schema_version": STABLE_ROUTE_LEDGER_SCHEMA_VERSION,
        "source_report_schema_version": ROUTE_REPORT_SCHEMA_VERSION,
        "report_count": len(reports),
        "promotion_rules": {
            "minimum_speedup_fraction_each_run": min_speedup,
            "maximum_cross_run_drift_fraction": max_run_drift,
            "absolute_time_drift_is_diagnostic_only": True,
            "minimum_paired_win_fraction_each_run": min_paired_win_fraction,
            "maximum_cross_run_speedup_spread": max_speedup_spread,
            "requires_native_dispatch": True,
            "requires_numerical_validation": True,
            "requires_repeated_measurement": True,
            "requires_interleaved_paired_trials": True,
            "requires_resource_evidence": True,
        },
        "decisions": decisions,
    }


__all__ = [
    "AppleRouteContext",
    "AppleRouteMeasurement",
    "PACKAGE_ROUTE",
    "ProductionRouteDecision",
    "ROUTE_REPORT_SCHEMA_VERSION",
    "STABLE_ROUTE_LEDGER_SCHEMA_VERSION",
    "STRICT_ROUTE_LEDGER_SCHEMA",
    "StrictRouteLedger",
    "aggregate_stable_route_reports",
    "live_apple_route_context",
    "load_route_measurements",
    "load_strict_route_ledger",
    "package_route_selected",
    "production_route_for",
    "production_route_decision",
    "live_apple_device_tag",
    "select_route",
]
