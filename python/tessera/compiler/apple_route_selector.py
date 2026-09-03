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
from datetime import datetime, timedelta, timezone
from functools import lru_cache
import hashlib
import json
import math
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
STRICT_RUNTIME_ROUTE_SCOPE = "runtime_route"
STRICT_PACKAGE_SUBGRAPH_SCOPE = "package_subgraph"
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


@dataclass(frozen=True)
class LegacyRouteLedgerInventory:
    """One legacy Apple route ledger that must be remeasured, not promoted.

    The inventory intentionally contains no route decision.  Schema-v1 reports
    do not carry the strict context and scope envelope, so using their selected
    route here would create a production-evidence bypass.
    """

    path: Path
    schema: str | None
    decision_count: int
    migration_state: str
    strict_ledger_path: Path | None = None


def _parse_utc(value: Any) -> datetime | None:
    if not isinstance(value, str):
        return None
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
    return parsed if parsed.tzinfo is not None else parsed.replace(tzinfo=timezone.utc)


# One-sided Student-t 95% multipliers, indexed by degrees of freedom (n - 1).
# Table rather than SciPy: this module is imported on the runtime path and
# Decision #23 keeps the shipped package free of heavyweight dependencies.
_STUDENT_T_ONE_SIDED_95 = {
    1: 6.314, 2: 2.920, 3: 2.353, 4: 2.132, 5: 2.015, 6: 1.943, 7: 1.895,
    8: 1.860, 9: 1.833, 10: 1.812, 11: 1.796, 12: 1.782, 13: 1.771,
    14: 1.761, 15: 1.753, 16: 1.746, 17: 1.740, 18: 1.734, 19: 1.729,
    20: 1.725, 21: 1.721, 22: 1.717, 23: 1.714, 24: 1.711, 25: 1.708,
    26: 1.706, 27: 1.703, 28: 1.701, 29: 1.699, 30: 1.697,
}
_STUDENT_T_ONE_SIDED_95_LARGE_SAMPLE = 1.645

SPEEDUP_CONFIDENCE_LEVEL = 0.95


def speedup_confidence_interval(
    per_run_speedups: Sequence[float],
) -> tuple[float, float] | None:
    """95% interval for the mean per-run median speedup, or ``None`` under two runs.

    The bounds answer the two questions a route decision actually asks: is the
    win big enough to be worth taking (lower bound), and can a qualifying win
    be ruled out at all (upper bound)? A row where the interval straddles the
    threshold is inconclusive -- see ``speedup_lower_confidence_bound``.
    """
    values = [float(value) for value in per_run_speedups]
    if len(values) < 2:
        return None
    multiplier = _STUDENT_T_ONE_SIDED_95.get(
        len(values) - 1, _STUDENT_T_ONE_SIDED_95_LARGE_SAMPLE)
    margin = multiplier * statistics.stdev(values) / math.sqrt(len(values))
    mean = statistics.mean(values)
    return (mean - margin, mean + margin)


def speedup_lower_confidence_bound(per_run_speedups: Sequence[float]) -> float | None:
    """One-sided 95% lower bound on the mean of per-run median speedups.

    **This replaces a range, and the difference is the whole point.** The gate
    it supersedes capped ``max - min`` of the per-run speedups at a fixed
    0.05. A range is monotonically non-decreasing in the number of runs, so
    that gate got *harder* the more evidence you collected: measured on this
    M1 Max, a route that is 40% faster and wins 144 of 144 paired trials
    promoted 69% of the time at two runs and 7% at eight. Adding runs -- the
    obvious response to an irreproducible decision -- made a true winner less
    promotable, and re-recording until it passed was selection on noise.

    A lower confidence bound converges instead: more runs shrink the interval
    toward the true mean, so evidence can only help a route that is genuinely
    faster, and can never rescue one whose speedup is indistinguishable from
    noise. Returns ``None`` when fewer than two runs make dispersion
    unmeasurable -- unprovable is not the permissive answer (Decision #30).
    """
    interval = speedup_confidence_interval(per_run_speedups)
    return None if interval is None else interval[0]


def promotion_rule_violations(
    row: Mapping[str, Any], rules: Mapping[str, Any],
    *, source_report_count: int | None = None,
) -> list[str]:
    """Which of the ledger's own ``promotion_rules`` a promoted row breaks.

    **The rules were a declaration with no consumer.**
    ``aggregate_stable_route_reports`` computes the thresholds, applies them,
    and writes them into every sealed ledger; ``seal_strict_route_ledger``
    copies them forward for audit; and until now nothing read them back. Twelve
    committed strict ledgers carry this block, and a ``promote_candidate`` row
    was admitted on the strength of its ``status`` string alone -- so a row
    naming a route that lost every paired trial would have been served, as long
    as its provenance fields were right. That is Decision #29 exactly: a
    contract that reads as closed in review and carries nothing.

    Re-deriving the verdict from the retained evidence is what makes the
    ledger checkable by someone who did not run the benchmark. It is
    deliberately a *re-derivation*, not a recomputation from raw times: the
    aggregate's own numbers are the thing under audit.

    Returns an empty list for a row that is not a promotion.
    """
    if row.get("status") != "promote_candidate":
        return []
    selected = row.get("selected_route")
    evidence = row.get("route_evidence")
    if not isinstance(evidence, Mapping) or not isinstance(selected, str):
        return ["missing_route_evidence"]
    chosen = evidence.get(selected)
    if not isinstance(chosen, Mapping):
        return [f"missing_evidence_for:{selected}"]

    violations: list[str] = []

    def _threshold(name: str) -> float | None:
        value = rules.get(name)
        return float(value) if isinstance(value, (int, float)) else None

    min_speedup = _threshold("minimum_speedup_fraction_each_run")
    min_win = _threshold("minimum_paired_win_fraction_each_run")
    min_pooled_win = _threshold("minimum_pooled_paired_win_fraction")
    max_spread = _threshold("maximum_cross_run_speedup_spread")
    min_bound = _threshold("minimum_speedup_lower_confidence_bound")
    min_runs = _threshold("minimum_promotion_runs")
    # A ledger is held to the stability rule IT was sealed under. Ledgers
    # sealed before the confidence bound carry only the cross-run range cap;
    # ledgers sealed after it mark that range diagnostic and carry a bound.
    # Checking whichever one the ledger declares is what lets this verifier
    # audit both without either grandfathering old rows or retroactively
    # inventing a threshold the sealer never applied.
    spread_is_diagnostic = rules.get(
        "cross_run_speedup_spread_is_diagnostic_only") is True
    # A missing threshold is not a pass. Without it there is nothing to hold
    # the promotion to, and the honest verdict is "unverifiable", not "fine".
    if min_speedup is None or min_win is None:
        violations.append("promotion_rules_incomplete")
    if min_bound is None and (max_spread is None or spread_is_diagnostic):
        violations.append("no_stability_rule_declared")

    medians = chosen.get("paired_median_speedups")
    fractions = chosen.get("paired_win_fractions")

    # The rules are spelled `*_each_run`, so a per-run metric list that is
    # SHORTER than the run count has not been checked for every run -- and
    # non-emptiness does not catch that. Review finding on PR #673: a promotion
    # truncated to a single median and win fraction returns no violations
    # today, and truncation makes the row look *better*, because
    # `cross_run_speedup_spread` over one element is 0.0 and clears any cap.
    # Dropping evidence must never improve a verdict.
    #
    # `source_report_count` is the authority when the ledger states it
    # (`seal_strict_route_ledger` writes one report per independent run). When
    # it does not, fall back to the sealer's own floor of two, and require the
    # two lists to agree with each other either way.
    expected_runs = (source_report_count
                     if isinstance(source_report_count, int)
                     and source_report_count >= 2 else None)
    if isinstance(medians, list) and isinstance(fractions, list):
        if len(medians) != len(fractions):
            violations.append(
                f"per_run_metric_length_mismatch:{len(medians)}!={len(fractions)}")
        elif medians:
            required = expected_runs if expected_runs is not None else 2
            if len(medians) < required:
                violations.append(
                    f"per_run_metrics_short:{len(medians)}<{required}")
            elif expected_runs is not None and len(medians) != expected_runs:
                violations.append(
                    f"per_run_metrics_exceed_reports:"
                    f"{len(medians)}!={expected_runs}")

    if not isinstance(medians, list) or not medians:
        violations.append("no_paired_median_speedups")
    elif min_speedup is not None and any(
            not isinstance(v, (int, float)) or v < min_speedup for v in medians):
        violations.append(
            f"speedup_below_minimum:{min(v for v in medians if isinstance(v, (int, float)))!r}"
            if any(isinstance(v, (int, float)) for v in medians)
            else "speedup_below_minimum")

    if not isinstance(fractions, list) or not fractions:
        violations.append("no_paired_win_fractions")
    elif min_win is not None and any(
            not isinstance(v, (int, float)) or v < min_win for v in fractions):
        # Ledgers sealed before pooling carry 0.75 here and this is the whole
        # win rule; ledgers sealed after carry 0.5 and this is the per-run
        # floor beneath the pooled rule checked next.
        violations.append("paired_win_fraction_below_minimum")

    if min_pooled_win is not None:
        pooled = chosen.get("pooled_paired_win_fraction")
        if not isinstance(pooled, (int, float)):
            violations.append("no_pooled_paired_win_fraction")
        elif pooled < min_pooled_win:
            violations.append(f"pooled_paired_win_fraction_below_minimum:{pooled!r}")

    if min_bound is not None:
        # Current rule: the win must survive its own measurement error.
        bound = chosen.get("speedup_lower_confidence_bound")
        if not isinstance(bound, (int, float)):
            violations.append("no_speedup_lower_confidence_bound")
        elif bound < min_bound:
            violations.append(f"speedup_lower_bound_below_minimum:{bound!r}")
        if (min_runs is not None and isinstance(medians, list)
                and len(medians) < int(min_runs)):
            violations.append(
                f"promotion_runs_below_minimum:{len(medians)}<{int(min_runs)}")
    elif max_spread is not None and not spread_is_diagnostic:
        # Superseded rule, still enforced for ledgers sealed under it.
        spread = chosen.get("cross_run_speedup_spread")
        if not isinstance(spread, (int, float)):
            violations.append("no_cross_run_speedup_spread")
        elif spread > max_spread:
            violations.append(f"speedup_spread_above_maximum:{spread!r}")

    # The `requires_*` booleans name evidence that must be PRESENT, not a
    # threshold to clear. Each maps to the field the aggregator set.
    for rule, field in (
        ("requires_native_dispatch", "placement_and_numerical_proof"),
        ("requires_numerical_validation", "placement_and_numerical_proof"),
        ("requires_repeated_measurement", "repeated_measurement"),
        ("requires_interleaved_paired_trials", "paired_measurement"),
        ("requires_resource_evidence", "resource_evidence_retained"),
    ):
        if rules.get(rule) is True and chosen.get(field) is not True:
            violations.append(f"{rule}:unmet")
    return violations


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
    if payload.get("selection_scope") != STRICT_RUNTIME_ROUTE_SCOPE:
        return StrictRouteLedger({}, {}, ("wrong_selection_scope",))
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
    report_count = payload.get("source_report_count")
    report_digests = payload.get("source_report_digests")
    if (not isinstance(report_count, int) or report_count < 2
            or not isinstance(report_digests, list)
            or len(report_digests) != report_count
            or any(not isinstance(digest, str)
                   or re.fullmatch(r"sha256:[0-9a-f]{64}", digest) is None
                   for digest in report_digests)):
        return StrictRouteLedger({}, {}, ("missing_or_invalid_source_reports",))
    decisions = payload.get("decisions")
    if not isinstance(decisions, list):
        return StrictRouteLedger({}, {}, ("missing_decisions",))
    # The ledger carries the thresholds it was sealed under; hold it to them.
    # Without this, `status: "promote_candidate"` was self-certifying -- the
    # loader checked provenance thoroughly and the promotion criteria not at
    # all, so a row naming a route that lost every trial would be served.
    promotion_rules = payload.get("promotion_rules")
    if not isinstance(promotion_rules, Mapping):
        promotion_rules = {}
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
        # `retain_incumbent_unstable_candidate` is admitted on exactly the
        # same terms as `retain_incumbent`: the incumbent serves either way.
        # It is a separate status so the ledger records WHY the candidate was
        # refused -- unreproducible rather than slower -- which a bare
        # `retain_incumbent` could not express.
        if row.get("status") not in {"promote_candidate", "retain_incumbent",
                                     "retain_incumbent_unstable_candidate"}:
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
        violations = promotion_rule_violations(
            row, promotion_rules, source_report_count=report_count)
        if violations:
            rejected.append(f"{prefix}:promotion_rule:{violations[0]}")
            continue
        routes[key] = selected
        citations[key] = f"{Path(path)}#decision[{index}]"
    return StrictRouteLedger(routes, citations, tuple(rejected))


def legacy_route_ledger_inventory(root: str | Path | None = None) -> tuple[LegacyRouteLedgerInventory, ...]:
    """Inventory old Apple route ledgers without treating them as v2 evidence."""
    base = Path(root) if root is not None else _DEFAULT_STRICT_LEDGER.parent
    records: list[LegacyRouteLedgerInventory] = []
    for path in sorted(base.glob("apple*route_ledger.json")):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            records.append(LegacyRouteLedgerInventory(path, None, 0, "unreadable"))
            continue
        if payload.get("schema") == STRICT_ROUTE_LEDGER_SCHEMA:
            continue
        decisions = payload.get("decisions")
        strict_path = path.with_name(
            path.name.replace("_route_ledger.json", "_strict_v2_route_ledger.json"))
        try:
            strict_payload = json.loads(strict_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            strict_payload = None
        strict_valid = (
            isinstance(strict_payload, Mapping)
            and strict_payload.get("schema") == STRICT_ROUTE_LEDGER_SCHEMA
            and strict_payload.get("selection_scope") == STRICT_RUNTIME_ROUTE_SCOPE
            and isinstance(strict_payload.get("source_report_count"), int)
            and strict_payload["source_report_count"] >= 2
            and isinstance(strict_payload.get("source_report_digests"), list)
            and len(strict_payload["source_report_digests"])
                == strict_payload["source_report_count"]
        )
        records.append(LegacyRouteLedgerInventory(
            path=path,
            schema=(str(payload.get("schema_version"))
                    if payload.get("schema_version") is not None else None),
            decision_count=len(decisions) if isinstance(decisions, list) else 0,
            migration_state=("remeasured_strict_v2" if strict_valid
                             else "remeasure_required_strict_v2_context_and_scope"),
            strict_ledger_path=strict_path if strict_valid else None,
        ))
    return tuple(records)


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
    min_promotion_runs: int = 3,
) -> dict[str, Any]:
    """Build an evidence ledger from two or more independent warm reports.

    A candidate is promoted only when it and the incumbent have exact matching
    rows in every report, retain placement/numerical/resource evidence, and are
    collected in paired interleaved trials. The candidate must win at least
    ``min_paired_win_fraction`` of trials and clear ``min_speedup`` in every
    independent run -- and then its *lower 95% confidence bound* across runs
    must also clear ``min_speedup``, over at least ``min_promotion_runs`` runs.

    **The confidence bound replaced a cross-run range cap, because the range
    made promotions irreproducible.** ``max_speedup_spread`` capped
    ``max - min`` of the per-run speedups at 0.05. Re-running this recorder
    twelve times on one M1 Max from one unchanged binary flipped two of the
    sixteen decisions, and the range was the only gate that ever fired: the
    ``retune_mla_decode`` candidate was 32-55% faster in all 24 runs and won
    144 of 144 paired trials, yet promoted in only 7 of 12 recordings. Because
    a range never shrinks as samples are added, collecting more evidence made
    that true winner *less* promotable (69% at two runs, 7% at eight), so the
    only way to land a promotion was to re-record until the draw was
    favourable -- selecting on noise, which is what this gate existed to stop.

    The bound converges on the true mean instead, so more runs can only help a
    route that is really faster and can never rescue one whose speedup is
    noise. ``max_speedup_spread`` is still computed and retained, but it is now
    diagnostic: like absolute clock drift, it describes the measurement without
    deciding the route.
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
    if min_promotion_runs < 2:
        raise ValueError("min_promotion_runs must be at least two")
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
                    lower_bound = speedup_lower_confidence_bound(median_speedups)
                    route_evidence["paired_speedups_vs_incumbent"] = paired_speedups
                    route_evidence["paired_median_speedups"] = median_speedups
                    route_evidence["paired_win_fractions"] = win_fractions
                    # Retained for audit, no longer decisive: a range grows
                    # with the sample, so gating on it punished evidence.
                    route_evidence["cross_run_speedup_spread"] = spread
                    route_evidence["speedup_lower_confidence_bound"] = lower_bound
                    # Every run individually shows a win of the required size.
                    # This is the point estimate; it is necessary, not
                    # sufficient, because it says nothing about reproducibility.
                    # Pooled across every paired trial in every run, with a
                    # per-run floor that no run may actually lose on balance.
                    #
                    # This replaced `all(fraction >= min_paired_win_fraction)`,
                    # which had the same non-convergence as the range cap it
                    # sits beside. With three trials a run's win fraction can
                    # only be 0, 1/3, 2/3 or 1, so a 0.75 threshold means "win
                    # all three", and requiring that of every run means winning
                    # 3n consecutive trials: for a route that truly wins 95% of
                    # trials, P(promote) falls from 0.74 at two runs to 0.29 at
                    # eight. Measured here, `retune_moe_swiglu` 16x32x64x32_e4
                    # promoted in 2 of 6 recordings on a single lost trial out
                    # of fifteen. A pooled proportion is a consistent estimator
                    # and settles as runs are added; the floor keeps one strong
                    # run from carrying a run the candidate lost.
                    trial_wins = sum(sum(value > 0.0 for value in values)
                                     for values in paired_speedups)
                    trial_count = sum(len(values) for values in paired_speedups)
                    pooled_win_fraction = (trial_wins / trial_count
                                           if trial_count else None)
                    route_evidence["pooled_paired_win_fraction"] = pooled_win_fraction
                    route_evidence["paired_trial_count"] = trial_count
                    wins_on_point_estimates = bool(
                        median_speedups
                        and all(speedup >= min_speedup
                                for speedup in median_speedups)
                        and pooled_win_fraction is not None
                        and pooled_win_fraction >= min_paired_win_fraction
                        and all(fraction > 0.5 for fraction in win_fractions))
                    # ...and the win survives its own measurement error, over
                    # enough independent runs to have measured that error.
                    reproducible = bool(
                        wins_on_point_estimates
                        and len(median_speedups) >= min_promotion_runs
                        and lower_bound is not None
                        and lower_bound >= min_speedup)
                    # Three distinct states, and collapsing the last two is
                    # how a re-record launders one into a promotion. A route
                    # that averages a win of the required size but cannot hold
                    # it across runs is *unmeasured*, not settled; a route that
                    # is simply slower is settled.
                    #
                    # The state is reached whenever runs disagree about a win
                    # the point estimates like -- a candidate whose per-run
                    # speedups scatter across the threshold, or one run landing
                    # somewhere the others did not. Host contention is *not*
                    # such a case and must not be confused for one: the trials
                    # are paired and interleaved, so eight busy cores slow both
                    # routes together and leave the verdict intact (measured on
                    # this M1 Max, `retune_moe_swiglu` 16x32x64x32_e4 records
                    # +50.6% loaded against +52% quiet). What this state is for
                    # is evidence that genuinely does not agree with itself.
                    interval = speedup_confidence_interval(median_speedups)
                    route_evidence["speedup_confidence_interval"] = (
                        list(interval) if interval else None)
                    route_evidence["stability_verdict"] = (
                        "stable_win" if reproducible
                        # The interval still admits a qualifying win but cannot
                        # confirm one: inconclusive, so explicitly unpromotable.
                        else "unstable_evidence"
                        if interval is not None and interval[1] >= min_speedup
                        # The interval excludes a qualifying win: settled.
                        else "not_a_stable_win")
                    route_evidence["promotable"] = reproducible
                    if reproducible:
                        winners.append((min(median_speedups), route))
                if winners:
                    _, selected = max(winners)
                    status = "promote_candidate"
                    reason = "candidate met paired stable-win gates in every run"
                else:
                    # A candidate that wins every run on point estimates but
                    # cannot clear its own confidence bound is NOT the same
                    # state as a candidate that is simply slower, and recording
                    # both as a bare `retain_incumbent` hid the difference. Say
                    # which one this is, so a re-record cannot quietly convert
                    # "we could not tell" into "promoted on a good draw".
                    unstable = sorted(
                        route for route, route_evidence in evidence.items()
                        if route_evidence.get("stability_verdict")
                        == "unstable_evidence")
                    if unstable:
                        status = "retain_incumbent_unstable_candidate"
                        reason = (
                            "candidate(s) " + ", ".join(unstable) +
                            " won every run but missed the 95% lower bound on "
                            "cross-run speedup; not promotable without "
                            "measurement that reproduces")
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
            "minimum_pooled_paired_win_fraction": min_paired_win_fraction,
            "minimum_paired_win_fraction_each_run": 0.5,
            "minimum_speedup_lower_confidence_bound": min_speedup,
            "speedup_confidence_level": SPEEDUP_CONFIDENCE_LEVEL,
            "minimum_promotion_runs": min_promotion_runs,
            "cross_run_speedup_spread_is_diagnostic_only": True,
            "maximum_cross_run_speedup_spread": max_speedup_spread,
            "requires_native_dispatch": True,
            "requires_numerical_validation": True,
            "requires_repeated_measurement": True,
            "requires_interleaved_paired_trials": True,
            "requires_resource_evidence": True,
        },
        "decisions": decisions,
    }


def seal_strict_route_ledger(
    stable: Mapping[str, Any], reports: Sequence[Mapping[str, Any]], *,
    valid_days: int = 30, selection_scope: str = STRICT_RUNTIME_ROUTE_SCOPE,
) -> dict[str, Any]:
    """Turn an aggregate into production-readable v2 evidence.

    Context must have been captured by each producer at measurement time; a
    selector must never synthesize it after the fact.
    """
    if len(reports) < 2:
        raise ValueError("strict sealing requires two independent reports")
    if selection_scope not in {STRICT_RUNTIME_ROUTE_SCOPE, STRICT_PACKAGE_SUBGRAPH_SCOPE}:
        raise ValueError(f"unsupported strict route selection scope: {selection_scope!r}")
    report_scopes = {report.get("selection_scope", STRICT_RUNTIME_ROUTE_SCOPE)
                     for report in reports}
    if report_scopes != {selection_scope}:
        raise ValueError("strict sealing scope must match every producer report")
    contexts: list[Mapping[str, Any]] = []
    for report in reports:
        context = report.get("context")
        if not isinstance(context, Mapping):
            raise ValueError("strict sealing requires producer-captured context")
        contexts.append(context)
    if any(dict(context) != dict(contexts[0]) for context in contexts[1:]):
        raise ValueError("strict sealing requires identical exact contexts")
    now = datetime.now(timezone.utc)
    decisions = []
    ineligible_decisions = []
    for row in stable.get("decisions", []):
        if not isinstance(row, Mapping):
            continue
        if row.get("selected_route") is None:
            # Preserve the aggregate's negative result for audit and future
            # remeasurement, but deliberately keep it out of ``decisions``:
            # the loader only sees admissible production rows.
            ineligible_decisions.append({
                "op": row.get("op"), "shape": row.get("shape"),
                "dtype": row.get("dtype"), "device": row.get("device"),
                "timing_domain": row.get("timing_domain"),
                "incumbent_route": row.get("incumbent_route"),
                "status": "ineligible",
                "reason": row.get("reason", "no selectable route"),
            })
            continue
        selected = str(row["selected_route"])
        evidence = row.get("route_evidence", {}).get(selected, {})
        decisions.append({**row, "selected_evidence": {
            "provenance": "native_gpu" if evidence.get("placement_and_numerical_proof") else "reference_cpu",
            "correctness": bool(evidence.get("placement_and_numerical_proof")),
            "device": row.get("device"), "timing_domain": row.get("timing_domain"),
        }})
    report_digests = [
        "sha256:" + hashlib.sha256(
            json.dumps(report, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()
        for report in reports
    ]
    return {"schema": STRICT_ROUTE_LEDGER_SCHEMA, "selection_scope": selection_scope,
            "measured_at": now.isoformat().replace("+00:00", "Z"),
            "expires_at": (now + timedelta(days=valid_days)).isoformat().replace("+00:00", "Z"),
            "context": dict(contexts[0]), "source_report_count": len(reports),
            "source_report_digests": report_digests,
            "promotion_rules": stable.get("promotion_rules", {}),
            "decisions": decisions, "ineligible_decisions": ineligible_decisions}


__all__ = [
    "AppleRouteContext",
    "AppleRouteMeasurement",
    "PACKAGE_ROUTE",
    "ProductionRouteDecision",
    "ROUTE_REPORT_SCHEMA_VERSION",
    "STABLE_ROUTE_LEDGER_SCHEMA_VERSION",
    "STRICT_ROUTE_LEDGER_SCHEMA",
    "STRICT_RUNTIME_ROUTE_SCOPE",
    "STRICT_PACKAGE_SUBGRAPH_SCOPE",
    "StrictRouteLedger",
    "LegacyRouteLedgerInventory",
    "aggregate_stable_route_reports",
    "live_apple_route_context",
    "legacy_route_ledger_inventory",
    "load_route_measurements",
    "load_strict_route_ledger",
    "package_route_selected",
    "production_route_for",
    "production_route_decision",
    "promotion_rule_violations",
    "live_apple_device_tag",
    "select_route",
    "seal_strict_route_ledger",
    "speedup_confidence_interval",
    "speedup_lower_confidence_bound",
    "SPEEDUP_CONFIDENCE_LEVEL",
]
