"""Compiler-facing profiler and model-analyzer planning contracts.

This module is deliberately hardware-free.  It lets the compiler, CLI, and
reporting tools ask one stable question:

    "For this target and these profiling features, which provider owns the
    trace, what is available now, and what artifact should downstream tools
    expect?"

The provider table is based on the active Tessera backends plus public profiler
surfaces used as design anchors:

* Triton Proton: Python context, annotations, launch metadata, GPU metrics,
  CUPTI PC sampling, and an intra-kernel instrumentation backend.
* Intel Unitrace: host API call logging, device/kernel timelines, metric query
  and sampling, filters, and paused/resumed collection.
* NVIDIA CUPTI / Triton Model Analyzer: runtime callbacks, activity records,
  PC/range profiling, and model-configuration sweeps with reports.
* AMD ROCprofiler-SDK: HIP/HSA/marker/memory tracing, hardware counters,
  dispatch/device counter collection, PC sampling, and thread trace.
* Apple Metal counters: counter sample buffers and timestamp correlation.

The contract does not claim that all vendor collectors are linked into Tessera
today; rows carry ``available`` / ``planned`` / ``unsupported`` status so reports
can be honest while the integration lands backend-by-backend.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping, Sequence


TRACE_SCHEMA_VERSION = "tessera.compiler.profiling_plan.v1"
MODEL_ANALYZER_SCHEMA_VERSION = "tessera.compiler.model_analyzer_manifest.v1"


RUNTIME_API = "runtime_api"
DEVICE_ACTIVITY = "device_activity"
COUNTERS = "counters"
INTRA_KERNEL = "intra_kernel"
MODEL_ANALYZER = "model_analyzer"
HOST_CONTEXT = "host_context"
ROOFLINE = "roofline"

TRACE_FEATURES: frozenset[str] = frozenset({
    RUNTIME_API,
    DEVICE_ACTIVITY,
    COUNTERS,
    INTRA_KERNEL,
    MODEL_ANALYZER,
    HOST_CONTEXT,
    ROOFLINE,
})

AVAILABLE = "available"
PLANNED = "planned"
UNSUPPORTED = "unsupported"
CAPABILITY_STATUSES: frozenset[str] = frozenset({
    AVAILABLE,
    PLANNED,
    UNSUPPORTED,
})


@dataclass(frozen=True)
class ProviderCapability:
    """One provider's support for one profiling feature."""

    feature: str
    provider: str
    status: str
    artifact: str
    notes: str = ""
    controls: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.feature not in TRACE_FEATURES:
            raise ValueError(
                f"unknown profiling feature {self.feature!r}; "
                f"valid features are {sorted(TRACE_FEATURES)}"
            )
        if self.status not in CAPABILITY_STATUSES:
            raise ValueError(
                f"unknown provider status {self.status!r}; "
                f"valid statuses are {sorted(CAPABILITY_STATUSES)}"
            )

    def to_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {
            "feature": self.feature,
            "provider": self.provider,
            "status": self.status,
            "artifact": self.artifact,
        }
        if self.notes:
            out["notes"] = self.notes
        if self.controls:
            out["controls"] = list(self.controls)
        return out


@dataclass(frozen=True)
class ModelAnalyzerSweep:
    """Tessera analogue of Triton Model Analyzer's config search contract."""

    mode: str = "quick"
    batch_sizes: tuple[int, ...] = (1, 2, 4, 8)
    instance_counts: tuple[int, ...] = (1,)
    dynamic_batching: tuple[bool, ...] = (False, True)
    latency_budget_ms: float | None = None
    memory_budget_bytes: int | None = None

    def __post_init__(self) -> None:
        if self.mode not in {"quick", "brute", "manual", "optuna"}:
            raise ValueError("mode must be one of quick, brute, manual, optuna")
        if not self.batch_sizes or any(int(v) <= 0 for v in self.batch_sizes):
            raise ValueError("batch_sizes must contain positive integers")
        if not self.instance_counts or any(int(v) <= 0 for v in self.instance_counts):
            raise ValueError("instance_counts must contain positive integers")
        if self.latency_budget_ms is not None and self.latency_budget_ms <= 0:
            raise ValueError("latency_budget_ms must be positive when provided")
        if self.memory_budget_bytes is not None and self.memory_budget_bytes <= 0:
            raise ValueError("memory_budget_bytes must be positive when provided")

    def to_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {
            "mode": self.mode,
            "batch_sizes": list(self.batch_sizes),
            "instance_counts": list(self.instance_counts),
            "dynamic_batching": list(self.dynamic_batching),
        }
        if self.latency_budget_ms is not None:
            out["latency_budget_ms"] = self.latency_budget_ms
        if self.memory_budget_bytes is not None:
            out["memory_budget_bytes"] = self.memory_budget_bytes
        return out


@dataclass(frozen=True)
class IntraKernelProbe:
    """Compiler-inserted probe point planned before backend lowering.

    This is intentionally backend-neutral.  NVIDIA/ROCm may later map probes to
    PC sampling or thread trace correlation, while Apple and CPU targets can use
    compiler-inserted phase markers/counters where native samplers do not exist.
    """

    kernel: str
    phase: str
    metric: str = "elapsed_cycles"
    aggregation: str = "sample"
    payload_fields: tuple[str, ...] = ("kernel", "phase", "tile", "program_id")

    def __post_init__(self) -> None:
        if not str(self.kernel).strip():
            raise ValueError("kernel must be non-empty")
        if not str(self.phase).strip():
            raise ValueError("phase must be non-empty")
        if self.metric not in {"elapsed_cycles", "active_threads", "bytes", "flops", "occupancy"}:
            raise ValueError("unsupported intra-kernel probe metric")
        if self.aggregation not in {"sample", "sum", "max", "min", "avg"}:
            raise ValueError("unsupported intra-kernel probe aggregation")
        if not self.payload_fields:
            raise ValueError("payload_fields must be non-empty")

    def to_dict(self) -> dict[str, Any]:
        return {
            "kernel": self.kernel,
            "phase": self.phase,
            "metric": self.metric,
            "aggregation": self.aggregation,
            "payload_fields": list(self.payload_fields),
        }


@dataclass(frozen=True)
class ProfilerPlan:
    """A deterministic profiling request artifact consumed by runtime tools."""

    target: str
    requested_features: tuple[str, ...]
    capabilities: tuple[ProviderCapability, ...]
    model_name: str | None = None
    kernels: tuple[str, ...] = ()
    analyzer_sweep: ModelAnalyzerSweep | None = None
    intra_kernel_probes: tuple[IntraKernelProbe, ...] = ()
    source_notes: tuple[str, ...] = field(default_factory=tuple)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": TRACE_SCHEMA_VERSION,
            "target": self.target,
            "model_name": self.model_name,
            "kernels": list(self.kernels),
            "requested_features": list(self.requested_features),
            "capabilities": [cap.to_dict() for cap in self.capabilities],
            "analyzer_sweep": (
                self.analyzer_sweep.to_dict()
                if self.analyzer_sweep is not None
                else None
            ),
            "intra_kernel_probes": [probe.to_dict() for probe in self.intra_kernel_probes],
            "source_notes": list(self.source_notes),
            "summary": summarize_capabilities(self.capabilities),
        }

    def to_json(self, *, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, sort_keys=True)


def plan_profile(
    target: str,
    *,
    features: Iterable[str] | None = None,
    model_name: str | None = None,
    kernels: Sequence[str] = (),
    analyzer_sweep: ModelAnalyzerSweep | Mapping[str, Any] | None = None,
) -> ProfilerPlan:
    """Create a backend-aware profiler plan for the compiler.

    ``target`` accepts Tessera target spellings such as ``"cuda"``,
    ``"nvidia_sm90"``, ``"rocm_gfx942"``, ``"apple_gpu"``, or ``"cpu"``.
    Unknown targets are allowed but every requested feature is marked planned or
    unsupported through the generic provider so callers can still render a clear
    diagnostic.
    """

    normalized = normalize_profiler_target(target)
    requested = _normalize_features(features)
    sweep = _normalize_sweep(analyzer_sweep)
    caps = provider_capabilities(normalized)
    selected = tuple(_select_capability(caps, feature) for feature in requested)
    return ProfilerPlan(
        target=normalized,
        requested_features=requested,
        capabilities=selected,
        model_name=model_name,
        kernels=tuple(str(k) for k in kernels),
        analyzer_sweep=sweep,
        intra_kernel_probes=(
            plan_intra_kernel_probes(kernels)
            if INTRA_KERNEL in requested
            else ()
        ),
        source_notes=_SOURCE_NOTES,
    )


@dataclass(frozen=True)
class ModelAnalyzerManifest:
    """Runner-facing config-search manifest derived from a profiler plan."""

    plan: ProfilerPlan
    objective: str = "latency_ms"
    warmup_runs: int = 3
    measurement_window_s: float = 30.0
    output_artifacts: tuple[str, ...] = (
        "model_analyzer_summary_json",
        "trace_event_json",
        "telemetry_json",
    )

    def __post_init__(self) -> None:
        if self.objective not in {"latency_ms", "throughput_qps", "memory_bytes"}:
            raise ValueError("objective must be latency_ms, throughput_qps, or memory_bytes")
        if self.warmup_runs < 0:
            raise ValueError("warmup_runs must be non-negative")
        if self.measurement_window_s <= 0:
            raise ValueError("measurement_window_s must be positive")
        if not self.output_artifacts:
            raise ValueError("output_artifacts must be non-empty")

    def to_dict(self) -> dict[str, Any]:
        sweep = self.plan.analyzer_sweep or ModelAnalyzerSweep()
        analyzer_cap = _capability_for_plan(self.plan, MODEL_ANALYZER)
        trace_caps = [
            _capability_for_plan(self.plan, feature).to_dict()
            for feature in (RUNTIME_API, DEVICE_ACTIVITY, COUNTERS, INTRA_KERNEL)
        ]
        return {
            "schema": MODEL_ANALYZER_SCHEMA_VERSION,
            "source_plan_schema": TRACE_SCHEMA_VERSION,
            "target": self.plan.target,
            "model_name": self.plan.model_name,
            "kernels": list(self.plan.kernels),
            "search": sweep.to_dict(),
            "objective": {
                "primary": self.objective,
                "latency_budget_ms": sweep.latency_budget_ms,
                "memory_budget_bytes": sweep.memory_budget_bytes,
            },
            "runner": {
                "provider": analyzer_cap.provider,
                "status": analyzer_cap.status,
                "artifact": analyzer_cap.artifact,
            },
            "telemetry": {
                "required_features": [
                    RUNTIME_API,
                    DEVICE_ACTIVITY,
                ],
                "capabilities": trace_caps,
            },
            "intra_kernel_probes": [
                probe.to_dict() for probe in self.plan.intra_kernel_probes
            ],
            "execution": {
                "warmup_runs": self.warmup_runs,
                "measurement_window_s": self.measurement_window_s,
            },
            "output_artifacts": list(self.output_artifacts),
        }

    def to_json(self, *, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, sort_keys=True)


def model_analyzer_manifest(
    plan: ProfilerPlan,
    *,
    objective: str = "latency_ms",
    warmup_runs: int = 3,
    measurement_window_s: float = 30.0,
    output_artifacts: Sequence[str] = (
        "model_analyzer_summary_json",
        "trace_event_json",
        "telemetry_json",
    ),
) -> ModelAnalyzerManifest:
    """Create a runner-facing Model Analyzer manifest from a profiler plan."""

    return ModelAnalyzerManifest(
        plan=plan,
        objective=objective,
        warmup_runs=warmup_runs,
        measurement_window_s=measurement_window_s,
        output_artifacts=tuple(output_artifacts),
    )


def plan_intra_kernel_probes(
    kernels: Sequence[str],
    *,
    phases: Sequence[str] = ("prologue", "mainloop", "epilogue"),
    metrics: Sequence[str] = ("elapsed_cycles",),
) -> tuple[IntraKernelProbe, ...]:
    """Create deterministic compiler-inserted probe specs for kernels."""

    kernel_names = tuple(str(k).strip() for k in kernels if str(k).strip()) or ("*",)
    probes: list[IntraKernelProbe] = []
    for kernel in kernel_names:
        for phase in phases:
            for metric in metrics:
                probes.append(IntraKernelProbe(kernel=kernel, phase=str(phase), metric=str(metric)))
    return tuple(probes)


def provider_capabilities(target: str) -> tuple[ProviderCapability, ...]:
    normalized = normalize_profiler_target(target)
    try:
        return _PROVIDER_TABLE[normalized]
    except KeyError:
        return _generic_capabilities(normalized)


def summarize_capabilities(
    capabilities: Sequence[ProviderCapability],
) -> dict[str, Any]:
    counts = {AVAILABLE: 0, PLANNED: 0, UNSUPPORTED: 0}
    missing: list[str] = []
    for cap in capabilities:
        counts[cap.status] += 1
        if cap.status != AVAILABLE:
            missing.append(cap.feature)
    return {
        "available": counts[AVAILABLE],
        "planned": counts[PLANNED],
        "unsupported": counts[UNSUPPORTED],
        "missing_features": missing,
    }


def normalize_profiler_target(target: str) -> str:
    value = str(target).strip().lower().replace("-", "_")
    if value in {"cuda", "nvidia", "sm80", "sm90", "sm100"}:
        return "nvidia"
    if value.startswith("nvidia_") or value.startswith("sm_"):
        return "nvidia"
    if value in {"hip", "amd", "rocm"} or value.startswith("rocm_") or value.startswith("gfx"):
        return "rocm"
    if value in {"apple", "metal", "mps", "apple_gpu"}:
        return "apple_gpu"
    if value in {"cpu", "x86", "x86_64", "apple_cpu"}:
        return "cpu"
    return value or "generic"


def _normalize_features(features: Iterable[str] | None) -> tuple[str, ...]:
    if features is None:
        return (
            HOST_CONTEXT,
            RUNTIME_API,
            DEVICE_ACTIVITY,
            COUNTERS,
            ROOFLINE,
        )
    seen: list[str] = []
    for feature in features:
        key = str(feature).strip().lower().replace("-", "_")
        if key not in TRACE_FEATURES:
            raise ValueError(
                f"unknown profiling feature {feature!r}; "
                f"valid features are {sorted(TRACE_FEATURES)}"
            )
        if key not in seen:
            seen.append(key)
    return tuple(seen)


def _normalize_sweep(
    sweep: ModelAnalyzerSweep | Mapping[str, Any] | None,
) -> ModelAnalyzerSweep | None:
    if sweep is None:
        return None
    if isinstance(sweep, ModelAnalyzerSweep):
        return sweep
    kwargs = dict(sweep)
    for key in ("batch_sizes", "instance_counts", "dynamic_batching"):
        if key in kwargs:
            kwargs[key] = tuple(kwargs[key])
    return ModelAnalyzerSweep(**kwargs)


def _select_capability(
    capabilities: Sequence[ProviderCapability],
    feature: str,
) -> ProviderCapability:
    for cap in capabilities:
        if cap.feature == feature:
            return cap
    return ProviderCapability(
        feature=feature,
        provider="tprof",
        status=UNSUPPORTED,
        artifact="trace-event-json",
        notes="No provider mapping exists for this target/feature pair.",
    )


def _capability_for_plan(plan: ProfilerPlan, feature: str) -> ProviderCapability:
    for cap in plan.capabilities:
        if cap.feature == feature:
            return cap
    return _select_capability(provider_capabilities(plan.target), feature)


def _generic_capabilities(target: str) -> tuple[ProviderCapability, ...]:
    return (
        ProviderCapability(
            HOST_CONTEXT,
            "tprof",
            AVAILABLE,
            "chrome_trace/perfetto_json",
            "Portable Tessera host ranges and annotations.",
        ),
        ProviderCapability(
            RUNTIME_API,
            "tprof-runtime-shim",
            PLANNED,
            "chrome_trace/perfetto_json",
            f"Runtime API interception is not backend-specialized for {target}.",
        ),
        ProviderCapability(
            DEVICE_ACTIVITY,
            "backend-provider",
            PLANNED,
            "trace-event-json",
            f"Device activity collection needs a native provider for {target}.",
        ),
        ProviderCapability(
            COUNTERS,
            "backend-provider",
            PLANNED,
            "telemetry_json",
            f"Hardware counters need a native provider for {target}.",
        ),
        ProviderCapability(
            INTRA_KERNEL,
            "compiler-instrumentation",
            PLANNED,
            "target_ir_instrumented",
            "Compiler can plan instrumentation points before backend lowering.",
        ),
        ProviderCapability(
            MODEL_ANALYZER,
            "tessera-model-analyzer",
            PLANNED,
            "model_analyzer_plan_json",
            "Search contract exists; execution needs target-specific runner.",
        ),
        ProviderCapability(
            ROOFLINE,
            "tprof-roofline",
            AVAILABLE,
            "roofline_html/json",
            "Portable roofline ingestion can consume trace/metric artifacts.",
        ),
    )


_SOURCE_NOTES = (
    "Triton Proton shape: Python context, annotations, launch metadata, GPU "
    "metrics, CUPTI PC sampling, and intra-kernel instrumentation.",
    "Unitrace shape: host call logging, kernel/device timelines, metrics, "
    "filters, and paused/resumed collection.",
    "Backend providers remain explicit so reports never confuse planned "
    "instrumentation with native runtime proof.",
)


_PROVIDER_TABLE: dict[str, tuple[ProviderCapability, ...]] = {
    "cpu": (
        ProviderCapability(
            HOST_CONTEXT,
            "tprof",
            AVAILABLE,
            "chrome_trace/perfetto_json",
            "Host ranges, counters, and compiler annotations are portable.",
        ),
        ProviderCapability(
            RUNTIME_API,
            "tprof-runtime-shim",
            AVAILABLE,
            "chrome_trace/perfetto_json",
            "Tessera runtime C ABI calls can be traced without vendor tools.",
        ),
        ProviderCapability(
            DEVICE_ACTIVITY,
            "cpu-timers",
            AVAILABLE,
            "trace-event-json",
            "CPU activity is represented as host/runtime spans.",
        ),
        ProviderCapability(
            COUNTERS,
            "tprof-counters",
            AVAILABLE,
            "telemetry_json",
            "Portable counters and wall-clock derived throughput.",
        ),
        ProviderCapability(
            INTRA_KERNEL,
            "compiler-instrumentation",
            PLANNED,
            "target_ir_instrumented",
            "Tile-level probes can be inserted, but CPU kernels do not expose "
            "GPU-style instruction sampling.",
        ),
        ProviderCapability(
            MODEL_ANALYZER,
            "tessera-model-analyzer",
            AVAILABLE,
            "model_analyzer_plan_json",
            "Compiler can sweep batch/instance/dynamic-batching style configs "
            "against CPU/runtime telemetry.",
        ),
        ProviderCapability(
            ROOFLINE,
            "tprof-roofline",
            AVAILABLE,
            "roofline_html/json",
            "Existing roofline tooling can classify CPU or imported samples.",
        ),
    ),
    "nvidia": (
        ProviderCapability(
            HOST_CONTEXT,
            "tprof+nvtx",
            AVAILABLE,
            "chrome_trace/perfetto_json",
            "Tessera ranges map to NVTX when built with NVTX.",
        ),
        ProviderCapability(
            RUNTIME_API,
            "cupti-callback-api",
            PLANNED,
            "chrome_trace/perfetto_json",
            "CUPTI Callback API maps CUDA runtime/driver calls.",
            controls=("include_api", "exclude_api", "start_paused"),
        ),
        ProviderCapability(
            DEVICE_ACTIVITY,
            "cupti-activity-api",
            PLANNED,
            "chrome_trace/perfetto_json",
            "CUPTI Activity API maps kernels, memcpy, device, and correlation "
            "records.",
            controls=("include_kernels", "exclude_kernels"),
        ),
        ProviderCapability(
            COUNTERS,
            "cupti-profiler-range-api",
            PLANNED,
            "metrics_csv+telemetry_json",
            "Range profiling and GPU metrics back roofline and reports.",
        ),
        ProviderCapability(
            INTRA_KERNEL,
            "cupti-pc-sampling+compiler-instrumentation",
            PLANNED,
            "pc_sampling_json+target_ir_instrumented",
            "Proton-like path: PC sampling for NVIDIA plus optional inserted "
            "Target IR probes.",
        ),
        ProviderCapability(
            MODEL_ANALYZER,
            "tessera-model-analyzer+nvidia-triton-model-analyzer",
            PLANNED,
            "model_analyzer_plan_json",
            "Tessera can emit config-search plans; Triton deployment can "
            "consume exported models for NVIDIA/Triton serving sweeps.",
        ),
        ProviderCapability(
            ROOFLINE,
            "tprof-roofline+nsight-import",
            AVAILABLE,
            "roofline_html/json",
            "Existing roofline tools ingest Nsight-style CSV examples.",
        ),
    ),
    "rocm": (
        ProviderCapability(
            HOST_CONTEXT,
            "tprof+roctx",
            PLANNED,
            "chrome_trace/perfetto_json",
            "Tessera ranges should map to ROCTx markers for ROCm traces.",
        ),
        ProviderCapability(
            RUNTIME_API,
            "rocprofiler-sdk-tracing",
            PLANNED,
            "chrome_trace/perfetto_json",
            "ROCprofiler-SDK supports HIP/HSA/API, memory, marker, and RCCL "
            "tracing surfaces.",
            controls=("include_api", "exclude_api", "start_paused"),
        ),
        ProviderCapability(
            DEVICE_ACTIVITY,
            "rocprofiler-sdk-dispatch-tracing",
            PLANNED,
            "chrome_trace/perfetto_json",
            "Dispatch/device traces map kernels, copies, queues, and "
            "correlation IDs.",
            controls=("include_kernels", "exclude_kernels"),
        ),
        ProviderCapability(
            COUNTERS,
            "rocprofiler-sdk-counters",
            PLANNED,
            "metrics_csv+telemetry_json",
            "Dispatch/device counter collection maps AMD hardware metrics.",
        ),
        ProviderCapability(
            INTRA_KERNEL,
            "rocprofiler-sdk-pc-sampling+thread-trace",
            PLANNED,
            "pc_sampling_json+thread_trace",
            "AMD PC sampling/thread trace is backend-specific and should be "
            "guarded by capability probes.",
        ),
        ProviderCapability(
            MODEL_ANALYZER,
            "tessera-model-analyzer",
            PLANNED,
            "model_analyzer_plan_json",
            "Compiler sweeps can run on ROCm once the runner consumes HIP "
            "telemetry and counters.",
        ),
        ProviderCapability(
            ROOFLINE,
            "tprof-roofline",
            AVAILABLE,
            "roofline_html/json",
            "Roofline tooling can consume ROCm metric exports once normalized.",
        ),
    ),
    "apple_gpu": (
        ProviderCapability(
            HOST_CONTEXT,
            "tprof+os-signpost",
            PLANNED,
            "chrome_trace/perfetto_json",
            "Tessera ranges should bridge to os_signpost/Instruments metadata.",
        ),
        ProviderCapability(
            RUNTIME_API,
            "tprof-apple-runtime",
            PLANNED,
            "chrome_trace/perfetto_json",
            "Trace MPSGraph/Metal command submission through Tessera runtime "
            "wrappers; native proof must run outside this sandbox on this host.",
        ),
        ProviderCapability(
            DEVICE_ACTIVITY,
            "metal-system-trace",
            PLANNED,
            "instruments_trace+trace_event_json",
            "Metal System Trace is the native activity timeline; Tessera should "
            "correlate command buffers to compiler kernel IDs.",
        ),
        ProviderCapability(
            COUNTERS,
            "metal-counter-sample-buffer",
            PLANNED,
            "counter_sample_json+telemetry_json",
            "Metal counter sample buffers expose runtime GPU counter data when "
            "the device reports support.",
        ),
        ProviderCapability(
            INTRA_KERNEL,
            "compiler-instrumentation",
            PLANNED,
            "target_ir_instrumented",
            "Apple has no CUPTI-style public PC sampler; use compiler-inserted "
            "phase markers/counters and Metal counters where supported.",
        ),
        ProviderCapability(
            MODEL_ANALYZER,
            "tessera-model-analyzer",
            PLANNED,
            "model_analyzer_plan_json",
            "Compiler sweeps can compare schedules/configs against Apple GPU "
            "telemetry after native runner integration.",
        ),
        ProviderCapability(
            ROOFLINE,
            "tprof-roofline",
            AVAILABLE,
            "roofline_html/json",
            "Existing roofline tooling can classify Apple samples once imported.",
        ),
    ),
}


__all__ = [
    "AVAILABLE",
    "CAPABILITY_STATUSES",
    "COUNTERS",
    "DEVICE_ACTIVITY",
    "HOST_CONTEXT",
    "INTRA_KERNEL",
    "IntraKernelProbe",
    "MODEL_ANALYZER",
    "MODEL_ANALYZER_SCHEMA_VERSION",
    "ModelAnalyzerSweep",
    "ModelAnalyzerManifest",
    "PLANNED",
    "ProfilerPlan",
    "ProviderCapability",
    "ROOFLINE",
    "RUNTIME_API",
    "TRACE_FEATURES",
    "TRACE_SCHEMA_VERSION",
    "UNSUPPORTED",
    "model_analyzer_manifest",
    "normalize_profiler_target",
    "plan_intra_kernel_probes",
    "plan_profile",
    "provider_capabilities",
    "summarize_capabilities",
]
