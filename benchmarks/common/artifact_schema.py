"""Standard benchmark row schema for compiler/runtime visibility."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any

from tessera.telemetry import make_event


class CompilerPath(str, Enum):
    TESSERA_JIT_CPU = "tessera_jit_cpu"
    TESSERA_JIT_APPLE_GPU = "tessera_jit_apple_gpu"
    GRAPH_IR_ONLY = "graph_ir_only"
    RUNTIME_UNAVAILABLE = "runtime_unavailable"
    REFERENCE = "reference"
    ARTIFACT_ONLY = "artifact_only"
    UNSUPPORTED = "unsupported"


class RuntimeStatus(str, Enum):
    EXECUTABLE = "executable"
    ARTIFACT_ONLY = "artifact_only"
    SKIPPED = "skipped"
    UNSUPPORTED = "unsupported"
    MISSING_BACKEND = "missing_backend"


class ExecutionKind(str, Enum):
    """Reference vs. optimized split for benchmark rows.

    Closes the M3 follow-up "per-target reference/optimized
    distinction in benchmark JSON".  Independent of
    :class:`CompilerPath` (which records how we got from source
    to runtime) and :class:`RuntimeStatus` (which records whether
    the runtime executed).  This axis records *which kind of
    execution produced the numbers*:

      * ``REFERENCE`` — naive NumPy / SciPy fallback.  Numbers are
        correctness-bearing but performance-meaningless.
      * ``OPTIMIZED_NATIVE`` — a real backend kernel ran
        (Accelerate cblas, BNNS, MPS, MSL, x86 AMX, NVIDIA WGMMA,
        ROCm MFMA, ...).  Numbers are performance-bearing.
      * ``ARTIFACT_ONLY`` — IR / lowering succeeded but no
        runtime executed; performance fields are unset.
      * ``UNKNOWN`` — legacy benchmark rows that pre-date this
        axis.  New rows should never use this.
    """
    REFERENCE = "reference"
    OPTIMIZED_NATIVE = "optimized_native"
    ARTIFACT_ONLY = "artifact_only"
    UNKNOWN = "unknown"


@dataclass(frozen=True)
class ArtifactLevels:
    graph: bool = False
    schedule: bool = False
    tile: bool = False
    target: bool = False
    artifact_hash: str | None = None


@dataclass(frozen=True)
class Correctness:
    max_error: float | None = None
    relative_error: float | None = None
    tolerance: float | None = None
    passed: bool | None = None


@dataclass(frozen=True)
class Profile:
    cpu_wall_ms: float | None = None
    kernel_elapsed_ms: float | None = None
    memory_bytes: int | None = None
    launch_overhead_ms: float | None = None


@dataclass(frozen=True)
class BenchmarkOperator:
    name: str
    dtype: str
    shape: str
    target: str = "cpu"


@dataclass(frozen=True)
class BenchmarkRow:
    operator: BenchmarkOperator
    compiler_path: CompilerPath
    runtime_status: RuntimeStatus
    artifact_levels: ArtifactLevels = field(default_factory=ArtifactLevels)
    correctness: Correctness = field(default_factory=Correctness)
    profile: Profile = field(default_factory=Profile)
    metrics: dict[str, Any] = field(default_factory=dict)
    telemetry: dict[str, Any] = field(default_factory=dict)
    reason: str = ""
    execution_kind: ExecutionKind = ExecutionKind.UNKNOWN

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["compiler_path"] = self.compiler_path.value
        data["runtime_status"] = self.runtime_status.value
        data["execution_kind"] = self.execution_kind.value
        return data

    def flat_dict(self) -> dict[str, Any]:
        data = {
            "operator": self.operator.name,
            "dtype": self.operator.dtype,
            "shape": self.operator.shape,
            "target": self.operator.target,
            "compiler_path": self.compiler_path.value,
            "runtime_status": self.runtime_status.value,
            "execution_kind": self.execution_kind.value,
            "reason": self.reason,
        }
        data.update({f"artifact_{k}": v for k, v in asdict(self.artifact_levels).items()})
        data.update({f"correctness_{k}": v for k, v in asdict(self.correctness).items()})
        data.update({f"profile_{k}": v for k, v in asdict(self.profile).items()})
        data.update(self.metrics)
        if self.telemetry:
            data["telemetry"] = dict(self.telemetry)
        return data


def telemetry_for_row(
    row: BenchmarkRow,
    *,
    name: str | None = None,
    source: str = "tessera_superbench",
    graph_hash: str | None = None,
    schedule_hash: str | None = None,
    kernel_id: str | None = None,
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build the shared Tessera telemetry event for a benchmark row."""

    latency_ms = row.metrics.get("latency_ms") or row.profile.cpu_wall_ms or row.profile.kernel_elapsed_ms
    throughput = row.metrics.get("throughput_flops")
    tflops = float(throughput) / 1.0e12 if throughput is not None else None
    bytes_per_s = row.metrics.get("bytes_per_s") or row.metrics.get("bandwidth_Bps")
    bandwidth_gbps = float(bytes_per_s) / 1.0e9 if bytes_per_s is not None else None
    return make_event(
        name or row.operator.name,
        source=source,
        op=_telemetry_op(row.operator.name),
        shape=None,
        dtype=row.operator.dtype,
        arch=row.operator.target,
        graph_hash=graph_hash or row.artifact_levels.artifact_hash,
        schedule_hash=schedule_hash,
        kernel_id=kernel_id or row.operator.name,
        latency_ms=float(latency_ms) if latency_ms is not None else None,
        tflops=tflops,
        bandwidth_gbps=bandwidth_gbps,
        memory_bytes=row.profile.memory_bytes,
        status=_telemetry_status(row.runtime_status),
        metadata={
            "compiler_path": row.compiler_path.value,
            "runtime_status": row.runtime_status.value,
            "execution_kind": row.execution_kind.value,
            "reason": row.reason,
            "shape_signature": row.operator.shape,
            **dict(metadata or {}),
        },
    )


def _telemetry_op(name: str) -> str:
    return {
        "gemm": "matmul",
        "conv2d_nhwc": "conv2d",
        "flash_attn": "flash_attention",
        "all_reduce": "all_reduce",
    }.get(name, name)


def _telemetry_status(status: RuntimeStatus) -> str:
    if status == RuntimeStatus.EXECUTABLE:
        return "ok"
    if status == RuntimeStatus.ARTIFACT_ONLY:
        return "unmeasured"
    if status == RuntimeStatus.MISSING_BACKEND:
        return "backend_unavailable"
    return status.value
