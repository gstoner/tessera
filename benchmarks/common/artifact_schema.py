"""Standard benchmark row schema for compiler/runtime visibility."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any


class CompilerPath(str, Enum):
    TESSERA_JIT_CPU = "tessera_jit_cpu"
    GRAPH_IR_ONLY = "graph_ir_only"
    RUNTIME_UNAVAILABLE = "runtime_unavailable"
    REFERENCE = "reference"
    ARTIFACT_ONLY = "artifact_only"
    UNSUPPORTED = "unsupported"


class RuntimeStatus(str, Enum):
    EXECUTABLE = "executable"
    SKIPPED = "skipped"
    UNSUPPORTED = "unsupported"
    MISSING_BACKEND = "missing_backend"


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
    reason: str = ""

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["compiler_path"] = self.compiler_path.value
        data["runtime_status"] = self.runtime_status.value
        return data

    def flat_dict(self) -> dict[str, Any]:
        data = {
            "operator": self.operator.name,
            "dtype": self.operator.dtype,
            "shape": self.operator.shape,
            "target": self.operator.target,
            "compiler_path": self.compiler_path.value,
            "runtime_status": self.runtime_status.value,
            "reason": self.reason,
        }
        data.update({f"artifact_{k}": v for k, v in asdict(self.artifact_levels).items()})
        data.update({f"correctness_{k}": v for k, v in asdict(self.correctness).items()})
        data.update({f"profile_{k}": v for k, v in asdict(self.profile).items()})
        data.update(self.metrics)
        return data
