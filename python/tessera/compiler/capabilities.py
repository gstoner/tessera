"""Shared compiler capability registry for targets, ops, and runtime status."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Optional

from .op_catalog import GRAPH_OP_TO_SPEC, LEGACY_GRAPH_OP_ALIASES, canonical_graph_op_name


CAPABILITY_REGISTRY_VERSION = "tessera.capabilities.v1"
RuntimeStatus = str


@dataclass(frozen=True)
class OpCapability:
    op_name: str
    runtime_status: RuntimeStatus
    dtypes: tuple[str, ...] = ()
    ranks: tuple[int, ...] = ()
    layouts: tuple[str, ...] = ("row_major",)
    reason: str = ""

    @property
    def executable(self) -> bool:
        return self.runtime_status == "ready"


@dataclass(frozen=True)
class TargetCapability:
    name: str
    aliases: tuple[str, ...]
    family: str
    runtime_backend: str
    default_runtime_status: RuntimeStatus
    supported_ops: Mapping[str, OpCapability] = field(default_factory=dict)
    supported_dtypes: tuple[str, ...] = ()
    features: tuple[str, ...] = ()
    reason: str = ""

    def op(self, op_name: str) -> OpCapability:
        canonical = canonical_op(op_name)
        return self.supported_ops.get(canonical, OpCapability(
            op_name=canonical,
            runtime_status=self.default_runtime_status,
            dtypes=self.supported_dtypes,
            reason=self.reason or f"{canonical} uses {self.default_runtime_status} on {self.name}",
        ))


@dataclass(frozen=True)
class CapabilityResult:
    target: str
    op_name: str
    supported: bool
    runtime_status: RuntimeStatus
    reason: str = ""
    capability_version: str = CAPABILITY_REGISTRY_VERSION


def canonical_op(op_name: str) -> str:
    if op_name.startswith("tessera.ops."):
        op_name = op_name.removeprefix("tessera.ops.")
    if not op_name.startswith("tessera."):
        spec = GRAPH_OP_TO_SPEC.get(op_name)
        if spec is not None:
            op_name = spec.graph_name
        else:
            from .op_catalog import graph_name_for
            op_name = graph_name_for(op_name) or op_name
    return canonical_graph_op_name(LEGACY_GRAPH_OP_ALIASES.get(op_name, op_name))


def normalize_target(target: object = "cpu") -> str:
    if target is None:
        return "cpu"
    if isinstance(target, str):
        normalized = target.lower().replace("-", "_")
        normalized = _ALIASES.get(normalized, normalized)
        if normalized not in TARGET_CAPABILITIES:
            allowed = sorted(TARGET_CAPABILITIES)
            raise ValueError(f"unsupported Tessera target {target!r}; expected one of {allowed}")
        return normalized
    from .gpu_target import GPUTargetProfile, ISA
    if isinstance(target, GPUTargetProfile):
        if target.isa >= ISA.SM_120:
            return "nvidia_sm120"
        if target.isa >= ISA.SM_100:
            return "nvidia_sm100"
        if target.isa >= ISA.SM_90:
            return "nvidia_sm90"
        return "nvidia_sm80"
    raise TypeError(f"unsupported Tessera target object {type(target)!r}")


def get_target_capability(target: object = "cpu") -> TargetCapability:
    return TARGET_CAPABILITIES[normalize_target(target)]


def supports_op(
    target: object,
    op: str,
    *,
    dtype: Optional[str] = None,
    rank: Optional[int] = None,
) -> CapabilityResult:
    target_name = normalize_target(target)
    cap = TARGET_CAPABILITIES[target_name]
    op_cap = cap.op(op)
    dtype_ok = dtype is None or not op_cap.dtypes or dtype in op_cap.dtypes
    rank_ok = rank is None or not op_cap.ranks or rank in op_cap.ranks
    supported = op_cap.runtime_status != "unsupported" and dtype_ok and rank_ok
    reason = op_cap.reason
    if not dtype_ok:
        reason = f"dtype {dtype!r} is not supported for {op_cap.op_name} on {target_name}"
    elif not rank_ok:
        reason = f"rank {rank!r} is not supported for {op_cap.op_name} on {target_name}"
    return CapabilityResult(
        target=target_name,
        op_name=op_cap.op_name,
        supported=supported,
        runtime_status=op_cap.runtime_status if supported else "unsupported",
        reason=reason,
    )


def runtime_status(target: object, op: str | None = None) -> RuntimeStatus:
    cap = get_target_capability(target)
    if op is None:
        return cap.default_runtime_status
    return supports_op(target, op).runtime_status


def target_aliases() -> dict[str, str]:
    return dict(_ALIASES)


def _ops(status: RuntimeStatus, names: tuple[str, ...], *, reason: str = "", dtypes: tuple[str, ...] = ("fp32", "f32")) -> dict[str, OpCapability]:
    return {
        canonical_op(name): OpCapability(canonical_op(name), status, dtypes=dtypes, reason=reason)
        for name in names
    }


_CPU_OPS = tuple(sorted(GRAPH_OP_TO_SPEC))
_APPLE_GPU_READY = ("tessera.matmul", "tessera.flash_attn", "tessera.softmax", "tessera.softmax_safe", "tessera.gelu", "tessera.rope")
_NVIDIA_ARTIFACT = ("tessera.matmul", "tessera.flash_attn", "tessera.gelu", "tessera.softmax", "tessera.softmax_safe")


TARGET_CAPABILITIES: dict[str, TargetCapability] = {
    "cpu": TargetCapability(
        name="cpu",
        aliases=("cpu", "x86", "x86_64"),
        family="cpu",
        runtime_backend="numpy",
        default_runtime_status="ready",
        supported_ops=_ops("ready", _CPU_OPS, reason="CPU reference/runtime path is available"),
        supported_dtypes=("fp32", "f32", "fp16", "bf16"),
        features=("reference_execution", "host_runtime"),
    ),
    "nvidia_sm80": TargetCapability(
        name="nvidia_sm80",
        aliases=("sm80", "sm_80"),
        family="nvidia",
        runtime_backend="cuda",
        default_runtime_status="artifact_only",
        supported_ops=_ops("artifact_only", _NVIDIA_ARTIFACT, reason="NVIDIA target artifact exists; executable smoke is hardware-gated"),
        supported_dtypes=("bf16", "fp16", "fp32", "f32"),
        features=("wmma", "cuda_artifacts"),
    ),
    "nvidia_sm90": TargetCapability(
        name="nvidia_sm90",
        aliases=("cuda", "nvidia", "gpu", "sm90", "sm_90", "sm90a", "sm_90a", "hopper"),
        family="nvidia",
        runtime_backend="cuda",
        default_runtime_status="artifact_only",
        supported_ops=_ops("artifact_only", _NVIDIA_ARTIFACT, reason="NVIDIA SM90 WGMMA/TMA artifact exists; executable smoke is hardware-gated"),
        supported_dtypes=("bf16", "fp16", "fp32", "f32"),
        features=("wgmma", "tma", "cuda_artifacts"),
    ),
    "nvidia_sm100": TargetCapability(
        name="nvidia_sm100",
        aliases=("sm100", "sm_100", "sm100a", "sm_100a", "blackwell"),
        family="nvidia",
        runtime_backend="cuda",
        default_runtime_status="artifact_only",
        supported_ops=_ops("artifact_only", _NVIDIA_ARTIFACT, reason="Blackwell target artifact exists; executable smoke is hardware-gated"),
        supported_dtypes=("bf16", "fp16", "fp32", "f32", "nvfp4"),
        features=("tcgen05", "tmem", "cuda_artifacts"),
    ),
    "nvidia_sm120": TargetCapability(
        name="nvidia_sm120",
        aliases=("sm120", "sm_120"),
        family="nvidia",
        runtime_backend="cuda",
        default_runtime_status="artifact_only",
        supported_ops=_ops("artifact_only", _NVIDIA_ARTIFACT, reason="SM120 target artifact exists; executable smoke is hardware-gated"),
        supported_dtypes=("bf16", "fp16", "fp32", "f32", "nvfp4"),
        features=("tcgen05", "tmem", "cuda_artifacts"),
    ),
    "rocm": TargetCapability(
        name="rocm",
        aliases=("rocm", "amd", "hip"),
        family="rocm",
        runtime_backend="hip",
        default_runtime_status="artifact_only",
        supported_ops=_ops("artifact_only", ("tessera.matmul", "tessera.gelu", "tessera.softmax"), reason="ROCm MFMA artifact exists; HIP execution remains gated"),
        supported_dtypes=("bf16", "fp16", "fp32", "f32"),
        features=("mfma", "hip_artifacts"),
    ),
    "metalium": TargetCapability(
        name="metalium",
        aliases=("metalium", "tt_metalium", "tt"),
        family="metalium",
        runtime_backend="metalium",
        default_runtime_status="artifact_only",
        supported_ops=_ops("artifact_only", ("tessera.matmul",), reason="Metalium target contract is scaffolded"),
        supported_dtypes=("bf16", "fp32", "f32"),
        features=("dma_contract",),
    ),
    "apple_cpu": TargetCapability(
        name="apple_cpu",
        aliases=("macos_cpu", "m_series_cpu"),
        family="apple",
        runtime_backend="accelerate",
        default_runtime_status="ready",
        supported_ops=_ops("ready", _CPU_OPS, reason="Apple CPU uses Accelerate where available and reference fallback otherwise"),
        supported_dtypes=("fp32", "f32", "bf16", "fp16"),
        features=("accelerate", "reference_fallback"),
    ),
    "apple_gpu": TargetCapability(
        name="apple_gpu",
        aliases=("apple", "mac", "macos_gpu", "m_series_gpu"),
        family="apple",
        runtime_backend="metal",
        default_runtime_status="artifact_only",
        supported_ops={
            **_ops("ready", _APPLE_GPU_READY, reason="Apple GPU runtime shim supports this single-op smoke path"),
            **_ops("artifact_only", ("tessera.moe", "tessera.add", "tessera.mul"), reason="Apple GPU target contract exists but native execution is not wired for this op"),
        },
        supported_dtypes=("fp32", "f32"),
        features=("metal", "mps", "msl"),
    ),
}

_ALIASES = {
    alias: target.name
    for target in TARGET_CAPABILITIES.values()
    for alias in target.aliases
}


__all__ = [
    "CAPABILITY_REGISTRY_VERSION",
    "CapabilityResult",
    "OpCapability",
    "TargetCapability",
    "canonical_op",
    "get_target_capability",
    "normalize_target",
    "runtime_status",
    "supports_op",
    "target_aliases",
]
