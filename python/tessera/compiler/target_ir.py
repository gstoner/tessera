"""Target IR object model and Tile IR lowering.

Target IR is the backend-specific contract layer below Tile IR. This module
keeps hardware-free CPU/x86, NVIDIA/CUDA, Apple Silicon, and ROCm artifacts
object-backed and verifiable while preserving the textual MLIR inspection
surface used by the Python compiler tests.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Iterable, Optional

from ..diagnostics import DiagnosticLevel, DiagnosticWhere, TesseraDiagnostic, TesseraErrorCode
from .tile_ir import TileIRModule, TileIRVerificationError, TileOp


APPLE_CPU_TARGET = "apple_cpu"
APPLE_GPU_TARGET = "apple_gpu"
CPU_TARGET = "cpu"
ROCM_TARGET = "rocm"
NVIDIA_TARGETS = {"nvidia_sm80", "nvidia_sm90", "nvidia_sm100", "nvidia_sm120"}
SUPPORTED_TARGETS = {APPLE_CPU_TARGET, APPLE_GPU_TARGET, CPU_TARGET, ROCM_TARGET, *NVIDIA_TARGETS}


def _diagnostic_level(severity: str) -> DiagnosticLevel:
    return {
        "fatal": DiagnosticLevel.FATAL,
        "error": DiagnosticLevel.ERROR,
        "warning": DiagnosticLevel.WARNING,
        "info": DiagnosticLevel.INFO,
        "note": DiagnosticLevel.NOTE,
    }.get(severity.lower(), DiagnosticLevel.ERROR)


@dataclass(frozen=True)
class TargetIRDiagnostic:
    severity: str
    message: str
    code: str = "TARGET_IR"

    def format(self) -> str:
        structured = self.to_tessera_diagnostic()
        return f"{structured.level.value.upper()} [{structured.code.value}] [{self.code}]: {self.message}\n  where: {structured.where}"

    def to_tessera_diagnostic(self) -> TesseraDiagnostic:
        return TesseraDiagnostic(
            level=_diagnostic_level(self.severity),
            message=self.message,
            code=TesseraErrorCode.TARGET_CODEGEN,
            where=DiagnosticWhere(ir_level="target-ir", pass_name="verifier"),
            hints=["inspect Target IR, target profile, and backend feature requirements"],
        )


@dataclass(frozen=True)
class TargetIRVerificationResult:
    diagnostics: tuple[TargetIRDiagnostic, ...] = ()

    @property
    def ok(self) -> bool:
        return not any(d.severity == "error" for d in self.diagnostics)

    def format(self) -> str:
        return "\n".join(d.format() for d in self.diagnostics)

    def structured_diagnostics(self) -> tuple[TesseraDiagnostic, ...]:
        return tuple(d.to_tessera_diagnostic() for d in self.diagnostics)


class TargetIRVerificationError(ValueError):
    pass


@dataclass
class TargetOp:
    op_name: str
    attrs: dict[str, Any] = field(default_factory=dict)
    operands: list[str] = field(default_factory=list)
    result: Optional[str] = None

    def to_mlir(self, indent: str = "  ") -> str:
        result_text = f"%{self.result} = " if self.result else ""
        operands = ", ".join(self.operands)
        return f"{indent}{result_text}\"{self.op_name}\"({operands}) {_format_attr_dict(self.attrs)} : () -> ()"


@dataclass
class TargetFunction:
    name: str
    body: list[TargetOp] = field(default_factory=list)
    target: str = "cpu"

    def to_mlir(self, indent: str = "  ") -> str:
        func_op = {
            APPLE_CPU_TARGET: "tessera_apple.cpu.func",
            APPLE_GPU_TARGET: "tessera_apple.gpu.func",
            CPU_TARGET: "tessera.cpu.func",
            ROCM_TARGET: "tessera_rocm.func",
        }.get(self.target, "tessera_nvidia.func" if self.target in NVIDIA_TARGETS else "tessera.target.func")
        lines = [f"{indent}\"{func_op}\"() ({{"]
        for op in self.body:
            lines.append(op.to_mlir(indent + "  "))
        lines.append(f"{indent}}}) {{sym_name = {json.dumps(self.name)}}} : () -> ()")
        return "\n".join(lines)


@dataclass
class TargetIRModule:
    functions: list[TargetFunction] = field(default_factory=list)
    attrs: dict[str, Any] = field(default_factory=lambda: {"tessera.ir.level": "target"})

    def verify(self) -> TargetIRVerificationResult:
        return TargetIRVerifier().verify_module(self)

    def to_mlir(self, *, verify: bool = True) -> str:
        if verify:
            result = self.verify()
            if not result.ok:
                raise TargetIRVerificationError(result.format())
        lines = [f"module attributes {_format_attr_dict(self.attrs)} {{"]
        for fn in self.functions:
            lines.append(fn.to_mlir())
        lines.append("}")
        return "\n".join(lines)


class TargetIRVerifier:
    def verify_module(self, module: TargetIRModule) -> TargetIRVerificationResult:
        diagnostics: list[TargetIRDiagnostic] = []
        target = module.attrs.get("target")
        if target not in SUPPORTED_TARGETS:
            diagnostics.append(TargetIRDiagnostic("error", f"unsupported target {target!r}", "TARGET_IR_TARGET"))
        for fn in module.functions:
            diagnostics.extend(self.verify_function(fn, target=str(target)).diagnostics)
        return TargetIRVerificationResult(tuple(diagnostics))

    def verify_function(self, fn: TargetFunction, *, target: str) -> TargetIRVerificationResult:
        diagnostics: list[TargetIRDiagnostic] = []
        for op in fn.body:
            if target == APPLE_CPU_TARGET:
                self._verify_apple_cpu_op(op, diagnostics)
            elif target == APPLE_GPU_TARGET:
                self._verify_apple_gpu_op(op, diagnostics)
            elif target == CPU_TARGET:
                self._verify_cpu_op(op, diagnostics)
            elif target == ROCM_TARGET:
                self._verify_rocm_op(op, diagnostics)
            elif target in NVIDIA_TARGETS:
                self._verify_nvidia_op(op, diagnostics)
        return TargetIRVerificationResult(tuple(diagnostics))

    def _verify_cpu_op(self, op: TargetOp, diagnostics: list[TargetIRDiagnostic]) -> None:
        if not op.op_name.startswith("tessera.cpu."):
            diagnostics.append(TargetIRDiagnostic("error", f"invalid CPU op {op.op_name!r}", "TARGET_IR_CPU_OP"))
            return
        self._require(op, diagnostics, "source", "result", "ordinal", "abi")

    def _verify_apple_cpu_op(self, op: TargetOp, diagnostics: list[TargetIRDiagnostic]) -> None:
        if op.op_name == "tessera_apple.diagnostic":
            self._require(op, diagnostics, "severity", "reason")
            return
        if not op.op_name.startswith("tessera_apple.cpu."):
            diagnostics.append(TargetIRDiagnostic("error", f"invalid Apple CPU op {op.op_name!r}", "TARGET_IR_APPLE_CPU_OP"))
        self._require(op, diagnostics, "framework", "abi", "dtype")

    def _verify_apple_gpu_op(self, op: TargetOp, diagnostics: list[TargetIRDiagnostic]) -> None:
        if op.op_name == "tessera_apple.diagnostic":
            self._require(op, diagnostics, "severity", "reason")
            return
        if op.op_name == "tessera_apple.gpu.metal_kernel":
            self._require(op, diagnostics, "kernel", "framework", "status", "dtype")
        elif op.op_name == "tessera_apple.gpu.dispatch":
            self._require(op, diagnostics, "queue", "artifact", "execution_mode")
        else:
            diagnostics.append(TargetIRDiagnostic("error", f"invalid Apple GPU op {op.op_name!r}", "TARGET_IR_APPLE_GPU_OP"))

    def _verify_rocm_op(self, op: TargetOp, diagnostics: list[TargetIRDiagnostic]) -> None:
        if op.op_name == "tessera.target.diagnostic":
            self._require(op, diagnostics, "target", "severity", "reason")
            return
        if not op.op_name.startswith("tessera_rocm."):
            diagnostics.append(TargetIRDiagnostic("error", f"invalid ROCm op {op.op_name!r}", "TARGET_IR_ROCM_OP"))
            return
        if op.op_name == "tessera_rocm.mfma":
            self._require(op, diagnostics, "arch", "shape", "accum")
        elif op.op_name == "tessera_rocm.async_copy":
            self._require(op, diagnostics, "src_space", "dst_space", "bytes")
        elif op.op_name == "tessera_rocm.wait":
            self._require(op, diagnostics, "ordinal")
        elif op.op_name == "tessera_rocm.elementwise":
            self._require(op, diagnostics, "arch")

    def _verify_nvidia_op(self, op: TargetOp, diagnostics: list[TargetIRDiagnostic]) -> None:
        if not op.op_name.startswith("tessera_nvidia."):
            diagnostics.append(TargetIRDiagnostic("error", f"invalid NVIDIA op {op.op_name!r}", "TARGET_IR_NVIDIA_OP"))
            return
        if op.op_name == "tessera_nvidia.wgmma":
            self._require(op, diagnostics, "arch", "shape", "dtype_ab", "dtype_c", "warpgroup")
        elif op.op_name == "tessera_nvidia.tma_async_copy":
            self._require(op, diagnostics, "arch", "src_space", "dst_space", "bytes")
        elif op.op_name == "tessera_nvidia.mbarrier":
            self._require(op, diagnostics, "arch", "scope", "ordinal")
        elif op.op_name == "tessera_nvidia.tmem_alloc":
            self._require(op, diagnostics, "arch", "columns")
        elif op.op_name == "tessera_nvidia.tcgen05_mma":
            self._require(op, diagnostics, "arch", "shape", "accum", "cta_group")
        elif op.op_name == "tessera_nvidia.cuda_kernel":
            self._require(op, diagnostics, "arch", "kernel", "status")

    def _require(self, op: TargetOp, diagnostics: list[TargetIRDiagnostic], *attrs: str) -> None:
        missing = [attr for attr in attrs if attr not in op.attrs]
        if missing:
            diagnostics.append(TargetIRDiagnostic("error", f"{op.op_name} missing attrs: {', '.join(missing)}", "TARGET_IR_MISSING_ATTR"))


def lower_tile_to_target_ir(tile_module: TileIRModule, *, target_kind: str) -> TargetIRModule:
    tile_result = tile_module.verify()
    if not tile_result.ok:
        raise TileIRVerificationError(tile_result.format())
    if target_kind not in SUPPORTED_TARGETS:
        raise ValueError(f"target_ir does not support target {target_kind!r}")
    attrs = {"tessera.ir.level": "target", "target": target_kind}
    if target_kind == CPU_TARGET:
        attrs.update({"arch": "x86_64", "execution_mode": "numpy"})
    elif target_kind in NVIDIA_TARGETS:
        attrs["arch"] = _nvidia_arch(target_kind)
    elif target_kind == ROCM_TARGET:
        attrs["arch"] = "gfx90a"
    elif target_kind == APPLE_CPU_TARGET:
        attrs.update({"arch": "arm64-apple-silicon", "execution_mode": "cpu_accelerate"})
    else:
        attrs.update({"arch": "apple-metal", "execution_mode": "metal_artifact"})
    target_module = TargetIRModule(attrs=attrs)
    for tile_fn in tile_module.functions:
        target_module.functions.append(TargetFunction(
            name=tile_fn.name,
            target=target_kind,
            body=_lower_tile_ops(tile_fn.body, target_kind=target_kind),
        ))
    return target_module


def _lower_tile_ops(ops: Iterable[TileOp], *, target_kind: str) -> list[TargetOp]:
    lowered: list[TargetOp] = []
    for tile_op in _flatten_tile_ops(ops):
        if target_kind == ROCM_TARGET:
            lowered.extend(_lower_rocm_op(tile_op))
        elif target_kind == APPLE_CPU_TARGET:
            lowered.extend(_lower_apple_cpu_op(tile_op))
        elif target_kind == APPLE_GPU_TARGET:
            lowered.extend(_lower_apple_gpu_op(tile_op))
        elif target_kind == CPU_TARGET:
            lowered.extend(_lower_cpu_op(tile_op))
        elif target_kind in NVIDIA_TARGETS:
            lowered.extend(_lower_nvidia_op(tile_op, target_kind=target_kind))
    return lowered


def _flatten_tile_ops(ops: Iterable[TileOp]) -> Iterable[TileOp]:
    for op in ops:
        if op.op_name == "tile.group":
            yield from _flatten_tile_ops(op.body)
        else:
            yield op
            yield from _flatten_tile_ops(op.body)


def _lower_rocm_op(op: TileOp) -> list[TargetOp]:
    source = str(op.attrs.get("source", _source_from_tile_op(op)))
    base = _base_attrs(op, target="rocm")
    if op.op_name == "tile.mma":
        return [
            TargetOp("tessera_rocm.mfma", {**base, "arch": "gfx90a", "shape": "m16n16k16", "accum": "f32"}),
            TargetOp("tessera_rocm.async_copy", {**base, "src_space": "global", "dst_space": "lds", "bytes": 16}),
            TargetOp("tessera_rocm.wait", {"ordinal": base["ordinal"]}),
        ]
    if source == "tessera.flash_attn" or op.op_name.startswith("tessera.attn."):
        return [TargetOp("tessera.target.diagnostic", {
            **base,
            "target": "rocm",
            "severity": "unsupported",
            "reason": "flash_attn target kernel contract is not implemented for ROCm in this phase",
        })]
    if source.startswith("tessera.kv_cache.") or op.op_name == "tile.kv_cache":
        return [TargetOp("tessera.target.diagnostic", {
            **base,
            "target": "rocm",
            "severity": "unsupported",
            "reason": "KV-cache target lowering is not implemented for ROCm in this phase",
        })]
    if op.op_name == "tile.async_copy":
        return [
            TargetOp("tessera_rocm.async_copy", {**base, "src_space": "global", "dst_space": "lds", "bytes": 16}),
            TargetOp("tessera_rocm.wait", {"ordinal": base["ordinal"]}),
        ]
    if op.op_name.startswith("tessera.queue.") or op.op_name == "tile.wait_async":
        return []
    return [
        TargetOp("tessera_rocm.elementwise", {**base, "arch": "gfx90a"}),
        TargetOp("tessera_rocm.async_copy", {**base, "src_space": "global", "dst_space": "lds", "bytes": 16}),
        TargetOp("tessera_rocm.wait", {"ordinal": base["ordinal"]}),
    ]


def _lower_cpu_op(op: TileOp) -> list[TargetOp]:
    if op.op_name.startswith("tessera.queue.") or op.op_name in {"tile.async_copy", "tile.wait_async"}:
        return []
    source = str(op.attrs.get("source", _source_from_tile_op(op)))
    base = _base_attrs(op)
    return [TargetOp(_cpu_target_op_name(source), {**base, "abi": "numpy"})]


def _lower_nvidia_op(op: TileOp, *, target_kind: str) -> list[TargetOp]:
    if op.op_name.startswith("tessera.queue.") or op.op_name in {"tile.async_copy", "tile.wait_async"}:
        return []
    source = str(op.attrs.get("source", _source_from_tile_op(op)))
    base = _base_attrs(op)
    arch = _nvidia_arch(target_kind)
    if op.op_name == "tile.mma":
        if target_kind in {"nvidia_sm100", "nvidia_sm120"}:
            return [
                TargetOp("tessera_nvidia.tmem_alloc", {**base, "arch": arch, "columns": 128}),
                TargetOp("tessera_nvidia.tcgen05_mma", {
                    **base,
                    "arch": arch,
                    "shape": "m128n128k32",
                    "accum": "tmem_f32",
                    "cta_group": 2,
                    "block_scaled": True,
                }),
            ]
        return [
            TargetOp("tessera_nvidia.wgmma", {
                **base,
                "arch": arch,
                "shape": "m64n64k16",
                "dtype_ab": "bf16",
                "dtype_c": "f32",
                "warpgroup": 4,
            }),
            TargetOp("tessera_nvidia.tma_async_copy", {
                **base,
                "arch": arch,
                "src_space": "global",
                "dst_space": "shared",
                "bytes": 16,
            }),
            TargetOp("tessera_nvidia.mbarrier", {"ordinal": base["ordinal"], "arch": arch, "scope": "cta"}),
        ]
    kernel = "flash_attn_contract" if source == "tessera.flash_attn" or op.op_name.startswith("tessera.attn.") else "elementwise_contract"
    return [TargetOp("tessera_nvidia.cuda_kernel", {**base, "arch": arch, "kernel": kernel, "status": "artifact_only"})]


def _lower_apple_cpu_op(op: TileOp) -> list[TargetOp]:
    source = str(op.attrs.get("source", _source_from_tile_op(op)))
    base = _base_attrs(op)
    if source.startswith("tessera.kv_cache.") or op.op_name == "tile.kv_cache":
        return [TargetOp("tessera_apple.diagnostic", {
            **base,
            "severity": "unsupported",
            "reason": "KV-cache target lowering is not implemented for Apple CPU in this phase",
        })]
    if op.op_name == "tile.mma":
        return [TargetOp("tessera_apple.cpu.accelerate_gemm", {**base, "framework": "Accelerate", "abi": "cblas_sgemm", "dtype": "f32"})]
    if source == "tessera.moe":
        return [TargetOp("tessera_apple.cpu.moe_solver", {
            **base,
            "framework": "Accelerate",
            "abi": "moe_top1_round_robin",
            "dtype": "f32",
            "routing": "deterministic_top1",
        })]
    if source in {"tessera.softmax", "tessera.softmax_safe"}:
        return [TargetOp("tessera_apple.cpu.vector_reduce", {**base, "framework": "Accelerate", "abi": "vDSP", "dtype": "f32"})]
    if source == "tessera.rope":
        return [TargetOp("tessera_apple.cpu.vector_op", {**base, "framework": "Accelerate", "abi": "vecLib", "pattern": "rotary_pairs", "dtype": "f32"})]
    if op.op_name.startswith("tessera.queue.") or op.op_name in {"tile.async_copy", "tile.wait_async"}:
        return []
    return [TargetOp("tessera_apple.cpu.vector_op", {**base, "framework": "Accelerate", "abi": "vecLib", "dtype": "f32"})]


def _lower_apple_gpu_op(op: TileOp) -> list[TargetOp]:
    source = str(op.attrs.get("source", _source_from_tile_op(op)))
    base = _base_attrs(op)
    if source.startswith("tessera.kv_cache.") or op.op_name == "tile.kv_cache":
        return [TargetOp("tessera_apple.diagnostic", {
            **base,
            "severity": "unsupported",
            "reason": "KV-cache target lowering is not implemented for Apple GPU in this phase",
        })]
    if op.op_name.startswith("tessera.queue.") or op.op_name in {"tile.async_copy", "tile.wait_async"}:
        return []
    kernel, framework, extra = _apple_gpu_kernel_contract(source)
    return [
        TargetOp("tessera_apple.gpu.metal_kernel", {**base, "kernel": kernel, "framework": framework, "dtype": "f32", **extra}),
        TargetOp("tessera_apple.gpu.dispatch", {
            "ordinal": base["ordinal"],
            "queue": "MTLCommandQueue",
            "artifact": "metallib",
            "execution_mode": "metal_artifact",
        }),
    ]


def _apple_gpu_kernel_contract(source: str) -> tuple[str, str, dict[str, Any]]:
    if source == "tessera.flash_attn":
        return "flash_attn_contract", "Metal", {"status": "artifact_only", "grid": "bhn", "threadgroup": "64x1x1", "temporary_memory": "scores_lse"}
    if source in {"tessera.matmul", "tessera.gemm"}:
        return "matmul_contract", "MPSGraph", {"status": "artifact_only", "grid": "mn_tiles", "threadgroup": "16x16x1", "temporary_memory": "none"}
    if source in {"tessera.softmax", "tessera.softmax_safe"}:
        return "softmax_contract", "MPSGraph", {"status": "artifact_only", "grid": "rows", "threadgroup": "256x1x1", "temporary_memory": "row_max_sum"}
    if source == "tessera.gelu":
        return "gelu_contract", "MPSGraph", {"status": "artifact_only", "grid": "elements", "threadgroup": "256x1x1", "temporary_memory": "none"}
    if source == "tessera.rope":
        return "rope_contract", "Metal", {"status": "artifact_only", "grid": "tokens_heads", "threadgroup": "128x1x1", "temporary_memory": "none"}
    if source == "tessera.moe":
        return "moe_contract", "MPSGraph", {"status": "artifact_only", "grid": "tokens_experts", "threadgroup": "128x1x1", "temporary_memory": "routing"}
    return "elementwise_contract", "Metal", {"status": "artifact_only", "grid": "elements", "threadgroup": "256x1x1", "temporary_memory": "none"}


def _base_attrs(op: TileOp, *, target: Optional[str] = None) -> dict[str, Any]:
    attrs = {
        "source": op.attrs.get("source", _source_from_tile_op(op)),
        "result": op.attrs.get("result", "value"),
        "ordinal": int(op.attrs.get("ordinal", 0)),
    }
    if target is not None:
        attrs["target"] = target
    return attrs


def _source_from_tile_op(op: TileOp) -> str:
    if op.op_name == "tile.mma":
        return "tessera.matmul"
    if op.op_name.startswith("tessera.attn."):
        return "tessera.flash_attn"
    if op.op_name.startswith("tile."):
        return "tessera." + op.op_name.removeprefix("tile.")
    return op.op_name


def _cpu_target_op_name(source: str) -> str:
    bare = source.removeprefix("tessera.").replace(".", "_")
    if source in {"tessera.matmul", "tessera.gemm"}:
        bare = "matmul"
    elif source in {"tessera.conv2d", "tessera.conv2d_nhwc"}:
        bare = "conv2d_nhwc"
    return f"tessera.cpu.{bare}"


def _nvidia_arch(target_kind: str) -> str:
    return {
        "nvidia_sm80": "sm_80",
        "nvidia_sm90": "sm_90a",
        "nvidia_sm100": "sm_100a",
        "nvidia_sm120": "sm_120",
    }[target_kind]


def _format_attr_dict(attrs: dict[str, Any]) -> str:
    if not attrs:
        return "{}"
    return "{" + ", ".join(f"{key} = {_format_attr_value(value)}" for key, value in attrs.items()) + "}"


def _format_attr_value(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, str):
        return json.dumps(value)
    if isinstance(value, int):
        return f"{value} : i64"
    if isinstance(value, float):
        return repr(value)
    if value is None:
        return "none"
    if isinstance(value, list | tuple):
        return "[" + ", ".join(_format_attr_value(item) for item in value) + "]"
    if isinstance(value, dict):
        return json.dumps(json.dumps(value, sort_keys=True))
    return json.dumps(str(value))
