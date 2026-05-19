"""Compiler driver contracts and instrumentation for Tessera frontend lowering.

The driver is the narrow boundary between Python frontend lowering and backend
artifact/codegen paths.  It intentionally records what happened at each layer so
tests and examples can distinguish executable CPU lowering from target-only
artifacts and explicit fallback paths.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

from .graph_ir import GraphIRModule
from .capabilities import CAPABILITY_REGISTRY_VERSION, supports_op
from .matmul_pipeline import CPUPlan, JitDiagnostic, LoweringArtifact, build_cpu_plan, explain_cpu_plan, normalize_target_kind


COMPILER_TRACE_SCHEMA_VERSION = "tessera.compiler.trace.v1"

PIPELINE_BY_TARGET = {
    "cpu": "tessera-lower-to-x86",
    "nvidia_sm80": "tessera-lower-to-gpu",
    "nvidia_sm90": "tessera-lower-to-gpu",
    "nvidia_sm100": "tessera-lower-to-gpu",
    "nvidia_sm120": "tessera-lower-to-gpu",
    "rocm": "tessera-lower-to-rocm",
    "apple_cpu": "tessera-lower-to-apple_cpu",
    "apple_gpu": "tessera-lower-to-apple_gpu",
    "metalium": "tessera-lower-to-metalium",
}


def stable_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class CompileRequest:
    source_origin: str
    function_name: str
    graph_ir: str
    target: str = "cpu"
    pipeline_name: str = ""
    options: Mapping[str, Any] = field(default_factory=dict)
    example_id: str | None = None

    def __post_init__(self) -> None:
        target = normalize_target_kind(self.target)
        object.__setattr__(self, "target", target)
        if not self.pipeline_name:
            object.__setattr__(self, "pipeline_name", PIPELINE_BY_TARGET.get(target, "tessera-target-artifact"))

    @property
    def graph_hash(self) -> str:
        return stable_hash(self.graph_ir)

    def to_dict(self) -> dict[str, Any]:
        return {
            "source_origin": self.source_origin,
            "function_name": self.function_name,
            "graph_ir_hash": self.graph_hash,
            "target": self.target,
            "pipeline_name": self.pipeline_name,
            "options": dict(self.options),
            "example_id": self.example_id,
        }


@dataclass(frozen=True)
class CompileTraceEvent:
    pass_name: str
    target: str
    input_hash: str
    output_hash: str
    elapsed_ms: float
    status: str
    diagnostic_count: int = 0
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": COMPILER_TRACE_SCHEMA_VERSION,
            "pass_name": self.pass_name,
            "target": self.target,
            "input_hash": self.input_hash,
            "output_hash": self.output_hash,
            "elapsed_ms": self.elapsed_ms,
            "status": self.status,
            "diagnostic_count": self.diagnostic_count,
            "metadata": dict(self.metadata),
        }

    def to_chrome_trace_event(self, *, pid: int = 1, tid: int = 0) -> dict[str, Any]:
        return {
            "name": self.pass_name,
            "cat": "tessera.compiler",
            "ph": "X",
            "pid": pid,
            "tid": tid,
            "ts": 0,
            "dur": self.elapsed_ms * 1000.0,
            "args": self.to_dict(),
        }


@dataclass(frozen=True)
class ToolInvocation:
    tool: str
    pipeline_name: str
    available: bool
    status: str
    elapsed_ms: float = 0.0
    stdout_hash: str = ""
    stderr: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "tool": self.tool,
            "pipeline_name": self.pipeline_name,
            "available": self.available,
            "status": self.status,
            "elapsed_ms": self.elapsed_ms,
            "stdout_hash": self.stdout_hash,
            "stderr": self.stderr,
        }


@dataclass(frozen=True)
class CompileArtifactBundle:
    request: CompileRequest
    graph: LoweringArtifact
    schedule: LoweringArtifact | None = None
    tile: LoweringArtifact | None = None
    target_ir: LoweringArtifact | None = None
    backend: LoweringArtifact | None = None
    executable: bool = False
    runtime_status: str = "unsupported"
    execution_mode: str = "artifact_only"
    execution_kind: str = "artifact_only"
    diagnostics: tuple[JitDiagnostic, ...] = ()
    trace_events: tuple[CompileTraceEvent, ...] = ()
    tool_invocations: tuple[ToolInvocation, ...] = ()
    cpu_plan: CPUPlan | None = None

    def artifact(self, level: str) -> LoweringArtifact | None:
        if level == "graph":
            return self.graph
        if level == "schedule":
            return self.schedule
        if level == "tile":
            return self.tile
        if level == "target":
            return self.target_ir
        if level == "backend":
            return self.backend
        return None

    def lowering_artifacts(self) -> tuple[LoweringArtifact, ...]:
        return tuple(artifact for artifact in (self.graph, self.schedule, self.tile, self.target_ir, self.backend) if artifact is not None)

    @property
    def artifact_hashes(self) -> dict[str, str]:
        return {artifact.level: stable_hash(artifact.text) for artifact in self.lowering_artifacts()}

    def diagnostics_text(self) -> tuple[str, ...]:
        return tuple(d.format() for d in self.diagnostics)

    def trace_json(self) -> str:
        return json.dumps([event.to_dict() for event in self.trace_events], sort_keys=True)

    def chrome_trace_json(self) -> str:
        return json.dumps({"traceEvents": [event.to_chrome_trace_event(tid=i) for i, event in enumerate(self.trace_events)]}, sort_keys=True)

    def to_metadata(self) -> dict[str, Any]:
        artifact_hashes = self.artifact_hashes
        profiling = {
            "schema": COMPILER_TRACE_SCHEMA_VERSION,
            "op": self.request.function_name,
            "target": self.request.target,
            "graph_hash": self.request.graph_hash,
            "schedule_hash": artifact_hashes.get("schedule"),
            "tile_hash": artifact_hashes.get("tile"),
            "target_hash": artifact_hashes.get("target"),
            "runtime_status": self.runtime_status,
            "execution_mode": self.execution_mode,
            "execution_kind": self.execution_kind,
            "capability_version": CAPABILITY_REGISTRY_VERSION,
        }
        if self.cpu_plan is not None:
            profiling["selected_schedule"] = self.cpu_plan.selected_schedule
        return {
            "target": self.request.target,
            "function_name": self.request.function_name,
            "source_origin": self.request.source_origin,
            "pipeline_name": self.request.pipeline_name,
            "example_id": self.request.example_id,
            "options": dict(self.request.options),
            "artifact_hashes": artifact_hashes,
            "profiling": profiling,
            "executable": self.executable,
            "runtime_status": self.runtime_status,
            "execution_mode": self.execution_mode,
            "execution_kind": self.execution_kind,
            "capability_version": CAPABILITY_REGISTRY_VERSION,
            "capability_reason": _capability_reason(self),
            "selected_schedule": self.cpu_plan.selected_schedule if self.cpu_plan is not None else None,
            "diagnostics": list(self.diagnostics_text()),
            "trace": [event.to_dict() for event in self.trace_events],
            "tool_invocations": [invocation.to_dict() for invocation in self.tool_invocations],
        }


def compile_graph_module(
    module: GraphIRModule,
    *,
    source_origin: str,
    target: object = "cpu",
    cpu_tile: tuple[int, int, int] = (128, 128, 64),
    options: Mapping[str, Any] | None = None,
    example_id: str | None = None,
    enable_tool_validation: bool = True,
) -> CompileArtifactBundle:
    target_kind = normalize_target_kind(target)
    graph_text = module.to_mlir()
    function_name = module.functions[0].name if module.functions else "<unknown>"
    request = CompileRequest(
        source_origin=source_origin,
        function_name=function_name,
        graph_ir=graph_text,
        target=target_kind,
        options=options or {},
        example_id=example_id,
    )

    diagnostics: list[JitDiagnostic] = []
    trace_events: list[CompileTraceEvent] = []
    tool_invocations: list[ToolInvocation] = []

    start = time.perf_counter()
    cpu_plan = build_cpu_plan(module, tile=cpu_tile, target_kind=target_kind)
    diagnostics.append(explain_cpu_plan(module, target=target_kind))
    elapsed_ms = (time.perf_counter() - start) * 1000.0

    output_text = cpu_plan.target_ir if cpu_plan is not None else graph_text
    status = "ok" if cpu_plan is not None else "fallback"
    execution_mode = _execution_mode_for(target_kind, cpu_plan is not None)
    execution_kind = _execution_kind_for(target_kind, cpu_plan)
    trace_events.append(CompileTraceEvent(
        pass_name="python-frontend-artifact-builder",
        target=target_kind,
        input_hash=stable_hash(graph_text),
        output_hash=stable_hash(output_text),
        elapsed_ms=elapsed_ms,
        status=status,
        diagnostic_count=len(diagnostics),
        metadata={"pipeline_name": request.pipeline_name, "execution_mode": execution_mode},
    ))

    if enable_tool_validation:
        invocation, event = _try_validate_with_tessera_opt(request)
        tool_invocations.append(invocation)
        trace_events.append(event)

    graph = LoweringArtifact("graph", graph_text)
    schedule = LoweringArtifact("schedule", cpu_plan.schedule_ir) if cpu_plan is not None else None
    tile = LoweringArtifact("tile", cpu_plan.tile_ir) if cpu_plan is not None else None
    target_artifact = LoweringArtifact("target", cpu_plan.target_ir) if cpu_plan is not None else None
    backend_artifact = _backend_artifact_for(target_kind, cpu_plan)

    executable = cpu_plan is not None and (
        target_kind == "cpu"
        or _is_apple_cpu_accelerate_executable(cpu_plan)
        or _is_apple_gpu_mps_executable(cpu_plan)
    )
    if executable:
        runtime_status = "ready"
    elif cpu_plan is not None:
        runtime_status = "artifact_only"
    else:
        runtime_status = "unsupported"
    if _is_apple_gpu_mps_executable(cpu_plan):
        execution_mode = "metal_runtime"
    execution_kind = _execution_kind_for(target_kind, cpu_plan, executable=executable)

    bundle = CompileArtifactBundle(
        request=request,
        graph=graph,
        schedule=schedule,
        tile=tile,
        target_ir=target_artifact,
        backend=backend_artifact,
        executable=executable,
        runtime_status=runtime_status,
        execution_mode=execution_mode,
        execution_kind=execution_kind,
        diagnostics=tuple(diagnostics),
        trace_events=tuple(trace_events),
        tool_invocations=tuple(tool_invocations),
        cpu_plan=cpu_plan,
    )
    _maybe_dump_debug_artifacts(bundle)
    return bundle


def _capability_reason(bundle: CompileArtifactBundle) -> str:
    if bundle.cpu_plan is None:
        return "no supported compiler plan was emitted"
    return supports_op(bundle.request.target, bundle.cpu_plan.op_name).reason


def _execution_mode_for(target_kind: str, has_plan: bool) -> str:
    if target_kind == "apple_cpu" and has_plan:
        return "cpu_accelerate"
    if target_kind == "apple_gpu" and has_plan:
        return "metal_artifact"
    if target_kind == "cpu" and has_plan:
        return "jit_cpu_numpy"
    return "artifact_only"


def _execution_kind_for(target_kind: str, cpu_plan: CPUPlan | None, *, executable: bool | None = None) -> str:
    if cpu_plan is None:
        return "fallback_eager"
    if target_kind == "cpu":
        return "native_cpu" if _is_native_cpu_gemm_plan(cpu_plan) else "reference_cpu"
    if target_kind == "apple_cpu":
        return "native_cpu" if executable is not False else "artifact_only"
    if target_kind == "apple_gpu":
        return "native_gpu" if executable else "artifact_only"
    return "artifact_only"


def _is_native_cpu_gemm_plan(cpu_plan: CPUPlan | None) -> bool:
    if cpu_plan is None or cpu_plan.target_kind != "cpu" or len(cpu_plan.ops) != 1:
        return False
    return cpu_plan.ops[0].op_name in {"tessera.matmul", "tessera.gemm"}


_APPLE_CPU_ACCELERATE_OPS: frozenset[str] = frozenset({
    "tessera.matmul",
    "tessera.gemm",
})


def is_apple_cpu_accelerate_op(op_name: str) -> bool:
    """Return True when an op is dispatched via Accelerate's CBLAS path."""

    return op_name in _APPLE_CPU_ACCELERATE_OPS


def _is_apple_cpu_accelerate_executable(cpu_plan: CPUPlan | None) -> bool:
    """An apple_cpu plan is executable as long as at least one op exists; the
    runtime dispatches matmul/gemm via Accelerate (cblas_sgemm) and falls
    through to the numpy reference path for every other supported op. Multi-op
    programs are now first-class — chain order is preserved by `cpu_plan.ops`.
    """

    if cpu_plan is None or cpu_plan.target_kind != "apple_cpu":
        return False
    return len(cpu_plan.ops) > 0


_APPLE_GPU_MPS_OPS: frozenset[str] = frozenset({
    "tessera.matmul",
    "tessera.gemm",
})

_APPLE_GPU_MSL_OPS: frozenset[str] = frozenset({
    "tessera.rope",
    "tessera.flash_attn",
    "tessera.softmax",
    "tessera.softmax_safe",
    "tessera.gelu",
    # `tessera.silu_mul` doesn't ship a standalone MSL kernel today (it's
    # only meaningful as part of the SwiGLU fusion chain). Keeping it out
    # of this set means a lone silu_mul falls back to the metal_artifact
    # contract — exactly the behavior we want until per-op coverage lands.
})

_APPLE_GPU_RUNTIME_OPS: frozenset[str] = _APPLE_GPU_MPS_OPS | _APPLE_GPU_MSL_OPS


def is_apple_gpu_mps_op(op_name: str) -> bool:
    """Return True when an op is dispatched via Metal Performance Shaders."""

    return op_name in _APPLE_GPU_MPS_OPS


def is_apple_gpu_msl_op(op_name: str) -> bool:
    """Phase 8.4: return True when an op is dispatched via a custom MSL kernel."""

    return op_name in _APPLE_GPU_MSL_OPS


def _is_apple_gpu_mps_executable(cpu_plan: CPUPlan | None) -> bool:
    """Phase 8.3 + 8.4: an apple_gpu plan is executable when:
      - it is a single op in the runtime envelope (matmul/gemm via MPS,
        rope/flash_attn/softmax/gelu via custom MSL), OR
      - it is a recognized op chain (Phase 8.4.3: matmul -> softmax fused
        into a single MSL kernel).

    Multi-op programs that don't match a recognized fusion pattern stay on
    the metal_artifact contract.

    Name kept for backward compatibility — it now spans MPS + MSL +
    fusion paths.
    """

    if cpu_plan is None or cpu_plan.target_kind != "apple_gpu":
        return False
    if len(cpu_plan.ops) == 1:
        return cpu_plan.ops[0].op_name in _APPLE_GPU_RUNTIME_OPS
    return _apple_gpu_chain_kind(cpu_plan) is not None


def _apple_gpu_matmul_dtype_suffix(cpu_plan: CPUPlan | None) -> str:
    """Phase 8.4.4 — extract the matmul element type from a CPUPlan and
    map it to the runtime symbol's dtype suffix.

    The graph IR operand types are strings like "tensor<*xf32>", "tensor<*xf16>",
    or "tensor<*xbf16>". The matching runtime symbol is one of:
      - tessera_apple_gpu_mps_matmul_f32   (Phase 8.3, native MPSDataTypeFloat32)
      - tessera_apple_gpu_mps_matmul_f16   (Phase 8.4.4, native MPSDataTypeFloat16)
      - tessera_apple_gpu_mps_matmul_bf16  (Phase 8.4.4, fp32-conversion path)

    Defaults to f32 when the operand type can't be parsed — matches the pre-
    Phase 8.4.4 behavior so existing single-op f32 programs are unchanged.
    """

    if cpu_plan is None or not cpu_plan.ops:
        return "f32"
    op = cpu_plan.ops[0]
    operand_types = list(getattr(op, "operand_types", ()) or ())
    for t in operand_types:
        if "bf16" in t:
            return "bf16"
        if "f16" in t and "bf16" not in t:
            return "f16"
    return "f32"


def _apple_gpu_chain_kind(cpu_plan: CPUPlan | None) -> str | None:
    """Phase 8.4.3 + 8.4.5: classify multi-op apple_gpu plans against the
    recognized fusion patterns. Returns one of:
      - "matmul_softmax_matmul": 3-op chain matmul -> softmax -> matmul
        (full attention block, Phase 8.4.5)
      - "matmul_softmax": 2-op chain matmul -> softmax (Phase 8.4.3)
      - None: plan doesn't match any recognized fusion pattern.

    Longer chains are checked first so the most-specific fusion wins.
    Each chain link enforces single-use consumer (the intermediate is
    materialized only inside the fused kernel, never observed elsewhere)
    and matching dtypes/axes.
    """

    if cpu_plan is None or cpu_plan.target_kind != "apple_gpu":
        return None
    ops = cpu_plan.ops

    def _operand_matches(consumer, expected_name: str) -> bool:
        if not consumer.operands:
            return False
        op0 = consumer.operands[0]
        if op0.startswith("%"):
            op0 = op0[1:]
        return op0 == expected_name

    def _softmax_axis_ok(softmax_op) -> bool:
        axis = softmax_op.kwargs.get("axis", -1) if hasattr(softmax_op, "kwargs") else -1
        if axis in {-1, None}:
            return True
        try:
            return int(axis) in {-1, 1}
        except (TypeError, ValueError):
            return False

    # Phase 8.4.8 (Stage 3 of the SwiGLU Performance Plan): 4-op fusion.
    # matmul(x, Wg) -> matmul(x, Wu) -> silu_mul(gate, up) -> matmul(_, Wd).
    # Both gate and up matmuls must consume the same `x` SSA value.
    if len(ops) == 4:
        m_gate, m_up, sm_op, m_down = ops[0], ops[1], ops[2], ops[3]
        # Single-use chain on intermediates: m_gate.result is consumed only
        # by silu_mul; m_up.result only by silu_mul; silu_mul.result only by
        # m_down. We can't ask the IR directly for use counts here, so
        # (mirroring the existing 2-op / 3-op detectors) the operand-name
        # equality check stands in as a structural single-use proxy.
        if (
            m_gate.op_name in {"tessera.matmul", "tessera.gemm"}
            and m_up.op_name in {"tessera.matmul", "tessera.gemm"}
            and sm_op.op_name == "tessera.silu_mul"
            and m_down.op_name in {"tessera.matmul", "tessera.gemm"}
            and m_gate.result and m_up.result and sm_op.result
            and len(m_gate.operands) >= 1 and len(m_up.operands) >= 1
            and m_gate.operands[0] == m_up.operands[0]  # shared %x
            and len(sm_op.operands) == 2
            and _operand_matches(sm_op, m_gate.result)  # silu_mul[0] == gate
            # silu_mul[1] == up.result (operand 1, not 0).
            and (sm_op.operands[1].lstrip("%") == m_up.result)
            and _operand_matches(m_down, sm_op.result)
        ):
            return "swiglu"

    # Phase 8.4.5: 3-op fusion. matmul -> softmax -> matmul.
    if len(ops) == 3:
        m1, sm, m2 = ops[0], ops[1], ops[2]
        if (
            m1.op_name in {"tessera.matmul", "tessera.gemm"}
            and sm.op_name in {"tessera.softmax", "tessera.softmax_safe"}
            and m2.op_name in {"tessera.matmul", "tessera.gemm"}
            and m1.result and sm.result
            and _operand_matches(sm, m1.result)
            and _operand_matches(m2, sm.result)
            and _softmax_axis_ok(sm)
        ):
            return "matmul_softmax_matmul"

    # Phase 8.4.3 + 8.4.7: 2-op fusions. matmul -> {softmax | gelu | rmsnorm}.
    if len(ops) == 2:
        first, second = ops[0], ops[1]
        if (
            first.op_name in {"tessera.matmul", "tessera.gemm"}
            and first.result
            and _operand_matches(second, first.result)
        ):
            if second.op_name in {"tessera.softmax", "tessera.softmax_safe"} \
                    and _softmax_axis_ok(second):
                return "matmul_softmax"
            if second.op_name == "tessera.gelu":
                return "matmul_gelu"
            if second.op_name in {"tessera.rmsnorm", "tessera.rmsnorm_safe"}:
                return "matmul_rmsnorm"
    return None


def _backend_artifact_for(target_kind: str, cpu_plan: CPUPlan | None) -> LoweringArtifact | None:
    if cpu_plan is None:
        return None
    if target_kind == "apple_cpu":
        text = "\n".join([
            'module attributes {tessera.ir.level = "backend", target = "apple_cpu", execution_mode = "cpu_accelerate"} {',
            '  "tessera_apple.cpu.runtime_pipeline"() {',
            '    pipeline = "tessera-lower-to-apple_cpu-runtime",',
            '    symbol = "tessera_apple_cpu_gemm_f32",',
            '    framework = "Accelerate",',
            '    abi = "cblas_sgemm",',
            '    dtype = "f32"',
            '  } : () -> ()',
            "}",
        ])
        return LoweringArtifact("backend", text)
    if target_kind == "apple_gpu" and _is_apple_gpu_mps_executable(cpu_plan):
        # Pick the runtime symbol/framework pair based on which op (or chain)
        # is in the plan. Single-op cases route to one of the per-op kernels;
        # recognized fusion chains route to a fused kernel (Phase 8.4.3).
        chain = _apple_gpu_chain_kind(cpu_plan)
        if chain == "swiglu":
            symbol = "tessera_apple_gpu_swiglu_f32"
            framework = "Metal"
            abi = "MSLComputePipelineState"
        elif chain == "matmul_softmax_matmul":
            symbol = "tessera_apple_gpu_matmul_softmax_matmul_f32"
            framework = "Metal"
            abi = "MSLComputePipelineState"
        elif chain == "matmul_softmax":
            symbol = "tessera_apple_gpu_matmul_softmax_f32"
            framework = "Metal"
            abi = "MSLComputePipelineState"
        elif chain == "matmul_gelu":
            symbol = "tessera_apple_gpu_matmul_gelu_f32"
            framework = "Metal"
            abi = "MSLComputePipelineState"
        elif chain == "matmul_rmsnorm":
            symbol = "tessera_apple_gpu_matmul_rmsnorm_f32"
            framework = "Metal"
            abi = "MSLComputePipelineState"
        else:
            only_op = cpu_plan.ops[0].op_name
            if only_op in _APPLE_GPU_MPS_OPS:
                # Phase 8.4.4 — pick the matmul symbol by element type. The
                # actual element type comes from the Graph IR operand types
                # ("tensor<*xf16>" etc); fall back to f32 if the parse fails.
                dtype_suffix = _apple_gpu_matmul_dtype_suffix(cpu_plan)
                symbol = f"tessera_apple_gpu_mps_matmul_{dtype_suffix}"
                framework = "MetalPerformanceShaders"
                abi = "MPSMatrixMultiplication"
            elif only_op == "tessera.rope":
                symbol = "tessera_apple_gpu_rope_f32"
                framework = "Metal"
                abi = "MSLComputePipelineState"
            elif only_op == "tessera.flash_attn":
                symbol = "tessera_apple_gpu_flash_attn_f32"
                framework = "Metal"
                abi = "MSLComputePipelineState"
            elif only_op in {"tessera.softmax", "tessera.softmax_safe"}:
                symbol = "tessera_apple_gpu_softmax_f32"
                framework = "Metal"
                abi = "MSLComputePipelineState"
            elif only_op == "tessera.gelu":
                symbol = "tessera_apple_gpu_gelu_f32"
                framework = "Metal"
                abi = "MSLComputePipelineState"
            else:
                symbol = "tessera_apple_gpu_unknown"
                framework = "Metal"
                abi = "unknown"
        text = "\n".join([
            'module attributes {tessera.ir.level = "backend", target = "apple_gpu", execution_mode = "metal_runtime"} {',
            '  "tessera_apple.gpu.runtime_pipeline"() {',
            '    pipeline = "tessera-lower-to-apple_gpu-runtime",',
            f'    symbol = "{symbol}",',
            f'    framework = "{framework}",',
            f'    abi = "{abi}",',
            '    dtype = "f32"',
            '  } : () -> ()',
            "}",
        ])
        return LoweringArtifact("backend", text)
    return None


def _try_validate_with_tessera_opt(request: CompileRequest) -> tuple[ToolInvocation, CompileTraceEvent]:
    tool = _find_tessera_opt()
    input_hash = stable_hash(request.graph_ir)
    if tool is None:
        invocation = ToolInvocation(
            tool="tessera-opt",
            pipeline_name=request.pipeline_name,
            available=False,
            status="missing",
        )
        event = CompileTraceEvent(
            pass_name=request.pipeline_name,
            target=request.target,
            input_hash=input_hash,
            output_hash=input_hash,
            elapsed_ms=0.0,
            status="tool-missing",
            metadata={"tool": "tessera-opt"},
        )
        return invocation, event

    start = time.perf_counter()
    try:
        proc = subprocess.run(
            [tool, f"-{request.pipeline_name}", "--allow-unregistered-dialect"],
            input=request.graph_ir,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=10,
            check=False,
        )
        elapsed_ms = (time.perf_counter() - start) * 1000.0
        stdout_hash = stable_hash(proc.stdout)
        status = "ok" if proc.returncode == 0 else "failed"
        invocation = ToolInvocation(
            tool=tool,
            pipeline_name=request.pipeline_name,
            available=True,
            status=status,
            elapsed_ms=elapsed_ms,
            stdout_hash=stdout_hash,
            stderr=proc.stderr.strip()[:1000],
        )
        event = CompileTraceEvent(
            pass_name=request.pipeline_name,
            target=request.target,
            input_hash=input_hash,
            output_hash=stdout_hash if proc.stdout else input_hash,
            elapsed_ms=elapsed_ms,
            status=f"tool-{status}",
            diagnostic_count=0 if proc.returncode == 0 else 1,
            metadata={"tool": tool, "returncode": proc.returncode},
        )
        return invocation, event
    except Exception as exc:
        elapsed_ms = (time.perf_counter() - start) * 1000.0
        invocation = ToolInvocation(
            tool=tool,
            pipeline_name=request.pipeline_name,
            available=True,
            status="error",
            elapsed_ms=elapsed_ms,
            stderr=str(exc),
        )
        event = CompileTraceEvent(
            pass_name=request.pipeline_name,
            target=request.target,
            input_hash=input_hash,
            output_hash=input_hash,
            elapsed_ms=elapsed_ms,
            status="tool-error",
            diagnostic_count=1,
            metadata={"tool": tool, "reason": str(exc)},
        )
        return invocation, event


def _find_tessera_opt() -> str | None:
    env = os.environ.get("TESSERA_OPT")
    if env:
        return env
    found = shutil.which("tessera-opt")
    if found:
        return found
    return None


def _maybe_dump_debug_artifacts(bundle: CompileArtifactBundle) -> None:
    debug_ir = os.environ.get("TESSERA_DEBUG_IR") == "1"
    dump_state = os.environ.get("TESSERA_DUMP_STATE") == "1"
    if not debug_ir and not dump_state:
        return
    dump_root = Path(os.environ.get("TESSERA_DUMP_DIR", Path(tempfile.gettempdir()) / "tessera_debug"))
    label = _safe_dump_label(bundle)
    out_dir = dump_root / label
    out_dir.mkdir(parents=True, exist_ok=True)
    if debug_ir:
        for artifact in bundle.lowering_artifacts():
            (out_dir / f"{artifact.level}.mlir").write_text(artifact.text + "\n", encoding="utf-8")
    if dump_state:
        (out_dir / "metadata.json").write_text(
            json.dumps(bundle.to_metadata(), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        (out_dir / "trace.json").write_text(bundle.chrome_trace_json() + "\n", encoding="utf-8")


def _safe_dump_label(bundle: CompileArtifactBundle) -> str:
    function = "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in bundle.request.function_name)
    return f"{function}-{bundle.request.target}-{bundle.request.graph_hash[:12]}"


__all__ = [
    "COMPILER_TRACE_SCHEMA_VERSION",
    "PIPELINE_BY_TARGET",
    "CompileArtifactBundle",
    "CompileRequest",
    "CompileTraceEvent",
    "ToolInvocation",
    "compile_graph_module",
    "stable_hash",
]
