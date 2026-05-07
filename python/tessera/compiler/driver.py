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
from typing import Any, Mapping, Sequence

from .graph_ir import GraphIRModule
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
        }
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
        diagnostics=tuple(diagnostics),
        trace_events=tuple(trace_events),
        tool_invocations=tuple(tool_invocations),
        cpu_plan=cpu_plan,
    )
    _maybe_dump_debug_artifacts(bundle)
    return bundle


def _execution_mode_for(target_kind: str, has_plan: bool) -> str:
    if target_kind == "apple_cpu" and has_plan:
        return "cpu_accelerate"
    if target_kind == "apple_gpu" and has_plan:
        return "metal_artifact"
    if target_kind == "cpu" and has_plan:
        return "jit_cpu_numpy"
    return "artifact_only"


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
})

_APPLE_GPU_RUNTIME_OPS: frozenset[str] = _APPLE_GPU_MPS_OPS | _APPLE_GPU_MSL_OPS


def is_apple_gpu_mps_op(op_name: str) -> bool:
    """Return True when an op is dispatched via Metal Performance Shaders."""

    return op_name in _APPLE_GPU_MPS_OPS


def is_apple_gpu_msl_op(op_name: str) -> bool:
    """Phase 8.4: return True when an op is dispatched via a custom MSL kernel."""

    return op_name in _APPLE_GPU_MSL_OPS


def _is_apple_gpu_mps_executable(cpu_plan: CPUPlan | None) -> bool:
    """Phase 8.3 + 8.4: a single-op apple_gpu plan is executable when the op is
    in the runtime envelope — currently matmul/gemm via MPS (Phase 8.3) or rope
    via custom MSL (Phase 8.4). Multi-op programs keep the metal_artifact
    contract until the runtime grows fused-kernel coverage.

    Name kept for backward compatibility — it now spans MPS + MSL paths.
    """

    if cpu_plan is None or cpu_plan.target_kind != "apple_gpu":
        return False
    if len(cpu_plan.ops) != 1:
        return False
    return cpu_plan.ops[0].op_name in _APPLE_GPU_RUNTIME_OPS


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
        # Pick the runtime symbol/framework pair based on which op is in the
        # plan. matmul/gemm route to MPS (Phase 8.3); rope routes to the
        # custom MSL kernel emitted by the runtime shim (Phase 8.4).
        only_op = cpu_plan.ops[0].op_name
        if only_op in _APPLE_GPU_MPS_OPS:
            symbol = "tessera_apple_gpu_mps_matmul_f32"
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
