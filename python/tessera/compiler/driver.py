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
import re
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
    # Sprint 4 (S4): when value mode was requested but the `-full` lowering
    # failed, the reason is recorded here and surfaced as
    # `apple_value_target_ir_error` — the front door keeps the artifact IR but
    # the failure is observable, not silent.
    value_mode_error: str | None = None

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
        profiling: dict[str, Any] = {
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
        extra: dict[str, Any] = {}
        if self.value_mode_error is not None:
            extra["apple_value_target_ir_error"] = self.value_mode_error
        return {
            **extra,
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

    # Apple Value Target IR sprint 3 — explicit value-mode front door. When the
    # caller asks for value mode on an Apple target, run the value-preserving
    # `-full` pipeline and use the emitted value Target IR as the target
    # artifact (so to_runtime_artifact routes to apple_value_target_ir). Default
    # (artifact) behavior is untouched — no opt-in, no change.
    value_mode_error: str | None = None
    if str((options or {}).get("apple_target_ir_mode", "")) == "value" \
            and target_kind in ("apple_cpu", "apple_gpu"):
        # Feed the *canonical* (parseable custom-assembly) Graph IR straight to
        # the value pipeline — no text rewrite. `graph_text` above is the paren
        # form kept for hashing / display; the canonical render is parser-ready.
        value_ir, value_mode_error = _lower_apple_value_target_ir(
            module.to_mlir(verify=False, canonical=True), target_kind)
        if value_ir:
            target_artifact = LoweringArtifact("target", value_ir)
            # CPU value calls are executable now; the GPU value row stays gated
            # in the execution matrix regardless, so this flag is safe there.
            if target_kind == "apple_cpu":
                executable = True
                runtime_status = "ready"
                execution_kind = "native_cpu"
                execution_mode = "cpu_accelerate"
        # Sprint 4 (S4): a failure (value_mode_error set) is recorded on the
        # bundle and surfaced as apple_value_target_ir_error — the artifact IR
        # is preserved but the failure is observable, never silent.

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
        value_mode_error=value_mode_error,
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
    # 2026-05-29 — batched / rank-3 matmul via the MPSGraph bmm lane (Tier-2).
    "tessera.batched_gemm",
})

_APPLE_GPU_MSL_OPS: frozenset[str] = frozenset({
    "tessera.rope",
    "tessera.flash_attn",
    "tessera.softmax",
    "tessera.softmax_safe",
    "tessera.gelu",
})

# 2026-05-29 — Tier-1 activations / normalizations dispatched through the
# MetalPerformanceShadersGraph lane (one parametrized runner per shape class,
# fp32 compute, f16 native + bf16 host-upcast). These previously fell back to
# the numpy reference path; they now execute on the GPU. `silu_mul` runs as a
# standalone binary op here (silu(a)*b); the fused SwiGLU chain still wins via
# the 4-op fusion check when the surrounding matmuls are present.
_APPLE_GPU_MPSGRAPH_OPS: frozenset[str] = frozenset({
    "tessera.relu",
    "tessera.sigmoid",
    "tessera.sigmoid_safe",
    "tessera.tanh",
    "tessera.softplus",
    "tessera.silu",
    "tessera.exp",
    "tessera.log",
    "tessera.sqrt",
    "tessera.rsqrt",
    "tessera.neg",
    "tessera.negative",
    "tessera.abs",
    "tessera.absolute",
    # Batch 1 (2026-06-08) — float-output elementwise math (unary + binary +
    # comparison) on the MPSGraph opcode lane.
    "tessera.sin", "tessera.cos", "tessera.tan",
    "tessera.asin", "tessera.acos", "tessera.atan",
    "tessera.sinh", "tessera.cosh", "tessera.erf", "tessera.erfc",
    "tessera.expm1", "tessera.log1p", "tessera.reciprocal",
    "tessera.sign", "tessera.floor", "tessera.ceil",
    "tessera.round", "tessera.trunc",
    "tessera.add", "tessera.sub", "tessera.mul", "tessera.div",
    "tessera.maximum", "tessera.minimum",
    "tessera.pow", "tessera.atan2", "tessera.mod", "tessera.floor_div",
    "tessera.eq", "tessera.ne", "tessera.lt", "tessera.le",
    "tessera.gt", "tessera.ge",
    "tessera.layer_norm",
    "tessera.rmsnorm",
    "tessera.rmsnorm_safe",
    "tessera.log_softmax",
    "tessera.silu_mul",
})

# 2026-05-29 — Tier-2 projection ops routed through the matmul / bmm lane:
# linear_general (x @ W (+ bias), last-axis contraction) and qkv_projection
# (x @ W_qkv then split-3).
_APPLE_GPU_PROJECTION_OPS: frozenset[str] = frozenset({
    "tessera.linear_general",
    "tessera.qkv_projection",
})

# 2026-05-29 — Tier-3 reductions / scans via the MPSGraph reduce lane.
_APPLE_GPU_REDUCTION_OPS: frozenset[str] = frozenset({
    "tessera.reduce", "tessera.mean", "tessera.amax", "tessera.amin",
    "tessera.prod", "tessera.var", "tessera.std", "tessera.argmax",
    "tessera.argmin", "tessera.cumsum", "tessera.cumprod",
})

# 2026-05-30 — Tier-3 convolutions: conv2d via the MPSGraph convolution2D node
# (NHWC/HWIO); conv3d via im2col + a GPU MPSGraph batched matmul (NDHWC/DHWIO).
_APPLE_GPU_CONV_OPS: frozenset[str] = frozenset({"tessera.conv2d", "tessera.conv3d"})

# 2026 — GPU linear-algebra lane (MPSMatrix decomposition/solve — the one
# capability MPSGraph lacks). Only the registered Graph IR ops are wired here:
# `tessera.cholesky` (factor) and `tessera.tri_solve` (triangular solve). Dense
# f32; batched/non-f32 fall back to numpy inside the dispatcher.
_APPLE_GPU_LINALG_OPS: frozenset[str] = frozenset({"tessera.cholesky", "tessera.tri_solve"})

# Mamba-2 selective state-space scan. Executes via the chunked-parallel SSD form
# whose batched contractions run on the Metal bmm lane (scalar-state A; general
# (D,N) A falls back to the sequential numpy reference inside the dispatcher).
_APPLE_GPU_SSM_OPS: frozenset[str] = frozenset({"tessera.selective_ssm"})

# Ragged grouped matmul — the MoE expert-FFN compute core. Each contiguous token
# group runs as a per-expert MPS matmul on the Metal lane.
_APPLE_GPU_MOE_OPS: frozenset[str] = frozenset({"tessera.grouped_gemm"})

# LDT candidate-axis ops with dedicated Metal kernels.
_APPLE_GPU_LDT_OPS: frozenset[str] = frozenset({"tessera.popcount", "tessera.count_nonzero", "tessera.loss.z_loss", "tessera.loss.asymmetric_bce", "tessera.loss.load_balance_loss", "tessera.masked_categorical"})
# Geometric-algebra (Clifford Cl(3,0)) flat-coefficient lane — canonical
# tessera.ops projection of tessera.ga.*; routed to the cl30 MSL kernels by
# runtime.py::_apple_gpu_dispatch_clifford.
_APPLE_GPU_CLIFFORD_OPS: frozenset[str] = frozenset({
    "tessera.clifford_geometric_product", "tessera.clifford_wedge",
    "tessera.clifford_left_contraction", "tessera.clifford_inner",
    "tessera.clifford_rotor_sandwich",
    "tessera.clifford_reverse", "tessera.clifford_grade_involution",
    "tessera.clifford_conjugate", "tessera.clifford_grade_projection",
    "tessera.clifford_hodge_star",
    "tessera.clifford_ext_deriv", "tessera.clifford_vec_deriv",
    "tessera.clifford_codiff",
    "tessera.clifford_exp", "tessera.clifford_log",
    "tessera.clifford_norm", "tessera.clifford_norm_squared",
})

# Energy-based-model flat-array lane — canonical tessera.ops projection of the
# tensor-clean tessera.ebm.* subset; routed to the EBM MSL kernels by
# runtime.py::_apple_gpu_dispatch_ebm.
_APPLE_GPU_EBM_OPS: frozenset[str] = frozenset({
    "tessera.ebm_energy_quadratic", "tessera.ebm_self_verify",
    "tessera.ebm_refinement", "tessera.ebm_inner_step",
})

# EBM training losses (CD / PCD / score-matching / ISM / DSM) — MPSGraph
# reductions; routed to the EBM-loss kernels by runtime.py::_apple_gpu_dispatch_ebm_loss.
_APPLE_GPU_EBM_LOSS_OPS: frozenset[str] = frozenset({
    "tessera.loss.contrastive_divergence", "tessera.loss.persistent_cd",
    "tessera.loss.score_matching", "tessera.loss.implicit_score_matching",
    "tessera.loss.denoising_score_matching",
})
_APPLE_GPU_RUNTIME_OPS: frozenset[str] = (
    _APPLE_GPU_MPS_OPS | _APPLE_GPU_MSL_OPS | _APPLE_GPU_MPSGRAPH_OPS
    | _APPLE_GPU_PROJECTION_OPS | _APPLE_GPU_REDUCTION_OPS | _APPLE_GPU_CONV_OPS
    | _APPLE_GPU_LINALG_OPS | _APPLE_GPU_SSM_OPS | _APPLE_GPU_MOE_OPS
    | _APPLE_GPU_LDT_OPS | _APPLE_GPU_CLIFFORD_OPS | _APPLE_GPU_EBM_OPS
    | _APPLE_GPU_EBM_LOSS_OPS
)


# ── Apple Value Target IR (sprint, 2026-06-03) ──────────────────────────────
# The value-preserving `-full` pipelines emit value-producing target ops that
# carry real SSA operands + results; the artifact pipelines emit attribute-only
# metadata ops.  These helpers let the front door (canonical_compile / JIT) tag
# the lowered module as `value_target_ir` vs `target_ir_artifact`, and let the
# runtime dispatcher read the dispatch tuple straight off each value op (the
# seam-closure contract: the IR names the C ABI `symbol` that executes it).
_APPLE_VALUE_OPS: tuple[str, ...] = (
    "tessera_apple.cpu.call",
    "tessera_apple.gpu.kernel_call",
    "tessera_apple.gpu.package_call",
)
_APPLE_ARTIFACT_OPS: tuple[str, ...] = (
    "tessera_apple.cpu.vector_op",
    "tessera_apple.cpu.accelerate_gemm",
    "tessera_apple.cpu.vector_reduce",
    "tessera_apple.gpu.metal_kernel",
)

# String attr `key = "value"` pairs inside an already-isolated attr dict.
# `(?:\\.|[^"\\])*` matches a quoted body with escaped quotes/backslashes, so a
# value containing JSON-like braces (e.g. argument_layout = "{\"buffers\":[…]}")
# round-trips intact.
_APPLE_ATTR_RE = re.compile(r'(\w+)\s*=\s*"((?:\\.|[^"\\])*)"')
# Bool attrs print unquoted (`lower = true`).  Sprint 3: the linalg semantic
# attrs (lower/trans/unit_diag/full_matrices) ride the value op as BoolAttrs, so
# the extractor must surface them — runtime dispatch must not assume defaults.
_APPLE_BOOL_ATTR_RE = re.compile(r"(\w+)\s*=\s*(true|false)\b")
_APPLE_INT_ATTR_RE = re.compile(r"(\w+)\s*=\s*(-?\d+)\b")
_APPLE_FLOAT_ATTR_RE = re.compile(
    r"(\w+)\s*=\s*(-?(?:\d+\.\d*|\d*\.\d+)(?:[eE][+-]?\d+)?)\s*:\s*f(?:32|64)\b"
)

_APPLE_VALUE_CPU_EXECUTABLE_SYMBOLS: frozenset[str] = frozenset({
    "tessera_apple_cpu_cholesky_f32",
    "tessera_apple_cpu_tri_solve_f32",
    "tessera_apple_cpu_cholesky_solve_f32",
    "tessera_apple_cpu_lu_f32",
    "tessera_apple_cpu_qr_f32",
    "tessera_apple_cpu_svd_f32",
    "tessera_apple_cpu_gemm_f32",
    "tessera_apple_cpu_gemm_f32_batched",
    "tessera_apple_cpu_gemm_f16",
    "tessera_apple_cpu_gemm_bf16",
})

# Stage 16D: Apple GPU value executable truth is per C ABI symbol, not per op
# family. Symbols with status-returning/probed runtime helpers use those probes;
# legacy value symbols that predate probe helpers still require the runtime
# resolver to bind and are tracked as follow-ons for numerical-probe parity.
_APPLE_VALUE_GPU_SYMBOL_PROBES: Mapping[str, str] = {
    "tessera_apple_gpu_bmm_f32": "_apple_gpu_bmm_f32",
    "tessera_apple_gpu_bmm_f16": "_apple_gpu_bmm_f16",
    "tessera_apple_gpu_bmm_bf16": "_apple_gpu_bmm_bf16",
    "tessera_apple_gpu_native_sparse_attn_f32": "_apple_gpu_native_sparse_attn_f32",
    "tessera_apple_gpu_ppo_policy_loss_f32": "_apple_gpu_ppo_policy_loss_available",
    "tessera_apple_gpu_ppo_policy_loss_ex_f32": "_apple_gpu_ppo_policy_loss_ex_available",
    "tessera_apple_gpu_ebm_energy_quadratic_value_f32":
        "_apple_gpu_ebm_energy_quadratic_value_available",
    "tessera_apple_gpu_ebm_langevin_step_value_f32":
        "_apple_gpu_ebm_langevin_step_value_available",
    "tessera_apple_gpu_ebm_refinement_value_f32":
        "_apple_gpu_ebm_refinement_value_available",
    "tessera_apple_gpu_ebm_partition_exact_value_f32":
        "_apple_gpu_ebm_partition_exact_value_available",
    "tessera_apple_gpu_clifford_geo_product_cl30_value_f32":
        "_apple_gpu_clifford_geo_product_cl30_value_available",
}


def _scan_value_call_attr_dicts(ir_text: str) -> list[tuple[str, str]]:
    """Brace-safe scanner (Sprint 2, S2-1): for every value-op mnemonic, walk
    forward to its attribute dict and return ``(op_name, attr_dict_body)``.

    Replaces the old regex (which matched only same-line / no-nested-brace attr
    dicts).  Walks to the matching top-level ``}`` while *skipping braces inside
    quoted strings* (respecting ``\\`` escapes), so an ``argument_layout`` whose
    value is itself a JSON object with braces does not terminate the dict early.
    """
    found: list[tuple[int, str, str]] = []  # (position, op, attr_body)
    n = len(ir_text)
    for op in _APPLE_VALUE_OPS:
        start = 0
        while True:
            hit = ir_text.find(op, start)
            if hit < 0:
                break
            start = hit + len(op)
            i = ir_text.find("{", start)  # the op's attr dict opens here
            if i < 0:
                break
            # Walk to the matching close brace, honoring quoted strings + escapes.
            depth, j, in_str, esc = 0, i, False, False
            body_start = i + 1
            while j < n:
                ch = ir_text[j]
                if in_str:
                    if esc:
                        esc = False
                    elif ch == "\\":
                        esc = True
                    elif ch == '"':
                        in_str = False
                elif ch == '"':
                    in_str = True
                elif ch == "{":
                    depth += 1
                elif ch == "}":
                    depth -= 1
                    if depth == 0:
                        found.append((hit, op, ir_text[body_start:j]))
                        break
                j += 1
            start = j + 1
    found.sort(key=lambda t: t[0])  # source order, not grouped by op-type
    return [(op, body) for _, op, body in found]


def classify_apple_target_ir(ir_text: str) -> str:
    """Classify a lowered Apple module as 'value_target_ir' (contains
    value-producing call ops — the `-full` semantics-preserving lane),
    'target_ir_artifact' (attribute-only metadata ops — the inspection lane),
    or 'none'. Value classification wins if both appear."""
    if any(op in ir_text for op in _APPLE_VALUE_OPS):
        return "value_target_ir"
    if any(op in ir_text for op in _APPLE_ARTIFACT_OPS):
        return "target_ir_artifact"
    return "none"


def extract_apple_value_calls(ir_text: str) -> list[dict[str, object]]:
    """Read the dispatch tuple off every value-producing Apple target op.

    The runtime dispatcher consumes this: for cpu.call it invokes the named
    Accelerate/LAPACK C ABI `symbol`; for gpu.kernel_call / gpu.package_call it
    reports native_gpu execution only when ``status == "executable"``. Returns
    one dict per value op with at least {op, op_kind, symbol, status}.
    """
    calls: list[dict[str, object]] = []
    for op_name, attr_blob in _scan_value_call_attr_dicts(ir_text):
        attrs: dict[str, object] = {
            k: v for k, v in _APPLE_ATTR_RE.findall(attr_blob)
        }
        # Bool attrs (unquoted) — don't clobber a same-named string attr.
        for k, v in _APPLE_BOOL_ATTR_RE.findall(attr_blob):
            attrs.setdefault(k, v == "true")
        for k, v in _APPLE_FLOAT_ATTR_RE.findall(attr_blob):
            attrs.setdefault(k, float(v))
        for k, v in _APPLE_INT_ATTR_RE.findall(attr_blob):
            attrs.setdefault(k, int(v))
        attrs["op"] = op_name
        attrs.setdefault("status", "")
        calls.append(attrs)
    return calls


def _gpu_value_symbol_available(symbol: str) -> bool:
    probe_name = _APPLE_VALUE_GPU_SYMBOL_PROBES.get(symbol)
    if probe_name is None:
        return False
    try:
        from tessera import runtime as _rt

        probe = getattr(_rt, probe_name, None)
        if probe is None:
            return False
        return bool(probe())
    except Exception:
        return False


def apple_value_call_is_executable(call: Mapping[str, object]) -> bool:
    """Return true only for per-symbol allowlisted Apple value calls.

    ``status="executable"`` is necessary but not sufficient: the call must use
    the expected value-op mnemonic and a C ABI symbol on the exact CPU/GPU
    allowlist. GPU symbols also need their runtime resolver/probe to succeed, so
    a stub or stale dylib cannot become executable by metadata alone.
    """
    if call.get("status") != "executable":
        return False
    symbol = str(call.get("symbol") or "")
    op = str(call.get("op") or "")
    if op == "tessera_apple.cpu.call":
        return symbol in _APPLE_VALUE_CPU_EXECUTABLE_SYMBOLS
    if op == "tessera_apple.gpu.kernel_call":
        return _gpu_value_symbol_available(symbol)
    return False


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

    # P6 (Metal 4 epilogue fusion): linear + bias + activation. Decomposed by
    # nn.functional.linear as gemm -> add(bias); an MLP block adds an activation:
    # gemm -> add(bias) -> {gelu | relu | silu}. Recognized STRUCTURALLY (the trace
    # leaves operand element types as `?`, so dtype is only known at call time).
    # The runtime dispatcher routes f16/bf16 to the MPP matmul2d epilogue (bias +
    # act fused in-register) and otherwise runs the matmul on MPS + bias/act —
    # either way correct, and a strict win over the all-numpy eager path these
    # multi-op chains hit today.
    _MM = {"tessera.matmul", "tessera.gemm"}
    _ACT = {"tessera.gelu": "matmul_bias_gelu", "tessera.relu": "matmul_bias_relu",
            "tessera.silu": "matmul_bias_silu"}
    if len(ops) == 3:
        m, addop, act = ops[0], ops[1], ops[2]
        if (m.op_name in _MM and addop.op_name == "tessera.add"
                and m.result and addop.result
                and _operand_matches(addop, m.result)
                and _operand_matches(act, addop.result)
                and act.op_name in _ACT):
            return _ACT[act.op_name]
    if len(ops) == 2:
        m, addop = ops[0], ops[1]
        if (m.op_name in _MM and addop.op_name == "tessera.add"
                and m.result and _operand_matches(addop, m.result)):
            return "matmul_bias"

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
        elif chain in {"matmul_bias", "matmul_bias_gelu", "matmul_bias_relu",
                       "matmul_bias_silu"}:
            # P6 — linear+bias(+act) fused via the MPP matmul2d epilogue (f16/bf16).
            dtype_suffix = _apple_gpu_matmul_dtype_suffix(cpu_plan)
            symbol = f"tessera_apple_gpu_mtl4_matmul2d_epilogue_{dtype_suffix}"
            framework = "MetalPerformancePrimitives"
            abi = "MTL4MatMul2dCooperative"
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


def _tessera_repo_root() -> Path | None:
    """Locate the Tessera repo root by walking up from this file until a
    directory containing both ``python/tessera`` and ``src/compiler`` is found.

    This is robust to where ``driver.py`` lives in the package tree (the old
    ``parents[2]`` assumption pointed at ``python/`` rather than the repo root,
    so the in-repo fallback never resolved). Returns None if no such ancestor
    exists (e.g. an installed wheel with no source tree alongside it)."""
    here = Path(__file__).resolve()
    for parent in here.parents:
        if (parent / "python" / "tessera").is_dir() and (parent / "src" / "compiler").is_dir():
            return parent
    return None


def _resolve_tessera_opt() -> str | None:
    """Find tessera-opt with precedence: TESSERA_OPT env, then PATH (both via
    _find_tessera_opt), then the in-repo build at
    ``<repo>/build/tools/tessera-opt/tessera-opt`` — so the value front door
    works without any environment setup when run from a source checkout."""
    found = _find_tessera_opt()
    if found:
        return found
    repo = _tessera_repo_root()
    if repo is not None:
        built = repo / "build" / "tools" / "tessera-opt" / "tessera-opt"
        if built.is_file() and os.access(built, os.X_OK):
            return str(built)
    return None


def _lower_apple_value_target_ir(
    graph_text: str, target_kind: str
) -> tuple[str | None, str | None]:
    """Run the value-preserving ``tessera-lower-to-apple_{cpu,gpu}-full``
    pipeline; return ``(value_ir, error_reason)``.

    On success ``error_reason`` is None. On any failure ``value_ir`` is None and
    ``error_reason`` is a short, named explanation (so the front door can record
    ``apple_value_target_ir_error`` rather than silently degrading to artifact
    metadata). Sprint 4 (S4): the failure path is observable, not silent."""
    if target_kind not in ("apple_cpu", "apple_gpu"):
        return None, f"value mode is only defined for apple_cpu/apple_gpu, not {target_kind!r}"
    tool = _resolve_tessera_opt()
    if not tool:
        return None, "tessera-opt not found (set TESSERA_OPT, add to PATH, or build build/tools/tessera-opt)"
    pipeline = f"tessera-lower-to-{target_kind}-full"
    # `graph_text` is expected to be the CANONICAL (parseable custom-assembly)
    # Graph IR render — `GraphIRModule.to_mlir(canonical=True)`. No text rewrite
    # happens here anymore: the previous `tessera.op(...)` -> `tessera.op ...`
    # regex was a fragile pre-parse seam (it mishandled nested operand parens and
    # — until recently — dotted op names). The canonical render emits the form the
    # parser already accepts, so the seam is gone.
    parse_text = graph_text
    try:
        # Sprint 9: the Apple value lane runs against the *registered* Tile IR
        # dialect — no `--allow-unregistered-dialect`. If a tile.* op were
        # unregistered, tessera-opt would now fail loudly (the guard test
        # `test_apple_value_lowering_uses_no_unregistered_dialect_flag` enforces
        # this is never reintroduced).
        proc = subprocess.run(
            [tool, f"-{pipeline}", "-"],
            input=parse_text, capture_output=True, text=True, timeout=60)
    except Exception as exc:  # noqa: BLE001 — surface the reason, don't swallow
        return None, f"tessera-opt invocation failed: {exc}"
    if proc.returncode != 0:
        detail = (proc.stderr or "").strip().splitlines()
        tail = detail[-1] if detail else f"returncode {proc.returncode}"
        return None, f"{pipeline} failed: {tail}"
    if not proc.stdout.strip():
        return None, f"{pipeline} produced empty output"
    return proc.stdout, None


def lower_apple_value_target_ir(graph_text: str, target_kind: str) -> str | None:
    """Apple Value Target IR sprint 3 — front-door value mode.

    Run the value-preserving ``tessera-lower-to-apple_{cpu,gpu}-full`` pipeline
    on the Graph IR and return the emitted value Target IR (the
    tessera_apple.{cpu.call,gpu.kernel_call} ops). Returns None when tessera-opt
    is unavailable or the lowering fails — the caller then keeps the artifact
    target IR (default behavior never drifts). Use
    :func:`_lower_apple_value_target_ir` when the failure reason is needed."""
    ir, _reason = _lower_apple_value_target_ir(graph_text, target_kind)
    return ir


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
