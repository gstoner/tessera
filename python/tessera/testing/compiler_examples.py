"""Maintained compiler example qualification harness.

The harness compiles a compact set of frontend examples across the compiler
foundation targets and verifies each claimed stage emits an artifact or an
explicit non-executable status.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Mapping

import numpy as np

import tessera as ts
from tessera.compiler.jit import JitFn, jit
from tessera.compiler.matmul_pipeline import normalize_target_kind
from tessera.runtime import RuntimeArtifact, launch


COMPILER_STAGES = (
    "frontend",
    "graph-ir",
    "schedule-ir",
    "tile-ir",
    "target-ir",
    "backend-artifact",
    "runtime-executable",
)

FOUNDATION_TARGETS = ("x86", "cuda", "rocm", "apple_cpu", "apple_gpu")


@dataclass(frozen=True)
class CompilerExample:
    example_id: str
    fn: Callable[..., Any]
    stages_by_target: Mapping[str, tuple[str, ...]]
    runtime_args: tuple[Any, ...] = ()


@dataclass(frozen=True)
class CompilerExampleResult:
    example_id: str
    target: str
    compiled: JitFn
    artifact: RuntimeArtifact
    claimed_stages: tuple[str, ...]
    trace: tuple[dict[str, Any], ...]
    launch_result: dict[str, Any] | None = None


def mlp_path(A, B):
    return ts.ops.relu(ts.ops.matmul(A, B))


def attention_like_path(A, B):
    return ts.ops.softmax(ts.ops.matmul(A, B))


def conv2d_path(X, W):
    return ts.ops.conv2d(X, W)


def rmsnorm_path(X):
    return ts.ops.rmsnorm_safe(X)


def flash_attn_path(Q, K, V):
    return ts.ops.flash_attn(Q, K, V, causal=True)


def _common_artifact_stages(*, runtime: bool = False) -> tuple[str, ...]:
    stages = ("frontend", "graph-ir", "schedule-ir", "tile-ir", "target-ir", "backend-artifact")
    return stages + (("runtime-executable",) if runtime else ())


COMPILER_EXAMPLE_MANIFEST: tuple[CompilerExample, ...] = (
    CompilerExample(
        "mlp_matmul_relu",
        mlp_path,
        {target: _common_artifact_stages(runtime=target == "x86") for target in FOUNDATION_TARGETS},
        runtime_args=(
            np.arange(6, dtype=np.float32).reshape(2, 3),
            np.arange(12, dtype=np.float32).reshape(3, 4),
        ),
    ),
    CompilerExample(
        "attention_matmul_softmax",
        attention_like_path,
        {target: _common_artifact_stages(runtime=target == "x86") for target in FOUNDATION_TARGETS},
        runtime_args=(
            np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32),
            np.array([[0.5, -1.0], [1.5, 0.25]], dtype=np.float32),
        ),
    ),
    CompilerExample(
        "conv2d_reference",
        conv2d_path,
        {target: _common_artifact_stages(runtime=target == "x86") for target in FOUNDATION_TARGETS},
        runtime_args=(
            np.arange(1 * 4 * 4 * 1, dtype=np.float32).reshape(1, 4, 4, 1),
            np.ones((3, 3, 1, 2), dtype=np.float32),
        ),
    ),
    CompilerExample(
        "rmsnorm_safe",
        rmsnorm_path,
        {target: _common_artifact_stages(runtime=target == "x86") for target in FOUNDATION_TARGETS},
        runtime_args=(np.array([[1.0, 2.0, 4.0]], dtype=np.float32),),
    ),
    CompilerExample(
        "flash_attn_contract",
        flash_attn_path,
        {target: _common_artifact_stages(runtime=target == "x86") for target in FOUNDATION_TARGETS},
        runtime_args=(
            np.ones((1, 2, 4), dtype=np.float32),
            np.ones((1, 2, 4), dtype=np.float32),
            np.ones((1, 2, 4), dtype=np.float32),
        ),
    ),
)


def qualify_compiler_example(example: CompilerExample, target: str, *, run: bool = False) -> CompilerExampleResult:
    target_kind = normalize_target_kind(target)
    compiled = jit(example.fn, target=target_kind)
    artifact = compiled.runtime_artifact()
    claimed = tuple(example.stages_by_target[target])
    _assert_claimed_stages(compiled, artifact, claimed)
    launch_result = launch(artifact, example.runtime_args) if run and "runtime-executable" in claimed else None
    if launch_result is not None and not launch_result.get("ok", False):
        raise AssertionError(f"{example.example_id} expected executable {target} launch, got {launch_result}")
    return CompilerExampleResult(
        example_id=example.example_id,
        target=target_kind,
        compiled=compiled,
        artifact=artifact,
        claimed_stages=claimed,
        trace=compiled.lowering_trace(),
        launch_result=launch_result,
    )


def qualify_compiler_examples(*, targets: tuple[str, ...] = FOUNDATION_TARGETS, run_runtime: bool = False) -> tuple[CompilerExampleResult, ...]:
    results: list[CompilerExampleResult] = []
    for example in COMPILER_EXAMPLE_MANIFEST:
        for target in targets:
            results.append(qualify_compiler_example(example, target, run=run_runtime))
    return tuple(results)


def _assert_claimed_stages(compiled: JitFn, artifact: RuntimeArtifact, claimed: tuple[str, ...]) -> None:
    metadata = artifact.metadata or {}
    if "frontend" in claimed and not compiled.graph_ir.functions:
        raise AssertionError("frontend stage was claimed but no Graph IR function was emitted")
    if "graph-ir" in claimed and not artifact.graph_ir:
        raise AssertionError("graph-ir stage was claimed but graph_ir is empty")
    if "schedule-ir" in claimed and not artifact.schedule_ir:
        raise AssertionError("schedule-ir stage was claimed but schedule_ir is empty")
    if "tile-ir" in claimed and not artifact.tile_ir:
        raise AssertionError("tile-ir stage was claimed but tile_ir is empty")
    if "target-ir" in claimed and not artifact.target_ir:
        raise AssertionError("target-ir stage was claimed but target_ir is empty")
    if "backend-artifact" in claimed and metadata.get("runtime_status") not in {"ready", "artifact_only"}:
        raise AssertionError(f"backend artifact stage expected ready/artifact_only, got {metadata.get('runtime_status')!r}")
    if "runtime-executable" in claimed and metadata.get("executable") is not True:
        raise AssertionError("runtime-executable stage was claimed but artifact is not executable")
    if not compiled.lowering_trace():
        raise AssertionError("compile trace must contain at least one event")


__all__ = [
    "COMPILER_EXAMPLE_MANIFEST",
    "COMPILER_STAGES",
    "FOUNDATION_TARGETS",
    "CompilerExample",
    "CompilerExampleResult",
    "qualify_compiler_example",
    "qualify_compiler_examples",
]
