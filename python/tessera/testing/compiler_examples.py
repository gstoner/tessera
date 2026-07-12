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


def _load_s8_compilers():
    """Lazy import of the S8 tiny-model compilers.

    The previous eager ``from examples.conformance.s8_tiny_models...``
    import made ``tessera.testing`` (and therefore the top-level
    ``tessera`` package, which re-exports it) depend on
    ``examples/`` being on ``sys.path``.  That broke direct example
    execution via the documented ``PYTHONPATH=python``.  Importing
    only when the manifest is materialized keeps the test surface
    available while making the package import standalone again.
    """
    from examples.conformance.s8_tiny_models.models import (
        compile_attention_slice,
        compile_mlp_slice,
        compile_qwen3_moe_slice,
    )
    return compile_mlp_slice, compile_attention_slice, compile_qwen3_moe_slice


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


def _build_manifest() -> tuple[CompilerExample, ...]:
    """Construct the manifest on demand.

    Defers the ``examples.conformance`` import to the first call so
    importing ``tessera.testing`` does not require ``examples/`` to be
    on ``sys.path``.  Cached by ``__getattr__`` below; this function
    runs exactly once per process.
    """
    s8_compile_mlp_slice, s8_compile_attention_slice, s8_compile_qwen3_moe_slice = _load_s8_compilers()
    return (
    CompilerExample(
        "mlp_matmul_relu",
        mlp_path,
        {target: _common_artifact_stages(runtime=False) for target in FOUNDATION_TARGETS},
        runtime_args=(
            np.arange(6, dtype=np.float32).reshape(2, 3),
            np.arange(12, dtype=np.float32).reshape(3, 4),
        ),
    ),
    CompilerExample(
        "attention_matmul_softmax",
        attention_like_path,
        # Exact Apple GPU and x86 kernel proofs are tracked by architecture-
        # aligned fixtures; this generic artifact manifest carries no device
        # execution provenance and therefore makes no runtime claim.
        {target: _common_artifact_stages(runtime=False) for target in FOUNDATION_TARGETS},
        runtime_args=(
            np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32),
            np.array([[0.5, -1.0], [1.5, 0.25]], dtype=np.float32),
        ),
    ),
    CompilerExample(
        "conv2d_reference",
        conv2d_path,
        {target: _common_artifact_stages(runtime=False) for target in FOUNDATION_TARGETS},
        runtime_args=(
            np.arange(1 * 4 * 4 * 1, dtype=np.float32).reshape(1, 4, 4, 1),
            np.ones((3, 3, 1, 2), dtype=np.float32),
        ),
    ),
    CompilerExample(
        "rmsnorm_safe",
        rmsnorm_path,
        # Exact Apple GPU and x86 proofs remain op- and architecture-specific.
        {target: _common_artifact_stages(runtime=False) for target in FOUNDATION_TARGETS},
        runtime_args=(np.array([[1.0, 2.0, 4.0]], dtype=np.float32),),
    ),
    CompilerExample(
        "flash_attn_contract",
        flash_attn_path,
        # Exact Apple GPU and x86 proofs remain op- and architecture-specific.
        {target: _common_artifact_stages(runtime=False) for target in FOUNDATION_TARGETS},
        runtime_args=(
            np.ones((1, 2, 4), dtype=np.float32),
            np.ones((1, 2, 4), dtype=np.float32),
            np.ones((1, 2, 4), dtype=np.float32),
        ),
    ),
    CompilerExample(
        "s8_tiny_diffusion_compile_slice",
        s8_compile_mlp_slice,
        {target: _common_artifact_stages(runtime=False) for target in FOUNDATION_TARGETS},
        runtime_args=(
            np.arange(6, dtype=np.float32).reshape(2, 3) / 10.0,
            np.ones((3, 4), dtype=np.float32) * 0.25,
        ),
    ),
    CompilerExample(
        "s8_tiny_attention_compile_slice",
        s8_compile_attention_slice,
        {target: _common_artifact_stages(runtime=False) for target in FOUNDATION_TARGETS},
        runtime_args=(
            np.arange(6, dtype=np.float32).reshape(2, 3) / 10.0,
            np.ones((3, 3), dtype=np.float32) * 0.25,
        ),
    ),
    CompilerExample(
        "s8_current_gen_qwen3_moe_compile_slice",
        s8_compile_qwen3_moe_slice,
        # MoE lowers through the compiler artifact path today, but the
        # reference launcher cannot yet materialize the symbolic route operand
        # for runtime execution. Keep this example as an honest artifact-only
        # current-gen conformance rung until that launcher gap closes.
        {target: _common_artifact_stages(runtime=False) for target in FOUNDATION_TARGETS},
        runtime_args=(
            np.linspace(-0.2, 0.4, 9, dtype=np.float32).reshape(3, 3),
            np.ones((3, 5), dtype=np.float32) * 0.10,
            np.ones((3, 5), dtype=np.float32) * 0.20,
            np.ones((5, 3), dtype=np.float32) * 0.15,
            np.stack([np.eye(3, dtype=np.float32), np.ones((3, 3), dtype=np.float32) * 0.25], axis=0),
            np.array([0, 0, 1], dtype=np.int64),
        ),
    ),
    )


# Module-level lazy accessor (PEP 562) so existing
# ``from tessera.testing.compiler_examples import COMPILER_EXAMPLE_MANIFEST``
# call sites still work, but ``examples/`` is only required on
# ``sys.path`` when the manifest is actually consulted.
_CACHED_MANIFEST: tuple[CompilerExample, ...] | None = None


def __getattr__(name):  # noqa: F811 (PEP 562 module __getattr__)
    if name == "COMPILER_EXAMPLE_MANIFEST":
        global _CACHED_MANIFEST
        if _CACHED_MANIFEST is None:
            _CACHED_MANIFEST = _build_manifest()
        return _CACHED_MANIFEST
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


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
    # Materialize the manifest lazily — same surface as the
    # module-level ``COMPILER_EXAMPLE_MANIFEST`` PEP 562 attribute,
    # but the explicit ``_build_manifest()`` call here documents that
    # we're triggering the ``examples.conformance`` import on demand.
    global _CACHED_MANIFEST
    if _CACHED_MANIFEST is None:
        _CACHED_MANIFEST = _build_manifest()
    results: list[CompilerExampleResult] = []
    for example in _CACHED_MANIFEST:
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


# ``COMPILER_EXAMPLE_MANIFEST`` is intentionally NOT a module-level
# binding — it's resolved lazily via PEP 562 ``__getattr__`` above
# so importing this module doesn't force ``examples.conformance``
# onto ``sys.path``.  The ``noqa: F822`` below tells ruff this is a
# deliberate dynamic re-export.
__all__ = [
    "COMPILER_EXAMPLE_MANIFEST",  # noqa: F822 (resolved via PEP 562 __getattr__)
    "COMPILER_STAGES",
    "FOUNDATION_TARGETS",
    "CompilerExample",
    "CompilerExampleResult",
    "qualify_compiler_example",
    "qualify_compiler_examples",
]
