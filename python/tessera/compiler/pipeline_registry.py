"""Arch-5 (2026-05-22) — pass pipeline registry.

Named pipelines like ``tessera-lower-to-x86`` and
``tessera-nvidia-pipeline-sm90`` are registered in C++ via
``mlir::PassPipelineRegistration``.  That's the *runtime* layer.
Tessera grew several pipelines and the metadata MLIR doesn't track —
required dialects, verifier insertion order, target applicability,
lit-fixture coverage — was scattered across comments and tests.

This module is the Python-side meta-layer that captures the metadata
once.  The drift gate at ``tests/unit/test_pipeline_registry.py``
asserts the registry stays in sync with the C++
``PassPipelineRegistration`` calls in
``src/transforms/lib/Passes.cpp``.

The registry doesn't replace MLIR's pipeline machinery — it layers on
top, so doc generation, audit dashboards, and pass-ordering decisions
("insert SymbolicDimEqualityPass after DistributionLowering") become
data changes instead of hunts through C++.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class PipelineSpec:
    """One named MLIR pass pipeline.

    Fields
    ------
    name
        The string registered via ``mlir::PassPipelineRegistration``
        (e.g., ``"tessera-lower-to-x86"``).  Drift-gated against
        ``src/transforms/lib/Passes.cpp``.
    passes
        Ordered tuple of pass names (as registered via
        ``OPT_PASS_REGISTRATION``).  Documents the intended chain;
        the C++ source remains authoritative for the actual
        composition.
    required_dialects
        Dialect names that must be loaded into the MLIRContext before
        the pipeline runs.  Cross-checked against
        :data:`tessera.compiler.dialects_manifest.REGISTERED_DIALECTS`.
    targets
        Hardware targets this pipeline produces artifacts for.
    verifier_passes
        Subset of ``passes`` that act as verifiers (don't transform
        IR — only check invariants).  Useful for the audit dashboard
        and for pass-ordering reasoning ("insert this before the
        first verifier").
    lit_fixtures
        Repo-relative paths to lit fixtures that exercise the
        pipeline.  Drift gate verifies each file exists and contains
        the pipeline name in a `RUN:` line.
    phase
        ``"lowering"`` / ``"tile_opt"`` / ``"target"`` / ``"verify"`` —
        which phase of the compile this pipeline owns.
    status
        ``"lit_verified"`` / ``"wired"`` / ``"planned"`` — how
        thoroughly the pipeline has been exercised.
    sprint
        Sprint label for archaeology.
    """

    name: str
    passes: tuple[str, ...]
    required_dialects: tuple[str, ...]
    targets: tuple[str, ...]
    verifier_passes: tuple[str, ...] = ()
    lit_fixtures: tuple[str, ...] = ()
    phase: str = "lowering"
    status: str = "wired"
    sprint: str = ""


# ─────────────────────────────────────────────────────────────────────────
# Registry — alphabetised by pipeline name.
# ─────────────────────────────────────────────────────────────────────────


REGISTERED_PIPELINES: tuple[PipelineSpec, ...] = (
    PipelineSpec(
        name="tessera-lower-to-apple_cpu",
        passes=(
            "tessera-effect-annotate",
            "canonicalize",
            "tessera-symdim-equality",
            "tessera-lower-to-apple_cpu",  # internal pass driving the lowering
        ),
        required_dialects=("tessera", "func", "scf", "arith"),
        targets=("apple_cpu",),
        verifier_passes=("tessera-symdim-equality",),
        lit_fixtures=("tests/tessera-ir/phase8/apple_cpu_lowering.mlir",),
        phase="lowering",
        status="lit_verified",
        sprint="Phase 8.1",
    ),
    PipelineSpec(
        name="tessera-lower-to-apple_cpu-runtime",
        passes=(
            "tessera-effect-annotate",
            "canonicalize",
            "tessera-matmul-to-apple-cpu",
        ),
        required_dialects=("tessera", "tessera_apple", "func", "scf", "arith"),
        targets=("apple_cpu",),
        lit_fixtures=("tests/tessera-ir/phase8/apple_cpu_runtime.mlir",),
        phase="lowering",
        status="lit_verified",
        sprint="Phase 8.2",
    ),
    PipelineSpec(
        name="tessera-lower-to-apple_gpu",
        passes=(
            "tessera-effect-annotate",
            "canonicalize",
            "tessera-lower-to-apple_gpu",
        ),
        required_dialects=("tessera", "tessera_apple", "func", "scf", "arith"),
        targets=("apple_gpu",),
        lit_fixtures=("tests/tessera-ir/phase8/apple_gpu_lowering.mlir",),
        phase="lowering",
        status="lit_verified",
        sprint="Phase 8.1",
    ),
    PipelineSpec(
        name="tessera-lower-to-apple_gpu-runtime",
        passes=(
            "tessera-effect-annotate",
            "canonicalize",
            "tessera-fuse-matmul-softmax-matmul",
            "tessera-fuse-matmul-softmax",
            "tessera-fuse-matmul-gelu",
            "tessera-fuse-matmul-rmsnorm",
            "tessera-lower-matmul-to-apple-gpu-mps",
            "tessera-lower-rope-to-apple-gpu-msl",
            "tessera-lower-flash-attn-to-apple-gpu-msl",
            "tessera-lower-softmax-to-apple-gpu-msl",
            "tessera-lower-gelu-to-apple-gpu-msl",
        ),
        required_dialects=("tessera", "tessera_apple", "func", "scf", "arith"),
        targets=("apple_gpu",),
        # No standalone lit fixture today — the runtime pipeline is
        # exercised end-to-end through tests/unit/test_apple_gpu_*.py
        # (the MPS + MSL dispatchers can't be invoked from raw IR).
        lit_fixtures=(),
        phase="lowering",
        status="lit_verified",
        sprint="Phase 8.3-8.4.7",
    ),
    PipelineSpec(
        name="tessera-lower-to-gpu",
        passes=(
            "tessera-effect-annotate",
            "canonicalize",
            "tessera-distribution-lower",
            "tessera-symdim-equality",
            "tessera-tile-ir-lowering",
            "tessera-warp-specialize",
            "tessera-async-copy-lowering",
        ),
        required_dialects=("tessera", "func", "scf", "arith"),
        targets=("nvidia_sm90", "nvidia_sm100", "nvidia_sm120"),
        verifier_passes=("tessera-symdim-equality",),
        lit_fixtures=(),
        phase="lowering",
        status="wired",
        sprint="Phase 3",
    ),
    PipelineSpec(
        name="tessera-lower-to-metalium",
        passes=(
            "tessera-effect-annotate",
            "canonicalize",
            "tessera-distribution-lower",
            "tessera-lower-to-metalium",
        ),
        required_dialects=("tessera", "func", "scf", "arith"),
        targets=("metalium",),
        lit_fixtures=(),
        phase="lowering",
        status="wired",
        sprint="Phase 7",
    ),
    PipelineSpec(
        name="tessera-lower-to-rocm",
        passes=(
            "tessera-effect-annotate",
            "canonicalize",
            "tessera-distribution-lower",
            "tessera-lower-to-rocm",
        ),
        required_dialects=("tessera", "func", "scf", "arith"),
        targets=("rocm_gfx90a", "rocm_gfx940", "rocm_gfx942", "rocm_gfx950"),
        lit_fixtures=(),
        phase="lowering",
        status="wired",
        sprint="Phase 8",
    ),
    PipelineSpec(
        name="tessera-lower-to-x86",
        passes=(
            "tessera-effect-annotate",
            "canonicalize",
            "tessera-distribution-lower",
            "tessera-symdim-equality",
            "tessera-tile",
            "tessera-tile-to-x86",
        ),
        required_dialects=("tessera", "func", "scf", "arith"),
        targets=("x86_amx", "x86_avx512"),
        verifier_passes=("tessera-symdim-equality",),
        lit_fixtures=("tests/tessera-ir/phase2/tile_to_x86.mlir",),
        phase="lowering",
        status="lit_verified",
        sprint="Phase 2",
    ),
    PipelineSpec(
        name="tessera-nvidia-pipeline",
        passes=(
            "tessera-effect-annotate",
            "canonicalize",
            "tessera-swiglu-fusion",
            "tessera-mla-fusion",
            "tessera-nsa-fusion",
            "tessera-hybrid-attention-fusion",
            "tessera-lightning-attention-fusion",
            "tessera-delta-attention-fusion",
            "tessera-distribution-lower",
            "tessera-symdim-equality",
            "tessera-tile-ir-lowering",
            "tessera-warp-specialize",
            "tessera-async-copy-lowering",
            "tessera-wgmma-lowering",
            "tessera-tma-descriptor",
            "tessera-nv-flash-attn-emitter",
        ),
        required_dialects=("tessera", "func", "scf", "arith"),
        targets=("nvidia_sm90",),  # default SM_90 chain
        verifier_passes=("tessera-symdim-equality",),
        lit_fixtures=("tests/tessera-ir/phase3/cuda13/nvidia_pipeline_alias.mlir",),
        phase="lowering",
        status="lit_verified",
        sprint="G-5",
    ),
    PipelineSpec(
        name="tessera-nvidia-pipeline-sm100",
        passes=(
            "tessera-effect-annotate",
            "canonicalize",
            "tessera-swiglu-fusion",
            "tessera-mla-fusion",
            "tessera-distribution-lower",
            "tessera-symdim-equality",
            "tessera-tile-ir-lowering",
            "tessera-warp-specialize",
            "tessera-async-copy-lowering",
            "tessera-tcgen05-lowering",  # Blackwell-specific
            "tessera-tma-descriptor",
        ),
        required_dialects=("tessera", "func", "scf", "arith"),
        targets=("nvidia_sm100",),
        verifier_passes=("tessera-symdim-equality",),
        lit_fixtures=("tests/tessera-ir/phase3/cuda13/nvidia_pipeline_alias.mlir",),
        phase="lowering",
        status="lit_verified",
        sprint="G-5",
    ),
    PipelineSpec(
        name="tessera-nvidia-pipeline-sm120",
        passes=(
            "tessera-effect-annotate",
            "canonicalize",
            "tessera-swiglu-fusion",
            "tessera-mla-fusion",
            "tessera-distribution-lower",
            "tessera-symdim-equality",
            "tessera-tile-ir-lowering",
            "tessera-warp-specialize",
            "tessera-async-copy-lowering",
            "tessera-tcgen05-lowering",
            "tessera-tma-descriptor",
        ),
        required_dialects=("tessera", "func", "scf", "arith"),
        targets=("nvidia_sm120",),
        verifier_passes=("tessera-symdim-equality",),
        lit_fixtures=("tests/tessera-ir/phase3/cuda13/nvidia_pipeline_alias.mlir",),
        phase="lowering",
        status="lit_verified",
        sprint="G-5",
    ),
    PipelineSpec(
        name="tessera-nvidia-pipeline-sm90",
        passes=(
            "tessera-effect-annotate",
            "canonicalize",
            "tessera-swiglu-fusion",
            "tessera-mla-fusion",
            "tessera-nsa-fusion",
            "tessera-hybrid-attention-fusion",
            "tessera-lightning-attention-fusion",
            "tessera-delta-attention-fusion",
            "tessera-distribution-lower",
            "tessera-symdim-equality",
            "tessera-tile-ir-lowering",
            "tessera-warp-specialize",
            "tessera-async-copy-lowering",
            "tessera-wgmma-lowering",
            "tessera-tma-descriptor",
            "tessera-nv-flash-attn-emitter",
        ),
        required_dialects=("tessera", "func", "scf", "arith"),
        targets=("nvidia_sm90",),
        verifier_passes=("tessera-symdim-equality",),
        lit_fixtures=("tests/tessera-ir/phase3/cuda13/nvidia_pipeline_alias.mlir",),
        phase="lowering",
        status="lit_verified",
        sprint="G-5",
    ),
)


# ─────────────────────────────────────────────────────────────────────────
# Public helpers
# ─────────────────────────────────────────────────────────────────────────


def all_pipeline_names() -> tuple[str, ...]:
    return tuple(sorted(p.name for p in REGISTERED_PIPELINES))


def pipeline_lookup(name: str) -> PipelineSpec | None:
    for spec in REGISTERED_PIPELINES:
        if spec.name == name:
            return spec
    return None


def pipelines_for_target(target: str) -> tuple[PipelineSpec, ...]:
    """Return all pipelines that produce artifacts for ``target``."""
    return tuple(p for p in REGISTERED_PIPELINES if target in p.targets)


__all__ = [
    "PipelineSpec",
    "REGISTERED_PIPELINES",
    "all_pipeline_names",
    "pipeline_lookup",
    "pipelines_for_target",
]
