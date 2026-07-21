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

from dataclasses import dataclass


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


TARGET_PIPELINE_SCHEMA_VERSION = "tessera.target_pipeline.v1"


@dataclass(frozen=True)
class TargetPipelineResolution:
    """Truthful compiler-pipeline ownership for one canonical target.

    ``current_driver_pipeline`` preserves the behavior of the existing Python
    driver. ``declared_pipeline`` names the best matching registered C++
    pipeline when one exists; keeping the two fields separate makes legacy
    family routing visible without silently changing compilation behavior.
    """

    target: str
    current_driver_pipeline: str
    declared_pipeline: str | None
    resolution_state: str
    target_scope: str
    owner: str
    registration_source: str
    level_b: str
    level_c: str
    reason: str
    driver_registration_source: str | None = None

    @property
    def has_declared_pipeline(self) -> bool:
        return self.declared_pipeline is not None


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
    # L-series linalg pilot (2026-06-02): full Graph→Schedule→Tile→Target Apple
    # CPU spine in one alias. Drives the whole stack from Graph IR (unlike the
    # artifact-only / op-direct aliases). cholesky is the carrier op + template
    # for the other linalg ops (tri_solve, svd).
    PipelineSpec(
        name="tessera-lower-to-apple_cpu-full",
        passes=(
            "tessera-effect-annotation",
            "tessera-distribution-lowering",
            "tessera-tiling",
            "tile-to-apple_cpu",
        ),
        required_dialects=("tessera", "tessera_apple", "func", "scf", "arith"),
        targets=("apple_cpu",),
        lit_fixtures=("tests/tessera-ir/phase8/apple_cholesky_full_spine.mlir",),
        phase="lowering",
        status="lit_verified",
        sprint="L-series",
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
    # L-series linalg pilot (2026-06-02): full Graph→Schedule→Tile→Target Apple
    # GPU spine in one alias (parallel to apple_cpu-full).
    PipelineSpec(
        name="tessera-lower-to-apple_gpu-full",
        passes=(
            "tessera-effect-annotation",
            "tessera-distribution-lowering",
            "tessera-tiling",
            "tile-to-apple_gpu",
        ),
        required_dialects=("tessera", "tessera_apple", "func", "scf", "arith"),
        targets=("apple_gpu",),
        lit_fixtures=("tests/tessera-ir/phase8/apple_cholesky_full_spine.mlir",),
        phase="lowering",
        status="lit_verified",
        sprint="L-series",
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
        # C4 (TIRx): the optional `legalize-dtypes` pipeline option inserts
        # tessera-compute-legalize before the contract check and
        # tessera-storage-legalize terminally (default off → byte-identical).
        # C2/C3/C6 (TIRx): the warp-spec legality gates run after
        # WarpSpecialization (which emits the #tile.* markers).
        passes=(
            "tessera-effect-annotate",
            "canonicalize",
            "tessera-distribution-lower",
            "tessera-layout-legality",
            "tessera-ir-contracts",
            "tessera-symdim-equality",
            "tessera-tile-ir-lowering",
            "tessera-warp-specialize",
            "tessera-tile-pipeline-legality",
            "tessera-warpspec-legality",
            "tessera-tile-barrier-reuse-legality",
            "tessera-async-copy-lowering",
            "tessera-tma-descriptor",
            # Second placement — over the typed #tile.barrier markers
            # NVTMADescriptor emits (kind consistency + arrival-count).
            "tessera-tile-pipeline-legality",
            "tessera-warpspec-legality",
        ),
        required_dialects=("tessera", "tile", "func", "scf", "arith"),
        targets=("nvidia_sm90", "nvidia_sm100", "nvidia_sm120"),
        verifier_passes=(
            "tessera-layout-legality",
            "tessera-ir-contracts",
            "tessera-symdim-equality",
            "tessera-tile-pipeline-legality",
            "tessera-warpspec-legality",
            "tessera-tile-barrier-reuse-legality",
        ),
        lit_fixtures=(),
        phase="lowering",
        status="wired",
        sprint="Phase 3 + C2/C3/C4/C6 (TIRx)",
    ),
    PipelineSpec(
        name="tessera-lower-to-nvidia-sm100",
        passes=("lower-tile-to-nvidia", "lower-tessera-nvidia-to-nvvm"),
        required_dialects=(
            "tile", "tessera_nvidia", "llvm", "nvvm", "func", "scf", "arith",
        ),
        targets=("nvidia_sm100",),
        lit_fixtures=(
            "src/compiler/codegen/tessera_gpu_backend_NVIDIA/test/nvidia/blackwell_to_nvvm_contract.mlir",
        ),
        phase="target",
        status="lit_verified",
        sprint="NVIDIA-E2E-2",
    ),
    PipelineSpec(
        name="tessera-lower-to-nvidia-sm120",
        passes=("lower-tile-to-nvidia", "lower-tessera-nvidia-to-nvvm"),
        required_dialects=(
            "tile", "tessera_nvidia", "llvm", "nvvm", "func", "scf", "arith",
        ),
        targets=("nvidia_sm120",),
        lit_fixtures=(
            "src/compiler/codegen/tessera_gpu_backend_NVIDIA/test/nvidia/sm120_nvfp4_matmul_kernel.mlir",
        ),
        phase="target",
        status="lit_verified",
        sprint="NVIDIA-E2E-2",
    ),
    PipelineSpec(
        name="tessera-lower-to-nvidia-sm90",
        passes=("lower-tile-to-nvidia", "lower-tessera-nvidia-to-nvvm"),
        required_dialects=(
            "tile", "tessera_nvidia", "llvm", "nvvm", "func", "scf", "arith",
        ),
        targets=("nvidia_sm90",),
        lit_fixtures=(
            "src/compiler/codegen/tessera_gpu_backend_NVIDIA/test/nvidia/hopper_to_nvvm_contract.mlir",
        ),
        phase="target",
        status="lit_verified",
        sprint="NVIDIA-E2E-2",
    ),
    PipelineSpec(
        name="tessera-lower-to-rocm",
        passes=(
            "tessera-effect-annotate",
            "canonicalize",
            "tessera-distribution-lower",
            "rocm-wave-lds-pipeline",
            "rocm-wave-lds-legality",
            "tessera-lower-to-rocm",
            "generate-rocm-softmax-kernel",
            "generate-rocm-reduce-kernel",
            "generate-rocm-paged-kv-read-kernel",
        ),
        required_dialects=("tessera", "tile", "tessera_rocm", "func", "scf", "arith"),
        targets=(
            "rocm",
            "rocm_gfx90a",
            "rocm_gfx940",
            "rocm_gfx942",
            "rocm_gfx950",
            "rocm_gfx1100",
            "rocm_gfx1151",
            "rocm_gfx1200",
            "rocm_gfx1201",
            "rocm_gfx1250",
        ),
        verifier_passes=("rocm-wave-lds-legality",),
        lit_fixtures=(
            "src/compiler/codegen/Tessera_ROCM_Backend/test/rocm/gfx1151_tile_softmax_kernel.mlir",
            "src/compiler/codegen/Tessera_ROCM_Backend/test/rocm/gfx1151_tile_reduce_kernel.mlir",
            "src/compiler/codegen/Tessera_ROCM_Backend/test/rocm/gfx1151_tile_paged_kv_read_kernel.mlir",
        ),
        phase="lowering",
        status="lit_verified",
        sprint="Phase 8 + ROCm Tile-IR convergence + ROCM-E2E-1/-2",
    ),
    PipelineSpec(
        name="tessera-lower-to-x86",
        passes=(
            "tessera-effect-annotate",
            "canonicalize",
            "tessera-distribution-lower",
            "tessera-layout-legality",
            "tessera-ir-contracts",
            "tessera-symdim-equality",
            "tessera-tile",
            "tessera-tile-to-x86",
        ),
        required_dialects=("tessera", "tile", "func", "scf", "arith"),
        targets=("cpu", "x86", "x86_amx", "x86_avx512"),
        verifier_passes=("tessera-symdim-equality",),
        lit_fixtures=(
            "tests/tessera-ir/phase2/tile_to_x86.mlir",
            "tests/tessera-ir/phase2/tile_x86_e2e.mlir",
        ),
        phase="lowering",
        status="lit_verified",
        sprint="Phase 2 + X86-E2E-1",
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
            "tessera-layout-legality",
            "tessera-ir-contracts",
            "tessera-symdim-equality",
            "tessera-tile-ir-lowering",
            "tessera-warp-specialize",
            # C2/C3/C6 (TIRx): warp-spec legality gates on the markers
            # WarpSpecialization emits (phase asymmetry, reuse, structure).
            "tessera-tile-pipeline-legality",
            "tessera-warpspec-legality",
            "tessera-tile-barrier-reuse-legality",
            "tessera-async-copy-lowering",
            "tessera-wgmma-lowering",
            "tessera-tma-descriptor",
            # Second placement — over the typed #tile.barrier markers
            # NVTMADescriptor emits (kind consistency + arrival-count).
            "tessera-tile-pipeline-legality",
            "tessera-warpspec-legality",
            "tessera-nv-flash-attn-emitter",
        ),
        required_dialects=("tessera", "tile", "func", "scf", "arith"),
        targets=("nvidia_sm90",),  # default SM_90 chain
        verifier_passes=(
            "tessera-layout-legality",
            "tessera-ir-contracts",
            "tessera-symdim-equality",
            "tessera-tile-pipeline-legality",
            "tessera-warpspec-legality",
            "tessera-tile-barrier-reuse-legality",
        ),
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
            "tessera-nsa-fusion",
            "tessera-hybrid-attention-fusion",
            "tessera-lightning-attention-fusion",
            "tessera-delta-attention-fusion",
            "tessera-distribution-lower",
            "tessera-layout-legality",
            "tessera-ir-contracts",
            "tessera-symdim-equality",
            "tessera-tile-ir-lowering",
            "tessera-warp-specialize",
            # Datacenter Blackwell retains typed MMA/attention carriers until
            # the exact TCGEN05/TMEM backend pipeline consumes them.
            "tessera-tile-pipeline-legality",
            "tessera-warpspec-legality",
            "tessera-tile-barrier-reuse-legality",
            "tessera-async-copy-lowering",
            "tessera-tma-descriptor",
            # Second placement — over the typed #tile.barrier markers.
            "tessera-tile-pipeline-legality",
            "tessera-warpspec-legality",
        ),
        required_dialects=("tessera", "tile", "func", "scf", "arith"),
        targets=("nvidia_sm100",),
        verifier_passes=(
            "tessera-layout-legality",
            "tessera-ir-contracts",
            "tessera-symdim-equality",
            "tessera-tile-pipeline-legality",
            "tessera-warpspec-legality",
            "tessera-tile-barrier-reuse-legality",
        ),
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
            "tessera-nsa-fusion",
            "tessera-hybrid-attention-fusion",
            "tessera-lightning-attention-fusion",
            "tessera-delta-attention-fusion",
            "tessera-distribution-lower",
            "tessera-layout-legality",
            "tessera-ir-contracts",
            "tessera-symdim-equality",
            "tessera-tile-ir-lowering",
            "tessera-warp-specialize",
            # Consumer Blackwell retains typed warp-MMA/attention carriers
            # until the exact SM120 backend pipeline consumes them.
            "tessera-tile-pipeline-legality",
            "tessera-warpspec-legality",
            "tessera-tile-barrier-reuse-legality",
            "tessera-async-copy-lowering",
            "tessera-tma-descriptor",
            # Second placement — over the typed #tile.barrier markers.
            "tessera-tile-pipeline-legality",
            "tessera-warpspec-legality",
        ),
        required_dialects=("tessera", "tile", "func", "scf", "arith"),
        targets=("nvidia_sm120",),
        verifier_passes=(
            "tessera-layout-legality",
            "tessera-ir-contracts",
            "tessera-symdim-equality",
            "tessera-tile-pipeline-legality",
            "tessera-warpspec-legality",
            "tessera-tile-barrier-reuse-legality",
        ),
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
            "tessera-layout-legality",
            "tessera-ir-contracts",
            "tessera-symdim-equality",
            "tessera-tile-ir-lowering",
            "tessera-warp-specialize",
            # C2/C3/C6 (TIRx): warp-spec legality gates on the markers
            # WarpSpecialization emits (phase asymmetry, reuse, structure).
            "tessera-tile-pipeline-legality",
            "tessera-warpspec-legality",
            "tessera-tile-barrier-reuse-legality",
            "tessera-async-copy-lowering",
            "tessera-wgmma-lowering",
            "tessera-tma-descriptor",
            # Second placement — over the typed #tile.barrier markers
            # NVTMADescriptor emits (kind consistency + arrival-count).
            "tessera-tile-pipeline-legality",
            "tessera-warpspec-legality",
            "tessera-nv-flash-attn-emitter",
        ),
        required_dialects=("tessera", "tile", "func", "scf", "arith"),
        targets=("nvidia_sm90",),
        verifier_passes=(
            "tessera-layout-legality",
            "tessera-ir-contracts",
            "tessera-symdim-equality",
            "tessera-tile-pipeline-legality",
            "tessera-warpspec-legality",
            "tessera-tile-barrier-reuse-legality",
        ),
        lit_fixtures=("tests/tessera-ir/phase3/cuda13/nvidia_pipeline_alias.mlir",),
        phase="lowering",
        status="lit_verified",
        sprint="G-5",
    ),
    # Pipeline-parallel layer (2026-06-23): partition into stages → insert
    # send/recv SSA rewrites → prove the 1F1B schedule. Drives a function from
    # unpartitioned IR to a verified 1F1B pipeline.
    PipelineSpec(
        name="tessera-pipeline",
        passes=(
            "tessera-pipeline-partition",
            "tessera-pipeline-stage-insertion",
            "tessera-pipeline-schedule-legality",
        ),
        required_dialects=("tessera", "func"),
        targets=("nvidia_sm90", "nvidia_sm100", "nvidia_sm120"),
        verifier_passes=("tessera-pipeline-schedule-legality",),
        lit_fixtures=("tests/tessera-ir/phase4/pipeline_schedule_legality.mlir",),
        phase="lowering",
        status="lit_verified",
        sprint="Pipeline-PP",
    ),
)


# Exact-target ownership is intentionally separate from ``PipelineSpec.targets``.
# A target may have a registered C++ pipeline while the current Python driver
# still uses a legacy family alias or artifact-only sentinel.  E2E-SPINE-0 makes
# that mismatch data instead of changing behavior while auditing it.
TARGET_PIPELINE_RESOLUTIONS: tuple[TargetPipelineResolution, ...] = (
    TargetPipelineResolution(
        "apple_cpu", "tessera-lower-to-apple_cpu",
        "tessera-lower-to-apple_cpu", "declared_exact", "host",
        "apple", "src/compiler/codegen/Tessera_Apple_Backend/lib/Target/Apple/Passes.cpp",
        "partial", "partial",
        "Canonical compilation packages the static single-result f32 BMM and linalg ABI scope; multi-result and unsupported contracts retain their explicit value/reference routes.",
    ),
    TargetPipelineResolution(
        "apple_gpu", "tessera-lower-to-apple_gpu-runtime",
        "tessera-lower-to-apple_gpu-runtime", "declared_exact", "exact_architecture",
        "apple", "src/compiler/codegen/Tessera_Apple_Backend/lib/Target/Apple/Passes.cpp",
        "partial", "partial",
        "Canonical compilation promotes the statically packageable, fresh-runtime Apple ABI scope to native images and descriptors; unsupported, composite, and multi-result routes retain their explicit legacy state.",
    ),
    TargetPipelineResolution(
        "cpu", "tessera-lower-to-x86", "tessera-lower-to-x86",
        "family_selector", "host", "shared_x86", "src/transforms/lib/Passes.cpp",
        "partial", "absent",
        "The generic CPU frontend records the x86 lowering alias while runtime execution may remain reference-backed.",
    ),
    TargetPipelineResolution(
        "nvidia_sm100", "tessera-lower-to-gpu",
        "tessera-lower-to-nvidia-sm100", "declared_exact",
        "exact_architecture", "nvidia", "src/compiler/codegen/tessera_gpu_backend_NVIDIA/lib/Conversion/NVIDIALowering.cpp",
        "partial", "absent",
        "The exact SM100 Tile-to-NVVM builder is registered; native packaging and exact-device execution remain gated.",
        driver_registration_source="src/transforms/lib/Passes.cpp",
    ),
    TargetPipelineResolution(
        "nvidia_sm120", "tessera-lower-to-gpu",
        "tessera-lower-to-nvidia-sm120", "declared_exact",
        "exact_architecture", "nvidia", "src/compiler/codegen/tessera_gpu_backend_NVIDIA/lib/Conversion/NVIDIALowering.cpp",
        "partial", "absent",
        "The exact SM120 Tile-to-NVVM builder produces the compiler-owned f16/NVFP4 native-image slice.",
        driver_registration_source="src/transforms/lib/Passes.cpp",
    ),
    TargetPipelineResolution(
        "nvidia_sm80", "tessera-lower-to-gpu", None,
        "unsupported_no_exact_pipeline", "exact_architecture", "nvidia",
        "src/transforms/lib/Passes.cpp", "absent", "absent",
        "No exact SM80 C++ pipeline is registered; the current driver retains its legacy artifact alias.",
    ),
    TargetPipelineResolution(
        "nvidia_sm90", "tessera-lower-to-gpu",
        "tessera-lower-to-nvidia-sm90", "declared_exact",
        "exact_architecture", "nvidia", "src/compiler/codegen/tessera_gpu_backend_NVIDIA/lib/Conversion/NVIDIALowering.cpp",
        "partial", "absent",
        "The exact SM90 Tile-to-NVVM builder is registered; canonical native-image packaging and exact-device proof are absent.",
        driver_registration_source="src/transforms/lib/Passes.cpp",
    ),
    TargetPipelineResolution(
        "rocm", "tessera-lower-to-rocm", "tessera-lower-to-rocm",
        "family_selector", "family_selector", "rocm",
        "src/compiler/codegen/Tessera_ROCM_Backend/lib/Conversion/Passes.cpp",
        "partial", "absent",
        "The family pipeline lowers the typed GEMM subset; other native lanes still use runtime-authored directives.",
    ),
    TargetPipelineResolution(
        "rocm_gfx1100", "tessera-target-artifact", "tessera-lower-to-rocm",
        "declared_shared_builder", "exact_architecture", "rocm",
        "src/compiler/codegen/Tessera_ROCM_Backend/lib/Conversion/Passes.cpp",
        "partial", "absent", "The ROCm pipeline is family-shared; the exact driver route remains artifact-only.",
    ),
    TargetPipelineResolution(
        "rocm_gfx1151", "tessera-target-artifact", "tessera-lower-to-rocm",
        "declared_shared_builder", "exact_architecture", "rocm",
        "src/compiler/codegen/Tessera_ROCM_Backend/lib/Conversion/Passes.cpp",
        "partial", "absent", "The family-shared pipeline now owns typed gfx1151 softmax and f32 arbitrary-axis reduction producers and HSACO packages; target-wide Level C remains absent while other families use legacy routes.",
    ),
    TargetPipelineResolution(
        "rocm_gfx1200", "tessera-target-artifact", "tessera-lower-to-rocm",
        "declared_shared_builder", "exact_architecture", "rocm",
        "src/compiler/codegen/Tessera_ROCM_Backend/lib/Conversion/Passes.cpp",
        "partial", "absent", "The ROCm pipeline is family-shared and exact-device execution remains gated.",
    ),
    TargetPipelineResolution(
        "rocm_gfx1201", "tessera-target-artifact", "tessera-lower-to-rocm",
        "declared_shared_builder", "exact_architecture", "rocm",
        "src/compiler/codegen/Tessera_ROCM_Backend/lib/Conversion/Passes.cpp",
        "partial", "absent", "The ROCm pipeline is family-shared and exact-device execution remains gated.",
    ),
    TargetPipelineResolution(
        "rocm_gfx1250", "tessera-target-artifact", "tessera-lower-to-rocm",
        "declared_shared_builder", "exact_architecture", "rocm",
        "src/compiler/codegen/Tessera_ROCM_Backend/lib/Conversion/Passes.cpp",
        "partial", "absent", "The ROCm pipeline is family-shared and exact-device execution remains gated.",
    ),
    TargetPipelineResolution(
        "rocm_gfx90a", "tessera-target-artifact", "tessera-lower-to-rocm",
        "declared_shared_builder", "exact_architecture", "rocm",
        "src/compiler/codegen/Tessera_ROCM_Backend/lib/Conversion/Passes.cpp",
        "partial", "absent", "The ROCm pipeline is family-shared and exact-device execution remains gated.",
    ),
    TargetPipelineResolution(
        "rocm_gfx940", "tessera-target-artifact", "tessera-lower-to-rocm",
        "declared_shared_builder", "exact_architecture", "rocm",
        "src/compiler/codegen/Tessera_ROCM_Backend/lib/Conversion/Passes.cpp",
        "partial", "absent", "The ROCm pipeline is family-shared and exact-device execution remains gated.",
    ),
    TargetPipelineResolution(
        "rocm_gfx942", "tessera-target-artifact", "tessera-lower-to-rocm",
        "declared_shared_builder", "exact_architecture", "rocm",
        "src/compiler/codegen/Tessera_ROCM_Backend/lib/Conversion/Passes.cpp",
        "partial", "absent", "The ROCm pipeline is family-shared and exact-device execution remains gated.",
    ),
    TargetPipelineResolution(
        "rocm_gfx950", "tessera-target-artifact", "tessera-lower-to-rocm",
        "declared_shared_builder", "exact_architecture", "rocm",
        "src/compiler/codegen/Tessera_ROCM_Backend/lib/Conversion/Passes.cpp",
        "partial", "absent", "The ROCm pipeline is family-shared and exact-device execution remains gated.",
    ),
    TargetPipelineResolution(
        "x86", "tessera-lower-to-x86", "tessera-lower-to-x86",
        "declared_exact", "host", "x86", "src/transforms/lib/Passes.cpp",
        "partial", "absent",
        "X86-E2E-1 emits typed softmax, reduction, rank-2 f32 matmul, and basic/extended f32 MHA C-ABI calls, then packages the stable AVX-512 shared object with canonical launch descriptors; other families remain executor-owned.",
    ),
)


@dataclass(frozen=True)
class CompilationSpineInventoryRow:
    target: str
    family: str
    runtime_backend: str
    current_driver_pipeline: str
    declared_pipeline: str
    resolution_state: str
    target_scope: str
    level_a: str
    level_b: str
    level_c: str
    owner: str
    reason: str


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


def target_pipeline_lookup(target: str) -> TargetPipelineResolution | None:
    for resolution in TARGET_PIPELINE_RESOLUTIONS:
        if resolution.target == target:
            return resolution
    return None


def current_driver_pipeline_map() -> dict[str, str]:
    """Return the behavior-preserving driver map from the ownership registry."""
    return {
        resolution.target: resolution.current_driver_pipeline
        for resolution in TARGET_PIPELINE_RESOLUTIONS
    }


def _level_a_status(target: str) -> str:
    """Derive Level-A runtime truth from the canonical execution matrix."""
    from .execution_matrix import all_rows

    rows = [
        row for row in all_rows()
        if row.executable and (row.target == target or row.evidence_target == target)
    ]
    native_kinds = {"native_cpu", "native_gpu", "cpu_accelerate"}
    if any(row.execution_kind in native_kinds for row in rows):
        return "native"
    if rows:
        return "reference"
    return "absent"


def compilation_spine_inventory() -> tuple[CompilationSpineInventoryRow, ...]:
    """Join target, pipeline, and runtime registries into Level-A/B/C truth."""
    from .capabilities import TARGET_CAPABILITIES

    rows: list[CompilationSpineInventoryRow] = []
    for resolution in TARGET_PIPELINE_RESOLUTIONS:
        capability = TARGET_CAPABILITIES[resolution.target]
        rows.append(CompilationSpineInventoryRow(
            target=resolution.target,
            family=capability.family,
            runtime_backend=capability.runtime_backend,
            current_driver_pipeline=resolution.current_driver_pipeline,
            declared_pipeline=resolution.declared_pipeline or "",
            resolution_state=resolution.resolution_state,
            target_scope=resolution.target_scope,
            level_a=_level_a_status(resolution.target),
            level_b=resolution.level_b,
            level_c=resolution.level_c,
            owner=resolution.owner,
            reason=resolution.reason,
        ))
    return tuple(rows)


COMPILATION_SPINE_CSV_COLUMNS: tuple[str, ...] = (
    "schema", "target", "family", "runtime_backend",
    "current_driver_pipeline", "declared_pipeline", "resolution_state",
    "target_scope", "level_a", "level_b", "level_c", "owner", "reason",
)


def render_compilation_spine_csv() -> str:
    import csv
    import io

    buffer = io.StringIO()
    writer = csv.writer(buffer, lineterminator="\n")
    writer.writerow(COMPILATION_SPINE_CSV_COLUMNS)
    for row in compilation_spine_inventory():
        writer.writerow((
            TARGET_PIPELINE_SCHEMA_VERSION,
            row.target,
            row.family,
            row.runtime_backend,
            row.current_driver_pipeline,
            row.declared_pipeline,
            row.resolution_state,
            row.target_scope,
            row.level_a,
            row.level_b,
            row.level_c,
            row.owner,
            row.reason,
        ))
    return buffer.getvalue()


def render_compilation_spine_markdown() -> str:
    lines = [
        "# Canonical compilation spine inventory",
        "",
        "**Generated from target capabilities, pipeline ownership, and the runtime execution matrix. Do not hand-edit.**",
        "",
        "Level A is native/reference runtime execution, Level B is a typed compiler seam, and Level C is the canonical Graph→native-image→launch path. `partial` never implies fleet closure.",
        "",
        "| Target | Driver pipeline | Declared pipeline | Resolution | A | B | C | Owner |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for row in compilation_spine_inventory():
        lines.append(
            f"| `{row.target}` | `{row.current_driver_pipeline}` | "
            f"`{row.declared_pipeline or '-'}` | `{row.resolution_state}` | "
            f"`{row.level_a}` | `{row.level_b}` | `{row.level_c}` | `{row.owner}` |"
        )
    lines.extend((
        "",
        "The canonical CSV companion retains target scope, runtime backend, and the complete resolution reason.",
        "",
    ))
    return "\n".join(lines)


__all__ = [
    "COMPILATION_SPINE_CSV_COLUMNS",
    "CompilationSpineInventoryRow",
    "PipelineSpec",
    "REGISTERED_PIPELINES",
    "TARGET_PIPELINE_RESOLUTIONS",
    "TARGET_PIPELINE_SCHEMA_VERSION",
    "TargetPipelineResolution",
    "all_pipeline_names",
    "compilation_spine_inventory",
    "current_driver_pipeline_map",
    "pipeline_lookup",
    "pipelines_for_target",
    "render_compilation_spine_csv",
    "render_compilation_spine_markdown",
    "target_pipeline_lookup",
]
