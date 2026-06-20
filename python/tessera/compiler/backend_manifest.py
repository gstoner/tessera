"""Sprint E — Backend kernel manifest (2026-05-11).

Synthesizes the per-op × per-target × per-dtype backend kernel matrix
from ``capabilities.TARGET_CAPABILITIES`` + the Apple GPU kernel inventory
(``docs/apple_backend.md``)
+ explicit registrations.  Lets the primitive coverage registry promote
the ``backend_kernel`` axis from a binary partial/planned status to a
per-target tracking with explicit kernel availability per dtype.

Schema:

    BackendKernelEntry(
        target="apple_gpu",     # normalized target name
        status="fused"          # "fused" | "reference" | "artifact_only" | "planned"
        dtypes=("fp32", "fp16", "bf16"),
        feature_flags=("metal", "mps", "msl"),
        notes="...",
    )

    BackendKernelManifest = dict[op_name, list[BackendKernelEntry]]

Status semantics:
    "fused"          : backend ships an optimized fused kernel
                       (e.g., Apple GPU matmul→softmax MSL kernel; x86 AMX GEMM)
    "reference"      : backend can execute via numpy reference / cblas /
                       Accelerate.cblas_sgemm (correct but not perf-tuned)
    "artifact_only"  : Target IR artifact lit-testable; execution gated
                       on hardware availability (NVIDIA/ROCm without GPU)
    "planned"        : no implementation today; intended for the target

The manifest does NOT change any axis values in the primitive coverage
registry by itself.  Sprint E-3 attaches it as
``metadata["backend_kernel_manifest"] = [entry_dict, ...]`` for the
dashboard to surface.  The ``backend_kernel`` contract axis stays at
``partial``/``planned`` until real GPU execution lights up (Phase G/H/I).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Optional, Tuple

from .capabilities import TARGET_CAPABILITIES, canonical_op
from .op_catalog import OP_SPECS

# Sprint G-3 type aliases — keep imports minimal at the entry point and
# expose them as module-level shorthands for the new BackendKernelEntry
# optional fields.
Optional_str = Optional[str]
Optional_float = Optional[float]
Optional_triple = Optional[Tuple[int, int, int]]
Optional_quad = Optional[Tuple[int, int, int, int]]


_FUSED_KERNEL_STATUS = "fused"
_REFERENCE_STATUS = "reference"
_ARTIFACT_STATUS = "artifact_only"
_COMPILEABLE_STATUS = "compileable"   # Sprint G/H follow-up: passes ptxas/hipcc
_PLANNED_STATUS = "planned"
# PK5 (2026-05-31) — Apple Metal `.mtlpackage` packaged kernel. The
# kernel ships as a pre-compiled Metal package (output of Core ML
# Tools / Xcode); Tessera loads it at runtime via PK1's
# `tessera_apple_gpu_mlpkg_compile`, reflects + dispatches per PK2-PK4.
# Distinct from `fused` (in-tree MSL source) and `artifact_only` (IR
# emits but no runtime). When status is "packaged" the entry MUST
# carry a non-empty ``packaged_pipeline_path`` field naming the
# `.mtlpackage` directory.
_PACKAGED_STATUS = "packaged"

# Arch-3 (2026-05-22) — top rung of the readiness ladder.  An entry
# only qualifies for ``hardware_verified`` when it carries BOTH a
# ``runtime_symbol`` AND an ``execute_compare_fixture`` checked into
# the test tree.  This is the missing definition that makes the
# registry's per-primitive ``backend_kernel = "complete"`` axis a
# computable property (a primitive is complete iff every declared
# target row is ``hardware_verified``).  Today (2026-05-22) zero
# entries claim this status — flipping the first one requires real
# NVIDIA / ROCm hardware proof.
_HARDWARE_VERIFIED_STATUS = "hardware_verified"

_VALID_STATUSES = frozenset({
    _FUSED_KERNEL_STATUS,
    _REFERENCE_STATUS,
    _ARTIFACT_STATUS,
    _COMPILEABLE_STATUS,
    _PLANNED_STATUS,
    _HARDWARE_VERIFIED_STATUS,
    _PACKAGED_STATUS,
})


# ─────────────────────────────────────────────────────────────────────────────
# Benchmark metadata (P1, 2026-06-10 — DEEP_COMPILER_AUDIT follow-on).
#
# A structured, uniform descriptor attached to every benchmarked manifest row.
# Before this, hot-path benchmark linkage was just a loose ``benchmark_json``
# pointer — it could point at a baseline that doesn't even contain the op's row
# (softmax / rmsnorm / flash_attn / bmm all pointed at apple_gpu_hot_paths.json
# but have no row there). ``benchmark_metadata`` makes the truth explicit and
# even: which hot-path group, which harness exercises the op, and whether it is
# *actually* ratcheted (has a recorded baseline row) vs merely benchmarked.
# ─────────────────────────────────────────────────────────────────────────────

#: Hot-path groups a benchmarked kernel can belong to.
BENCHMARK_HOT_PATH_GROUPS: frozenset[str] = frozenset({
    "gemm", "norm", "activation", "attention", "conv", "moe",
    "fused_epilogue", "packaged", "ga_ebm",
})


@dataclass(frozen=True)
class BenchmarkMetadata:
    """Structured hot-path benchmark descriptor for a manifest row.

    hot_path_group : coarse group the kernel belongs to (BENCHMARK_HOT_PATH_GROUPS)
    harness        : repo-relative benchmark file that exercises the op
    ratcheted      : True iff a recorded baseline row gates this op's latency
    ratchet_key    : the ``op`` key in the baseline (required when ratcheted)
    notes          : free-form provenance
    """
    hot_path_group: str
    harness: str
    ratcheted: bool = False
    ratchet_key: Optional[str] = None
    notes: str = ""

    def __post_init__(self) -> None:
        if self.hot_path_group not in BENCHMARK_HOT_PATH_GROUPS:
            raise ValueError(
                f"BenchmarkMetadata.hot_path_group must be one of "
                f"{sorted(BENCHMARK_HOT_PATH_GROUPS)}, got {self.hot_path_group!r}")
        if self.ratcheted and not self.ratchet_key:
            raise ValueError(
                "BenchmarkMetadata.ratcheted=True requires a ratchet_key "
                "(the op's row key in the ratchet baseline)")

    def to_dict(self) -> dict[str, object]:
        out: dict[str, object] = {
            "hot_path_group": self.hot_path_group,
            "harness": self.harness,
            "ratcheted": self.ratcheted,
        }
        if self.ratchet_key is not None:
            out["ratchet_key"] = self.ratchet_key
        if self.notes:
            out["notes"] = self.notes
        return out


@dataclass(frozen=True)
class BackendKernelEntry:
    """One row of the per-op × per-target × per-dtype matrix.

    All dtype strings are canonical (the dataclass normalizes at
    ``__post_init__``).

    Sprint G-3 (2026-05-11) added the toolchain-pin + tile-shape fields:

    Attributes
    ----------
    target : str
        Normalized target name (``"nvidia_sm90"`` / ``"rocm_gfx942"`` /
        ``"apple_gpu"`` / etc.).
    status : str
        One of ``fused`` / ``reference`` / ``compileable`` /
        ``artifact_only`` / ``planned``.
    dtypes : tuple[str, ...]
        Canonical dtype set for this entry.  Aliases normalized at
        construction.

        **Semantics depend on ``status`` (do not overread!):**

          * ``fused`` → dtypes the fused kernel **actually supports
            today** (e.g., Apple GPU matmul ships
            ``("fp32", "fp16", "bf16")``).
          * ``reference`` → dtypes the **Python / numpy reference path
            runs today**.  The M7 ``complex_*`` family is fp32-only on
            the reference path; tagging fp16/bf16 here would be an
            overclaim.
          * ``compileable`` → dtypes the lowering pipeline emits IR for.
            Compilation works; execution may not.
          * ``artifact_only`` → dtypes the Target IR artifact compiles
            for under the pinned toolchain (CUDA 13.2 U1 / ROCm 7.2.3).
            No host execution.
          * ``planned`` → **target kernel dtypes for the unbuilt
            kernel** — the matrix the future native kernel will
            support, NOT the dtypes that run today.  When ``status``
            flips to ``fused``, this set becomes the live kernel's
            dtype matrix unchanged.

        Concretely: a row reading
        ``BackendKernelEntry(target="apple_gpu", status="planned",
        dtypes=("fp32", "fp16", "bf16"))`` means "the planned MSL
        kernel will support fp32/fp16/bf16 — today only fp32 runs,
        via the matching ``status=reference`` row on a CPU target."
    feature_flags : tuple[str, ...]
        Target-specific feature flags (``wgmma`` / ``tcgen05_pair`` /
        ``mfma_f8`` / ``msl`` / etc.).
    notes : str
        Free-form context.

    Sprint G-3 fields (all optional — meaningful only on tensor-core
    targets):
    cuda_arch_min : str | None
        Minimum NVIDIA SM arch the kernel compiles for, e.g., ``"sm_90a"``
        (Hopper WGMMA) or ``"sm_100a"`` (Blackwell tcgen05).  ``None``
        for non-NVIDIA targets.
    nvcc_version_min : str | None
        Minimum nvcc release that emits this kernel correctly.  Today
        all NVIDIA entries pin to ``"13.3"``.
    wgmma_shape : tuple[int, int, int] | None
        ``(M, N, K)`` for the WGMMA tile.  ``(64, 256, 16)`` is the
        canonical bf16/fp16 Hopper shape; FP8 lowers to
        ``(64, 256, 32)``; FP4/NVFP4 to ``(64, 256, 64)``.  ``None``
        for non-NVIDIA targets or non-tensor-core kernels.
    cluster_size : tuple[int, int, int] | None
        Thread-block cluster shape for SM_90+ (``(1, 1, 1)`` when
        clusters disabled).  ``None`` on pre-Hopper or non-NVIDIA.
    mfma_shape : tuple[int, int, int, int] | None
        ``(M, N, K, K_blocks)`` for AMD MFMA instructions.  Looked up
        against ``rocm_target.mfma_variants(arch)``.  ``None`` for
        non-AMD targets.
    hipcc_version_min : str | None
        Minimum hipcc release.  Today all ROCm entries pin to ``"7.2.3"``.
    expected_mfu : float | None
        Target MFU as a fraction of peak (e.g., ``0.65`` = 65%).  Used
        by ``perf_gate.py`` once execution lights up.
    roofline_target : str | None
        Free-form roofline characterization (e.g., ``"compute-bound at
        N >= 2048"``, ``"memory-bound at K < 256"``).
    """

    target: str
    status: str
    dtypes: tuple[str, ...] = ()
    feature_flags: tuple[str, ...] = ()
    notes: str = ""
    # Sprint G-3 — toolchain pins + tile-shape contracts
    cuda_arch_min: Optional_str = None
    nvcc_version_min: Optional_str = None
    wgmma_shape: Optional_triple = None
    cluster_size: Optional_triple = None
    mfma_shape: Optional_quad = None
    hipcc_version_min: Optional_str = None
    expected_mfu: Optional_float = None
    roofline_target: Optional_str = None
    mma_descriptor: Optional[object] = None
    """A1 (2026-06-18) — the unified cooperative-matrix descriptor
    (:class:`rocm_mma.MmaDescriptor`) for a representative ``(arch, dtype)`` of
    this entry: the MFMA(CDNA)/WMMA(RDNA) instruction with its *derived* A/B
    operand layouts + ``k_width`` (the "shape is the anchor" contract from
    rocWMMA / AMD Gluon).  Complements the coarser :attr:`mfma_shape` with the
    operand-layout/packing detail.  Type-erased to ``object`` to keep this a
    leaf-importable module; validated structurally in ``__post_init__`` (ROCm
    targets only).  ``None`` for non-GEMM / non-ROCm entries."""
    # Arch-3 (2026-05-22) — execute-and-compare hooks.  All optional so
    # existing constructors continue to work.  When ``status ==
    # "hardware_verified"`` the validator below requires
    # ``runtime_symbol`` AND ``execute_compare_fixture`` to be set.
    shape_envelope: Optional_str = None
    """Free-form envelope description (e.g., ``"M*N*K <= 2^30"``,
    ``"head_dim <= 256"``, ``"single tile, threadgroup <= 1024"``).
    Documents the validated shape range without imposing parseable
    syntax — the prose form is enough for human review and audit
    citation."""
    runtime_symbol: Optional_str = None
    """C ABI symbol of the kernel that runs at dispatch time (e.g.,
    ``"tessera_apple_gpu_matmul_softmax_matmul_f32"``).  ``None`` for
    artifact-only / planned entries that don't expose a runtime entry
    point."""
    lit_fixture: Optional_str = None
    """Path (relative to repo root) to the lit fixture that exercises
    the kernel at the IR layer.  ``None`` when the entry's coverage
    is only Python-level."""
    execute_compare_fixture: Optional_str = None
    """Path (relative to repo root) to the Python test that runs the
    kernel and compares against a numpy / reference oracle.  This is
    the binary proof of hardware execution.  ``None`` until the
    proof lands.  Required for ``status == "hardware_verified"``."""
    benchmark_json: Optional_str = None
    """Path to a benchmark JSON file in the canonical
    ``benchmarks/...`` tree that records latency / MFU for this
    kernel.  ``None`` until benchmarked.  Recommended for
    ``hardware_verified`` entries but not strictly required."""
    benchmark_metadata: Optional[object] = None
    """P1 (2026-06-10) — structured :class:`BenchmarkMetadata` for the row
    (hot-path group / harness / ratchet status). Erased to ``object`` to keep
    the field annotation simple; validated in ``__post_init__`` to be a
    ``BenchmarkMetadata`` when set. Carries the honest per-row truth that the
    loose ``benchmark_json`` pointer can't (e.g. benchmarked-but-not-ratcheted)."""
    packaged_pipeline_path: Optional_str = None
    """PK5 (2026-05-31) — Apple Metal ``.mtlpackage`` path for entries
    with ``status == "packaged"``. Repo-relative or absolute path to
    the directory the runtime loads via ``[device newLibraryWithURL:]``.
    REQUIRED when ``status == "packaged"`` (validated in
    ``__post_init__``); ignored otherwise. The full lifecycle (load
    → compile → reflect → prepare → dispatch) lives in
    ``tessera.apple_mlpkg`` and uses this path as the input."""
    apple_binding_spec: Optional[object] = None
    """Audit Action 2 (2026-06-01) — compiler-emitted
    ``AppleKernelBindingSpec`` for entries with ``status == "packaged"``.
    Type erased to ``object`` to avoid an import cycle
    (``apple_mlpkg`` imports nothing from this module today, but the
    transitive graph could shift; ``object`` keeps that lane
    bidirectional). Validated structurally in ``__post_init__``: when
    set, must be an instance of ``AppleKernelBindingSpec`` AND its
    ``function_name`` / ``package_path`` must be consistent with the
    enclosing manifest entry. Only meaningful for ``status="packaged"``;
    rejected (with a precise diagnostic) otherwise."""

    def __post_init__(self) -> None:
        from ..dtype import canonicalize_dtype

        if self.status not in _VALID_STATUSES:
            raise ValueError(
                f"BackendKernelEntry.status must be one of {sorted(_VALID_STATUSES)}, "
                f"got {self.status!r}"
            )
        # Normalize dtype aliases to canonical names + dedupe (insertion order).
        seen: dict[str, None] = {}
        for d in self.dtypes:
            canon = canonicalize_dtype(d, allow_planned_gated=True)
            seen[canon] = None
        normalized = tuple(seen)
        if normalized != tuple(self.dtypes):
            object.__setattr__(self, "dtypes", normalized)

        # P1 validation: benchmark_metadata, when set, must be a BenchmarkMetadata.
        if self.benchmark_metadata is not None and not isinstance(
            self.benchmark_metadata, BenchmarkMetadata
        ):
            raise ValueError(
                "BackendKernelEntry.benchmark_metadata must be a "
                f"BenchmarkMetadata, got {type(self.benchmark_metadata).__name__}")

        # Sprint G-3 validation: WGMMA shape only on NVIDIA targets.
        if self.wgmma_shape is not None:
            if not self.target.startswith("nvidia"):
                raise ValueError(
                    f"wgmma_shape only applies to NVIDIA targets, got "
                    f"target={self.target!r}"
                )
            if len(self.wgmma_shape) != 3:
                raise ValueError(
                    f"wgmma_shape must be (M, N, K), got {self.wgmma_shape!r}"
                )
        # cuda_arch_min validation — must be in the known set.
        if self.cuda_arch_min is not None:
            _valid_arches = {
                "sm_70", "sm_75", "sm_80", "sm_86", "sm_89",
                "sm_90", "sm_90a", "sm_100", "sm_100a", "sm_120", "sm_120a",
            }
            if self.cuda_arch_min not in _valid_arches:
                raise ValueError(
                    f"cuda_arch_min must be one of {sorted(_valid_arches)}, "
                    f"got {self.cuda_arch_min!r}"
                )
        # MFMA shape only on AMD targets.
        if self.mfma_shape is not None:
            if not self.target.startswith("rocm"):
                raise ValueError(
                    f"mfma_shape only applies to ROCm targets, got "
                    f"target={self.target!r}"
                )
            if len(self.mfma_shape) != 4:
                raise ValueError(
                    f"mfma_shape must be (M, N, K, K_blocks), got "
                    f"{self.mfma_shape!r}"
                )
        # expected_mfu in [0, 1]
        if self.expected_mfu is not None:
            if not (0.0 <= self.expected_mfu <= 1.0):
                raise ValueError(
                    f"expected_mfu must be in [0, 1], got {self.expected_mfu}"
                )

        # A1 (2026-06-18) — mma_descriptor must be a rocm_mma.MmaDescriptor and
        # only applies to ROCm targets (lazy import avoids an import cycle).
        if self.mma_descriptor is not None:
            from .rocm_mma import MmaDescriptor
            if not isinstance(self.mma_descriptor, MmaDescriptor):
                raise TypeError(
                    f"mma_descriptor must be a rocm_mma.MmaDescriptor, got "
                    f"{type(self.mma_descriptor).__name__}")
            if not self.target.startswith("rocm"):
                raise ValueError(
                    f"mma_descriptor only applies to ROCm targets, got "
                    f"target={self.target!r}")

        # Arch-3 (2026-05-22) — hardware_verified contract.  This is
        # the top rung of the readiness ladder; an entry only qualifies
        # when it carries both an executable runtime entry point AND a
        # checked-in numerical-proof test.  Without the test fixture
        # there's no evidence the kernel actually produces correct
        # output on real hardware — so we refuse to let the status
        # claim more than ``fused`` / ``reference``.
        if self.status == _HARDWARE_VERIFIED_STATUS:
            if not self.runtime_symbol:
                raise ValueError(
                    f"status='hardware_verified' requires runtime_symbol "
                    f"to be set; got target={self.target!r}"
                )
            if not self.execute_compare_fixture:
                raise ValueError(
                    f"status='hardware_verified' requires "
                    f"execute_compare_fixture to point at a Python "
                    f"test that numerically validates the kernel; got "
                    f"target={self.target!r}, runtime_symbol="
                    f"{self.runtime_symbol!r}"
                )

        # PK5 (2026-05-31) — packaged-kernel contract. Status
        # ``packaged`` means "this kernel ships as an `.mtlpackage`
        # the runtime loads via `tessera.apple_mlpkg.compile_mlpackage`."
        # The path is mandatory — without it the manifest entry is a
        # promise without a deliverable, and the dashboard would mark
        # it executable without anything to execute.
        if self.status == _PACKAGED_STATUS:
            if not self.packaged_pipeline_path:
                raise ValueError(
                    f"status='packaged' requires packaged_pipeline_path "
                    f"to name a `.mtlpackage` directory; got "
                    f"target={self.target!r}, "
                    f"packaged_pipeline_path={self.packaged_pipeline_path!r}"
                )
            # The path may be repo-relative or absolute; both work.
            # Filesystem-existence checks live in a separate drift
            # gate (audit follow-up) — we don't validate here so a
            # registry entry can land before the artifact lands.

        # Audit Action 2 (2026-06-01) — AppleKernelBindingSpec
        # consistency check. The spec is the compiler's declarative
        # binding contract; if attached it must agree with the
        # manifest's other packaged-kernel fields (function_name
        # comes from the spec; package_path must match the manifest's
        # packaged_pipeline_path). Late-import keeps the dependency
        # one-way (manifest → apple_mlpkg, not the reverse).
        if self.apple_binding_spec is not None:
            # Import here so a non-Apple manifest module doesn't pay
            # the cost / can't trigger a cycle.
            from ..apple_mlpkg import AppleKernelBindingSpec
            spec = self.apple_binding_spec
            if not isinstance(spec, AppleKernelBindingSpec):
                raise TypeError(
                    f"apple_binding_spec must be an AppleKernelBindingSpec, "
                    f"got {type(spec).__name__}"
                )
            if self.status != _PACKAGED_STATUS:
                raise ValueError(
                    f"apple_binding_spec is only valid when "
                    f"status='packaged', got status={self.status!r}"
                )
            # The spec's package_path must match the manifest's
            # packaged_pipeline_path — they're the same artifact viewed
            # from two angles, and disagreement would silently route
            # the runtime to a different package than the compiler
            # contract declared.
            if (self.packaged_pipeline_path is not None
                    and spec.package_path
                    and spec.package_path != self.packaged_pipeline_path):
                raise ValueError(
                    f"apple_binding_spec.package_path "
                    f"({spec.package_path!r}) does not match "
                    f"packaged_pipeline_path "
                    f"({self.packaged_pipeline_path!r})"
                )

    def as_dict(self) -> dict[str, object]:
        out: dict[str, object] = {
            "target": self.target,
            "status": self.status,
            "dtypes": list(self.dtypes),
            "feature_flags": list(self.feature_flags),
            "notes": self.notes,
        }
        # Only emit non-None Sprint G-3 fields to keep the JSON compact.
        if self.cuda_arch_min is not None:
            out["cuda_arch_min"] = self.cuda_arch_min
        if self.nvcc_version_min is not None:
            out["nvcc_version_min"] = self.nvcc_version_min
        if self.wgmma_shape is not None:
            out["wgmma_shape"] = list(self.wgmma_shape)
        if self.cluster_size is not None:
            out["cluster_size"] = list(self.cluster_size)
        if self.mfma_shape is not None:
            out["mfma_shape"] = list(self.mfma_shape)
        if self.hipcc_version_min is not None:
            out["hipcc_version_min"] = self.hipcc_version_min
        if self.expected_mfu is not None:
            out["expected_mfu"] = self.expected_mfu
        if self.roofline_target is not None:
            out["roofline_target"] = self.roofline_target
        if self.mma_descriptor is not None:
            from .rocm_mma import MmaDescriptor
            if isinstance(self.mma_descriptor, MmaDescriptor):
                out["mma_descriptor"] = self.mma_descriptor.as_metadata_dict()
        # Arch-3 (2026-05-22) — execute-and-compare hooks.
        if self.shape_envelope is not None:
            out["shape_envelope"] = self.shape_envelope
        if self.runtime_symbol is not None:
            out["runtime_symbol"] = self.runtime_symbol
        if self.lit_fixture is not None:
            out["lit_fixture"] = self.lit_fixture
        if self.execute_compare_fixture is not None:
            out["execute_compare_fixture"] = self.execute_compare_fixture
        if self.benchmark_json is not None:
            out["benchmark_json"] = self.benchmark_json
        if isinstance(self.benchmark_metadata, BenchmarkMetadata):
            out["benchmark_metadata"] = self.benchmark_metadata.to_dict()
        # PK5 (2026-05-31) — packaged-kernel path.
        if self.packaged_pipeline_path is not None:
            out["packaged_pipeline_path"] = self.packaged_pipeline_path
        # Audit Action 2 (2026-06-01) — render the compiler-emitted
        # binding spec into JSON-friendly form when present. The
        # dashboard / drift tooling can then diff manifest entries
        # without importing the apple_mlpkg dataclasses.
        if self.apple_binding_spec is not None:
            # Forward to the spec's own ``to_dict`` (already
            # JSON-friendly: lists, None for wildcards, no tuples).
            # The field is typed ``Optional[object]`` to avoid an import
            # cycle, so route through ``Any`` for the duck-typed call.
            spec: Any = self.apple_binding_spec
            out["apple_binding_spec"] = spec.to_dict()
        return out

    @property
    def is_hardware_verified(self) -> bool:
        """Arch-3 (2026-05-22) — convenience predicate for status checks."""
        return self.status == _HARDWARE_VERIFIED_STATUS


def primitive_is_complete(entries: tuple["BackendKernelEntry", ...]) -> bool:
    """Arch-3 (2026-05-22) — compute the registry-level
    ``backend_kernel = "complete"`` axis from a primitive's full target
    row set.

    A primitive's backend_kernel axis flips to ``"complete"`` iff
    EVERY declared target is ``hardware_verified``.  This makes
    ``backend_kernel`` a *computed* property of the manifest rather
    than a hand-flipped status field — closing the definition gap
    that the V8 Phase G/H/I audit doc surfaced.

    Returns False (the registry stays ``partial``) if any target row
    is below ``hardware_verified``.  An empty tuple returns False
    (no declared targets ⇒ nothing to verify ⇒ not complete).
    """
    if not entries:
        return False
    return all(e.status == _HARDWARE_VERIFIED_STATUS for e in entries)


# ─────────────────────────────────────────────────────────────────────────────
# Apple GPU shipped MSL kernels — per `docs/apple_backend.md` (GPU kernel inventory)
# (Phase 8.3 → 8.4.7).  Each entry below corresponds to one ABI symbol
# (or fusion).  Status="fused" when the MSL kernel is a fused chain;
# "reference" when the kernel is a single-op MPS dispatch.
# ─────────────────────────────────────────────────────────────────────────────
_APPLE_GPU_FUSED = ("fp32", "fp16", "bf16")

_APPLE_GPU_KERNELS: dict[str, dict[str, Any]] = {
    "matmul": {
        "status": _FUSED_KERNEL_STATUS,
        "dtypes": _APPLE_GPU_FUSED,
        "notes": "MPSMatrixMultiplication + bf16 conversion path",
        "benchmark_json": "benchmarks/baselines/apple_gpu_hot_paths.json",
    },
    # Project 3 (2026-06-01) — 8 encode-eligible ops promoted to
    # ``hardware_verified``. Each carries:
    #   * runtime_symbol = the per-op encode-session C ABI symbol
    #     (the actual dispatch entry point, not the legacy MPS one)
    #   * shape_envelope = free-form documentation of the validated
    #     shape range
    # ``execute_compare_fixture`` is attached at construction time
    # from ``_NUMERICAL_FIXTURES`` BEFORE the BackendKernelEntry
    # validator runs (see ``manifest_for`` Apple GPU branch).
    "softmax": {
        "status": _HARDWARE_VERIFIED_STATUS,
        "dtypes": _APPLE_GPU_FUSED,
        "notes": "Custom MSL softmax kernel (Phase 8.4.2)",
        "runtime_symbol": "tessera_apple_gpu_softmax_dev_f32_enc",
        "benchmark_json": "benchmarks/baselines/apple_gpu_hot_paths.json",
        "shape_envelope": "rows*cols, no per-row limit (MPSGraph rowop)",
    },
    "softmax_safe": {
        "status": _HARDWARE_VERIFIED_STATUS,
        "dtypes": _APPLE_GPU_FUSED,
        "notes": "Aliases softmax MSL kernel",
        "runtime_symbol": "tessera_apple_gpu_softmax_dev_f32_enc",
        "shape_envelope": "rows*cols, no per-row limit (MPSGraph rowop)",
    },
    "gelu": {
        "status": _HARDWARE_VERIFIED_STATUS,
        "dtypes": _APPLE_GPU_FUSED,
        "notes": (
            "Custom MSL gelu (Phase 8.4.2); the encode-session path "
            "dispatches via the unary opcode router (op_code=5, "
            "tanh approximation)."
        ),
        "runtime_symbol": "tessera_apple_gpu_unary_dev_f32_enc",
        "shape_envelope": "n elements, no limit (elementwise unary)",
    },
    "rope": {
        "status": _HARDWARE_VERIFIED_STATUS,
        "dtypes": _APPLE_GPU_FUSED,
        "notes": "Custom MSL rope (Phase 8.4.0)",
        "runtime_symbol": "tessera_apple_gpu_rope_dev_f32_enc",
        "shape_envelope": "M*K, K must be even (rope dim-pair structure)",
    },
    "rmsnorm": {
        "status": _HARDWARE_VERIFIED_STATUS,
        "dtypes": _APPLE_GPU_FUSED,
        "notes": (
            "MSL rmsnorm + Phase 8.4.7 matmul→rmsnorm fusion. Phase 3b "
            "(2026-06-01) adds f16/bf16 via MPSGraph rowop encode-session"
        ),
        "runtime_symbol": "tessera_apple_gpu_rmsnorm_dev_f32_enc",
        "benchmark_json": "benchmarks/baselines/apple_gpu_hot_paths.json",
        "shape_envelope": "rows*cols, no per-row limit (MPSGraph rowop)",
    },
    "flash_attn": {
        "status": _HARDWARE_VERIFIED_STATUS,
        "dtypes": _APPLE_GPU_FUSED,
        "notes": "Online-softmax MSL kernel; head_dim ≤ 256 (Phase 8.4.1)",
        "runtime_symbol": "tessera_apple_gpu_flash_attn_dev_f32_enc",
        "benchmark_json": "benchmarks/baselines/apple_gpu_hot_paths.json",
        "shape_envelope": "head_dim <= 256 (MSL stack array, Phase 8.4.1)",
    },
    # Phase 2.1c + 3b (2026-06-01) — encode-session ops with full dtype
    # coverage. layer_norm/silu/bmm landed as part of the single-cb
    # decode-chain work; all 8 encode-eligible ops cover {f32, f16, bf16}.
    "layer_norm": {
        "status": _HARDWARE_VERIFIED_STATUS,
        "dtypes": _APPLE_GPU_FUSED,
        "notes": (
            "MPSGraph rowop encode-session (encode_rowop_dev kind=0). "
            "Available in {f32, f16, bf16} via Project-3 + Phase 3b."
        ),
        "runtime_symbol": "tessera_apple_gpu_layer_norm_dev_f32_enc",
        "shape_envelope": "rows*cols, no per-row limit (MPSGraph rowop)",
    },
    "silu": {
        "status": _HARDWARE_VERIFIED_STATUS,
        "dtypes": _APPLE_GPU_FUSED,
        "notes": (
            "MPSGraph unary node op=4 (silu = x * sigmoid(x)). "
            "Encode-session reachable in {f32, f16, bf16}."
        ),
        "runtime_symbol": "tessera_apple_gpu_unary_dev_f32_enc",
        "shape_envelope": "n elements, no limit (elementwise unary)",
    },
    "bmm": {
        "status": _HARDWARE_VERIFIED_STATUS,
        "dtypes": _APPLE_GPU_FUSED,
        "notes": (
            "MPSGraph batched matmul (encode_bmm_dev). Honors batch > 1 "
            "and b_broadcast for K/V reuse across heads in attention."
        ),
        "runtime_symbol": "tessera_apple_gpu_bmm_dev_f32_enc",
        "benchmark_json": "benchmarks/baselines/apple_gpu_hot_paths.json",
        "shape_envelope": "batch*M*N*K (MPSGraph batched matmul)",
    },
    # Audit follow-up (2026-05-31): these three ops are dispatched by the
    # Apple GPU runtime envelope (driver._APPLE_GPU_{MPS,MSL,MPSGRAPH}_OPS
    # and the conv2d native multi-tile spike), but had no manifest entry —
    # surfaced as a gap by the op×target conformance matrix.
    "relu": {
        "status": _FUSED_KERNEL_STATUS,
        "dtypes": _APPLE_GPU_FUSED,
        "notes": "MPSGraph relu node (apple_gpu_runtime.mm MPSGraph lane)",
    },
    "conv2d": {
        "status": _HARDWARE_VERIFIED_STATUS,
        "dtypes": _APPLE_GPU_FUSED,  # Sprint A (2026-06-01): full {f32,f16,bf16}.
        "notes": (
            "Native multi-tile MSL convolution2d via MPP cooperative op "
            "(landed 2026-05-29). Project 5 (2026-06-01) added the "
            "encode-session lane via `tessera_apple_gpu_conv2d_dev_f32_enc`. "
            "Sprint A (2026-06-01) extended the encode lane to the full "
            "3-dtype matrix via `_dev_f16_enc` + `_dev_bf16_enc` siblings."
        ),
        "runtime_symbol": "tessera_apple_gpu_conv2d_dev_f32_enc",
        "benchmark_json": "benchmarks/baselines/apple_gpu_hot_paths.json",
        "shape_envelope": (
            "NHWC source + HWIO weights; bias optional; honors stride/"
            "pad/dilation/groups (depthwise covered); {f32, f16, bf16}"
        ),
    },
    "kv_cache_read": {
        "status": _FUSED_KERNEL_STATUS,
        "dtypes": _APPLE_GPU_FUSED,
        "notes": (
            "KVCacheHandle.read() dispatches to the Apple GPU MPS path "
            "for fp32/fp16/bf16 cache pages"
        ),
    },
    # Audit follow-up (2026-06-10) — MoE expert-FFN ops. Both are runtime
    # envelope ops (apple_gpu_envelope.APPLE_GPU_LANE_BY_OP: lanes grouped_gemm
    # / moe_swiglu_block) with dedicated fused MSL kernels + execute-compare
    # fixtures, but had no manifest row — surfaced as a blind spot by the
    # DEEP_COMPILER_AUDIT_2026_06_10 benchmark/manifest coverage pass.
    "grouped_gemm": {
        "status": _FUSED_KERNEL_STATUS,
        "dtypes": ("fp32",),
        "notes": (
            "Fused ragged grouped-GEMM MSL kernel "
            "(tessera_apple_gpu_grouped_gemm_f32, one dispatch over the whole "
            "(T,N) output, folds the per-token expert id in); f16/bf16 inputs "
            "route through the composed per-group bmm lane (f32 compute)."
        ),
        "runtime_symbol": "tessera_apple_gpu_grouped_gemm_f32",
        "benchmark_json": "benchmarks/apple_gpu/benchmark_megamoe_overlap.py",
        "shape_envelope": "x (T,K) + w (E,K,N) + group_sizes (E,); sum(gs)==T",
    },
    "moe_swiglu_block": {
        "status": _FUSED_KERNEL_STATUS,
        "dtypes": ("fp32",),
        "notes": (
            "Fused ragged SwiGLU MoE expert-FFN MSL kernel "
            "(tessera_apple_gpu_moe_swiglu_f32 — gate/up/silu_mul/down in one "
            "dispatch, the grouped analog of swiglu_f32); H,Kout<=256 fast path, "
            "else the composed grouped-GEMM + silu_mul lanes. The local MegaMoE "
            "expert-FFN core (see docs/distributed_megamoe.md)."
        ),
        "runtime_symbol": "tessera_apple_gpu_moe_swiglu_f32",
        "benchmark_json": "benchmarks/apple_gpu/benchmark_megamoe_overlap.py",
        "shape_envelope": (
            "x (T,K) + Wg/Wu (E,K,H) + Wd (E,H,Kout) + group_sizes (E,); "
            "fused kernel H,Kout<=256"
        ),
    },
    # Frontier MoE model-class track (2026-06-13) — fused dequantize-into-GEMM.
    # The genuine fused kernel behind stdlib.quant.dequant_matmul(backend=
    # "apple_gpu"): packed-INT4 codes + group scales dequantized in-register,
    # f32 accumulate, one dispatch. README references the runtime symbol; the
    # claim-lint requires the matching manifest row.
    "dequant_matmul": {
        "status": _FUSED_KERNEL_STATUS,
        "dtypes": ("fp32",),
        "notes": (
            "Fused dequantize-into-GEMM MSL kernel "
            "(tessera_apple_gpu_dequant_matmul_f32 — packed-INT4 weight codes + "
            "group scales dequantized in-register, fp32 accumulate, one "
            "dispatch). The quant pillar of the frontier MoE model-class track."
        ),
        "runtime_symbol": "tessera_apple_gpu_dequant_matmul_f32",
        "shape_envelope": "x (M,K) + codes/scales packed (K%GS==0); O (M,N) fp32",
    },
}


# ─────────────────────────────────────────────────────────────────────────────
# Audit follow-up A.3 (2026-05-31) — Per-(op, target) test fixtures that
# exercise the numerical-correctness path end-to-end. Replaces the
# conformance matrix's filename/content heuristic with first-class
# manifest data.
#
# Each value is a repo-relative path to a Python test file that ships an
# assertion comparing the kernel's output to numpy / a reference at a
# tolerance the test declares. The conformance matrix consults this map
# via ``manifest_for(op)`` → ``BackendKernelEntry.execute_compare_fixture``.
#
# Strict rules for additions:
#   - The file MUST exist in tests/unit/ at the moment of addition (the
#     ``_validate_numerical_fixtures`` walker would otherwise complain).
#   - The file MUST contain a numpy-comparison-style assertion for THIS
#     op on THIS target — not just "imports both names".
#   - Prefer a small, focused fixture over a 1000-line model test, so a
#     downstream consumer reading the dashboard finds the proof quickly.
# ─────────────────────────────────────────────────────────────────────────────

_NUMERICAL_FIXTURES: dict[tuple[str, str], str] = {
    # cpu
    ("matmul", "cpu"): "tests/unit/test_end_to_end_matmul_cpu_path.py",
    # conv2d on the CPU reference path: @jit conv2d_nhwc executes and is
    # assert_allclose'd against a hand-computed expected output (audit
    # 2026-06-10 — promotes conv2d/cpu off the keyword heuristic).
    ("conv2d", "cpu"): "tests/unit/test_operator_registry_foundation.py",
    # KV-cache read on the reference handle: append → read(start, end) and
    # assert_allclose the returned keys/values against the appended slice.
    ("kv_cache_read", "cpu"): "tests/unit/test_kv_cache_handle.py",
    # apple_cpu
    ("matmul", "apple_cpu"): "tests/unit/test_apple_backend_roadmap.py",
    # apple_gpu — landed runtime paths
    # GPU matmul vs both the CPU lane and numpy (gpu == a @ b). The
    # buffer-pool test executes matmul but asserts RAII invariants, not
    # numerical equality — so it is NOT a valid execute_compare_fixture.
    ("matmul", "apple_gpu"):
        "tests/unit/test_production_jit_phase3_apple_gpu.py",
    ("softmax", "apple_gpu"): "tests/unit/test_apple_gpu_mpsgraph_lane.py",
    ("softmax_safe", "apple_gpu"): "tests/unit/test_apple_gpu_mpsgraph_lane.py",
    ("flash_attn", "apple_gpu"): "tests/unit/test_apple_gpu_fused_attention.py",
    # Fused matmul→softmax single MSL kernel: the fixture runs
    # ``agb.gpu_matmul_softmax(a, b)`` and compares it to both the un-fused
    # CPU-lane composition and the numpy reference ``softmax(a @ b)``.
    ("matmul_softmax", "apple_gpu"):
        "tests/unit/test_production_jit_phase3_apple_gpu.py",
    ("conv2d", "apple_gpu"): "tests/unit/test_apple_gpu_conv2d.py",
    # Mamba-2 selective scan: chunked-parallel SSD with Metal bmm contractions,
    # validated bit-exact against the sequential numpy reference.
    ("selective_ssm", "apple_gpu"): "tests/unit/test_mamba_ssd_gpu.py",
    # Ragged grouped matmul (MoE expert FFN): per-group MPS matmul.
    ("grouped_gemm", "apple_gpu"): "tests/unit/test_grouped_gemm_gpu.py",
    # Fused dequantize-into-GEMM: dequant_matmul(backend="apple_gpu") output
    # compared to the full-precision x @ dequant(W) oracle.
    ("dequant_matmul", "apple_gpu"): "tests/unit/test_stdlib_quant.py",
    ("rmsnorm", "apple_gpu"): "tests/unit/test_apple_gpu_mpsgraph_lane.py",
    ("gelu", "apple_gpu"): "tests/unit/test_apple_gpu_mpsgraph_lane.py",
    # rope(q)/rope(k) vs a numpy rotary reference (execute-compare), not the
    # buffer-pool RAII test.
    ("rope", "apple_gpu"): "tests/unit/test_apple_gpu_ops_interception.py",
    ("relu", "apple_gpu"): "tests/unit/test_apple_gpu_mpsgraph_lane.py",
    # Phase 2.1c + 3b (2026-06-01) — encode-session ops, multi-dtype.
    # Each fixture compares the encode-session output to a numpy
    # reference at the dtype-appropriate tolerance.
    ("layer_norm", "apple_gpu"):
        "tests/unit/test_apple_gpu_f16_encode_session.py",
    ("silu", "apple_gpu"):
        "tests/unit/test_apple_gpu_full_decoder_layer.py",
    ("bmm", "apple_gpu"):
        "tests/unit/test_apple_gpu_f16_encode_session.py",
    # Fused ragged SwiGLU MoE expert-FFN block: the fused MSL kernel +
    # composed-lane fast paths vs a numpy f64 reference (incl. E=1 reduces to
    # dense swiglu, large-H fallback).
    ("moe_swiglu_block", "apple_gpu"): "tests/unit/test_moe_swiglu_block.py",
    # ── Audit 2026-06-10 — record numerical proof for Apple GPU `fused`
    #    rows that had a genuine dedicated execute-compare test but no wired
    #    fixture (the "numerical-proof discipline" gap). Each fixture below was
    #    confirmed (Decision #27) to run the op's GPU kernel and assert_allclose
    #    it against a numpy / GA / reference. Geometric-algebra (Cl(3,0)) family:
    ("clifford_reverse", "apple_gpu"): "tests/unit/test_apple_gpu_clifford_msl_full.py",
    ("clifford_grade_involution", "apple_gpu"): "tests/unit/test_apple_gpu_clifford_msl_full.py",
    ("clifford_conjugate", "apple_gpu"): "tests/unit/test_apple_gpu_clifford_msl_full.py",
    ("clifford_hodge_star", "apple_gpu"): "tests/unit/test_apple_gpu_clifford_msl_full.py",
    ("clifford_norm", "apple_gpu"): "tests/unit/test_apple_gpu_clifford_msl_full.py",
    ("clifford_wedge", "apple_gpu"): "tests/unit/test_apple_gpu_clifford_msl_full.py",
    ("clifford_left_contraction", "apple_gpu"): "tests/unit/test_apple_gpu_clifford_msl_full.py",
    ("clifford_inner", "apple_gpu"): "tests/unit/test_apple_gpu_clifford_msl_full.py",
    ("clifford_grade_projection", "apple_gpu"): "tests/unit/test_apple_gpu_clifford_msl_full.py",
    ("clifford_geometric_product", "apple_gpu"): "tests/unit/test_apple_gpu_clifford_lane.py",
    ("clifford_rotor_sandwich", "apple_gpu"): "tests/unit/test_apple_gpu_clifford_msl.py",
    # GA differential-operator family (GA(1,1) field ops):
    ("clifford_codiff", "apple_gpu"): "tests/unit/test_apple_gpu_clifford_msl_ga11.py",
    ("clifford_exp", "apple_gpu"): "tests/unit/test_apple_gpu_clifford_msl_ga11.py",
    ("clifford_log", "apple_gpu"): "tests/unit/test_apple_gpu_clifford_msl_ga11.py",
    ("clifford_ext_deriv", "apple_gpu"): "tests/unit/test_apple_gpu_clifford_msl_ga11.py",
    ("clifford_vec_deriv", "apple_gpu"): "tests/unit/test_apple_gpu_clifford_msl_ga11.py",
    ("clifford_integral", "apple_gpu"): "tests/unit/test_apple_gpu_clifford_msl_ga11.py",
    # Complex-number kernels:
    ("complex_mul", "apple_gpu"): "tests/unit/test_complex_runtime.py",
    ("complex_exp", "apple_gpu"): "tests/unit/test_complex_runtime.py",
    # EBM refinement / inner-step (Langevin descent), vs the numpy reference:
    ("ebm_refinement", "apple_gpu"): "tests/unit/test_apple_gpu_ebm_lane.py",
    ("ebm_inner_step", "apple_gpu"): "tests/unit/test_apple_gpu_ebm_lane.py",
}


# ─────────────────────────────────────────────────────────────────────────────
# P1 (2026-06-10) — uniform hot-path benchmark metadata for Apple GPU rows.
#
# Attached to the BackendKernelEntry at construction (parallel to
# _NUMERICAL_FIXTURES). ``ratcheted`` is the HONEST per-row truth: only ops with
# a recorded standalone row in benchmarks/baselines/apple_gpu_hot_paths.json are
# True (matmul, conv2d). softmax/rmsnorm/flash_attn/bmm/grouped_gemm/
# moe_swiglu_block are benchmarked by their harness but have no standalone
# baseline row yet — recording those is an environment-aware follow-on (needs a
# live Metal host). The fused-epilogue chains (matmul_softmax/gelu/rmsnorm/
# softmax_matmul, swiglu) ARE ratcheted but are benchmark-only aliases with no
# manifest row (see benchmark_coverage.FUSED_CHAIN_BENCH_ALIASES).
# ─────────────────────────────────────────────────────────────────────────────

_HOT_PATHS_BASELINE = "benchmarks/baselines/apple_gpu_hot_paths.json"

_APPLE_GPU_HOT_PATH_METADATA: dict[str, BenchmarkMetadata] = {
    "matmul": BenchmarkMetadata(
        hot_path_group="gemm", harness="benchmarks/benchmark_gemm.py",
        ratcheted=True, ratchet_key="matmul"),
    "conv2d": BenchmarkMetadata(
        hot_path_group="conv",
        harness="benchmarks/apple_gpu/record_hot_path_baseline.py",
        ratcheted=True, ratchet_key="conv2d"),
    "softmax": BenchmarkMetadata(
        hot_path_group="norm", harness="benchmarks/apple_gpu/benchmark_fusion.py",
        ratcheted=False, notes="fused matmul_softmax epilogue is ratcheted; standalone softmax pending baseline row"),
    "rmsnorm": BenchmarkMetadata(
        hot_path_group="norm", harness="benchmarks/apple_gpu/benchmark_fusion.py",
        ratcheted=False, notes="fused matmul_rmsnorm epilogue is ratcheted; standalone rmsnorm pending baseline row"),
    "flash_attn": BenchmarkMetadata(
        hot_path_group="attention", harness="benchmarks/benchmark_attention.py",
        ratcheted=False, notes="benchmarked; standalone baseline row pending"),
    "bmm": BenchmarkMetadata(
        hot_path_group="gemm", harness="benchmarks/benchmark_gemm.py",
        ratcheted=False, notes="benchmarked; standalone baseline row pending"),
    "grouped_gemm": BenchmarkMetadata(
        hot_path_group="moe",
        harness="benchmarks/apple_gpu/benchmark_grouped_gemm.py",
        ratcheted=True, ratchet_key="grouped_gemm",
        notes="MoE expert-FFN core; isolated ratchet row (DeepGEMM keystone) — "
              "timed standalone, not via the MegaMoE overlap harness"),
    "moe_swiglu_block": BenchmarkMetadata(
        hot_path_group="moe",
        harness="benchmarks/apple_gpu/benchmark_grouped_gemm.py",
        ratcheted=True, ratchet_key="moe_swiglu_block",
        notes="MoE expert-FFN block; isolated ratchet row (DeepGEMM keystone) — "
              "timed standalone, not via the MegaMoE overlap harness"),
}


def _attach_numerical_fixtures(
    op_name: str, entries: "list[BackendKernelEntry]",
) -> "list[BackendKernelEntry]":
    """Post-process a manifest's entries to attach the audit-recorded
    ``execute_compare_fixture`` for any (op, target) pair the
    ``_NUMERICAL_FIXTURES`` map knows about. The field is honored
    additively — entries that already declare a fixture (via a
    constructor argument upstream) are left untouched."""
    out: "list[BackendKernelEntry]" = []
    for e in entries:
        if e.execute_compare_fixture is not None:
            out.append(e)
            continue
        fixture = _NUMERICAL_FIXTURES.get((op_name, e.target))
        if fixture is None:
            out.append(e)
            continue
        # BackendKernelEntry is a frozen dataclass; rebuild with the
        # fixture field set. We use ``replace`` from dataclasses to be
        # explicit about field-by-field copy.
        from dataclasses import replace as _replace
        out.append(_replace(e, execute_compare_fixture=fixture))
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Sprint G-3 (2026-05-11) — Per-kernel WGMMA tile shape + cluster + MFU
# targets for NVIDIA SM_90+ (CUDA 13.2 U1).
#
# The canonical bf16/fp16 Hopper tile is (M=64, N=256, K=16) — this is
# what cuBLAS uses for its WGMMA-based GEMM kernels. FP8 lowers to
# K=32; FP4/NVFP4 to K=64.  Tessera's WGMMA lowering pass uses these
# shapes to drive `tile_q`/`tile_kv` selection in attention kernels.
#
# Per-kernel MFU targets come from published benchmarks where available
# (FA-4 on H100 hits ~75% of FP16 peak; cuBLAS GEMM hits ~80% MFU on
# large M/N) and conservative estimates otherwise.  These are tracked
# by `perf_gate.py` once execution lights up under Phase G.
# ─────────────────────────────────────────────────────────────────────────────

_NVIDIA_KERNEL_TILE_SHAPES: dict[str, dict[str, tuple]] = {
    # ── Matmul / contraction family ──────────────────────────────────────
    "matmul":           {"wgmma_shape": (64, 256, 16), "cluster": (1, 1, 1)},
    "gemm":             {"wgmma_shape": (64, 256, 16), "cluster": (1, 1, 1)},
    "batched_gemm":     {"wgmma_shape": (64, 256, 16), "cluster": (1, 1, 1)},
    "einsum":           {"wgmma_shape": (64, 256, 16), "cluster": (1, 1, 1)},
    "linear_general":   {"wgmma_shape": (64, 256, 16), "cluster": (1, 1, 1)},
    "qkv_projection":   {"wgmma_shape": (64, 256, 16), "cluster": (1, 1, 1)},
    "fused_epilogue":   {"wgmma_shape": (64, 256, 16), "cluster": (1, 1, 1)},
    "factorized_matmul":{"wgmma_shape": (64, 128, 16), "cluster": (1, 1, 1)},
    # ── Attention family ─────────────────────────────────────────────────
    # FA-4 uses two WGMMA passes per outer-step: tile_q=128, tile_kv=128,
    # head_dim=128 maps to wgmma (M=128, N=128, K=16) on bf16 inputs with
    # an fp32 accumulator.  Cluster (2, 1, 1) for producer-consumer
    # warp specialization across paired CTAs.
    "flash_attn":               {"wgmma_shape": (64, 128, 16), "cluster": (2, 1, 1)},
    "multi_head_attention":     {"wgmma_shape": (64, 128, 16), "cluster": (2, 1, 1)},
    "gqa_attention":            {"wgmma_shape": (64, 128, 16), "cluster": (2, 1, 1)},
    "mqa_attention":            {"wgmma_shape": (64, 128, 16), "cluster": (2, 1, 1)},
    "mla_decode":               {"wgmma_shape": (64, 128, 16), "cluster": (2, 1, 1)},
    "mla_decode_fused":         {"wgmma_shape": (64, 128, 16), "cluster": (2, 1, 1)},
    # DeepSeek NSA — top-k sparse selection then WGMMA over the chosen
    # blocks.  Reuses the FA tile; cluster=1 since each block runs
    # independently.
    "deepseek_sparse_attention":{"wgmma_shape": (64, 128, 16), "cluster": (1, 1, 1)},
    "attn_top_k_blocks":        {"wgmma_shape": (64, 128, 16), "cluster": (1, 1, 1)},
    "attn_compressed_blocks":   {"wgmma_shape": (64, 128, 16), "cluster": (1, 1, 1)},
    "attn_sliding_window":      {"wgmma_shape": (64, 128, 16), "cluster": (1, 1, 1)},
    # 2D local-window attention — H×W spatial neighborhoods for weather/grid
    # workloads.  Same WGMMA tile as sliding-window 1D; cluster=1 since each
    # (h, w) center attends to a small fixed patch.
    "attn_local_window_2d":     {"wgmma_shape": (64, 128, 16), "cluster": (1, 1, 1)},
    # MiniMax Lightning — linear-attention with delta-rule recurrence.
    # The recurrence runs as a sequence of small (32, 32, 16) WGMMAs.
    "lightning_attention":      {"wgmma_shape": (32, 32, 16),  "cluster": (1, 1, 1)},
    "linear_attn":              {"wgmma_shape": (32, 32, 16),  "cluster": (1, 1, 1)},
    "gated_deltanet":           {"wgmma_shape": (32, 32, 16),  "cluster": (1, 1, 1)},
    "kimi_delta_attention":     {"wgmma_shape": (32, 32, 16),  "cluster": (1, 1, 1)},
    "modified_delta_attention": {"wgmma_shape": (32, 32, 16),  "cluster": (1, 1, 1)},
    "gated_attention":          {"wgmma_shape": (64, 128, 16), "cluster": (1, 1, 1)},
    "hybrid_attention":         {"wgmma_shape": (64, 128, 16), "cluster": (2, 1, 1)},
    # ── Normalization (fused, no WGMMA — single-tile reductions) ─────────
    # rmsnorm / layer_norm / softmax don't use WGMMA in the canonical
    # fused kernel; they use cooperative warp-shuffle reductions.
    # Recorded shape None.
}


# Per-(op, target) expected MFU targets.  Conservative for SM_120 (Blackwell
# consumer, RTX 50-series): it lacks the datacenter sm_100 tcgen05/TMEM path and
# uses warp-level mma.sync.block_scale, so its achievable MFU differs from sm_100
# — pending on-silicon measurement on the RTX 5070 Ti.
_NVIDIA_KERNEL_MFU: dict[tuple[str, str], float] = {
    # cuBLAS WGMMA GEMM hits ~80% MFU on large M/N.
    ("matmul",     "nvidia_sm90"):  0.80,
    ("matmul",     "nvidia_sm100"): 0.82,
    ("matmul",     "nvidia_sm120"): 0.80,
    ("gemm",       "nvidia_sm90"):  0.80,
    ("gemm",       "nvidia_sm100"): 0.82,
    ("batched_gemm","nvidia_sm90"): 0.78,
    ("batched_gemm","nvidia_sm100"):0.80,
    # FA-4 on H100 hits ~75% of FP16 peak; B100 expected slightly higher.
    ("flash_attn", "nvidia_sm90"):  0.75,
    ("flash_attn", "nvidia_sm100"): 0.78,
    ("flash_attn", "nvidia_sm120"): 0.75,
    # MLA decode — KV-bound, lower MFU; perf target is decode tokens/sec.
    ("mla_decode", "nvidia_sm90"):  0.55,
    ("mla_decode", "nvidia_sm100"): 0.60,
    # Lightning + delta variants — linear-attention, recurrence-bound.
    ("lightning_attention", "nvidia_sm90"):  0.40,
    ("kimi_delta_attention", "nvidia_sm90"): 0.40,
    # Sparse attention is gather-bound until WGMMA kicks in.
    ("deepseek_sparse_attention", "nvidia_sm90"): 0.50,
}


_NVIDIA_KERNEL_ROOFLINE: dict[str, str] = {
    "matmul":    "compute-bound at M*N >= 8192*8192 on SM_90; memory-bound at K <= 256",
    "gemm":      "compute-bound at M*N >= 8192*8192 on SM_90; memory-bound at K <= 256",
    "flash_attn":"compute-bound at seq_len >= 1024 + head_dim >= 64; memory-bound otherwise",
    "mla_decode":"KV-cache-memory-bound (compressed latent KV reduces bandwidth ~4x vs MHA)",
    "lightning_attention": "recurrence-serial; memory-bound on the state update step",
    "deepseek_sparse_attention": "gather-bound at top-k <= 32; compute-bound at top-k >= 64",
    "matmul_softmax":     "fused — saves the score-matrix DRAM round-trip; compute-bound after fusion",
    "matmul_softmax_matmul": "fused 3-op chain — saves both score + softmax round-trips",
}


# ─────────────────────────────────────────────────────────────────────────────
# Sprint H-3 (2026-05-11) — Per-kernel MFMA shape + LDS layout + MFU for
# ROCm 7.2.3.  Mirrors the NVIDIA tables above.
#
# Canonical MFMA shapes for bf16 on CDNA 3 (gfx94x): (32, 32, 8, 1) and
# (16, 16, 16, 1).  FP8 variants are (32, 32, 16, 1) / (16, 16, 32, 1).
# CDNA 4 (gfx950) adds FP4 lanes at (32, 32, 32, 1) / (16, 16, 64, 1).
# ─────────────────────────────────────────────────────────────────────────────

_ROCM_KERNEL_MFMA_SHAPES: dict[str, tuple[int, int, int, int]] = {
    # Matmul family — canonical CDNA bf16 shape
    "matmul":           (32, 32, 8, 1),
    "gemm":             (32, 32, 8, 1),
    "batched_gemm":     (32, 32, 8, 1),
    "einsum":           (32, 32, 8, 1),
    "linear_general":   (32, 32, 8, 1),
    "qkv_projection":   (32, 32, 8, 1),
    "fused_epilogue":   (32, 32, 8, 1),
    "factorized_matmul":(16, 16, 16, 1),
    # Attention family — smaller MFMA tile (16x16) for the score matrix
    # because the matrix is typically narrow in N (head_dim ≤ 128).
    "flash_attn":               (16, 16, 16, 1),
    "multi_head_attention":     (16, 16, 16, 1),
    "gqa_attention":            (16, 16, 16, 1),
    "mqa_attention":            (16, 16, 16, 1),
    "mla_decode":               (16, 16, 16, 1),
    "mla_decode_fused":         (16, 16, 16, 1),
    "deepseek_sparse_attention":(16, 16, 16, 1),
    "attn_top_k_blocks":        (16, 16, 16, 1),
    "attn_compressed_blocks":   (16, 16, 16, 1),
    "attn_sliding_window":      (16, 16, 16, 1),
    "attn_local_window_2d":     (16, 16, 16, 1),
    "lightning_attention":      (16, 16, 16, 1),
    "linear_attn":              (16, 16, 16, 1),
    "gated_deltanet":           (16, 16, 16, 1),
    "kimi_delta_attention":     (16, 16, 16, 1),
    "modified_delta_attention": (16, 16, 16, 1),
    "gated_attention":          (16, 16, 16, 1),
    "hybrid_attention":         (16, 16, 16, 1),
}


_ROCM_KERNEL_MFU: dict[tuple[str, str], float] = {
    # rocBLAS MFMA GEMM hits ~75% MFU on MI300X.
    ("matmul", "rocm_gfx942"): 0.75,
    ("matmul", "rocm_gfx950"): 0.78,
    ("gemm",   "rocm_gfx942"): 0.75,
    ("gemm",   "rocm_gfx950"): 0.78,
    ("batched_gemm", "rocm_gfx942"): 0.72,
    # FA on MI300X via rocm-FA2 hits ~65% of FP16 peak.
    ("flash_attn", "rocm_gfx942"): 0.65,
    ("flash_attn", "rocm_gfx950"): 0.70,
    ("mla_decode", "rocm_gfx942"): 0.50,
    ("lightning_attention", "rocm_gfx942"): 0.35,
}


# ─────────────────────────────────────────────────────────────────────────────
# x86 backend — AMX BF16 GEMM is the only currently-real execution path
# (Phase 2).
# ─────────────────────────────────────────────────────────────────────────────
_X86_KERNELS: dict[str, dict[str, Any]] = {
    "matmul": {
        "status": _FUSED_KERNEL_STATUS,
        "dtypes": ("bf16",),
        "notes": "AMX BF16 GEMM (Phase 2; the only fully-wired exec path)",
    },
    "gemm": {
        "status": _FUSED_KERNEL_STATUS,
        "dtypes": ("bf16",),
        "notes": "AMX BF16 GEMM",
    },
}


# ─────────────────────────────────────────────────────────────────────────────
# Apple CPU — Accelerate cblas_sgemm + BNNS fp16/bf16
# ─────────────────────────────────────────────────────────────────────────────
_APPLE_CPU_KERNELS: dict[str, dict[str, Any]] = {
    "matmul": {
        "status": _FUSED_KERNEL_STATUS,
        "dtypes": ("fp32", "fp16", "bf16"),
        "notes": "Accelerate cblas_sgemm (f32) + BNNS f16/bf16 (Phase 8.2)",
    },
    "gemm": {
        "status": _FUSED_KERNEL_STATUS,
        "dtypes": ("fp32", "fp16", "bf16"),
        "notes": "Accelerate cblas_sgemm + BNNS",
    },
}


# ─────────────────────────────────────────────────────────────────────────────
# GA9 (2026-05-17) — Clifford / Geometric-Algebra kernel manifest.
#
# Parallel to the tensor-op manifest above: the 17 `clifford_*` primitives
# in `primitive_coverage.py` are not part of the tensor `OP_SPECS` catalog,
# so they need a dedicated dispatch table.  `manifest_for()` checks the
# `clifford_*` prefix and routes to `clifford_manifest_for()` below.
#
# Per Q4 of `ga_scope_lock.md`, the backend priority is:
#   x86       → reference status (Python GA reference runs on x86 CPU).
#   apple_cpu → reference status (same Python path; Accelerate hand-off
#               for matmul-flavored ops is a GA9-followup).
#   apple_gpu → planned status for v1 (custom MSL kernels for fused
#               geo_product / rotor_sandwich are post-GA9 work).
#   NVIDIA / ROCm → planned (post-Phase G/H).
#
# Two headline ops (geo_product, rotor_sandwich) get expanded fp16/bf16
# coverage on Apple GPU because they're the load-bearing primitives the
# MSL kernels will fuse first; the rest carry fp32-only as the v1 baseline.
# ─────────────────────────────────────────────────────────────────────────────

# Canonical CPU dtype coverage for all 17 GA primitives.
_CLIFFORD_CPU_DTYPES = ("fp32", "fp64")

# Apple GPU dtype coverage for the two headline ops (geo_product, rotor_sandwich)
# that the post-GA9 MSL fused kernels will target first.
_CLIFFORD_APPLE_GPU_HEADLINE_DTYPES = ("fp32", "fp16", "bf16")

# Apple GPU dtype coverage for the v1 baseline (fp32 only — MSL ports of
# every GA op are GA10/GA11 follow-on conformance work).
_CLIFFORD_APPLE_GPU_BASELINE_DTYPES = ("fp32",)

# NVIDIA / ROCm dtype targets for the planned-only entries.
_CLIFFORD_PLANNED_GPU_DTYPES = ("fp32", "fp16", "bf16")


# GA10 conformance follow-on (2026-05-17): the two headline ops also
# carry fp16 + bf16 MSL ports.  All other shipped-MSL ops are f32-only;
# the dtype set below mirrors what `apple_gpu_runtime.mm` actually
# exports.
_CLIFFORD_HEADLINE_OPS = frozenset({
    "clifford_geometric_product",
    "clifford_rotor_sandwich",
})

# Ops that ship fused MSL kernels on Apple GPU (2026-05-17 follow-on).
# Each maps to the exported runtime C ABI symbol name + dtype set.
_CLIFFORD_APPLE_GPU_FUSED = {
    "clifford_geometric_product": {
        "symbol_prefix": "tessera_apple_gpu_clifford_geo_product_cl30_",
        "dtypes": ("fp32", "fp16", "bf16"),
    },
    "clifford_rotor_sandwich": {
        "symbol_prefix": "tessera_apple_gpu_clifford_rotor_sandwich_cl30_",
        "dtypes": ("fp32", "fp16", "bf16"),
    },
    # Fused rotor-invariant ‖R x R†‖ (gap #6) — one dispatch for the
    # rotor_sandwich→norm chain. Not a GA primitive (excluded from
    # _CLIFFORD_PRIMITIVES); it is a fusion op the clifford_jit IR pass emits.
    "clifford_rotor_sandwich_norm": {
        "symbol_prefix": "tessera_apple_gpu_clifford_rotor_sandwich_norm_cl30_",
        "dtypes": ("fp32",),
    },
    "clifford_reverse": {
        "symbol_prefix": "tessera_apple_gpu_clifford_reverse_cl30_",
        "dtypes": ("fp32",),
    },
    "clifford_grade_involution": {
        "symbol_prefix": "tessera_apple_gpu_clifford_grade_involution_cl30_",
        "dtypes": ("fp32",),
    },
    "clifford_conjugate": {
        "symbol_prefix": "tessera_apple_gpu_clifford_conjugate_cl30_",
        "dtypes": ("fp32",),
    },
    "clifford_hodge_star": {
        "symbol_prefix": "tessera_apple_gpu_clifford_hodge_star_cl30_",
        "dtypes": ("fp32",),
    },
    "clifford_norm": {
        "symbol_prefix": "tessera_apple_gpu_clifford_norm_cl30_",
        "dtypes": ("fp32",),
    },
    # ‖x‖² (norm without the sqrt) — derived reduction op with its own fused
    # MSL kernel (apple_gpu_runtime.mm:tessera_apple_gpu_clifford_norm_squared_cl30_f32).
    # In _CLIFFORD_FUSION_OPS, not _CLIFFORD_PRIMITIVES (keeps the 17-primitive count).
    "clifford_norm_squared": {
        "symbol_prefix": "tessera_apple_gpu_clifford_norm_squared_cl30_",
        "dtypes": ("fp32",),
    },
    "clifford_wedge": {
        "symbol_prefix": "tessera_apple_gpu_clifford_wedge_cl30_",
        "dtypes": ("fp32",),
    },
    "clifford_left_contraction": {
        "symbol_prefix": "tessera_apple_gpu_clifford_left_contraction_cl30_",
        "dtypes": ("fp32",),
    },
    "clifford_inner": {
        "symbol_prefix": "tessera_apple_gpu_clifford_inner_cl30_",
        "dtypes": ("fp32",),
    },
    "clifford_grade_projection": {
        "symbol_prefix": "tessera_apple_gpu_clifford_grade_projection_cl30_",
        "dtypes": ("fp32",),
    },
    # GA11 trig-MSL closed-form ops (2026-05-17).
    "clifford_exp": {
        "symbol_prefix": "tessera_apple_gpu_clifford_exp_cl30_",
        "dtypes": ("fp32",),
    },
    "clifford_log": {
        "symbol_prefix": "tessera_apple_gpu_clifford_log_cl30_",
        "dtypes": ("fp32",),
    },
    # GA11 field-signature ops (2026-05-17) — different ABI
    # (F, Out, D0, D1, D2, h0, h1, h2) for ext_deriv / vec_deriv / codiff;
    # (field, weights, out, n) for integral.
    "clifford_ext_deriv": {
        "symbol_prefix": "tessera_apple_gpu_clifford_ext_deriv_cl30_",
        "dtypes": ("fp32",),
    },
    "clifford_vec_deriv": {
        "symbol_prefix": "tessera_apple_gpu_clifford_vec_deriv_cl30_",
        "dtypes": ("fp32",),
    },
    "clifford_codiff": {
        "symbol_prefix": "tessera_apple_gpu_clifford_codiff_cl30_",
        "dtypes": ("fp32",),
    },
    "clifford_integral": {
        "symbol_prefix": "tessera_apple_gpu_clifford_integral_cl30_",
        "dtypes": ("fp32",),
    },
}

# All 17 GA primitives (12 GA3 core + 5 GA5 differential-form) registered
# in `primitive_coverage.py` under `category="geometric_algebra"`.
_CLIFFORD_PRIMITIVES = (
    # GA3 core ops
    "clifford_geometric_product",
    "clifford_grade_projection",
    "clifford_wedge",
    "clifford_left_contraction",
    "clifford_inner",
    "clifford_reverse",
    "clifford_grade_involution",
    "clifford_conjugate",
    "clifford_norm",
    "clifford_exp",
    "clifford_log",
    "clifford_rotor_sandwich",
    # GA5 differential-form ops
    "clifford_hodge_star",
    "clifford_ext_deriv",
    "clifford_codiff",
    "clifford_vec_deriv",
    "clifford_integral",
)

# Fused GA chains (gap #6) — NOT primitives, but they ship a fused Apple-GPU
# kernel + a CPU reference (the unfused composition), so `clifford_manifest_for`
# reports them. Kept separate from `_CLIFFORD_PRIMITIVES` so the "17 primitives"
# audits/counts stay exact.
_CLIFFORD_FUSION_OPS = frozenset({"clifford_rotor_sandwich_norm",
                                  "clifford_norm_squared"})


def clifford_manifest_for(op_name: str) -> list[BackendKernelEntry]:
    """Return the backend manifest entries for a `clifford_*` primitive.

    All 17 GA primitives ship with `x86` + `apple_cpu` reference status
    (the Python GA implementation in `tessera.ga.ops` is the v1
    execution path on these targets — verified end-to-end by the GA10
    Python conformance suite).  Apple GPU coverage is planned for v1;
    the two headline ops (geo_product, rotor_sandwich) declare
    fp32/fp16/bf16 slots so the post-GA9 MSL kernel work has a
    pre-locked dtype target.  NVIDIA / ROCm entries are planned,
    gated on Phase G/H respectively.
    """
    if op_name not in _CLIFFORD_PRIMITIVES and op_name not in _CLIFFORD_FUSION_OPS:
        return []
    entries: list[BackendKernelEntry] = []

    # x86 — reference status (the Python GA ops run on every CPU target).
    entries.append(BackendKernelEntry(
        target="x86",
        status=_REFERENCE_STATUS,
        dtypes=_CLIFFORD_CPU_DTYPES,
        feature_flags=("clifford_dialect", "numpy_reference"),
        notes="Python GA reference path; GA8 unrolled arith.mulf via ExpandProductTable lit-tested",
    ))

    # Apple CPU — reference status, same Python path as x86. The
    # Accelerate hand-off for matmul-flavored GA ops (geo_product
    # batched contractions) is a GA9-followup performance optimization.
    entries.append(BackendKernelEntry(
        target="apple_cpu",
        status=_REFERENCE_STATUS,
        dtypes=_CLIFFORD_CPU_DTYPES,
        feature_flags=("clifford_dialect", "numpy_reference"),
        notes="Python GA reference; Accelerate hand-off for batched products pending GA9-followup",
    ))

    # Apple GPU — all 17 GA primitives ship fused MSL kernels as of
    # GA11 (2026-05-17). geo_product + rotor_sandwich additionally carry
    # fp16 + bf16 ports; the other 15 ops are fp32-only for v1.
    # exp_mv / log_mv use closed-form trigonometric MSL (cos/sin/atan2)
    # with a power-series fallback for the general (non-bivector) case.
    # ext_deriv / vec_deriv / codiff / integral are field-signature
    # kernels — they take a sampled 3D grid + per-axis spacings
    # (D0, D1, D2, h0, h1, h2) instead of the (in, out, batch) ABI.
    fused_spec = _CLIFFORD_APPLE_GPU_FUSED[op_name]
    entries.append(BackendKernelEntry(
        target="apple_gpu",
        status=_FUSED_KERNEL_STATUS,
        dtypes=tuple(fused_spec["dtypes"]),
        feature_flags=("clifford_dialect", "msl", "metal"),
        notes=(
            "Fused MSL kernel(s): "
            + ", ".join(
                f"{fused_spec['symbol_prefix']}{dt}"
                for dt in fused_spec["dtypes"]
            )
            + " — apple_gpu_runtime.mm; verified bitwise vs Python GA reference."
        ),
    ))

    # NVIDIA — planned, gated on Phase G.  No per-arch breakout yet;
    # the artifact will land when Phase G H100 BF16 GEMM is green.
    entries.append(BackendKernelEntry(
        target="nvidia_sm90",
        status=_PLANNED_STATUS,
        dtypes=_CLIFFORD_PLANNED_GPU_DTYPES,
        feature_flags=("clifford_dialect", "wgmma"),
        notes="Gated on Phase G; canonical bf16 Cl(3,0) bivector kernel is the first target",
    ))

    # ROCm — planned, gated on Phase H.
    entries.append(BackendKernelEntry(
        target="rocm",
        status=_PLANNED_STATUS,
        dtypes=_CLIFFORD_PLANNED_GPU_DTYPES,
        feature_flags=("clifford_dialect", "mfma"),
        notes="Gated on Phase H",
    ))

    return entries


# ─────────────────────────────────────────────────────────────────────────────
# EBM (energy-based model) kernel manifest — 2026-05-17.
#
# Parallel to the GA dispatch table above: EBM primitives live in
# `tessera.ebm.*` (`primitive_coverage.py` registers them under
# `category="ebm"`).  They are not in the tensor `OP_SPECS` catalog, so
# they need a dedicated table that `manifest_for()` routes to via the
# `ebm_*` prefix.
#
# Status as of the GA + EBM native benchmark milestone:
#   - The Python reference path in `tessera.ebm` runs on every CPU host
#     (x86 / apple_cpu) — both targets declare `reference`.
#   - Eight EBM rows have fused Apple GPU MSL dispatch:
#       ebm_inner_step, ebm_refinement, ebm_langevin_step,
#       ebm_decode_init, ebm_bivector_langevin, ebm_sphere_langevin,
#       ebm_self_verify, ebm_energy (quadratic specialization).
#   - One core benchmark row remains Python-only on Apple GPU:
#       ebm_partition_exact.
#   - Other registered partition-estimator variants remain planned until
#     they get dedicated benchmark rows and native kernels.
# ─────────────────────────────────────────────────────────────────────────────

_EBM_CPU_DTYPES = ("fp32", "fp64")
_EBM_APPLE_GPU_BASELINE_DTYPES = ("fp32",)
_EBM_PLANNED_GPU_DTYPES = ("fp32", "fp16", "bf16")

_EBM_APPLE_GPU_FUSED: dict[str, dict[str, Any]] = {
    # Pointwise affine: out = y - eta * grad. ABI: (y, grad, eta, out, n).
    "ebm_inner_step": {
        "symbol": "tessera_apple_gpu_ebm_inner_step_f32",
        "dtypes": ("fp32",),
        "abi": "(y:f32*, grad:f32*, eta:f32, out:f32*, n:i32)",
        "notes": (
            "Pointwise EBM inner step on Apple GPU — out[i] = y[i] - "
            "eta * grad[i]. First native EBM primitive."
        ),
    },
    # EBT refinement chain: T inner-step iterations on-device with
    # ping-pong buffers. Same kernel body as inner_step, dispatched in a
    # loop with std::swap of the working buffers between iterations.
    "ebm_refinement": {
        "symbol": "tessera_apple_gpu_ebm_refinement_f32",
        "dtypes": ("fp32",),
        "abi": "(y0:f32*, grad:f32*, eta:f32, T:i32, y_out:f32*, n:i32)",
        "notes": (
            "EBT-style refinement on Apple GPU — T iterations of "
            "inner_step on-device with ping-pong buffers."
        ),
    },
    # Pointwise Langevin step with caller-supplied noise buffer.
    # ABI: (y, grad, noise, eta, noise_scale, out, n).
    "ebm_langevin_step": {
        "symbol": "tessera_apple_gpu_ebm_langevin_step_f32",
        "dtypes": ("fp32",),
        "abi": ("(y:f32*, grad:f32*, noise:f32*, eta:f32, noise_scale:f32, "
                "out:f32*, n:i32)"),
        "notes": (
            "Affine Langevin step on Apple GPU — out[i] = y[i] - eta * "
            "grad[i] + noise_scale * noise[i]. Caller pre-generates noise "
            "from tessera.rng (deterministic Philox)."
        ),
    },
    # Manifold Langevin steps (geo_sampling) — real Apple GPU kernels,
    # 2026-06-02. ``sphere`` has a dedicated tangent-projected MSL kernel;
    # ``bivector`` reuses the affine ``ebm_langevin_step`` kernel on the
    # grade-2 coefficient vector (the grade projection is the identity on an
    # already-bivector state, so the on-device update is exactly the affine
    # step). Both flip planned→partial: real native kernel + numpy reference
    # in tessera.ebm.geo_sampling; contract axes (transpose/sharding) stay
    # partial pending Phase G mesh work.
    "ebm_sphere_langevin_step": {
        "symbol": "tessera_apple_gpu_ebm_sphere_langevin_step_f32",
        "dtypes": ("fp32",),
        "abi": ("(x:f32*, grad:f32*, noise:f32*, eta:f32, noise_scale:f32, "
                "out:f32*, d:i32)"),
        "notes": (
            "One Langevin step on S^{d-1} — tangent-project the affine "
            "y - eta*grad + noise_scale*noise update, then renormalize to "
            "the sphere. Caller pre-generates noise from tessera.rng."
        ),
    },
    "ebm_bivector_langevin_step": {
        "symbol": "tessera_apple_gpu_ebm_langevin_step_f32",
        "dtypes": ("fp32",),
        "abi": ("(y:f32*, grad:f32*, noise:f32*, eta:f32, noise_scale:f32, "
                "out:f32*, n:i32)"),
        "notes": (
            "One Langevin step on the grade-2 (bivector) subspace of "
            "Cl(p,0) — the on-device update reuses the affine "
            "ebm_langevin_step kernel on the bivector coefficient vector "
            "(grade projection is identity on an already-bivector state)."
        ),
    },
    # M6 Step 4 (2026-05-18): on-device Philox-4x32-10 variant of
    # langevin_step.  Generates standard-normal noise inside the MSL
    # kernel from a (key, counter) tuple, eliminating the host noise
    # buffer + upload.  Constants match tessera.compiler.philox.
    "ebm_langevin_step_philox": {
        "symbol": "tessera_apple_gpu_ebm_langevin_step_philox_f32",
        "dtypes": ("fp32",),
        "abi": ("(y:f32*, grad:f32*, eta:f32, noise_scale:f32, "
                "key:u32*, counter:u32*, out:f32*, n:i32)"),
        "notes": (
            "Affine Langevin step with on-device Philox-4x32-10 RNG. "
            "Per-thread counter = (counter[0]+i, counter[1..3]); Box-Muller "
            "maps the first two uniforms to a standard normal.  Same "
            "math as ebm_langevin_step but no host noise buffer."
        ),
    },
    # decode_init noise-apply: out[i] = base[i % base_len] + std * noise[i].
    # Caller pre-generates `noise` via tessera.rng; `base_len=0` means
    # mean=0 (pure-noise init), `base_len=n` means no broadcasting.
    "ebm_decode_init": {
        "symbol": "tessera_apple_gpu_ebm_decode_init_noise_apply_f32",
        "dtypes": ("fp32",),
        "abi": ("(base:f32*, base_len:i32, noise:f32*, std:f32, "
                "out:f32*, n:i32)"),
        "notes": (
            "decode_init(strategy='noise') applied on Apple GPU — "
            "broadcasts `base` across K × event dims and adds `std * "
            "noise[i]`. Caller pre-generates noise from RNGKey."
        ),
    },
    # Geometric Langevin on bivectors — reuses ebm_langevin_step kernel
    # because grade-restricted state + projected gradient + projected
    # noise are all 8-coefficient affine ops. The "composition" entry
    # documents that this op dispatches the same MSL kernel as
    # ebm_langevin_step (no separate symbol), so the manifest reports
    # the same symbol but a different ABI shape (grade-projected inputs).
    "ebm_bivector_langevin": {
        "symbol": "tessera_apple_gpu_ebm_langevin_step_f32",
        "dtypes": ("fp32",),
        "abi": ("(y:f32*[8], grad_proj:f32*[8], noise_proj:f32*[8], "
                "eta:f32, noise_scale:f32, out:f32*[8])"),
        "notes": (
            "Bivector Langevin step on Apple GPU — composition of GA "
            "grade_projection (already native; applied host-side via "
            "tessera.ga.grade_projection or natively via "
            "tessera_apple_gpu_clifford_grade_projection) + the affine "
            "Langevin kernel above. Demonstrates GA-kernel reuse for "
            "manifold-aware EBM sampling."
        ),
    },
    # Sphere Langevin step — full tangent-projection + retract in one
    # MSL kernel.  d is the ambient dimension (3 for S^2, etc.).
    "ebm_sphere_langevin": {
        "symbol": "tessera_apple_gpu_ebm_sphere_langevin_step_f32",
        "dtypes": ("fp32",),
        "abi": ("(x:f32*, grad:f32*, noise:f32*, eta:f32, noise_scale:f32, "
                "out:f32*, d:i32)"),
        "notes": (
            "Sphere Langevin step on Apple GPU — tangent projection + "
            "Euler-Maruyama update + retract to S^{d-1}, all in a single "
            "MSL kernel. Single-thread dispatch (d is small)."
        ),
    },
    # Hard-argmin self_verify — one threadgroup per batch row scans K
    # energies + gathers the winning candidate row.
    "ebm_self_verify": {
        "symbol": "tessera_apple_gpu_ebm_self_verify_hard_argmin_f32",
        "dtypes": ("fp32",),
        "abi": ("(energies:f32*[BxK], candidates:f32*[BxKxD], "
                "out:f32*[BxD], B:i32, K:i32, D:i32)"),
        "notes": (
            "Hard-argmin self_verify on Apple GPU — for each batch row, "
            "find k* = argmin_k(energies[b, k]) and copy "
            "candidates[b, k*, :] into out[b, :]. Soft-min (beta > 0) "
            "is a separate kernel pending follow-up."
        ),
    },
    # ebm_energy — quadratic specialization E_b = 0.5 * ||x_b - y_b||^2.
    # Caller opts in when their model_fn matches this shape.
    "ebm_energy": {
        "symbol": "tessera_apple_gpu_ebm_energy_quadratic_f32",
        "dtypes": ("fp32",),
        "abi": "(x:f32*[BxD], y:f32*[BxD], energies:f32*[B], B:i32, D:i32)",
        "notes": (
            "Quadratic energy specialization E_b = 0.5 * ||x_b - y_b||^2 "
            "on Apple GPU — the dominant EBT / diffusion energy form. "
            "Arbitrary user energy_fn lifts to MSL is a follow-up sprint."
        ),
    },
    # ebm_energy_quadratic — the tensor-clean registry name for the SAME fused
    # kernel as ``ebm_energy`` (runtime.py uses this canonical name + the
    # value-call lane ``tessera_apple_gpu_ebm_energy_quadratic_value_f32``).
    # Both names map to one Metal kernel; carried here so the registry op
    # ``ebm_energy_quadratic`` reports fused instead of planned.
    "ebm_energy_quadratic": {
        "symbol": "tessera_apple_gpu_ebm_energy_quadratic_f32",
        "dtypes": ("fp32",),
        "abi": "(x:f32*[BxD], y:f32*[BxD], energies:f32*[B], B:i32, D:i32)",
        "notes": (
            "Tensor-clean quadratic energy E_b = 0.5 * ||x_b - y_b||^2; same "
            "fused MSL kernel as ``ebm_energy`` + a value-call lane "
            "(``ebm_energy_quadratic_value_f32``) for @jit(target='apple_gpu')."
        ),
    },
    # ebm_partition_exact — stable logsumexp on a precomputed energies
    # array.  Caller hands the kernel the per-state energies (typically
    # produced by `ebm.energy_quadratic` or another `ebm_energy_*` call)
    # plus a temperature; the kernel returns Z = Σ_i exp(-E_i/T).
    # Single-thread MSL today (N is typically small for exhaustive
    # state enumeration); parallel tree-reduction is a follow-up.
    "ebm_partition_exact": {
        "symbol": "tessera_apple_gpu_ebm_partition_exact_f32",
        "dtypes": ("fp32",),
        "abi": ("(energies:f32*[N], N:i32, temperature:f32, out:f32*)"),
        "notes": (
            "Stable logsumexp over a precomputed energies array — "
            "Z = Σ_i exp(-E_i/T).  Closes the 8/9 → 9/9 native EBM "
            "gap.  Caller provides the energies (any other EBM "
            "energy kernel produces them)."
        ),
    },
    # ebm_ebt_tiny — fused EBT refinement + energy + hard-argmin in a
    # single MSL dispatch.  Optimization for the ebt_tiny workload —
    # the standalone refinement + self_verify chain loses at small
    # shapes because per-dispatch overhead dominates; this fused
    # kernel collapses both into one dispatch.
    "ebm_ebt_tiny": {
        "symbol": "tessera_apple_gpu_ebm_ebt_tiny_refinement_argmin_f32",
        "dtypes": ("fp32",),
        "abi": ("(y0:f32*[BxKxD], grad:f32*[BxKxD], eta:f32, T:i32, "
                "out:f32*[BxD], B:i32, K:i32, D:i32)"),
        "notes": (
            "Fused EBT-tiny pipeline: streaming closed-form refinement "
            "+ per-row squared-norm energy + K-way hard argmin, all in "
            "one Metal dispatch.  K <= 256 (threadgroup-size budget for "
            "the argmin reduction); D is unbounded after the 2026-05-17 "
            "register-vector elimination.  Bit-equivalent to the "
            "ebm.refinement → ebm.self_verify chain but with one "
            "dispatch instead of two."
        ),
    },
}

# All EBM primitives currently covered by the manifest (the union of the
# fused set + every Python-reference-only entry).  Kept explicit so the
# manifest is self-documenting and `audit_backend_dtypes()` can walk it.
_EBM_PRIMITIVES: tuple[str, ...] = (
    # EBM1 — core energy/inner-loop primitives.
    "ebm_energy",
    "ebm_energy_quadratic",  # tensor-clean alias of ebm_energy (same fused kernel)
    "ebm_inner_step",
    "ebm_refinement",       # EBT-style refinement loop
    "ebm_langevin_step",
    "ebm_self_verify",
    "ebm_decode_init",
    # EBM3 — partition function family.
    "ebm_partition_exact",
    "ebm_partition_monte_carlo",
    "ebm_partition_ais",
    # EBM7 — manifold-aware integrators. Both the legacy no-suffix labels
    # and the canonical registry/runtime ``_step`` names (2026-06-02: the
    # ``_step`` ops are the ones the primitive registry + Apple GPU runtime
    # symbols actually use, so they must be in the manifest set to flip
    # planned→partial).
    "ebm_bivector_langevin",
    "ebm_sphere_langevin",
    "ebm_bivector_langevin_step",
    "ebm_sphere_langevin_step",
    # The chain wrappers (sample = loop of step) — reference-backed
    # (real numpy chain in tessera.ebm.geo_sampling, GPU per-step), so they
    # carry a reference manifest slot and resolve partial. No *dedicated*
    # fused chain kernel yet (that would be a future on-device-loop kernel
    # like ebm_refinement), so apple_gpu stays planned for these two.
    "ebm_bivector_langevin_sample",
    "ebm_sphere_langevin_sample",
    # EBT-tiny fused-pipeline optimization (2026-05-17).
    "ebm_ebt_tiny",
    # M6 Step 4 (2026-05-18) — on-device Philox RNG variant of
    # langevin_step (Apple GPU fused only; CPU reference path
    # provided by the Python wrapper).
    "ebm_langevin_step_philox",
)


def ebm_manifest_for(op_name: str) -> list[BackendKernelEntry]:
    """Return the backend manifest entries for an ``ebm_*`` primitive.

    Status semantics:
      - ``x86`` + ``apple_cpu``: ``reference`` (Python implementation
        in ``tessera.ebm.*`` runs on every CPU host).
      - ``apple_gpu``: ``fused`` for primitives in
        ``_EBM_APPLE_GPU_FUSED``; ``planned`` for everything else.
      - ``nvidia_sm90`` + ``rocm``: ``planned`` (Phase G / H gating).

    The benchmark driver uses this to label each EBM row's backend
    column instead of carrying a row-local ``python_reference_only``
    string.
    """
    if op_name not in _EBM_PRIMITIVES:
        return []
    entries: list[BackendKernelEntry] = []

    # CPU targets — Python reference path on every host.
    entries.append(BackendKernelEntry(
        target="x86",
        status=_REFERENCE_STATUS,
        dtypes=_EBM_CPU_DTYPES,
        feature_flags=("ebm_namespace", "numpy_reference"),
        notes="Python EBM reference (tessera.ebm.*)",
    ))
    entries.append(BackendKernelEntry(
        target="apple_cpu",
        status=_REFERENCE_STATUS,
        dtypes=_EBM_CPU_DTYPES,
        feature_flags=("ebm_namespace", "numpy_reference"),
        notes="Python EBM reference; Accelerate hand-off pending follow-up",
    ))

    # Apple GPU — first native EBM primitives landed 2026-05-17.
    fused_spec = _EBM_APPLE_GPU_FUSED.get(op_name)
    if fused_spec is not None:
        entries.append(BackendKernelEntry(
            target="apple_gpu",
            status=_FUSED_KERNEL_STATUS,
            dtypes=tuple(fused_spec["dtypes"]),
            feature_flags=("ebm_namespace", "msl", "metal"),
            notes=(
                f"Fused MSL kernel: {fused_spec['symbol']} "
                f"— ABI {fused_spec['abi']}. {fused_spec['notes']}"
            ),
        ))
    else:
        entries.append(BackendKernelEntry(
            target="apple_gpu",
            status=_PLANNED_STATUS,
            dtypes=_EBM_APPLE_GPU_BASELINE_DTYPES,
            feature_flags=("ebm_namespace", "msl"),
            notes=(
                "Python reference is the v1 execution path on Apple GPU; "
                "native MSL kernel pending."
            ),
        ))

    # NVIDIA / ROCm — both planned, gated on Phase G / H.
    entries.append(BackendKernelEntry(
        target="nvidia_sm90",
        status=_PLANNED_STATUS,
        dtypes=_EBM_PLANNED_GPU_DTYPES,
        feature_flags=("ebm_namespace",),
        notes="Gated on Phase G",
    ))
    entries.append(BackendKernelEntry(
        target="rocm",
        status=_PLANNED_STATUS,
        dtypes=_EBM_PLANNED_GPU_DTYPES,
        feature_flags=("ebm_namespace",),
        notes="Gated on Phase H",
    ))

    return entries


def _public_to_graph_name(public: str) -> str:
    """Convert a public op name (e.g., ``"matmul"``) to its catalog
    graph_name (e.g., ``"tessera.matmul"``)."""
    spec = OP_SPECS.get(public)
    return spec.graph_name if spec is not None else f"tessera.{public}"


def _capability_status(target_name: str, op_name: str) -> tuple[str, tuple[str, ...]] | None:
    """Pull (runtime_status, dtypes) for an op on a target.

    Returns ``None`` when the target has no entry for this op.
    """
    cap = TARGET_CAPABILITIES.get(target_name)
    if cap is None:
        return None
    graph_name = _public_to_graph_name(op_name)
    canonical = canonical_op(graph_name)
    if canonical not in cap.supported_ops:
        return None
    op_cap = cap.supported_ops[canonical]
    return (op_cap.runtime_status, tuple(op_cap.dtypes))


# M7 follow-up (2026-05-18) — MSL kernels for the conformal-primitive
# surface.  Only the 4 ops with real GPU benefit ship native kernels;
# ``conformal_jacobian`` (4-call host composition, no GPU win) and
# ``laplacian_2d`` (small stencil, host numpy is faster at our typical
# sizes) intentionally don't get manifest entries — they stay CPU-only
# until a real workload demonstrates a GPU benefit.
_COMPLEX_APPLE_GPU_FUSED: dict[str, dict[str, Any]] = {
    "complex_mul": {
        "symbol": "tessera_apple_gpu_complex_mul_f32",
        "dtypes": ("fp32",),
        "abi": ("(a_re:f32*, a_im:f32*, b_re:f32*, b_im:f32*, "
                "out_re:f32*, out_im:f32*, n:i32)"),
        "notes": (
            "Elementwise complex multiplication on Apple GPU — "
            "(a + bi)(c + di) = (ac − bd) + (ad + bc)i."
        ),
    },
    "complex_exp": {
        "symbol": "tessera_apple_gpu_complex_exp_f32",
        "dtypes": ("fp32",),
        "abi": "(re:f32*, im:f32*, out_re:f32*, out_im:f32*, n:i32)",
        "notes": (
            "Elementwise complex exponential via Euler form — "
            "e^(a+bi) = e^a · (cos b, sin b)."
        ),
    },
    "complex_stereographic": {
        "symbol": "tessera_apple_gpu_complex_stereographic_f32",
        "dtypes": ("fp32",),
        "abi": ("(x:f32*, y:f32*, z:f32*, out_re:f32*, out_im:f32*, n:i32)"),
        "notes": (
            "Stereographic projection S² → ℂ — f(x,y,z) = (x + iy) / "
            "(1 − z).  North pole → ∞."
        ),
    },
    "complex_mobius": {
        "symbol": "tessera_apple_gpu_complex_mobius_f32",
        "dtypes": ("fp32",),
        "abi": ("(z_re:f32*, z_im:f32*, a_re:f32, a_im:f32, "
                "b_re:f32, b_im:f32, c_re:f32, c_im:f32, "
                "d_re:f32, d_im:f32, out_re:f32*, out_im:f32*, n:i32)"),
        "notes": (
            "Möbius transformation (az+b)/(cz+d) — broadcasts scalar "
            "(a, b, c, d) across the input batch."
        ),
    },
}


_COMPLEX_PRIMITIVES: tuple[str, ...] = (
    "complex_mul",
    "complex_exp",
    "complex_stereographic",
    "complex_mobius",
    # Host-only by design (no manifest entry above):
    #   complex_conjugate, complex_abs, conformal_jacobian, laplacian_2d
)


# E3 (2026-05-20) — every M7 Visual Complex op (4 fused + 16 long-tail)
# routes through ``complex_manifest_for``.  The 16 long-tail ops don't
# yet ship native MSL / WGMMA / MFMA kernels; the manifest declares
# ``reference`` status on CPU targets and ``planned`` slots on
# apple_gpu / nvidia_sm90+ / rocm so the IR has reserved slots ready
# for Phase G / H / M7-follow-up kernel work to slot into.  The audit
# walker now sees a populated manifest for the entire M7 surface
# (not just the 4 fused ops), which is what flips them out of
# ``target_ir=planned``.
_M7_LONG_TAIL: tuple[str, ...] = (
    # Pointwise complex math (7) — all elementwise on packed (re, im).
    "complex_div",
    "complex_log",
    "complex_sqrt",
    "complex_pow",
    "complex_conjugate",
    "complex_abs",
    "complex_arg",
    # Möbius / projective family (1 — the other 2 are already fused).
    "mobius_from_three_points",
    # Cross-ratio / cocircularity / Cauchy-Riemann certificate (3).
    "cross_ratio",
    "is_concyclic",
    "check_cauchy_riemann",
    # Wirtinger derivatives + Laplacian (3 stencils on (re, im) grid).
    "dz",
    "dbar",
    "laplacian_2d",
    # Conformal Jacobian + energy on sphere (2).
    "conformal_jacobian",
    "conformal_energy_on_sphere",
)

# Per-op dtype matrix for M7 long-tail ops.
#
# ``_M7_LONG_TAIL_DTYPES`` is what the **Python reference path supports
# today** — fp32 only.  This is the dtype set we use on
# ``status="reference"`` entries (cpu / x86 / apple_cpu).  Treating
# numpy's complex-pair representation as supporting fp16/bf16 today
# would be an overclaim — those need explicit storage/accum-split
# implementations.
#
# ``_M7_LONG_TAIL_PLANNED_GPU_DTYPES`` is the **target dtype matrix for
# the unbuilt native kernels** — fp32 + fp16 + bf16.  This is what we
# attach to ``status="planned"`` entries on apple_gpu / nvidia_sm90+ /
# rocm so Phase G / H / M7-follow-up kernel work has a clear dtype
# contract to satisfy.  When those kernels land and ``status`` flips
# to ``fused``, this set becomes the live kernel's dtype matrix
# unchanged (the storage/accum split for complex math typically wants
# fp16 storage + fp32 accum, captured via the op's
# ``NumericPolicy(storage=..., accum=...)`` in
# ``primitive_coverage.py``).
_M7_LONG_TAIL_DTYPES: tuple[str, ...] = ("fp32",)
_M7_LONG_TAIL_PLANNED_GPU_DTYPES: tuple[str, ...] = ("fp32", "fp16", "bf16")


def complex_manifest_for(op_name: str) -> list[BackendKernelEntry]:
    """Return backend manifest entries for a M7 Visual Complex primitive.

    Coverage tiers:
      * **Fused (4)** — ``complex_mul``, ``complex_exp``,
        ``complex_mobius``, ``complex_stereographic``.  Ship native MSL
        kernels on Apple GPU; the audit reads ``target_ir=fused``.
      * **Long-tail (16)** — the rest of the M7 surface
        (``complex_div``/``log``/``sqrt``/``pow``/``conjugate``/``abs``/
        ``arg``, ``mobius_from_three_points``, ``cross_ratio``,
        ``is_concyclic``, ``check_cauchy_riemann``, ``dz``, ``dbar``,
        ``laplacian_2d``, ``conformal_jacobian``,
        ``conformal_energy_on_sphere``).  These run today via the
        Python reference path in ``tessera.complex.*``; native kernel
        slots on apple_gpu / nvidia_sm90+ / rocm are reserved as
        ``planned`` so the audit walker reflects the future intent
        (and Phase G / H / M7-follow-up work knows which slots to
        target).

    Returns an empty list for ops outside the M7 inventory.
    """
    in_fused = op_name in _COMPLEX_PRIMITIVES
    in_long_tail = op_name in _M7_LONG_TAIL
    if not (in_fused or in_long_tail):
        return []
    entries: list[BackendKernelEntry] = []
    # CPU targets — Python reference always available for the entire
    # M7 surface (the numpy code in ``tessera.complex.*`` runs on
    # cpu / x86 / apple_cpu without any backend-specific dispatch).
    # The ``cpu`` entry mirrors the generic catalog path's "numpy
    # reference always available as fallback" emission — needed so
    # ``support.tier()`` sees a ``target_ir=reference`` row on the
    # ``cpu`` target and reports REFERENCE_ONLY instead of falling
    # through to the runtime=ready NATIVE_READY branch.
    entries.append(BackendKernelEntry(
        target="cpu",
        status=_REFERENCE_STATUS,
        dtypes=("fp32",),
        feature_flags=("numpy", "reference_execution"),
        notes="numpy reference path (tessera.complex.*)",
    ))
    entries.append(BackendKernelEntry(
        target="x86",
        status=_REFERENCE_STATUS,
        dtypes=("fp32",),
        feature_flags=("complex_namespace", "numpy_reference"),
        notes="Python complex reference (tessera.complex.*)",
    ))
    entries.append(BackendKernelEntry(
        target="apple_cpu",
        status=_REFERENCE_STATUS,
        dtypes=("fp32",),
        feature_flags=("complex_namespace", "numpy_reference"),
        notes="Python complex reference (tessera.complex.*)",
    ))
    # Apple GPU — fused MSL kernel for the 4-op fused subset; planned
    # slot (with full dtype matrix) for the long-tail.
    fused = _COMPLEX_APPLE_GPU_FUSED.get(op_name)
    if fused is not None:
        entries.append(BackendKernelEntry(
            target="apple_gpu",
            status=_FUSED_KERNEL_STATUS,
            dtypes=tuple(fused["dtypes"]),
            feature_flags=("complex_namespace", "msl", "metal"),
            notes=(
                f"Fused MSL kernel: {fused['symbol']} — "
                f"ABI {fused['abi']}. {fused['notes']}"
            ),
        ))
    elif in_long_tail:
        entries.append(BackendKernelEntry(
            target="apple_gpu",
            status=_PLANNED_STATUS,
            dtypes=_M7_LONG_TAIL_PLANNED_GPU_DTYPES,
            feature_flags=("complex_namespace", "msl", "metal"),
            notes=(
                "Planned kernel target dtypes (fp32 + fp16 + bf16). "
                "Today: this op runs only via the Python reference path "
                "above (fp32-only); the fp16/bf16 entries here describe "
                "what the future MSL kernel will support, not what runs "
                "now. Promotion gated on the M7 follow-up kernel sprint."
            ),
        ))
    # NVIDIA — planned slots across SM_80 / SM_90 / SM_100 / SM_120.
    # The 4 fused ops are the canonical first kernels for Phase G; the
    # long-tail is gated on the same Phase G milestone.  cuComplex.h-style
    # PTX intrinsics cover complex_log / sqrt / pow / div / conjugate /
    # abs / arg trivially once the SM_90 BF16 GEMM baseline is green.
    for target_name, flags, arch_min in (
        ("nvidia_sm80",  ("complex_namespace", "wmma"),      "sm_80"),
        ("nvidia_sm90",  ("complex_namespace", "wgmma"),     "sm_90a"),
        ("nvidia_sm100", ("complex_namespace", "tcgen05"),   "sm_100a"),
        ("nvidia_sm120", ("complex_namespace", "tcgen05"),   "sm_120a"),
    ):
        entries.append(BackendKernelEntry(
            target=target_name,
            status=_PLANNED_STATUS,
            dtypes=_M7_LONG_TAIL_PLANNED_GPU_DTYPES,
            feature_flags=flags,
            notes=(
                "Planned WGMMA/tcgen05 kernel target dtypes "
                "(fp32 + fp16 + bf16). Today: this op runs only via the "
                "Python reference path (fp32-only); fp16/bf16 lanes "
                "land with the actual Phase G kernel work."
            ),
            cuda_arch_min=arch_min,
            nvcc_version_min="13.3",
        ))
    # ROCm — planned slot, gated on Phase H.
    entries.append(BackendKernelEntry(
        target="rocm",
        status=_PLANNED_STATUS,
        dtypes=_M7_LONG_TAIL_PLANNED_GPU_DTYPES,
        feature_flags=("complex_namespace", "mfma"),
        notes=(
            "Planned MFMA kernel target dtypes (fp32 + fp16 + bf16). "
            "Today: this op runs only via the Python reference path "
            "(fp32-only); fp16/bf16 lanes land with the Phase H "
            "kernel work."
        ),
        hipcc_version_min="7.2.3",
    ))
    return entries


# A1 (2026-06-18) — GEMM-family ops that carry a unified MMA descriptor.
# Mirrors ``primitive_coverage._ROCM_MMA_OPS`` (kept as a local literal so this
# module stays importable without pulling in the audit registry).
_ROCM_MMA_OP_NAMES: frozenset[str] = frozenset({
    "matmul", "batched_gemm", "grouped_gemm", "dequant_grouped_gemm",
    "linear_general", "qkv_projection", "factorized_matmul",
})


def _rocm_mma_descriptor_for(op_name: str, dtypes: tuple[str, ...]):
    """Build the unified MMA descriptor for a ROCm GEMM-family entry, using the
    first dtype in ``dtypes`` that has a matrix-core path on gfx942 (the ``rocm``
    alias).  Returns ``None`` for non-GEMM ops or when no dtype resolves."""
    if op_name not in _ROCM_MMA_OP_NAMES:
        return None
    from .rocm_mma import select_mma
    from .rocm_target import AMDArch, TesseraROCmTargetError
    for dt in dtypes:
        try:
            return select_mma(AMDArch.GFX_942, dt)
        except TesseraROCmTargetError:
            continue
    return None


def manifest_for(op_name: str) -> list[BackendKernelEntry]:
    """Return the backend manifest entries for ``op_name``.

    Order: cpu / x86 / apple_cpu / apple_gpu / nvidia_sm80 / sm90 / sm100 /
    sm120 / rocm.  Each entry's status reflects whether the
    target ships a fused kernel, a reference path, an artifact-only stub,
    or has no plan.

    GA9 (2026-05-17): `clifford_*` op names are dispatched to a
    parallel `clifford_manifest_for()` table since they aren't part of
    the tensor `OP_SPECS` catalog.
    M7 follow-up (2026-05-18): same pattern for `complex_*`.
    """
    # Domain manifests (clifford / ebm / complex) are built by parallel
    # tables outside OP_SPECS. They must still honor _NUMERICAL_FIXTURES —
    # audit 2026-06-10 found these early returns bypassed the attach on the
    # main path below, so a domain op could never receive a numerical fixture.
    if op_name.startswith("clifford_"):
        return _attach_numerical_fixtures(op_name, clifford_manifest_for(op_name))
    if op_name.startswith("ebm_"):
        return _attach_numerical_fixtures(op_name, ebm_manifest_for(op_name))
    # E3 (2026-05-20): route every M7 Visual Complex op through
    # ``complex_manifest_for`` — the 9 non-prefixed names (mobius /
    # cross_ratio / dz / dbar / laplacian_2d / ...) need the same
    # CPU-reference + GPU-planned-slot treatment as the prefixed
    # ones, but they don't carry a ``complex_`` prefix.
    if (op_name.startswith("complex_")
            or op_name in _COMPLEX_APPLE_GPU_FUSED
            or op_name in _M7_LONG_TAIL):
        return _attach_numerical_fixtures(op_name, complex_manifest_for(op_name))
    entries: list[BackendKernelEntry] = []

    # x86 AMX
    x86 = _X86_KERNELS.get(op_name)
    if x86 is not None:
        entries.append(BackendKernelEntry(
            target="x86",
            status=str(x86["status"]),
            dtypes=tuple(x86["dtypes"]),
            feature_flags=("amx", "avx512"),
            notes=str(x86.get("notes", "")),
        ))

    # Apple CPU
    apple_cpu = _APPLE_CPU_KERNELS.get(op_name)
    if apple_cpu is not None:
        entries.append(BackendKernelEntry(
            target="apple_cpu",
            status=str(apple_cpu["status"]),
            dtypes=tuple(apple_cpu["dtypes"]),
            feature_flags=("accelerate", "bnns"),
            notes=str(apple_cpu.get("notes", "")),
        ))
    else:
        # Apple CPU falls back to reference for everything else (Accelerate
        # has cblas_sgemm + numpy reference for the rest of the catalog).
        cap = _capability_status("apple_cpu", op_name)
        if cap is not None:
            status, dtypes = cap
            mapped = _REFERENCE_STATUS if status == "ready" else _ARTIFACT_STATUS
            entries.append(BackendKernelEntry(
                target="apple_cpu",
                status=mapped,
                dtypes=dtypes,
                feature_flags=("accelerate",),
                notes="reference path via numpy/Accelerate",
            ))

    # Apple GPU (shipped MSL kernels)
    apple_gpu = _APPLE_GPU_KERNELS.get(op_name)
    if apple_gpu is not None:
        # Project 3 (2026-06-01) — when an op is promoted to
        # ``hardware_verified``, both ``runtime_symbol`` and
        # ``execute_compare_fixture`` are required at construction.
        # Pull both from the source-of-truth tables BEFORE building
        # the entry so the validator sees a complete contract.
        _ag_status = str(apple_gpu["status"])
        _ag_runtime_symbol: Optional_str = apple_gpu.get("runtime_symbol")
        _ag_shape_envelope: Optional_str = apple_gpu.get("shape_envelope")
        _ag_fixture: Optional_str = (
            _NUMERICAL_FIXTURES.get((op_name, "apple_gpu"))
            if _ag_status == _HARDWARE_VERIFIED_STATUS else None)
        entries.append(BackendKernelEntry(
            target="apple_gpu",
            status=_ag_status,
            dtypes=tuple(apple_gpu["dtypes"]),
            feature_flags=("metal", "mps", "msl"),
            notes=str(apple_gpu.get("notes", "")),
            runtime_symbol=_ag_runtime_symbol,
            shape_envelope=_ag_shape_envelope,
            execute_compare_fixture=_ag_fixture,
            # P2 (2026-06-09) — hot-path rows carry their ratchet baseline.
            benchmark_json=apple_gpu.get("benchmark_json"),
            # P1 (2026-06-10) — uniform structured hot-path benchmark metadata.
            benchmark_metadata=_APPLE_GPU_HOT_PATH_METADATA.get(op_name),
        ))
    else:
        cap = _capability_status("apple_gpu", op_name)
        if cap is not None:
            status, dtypes = cap
            mapped = (
                _REFERENCE_STATUS if status == "ready"
                else _ARTIFACT_STATUS if status == "artifact_only"
                else _PLANNED_STATUS
            )
            entries.append(BackendKernelEntry(
                target="apple_gpu",
                status=mapped,
                dtypes=dtypes,
                feature_flags=("metal",),
                notes="capability-registered Apple GPU coverage",
            ))

    # NVIDIA SM_80 / SM_90 / SM_100 / SM_120 — artifact-only until Phase G.
    # Sprint G-3 (2026-05-11): each entry carries cuda_arch_min /
    # nvcc_version_min + WGMMA shape (Hopper+ only).  Per-kernel shape
    # overrides come from `_NVIDIA_KERNEL_TILE_SHAPES` below.
    _kernel_shapes = _NVIDIA_KERNEL_TILE_SHAPES.get(op_name, {})
    for target_name, flags, arch_min in (
        ("nvidia_sm80",  ("wmma",),                  "sm_80"),
        ("nvidia_sm90",  ("wgmma", "tma"),           "sm_90a"),
        ("nvidia_sm100", ("tcgen05", "tmem"),        "sm_100a"),
        ("nvidia_sm120", ("tcgen05", "tmem"),        "sm_120a"),
    ):
        cap = _capability_status(target_name, op_name)
        if cap is not None:
            status, dtypes = cap
            mapped = (
                _FUSED_KERNEL_STATUS if status == "ready"
                else _ARTIFACT_STATUS if status == "artifact_only"
                else _PLANNED_STATUS
            )
            # WGMMA only kicks in at SM_90+.  Use the per-kernel shape
            # if registered, otherwise default to the canonical
            # bf16/fp16 Hopper tile.
            wgmma_shape = None
            cluster = None
            if target_name != "nvidia_sm80" and op_name in _NVIDIA_KERNEL_TILE_SHAPES:
                wgmma_shape = _kernel_shapes.get("wgmma_shape")
                cluster = _kernel_shapes.get("cluster")
            mfu = _NVIDIA_KERNEL_MFU.get((op_name, target_name))
            roofline = _NVIDIA_KERNEL_ROOFLINE.get(op_name)
            entries.append(BackendKernelEntry(
                target=target_name,
                status=mapped,
                dtypes=dtypes,
                feature_flags=flags,
                notes=(
                    "Target IR artifact ships under CUDA 13.2 U1; "
                    "execution gated on Phase G"
                    if mapped == _ARTIFACT_STATUS
                    else ""
                ),
                cuda_arch_min=arch_min,
                nvcc_version_min="13.3",
                wgmma_shape=wgmma_shape,
                cluster_size=cluster,
                expected_mfu=mfu,
                roofline_target=roofline,
            ))

    # ROCm MFMA — Sprint H-3 (2026-05-11): attach MFMA shape + hipcc
    # version pin per kernel.
    cap = _capability_status("rocm", op_name)
    if cap is not None:
        status, dtypes = cap
        mapped = (
            _FUSED_KERNEL_STATUS if status == "ready"
            else _ARTIFACT_STATUS if status == "artifact_only"
            else _PLANNED_STATUS
        )
        mfma = _ROCM_KERNEL_MFMA_SHAPES.get(op_name)
        mfu = _ROCM_KERNEL_MFU.get((op_name, "rocm_gfx942"))
        # A1 (2026-06-18) — attach the unified MMA descriptor for GEMM-family
        # ops, derived from a representative dtype on gfx942 (the `rocm` alias).
        mma_desc = _rocm_mma_descriptor_for(op_name, dtypes)
        entries.append(BackendKernelEntry(
            target="rocm",
            status=mapped,
            dtypes=dtypes,
            feature_flags=("mfma",),
            notes=(
                "ROCm 7.2.3 MFMA artifact ships; HIP execution gated on Phase H"
                if mapped == _ARTIFACT_STATUS else ""
            ),
            mfma_shape=mfma,
            hipcc_version_min="7.2.3",
            expected_mfu=mfu,
            mma_descriptor=mma_desc,
        ))

    # CPU numpy reference — always available as fallback
    cap = _capability_status("cpu", op_name)
    if cap is not None:
        status, dtypes = cap
        if status == "ready":
            entries.append(BackendKernelEntry(
                target="cpu",
                status=_REFERENCE_STATUS,
                dtypes=dtypes,
                feature_flags=("numpy", "reference_execution"),
                notes="numpy reference path",
            ))

    # Audit follow-up A.3 (2026-05-31) — attach known
    # ``execute_compare_fixture`` paths to entries that ship a real
    # numerical-correctness test. Replaces the conformance matrix's
    # filename/content heuristic with first-class manifest data.
    entries = _attach_numerical_fixtures(op_name, entries)

    return entries


def all_manifests() -> Mapping[str, list[BackendKernelEntry]]:
    """Return the manifest for every op in ``OP_SPECS`` plus every
    `clifford_*` GA9 primitive.

    Useful for the dashboard renderer and for the audit walker that
    verifies dtype canonicalness across the full per-target matrix.
    """
    out: dict[str, list[BackendKernelEntry]] = {}
    for name in OP_SPECS:
        m = manifest_for(name)
        if m:
            out[name] = m
    # GA9: include clifford primitives that aren't in OP_SPECS.
    for name in _CLIFFORD_PRIMITIVES:
        m = manifest_for(name)
        if m:
            out[name] = m
    return out


def manifest_summary() -> dict[str, dict[str, int]]:
    """Roll up the manifest by (target, status) — useful for CLAUDE.md
    headline reporting.

    Returns
    -------
    dict[target, dict[status, count]]
        Count of ops by target × status.
    """
    summary: dict[str, dict[str, int]] = {}
    for entries in all_manifests().values():
        for e in entries:
            tgt = summary.setdefault(e.target, {})
            tgt[e.status] = tgt.get(e.status, 0) + 1
    return summary


def audit_backend_dtypes() -> dict[str, list[tuple[str, str, str]]]:
    """Walk the manifest, classify every dtype mention into canonical /
    alias / planned_gated / unknown buckets.

    Used as a parallel check to `primitive_coverage.audit_canonical_dtypes`
    — backend-side dtype hygiene gate.
    """
    from ..dtype import (
        is_canonical_dtype,
        is_planned_gated_dtype,
        dtype_aliases,
    )

    aliases = dtype_aliases()
    buckets: dict[str, list[tuple[str, str, str]]] = {
        "canonical": [],
        "alias": [],
        "planned_gated": [],
        "unknown": [],
    }
    for op_name, entries in all_manifests().items():
        for e in entries:
            for dt in e.dtypes:
                key = f"{op_name}::{e.target}"
                if is_canonical_dtype(dt):
                    buckets["canonical"].append((op_name, key, dt))
                elif dt in aliases:
                    buckets["alias"].append((op_name, key, dt))
                elif is_planned_gated_dtype(dt):
                    buckets["planned_gated"].append((op_name, key, dt))
                else:
                    buckets["unknown"].append((op_name, key, dt))
    return buckets


__all__ = [
    "BackendKernelEntry",
    "BenchmarkMetadata",
    "BENCHMARK_HOT_PATH_GROUPS",
    "manifest_for",
    "clifford_manifest_for",
    "ebm_manifest_for",
    "all_manifests",
    "manifest_summary",
    "audit_backend_dtypes",
]
