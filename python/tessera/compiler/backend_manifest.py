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
``partial``/``planned`` until real GPU execution lights up (Phase G/H).
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
# 2026-06-25 — a rung BELOW ``hardware_verified`` for kernels that EXECUTE on
# real hardware via ``runtime.launch()`` but ship as a COMPILER-GENERATED hsaco
# (tessera-opt → ROCDL → hsaco, loaded + launched in-process), NOT as a standalone
# C-ABI ``runtime_symbol`` in a shipped ``.so``. It still requires a checked-in
# ``execute_compare_fixture`` (the numerical proof) — only the C-symbol half of
# the ``hardware_verified`` contract is absent. This is the honest status for the
# ROCm compiled-lane attention/epilogue family (rocm_*_compiled executors): the
# kernel is real and verified on gfx1151, but there is no shipped C entry point.
_COMPILED_STATUS = "compiled"
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
    _COMPILED_STATUS,
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
            for under the pinned toolchain (CUDA 13.3 / ROCm 7.2.4).
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
        Minimum hipcc release.  Today all ROCm entries pin to ``"7.2.4"``.
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

        # 2026-06-25 — ``compiled`` contract. The kernel executes on hardware via
        # runtime.launch() as a compiler-generated hsaco (no shipped C-ABI
        # symbol), so unlike ``hardware_verified`` it does NOT require a
        # ``runtime_symbol`` — but it MUST still carry the numerical-proof
        # ``execute_compare_fixture`` (the kernel really runs + matches a
        # reference), else it's no better than ``compileable``.
        if self.status == _COMPILED_STATUS and not self.execute_compare_fixture:
            raise ValueError(
                f"status='compiled' requires execute_compare_fixture to point "
                f"at a Python test that numerically validates the "
                f"compiler-generated kernel on hardware; got "
                f"target={self.target!r}"
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
    that the V8 Phase G/H audit doc surfaced.

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

_APPLE_GPU_STRUCTURED_COMPUTE_FIXTURE = (
    "tests/unit/test_apple_gpu_structured_compute_compiled.py")

_APPLE_GPU_KERNELS: dict[str, dict[str, Any]] = {
    "matmul": {
        "status": _FUSED_KERNEL_STATUS,
        "dtypes": _APPLE_GPU_FUSED,
        "notes": "MPSMatrixMultiplication + bf16 conversion path",
        "benchmark_json": "benchmarks/baselines/apple_gpu_hot_paths.json",
    },
    # Structured-compute convolution family (2026-07-09) — parity with the
    # x86/ROCm structured-compute tails. These reach an executable apple_gpu
    # path via apple_gpu_structured_compute_compiled and match the reference
    # primitive; host-structured im2col/layout bookkeeping. ``compiled`` (direct
    # execute/compare evidence), NOT a bespoke fused Metal kernel.
    "conv1d": {
        "status": _COMPILED_STATUS,
        "dtypes": ("fp32",),
        "notes": ("Structured-compute conv1d via "
                  "apple_gpu_structured_compute_compiled; matches "
                  "tessera.nn.functional.conv1d."),
        "execute_compare_fixture": _APPLE_GPU_STRUCTURED_COMPUTE_FIXTURE,
    },
    "conv_transpose": {
        "status": _COMPILED_STATUS,
        "dtypes": ("fp32",),
        "notes": ("Structured-compute conv_transpose via "
                  "apple_gpu_structured_compute_compiled; matches "
                  "tessera.nn.functional.conv_transpose."),
        "execute_compare_fixture": _APPLE_GPU_STRUCTURED_COMPUTE_FIXTURE,
    },
    "depthwise_conv1d": {
        "status": _COMPILED_STATUS,
        "dtypes": ("fp32",),
        "notes": ("Structured-compute depthwise_conv1d via "
                  "apple_gpu_structured_compute_compiled; matches "
                  "tessera.ops.depthwise_conv1d."),
        "execute_compare_fixture": _APPLE_GPU_STRUCTURED_COMPUTE_FIXTURE,
    },
    # Pointwise-regression loss lane (2026-07-09) — parity with the x86/ROCm
    # loss lanes. Residual + none/mean/sum reduction on the MPSGraph binary +
    # reduce lanes (mse/mae also on the GPU mul/abs opcodes; huber/smooth_l1/
    # log_cosh apply the piecewise/transcendental middle host-side). ``compiled``
    # (direct execute/compare vs tessera.losses), NOT a bespoke fused kernel.
    **{op: {
        "status": _COMPILED_STATUS,
        "dtypes": ("fp32",),
        "notes": (f"Pointwise-regression loss {op} via apple_gpu_loss_compiled "
                  "(MPSGraph binary + reduce lanes, host piecewise middle); "
                  "matches tessera.losses."),
        "execute_compare_fixture": "tests/unit/test_apple_gpu_loss_compiled.py",
    } for op in ("mse_loss", "mae_loss", "huber_loss", "smooth_l1_loss",
                 "log_cosh_loss")},
    # Loss-family lane (2026-07-09) — binary-CE / class-axis / RL-policy /
    # EBM-diffusion. Per-sample loss via the standalone reference (host
    # structure); none/mean/sum reduction on the MPSGraph reduce lane. Parity
    # with the x86/ROCm binary/class/rl/ebm loss lanes. ``compiled`` (direct
    # execute/compare vs tessera.losses / tessera.rl), NOT a bespoke fused kernel.
    **{op: {
        "status": _COMPILED_STATUS,
        "dtypes": ("fp32",),
        "notes": (f"Loss-family {op} via apple_gpu_loss_family_compiled "
                  "(reference per-sample loss + MPSGraph reduce lane); matches "
                  "tessera.losses / tessera.rl."),
        "execute_compare_fixture": "tests/unit/test_apple_gpu_loss_family_compiled.py",
    } for op in ("binary_cross_entropy_loss",
                 "cross_entropy_loss", "kl_divergence", "js_divergence", "z_loss",
                 "ppo_policy_loss", "cispo_policy_loss", "grpo_policy_loss",
                 "score_matching_loss", "denoising_score_matching_loss",
                 "implicit_score_matching_loss", "contrastive_divergence_loss",
                 "persistent_cd_loss", "ddpm_noise_pred_loss", "vlb_loss",
                 "load_balance_loss")},
    # ctc_loss + edm_loss_weight ride the existing apple_gpu structured-compute
    # lane (they are in _SINGLE_GPU_COMPUTE_REFERENCE_OPS -> the structured
    # manifest path + apple_gpu_structured_compute_compiled executor).
    **{op: {
        "status": _COMPILED_STATUS,
        "dtypes": ("fp32",),
        "notes": (f"Structured-compute loss/schedule {op} via "
                  "apple_gpu_structured_compute_compiled; matches "
                  "tessera.losses / diffusion_schedule."),
        "execute_compare_fixture": _APPLE_GPU_STRUCTURED_COMPUTE_FIXTURE,
    } for op in ("ctc_loss", "edm_loss_weight")},
    # Structured-compute tail (2026-07-09) — vision/layout transforms, recurrent
    # cells, MoR routing, VLM resamplers, RoPE split/merge, and other
    # host-structured primitives. All reach an executable apple_gpu path via
    # apple_gpu_structured_compute_compiled and match the reference (ops.* /
    # nn.functional.* / memory.*). ``compiled`` (direct execute/compare), NOT a
    # bespoke fused Metal kernel — parity with the x86/ROCm structured-compute
    # lanes.
    **{op: {
        "status": _COMPILED_STATUS,
        "dtypes": ("fp32",),
        "notes": (f"Structured-compute {op} via "
                  "apple_gpu_structured_compute_compiled; matches the "
                  "tessera reference primitive."),
        "execute_compare_fixture": _APPLE_GPU_STRUCTURED_COMPUTE_FIXTURE,
    } for op in (
        "arange", "bidirectional_scan", "center_crop", "cross_attention",
        "edm_precondition", "factorized_pos_emb", "gru_cell", "lstm_cell", "image_resize",
        "interpolate", "lora_linear", "masked_fill", "masked_scatter",
        "memory_read", "mor_partition", "mor_router", "mor_scatter", "mrope_2d",
        "online_softmax_state", "pack", "patchify", "perceiver_resampler",
        "pixel_shuffle", "pixel_unshuffle", "rearrange", "rope_merge",
        "rope_split", "simple_rnn_cell", "spectral_norm", "tile_view", "unpack")},
    # Conformal-geometry lane (2026-07-09) — mobius f(z)=(az+b)/(cz+d) composed
    # on the interleaved-f32 Apple GPU complex_mul/complex_div lanes. ``compiled``
    # (direct execute/compare vs tessera.complex), parity with x86/rocm.
    "mobius": {
        "status": _COMPILED_STATUS,
        "dtypes": ("fp32",),
        "notes": ("Conformal mobius via apple_gpu_conformal_compiled "
                  "(interleaved-f32 complex_mul/complex_div lanes); matches "
                  "tessera.complex."),
        "execute_compare_fixture": "tests/unit/test_apple_gpu_complex_compiled.py",
    },
    # Optimizer lane (2026-07-10) — sgd/momentum/adam/adamw/lion per-parameter
    # update via apple_gpu_optimizer_compiled. Apple ships no device optimizer
    # kernel; the elementwise update rules run on the numpy reference the x86/ROCm
    # device kernels are matched against. ``compiled`` (direct execute/compare),
    # NOT a bespoke fused Metal kernel.
    **{op: {
        "status": _COMPILED_STATUS,
        "dtypes": ("fp32",),
        "notes": (f"Optimizer {op} via apple_gpu_optimizer_compiled (numpy "
                  "reference update rules); matches tessera.optim."),
        "execute_compare_fixture": "tests/unit/test_apple_gpu_optimizer_compiled.py",
    } for op in ("sgd", "momentum", "adam", "adamw", "lion")},
    # 0-move + sort lane (2026-07-10) — pad/roll/flip/tile/repeat/stack +
    # sort/argsort via apple_gpu_shape_compiled (host index-map + numpy gather /
    # numpy stable sort; Apple ships no device gather/sort kernel). ``compiled``
    # (direct execute/compare), NOT a bespoke fused Metal kernel.
    **{op: {
        "status": _COMPILED_STATUS,
        "dtypes": ("fp32",),
        "notes": (f"0-move/sort {op} via apple_gpu_shape_compiled (numpy gather / "
                  "stable sort reference); matches tessera.ops / numpy."),
        "execute_compare_fixture": "tests/unit/test_apple_gpu_shape_compiled.py",
    } for op in ("pad", "roll", "flip", "tile", "repeat", "stack",
                 "sort", "argsort")},
    # Reduce lane (2026-07-10) — sum genuinely on the MPSGraph reduce lane
    # (apple_gpu_reduce_compiled; numpy fallback when Metal is unavailable).
    "sum": {
        "status": _COMPILED_STATUS,
        "dtypes": ("fp32",),
        "notes": ("Reduce sum via apple_gpu_reduce_compiled (MPSGraph reduce "
                  "lane); matches numpy.sum."),
        "execute_compare_fixture": "tests/unit/test_apple_gpu_reduce_compiled.py",
    },
    # Scatter lane (2026-07-10) — scatter/scatter_add/scatter_reduce via
    # apple_gpu_scatter_compiled (numpy indexed store; Apple ships no device
    # scatter kernel). ``compiled`` (direct execute/compare).
    **{op: {
        "status": _COMPILED_STATUS,
        "dtypes": ("fp32",),
        "notes": (f"Scatter {op} via apple_gpu_scatter_compiled (numpy indexed "
                  "store reference); matches numpy scatter."),
        "execute_compare_fixture": "tests/unit/test_apple_gpu_scatter_compiled.py",
    } for op in ("scatter", "scatter_add", "scatter_reduce")},
    # Sparse + MoE lane (2026-07-10) — spmm_csr/spmm_coo/sddmm/bsmm/moe via
    # apple_gpu_sparse_compiled (numpy CSR SpMM / (a@b)*mask / a@b / routed
    # per-token expert GEMVs; Apple ships no device sparse/moe kernel).
    **{op: {
        "status": _COMPILED_STATUS,
        "dtypes": ("fp32",),
        "notes": (f"Sparse/MoE {op} via apple_gpu_sparse_compiled (numpy "
                  "reference); matches numpy / tessera."),
        "execute_compare_fixture": "tests/unit/test_apple_gpu_sparse_compiled.py",
    } for op in ("spmm_csr", "spmm_coo", "sddmm", "bsmm", "moe")},
    # Reference tail lane (2026-07-10) — the heterogeneous remainder (MLA
    # latent-KV, alibi, lgamma/digamma, fused_epilogue, asymmetric_bce,
    # normalize_group_advantages, speculative-decode accept) via
    # apple_gpu_tail_compiled (public tessera reference; Apple ships no device
    # kernel). ``compiled`` (direct execute/compare).
    **{op: {
        "status": _COMPILED_STATUS,
        "dtypes": ("fp32",),
        "notes": (f"Reference tail {op} via apple_gpu_tail_compiled (public "
                  "tessera reference); matches tessera.ops / losses / rl."),
        "execute_compare_fixture": "tests/unit/test_apple_gpu_tail_compiled.py",
    } for op in ("latent_kv_compress", "latent_kv_expand_k", "latent_kv_expand_v",
                 "alibi", "lgamma", "digamma", "fused_epilogue", "asymmetric_bce",
                 "normalize_group_advantages", "spec_accept",
                 "spec_accept_sample", "spec_accept_tree_sample")},
    # stereographic rides the existing apple_gpu_conformal_compiled lane
    # (_conformal_compute handles mobius + stereographic; genuine composition on
    # the interleaved-f32 complex/binary-div lanes -> native_gpu).
    "stereographic": {
        "status": _COMPILED_STATUS,
        "dtypes": ("fp32",),
        "notes": ("Conformal stereographic via apple_gpu_conformal_compiled "
                  "(sphere 3-vector -> C on the binary-div lane); matches "
                  "tessera.complex."),
        "execute_compare_fixture": "tests/unit/test_apple_gpu_complex_compiled.py",
    },
    # Philox RNG base lane (2026-07-10) — rng_uniform / rng_normal / dropout via
    # apple_gpu_rng_compiled. Apple ships no device Philox kernel; the lane draws
    # from the counter-based Philox-4x32-10 reference (tessera.rng_device) the
    # x86/ROCm device kernels are bit-matched against. ``compiled`` (direct
    # execute/compare), NOT a bespoke fused Metal kernel.
    **{op: {
        "status": _COMPILED_STATUS,
        "dtypes": dts,
        "notes": (f"Philox RNG {op} via apple_gpu_rng_compiled "
                  "(Philox-4x32-10 reference core); matches "
                  "tessera.rng_device."),
        "execute_compare_fixture": "tests/unit/test_apple_gpu_rng_compiled.py",
    } for op, dts in (("rng_uniform", ("fp32",)), ("rng_normal", ("fp32",)),
                      ("dropout", ("fp32",)))},
    # Linalg decomposition lane (2026-07-10) — cholesky_solve/lu/qr/svd via
    # apple_gpu_linalg_compiled. Apple ships no MPS lu/qr/svd primitive; the
    # decompositions resolve on the numpy reference (np.linalg + a standalone
    # partial-pivot LU) the x86/ROCm device kernels match. ``compiled`` (direct
    # execute/compare), NOT a bespoke fused Metal kernel.
    **{op: {
        "status": _COMPILED_STATUS,
        "dtypes": ("fp32",),
        "notes": (f"Linalg {op} via apple_gpu_linalg_compiled (numpy reference; "
                  "no MPS lu/qr/svd primitive); matches np.linalg."),
        "execute_compare_fixture": "tests/unit/test_apple_gpu_linalg_compiled.py",
    } for op in ("cholesky_solve", "lu", "qr", "svd")},
    # Matmul-family lane (2026-07-10) — einsum / factorized_matmul via
    # apple_gpu_matmul_family_compiled (numpy reference the GEMM lanes match).
    **{op: {
        "status": _COMPILED_STATUS,
        "dtypes": ("fp32",),
        "notes": (f"Matmul-family {op} via apple_gpu_matmul_family_compiled "
                  "(numpy reference); matches numpy."),
        "execute_compare_fixture": "tests/unit/test_apple_gpu_matmul_family_compiled.py",
    } for op in ("einsum", "factorized_matmul")},
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
    "attn_compressed_blocks": {
        "status": _FUSED_KERNEL_STATUS,
        "dtypes": ("fp32",),
        "notes": "NSA compressed-block attention: QKᵀ / softmax / PV execute "
                 "on the Apple GPU batched-attention lane over compressed K/V.",
        "shape_envelope": "Q,K,V rank-4 [B,H,S,D], fp32 attention core",
    },
    "attn_top_k_blocks": {
        "status": _FUSED_KERNEL_STATUS,
        "dtypes": ("fp32",),
        "notes": "NSA top-k block attention: host top-k/gather selects blocks; "
                 "per-query dense attention FLOPs execute on Apple GPU.",
        "shape_envelope": "rank-4 Q/K/V; S_k divisible by block_size",
    },
    "attn_local_window_2d": {
        "status": _FUSED_KERNEL_STATUS,
        "dtypes": ("fp32",),
        "notes": "2D local-window attention: structured host im2col gather + "
                 "Apple GPU bmm/add-mask/softmax/bmm attention core.",
        "shape_envelope": "rank-5 Q/K/V [B,H,Hq,Wq,D], non-negative window",
    },
    "lookahead_sparse_attention": {
        "status": _FUSED_KERNEL_STATUS,
        "dtypes": ("fp32",),
        "notes": "Lookahead sparse attention: host data-dependent footprint "
                 "selection with Apple GPU per-footprint attention; uses fused "
                 "MSL single-dispatch path when the symbol/envelope is available.",
        "shape_envelope": "rank-4 Q/K/V; positive window/block/tau; footprint <= 256 for fused path",
    },
    "msa_sparse_attention": {
        "status": _FUSED_KERNEL_STATUS,
        "dtypes": ("fp32", "fp16"),
        "notes": "MiniMax Sparse Attention on Apple GPU: index/select may run on "
                 "host or GPU selector; main sparse attention uses native fused "
                 "block-sparse MSL when D<=256, otherwise composed GPU bmm lane.",
        "shape_envelope": "rank-4 Q/K/V; S_k divisible by block_size; D<=256 native fast path",
    },
    "linear_attn_state": {
        "status": _FUSED_KERNEL_STATUS,
        "dtypes": ("fp32",),
        "notes": "Linear-attention recurrent state via quadratic-parallel "
                 "Apple GPU batched matmul path with structured host masks.",
        "shape_envelope": "rank-4 Q/K/V [B,H,S,D], fp32 state [B,H,D,D]",
    },
    "memory_index_score": {
        "status": _FUSED_KERNEL_STATUS,
        "dtypes": ("fp32",),
        "notes": "LSA memory-index scoring: Apple GPU batched matmul + sigmoid "
                 "composition; host only supplies scale/shape metadata.",
        "shape_envelope": "indexer_keys/query rank-4 [B,H,nb|Sq,Dk]",
    },
    "msa_index_scores": {
        "status": _FUSED_KERNEL_STATUS,
        "dtypes": ("fp32",),
        "notes": "MSA index-branch scoring: host block/head mean reductions plus "
                 "Apple GPU batched matmul over grouped queries and block keys.",
        "shape_envelope": "rank-4 Q/K; Hq divisible by Hkv; positive block_size",
    },
    "varlen_sdpa": {
        "status": _FUSED_KERNEL_STATUS,
        "dtypes": ("fp32",),
        "notes": "Packed-sequence SDPA decomposes to per-segment Apple GPU "
                 "flash-attention calls with host cu_seqlens metadata.",
        "shape_envelope": "rank-3 packed [H,total,Dh] Q/K/V and monotonic cu_seqlens",
    },
    "score_combine": {
        "status": _FUSED_KERNEL_STATUS,
        "dtypes": ("fp32",),
        "notes": "Diffusion score composition base + gamma*delta via Apple GPU "
                 "binary elementwise multiply/add lanes.",
        "shape_envelope": "base/delta identical shape, scalar gamma",
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
    "transpose": {
        "status": _FUSED_KERNEL_STATUS,
        "dtypes": _APPLE_GPU_FUSED,
        "notes": (
            "MPSGraph N-D permute lane "
            "(tessera_apple_gpu_mpsgraph_transpose_{f32,f16}; bf16 rides the "
            "raw f16 path). Value-preserving data movement."
        ),
    },
    "gather": {
        "status": _FUSED_KERNEL_STATUS,
        "dtypes": _APPLE_GPU_FUSED,
        "notes": (
            "MPSGraph axis-0 row gather lane "
            "(tessera_apple_gpu_mpsgraph_gather_{f32,f16}; 2D table + int32 "
            "indices; bf16 rides the raw f16 path)."
        ),
    },
    "slice": {
        "status": _FUSED_KERNEL_STATUS,
        "dtypes": _APPLE_GPU_FUSED,
        "notes": (
            "MPSGraph static N-D slice lane "
            "(tessera_apple_gpu_mpsgraph_slice_{f32,f16}; starts/sizes attrs, "
            "stride 1; bf16 rides the raw f16 path)."
        ),
    },
    "cast": {
        "status": _FUSED_KERNEL_STATUS,
        "dtypes": _APPLE_GPU_FUSED,
        "notes": (
            "MPSGraph dtype-conversion lane used by encode-session bf16/f16 "
            "bridges; bf16 rides raw f16 storage with fp32 internal compute."
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
    "quantized_matmul": {
        "status": _HARDWARE_VERIFIED_STATUS,
        "dtypes": ("fp32", "fp16"),
        "notes": (
            "Packed INT4 quantized matmul Apple GPU lane: i4 weights + "
            "per-group affine scale/bias dequantized in-register, f32 "
            "accumulate. Covers untiled f32, f16 activation upload, tiled, "
            "and split-K variants through the quant_matmul runtime lane."
        ),
        "runtime_symbol": "tessera_apple_gpu_quantized_matmul_i4_f32",
        "shape_envelope": (
            "x (M,K), packed weights (N,ceil(K/2)), scales/biases "
            "(N,ceil(K/group_size)); O (M,N) fp32"
        ),
    },
    "masked_categorical": {
        "status": _FUSED_KERNEL_STATUS,
        "dtypes": ("fp32", "int32"),
        "notes": "Masked categorical greedy/select path via Apple GPU MPSGraph "
                 "select + argmax subgraph; stochastic-key path remains reference.",
        "shape_envelope": "rank-2 logits/mask; keyless greedy path",
    },
}


# ─────────────────────────────────────────────────────────────────────────────
# Strix Halo bring-up (2026-06-22) — ROCm ops promoted to ``hardware_verified``.
#
# An op lands here only once it ships BOTH a real C-ABI ``runtime_symbol`` (an
# auto-built backend lib that actually runs the kernel on an AMD GPU) AND a
# checked-in ``execute_compare_fixture`` (see ``_NUMERICAL_FIXTURES``). This is
# the ROCm analog of the per-op ``_APPLE_GPU_KERNELS`` hardware_verified rows.
#
# When an op is in this table the generic ROCm artifact_only row is REPLACED by
# the hardware_verified row below — there is exactly one ``rocm`` entry per op.
#
# Honesty (Decision #25): ``dtypes``/``shape_envelope`` describe ONLY what the
# shipped symbol actually proves. matmul is a general tiled/K-looped RDNA
# **WMMA** GEMM, f32<-{f16,bf16} (ragged M/N/K zero-padded), and carries no MFMA
# shape (WMMA, not the CDNA matrix-core path). It also backs the
# runtime.launch() ``rocm_wmma`` execution lane.
# ─────────────────────────────────────────────────────────────────────────────
_ROCM_HARDWARE_VERIFIED: dict[str, dict[str, Any]] = {
    "matmul": {
        "runtime_symbol": "tessera_rocm_wmma_gemm_f16",
        "dtypes": ("fp16", "bf16"),
        "feature_flags": ("wmma",),
        "shape_envelope": (
            "general tiled/K-looped GEMM, f32<-{f16,bf16}, any positive M/N/K "
            "(ragged edges zero-padded), 16x16x16 WMMA tiles, size-adaptive "
            "register macro-tile (2x4 small / 3x4 large — the occupancy lever) "
            "(gfx1151 Strix Halo / gfx1100 WSL enumeration)"
        ),
        # E2 (2026-07-06) — WMMA matmul ladder ratchet baseline, recorded on the
        # gfx1151 box (real medians, repo Decision #26).
        "benchmark_json": "benchmarks/baselines/rocm_gfx1151_hot_paths.json",
        "notes": (
            "RDNA 3.5 WMMA matrix-core GEMM, f32<-{f16,bf16}, ROCm 7.2.4. "
            "DEFAULT execution lane (Stage L4): @jit(target='rocm') matmul runs "
            "the COMPILER-GENERATED WMMA kernel — tessera-opt generates + "
            "serializes it to hsaco in-process (no mlir-opt), then HIP launches "
            "it (runtime.launch() rocm_compiled lane); reaches parity-or-better "
            "vs the hand-written kernel across aligned/ragged/f16/bf16. The "
            "shipped hand-written libtessera_rocm_gemm.so symbol "
            "(tessera_rocm_wmma_gemm_f16, the runtime_symbol above) is the "
            "reference ORACLE the compiled kernel is checked bit-identical "
            "against AND the availability fallback when the compiled lane can't "
            "run on a host. Numerically validated vs numpy by the "
            "execute_compare_fixture; the compiled lane is validated vs the "
            "oracle by tests/unit/test_rocm_compiled_launch_execute.py."
        ),
    },
    # `gemm` is the BLAS-vocabulary matmul (same tessera.matmul op + the same
    # tessera_rocm_wmma_gemm_f16 symbol matmul uses; rt._execute_rocm_compiled_gemm
    # executes it). It is the matrix-core GEMM across BOTH AMD families — so unlike
    # `matmul` (kept a pure RDNA-WMMA row) it ALSO carries the CDNA MFMA artifact
    # shape as metadata. Status = hardware_verified for the RDNA WMMA execution
    # proven on gfx1151; RDNA4 (gfx1200) / gfx12.5 (gfx1250) WMMA + CDNA MFMA are
    # the arch targets, gated on their fragment-layout ISA + silicon (the RDNA3
    # V_WMMA_16x16x16 intrinsic does not select on gfx12 — RDNA4 uses 16x16x32).
    "gemm": {
        "runtime_symbol": "tessera_rocm_wmma_gemm_f16",
        "dtypes": ("fp16", "bf16"),
        "feature_flags": ("wmma", "mfma"),
        "mfma_shape": (32, 32, 8, 1),
        "shape_envelope": (
            "matrix-core GEMM, f32<-{f16,bf16}. RDNA WMMA 16x16x16 "
            "hardware-verified on gfx1151 (RDNA3.5); RDNA4 gfx1200 (WMMA "
            "16x16x32) + gfx12.5 gfx1250 (large-K WMMA) + CDNA MFMA (32x32x8) "
            "are arch targets gated on their fragment-layout ISA + silicon"
        ),
        # No manifest `benchmark_json` — gemm's dedicated perf harness is
        # benchmarks/benchmark_gemm.py (more specific than the shared matmul
        # hot-paths baseline), which the benchmark-source router prefers.
        "notes": (
            "GEMM = the matrix-core matmul (alias of tessera.matmul, same WMMA "
            "symbol). RDNA WMMA execution hardware-verified on gfx1151; carries "
            "the CDNA MFMA artifact shape (32x32x8) for the datacenter target. "
            "RDNA4/gfx12.5 WMMA + CDNA execution are hardware-gated."
        ),
    },
    "flash_attn": {
        "runtime_symbol": "tessera_rocm_wmma_flash_attn_f16",
        "dtypes": ("fp16", "bf16"),
        "feature_flags": ("wmma",),
        "shape_envelope": (
            "FA-2 forward, f32<-{f16,bf16}, Q[B,H,Sq,D] x K/V[B,H,Sk,D], "
            "head_dim D multiple of 16, any positive B/H/Sq/Sk (ragged Sq/Sk "
            "zero-padded + masked), optional causal mask, both QK^T and P@V on "
            "16x16x16 WMMA, online softmax, one wave per (query-tile, b*h). "
            "Correctness-first (no perf ladder yet)"
        ),
        # E2 (2026-07-06) — compiled FA-2 flash_attn ladder ratchet baseline
        # (rt._rocm_flash_attn), recorded on the gfx1151 box alongside matmul.
        "benchmark_json": "benchmarks/baselines/rocm_gfx1151_hot_paths.json",
        "notes": (
            "RDNA 3.5 WMMA flash-attention forward executes on the AMD GPU "
            "through the shipped libtessera_rocm_flash_attn.so symbols "
            "(tessera_rocm_wmma_flash_attn_{f16,bf16}, HIPRTC-compiled for the "
            "device arch at load); ROCm 7.2.4. The second op after matmul to run "
            "natively on a non-Apple backend. Numerically validated vs a numpy "
            "attention reference by the execute_compare_fixture. The FA-2 "
            "BACKWARD (dQ/dK/dV) also executes on gfx1151 via the "
            "compiler-generated rocm_flash_attn_bwd_compiled lane "
            "(generate-wmma-flash-attn-bwd-kernel -> fa_pre/fa_dkdv/fa_dq; MHA + "
            "GQA/MQA + additive attn_bias + sliding-window + logit-softcap, "
            "scale+causal), validated vs autodiff vjp_flash_attn — see the "
            "runtime_execution_matrix."
        ),
    },
}


# ─────────────────────────────────────────────────────────────────────────────
# Compiler-generated executing lane (2026-06-25) — ROCm ops promoted to
# ``compiled``. These execute on gfx1151 via ``runtime.launch()`` as a
# COMPILER-GENERATED hsaco (tessera-opt → ROCDL → hsaco, loaded + launched
# in-process), each with a checked-in ``execute_compare_fixture`` — but NO
# shipped C-ABI ``runtime_symbol``, so they are a rung below ``hardware_verified``
# (which matmul/flash_attn earn via their shipped libtessera_rocm_*.so symbols).
#
# Honesty (Decision #25): these are the compiled-lane CAPABILITIES proven on the
# box. The flash_attn-family rows (gqa/mqa/mha/sliding-window) are realized by the
# flash_attn kernel with directive attrs (+ the runtime detects/forwards them);
# fused_epilogue is the matmul kernel + a fused bias/activation epilogue; the
# linear-attn family (linear_attn/lightning_attention/retention) is the dedicated
# linear-attn kernel dispatched by op name. Soft-capping is a flash_attn attr with
# no standalone op row, so it is covered by the flash_attn row, not listed here.
# ─────────────────────────────────────────────────────────────────────────────
_ROCM_COMPILED: dict[str, dict[str, Any]] = {
    "spec_accept": {
        "dtypes": ("int32",),
        "feature_flags": ("control_flow", "speculative_decode"),
        "notes": "Speculative-decoding accept mask kernel — compiler-generated "
                 "ROCm control-flow lane (generate-rocm-spec-accept-kernel). "
                 "Executes via runtime.launch() (rocm_spec_accept_compiled).",
    },
    "spec_accept_sample": {
        "dtypes": ("int32", "fp32"),
        "feature_flags": ("control_flow", "speculative_decode", "explicit_rng"),
        "notes": "Speculative-decoding accept+sample kernel — compares proposal "
                 "and target probabilities, emits accept flags plus sampled "
                 "fallback ids via the explicit RNG stream. Executes via "
                 "runtime.launch() (rocm_spec_accept_sample_compiled).",
    },
    "spec_accept_tree_sample": {
        "dtypes": ("fp32",),
        "feature_flags": ("control_flow", "speculative_decode", "explicit_rng"),
        "notes": "Tree speculative-decoding sampler — compiler-generated ROCm "
                 "lane for per-node probabilities and parent links. Executes via "
                 "runtime.launch() (rocm_spec_accept_tree_sample_compiled).",
    },
    "gqa_attention": {
        "dtypes": ("fp16", "bf16"),
        "notes": "GQA/MQA via the flash_attn WMMA kernel (gqa directive attr; "
                 "fwd+bwd, grouped K/V; runtime detects from operand shapes). "
                 "Executes on gfx1151 via runtime.launch() (rocm_flash_attn_"
                 "compiled); no shipped C-ABI symbol.",
    },
    "mqa_attention": {
        "dtypes": ("fp16", "bf16"),
        "notes": "Multi-query attention = the GQA flash_attn kernel at "
                 "kv_ratio=H (one shared KV head). Executes via runtime.launch().",
    },
    "multi_head_attention": {
        "dtypes": ("fp16", "bf16"),
        "notes": "Full multi-head = the flash_attn WMMA kernel itself. Executes "
                 "via runtime.launch() (rocm_flash_attn_compiled).",
    },
    "attn_sliding_window": {
        "dtypes": ("fp16", "bf16"),
        "notes": "Mistral sliding-window via the flash_attn WMMA kernel "
                 "(sliding_window attr, causal band of width W; KV-tile skip). "
                 "Executes via runtime.launch() (window kwarg).",
    },
    "linear_attn": {
        "dtypes": ("fp16", "bf16"),
        "notes": "Quadratic-parallel linear attention O=(φ(Q)φ(K)ᵀ⊙causal)@V, no "
                 "softmax (dedicated generate-wmma-linear-attn-kernel). φ ∈ "
                 "{identity,relu,polynomial_2}. Executes via runtime.launch() "
                 "(rocm_linear_attn_compiled).",
    },
    "lightning_attention": {
        "dtypes": ("fp16", "bf16"),
        "notes": "Decay-masked linear attention (φ=identity, A[i,j]*=λ^(i-j)) via "
                 "the linear-attn kernel, dispatched by op name. Executes via "
                 "runtime.launch(). (Degree-2 retention — φ=x² + decay — shares "
                 "this kernel via op-name dispatch but has no separate rocm op "
                 "row.)",
    },
    "retention": {
        "dtypes": ("fp16", "bf16"),
        "notes": "Degree-2 retention (φ=x² plus optional decay) via the "
                 "linear-attn kernel, dispatched by op name. Executes via "
                 "runtime.launch() (rocm_linear_attn_compiled).",
    },
    "fused_epilogue": {
        "dtypes": ("fp16", "bf16"),
        "notes": "Matmul + fused bias/relu/gelu/silu epilogue on the f32 "
                 "accumulator (generate-wmma-gemm-kernel). Executes via "
                 "runtime.launch() (rocm_compiled + activation kwarg); float-only.",
    },
    "softmax": {
        "dtypes": ("fp32", "fp16", "bf16"),
        "feature_flags": ("reduction",),
        "notes": "Row-wise stable softmax over the last axis — the first "
                 "non-matmul/non-WMMA compiled ROCm kernel "
                 "(generate-rocm-softmax-kernel: one workgroup per row, LDS "
                 "tree-reduce, f32 reduce). Executes via runtime.launch() "
                 "(rocm_softmax_compiled).",
    },
    "online_softmax": {
        "dtypes": ("fp32", "fp16", "bf16"),
        "feature_flags": ("reduction",),
        "notes": "Stateless online_softmax (== softmax over the last axis) rides "
                 "the compiled softmax kernel (generate-rocm-softmax-kernel) via "
                 "runtime.launch() (rocm_softmax_compiled); the streaming-state "
                 "form is declined (Decision #21).",
    },
    # KV-cache paged-movement core (§5.6). The append/read/prune tensor movement
    # over a resident cache buffer executes on gfx1151 by COMPOSING the existing
    # device scatter (append row write) + masked-gather (read/prune) kernels with
    # host page-index math — no bespoke kernel. Mirrors KVCacheHandle.{append,
    # read,prune} on the non-quantized fp path; quantize_kv rides the intquant
    # lane. Executes via runtime.launch() (rocm_kv_cache_compiled). f32.
    **{op: {
        "dtypes": ("fp32",),
        "feature_flags": ("kv_cache", "paged"),
        "notes": f"KV-cache {op.split('_')[-1]} paged-movement core over a "
                 "resident cache buffer (max_seq, H, D): append = row scatter-"
                 "write at [start, start+n) (generate-rocm-scatter-kernel, set "
                 "mode); read = row gather of [start, end); prune = trailing-"
                 "window gather + zero-fill (generate-rocm-gather-kernel). Host "
                 "owns the page-index math; quantize_kv rides the intquant lane. "
                 "Executes via runtime.launch() (rocm_kv_cache_compiled). f32, "
                 "matches the KVCacheHandle reference.",
    } for op in ("kv_cache_append", "kv_cache_read", "kv_cache_prune")},
    **{op: {
        "dtypes": ("fp32", "fp16", "bf16"),
        "feature_flags": ("reduction",),
        "notes": f"Row {op} reduction over the last axis (generate-rocm-reduce-"
                 "kernel: one workgroup per row, LDS tree-reduce, f32 reduce) — "
                 "the ROCm analog of the x86 AVX-512 reduction lane. Executes "
                 "via runtime.launch() (rocm_reduce_compiled).",
    } for op in ("sum", "mean", "max", "min", "amax", "amin")},
    # argmax/argmin — CUB ArgMax-style warp-shuffle arg-reduce (value+index pair),
    # i32 index output. Executes via runtime.launch() (rocm_argreduce_compiled).
    **{op: {
        "dtypes": ("fp32", "fp16", "bf16"),
        "feature_flags": ("reduction",),
        "notes": f"Row {op} along one axis (generate-rocm-argreduce-kernel: "
                 "warp-shuffle (value,index) reduce, first-occurrence tie-break, "
                 "i32 output). Executes via runtime.launch() "
                 "(rocm_argreduce_compiled).",
    } for op in ("argmax", "argmin")},
    # cumsum/cumprod/cummax/cummin — CUB BlockScan (gpu.shuffle up Kogge-Stone +
    # cross-tile carry); inclusive prefix, same-shape output. Executes via
    # runtime.launch() (rocm_scan_compiled).
    **{op: {
        "dtypes": ("fp32", "fp16", "bf16"),
        "feature_flags": ("scan",),
        "notes": f"Row inclusive scan {op} (generate-rocm-scan-kernel: block-scan "
                 "via gpu.shuffle up + cross-tile carry, f32 scan). Executes via "
                 "runtime.launch() (rocm_scan_compiled).",
    } for op in ("cumsum", "cumprod", "cummax", "cummin")},
    "rmsnorm": {
        "dtypes": ("fp32", "fp16", "bf16"),
        "feature_flags": ("reduction",),
        "notes": "Unweighted row rmsnorm over the last axis (row-reduction "
                 "kernel, generate-rocm-norm-kernel): x / sqrt(mean(x²) + eps). "
                 "Executes via runtime.launch() (rocm_norm_compiled).",
    },
    "layer_norm": {
        "dtypes": ("fp32", "fp16", "bf16"),
        "feature_flags": ("reduction",),
        "notes": "Unweighted row layer_norm over the last axis (row-reduction "
                 "kernel, generate-rocm-norm-kernel): (x − μ) / sqrt(var + eps). "
                 "Executes via runtime.launch() (rocm_norm_compiled).",
    },
    "rmsnorm_safe": {
        "dtypes": ("fp32", "fp16", "bf16"),
        "feature_flags": ("reduction",),
        "notes": "Unweighted row rmsnorm_safe over the last axis (row-reduction "
                 "kernel, generate-rocm-norm-kernel) with the safe default "
                 "eps=1e-6. Executes via runtime.launch() (rocm_norm_compiled).",
    },
    # P5 — group/instance/weight norm composed on the gfx1151 layer_norm (row
    # mean/var) + reduce (sum-of-squares) lanes; host does the reshape / divide.
    **{op: {
        "dtypes": ("fp32",),
        "feature_flags": ("normalization",),
        "notes": f"Standalone {op} — composed on the gfx1151 layer_norm / reduce "
                 "kernels (no new kernel; host reshape/affine). Executes via "
                 "runtime.launch() (rocm_normcompose_compiled).",
    } for op in ("group_norm", "instance_norm", "weight_norm")},
    "grad_clip_norm": {
        "dtypes": ("fp32",),
        "feature_flags": ("reduction", "grad_transform"),
        "notes": "Global gradient-norm clipping g*min(1, max_norm/||g||) — the "
                 "L2 norm's global sum-of-squares runs on the gfx1151 reduce "
                 "kernel (FLOP-heavy O(n) part); host does sqrt + the clip "
                 "scale; norm_type=inf uses max|g|. Executes via runtime.launch() "
                 "(rocm_grad_clip_compiled).",
    },
    "gelu": {
        "dtypes": ("fp32", "fp16", "bf16"),
        "feature_flags": ("elementwise",),
        "notes": "Standalone elementwise gelu (tanh approximation) — flat "
                 "per-element kernel (generate-rocm-activation-kernel). Executes "
                 "via runtime.launch() (rocm_activation_compiled).",
    },
    "silu": {
        "dtypes": ("fp32", "fp16", "bf16"),
        "feature_flags": ("elementwise",),
        "notes": "Standalone elementwise silu (x·σ(x)) — flat per-element kernel "
                 "(generate-rocm-activation-kernel). Executes via runtime.launch()"
                 " (rocm_activation_compiled).",
    },
    "relu": {
        "dtypes": ("fp32", "fp16", "bf16"),
        "feature_flags": ("elementwise",),
        "notes": "Standalone elementwise relu (max(0,x)) — flat per-element kernel "
                 "(generate-rocm-activation-kernel). Executes via runtime.launch() "
                 "(rocm_activation_compiled). (Reconciliation close: was runtime-"
                 "native but manifest-undeclared — manifest_runtime_reconciliation.)",
    },
    "silu_mul": {
        "dtypes": ("fp32", "fp16", "bf16"),
        "feature_flags": ("elementwise",),
        "notes": "SwiGLU gate-multiply silu(a)·b — flat 2-operand elementwise "
                 "kernel (generate-rocm-silu-mul-kernel). The standalone analog "
                 "of the fused SwiGLU gate-multiply. Executes via "
                 "runtime.launch() (rocm_silu_mul_compiled).",
    },
    # S11 pointwise regression losses — per-element loss kernel
    # (generate-rocm-pointwise-loss-kernel) + host none/mean/sum reduction. The
    # ROCm mirror of the x86_loss lane. Executes via rocm_loss_compiled.
    **{op: {
        "dtypes": ("fp32", "fp16", "bf16"),
        "feature_flags": ("elementwise",),
        "notes": f"Pointwise regression loss {op} — per-element kernel "
                 "(generate-rocm-pointwise-loss-kernel; exp/log1p via "
                 "math->rocdl) + reduction. Executes via runtime.launch() "
                 "(rocm_loss_compiled).",
    } for op in ("mse_loss", "mae_loss", "huber_loss", "smooth_l1_loss",
                 "log_cosh_loss")},
    # S11 binary-cross-entropy losses (generate-rocm-binary-loss-kernel) — ROCm
    # mirror of x86_binary_loss. Executes via rocm_binary_loss_compiled.
    **{op: {
        "dtypes": ("fp32", "fp16", "bf16"),
        "feature_flags": ("elementwise",),
        "notes": f"Binary-cross-entropy loss {op} — per-element kernel "
                 "(generate-rocm-binary-loss-kernel, stable softplus) + "
                 "reduction. Executes via runtime.launch() "
                 "(rocm_binary_loss_compiled).",
    } for op in ("binary_cross_entropy_loss", "asymmetric_bce")},
    # S11 RL policy losses — ppo/cispo/grpo surrogate
    # (generate-rocm-policy-loss-kernel) + normalize_group_advantages on the norm
    # lane. ROCm mirror of x86_rl_loss. Executes via rocm_rl_loss_compiled.
    **{op: {
        "dtypes": ("fp32", "fp16", "bf16"),
        "feature_flags": ("elementwise",),
        "notes": f"RL policy loss {op} — surrogate kernel "
                 "(generate-rocm-policy-loss-kernel) / norm-lane normalize + "
                 "reduction. Executes via runtime.launch() "
                 "(rocm_rl_loss_compiled).",
    } for op in ("ppo_policy_loss", "cispo_policy_loss", "grpo_policy_loss",
                 "normalize_group_advantages")},
    # S11 class-axis losses — exp/log on the rocm unary lane + host class-axis
    # structure. ROCm mirror of x86_class_loss. Executes via
    # rocm_class_loss_compiled.
    **{op: {
        "dtypes": ("fp32",),
        "feature_flags": ("elementwise",),
        "notes": f"Class-axis loss {op} — exp/log on the rocm unary lane + host "
                 "class-axis max/sum/gather. Executes via runtime.launch() "
                 "(rocm_class_loss_compiled).",
    } for op in ("cross_entropy_loss", "kl_divergence", "js_divergence",
                 "focal_loss", "label_smoothed_cross_entropy", "z_loss")},
    **{op: {
        "dtypes": ("fp32",),
        "feature_flags": ("elementwise", "reduction", "hip_runtime"),
        "notes": f"Metric/contrastive loss {op} — generated ROCm reduce + "
                 "exp/log kernels with host label/mask/sort/matrix structure. "
                 "Executes via runtime.launch() (rocm_metric_loss_compiled).",
    } for op in ("wasserstein_distance", "cosine_embedding_loss",
                 "contrastive_loss", "triplet_loss", "nt_xent_loss",
                 "info_nce_loss", "seq2seq_loss")},
    # P7 — EBM / diffusion losses composed on the gfx1151 binary + reduce kernels
    # (diff/square + reductions on device; host structure). Executes via
    # rocm_ebm_loss_compiled.
    **{op: {
        "dtypes": ("fp32",),
        "feature_flags": ("elementwise",),
        "notes": f"EBM/diffusion loss {op} — diff/square + reductions on the "
                 "rocm binary + reduce kernels (host structure). Executes via "
                 "runtime.launch() (rocm_ebm_loss_compiled).",
    } for op in ("score_matching_loss", "denoising_score_matching_loss",
                 "implicit_score_matching_loss", "contrastive_divergence_loss",
                 "persistent_cd_loss", "ddpm_noise_pred_loss", "vlb_loss",
                 "load_balance_loss")},
    # (EBM energy/step-compute + Langevin ops route through ebm_manifest_for() —
    # their compiled ROCm status is emitted there, not in this generic table,
    # since manifest_for() returns via ebm_manifest_for for every ebm_* name.)
    # S9 low-precision float quantization (generate-rocm-fpquant-kernel) — ROCm
    # mirror of x86_fpquant. Executes via rocm_fpquant_compiled / rocm_nvfp4.
    **{op: {
        "dtypes": ("fp32",),
        "feature_flags": ("elementwise",),
        "notes": f"Low-precision float {op} — grid-snap kernel "
                 "(generate-rocm-fpquant-kernel; log2/exp2/roundeven) + scale. "
                 "Executes via runtime.launch() (rocm_fpquant_compiled).",
    } for op in ("quantize_fp8", "dequantize_fp8", "quantize_fp6",
                 "dequantize_fp6", "quantize_fp4", "dequantize_fp4")},
    # Integer quantization — scalar qparam selection + int8 container conversion
    # around generated ROCm unary/binary kernels. int4 is signed int4 values
    # stored in int8 containers (not packed weights).
    **{op: {
        "dtypes": ("fp32", "int8") if "int8" in op else ("fp32",),
        "feature_flags": ("elementwise", "hip_runtime"),
        "notes": f"Integer quantization {op} — round/max/min/mul on generated "
                 "ROCm unary/binary kernels + host qparam structure. Executes "
                 "via runtime.launch() (rocm_intquant_compiled).",
    } for op in ("quantize_int8", "dequantize_int8", "quantize_int4",
                 "dequantize_int4", "fake_quantize")},
    **{op: {
        "dtypes": ("fp32",),
        "feature_flags": ("reduction", "hip_runtime"),
        "notes": f"Pooling {op} — host window matrix + generated ROCm reduce "
                 "kernel max/min/mean. Executes via runtime.launch() "
                 "(rocm_pooling_compiled).",
    } for op in ("max_pool", "avg_pool", "min_pool", "adaptive_pool")},
    "image_normalize": {
        "dtypes": ("fp32",),
        "feature_flags": ("elementwise", "hip_runtime"),
        "notes": "image_normalize ((x-mean)/std) composed on generated ROCm "
                 "binary sub/div kernels with host layout and per-channel "
                 "broadcast. Executes via runtime.launch() "
                 "(rocm_image_affine_compiled).",
    },
    **{op: {
        "dtypes": ("fp32",),
        "feature_flags": ("elementwise",),
        "notes": f"NVFP4 block-scaled fp4 {op} — per-block fp8-E4M3 scale + E2M1 "
                 "codes on the fpquant kernel + host block structure. Executes "
                 "via runtime.launch() (rocm_nvfp4_compiled).",
    } for op in ("quantize_nvfp4", "dequantize_nvfp4")},
    # S2 reduction foundation — prod via the warp-shuffle reduce kernel (new
    # combine); var/std/count_nonzero composed from it. rocm_reduce_compiled /
    # rocm_stat_reduce_compiled.
    "prod": {
        "dtypes": ("fp32", "fp16", "bf16"),
        "feature_flags": ("reduction",),
        "notes": "Row reduction prod (Π_k X[m,k]) — warp-shuffle reduce kernel "
                 "(generate-rocm-reduce-kernel, prod combine). Executes via "
                 "runtime.launch() (rocm_reduce_compiled).",
    },
    **{op: {
        "dtypes": ("fp32",),
        "feature_flags": ("reduction",),
        "notes": f"Statistical reduction {op} — composed from the warp-shuffle "
                 "reduce kernel (var=mean(x^2)-mean(x)^2). Executes via "
                 "runtime.launch() (rocm_stat_reduce_compiled).",
    } for op in ("var", "std", "count_nonzero")},
    # S2 stable-reduction foundation — logsumexp/log_softmax (max-shifted reduce
    # + exp/log lane), softmax_safe/sigmoid_safe (alias stable softmax/sigmoid).
    **{op: {
        "dtypes": ("fp32",),
        "feature_flags": ("reduction",),
        "notes": f"Stable reduction {op} — max-shifted reduce (max/sum) + "
                 "unary exp/log lane. Executes via runtime.launch() "
                 "(rocm_stable_reduce_compiled).",
    } for op in ("logsumexp", "log_softmax", "softmax_safe", "sigmoid_safe")},
    # Spectral FFT (PR4) — fft/ifft/rfft/irfft on the COMPILER-GENERATED DFT
    # kernel (generate-rocm-dft-kernel) on gfx1151 (rocm_fft_compiled).
    **{op: {
        "dtypes": ("fp32",),
        "feature_flags": ("spectral",),
        "notes": f"Spectral {op} — direct DFT kernel (generate-rocm-dft-kernel, "
                 "cos/sin via math->rocdl) any length + r2c/c2r + plan scale. "
                 "Executes via runtime.launch() (rocm_fft_compiled).",
    } for op in ("fft", "ifft", "rfft", "irfft")},
    # Spectral composites (PR5) — dct/stft/istft/spectral_conv/spectral_filter
    # composed on the gfx1151 FFT (DFT) lane (rocm_spectral_compiled).
    **{op: {
        "dtypes": ("fp32",),
        "feature_flags": ("spectral",),
        "notes": f"Spectral {op} — composes the rocm_fft_compiled DFT lane "
                 "(frame/window/overlap-add/pointwise on host). Executes via "
                 "runtime.launch() (rocm_spectral_compiled).",
    } for op in ("dct", "stft", "istft", "spectral_conv", "spectral_filter")},
    # Sparse (PR) — COMPILER-GENERATED gfx1151 sparse kernels (spmm CSR row-wise,
    # sddmm sampled dense-dense) + the WMMA matmul for bsmm (rocm_sparse_compiled).
    **{op: {
        "dtypes": ("fp32",),
        "feature_flags": ("sparse",),
        "notes": f"Sparse {op} — direct sparse kernel (generate-rocm-spmm/sddmm-"
                 "kernel, iterates the nonzero structure) on gfx1151; bsmm via the "
                 "WMMA matmul (bf16). Executes via runtime.launch() "
                 "(rocm_sparse_compiled).",
    } for op in ("spmm_csr", "spmm_coo", "sddmm", "bsmm")},
    **{op: {
        "dtypes": ("fp32",),
        "feature_flags": ("composite_helper", "wmma"),
        "notes": f"Composite helper {op} — Target IR keeps the helper "
                 "compiler-visible while composing existing ROCm matmul/"
                 "flash-attn/binary helper semantics plus host metadata logic. "
                 "Executes via runtime.launch() (rocm_composite_helper_compiled) "
                 "with exact reference fallback until HIP-native proof lands.",
    } for op in ("memory_index_score", "msa_index_scores", "varlen_sdpa",
                 "score_combine")},
    # MoE compute (PR) — routed per-token expert GEMVs (top-1) on gfx1151
    # (rocm_moe_compiled). dispatch/combine = transport (mesh-gated), unchanged.
    "moe": {
        "dtypes": ("fp32",),
        "feature_flags": ("moe",),
        "notes": "MoE compute (moe) — direct routed per-token expert GEMV kernel "
                 "(generate-rocm-moe-kernel, one thread per (token, out-col); "
                 "routing resolved on host) on gfx1151. Executes via "
                 "runtime.launch() (rocm_moe_compiled).",
    },
    # Optimizer steps (P3) — fused per-parameter update on gfx1151
    # (rocm_optimizer_compiled). adafactor (factored moments) is a follow-up.
    **{op: {
        "dtypes": ("fp32",),
        "feature_flags": ("optimizer",),
        "notes": f"Optimizer {op} — direct fused per-parameter update kernel "
                 "(generate-rocm-optimizer-kernel, kind StrAttr-selected, one "
                 "thread per element; host computes the 1-β^t bias correction) on "
                 "gfx1151. Executes via runtime.launch() (rocm_optimizer_compiled).",
    } for op in ("sgd", "momentum", "adam", "adamw", "lion", "nesterov")},
    # P3 tail — LAMB: device adam update + a per-tensor trust ratio ‖p‖/‖update‖
    # on host (the reduction the elementwise lane can't do).
    "lamb": {
        "dtypes": ("fp32",),
        "feature_flags": ("optimizer",),
        "notes": "Optimizer lamb — gfx1151 adam kernel (lr=1/wd=0) + host "
                 "layer-wise trust ratio ‖p‖/‖update‖. Executes via "
                 "runtime.launch() (rocm_lamb_compiled).",
    },
    # P3 tail — Muon: momentum + orthogonal polar factor U·Vh from a gfx1151
    # device SVD (host does the small U@Vh + momentum/sgd). <2-D normalizes.
    "muon": {
        "dtypes": ("fp32",),
        "feature_flags": ("optimizer",),
        "notes": "Optimizer muon — momentum then U·Vh orthogonalization via the "
                 "gfx1151 SVD kernel (rocm_linalg svd); host U@Vh + sgd. Executes "
                 "via runtime.launch() (rocm_muon_compiled).",
    },
    # State-space (PR) — Mamba2 selective scan, one thread per (b,d) channel on
    # gfx1151 (rocm_selective_ssm_compiled).
    "selective_ssm": {
        "dtypes": ("fp32", "fp16", "bf16"),
        "feature_flags": ("state_space",),
        "notes": "Mamba2 selective_ssm — direct selective-scan kernel "
                 "(generate-rocm-selective-ssm-kernel, one thread per (b,d) "
                 "channel, exp via math->rocdl) on gfx1151. Executes via "
                 "runtime.launch() (rocm_selective_ssm_compiled). f16/bf16 "
                 "storage (loads extf->f32, y truncf; state+exp+accumulate f32). "
                 "Reverse-mode adjoint: generate-rocm-selective-ssm-bwd-kernel "
                 "(reverse scan, atomic_rmw cross-channel reductions), matches "
                 "the numpy VJP. (The chunked-parallel SSD form is x86-only: ROCm's "
                 "WMMA bmm is f16/bf16, so f32/low-precision stay on this "
                 "exact sequential scan rather than overflowing an f16 bmm.)",
    },
    # Linalg PR-A — Cholesky + triangular solve on gfx1151 (one thread per matrix
    # / per matrix-RHS-column) (rocm_linalg_compiled).
    **{op: {
        "dtypes": ("fp32",),
        "feature_flags": ("linalg",),
        "notes": f"Linalg {op} — direct factorization/solve kernel (generate-"
                 "rocm-cholesky/tri-solve-kernel) on gfx1151; cholesky_solve = "
                 "two triangular solves. Executes via runtime.launch() "
                 "(rocm_linalg_compiled).",
    } for op in ("cholesky", "tri_solve", "cholesky_solve")},
    # Linalg PR-B — LU (partial pivot) + Householder QR on gfx1151, one thread
    # per matrix (rocm_linalg_compiled).
    **{op: {
        "dtypes": ("fp32",),
        "feature_flags": ("linalg",),
        "notes": f"Linalg {op} — direct factorization kernel (generate-rocm-{op}-"
                 "kernel, one thread per matrix) on gfx1151. Executes via "
                 "runtime.launch() (rocm_linalg_compiled).",
    } for op in ("lu", "qr")},
    # Linalg PR-C — one-sided Jacobi SVD on gfx1151 (rocm_linalg_compiled).
    "svd": {
        "dtypes": ("fp32",),
        "feature_flags": ("linalg",),
        "notes": "Linalg svd — direct one-sided Jacobi SVD kernel (generate-rocm-"
                 "svd-kernel, one thread per matrix, m>=n; wide case via "
                 "transpose) on gfx1151. Executes via runtime.launch() "
                 "(rocm_linalg_compiled).",
    },
    # S2 scalar-math / stability family — flat per-element unary math kernel
    # (generate-rocm-unary-kernel), the unary sibling of the activation lane.
    # Executes via runtime.launch() (rocm_unary_compiled). f32/f16/bf16, f32
    # compute. The transcendentals lower through the math → ROCDL path.
    **{op: {
        "dtypes": ("fp32", "fp16", "bf16"),
        "feature_flags": ("elementwise",),
        "notes": f"Standalone elementwise {op} — flat per-element unary-math "
                 "kernel (generate-rocm-unary-kernel). Executes via "
                 "runtime.launch() (rocm_unary_compiled).",
    } for op in ("exp", "log", "sqrt", "rsqrt", "reciprocal", "absolute",
                 "abs", "sign", "erf", "tanh", "sigmoid", "log1p", "expm1",
                 "softplus", "sin", "cos", "tan", "sinh", "cosh", "asin", "acos",
                 "atan", "erfc", "floor", "ceil", "round", "trunc")},
    # P2e — lgamma: ln Γ(x) via an MLIR-built Lanczos g=5 series + reflection
    # (no math.lgamma exists). fp32 only (the series is f32-tuned).
    "lgamma": {
        "dtypes": ("fp32",),
        "feature_flags": ("elementwise",),
        "notes": "Standalone elementwise lgamma — MLIR-built Lanczos g=5 series "
                 "(generate-rocm-unary-kernel; reflection via math.sin for "
                 "x<0.5). Executes via runtime.launch() (rocm_unary_compiled).",
    },
    # P2e — digamma: ψ(x) via an MLIR-built recurrence + asymptotic series
    # (no math.digamma); reflection (math.tan) + pole->NaN for x<=0. fp32 only.
    "digamma": {
        "dtypes": ("fp32",),
        "feature_flags": ("elementwise",),
        "notes": "Standalone elementwise digamma — MLIR-built recurrence + "
                 "asymptotic series (generate-rocm-unary-kernel; reflection via "
                 "math.tan, poles->NaN for x<=0). Executes via runtime.launch() "
                 "(rocm_unary_compiled).",
    },
    # S2 binary-arithmetic family — flat 2-operand per-element kernel
    # (generate-rocm-binary-kernel), the binary sibling of the unary-math lane.
    # Executes via runtime.launch() (rocm_binary_compiled). f32/f16/bf16, f32
    # compute. maximum/minimum are IEEE NaN-propagating; pow lowers via ROCDL.
    **{op: {
        "dtypes": ("fp32", "fp16", "bf16"),
        "feature_flags": ("elementwise",),
        "notes": f"Standalone elementwise binary {op} — flat 2-operand "
                 "per-element kernel (generate-rocm-binary-kernel). Executes via "
                 "runtime.launch() (rocm_binary_compiled).",
    } for op in ("sub", "div", "pow", "maximum", "minimum",
                 "add", "mul", "mod", "floor_div")},
    # S2 comparison family — flat 2-operand per-element kernel
    # (generate-rocm-compare-kernel) with boolean (i8 0/1) output, NaN semantics
    # matching numpy. Executes via runtime.launch() (rocm_compare_compiled).
    # f32/f16/bf16 input storage, f32 compare, bool output.
    **{op: {
        "dtypes": ("fp32", "fp16", "bf16"),
        "feature_flags": ("elementwise",),
        "notes": f"Standalone elementwise comparison {op} — flat 2-operand "
                 "per-element kernel (generate-rocm-compare-kernel), bool output. "
                 "Executes via runtime.launch() (rocm_compare_compiled).",
    } for op in ("eq", "ne", "lt", "le", "gt", "ge")},
    # P2b — unary predicate family (isnan/isinf/isfinite), f32 in / i8 bool out
    # (generate-rocm-predicate-kernel). Executes via rocm_predicate_compiled.
    **{op: {
        "dtypes": ("fp32",),
        "feature_flags": ("elementwise",),
        "notes": f"Standalone unary predicate {op} — flat per-element kernel "
                 "(generate-rocm-predicate-kernel), bool output. Executes via "
                 "runtime.launch() (rocm_predicate_compiled).",
    } for op in ("isnan", "isinf", "isfinite")},
    # P2c — clamp/clip composed on the gfx1151 binary max/min lane (no new
    # kernel; scalar bounds broadcast on host). Executes via rocm_clamp_compiled.
    **{op: {
        "dtypes": ("fp32",),
        "feature_flags": ("elementwise",),
        "notes": f"Standalone {op} — min(max(x, lo), hi) composed on the "
                 "rocm_binary_compiled max/min kernel (either bound optional). "
                 "Executes via runtime.launch() (rocm_clamp_compiled).",
    } for op in ("clamp", "clip")},
    # (complex_* ops route through complex_manifest_for() — their compiled
    # device-lane status is emitted there, not in this generic _ROCM table.)
    # P2e — softcap composed on the gfx1151 unary tanh lane (no new kernel;
    # scalar cap broadcast on host). Executes via rocm_softcap_compiled.
    "softcap": {
        "dtypes": ("fp32",),
        "feature_flags": ("elementwise",),
        "notes": "Standalone softcap — cap*tanh(x/cap) composed on the "
                 "rocm_unary_compiled tanh kernel (scalar cap on host). "
                 "Executes via runtime.launch() (rocm_softcap_compiled).",
    },
    # P4 — 0-move / strided-copy lane: pad/cat/roll/flip/tile/repeat/stack via
    # the gfx1151 masked-gather kernel (host computes the index map; device moves
    # the f32 data). Executes via rocm_strided_compiled.
    **{op: {
        "dtypes": ("fp32",),
        "feature_flags": ("layout_transform",),
        "notes": f"Standalone {op} — 0-move op realized by the gfx1151 "
                 "masked-gather kernel (generate-rocm-gather-kernel; host index "
                 "map). Executes via runtime.launch() (rocm_strided_compiled).",
    } for op in ("pad", "cat", "roll", "flip", "tile", "repeat", "stack")},
    # P8 — scatter family (0-reduce indexed store) via the COMPILER-GENERATED
    # gfx1151 kernel (generate-rocm-scatter-kernel; one thread per element;
    # atomic_rmw for add/min/max). Executes via rocm_scatter_compiled.
    **{op: {
        "dtypes": ("fp32",),
        "feature_flags": ("data_movement", "atomics"),
        "notes": f"Standalone {op} — 0-reduce indexed store realized by the "
                 "COMPILER-GENERATED gfx1151 kernel (generate-rocm-scatter-"
                 "kernel; one thread per element; atomic_rmw add/min/max). "
                 "Executes via runtime.launch() (rocm_scatter_compiled).",
    } for op in ("scatter", "scatter_add", "scatter_reduce")},
    # P9 — sort / argsort / top_k via the COMPILER-GENERATED cooperative bitonic
    # kernel (generate-rocm-sort-kernel, one block per row; host pads/flips).
    **{op: {
        "dtypes": ("fp32",),
        "feature_flags": ("data_movement",),
        "notes": f"Standalone {op} — data-independent bitonic sort network "
                 "(generate-rocm-sort-kernel; one block per row; host pads to a "
                 "power of two + flips for descending). Executes via "
                 "runtime.launch() (rocm_sort_compiled).",
    } for op in ("sort", "argsort", "top_k")},
    # P13 — conv2d / conv3d via im2col + the gfx1151 WMMA GEMM (host im2col,
    # device WMMA f16/bf16 with f32 accumulate). Executes via rocm_conv_compiled.
    **{op: {
        "dtypes": ("fp16", "bf16"),
        "feature_flags": ("stencil", "wmma"),
        "notes": f"Standalone {op} — im2col + the COMPILER-GENERATED gfx1151 "
                 "WMMA GEMM (host lays out the patch matrix; f16/bf16 storage, "
                 "f32 accumulate). Executes via runtime.launch() "
                 "(rocm_conv_compiled).",
    } for op in ("conv2d", "conv3d")},
    # P5 — conformal geometry: mobius / stereographic composed on the gfx1151
    # complex (mul/div) + binary (div) lanes (no new kernel; host orchestration).
    **{op: {
        "dtypes": ("fp32",),
        "feature_flags": ("complex_namespace",),
        "notes": f"Standalone {op} — composed on the gfx1151 complex / binary "
                 "lanes (interleaved-f32; host orchestration). Executes via "
                 "runtime.launch() (rocm_conformal_compiled).",
    } for op in ("mobius", "stereographic")},
    # P6 — device RNG: counter-based Philox-4x32-10 (generate-rocm-philox-kernel)
    # produces the uniform bits; host applies the distribution transform. A
    # SEPARATE deterministic stream from the host numpy-Generator path.
    **{op: {
        "dtypes": ("fp32",),
        "feature_flags": ("random",),
        "notes": f"Standalone {op} — Philox-4x32-10 uniform RNG kernel "
                 "(generate-rocm-philox-kernel) + host transform. Executes via "
                 "runtime.launch() (rocm_rng_compiled).",
    } for op in ("rng_uniform", "rng_normal", "dropout")},
    # P2e — atan2 composed on the gfx1151 unary atan lane (no new kernel;
    # quadrant/sign logic on host). Executes via rocm_atan2_compiled.
    "atan2": {
        "dtypes": ("fp32",),
        "feature_flags": ("elementwise",),
        "notes": "Standalone atan2 — quadrant-aware atan2(y, x) composed on the "
                 "rocm_unary_compiled atan kernel (sign/quadrant on host). "
                 "Executes via runtime.launch() (rocm_atan2_compiled).",
    },
    # S2 logical family — flat elementwise kernel over i8 booleans
    # (generate-rocm-logical-kernel). and/or/xor binary, not unary; inputs
    # normalized via != 0 (numpy semantics). Executes via runtime.launch()
    # (rocm_logical_compiled). bool in/out.
    **{op: {
        "dtypes": ("bool",),
        "feature_flags": ("elementwise",),
        "notes": f"Standalone elementwise {op} — flat per-element logical kernel "
                 "(generate-rocm-logical-kernel) over i8 booleans. Executes via "
                 "runtime.launch() (rocm_logical_compiled).",
    } for op in ("logical_and", "logical_or", "logical_xor", "logical_not")},
    # S2 bitwise family — flat elementwise kernel over i32 integers
    # (generate-rocm-bitwise-kernel), acting on the full bit pattern (no
    # normalization). and/or/xor binary, not unary. Executes via
    # runtime.launch() (rocm_bitwise_compiled). i32 in/out.
    **{op: {
        "dtypes": ("int32",),
        "feature_flags": ("elementwise",),
        "notes": f"Standalone elementwise {op} — flat per-element bitwise kernel "
                 "(generate-rocm-bitwise-kernel) over i32 integers. Executes via "
                 "runtime.launch() (rocm_bitwise_compiled).",
    } for op in ("bitwise_and", "bitwise_or", "bitwise_xor", "bitwise_not")},
    # P2e — popcount: set-bit count per i32 element, unary, on the bitwise lane
    # via math.ctpop (RDNA v_bcnt). Executes via runtime.launch().
    "popcount": {
        "dtypes": ("int32",),
        "feature_flags": ("elementwise",),
        "notes": "Standalone elementwise popcount — flat per-element set-bit "
                 "count (generate-rocm-bitwise-kernel, math.ctpop / v_bcnt) over "
                 "i32 integers. Executes via runtime.launch() "
                 "(rocm_bitwise_compiled).",
    },
    # Ternary select where(cond,a,b)=cond?a:b — flat 3-operand elementwise
    # kernel (generate-rocm-where-kernel). cond i8 normalized != 0, a/b/out
    # float. Executes via runtime.launch() (rocm_where_compiled).
    "where": {
        "dtypes": ("fp32", "fp16", "bf16"),
        "feature_flags": ("elementwise",),
        "notes": "Standalone ternary select where(cond,a,b)=cond?a:b — flat "
                 "3-operand per-element kernel (generate-rocm-where-kernel); "
                 "cond i8 normalized != 0, a/b/out float. Executes via "
                 "runtime.launch() (rocm_where_compiled).",
    },
    "rope": {
        "dtypes": ("fp32", "fp16", "bf16"),
        "feature_flags": ("elementwise",),
        "notes": "Rotary position embedding (interleaved pairs) over [M, D] — "
                 "elementwise-per-pair kernel (generate-rocm-rope-kernel: one "
                 "workgroup per row, f32 cos/sin). Executes via runtime.launch() "
                 "(rocm_rope_compiled).",
    },
    "alibi": {
        "dtypes": ("fp32", "fp16", "bf16"),
        "feature_flags": ("elementwise",),
        "notes": "ALiBi positional-bias generator bias[h,i,j]=slope[h]·(j−i) over "
                 "[H, S, S] — flat elementwise kernel (generate-rocm-alibi-"
                 "kernel). Slopes default to the 2^(-8(k+1)/H) ramp. Executes via "
                 "runtime.launch() (rocm_alibi_compiled).",
    },
    "dequant_matmul": {
        "dtypes": ("fp32",),
        "feature_flags": ("quantization", "gemm"),
        "notes": "Packed dequantize-then-GEMM over int4/int8 grouped weights — "
                 "compiler-generated ROCm dequant GEMM executor. Executes via "
                 "runtime.launch() (rocm_dequant_gemm_compiled).",
    },
    "dequant_grouped_gemm": {
        "dtypes": ("fp32",),
        "feature_flags": ("quantization", "grouped_gemm"),
        "notes": "Grouped packed dequantize-then-GEMM, one expert slice per "
                 "group size; reuses the ROCm dequant GEMM executor per expert. "
                 "Executes via runtime.launch() (rocm_dequant_gemm_compiled).",
    },
    "batched_gemm": {
        "dtypes": ("fp16", "bf16"),
        "notes": "Batched matmul A[...,M,K]@B[...,K,N] — the WMMA GEMM kernel "
                 "looped over leading batch dims. Executes via runtime.launch() "
                 "(rocm_matmul_family_compiled).",
    },
    "linear_general": {
        "dtypes": ("fp16", "bf16"),
        "notes": "Axis-flexible linear projection x[...,K]@W[K,N] (+bias), "
                 "axis=-1 — reshape + WMMA GEMM. Executes via runtime.launch() "
                 "(rocm_matmul_family_compiled).",
    },
    "qkv_projection": {
        "dtypes": ("fp16", "bf16"),
        "notes": "Fused QKV projection x[...,D]@W_qkv[D,3N] (the 3-way split is a "
                 "trivial host view) — reshape + WMMA GEMM. Executes via "
                 "runtime.launch() (rocm_matmul_family_compiled).",
    },
    "factorized_matmul": {
        "dtypes": ("fp16", "bf16"),
        "notes": "Low-rank A@B with rank-r truncation — the WMMA GEMM on GPU + an "
                 "exact host SVD-truncate epilogue. Executes via runtime.launch() "
                 "(rocm_matmul_family_compiled).",
    },
    "einsum": {
        "dtypes": ("fp16", "bf16"),
        "notes": "Single-contraction two-operand einsum mapped to a (batched) "
                 "WMMA GEMM (canonicalize + transpose); other specs emit a stable "
                 "unsupported diagnostic. Executes via runtime.launch() "
                 "(rocm_matmul_family_compiled).",
    },
    "gated_attention": {
        "dtypes": ("fp16", "bf16"),
        "notes": "Softmax attention × a learned gate — the WMMA flash_attn kernel "
                 "+ an elementwise sigmoid-gate multiply. Executes via "
                 "runtime.launch() (rocm_exotic_attn_compiled).",
    },
    "mla_decode": {
        "dtypes": ("fp16", "bf16"),
        "notes": "Multi-head latent attention decode — latent K/V projections "
                 "(WMMA GEMM) + the WMMA flash_attn kernel. Executes via "
                 "runtime.launch() (rocm_exotic_attn_compiled).",
    },
    "mla_decode_fused": {
        "dtypes": ("fp16", "bf16"),
        "notes": "Fused MLA decode — down/up latent projections (c=x@w_dkv; "
                 "K=c@w_uk; V=c@w_uv on the WMMA GEMM) + the WMMA flash_attn "
                 "kernel. Executes via runtime.launch() "
                 "(rocm_exotic_attn_compiled).",
    },
    "deepseek_sparse_attention": {
        "dtypes": ("fp32",),
        "feature_flags": ("sparse_attention",),
        "notes": "DeepSeek/NSA composition — sliding + compressed-block "
                 "branches remain reference compositions while the top-k branch "
                 "uses the GPU-resident top-k selector plus selected-block "
                 "sparse-attention kernel when ROCm is available. Executes via "
                 "runtime.launch() (rocm_sparse_attn_compiled), with exact "
                 "reference fallback off hardware.",
    },
    "msa_sparse_attention": {
        "dtypes": ("fp32",),
        "feature_flags": ("sparse_attention",),
        "notes": "MiniMax Sparse Attention: block scores/select may use the ROCm "
                 "GPU selector; selected-block attention uses the native sparse "
                 "attention kernel when hardware is available. Executes via "
                 "runtime.launch() (rocm_sparse_attn_compiled), with exact "
                 "reference fallback off hardware.",
    },
    "hybrid_attention": {
        "dtypes": ("fp16", "bf16", "fp32"),
        "feature_flags": ("attention_policy",),
        "notes": "Named Ling/Kimi hybrid attention policy wrapper — delegates "
                 "Lightning slots to rocm_linear_attn_compiled and Delta/Kimi "
                 "slots to rocm_deltanet_compiled when ROCm is available; MLA "
                 "softmax/weight slots fall back to the public reference until "
                 "the fused hybrid slot is promoted.",
    },
    "gated_deltanet": {
        "dtypes": ("fp32", "fp16", "bf16"),
        "feature_flags": ("recurrent",),
        "notes": "Gated/delta linear-attention recurrence as a causal "
                 "SEQUENTIAL-SCAN kernel (generate-rocm-deltanet-kernel: one "
                 "workgroup per (b,h), one thread per value-column, LDS state) — "
                 "the first recurrent compiled ROCm kernel. erase/gate/beta/decay "
                 "flags. Executes via runtime.launch() (rocm_deltanet_compiled).",
    },
    "kimi_delta_attention": {
        "dtypes": ("fp32", "fp16", "bf16"),
        "feature_flags": ("recurrent",),
        "notes": "Kimi Delta Attention — the gated/delta scan kernel "
                 "(generate-rocm-deltanet-kernel), same recurrence as "
                 "gated_deltanet. Executes via runtime.launch() "
                 "(rocm_deltanet_compiled).",
    },
    "modified_delta_attention": {
        "dtypes": ("fp32", "fp16", "bf16"),
        "feature_flags": ("recurrent",),
        "notes": "Modified Delta Attention — the gated/delta scan kernel with a "
                 "bounded delta update (delta / (1 + ‖k‖·‖target‖)). Executes via "
                 "runtime.launch() (rocm_deltanet_compiled).",
    },
}


# ─────────────────────────────────────────────────────────────────────────────
# Consumer-Blackwell bring-up (2026-06-25) — NVIDIA ops promoted to
# ``hardware_verified`` on sm_120 (RTX 5070 Ti, CC 12.0, CUDA 13.3).
#
# Mirror of ``_ROCM_HARDWARE_VERIFIED``: an op lands here once it ships BOTH a
# real C-ABI ``runtime_symbol`` (the auto-built ``libtessera_nvidia_gemm.so``,
# CMake target ``tessera_nvidia_gemm``, which NVRTC-compiles the warp-level
# mma.sync kernel for the device arch and runs it on the GPU) AND a checked-in
# ``execute_compare_fixture`` (see ``_NUMERICAL_FIXTURES``). When present, the
# generic nvidia_sm120 artifact_only row is REPLACED by this hardware_verified
# row (the other sm_80/90/100 rows stay artifact_only — proven only on sm_120).
#
# Honesty (Decision #25): the shape_envelope/dtypes describe ONLY what the shipped
# symbols prove. matmul is a general tiled/K-looped warp-level **mma.sync** GEMM
# (NOT tcgen05/TMEM — consumer Blackwell lacks those), f32 accumulate, ragged
# M/N/K zero-padded. TF32 is fp32-storage + tf32-math (Decision #15a), so it is
# listed as ``fp32``; fp8 e4m3/e5m2 are first-class storage dtypes here.
# ─────────────────────────────────────────────────────────────────────────────
_NVIDIA_HARDWARE_VERIFIED: dict[str, dict[str, Any]] = {
    "matmul": {
        "runtime_symbol": "tessera_nvidia_mma_gemm_bf16",
        "dtypes": ("bf16", "fp16", "fp32", "fp8_e4m3", "fp8_e5m2"),
        "feature_flags": ("mma_sync",),
        "shape_envelope": (
            "general tiled/K-looped warp-level mma.sync GEMM, f32 accumulate, any "
            "positive M/N/K (ragged edges zero-padded), one warp per 16x8 output "
            "tile. Per-dtype MMA shape: bf16/fp16 m16n8k16, fp32(tf32-math) "
            "m16n8k8, fp8 e4m3/e5m2 m16n8k32 (sm_120 consumer Blackwell, "
            "RTX 5070 Ti)"
        ),
        "notes": (
            "Warp-level mma.sync GEMM on consumer Blackwell (sm_120, CC 12.0), "
            "CUDA 13.3. Ships five C-ABI symbols in libtessera_nvidia_gemm.so "
            "(CMake target tessera_nvidia_gemm): tessera_nvidia_mma_gemm_"
            "{bf16,f16,tf32,e4m3,e5m2}, each NVRTC-compiled (compute_XX from "
            "cuDeviceGetAttribute) at first call and launched via the CUDA driver "
            "API. fp32 storage runs tf32-math (mma.sync m16n8k8.tf32). Numerically "
            "validated vs numpy/ml_dtypes references by the execute_compare_fixture "
            "across aligned + ragged shapes. NOT the @jit default lane yet — the "
            "runtime dispatch / execution_matrix row is the follow-up (cf. ROCm's "
            "rocm_wmma symbol vs the rocm_compiled lane)."
        ),
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
    # rocm — Strix Halo (gfx1151 / gfx1100 WSL) RDNA WMMA. The shipped
    # `tessera_rocm_wmma_gemm_f16` C-ABI symbol (libtessera_rocm_gemm.so) is
    # dlopened and its f32<-f16 16x16x16 WMMA GEMM compared to a numpy
    # reference (maxerr < 1e-2). Skip-clean when no AMD GPU / HIPRTC. This is
    # the numerical-proof half of the rocm matmul `hardware_verified` row.
    ("matmul", "rocm"): "tests/unit/test_rocm_wmma_runtime_symbol.py",
    # gemm shares matmul's WMMA symbol (same tessera_rocm_wmma_gemm_f16) — the same
    # numerical proof covers it.
    ("gemm", "rocm"): "tests/unit/test_rocm_wmma_runtime_symbol.py",
    # rocm flash_attn — the shipped `tessera_rocm_wmma_flash_attn_{f16,bf16}`
    # C-ABI symbols (libtessera_rocm_flash_attn.so) are dlopened and the FA-2
    # forward (both QK^T and P@V on 16x16x16 WMMA, online softmax, causal +
    # ragged) is compared to a numpy attention reference. Skip-clean when no AMD
    # GPU / HIPRTC. The numerical-proof half of the rocm flash_attn
    # `hardware_verified` row — the second op after matmul to execute on ROCm.
    ("flash_attn", "rocm"): "tests/unit/test_rocm_flash_attn_runtime_symbol.py",
    ("spec_accept", "rocm"): "tests/unit/test_rocm_spec_accept_exec.py",
    ("spec_accept_sample", "rocm"): "tests/unit/test_rocm_spec_accept_sample_exec.py",
    ("spec_accept_tree_sample", "rocm"):
        "tests/unit/test_rocm_spec_accept_tree_sample_exec.py",
    # P10 — x86 AVX-512 flash_attn partner (online-softmax FA forward), compared
    # to the dense attention reference on the AVX-512 box. Skip-clean w/o the .so.
    ("flash_attn", "x86"): "tests/unit/test_x86_flash_attn_compiled.py",
    ("msa_sparse_attention", "x86"): "tests/unit/test_x86_msa_compiled.py",
    # P13 — conv2d/conv3d via im2col + the device GEMM, compared to the conv
    # reference on x86 (f32) + gfx1151 (WMMA f16). Skip-clean w/o the .so / GPU.
    ("conv2d", "x86"): "tests/unit/test_x86_conv_compiled.py",
    ("conv3d", "x86"): "tests/unit/test_x86_conv_compiled.py",
    ("conv2d", "rocm"): "tests/unit/test_rocm_conv_compiled.py",
    ("conv3d", "rocm"): "tests/unit/test_rocm_conv_compiled.py",
    # P8 — scatter family (0-reduce indexed store) on x86 + ROCm, each compared
    # to the numpy scatter reference. Skip-clean w/o the .so / GPU.
    **{(op, "x86"): "tests/unit/test_x86_scatter_compiled.py"
       for op in ("scatter", "scatter_add", "scatter_reduce")},
    **{(op, "rocm"): "tests/unit/test_rocm_scatter_compiled.py"
       for op in ("scatter", "scatter_add", "scatter_reduce")},
    # §5.6 KV-cache paged-movement core — append/read/prune executed on the
    # gfx1151 scatter+gather kernels, compared to the KVCacheHandle reference.
    **{(op, "rocm"): "tests/unit/test_rocm_kv_cache_compiled.py"
       for op in ("kv_cache_append", "kv_cache_read", "kv_cache_prune")},
    # P11 — x86 MLA latent-KV lane (compress/expand/decode composed on the GEMM +
    # flash_attn lanes), compared to the numpy MLA reference. Skip-clean w/o .so.
    **{(op, "x86"): "tests/unit/test_x86_mla_compiled.py"
       for op in ("latent_kv_compress", "latent_kv_expand_k",
                  "latent_kv_expand_v", "mla_decode_fused")},
    # rocm compiled-lane family (2026-06-25) — compiler-generated hsaco executing
    # via runtime.launch(), each compared to a numpy reference on gfx1151. These
    # back the ``compiled`` status (no shipped C-ABI symbol). Skip-clean w/o GPU.
    ("gqa_attention", "rocm"): "tests/unit/test_rocm_gqa_compiled.py",
    ("mqa_attention", "rocm"): "tests/unit/test_rocm_gqa_compiled.py",
    ("multi_head_attention", "rocm"): "tests/unit/test_rocm_flash_attn_compiled.py",
    ("attn_sliding_window", "rocm"):
        "tests/unit/test_rocm_sliding_window_compiled.py",
    ("linear_attn", "rocm"): "tests/unit/test_rocm_linear_attn_compiled.py",
    ("lightning_attention", "rocm"):
        "tests/unit/test_rocm_linear_attn_compiled.py",
    ("retention", "rocm"): "tests/unit/test_rocm_linear_attn_compiled.py",
    ("fused_epilogue", "rocm"):
        "tests/unit/test_rocm_fused_epilogue_compiled.py",
    ("softmax", "rocm"): "tests/unit/test_rocm_softmax_compiled.py",
    **{(op, "rocm"): "tests/unit/test_rocm_reduce_compiled.py"
       for op in ("sum", "mean", "max", "min", "amax", "amin")},
    **{(op, "rocm"): "tests/unit/test_rocm_argreduce_compiled.py"
       for op in ("argmax", "argmin")},
    **{(op, "rocm"): "tests/unit/test_rocm_scan_compiled.py"
       for op in ("cumsum", "cumprod", "cummax", "cummin")},
    **{(op, "x86"): "tests/unit/test_x86_scan_compiled.py"
       for op in ("cumsum", "cumprod", "cummax", "cummin")},
    **{(op, "x86"): "tests/unit/test_x86_argreduce_compiled.py"
       for op in ("argmax", "argmin")},
    **{(op, "x86"): "tests/unit/test_x86_reduce_compiled.py"
       for op in ("sum", "mean", "max", "min", "amax", "amin")},
    **{(op, "x86"): "tests/unit/test_x86_unary_compiled.py"
       for op in ("sqrt", "rsqrt", "reciprocal", "absolute", "sign",
                  "floor", "ceil", "round", "trunc")},
    **{(op, "x86"): "tests/unit/test_x86_binary_compiled.py"
       for op in ("sub", "div", "maximum", "minimum")},
    # P2a — binary arithmetic completion (add/mul/mod/floor_div) + abs alias.
    **{(op, "x86"): "tests/unit/test_x86_elementwise_p2_compiled.py"
       for op in ("add", "mul", "mod", "floor_div", "abs")},
    **{(op, "rocm"): "tests/unit/test_rocm_elementwise_p2_compiled.py"
       for op in ("add", "mul", "mod", "floor_div", "abs")},
    **{(op, "x86"): "tests/unit/test_x86_clamp_compiled.py"
       for op in ("clamp", "clip")},
    **{(op, "rocm"): "tests/unit/test_rocm_clamp_compiled.py"
       for op in ("clamp", "clip")},
    **{(op, "x86"): "tests/unit/test_x86_complex_compiled.py"
       for op in ("complex_mul", "complex_div", "complex_conjugate",
                  "complex_abs", "complex_arg", "complex_exp", "complex_log",
                  "complex_sqrt", "complex_pow")},
    **{(op, "rocm"): "tests/unit/test_rocm_complex_compiled.py"
       for op in ("complex_mul", "complex_div", "complex_conjugate",
                  "complex_abs", "complex_arg", "complex_exp", "complex_log",
                  "complex_sqrt", "complex_pow")},
    **{(op, "x86"): "tests/unit/test_x86_complex_compiled.py"
       for op in ("check_cauchy_riemann", "conformal_jacobian", "dbar", "dz",
                  "laplacian_2d")},
    **{(op, "rocm"): "tests/unit/test_rocm_complex_compiled.py"
       for op in ("check_cauchy_riemann", "conformal_jacobian", "dbar", "dz",
                  "laplacian_2d")},
    ("softcap", "x86"): "tests/unit/test_x86_softcap_compiled.py",
    ("softcap", "rocm"): "tests/unit/test_rocm_softcap_compiled.py",
    **{(op, "x86"): "tests/unit/test_x86_rng_compiled.py"
       for op in ("rng_uniform", "rng_normal", "dropout")},
    **{(op, "rocm"): "tests/unit/test_rocm_rng_compiled.py"
       for op in ("rng_uniform", "rng_normal", "dropout")},
    **{(op, "x86"): "tests/unit/test_x86_strided_compiled.py"
       for op in ("pad", "cat", "roll", "flip", "tile", "repeat", "stack")},
    **{(op, "rocm"): "tests/unit/test_rocm_strided_compiled.py"
       for op in ("pad", "cat", "roll", "flip", "tile", "repeat", "stack")},
    **{(op, "x86"): "tests/unit/test_x86_conformal_compiled.py"
       for op in ("mobius", "stereographic")},
    **{(op, "rocm"): "tests/unit/test_rocm_conformal_compiled.py"
       for op in ("mobius", "stereographic")},
    **{(op, "x86"): "tests/unit/test_x86_sort_compiled.py"
       for op in ("sort", "argsort", "top_k")},
    **{(op, "rocm"): "tests/unit/test_rocm_sort_compiled.py"
       for op in ("sort", "argsort", "top_k")},
    ("atan2", "x86"): "tests/unit/test_x86_atan2_compiled.py",
    ("atan2", "rocm"): "tests/unit/test_rocm_atan2_compiled.py",
    ("sin", "x86"): "tests/unit/test_x86_sin_compiled.py",
    ("sin", "rocm"): "tests/unit/test_rocm_sin_compiled.py",
    ("lgamma", "x86"): "tests/unit/test_x86_lgamma_compiled.py",
    ("lgamma", "rocm"): "tests/unit/test_rocm_lgamma_compiled.py",
    ("digamma", "x86"): "tests/unit/test_x86_digamma_compiled.py",
    ("digamma", "rocm"): "tests/unit/test_rocm_digamma_compiled.py",
    **{(op, "x86"): "tests/unit/test_x86_predicate_compiled.py"
       for op in ("isnan", "isinf", "isfinite")},
    **{(op, "rocm"): "tests/unit/test_rocm_predicate_compiled.py"
       for op in ("isnan", "isinf", "isfinite")},
    **{(op, "x86"): "tests/unit/test_x86_compare_compiled.py"
       for op in ("eq", "ne", "lt", "le", "gt", "ge")},
    **{(op, "x86"): "tests/unit/test_x86_logical_compiled.py"
       for op in ("logical_and", "logical_or", "logical_xor", "logical_not")},
    **{(op, "x86"): "tests/unit/test_x86_bitwise_compiled.py"
       for op in ("bitwise_and", "bitwise_or", "bitwise_xor", "bitwise_not")},
    ("popcount", "x86"): "tests/unit/test_x86_popcount_compiled.py",
    ("where", "x86"): "tests/unit/test_x86_where_compiled.py",
    ("where", "rocm"): "tests/unit/test_rocm_where_compiled.py",
    **{(op, "x86"): "tests/unit/test_x86_transcendental_compiled.py"
       for op in ("exp", "log", "tanh", "sigmoid", "silu", "gelu", "erf",
                  "softplus", "expm1", "log1p", "cos", "tan", "sinh", "cosh",
                  "asin", "acos", "atan", "erfc")},
    **{(op, "x86"): "tests/unit/test_x86_binary_math_compiled.py"
       for op in ("pow", "silu_mul")},
    **{(op, "x86"): "tests/unit/test_x86_norm_softmax_compiled.py"
       for op in ("rmsnorm", "layer_norm", "softmax")},
    # Normalization tail closed on the existing kernels (dispatch only, no new
    # codegen): online_softmax(no-state) == softmax; rmsnorm_safe == rmsnorm.
    ("online_softmax", "rocm"):
        "tests/unit/test_online_softmax_rmsnorm_safe_compiled.py",
    ("online_softmax", "x86"):
        "tests/unit/test_online_softmax_rmsnorm_safe_compiled.py",
    ("rmsnorm_safe", "x86"):
        "tests/unit/test_online_softmax_rmsnorm_safe_compiled.py",
    **{(op, "x86"): "tests/unit/test_x86_matmul_family_compiled.py"
       for op in ("batched_gemm", "linear_general", "qkv_projection",
                  "factorized_matmul", "einsum")},
    ("rope", "x86"): "tests/unit/test_x86_posenc_compiled.py",
    ("alibi", "x86"): "tests/unit/test_x86_posenc_compiled.py",
    **{(op, "x86"): "tests/unit/test_x86_attention_compiled.py"
       for op in ("multi_head_attention", "gqa_attention", "mqa_attention",
                  "mla_decode")},
    ("attn_sliding_window", "x86"):
        "tests/unit/test_x86_flash_attn_compiled.py",
    ("deepseek_sparse_attention", "x86"):
        "tests/unit/test_x86_nsa_compiled.py",
    **{(op, "x86"): "tests/unit/test_x86_linear_attn_compiled.py"
       for op in ("linear_attn", "power_attn", "retention")},
    **{(op, "x86"): "tests/unit/test_x86_deltanet_compiled.py"
       for op in ("gated_deltanet", "kimi_delta_attention",
                  "modified_delta_attention")},
    **{(op, "x86"): "tests/unit/test_x86_rng_compiled.py"
       for op in ("rng_uniform", "rng_normal", "rng_bernoulli", "rng_beta",
                  "rng_categorical", "rng_dirichlet", "rng_gamma",
                  "rng_gibbs_sample", "rng_hmc_sample", "rng_langevin_sample",
                  "rng_mala_sample", "rng_multinomial", "rng_permutation",
                  "rng_poisson", "rng_randint", "rng_truncated_normal",
                  "rng_key", "rng_split", "rng_fold_in", "rng_clone")},
    **{(op, "rocm"): "tests/unit/test_rocm_rng_compiled.py"
       for op in ("rng_uniform", "rng_normal", "rng_bernoulli", "rng_beta",
                  "rng_categorical", "rng_dirichlet", "rng_gamma",
                  "rng_gibbs_sample", "rng_hmc_sample", "rng_langevin_sample",
                  "rng_mala_sample", "rng_multinomial", "rng_permutation",
                  "rng_poisson", "rng_randint", "rng_truncated_normal",
                  "rng_key", "rng_split", "rng_fold_in", "rng_clone")},
    **{(op, "x86"): "tests/unit/test_x86_structured_compute_compiled.py"
       for op in ("attn_compressed_blocks", "attn_local_window_2d",
                  "attn_top_k_blocks", "linear_attn_state",
                  "lookahead_sparse_attention", "transpose")},
    **{(op, "rocm"): "tests/unit/test_rocm_structured_compute_compiled.py"
       for op in ("attn_compressed_blocks", "attn_local_window_2d",
                  "attn_top_k_blocks", "linear_attn_state",
                  "lookahead_sparse_attention", "power_attn", "transpose")},
    **{(op, "x86"): "tests/unit/test_x86_loss_compiled.py"
       for op in ("mse_loss", "mae_loss", "huber_loss", "smooth_l1_loss",
                  "log_cosh_loss")},
    **{(op, "x86"): "tests/unit/test_x86_binary_loss_compiled.py"
       for op in ("binary_cross_entropy_loss", "asymmetric_bce")},
    **{(op, "x86"): "tests/unit/test_x86_rl_loss_compiled.py"
       for op in ("ppo_policy_loss", "cispo_policy_loss", "grpo_policy_loss",
                  "normalize_group_advantages")},
    **{(op, "x86"): "tests/unit/test_x86_class_loss_compiled.py"
       for op in ("cross_entropy_loss", "kl_divergence", "js_divergence",
                  "focal_loss", "label_smoothed_cross_entropy", "z_loss")},
    **{(op, "x86"): "tests/unit/test_x86_metric_loss_compiled.py"
       for op in ("wasserstein_distance", "cosine_embedding_loss",
                  "contrastive_loss", "triplet_loss", "nt_xent_loss",
                  "info_nce_loss", "seq2seq_loss")},
    **{(op, "x86"): "tests/unit/test_x86_ebm_loss_compiled.py"
       for op in ("score_matching_loss", "denoising_score_matching_loss",
                  "implicit_score_matching_loss", "contrastive_divergence_loss",
                  "persistent_cd_loss", "ddpm_noise_pred_loss", "vlb_loss",
                  "load_balance_loss")},
    **{(op, "x86"): "tests/unit/test_x86_ebm_compute_compiled.py"
       for op in ("ebm_energy_quadratic", "ebm_inner_step", "ebm_refinement",
                  "ebm_self_verify")},
    ("ebm_langevin_step", "x86"):
        "tests/unit/test_x86_ebm_langevin_compiled.py",
    ("ebm_langevin_step", "rocm"):
        "tests/unit/test_rocm_ebm_langevin_compiled.py",
    **{(op, "x86"): "tests/unit/test_x86_fpquant_compiled.py"
       for op in ("quantize_fp8", "dequantize_fp8", "quantize_fp6",
                  "dequantize_fp6", "quantize_fp4", "dequantize_fp4")},
    **{(op, "x86"): "tests/unit/test_x86_intquant_compiled.py"
       for op in ("quantize_int8", "dequantize_int8", "quantize_int4",
                  "dequantize_int4", "fake_quantize")},
    **{(op, "x86"): "tests/unit/test_x86_pooling_vision_compiled.py"
       for op in ("max_pool", "avg_pool", "min_pool", "adaptive_pool",
                  "image_normalize")},
    **{(op, "x86"): "tests/unit/test_x86_nvfp4_compiled.py"
       for op in ("quantize_nvfp4", "dequantize_nvfp4")},
    **{(op, "x86"): "tests/unit/test_x86_reduce_foundation_compiled.py"
       for op in ("prod", "var", "std", "count_nonzero", "logsumexp",
                  "log_softmax", "softmax_safe", "sigmoid_safe")},
    **{(op, "x86"): "tests/unit/test_x86_fft_compiled.py"
       for op in ("fft", "ifft", "rfft", "irfft")},
    **{(op, "rocm"): "tests/unit/test_rocm_fft_compiled.py"
       for op in ("fft", "ifft", "rfft", "irfft")},
    **{(op, "x86"): "tests/unit/test_x86_spectral_compiled.py"
       for op in ("dct", "stft", "istft", "spectral_conv", "spectral_filter")},
    **{(op, "rocm"): "tests/unit/test_rocm_spectral_compiled.py"
       for op in ("dct", "stft", "istft", "spectral_conv", "spectral_filter")},
    **{(op, "x86"): "tests/unit/test_x86_sparse_compiled.py"
       for op in ("spmm_csr", "spmm_coo", "sddmm", "bsmm")},
    **{(op, "rocm"): "tests/unit/test_rocm_sparse_compiled.py"
       for op in ("spmm_csr", "spmm_coo", "sddmm", "bsmm")},
    **{(op, "x86"): "tests/unit/test_composite_helper_backend_parity.py"
       for op in ("memory_index_score", "msa_index_scores", "varlen_sdpa",
                  "score_combine")},
    **{(op, "rocm"): "tests/unit/test_composite_helper_backend_parity.py"
       for op in ("memory_index_score", "msa_index_scores", "varlen_sdpa",
                  "score_combine")},
    **{(op, "apple_gpu"): "tests/unit/test_apple_gpu_composite_helpers.py"
       for op in ("memory_index_score", "msa_index_scores", "varlen_sdpa",
                  "score_combine")},
    ("selective_ssm", "x86"): "tests/unit/test_x86_state_space_compiled.py",
    ("selective_ssm", "rocm"): "tests/unit/test_rocm_state_space_compiled.py",
    ("moe", "x86"): "tests/unit/test_x86_moe_compiled.py",
    ("moe", "rocm"): "tests/unit/test_rocm_moe_compiled.py",
    **{(op, "x86"): "tests/unit/test_x86_optimizer_compiled.py"
       for op in ("sgd", "momentum", "adam", "adamw", "lion", "nesterov")},
    **{(op, "rocm"): "tests/unit/test_rocm_optimizer_compiled.py"
       for op in ("sgd", "momentum", "adam", "adamw", "lion", "nesterov")},
    ("lamb", "x86"): "tests/unit/test_x86_lamb_compiled.py",
    ("lamb", "rocm"): "tests/unit/test_rocm_lamb_compiled.py",
    ("muon", "x86"): "tests/unit/test_x86_muon_compiled.py",
    ("muon", "rocm"): "tests/unit/test_rocm_muon_compiled.py",
    **{(op, "x86"): "tests/unit/test_x86_linalg_compiled.py"
       for op in ("cholesky", "tri_solve", "cholesky_solve")},
    **{(op, "rocm"): "tests/unit/test_rocm_linalg_compiled.py"
       for op in ("cholesky", "tri_solve", "cholesky_solve")},
    **{(op, "x86"): "tests/unit/test_x86_lu_qr_compiled.py" for op in ("lu", "qr")},
    **{(op, "rocm"): "tests/unit/test_rocm_lu_qr_compiled.py"
       for op in ("lu", "qr")},
    ("svd", "x86"): "tests/unit/test_x86_svd_compiled.py",
    ("svd", "rocm"): "tests/unit/test_rocm_svd_compiled.py",
    ("rmsnorm", "rocm"): "tests/unit/test_rocm_norm_compiled.py",
    ("rmsnorm_safe", "rocm"): "tests/unit/test_rocm_norm_compiled.py",
    ("layer_norm", "rocm"): "tests/unit/test_rocm_norm_compiled.py",
    **{(op, "x86"): "tests/unit/test_x86_normcompose_compiled.py"
       for op in ("group_norm", "instance_norm", "weight_norm")},
    **{(op, "rocm"): "tests/unit/test_rocm_normcompose_compiled.py"
       for op in ("group_norm", "instance_norm", "weight_norm")},
    ("grad_clip_norm", "x86"): "tests/unit/test_x86_grad_clip_compiled.py",
    ("grad_clip_norm", "rocm"): "tests/unit/test_rocm_grad_clip_compiled.py",
    ("gelu", "rocm"): "tests/unit/test_rocm_activation_compiled.py",
    ("silu", "rocm"): "tests/unit/test_rocm_activation_compiled.py",
    ("relu", "rocm"): "tests/unit/test_rocm_activation_compiled.py",
    ("silu_mul", "rocm"): "tests/unit/test_rocm_silu_mul_compiled.py",
    **{(op, "rocm"): "tests/unit/test_rocm_loss_compiled.py"
       for op in ("mse_loss", "mae_loss", "huber_loss", "smooth_l1_loss",
                  "log_cosh_loss")},
    **{(op, "rocm"): "tests/unit/test_rocm_binary_loss_compiled.py"
       for op in ("binary_cross_entropy_loss", "asymmetric_bce")},
    **{(op, "rocm"): "tests/unit/test_rocm_rl_loss_compiled.py"
       for op in ("ppo_policy_loss", "cispo_policy_loss", "grpo_policy_loss",
                  "normalize_group_advantages")},
    **{(op, "rocm"): "tests/unit/test_rocm_class_loss_compiled.py"
       for op in ("cross_entropy_loss", "kl_divergence", "js_divergence",
                  "focal_loss", "label_smoothed_cross_entropy", "z_loss")},
    **{(op, "rocm"): "tests/unit/test_rocm_metric_loss_compiled.py"
       for op in ("wasserstein_distance", "cosine_embedding_loss",
                  "contrastive_loss", "triplet_loss", "nt_xent_loss",
                  "info_nce_loss", "seq2seq_loss")},
    **{(op, "rocm"): "tests/unit/test_rocm_ebm_loss_compiled.py"
       for op in ("score_matching_loss", "denoising_score_matching_loss",
                  "implicit_score_matching_loss", "contrastive_divergence_loss",
                  "persistent_cd_loss", "ddpm_noise_pred_loss", "vlb_loss",
                  "load_balance_loss")},
    **{(op, "rocm"): "tests/unit/test_rocm_ebm_compute_compiled.py"
       for op in ("ebm_energy_quadratic", "ebm_inner_step", "ebm_refinement",
                  "ebm_self_verify")},
    **{(op, "rocm"): "tests/unit/test_rocm_fpquant_compiled.py"
       for op in ("quantize_fp8", "dequantize_fp8", "quantize_fp6",
                  "dequantize_fp6", "quantize_fp4", "dequantize_fp4",
                  "quantize_nvfp4", "dequantize_nvfp4")},
    **{(op, "rocm"): "tests/unit/test_rocm_intquant_compiled.py"
       for op in ("quantize_int8", "dequantize_int8", "quantize_int4",
                  "dequantize_int4", "fake_quantize")},
    **{(op, "rocm"): "tests/unit/test_rocm_pooling_vision_compiled.py"
       for op in ("max_pool", "avg_pool", "min_pool", "adaptive_pool",
                  "image_normalize")},
    **{(op, "rocm"): "tests/unit/test_rocm_reduce_foundation_compiled.py"
       for op in ("prod", "var", "std", "count_nonzero", "logsumexp",
                  "log_softmax", "softmax_safe", "sigmoid_safe")},
    **{(op, "rocm"): "tests/unit/test_rocm_unary_compiled.py"
       for op in ("exp", "log", "sqrt", "rsqrt", "reciprocal", "absolute",
                  "sign", "erf", "tanh", "sigmoid", "log1p", "expm1",
                  "softplus", "cos", "tan", "sinh", "cosh", "asin", "acos",
                  "atan", "erfc", "floor", "ceil", "round", "trunc")},
    **{(op, "rocm"): "tests/unit/test_rocm_binary_compiled.py"
       for op in ("sub", "div", "pow", "maximum", "minimum")},
    **{(op, "rocm"): "tests/unit/test_rocm_compare_compiled.py"
       for op in ("eq", "ne", "lt", "le", "gt", "ge")},
    **{(op, "rocm"): "tests/unit/test_rocm_logical_compiled.py"
       for op in ("logical_and", "logical_or", "logical_xor", "logical_not")},
    **{(op, "rocm"): "tests/unit/test_rocm_bitwise_compiled.py"
       for op in ("bitwise_and", "bitwise_or", "bitwise_xor", "bitwise_not")},
    ("popcount", "rocm"): "tests/unit/test_rocm_popcount_compiled.py",
    ("rope", "rocm"): "tests/unit/test_rocm_rope_compiled.py",
    ("alibi", "rocm"): "tests/unit/test_rocm_alibi_compiled.py",
    ("dequant_matmul", "rocm"): "tests/unit/test_rocm_dequant_gemm_compiled.py",
    ("dequant_grouped_gemm", "rocm"):
        "tests/unit/test_rocm_dequant_gemm_compiled.py",
    ("batched_gemm", "rocm"): "tests/unit/test_rocm_matmul_family_compiled.py",
    ("linear_general", "rocm"): "tests/unit/test_rocm_matmul_family_compiled.py",
    ("qkv_projection", "rocm"): "tests/unit/test_rocm_matmul_family_compiled.py",
    ("factorized_matmul", "rocm"):
        "tests/unit/test_rocm_matmul_family_compiled.py",
    ("einsum", "rocm"): "tests/unit/test_rocm_matmul_family_compiled.py",
    ("gated_attention", "rocm"):
        "tests/unit/test_rocm_exotic_attn_compiled.py",
    ("mla_decode", "rocm"): "tests/unit/test_rocm_exotic_attn_compiled.py",
    ("mla_decode_fused", "rocm"):
        "tests/unit/test_rocm_exotic_attn_compiled.py",
    ("deepseek_sparse_attention", "rocm"):
        "tests/unit/test_rocm_sparse_attn_compiled.py",
    ("msa_sparse_attention", "rocm"):
        "tests/unit/test_rocm_sparse_attn_compiled.py",
    ("hybrid_attention", "rocm"):
        "tests/unit/test_rocm_exotic_attn_compiled.py",
    ("gated_deltanet", "rocm"): "tests/unit/test_rocm_deltanet_compiled.py",
    ("kimi_delta_attention", "rocm"):
        "tests/unit/test_rocm_deltanet_compiled.py",
    ("modified_delta_attention", "rocm"):
        "tests/unit/test_rocm_deltanet_compiled.py",
    # nvidia_sm120 — consumer Blackwell (RTX 5070 Ti) warp-level mma.sync. The
    # shipped `tessera_nvidia_mma_gemm_{bf16,f16,tf32,e4m3,e5m2}` C-ABI symbols
    # (libtessera_nvidia_gemm.so) are dlopened and each dtype's GEMM compared to a
    # numpy/ml_dtypes reference. Skip-clean when no NVIDIA GPU / NVRTC. The
    # numerical-proof half of the nvidia_sm120 matmul `hardware_verified` row.
    ("matmul", "nvidia_sm120"): "tests/unit/test_nvidia_mma_runtime_symbol.py",
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
    ("attn_compressed_blocks", "apple_gpu"):
        "tests/unit/test_apple_gpu_masked_attn.py",
    ("attn_top_k_blocks", "apple_gpu"):
        "tests/unit/test_apple_gpu_sparse_attn.py",
    ("attn_local_window_2d", "apple_gpu"):
        "tests/unit/test_apple_gpu_sparse_attn.py",
    ("lookahead_sparse_attention", "apple_gpu"):
        "tests/unit/test_apple_gpu_lookahead_envelope.py",
    ("msa_sparse_attention", "apple_gpu"):
        "tests/unit/test_apple_gpu_sparse_attn.py",
    ("linear_attn_state", "apple_gpu"):
        "tests/unit/test_apple_gpu_linear_attn.py",
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
    ("quantized_matmul", "apple_gpu"):
        "tests/unit/test_apple_gpu_quantized_matmul.py",
    ("masked_categorical", "apple_gpu"):
        "tests/unit/test_apple_gpu_ldt_loss_ops.py",
    ("rmsnorm", "apple_gpu"): "tests/unit/test_apple_gpu_mpsgraph_lane.py",
    ("gelu", "apple_gpu"): "tests/unit/test_apple_gpu_mpsgraph_lane.py",
    ("transpose", "apple_gpu"): "tests/unit/test_apple_gpu_transpose.py",
    ("gather", "apple_gpu"): "tests/unit/test_apple_gpu_gather.py",
    ("slice", "apple_gpu"): "tests/unit/test_apple_gpu_slice.py",
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
    # ‖x‖² fused reduction (test_norm_squared_matches_python_reference):
    ("clifford_norm_squared", "apple_gpu"): "tests/unit/test_apple_gpu_clifford_msl_full.py",
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
    # EBM quadratic energy + refinement / inner-step (Langevin descent), vs the
    # numpy reference (test_energy_quadratic_jit_metal_runtime et al.):
    ("ebm_energy_quadratic", "apple_gpu"): "tests/unit/test_apple_gpu_ebm_lane.py",
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
# targets for NVIDIA SM_90+ (CUDA 13.3).
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
# ROCm 7.2.4.  Mirrors the NVIDIA tables above.
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
# x86 backend — two honest readiness tiers:
#
#   * matmul/gemm: AMX BF16 fused backend row.
#   * AVX-512 f32/i32/bool lanes: runtime-loaded compiled kernels from
#     libtessera_x86_elementwise.so, each backed by an execute-compare fixture.
#     These are ``compiled`` rather than ``hardware_verified`` because the
#     manifest does not claim per-op C ABI runtime_symbol contracts.
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
    # P10 — flash_attn x86 AVX-512 partner to the ROCm WMMA flash_attn. FA-style
    # streaming/online-softmax forward (tessera_x86_flash_attn_f32, runtime-
    # loaded; x86_flash_attn_compiled lane). f32; MHA core path (scale + causal);
    # validated on-device (execute_compare_fixture).
    "flash_attn": {
        "status": _FUSED_KERNEL_STATUS,
        "dtypes": ("fp32",),
        "notes": "AVX-512 FA-style online-softmax forward "
                 "(tessera_x86_flash_attn_f32, runtime-loaded; "
                 "x86_flash_attn_compiled lane; MHA scale+causal; f32)",
    },
    # P13 — conv2d / conv3d via im2col + the AVX-512 f32 GEMM (host im2col,
    # device GEMM; x86_conv_compiled lane). f32.
    **{op: {
        "status": _FUSED_KERNEL_STATUS,
        "dtypes": ("fp32",),
        "notes": f"{op} — im2col + the AVX-512 f32 GEMM (host lays out the "
                 "patch matrix; the device runs the GEMM; bias/groups on host; "
                 "x86_conv_compiled lane; f32, matches the conv reference)",
    } for op in ("conv2d", "conv3d")},
    # P11 — MLA latent-KV building blocks composed on the AVX-512 GEMM (compress/
    # expand = batched matmul) + the flash_attn lane (mla_decode_fused chains
    # compress→expand→flash_attn). x86_mla_compiled lane; f32; on-device fixture.
    **{op: {
        "status": _FUSED_KERNEL_STATUS,
        "dtypes": ("fp32",),
        "notes": f"{op} — MLA latent-KV composed on the AVX-512 GEMM "
                 "(+ flash_attn for mla_decode_fused; x86_mla_compiled lane; "
                 "f32)",
    } for op in ("latent_kv_compress", "latent_kv_expand_k",
                 "latent_kv_expand_v", "mla_decode_fused")},
    # P8 — scatter family (0-reduce indexed store): scatter/scatter_add/
    # scatter_reduce via the AVX-512 row-scatter kernel (tessera_x86_scatter_f32,
    # runtime-loaded; x86_scatter_compiled lane). f32; on-device fixture.
    **{op: {
        "status": _FUSED_KERNEL_STATUS,
        "dtypes": ("fp32",),
        "notes": f"{op} — 0-reduce indexed store via the AVX-512 row-scatter "
                 "kernel (tessera_x86_scatter_f32, runtime-loaded; "
                 "x86_scatter_compiled lane; f32)",
    } for op in ("scatter", "scatter_add", "scatter_reduce")},
    # S2 reduction family — hand-written AVX-512 row-reduction kernel
    # (tessera_x86_avx512_reduce_f32) the runtime ctypes-loads from
    # libtessera_x86_elementwise.so and executes (x86_reduce_compiled). f32,
    # NaN-propagating max/min; validated on-device (execute_compare_fixture).
    **{op: {
        "status": _FUSED_KERNEL_STATUS,
        "dtypes": ("fp32",),
        "notes": f"AVX-512 row reduction {op} (tessera_x86_avx512_reduce_f32, "
                 "runtime-loaded; x86_reduce_compiled lane)",
    } for op in ("sum", "mean", "max", "min", "amax", "amin")},
    # scan + argreduce — runtime-loaded x86 kernels (x86_scan_compiled /
    # x86_argreduce_compiled lanes), the CPU analog of the ROCm block-scan /
    # arg-reduce. f32; validated on-device (execute_compare_fixture).
    **{op: {
        "status": _FUSED_KERNEL_STATUS,
        "dtypes": ("fp32",),
        "notes": f"AVX-512 inclusive scan {op} (tessera_x86_avx512_scan_f32, "
                 "runtime-loaded; x86_scan_compiled lane)",
    } for op in ("cumsum", "cumprod", "cummax", "cummin")},
    **{op: {
        "status": _FUSED_KERNEL_STATUS,
        "dtypes": ("fp32",),
        "notes": f"AVX-512 {op} (tessera_x86_avx512_argreduce_f32, runtime-"
                 "loaded; x86_argreduce_compiled lane; i32 index output)",
    } for op in ("argmax", "argmin")},
    # S2 unary-math algebraic + rounding subset — hand-written AVX-512 kernel
    # (tessera_x86_avx512_unary_f32) the runtime ctypes-loads and executes
    # (x86_unary_compiled). f32; transcendentals stay numpy-reference on CPU.
    **{op: {
        "status": _FUSED_KERNEL_STATUS,
        "dtypes": ("fp32",),
        "notes": f"AVX-512 unary {op} (tessera_x86_avx512_unary_f32, direct "
                 "intrinsic; runtime-loaded; x86_unary_compiled lane)",
    } for op in ("sqrt", "rsqrt", "reciprocal", "absolute", "abs", "sign",
                 "floor", "ceil", "round", "trunc")},
    # S2 binary-arithmetic direct-intrinsic subset — hand-written AVX-512 kernel
    # (tessera_x86_avx512_binary_f32) the runtime ctypes-loads and executes
    # (x86_binary_compiled). f32; `pow` is transcendental → numpy-reference.
    # P2a adds add/mul (arithmetic) + mod/floor_div (floor-based, numpy semantics).
    **{op: {
        "status": _FUSED_KERNEL_STATUS,
        "dtypes": ("fp32",),
        "notes": f"AVX-512 binary {op} (tessera_x86_avx512_binary_f32, direct "
                 "intrinsic; runtime-loaded; x86_binary_compiled lane)",
    } for op in ("sub", "div", "maximum", "minimum",
                 "add", "mul", "mod", "floor_div")},
    # S2 comparison family — hand-written AVX-512 kernel
    # (tessera_x86_avx512_compare_f32) the runtime ctypes-loads and executes
    # (x86_compare_compiled). f32 in, bool out; NaN semantics match numpy.
    **{op: {
        "status": _FUSED_KERNEL_STATUS,
        "dtypes": ("fp32",),
        "notes": f"AVX-512 comparison {op} (tessera_x86_avx512_compare_f32, "
                 "runtime-loaded; x86_compare_compiled lane; bool output)",
    } for op in ("eq", "ne", "lt", "le", "gt", "ge")},
    # P2b — unary predicate family (isnan/isinf/isfinite), AVX-512 kernel
    # (tessera_x86_avx512_predicate_f32; x86_predicate_compiled). f32 in, bool out.
    **{op: {
        "status": _FUSED_KERNEL_STATUS,
        "dtypes": ("fp32",),
        "notes": f"AVX-512 predicate {op} (tessera_x86_avx512_predicate_f32, "
                 "runtime-loaded; x86_predicate_compiled lane; bool output)",
    } for op in ("isnan", "isinf", "isfinite")},
    # P2c — clamp/clip composed on the AVX-512 binary max/min kernel (no new
    # kernel; scalar bounds broadcast on host; x86_clamp_compiled lane).
    **{op: {
        "status": _FUSED_KERNEL_STATUS,
        "dtypes": ("fp32",),
        "notes": f"{op} — min(max(x, lo), hi) composed on the x86_binary_compiled "
                 "AVX-512 max/min kernel (either bound optional; matches np.clip)",
    } for op in ("clamp", "clip")},
    # (complex_* ops route through complex_manifest_for() — their fused
    # device-lane status is emitted there, not in this generic _X86 table.)
    # P2e — softcap composed on the AVX-512 transcendental tanh kernel (no new
    # kernel; scalar cap broadcast on host; x86_softcap_compiled lane).
    "softcap": {
        "status": _FUSED_KERNEL_STATUS,
        "dtypes": ("fp32",),
        "notes": "softcap — cap*tanh(x/cap) composed on the "
                 "x86_transcendental_compiled AVX-512 tanh kernel "
                 "(scalar cap on host; matches cap*tanh(x/cap))",
    },
    # P4 — 0-move / strided-copy lane: pad/cat/roll/flip/tile/repeat/stack via
    # the AVX-512 masked-gather kernel (host index map; device data movement).
    **{op: {
        "status": _FUSED_KERNEL_STATUS,
        "dtypes": ("fp32",),
        "notes": f"{op} — 0-move op realized by the AVX-512 masked-gather "
                 "kernel (tessera_x86_gather_f32, runtime-loaded; host index "
                 "map; x86_strided_compiled lane; f32, matches numpy)",
    } for op in ("pad", "cat", "roll", "flip", "tile", "repeat", "stack")},
    # P9 — sort / argsort / top_k via the AVX-512 bitonic sort network kernel
    # (tessera_x86_bitonic_sort_kv_f32; host pads to a power of two + flips).
    **{op: {
        "status": _FUSED_KERNEL_STATUS,
        "dtypes": ("fp32",),
        "notes": f"{op} — data-independent bitonic sort network "
                 "(tessera_x86_bitonic_sort_kv_f32, runtime-loaded; AVX-512 wide "
                 "stages + scalar tail; host pads to a power of two + flips for "
                 "descending; x86_sort_compiled lane; f32, matches numpy)",
    } for op in ("sort", "argsort", "top_k")},
    # P5 — conformal geometry: mobius / stereographic composed on the AVX-512
    # complex (mul/div) + binary (div) lanes (no new kernel; host orchestration).
    **{op: {
        "status": _FUSED_KERNEL_STATUS,
        "dtypes": ("fp32",),
        "notes": f"{op} — composed on the AVX-512 complex / binary lanes "
                 "(interleaved-f32; host orchestration; x86_conformal_compiled "
                 "lane; f32, matches tessera.complex)",
    } for op in ("mobius", "stereographic")},
    # P6 — device RNG: counter-based Philox-4x32-10 (tessera_x86_philox_uniform_f32)
    # produces the uniform bits; host applies the distribution transform.
    **{op: {
        "status": _FUSED_KERNEL_STATUS,
        "dtypes": ("fp32",),
        "notes": f"{op} — Philox-4x32-10 uniform RNG kernel "
                 "(tessera_x86_philox_uniform_f32, runtime-loaded) + host "
                 "transform; x86_rng_compiled lane; bit-exact vs "
                 "tessera.rng_device (a deterministic stream, distinct from the "
                 "host numpy-Generator path)",
    } for op in ("rng_uniform", "rng_normal", "dropout")},
    # P2e — atan2 composed on the AVX-512 transcendental atan kernel (no new
    # kernel; quadrant/sign logic on host; x86_atan2_compiled lane).
    "atan2": {
        "status": _FUSED_KERNEL_STATUS,
        "dtypes": ("fp32",),
        "notes": "atan2 — quadrant-aware atan2(y, x) composed on the "
                 "x86_transcendental_compiled AVX-512 atan kernel "
                 "(sign/quadrant on host; matches np.arctan2)",
    },
    # S2 logical family — hand-written AVX-512 kernel
    # (tessera_x86_avx512_logical_i8) the runtime ctypes-loads and executes
    # (x86_logical_compiled). i8 bool in/out; inputs normalized via != 0.
    **{op: {
        "status": _FUSED_KERNEL_STATUS,
        "dtypes": ("bool",),
        "notes": f"AVX-512 logical {op} (tessera_x86_avx512_logical_i8, "
                 "runtime-loaded; x86_logical_compiled lane; bool in/out)",
    } for op in ("logical_and", "logical_or", "logical_xor", "logical_not")},
    # S2 bitwise family — hand-written AVX-512 kernel
    # (tessera_x86_avx512_bitwise_i32) the runtime ctypes-loads and executes
    # (x86_bitwise_compiled). i32 in/out; full bit pattern (no normalization).
    **{op: {
        "status": _FUSED_KERNEL_STATUS,
        "dtypes": ("int32",),
        "notes": f"AVX-512 bitwise {op} (tessera_x86_avx512_bitwise_i32, "
                 "runtime-loaded; x86_bitwise_compiled lane; i32 in/out)",
    } for op in ("bitwise_and", "bitwise_or", "bitwise_xor", "bitwise_not")},
    # P2e — popcount: set-bit count per i32, unary, on the bitwise lane via the
    # AVX-512 VPOPCNTDQ instruction (_mm512_popcnt_epi32).
    "popcount": {
        "status": _FUSED_KERNEL_STATUS,
        "dtypes": ("int32",),
        "notes": "Standalone elementwise popcount — set-bit count per i32 "
                 "element via AVX-512 VPOPCNTDQ (tessera_x86_avx512_bitwise_i32, "
                 "runtime-loaded; x86_bitwise_compiled lane; i32 in/out)",
    },
    # Ternary select where(cond,a,b) — hand-written AVX-512 kernel
    # (tessera_x86_avx512_where_f32, _mm512_cmpneq_epi8_mask +
    # _mm512_mask_blend_ps) the runtime ctypes-loads (x86_where_compiled).
    # cond i8 normalized != 0, a/b/out f32.
    "where": {
        "status": _FUSED_KERNEL_STATUS,
        "dtypes": ("fp32",),
        "notes": "AVX-512 ternary select where(cond,a,b) "
                 "(tessera_x86_avx512_where_f32, runtime-loaded; "
                 "x86_where_compiled lane; cond i8 != 0, a/b/out f32)",
    },
    # S2 transcendental / activation family — hand-written AVX-512 vectorized
    # kernel (tessera_x86_avx512_transcendental_f32) the runtime ctypes-loads
    # (x86_transcendental_compiled). Cephes exp/log minimax cores + A&S erf;
    # activations compose. The CPU analog reaching ROCm math->ROCDL parity.
    # gelu uses the tanh approximation (matches the ROCm activation reference).
    **{op: {
        "status": _FUSED_KERNEL_STATUS,
        "dtypes": ("fp32",),
        "notes": f"AVX-512 transcendental/activation {op} "
                 "(tessera_x86_avx512_transcendental_f32, runtime-loaded; "
                 "x86_transcendental_compiled lane; Cephes exp/log cores + A&S "
                 "erf; f32, matches numpy 2e-5)",
    } for op in ("exp", "log", "tanh", "sigmoid", "silu", "gelu", "erf",
                 "softplus", "expm1", "log1p", "sin", "cos", "tan", "sinh",
                 "cosh", "asin", "acos", "atan", "erfc")},
    # P2e — lgamma: ln Γ(x) via the AVX-512 NR-Lanczos g=5 SIMD core (positive
    # domain) + std::lgamma fallback for reflection lanes (x<0.5).
    "lgamma": {
        "status": _FUSED_KERNEL_STATUS,
        "dtypes": ("fp32",),
        "notes": "AVX-512 lgamma — NR-Lanczos g=5 SIMD core "
                 "(tessera_x86_avx512_transcendental_f32, runtime-loaded; "
                 "x86_transcendental_compiled lane; std::lgamma fallback for "
                 "x<0.5; f32, matches math.lgamma rel 1e-4)",
    },
    # P2e — digamma: ψ(x) AVX-512 recurrence + asymptotic SIMD core (x>0) +
    # scalar digamma_d fallback for x<=0 (reflection / poles).
    "digamma": {
        "status": _FUSED_KERNEL_STATUS,
        "dtypes": ("fp32",),
        "notes": "AVX-512 digamma — recurrence + asymptotic SIMD core "
                 "(tessera_x86_avx512_transcendental_f32, runtime-loaded; "
                 "x86_transcendental_compiled lane; scalar fallback for x<=0; "
                 "f32, matches tessera.ops.digamma rel 1e-4)",
    },
    # Transcendental-backed BINARY ops — pow(a,b) (positive base) and
    # silu_mul(a,b)=silu(a)*b (SwiGLU gate-multiply); share the exp/log/sigmoid
    # cores. Runtime ctypes-loads them (x86_binary_math_compiled lane).
    **{op: {
        "status": _FUSED_KERNEL_STATUS,
        "dtypes": ("fp32",),
        "notes": f"AVX-512 transcendental-backed binary {op} "
                 "(tessera_x86_avx512_{pow,silu_mul}_f32, runtime-loaded; "
                 "x86_binary_math_compiled lane; f32, matches numpy 2e-5)",
    } for op in ("pow", "silu_mul")},
    # Row-reduction norm / softmax — unweighted rmsnorm / layer_norm and stable
    # softmax over the last axis (AVX-512 horizontal reduce). The CPU analog of
    # the ROCm warp-shuffle norm/softmax lanes (x86_norm_compiled /
    # x86_softmax_compiled).
    "rmsnorm": {
        "status": _FUSED_KERNEL_STATUS,
        "dtypes": ("fp32",),
        "notes": "AVX-512 unweighted rmsnorm row-reduction "
                 "(tessera_x86_avx512_rmsnorm_f32, runtime-loaded; "
                 "x86_norm_compiled lane; f32, matches numpy 2e-5)",
    },
    "rmsnorm_safe": {
        "status": _FUSED_KERNEL_STATUS,
        "dtypes": ("fp32",),
        "notes": "rmsnorm_safe (== rmsnorm, tighter eps default) rides the "
                 "AVX-512 rmsnorm row-reduction (tessera_x86_avx512_rmsnorm_f32; "
                 "x86_norm_compiled lane; f32, matches nn.functional)",
    },
    "layer_norm": {
        "status": _FUSED_KERNEL_STATUS,
        "dtypes": ("fp32",),
        "notes": "AVX-512 unweighted layer_norm row-reduction "
                 "(tessera_x86_avx512_layernorm_f32, runtime-loaded; "
                 "x86_norm_compiled lane; f32, matches numpy 2e-5)",
    },
    # P5 — group/instance/weight norm composed on the AVX-512 layer_norm (row
    # mean/var) + reduce (sum-of-squares) lanes; host does the reshape / divide.
    **{op: {
        "status": _FUSED_KERNEL_STATUS,
        "dtypes": ("fp32",),
        "notes": f"{op} composed on the AVX-512 layer_norm / reduce kernels "
                 "(no new kernel; host reshape/affine; x86_normcompose_compiled "
                 "lane; f32, matches nn.functional)",
    } for op in ("group_norm", "instance_norm", "weight_norm")},
    "grad_clip_norm": {
        "status": _FUSED_KERNEL_STATUS,
        "dtypes": ("fp32",),
        "notes": "Global gradient-norm clipping g*min(1, max_norm/||g||) — the "
                 "L2 sum-of-squares runs on the AVX-512 reduce kernel; host "
                 "sqrt + scale; x86_grad_clip_compiled lane; f32, matches "
                 "optim.clip_grad_norm",
    },
    "softmax": {
        "status": _FUSED_KERNEL_STATUS,
        "dtypes": ("fp32",),
        "notes": "AVX-512 stable softmax row-reduction "
                 "(tessera_x86_avx512_softmax_f32, runtime-loaded; "
                 "x86_softmax_compiled lane; f32, matches numpy 2e-5)",
    },
    "online_softmax": {
        "status": _FUSED_KERNEL_STATUS,
        "dtypes": ("fp32",),
        "notes": "Stateless online_softmax (== softmax) rides the AVX-512 stable "
                 "softmax row-reduction (tessera_x86_avx512_softmax_f32; "
                 "x86_softmax_compiled lane); the streaming-state form is declined "
                 "(Decision #21). f32, matches numpy",
    },
    # GEMM family — batched_gemm / linear_general / qkv_projection /
    # factorized_matmul / einsum, all on the AVX-512 f32 GEMM microkernel
    # (tessera_x86_avx512_gemm_f32) with reshape/batch/einsum in Python. The CPU
    # analog of the ROCm WMMA matmul-family lane (x86_matmul_family_compiled).
    **{op: {
        "status": _FUSED_KERNEL_STATUS,
        "dtypes": ("fp32",),
        "notes": f"AVX-512 f32 GEMM-family {op} "
                 "(tessera_x86_avx512_gemm_f32 microkernel + Python "
                 "reshape/batch/einsum; x86_matmul_family_compiled lane; f32)",
    } for op in ("batched_gemm", "linear_general", "qkv_projection",
                 "factorized_matmul", "einsum")},
    # Position encodings — interleaved-pair rope and the ALiBi bias generator.
    # The CPU analog of the ROCm rope/alibi lanes (x86_rope_compiled /
    # x86_alibi_compiled).
    "rope": {
        "status": _FUSED_KERNEL_STATUS,
        "dtypes": ("fp32",),
        "notes": "AVX-512 interleaved-pair rotary position embedding "
                 "(tessera_x86_avx512_rope_f32, runtime-loaded; deinterleave + "
                 "Cephes sincos; x86_rope_compiled lane; f32, matches numpy 2e-5)",
    },
    "alibi": {
        "status": _FUSED_KERNEL_STATUS,
        "dtypes": ("fp32",),
        "notes": "AVX-512 ALiBi positional-bias generator bias[h,i,j]="
                 "slope[h]*(j-i) (tessera_x86_avx512_alibi_f32, runtime-loaded; "
                 "x86_alibi_compiled lane; f32)",
    },
    # Softmax-attention family — multi_head / gqa / mqa / mla_decode, composed
    # from the AVX-512 GEMM (QK^T, probs*V) + the row-softmax kernel. The CPU
    # analog of the ROCm flash-attention family (x86_attention_compiled).
    **{op: {
        "status": _FUSED_KERNEL_STATUS,
        "dtypes": ("fp32",),
        "notes": f"AVX-512 softmax-attention {op} "
                 "(O=softmax(QK^T*scale[+causal])V on the f32 GEMM + row-softmax "
                 "kernels; x86_attention_compiled lane; f32)",
    } for op in ("multi_head_attention", "gqa_attention", "mqa_attention",
                 "mla_decode")},
    # P10 extras — sliding-window attention on the extended AVX-512 flash_attn
    # kernel (causal band of width W; GQA/softcap/bias also ride this lane).
    "attn_sliding_window": {
        "status": _FUSED_KERNEL_STATUS,
        "dtypes": ("fp32",),
        "notes": "AVX-512 sliding-window attention "
                 "(tessera_x86_flash_attn_ext_f32; causal band of width W; the "
                 "online-softmax FA forward with window/softcap/bias support; "
                 "x86_flash_attn_compiled lane; f32)",
    },
    # P10 scan-family — linear-attention backbone (linear_attn / power_attn /
    # retention) via the quadratic-parallel form (φQ·φKᵀ ⊙ causal ⊙ decay)@V on
    # the AVX-512 GEMM; feature map / mask / decay on host (x86_linear_attn_
    # compiled lane). The AVX-512 partner to the ROCm linear_attn lane.
    **{op: {
        "status": _FUSED_KERNEL_STATUS,
        "dtypes": ("fp32",),
        "notes": f"AVX-512 {op} — quadratic-parallel linear attention "
                 "(φ(Q)·φ(K)ᵀ ⊙ causal ⊙ decay)@V on two batched GEMMs; feature "
                 "map / mask / decay on host; x86_linear_attn_compiled lane; f32",
    } for op in ("linear_attn", "power_attn", "retention")},
    # DeltaNet / gated-delta linear attention — the hand-written AVX-512 causal
    # delta-rule sequential scan (avx512_deltanet_f32; x86_deltanet_compiled
    # lane), the x86 partner to the ROCm rocm_deltanet_compiled recurrence.
    **{op: {
        "status": _FUSED_KERNEL_STATUS,
        "dtypes": ("fp32",),
        "notes": f"AVX-512 {op} — causal delta-rule sequential scan "
                 "(avx512_deltanet_f32: per (b,h) a Dqk x Dv state over S with "
                 "erase/decay/beta/modified/gate); x86_deltanet_compiled lane; "
                 "f32, matches numpy _delta_attention_impl",
    } for op in ("gated_deltanet", "kimi_delta_attention",
                 "modified_delta_attention")},
    # P11 — NSA (DeepSeek native sparse attention): the sliding / compressed /
    # top-k branches all run their attention on the AVX-512 flash_attn kernels;
    # block compression / top-k selection / gather / gate blend on the host.
    "deepseek_sparse_attention": {
        "status": _FUSED_KERNEL_STATUS,
        "dtypes": ("fp32",),
        "notes": "AVX-512 NSA — sliding (windowed FA) + compressed-block (dense "
                 "FA over mean summaries) + top-k-block (host select/gather + "
                 "dense FA) branches blended by the gate; x86_nsa_compiled lane; "
                 "f32, matches the dense-masked reference",
    },
    "msa_sparse_attention": {
        "status": _FUSED_KERNEL_STATUS,
        "dtypes": ("fp32",),
        "notes": "AVX-512 MSA — exp-free index scoring + per-GQA-group top-k "
                 "block selection on host (bit-identical to the reference ops); "
                 "exact attend on the flash_attn kernel as dense attention with "
                 "a non-selected/causal additive -inf mask; x86_msa_compiled "
                 "lane; f32, matches the reference; dense-equivalence "
                 "(top_k==num_blocks) → dense GQA",
    },
    # Pointwise regression losses — per-element loss on the AVX-512 loss kernel
    # + none/mean/sum on the reduce kernel (x86_loss_compiled).
    **{op: {
        "status": _FUSED_KERNEL_STATUS,
        "dtypes": ("fp32",),
        "notes": f"AVX-512 pointwise loss {op} "
                 "(tessera_x86_avx512_pointwise_loss_f32 per-element + reduce "
                 "kernel; x86_loss_compiled lane; f32, matches numpy 2e-5)",
    } for op in ("mse_loss", "mae_loss", "huber_loss", "smooth_l1_loss",
                 "log_cosh_loss")},
    # Binary-cross-entropy losses — per-element on the AVX-512 binary-loss kernel
    # (stable softplus) + reduce kernel (x86_binary_loss_compiled).
    **{op: {
        "status": _FUSED_KERNEL_STATUS,
        "dtypes": ("fp32",),
        "notes": f"AVX-512 binary-cross-entropy loss {op} "
                 "(tessera_x86_avx512_binary_loss_f32 per-element + reduce "
                 "kernel; x86_binary_loss_compiled lane; f32, matches numpy 2e-5)",
    } for op in ("binary_cross_entropy_loss", "asymmetric_bce")},
    # RL policy losses — ppo/cispo/grpo core surrogate on the AVX-512 policy-loss
    # kernel; normalize_group_advantages on the layer_norm kernel over the group
    # axis (x86_rl_loss_compiled). KL/entropy/mask add-ons diagnose out.
    **{op: {
        "status": _FUSED_KERNEL_STATUS,
        "dtypes": ("fp32",),
        "notes": f"AVX-512 RL policy loss {op} "
                 "(tessera_x86_avx512_policy_loss_f32 surrogate / layer_norm "
                 "kernel; x86_rl_loss_compiled lane; core path, f32, numpy 2e-5)",
    } for op in ("ppo_policy_loss", "cispo_policy_loss", "grpo_policy_loss",
                 "normalize_group_advantages")},
    # Class-axis losses — exp/log on the AVX-512 transcendental kernel +
    # class-axis structure on the host + reduce kernel (x86_class_loss_compiled).
    **{op: {
        "status": _FUSED_KERNEL_STATUS,
        "dtypes": ("fp32",),
        "notes": f"AVX-512 class-axis loss {op} "
                 "(exp/log on the transcendental kernel + host class-axis "
                 "structure + reduce kernel; x86_class_loss_compiled lane; f32, "
                 "matches numpy 2e-4)",
    } for op in ("cross_entropy_loss", "kl_divergence", "js_divergence",
                 "focal_loss", "label_smoothed_cross_entropy", "z_loss")},
    **{op: {
        "status": _FUSED_KERNEL_STATUS,
        "dtypes": ("fp32",),
        "notes": f"AVX-512 metric/contrastive loss {op} "
                 "(reduce + exp/log kernels with host label/mask/sort/matrix "
                 "structure; x86_metric_loss_compiled lane; f32, matches "
                 "tessera.losses)",
    } for op in ("wasserstein_distance", "cosine_embedding_loss",
                 "contrastive_loss", "triplet_loss", "nt_xent_loss",
                 "info_nce_loss", "seq2seq_loss")},
    # P7 — EBM / diffusion losses composed on the AVX-512 binary + reduce kernels
    # (diff/square + reductions on device; argmax/one-hot/scalar scale on host;
    # x86_ebm_loss_compiled lane). f32.
    **{op: {
        "status": _FUSED_KERNEL_STATUS,
        "dtypes": ("fp32",),
        "notes": f"AVX-512 EBM/diffusion loss {op} (diff/square + reductions on "
                 "the binary + reduce kernels; host structure; "
                 "x86_ebm_loss_compiled lane; f32, matches numpy)",
    } for op in ("score_matching_loss", "denoising_score_matching_loss",
                 "implicit_score_matching_loss", "contrastive_divergence_loss",
                 "persistent_cd_loss", "ddpm_noise_pred_loss", "vlb_loss",
                 "load_balance_loss")},
    # (EBM energy/step-compute + Langevin ops route through ebm_manifest_for() —
    # their fused x86 status is emitted there, not in this generic table.)
    # Low-precision float quantization — quantize/dequantize fp8/fp6/fp4 on the
    # AVX-512 fpquant kernel (per-tensor symmetric grid-snap, fake-quant in f32
    # storage; x86_fpquant_compiled). The lane is f32 in/out (the fpN grid is
    # the quantization target, stored back as f32 — the dtypes the KERNEL runs).
    **{op: {
        "status": _FUSED_KERNEL_STATUS,
        "dtypes": ("fp32",),
        "notes": f"AVX-512 low-precision float {op} "
                 "(tessera_x86_avx512_fpquant_f32 grid-snap, per-tensor scale; "
                 "x86_fpquant_compiled lane; f32 fake-quant, matches reference)",
    } for op in ("quantize_fp8", "dequantize_fp8", "quantize_fp6",
                 "dequantize_fp6", "quantize_fp4", "dequantize_fp4")},
    # Integer quantization — scalar qparam selection + int8 container conversion
    # around AVX-512 round/max/min/mul kernels. int4 is signed int4 values stored
    # in int8 containers (not packed weights).
    **{op: {
        "status": _FUSED_KERNEL_STATUS,
        "dtypes": ("fp32", "int8") if "int8" in op else ("fp32",),
        "notes": f"AVX-512 integer quantization {op} "
                 "(round/max/min/mul on libtessera_x86_elementwise.so; "
                 "x86_intquant_compiled composite lane; matches reference)",
    } for op in ("quantize_int8", "dequantize_int8", "quantize_int4",
                 "dequantize_int4", "fake_quantize")},
    **{op: {
        "status": _FUSED_KERNEL_STATUS,
        "dtypes": ("fp32",),
        "notes": f"AVX-512 pooling {op} "
                 "(host window matrix + tessera_x86_avx512_reduce_f32 "
                 "max/min/mean; x86_pooling_compiled lane; matches reference)",
    } for op in ("max_pool", "avg_pool", "min_pool", "adaptive_pool")},
    "image_normalize": {
        "status": _FUSED_KERNEL_STATUS,
        "dtypes": ("fp32",),
        "notes": "AVX-512 image_normalize ((x-mean)/std) composed on "
                 "tessera_x86_avx512_binary_f32 sub/div with host layout and "
                 "per-channel broadcast; x86_image_affine_compiled lane",
    },
    # NVFP4 — block-scaled fp4 (E2M1 codes + per-block fp8-E4M3 scale) on the
    # AVX-512 fpquant kernel + host block structure (x86_nvfp4_compiled).
    **{op: {
        "status": _FUSED_KERNEL_STATUS,
        "dtypes": ("fp32",),
        "notes": f"AVX-512 NVFP4 block-scaled fp4 {op} (per-block fp8-E4M3 scale "
                 "+ E2M1 codes on the fpquant kernel; x86_nvfp4_compiled lane; "
                 "f32 fake-quant, matches the microscaling reference)",
    } for op in ("quantize_nvfp4", "dequantize_nvfp4")},
    # S2 reduce/stable-reduce foundation — prod (new AVX-512 reduce kind);
    # var/std/count_nonzero composed from the reduce kernel; logsumexp/
    # log_softmax/softmax_safe/sigmoid_safe (max-shifted reduce + exp/log lane).
    # x86 mirror of the ROCm reduce-foundation lane.
    "prod": {
        "status": _FUSED_KERNEL_STATUS,
        "dtypes": ("fp32",),
        "notes": "AVX-512 row reduction prod (tessera_x86_avx512_reduce_f32 "
                 "kind 4; x86_reduce_compiled lane; f32)",
    },
    **{op: {
        "status": _FUSED_KERNEL_STATUS,
        "dtypes": ("fp32",),
        "notes": f"AVX-512 statistical reduction {op} (composed from the reduce "
                 "kernel; x86_stat_reduce_compiled lane; f32)",
    } for op in ("var", "std", "count_nonzero")},
    **{op: {
        "status": _FUSED_KERNEL_STATUS,
        "dtypes": ("fp32",),
        "notes": f"AVX-512 stable reduction {op} (max-shifted reduce + exp/log "
                 "lane; x86_stable_reduce_compiled lane; f32)",
    } for op in ("logsumexp", "log_softmax", "softmax_safe", "sigmoid_safe")},
    # Spectral FFT (PR2) — fft/ifft/rfft/irfft over a power-of-two axis on the
    # AVX-512 radix-2 C2C kernel + r2c/c2r pack-unpack (x86_fft_compiled).
    **{op: {
        "status": _FUSED_KERNEL_STATUS,
        "dtypes": ("fp32",),
        "notes": f"AVX-512 FFT {op} (radix-2 C2C kernel for power-of-two; tiny "
                 "non-pow2 via the DFT-matrix on the GEMM; other non-pow2 via "
                 "Bluestein; x86_fft_compiled lane; complex64/f32, matches "
                 "np.fft)",
    } for op in ("fft", "ifft", "rfft", "irfft")},
    # Spectral composites (PR5) — dct/stft/istft/spectral_conv/spectral_filter
    # composed on the AVX-512 FFT lane (x86_spectral_compiled).
    **{op: {
        "status": _FUSED_KERNEL_STATUS,
        "dtypes": ("fp32",),
        "notes": f"Spectral {op} — composes the x86_fft_compiled radix-2 lane "
                 "(frame/window/overlap-add/pointwise on host; "
                 "x86_spectral_compiled lane; f32, matches np.fft)",
    } for op in ("dct", "stft", "istft", "spectral_conv", "spectral_filter")},
    # Sparse (PR) — genuinely sparse AVX-512 kernels (spmm_csr row-AXPY, sddmm
    # sampled-dot) + the GEMM microkernel for bsmm (x86_sparse_compiled).
    **{op: {
        "status": _FUSED_KERNEL_STATUS,
        "dtypes": ("fp32",),
        "notes": f"Sparse {op} — AVX-512 sparse kernel (spmm = row-wise AXPY over "
                 "CSR nonzeros, sddmm = sampled dense-dense dot; bsmm via the "
                 "GEMM microkernel; x86_sparse_compiled lane; f32, matches numpy)",
    } for op in ("spmm_csr", "spmm_coo", "sddmm", "bsmm")},
    **{op: {
        "status": _FUSED_KERNEL_STATUS,
        "dtypes": ("fp32",),
        "notes": f"Composite helper {op} — host shape/metadata logic composes "
                 "existing AVX-512-compatible matmul/attention/binary runtime "
                 "semantics; x86_composite_helper_compiled lane; f32, matches "
                 "the public op reference",
    } for op in ("memory_index_score", "msa_index_scores", "varlen_sdpa",
                 "score_combine")},
    # MoE compute (PR) — routed per-token expert GEMVs (top-1), AVX-512
    # (x86_moe_compiled). dispatch/combine = transport (mesh-gated), unchanged.
    "moe": {
        "status": _FUSED_KERNEL_STATUS,
        "dtypes": ("fp32",),
        "notes": "MoE compute (moe) — AVX-512 routed per-token expert GEMV kernel "
                 "(top-1; routing resolved on host, out_dim vectorized); "
                 "x86_moe_compiled lane; f32, matches numpy",
    },
    # Optimizer steps (P3) — fused per-parameter update, AVX-512
    # (x86_optimizer_compiled). adafactor (factored moments) is a follow-up.
    **{op: {
        "status": _FUSED_KERNEL_STATUS,
        "dtypes": ("fp32",),
        "notes": f"Optimizer {op} — AVX-512 fused per-parameter update kernel "
                 "(state m/v in-place; host computes the 1-β^t bias correction); "
                 "x86_optimizer_compiled lane; f32, matches the optim.py reference",
    } for op in ("sgd", "momentum", "adam", "adamw", "lion", "nesterov")},
    # P3 tail — LAMB: AVX-512 adam update + host per-tensor trust ratio.
    "lamb": {
        "status": _FUSED_KERNEL_STATUS,
        "dtypes": ("fp32",),
        "notes": "Optimizer lamb — AVX-512 adam kernel (lr=1/wd=0) + host "
                 "layer-wise trust ratio ‖p‖/‖update‖; x86_lamb_compiled lane; "
                 "f32, matches optim.lamb",
    },
    # P3 tail — Muon: momentum + orthogonal polar factor U·Vh from the AVX-512
    # device SVD (host does U@Vh + momentum/sgd). <2-D normalizes.
    "muon": {
        "status": _FUSED_KERNEL_STATUS,
        "dtypes": ("fp32",),
        "notes": "Optimizer muon — momentum then U·Vh orthogonalization via the "
                 "AVX-512 SVD kernel; host U@Vh + sgd; x86_muon_compiled lane; "
                 "f32, matches optim.muon",
    },
    # State-space (PR) — Mamba2 selective scan, AVX-512 fused single-pass scan
    # vectorized over the state dim N (x86_selective_ssm_compiled).
    "selective_ssm": {
        "status": _FUSED_KERNEL_STATUS,
        "dtypes": ("fp32", "fp16", "bf16"),
        "notes": "Mamba2 selective_ssm — AVX-512 fused selective-scan kernel "
                 "(single pass over S, vectorized over the state dim N, exp via "
                 "the Cephes core; x86_selective_ssm_compiled lane; matches the "
                 "numpy reference). f16/bf16 storage (vcvtph2ps / vcvtpbh_ps "
                 "load-convert, y truncated back; state+exp+accumulate f32). "
                 "Reverse-mode adjoint tessera_x86_selective_ssm_bwd_f32 "
                 "(sequential reverse scan vectorized over N) matches the "
                 "numpy VJP. Scalar-state A (D,) f32 routes through the "
                 "chunked-parallel SSD form (AVX-512 GEMM bmms)",
    },
    # Linalg PR-A — Cholesky + triangular solve (SPD/triangular family). Genuine
    # AVX-512 factorization/substitution kernels (x86_linalg_compiled).
    **{op: {
        "status": _FUSED_KERNEL_STATUS,
        "dtypes": ("fp32",),
        "notes": f"Linalg {op} — AVX-512 kernel (Cholesky–Banachiewicz "
                 "factorization / forward-back triangular substitution; "
                 "cholesky_solve = two triangular solves; batched; "
                 "x86_linalg_compiled lane; f32, matches numpy)",
    } for op in ("cholesky", "tri_solve", "cholesky_solve")},
    # Linalg PR-B — LU (partial pivot) + Householder QR. Genuine AVX-512
    # factorization kernels (x86_linalg_compiled).
    **{op: {
        "status": _FUSED_KERNEL_STATUS,
        "dtypes": ("fp32",),
        "notes": f"Linalg {op} — AVX-512 kernel (getrf partial-pivot LU / "
                 "Householder QR with vectorized rank-1 / reflector updates; "
                 "batched; x86_linalg_compiled lane; f32, matches numpy)",
    } for op in ("lu", "qr")},
    # Linalg PR-C — one-sided Jacobi SVD (x86_linalg_compiled).
    "svd": {
        "status": _FUSED_KERNEL_STATUS,
        "dtypes": ("fp32",),
        "notes": "Linalg svd — AVX-512 one-sided Jacobi (column-major working "
                 "copy, vectorized column dots/rotations, descending sort; wide "
                 "case via transpose; batched; x86_linalg_compiled lane; f32, "
                 "matches numpy by invariants)",
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

# P12 (S_SERIES_GAP_CLOSURE_PLAN) — the GA ops with a native x86 + ROCm device
# lane: table-driven Cl(3,0) bilinear products plus composite unary / field
# wrappers that execute through runtime.launch() and compare against the
# canonical flat Clifford shim.
_CLIFFORD_DEVICE_COMPILED = frozenset({
    "clifford_geometric_product",
    "clifford_wedge",
    "clifford_left_contraction",
    "clifford_inner",
    "clifford_rotor_sandwich",
    "clifford_reverse",
    "clifford_grade_involution",
    "clifford_conjugate",
    "clifford_grade_projection",
    "clifford_hodge_star",
    "clifford_ext_deriv",
    "clifford_codiff",
    "clifford_vec_deriv",
    "clifford_exp",
    "clifford_integral",
    "clifford_log",
    "clifford_norm",
    "clifford_norm_squared",
})


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
    _device = op_name in _CLIFFORD_DEVICE_COMPILED

    # x86 — native AVX-512 bilinear lane (P12) for the table-driven products;
    # Python GA reference for the rest.
    if _device:
        entries.append(BackendKernelEntry(
            target="x86",
            status=_FUSED_KERNEL_STATUS,
            dtypes=("fp32",),
            feature_flags=("clifford_dialect", "cayley_table", "avx512"),
            notes="Cl(3,0) bilinear product on the AVX-512 kernel "
                  "(tessera_x86_clifford_bilinear_f32; blade-major [8,n]; "
                  "compile-time Cayley table; x86_clifford_compiled lane)",
            execute_compare_fixture="tests/unit/test_x86_clifford_compiled.py",
        ))
    else:
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

    # ROCm — native compiled bilinear lane (P12) for the table-driven products;
    # planned (Phase H) for the rest.
    if _device:
        entries.append(BackendKernelEntry(
            target="rocm",
            status=_COMPILED_STATUS,
            dtypes=("fp32",),
            feature_flags=("clifford_dialect", "cayley_table", "hip_runtime"),
            notes="Cl(3,0) bilinear product on the COMPILER-GENERATED gfx1151 "
                  "kernel (generate-rocm-clifford-kernel; one thread per batch "
                  "element; triples unrolled at generation time; "
                  "rocm_clifford_compiled lane)",
            execute_compare_fixture="tests/unit/test_rocm_clifford_compiled.py",
            hipcc_version_min="7.2.4",
        ))
    else:
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


# P7 follow-up — EBM ops with a native x86 + ROCm device lane: the energy /
# step-compute ops (composed on the device binary + reduce lanes) and the
# Langevin sampling step (on-device Philox noise). For these, x86 = fused and
# ROCm = compiled (with the numerical fixture) instead of reference / Phase-H.
# Routed through ebm_manifest_for (NOT the generic _X86_KERNELS / _ROCM_COMPILED
# tables, which manifest_for never reaches for ebm_* names).
_EBM_DEVICE_COMPILED: dict[str, tuple[str, str]] = {
    # Quadratic energy 0.5*||x-y||^2 (the dominant EBT / diffusion energy). Both
    # the generic `ebm_energy` op (Apple opts its quadratic form into the same
    # kernel) and the tensor-clean `ebm_energy_quadratic` share ONE dedicated
    # fused per-row reduction: AVX-512 (double-accumulated) + a gfx1151
    # one-workgroup-per-row warp-shuffle sum-of-squares
    # (generate-rocm-ebm-energy-quadratic-kernel). tessera.ebm.energy_quadratic
    # routes x86 → ROCm → Apple → numpy.
    "ebm_energy": ("tests/unit/test_x86_ebm_energy_quadratic_compiled.py",
                   "tests/unit/test_rocm_ebm_energy_quadratic_compiled.py"),
    "ebm_energy_quadratic": (
        "tests/unit/test_x86_ebm_energy_quadratic_compiled.py",
        "tests/unit/test_rocm_ebm_energy_quadratic_compiled.py"),
    # EBT-tiny fused inference step (refine→energy→hard-argmin→gather) over B
    # batches of K≤256 candidates: AVX-512 (double-accumulated energy, first-min
    # tie-break) + a gfx1151 one-workgroup-per-batch kernel with a shared-memory
    # tree argmin (generate-rocm-ebm-ebt-tiny-kernel). Matches Apple's fused f32
    # ebt_tiny dispatch. tessera.ebm.ebt_tiny routes x86 → ROCm → Apple → numpy.
    "ebm_ebt_tiny": (
        "tests/unit/test_x86_ebm_ebt_tiny_compiled.py",
        "tests/unit/test_rocm_ebm_ebt_tiny_compiled.py"),
    "ebm_inner_step": ("tests/unit/test_x86_ebm_compute_compiled.py",
                       "tests/unit/test_rocm_ebm_compute_compiled.py"),
    "ebm_refinement": ("tests/unit/test_x86_ebm_compute_compiled.py",
                       "tests/unit/test_rocm_ebm_compute_compiled.py"),
    "ebm_self_verify": ("tests/unit/test_x86_ebm_compute_compiled.py",
                        "tests/unit/test_rocm_ebm_compute_compiled.py"),
    "ebm_langevin_step": ("tests/unit/test_x86_ebm_langevin_compiled.py",
                          "tests/unit/test_rocm_ebm_langevin_compiled.py"),
    # On-device-Philox Langevin step — SAME affine math as ebm_langevin_step but
    # the noise is drawn IN-KERNEL from (key, counter) via Philox-4x32-10 +
    # Box-Muller (no host noise buffer). The x86 `tessera_x86_ebm_langevin_philox_f32`
    # and ROCm `generate-rocm-ebm-langevin-kernel` already implement this exact
    # math (verified byte-tight vs the numpy Philox reference); tessera.ebm.
    # langevin_step_philox routes x86 → ROCm → Apple → numpy.
    "ebm_langevin_step_philox": (
        "tests/unit/test_x86_ebm_langevin_philox_compiled.py",
        "tests/unit/test_rocm_ebm_langevin_philox_compiled.py"),
    # Manifold Langevin STEP — reuses the native affine-Langevin kernel (host-drawn,
    # grade-projected noise as an input). x86 = AVX-512 affine kernel, ROCm =
    # generate-rocm-ebm-affine-langevin-kernel. Both the `_step` ops and their
    # legacy no-`_step` registry aliases (ebm_bivector_langevin / ebm_sphere_langevin)
    # resolve to the same native core. The `_sample` chain wrappers compose that
    # native per-step kernel in a host Markov loop (burn-in/thin/RNG on host) — like
    # the spectral dct/stft composites over the device FFT executor — so they are
    # native (compiled) too, proven by an on-device chain fixture in the geo tests.
    "ebm_bivector_langevin_step": (
        "tests/unit/test_x86_ebm_geo_langevin_compiled.py",
        "tests/unit/test_rocm_ebm_geo_langevin_compiled.py"),
    "ebm_bivector_langevin": (
        "tests/unit/test_x86_ebm_geo_langevin_compiled.py",
        "tests/unit/test_rocm_ebm_geo_langevin_compiled.py"),
    "ebm_bivector_langevin_sample": (
        "tests/unit/test_x86_ebm_geo_langevin_compiled.py",
        "tests/unit/test_rocm_ebm_geo_langevin_compiled.py"),
    # Sphere step reuses the SAME affine kernel: host tangent-projection + affine
    # core + host normalize (retract). No dedicated kernel — the affine core is
    # native, the projection/retract are host (like bivector's grade projection).
    "ebm_sphere_langevin_step": (
        "tests/unit/test_x86_ebm_geo_langevin_compiled.py",
        "tests/unit/test_rocm_ebm_geo_langevin_compiled.py"),
    "ebm_sphere_langevin": (
        "tests/unit/test_x86_ebm_geo_langevin_compiled.py",
        "tests/unit/test_rocm_ebm_geo_langevin_compiled.py"),
    "ebm_sphere_langevin_sample": (
        "tests/unit/test_x86_ebm_geo_langevin_compiled.py",
        "tests/unit/test_rocm_ebm_geo_langevin_compiled.py"),
    # Exact-partition (from_energies f32 fast path): a dedicated log-sum-exp
    # reduction kernel — AVX-512 (double-accumulated) + a gfx1151 warp-shuffle
    # reduction (generate-rocm-ebm-partition-kernel). Matches Apple's fused f32
    # partition lane. The EXACT f64 partition_function_exact stays host (it must
    # represent Z that overflows f32 and carries a 1e-10 contract).
    "ebm_partition_exact": (
        "tests/unit/test_x86_ebm_partition_compiled.py",
        "tests/unit/test_rocm_ebm_partition_compiled.py"),
    # Decode-init noise-apply (DFlash / EBM speculative-decode seeding): a
    # dedicated elementwise `out = base + std*noise` kernel — AVX-512
    # (double-accumulated) + a gfx1151 one-thread-per-element kernel
    # (generate-rocm-ebm-decode-init-kernel). Matches Apple's fused f32
    # decode-init lane. Host draws the unit-variance Gaussian so the fast path
    # shares the numpy reference's samples exactly.
    "ebm_decode_init": (
        "tests/unit/test_x86_ebm_decode_init_compiled.py",
        "tests/unit/test_rocm_ebm_decode_init_compiled.py"),
}


# EBM primitives with NO native kernel on ANY backend — host-orchestrated samplers
# over a user ``energy_fn`` (an importance-sampling / annealed-IS loop whose inner
# energy is user code). Apple GPU — the most-developed GPU backend — is itself
# ``reference`` for these (it found no kernel to fuse), which is the tell that no
# kernel exists or is plannable: the Python reference is the *terminal* execution
# path, not a gap awaiting one. Marking them ``planned`` on ROCm/NVIDIA overstated
# the open backend-kernel work in the automated Backend-Proof tally; they are
# honestly ``reference`` on every target, exactly like x86/apple_cpu/apple_gpu.
#
# The ONLY EBM ops honestly `reference` on every target (this frozenset) are the
# two user-energy_fn samplers above — everything else in the family now has a real
# x86+ROCm lane. Graduated OUT of "planned" and into _EBM_DEVICE_COMPILED: the
# f32 log-sum-exp reduction (`ebm_partition_exact`), base+std*noise decode-init
# (`ebm_decode_init`), per-row 0.5*||x-y||^2 quadratic energy (`ebm_energy`),
# fused EBT-tiny (`ebm_ebt_tiny`), the on-device-Philox Langevin step
# (`ebm_langevin_step_philox`), the manifold Langevin steps + their no-`_step`
# aliases (`ebm_{bivector,sphere}_langevin{,_step}`), and — following the spectral
# composite precedent (host loop over a native device executor) — the manifold
# `_sample` chains (`ebm_{bivector,sphere}_langevin_sample`), each proven by an
# on-device chain fixture (see _EBM_DEVICE_COMPILED above).
_EBM_USER_FUNCTION_OPS: frozenset[str] = frozenset({
    "ebm_partition_monte_carlo",  # importance-sampled Z over a user energy_fn
    "ebm_partition_ais",          # annealed IS over a user energy_fn + host schedule
})


def ebm_manifest_for(op_name: str) -> list[BackendKernelEntry]:
    """Return the backend manifest entries for an ``ebm_*`` primitive.

    Status semantics:
      - ``x86`` + ``apple_cpu``: ``reference`` (Python implementation
        in ``tessera.ebm.*`` runs on every CPU host), EXCEPT the P7
        device-lane ops (``_EBM_DEVICE_COMPILED``) whose x86 slot is
        ``fused`` (native AVX-512 compute / Langevin kernels).
      - ``apple_gpu``: ``fused`` for primitives in
        ``_EBM_APPLE_GPU_FUSED``; ``planned`` for everything else.
      - ``nvidia_sm90``: ``planned`` (Phase G).  ``rocm``: ``compiled``
        for the P7 device-lane ops, else ``planned`` (Phase H).

    The benchmark driver uses this to label each EBM row's backend
    column instead of carrying a row-local ``python_reference_only``
    string.
    """
    if op_name not in _EBM_PRIMITIVES:
        return []
    entries: list[BackendKernelEntry] = []
    device = _EBM_DEVICE_COMPILED.get(op_name)
    # A user-function op has no possible native kernel on ANY backend — the GPU
    # slots are honestly `reference` (the numpy path is terminal), not `planned`.
    user_fn = op_name in _EBM_USER_FUNCTION_OPS
    _user_fn_note = ("host-orchestrated sampler over a user energy_fn — no native "
                     "kernel on any backend (Apple GPU is reference too); the "
                     "Python reference is the terminal execution path, not a gap "
                     "awaiting a kernel")

    # CPU targets — Python reference path on every host; the P7 device-lane ops
    # carry a native AVX-512 x86 kernel (fused) instead.
    if device is not None:
        entries.append(BackendKernelEntry(
            target="x86",
            status=_FUSED_KERNEL_STATUS,
            dtypes=("fp32",),
            feature_flags=("ebm_namespace", "avx512"),
            notes="AVX-512 EBM device lane — diff/square/reduce (compute), "
                  "Philox Box-Muller (langevin), stable log-sum-exp "
                  "(partition), base+std*noise (decode-init), or the fused "
                  "refine→energy→argmin→gather EBT-tiny pipeline on the "
                  "runtime-loaded kernels; see the execute-compare fixture",
            execute_compare_fixture=device[0],
        ))
    else:
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
    elif user_fn:
        entries.append(BackendKernelEntry(
            target="apple_gpu",
            status=_REFERENCE_STATUS,
            dtypes=_EBM_CPU_DTYPES,
            feature_flags=("ebm_namespace", "numpy_reference"),
            notes=_user_fn_note,
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

    # NVIDIA — a user-function op is `reference` (no kernel possible); else planned
    # (gated on Phase G).
    entries.append(BackendKernelEntry(
        target="nvidia_sm90",
        status=_REFERENCE_STATUS if user_fn else _PLANNED_STATUS,
        dtypes=_EBM_CPU_DTYPES if user_fn else _EBM_PLANNED_GPU_DTYPES,
        feature_flags=(("ebm_namespace", "numpy_reference") if user_fn
                       else ("ebm_namespace",)),
        notes=_user_fn_note if user_fn else "Gated on Phase G",
    ))
    # ROCm — compiled device lane for the P7 ops; `reference` for the user-function
    # ops (no kernel possible); else planned (Phase H).
    if device is not None:
        entries.append(BackendKernelEntry(
            target="rocm",
            status=_COMPILED_STATUS,
            dtypes=("fp32",),
            feature_flags=("ebm_namespace", "hip_runtime"),
            notes="COMPILER-GENERATED gfx1151 EBM device lane — diff/square/"
                  "reduce (compute), Philox Box-Muller (langevin), warp-shuffle "
                  "log-sum-exp (partition), base+std*noise (decode-init), or the "
                  "fused refine→energy→argmin→gather EBT-tiny pipeline "
                  "(one-workgroup-per-batch, shared-memory tree argmin); "
                  "executes via runtime.launch(); see the execute-compare "
                  "fixture",
            execute_compare_fixture=device[1],
            hipcc_version_min="7.2.4",
        ))
    elif user_fn:
        entries.append(BackendKernelEntry(
            target="rocm",
            status=_REFERENCE_STATUS,
            dtypes=_EBM_CPU_DTYPES,
            feature_flags=("ebm_namespace", "numpy_reference"),
            notes=_user_fn_note,
        ))
    else:
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

# P5 (2026-06-28) — the 9 pointwise complex ops that now ship a REAL device lane
# (interleaved-f32 composed on the AVX-512 / gfx1151 transcendental / unary /
# binary / atan2 kernels; runtime x86_complex_compiled / rocm_complex_compiled).
# complex_manifest_for() emits these as fused (x86) / compiled (rocm).
_COMPLEX_DEVICE_COMPILED: frozenset[str] = frozenset({
    "complex_mul", "complex_div", "complex_conjugate", "complex_abs",
    "complex_arg", "complex_exp", "complex_log", "complex_sqrt", "complex_pow",
    "check_cauchy_riemann", "conformal_jacobian",
    "conformal_energy_on_sphere", "cross_ratio", "dbar", "dz",
    "is_concyclic", "laplacian_2d", "mobius_from_three_points",
})


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
    # P5 (2026-06-28): the 9 pointwise complex ops now ship REAL device lanes —
    # interleaved-f32 composed on the AVX-512 / gfx1151 transcendental / unary /
    # binary / atan2 kernels (runtime x86_complex_compiled / rocm_complex_compiled).
    # Emit them as fused (x86) / compiled (rocm) HERE — manifest_for() routes all
    # complex_* ops through this function before the generic _X86_KERNELS /
    # _ROCM_COMPILED tables, so the compiled status must live here to be seen by
    # support / conformance / gating.
    _complex_compiled = op_name in _COMPLEX_DEVICE_COMPILED
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
        status=_FUSED_KERNEL_STATUS if _complex_compiled else _REFERENCE_STATUS,
        dtypes=("fp32",),
        feature_flags=(("complex_namespace", "interleaved_f32", "avx512")
                       if _complex_compiled
                       else ("complex_namespace", "numpy_reference")),
        notes=("interleaved-f32 complex op composed on the AVX-512 "
               "transcendental/unary/binary/atan2 kernels "
               "(x86_complex_compiled lane)" if _complex_compiled
               else "Python complex reference (tessera.complex.*)"),
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
    elif in_long_tail and _complex_compiled:
        # P6 (2026-07-09) — the long-tail ops now ship a REAL Apple GPU lane
        # (apple_gpu_complex_compiled): the 9 pointwise ops compose interleaved-
        # f32 on the MSL unary/binary/atan2 lanes; the geometric/certificate ops
        # reuse the tessera.complex reference (host structure — the same path
        # x86/ROCm take). Direct execute/compare, not a bespoke fused MSL kernel;
        # fp16/bf16 native storage stays the M7 follow-up.
        entries.append(BackendKernelEntry(
            target="apple_gpu",
            status=_COMPILED_STATUS,
            dtypes=("fp32",),
            feature_flags=("complex_namespace", "interleaved_f32", "msl", "metal"),
            notes=(
                "interleaved-f32 complex op composed on the Apple GPU unary/"
                "binary/atan2 lanes (apple_gpu_complex_compiled); geometric ops "
                "reuse the tessera.complex reference. fp16/bf16 native storage "
                "remains the M7 follow-up."
            ),
            execute_compare_fixture="tests/unit/test_apple_gpu_complex_compiled.py",
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
    # ROCm — the 9 pointwise ops ship a compiled device lane (interleaved-f32
    # composed on the gfx1151 unary/binary/atan2 kernels, rocm_complex_compiled);
    # the long-tail stays planned (Phase H).
    if _complex_compiled:
        entries.append(BackendKernelEntry(
            target="rocm",
            status=_COMPILED_STATUS,
            dtypes=("fp32",),
            feature_flags=("complex_namespace", "interleaved_f32", "hip_runtime"),
            notes=(
                "interleaved-f32 complex op composed on the gfx1151 "
                "unary/binary/atan2 kernels (rocm_complex_compiled lane)"
            ),
            execute_compare_fixture="tests/unit/test_rocm_complex_compiled.py",
            hipcc_version_min="7.2.4",
        ))
    else:
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
            hipcc_version_min="7.2.4",
        ))
    return entries


# A1 (2026-06-18) — GEMM-family ops that carry a unified MMA descriptor.
# Mirrors ``primitive_coverage._ROCM_MMA_OPS`` (kept as a local literal so this
# module stays importable without pulling in the audit registry).
_ROCM_MMA_OP_NAMES: frozenset[str] = frozenset({
    "matmul", "batched_gemm", "grouped_gemm", "dequant_matmul",
    "dequant_grouped_gemm",
    "linear_general", "qkv_projection", "factorized_matmul",
})


# Single-GPU closeout C.4 (2026-07-01): planned/partial compute-tail
# primitives that are outside OP_SPECS still need explicit backend pathway
# ownership. These rows do not claim native kernels; they pin the conservative
# CPU reference execution lane so the closeout dashboard no longer treats them
# as ownerless backend work.
_SINGLE_GPU_COMPUTE_REFERENCE_OPS: frozenset[str] = frozenset({
    # rng
    "rng_bernoulli", "rng_beta", "rng_categorical", "rng_dirichlet",
    "rng_clone", "rng_fold_in", "rng_key", "rng_split",
    "rng_gamma", "rng_gibbs_sample", "rng_hmc_sample", "rng_langevin_sample",
    "rng_mala_sample", "rng_multinomial", "rng_permutation", "rng_poisson",
    "rng_randint", "rng_truncated_normal",
    # loss
    "ctc_loss",
    # vision / pooling
    "center_crop", "image_resize", "interpolate",
    # recurrent / model / layout
    "bidirectional_scan", "conv1d", "conv_transpose", "gru_cell", "lstm_cell",
    "lora_linear", "patchify", "pixel_shuffle", "pixel_unshuffle",
    "simple_rnn_cell",
    "arange", "cast", "masked_fill", "mor_partition", "mor_router",
    "mor_scatter", "pack", "rearrange", "rope_merge", "rope_split",
    "tile_view", "unpack",
    # smaller compute tails
    "cross_attention", "depthwise_conv1d", "edm_loss_weight",
    "edm_precondition", "factorized_pos_emb", "masked_scatter",
    "memory_read", "mrope_2d", "online_softmax_state", "perceiver_resampler",
    "spectral_norm",
})

_SINGLE_GPU_COMPUTE_REFERENCE_DTYPES: Mapping[str, tuple[str, ...]] = {
    "dequantize_int4": ("fp32",),
    "dequantize_int8": ("fp32", "int8"),
    "fake_quantize": ("fp32",),
    "quantize_int4": ("fp32",),
    "quantize_int8": ("fp32", "int8"),
    "rng_bernoulli": ("bool", "fp32"),
    "rng_categorical": ("int32", "fp32"),
    "rng_multinomial": ("int32", "fp32"),
    "rng_permutation": ("int32",),
    "rng_randint": ("int32",),
}


_STRUCTURED_COMPUTE_COMPILED_OPS: frozenset[str] = frozenset({
    "ctc_loss",
    "center_crop", "image_resize", "interpolate",
    "patchify", "pixel_shuffle", "pixel_unshuffle",
    "conv1d", "conv_transpose", "lora_linear",
    "gru_cell", "lstm_cell", "simple_rnn_cell",
    "depthwise_conv1d",
    "cross_attention", "perceiver_resampler",
    "bidirectional_scan",
    "arange", "cast", "masked_fill", "mor_partition", "mor_router",
    "mor_scatter", "pack", "rearrange", "rope_merge", "rope_split",
    "tile_view", "unpack",
    "edm_loss_weight", "edm_precondition", "factorized_pos_emb",
    "masked_scatter", "memory_read", "mrope_2d", "online_softmax_state",
    "spectral_norm",
})

_STRUCTURED_COMPUTE_CUDA_PLANNED_OPS: frozenset[str] = frozenset({
    "edm_loss_weight", "edm_precondition", "factorized_pos_emb",
    "masked_scatter", "memory_read", "mrope_2d", "online_softmax_state",
    "spectral_norm",
})

_RNG_DISTRIBUTION_COMPILED_OPS: frozenset[str] = frozenset({
    "rng_bernoulli", "rng_beta", "rng_categorical", "rng_clone",
    "rng_dirichlet", "rng_fold_in", "rng_gamma", "rng_gibbs_sample",
    "rng_hmc_sample", "rng_key", "rng_langevin_sample", "rng_mala_sample",
    "rng_multinomial", "rng_permutation", "rng_poisson", "rng_randint",
    "rng_split", "rng_truncated_normal",
})

_STRUCTURED_COMPUTE_OVERLAY_OPS: frozenset[str] = frozenset({
    "attn_compressed_blocks", "attn_local_window_2d", "attn_top_k_blocks",
    "linear_attn_state", "lookahead_sparse_attention", "power_attn",
    "transpose",
})


def _overlay_structured_compute_entries(
    op_name: str,
    entries: list[BackendKernelEntry],
) -> list[BackendKernelEntry]:
    if op_name not in _STRUCTURED_COMPUTE_OVERLAY_OPS:
        return entries
    dtypes = _SINGLE_GPU_COMPUTE_REFERENCE_DTYPES.get(op_name, ("fp32",))
    notes = (
        "Single-GPU structured-compute executable lane: runtime.launch() "
        "dispatches target-specific artifacts for attention/layout structure. "
        "This is direct execute/compare evidence, not a claim of a bespoke "
        "fused kernel."
    )
    out = [e for e in entries if e.target not in {"x86", "rocm"}]
    out.append(BackendKernelEntry(
        target="x86",
        status=_COMPILED_STATUS,
        dtypes=dtypes,
        feature_flags=("avx512",),
        notes=notes + " Executes via x86_structured_compute_compiled.",
        execute_compare_fixture="tests/unit/test_x86_structured_compute_compiled.py",
    ))
    out.append(BackendKernelEntry(
        target="rocm",
        status=_COMPILED_STATUS,
        dtypes=dtypes,
        feature_flags=("hip_runtime", "structured_compute"),
        notes=notes + " Executes via rocm_structured_compute_compiled.",
        execute_compare_fixture="tests/unit/test_rocm_structured_compute_compiled.py",
        hipcc_version_min="7.2.4",
    ))
    return out


def _single_gpu_compute_reference_manifest_for(
    op_name: str,
) -> list[BackendKernelEntry]:
    if op_name not in _SINGLE_GPU_COMPUTE_REFERENCE_OPS:
        return []
    dtypes = _SINGLE_GPU_COMPUTE_REFERENCE_DTYPES.get(op_name, ("fp32",))
    if op_name in _RNG_DISTRIBUTION_COMPILED_OPS:
        notes = (
            "Single-GPU RNG executable lane: runtime.launch() dispatches the "
            "compiler-visible RNGKey/Philox sampler contract. Uniform/normal "
            "exercise backend Philox directly; distribution/key-state ops are "
            "structured transforms over that deterministic stream."
        )
        return [
            BackendKernelEntry(
                target="cpu",
                status=_REFERENCE_STATUS,
                dtypes=dtypes,
                feature_flags=("numpy", "philox", "reference_execution"),
                notes="RNGKey Philox reference path",
            ),
            BackendKernelEntry(
                target="x86",
                status=_COMPILED_STATUS,
                dtypes=dtypes,
                feature_flags=("avx512",),
                notes=notes + " Executes via x86_rng_compiled.",
                execute_compare_fixture="tests/unit/test_x86_rng_compiled.py",
            ),
            BackendKernelEntry(
                target="apple_cpu",
                status=_REFERENCE_STATUS,
                dtypes=dtypes,
                feature_flags=("numpy", "philox", "reference_execution"),
                notes="RNGKey Philox reference path",
            ),
            BackendKernelEntry(
                target="apple_gpu",
                status=_COMPILED_STATUS,
                dtypes=dtypes,
                feature_flags=("metal", "philox", "rng_distribution"),
                notes=notes + " Executes via apple_gpu_rng_compiled "
                              "(Philox reference core; Apple ships no device "
                              "Philox kernel).",
                execute_compare_fixture="tests/unit/test_apple_gpu_rng_compiled.py",
            ),
            BackendKernelEntry(
                target="rocm",
                status=_COMPILED_STATUS,
                dtypes=dtypes,
                feature_flags=("hip_runtime", "philox", "rng_distribution"),
                notes=notes + " Executes via rocm_rng_compiled.",
                execute_compare_fixture="tests/unit/test_rocm_rng_compiled.py",
                hipcc_version_min="7.2.4",
            ),
        ]
    if op_name in _STRUCTURED_COMPUTE_COMPILED_OPS:
        notes = (
            "Single-GPU structured-compute executable lane: runtime.launch() "
            "dispatches target-specific artifacts for host-structured dynamic "
            "programming, image/layout indexing, conv/recurrent cells, and "
            "streaming depthwise convolution. This is direct execute/compare "
            "evidence, not a claim of a bespoke fused kernel."
        )
        entries = [
            BackendKernelEntry(
                target="cpu",
                status=_REFERENCE_STATUS,
                dtypes=dtypes,
                feature_flags=("numpy", "reference_execution"),
                notes="numpy reference path for structured compute",
            ),
            BackendKernelEntry(
                target="x86",
                status=_COMPILED_STATUS,
                dtypes=dtypes,
                feature_flags=("avx512",),
                notes=notes + " Executes via x86_structured_compute_compiled.",
                execute_compare_fixture="tests/unit/test_x86_structured_compute_compiled.py",
            ),
            BackendKernelEntry(
                target="apple_cpu",
                status=_REFERENCE_STATUS,
                dtypes=dtypes,
                feature_flags=("numpy", "reference_execution"),
                notes="numpy reference path for structured compute",
            ),
        ]
        apple_gpu = _APPLE_GPU_KERNELS.get(op_name)
        if apple_gpu is not None:
            entries.append(BackendKernelEntry(
                target="apple_gpu",
                status=str(apple_gpu["status"]),
                dtypes=tuple(apple_gpu["dtypes"]),
                feature_flags=("metal", "mps", "msl"),
                notes=str(apple_gpu.get("notes", "")),
                runtime_symbol=apple_gpu.get("runtime_symbol"),
                shape_envelope=apple_gpu.get("shape_envelope"),
                # A ``compiled`` structured-compute row carries its numerical
                # proof (execute_compare_fixture) directly — mirrors the x86/
                # ROCm entries above. Required at construction for compiled.
                execute_compare_fixture=apple_gpu.get("execute_compare_fixture"),
                benchmark_json=apple_gpu.get("benchmark_json"),
                benchmark_metadata=_APPLE_GPU_HOT_PATH_METADATA.get(op_name),
            ))
        entries.extend([
            BackendKernelEntry(
                target="rocm",
                status=_COMPILED_STATUS,
                dtypes=dtypes,
                feature_flags=("hip_runtime", "structured_compute"),
                notes=notes + " Executes via rocm_structured_compute_compiled.",
                execute_compare_fixture="tests/unit/test_rocm_structured_compute_compiled.py",
                hipcc_version_min="7.2.4",
            ),
        ])
        if op_name in _STRUCTURED_COMPUTE_CUDA_PLANNED_OPS:
            for target_name, flags, arch_min in (
                ("nvidia_sm80", ("cuda", "wmma", "planned_kernel"), "sm_80"),
                ("nvidia_sm90", ("cuda", "wgmma", "planned_kernel"), "sm_90a"),
                ("nvidia_sm100", ("cuda", "tcgen05", "planned_kernel"), "sm_100a"),
                ("nvidia_sm120", ("cuda", "tcgen05", "planned_kernel"), "sm_120a"),
            ):
                entries.append(BackendKernelEntry(
                    target=target_name,
                    status=_PLANNED_STATUS,
                    dtypes=dtypes,
                    feature_flags=flags,
                    notes=(
                        "Single-GPU closeout compute-tail CUDA owner: planned "
                        "kernel lane. X86/ROCm have compiled structured-compute "
                        "proof; CUDA remains explicitly open."
                    ),
                    cuda_arch_min=arch_min,
                    nvcc_version_min="13.3",
                ))
        return entries
    notes = (
        "Single-GPU closeout compute-tail reference owner: Python/numpy "
        "execution path. Native fused kernel remains tracked separately."
    )
    entries = [
        BackendKernelEntry(
            target="cpu",
            status=_REFERENCE_STATUS,
            dtypes=dtypes,
            feature_flags=("numpy", "reference_execution"),
            notes=notes,
        ),
        BackendKernelEntry(
            target="x86",
            status=_REFERENCE_STATUS,
            dtypes=dtypes,
            feature_flags=("numpy", "reference_execution"),
            notes=notes,
        ),
        BackendKernelEntry(
            target="apple_cpu",
            status=_REFERENCE_STATUS,
            dtypes=dtypes,
            feature_flags=("numpy", "reference_execution"),
            notes=notes,
        ),
        BackendKernelEntry(
            target="rocm",
            status=_PLANNED_STATUS,
            dtypes=dtypes,
            feature_flags=("hip", "rocm", "planned_kernel"),
            notes=(
                "Single-GPU closeout compute-tail ROCm owner: HIP backend "
                "planned lane. No compiled hsaco/runtime proof claimed yet."
            ),
        ),
    ]
    for target_name, flags, arch_min in (
        ("nvidia_sm80", ("cuda", "wmma", "planned_kernel"), "sm_80"),
        ("nvidia_sm90", ("cuda", "wgmma", "planned_kernel"), "sm_90a"),
        ("nvidia_sm100", ("cuda", "tcgen05", "planned_kernel"), "sm_100a"),
        ("nvidia_sm120", ("cuda", "tcgen05", "planned_kernel"), "sm_120a"),
    ):
        entries.append(BackendKernelEntry(
            target=target_name,
            status=_PLANNED_STATUS,
            dtypes=dtypes,
            feature_flags=flags,
            notes=(
                "Single-GPU closeout compute-tail NVIDIA owner: CUDA "
                "compiler path planned lane. No PTX/SASS/runtime proof "
                "claimed yet."
            ),
            cuda_arch_min=arch_min,
            nvcc_version_min="13.3",
        ))
    return entries


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
    if op_name in _SINGLE_GPU_COMPUTE_REFERENCE_OPS:
        return _attach_numerical_fixtures(
            op_name, _single_gpu_compute_reference_manifest_for(op_name))
    entries: list[BackendKernelEntry] = []

    # x86 AMX / AVX-512
    x86 = _X86_KERNELS.get(op_name)
    if x86 is not None:
        x86_fixture = _NUMERICAL_FIXTURES.get((op_name, "x86"))
        x86_status = str(x86["status"])
        is_amx_gemm = op_name in {"matmul", "gemm"}
        if x86_fixture is not None and not is_amx_gemm:
            x86_status = _COMPILED_STATUS
        entries.append(BackendKernelEntry(
            target="x86",
            status=x86_status,
            dtypes=tuple(x86["dtypes"]),
            feature_flags=("amx", "avx512") if is_amx_gemm else ("avx512",),
            notes=str(x86.get("notes", "")),
            execute_compare_fixture=x86_fixture,
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
        # hardware_verified pulls its fixture from _NUMERICAL_FIXTURES; a
        # ``compiled`` lane (e.g. the pointwise-loss lane) carries its own
        # execute_compare_fixture in the kernel dict — both are required at
        # construction for their status.
        _ag_fixture: Optional_str = (
            _NUMERICAL_FIXTURES.get((op_name, "apple_gpu"))
            if _ag_status == _HARDWARE_VERIFIED_STATUS
            else apple_gpu.get("execute_compare_fixture"))
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
        # Consumer-Blackwell bring-up (2026-06-25): a shipped runtime symbol +
        # execute-compare fixture promotes the sm_120 row to hardware_verified
        # (warp-level mma.sync). Replaces the artifact_only row for this arch
        # only; sm_80/90/100 stay artifact_only (proven only on sm_120).
        nv_hv = _NVIDIA_HARDWARE_VERIFIED.get(op_name) if target_name == "nvidia_sm120" else None
        if nv_hv is not None:
            entries.append(BackendKernelEntry(
                target="nvidia_sm120",
                status=_HARDWARE_VERIFIED_STATUS,
                dtypes=tuple(nv_hv["dtypes"]),
                feature_flags=tuple(nv_hv.get("feature_flags", ("mma_sync",))),
                notes=str(nv_hv.get("notes", "")),
                runtime_symbol=nv_hv["runtime_symbol"],
                execute_compare_fixture=_NUMERICAL_FIXTURES.get((op_name, "nvidia_sm120")),
                shape_envelope=nv_hv.get("shape_envelope"),
                cuda_arch_min="sm_120a",
                nvcc_version_min="13.3",
                expected_mfu=_NVIDIA_KERNEL_MFU.get((op_name, "nvidia_sm120")),
                # E2 — plumbing ready; stays None until the sm_120 box records
                # benchmarks/baselines/nvidia_sm120_hot_paths.json (Decision #26
                # forbids pointing at a baseline that does not yet exist).
                benchmark_json=nv_hv.get("benchmark_json"),
            ))
            continue
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
                    "Target IR artifact ships under CUDA 13.3; "
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

    # ROCm — Strix Halo bring-up (2026-06-22): ops with a shipped runtime
    # symbol + execute-compare fixture are ``hardware_verified`` (RDNA WMMA);
    # all others ride the generic MFMA artifact row (Sprint H-3, 2026-05-11:
    # MFMA shape + hipcc version pin per kernel, HIP execution gated on Phase H).
    rocm_hv = _ROCM_HARDWARE_VERIFIED.get(op_name)
    if rocm_hv is not None:
        # Both halves of the hardware_verified contract are pulled in BEFORE
        # construction so the validator sees a complete (symbol + fixture) row.
        entries.append(BackendKernelEntry(
            target="rocm",
            status=_HARDWARE_VERIFIED_STATUS,
            dtypes=tuple(rocm_hv["dtypes"]),
            feature_flags=tuple(rocm_hv.get("feature_flags", ("wmma",))),
            notes=str(rocm_hv.get("notes", "")),
            runtime_symbol=rocm_hv["runtime_symbol"],
            execute_compare_fixture=_NUMERICAL_FIXTURES.get((op_name, "rocm")),
            shape_envelope=rocm_hv.get("shape_envelope"),
            hipcc_version_min="7.2.4",
            expected_mfu=_ROCM_KERNEL_MFU.get((op_name, "rocm_gfx942")),
            # A hardware_verified matrix-core op may ALSO carry a CDNA MFMA artifact
            # shape (gemm does; matmul stays pure-RDNA-WMMA with None) — the CDNA
            # datacenter target alongside the proven RDNA WMMA row.
            mfma_shape=rocm_hv.get("mfma_shape"),
            # E2 — hot-path rows carry their ratchet baseline (Apple-style
            # layer-1 linkage); None for ops with no recorded gfx1151 baseline.
            benchmark_json=rocm_hv.get("benchmark_json"),
        ))
    elif (rocm_c := _ROCM_COMPILED.get(op_name)) is not None:
        # Compiler-generated executing lane: runs on gfx1151 via runtime.launch()
        # as an hsaco (no C-ABI symbol), with a checked-in execute_compare_fixture
        # — status ``compiled`` (a rung below the C-symbol hardware_verified).
        entries.append(BackendKernelEntry(
            target="rocm",
            status=_COMPILED_STATUS,
            dtypes=tuple(rocm_c["dtypes"]),
            feature_flags=tuple(rocm_c.get("feature_flags", ("wmma",))),
            notes=str(rocm_c.get("notes", "")),
            execute_compare_fixture=_NUMERICAL_FIXTURES.get((op_name, "rocm")),
            shape_envelope=rocm_c.get("shape_envelope"),
            hipcc_version_min="7.2.4",
            expected_mfu=_ROCM_KERNEL_MFU.get((op_name, "rocm_gfx942")),
            # GEMM-family compiled ops keep the unified MMA descriptor the old
            # artifact entry carried (None for non-GEMM ops like softmax/norm/
            # activation/rope) — promoting batched_gemm etc. to `compiled` must
            # not drop it (test_backend_manifest_rocm_gemm_carries_mma_descriptor).
            mma_descriptor=_rocm_mma_descriptor_for(
                op_name, tuple(rocm_c["dtypes"])),
        ))
    else:
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
            # A1 (2026-06-18) — attach the unified MMA descriptor for
            # GEMM-family ops, derived from a representative dtype on gfx942
            # (the `rocm` alias).
            mma_desc = _rocm_mma_descriptor_for(op_name, dtypes)
            entries.append(BackendKernelEntry(
                target="rocm",
                status=mapped,
                dtypes=dtypes,
                feature_flags=("mfma",),
                notes=(
                    "ROCm 7.2.4 MFMA artifact ships; HIP execution gated on "
                    "Phase H" if mapped == _ARTIFACT_STATUS else ""
                ),
                mfma_shape=mfma,
                hipcc_version_min="7.2.4",
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
    entries = _overlay_structured_compute_entries(op_name, entries)
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
    for name in _SINGLE_GPU_COMPUTE_REFERENCE_OPS:
        m = manifest_for(name)
        if m:
            out[name] = m
    for name in _STRUCTURED_COMPUTE_OVERLAY_OPS:
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
