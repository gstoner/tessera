"""Sprint E — Backend kernel manifest (2026-05-11).

Synthesizes the per-op × per-target × per-dtype backend kernel matrix
from ``capabilities.TARGET_CAPABILITIES`` + ``apple_gpu_kernel_inventory``
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

from dataclasses import dataclass, field
from typing import Mapping

from .capabilities import TARGET_CAPABILITIES, canonical_op
from .op_catalog import OP_SPECS


_FUSED_KERNEL_STATUS = "fused"
_REFERENCE_STATUS = "reference"
_ARTIFACT_STATUS = "artifact_only"
_COMPILEABLE_STATUS = "compileable"   # Sprint G/H follow-up: passes ptxas/hipcc
_PLANNED_STATUS = "planned"

_VALID_STATUSES = frozenset({
    _FUSED_KERNEL_STATUS,
    _REFERENCE_STATUS,
    _ARTIFACT_STATUS,
    _COMPILEABLE_STATUS,
    _PLANNED_STATUS,
})


@dataclass(frozen=True)
class BackendKernelEntry:
    """One row of the per-op × per-target × per-dtype matrix.

    All dtype strings are canonical (the dataclass normalizes at
    ``__post_init__``).
    """

    target: str
    status: str
    dtypes: tuple[str, ...] = ()
    feature_flags: tuple[str, ...] = ()
    notes: str = ""

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

    def as_dict(self) -> dict[str, object]:
        return {
            "target": self.target,
            "status": self.status,
            "dtypes": list(self.dtypes),
            "feature_flags": list(self.feature_flags),
            "notes": self.notes,
        }


# ─────────────────────────────────────────────────────────────────────────────
# Apple GPU shipped MSL kernels — per `docs/apple_gpu_kernel_inventory.md`
# (Phase 8.3 → 8.4.7).  Each entry below corresponds to one ABI symbol
# (or fusion).  Status="fused" when the MSL kernel is a fused chain;
# "reference" when the kernel is a single-op MPS dispatch.
# ─────────────────────────────────────────────────────────────────────────────
_APPLE_GPU_FUSED = ("fp32", "fp16", "bf16")

_APPLE_GPU_KERNELS: dict[str, dict[str, object]] = {
    "matmul": {
        "status": _FUSED_KERNEL_STATUS,
        "dtypes": _APPLE_GPU_FUSED,
        "notes": "MPSMatrixMultiplication + bf16 conversion path",
    },
    "softmax": {
        "status": _FUSED_KERNEL_STATUS,
        "dtypes": _APPLE_GPU_FUSED,
        "notes": "Custom MSL softmax kernel (Phase 8.4.2)",
    },
    "softmax_safe": {
        "status": _FUSED_KERNEL_STATUS,
        "dtypes": _APPLE_GPU_FUSED,
        "notes": "Aliases softmax MSL kernel",
    },
    "gelu": {
        "status": _FUSED_KERNEL_STATUS,
        "dtypes": _APPLE_GPU_FUSED,
        "notes": "Custom MSL gelu (Phase 8.4.2)",
    },
    "rope": {
        "status": _FUSED_KERNEL_STATUS,
        "dtypes": _APPLE_GPU_FUSED,
        "notes": "Custom MSL rope (Phase 8.4.0)",
    },
    "rmsnorm": {
        "status": _FUSED_KERNEL_STATUS,
        "dtypes": ("fp32",),
        "notes": "Available as part of matmul→rmsnorm fusion (Phase 8.4.7)",
    },
    "flash_attn": {
        "status": _FUSED_KERNEL_STATUS,
        "dtypes": _APPLE_GPU_FUSED,
        "notes": "Online-softmax MSL kernel; head_dim ≤ 256 (Phase 8.4.1)",
    },
}


# ─────────────────────────────────────────────────────────────────────────────
# x86 backend — AMX BF16 GEMM is the only currently-real execution path
# (Phase 2).
# ─────────────────────────────────────────────────────────────────────────────
_X86_KERNELS: dict[str, dict[str, object]] = {
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
_APPLE_CPU_KERNELS: dict[str, dict[str, object]] = {
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
# Sprint I-2 (2026-05-11) — Tenstorrent Metalium kernel inventory.
#
# Metalium is the canonical user of the planned/gated `bfp*` block-FP
# dtype family (per `docs/reference/tessera_tensor_attributes.md`).
# Each entry below declares `metadata.dtype_status="planned_gated"`
# implicitly via the dtype set membership — the registry walker is the
# enforcement point for that.
#
# Matmul + DMA already shipped (Phase 7).  Softmax / LayerNorm / RMSNorm
# land as Sprint I-1 lit fixtures with `_ARTIFACT_STATUS` (the Metalium
# pass currently lowers `tessera.tile.softmax` to a `dma + matmul`
# decomposition; lit-testable, execution gated on Wormhole hardware).
# ─────────────────────────────────────────────────────────────────────────────
_METALIUM_KERNELS: dict[str, dict[str, object]] = {
    "matmul": {
        "status": _ARTIFACT_STATUS,
        "dtypes": ("bf16", "fp32"),
        "notes": "Tile-local matmul intrinsic via `tessera_metalium.matmul` (Phase 7); FP4/FP6 lanes via `bfp4`/`bfp8` block-FP (planned/gated)",
    },
    "softmax": {
        "status": _ARTIFACT_STATUS,
        "dtypes": ("bf16",),
        "notes": "DMA→tile_local_reduce(via matmul)→exp/scale→DMA (Sprint I-1)",
    },
    "softmax_safe": {
        "status": _ARTIFACT_STATUS,
        "dtypes": ("bf16",),
        "notes": "Alias of softmax with explicit max-subtract for numerical safety",
    },
    "layer_norm": {
        "status": _ARTIFACT_STATUS,
        "dtypes": ("bf16",),
        "notes": "Two tile-local reductions (Σx, Σ(x-μ)²) via matmul (Sprint I-1)",
    },
    "rmsnorm": {
        "status": _ARTIFACT_STATUS,
        "dtypes": ("bf16",),
        "notes": "One tile-local Σx² reduction via matmul (Sprint I-1)",
    },
}


# Metalium planned/gated dtype declarations.  `bfp8` / `bfp4` are the
# Tenstorrent block-FP formats; entries that reference them carry
# `metadata.dtype_status="planned_gated"` enforced by the registry walker.
_METALIUM_PLANNED_GATED_KERNELS: dict[str, dict[str, object]] = {
    "matmul": {
        "status": _PLANNED_STATUS,
        "dtypes": ("bfp8", "bfp4"),
        "notes": "Tenstorrent block-FP MFMA-style matmul; planned/gated dtype family",
    },
}


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


def manifest_for(op_name: str) -> list[BackendKernelEntry]:
    """Return the backend manifest entries for ``op_name``.

    Order: cpu / x86 / apple_cpu / apple_gpu / nvidia_sm80 / sm90 / sm100 /
    sm120 / rocm / metalium.  Each entry's status reflects whether the
    target ships a fused kernel, a reference path, an artifact-only stub,
    or has no plan.
    """
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
        entries.append(BackendKernelEntry(
            target="apple_gpu",
            status=str(apple_gpu["status"]),
            dtypes=tuple(apple_gpu["dtypes"]),
            feature_flags=("metal", "mps", "msl"),
            notes=str(apple_gpu.get("notes", "")),
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

    # NVIDIA SM_80 / SM_90 / SM_100 / SM_120 — artifact-only until Phase G
    for target_name, flags in (
        ("nvidia_sm80",  ("wmma",)),
        ("nvidia_sm90",  ("wgmma", "tma")),
        ("nvidia_sm100", ("tcgen05", "tmem")),
        ("nvidia_sm120", ("tcgen05", "tmem")),
    ):
        cap = _capability_status(target_name, op_name)
        if cap is not None:
            status, dtypes = cap
            mapped = (
                _FUSED_KERNEL_STATUS if status == "ready"
                else _ARTIFACT_STATUS if status == "artifact_only"
                else _PLANNED_STATUS
            )
            entries.append(BackendKernelEntry(
                target=target_name,
                status=mapped,
                dtypes=dtypes,
                feature_flags=flags,
                notes=(
                    "Target IR artifact ships; execution gated on Phase G"
                    if mapped == _ARTIFACT_STATUS
                    else ""
                ),
            ))

    # ROCm MFMA
    cap = _capability_status("rocm", op_name)
    if cap is not None:
        status, dtypes = cap
        mapped = (
            _FUSED_KERNEL_STATUS if status == "ready"
            else _ARTIFACT_STATUS if status == "artifact_only"
            else _PLANNED_STATUS
        )
        entries.append(BackendKernelEntry(
            target="rocm",
            status=mapped,
            dtypes=dtypes,
            feature_flags=("mfma",),
            notes=(
                "ROCm MFMA artifact ships; HIP execution gated on Phase H"
                if mapped == _ARTIFACT_STATUS else ""
            ),
        ))

    # Tenstorrent Metalium — Sprint I-1/I-2 (2026-05-11): shipped kernel
    # inventory takes precedence over the generic capability lookup so
    # the manifest reflects the actual dialect coverage (softmax /
    # layer_norm / rmsnorm via `dma + matmul` decomposition).
    metalium = _METALIUM_KERNELS.get(op_name)
    if metalium is not None:
        entries.append(BackendKernelEntry(
            target="metalium",
            status=str(metalium["status"]),
            dtypes=tuple(metalium["dtypes"]),
            feature_flags=("dma_contract", "tile_local_matmul", "risc_v_grid"),
            notes=str(metalium.get("notes", "")),
        ))
        # Planned/gated dtype variants (bfp8 / bfp4) are recorded as a
        # separate entry so the audit walker can verify the
        # `metadata.dtype_status="planned_gated"` annotation.
        gated = _METALIUM_PLANNED_GATED_KERNELS.get(op_name)
        if gated is not None:
            entries.append(BackendKernelEntry(
                target="metalium_blockfp",
                status=str(gated["status"]),
                dtypes=tuple(gated["dtypes"]),
                feature_flags=("dma_contract", "block_fp", "tt_native"),
                notes=str(gated.get("notes", "")),
            ))
    else:
        cap = _capability_status("metalium", op_name)
        if cap is not None:
            status, dtypes = cap
            mapped = (
                _FUSED_KERNEL_STATUS if status == "ready"
                else _ARTIFACT_STATUS if status == "artifact_only"
                else _PLANNED_STATUS
            )
            entries.append(BackendKernelEntry(
                target="metalium",
                status=mapped,
                dtypes=dtypes,
                feature_flags=("dma_contract",),
                notes=(
                    "Metalium target artifact scaffolded"
                    if mapped == _ARTIFACT_STATUS else ""
                ),
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

    return entries


def all_manifests() -> Mapping[str, list[BackendKernelEntry]]:
    """Return the manifest for every op in ``OP_SPECS``.

    Useful for the dashboard renderer and for the audit walker that
    verifies dtype canonicalness across the full per-target matrix.
    """
    out: dict[str, list[BackendKernelEntry]] = {}
    for name in OP_SPECS:
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
    "manifest_for",
    "all_manifests",
    "manifest_summary",
    "audit_backend_dtypes",
]
