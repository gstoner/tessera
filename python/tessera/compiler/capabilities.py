"""Shared compiler capability registry for targets, ops, and runtime status."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping, Optional

from .op_catalog import GRAPH_OP_TO_SPEC, LEGACY_GRAPH_OP_ALIASES, canonical_graph_op_name


CAPABILITY_REGISTRY_VERSION = "tessera.capabilities.v1"
RuntimeStatus = str


@dataclass(frozen=True)
class OpCapability:
    op_name: str
    runtime_status: RuntimeStatus
    dtypes: tuple[str, ...] = ()
    ranks: tuple[int, ...] = ()
    layouts: tuple[str, ...] = ("row_major",)
    reason: str = ""

    @property
    def executable(self) -> bool:
        return self.runtime_status == "ready"


@dataclass(frozen=True)
class TargetCapability:
    name: str
    aliases: tuple[str, ...]
    family: str
    runtime_backend: str
    default_runtime_status: RuntimeStatus
    supported_ops: Mapping[str, OpCapability] = field(default_factory=dict)
    supported_dtypes: tuple[str, ...] = ()
    features: tuple[str, ...] = ()
    reason: str = ""

    def op(self, op_name: str) -> OpCapability:
        canonical = canonical_op(op_name)
        return self.supported_ops.get(canonical, OpCapability(
            op_name=canonical,
            runtime_status=self.default_runtime_status,
            dtypes=self.supported_dtypes,
            reason=self.reason or f"{canonical} uses {self.default_runtime_status} on {self.name}",
        ))


@dataclass(frozen=True)
class CapabilityResult:
    target: str
    op_name: str
    supported: bool
    runtime_status: RuntimeStatus
    reason: str = ""
    capability_version: str = CAPABILITY_REGISTRY_VERSION


def canonical_op(op_name: str) -> str:
    if op_name.startswith("tessera.ops."):
        op_name = op_name.removeprefix("tessera.ops.")
    if not op_name.startswith("tessera."):
        spec = GRAPH_OP_TO_SPEC.get(op_name)
        if spec is not None:
            op_name = spec.graph_name
        else:
            from .op_catalog import graph_name_for
            op_name = graph_name_for(op_name) or op_name
    return canonical_graph_op_name(LEGACY_GRAPH_OP_ALIASES.get(op_name, op_name))


def normalize_target(target: object = "cpu") -> str:
    if target is None:
        return "cpu"
    if isinstance(target, str):
        normalized = target.lower().replace("-", "_")
        normalized = _ALIASES.get(normalized, normalized)
        if normalized not in TARGET_CAPABILITIES:
            allowed = sorted(TARGET_CAPABILITIES)
            raise ValueError(f"unsupported Tessera target {target!r}; expected one of {allowed}")
        return normalized
    from .gpu_target import GPUTargetProfile, ISA
    if isinstance(target, GPUTargetProfile):
        if target.isa >= ISA.SM_120:
            return "nvidia_sm120"
        if target.isa >= ISA.SM_100:
            return "nvidia_sm100"
        if target.isa >= ISA.SM_90:
            return "nvidia_sm90"
        return "nvidia_sm80"
    raise TypeError(f"unsupported Tessera target object {type(target)!r}")


def get_target_capability(target: object = "cpu") -> TargetCapability:
    return TARGET_CAPABILITIES[normalize_target(target)]


def supports_op(
    target: object,
    op: str,
    *,
    dtype: Optional[str] = None,
    rank: Optional[int] = None,
) -> CapabilityResult:
    target_name = normalize_target(target)
    cap = TARGET_CAPABILITIES[target_name]
    op_cap = cap.op(op)
    dtype_ok = dtype is None or not op_cap.dtypes or dtype in op_cap.dtypes
    rank_ok = rank is None or not op_cap.ranks or rank in op_cap.ranks
    supported = op_cap.runtime_status != "unsupported" and dtype_ok and rank_ok
    reason = op_cap.reason
    if not dtype_ok:
        reason = f"dtype {dtype!r} is not supported for {op_cap.op_name} on {target_name}"
    elif not rank_ok:
        reason = f"rank {rank!r} is not supported for {op_cap.op_name} on {target_name}"
    return CapabilityResult(
        target=target_name,
        op_name=op_cap.op_name,
        supported=supported,
        runtime_status=op_cap.runtime_status if supported else "unsupported",
        reason=reason,
    )


def runtime_status(target: object, op: str | None = None) -> RuntimeStatus:
    cap = get_target_capability(target)
    if op is None:
        return cap.default_runtime_status
    return supports_op(target, op).runtime_status


def target_aliases() -> dict[str, str]:
    return dict(_ALIASES)


def _ops(status: RuntimeStatus, names: tuple[str, ...], *, reason: str = "", dtypes: tuple[str, ...] = ("fp32", "f32")) -> dict[str, OpCapability]:
    return {
        canonical_op(name): OpCapability(canonical_op(name), status, dtypes=dtypes, reason=reason)
        for name in names
    }


_CPU_OPS = tuple(sorted(GRAPH_OP_TO_SPEC))
_APPLE_GPU_READY = ("tessera.matmul", "tessera.flash_attn", "tessera.softmax", "tessera.softmax_safe", "tessera.gelu", "tessera.rope")

# E1 (partial-ops uplift, 2026-05-20).  apple_gpu ships fused MSL kernels for
# all 17 GA primitives (12 GA3 core + 5 GA5 differential-form) and 9 EBM
# primitives.  This table mirrors ``backend_manifest._CLIFFORD_APPLE_GPU_FUSED``
# + ``_EBM_APPLE_GPU_FUSED`` so the e2e_coverage audit walker's ``runtime``
# axis resolves to ``fused`` (instead of ``unknown``) for these ops — that's
# what flips them from PARTIAL → COMPLETE.  Per-op dtype tuples reflect the
# actual kernel inventory: only geometric_product + rotor_sandwich ship
# fp16/bf16 ports today; the other 15 GA ops and every EBM op are fp32-only
# for v1.  Adding more dtype lanes is a backend-side change; this table just
# teaches the capability registry what the manifest already says.
_APPLE_GPU_FUSED_OPS: dict[str, tuple[str, ...]] = {
    # GA3 core (12 ops)
    "clifford_geometric_product": ("fp32", "fp16", "bf16"),
    "clifford_rotor_sandwich":    ("fp32", "fp16", "bf16"),
    "clifford_reverse":           ("fp32",),
    "clifford_grade_involution":  ("fp32",),
    "clifford_conjugate":         ("fp32",),
    "clifford_norm":              ("fp32",),
    "clifford_wedge":             ("fp32",),
    "clifford_left_contraction":  ("fp32",),
    "clifford_inner":             ("fp32",),
    "clifford_grade_projection":  ("fp32",),
    "clifford_exp":               ("fp32",),
    "clifford_log":               ("fp32",),
    # GA5 differential-form (5 ops)
    "clifford_hodge_star":        ("fp32",),
    "clifford_ext_deriv":         ("fp32",),
    "clifford_vec_deriv":         ("fp32",),
    "clifford_codiff":            ("fp32",),
    "clifford_integral":          ("fp32",),
    # EBM (9 ops with shipped MSL kernels)
    "ebm_inner_step":             ("fp32",),
    "ebm_refinement":             ("fp32",),
    "ebm_langevin_step":          ("fp32",),
    "ebm_decode_init":            ("fp32",),
    "ebm_bivector_langevin":      ("fp32",),
    "ebm_sphere_langevin":        ("fp32",),
    "ebm_self_verify":            ("fp32",),
    "ebm_energy":                 ("fp32",),
    "ebm_partition_exact":        ("fp32",),
    # E2 (partial-ops uplift, 2026-05-20) — M7 ``complex_*`` ops with
    # fused Apple GPU MSL kernels.  Keyed by the **public** name (the
    # backend manifest uses a ``complex_`` prefix on two of them —
    # ``mobius`` → ``complex_mobius``, ``stereographic`` →
    # ``complex_stereographic`` — but ``audit._axis_runtime`` looks the
    # capability registry up by the public name, not the backend alias).
    "complex_mul":                ("fp32",),
    "complex_exp":                ("fp32",),
    "mobius":                     ("fp32",),
    "stereographic":              ("fp32",),
}


def _apple_gpu_fused_caps() -> dict[str, OpCapability]:
    """Build per-op ``OpCapability(runtime_status='fused')`` entries from
    ``_APPLE_GPU_FUSED_OPS``.  Keyed by ``canonical_op(name)`` to match
    ``audit._axis_runtime``'s lookup path.
    """
    out: dict[str, OpCapability] = {}
    for name, dtypes in _APPLE_GPU_FUSED_OPS.items():
        key = canonical_op(name)
        out[key] = OpCapability(
            op_name=key,
            runtime_status="fused",
            dtypes=dtypes,
            reason=(
                f"Apple GPU ships a fused MSL kernel for {name} "
                f"(backend_manifest._{'CLIFFORD' if name.startswith('clifford_') else 'EBM'}_APPLE_GPU_FUSED)"
            ),
        )
    return out

# Sprint G-2 (2026-05-11): expanded to match the planned NVIDIA kernel
# inventory in `docs/nvidia_cuda13_kernel_inventory.md`.  Each entry
# here ships a Target IR artifact under CUDA 13.2 U1.
_NVIDIA_ARTIFACT = (
    # Matmul / contraction family
    "tessera.matmul", "tessera.batched_gemm", "tessera.einsum",
    "tessera.linear_general", "tessera.qkv_projection",
    "tessera.fused_epilogue", "tessera.factorized_matmul",
    # Attention family
    "tessera.flash_attn",
    "tessera.multi_head_attention", "tessera.gqa_attention",
    "tessera.mqa_attention",
    "tessera.mla_decode", "tessera.mla_decode_fused",
    "tessera.deepseek_sparse_attention",
    "tessera.attn_top_k_blocks", "tessera.attn_compressed_blocks",
    "tessera.attn_sliding_window",
    "tessera.lightning_attention", "tessera.linear_attn",
    "tessera.gated_deltanet", "tessera.kimi_delta_attention",
    "tessera.modified_delta_attention", "tessera.gated_attention",
    "tessera.hybrid_attention",
    # Normalization / activation / position encoding
    "tessera.layer_norm", "tessera.rmsnorm", "tessera.rmsnorm_safe",
    "tessera.softmax", "tessera.softmax_safe", "tessera.online_softmax",
    "tessera.gelu", "tessera.silu", "tessera.silu_mul",
    "tessera.rope", "tessera.alibi",
)

# Sprint H-3 (2026-05-11): expanded to match the planned ROCm kernel
# inventory in `docs/rocm_mfma_kernel_inventory.md`.
_ROCM_ARTIFACT = (
    "tessera.matmul", "tessera.batched_gemm", "tessera.einsum",
    "tessera.linear_general", "tessera.qkv_projection",
    "tessera.fused_epilogue", "tessera.factorized_matmul",
    "tessera.flash_attn",
    "tessera.multi_head_attention", "tessera.gqa_attention",
    "tessera.mqa_attention",
    "tessera.mla_decode", "tessera.mla_decode_fused",
    "tessera.deepseek_sparse_attention",
    "tessera.attn_sliding_window",
    "tessera.lightning_attention", "tessera.linear_attn",
    "tessera.gated_deltanet", "tessera.kimi_delta_attention",
    "tessera.modified_delta_attention", "tessera.gated_attention",
    "tessera.hybrid_attention",
    "tessera.layer_norm", "tessera.rmsnorm", "tessera.rmsnorm_safe",
    "tessera.softmax", "tessera.softmax_safe",
    "tessera.gelu", "tessera.silu", "tessera.silu_mul",
    "tessera.rope", "tessera.alibi",
)


TARGET_CAPABILITIES: dict[str, TargetCapability] = {
    "cpu": TargetCapability(
        name="cpu",
        aliases=("cpu", "x86", "x86_64"),
        family="cpu",
        runtime_backend="numpy",
        default_runtime_status="ready",
        supported_ops=_ops("ready", _CPU_OPS, reason="CPU reference/runtime path is available"),
        supported_dtypes=("fp32", "f32", "fp16", "bf16"),
        features=("reference_execution", "host_runtime"),
    ),
    # Sprint G-1 (2026-05-11): NVIDIA capability matrix pinned to
    # CUDA 13.2 Update 1.  Each entry's `features` tuple records the
    # subset of `cuda_feature_set(isa)` flags that are functional under
    # 13.2 U1; `supported_dtypes` mirrors `_TENSOR_CORE_DTYPES[isa]`
    # filtered to canonical Tessera dtype strings (TF32 is a math_mode,
    # not a storage dtype — see `tessera.dtype.canonicalize_dtype`).
    "nvidia_sm80": TargetCapability(
        name="nvidia_sm80",
        aliases=("sm80", "sm_80"),
        family="nvidia",
        runtime_backend="cuda",
        default_runtime_status="artifact_only",
        supported_ops=_ops("artifact_only", _NVIDIA_ARTIFACT, reason="NVIDIA Ampere artifact ships under CUDA 13.2 U1; executable smoke is hardware-gated"),
        supported_dtypes=("bf16", "fp16", "fp32", "fp64", "int8"),
        features=("wmma", "cp_async", "mbarrier", "cuda_13_2_u1"),
    ),
    "nvidia_sm90": TargetCapability(
        name="nvidia_sm90",
        aliases=("cuda", "nvidia", "gpu", "sm90", "sm_90", "sm90a", "sm_90a", "hopper"),
        family="nvidia",
        runtime_backend="cuda",
        default_runtime_status="artifact_only",
        supported_ops=_ops("artifact_only", _NVIDIA_ARTIFACT, reason="NVIDIA SM90 WGMMA/TMA artifact ships under CUDA 13.2 U1; executable smoke is hardware-gated"),
        supported_dtypes=("bf16", "fp16", "fp32", "fp64", "fp8_e4m3", "fp8_e5m2", "int8"),
        features=(
            "wgmma", "wgmma_sparse", "tma", "tma_swizzle_128b",
            "cluster_launch", "mbarrier_arrive_tx", "cp_async_bulk",
            "async_proxy_fence", "cuda_13_2_u1",
        ),
    ),
    "nvidia_sm100": TargetCapability(
        name="nvidia_sm100",
        aliases=("sm100", "sm_100", "sm100a", "sm_100a", "blackwell"),
        family="nvidia",
        runtime_backend="cuda",
        default_runtime_status="artifact_only",
        supported_ops=_ops("artifact_only", _NVIDIA_ARTIFACT, reason="Blackwell artifact ships under CUDA 13.2 U1; executable smoke is hardware-gated"),
        supported_dtypes=(
            "bf16", "fp16", "fp32", "fp64",
            "fp8_e4m3", "fp8_e5m2",
            "fp6_e2m3", "fp6_e3m2",
            "fp4_e2m1", "nvfp4",
            "int8",
        ),
        features=(
            "wgmma", "wgmma_sparse", "tma", "tma_swizzle_128b",
            "cluster_launch", "mbarrier_arrive_tx",
            "tcgen05", "tcgen05_pair", "tmem", "block_scaled_mma",
            "cp_async_bulk", "async_proxy_fence", "cuda_13_2_u1",
        ),
    ),
    "nvidia_sm120": TargetCapability(
        name="nvidia_sm120",
        aliases=("sm120", "sm_120"),
        family="nvidia",
        runtime_backend="cuda",
        default_runtime_status="artifact_only",
        supported_ops=_ops("artifact_only", _NVIDIA_ARTIFACT, reason="SM120 (Rubin) artifact ships under CUDA 13.2 U1 with preliminary intrinsics; executable smoke is hardware-gated"),
        supported_dtypes=(
            "bf16", "fp16", "fp32", "fp64",
            "fp8_e4m3", "fp8_e5m2",
            "fp6_e2m3", "fp6_e3m2",
            "fp4_e2m1", "nvfp4",
            "int8",
        ),
        features=(
            "wgmma", "wgmma_sparse", "tma", "tma_swizzle_128b",
            "cluster_launch", "mbarrier_arrive_tx",
            "tcgen05", "tcgen05_pair", "tmem", "block_scaled_mma",
            "cp_async_bulk", "async_proxy_fence", "cuda_13_2_u1",
        ),
    ),
    # Sprint H-1 (2026-05-11): ROCm 7.2.3 capability matrix.  Per-arch
    # entries replace the single "rocm" alias; the legacy "rocm" name
    # routes to gfx942 (MI300X) as the canonical default.  Each entry's
    # `features` reflects `rocm_feature_set(arch)` from `rocm_target.py`.
    "rocm": TargetCapability(
        name="rocm",
        aliases=("rocm", "amd", "hip"),
        family="rocm",
        runtime_backend="hip",
        default_runtime_status="artifact_only",
        # Sprint H-3 (2026-05-11): full planned kernel set from
        # `docs/rocm_mfma_kernel_inventory.md`.
        supported_ops=_ops("artifact_only", _ROCM_ARTIFACT, reason="ROCm 7.2.3 MFMA artifact exists; HIP execution remains gated"),
        supported_dtypes=("bf16", "fp16", "fp32", "fp64", "fp8_e4m3", "fp8_e5m2", "int8"),
        features=(
            "mfma", "mfma_f8", "mfma_xf32",
            "lds_async_copy", "buffer_load_lds", "global_load_lds",
            "xnack", "sram_ecc", "rocm_7_2_3",
        ),
    ),
    "rocm_gfx90a": TargetCapability(
        name="rocm_gfx90a",
        aliases=("gfx90a", "mi250", "mi250x"),
        family="rocm",
        runtime_backend="hip",
        default_runtime_status="artifact_only",
        supported_ops=_ops("artifact_only", ("tessera.matmul",), reason="ROCm 7.2.3 CDNA 2 baseline MFMA artifact"),
        supported_dtypes=("bf16", "fp16", "fp32", "fp64", "int8"),
        features=("mfma", "buffer_load_lds", "xnack", "sram_ecc", "rocm_7_2_3"),
    ),
    "rocm_gfx940": TargetCapability(
        name="rocm_gfx940",
        aliases=("gfx940", "mi300a"),
        family="rocm",
        runtime_backend="hip",
        default_runtime_status="artifact_only",
        supported_ops=_ops("artifact_only", ("tessera.matmul", "tessera.gelu", "tessera.softmax"), reason="ROCm 7.2.3 CDNA 3 unified-APU artifact; HIP execution gated"),
        supported_dtypes=("bf16", "fp16", "fp32", "fp64", "fp8_e4m3", "fp8_e5m2", "int8"),
        features=(
            "mfma", "mfma_f8", "mfma_xf32",
            "lds_async_copy", "buffer_load_lds", "global_load_lds",
            "xnack", "sram_ecc", "rocm_7_2_3",
        ),
    ),
    "rocm_gfx942": TargetCapability(
        name="rocm_gfx942",
        aliases=("gfx942", "mi300x"),
        family="rocm",
        runtime_backend="hip",
        default_runtime_status="artifact_only",
        supported_ops=_ops("artifact_only", ("tessera.matmul", "tessera.gelu", "tessera.softmax", "tessera.flash_attn"), reason="ROCm 7.2.3 CDNA 3 discrete MI300X MFMA artifact; HIP execution gated"),
        supported_dtypes=("bf16", "fp16", "fp32", "fp64", "fp8_e4m3", "fp8_e5m2", "int8"),
        features=(
            "mfma", "mfma_f8", "mfma_xf32",
            "lds_async_copy", "buffer_load_lds", "global_load_lds",
            "xnack", "sram_ecc", "rocm_7_2_3",
        ),
    ),
    "rocm_gfx950": TargetCapability(
        name="rocm_gfx950",
        aliases=("gfx950", "mi325x"),
        family="rocm",
        runtime_backend="hip",
        default_runtime_status="artifact_only",
        supported_ops=_ops("artifact_only", ("tessera.matmul", "tessera.gelu", "tessera.softmax", "tessera.flash_attn"), reason="ROCm 7.2.3 CDNA 4 MI325X MFMA artifact (with FP4/FP6 lanes); HIP execution gated"),
        supported_dtypes=(
            "bf16", "fp16", "fp32", "fp64",
            "fp8_e4m3", "fp8_e5m2",
            "fp6_e2m3", "fp6_e3m2",
            "fp4_e2m1",
            "int8",
        ),
        features=(
            "mfma", "mfma_f8", "mfma_xf32", "mfma_f4", "mfma_f6",
            "lds_async_copy", "buffer_load_lds", "global_load_lds",
            "cluster_mode", "xnack", "sram_ecc", "rocm_7_2_3",
        ),
    ),
    "rocm_gfx1100": TargetCapability(
        name="rocm_gfx1100",
        aliases=("gfx1100", "rdna3", "rx7900"),
        family="rocm",
        runtime_backend="hip",
        default_runtime_status="artifact_only",
        supported_ops=_ops("artifact_only", ("tessera.matmul",), reason="ROCm 7.2.3 RDNA 3 WMMA artifact (prosumer line)"),
        supported_dtypes=("bf16", "fp16", "fp32", "int8"),
        features=("wmma_f16", "wmma_bf16", "buffer_load_lds", "rocm_7_2_3"),
    ),
    "metalium": TargetCapability(
        name="metalium",
        aliases=("metalium", "tt_metalium", "tt"),
        family="metalium",
        runtime_backend="metalium",
        default_runtime_status="artifact_only",
        supported_ops=_ops("artifact_only", ("tessera.matmul",), reason="Metalium target contract is scaffolded"),
        supported_dtypes=("bf16", "fp32", "f32"),
        features=("dma_contract",),
    ),
    "apple_cpu": TargetCapability(
        name="apple_cpu",
        aliases=("macos_cpu", "m_series_cpu"),
        family="apple",
        runtime_backend="accelerate",
        default_runtime_status="ready",
        supported_ops=_ops("ready", _CPU_OPS, reason="Apple CPU uses Accelerate where available and reference fallback otherwise"),
        supported_dtypes=("fp32", "f32", "bf16", "fp16"),
        features=("accelerate", "reference_fallback"),
    ),
    "apple_gpu": TargetCapability(
        name="apple_gpu",
        aliases=("apple", "mac", "macos_gpu", "m_series_gpu"),
        family="apple",
        runtime_backend="metal",
        default_runtime_status="artifact_only",
        supported_ops={
            **_ops("ready", _APPLE_GPU_READY, reason="Apple GPU runtime shim supports this single-op smoke path"),
            **_ops("artifact_only", ("tessera.moe", "tessera.add", "tessera.mul"), reason="Apple GPU target contract exists but native execution is not wired for this op"),
            # E1 (2026-05-20) — 26 fused GA + EBM MSL kernels.  These are
            # the same kernels documented in ``backend_manifest`` and
            # benchmarked end-to-end by ``benchmarks/apple_gpu/benchmark_ga_ebm.py``.
            **_apple_gpu_fused_caps(),
        },
        supported_dtypes=("fp32", "f32"),
        features=("metal", "mps", "msl"),
    ),
}

_ALIASES = {
    alias: target.name
    for target in TARGET_CAPABILITIES.values()
    for alias in target.aliases
}


__all__ = [
    "CAPABILITY_REGISTRY_VERSION",
    "CapabilityResult",
    "OpCapability",
    "TargetCapability",
    "canonical_op",
    "get_target_capability",
    "normalize_target",
    "runtime_status",
    "supports_op",
    "target_aliases",
]
