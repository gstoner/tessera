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
# here ships a Target IR artifact under CUDA 13.3.
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
    # Structured CUDA-kernel Target IR contracts. These are artifact-only:
    # lowering/verification is hardware-free; execution remains gated.
    "tessera.conv2d_nhwc", "tessera.kv_cache.read",
    # Normalization / activation / position encoding
    "tessera.layer_norm", "tessera.rmsnorm", "tessera.rmsnorm_safe",
    "tessera.softmax", "tessera.softmax_safe", "tessera.online_softmax",
    "tessera.gelu", "tessera.silu", "tessera.silu_mul",
    "tessera.rope", "tessera.alibi",
    # Composite helpers promoted during single-GPU closeout. CUDA carries
    # Target IR artifacts; executable smoke remains hardware-gated.
    "tessera.memory_index_score", "tessera.msa_index_scores",
    "tessera.varlen_sdpa", "tessera.score_combine",
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
    "tessera.memory_index_score", "tessera.msa_index_scores",
    "tessera.varlen_sdpa", "tessera.score_combine",
)


TARGET_CAPABILITIES: dict[str, TargetCapability] = {
    "cpu": TargetCapability(
        name="cpu",
        aliases=("cpu", "x86_64"),
        family="cpu",
        runtime_backend="numpy",
        default_runtime_status="ready",
        supported_ops={
            **_ops("ready", _CPU_OPS, reason="CPU reference/runtime path is available"),
            # Sprint 7 (2026-06-03): the numpy reference matmul executes f16/bf16
            # as well as f32, so the per-op dtype tuple is widened to match. The
            # target-agnostic Graph IR verifier consults the `cpu` capability
            # (it has no real-target context); the authoritative per-target gate
            # runs later in pipeline_gates. (batched_gemm stays fp32 — no batched
            # f16/bf16 yet.)
            canonical_op("tessera.matmul"): OpCapability(
                canonical_op("tessera.matmul"), "ready",
                dtypes=("fp32", "f32", "fp16", "bf16"),
                reason="CPU reference matmul executes f32/f16/bf16"),
            # Sprint 8 (2026-06-03): batched_gemm dtype tuple widened so f16/bf16
            # rank-3 batched matmul passes the target-agnostic Graph IR verifier
            # (which consults the `cpu` capability). The CPU value lane still
            # gates non-f32 batched in TileToApple; the GPU value lane executes
            # f16/bf16 batched via the bmm symbols.
            canonical_op("tessera.batched_gemm"): OpCapability(
                canonical_op("tessera.batched_gemm"), "ready",
                dtypes=("fp32", "f32", "fp16", "bf16"),
                reason="CPU reference batched matmul executes f32/f16/bf16; "
                       "CPU value lane is fp32-only, GPU value lane f32/f16/bf16"),
        },
        supported_dtypes=("fp32", "f32", "fp16", "bf16"),
        features=("reference_execution", "host_runtime"),
    ),
    # Native host lane. Keep this distinct from ``cpu`` so exact-target audit
    # rows cannot inherit reference execution or a numpy Target IR by alias.
    "x86": TargetCapability(
        name="x86",
        aliases=("x86",),
        family="x86",
        runtime_backend="native_cpu",
        default_runtime_status="ready",
        supported_ops=_ops(
            "ready", _CPU_OPS,
            reason="x86 target contract is available; per-op native proof is manifest-gated",
        ),
        supported_dtypes=("fp32", "f32", "bf16"),
        features=("avx512", "amx", "native_cpu"),
    ),
    # Sprint G-1 (2026-05-11): NVIDIA capability matrix pinned to
    # CUDA 13.3.  Each entry's `features` tuple records the
    # subset of `cuda_feature_set(isa)` flags that are functional under
    # 13.3; `supported_dtypes` mirrors `_TENSOR_CORE_DTYPES[isa]`
    # filtered to canonical Tessera dtype strings (TF32 is a math_mode,
    # not a storage dtype — see `tessera.dtype.canonicalize_dtype`).
    "nvidia_sm80": TargetCapability(
        name="nvidia_sm80",
        aliases=("sm80", "sm_80"),
        family="nvidia",
        runtime_backend="cuda",
        default_runtime_status="artifact_only",
        supported_ops=_ops("artifact_only", _NVIDIA_ARTIFACT, reason="NVIDIA Ampere artifact ships under CUDA 13.3; executable smoke is hardware-gated"),
        supported_dtypes=("bf16", "fp16", "fp32", "fp64", "int8"),
        features=("wmma", "cp_async", "mbarrier", "cuda_13_3"),
    ),
    "nvidia_sm90": TargetCapability(
        name="nvidia_sm90",
        aliases=("cuda", "nvidia", "gpu", "sm90", "sm_90", "sm90a", "sm_90a", "hopper"),
        family="nvidia",
        runtime_backend="cuda",
        default_runtime_status="artifact_only",
        supported_ops=_ops("artifact_only", _NVIDIA_ARTIFACT, reason="NVIDIA SM90 WGMMA/TMA artifact ships under CUDA 13.3; executable smoke is hardware-gated"),
        supported_dtypes=("bf16", "fp16", "fp32", "fp64", "fp8_e4m3", "fp8_e5m2", "int8"),
        features=(
            "wgmma", "wgmma_sparse", "tma", "tma_swizzle_128b",
            "cluster_launch", "mbarrier_arrive_tx", "cp_async_bulk",
            "async_proxy_fence", "cuda_13_3",
        ),
    ),
    "nvidia_sm100": TargetCapability(
        name="nvidia_sm100",
        aliases=("sm100", "sm_100", "sm100a", "sm_100a", "blackwell"),
        family="nvidia",
        runtime_backend="cuda",
        default_runtime_status="artifact_only",
        supported_ops=_ops("artifact_only", _NVIDIA_ARTIFACT, reason="Blackwell artifact ships under CUDA 13.3; executable smoke is hardware-gated"),
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
            "cp_async_bulk", "async_proxy_fence", "cuda_13_3",
        ),
    ),
    "nvidia_sm120": TargetCapability(
        name="nvidia_sm120",
        aliases=("sm120", "sm_120", "blackwell_consumer", "rtx50", "gb20x"),
        family="nvidia",
        runtime_backend="cuda",
        default_runtime_status="artifact_only",
        supported_ops=_ops("artifact_only", _NVIDIA_ARTIFACT, reason="SM120 Blackwell consumer (RTX 50-series / GB20x) artifact under CUDA 13.3; FP4 via mma.sync.block_scale (no tcgen05/TMEM — those are sm_100a); executable smoke is hardware-gated"),
        supported_dtypes=(
            "bf16", "fp16", "fp32", "fp64",
            "fp8_e4m3", "fp8_e5m2",
            "fp6_e2m3", "fp6_e3m2",
            "fp4_e2m1", "nvfp4",
            "int8",
        ),
        features=(
            # Consumer Blackwell (RTX 50-series / GB20x) is NOT a superset of
            # datacenter sm_100: NO Hopper `wgmma`/`wgmma_sparse`, NO `tcgen05`/
            # `tcgen05_pair`/`tmem` (those are sm_100a only).  FP4 block-scaling
            # goes through warp-level `mma.sync.block_scale`.  Mirrors the
            # `cuda_feature_set(SM_120)` "ready" flags (guarded by
            # test_compiler_capabilities.py::test_nvidia_features_match_cuda_matrix).
            "tma", "tma_swizzle_128b", "cluster_launch", "mbarrier_arrive_tx",
            "block_scaled_mma", "cp_async_bulk", "async_proxy_fence", "cuda_13_3",
        ),
    ),
    # Sprint H-1 (2026-05-11): ROCm 7.2.4 capability matrix.  Per-arch
    # entries replace the single proof-bearing "rocm" alias. The legacy
    # "rocm" name is a family-level compilation selector only; generated audit
    # maps derive exact gfx* proof rows separately. Each entry's
    # `features` reflects `rocm_feature_set(arch)` from `rocm_target.py`.
    "rocm": TargetCapability(
        name="rocm",
        aliases=("rocm", "amd", "hip"),
        family="rocm",
        runtime_backend="hip",
        default_runtime_status="artifact_only",
        # Sprint H-3 (2026-05-11): full planned kernel set from
        # `docs/rocm_mfma_kernel_inventory.md`.
        supported_ops=_ops("artifact_only", _ROCM_ARTIFACT, reason="ROCm 7.2.4 MFMA artifact exists; HIP execution remains gated"),
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
        supported_ops=_ops("artifact_only", ("tessera.matmul",), reason="ROCm 7.2.4 CDNA 2 baseline MFMA artifact"),
        supported_dtypes=("bf16", "fp16", "fp32", "fp64", "int8"),
        features=("mfma", "buffer_load_lds", "xnack", "sram_ecc", "rocm_7_2_3"),
    ),
    "rocm_gfx940": TargetCapability(
        name="rocm_gfx940",
        aliases=("gfx940", "mi300a"),
        family="rocm",
        runtime_backend="hip",
        default_runtime_status="artifact_only",
        supported_ops=_ops("artifact_only", ("tessera.matmul", "tessera.gelu", "tessera.softmax"), reason="ROCm 7.2.4 CDNA 3 unified-APU artifact; HIP execution gated"),
        supported_dtypes=("bf16", "fp16", "fp32", "fp64", "fp8_e4m3", "fp8_e5m2", "int8"),
        features=(
            "mfma", "mfma_f8", "mfma_xf32",
            "lds_async_copy", "buffer_load_lds", "global_load_lds",
            "xnack", "sram_ecc", "rocm_7_2_3",
        ),
    ),
    "rocm_gfx942": TargetCapability(
        name="rocm_gfx942",
        aliases=("gfx942", "mi300x", "mi325x"),
        family="rocm",
        runtime_backend="hip",
        default_runtime_status="artifact_only",
        supported_ops=_ops("artifact_only", ("tessera.matmul", "tessera.gelu", "tessera.softmax", "tessera.flash_attn"), reason="ROCm 7.2.4 CDNA 3 discrete MI300X MFMA artifact; HIP execution gated"),
        supported_dtypes=("bf16", "fp16", "fp32", "fp64", "fp8_e4m3", "fp8_e5m2", "int8"),
        features=(
            "mfma", "mfma_f8", "mfma_xf32",
            "lds_async_copy", "buffer_load_lds", "global_load_lds",
            "xnack", "sram_ecc", "rocm_7_2_3",
        ),
    ),
    "rocm_gfx950": TargetCapability(
        name="rocm_gfx950",
        aliases=("gfx950", "mi350x", "mi355x", "mi350p"),
        family="rocm",
        runtime_backend="hip",
        default_runtime_status="artifact_only",
        supported_ops=_ops("artifact_only", ("tessera.matmul", "tessera.gelu", "tessera.softmax", "tessera.flash_attn"), reason="ROCm 7.2.4 CDNA 4 MI350-series MFMA artifact (with FP4/FP6 lanes); exact-device execution gated"),
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
        supported_ops=_ops("artifact_only", ("tessera.matmul",), reason="ROCm 7.2.4 RDNA 3 WMMA artifact (prosumer line)"),
        supported_dtypes=("bf16", "fp16", "fp32", "int8"),
        features=("wmma_f16", "wmma_bf16", "buffer_load_lds", "rocm_7_2_3"),
    ),
    "rocm_gfx1151": TargetCapability(
        name="rocm_gfx1151",
        aliases=("gfx1151", "rdna35", "strixhalo", "radeon8060s", "ryzenaimax395"),
        family="rocm",
        runtime_backend="hip",
        default_runtime_status="artifact_only",
        supported_ops=_ops(
            "artifact_only",
            ("tessera.matmul",),
            reason="ROCm 7.2.4 RDNA 3.5 (Strix Halo APU) WMMA artifact; HIP execution gated on real gfx1151 silicon",
        ),
        # ISA §7.9 Table 33: F16/BF16/IU8 executable surface; no FP8 WMMA on
        # RDNA 3.5 (the load-bearing difference from gfx1200).
        supported_dtypes=("bf16", "fp16", "fp32", "int8"),
        features=("wmma_f16", "wmma_bf16", "buffer_load_lds", "rocm_7_2_3"),
    ),
    "rocm_gfx1200": TargetCapability(
        name="rocm_gfx1200",
        aliases=("gfx1200", "gfx12", "rdna4", "rx9000"),
        family="rocm",
        runtime_backend="hip",
        default_runtime_status="artifact_only",
        supported_ops=_ops(
            "artifact_only",
            ("tessera.matmul",),
            reason=(
                "ROCm 7.2.4 GFX12/RDNA 4 WMMA artifact planning target; "
                "HIP execution remains gated"
            ),
        ),
        supported_dtypes=(
            "bf16", "fp16", "fp32",
            "fp8_e4m3", "fp8_e5m2",
            "int8", "int32", "int4",
        ),
        features=("wmma_f16", "wmma_bf16", "wmma_f8", "buffer_load_lds", "rocm_7_2_3"),
    ),
    "rocm_gfx1201": TargetCapability(
        name="rocm_gfx1201",
        aliases=("gfx1201", "radeon_ai_pro_r9700", "r9700"),
        family="rocm",
        runtime_backend="hip",
        default_runtime_status="artifact_only",
        supported_ops=_ops(
            "artifact_only", ("tessera.matmul",),
            reason=(
                "ROCm RDNA 4 gfx1201 / Radeon AI PRO R9700 WMMA artifact; "
                "exact-device compile and execution proof remains gated"
            ),
        ),
        supported_dtypes=(
            "bf16", "fp16", "fp32", "fp8_e4m3", "fp8_e5m2",
            "int8", "int32", "int4",
        ),
        features=("wmma_f16", "wmma_bf16", "wmma_f8", "buffer_load_lds", "rocm_7_2_3"),
    ),
    "rocm_gfx1250": TargetCapability(
        name="rocm_gfx1250",
        aliases=("gfx1250", "mi455x"),
        family="rocm",
        runtime_backend="hip",
        default_runtime_status="artifact_only",
        supported_ops=_ops(
            "artifact_only", ("tessera.matmul",),
            reason=(
                "AMD Instinct MI455X / gfx1250 Wave32 WMMA-v2 artifact target; "
                "exact-device execution proof remains gated"
            ),
        ),
        supported_dtypes=("bf16", "fp16", "fp32", "int8"),
        features=("wmma_v2", "wave32", "buffer_load_lds", "rocm_7_2_3"),
    ),
    "apple_cpu": TargetCapability(
        name="apple_cpu",
        aliases=("macos_cpu", "m_series_cpu"),
        family="apple",
        runtime_backend="accelerate",
        default_runtime_status="ready",
        supported_ops={
            **_ops("ready", _CPU_OPS, reason="Apple CPU uses Accelerate where available and reference fallback otherwise"),
            # Sprint 7 (2026-06-03): rank-2 matmul executes in the value lane for
            # f32 (cblas_sgemm) + f16/bf16 (BNNS). The per-op dtype tuple is
            # widened so f16/bf16 rank-2 matmul reaches the value lane instead of
            # being rejected by the frontend capability gate. Rank-2 / no-transpose
            # / no-dynamic is enforced downstream in TilingPass; batched_gemm stays
            # fp32-only (no batched f16/bf16 yet).
            canonical_op("tessera.matmul"): OpCapability(
                canonical_op("tessera.matmul"), "ready",
                dtypes=("fp32", "f32", "fp16", "bf16"),
                reason="Apple CPU rank-2 matmul: f32 via cblas_sgemm, f16/bf16 via BNNS"),
        },
        supported_dtypes=("fp32", "f32", "bf16", "fp16"),
        features=("accelerate", "reference_fallback"),
    ),
    "apple_gpu": TargetCapability(
        name="apple_gpu",
        aliases=("apple", "mac", "macos_gpu", "m_series_gpu"),
        family="apple",
        runtime_backend="metal",
        default_runtime_status="artifact_only",
        # C-fix (Apple plan, 2026-05-20): expand the target-level
        # dtype tuple so it matches what per-op manifest entries
        # actually advertise.  Previously this said only
        # ("fp32", "f32"), but shipped MSL kernels for matmul /
        # softmax / gelu / rope / flash_attn carry fp16 + bf16 ports
        # (mixed precision: half I/O + fp32 accumulators).  Honest
        # tuple = union of dtypes any apple_gpu kernel supports.
        # Per-op truth still lives in ``backend_manifest._APPLE_GPU_KERNELS``
        # / ``_CLIFFORD_APPLE_GPU_FUSED`` / ``_EBM_APPLE_GPU_FUSED``
        # / ``_COMPLEX_APPLE_GPU_FUSED`` and remains the source of
        # truth for "does this specific op support this dtype today".
        supported_ops={
            **_ops("ready", _APPLE_GPU_READY, reason="Apple GPU runtime shim supports this single-op smoke path"),
            **_ops("artifact_only", ("tessera.moe", "tessera.add", "tessera.mul"), reason="Apple GPU target contract exists but native execution is not wired for this op"),
            # E1 (2026-05-20) — 26 fused GA + EBM MSL kernels.  These are
            # the same kernels documented in ``backend_manifest`` and
            # benchmarked end-to-end by ``benchmarks/apple_gpu/benchmark_ga_ebm.py``.
            **_apple_gpu_fused_caps(),
        },
        supported_dtypes=("fp32", "f32", "fp16", "bf16"),
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
