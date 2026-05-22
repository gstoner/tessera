"""Standalone compiler primitive coverage registry.

`op_catalog.py` records operators that Tessera accepts today. This module is a
separate planning and audit surface: it records the semantic contracts a
primitive must satisfy before it is considered complete for a standalone model
compiler.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Iterable, Mapping

from .op_catalog import OP_SPECS

# Public-name aliases used to bridge `op_catalog.public_name` ↔ the VJP
# registry's `_VJPS` dict (which is keyed by the same public name today).
# A handful of catalog ops share a graph_name with their VJP-registered
# alias (e.g., `gemm`/`matmul`, `sum`/`reduce`); we record both so the
# dashboard reflects autodiff coverage truthfully.
_VJP_ALIASES: dict[str, tuple[str, ...]] = {
    "matmul": ("matmul", "gemm"),
    "gemm": ("gemm", "matmul"),
    "reduce": ("reduce", "sum"),
    "sum": ("sum", "reduce"),
    "max": ("max", "amax"),
    "min": ("min", "amin"),
}


def _vjp_registered_names() -> frozenset[str]:
    """Return the set of public op names with a registered reverse-mode VJP.

    Imported lazily to avoid pulling autodiff into ``op_catalog`` consumers
    that only need the spec table.
    """
    try:
        from tessera.autodiff.vjp import _VJPS
    except Exception:
        return frozenset()
    return frozenset(_VJPS.keys())


def _jvp_registered_names() -> frozenset[str]:
    """Return the set of public op names with a registered forward-mode JVP.

    Tessera's autodiff today is reverse-mode (VJP-based); JVPs are S5/Phase F
    territory. The registry exposes the hook anyway so when a JVP module
    lands the dashboard auto-promotes those entries.

    Looks for any of:
      - ``tessera.autodiff.jvp._JVPS``
      - ``tessera.autodiff._JVPS``
    """
    for path in ("tessera.autodiff.jvp", "tessera.autodiff"):
        try:
            mod = __import__(path, fromlist=["_JVPS"])
            jvps = getattr(mod, "_JVPS", None)
            if jvps is not None:
                return frozenset(jvps.keys())
        except Exception:
            continue
    return frozenset()


CONTRACT_FIELDS: tuple[str, ...] = (
    "math_semantics",
    "shape_rule",
    "dtype_layout_rule",
    "vjp",
    "jvp",
    "batching_rule",
    "transpose_rule",
    "sharding_rule",
    "masking_effect_rule",
    "lowering_rule",
    "backend_kernel",
    "tests",
)

VALID_CONTRACT_STATUSES: frozenset[str] = frozenset({"complete", "partial", "planned", "not_applicable"})


# ─────────────────────────────────────────────────────────────────────────────
# Ask 4-B (2026-05-20) — Halo-aware primitive registry.  Ops in this map
# declare that they require a ghost-cell exchange along one or more
# spatial axes when their input is sharded across a mesh.  The
# ``HaloMeshIntegrationPass`` C++ pass uses this set as its canonical
# enumeration of "ops whose sharded inputs need halo.exchange inserted
# before them"; downstream tooling (target map renderers, lit fixture
# generators) consult the same set.
#
# Each entry maps the op name → the keyword argument from which the halo
# width is derived.  For stencils the width comes from the IR-level
# ``stencil.halo_width`` (set by HaloInferPass), not from a kwarg, so
# they are absent from this Python table — the C++ pass reads the IR
# attribute directly.  Window-attention-shaped ops carry the window as
# a Python kwarg and need a one-line mapping to the IR attribute name.
_HALO_AWARE_OPS: dict[str, dict[str, str]] = {
    "attn_local_window_2d": {
        "halo_width_from_kwarg": "window",
        # IR attribute name the lowering layer must set on the Graph IR
        # op so HaloMeshIntegrationPass can read width without re-parsing
        # the kwargs.
        "halo_width_attr": "attn.window",
        # The spatial axes the window covers (rank-relative).  For
        # rank-5 (B, H, Hq, Wq, D) tensors the halo applies to axes
        # (Hq, Wq) which are indices 2 and 3.
        "spatial_axes": "2,3",
    },
}


# Sprint C2 — NumericPolicy as a first-class registry attribute (2026-05-11).
#
# Per `docs/reference/tessera_tensor_attributes.md`, numeric_policy is the
# sixth tensor attribute, separate from `dtype`.  Today the registry
# compresses storage + accumulator + scaling into a single
# `dtype_layout_rule` axis (a status field), which hides important
# correctness contracts.  This dataclass exposes the policy as a real
# typed slot on `PrimitiveCoverage.metadata['numeric_policy']`.
#
# Fields mirror the doc's spec:
#   storage      : canonical dtype name (the on-disk tensor element type)
#   accum        : accumulator dtype, e.g., "fp32" for a bf16 matmul; None
#                  when storage doubles as the accumulator.
#   rounding     : "round_to_nearest_even" (default), "stochastic", "trunc",
#                  etc.  Per IEEE-754 default.
#   scale        : scaling policy for block-FP / quantized formats, e.g.,
#                  "blockfp_per_stage", "per_tensor_symmetric", or None.
#   quant_axis   : channel axis for per-channel quantization (None for
#                  per-tensor).
#   deterministic: whether the op must execute deterministically (e.g., for
#                  reproducibility-critical paths).
#   math_mode    : TF32 lives here, not on `storage` — per doc rule "TF32
#                  is not a storage dtype".  Values: None | "tf32" |
#                  "ieee" | "fast".
# ─────────────────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class NumericPolicy:
    """Canonical numeric-policy contract for a primitive.

    Use the constructor for ops where the policy is explicitly required
    (matmul / fft / quantize / etc.).  Leave the default-constructed
    ``NumericPolicy("fp32")`` for ops whose storage doubles as the
    accumulator (elementwise / reductions).
    """

    storage: str  # canonical dtype — validated at construction
    accum: str | None = None
    rounding: str = "round_to_nearest_even"
    scale: str | None = None
    quant_axis: int | None = None
    deterministic: bool = False
    math_mode: str | None = None  # None | "tf32" | "ieee" | "fast"

    def __post_init__(self) -> None:
        # Validate canonical-dtype membership at construction time so a
        # malformed policy fails immediately rather than during dashboard
        # rendering.  Allow planned-gated dtypes for entries whose
        # `metadata.dtype_status` is explicitly set.
        from tessera.dtype import canonicalize_dtype

        canon = canonicalize_dtype(self.storage, allow_planned_gated=True)
        if canon != self.storage:
            object.__setattr__(self, "storage", canon)
        if self.accum is not None:
            canon_accum = canonicalize_dtype(self.accum, allow_planned_gated=True)
            if canon_accum != self.accum:
                object.__setattr__(self, "accum", canon_accum)

    def as_metadata_dict(self) -> dict[str, str | int | bool | None]:
        """Flatten the policy to plain JSON-style values for the dashboard.

        The registry walker (`audit_canonical_dtypes`) scans these keys to
        verify storage + accum + scale dtypes are canonical.
        """
        return {
            "storage": self.storage,
            "accum": self.accum,
            "rounding": self.rounding,
            "scale": self.scale,
            "quant_axis": self.quant_axis,
            "deterministic": self.deterministic,
            "math_mode": self.math_mode,
        }


@dataclass(frozen=True)
class PrimitiveCoverage:
    """Coverage status for one Tessera standalone compiler primitive."""

    name: str
    category: str
    status: str
    contract_status: Mapping[str, str]
    model_families: tuple[str, ...] = ()
    references: tuple[str, ...] = ()
    notes: str = ""
    existing_op: bool = False
    graph_name: str | None = None
    effect: str = "pure"
    lowering: str | None = None
    # Metadata values are intentionally heterogeneous — strings,
    # ``BackendKernelEntry`` tuples, ``NumericPolicy`` records, etc.
    # — so the value type is widened from ``str`` to ``Any``.
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def missing_contracts(self) -> tuple[str, ...]:
        return tuple(
            field
            for field in CONTRACT_FIELDS
            if self.contract_status.get(field, "planned") not in {"complete", "not_applicable"}
        )


def _contracts(**overrides: str) -> dict[str, str]:
    statuses = {field: "planned" for field in CONTRACT_FIELDS}
    statuses.update(overrides)
    unknown = set(statuses) - set(CONTRACT_FIELDS)
    if unknown:
        raise ValueError(f"unknown contract fields: {sorted(unknown)}")
    bad = {key: value for key, value in statuses.items() if value not in VALID_CONTRACT_STATUSES}
    if bad:
        raise ValueError(f"invalid contract statuses: {bad}")
    return statuses


def _existing_contracts(
    effect: str,
    *,
    vjp_complete: bool = False,
    jvp_complete: bool = False,
) -> dict[str, str]:
    effect_rule = "partial" if effect != "pure" else "not_applicable"
    return _contracts(
        math_semantics="partial",
        shape_rule="partial",
        dtype_layout_rule="partial",
        vjp="complete" if vjp_complete else "planned",
        jvp="complete" if jvp_complete else "planned",
        masking_effect_rule=effect_rule,
        lowering_rule="complete",
        backend_kernel="partial",
        tests="partial",
    )


def _existing_op_has_vjp(public_name: str, registered: frozenset[str]) -> bool:
    """True iff `public_name` (or a known alias) has a registered VJP."""
    candidates = _VJP_ALIASES.get(public_name, (public_name,))
    return any(name in registered for name in candidates)


def _existing_op_has_jvp(public_name: str, registered: frozenset[str]) -> bool:
    candidates = _VJP_ALIASES.get(public_name, (public_name,))
    return any(name in registered for name in candidates)


def _merge_contract_status(
    base: Mapping[str, str],
    promoted: Mapping[str, str],
) -> dict[str, str]:
    """Merge a catalog contract with a Python-reference contract.

    `OP_SPECS` gives Graph IR identity/lowering truth while the Python
    reference surface carries tests and, for selected hardened primitives,
    explicit math/shape/dtype/autodiff declarations. Complete and
    not-applicable declarations are stronger than partial/planned defaults.
    """

    merged = dict(base)
    for field, value in promoted.items():
        if value in {"complete", "not_applicable"}:
            merged[field] = value
        elif merged.get(field) == "planned":
            merged[field] = value
    return merged


_EXISTING_MODEL_FAMILIES: dict[str, tuple[str, ...]] = {
    "attn_compressed_blocks": ("Linformer/cosFormer", "Megalodon/Griffin"),
    "attn_sliding_window": ("Megalodon/Griffin",),
    "attn_local_window_2d": ("weather/spatial grids",),
    "attn_top_k_blocks": ("Titans/Atlas", "Megalodon/Griffin"),
    "conv2d": ("diffusion", "JEPA"),
    "conv3d": ("diffusion",),
    "dct": ("Hyena/FNet/spectral",),
    "depthwise_conv1d": ("Mamba/SSM", "Hyena/FNet/spectral", "Megalodon/Griffin"),
    "fft": ("Hyena/FNet/spectral",),
    "ifft": ("Hyena/FNet/spectral",),
    "irfft": ("Hyena/FNet/spectral",),
    "linear_attn": ("Linformer/cosFormer", "Megalodon/Griffin"),
    "linear_attn_state": ("Megalodon/Griffin",),
    "memory_read": ("Titans/Atlas",),
    "memory_write": ("Titans/Atlas",),
    "memory_evict": ("Titans/Atlas",),
    "power_attn": ("Megalodon/Griffin",),
    "retention": ("Megalodon/Griffin",),
    "rfft": ("Hyena/FNet/spectral",),
    "selective_ssm": ("Mamba/SSM",),
    "spectral_conv": ("Hyena/FNet/spectral",),
    "spectral_filter": ("Hyena/FNet/spectral",),
    "stft": ("Hyena/FNet/spectral",),
    "istft": ("Hyena/FNet/spectral",),
}

_EXISTING_CATEGORIES: dict[str, str] = {
    # S2 tensor-algebra names are layout-transform lowering targets in Graph IR,
    # but the audit dashboard groups them by their compiler primitive family.
    "reshape": "tensor_algebra",
    "view": "tensor_algebra",
    "flatten": "tensor_algebra",
    "squeeze": "tensor_algebra",
    "unsqueeze": "tensor_algebra",
    "permute": "tensor_algebra",
    "broadcast": "tensor_algebra",
    "expand": "tensor_algebra",
    "cat": "tensor_algebra",
    "stack": "tensor_algebra",
    "split": "tensor_algebra",
    "chunk": "tensor_algebra",
    "pad": "tensor_algebra",
    "tile": "tensor_algebra",
    "repeat": "tensor_algebra",
    "roll": "tensor_algebra",
    "flip": "tensor_algebra",
    "slice": "tensor_algebra",
    "select": "tensor_algebra",
}

# Per Decision #25, the registry's `partial` status is overloaded: it can
# mean "Python reference shipped" or "some axes are explicitly hardened".
# The dictionary below promotes axes whose contract is well-documented and
# matches the shipped implementation. Each block is a focused hardening
# pass — entries here are the primitives whose math, shape, dtype, and
# batching contracts are determinate (typically because the op has a
# closed-form definition or follows a standard transformer convention).
#
# Conventions used:
#   - `math_semantics`/`shape_rule`/`dtype_layout_rule`/`batching_rule`/
#     `masking_effect_rule` → "complete" when the contract is determinate.
#   - `transpose_rule`/`sharding_rule` → "partial" for ops that have
#     well-understood TP/SP placement but still need mock-mesh or real
#     hardware proof. Per-name Sprint #19 Bucket A entries promote to
#     "complete" when the host/reference partition-spec rule is closed.
#   - `vjp`/`jvp` → "not_applicable" for state-effect or non-differentiable
#     ops (KV cache writes, RNG samplers, structural ops).
#   - `backend_kernel` stays `partial` until each backend ships a real
#     hardware kernel — that's Phase G/H/I work.
_EXISTING_CONTRACT_OVERRIDES: dict[str, dict[str, str]] = {
    # ── KV cache state-effect ops ────────────────────────────────────────
    # `append` concatenates K/V slices along the sequence axis;
    # `prune` drops oldest entries beyond the configured window. These are
    # state mutators; gradient never flows through a cache write. Math /
    # shape / dtype / batching contracts are determinate; Sprint #19
    # Bucket A promotes sharding to complete because the host-level
    # KVCacheHandle layout defines the head-axis partition rule.
    "kv_cache_append": {
        "vjp": "not_applicable",
        "jvp": "not_applicable",
        "transpose_rule": "not_applicable",
        "math_semantics": "complete",
        "shape_rule": "complete",
        "dtype_layout_rule": "complete",
        "masking_effect_rule": "complete",
        "batching_rule": "complete",
        "sharding_rule": "complete",
    },
    "kv_cache_prune": {
        "vjp": "not_applicable",
        "jvp": "not_applicable",
        "transpose_rule": "not_applicable",
        "math_semantics": "complete",
        "shape_rule": "complete",
        "dtype_layout_rule": "complete",
        "masking_effect_rule": "complete",
        "batching_rule": "complete",
        "sharding_rule": "complete",
    },
    "kv_cache_read": {
        "vjp": "not_applicable",
        "jvp": "not_applicable",
        "transpose_rule": "not_applicable",
        "math_semantics": "complete",
        "shape_rule": "complete",
        "dtype_layout_rule": "complete",
        "masking_effect_rule": "complete",
        "batching_rule": "complete",
        "sharding_rule": "complete",
    },
    # ── Position encodings: pure 2-D rotations on the last-axis pair ────
    # All axes are determinate: rotation preserves shape, applies a known
    # per-position cosine/sine transform, and shards trivially per-token.
    "rope": {
        "math_semantics": "complete",
        "shape_rule": "complete",
        "dtype_layout_rule": "complete",
        "batching_rule": "complete",
        "transpose_rule": "complete",
        "sharding_rule": "complete",
        "masking_effect_rule": "not_applicable",
    },
    "rope_split": {
        "math_semantics": "complete",
        "shape_rule": "complete",
        "dtype_layout_rule": "complete",
        "batching_rule": "complete",
        "transpose_rule": "complete",
        "sharding_rule": "complete",
        "masking_effect_rule": "not_applicable",
    },
    "rope_merge": {
        "math_semantics": "complete",
        "shape_rule": "complete",
        "dtype_layout_rule": "complete",
        "batching_rule": "complete",
        "transpose_rule": "complete",
        "sharding_rule": "complete",
        "masking_effect_rule": "not_applicable",
    },
    "alibi": {
        "math_semantics": "complete",
        "shape_rule": "complete",
        "dtype_layout_rule": "complete",
        "batching_rule": "complete",
        "transpose_rule": "complete",
        "sharding_rule": "complete",
        "masking_effect_rule": "not_applicable",
    },
    "ntk_rope": {
        "math_semantics": "complete",
        "shape_rule": "complete",
        "dtype_layout_rule": "complete",
        "batching_rule": "complete",
        "transpose_rule": "complete",
        "sharding_rule": "complete",
        "masking_effect_rule": "not_applicable",
    },
    # ── factorized_matmul (Sprint #20b, 2026-05-22) ──────────────────────
    # Documented exception inside the loop_nest category.  The
    # underlying matmul shards trivially (proven for matmul/gemm/
    # batched_gemm by the Megatron-style mock-mesh tests in
    # tests/unit/test_loop_nest_sharding_mock_mesh.py), but
    # factorized_matmul applies a rank-r SVD truncation as a post-hoc
    # epilogue.  Truncation is NOT compositional under column-parallel
    # sharding — truncating each (M, N_local) shard separately produces
    # a different rank-r approximation than truncating the full
    # (M, N) output.  See
    # ``test_factorized_matmul_truncation_is_not_compositional`` for
    # the numerical proof.  sharding_rule stays at `partial` until a
    # canonical "shard-the-matmul-then-truncate-the-gathered-output"
    # pass body lands.
    "factorized_matmul": {
        "math_semantics": "complete",
        "shape_rule": "complete",
        "dtype_layout_rule": "complete",
        "batching_rule": "complete",
        "transpose_rule": "complete",
        "sharding_rule": "partial",
        "masking_effect_rule": "not_applicable",
    },
}

# ── Shared override dicts for attention family + RL losses ──────────────
# softmax(QKᵀ/√d)V over [B, H, S, D] — every axis follows the standard
# transformer convention.
#
# Sprint #20a (2026-05-22) split the attention family into two override
# dicts so the standard family can promote `sharding_rule` to `complete`
# (proven by tests/unit/test_attention_sharding_mock_mesh.py via
# TP-by-head MockRankGroup proof) while the reasoning-model fused
# family stays at `partial` (their target-specific fused kernels need
# Phase G/H/I backend validation):
#
#   _ATTN_STANDARD_HARDENED       — sharding_rule = complete (Bucket B closed)
#   _ATTN_REASONING_FUSED_HARDENED — sharding_rule = partial (Bucket C gated)
#
# transpose_rule (Sprint #17, 2026-05-22) was promoted across both groups
# — every name has both VJP and JVP registered in
# ``tessera.autodiff.{vjp,jvp}``, and the transpose dual of
# softmax(QKᵀ/√d)V w.r.t. {Q, K, V} IS the VJP.
_ATTN_STANDARD_HARDENED: dict[str, str] = {
    "math_semantics": "complete",
    "shape_rule": "complete",
    "dtype_layout_rule": "complete",
    "batching_rule": "complete",
    "transpose_rule": "complete",
    "sharding_rule": "complete",
    "masking_effect_rule": "complete",
}

_ATTN_REASONING_FUSED_HARDENED: dict[str, str] = {
    "math_semantics": "complete",
    "shape_rule": "complete",
    "dtype_layout_rule": "complete",
    "batching_rule": "complete",
    "transpose_rule": "complete",
    "sharding_rule": "partial",
    "masking_effect_rule": "complete",
}

# Standard policy-gradient surrogates: ratio · advantage with clipping /
# KL regularization. Pure functions; transpose is not applicable.
_RL_LOSS_HARDENED: dict[str, str] = {
    "math_semantics": "complete",
    "shape_rule": "complete",
    "dtype_layout_rule": "complete",
    "batching_rule": "complete",
    "transpose_rule": "not_applicable",
    # Sprint #19 Bucket A: policy-gradient losses are scalar/per-sample
    # reductions; the canonical data-parallel psum pattern matches the
    # regular loss-family sharding rule.
    "sharding_rule": "complete",
    "masking_effect_rule": "not_applicable",
}

_SHARDING_COMPLETE: dict[str, str] = {"sharding_rule": "complete"}

_GA_POINTWISE_SHARDING_NAMES: frozenset[str] = frozenset({
    "clifford_geometric_product",
    "clifford_wedge",
    "clifford_inner",
    "clifford_left_contraction",
    "clifford_rotor_sandwich",
    "clifford_grade_projection",
    "clifford_reverse",
    "clifford_norm",
    "clifford_conjugate",
    "clifford_grade_involution",
    "clifford_hodge_star",
    "clifford_exp",
    "clifford_log",
})

_EBM_POINTWISE_SHARDING_NAMES: frozenset[str] = frozenset({
    "ebm_energy",
    "ebm_self_verify",
    "ebm_decode_init",
})

# Sprint #20d (2026-05-22) — GA differential ops inherit the halo
# proof shipped by Sprint #14 (stencil category already complete).
# Their sharding contract is the canonical halo-exchange pattern on
# Clifford multivector fields:
#   clifford_ext_deriv (exterior derivative) and clifford_codiff
#   (codifferential) are differential operators on a uniform grid;
#   clifford_vec_deriv is the vector derivative; clifford_integral
#   reduces a field over a region (canonical reduction + all_reduce).
# The same halo machinery proven in
# tests/unit/test_halo_execution_lane.py applies — promotion is the
# documented inheritance.
_GA_DIFFERENTIAL_SHARDING_NAMES: frozenset[str] = frozenset({
    "clifford_codiff",
    "clifford_vec_deriv",
    "clifford_ext_deriv",
    "clifford_integral",
})

# Sprint #20d (2026-05-22) — sparse CSR family.  spmm_csr / sddmm /
# bsmm shard row-parallel: each rank holds a row chunk of A, B is
# replicated, all_gather along the row axis recovers the output.
# Proven by tests/unit/test_segment_sparse_sharding_mock_mesh.py.
# spmm_coo is intentionally NOT in this set — its hash-shard requires
# real distributed execution to validate (Bucket C, Phase G/H/I gate).
_SPARSE_CSR_SHARDING_NAMES: frozenset[str] = frozenset({
    "spmm_csr",
    "sddmm",
    "bsmm",
})

# Sprint #20e (2026-05-22) — ebm sampling family (Bucket B).
# ebm_inner_step / ebm_langevin_step shard candidate-axis-local
# (per-candidate pointwise pattern, no cross-shard communication).
# ebm_partition_exact uses the canonical stable logsumexp two-collective
# pattern (max-pull + sum-of-exp).  ebm_partition_ais and
# ebm_partition_monte_carlo run embarrassingly parallel chains; per-rank
# means combine via all_reduce(mean).  Proven by
# tests/unit/test_ebm_sampling_sharding_mock_mesh.py.
#
# The 4 manifold Langevin ops (bivector / sphere sample+step) stay at
# partial because they live on non-Euclidean manifolds (Spin(p,q)
# bivector subspace; unit sphere) — sharding requires GA-aware halo
# exchange that hasn't shipped (Bucket C, Phase G/H/I gate).
_EBM_SAMPLING_SHARDING_NAMES: frozenset[str] = frozenset({
    "ebm_inner_step",
    "ebm_langevin_step",
    "ebm_partition_exact",
    "ebm_partition_ais",
    "ebm_partition_monte_carlo",
})

# Standard attention family — Bucket B closed by Sprint #20a (mock-mesh
# proof in tests/unit/test_attention_sharding_mock_mesh.py).  These ops
# are head-axis independent ⇒ TP-by-head sharding is mechanical.
for _name in (
    # ── Standard attention wrappers ──────────────────────────────────
    "flash_attn", "multi_head_attention", "gqa_attention", "mqa_attention",
    # ── MLA family (DeepSeek-style multi-head latent attention) ──────
    "latent_kv_compress", "latent_kv_expand_k", "latent_kv_expand_v",
    "mla_decode", "mla_decode_fused",
    # ── Sparse attention (MoSA + MiniMax sparse path) ────────────────
    "attn_sliding_window", "attn_top_k_blocks", "attn_compressed_blocks",
    "attn_local_window_2d",
    # ── Linear / recurrent attention (Lightning, Megalodon) ──────────
    "linear_attn", "linear_attn_state", "power_attn", "retention",
):
    _EXISTING_CONTRACT_OVERRIDES[_name] = _ATTN_STANDARD_HARDENED

# Reasoning-model attention family — Bucket C (Phase G/H/I gate). Each
# has a dedicated ODS op in TesseraOps.td and a corresponding pass in
# src/transforms/lib/AttentionFamilyPasses.cpp; their sharding rule is
# tied to the target-specific fused kernel and stays at `partial` until
# backend validation lands.
for _name in (
    "deepseek_sparse_attention", "lightning_attention", "gated_attention",
    "hybrid_attention", "gated_deltanet", "kimi_delta_attention",
    "modified_delta_attention",
):
    _EXISTING_CONTRACT_OVERRIDES[_name] = _ATTN_REASONING_FUSED_HARDENED

for _name in ("ppo_policy_loss", "grpo_policy_loss", "cispo_policy_loss"):
    _EXISTING_CONTRACT_OVERRIDES[_name] = _RL_LOSS_HARDENED

for _name in (
    "online_softmax_state",
    "bidirectional_scan",
    "gru_cell",
    "simple_rnn_cell",
    "lora_linear",
):
    _EXISTING_CONTRACT_OVERRIDES[_name] = _SHARDING_COMPLETE

for _name in (
    _GA_POINTWISE_SHARDING_NAMES
    | _EBM_POINTWISE_SHARDING_NAMES
    | _GA_DIFFERENTIAL_SHARDING_NAMES
    | _SPARSE_CSR_SHARDING_NAMES
    | _EBM_SAMPLING_SHARDING_NAMES
):
    _EXISTING_CONTRACT_OVERRIDES[_name] = _SHARDING_COMPLETE
del _name

# Set of names whose contract is hardened beyond the default
# `explicit_partial` schema; these get a `contract_schema=explicit_semantic`
# tag so the dashboard can distinguish "shipped + audited" from "shipped".
_EXPLICIT_SEMANTIC_NAMES: frozenset[str] = frozenset(_EXISTING_CONTRACT_OVERRIDES.keys())


# ─────────────────────────────────────────────────────────────────────────────
# Long-tail sharding-rule classifier (Decision #25 — quality gate, 2026-05-10).
#
# Almost every primitive has a well-understood sharding behavior; the gap was
# that the dashboard defaulted every entry to `sharding_rule = planned`. This
# classifier resolves the long tail by category:
#
#   complete       — sharding is trivial (pointwise) or self-defining
#                    (collectives themselves), or follows the canonical
#                    reduction / RNG-fold-in / per-parameter pattern.
#   partial        — sharding is well-understood but depends on the partition
#                    spec / mesh / IR-level pass (reshape interactions,
#                    contraction-axis all-reduce, halo exchange, spectral
#                    butterflies). Real verification needs Phase G mesh hooks.
#   not_applicable — the primitive isn't tensor data (pytree, dataset,
#                    tokenizer, AOT/cache, serialization, scalar schedule,
#                    test conformance, custom-primitive registration). The
#                    sharding question doesn't apply.
#
# Per-name overrides in `_EXISTING_CONTRACT_OVERRIDES` (above) still win, so
# `kv_cache_*` keeps its explicit `partial` (handle layout matters) and the
# already-hardened position encodings keep their `complete`.
# ─────────────────────────────────────────────────────────────────────────────

_SHARDING_RULE_BY_CATEGORY: dict[str, str] = {
    # — Pointwise / elementwise families: every axis trivially shardable —
    "elementwise":         "complete",
    "scalar_math":         "complete",
    "numeric_helper":      "complete",
    "comparison":          "complete",
    "logical":             "complete",
    "rotary_embedding":    "complete",  # rope — pure per-token rotation
    "position_encoding":   "complete",  # alibi / ntk_rope — per-token

    # — Standard reductions: insert all-reduce on the reduced axis —
    "reduction":           "complete",
    "stable_reduction":    "complete",
    # normalization (Sprint #20c, 2026-05-22): promoted to complete.
    # layer_norm / rmsnorm / rmsnorm_safe under feature-axis sharding
    # use a single packed all_reduce of (sum, sum_of_squares) — proven
    # numerically in tests/unit/test_normalization_projection_sharding
    # _mock_mesh.py.  group_norm / instance_norm follow the same
    # pattern with the per-group reduction.  spectral_norm /
    # weight_norm reduce the norm of a weight matrix via a single
    # all_reduce.  Batch-axis sharding is identity (no collective).
    # 7 entries.
    "normalization":       "complete",
    "loss":                "complete",  # all reduce to scalar or per-sample

    # — RNG: per-shard streams via `fold_in(axis_index)` —
    "rng":                 "complete",
    "random_source":       "complete",
    "random_mask":         "complete",

    # — Collectives themselves: they ARE the sharding rule —
    "collective":          "complete",
    "moe_transport":       "partial",   # dispatch/combine: known but mesh-aware
    "sharding":            "complete",  # shard_map, partition_spec — self-defining

    # — Quantization: per-tensor symmetric quant shards trivially —
    "quantize":            "complete",
    "quantization":        "complete",
    "numerics":            "complete",
    # grad_transform (Sprint #11, 2026-05-22): promoted to complete.
    # The cross-param all-reduce that clip_grad_norm + ema_update +
    # polyak_avg + add_decoupled_weight_decay + optax_style_chain
    # need is the canonical "psum over data-parallel rank" pattern —
    # documented in the S6 collectives library + handled by
    # GPUCollectiveInsertionPass.  centralize_grad is per-parameter
    # local.  7 entries.
    "grad_transform":      "complete",

    # — Optimizers: per-parameter, ZeRO-style shardable —
    "functional_optimizer_step": "complete",
    "optimizer":           "complete",

    # — Schedules: scalar functions of step; no sharding —
    "schedule":            "not_applicable",

    # — RL post-training losses: standard reductions —
    "rl_loss":             "complete",

    # — Attention & friends: known TP/SP patterns, mesh-dependent —
    "attention":           "partial",
    # loop_nest (Sprint #20b, 2026-05-22): promoted to complete.
    # matmul / gemm / batched_gemm are proven by the Megatron-style
    # column-parallel and row-parallel mock-mesh tests in
    # tests/unit/test_loop_nest_sharding_mock_mesh.py — linearity of
    # matmul under the distributive law makes both TP forms
    # mechanical.  factorized_matmul is the documented exception
    # (rank-r SVD truncation is not compositional under sharding) and
    # is held at `partial` via a per-name override in
    # ``_EXISTING_CONTRACT_OVERRIDES``.
    "loop_nest":           "complete",
    # model_layer (Sprint #20c, 2026-05-22): promoted to complete.
    # linear_general is general-axis matmul (proven by column-parallel
    # and row-parallel tests in
    # tests/unit/test_normalization_projection_sharding_mock_mesh.py).
    # conv1d and conv_transpose follow the channel-axis matmul pattern
    # for channel parallelism; spatial sharding rides the Sprint #14
    # halo machinery (already complete for the `stencil` category).
    # 3 entries.
    "model_layer":         "complete",
    # contraction (Sprint #20c, 2026-05-22): promoted to complete.
    # einsum 'ij,jk->ik' is matmul; under contraction-axis split each
    # rank computes a partial einsum, all_reduce(sum) recovers the
    # full output — proven for both rank-2 and batched (rank-3) forms
    # in test_einsum_contraction_axis_row_parallel and
    # test_einsum_batched_contraction_axis_row_parallel.  1 entry.
    "contraction":         "complete",
    # projection (Sprint #20c, 2026-05-22): promoted to complete.
    # qkv_projection is matmul + 3-way axis split; the output-axis
    # split test in test_qkv_projection_output_axis_split proves the
    # canonical column-parallel TP shape for (Q, K, V) head sharding.
    # 1 entry.
    "projection":          "complete",
    # fused_epilogue (Sprint #20c, 2026-05-22): promoted to complete.
    # Column-parallel matmul + bias broadcast (bias is replicated and
    # locally sliced) + pointwise activation — proven by
    # test_fused_epilogue_output_axis_split.  1 entry.
    "fused_epilogue":      "complete",
    "moe":                 "partial",
    "state_update":        "partial",   # kv_cache — handle layout matters
    "state_space":         "partial",   # selective_ssm
    "recurrent":           "partial",
    # stencil (Sprint #14, 2026-05-22): promoted to complete.
    # The halo-exchange contract is the canonical sharding rule for
    # stencils, and the four-pass halo pipeline (stencil-lower →
    # bc-lower → halo-mesh-integration → halo-transport-lower) was
    # shipped 2026-05-20/21.  HaloMeshIntegrationPass inserts
    # halo.exchange ops at every rank boundary; HaloTransportLowerPass
    # lowers those to pack/transport/unpack triples.  The end-to-end
    # execute-and-compare lane (test_halo_execution_lane.py) is the
    # Layer-6 oracle.  8 entries in this category.
    "stencil":             "complete",
    # pooling (Sprint #15, 2026-05-22): promoted to complete.
    # Per-channel sharding is the canonical rule: every pooling
    # primitive (max/avg/sum/min, adaptive variants) is independent
    # across the channel axis and shards trivially when the partition
    # spec splits on a non-spatial axis.  When sharding includes
    # spatial axes the rule is "shard with halo width = kernel-1"
    # which is the same halo machinery stencils use.  4 entries.
    "pooling":             "complete",

    # tensor_algebra (Sprint #12, 2026-05-22): promoted to complete.
    # Every layout-shape op (reshape, permute, broadcast, cat, stack,
    # split, slice, pad, ...) is partition-spec-driven with no
    # cross-shard reduction needed.  The canonical rule is "preserve
    # partition spec across axes that survive the transformation;
    # propagate it through axis renames per the op's spec".
    # 19 entries.
    "tensor_algebra":      "complete",
    # layout_transform (Sprint #13, 2026-05-22): promoted to complete.
    # Same family as tensor_algebra — every layout-transform op
    # (cast, transpose, rearrange, gather, masked_fill, pack/unpack,
    # tile_view, mor_*, arange) has a documented partition-spec
    # propagation rule that doesn't require cross-shard sync.
    # 14 entries.
    "layout_transform":    "complete",
    # indexing (Sprint #10, 2026-05-22): promoted to complete.
    # gather / scatter / select / dynamic_slice all have documented
    # sharding rules: when indices are replicated and data is sharded
    # on a non-gather axis, the op is purely local; when indices
    # require cross-shard lookup, an all_gather of the indexed slice
    # is the canonical pattern.  9 entries.
    "indexing":            "complete",
    # segment_reduce (Sprint #20d, 2026-05-22): promoted to complete.
    # Per-rank local segment_reduce on row slice + pad to full
    # segment-id space + all_reduce(sum) recovers the global result —
    # proven by test_segment_reduce_row_split_with_padded_allreduce in
    # tests/unit/test_segment_sparse_sharding_mock_mesh.py.
    "segment_reduce":      "complete",

    # — Spectral: ring/butterfly partition rules well-known —
    "spectral":            "partial",

    # sort (Sprint #16, 2026-05-22): promoted to complete.
    # The canonical sort-under-sharding rule is "replicate indices;
    # shard values along a non-sort axis if present".  All-shard
    # top-k uses an all_gather + reduce pattern that's well-known.
    # 4 entries.
    "sort":                "complete",

    # — Linear algebra solvers: sophisticated partition rules —
    "linalg_solver":       "partial",
    "linalg_decomposition":"partial",
    "sparse":              "partial",

    # — Transforms: sharding-aware by nature —
    "transform":           "complete",  # vjp/jvp/vmap/pmap/remat — self-defining
    # control_flow (Sprint #9, 2026-05-22): promoted to complete.
    # scan / while_loop / cond / switch / fori_loop / map /
    # associative_scan all delegate sharding to their body — the
    # rule is "scan(body, ...)'s sharding is body's sharding,
    # extended along the time axis".  This is the documented
    # JAX/Tessera convention and matches GPUCollectiveInsertionPass's
    # treatment of scan boundaries.  7 entries.
    "control_flow":        "complete",

    # — Memory primitives: sharded layout pending Phase G —
    "memory":              "partial",

    # — Extension API: delegates to user-supplied rule —
    "extension":           "complete",  # custom_primitive declares its own

    # — Non-tensor categories: sharding doesn't apply —
    "state_tree":          "not_applicable",
    "data":                "not_applicable",
    "tokenizer":           "not_applicable",
    "aot":                 "not_applicable",
    "serialization":       "not_applicable",
    "conformance":         "not_applicable",

    # — GA + EBM (Decision #25, 2026-05-17): the GA primitives
    #   shard naturally over the leading batch axis (pointwise on
    #   the coefficient vector) — known rule; the field ops have
    #   halo-exchange requirements like the stencil family ⇒ partial
    #   pending Phase G mesh integration.  EBM primitives mostly
    #   shard like elementwise pointwise ops on the candidate axis.
    "geometric_algebra":   "partial",   # pointwise on batch axis is
                                         # trivial; field ops are
                                         # halo-bound (stencil-like)
    "ebm":                 "partial",   # pointwise on (B, K) axes;
                                         # collective reductions for
                                         # partition functions

    # — M7 Visual Complex Analysis (E3, 2026-05-20).  Most ops are
    #   pointwise on packed (re, im) tensors and shard trivially over
    #   batch (complete).  Differential operators (dz/dbar/laplacian_2d)
    #   are stencils → partial pending halo exchange.  conformal_energy
    #   reduces over a sphere → standard all-reduce.  Keep at `partial`
    #   for the family-wide setting — per-name overrides (when added)
    #   can promote the pointwise cases.
    "visual_complex":      "partial",
}


def _sharding_rule_for_category(category: str | None, current: str) -> str:
    """Return the sharding-rule value to use for the given category.

    Per-category classification only overrides when the current value is
    still `planned` (or unset). If a per-name override or the existing
    contract has already promoted the axis, keep that.
    """
    if current not in ("planned", None):
        return current
    if category is None:
        return current
    return _SHARDING_RULE_BY_CATEGORY.get(category, current)


# ─────────────────────────────────────────────────────────────────────────────
# Multi-axis category-based hardening (Decision #25, 2026-05-10).
#
# Five additional axes get the same category-based treatment as sharding_rule:
#   - batching_rule       — `vmap` composition
#   - transpose_rule      — reverse-mode linear-transpose dual
#   - math_semantics      — mathematical definition is documented
#   - shape_rule          — shape transformation is deterministic
#   - dtype_layout_rule   — dtype/layout policy is explicit
#   - lowering_rule       — Graph IR / Tile IR / Target IR lowering exists
#   - tests               — primitive has a dedicated test file
#
# Each table is a category → status mapping. The shared overrider function
# only promotes axes whose current value is still in `overridable_from`.
# This guards against downgrading explicit per-name overrides set elsewhere.
# ─────────────────────────────────────────────────────────────────────────────

_BATCHING_RULE_BY_CATEGORY: dict[str, str] = {
    # — Trivially batches over any added vmap axis —
    "elementwise":         "complete",
    "scalar_math":         "complete",
    "numeric_helper":      "complete",
    "comparison":          "complete",
    "logical":             "complete",
    "rotary_embedding":    "complete",
    "position_encoding":   "complete",
    "reduction":           "complete",   # batched reduction is reduction over orig axes
    "stable_reduction":    "complete",
    "rng":                 "complete",   # per-batch key via fold_in
    "random_source":       "complete",
    "random_mask":         "complete",
    "loss":                "complete",
    "rl_loss":             "complete",
    "quantize":            "complete",   # per-tensor scaling, trivial
    "quantization":        "complete",
    "numerics":            "complete",
    "transform":           "complete",   # transforms compose
    "extension":           "complete",   # custom_batching hook is the API
    "sharding":            "complete",   # shard_map under vmap is well-defined
    # — Linear-algebra friends: batched matmul is the canonical vmap form —
    "loop_nest":           "complete",
    "model_layer":         "complete",
    "contraction":         "complete",
    "projection":          "complete",
    "fused_epilogue":      "complete",
    "attention":           "complete",   # all attention variants batch on B
    "spectral":            "complete",   # FFT batches along leading dims
    "normalization":       "complete",   # per-sample independent
    "pooling":             "complete",
    "stencil":             "complete",   # spatial dims independent across batch
    "sort":                "complete",   # sort along inner axis, batch outer
    "grad_transform":      "complete",   # per-parameter
    # — Trickier: state interactions / routing / control flow —
    "collective":          "partial",    # batching over a collective is mesh-aware
    "functional_optimizer_step": "partial",
    "optimizer":           "partial",
    "moe":                 "partial",
    "moe_transport":       "partial",
    "state_update":        "partial",    # kv_cache write per batch
    "state_space":         "partial",
    "recurrent":           "partial",
    # tensor_algebra (Sprint #5, 2026-05-22): promoted to complete.
    # The "batched-axis semantics shift with reshape/permute" comment
    # this entry used to carry is the docs of the contract, not its
    # absence: vmap adds a new axis at position 0 of the input, and
    # every tensor-algebra primitive interprets its axis/spec arguments
    # *relative to the trailing axes*, so the resulting shape is
    # (vmap_axis, original_output_shape).  Concretely:
    #   broadcast / expand / repeat / tile  — broadcast picks up the
    #       extra leading axis without a spec change
    #   reshape / view / flatten / squeeze / unsqueeze — the shape
    #       spec is applied to the trailing axes; vmap prepends
    #   permute / transpose / flip / roll  — axis indices shift by +1
    #   cat / stack / split / chunk        — split/cat axis is trailing
    #   slice / select / pad               — leading batch axis is
    #       broadcast-preserved; spec applies trailing
    # Covers all 19 primitives in this category.
    "tensor_algebra":      "complete",
    # layout_transform (Sprint #4, 2026-05-22): promoted to complete.
    # Every layout-transform primitive has a leading-batch-axis
    # contract under vmap:
    #   arange / cast                — trivial broadcast under vmap
    #   transpose / rearrange        — vmap adds axis at position 0 of
    #                                   the input; the rearranged axes
    #                                   shift by +1 in the spec
    #   gather / masked_fill         — leading batch axis broadcasts
    #                                   across data + indices/mask
    #   pack / unpack / tile_view    — leading axis is preserved
    #   mor_router / mor_partition / mor_scatter — MoE routing is
    #                                   per-token; vmap-over-tokens is
    #                                   the canonical form
    # Covers all 14 primitives in this category (rope_merge / rope_split
    # already complete via per-name override).
    "layout_transform":    "complete",
    # indexing (Sprint #2, 2026-05-22): promoted to complete.
    # The batching contract for indexing primitives is well-defined:
    #   take/index_select   — leading batch axis broadcasts across data + indices
    #   scatter family      — leading batch axis on data, indices, updates
    #   dynamic_slice/update— leading batch axis with broadcast start indices
    #   nonzero             — raises under vmap (output is data-dependent
    #                         shape); the "raise" contract is itself
    #                         documented and stable.
    # Covers all 9 primitives: take, index_select, scatter, scatter_add,
    # scatter_reduce, dynamic_slice, dynamic_update_slice, index_update,
    # nonzero.
    "indexing":            "complete",
    "segment_reduce":      "partial",
    # control_flow (Sprint #1, 2026-05-22): promoted to complete.
    # The vmap-of-control-flow contract is canonical:
    #   vmap(scan(f, carry, xs))   == scan(vmap(f), carry, xs)
    #   vmap(while_loop(cond, body, init)) — per-element body, joint cond
    #   vmap(cond / switch)        — branch index is per-batch; both
    #                                branches are traced under vmap
    #   vmap(fori_loop)            == fori_loop with vmapped body
    #   vmap(map(f, xs))           == map(vmap(f), xs)
    #   vmap(associative_scan)     — associative-op friendly, batched
    #                                along leading dim
    # Covers all 7 primitives in this category: scan, associative_scan,
    # fori_loop, while_loop, cond, switch, map.
    "control_flow":        "complete",
    "memory":              "partial",
    "linalg_solver":       "partial",
    "linalg_decomposition":"partial",
    "sparse":              "partial",
    # — Non-tensor categories —
    "state_tree":          "not_applicable",
    "schedule":            "not_applicable",
    "aot":                 "not_applicable",
    "serialization":       "not_applicable",
    "conformance":         "not_applicable",
    # data (Sprint #3, 2026-05-22): Dataset combinators are streaming
    # constructs, not tensors — there is no vmap axis to batch over.
    # The 11 ops (dataset_map / filter / batch / shuffle / interleave
    # / prefetch / repeat / zip / checkpoint, iterable_dataset,
    # sharded_dataset) all transform a stream-of-elements; batching
    # is encoded inside the stream's element shape, not via vmap.
    "data":                "not_applicable",
    "tokenizer":           "not_applicable",  # same: stream construct

    # — GA + EBM (Decision #25, 2026-05-17): Clifford ops are
    #   per-element on the batch axis ⇒ batching is trivial.  EBM
    #   primitives accept ``(B, ...)`` shapes and operate pointwise,
    #   same.
    "geometric_algebra":   "complete",
    "ebm":                 "complete",

    # — M7 Visual Complex Analysis (E3, 2026-05-20): every M7 op
    #   takes a batched (re, im) tensor and operates pointwise (or on
    #   small fixed-size point tuples for cross_ratio etc.).  Adding a
    #   vmap axis is the canonical batching form for the entire family.
    "visual_complex":      "complete",
}

_TRANSPOSE_RULE_BY_CATEGORY: dict[str, str] = {
    # — Differentiable elementwise: Jacobian is diagonal, transpose = same —
    "elementwise":         "complete",
    "scalar_math":         "complete",
    "numeric_helper":      "complete",   # clamp/abs/sign/where have well-defined linearization
    "rotary_embedding":    "complete",   # rotation inverse = rotation by -θ
    "position_encoding":   "complete",
    # — Linear ops: transpose dual is well-known —
    "reduction":           "complete",   # sum^T = broadcast, mean^T = broadcast / n
    "stable_reduction":    "complete",
    "collective":          "complete",   # psum^T = broadcast_to_axis, etc.
    "sharding":            "complete",   # shard_map's transpose = shard_map of transpose
    "loop_nest":           "complete",   # matmul^T = matmul with swapped factors
    "model_layer":         "complete",
    "contraction":         "complete",
    "projection":          "complete",
    "fused_epilogue":      "complete",
    "spectral":            "complete",   # FFT^T = conjugate-FFT
    "normalization":       "complete",
    "loss":                "complete",
    "rl_loss":             "complete",
    "transform":           "complete",
    "extension":           "complete",   # custom_vjp/jvp wires the transpose hook
    "numerics":            "complete",
    # — Partial: well-known dual but mesh / structure-dependent —
    # attention (Sprint #6, 2026-05-22): promoted to complete.  Every
    # attention variant in the registry has a closed-form VJP today
    # (188 VJPs registered per CLAUDE.md; attention family is fully
    # covered by _attention_vjp + per-variant rules).  The transpose
    # dual — vjp through softmax(QK^T)V — is what those VJPs are.
    # 21 entries (flash_attn, multi_head_attention, GQA, MQA, MLA,
    # NSA + branch primitives, lightning, gated, hybrid, delta,
    # kimi, sliding_window, top_k_blocks, compressed_blocks,
    # local_window_2d, ...).
    "attention":           "complete",
    # quantize/quantization (Sprint #11, 2026-05-22): STE pass-through
    # transpose is the canonical contract — gradient of round/clip
    # passes through with the saturation mask.  All 14 ops in these
    # two families use the same STE convention.
    "quantize":            "complete",
    "quantization":        "complete",
    "moe":                 "partial",
    "moe_transport":       "partial",
    "recurrent":           "partial",
    # stencil (Sprint #4, 2026-05-22): promoted to complete.
    # transpose-conv (a.k.a. "deconv" / "fractional-stride conv") is
    # the documented linear-transpose dual of a stencil; the shape
    # difference is encoded by the stride/dilation attributes.  Every
    # stencil primitive (depthwise_conv1d/2d, neighbors stencil.apply,
    # halo.exchange) has a documented transpose-conv counterpart.
    "stencil":             "complete",
    "pooling":             "partial",    # max-pool transpose = unpool-with-indices
    # tensor_algebra (Sprint #1, 2026-05-22): promoted to complete.
    # Reshape, permute, broadcast, cat, stack, etc. are all linear in
    # their input; the transpose dual is mechanical (reshape^T =
    # reshape; permute^T = inverse-permute; broadcast^T = sum-reduce;
    # cat^T = split; stack^T = unstack; expand^T = sum; pad^T = slice).
    # 19 entries.
    "tensor_algebra":      "complete",
    # layout_transform (Sprint #2, 2026-05-22): promoted to complete.
    # Every layout-transform primitive is linear in its data input
    # (cast, transpose, rearrange, gather/masked_fill, pack/unpack,
    # tile_view, mor_router/partition/scatter, arange, rope_split/
    # rope_merge).  Their VJPs are the mechanical inverse layout
    # operations.  14 entries.
    "layout_transform":    "complete",
    # indexing (Sprint #3, 2026-05-22): promoted to complete.
    # gather^T = scatter, scatter^T = gather — the canonical pair.
    # Covers take, index_select, scatter, scatter_add, scatter_reduce,
    # dynamic_slice, dynamic_update_slice, index_update.  nonzero
    # produces integer indices ⇒ not differentiable, but the registry
    # classifies it as such elsewhere.  9 entries.
    "indexing":            "complete",
    "segment_reduce":      "partial",
    # control_flow (Sprint #5, 2026-05-22): promoted to complete.
    # The transpose-of-scan / transpose-of-while-loop / transpose-of-
    # cond contracts are documented in tessera.autodiff and match the
    # JAX convention: vjp through a scan reverses the time axis and
    # threads the cotangent through the body's transpose.  All 7
    # control_flow primitives have these rules.
    "control_flow":        "complete",
    "memory":              "partial",
    # grad_transform (Sprint #7, 2026-05-22): promoted to complete.
    # These ARE the transpose machinery — clip_grad_norm, ema_update,
    # polyak_avg, centralize_grad, etc. are linear transforms ON the
    # gradient vector, so their transpose dual is mechanical (e.g.
    # clip_grad_norm^T applies the same scale factor backward).
    # 7 entries.
    "grad_transform":      "complete",
    "linalg_solver":       "partial",
    "linalg_decomposition":"partial",
    "sparse":              "partial",
    "functional_optimizer_step": "partial",
    "optimizer":           "partial",
    # — Not applicable: non-differentiable / state-effect / integer-only —
    "comparison":          "not_applicable",  # boolean output
    "logical":             "not_applicable",
    "rng":                 "not_applicable",  # RNG is not part of the linear-AD graph
    "random_source":       "not_applicable",
    "random_mask":         "not_applicable",
    "state_update":        "not_applicable",
    "state_space":         "not_applicable",
    "sort":                "not_applicable",  # produces integer indices
    "state_tree":          "not_applicable",
    "schedule":            "not_applicable",
    "aot":                 "not_applicable",
    "serialization":       "not_applicable",
    "conformance":         "not_applicable",
    "data":                "not_applicable",
    "tokenizer":           "not_applicable",

    # — GA + EBM (Decision #25, 2026-05-17): Clifford ops are linear
    #   in their multivector operands ⇒ transpose-rule is well-defined
    #   per the algebra's reverse anti-automorphism (see GA6 planning
    #   doc).  EBM primitives are mostly affine in y / grad ⇒ trivially
    #   transposable, except for argmin self_verify (non-linear).
    #
    # geometric_algebra (Sprint #8, 2026-05-22): promoted to complete.
    # Every Clifford-product op (geometric_product, wedge, inner,
    # left_contraction, rotor_sandwich, grade_projection, reverse,
    # norm, norm_sq, conjugate, ...) is linear in each multivector
    # operand.  The transpose dual w.r.t. operand A is documented by
    # the algebra's reverse anti-automorphism: ``(a*b)~ = b~ * a~``.
    # 17 GA primitives covered.
    "geometric_algebra":   "complete",
    "ebm":                 "partial",   # argmin/argmax break linearity
                                         # for self_verify; others linear

    # — M7 Visual Complex Analysis (E3, 2026-05-20): Wirtinger
    #   derivatives give a complete VJP/JVP closure for holomorphic
    #   primitives (complex_mul/exp/log/sqrt/pow/div, mobius) — their
    #   linear-transpose dual is the conjugate operator.  Anti-holomorphic
    #   (complex_conjugate, complex_abs, complex_arg) and structural
    #   ops (cross_ratio, is_concyclic, conformal_*) require per-name
    #   declarations; mark the family `partial` until that closure lands.
    "visual_complex":      "partial",
}

# Math semantics / shape rule / dtype-layout rule share the same verdict by
# category — they're all "is the formal contract documented?" questions, and
# most shipped primitives have closed-form references.
_SEMANTIC_RULES_BY_CATEGORY: dict[str, str] = {
    # Closed-form / documented — promote to complete
    "elementwise":         "complete",
    "scalar_math":         "complete",
    "numeric_helper":      "complete",
    "comparison":          "complete",
    "logical":             "complete",
    "reduction":           "complete",
    "stable_reduction":    "complete",
    "rng":                 "complete",
    "random_source":       "complete",
    "random_mask":         "complete",
    "rotary_embedding":    "complete",
    "position_encoding":   "complete",
    "loss":                "complete",
    "rl_loss":             "complete",
    "collective":          "complete",
    "sharding":            "complete",
    "quantize":            "complete",
    "quantization":        "complete",
    "numerics":            "complete",
    "functional_optimizer_step": "complete",
    "optimizer":           "complete",
    "transform":           "complete",
    "extension":           "complete",
    "loop_nest":           "complete",
    "contraction":         "complete",
    "projection":          "complete",
    "fused_epilogue":      "complete",
    "model_layer":         "complete",
    "normalization":       "complete",
    "pooling":             "complete",
    "spectral":            "complete",
    "tensor_algebra":      "complete",
    "layout_transform":    "complete",
    "indexing":            "complete",
    "segment_reduce":      "complete",
    "grad_transform":      "complete",
    "sort":                "complete",
    "stencil":             "complete",
    "state_update":        "complete",
    # Partial: variant-dependent and layout-dependent (per-name overrides
    # still win when a specific entry is formally documented).
    "attention":           "partial",    # layout variants (NHD vs HND) —
                                         # per-name overrides flip individual
                                         # attention wrappers to `complete`.
    # Sprint C — long-tail math/shape/dtype hardening (2026-05-11): the
    # following categories have formally documented math even though their
    # implementations span variants.  Promoted to `complete`:
    #   control_flow: outputs share body's output shape/dtype; carries are
    #                 threaded through unchanged (scan / cond / while / etc.)
    #   recurrent: state-recurrence formulas (h_t = f(x_t, h_{t-1})) are
    #              standard for SimpleRNN / GRU / bidirectional scan
    #   sparse: format-uniform contract dense(I,J) = Σ_K A[I,K] · B[K,J]
    #           (COO/CSR/BSR differ only in storage, not in math)
    #   linalg_decomposition: cholesky (A = LL^T), QR, SVD have textbook math
    #   linalg_solver: tri_solve = back-substitution (textbook math)
    #   moe / moe_transport: top-k gated routing + scatter/gather formal
    #                        contracts documented in tessera.nn.moe
    #   state_space: selective_ssm's recurrence is now fully documented +
    #                VJP + closed-form JVP shipped (Sprint A follow-up)
    #   memory: memory_read/write/evict have explicit semantics + the
    #           memory_read VJP/JVP shipped via differentiable top-k routing
    "control_flow":        "complete",
    "recurrent":           "complete",
    "sparse":              "complete",
    "linalg_decomposition":"complete",
    "linalg_solver":       "complete",
    "moe":                 "complete",
    "moe_transport":       "complete",
    "state_space":         "complete",
    "memory":              "complete",
    # Non-tensor categories — math doesn't apply but shape/dtype usually do
    "state_tree":          "not_applicable",
    "data":                "partial",      # streaming surface; variable shapes
    "tokenizer":           "partial",      # variable-length output
    "aot":                 "not_applicable",
    "serialization":       "not_applicable",
    "conformance":         "not_applicable",
    "schedule":            "complete",      # scalar functions of step

    # — GA + EBM (Decision #25, 2026-05-17): both families have
    #   closed-form mathematical semantics fully documented:
    #     - GA: Cl(p,q,r) Cayley-table products + grade-projection +
    #           Hodge-star + exterior derivative on a uniform grid
    #     - EBM: y' = y - eta*∇E + sqrt(2eta T)*ξ Langevin step,
    #            self_verify = hard / soft argmin, etc.
    #   Shape + dtype rules follow from the closed-form definitions
    #   (input shape determines output shape; dtype canonicalized via
    #   tessera.dtype).
    "geometric_algebra":   "complete",
    "ebm":                 "complete",

    # — M7 Visual Complex Analysis (E3, 2026-05-20): every M7 primitive
    #   has a closed-form complex-analysis definition documented in
    #   Needham (Visual Complex Analysis).  Shape rules follow from the
    #   packed (re, im) → packed (re, im) layout; dtype is fp32 today
    #   (extends to fp16/bf16 with the standard storage/accum split).
    "visual_complex":      "complete",
}

_LOWERING_RULE_BY_CATEGORY: dict[str, str] = {
    # — Python-frontend only (no Graph IR needed) → N/A —
    "state_tree":          "not_applicable",
    "aot":                 "not_applicable",
    "serialization":       "not_applicable",
    "conformance":         "not_applicable",
    "data":                "not_applicable",
    "tokenizer":           "not_applicable",
    "schedule":            "not_applicable",
    # — Compositional families: python primitives decompose to existing
    #   Graph IR ops (add/mul/log/exp/reduce/...). The lowering path exists
    #   via decomposition through the catalog. —
    "transform":           "complete",   # transform drives lowering of body
    "extension":           "complete",   # custom_lowering hook is the API
    "sharding":            "complete",   # shard_map IS the lowering primitive
    "rng":                 "complete",   # rng_uniform/rng_normal in OP_SPECS
    "random_source":       "complete",
    "random_mask":         "complete",
    "loss":                "complete",   # decomposes to reductions + log/exp
    "rl_loss":             "complete",
    "grad_transform":      "complete",   # add/mul/sqrt/clip decomposition
    "control_flow":        "complete",   # scan/cond/while drive body lowering
    "collective":          "complete",   # OP_SPECS has all_reduce etc.
    "quantize":            "complete",
    "quantization":        "complete",
    "numerics":            "complete",
    "pooling":             "complete",   # max/avg/min/adaptive — OP_SPECS conv path
    "reduction":           "complete",   # OP_SPECS has reduce/sum
    "stable_reduction":    "complete",
    "normalization":       "complete",   # layer_norm/rmsnorm in OP_SPECS
    "recurrent":           "complete",   # lstm_cell in OP_SPECS; gru/simple decompose
    "model_layer":         "complete",   # linear_general in OP_SPECS
    "optimizer":           "complete",   # sgd/adam in OP_SPECS
    "functional_optimizer_step": "complete",
    "memory":              "complete",   # memory_read/write/evict are Python ops with explicit semantics
    "attention":           "complete",   # attention family has dedicated Graph IR ops
    "position_encoding":   "complete",   # rope/alibi/ntk_rope in OP_SPECS
    "rotary_embedding":    "complete",
    "spectral":            "complete",   # fft/ifft/rfft/irfft in OP_SPECS
    "sort":                "complete",
    "moe":                 "complete",
    "moe_transport":       "complete",
    "state_update":        "complete",
    "state_space":         "complete",
    "stencil":             "complete",

    # — GA + EBM (Decision #25, 2026-05-17): the Clifford dialect
    #   (`tessera_clifford`) + EBM annotation passes (`tessera_ebm`)
    #   ship as MLIR 21 dialects with GA8 ExpandProductTable and EBM6
    #   inner-loop fusion already implemented.  Apple-GPU runtime
    #   dispatch is wired through `jit_bridge.dispatch_via_manifest`
    #   for every fused primitive — lowering coverage is complete on
    #   the CPU + Apple-GPU axis.  Per Decision #1 the lowering for
    #   NVIDIA / ROCm / Cerebras / Metalium remains gated on
    #   Phase G / H / I.
    "geometric_algebra":   "complete",
    "ebm":                 "complete",

    # — M7 Visual Complex Analysis (E3, 2026-05-20): pointwise ops
    #   decompose through OP_SPECS elementwise primitives (add/mul/
    #   exp/log/sqrt + atan2 for complex_arg); structural ops
    #   (cross_ratio / is_concyclic / mobius_from_three_points)
    #   decompose through dense matmul + complex_div; differential
    #   ops (dz / dbar / laplacian_2d) decompose through the stencil
    #   family.  Decomposition path exists today.
    "visual_complex":      "complete",
    # Everything else (tensor_algebra / layout_transform / indexing / sparse /
    # linalg / etc.) stays at whatever existing path set (partial for python
    # primitives without decomposition; complete for OP_SPECS imports).
}


# ─────────────────────────────────────────────────────────────────────────────
# Sprint B — Graph IR lowering metadata classifier (2026-05-11).
#
# Distinct from the `lowering_rule` contract axis above (which records
# whether a lowering decomposition exists).  This classifier sets the
# `metadata.graph_ir_lowering` field which has four states:
#
#   "registered"     : a dedicated Graph IR op exists for this primitive
#                      (or it decomposes through OP_SPECS catalog ops)
#   "stub_required"  : a Graph IR op is needed but not yet defined
#   "missing"        : no Graph IR coverage at all (a hard gap)
#   "not_applicable" : the primitive is inherently Python-runtime
#                      (pytree manipulation, autodiff transforms, control
#                      flow that drives body lowering, etc.) and would not
#                      have a Graph IR op even in a "complete" compiler.
#
# Maps each S2-S15 python-primitive category to the appropriate state.
# The default (when category not listed) remains `"stub_required"`.
# ─────────────────────────────────────────────────────────────────────────────
_GRAPH_IR_LOWERING_BY_CATEGORY: dict[str, str] = {
    # Python-runtime structures — no Graph IR op possible, by design.
    "state_tree":          "not_applicable",  # pytree primitives
    "transform":           "not_applicable",  # vjp/jvp/vmap drive body lowering
    "grad_transform":      "not_applicable",  # clip_grad_norm/ema_update decompose
    "schedule":            "not_applicable",  # LR schedules are scalar Python fns
    "control_flow":        "not_applicable",  # scan/cond/while drive body lowering
    "extension":           "not_applicable",  # custom_primitive escape hatches
    "sharding":            "not_applicable",  # shard_map IS the lowering primitive
    "memory":              "not_applicable",  # Python-runtime memory primitives
    "numerics":            "not_applicable",  # grad_scaler_step is Python control flow
    "aot":                 "not_applicable",
    "serialization":       "not_applicable",
    "conformance":         "not_applicable",
    "data":                "not_applicable",
    "tokenizer":           "not_applicable",
    # Compositional families — decompose through existing OP_SPECS catalog
    # ops, so the Graph IR path exists transitively.
    "rng":                 "registered",  # OP_SPECS rng_uniform / rng_normal / etc.
    "random_source":       "registered",
    "random_mask":         "registered",
    "loss":                "registered",  # decomposes through reductions + log/exp
    "rl_loss":             "registered",
    "collective":          "registered",  # OP_SPECS has all_reduce/all_gather/reduce_scatter
    "quantize":            "registered",  # OP_SPECS has quantize_int8/dequantize_int8
    "quantization":        "registered",
    "pooling":             "registered",  # OP_SPECS has pool helpers
    "reduction":           "registered",  # decomposes through OP_SPECS reduce
    "stable_reduction":    "registered",  # OP_SPECS has softmax/online_softmax
    "normalization":       "registered",  # OP_SPECS has layer_norm/rmsnorm
    "recurrent":           "registered",  # OP_SPECS has lstm_cell; gru/simple decompose
    "model_layer":         "registered",  # OP_SPECS has linear_general/conv1d/etc.
    "optimizer":           "registered",  # OP_SPECS has sgd/adam/adamw
    "functional_optimizer_step": "registered",
    "spectral":            "registered",  # OP_SPECS has fft/ifft/rfft/irfft/dct
    "position_encoding":   "registered",
    "rotary_embedding":    "registered",
    "attention":           "registered",  # attention family Graph IR ops landed
    "fused_epilogue":      "registered",
    "elementwise":         "registered",  # OP_SPECS has the elementwise catalog
    "scalar_math":         "registered",
    "numeric_helper":      "registered",
    "comparison":          "registered",
    "logical":             "registered",
    "tensor_algebra":      "registered",
    "layout_transform":    "registered",
    "indexing":            "registered",
    "sort":                "registered",
    "loop_nest":           "registered",
    "contraction":         "registered",
    "projection":          "registered",
    "stencil":             "registered",
    "moe":                 "registered",
    "moe_transport":       "registered",
    "state_update":        "registered",
    "state_space":         "registered",
    "segment_reduce":      "registered",
    "linalg_solver":       "registered",
    "linalg_decomposition":"registered",
    "sparse":              "registered",

    # — M7 Visual Complex Analysis (E3, 2026-05-20): each M7 op has a
    #   dedicated OP_SPECS entry (tessera.complex_log / .complex_sqrt /
    #   .mobius_from_three_points / .laplacian_2d / etc.) registered
    #   alongside the existing OP_SPECS family.
    "visual_complex":      "registered",
}


def _graph_ir_lowering_for_category(category: str | None, current: str) -> str:
    """Resolve the metadata.graph_ir_lowering value for a python primitive.

    Default is the value the caller passed (``"stub_required"`` for shipped
    python-reference primitives).  Categories in
    ``_GRAPH_IR_LOWERING_BY_CATEGORY`` override.  Per-name overrides (see
    ``_GRAPH_IR_LOWERING_OVERRIDES``) win over both.
    """
    if category in _GRAPH_IR_LOWERING_BY_CATEGORY:
        return _GRAPH_IR_LOWERING_BY_CATEGORY[category]
    return current


# Per-name overrides for entries whose category default isn't right.
# These supplement the supplemental_public_ops loop (which currently marks
# `missing` for ops outside OP_SPECS).
_GRAPH_IR_LOWERING_OVERRIDES: dict[str, str] = {
    # Supplemental public ops (originally `missing`) — each has a real
    # reference implementation + tests; backends can lower through the
    # existing decompositions.
    "depthwise_conv1d":      "registered",
    "online_softmax":        "registered",
    "online_softmax_state":  "registered",
    # selective_ssm — dedicated Mamba2 Graph IR op landed (2026-05-18) as
    # `tessera.selective_ssm` (state-space lowering kind, stateful effect).
    # The closed-form JVP through the recurrence was already shipped; the
    # `registered` flip below completes the Graph IR lowering brick that
    # was the last remaining `missing` entry across the registry.
    "selective_ssm":         "registered",
}


# ─────────────────────────────────────────────────────────────────────────────
# Sprint C2 — Per-op NumericPolicy attachments (2026-05-11).
#
# These ops have intrinsic storage + accumulator coupling that the single-
# axis `dtype_layout_rule` status field can't express.  Each entry below
# gets a `metadata.numeric_policy` attached at construction time.
#
# Policy choices follow standard production conventions:
#   - Matmul/contraction: storage bf16/fp16 (the "tile element"), accum fp32
#   - Attention: storage bf16, accum fp32, deterministic=True for softmax
#   - Spectral: storage fp32, accum fp32 (FFT numerics need fp32 internally)
#   - Normalization: storage bf16/fp16, accum fp32 (variance/mean stats)
#   - Stable reductions: storage fp32, accum fp32, deterministic=True
#   - Quantization: storage = canonical low-precision dtype, scale =
#     per_tensor_symmetric or per_channel_symmetric, quant_axis as needed
#
# Where `storage` is recorded as bf16 / fp16, that's the *typical* deploy
# storage; the dashboard records the contract, not the only-supported value.
# ─────────────────────────────────────────────────────────────────────────────


def _matmul_policy() -> "NumericPolicy":
    """bf16 storage with fp32 accumulator — covers matmul / einsum /
    convolution / linear_general / projection ops."""
    return NumericPolicy(storage="bf16", accum="fp32")


def _attn_policy() -> "NumericPolicy":
    """Attention: bf16 tile, fp32 accumulator, deterministic softmax."""
    return NumericPolicy(storage="bf16", accum="fp32", deterministic=True)


def _spectral_policy() -> "NumericPolicy":
    """FFT family: fp32 throughout — FFT numerics need the full mantissa."""
    return NumericPolicy(storage="fp32", accum="fp32")


def _normalize_policy() -> "NumericPolicy":
    """Layer/RMS norm: bf16 storage, fp32 stat accumulation."""
    return NumericPolicy(storage="bf16", accum="fp32")


def _stable_reduce_policy() -> "NumericPolicy":
    """Softmax / logsumexp: fp32 throughout for numerical stability."""
    return NumericPolicy(storage="fp32", accum="fp32", deterministic=True)


def _quant_int8_policy(quant_axis: int | None = None) -> "NumericPolicy":
    return NumericPolicy(
        storage="int8",
        accum="fp32",
        scale="per_channel_symmetric" if quant_axis is not None else "per_tensor_symmetric",
        quant_axis=quant_axis,
    )


def _quant_low_fp_policy(storage: str) -> "NumericPolicy":
    """fp8 / fp6 / fp4 / nvfp4 quantizers — block-scaled storage,
    fp32 accumulator for the rescale."""
    return NumericPolicy(
        storage=storage,
        accum="fp32",
        scale="blockfp_per_stage",
    )


# Map op name → NumericPolicy.  Populated lazily because NumericPolicy is
# defined further up; we resolve at registry-build time via _policy_for_name.
_NUMERIC_POLICY_BY_NAME_FACTORIES: dict[str, "Callable[[], NumericPolicy]"] = {
    # ── Matmul / contraction / convolution family ──────────────────────
    "matmul":            _matmul_policy,
    "gemm":              _matmul_policy,
    "batched_gemm":      _matmul_policy,
    "einsum":            _matmul_policy,
    "factorized_matmul": _matmul_policy,
    "linear_general":    _matmul_policy,
    "conv1d":            _matmul_policy,
    "conv2d":            _matmul_policy,
    "conv3d":            _matmul_policy,
    "conv_transpose":    _matmul_policy,
    "depthwise_conv1d":  _matmul_policy,
    "depthwise_conv2d":  _matmul_policy,
    "qkv_projection":    _matmul_policy,
    "fused_epilogue":    _matmul_policy,
    # ── Attention family ───────────────────────────────────────────────
    "flash_attn":                 _attn_policy,
    "multi_head_attention":       _attn_policy,
    "gqa_attention":              _attn_policy,
    "mqa_attention":              _attn_policy,
    "mla_decode":                 _attn_policy,
    "mla_decode_fused":           _attn_policy,
    "linear_attn":                _attn_policy,
    "lightning_attention":        _attn_policy,
    "deepseek_sparse_attention":  _attn_policy,
    "gated_attention":            _attn_policy,
    "hybrid_attention":           _attn_policy,
    "gated_deltanet":             _attn_policy,
    "kimi_delta_attention":       _attn_policy,
    "modified_delta_attention":   _attn_policy,
    "attn_sliding_window":        _attn_policy,
    "attn_local_window_2d":       _attn_policy,
    "attn_compressed_blocks":     _attn_policy,
    "attn_top_k_blocks":          _attn_policy,
    # ── Spectral family ────────────────────────────────────────────────
    "fft":              _spectral_policy,
    "ifft":             _spectral_policy,
    "rfft":             _spectral_policy,
    "irfft":            _spectral_policy,
    "stft":             _spectral_policy,
    "istft":            _spectral_policy,
    "dct":              _spectral_policy,
    "spectral_conv":    _spectral_policy,
    "spectral_filter":  _spectral_policy,
    # ── Normalization family ───────────────────────────────────────────
    "layer_norm":      _normalize_policy,
    "rmsnorm":         _normalize_policy,
    "rmsnorm_safe":    _normalize_policy,
    "group_norm":      _normalize_policy,
    "instance_norm":   _normalize_policy,
    "weight_norm":     _normalize_policy,
    "spectral_norm":   _normalize_policy,
    # ── Stable reductions ──────────────────────────────────────────────
    "softmax":               _stable_reduce_policy,
    "softmax_safe":          _stable_reduce_policy,
    "online_softmax":        _stable_reduce_policy,
    "online_softmax_state":  _stable_reduce_policy,
    "logsumexp":             _stable_reduce_policy,
    "log_softmax":           _stable_reduce_policy,
    # ── Quantization family ────────────────────────────────────────────
    "quantize_int8":     lambda: _quant_int8_policy(),
    "dequantize_int8":   lambda: _quant_int8_policy(),
    "quantize_int4":     lambda: NumericPolicy(
        storage="int8",  # int4 packed into int8 today (per Decision #23)
        accum="fp32",
        scale="per_tensor_symmetric",
    ),
    "dequantize_int4":   lambda: NumericPolicy(
        storage="int8", accum="fp32", scale="per_tensor_symmetric",
    ),
    "quantize_fp8":      lambda: _quant_low_fp_policy("fp8_e4m3"),
    "dequantize_fp8":    lambda: _quant_low_fp_policy("fp8_e4m3"),
    "quantize_fp4":      lambda: _quant_low_fp_policy("fp4_e2m1"),
    "dequantize_fp4":    lambda: _quant_low_fp_policy("fp4_e2m1"),
    "quantize_fp6":      lambda: _quant_low_fp_policy("fp6_e2m3"),
    "dequantize_fp6":    lambda: _quant_low_fp_policy("fp6_e2m3"),
    "quantize_nvfp4":    lambda: _quant_low_fp_policy("nvfp4"),
    "dequantize_nvfp4":  lambda: _quant_low_fp_policy("nvfp4"),
    "fake_quantize":     lambda: _quant_int8_policy(),
    "calibration_observer": lambda: NumericPolicy(
        storage="fp32", accum="fp32",
        scale="per_tensor_symmetric",
        deterministic=True,
    ),
    # ── Mixed-precision optimizer/scaler ───────────────────────────────
    "grad_scaler_step":  lambda: NumericPolicy(
        storage="fp32", accum="fp32",
        scale="loss_scale",
        deterministic=True,
    ),
}


def _policy_for_name(name: str) -> "NumericPolicy | None":
    factory = _NUMERIC_POLICY_BY_NAME_FACTORIES.get(name)
    return factory() if factory is not None else None


def _manifest_for_name(name: str) -> list[dict[str, object]] | None:
    """Return the backend-kernel manifest entries for ``name`` as plain
    dicts (Sprint E, 2026-05-11).

    Looks up `backend_manifest.manifest_for(name)`; returns ``None`` when
    the manifest is empty (i.e., when ``name`` is not in OP_SPECS or has
    no per-target coverage worth recording).  Imported lazily to avoid
    cycles.
    """
    try:
        from . import backend_manifest as _bm
    except Exception:
        return None
    entries = _bm.manifest_for(name)
    if not entries:
        return None
    return [e.as_dict() for e in entries]


def _supplemental_metadata(name: str, graph_ir_state: str) -> dict[str, object]:
    """Build the metadata dict for a supplemental_public_ops entry.

    Sprint C2: attach numeric_policy when this op is in
    ``_NUMERIC_POLICY_BY_NAME_FACTORIES``.  Keeps the existing fields
    intact otherwise.
    """
    md: dict[str, object] = {
        "implementation": "python_reference",
        "contract_schema": "explicit_partial",
        "graph_ir_lowering": graph_ir_state,
        "backend_kernel": "reference_only",
    }
    policy = _policy_for_name(name)
    if policy is not None:
        md["numeric_policy"] = policy.as_metadata_dict()
    return md


# Categories whose primitives are inherently non-differentiable: VJP/JVP
# should resolve to `not_applicable`, not `planned`. RNG is non-diff through
# the sample; transforms ARE the autodiff primitives (vjp/jvp themselves);
# control-flow primitives have body-dependent rules that the framework
# handles separately; integer-output / boolean-output / state-mutating
# primitives don't have a linearization at all.
_NONDIFFERENTIABLE_CATEGORIES: frozenset[str] = frozenset({
    "rng", "random_source", "random_mask",
    "transform",
    "control_flow",
    "schedule",
    "comparison",
    "logical",
    "sharding",
    "grad_transform",
    "sort",
    "state_tree",
    "data",
    "tokenizer",
    "aot",
    "serialization",
    "conformance",
    "extension",  # custom_primitive declares its own rules; the catalog entry itself has no canonical VJP/JVP
})

# Specific primitive names that are inherently non-differentiable even
# though their category (numeric_helper / indexing / reduction / etc.) is
# differentiable in general. These are integer-output, boolean-output, or
# permutation-index operators where the gradient through their primary
# output is undefined.
_NONDIFFERENTIABLE_PER_NAME: frozenset[str] = frozenset({
    # numeric_helper integer-output / boolean-output
    "floor", "ceil", "round", "trunc",
    "isnan", "isinf", "isfinite",
    # reduction integer-output (indices)
    "argmax", "argmin",
    # indexing primitives that produce or use integer indices only
    "nonzero",
    # state-effect / movement ops without a canonical VJP
    "pack", "unpack",  # explicit memory-movement intrinsics
    "rearrange",       # axis-permutation; transpose handles the AD
    "tile_view",       # in-place view, no copy
    "arange",          # constant-generating
    "masked_fill",     # already has VJP; keeping placeholder is incorrect — drop from list
})
# Drop masked_fill from the non-diff set (it has a registered VJP).
_NONDIFFERENTIABLE_PER_NAME = _NONDIFFERENTIABLE_PER_NAME - {"masked_fill"}


_TESTS_BY_CATEGORY: dict[str, str] = {
    # Categories with comprehensive test files (see tests/unit/)
    "elementwise":         "complete",   # test_s2_primitives.py + test_autodiff_*
    "scalar_math":         "complete",
    "numeric_helper":      "complete",
    "comparison":          "complete",
    "logical":             "complete",
    "reduction":           "complete",   # test_s2_primitives + test_sprint_*
    "stable_reduction":    "complete",
    "rng":                 "complete",   # test_rng_keys.py
    "random_source":       "complete",
    "random_mask":         "complete",
    "rotary_embedding":    "complete",   # test_autodiff_loss_layer_coverage + test_reasoning_model_support
    "position_encoding":   "complete",
    "loss":                "complete",   # test_autodiff_loss_layer_coverage + test_deferred_vjps
    "rl_loss":             "complete",   # test_reasoning_model_support
    "collective":          "complete",   # test_sprint_collectives_optim_memory_cumextrema
    "sharding":            "complete",
    "quantize":            "complete",   # test_optimizer_mixed_precision_support
    "quantization":        "complete",
    "numerics":            "complete",
    "functional_optimizer_step": "complete",  # test_optimizer_mixed_precision_support
    "optimizer":           "complete",
    "attention":           "complete",   # test_attention_family_support + test_autodiff_*
    "transform":           "complete",
    "extension":           "complete",
    "loop_nest":           "complete",   # test_autodiff_lowering_gap_hardening
    "memory":              "complete",   # test_sprint_collectives_optim_memory_cumextrema
    "stencil":             "complete",   # test_conv1d_autodiff
    "pooling":             "complete",   # test_autodiff_loss_layer_coverage
    "normalization":       "complete",   # test_autodiff_loss_layer_coverage
    "state_tree":          "complete",   # test_state_tree.py
    "state_update":        "complete",   # KV cache tests
    "model_layer":         "complete",   # test_autodiff_lowering_gap_hardening
    "contraction":         "complete",
    # Long-tail categories now covered by `test_primitive_coverage_smoke.py`
    "moe":                 "complete",
    "moe_transport":       "complete",
    "state_space":         "complete",
    "spectral":            "complete",
    "tensor_algebra":      "complete",
    "layout_transform":    "complete",
    "indexing":            "complete",
    "segment_reduce":      "complete",
    "linalg_solver":       "complete",
    "linalg_decomposition":"complete",
    "sparse":              "complete",
    "sort":                "complete",
    "fused_epilogue":      "complete",
    "projection":          "complete",
    # Still partial: categories where the smoke file doesn't reach yet.
    "recurrent":           "partial",
    "control_flow":        "partial",
    "grad_transform":      "partial",
    "schedule":            "partial",
    # Non-tensor categories — tests live in other suites
    "aot":                 "not_applicable",
    "serialization":       "not_applicable",
    "conformance":         "not_applicable",
    "data":                "complete",       # test_data_pipeline.py (S15 surface)
    "tokenizer":           "complete",

    # — GA + EBM (Decision #25, 2026-05-17): GA has dedicated test
    #   files (test_ga_*, test_apple_gpu_clifford_*); EBM is tested
    #   by test_ebm_*, test_benchmark_ga_ebm.py, test_jit_bridge.py
    #   + the benchmark harness itself.  Every primitive has at
    #   least one correctness test (per-shape native vs Python ref
    #   in the benchmark sweep + dedicated VJP/JVP tests for the
    #   handful with autodiff already shipped).
    "geometric_algebra":   "complete",
    "ebm":                 "complete",

    # — M7 Visual Complex Analysis (E3, 2026-05-20): focused test
    #   coverage lives in ``tests/unit/test_complex_*`` (~94 tests
    #   across complex_mul/exp/log/sqrt/conjugate/abs/arg/mobius/
    #   stereographic/cross_ratio/dz/dbar/laplacian_2d/conformal_*).
    "visual_complex":      "complete",
}


def _apply_category_overrides(
    contract: dict[str, str], category: str | None,
) -> None:
    """In-place promote each axis based on `category` and per-axis tables.

    Override rules per axis:
      sharding_rule / batching_rule / transpose_rule / lowering_rule  →
          only override if current ∈ {"planned"} (the unset default).
      math_semantics / shape_rule / dtype_layout_rule / tests  →
          also override if current ∈ {"partial"} since those start at partial.
      Never downgrade `complete` or `not_applicable`.
    """
    if category is None:
        return

    def _promote(axis: str, table: dict[str, str], overridable: frozenset[str]) -> None:
        if contract.get(axis) not in overridable:
            return
        v = table.get(category)
        if v is None:
            return
        contract[axis] = v

    # Planned-only axes (preserve any earlier partial/complete decision)
    # sharding_rule (2026-05-22): widened in lockstep with batching +
    # transpose.  The S-series sprint #3 (sharding) honors the same
    # "category is authoritative, per-name exceptions win" pattern.
    _promote("sharding_rule",  _SHARDING_RULE_BY_CATEGORY,  frozenset({"planned", "partial"}))
    # batching_rule (2026-05-22): widened to also override ``partial`` —
    # matching lowering_rule's pattern.  This is what lets the S-series
    # batching promotion sprint move a category from partial → complete
    # via a single table edit rather than per-entry surgery.  Safe
    # because the table is authoritative for category-level decisions;
    # per-name exceptions live in ``_apply_per_name_overrides``.
    _promote("batching_rule",  _BATCHING_RULE_BY_CATEGORY,  frozenset({"planned", "partial"}))
    # transpose_rule (2026-05-22): widened in lockstep — the
    # S-series sprint #2 (transpose) follows the same shape as #1.
    _promote("transpose_rule", _TRANSPOSE_RULE_BY_CATEGORY, frozenset({"planned", "partial"}))
    _promote("lowering_rule",  _LOWERING_RULE_BY_CATEGORY,  frozenset({"planned", "partial"}))
    # Partial-and-planned axes (semantic axes start at partial by default)
    semantic_overridable = frozenset({"planned", "partial"})
    _promote("math_semantics",     _SEMANTIC_RULES_BY_CATEGORY, semantic_overridable)
    _promote("shape_rule",         _SEMANTIC_RULES_BY_CATEGORY, semantic_overridable)
    _promote("dtype_layout_rule",  _SEMANTIC_RULES_BY_CATEGORY, semantic_overridable)
    _promote("tests",              _TESTS_BY_CATEGORY,           semantic_overridable)

    # Mark vjp/jvp as not_applicable for inherently non-differentiable
    # categories. Only override when the current value is `planned` so we
    # never downgrade an explicit `complete` from `_VJPS`/`_JVPS`.
    if category in _NONDIFFERENTIABLE_CATEGORIES:
        if contract.get("vjp") == "planned":
            contract["vjp"] = "not_applicable"
        if contract.get("jvp") == "planned":
            contract["jvp"] = "not_applicable"


def _apply_per_name_overrides(contract: dict[str, str], name: str) -> None:
    """Per-name overrides for specific primitives whose category is
    differentiable in general but whose individual semantics aren't.

    Integer-output (`floor`/`ceil`/`argmax`/`nonzero`/...), boolean-output
    (`isnan`/`isinf`/`isfinite`), and explicit memory-movement intrinsics
    (`pack`/`unpack`) have undefined gradients on their primary output.
    """
    if name in _NONDIFFERENTIABLE_PER_NAME:
        if contract.get("vjp") == "planned":
            contract["vjp"] = "not_applicable"
        if contract.get("jvp") == "planned":
            contract["jvp"] = "not_applicable"


def _apply_effect_overrides(
    contract: dict[str, str], effect: str,
) -> None:
    """Promote `masking_effect_rule` based on the OpSpec's declared effect.

    Any non-pure effect (`state`, `random`, `collective`, `movement`, `io`)
    has its rule explicitly declared via `OpSpec.effect`; this is the
    canonical contract for masking/effect behavior. Only override when
    the current value is `partial` (the default for non-pure ops).
    """
    if contract.get("masking_effect_rule") != "partial":
        return
    if effect != "pure":
        contract["masking_effect_rule"] = "complete"


def _existing_coverage() -> dict[str, PrimitiveCoverage]:
    registered_vjps = _vjp_registered_names()
    registered_jvps = _jvp_registered_names()
    entries: dict[str, PrimitiveCoverage] = {}
    for name, spec in sorted(OP_SPECS.items()):
        has_vjp = _existing_op_has_vjp(name, registered_vjps)
        has_jvp = _existing_op_has_jvp(name, registered_jvps)
        contract_status = _existing_contracts(
            spec.effect, vjp_complete=has_vjp, jvp_complete=has_jvp
        )
        # Apply multi-axis category classifier before per-name overrides
        # so explicit overrides always win over the category default.
        category = _EXISTING_CATEGORIES.get(name, spec.lowering)
        _apply_category_overrides(contract_status, category)
        _apply_effect_overrides(contract_status, spec.effect)
        _apply_per_name_overrides(contract_status, name)
        contract_status.update(_EXISTING_CONTRACT_OVERRIDES.get(name, {}))
        schema = ("explicit_semantic" if name in _EXPLICIT_SEMANTIC_NAMES
                  else "explicit_partial")
        # Sprint C2 (2026-05-11): attach numeric_policy when this op has an
        # intrinsic storage/accumulator coupling.
        metadata: dict[str, object] = {
            "implementation": "op_catalog",
            "contract_schema": schema,
            "graph_ir_lowering": "registered",
            "backend_kernel": "partial",
        }
        policy = _policy_for_name(name)
        if policy is not None:
            metadata["numeric_policy"] = policy.as_metadata_dict()
        # Sprint E (2026-05-11): attach the per-target backend-kernel
        # manifest synthesized from the capability registry +
        # Apple GPU kernel inventory + x86 AMX backend.
        manifest = _manifest_for_name(name)
        if manifest is not None:
            metadata["backend_kernel_manifest"] = manifest
        # Ask 4-B (2026-05-20): attach halo_aware metadata for ops that
        # the HaloMeshIntegrationPass should wrap with halo.exchange when
        # their inputs are sharded.
        if name in _HALO_AWARE_OPS:
            metadata["halo_aware"] = dict(_HALO_AWARE_OPS[name])
        entries[name] = PrimitiveCoverage(
            name=name,
            category=_EXISTING_CATEGORIES.get(name, spec.lowering),
            status="partial",
            contract_status=contract_status,
            model_families=_EXISTING_MODEL_FAMILIES.get(name, ()),
            references=("tessera",),
            notes="Imported from the supported op catalog; S1 keeps missing semantic rules visible.",
            existing_op=True,
            graph_name=spec.graph_name,
            effect=spec.effect,
            lowering=spec.lowering,
            metadata=metadata,
        )
    supplemental_public_ops = {
        "depthwise_conv1d": ("stencil", "state", "streaming depthwise convolution"),
        "online_softmax": ("stable_reduction", "state", "streaming softmax helper"),
        "online_softmax_state": ("state_update", "state", "streaming softmax carry state"),
        "selective_ssm": ("state_space", "state", "Mamba-style selective state-space op"),
    }
    for name, (lowering, effect, notes) in supplemental_public_ops.items():
        has_vjp = _existing_op_has_vjp(name, registered_vjps)
        has_jvp = _existing_op_has_jvp(name, registered_jvps)
        contract_status = _existing_contracts(
            effect, vjp_complete=has_vjp, jvp_complete=has_jvp
        )
        _apply_category_overrides(contract_status, lowering)
        _apply_effect_overrides(contract_status, effect)
        _apply_per_name_overrides(contract_status, name)
        contract_status.update(_EXISTING_CONTRACT_OVERRIDES.get(name, {}))
        # Sprint B (2026-05-11): supplemental_public_ops default to "missing"
        # (they're outside OP_SPECS), but per-name overrides flip
        # depthwise_conv1d / online_softmax / online_softmax_state to
        # "registered" since they have reference implementations + tests
        # that downstream backends can lower through.  selective_ssm stays
        # "missing" pending a dedicated Mamba2 Graph IR op.
        graph_ir_state = _GRAPH_IR_LOWERING_OVERRIDES.get(name, "missing")
        entries.setdefault(
            name,
            PrimitiveCoverage(
                name=name,
                category=lowering,
                status="partial",
                contract_status=contract_status,
                model_families=_EXISTING_MODEL_FAMILIES.get(name, ()),
                references=("tessera",),
                notes=f"Public Python op outside OP_SPECS today; tracked for standalone coverage: {notes}.",
                existing_op=True,
                graph_name=f"tessera.{name}",
                effect=effect,
                lowering=lowering,
                metadata=_supplemental_metadata(name, graph_ir_state),
            ),
        )

    # ─────────────────────────────────────────────────────────────────────
    # S-series — Python-frontend primitives that are *shipped* (have numpy
    # reference implementations + tests) but live outside `op_catalog.py`
    # because they're structural primitives, not Graph IR ops. Covered at
    # the partial level: math/shape/dtype/lowering/tests = partial; VJP/JVP
    # /batching/transpose/sharding rules remain visible as missing until the
    # owning sprint closes them.
    # ─────────────────────────────────────────────────────────────────────
    python_primitives = {
        # S2 — reduction aliases and cumulative extrema
        "max": ("reduction", "first-class max reduction alias for amax — S2 hardened 2026-05-10"),
        "min": ("reduction", "first-class min reduction alias for amin — S2 hardened 2026-05-10"),
        "cummax": ("reduction", "cumulative max reference — S2 hardened 2026-05-10"),
        "cummin": ("reduction", "cumulative min reference — S2 hardened 2026-05-10"),
        # S3 — pytree state-tree primitives (python/tessera/state/tree.py)
        "tree_flatten": ("state_tree", "tree pytree flatten — S3 landed 2026-05-10"),
        "tree_unflatten": ("state_tree", "tree pytree unflatten — S3 landed 2026-05-10"),
        "tree_map": ("state_tree", "tree pytree map — S3 landed 2026-05-10"),
        "tree_reduce": ("state_tree", "tree pytree reduce — S3 landed 2026-05-10"),
        "tree_transpose": ("state_tree", "tree pytree transpose — S3 landed 2026-05-10"),
        "empty_state_tree": ("state_tree", "empty typed state tree — S3 landed 2026-05-10"),
        "module_state_tree": ("state_tree", "nn.Module state projection — S3 landed 2026-05-10"),
        "state_filter": ("state_tree", "state-collection filter — S3 landed 2026-05-10"),
        "state_partition": ("state_tree", "disjoint state partition — S3 landed 2026-05-10"),
        "state_collection_spec": ("state_tree", "typed state collection contracts — S3 landed 2026-05-10"),
        # S4 — RNG keys + samplers (python/tessera/rng.py)
        "rng_key": ("rng", "RNGKey.from_seed — S4 landed 2026-05-10"),
        "rng_split": ("rng", "RNGKey.split — S4 landed 2026-05-10"),
        "rng_fold_in": ("rng", "RNGKey.fold_in — S4 landed 2026-05-10"),
        "rng_clone": ("rng", "RNGKey.clone — S4 landed 2026-05-10"),
        "rng_truncated_normal": ("rng", "truncated normal sampler — S4 landed 2026-05-10"),
        "rng_bernoulli": ("rng", "bernoulli sampler — S4 landed 2026-05-10"),
        "rng_categorical": ("rng", "categorical (Gumbel-max) sampler — S4 landed 2026-05-10"),
        "rng_multinomial": ("rng", "multinomial sampler — S4 landed 2026-05-10"),
        "rng_randint": ("rng", "randint sampler — S4 landed 2026-05-10"),
        "rng_permutation": ("rng", "permutation sampler — S4 landed 2026-05-10"),
        "rng_gamma": ("rng", "gamma sampler — S4 landed 2026-05-10"),
        "rng_beta": ("rng", "beta sampler — S4 landed 2026-05-10"),
        "rng_dirichlet": ("rng", "dirichlet sampler — S4 landed 2026-05-10"),
        "rng_poisson": ("rng", "poisson sampler — S4 landed 2026-05-10"),
        # S5 — control-flow + transforms (python/tessera/control.py and autodiff/*)
        "scan": ("control_flow", "sequential scan — S5 landed 2026-05-10"),
        "associative_scan": ("control_flow", "associative prefix scan — S5 landed 2026-05-10"),
        "while_loop": ("control_flow", "structured while loop — S5 landed 2026-05-10"),
        "fori_loop": ("control_flow", "structured counted loop — S5 landed 2026-05-10"),
        "cond": ("control_flow", "structured conditional — S5 landed 2026-05-10"),
        "switch": ("control_flow", "indexed branch switch — S5 landed 2026-05-10"),
        "map": ("control_flow", "axis-aware sequential map — S5 landed 2026-05-10"),
        "value_and_grad": ("transform", "value and reverse-mode gradient — S5 landed 2026-05-10"),
        "vjp": ("transform", "pullback transform — S5 landed 2026-05-10"),
        "jvp": ("transform", "forward-mode JVP transform — S5 landed 2026-05-10"),
        "vmap": ("transform", "batched map transform — S5 landed 2026-05-10"),
        "pmap": ("transform", "axis-aware SPMD map — S5 landed 2026-05-10"),
        "remat": ("transform", "rematerialization transform — S5 landed 2026-05-10"),
        "checkpoint": ("transform", "checkpoint alias for rematerialization — S5 landed 2026-05-10"),
        "autocast": ("transform", "mixed-precision transform — S5 landed 2026-05-10"),
        "axis_index": ("transform", "mapped-axis index helper — S5 landed 2026-05-10"),
        "axis_size": ("transform", "mapped-axis size helper — S5 landed 2026-05-10"),
        "axis_name": ("transform", "mapped-axis name helper — S5 landed 2026-05-10"),
        # S6 — sharding + collectives (python/tessera/sharding.py)
        "shard_map": ("sharding", "CPU-reference shard_map — S6 landed 2026-05-10"),
        "named_sharding": ("sharding", "named sharding constructor — S6 landed 2026-05-10"),
        "partition_spec": ("sharding", "partition spec constructor — S6 landed 2026-05-10"),
        "psum": ("collective", "parallel sum collective — S6 landed 2026-05-10"),
        "pmean": ("collective", "parallel mean collective — S6 landed 2026-05-10"),
        "pmax": ("collective", "parallel max collective — S6 landed 2026-05-10"),
        "pmin": ("collective", "parallel min collective — S6 landed 2026-05-10"),
        "collective_permute": ("collective", "collective permute primitive — S6 landed 2026-05-10"),
        "broadcast_to_axis": ("collective", "broadcast to mapped axis — S6 landed 2026-05-10"),
        # S7 — model-layer reference surface (python/tessera/nn/functional.py + layers.py)
        "conv1d": ("model_layer", "NCL grouped Conv1d reference — S7 landed 2026-05-10"),
        "conv_transpose": ("model_layer", "NCL grouped ConvTranspose1d reference — S7 landed 2026-05-10"),
        "linear_general": ("model_layer", "axis-flexible LinearGeneral reference — S7 landed 2026-05-10"),
        "einsum": ("model_layer", "Einsum layer helper — S7 landed 2026-05-10"),
        "lora_linear": ("model_layer", "LoRA linear adapter — S7 landed 2026-05-10"),
        "group_norm": ("normalization", "GroupNorm reference — S7 landed 2026-05-10"),
        "instance_norm": ("normalization", "InstanceNorm reference — S7 landed 2026-05-10"),
        "weight_norm": ("normalization", "WeightNorm reference — S7 landed 2026-05-10"),
        "spectral_norm": ("normalization", "SpectralNorm reference — S7 landed 2026-05-10"),
        "max_pool": ("pooling", "max pool reference — S7 landed 2026-05-10"),
        "avg_pool": ("pooling", "average pool reference — S7 landed 2026-05-10"),
        "min_pool": ("pooling", "min pool reference — S7 landed 2026-05-10"),
        "adaptive_pool": ("pooling", "adaptive 2D pool reference — S7 landed 2026-05-10"),
        "gru_cell": ("recurrent", "GRU cell reference — S7 landed 2026-05-10"),
        "simple_rnn_cell": ("recurrent", "simple RNN cell reference — S7 landed 2026-05-10"),
        "bidirectional_scan": ("recurrent", "bidirectional scan helper — S7 landed 2026-05-10"),
        "alibi": ("position_encoding", "ALiBi bias helper — S7 landed 2026-05-10"),
        "ntk_rope": ("position_encoding", "NTK-scaled RoPE helper — S7 landed 2026-05-10"),
        "multi_head_attention": ("attention", "multi-head attention wrapper — S7 landed 2026-05-10"),
        "gqa_attention": ("attention", "grouped-query attention wrapper — S7 landed 2026-05-10"),
        "mqa_attention": ("attention", "multi-query attention wrapper — S7 landed 2026-05-10"),
        "mla_decode": ("attention", "latent KV decode attention wrapper — S7 landed 2026-05-10"),
        "gated_attention": ("attention", "gated softmax attention wrapper — attention-family batch 2026-05-10"),
        "hybrid_attention": ("attention", "named Ling/Kimi hybrid attention policy wrapper — attention-family batch 2026-05-10"),
        "deepseek_sparse_attention": ("attention", "DeepSeek/NSA three-branch sparse attention wrapper — attention-family batch 2026-05-10"),
        "lightning_attention": ("attention", "Lightning linear attention wrapper — attention-family batch 2026-05-10"),
        "gated_deltanet": ("attention", "Gated DeltaNet recurrence — attention-family batch 2026-05-10"),
        "kimi_delta_attention": ("attention", "Kimi Delta Attention recurrence — attention-family batch 2026-05-10"),
        "modified_delta_attention": ("attention", "modified bounded Delta Attention recurrence — attention-family batch 2026-05-10"),
        "memory_read": ("memory", "top-k weighted memory read — S7 memory hardened 2026-05-10"),
        "memory_write": ("memory", "functional memory append/update surface — S7 memory hardened 2026-05-10"),
        "memory_evict": ("memory", "functional memory eviction surface — S7 memory hardened 2026-05-10"),
        # S8 — tiny standalone conformance targets (tests/unit/test_s7_s8_s9.py)
        "tiny_diffusion_conformance": ("conformance", "diffusion-like forward/RNG/state smoke — S8 landed 2026-05-10"),
        "tiny_recurrent_conformance": ("conformance", "scan/RNN gradient smoke — S8 landed 2026-05-10"),
        "tiny_attention_conformance": ("conformance", "efficient-attention style smoke — S8 landed 2026-05-10"),
        # S9 — quantization + mixed precision references (python/tessera/quantization.py)
        "quantize_int8": ("quantization", "int8 reference quantizer — S9 landed 2026-05-10"),
        "dequantize_int8": ("quantization", "int8 reference dequantizer — S9 landed 2026-05-10"),
        "quantize_int4": ("quantization", "int4-in-int8 reference quantizer — S9 landed 2026-05-10"),
        "dequantize_int4": ("quantization", "int4-in-int8 reference dequantizer — S9 landed 2026-05-10"),
        "fake_quantize": ("quantization", "QAT fake quantization reference — S9 landed 2026-05-10"),
        "calibration_observer": ("quantization", "min/max calibration observer — S9 landed 2026-05-10"),
        "grad_scaler_step": ("numerics", "loss-scale update helper — S9 landed 2026-05-10"),
        # S10 — optimizers, schedules, and gradient transforms (python/tessera/optim.py)
        "sgd": ("optimizer", "functional SGD — S10 landed 2026-05-10"),
        "momentum": ("optimizer", "functional momentum SGD — S10 landed 2026-05-10"),
        "nesterov": ("optimizer", "functional Nesterov momentum — S10 landed 2026-05-10"),
        "adam": ("optimizer", "functional Adam — S10 refined 2026-05-10"),
        "adamw": ("optimizer", "functional AdamW — S10 landed 2026-05-10"),
        "adafactor": ("optimizer", "functional Adafactor — S10 landed 2026-05-10"),
        "lion": ("optimizer", "functional Lion — S10 landed 2026-05-10"),
        "muon": ("optimizer", "functional Muon-style orthogonalized update — S10 landed 2026-05-10"),
        "lamb": ("optimizer", "functional LAMB — S10 landed 2026-05-10"),
        "constant_lr": ("schedule", "constant learning-rate schedule — S10 landed 2026-05-10"),
        "cosine_lr": ("schedule", "cosine decay schedule — S10 landed 2026-05-10"),
        "cosine_warmup_lr": ("schedule", "warmup plus cosine decay schedule — S10 landed 2026-05-10"),
        "linear_warmup_lr": ("schedule", "linear warmup schedule — S10 landed 2026-05-10"),
        "polynomial_lr": ("schedule", "polynomial decay schedule — S10 landed 2026-05-10"),
        "inverse_sqrt_lr": ("schedule", "inverse-square-root schedule — S10 landed 2026-05-10"),
        "cyclical_lr": ("schedule", "cyclical learning-rate schedule — S10 refined 2026-05-10"),
        "chained_schedule": ("schedule", "composed schedule helper — S10 refined 2026-05-10"),
        "clip_grad_norm": ("grad_transform", "functional gradient norm clipping — S10 landed 2026-05-10"),
        "clip_grad_value": ("grad_transform", "functional gradient value clipping — S10 landed 2026-05-10"),
        "centralize_grad": ("grad_transform", "gradient centralization — S10 landed 2026-05-10"),
        "add_decoupled_weight_decay": ("grad_transform", "decoupled weight-decay transform — S10 landed 2026-05-10"),
        "ema_update": ("grad_transform", "EMA parameter update — S10 landed 2026-05-10"),
        "polyak_avg": ("grad_transform", "Polyak average update — S10 landed 2026-05-10"),
        "optax_style_chain": ("grad_transform", "small Optax-style transform chain — S10 landed 2026-05-10"),
        # S11 — losses (python/tessera/losses.py)
        "mse_loss": ("loss", "mean-squared error loss — S11 landed 2026-05-10"),
        "mae_loss": ("loss", "mean-absolute error loss — S11 landed 2026-05-10"),
        "huber_loss": ("loss", "Huber loss — S11 landed 2026-05-10"),
        "smooth_l1_loss": ("loss", "SmoothL1 loss — S11 landed 2026-05-10"),
        "log_cosh_loss": ("loss", "log-cosh regression loss — S11 landed 2026-05-10"),
        "cross_entropy_loss": ("loss", "stable cross entropy loss — S11 landed 2026-05-10"),
        "binary_cross_entropy_loss": ("loss", "stable binary cross entropy with logits — S11 landed 2026-05-10"),
        "focal_loss": ("loss", "focal classification loss — S11 landed 2026-05-10"),
        "label_smoothed_cross_entropy": ("loss", "label-smoothed cross entropy — S11 landed 2026-05-10"),
        "kl_divergence": ("loss", "categorical KL divergence — S11 landed 2026-05-10"),
        "js_divergence": ("loss", "Jensen-Shannon divergence — S11 landed 2026-05-10"),
        "wasserstein_distance": ("loss", "1D empirical Wasserstein distance — S11 landed 2026-05-10"),
        "nt_xent_loss": ("loss", "NT-Xent contrastive loss — S11 landed 2026-05-10"),
        "info_nce_loss": ("loss", "InfoNCE loss — S11 landed 2026-05-10"),
        "triplet_loss": ("loss", "triplet margin loss — S11 landed 2026-05-10"),
        "contrastive_loss": ("loss", "pairwise contrastive loss — S11 landed 2026-05-10"),
        "cosine_embedding_loss": ("loss", "cosine embedding loss — S11 landed 2026-05-10"),
        "ddpm_noise_pred_loss": ("loss", "DDPM noise-prediction loss — S11 landed 2026-05-10"),
        "vlb_loss": ("loss", "diffusion VLB term reducer — S11 landed 2026-05-10"),
        "score_matching_loss": ("loss", "score-matching loss — S11 landed 2026-05-10"),
        "ctc_loss": ("loss", "small CPU-reference CTC loss — S11 landed 2026-05-10"),
        "seq2seq_loss": ("loss", "masked seq2seq cross-entropy loss — S11 landed 2026-05-10"),
        # S12 — serialization + checkpointing (python/tessera/checkpoint.py)
        "save_state": ("serialization", "versioned state-tree binary save — S12 landed 2026-05-10"),
        "load_state": ("serialization", "versioned state-tree binary load — S12 landed 2026-05-10"),
        "save_sharded": ("serialization", "mock sharded state save — S12 landed 2026-05-10"),
        "load_sharded": ("serialization", "mock sharded state load — S12 landed 2026-05-10"),
        "state_migration": ("serialization", "registered checkpoint migration rule — S12 landed 2026-05-10"),
        "partial_state_load": ("serialization", "top-level state collection filtering — S12 landed 2026-05-10"),
        # S13 — custom primitive / extension API (python/tessera/custom.py)
        "custom_primitive": ("extension", "custom primitive decorator — S13 landed 2026-05-10"),
        "custom_call": ("extension", "opaque custom call decorator — S13 landed 2026-05-10"),
        "custom_vjp": ("extension", "custom VJP registration — S13 landed 2026-05-10"),
        "custom_jvp": ("extension", "custom JVP registration — S13 landed 2026-05-10"),
        "custom_batching": ("extension", "custom batching registration — S13 landed 2026-05-10"),
        "custom_lowering": ("extension", "per-target custom lowering registration — S13 landed 2026-05-10"),
        # S14 — compilation cache + AOT export (python/tessera/aot.py)
        "aot_export": ("aot", "reference AOT export — S14 landed 2026-05-10"),
        "aot_load": ("aot", "reference AOT load — S14 landed 2026-05-10"),
        "stablehlo_export": ("aot", "StableHLO reference text export — S14 landed 2026-05-10"),
        "gguf_export": ("aot", "GGUF reference metadata export — S14 landed 2026-05-10"),
        "safetensors_export": ("aot", "safetensors-like npz export — S14 landed 2026-05-10"),
        "compilation_cache": ("aot", "persistent AOT artifact cache — S14 landed 2026-05-10"),
        # S15 — data pipeline + tokenizers (python/tessera/data.py)
        "dataset_map": ("data", "Dataset.map — S15 landed 2026-05-10"),
        "dataset_filter": ("data", "Dataset.filter — S15 landed 2026-05-10"),
        "dataset_batch": ("data", "Dataset.batch — S15 landed 2026-05-10"),
        "dataset_prefetch": ("data", "Dataset.prefetch reference no-op — S15 landed 2026-05-10"),
        "dataset_shuffle": ("data", "RNGKey-backed deterministic shuffle — S15 landed 2026-05-10"),
        "dataset_interleave": ("data", "Dataset.interleave — S15 landed 2026-05-10"),
        "dataset_repeat": ("data", "Dataset.repeat — S15 landed 2026-05-10"),
        "dataset_zip": ("data", "Dataset.zip — S15 landed 2026-05-10"),
        "sharded_dataset": ("data", "mesh-axis sharded dataset — S15 landed 2026-05-10"),
        "iterable_dataset": ("data", "checkpointable iterable dataset — S15 landed 2026-05-10"),
        "dataset_checkpoint": ("data", "dataset checkpoint/restore metadata — S15 landed 2026-05-10"),
        "tokenizer_byte": ("tokenizer", "UTF-8 byte tokenizer — S15 landed 2026-05-10"),
        "tokenizer_bpe": ("tokenizer", "vocab-backed BPE-compatible tokenizer — S15 landed 2026-05-10"),
        "tokenizer_wordpiece": ("tokenizer", "vocab-backed WordPiece-compatible tokenizer — S15 landed 2026-05-10"),
        "tokenizer_unigram": ("tokenizer", "vocab-backed unigram-compatible tokenizer — S15 landed 2026-05-10"),
        "tokenizer_sentencepiece_compat": ("tokenizer", "SentencePiece-compatible vocab tokenizer — S15 landed 2026-05-10"),
        # S8 — expanded conformance target once S10-S15 exist.
        "tiny_training_step_conformance": ("conformance", "data/loss/optimizer/checkpoint training-step smoke — S8 expanded 2026-05-10"),
    }
    nondifferentiable_categories = {
        "aot",
        "conformance",
        "data",
        "extension",
        "serialization",
        "state_tree",
        "tokenizer",
    }
    contract_overrides: dict[str, dict[str, str]] = {
        "max": {"math_semantics": "complete", "shape_rule": "complete", "dtype_layout_rule": "complete", "vjp": "complete"},
        "min": {"math_semantics": "complete", "shape_rule": "complete", "dtype_layout_rule": "complete", "vjp": "complete"},
        "cummax": {"math_semantics": "complete", "shape_rule": "complete", "dtype_layout_rule": "complete"},
        "cummin": {"math_semantics": "complete", "shape_rule": "complete", "dtype_layout_rule": "complete"},
        "conv1d": {"math_semantics": "complete", "shape_rule": "complete", "dtype_layout_rule": "complete"},
        "linear_general": {"math_semantics": "complete", "shape_rule": "complete", "dtype_layout_rule": "complete"},
        "sgd": {"math_semantics": "complete", "shape_rule": "complete", "dtype_layout_rule": "complete"},
        "adam": {"math_semantics": "complete", "shape_rule": "complete", "dtype_layout_rule": "complete"},
        "adamw": {"math_semantics": "complete", "shape_rule": "complete", "dtype_layout_rule": "complete"},
        "momentum": {"math_semantics": "complete", "shape_rule": "complete", "dtype_layout_rule": "complete"},
        "adafactor": {"math_semantics": "complete", "shape_rule": "complete", "dtype_layout_rule": "complete"},
        "lion": {"math_semantics": "complete", "shape_rule": "complete", "dtype_layout_rule": "complete"},
        "gated_attention": {"math_semantics": "complete", "shape_rule": "complete", "dtype_layout_rule": "complete"},
        "hybrid_attention": {"math_semantics": "complete", "shape_rule": "complete", "dtype_layout_rule": "complete"},
        "deepseek_sparse_attention": {"math_semantics": "complete", "shape_rule": "complete", "dtype_layout_rule": "complete"},
        "lightning_attention": {"math_semantics": "complete", "shape_rule": "complete", "dtype_layout_rule": "complete"},
        "gated_deltanet": {"math_semantics": "complete", "shape_rule": "complete", "dtype_layout_rule": "complete"},
        "kimi_delta_attention": {"math_semantics": "complete", "shape_rule": "complete", "dtype_layout_rule": "complete"},
        "modified_delta_attention": {"math_semantics": "complete", "shape_rule": "complete", "dtype_layout_rule": "complete"},
        "mse_loss": {"math_semantics": "complete", "shape_rule": "complete", "dtype_layout_rule": "complete"},
        "mae_loss": {"math_semantics": "complete", "shape_rule": "complete", "dtype_layout_rule": "complete"},
        "huber_loss": {"math_semantics": "complete", "shape_rule": "complete", "dtype_layout_rule": "complete"},
        "smooth_l1_loss": {"math_semantics": "complete", "shape_rule": "complete", "dtype_layout_rule": "complete"},
        "log_cosh_loss": {"math_semantics": "complete", "shape_rule": "complete", "dtype_layout_rule": "complete"},
        "cross_entropy_loss": {"math_semantics": "complete", "shape_rule": "complete", "dtype_layout_rule": "complete"},
        "binary_cross_entropy_loss": {"math_semantics": "complete", "shape_rule": "complete", "dtype_layout_rule": "complete"},
        "ddpm_noise_pred_loss": {"math_semantics": "complete", "shape_rule": "complete", "dtype_layout_rule": "complete"},
        "score_matching_loss": {"math_semantics": "complete", "shape_rule": "complete", "dtype_layout_rule": "complete"},
        "vlb_loss": {"math_semantics": "complete", "shape_rule": "complete", "dtype_layout_rule": "complete"},
        # Sprint D (2026-05-11): memory primitives now have:
        #   - vmap_axis_map for shared-state batching semantics
        #     (tessera.memory.vmap_axis_for) → batching_rule complete
        #   - MemoryShardSpec for content-addressed sharding
        #     (tessera.sharding.MemoryShardSpec, KEY_HASH/BUCKET modes) +
        #     MemoryStateHandle persistent ABI → sharding_rule complete
        #   - transpose_rule promoted to complete (top-k indices treated as
        #     constants matches the shipped VJP convention)
        "memory_read": {
            "math_semantics": "complete",
            "shape_rule": "complete",
            "dtype_layout_rule": "complete",
            "batching_rule": "complete",
            "transpose_rule": "complete",
            "sharding_rule": "complete",
        },
        "memory_write": {
            "math_semantics": "complete",
            "shape_rule": "complete",
            "dtype_layout_rule": "complete",
            "vjp": "not_applicable",
            "jvp": "not_applicable",
            "batching_rule": "complete",
            "transpose_rule": "not_applicable",
            "sharding_rule": "complete",
        },
        "memory_evict": {
            "math_semantics": "complete",
            "shape_rule": "complete",
            "dtype_layout_rule": "complete",
            "vjp": "not_applicable",
            "jvp": "not_applicable",
            "batching_rule": "complete",
            "transpose_rule": "not_applicable",
            "sharding_rule": "complete",
        },
    }
    for name, (category, notes) in python_primitives.items():
        # Python primitives default to partial coverage. We then promote
        # individual axes to `complete` based on what's actually registered:
        #   - vjp/jvp: consult `_VJPS`/`_JVPS` — same hook as `OP_SPECS` ops.
        #   - other axes via `contract_overrides[name]` below.
        contract = _contracts(
            math_semantics="partial",
            shape_rule="partial",
            dtype_layout_rule="partial",
            lowering_rule="partial",  # Python-frontend only — no Graph IR yet.
            tests="complete",
            masking_effect_rule="not_applicable",
        )
        if category in nondifferentiable_categories:
            contract.update(vjp="not_applicable", jvp="not_applicable", transpose_rule="not_applicable")
        else:
            if _existing_op_has_vjp(name, registered_vjps):
                contract["vjp"] = "complete"
            if _existing_op_has_jvp(name, registered_jvps):
                contract["jvp"] = "complete"
        if category in {"data", "tokenizer"}:
            contract.update(
                math_semantics="complete",
                shape_rule="complete",
                dtype_layout_rule="complete",
                batching_rule="partial",
                sharding_rule="partial",
            )
        # Apply the multi-axis category classifier (sharding_rule / batching /
        # transpose / math / shape / dtype / lowering / tests); per-name
        # overrides in `contract_overrides` (next line) still win.
        _apply_category_overrides(contract, category)
        _apply_per_name_overrides(contract, name)
        contract.update(contract_overrides.get(name, {}))
        contract.update(_EXISTING_CONTRACT_OVERRIDES.get(name, {}))
        metadata = {
            "implementation": "python_reference",
            "contract_schema": "explicit_partial",
            "graph_ir_lowering": "stub_required",
            "backend_kernel": "reference_only",
        }
        # Sprint B (2026-05-11): category-based graph_ir_lowering classifier.
        # Replaces the old hard-coded {data, tokenizer, serialization, aot,
        # conformance} → not_applicable rule with a comprehensive table
        # covering all S2-S15 python-primitive categories.
        metadata["graph_ir_lowering"] = _graph_ir_lowering_for_category(
            category, str(metadata["graph_ir_lowering"])
        )
        # Per-name override wins over category default.
        if name in _GRAPH_IR_LOWERING_OVERRIDES:
            metadata["graph_ir_lowering"] = _GRAPH_IR_LOWERING_OVERRIDES[name]
        if category == "conformance":
            metadata["model_manifest"] = "examples.conformance.s8_tiny_models.manifest"
        if all(
            contract[field] == "complete"
            for field in ("math_semantics", "shape_rule", "dtype_layout_rule")
        ):
            metadata["contract_schema"] = "explicit_semantic"
        # Sprint C2 (2026-05-11): attach numeric_policy for python_primitives
        # that have intrinsic storage/accum coupling (e.g., logsumexp,
        # log_softmax, the quantization helpers under tessera.quantization).
        _python_policy = _policy_for_name(name)
        if _python_policy is not None:
            metadata["numeric_policy"] = _python_policy.as_metadata_dict()
        python_entry = PrimitiveCoverage(
            name=name,
            category=category,
            status="partial",
            contract_status=contract,
            model_families=_EXISTING_MODEL_FAMILIES.get(name, ("all",)),
            references=("tessera",),
            notes=notes,
            existing_op=True,
            graph_name=None,
            effect="pure",
            lowering=category,
            metadata=metadata,
        )
        existing = entries.get(name)
        if existing is not None:
            merged_contract = _merge_contract_status(existing.contract_status, python_entry.contract_status)
            entries[name] = PrimitiveCoverage(
                name=existing.name,
                category=existing.category,
                status=existing.status,
                contract_status=merged_contract,
                model_families=existing.model_families or python_entry.model_families,
                references=tuple(dict.fromkeys(existing.references + python_entry.references)),
                notes=f"{existing.notes} Python reference/tests shipped: {notes}",
                existing_op=True,
                graph_name=existing.graph_name,
                effect=existing.effect,
                lowering=existing.lowering,
                metadata={
                    **existing.metadata,
                    "implementation": "op_catalog+python_reference",
                    "contract_schema": (
                        "explicit_semantic"
                        if all(
                            merged_contract[field] == "complete"
                            for field in ("math_semantics", "shape_rule", "dtype_layout_rule")
                        )
                        else existing.metadata.get("contract_schema", "explicit_partial")
                    ),
                },
            )
        else:
            entries[name] = python_entry
    return entries


def _planned(
    name: str,
    category: str,
    families: Iterable[str],
    *,
    references: Iterable[str] = ("jax.lax", "jax.numpy", "flax.nnx"),
    notes: str = "",
) -> PrimitiveCoverage:
    return PrimitiveCoverage(
        name=name,
        category=category,
        status="planned",
        contract_status=_contracts(),
        model_families=tuple(families),
        references=tuple(references),
        notes=notes,
    )


def _partial(
    name: str,
    category: str,
    families: Iterable[str],
    *,
    references: Iterable[str] = ("jax.lax", "jax.numpy", "flax.nnx"),
    notes: str = "",
) -> PrimitiveCoverage:
    """Same shape as :func:`_planned`, but ``status="partial"`` —
    i.e., a Python reference + a fused native kernel both exist for
    this primitive (per Decision #25, contract-axis completeness is
    still the next gate)."""
    return PrimitiveCoverage(
        name=name,
        category=category,
        status="partial",
        contract_status=_contracts(),
        model_families=tuple(families),
        references=tuple(references),
        notes=notes,
    )


_PLANNED_ENTRIES: tuple[PrimitiveCoverage, ...] = (
    # ── S2: tensor algebra ───────────────────────────────────────────────
    _planned("reshape", "tensor_algebra", ("all",), references=("jax.numpy", "aten")),
    _planned("view", "tensor_algebra", ("all",), references=("aten",)),
    _planned("flatten", "tensor_algebra", ("all",), references=("jax.numpy", "flax.nnx")),
    _planned("squeeze", "tensor_algebra", ("all",), references=("jax.numpy", "aten")),
    _planned("unsqueeze", "tensor_algebra", ("all",), references=("aten",)),
    _planned("permute", "tensor_algebra", ("all",), references=("jax.numpy", "aten")),
    _planned("broadcast", "tensor_algebra", ("all",), references=("jax.lax", "jax.numpy")),
    _planned("expand", "tensor_algebra", ("all",), references=("aten",)),
    _planned("cat", "tensor_algebra", ("all",), references=("jax.numpy", "aten")),
    _planned("stack", "tensor_algebra", ("all",), references=("jax.numpy", "aten")),
    _planned("split", "tensor_algebra", ("all",), references=("jax.numpy", "aten")),
    _planned("chunk", "tensor_algebra", ("all",), references=("aten",)),
    _planned("slice", "tensor_algebra", ("all",), references=("jax.lax", "aten")),
    _planned("select", "tensor_algebra", ("all",), references=("aten",)),
    _planned("pad", "tensor_algebra", ("all", "diffusion"), references=("jax.lax", "aten")),
    _planned("tile", "tensor_algebra", ("all",), references=("jax.numpy", "aten")),
    _planned("repeat", "tensor_algebra", ("all",), references=("aten",)),
    _planned("roll", "tensor_algebra", ("all", "diffusion"), references=("jax.numpy", "aten")),
    _planned("flip", "tensor_algebra", ("all", "diffusion"), references=("jax.numpy", "aten")),
    _planned("dynamic_slice", "tensor_algebra", ("all", "RNN/xLSTM", "Mamba/SSM"), references=("jax.lax",)),
    _planned("dynamic_update_slice", "tensor_algebra", ("all", "Titans/Atlas"), references=("jax.lax",)),
    # ── S2: indexing ─────────────────────────────────────────────────────
    _planned("scatter", "indexing", ("all", "Titans/Atlas"), references=("jax.lax",)),
    _planned("scatter_add", "indexing", ("all", "Titans/Atlas", "JEPA"), references=("jax.lax", "aten")),
    _planned("scatter_reduce", "indexing", ("all", "Titans/Atlas"), references=("jax.lax", "aten")),
    _planned("take", "indexing", ("all",), references=("jax.numpy", "aten")),
    _planned("index_select", "indexing", ("all",), references=("aten",)),
    _planned("nonzero", "indexing", ("all", "Titans/Atlas"), references=("jax.numpy", "aten")),
    _planned("top_k", "indexing", ("Titans/Atlas", "Megalodon/Griffin"), references=("jax.lax", "aten")),
    _planned("sort", "indexing", ("all", "Titans/Atlas"), references=("jax.lax", "aten")),
    _planned("argsort", "indexing", ("all",), references=("jax.numpy", "aten")),
    _planned("index_update", "indexing", ("all", "Titans/Atlas"), references=("jax.numpy", "aten")),
    # ── S2: reductions ──────────────────────────────────────────────────
    _planned("mean", "reduction", ("all",), references=("jax.numpy", "aten")),
    _planned("prod", "reduction", ("all",), references=("jax.numpy", "aten")),
    _planned("max", "reduction", ("all",), references=("jax.numpy", "aten")),
    _planned("min", "reduction", ("all",), references=("jax.numpy", "aten")),
    _planned("var", "reduction", ("all",), references=("jax.numpy", "aten")),
    _planned("std", "reduction", ("all",), references=("jax.numpy", "aten")),
    _planned("argmax", "reduction", ("all",), references=("jax.numpy", "aten")),
    _planned("argmin", "reduction", ("all",), references=("jax.numpy", "aten")),
    _planned("cumsum", "reduction", ("all", "Mamba/SSM"), references=("jax.numpy", "aten")),
    _planned("cumprod", "reduction", ("all",), references=("jax.numpy", "aten")),
    _planned("cummax", "reduction", ("all",), references=("aten",)),
    _planned("cummin", "reduction", ("all",), references=("aten",)),
    # ── S2: numerical-stability primitives ──────────────────────────────
    _planned("logsumexp", "stable_reduction", ("all", "diffusion", "JEPA"), references=("jax.scipy", "aten")),
    _planned("log_softmax", "stable_reduction", ("all",), references=("jax.nn", "aten")),
    _planned("log1p", "scalar_math", ("all",), references=("jax.numpy", "aten")),
    _planned("expm1", "scalar_math", ("all",), references=("jax.numpy", "aten")),
    _planned("softplus", "scalar_math", ("all", "diffusion"), references=("jax.nn", "aten")),
    _planned("sigmoid_safe", "stable_reduction", ("all",), references=("jax.nn",)),
    # ── S2: scalar math breadth ─────────────────────────────────────────
    _planned("sub", "scalar_math", ("all",), references=("jax.numpy", "aten")),
    _planned("div", "scalar_math", ("all",), references=("jax.numpy", "aten")),
    _planned("floor_div", "scalar_math", ("all",), references=("jax.numpy", "aten")),
    _planned("mod", "scalar_math", ("all",), references=("jax.numpy", "aten")),
    _planned("exp", "scalar_math", ("all",), references=("jax.numpy", "aten")),
    _planned("log", "scalar_math", ("all",), references=("jax.numpy", "aten")),
    _planned("sqrt", "scalar_math", ("all",), references=("jax.numpy", "aten")),
    _planned("rsqrt", "scalar_math", ("all",), references=("jax.lax", "aten")),
    _planned("pow", "scalar_math", ("all",), references=("jax.numpy", "aten")),
    _planned("cos", "scalar_math", ("FNet/spectral", "diffusion"), references=("jax.numpy", "aten")),
    _planned("tan", "scalar_math", ("all",), references=("jax.numpy", "aten")),
    _planned("sinh", "scalar_math", ("all",), references=("jax.numpy", "aten")),
    _planned("cosh", "scalar_math", ("all",), references=("jax.numpy", "aten")),
    _planned("asin", "scalar_math", ("all",), references=("jax.numpy", "aten")),
    _planned("acos", "scalar_math", ("all",), references=("jax.numpy", "aten")),
    _planned("atan", "scalar_math", ("all",), references=("jax.numpy", "aten")),
    _planned("atan2", "scalar_math", ("all", "diffusion"), references=("jax.numpy", "aten")),
    _planned("erf", "scalar_math", ("all", "diffusion"), references=("jax.lax", "aten")),
    _planned("erfc", "scalar_math", ("all",), references=("jax.lax", "aten")),
    _planned("lgamma", "scalar_math", ("diffusion",), references=("jax.lax", "aten")),
    _planned("digamma", "scalar_math", ("diffusion",), references=("jax.lax", "aten")),
    # ── S2: comparisons + logical ───────────────────────────────────────
    _planned("eq", "comparison", ("all",), references=("jax.numpy", "aten")),
    _planned("ne", "comparison", ("all",), references=("jax.numpy", "aten")),
    _planned("lt", "comparison", ("all",), references=("jax.numpy", "aten")),
    _planned("le", "comparison", ("all",), references=("jax.numpy", "aten")),
    _planned("gt", "comparison", ("all",), references=("jax.numpy", "aten")),
    _planned("ge", "comparison", ("all",), references=("jax.numpy", "aten")),
    _planned("logical_and", "logical", ("all",), references=("jax.numpy", "aten")),
    _planned("logical_or", "logical", ("all",), references=("jax.numpy", "aten")),
    _planned("logical_not", "logical", ("all",), references=("jax.numpy", "aten")),
    _planned("logical_xor", "logical", ("all",), references=("jax.numpy", "aten")),
    _planned("bitwise_and", "logical", ("all",), references=("jax.numpy", "aten")),
    _planned("bitwise_or", "logical", ("all",), references=("jax.numpy", "aten")),
    _planned("bitwise_xor", "logical", ("all",), references=("jax.numpy", "aten")),
    _planned("bitwise_not", "logical", ("all",), references=("jax.numpy", "aten")),
    # ── S2: numeric helpers ─────────────────────────────────────────────
    _planned("clamp", "numeric_helper", ("all",), references=("jax.numpy", "aten")),
    _planned("minimum", "numeric_helper", ("all",), references=("jax.numpy", "aten")),
    _planned("maximum", "numeric_helper", ("all",), references=("jax.numpy", "aten")),
    _planned("sign", "numeric_helper", ("all",), references=("jax.numpy", "aten")),
    _planned("abs", "numeric_helper", ("all",), references=("jax.numpy", "aten")),
    _planned("reciprocal", "numeric_helper", ("all",), references=("jax.numpy", "aten")),
    _planned("floor", "numeric_helper", ("all",), references=("jax.numpy", "aten")),
    _planned("ceil", "numeric_helper", ("all",), references=("jax.numpy", "aten")),
    _planned("round", "numeric_helper", ("all",), references=("jax.numpy", "aten")),
    _planned("trunc", "numeric_helper", ("all",), references=("jax.numpy", "aten")),
    _planned("where", "numeric_helper", ("all",), references=("jax.numpy", "aten")),
    _planned("isnan", "numeric_helper", ("all",), references=("jax.numpy", "aten")),
    _planned("isinf", "numeric_helper", ("all",), references=("jax.numpy", "aten")),
    _planned("isfinite", "numeric_helper", ("all",), references=("jax.numpy", "aten")),
    # ── S5: control flow + transforms ───────────────────────────────────
    _planned("scan", "control_flow", ("RNN/xLSTM", "Mamba/SSM", "Megalodon/Griffin"), references=("jax.lax",)),
    _planned("associative_scan", "control_flow", ("Mamba/SSM", "Hyena/FNet/spectral"), references=("jax.lax",)),
    _planned("while_loop", "control_flow", ("RNN/xLSTM", "Titans/Atlas"), references=("jax.lax",)),
    _planned("fori_loop", "control_flow", ("all",), references=("jax.lax",)),
    _planned("cond", "control_flow", ("all",), references=("jax.lax",)),
    _planned("switch", "control_flow", ("all",), references=("jax.lax",)),
    _planned("map", "control_flow", ("all",), references=("jax.lax",)),
    _planned("value_and_grad", "transform", ("all",), references=("jax",)),
    _planned("vjp", "transform", ("all",), references=("jax",)),
    _planned("jvp", "transform", ("all",), references=("jax",)),
    _planned("vmap", "transform", ("all",), references=("jax",)),
    _planned("pmap", "transform", ("all",), references=("jax",)),
    _planned("remat", "transform", ("all",), references=("jax",)),
    _planned("checkpoint", "transform", ("all",), references=("jax",)),
    _planned("autocast", "transform", ("all",), references=("torch.autocast", "jax.numpy"),
             notes="S9 numerics — autocast is a transform over primitives."),
    _planned("axis_index", "transform", ("all",), references=("jax.lax",)),
    _planned("axis_size", "transform", ("all",), references=("jax.lax",)),
    _planned("axis_name", "transform", ("all",), references=("jax.lax",)),
    # ── S6: sharding + collectives ──────────────────────────────────────
    _planned("shard_map", "sharding", ("all",), references=("jax.shard_map",)),
    _planned("named_sharding", "sharding", ("all",), references=("jax.sharding",)),
    _planned("partition_spec", "sharding", ("all",), references=("jax.sharding",)),
    _planned("psum", "collective", ("all",), references=("jax.lax",)),
    _planned("pmean", "collective", ("all",), references=("jax.lax",)),
    _planned("pmax", "collective", ("all",), references=("jax.lax",)),
    _planned("pmin", "collective", ("all",), references=("jax.lax",)),
    _planned("collective_permute", "collective", ("all",), references=("jax.lax",)),
    _planned("broadcast_to_axis", "collective", ("all",), references=("jax.lax",)),
    # ── S3: state trees ─────────────────────────────────────────────────
    _planned("tree_flatten", "state_tree", ("all",), references=("jax.tree", "flax.nnx")),
    _planned("tree_unflatten", "state_tree", ("all",), references=("jax.tree", "flax.nnx")),
    _planned("tree_map", "state_tree", ("all",), references=("jax.tree", "flax.nnx")),
    _planned("tree_reduce", "state_tree", ("all",), references=("jax.tree",)),
    _planned("tree_transpose", "state_tree", ("all",), references=("jax.tree",)),
    _planned("state_filter", "state_tree", ("all", "Titans/Atlas"), references=("flax.nnx",)),
    _planned("state_partition", "state_tree", ("all",), references=("flax.nnx",)),
    # ── S4: RNG samplers ────────────────────────────────────────────────
    _planned("rng_key", "rng", ("all", "diffusion", "JEPA"), references=("jax.random",)),
    _planned("rng_split", "rng", ("all", "diffusion", "JEPA"), references=("jax.random",)),
    _planned("rng_fold_in", "rng", ("all", "sharding"), references=("jax.random",)),
    _planned("rng_clone", "rng", ("all",), references=("jax.random",)),
    _planned("rng_truncated_normal", "rng", ("diffusion", "JEPA"), references=("jax.random",)),
    _planned("rng_bernoulli", "rng", ("diffusion", "JEPA"), references=("jax.random",)),
    _planned("rng_categorical", "rng", ("diffusion", "inference"), references=("jax.random",)),
    _planned("rng_multinomial", "rng", ("inference",), references=("jax.random",)),
    _planned("rng_randint", "rng", ("all",), references=("jax.random",)),
    _planned("rng_permutation", "rng", ("JEPA", "diffusion"), references=("jax.random",)),
    _planned("rng_gamma", "rng", ("diffusion",), references=("jax.random",)),
    _planned("rng_beta", "rng", ("diffusion",), references=("jax.random",)),
    _planned("rng_dirichlet", "rng", ("diffusion", "JEPA"), references=("jax.random",)),
    _planned("rng_poisson", "rng", ("diffusion",), references=("jax.random",)),
    # ── S7: model layers ────────────────────────────────────────────────
    _planned("conv1d", "model_layer", ("all", "Mamba/SSM"), references=("flax.nnx", "aten")),
    _planned("conv_transpose", "model_layer", ("diffusion", "JEPA"), references=("flax.nnx", "aten")),
    _planned("linear_general", "model_layer", ("Linformer/cosFormer", "JEPA"), references=("flax.nnx",)),
    _planned("lora_linear", "model_layer", ("all",), references=("flax.nnx",)),
    _planned("group_norm", "normalization", ("diffusion", "JEPA"), references=("flax.nnx", "aten")),
    _planned("instance_norm", "normalization", ("diffusion", "JEPA"), references=("flax.nnx", "aten")),
    _planned("weight_norm", "normalization", ("all",), references=("flax.nnx",)),
    _planned("spectral_norm", "normalization", ("diffusion", "Hyena/FNet/spectral"), references=("flax.nnx",)),
    _planned("max_pool", "pooling", ("diffusion", "JEPA"), references=("flax.nnx", "aten")),
    _planned("avg_pool", "pooling", ("diffusion", "JEPA"), references=("flax.nnx", "aten")),
    _planned("min_pool", "pooling", ("diffusion",), references=("aten",)),
    _planned("adaptive_pool", "pooling", ("diffusion", "JEPA"), references=("flax.nnx", "aten")),
    _planned("gru_cell", "recurrent", ("RNN/xLSTM",), references=("flax.nnx", "aten")),
    _planned("simple_rnn_cell", "recurrent", ("RNN/xLSTM",), references=("flax.nnx",)),
    _planned("bidirectional_scan", "recurrent", ("RNN/xLSTM", "JEPA"), references=("flax.nnx", "jax.lax")),
    # ── S7: position encodings + attention library ──────────────────────
    _planned("alibi", "position_encoding", ("all",), references=("aten",)),
    _planned("ntk_rope", "position_encoding", ("all",), references=("aten",)),
    _planned("multi_head_attention", "attention", ("all",), references=("flax.nnx", "aten")),
    _planned("gqa_attention", "attention", ("all",), references=("aten",)),
    _planned("mqa_attention", "attention", ("all",), references=("aten",)),
    _planned("mla_decode", "attention", ("all",), references=("aten",)),
    _planned("gated_attention", "attention", ("all",), references=("flax.nnx", "aten")),
    _planned("hybrid_attention", "attention", ("Kimi", "Ling"), references=("Kimi Linear", "Ling 2.5")),
    _planned("deepseek_sparse_attention", "attention", ("DeepSeek",), references=("DeepSeek NSA",)),
    _planned("lightning_attention", "attention", ("Ling", "linear_attention"), references=("Lightning Attention",)),
    _planned("gated_deltanet", "attention", ("Kimi", "linear_attention"), references=("Gated DeltaNet",)),
    _planned("kimi_delta_attention", "attention", ("Kimi",), references=("Kimi Linear",)),
    _planned("modified_delta_attention", "attention", ("Kimi",), references=("Kimi Linear",)),
    # ── S7: Titans/Atlas memory ─────────────────────────────────────────
    _planned("memory_read", "memory", ("Titans/Atlas",), references=("flax.nnx",)),
    _planned("memory_write", "memory", ("Titans/Atlas",), references=("flax.nnx",)),
    _planned("memory_evict", "memory", ("Titans/Atlas",), references=("flax.nnx",)),
    # ── S9: numerics + quantization ─────────────────────────────────────
    _planned("quantize_int8", "quantization", ("all", "inference"),
             references=("torch.quantization", "jax.numpy")),
    _planned("dequantize_int8", "quantization", ("all", "inference"),
             references=("torch.quantization", "jax.numpy")),
    _planned("quantize_int4", "quantization", ("inference",), references=("aten",)),
    _planned("dequantize_int4", "quantization", ("inference",), references=("aten",)),
    _planned("fake_quantize", "quantization", ("all",), references=("torch.quantization",),
             notes="QAT — straight-through-estimator VJP."),
    _planned("calibration_observer", "quantization", ("all", "inference"),
             references=("torch.quantization",)),
    _planned("grad_scaler_step", "numerics", ("all",), references=("torch.cuda.amp",)),
    # ── S10: optimizers + schedules ─────────────────────────────────────
    _planned("sgd", "optimizer", ("all",), references=("optax", "torch.optim")),
    _planned("momentum", "optimizer", ("all",), references=("optax",)),
    _planned("nesterov", "optimizer", ("all",), references=("optax",)),
    _planned("adamw", "optimizer", ("all",), references=("optax", "torch.optim")),
    _planned("adafactor", "optimizer", ("all",), references=("optax",)),
    _planned("lion", "optimizer", ("all",), references=("optax",)),
    _planned("muon", "optimizer", ("all",), references=("torch.optim",)),
    _planned("lamb", "optimizer", ("all",), references=("optax",)),
    _planned("cosine_lr", "schedule", ("all",), references=("optax",)),
    _planned("cosine_warmup_lr", "schedule", ("all",), references=("optax",)),
    _planned("linear_warmup_lr", "schedule", ("all",), references=("optax",)),
    _planned("polynomial_lr", "schedule", ("all",), references=("optax",)),
    _planned("inverse_sqrt_lr", "schedule", ("all",), references=("optax",)),
    _planned("clip_grad_norm", "grad_transform", ("all",), references=("optax", "torch.nn.utils")),
    _planned("clip_grad_value", "grad_transform", ("all",), references=("optax", "torch.nn.utils")),
    _planned("ema_update", "grad_transform", ("all", "diffusion"), references=("optax",)),
    _planned("polyak_avg", "grad_transform", ("all",), references=("optax",)),
    # ── S11: losses ─────────────────────────────────────────────────────
    _planned("mse_loss", "loss", ("all", "diffusion", "JEPA"), references=("optax", "torch.nn.functional")),
    _planned("mae_loss", "loss", ("all",), references=("optax", "torch.nn.functional")),
    _planned("huber_loss", "loss", ("all",), references=("optax", "torch.nn.functional")),
    _planned("smooth_l1_loss", "loss", ("all",), references=("torch.nn.functional",)),
    _planned("log_cosh_loss", "loss", ("all",), references=("optax",)),
    _planned("binary_cross_entropy_loss", "loss", ("all",), references=("optax", "torch.nn.functional")),
    _planned("focal_loss", "loss", ("all",), references=("torch.nn.functional",)),
    _planned("label_smoothed_cross_entropy", "loss", ("all",), references=("optax",)),
    _planned("kl_divergence", "loss", ("all", "diffusion"), references=("optax", "torch.nn.functional")),
    _planned("js_divergence", "loss", ("all",), references=("optax",)),
    _planned("wasserstein_distance", "loss", ("diffusion",), references=("optax",)),
    _planned("nt_xent_loss", "loss", ("JEPA",), references=("torch.nn.functional",)),
    _planned("info_nce_loss", "loss", ("JEPA",), references=("torch.nn.functional",)),
    _planned("triplet_loss", "loss", ("JEPA",), references=("torch.nn.functional",)),
    _planned("contrastive_loss", "loss", ("JEPA",), references=("torch.nn.functional",)),
    _planned("cosine_embedding_loss", "loss", ("JEPA",), references=("torch.nn.functional",)),
    _planned("ddpm_noise_pred_loss", "loss", ("diffusion",), references=("torch.nn.functional",)),
    _planned("vlb_loss", "loss", ("diffusion",), references=("torch.nn.functional",)),
    _planned("score_matching_loss", "loss", ("diffusion",), references=("torch.nn.functional",)),
    _planned("ctc_loss", "loss", ("RNN/xLSTM",), references=("torch.nn.functional",)),
    _planned("seq2seq_loss", "loss", ("RNN/xLSTM",), references=("torch.nn.functional",)),
    # ── S12: serialization + checkpointing ──────────────────────────────
    _planned("save_state", "serialization", ("all",), references=("orbax", "torch.save")),
    _planned("load_state", "serialization", ("all",), references=("orbax", "torch.load")),
    _planned("save_sharded", "serialization", ("all",), references=("orbax",)),
    _planned("load_sharded", "serialization", ("all",), references=("orbax",)),
    _planned("state_migration", "serialization", ("all",), references=("orbax",),
             notes="Versioned checkpoint upgrades — explicit field renames + dtype upgrades."),
    # ── S13: custom-primitive / extension API ───────────────────────────
    _planned("custom_primitive", "extension", ("all",), references=("jax.custom_vjp", "torch.autograd.Function")),
    _planned("custom_call", "extension", ("all",), references=("jax.custom_call", "torch.ops")),
    _planned("custom_vjp", "extension", ("all",), references=("jax.custom_vjp",)),
    _planned("custom_jvp", "extension", ("all",), references=("jax.custom_jvp",)),
    _planned("custom_batching", "extension", ("all",), references=("jax",)),
    # ── S14: cache + AOT export ─────────────────────────────────────────
    _planned("aot_export", "aot", ("all", "inference"), references=("jax.export", "torch.export")),
    _planned("aot_load", "aot", ("all", "inference"), references=("jax.export", "torch.export")),
    _planned("stablehlo_export", "aot", ("all", "inference"), references=("jax.export",)),
    _planned("gguf_export", "aot", ("inference",), references=("llama.cpp",)),
    _planned("safetensors_export", "aot", ("all", "inference"), references=("safetensors",)),
    _planned("compilation_cache", "aot", ("all",), references=("jax.experimental.compilation_cache",)),
    # ── S15: data pipeline (in-scope per S0) ────────────────────────────
    _planned("dataset_map", "data", ("all",), references=("tf.data", "torch.utils.data")),
    _planned("dataset_filter", "data", ("all",), references=("tf.data",)),
    _planned("dataset_batch", "data", ("all",), references=("tf.data", "torch.utils.data")),
    _planned("dataset_prefetch", "data", ("all",), references=("tf.data",)),
    _planned("dataset_shuffle", "data", ("all",), references=("tf.data", "torch.utils.data")),
    _planned("dataset_interleave", "data", ("all",), references=("tf.data",)),
    _planned("dataset_repeat", "data", ("all",), references=("tf.data",)),
    _planned("dataset_zip", "data", ("all",), references=("tf.data",)),
    _planned("sharded_dataset", "data", ("all",), references=("grain",)),
    _planned("iterable_dataset", "data", ("all",), references=("torch.utils.data",)),
    _planned("dataset_checkpoint", "data", ("all",), references=("grain",),
             notes="Resume iteration after S12 checkpoint without re-shuffling."),
    _planned("tokenizer_byte", "tokenizer", ("all",), references=("tiktoken",)),
    _planned("tokenizer_bpe", "tokenizer", ("all",), references=("tiktoken", "tokenizers")),
    _planned("tokenizer_wordpiece", "tokenizer", ("all",), references=("tokenizers",)),
    _planned("tokenizer_unigram", "tokenizer", ("all",), references=("sentencepiece",)),
    _planned("tokenizer_sentencepiece_compat", "tokenizer", ("all",), references=("sentencepiece",),
             notes="Reads SentencePiece protobufs but the tokenizer runs in Tessera."),
    # ── EBM2: iterative Markov-chain samplers (extends rng) ──────────────
    # See docs/audit/ga_ebm_roadmap.md § EBM2. These extend the keyed
    # point-sampler surface from S4 with chain-based samplers that
    # produce sequences from energy-defined target distributions.
    _planned("rng_langevin_sample", "rng", ("EBT", "RBM", "score_matching", "diffusion"),
             references=("Welling & Teh 2011",),
             notes="EBM2: unadjusted Langevin (ULA) — biased but cheap."),
    _planned("rng_mala_sample", "rng", ("EBT", "RBM", "score_matching"),
             references=("Roberts & Stramer 2002",),
             notes="EBM2: Metropolis-adjusted Langevin — exact via MH correction."),
    _planned("rng_hmc_sample", "rng", ("EBT", "RBM", "bayesian"),
             references=("Neal 2010", "Betancourt 2017"),
             notes="EBM2: Hamiltonian Monte Carlo via leapfrog; volume-preserving."),
    _planned("rng_gibbs_sample", "rng", ("RBM",),
             references=("Geman & Geman 1984",),
             notes="EBM2: coordinate-wise Gibbs; user-supplied conditional sampler."),
    # ── EBM1: Energy-Based Model primitive surface (Euclidean baseline) ──
    # See docs/spec/EBM_SPEC.md and docs/audit/ga_ebm_roadmap.md § EBM1.
    # All five primitives are Tessera-native — no PyTorch/JAX wrappers
    # per Decision #23. References list reading-only inspiration.
    # 2026-05-18: all five plus `ebm_partition_exact` ship fused MSL
    # kernels on Apple GPU (see ``_EBM_APPLE_GPU_FUSED``), so they
    # promote from `planned` → `partial` per Decision #25 (Python
    # ref + native kernel + tests; contract-axis completeness pending).
    _partial("ebm_energy", "ebm", ("EBT", "RBM", "score_matching"),
             references=("LeCun 2006",),
             notes="EBM1: user-provided scalar energy E(x, y; theta). "
                   "Native fused MSL kernel (quadratic specialization) "
                   "in ``_EBM_APPLE_GPU_FUSED[\"ebm_energy\"]``."),
    _partial("ebm_inner_step", "ebm", ("EBT", "RBM", "score_matching"),
             references=("Du & Mordatch 2019",),
             notes="EBM1: pluggable inner-loop update y' = y - eta*grad [+ noise]. "
                   "Fused MSL kernel ``tessera_apple_gpu_ebm_inner_step_f32``."),
    _partial("ebm_langevin_step", "ebm", ("EBT", "RBM", "score_matching"),
             references=("Welling & Teh 2011",),
             notes="EBM1: one Langevin step; consumes one RNGKey, returns next. "
                   "Fused MSL kernel ``tessera_apple_gpu_ebm_langevin_step_f32``."),
    _partial("ebm_self_verify", "ebm", ("EBT",),
             references=("EBT 2025",),
             notes="EBM1: argmin over K candidates; soft-min variant when beta>0. "
                   "Fused MSL kernel ``tessera_apple_gpu_ebm_self_verify_f32``."),
    _partial("ebm_decode_init", "ebm", ("EBT",),
             references=("EBT 2025",),
             notes="EBM1: initialize K candidate trajectories (noise/base_model/copy). "
                   "Fused MSL kernel ``tessera_apple_gpu_ebm_decode_init_noise_apply_f32``."),
    # ── EBM3: partition function estimators ──────────────────────────────
    _partial("ebm_partition_exact", "ebm", ("RBM",),
             references=("LeCun 2006",),
             notes="EBM3: Z = Σ_s exp(-E(s)) over finite support — fused via "
                   "stable logsumexp in a single MSL dispatch. Native fast "
                   "path: ``tessera.ebm.partition_exact_from_energies`` → "
                   "``tessera_apple_gpu_ebm_partition_exact_f32``."),
    _planned("ebm_partition_monte_carlo", "ebm", ("EBT", "RBM", "score_matching"),
             references=("Robert & Casella 2004",),
             notes="EBM3: importance-sampled Z with effective-sample-size diagnostic."),
    _planned("ebm_partition_ais", "ebm", ("RBM", "diffusion", "bayesian"),
             references=("Neal 2001",),
             notes="EBM3: Annealed Importance Sampling estimator."),
    # ── EBM4: training losses ────────────────────────────────────────────
    _planned("contrastive_divergence_loss", "loss", ("RBM", "EBT"),
             references=("Hinton 2002",),
             notes="EBM4: L = E(x+) - E(x-); k-step CD."),
    _planned("persistent_cd_loss", "loss", ("RBM",),
             references=("Tieleman 2008",),
             notes="EBM4: PCD with persistent negative-sample chain."),
    _planned("implicit_score_matching_loss", "loss", ("score_matching", "EBT"),
             references=("Hyvärinen 2005",),
             notes="EBM4: L = 0.5 ||s||^2 + tr(grad s); no Z required."),
    _planned("denoising_score_matching_loss", "loss", ("score_matching", "diffusion"),
             references=("Vincent 2011",),
             notes="EBM4: L = 0.5 ||s + (y_noisy-y)/sigma^2||^2."),
    # ── EBM7: manifold-aware Langevin integrators ────────────────────────
    # The GA + EBM merge point. State lives on a non-flat manifold;
    # gradient and noise are projected to the appropriate tangent space.
    _partial("ebm_bivector_langevin_step", "ebm",
             ("orientation_diffusion", "SO_diffusion", "lie_group_ebm"),
             references=("Hestenes & Sobczyk", "Roberts & Stramer 2002"),
             notes="EBM7: Langevin on the bivector (so(n)) subspace via grade projection. "
                   "Fused via the shared ``tessera_apple_gpu_ebm_langevin_step_f32`` "
                   "kernel applied to grade-2-projected inputs "
                   "(manifest entry ``ebm_bivector_langevin``)."),
    _partial("ebm_sphere_langevin_step", "ebm",
             ("orientation_diffusion", "vMF_sampling", "manifold_EBM"),
             references=("Roberts & Stramer 2002", "Brubaker et al. 2012"),
             notes="EBM7: Riemannian Langevin on S^{d-1} via tangent projection + normalization retraction. "
                   "Fused MSL kernel ``tessera_apple_gpu_ebm_sphere_langevin_f32``."),
    _planned("ebm_bivector_langevin_sample", "ebm",
             ("orientation_diffusion", "lie_group_ebm"),
             references=("Hestenes & Sobczyk",),
             notes="EBM7: bivector Langevin chain (chain wrapper for EBM7 step)."),
    _planned("ebm_sphere_langevin_sample", "ebm",
             ("orientation_diffusion", "vMF_sampling"),
             references=("Brubaker et al. 2012",),
             notes="EBM7: sphere Langevin chain (chain wrapper)."),
    # ── GA4: Clifford geometric-algebra primitive surface ────────────────
    # See docs/audit/ga_ebm_roadmap.md § GA4. Each primitive corresponds
    # 1:1 to a tessera.clifford Graph IR op landing in GA7. The math /
    # shape / dtype rules are uniformly "complete" (the algebra defines
    # them precisely); other contract axes (batching/sharding/backend
    # kernel) are deferred to GA6+ / GA9.
    _planned("clifford_geometric_product", "geometric_algebra",
             ("GA-MLP", "equivariant_pointcloud", "lorentz_classifier"),
             references=("Hestenes & Sobczyk",),
             notes="GA3: a*b via Cayley table; the foundational Clifford op."),
    _planned("clifford_grade_projection", "geometric_algebra", ("all_GA",),
             references=("Doran & Lasenby",),
             notes="GA3: extract grade-k component."),
    _planned("clifford_wedge", "geometric_algebra", ("all_GA",),
             references=("Hestenes & Sobczyk",),
             notes="GA3: outer / exterior product a∧b."),
    _planned("clifford_left_contraction", "geometric_algebra", ("all_GA",),
             references=("Doran & Lasenby",),
             notes="GA3: left contraction a⌋b."),
    _planned("clifford_inner", "geometric_algebra", ("all_GA",),
             references=("Hestenes & Sobczyk",),
             notes="GA3: scalar inner product <a, b>."),
    _planned("clifford_reverse", "geometric_algebra", ("all_GA",),
             references=("Hestenes & Sobczyk",),
             notes="GA3: reversion anti-automorphism a†."),
    _planned("clifford_grade_involution", "geometric_algebra", ("all_GA",),
             references=("Hestenes & Sobczyk",),
             notes="GA3: grade-flipping involution; even grades preserved."),
    _planned("clifford_conjugate", "geometric_algebra", ("all_GA",),
             references=("Hestenes & Sobczyk",),
             notes="GA3: Clifford conjugation = reverse ∘ involution."),
    _planned("clifford_norm", "geometric_algebra", ("all_GA",),
             references=("Doran & Lasenby",),
             notes="GA3: |a| = sqrt(<a, a>)."),
    _planned("clifford_exp", "geometric_algebra",
             ("GA-MLP", "equivariant_pointcloud"),
             references=("Hestenes & Sobczyk",),
             notes="GA3: exp(B/2) → rotor; closed-form for Cl(3,0) bivectors."),
    _planned("clifford_log", "geometric_algebra",
             ("GA-MLP", "equivariant_pointcloud"),
             references=("Doran & Lasenby",),
             notes="GA3: log(R) → bivector; closed-form for Cl(3,0) rotors."),
    _planned("clifford_rotor_sandwich", "geometric_algebra",
             ("equivariant_pointcloud", "lorentz_classifier"),
             references=("Hestenes & Sobczyk",),
             notes="GA3: R·x·R† — the equivariance-from-algebra primitive."),
    _planned("clifford_hodge_star", "geometric_algebra", ("all_GA", "physics_ML"),
             references=("Frankel — Geometry of Physics",),
             notes="GA5: ⋆ω = reverse(ω)·I; pointwise duality."),
    _planned("clifford_ext_deriv", "geometric_algebra", ("physics_ML",),
             references=("Frankel — Geometry of Physics",),
             notes="GA5: exterior derivative d on a sampled field."),
    _planned("clifford_codiff", "geometric_algebra", ("physics_ML",),
             references=("Frankel — Geometry of Physics",),
             notes="GA5: codifferential d* = ±⋆d⋆."),
    _planned("clifford_vec_deriv", "geometric_algebra", ("physics_ML",),
             references=("Hestenes — Geometric Calculus",),
             notes="GA5: geometric gradient ∂F = Σ_i e_i ∂_i F."),
    _planned("clifford_integral", "geometric_algebra",
             ("physics_ML", "manifold_EBM"),
             references=("Frankel — Geometry of Physics",),
             notes="GA5: ∫_M ω — Riemann-sum integration over a Manifold."),
    # ── M7: Visual Complex Analysis (Needham) ────────────────────────────
    # Public surface lives in ``python/tessera/complex.py`` and
    # ``python/tessera/compiler/complex_jit.py``.  All entries below
    # ship a Python reference; the ``@analytic`` / ``@complex_jit``
    # decorators do Cauchy-Riemann verification at decoration time.
    # 94 focused complex tests pass; contract-axis completeness and
    # native (Apple GPU) lowering are the next gates.
    #
    # 2026-05-19: dropped ``complex_add`` — there is no
    # ``def complex_add`` in ``python/tessera/complex.py``.  Complex
    # addition is just ``+`` on numpy arrays; no Tessera-specific
    # primitive ships for it, so listing one was an overclaim.
    _partial("complex_mul", "visual_complex", ("complex_analysis",),
             references=("Needham — Visual Complex Analysis",),
             notes="M7: ℂ-mul; preserves Cauchy-Riemann under @analytic."),
    _partial("complex_div", "visual_complex", ("complex_analysis",),
             references=("Needham — Visual Complex Analysis",),
             notes="M7: ℂ-div; @analytic checks holomorphic at z."),
    _partial("complex_exp", "visual_complex", ("complex_analysis",),
             references=("Needham — Visual Complex Analysis",),
             notes="M7: exp(z) = e^x (cos y + i sin y); entire."),
    _partial("complex_log", "visual_complex", ("complex_analysis",),
             references=("Needham — Visual Complex Analysis",),
             notes="M7: principal log; branch cut at negative real axis."),
    _partial("complex_sqrt", "visual_complex", ("complex_analysis",),
             references=("Needham — Visual Complex Analysis",),
             notes="M7: principal sqrt; branch cut at negative real axis."),
    _partial("complex_pow", "visual_complex", ("complex_analysis",),
             references=("Needham — Visual Complex Analysis",),
             notes="M7: z^w via exp(w log z); principal branch."),
    _partial("complex_conjugate", "visual_complex", ("complex_analysis",),
             references=("Needham — Visual Complex Analysis",),
             notes="M7: z̄; anti-holomorphic (fails CR check intentionally)."),
    _partial("complex_abs", "visual_complex", ("complex_analysis",),
             references=("Needham — Visual Complex Analysis",),
             notes="M7: |z| = sqrt(x² + y²); not holomorphic."),
    _partial("complex_arg", "visual_complex", ("complex_analysis",),
             references=("Needham — Visual Complex Analysis",),
             notes="M7: arg(z) ∈ (-π, π]; branch cut at negative real."),
    _partial("mobius", "visual_complex",
             ("complex_analysis", "hyperbolic_geometry"),
             references=("Needham — Visual Complex Analysis",
                         "Marden — Hyperbolic Geometry"),
             notes="M7: f(z) = (az+b)/(cz+d); preserves cross-ratio."),
    _partial("mobius_from_three_points", "visual_complex",
             ("complex_analysis",),
             references=("Needham — Visual Complex Analysis",),
             notes="M7: uniquely determined by three-point image."),
    _partial("cross_ratio", "visual_complex", ("complex_analysis",),
             references=("Needham — Visual Complex Analysis",),
             notes="M7: [z1,z2,z3,z4]; Möbius-invariant."),
    _partial("is_concyclic", "visual_complex", ("complex_analysis",),
             references=("Needham — Visual Complex Analysis",),
             notes="M7: four points cocircular iff cross-ratio is real."),
    _partial("stereographic", "visual_complex",
             ("complex_analysis", "conformal_geometry"),
             references=("Needham — Visual Complex Analysis",),
             notes="M7: ℂ ↔ S² stereographic projection from the north pole."),
    _partial("check_cauchy_riemann", "visual_complex",
             ("complex_analysis",),
             references=("Needham — Visual Complex Analysis",),
             notes="M7: holomorphicity certificate via ∂f/∂z̄ residual; "
                   "backs @analytic + @complex_jit decoration-time gate."),
    _partial("conformal_jacobian", "visual_complex",
             ("conformal_geometry",),
             references=("Needham — Visual Complex Analysis",),
             notes="M7: |f'(z)|² for a holomorphic f; area scaling."),
    _partial("conformal_energy_on_sphere", "visual_complex",
             ("conformal_geometry",),
             references=("Needham — Visual Complex Analysis",),
             notes="M7: energy of a Möbius image on S² via stereographic pullback."),
    _partial("dz", "visual_complex", ("complex_analysis",),
             references=("Needham — Visual Complex Analysis",),
             notes="M7: Wirtinger derivative ∂/∂z = ½(∂x - i ∂y)."),
    _partial("dbar", "visual_complex", ("complex_analysis",),
             references=("Needham — Visual Complex Analysis",),
             notes="M7: Wirtinger derivative ∂/∂z̄ = ½(∂x + i ∂y); "
                   "vanishes iff f is holomorphic."),
    _partial("laplacian_2d", "visual_complex",
             ("complex_analysis", "harmonic_analysis"),
             references=("Needham — Visual Complex Analysis",),
             notes="M7: Δ = 4 ∂² /∂z∂z̄; harmonic ⇔ real-part of holomorphic."),
)


def all_primitive_coverages() -> dict[str, PrimitiveCoverage]:
    entries = _existing_coverage()
    planned_names: set[str] = set()
    for entry in _PLANNED_ENTRIES:
        if entry.name in planned_names:
            raise ValueError(f"duplicate planned primitive coverage entry: {entry.name}")
        planned_names.add(entry.name)
        # Decision #25 (2026-05-17): the planned-entry path didn't
        # apply the multi-axis category overrides — only the
        # OP_SPECS + python_primitive paths did.  Apply them here so
        # GA + EBM entries (and any future ``_planned()`` family)
        # get the same category-based axis promotion as the rest of
        # the registry.  We rebuild the frozen dataclass with the
        # promoted contract_status; the rest of the entry is
        # preserved verbatim.
        promoted = dict(entry.contract_status)
        _apply_category_overrides(promoted, entry.category)
        _apply_effect_overrides(promoted, entry.effect or "pure")
        _apply_per_name_overrides(promoted, entry.name)
        promoted.update(_EXISTING_CONTRACT_OVERRIDES.get(entry.name, {}))
        # E3 (2026-05-20): apply the same Graph IR lowering classifier
        # to planned/partial entries as the python-primitive path uses
        # (line 1859 area).  Without this, registry entries built via
        # ``_partial`` / ``_planned`` never get the
        # ``metadata.graph_ir_lowering`` field populated, which makes
        # the audit's ``_axis_graph_ir`` walker fall through to
        # ``missing`` even when the category-table classifier says
        # ``registered``.
        category_gir = _graph_ir_lowering_for_category(
            entry.category, str(entry.metadata.get("graph_ir_lowering", ""))
        )
        new_metadata = dict(entry.metadata)
        if category_gir and "graph_ir_lowering" not in new_metadata:
            new_metadata["graph_ir_lowering"] = category_gir
        metadata_changed = new_metadata != dict(entry.metadata)
        if promoted != entry.contract_status or metadata_changed:
            entry = PrimitiveCoverage(
                name=entry.name,
                category=entry.category,
                status=entry.status,
                contract_status=promoted,
                model_families=entry.model_families,
                references=entry.references,
                notes=entry.notes,
                existing_op=entry.existing_op,
                graph_name=entry.graph_name,
                effect=entry.effect,
                lowering=entry.lowering,
                metadata=new_metadata,
            )
        entries.setdefault(entry.name, entry)
    # GA9 (2026-05-17): planned entries don't go through
    # `_supplemental_metadata` so they miss the manifest attachment.
    # Walk planned entries that have a registered manifest (currently
    # the `clifford_*` GA9 primitives) and graft the backend matrix
    # onto their metadata so the dashboard + audit walkers can see it.
    for name, entry in list(entries.items()):
        if "backend_kernel_manifest" in entry.metadata:
            continue
        manifest = _manifest_for_name(name)
        if not manifest:
            continue
        new_metadata = dict(entry.metadata)
        new_metadata["backend_kernel_manifest"] = manifest
        entries[name] = PrimitiveCoverage(
            name=entry.name,
            category=entry.category,
            status=entry.status,
            contract_status=entry.contract_status,
            model_families=entry.model_families,
            references=entry.references,
            notes=entry.notes,
            existing_op=entry.existing_op,
            graph_name=entry.graph_name,
            effect=entry.effect,
            lowering=entry.lowering,
            metadata=new_metadata,
        )
    # ────────────────────────────────────────────────────────────────────
    # Backend kernel vertical slice (Sprint #19 + #19b, 2026-05-22):
    # honest reclassification of `backend_kernel = "planned"` rows
    # that ship at least one real implementation via the
    # `backend_kernel_manifest`.
    #
    # Semantics (Sprint #19b widening, 2026-05-22):
    #   "planned"  — no manifest, or every manifest slot is itself
    #                `planned`/`artifact_only` (no executable code path).
    #                `artifact_only` means Target IR ships but cannot
    #                execute (Phase G/H/I gate) — labeling that as
    #                `partial` would be misleading.
    #   "partial"  — at least one manifest slot ships an executable
    #                implementation: `fused`, `compileable`, or
    #                `reference` (the Python eval path).  This matches
    #                the `_existing_contracts` walker's default
    #                `backend_kernel="partial"` for OP_SPECS entries
    #                (every catalog op has at least a numpy reference);
    #                python_primitive rows with the same shape now
    #                label consistently.
    #   "complete" — every documented target ships (universal Phase G/H/I
    #                gate; the s_series_status drift test
    #                ``test_backend_kernel_is_universal_phase_g_gate``
    #                enforces that this stays universally open until the
    #                hardware lanes light up).
    #
    # Sprint #19a (initial pass) flipped 23 GA/EBM entries with fused
    # Apple GPU MSL kernels.  Sprint #19b widens to recognize
    # ``reference`` so the two EBM partition rows
    # (`ebm_partition_ais`, `ebm_partition_monte_carlo`) align with the
    # OP_SPECS walker.  This is a labeling honesty fix only — no
    # contract claim changes; `open_n = partial + planned` stays
    # invariant so the universal Phase G gate test is unaffected.
    # ────────────────────────────────────────────────────────────────────
    _REAL_IMPLEMENTATION_STATUSES = {"fused", "compileable", "reference"}
    for name, entry in list(entries.items()):
        if entry.contract_status.get("backend_kernel") != "planned":
            continue
        manifest = entry.metadata.get("backend_kernel_manifest")
        if not isinstance(manifest, list):
            continue
        ships_real_impl = any(
            isinstance(slot, dict)
            and slot.get("status") in _REAL_IMPLEMENTATION_STATUSES
            for slot in manifest
        )
        if not ships_real_impl:
            continue
        new_contract = dict(entry.contract_status)
        new_contract["backend_kernel"] = "partial"
        entries[name] = PrimitiveCoverage(
            name=entry.name,
            category=entry.category,
            status=entry.status,
            contract_status=new_contract,
            model_families=entry.model_families,
            references=entry.references,
            notes=entry.notes,
            existing_op=entry.existing_op,
            graph_name=entry.graph_name,
            effect=entry.effect,
            lowering=entry.lowering,
            metadata=entry.metadata,
        )
    return dict(sorted(entries.items()))


def coverage_for(name: str) -> PrimitiveCoverage:
    entries = all_primitive_coverages()
    try:
        return entries[name]
    except KeyError as exc:
        raise KeyError(f"unknown primitive coverage entry: {name}") from exc


def primitives_for_model_family(family: str) -> tuple[PrimitiveCoverage, ...]:
    return tuple(
        entry
        for entry in all_primitive_coverages().values()
        if family in entry.model_families or "all" in entry.model_families
    )


def coverage_summary() -> dict[str, int]:
    summary: dict[str, int] = {}
    for entry in all_primitive_coverages().values():
        summary[entry.status] = summary.get(entry.status, 0) + 1
    return summary


# ──────────────────────────────────────────────────────────────────────────
# Sprint A0 — Canonical-dtype enforcement (forward-looking)
#
# These helpers walk the registry looking for dtype identifiers stored on
# entries' metadata.  The intent is:
#
#   1. Today the registry stores dtype completeness as a *status* (the
#      `dtype_layout_rule` axis), not as identifiers.  This walker is a
#      no-op against the current registry.
#
#   2. Sprint C2 adds `metadata.numeric_policy = NumericPolicy(
#         storage=..., accum=..., scale=..., quant_axis=..., ...)` for
#      promoted ops (matmul/fft/quant/etc.).  Each storage/accumulator
#      slot is a dtype string that must be canonical.
#
#   3. Sprint A0+ planned-gated track: entries that reference
#      uint*/complex*/int4/mxfp*/bfp* must carry
#      ``metadata.dtype_status = "planned_gated"``.  This walker is the
#      gate.
#
# The walker is invoked from `tests/unit/test_canonical_dtype.py` and is
# also exposed publicly via `audit_canonical_dtypes()`.
# ──────────────────────────────────────────────────────────────────────────

# Metadata keys whose values, when strings, are interpreted as dtype
# identifiers.  Add to this set as future schemas land.
_DTYPE_METADATA_KEYS: frozenset[str] = frozenset({
    "dtype",
    "storage_dtype",
    "accum_dtype",
    "scale_dtype",
    "quant_dtype",
    "output_dtype",
    "input_dtype",
})


def _iter_dtype_strings_in_entry(entry: "PrimitiveCoverage") -> Iterable[tuple[str, str]]:
    """Yield (metadata_key, dtype_string) tuples found on a registry entry.

    Scans:
      - top-level metadata dict values whose key is in `_DTYPE_METADATA_KEYS`
      - nested values under `numeric_policy` (planned for Sprint C2)
    """
    md = entry.metadata or {}
    for key in _DTYPE_METADATA_KEYS:
        val = md.get(key)
        if isinstance(val, str):
            yield key, val
        elif isinstance(val, (list, tuple)):
            for item in val:
                if isinstance(item, str):
                    yield key, item
    # numeric_policy (forward-compat for Sprint C2)
    np_meta = md.get("numeric_policy")
    if isinstance(np_meta, Mapping):
        for sub_key in ("storage", "accum", "scale_dtype"):
            sub_val = np_meta.get(sub_key)
            if isinstance(sub_val, str):
                yield f"numeric_policy.{sub_key}", sub_val


def audit_canonical_dtypes() -> dict[str, list[tuple[str, str, str]]]:
    """Walk the live registry, classify every stored dtype string.

    Returns a dict with four buckets, each a list of
    ``(primitive_name, metadata_key, dtype_string)`` tuples:

      - ``"canonical"``    : dtype is a canonical spelling
      - ``"alias"``        : dtype is an accepted alias (e.g., ``"f32"``)
      - ``"planned_gated"``: dtype is a planned/gated spelling
      - ``"unknown"``      : dtype is not recognized

    The intent is: ``unknown`` MUST be empty.  ``alias`` SHOULD be empty
    (entries should store the canonical spelling), but is allowed during
    migration.  ``planned_gated`` entries must also declare
    ``metadata.dtype_status == "planned_gated"``.
    """
    from tessera.dtype import (
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
    for name, entry in all_primitive_coverages().items():
        for key, dt in _iter_dtype_strings_in_entry(entry):
            if is_canonical_dtype(dt):
                buckets["canonical"].append((name, key, dt))
            elif dt in aliases:
                buckets["alias"].append((name, key, dt))
            elif is_planned_gated_dtype(dt):
                buckets["planned_gated"].append((name, key, dt))
            else:
                buckets["unknown"].append((name, key, dt))
    return buckets


def assert_canonical_dtypes() -> None:
    """Assert the registry contains no unknown or unannounced gated dtypes.

    Rules enforced:
      - bucket ``unknown`` must be empty
      - every entry in ``planned_gated`` must carry
        ``metadata.dtype_status == "planned_gated"``

    Raises ``AssertionError`` with a precise list on violation.  Wired
    into the canonical-dtype guard test
    (``tests/unit/test_canonical_dtype.py``).
    """
    buckets = audit_canonical_dtypes()
    if buckets["unknown"]:
        bad = "; ".join(f"{n}.{k}={d!r}" for n, k, d in buckets["unknown"])
        raise AssertionError(
            f"registry has unknown dtype strings: {bad}.  Use "
            "tessera.dtype.canonicalize_dtype() or add to the canonical / "
            "planned-gated tables in tessera.dtype."
        )
    reg = all_primitive_coverages()
    for name, key, dt in buckets["planned_gated"]:
        entry = reg.get(name)
        # ``reg.get`` can return None for entries not in the registry;
        # treat missing metadata as "no status declared".
        status = (entry.metadata if entry is not None else {}).get("dtype_status")
        if status != "planned_gated":
            raise AssertionError(
                f"{name}.{key} = {dt!r} is a planned/gated dtype but the "
                f"entry's metadata.dtype_status is {status!r} "
                f"(must be 'planned_gated')."
            )


def render_markdown(entries: Iterable[PrimitiveCoverage] | None = None) -> str:
    rows = list(entries if entries is not None else all_primitive_coverages().values())
    lines = [
        "# Standalone Primitive Coverage",
        "",
        "This dashboard tracks Tessera-native compiler primitive completeness.",
        "External frameworks are references only; they are not runtime dependencies.",
        "",
        "| Primitive | Category | Status | Existing op | Lowering gate | Backend gate | Missing contracts | Model families |",
        "|-----------|----------|--------|-------------|---------------|--------------|-------------------|----------------|",
    ]
    for entry in rows:
        missing = ", ".join(entry.missing_contracts()) or "-"
        families = ", ".join(entry.model_families) or "-"
        existing = "yes" if entry.existing_op else "no"
        lowering_gate = entry.metadata.get("graph_ir_lowering", "-")
        backend_gate = entry.metadata.get("backend_kernel", "-")
        lines.append(
            f"| `{entry.name}` | {entry.category} | {entry.status} | "
            f"{existing} | {lowering_gate} | {backend_gate} | {missing} | {families} |"
        )
    return "\n".join(lines) + "\n"


__all__ = [
    "CONTRACT_FIELDS",
    "PrimitiveCoverage",
    "all_primitive_coverages",
    "coverage_for",
    "coverage_summary",
    "primitives_for_model_family",
    "render_markdown",
    # Sprint A0 — canonical-dtype audit
    "audit_canonical_dtypes",
    "assert_canonical_dtypes",
]
