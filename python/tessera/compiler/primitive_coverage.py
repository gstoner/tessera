"""Standalone compiler primitive coverage registry.

`op_catalog.py` records operators that Tessera accepts today. This module is a
separate planning and audit surface: it records the semantic contracts a
primitive must satisfy before it is considered complete for a standalone model
compiler.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Mapping

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
        from tessera.autodiff.vjp import _VJPS  # type: ignore
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
    metadata: Mapping[str, str] = field(default_factory=dict)

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
#     well-understood TP/SP placement but no compiler-level sharding pass
#     yet (this becomes "complete" with Phase G's mesh integration).
#   - `vjp`/`jvp` → "not_applicable" for state-effect or non-differentiable
#     ops (KV cache writes, RNG samplers, structural ops).
#   - `backend_kernel` stays `partial` until each backend ships a real
#     hardware kernel — that's Phase G/H/I work.
_EXISTING_CONTRACT_OVERRIDES: dict[str, dict[str, str]] = {
    # ── KV cache state-effect ops ────────────────────────────────────────
    # `append` concatenates K/V slices along the sequence axis;
    # `prune` drops oldest entries beyond the configured window. These are
    # state mutators; gradient never flows through a cache write. Math /
    # shape / dtype / batching contracts are determinate; sharding along
    # the head axis is well-understood but lives behind Phase G.
    "kv_cache_append": {
        "vjp": "not_applicable",
        "jvp": "not_applicable",
        "transpose_rule": "not_applicable",
        "math_semantics": "complete",
        "shape_rule": "complete",
        "dtype_layout_rule": "complete",
        "masking_effect_rule": "complete",
        "batching_rule": "complete",
        "sharding_rule": "partial",
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
        "sharding_rule": "partial",
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
        "sharding_rule": "partial",
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
}

# ── Shared override dicts for attention family + RL losses ──────────────
# softmax(QKᵀ/√d)V over [B, H, S, D] — every axis follows the standard
# transformer convention. TP along H is well-understood; sequence
# parallelism along S is staged behind Phase G mesh integration.
_ATTN_HARDENED: dict[str, str] = {
    "math_semantics": "complete",
    "shape_rule": "complete",
    "dtype_layout_rule": "complete",
    "batching_rule": "complete",
    "transpose_rule": "partial",
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
    "sharding_rule": "partial",
    "masking_effect_rule": "not_applicable",
}

for _name in (
    # ── Standard attention wrappers ──────────────────────────────────
    "flash_attn", "multi_head_attention", "gqa_attention", "mqa_attention",
    # ── MLA family (DeepSeek-style multi-head latent attention) ──────
    "latent_kv_compress", "latent_kv_expand_k", "latent_kv_expand_v",
    "mla_decode", "mla_decode_fused",
    # ── Sparse attention (MoSA + MiniMax sparse path) ────────────────
    "attn_sliding_window", "attn_top_k_blocks", "attn_compressed_blocks",
    # ── Linear / recurrent attention (Lightning, Megalodon) ──────────
    "linear_attn", "linear_attn_state", "power_attn", "retention",
    # ── Reasoning-model attention family (S-series 2026-05-10) ───────
    # Each has a dedicated ODS op in TesseraOps.td and a corresponding pass
    # in src/transforms/lib/AttentionFamilyPasses.cpp.
    "deepseek_sparse_attention", "lightning_attention", "gated_attention",
    "hybrid_attention", "gated_deltanet", "kimi_delta_attention",
    "modified_delta_attention",
):
    _EXISTING_CONTRACT_OVERRIDES[_name] = _ATTN_HARDENED

for _name in ("ppo_policy_loss", "grpo_policy_loss", "cispo_policy_loss"):
    _EXISTING_CONTRACT_OVERRIDES[_name] = _RL_LOSS_HARDENED
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
    "normalization":       "partial",   # all-reduce on feature axis when sharded
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
    "grad_transform":      "partial",   # clip_grad_norm needs cross-param sum

    # — Optimizers: per-parameter, ZeRO-style shardable —
    "functional_optimizer_step": "complete",
    "optimizer":           "complete",

    # — Schedules: scalar functions of step; no sharding —
    "schedule":            "not_applicable",

    # — RL post-training losses: standard reductions —
    "rl_loss":             "complete",

    # — Attention & friends: known TP/SP patterns, mesh-dependent —
    "attention":           "partial",
    "loop_nest":           "partial",   # matmul/gemm/conv — TP along contract axis
    "model_layer":         "partial",
    "contraction":         "partial",   # einsum
    "projection":          "partial",
    "fused_epilogue":      "partial",
    "moe":                 "partial",
    "state_update":        "partial",   # kv_cache — handle layout matters
    "state_space":         "partial",   # selective_ssm
    "recurrent":           "partial",
    "stencil":             "partial",   # depthwise_conv1d/2d — halo exchange
    "pooling":             "partial",

    # — Structural / layout / indexing: rule depends on partition spec —
    "tensor_algebra":      "partial",
    "layout_transform":    "partial",
    "indexing":            "partial",
    "segment_reduce":      "partial",

    # — Spectral: ring/butterfly partition rules well-known —
    "spectral":            "partial",

    # — Sort / top-k: indices must be replicated; partial-axis sharding —
    "sort":                "partial",

    # — Linear algebra solvers: sophisticated partition rules —
    "linalg_solver":       "partial",
    "linalg_decomposition":"partial",
    "sparse":              "partial",

    # — Transforms: sharding-aware by nature —
    "transform":           "complete",  # vjp/jvp/vmap/pmap/remat — self-defining
    "control_flow":        "partial",   # scan/while depend on body

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
    "tensor_algebra":      "partial",    # batched-axis semantics shift with reshape/permute
    "layout_transform":    "partial",
    "indexing":            "partial",
    "segment_reduce":      "partial",
    "control_flow":        "partial",    # scan body-dependent
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
    "attention":           "partial",
    "quantize":            "partial",    # STE transpose
    "quantization":        "partial",
    "moe":                 "partial",
    "moe_transport":       "partial",
    "recurrent":           "partial",
    "stencil":             "partial",    # transpose-conv is well-defined but distinct shape
    "pooling":             "partial",    # max-pool transpose = unpool-with-indices
    "tensor_algebra":      "partial",    # reshape^T = reshape; permute^T = inv-permute
    "layout_transform":    "partial",
    "indexing":            "partial",    # gather^T = scatter
    "segment_reduce":      "partial",
    "control_flow":        "partial",
    "memory":              "partial",
    "grad_transform":      "partial",
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
    # Partial: variant-dependent or storage-format-dependent
    "attention":           "partial",    # layout variants (NHD vs HND)
    "moe":                 "partial",
    "moe_transport":       "partial",
    "state_space":         "partial",    # selective state has variants
    "recurrent":           "partial",
    "memory":              "partial",
    "control_flow":        "partial",
    "linalg_solver":       "partial",
    "linalg_decomposition":"partial",
    "sparse":              "partial",
    # Non-tensor categories — math doesn't apply but shape/dtype usually do
    "state_tree":          "not_applicable",
    "data":                "partial",      # streaming surface; variable shapes
    "tokenizer":           "partial",      # variable-length output
    "aot":                 "not_applicable",
    "serialization":       "not_applicable",
    "conformance":         "not_applicable",
    "schedule":            "complete",      # scalar functions of step
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
    # Everything else (tensor_algebra / layout_transform / indexing / sparse /
    # linalg / etc.) stays at whatever existing path set (partial for python
    # primitives without decomposition; complete for OP_SPECS imports).
}

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
    _promote("sharding_rule",  _SHARDING_RULE_BY_CATEGORY,  frozenset({"planned"}))
    _promote("batching_rule",  _BATCHING_RULE_BY_CATEGORY,  frozenset({"planned"}))
    _promote("transpose_rule", _TRANSPOSE_RULE_BY_CATEGORY, frozenset({"planned"}))
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
            metadata={
                "implementation": "op_catalog",
                "contract_schema": schema,
                "graph_ir_lowering": "registered",
                "backend_kernel": "partial",
            },
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
                metadata={
                    "implementation": "python_reference",
                    "contract_schema": "explicit_partial",
                    "graph_ir_lowering": "missing",
                    "backend_kernel": "reference_only",
                },
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
        "memory_read": {
            "math_semantics": "complete",
            "shape_rule": "complete",
            "dtype_layout_rule": "complete",
            "batching_rule": "partial",
            # The top-k argmax-routing makes the transpose rule depend on
            # whether indices are treated as constants (standard) or as
            # straight-through. Mark `partial` matching the category default.
            "transpose_rule": "partial",
            # Memory-table sharding: keys/values split along the entries axis;
            # query-time top-k requires an all-gather of the scores before the
            # softmax. Well-understood but mesh-aware — `partial` per the
            # category convention for memory primitives.
            "sharding_rule": "partial",
        },
        "memory_write": {
            "math_semantics": "complete",
            "shape_rule": "complete",
            "dtype_layout_rule": "complete",
            "vjp": "not_applicable",
            "jvp": "not_applicable",
            "batching_rule": "partial",
            "transpose_rule": "not_applicable",
        },
        "memory_evict": {
            "math_semantics": "complete",
            "shape_rule": "complete",
            "dtype_layout_rule": "complete",
            "vjp": "not_applicable",
            "jvp": "not_applicable",
            "batching_rule": "partial",
            "transpose_rule": "not_applicable",
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
        metadata = {
            "implementation": "python_reference",
            "contract_schema": "explicit_partial",
            "graph_ir_lowering": "stub_required",
            "backend_kernel": "reference_only",
        }
        if category in {"data", "tokenizer", "serialization", "aot", "conformance"}:
            metadata["graph_ir_lowering"] = "not_applicable"
        if category == "conformance":
            metadata["model_manifest"] = "examples.conformance.s8_tiny_models.manifest"
        if all(
            contract[field] == "complete"
            for field in ("math_semantics", "shape_rule", "dtype_layout_rule")
        ):
            metadata["contract_schema"] = "explicit_semantic"
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
)


def all_primitive_coverages() -> dict[str, PrimitiveCoverage]:
    entries = _existing_coverage()
    planned_names: set[str] = set()
    for entry in _PLANNED_ENTRIES:
        if entry.name in planned_names:
            raise ValueError(f"duplicate planned primitive coverage entry: {entry.name}")
        planned_names.add(entry.name)
        entries.setdefault(entry.name, entry)
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
]
