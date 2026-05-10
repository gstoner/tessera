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


def _existing_contracts(effect: str, *, vjp_complete: bool = False) -> dict[str, str]:
    effect_rule = "partial" if effect != "pure" else "not_applicable"
    return _contracts(
        math_semantics="partial",
        shape_rule="partial",
        dtype_layout_rule="partial",
        vjp="complete" if vjp_complete else "planned",
        masking_effect_rule=effect_rule,
        lowering_rule="complete",
        backend_kernel="partial",
        tests="partial",
    )


def _existing_op_has_vjp(public_name: str, registered: frozenset[str]) -> bool:
    """True iff `public_name` (or a known alias) has a registered VJP."""
    candidates = _VJP_ALIASES.get(public_name, (public_name,))
    return any(name in registered for name in candidates)


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
    "power_attn": ("Megalodon/Griffin",),
    "retention": ("Megalodon/Griffin",),
    "rfft": ("Hyena/FNet/spectral",),
    "selective_ssm": ("Mamba/SSM",),
    "spectral_conv": ("Hyena/FNet/spectral",),
    "spectral_filter": ("Hyena/FNet/spectral",),
    "stft": ("Hyena/FNet/spectral",),
    "istft": ("Hyena/FNet/spectral",),
}


def _existing_coverage() -> dict[str, PrimitiveCoverage]:
    registered_vjps = _vjp_registered_names()
    entries: dict[str, PrimitiveCoverage] = {}
    for name, spec in sorted(OP_SPECS.items()):
        has_vjp = _existing_op_has_vjp(name, registered_vjps)
        entries[name] = PrimitiveCoverage(
            name=name,
            category=spec.lowering,
            status="partial",
            contract_status=_existing_contracts(spec.effect, vjp_complete=has_vjp),
            model_families=_EXISTING_MODEL_FAMILIES.get(name, ()),
            references=("tessera",),
            notes="Imported from the supported op catalog; S1 keeps missing semantic rules visible.",
            existing_op=True,
            graph_name=spec.graph_name,
            effect=spec.effect,
            lowering=spec.lowering,
        )
    supplemental_public_ops = {
        "depthwise_conv1d": ("stencil", "state", "streaming depthwise convolution"),
        "online_softmax": ("stable_reduction", "state", "streaming softmax helper"),
        "online_softmax_state": ("state_update", "state", "streaming softmax carry state"),
        "selective_ssm": ("state_space", "state", "Mamba-style selective state-space op"),
    }
    for name, (lowering, effect, notes) in supplemental_public_ops.items():
        has_vjp = _existing_op_has_vjp(name, registered_vjps)
        entries.setdefault(
            name,
            PrimitiveCoverage(
                name=name,
                category=lowering,
                status="partial",
                contract_status=_existing_contracts(effect, vjp_complete=has_vjp),
                model_families=_EXISTING_MODEL_FAMILIES.get(name, ()),
                references=("tessera",),
                notes=f"Public Python op outside OP_SPECS today; tracked for standalone coverage: {notes}.",
                existing_op=True,
                graph_name=f"tessera.{name}",
                effect=effect,
                lowering=lowering,
            ),
        )
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
        "| Primitive | Category | Status | Existing op | Missing contracts | Model families |",
        "|-----------|----------|--------|-------------|-------------------|----------------|",
    ]
    for entry in rows:
        missing = ", ".join(entry.missing_contracts()) or "-"
        families = ", ".join(entry.model_families) or "-"
        existing = "yes" if entry.existing_op else "no"
        lines.append(
            f"| `{entry.name}` | {entry.category} | {entry.status} | "
            f"{existing} | {missing} | {families} |"
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
