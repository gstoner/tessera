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


def _existing_contracts(effect: str) -> dict[str, str]:
    effect_rule = "partial" if effect != "pure" else "not_applicable"
    return _contracts(
        math_semantics="partial",
        shape_rule="partial",
        dtype_layout_rule="partial",
        masking_effect_rule=effect_rule,
        lowering_rule="complete",
        backend_kernel="partial",
        tests="partial",
    )


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
    entries: dict[str, PrimitiveCoverage] = {}
    for name, spec in sorted(OP_SPECS.items()):
        entries[name] = PrimitiveCoverage(
            name=name,
            category=spec.lowering,
            status="partial",
            contract_status=_existing_contracts(spec.effect),
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
        entries.setdefault(
            name,
            PrimitiveCoverage(
                name=name,
                category=lowering,
                status="partial",
                contract_status=_existing_contracts(effect),
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
    _planned("dynamic_slice", "tensor_algebra", ("all", "RNN/xLSTM", "Mamba/SSM"), references=("jax.lax",)),
    _planned("dynamic_update_slice", "tensor_algebra", ("all", "Titans/Atlas"), references=("jax.lax",)),
    _planned("scatter_add", "indexing", ("all", "Titans/Atlas", "JEPA"), references=("jax.lax", "aten")),
    _planned("scatter_reduce", "indexing", ("all", "Titans/Atlas"), references=("jax.lax", "aten")),
    _planned("nonzero", "indexing", ("all", "Titans/Atlas"), references=("jax.numpy", "aten")),
    _planned("top_k", "indexing", ("Titans/Atlas", "Megalodon/Griffin"), references=("jax.lax", "aten")),
    _planned("sort", "indexing", ("all", "Titans/Atlas"), references=("jax.lax", "aten")),
    _planned("argsort", "indexing", ("all",), references=("jax.numpy", "aten")),
    _planned("index_update", "indexing", ("all", "Titans/Atlas"), references=("jax.numpy", "aten")),
    _planned("exp", "scalar_math", ("all",), references=("jax.numpy", "aten")),
    _planned("log", "scalar_math", ("all",), references=("jax.numpy", "aten")),
    _planned("sqrt", "scalar_math", ("all",), references=("jax.numpy", "aten")),
    _planned("rsqrt", "scalar_math", ("all",), references=("jax.lax", "aten")),
    _planned("pow", "scalar_math", ("all",), references=("jax.numpy", "aten")),
    _planned("cos", "scalar_math", ("FNet/spectral", "diffusion"), references=("jax.numpy", "aten")),
    _planned("erf", "scalar_math", ("all", "diffusion"), references=("jax.lax", "aten")),
    _planned("scan", "control_flow", ("RNN/xLSTM", "Mamba/SSM", "Megalodon/Griffin"), references=("jax.lax",)),
    _planned("associative_scan", "control_flow", ("Mamba/SSM", "Hyena/FNet/spectral"), references=("jax.lax",)),
    _planned("while_loop", "control_flow", ("RNN/xLSTM", "Titans/Atlas"), references=("jax.lax",)),
    _planned("cond", "control_flow", ("all",), references=("jax.lax",)),
    _planned("switch", "control_flow", ("all",), references=("jax.lax",)),
    _planned("value_and_grad", "transform", ("all",), references=("jax",)),
    _planned("shard_map", "sharding", ("all",), references=("jax.shard_map",)),
    _planned("named_sharding", "sharding", ("all",), references=("jax.sharding",)),
    _planned("tree_flatten", "state_tree", ("all",), references=("jax.tree", "flax.nnx")),
    _planned("tree_map", "state_tree", ("all",), references=("jax.tree", "flax.nnx")),
    _planned("state_filter", "state_tree", ("all", "Titans/Atlas"), references=("flax.nnx",)),
    _planned("rng_key", "rng", ("all", "diffusion", "JEPA"), references=("jax.random",)),
    _planned("rng_split", "rng", ("all", "diffusion", "JEPA"), references=("jax.random",)),
    _planned("rng_fold_in", "rng", ("all", "sharding"), references=("jax.random",)),
    _planned("rng_bernoulli", "rng", ("diffusion", "JEPA"), references=("jax.random",)),
    _planned("rng_categorical", "rng", ("diffusion", "inference"), references=("jax.random",)),
    _planned("rng_permutation", "rng", ("JEPA", "diffusion"), references=("jax.random",)),
    _planned("conv_transpose", "model_layer", ("diffusion", "JEPA"), references=("flax.nnx", "aten")),
    _planned("linear_general", "model_layer", ("Linformer/cosFormer", "JEPA"), references=("flax.nnx",)),
    _planned("lora_linear", "model_layer", ("all",), references=("flax.nnx",)),
    _planned("group_norm", "normalization", ("diffusion", "JEPA"), references=("flax.nnx", "aten")),
    _planned("instance_norm", "normalization", ("diffusion", "JEPA"), references=("flax.nnx", "aten")),
    _planned("weight_norm", "normalization", ("all",), references=("flax.nnx",)),
    _planned("spectral_norm", "normalization", ("diffusion", "Hyena/FNet/spectral"), references=("flax.nnx",)),
    _planned("adaptive_pool", "pooling", ("diffusion", "JEPA"), references=("flax.nnx", "aten")),
    _planned("gru_cell", "recurrent", ("RNN/xLSTM",), references=("flax.nnx", "aten")),
    _planned("simple_rnn_cell", "recurrent", ("RNN/xLSTM",), references=("flax.nnx",)),
    _planned("bidirectional_scan", "recurrent", ("RNN/xLSTM", "JEPA"), references=("flax.nnx", "jax.lax")),
    _planned("memory_read", "memory", ("Titans/Atlas",), references=("flax.nnx",)),
    _planned("memory_write", "memory", ("Titans/Atlas",), references=("flax.nnx",)),
    _planned("memory_evict", "memory", ("Titans/Atlas",), references=("flax.nnx",)),
)


def all_primitive_coverages() -> dict[str, PrimitiveCoverage]:
    entries = _existing_coverage()
    for entry in _PLANNED_ENTRIES:
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
