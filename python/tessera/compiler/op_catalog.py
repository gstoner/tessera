"""Canonical Tessera frontend operator catalog.

This module is intentionally dependency-light so it can be shared by the
Python AST frontend, textual frontend, effect inference, and reference CPU
lowering without creating import cycles.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class OpSpec:
    public_name: str
    graph_name: str
    min_arity: int
    max_arity: int
    effect: str = "pure"
    lowering: str = "elementwise"

    def valid_arity(self, arity: int) -> bool:
        return self.min_arity <= arity <= self.max_arity


_SPECS = [
    OpSpec("gemm", "tessera.gemm", 2, 2, lowering="loop_nest"),
    OpSpec("matmul", "tessera.matmul", 2, 2, lowering="loop_nest"),
    OpSpec("conv2d", "tessera.conv2d", 2, 4, lowering="stencil"),
    OpSpec("layer_norm", "tessera.layer_norm", 1, 1, lowering="normalization"),
    OpSpec("softmax", "tessera.softmax", 1, 1, lowering="stable_reduction"),
    OpSpec("softmax_safe", "tessera.softmax_safe", 1, 1, lowering="stable_reduction"),
    OpSpec("gelu", "tessera.gelu", 1, 1),
    OpSpec("relu", "tessera.relu", 1, 1),
    OpSpec("sigmoid", "tessera.sigmoid", 1, 1),
    OpSpec("sin", "tessera.sin", 1, 1),
    OpSpec("adam", "tessera.adam", 4, 4, lowering="functional_optimizer_step"),
    OpSpec("transpose", "tessera.transpose", 1, 1, lowering="layout_transform"),
    OpSpec("cast", "tessera.cast", 1, 1, lowering="layout_transform"),
    OpSpec("dropout", "tessera.dropout", 1, 1, effect="random", lowering="random_mask"),
    OpSpec("flash_attn", "tessera.flash_attn", 3, 3, effect="state", lowering="attention"),
    OpSpec("all_reduce", "tessera.all_reduce", 1, 1, effect="collective", lowering="collective"),
    OpSpec("reduce_scatter", "tessera.reduce_scatter", 1, 1, effect="collective", lowering="collective"),
    OpSpec("all_gather", "tessera.all_gather", 1, 1, effect="collective", lowering="collective"),
    OpSpec("fused_epilogue", "tessera.fused_epilogue", 1, 3, lowering="fused_epilogue"),
    OpSpec("fft", "tessera.fft", 1, 1, lowering="spectral"),
    OpSpec("ifft", "tessera.ifft", 1, 1, lowering="spectral"),
    OpSpec("rfft", "tessera.rfft", 1, 1, lowering="spectral"),
    OpSpec("irfft", "tessera.irfft", 1, 1, lowering="spectral"),
    OpSpec("dct", "tessera.dct", 1, 1, lowering="spectral"),
    OpSpec("spectral_conv", "tessera.spectral_conv", 2, 2, lowering="spectral"),
    OpSpec("rmsnorm_safe", "tessera.rmsnorm_safe", 1, 1, lowering="normalization"),
    OpSpec("kv_cache_append", "tessera.kv_cache.append", 3, 3, effect="state", lowering="state_update"),
    OpSpec("kv_cache_prune", "tessera.kv_cache.prune", 1, 1, effect="state", lowering="state_update"),
]

OP_SPECS: dict[str, OpSpec] = {spec.public_name: spec for spec in _SPECS}
GRAPH_OP_TO_SPEC: dict[str, OpSpec] = {spec.graph_name: spec for spec in _SPECS}
GRAPH_OP_MAP: dict[str, str] = {spec.public_name: spec.graph_name for spec in _SPECS}
SUPPORTED_CPU_OPS: frozenset[str] = frozenset(GRAPH_OP_TO_SPEC)


def normalize_op_name(name: str) -> str:
    """Return the public Tessera op name from a bare or qualified call name."""

    if name.startswith("tessera.ops."):
        return name.removeprefix("tessera.ops.")
    if name.startswith("ts.ops."):
        return name.removeprefix("ts.ops.")
    if name.startswith("ops."):
        return name.removeprefix("ops.")
    if name.startswith("op."):
        return name.removeprefix("op.")
    if name.startswith("tessera."):
        tail = name.removeprefix("tessera.")
        if tail.startswith("kv_cache."):
            return tail.replace(".", "_")
        return tail
    return name


def get_op_spec(name: str) -> Optional[OpSpec]:
    return OP_SPECS.get(normalize_op_name(name))


def graph_name_for(name: str) -> Optional[str]:
    spec = get_op_spec(name)
    return spec.graph_name if spec is not None else None


__all__ = [
    "GRAPH_OP_MAP",
    "GRAPH_OP_TO_SPEC",
    "OP_SPECS",
    "SUPPORTED_CPU_OPS",
    "OpSpec",
    "get_op_spec",
    "graph_name_for",
    "normalize_op_name",
]
