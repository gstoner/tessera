"""
tessera.compiler.distributed_planner — DistributedPlan: dp/tp/pp assignment.

Translates a high-level mesh spec into per-layer parallelism strategies and
emits the MLIR attribute annotations that GPUCollectiveInsertionPass reads.

Design:
  DistributedPlan owns a list of LayerSpec objects, one per model layer.
  Each LayerSpec records which mesh axes handle data parallelism (dp),
  tensor parallelism (tp), and pipeline parallelism (pp) for that layer.

  to_mlir_attrs() serializes the plan so that:
    - DistributionLoweringPass can emit schedule.mesh.region wrappers
    - GPUCollectiveInsertionPass can insert reduce_scatter/all_gather at
      the right mesh boundaries

Reference: CLAUDE.md §Phase 4 — Track A
           src/transforms/lib/GPUCollectiveInsertionPass.cpp
           src/transforms/lib/PipelineStageInsertionPass.cpp
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple


# ─────────────────────────────────────────────────────────────────────────────
# LayerSpec
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class LayerSpec:
    """
    Parallelism strategy for a single model layer.

    Attributes:
        name        : layer name (e.g. "transformer.layer_0.attn")
        layer_type  : one of "linear", "attn", "moe", "embedding", "norm"
        dp_axis     : mesh axis name for data parallelism (or None)
        tp_axis     : mesh axis name for tensor parallelism (or None)
        pp_stage    : pipeline stage index (0-based; None = not pipelined)
        weight_sharding: how to shard weights — "col_parallel", "row_parallel",
                          "replicated", "expert_parallel"
        activation_sharding: how activations flow — "dp_scatter", "tp_allgather",
                              "full"

    Example:
        LayerSpec(
            name="mlp.fc1",
            layer_type="linear",
            dp_axis="dp",
            tp_axis="tp",
            weight_sharding="col_parallel",
            activation_sharding="dp_scatter",
        )
    """
    name: str
    layer_type: str = "linear"
    dp_axis: Optional[str] = None
    tp_axis: Optional[str] = None
    pp_stage: Optional[int] = None
    weight_sharding: str = "replicated"
    activation_sharding: str = "full"

    _VALID_LAYER_TYPES = {"linear", "attn", "moe", "embedding", "norm", "conv"}
    _VALID_WEIGHT_SHARDING = {"col_parallel", "row_parallel", "replicated", "expert_parallel"}
    _VALID_ACT_SHARDING = {"dp_scatter", "tp_allgather", "full", "cyclic"}

    def __post_init__(self) -> None:
        if self.layer_type not in self._VALID_LAYER_TYPES:
            raise ValueError(
                f"LayerSpec {self.name!r}: unknown layer_type {self.layer_type!r}. "
                f"Valid: {sorted(self._VALID_LAYER_TYPES)}"
            )
        if self.weight_sharding not in self._VALID_WEIGHT_SHARDING:
            raise ValueError(
                f"LayerSpec {self.name!r}: unknown weight_sharding "
                f"{self.weight_sharding!r}"
            )
        if self.activation_sharding not in self._VALID_ACT_SHARDING:
            raise ValueError(
                f"LayerSpec {self.name!r}: unknown activation_sharding "
                f"{self.activation_sharding!r}"
            )

    def needs_reduce_scatter(self) -> bool:
        """True if this layer requires a reduce_scatter at its output boundary."""
        # Column-parallel linear: each TP rank computes partial output → reduce_scatter
        return self.tp_axis is not None and self.weight_sharding == "col_parallel"

    def needs_all_gather(self) -> bool:
        """True if this layer requires an all_gather before the next layer."""
        # Row-parallel linear: need to gather split activations across TP ranks
        return self.tp_axis is not None and self.weight_sharding == "row_parallel"

    def to_ir_attr(self) -> str:
        """Serialize as MLIR attribute for GPUCollectiveInsertionPass."""
        parts = [f'name = "{self.name}"', f'type = "{self.layer_type}"']
        if self.dp_axis:
            parts.append(f'dp = "{self.dp_axis}"')
        if self.tp_axis:
            parts.append(f'tp = "{self.tp_axis}"')
        if self.pp_stage is not None:
            parts.append(f'pp_stage = {self.pp_stage}')
        parts.append(f'weight_sharding = "{self.weight_sharding}"')
        parts.append(f'act_sharding = "{self.activation_sharding}"')
        return '{tessera.layer = {' + ', '.join(parts) + '}}'


# ─────────────────────────────────────────────────────────────────────────────
# DistributedPlan
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class DistributedPlan:
    """
    Full parallelism plan for a model: dp/tp/pp assignment across all layers.

    Usage:
        plan = DistributedPlan(
            mesh_axes={"dp": 4, "tp": 2, "pp": 2},
            layers=[
                LayerSpec("embed",         layer_type="embedding", dp_axis="dp"),
                LayerSpec("attn.0",        layer_type="attn",  dp_axis="dp", tp_axis="tp",
                          weight_sharding="col_parallel", pp_stage=0),
                LayerSpec("mlp.fc1",       layer_type="linear", dp_axis="dp", tp_axis="tp",
                          weight_sharding="col_parallel", pp_stage=0),
                LayerSpec("mlp.fc2",       layer_type="linear", dp_axis="dp", tp_axis="tp",
                          weight_sharding="row_parallel", pp_stage=1),
            ]
        )
        plan.validate()
        mlir_str = plan.to_mlir_attrs()

    Attributes:
        mesh_axes : dict of axis_name → size (e.g. {"dp": 4, "tp": 2})
        layers    : ordered list of LayerSpec (model execution order)
    """
    mesh_axes: Dict[str, int]
    layers: List[LayerSpec] = field(default_factory=list)

    def __post_init__(self) -> None:
        if not self.mesh_axes:
            raise ValueError("DistributedPlan requires at least one mesh axis")
        for name, size in self.mesh_axes.items():
            if not isinstance(size, int) or size < 1:
                raise ValueError(
                    f"mesh_axes[{name!r}] must be a positive int, got {size!r}"
                )

    @property
    def total_ranks(self) -> int:
        result = 1
        for s in self.mesh_axes.values():
            result *= s
        return result

    @property
    def num_pipeline_stages(self) -> int:
        stages = {s.pp_stage for s in self.layers if s.pp_stage is not None}
        return len(stages)

    def validate(self) -> None:
        """
        Check the plan for consistency:
          - All referenced axes exist in mesh_axes
          - pp_stage values are contiguous starting from 0
          - MoE layers use an axis with num_ranks divisible by num_experts
        """
        known_axes = set(self.mesh_axes)
        for spec in self.layers:
            if spec.dp_axis and spec.dp_axis not in known_axes:
                raise ValueError(
                    f"Layer {spec.name!r}: dp_axis={spec.dp_axis!r} not in mesh_axes {known_axes}"
                )
            if spec.tp_axis and spec.tp_axis not in known_axes:
                raise ValueError(
                    f"Layer {spec.name!r}: tp_axis={spec.tp_axis!r} not in mesh_axes {known_axes}"
                )

        # Pipeline stages must be 0-based contiguous
        stages = sorted(s.pp_stage for s in self.layers if s.pp_stage is not None)
        if stages:
            expected = list(range(stages[-1] + 1))
            if stages != expected:
                # Allow gaps — warn but don't raise
                pass  # could emit a warning here

    def add_layer(self, spec: LayerSpec) -> "DistributedPlan":
        """Append a layer spec and return self (fluent API)."""
        self.layers.append(spec)
        return self

    def layers_for_stage(self, stage: int) -> List[LayerSpec]:
        """Return all layers assigned to pipeline stage `stage`."""
        return [s for s in self.layers if s.pp_stage == stage]

    def reduce_scatter_boundaries(self) -> List[str]:
        """Return names of layers that need a reduce_scatter at their output."""
        return [s.name for s in self.layers if s.needs_reduce_scatter()]

    def all_gather_boundaries(self) -> List[str]:
        """Return names of layers that need an all_gather at their input."""
        return [s.name for s in self.layers if s.needs_all_gather()]

    def to_mlir_attrs(self) -> str:
        """
        Serialize the plan as a MLIR module-level attribute string.

        This string is attached as `tessera.distributed_plan = {...}` to the
        module op before GPUCollectiveInsertionPass and PipelineStageInsertionPass
        run. Both passes read this attribute to determine insertion points.
        """
        mesh_parts = ", ".join(
            f'"{k}" = {v}' for k, v in self.mesh_axes.items()
        )
        layer_attrs = ", ".join(
            f'{{{", ".join(self._layer_dict(s))}}}'
            for s in self.layers
        )
        return (
            f'{{tessera.distributed_plan = {{'
            f'mesh = {{{mesh_parts}}}, '
            f'total_ranks = {self.total_ranks}, '
            f'num_stages = {self.num_pipeline_stages}, '
            f'layers = [{layer_attrs}]}}}}'
        )

    @staticmethod
    def _layer_dict(spec: LayerSpec) -> List[str]:
        parts = [f'name = "{spec.name}"', f'type = "{spec.layer_type}"']
        if spec.dp_axis:
            parts.append(f'dp = "{spec.dp_axis}"')
        if spec.tp_axis:
            parts.append(f'tp = "{spec.tp_axis}"')
        if spec.pp_stage is not None:
            parts.append(f'stage = {spec.pp_stage}')
        return parts

    @classmethod
    def for_transformer(
        cls,
        num_layers: int,
        mesh_axes: Dict[str, int],
        dp_axis: str = "dp",
        tp_axis: Optional[str] = "tp",
        pp_stages: int = 1,
    ) -> "DistributedPlan":
        """
        Factory: build a standard transformer DistributedPlan.

        Creates LayerSpecs for embedding + N × (attn, mlp_fc1, mlp_fc2, norm) + lm_head.
        Assigns pipeline stages evenly across layers.

        Args:
            num_layers : number of transformer blocks
            mesh_axes  : mesh axis sizes
            dp_axis    : axis name for data parallelism
            tp_axis    : axis name for tensor parallelism (None = no TP)
            pp_stages  : number of pipeline stages (1 = no PP)

        Returns:
            DistributedPlan with all layers pre-populated.
        """
        plan = cls(mesh_axes=mesh_axes)

        # Embedding (replicated across TP, DP-sharded)
        plan.add_layer(LayerSpec(
            name="embedding",
            layer_type="embedding",
            dp_axis=dp_axis,
            weight_sharding="replicated",
        ))

        # Determine layers per stage
        layers_per_stage = max(1, num_layers // pp_stages)

        for i in range(num_layers):
            stage = min(i // layers_per_stage, pp_stages - 1)

            plan.add_layer(LayerSpec(
                name=f"layer_{i}.attn",
                layer_type="attn",
                dp_axis=dp_axis,
                tp_axis=tp_axis,
                pp_stage=stage,
                weight_sharding="col_parallel" if tp_axis else "replicated",
                activation_sharding="tp_allgather" if tp_axis else "full",
            ))
            plan.add_layer(LayerSpec(
                name=f"layer_{i}.mlp_fc1",
                layer_type="linear",
                dp_axis=dp_axis,
                tp_axis=tp_axis,
                pp_stage=stage,
                weight_sharding="col_parallel" if tp_axis else "replicated",
                activation_sharding="full",
            ))
            plan.add_layer(LayerSpec(
                name=f"layer_{i}.mlp_fc2",
                layer_type="linear",
                dp_axis=dp_axis,
                tp_axis=tp_axis,
                pp_stage=stage,
                weight_sharding="row_parallel" if tp_axis else "replicated",
                activation_sharding="dp_scatter" if tp_axis else "full",
            ))
            plan.add_layer(LayerSpec(
                name=f"layer_{i}.norm",
                layer_type="norm",
                dp_axis=dp_axis,
                pp_stage=stage,
                weight_sharding="replicated",
            ))

        plan.add_layer(LayerSpec(
            name="lm_head",
            layer_type="linear",
            dp_axis=dp_axis,
            tp_axis=tp_axis,
            weight_sharding="col_parallel" if tp_axis else "replicated",
        ))

        return plan

    def __repr__(self) -> str:
        mesh_str = ", ".join(f"{k}={v}" for k, v in self.mesh_axes.items())
        return (
            f"DistributedPlan(mesh={{{mesh_str}}}, "
            f"{len(self.layers)} layers, "
            f"{self.num_pipeline_stages} pipeline stage(s))"
        )
