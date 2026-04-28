"""
Phase 4 — test_gpu_collective_insertion.py

Tests for GPUCollectiveInsertionPass via DistributedPlan IR annotations.
Validates that the Python plan correctly identifies collective insertion points.
"""
import pytest
from tessera.compiler.distributed_planner import DistributedPlan, LayerSpec


class TestCollectiveBoundaries:
    """Validate Python-layer collective boundary detection."""

    def test_col_parallel_is_reduce_scatter_boundary(self):
        plan = DistributedPlan(mesh_axes={"dp": 4, "tp": 2})
        plan.add_layer(LayerSpec("attn.qkv", "linear",
                                  tp_axis="tp", weight_sharding="col_parallel"))
        assert "attn.qkv" in plan.reduce_scatter_boundaries()

    def test_row_parallel_is_all_gather_boundary(self):
        plan = DistributedPlan(mesh_axes={"dp": 4, "tp": 2})
        plan.add_layer(LayerSpec("attn.proj", "linear",
                                  tp_axis="tp", weight_sharding="row_parallel"))
        assert "attn.proj" in plan.all_gather_boundaries()

    def test_replicated_is_no_collective_boundary(self):
        plan = DistributedPlan(mesh_axes={"dp": 4, "tp": 2})
        plan.add_layer(LayerSpec("norm", "norm", weight_sharding="replicated"))
        assert "norm" not in plan.reduce_scatter_boundaries()
        assert "norm" not in plan.all_gather_boundaries()

    def test_four_layer_mlp_collectives(self):
        """Standard Megatron-style MLP: fc1 col-parallel, fc2 row-parallel."""
        plan = DistributedPlan(mesh_axes={"dp": 4, "tp": 2})
        plan.add_layer(LayerSpec("mlp.fc1", "linear",
                                  dp_axis="dp", tp_axis="tp",
                                  weight_sharding="col_parallel"))
        plan.add_layer(LayerSpec("mlp.fc2", "linear",
                                  dp_axis="dp", tp_axis="tp",
                                  weight_sharding="row_parallel"))
        plan.validate()
        assert plan.reduce_scatter_boundaries() == ["mlp.fc1"]
        assert plan.all_gather_boundaries() == ["mlp.fc2"]

    def test_transformer_plan_has_collectives(self):
        plan = DistributedPlan.for_transformer(
            num_layers=4, mesh_axes={"dp": 4, "tp": 2}
        )
        rs = plan.reduce_scatter_boundaries()
        ag = plan.all_gather_boundaries()
        assert len(rs) > 0, "Expected reduce_scatter boundaries"
        assert len(ag) > 0, "Expected all_gather boundaries"

    def test_transformer_no_tp_has_no_collectives(self):
        plan = DistributedPlan.for_transformer(
            num_layers=4, mesh_axes={"dp": 4}, tp_axis=None
        )
        assert plan.reduce_scatter_boundaries() == []
        assert plan.all_gather_boundaries() == []


class TestCollectiveMLIRAnnotation:
    """Validate that to_mlir_attrs() includes collective insertion hints."""

    def test_mlir_attrs_lists_layers(self):
        plan = DistributedPlan(mesh_axes={"dp": 4, "tp": 2})
        plan.add_layer(LayerSpec("fc1", "linear",
                                  tp_axis="tp", weight_sharding="col_parallel"))
        attr = plan.to_mlir_attrs()
        assert "fc1" in attr

    def test_mlir_attrs_contains_mesh(self):
        plan = DistributedPlan(mesh_axes={"dp": 4, "tp": 2})
        attr = plan.to_mlir_attrs()
        assert "dp" in attr
        assert "tp" in attr

    def test_layer_ir_attr_col_parallel(self):
        s = LayerSpec("fc1", "linear", tp_axis="tp",
                      weight_sharding="col_parallel")
        attr = s.to_ir_attr()
        assert "col_parallel" in attr
        assert "tp" in attr

    def test_layer_ir_attr_row_parallel(self):
        s = LayerSpec("fc2", "linear", tp_axis="tp",
                      weight_sharding="row_parallel")
        attr = s.to_ir_attr()
        assert "row_parallel" in attr
