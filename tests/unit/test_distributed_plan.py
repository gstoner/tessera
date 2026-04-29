"""
Phase 4 — test_distributed_plan.py

Tests for DistributedPlan: LayerSpec validation, plan construction, IR attr
serialization, and the transformer factory.
"""
import pytest
from tessera.compiler.distributed_planner import DistributedPlan, LayerSpec


class TestLayerSpec:
    def test_basic_creation(self):
        s = LayerSpec(name="attn.0", layer_type="attn", dp_axis="dp", tp_axis="tp")
        assert s.name == "attn.0"
        assert s.layer_type == "attn"

    def test_invalid_layer_type_raises(self):
        with pytest.raises(ValueError, match="unknown layer_type"):
            LayerSpec(name="x", layer_type="unknown_type")

    def test_col_parallel_needs_reduce_scatter(self):
        s = LayerSpec(name="fc1", layer_type="linear",
                      tp_axis="tp", weight_sharding="col_parallel")
        assert s.needs_reduce_scatter() is True
        assert s.needs_all_gather() is False

    def test_row_parallel_needs_all_gather(self):
        s = LayerSpec(name="fc2", layer_type="linear",
                      tp_axis="tp", weight_sharding="row_parallel")
        assert s.needs_all_gather() is True
        assert s.needs_reduce_scatter() is False

    def test_replicated_needs_neither(self):
        s = LayerSpec(name="norm", layer_type="norm",
                      weight_sharding="replicated")
        assert s.needs_reduce_scatter() is False
        assert s.needs_all_gather() is False

    def test_to_ir_attr_contains_name(self):
        s = LayerSpec(name="mlp.fc1", layer_type="linear")
        attr = s.to_ir_attr()
        assert "mlp.fc1" in attr

    def test_to_ir_attr_contains_stage(self):
        s = LayerSpec(name="attn", layer_type="attn", pp_stage=2)
        attr = s.to_ir_attr()
        assert "pp_stage = 2" in attr


class TestDistributedPlan:
    def _simple_plan(self):
        return DistributedPlan(mesh_axes={"dp": 4, "tp": 2})

    def test_total_ranks(self):
        plan = DistributedPlan(mesh_axes={"dp": 4, "tp": 2})
        assert plan.total_ranks == 8

    def test_empty_layers(self):
        plan = self._simple_plan()
        assert len(plan.layers) == 0

    def test_add_layer_fluent(self):
        plan = self._simple_plan()
        plan.add_layer(LayerSpec("l0", "linear"))
        plan.add_layer(LayerSpec("l1", "linear"))
        assert len(plan.layers) == 2

    def test_validate_unknown_dp_axis_raises(self):
        plan = DistributedPlan(mesh_axes={"dp": 4})
        plan.add_layer(LayerSpec("l0", "linear", dp_axis="unknown"))
        with pytest.raises(ValueError, match="dp_axis="):
            plan.validate()

    def test_validate_unknown_tp_axis_raises(self):
        plan = DistributedPlan(mesh_axes={"dp": 4})
        plan.add_layer(LayerSpec("l0", "linear", tp_axis="tp"))
        with pytest.raises(ValueError, match="tp_axis="):
            plan.validate()

    def test_validate_known_axes_passes(self):
        plan = DistributedPlan(mesh_axes={"dp": 4, "tp": 2})
        plan.add_layer(LayerSpec("l0", "linear", dp_axis="dp", tp_axis="tp"))
        plan.validate()  # no exception

    def test_reduce_scatter_boundaries(self):
        plan = DistributedPlan(mesh_axes={"dp": 4, "tp": 2})
        plan.add_layer(LayerSpec("fc1", "linear",
                                  tp_axis="tp", weight_sharding="col_parallel"))
        plan.add_layer(LayerSpec("fc2", "linear",
                                  tp_axis="tp", weight_sharding="row_parallel"))
        rs_layers = plan.reduce_scatter_boundaries()
        assert "fc1" in rs_layers
        assert "fc2" not in rs_layers

    def test_all_gather_boundaries(self):
        plan = DistributedPlan(mesh_axes={"dp": 4, "tp": 2})
        plan.add_layer(LayerSpec("fc2", "linear",
                                  tp_axis="tp", weight_sharding="row_parallel"))
        ag_layers = plan.all_gather_boundaries()
        assert "fc2" in ag_layers

    def test_num_pipeline_stages_no_pp(self):
        plan = DistributedPlan(mesh_axes={"dp": 4})
        plan.add_layer(LayerSpec("l0", "linear"))
        assert plan.num_pipeline_stages == 0

    def test_num_pipeline_stages_with_pp(self):
        plan = DistributedPlan(mesh_axes={"dp": 4, "pp": 2})
        plan.add_layer(LayerSpec("l0", "linear", pp_stage=0))
        plan.add_layer(LayerSpec("l1", "linear", pp_stage=1))
        assert plan.num_pipeline_stages == 2

    def test_layers_for_stage(self):
        plan = DistributedPlan(mesh_axes={"dp": 4, "pp": 2})
        plan.add_layer(LayerSpec("l0", "linear", pp_stage=0))
        plan.add_layer(LayerSpec("l1", "attn",   pp_stage=0))
        plan.add_layer(LayerSpec("l2", "linear", pp_stage=1))
        stage0 = plan.layers_for_stage(0)
        assert len(stage0) == 2
        assert all(s.pp_stage == 0 for s in stage0)

    def test_to_mlir_attrs_is_string(self):
        plan = DistributedPlan(mesh_axes={"dp": 4, "tp": 2})
        plan.add_layer(LayerSpec("l0", "linear", dp_axis="dp"))
        attr = plan.to_mlir_attrs()
        assert isinstance(attr, str)
        assert "tessera.distributed_plan" in attr

    def test_to_mlir_attrs_contains_total_ranks(self):
        plan = DistributedPlan(mesh_axes={"dp": 4, "tp": 2})
        attr = plan.to_mlir_attrs()
        assert "total_ranks = 8" in attr


class TestDistributedPlanFactory:
    def test_transformer_factory_layer_count(self):
        plan = DistributedPlan.for_transformer(
            num_layers=4,
            mesh_axes={"dp": 4, "tp": 2},
        )
        # 1 embedding + 4 × (attn + fc1 + fc2 + norm) + 1 lm_head = 18
        assert len(plan.layers) == 1 + 4 * 4 + 1

    def test_transformer_factory_has_embedding(self):
        plan = DistributedPlan.for_transformer(2, {"dp": 4})
        assert any(s.name == "embedding" for s in plan.layers)

    def test_transformer_factory_has_lm_head(self):
        plan = DistributedPlan.for_transformer(2, {"dp": 4})
        assert any(s.name == "lm_head" for s in plan.layers)

    def test_transformer_factory_pp_stages(self):
        plan = DistributedPlan.for_transformer(
            num_layers=4, mesh_axes={"dp": 4, "pp": 2}, pp_stages=2
        )
        plan.validate()
        assert plan.num_pipeline_stages == 2

    def test_transformer_factory_no_tp(self):
        plan = DistributedPlan.for_transformer(
            num_layers=2, mesh_axes={"dp": 4}, tp_axis=None
        )
        # With no TP, no col/row-parallel sharding → no reduce_scatter needed
        assert plan.reduce_scatter_boundaries() == []
