"""
Phase 5 — test_deployment_manifest.py

Tests for DeploymentManifest — Python-layer mirror of
ExportDeploymentManifestPass output.
"""
import pytest
from tessera.compiler.solver_config import DeploymentManifest


class TestDeploymentManifestBasic:
    def test_default_version(self):
        m = DeploymentManifest()
        assert m.version == "v1.1"

    def test_empty_mesh(self):
        m = DeploymentManifest()
        assert m.mesh == {}

    def test_add_mesh_axis(self):
        m = DeploymentManifest()
        m.add_mesh_axis("dp", 4)
        assert m.mesh["dp"] == 4

    def test_add_mesh_axis_invalid_size(self):
        m = DeploymentManifest()
        with pytest.raises(ValueError):
            m.add_mesh_axis("dp", 0)

    def test_add_mesh_axis_fluent(self):
        m = DeploymentManifest()
        result = m.add_mesh_axis("dp", 4)
        assert result is m

    def test_total_ranks_empty(self):
        m = DeploymentManifest()
        assert m.total_ranks() == 1

    def test_total_ranks_single_axis(self):
        m = DeploymentManifest()
        m.add_mesh_axis("dp", 4)
        assert m.total_ranks() == 4

    def test_total_ranks_two_axes(self):
        m = DeploymentManifest()
        m.add_mesh_axis("dp", 4).add_mesh_axis("tp", 2)
        assert m.total_ranks() == 8


class TestDeploymentManifestCollectives:
    def test_add_collective(self):
        m = DeploymentManifest()
        m.add_collective("reduce_scatter")
        assert "reduce_scatter" in m.collectives

    def test_add_multiple_collectives(self):
        m = DeploymentManifest()
        m.add_collective("reduce_scatter").add_collective("all_gather")
        assert len(m.collectives) == 2

    def test_add_optimizer_shard(self):
        m = DeploymentManifest()
        m.add_optimizer_shard("fc1.weight", "dp")
        assert len(m.optimizer_shards) == 1
        assert "fc1.weight" in m.optimizer_shards[0]
        assert "dp" in m.optimizer_shards[0]

    def test_add_checkpoint(self):
        m = DeploymentManifest()
        m.add_checkpoint("layer_0.attn")
        assert len(m.checkpoints) == 1
        assert "layer_0.attn" in m.checkpoints[0]

    def test_add_checkpoint_with_policy(self):
        m = DeploymentManifest()
        m.add_checkpoint("layer_0.attn", policy="full")
        assert "full" in m.checkpoints[0]


class TestDeploymentManifestSerialization:
    def _full_manifest(self):
        m = DeploymentManifest()
        m.add_mesh_axis("dp", 4).add_mesh_axis("tp", 2)
        m.add_collective("reduce_scatter").add_collective("all_gather")
        m.add_optimizer_shard("fc1", "dp", "zero2")
        m.add_checkpoint("layer_0", "selective")
        return m

    def test_to_json_is_dict(self):
        assert isinstance(self._full_manifest().to_json(), dict)

    def test_to_json_has_version(self):
        j = self._full_manifest().to_json()
        assert j["version"] == "v1.1"

    def test_to_json_has_mesh(self):
        j = self._full_manifest().to_json()
        assert "dp" in j["mesh"]
        assert "tp" in j["mesh"]

    def test_to_json_total_ranks(self):
        j = self._full_manifest().to_json()
        assert j["total_ranks"] == 8

    def test_to_json_collectives(self):
        j = self._full_manifest().to_json()
        assert "reduce_scatter" in j["collectives"]

    def test_to_json_optimizer_shards(self):
        j = self._full_manifest().to_json()
        assert len(j["optimizer_shards"]) == 1

    def test_to_json_checkpoints(self):
        j = self._full_manifest().to_json()
        assert len(j["checkpoints"]) == 1

    def test_to_ir_attr_contains_version(self):
        attr = self._full_manifest().to_ir_attr()
        assert "v1.1" in attr

    def test_to_ir_attr_contains_total_ranks(self):
        attr = self._full_manifest().to_ir_attr()
        assert "total_ranks = 8" in attr

    def test_to_ir_attr_has_prefix(self):
        attr = self._full_manifest().to_ir_attr()
        assert "tessera.deployment_manifest" in attr

    def test_repr_contains_version(self):
        m = self._full_manifest()
        assert "v1.1" in repr(m)
