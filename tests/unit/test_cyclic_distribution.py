"""
Phase 4 — test_cyclic_distribution.py

Tests for Cyclic distribution: make_shard_spec() and DistributedArray.parts()
round-robin slicing.
"""
import pytest
import numpy as np
import tessera
from tessera.distributed.domain import Cyclic, Block, Rect
from tessera.distributed.shard import ShardSpec
from tessera.distributed.array import DistributedArray


class TestCyclicMakeShardSpec:
    def test_cyclic_returns_shard_spec(self):
        dist = Cyclic(mesh_axes=("dp",))
        spec = dist.make_shard_spec(Rect((8, 64)))
        assert isinstance(spec, ShardSpec)

    def test_cyclic_sets_cyclic_flag(self):
        dist = Cyclic(mesh_axes=("dp",))
        spec = dist.make_shard_spec(Rect((8, 64)))
        assert spec.cyclic is True

    def test_cyclic_partition_dim_zero(self):
        dist = Cyclic(mesh_axes=("dp",))
        spec = dist.make_shard_spec(Rect((8, 64)))
        assert spec.partition == (0,)
        assert spec.mesh_axes == ("dp",)

    def test_cyclic_two_axes(self):
        dist = Cyclic(mesh_axes=("dp", "tp"))
        spec = dist.make_shard_spec(Rect((8, 32, 64)))
        assert spec.partition == (0, 1)
        assert spec.mesh_axes == ("dp", "tp")
        assert spec.cyclic is True

    def test_cyclic_not_replicated(self):
        dist = Cyclic(mesh_axes=("dp",))
        spec = dist.make_shard_spec(Rect((8, 64)))
        assert spec.replicated is False

    def test_cyclic_ir_attr_contains_kind(self):
        dist = Cyclic(mesh_axes=("dp",))
        spec = dist.make_shard_spec(Rect((8, 64)))
        attr = spec.to_ir_attr()
        assert 'kind = "cyclic"' in attr

    def test_block_ir_attr_is_block(self):
        """Sanity: Block shard spec still says 'block', not 'cyclic'."""
        dist = Block(mesh_axes=("dp",))
        spec = dist.make_shard_spec(Rect((8, 64)))
        assert 'kind = "block"' in spec.to_ir_attr()

    def test_cyclic_repr(self):
        dist = Cyclic(mesh_axes=("dp",))
        assert "Cyclic" in repr(dist)

    def test_cyclic_requires_nonempty_axes(self):
        with pytest.raises((ValueError, TypeError)):
            Cyclic(mesh_axes=())


class TestCyclicParts:
    """Test DistributedArray.parts() with Cyclic distribution."""

    def _make_array(self, shape, num_ranks):
        dist = Cyclic(mesh_axes=("dp",))
        arr = DistributedArray.from_domain(
            Rect(shape), dtype="fp32", distribution=dist
        )
        arr._bind_mesh(tessera.distributed.shard.MeshSpec({"dp": num_ranks}))
        return arr

    def test_parts_count_equals_num_ranks(self):
        arr = self._make_array((8, 16), num_ranks=4)
        parts = arr.parts("dp")
        assert len(parts) == 4

    def test_parts_round_robin_rows(self):
        """rank k gets rows k, k+4, k+8, … for dp=4."""
        data = np.arange(16, dtype=np.float32).reshape(16, 1)
        dist = Cyclic(mesh_axes=("dp",))
        arr = DistributedArray(data=data, dtype="fp32",
                               shard_spec=dist.make_shard_spec(Rect((16, 1))))
        import tessera.distributed.shard as shard_mod
        arr._bind_mesh(shard_mod.MeshSpec({"dp": 4}))

        parts = arr.parts("dp")
        # rank 0 should have rows 0, 4, 8, 12
        assert np.allclose(parts[0]._data.flatten(), [0, 4, 8, 12])
        # rank 1 should have rows 1, 5, 9, 13
        assert np.allclose(parts[1]._data.flatten(), [1, 5, 9, 13])
        # rank 2 should have rows 2, 6, 10, 14
        assert np.allclose(parts[2]._data.flatten(), [2, 6, 10, 14])
        # rank 3 should have rows 3, 7, 11, 15
        assert np.allclose(parts[3]._data.flatten(), [3, 7, 11, 15])

    def test_cyclic_vs_block_different_shards(self):
        """Cyclic and Block parts() must differ for the same data."""
        data = np.arange(8, dtype=np.float32).reshape(8, 1)
        import tessera.distributed.shard as shard_mod
        mesh = shard_mod.MeshSpec({"dp": 2})

        block_arr = DistributedArray(
            data=data, dtype="fp32",
            shard_spec=Block(mesh_axes=("dp",)).make_shard_spec(Rect((8, 1)))
        )
        block_arr._bind_mesh(mesh)

        cyc_arr = DistributedArray(
            data=data, dtype="fp32",
            shard_spec=Cyclic(mesh_axes=("dp",)).make_shard_spec(Rect((8, 1)))
        )
        cyc_arr._bind_mesh(mesh)

        block_parts = block_arr.parts("dp")
        cyc_parts = cyc_arr.parts("dp")
        # Block rank 0 gets rows 0-3; Cyclic rank 0 gets rows 0, 2, 4, 6
        assert not np.allclose(block_parts[0]._data, cyc_parts[0]._data)

    def test_all_elements_covered(self):
        """Every element appears in exactly one cyclic shard."""
        data = np.arange(12, dtype=np.float32).reshape(12, 1)
        dist = Cyclic(mesh_axes=("dp",))
        arr = DistributedArray(data=data, dtype="fp32",
                               shard_spec=dist.make_shard_spec(Rect((12, 1))))
        import tessera.distributed.shard as shard_mod
        arr._bind_mesh(shard_mod.MeshSpec({"dp": 3}))

        parts = arr.parts("dp")
        all_vals = np.sort(np.concatenate([p._data.flatten() for p in parts]))
        assert np.allclose(all_vals, np.arange(12, dtype=np.float32))
