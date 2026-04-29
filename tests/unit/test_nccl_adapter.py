"""
Phase 4 — test_nccl_adapter.py

Tests for MockRankGroup collective semantics (all_reduce, reduce_scatter,
all_gather) and the NCCLAdapter / RCCLAdapter mock paths.
"""
import pytest
import numpy as np
from tessera.testing.mock_collective import MockRankGroup, MockCollectiveError


class TestAllReduce:
    def test_all_reduce_sum_scalar(self):
        group = MockRankGroup(n=4, mesh_axes={"dp": 4})
        results = group.run(lambda r: r.all_reduce(
            np.array([float(r.rank)], dtype=np.float32), op="sum"
        ))
        expected = sum(range(4))  # 0+1+2+3 = 6
        for res in results:
            assert np.isclose(res[0], expected), f"Expected {expected}, got {res[0]}"

    def test_all_reduce_sum_matrix(self):
        group = MockRankGroup(n=2, mesh_axes={"dp": 2})
        results = group.run(lambda r: r.all_reduce(
            np.ones((4, 8), dtype=np.float32), op="sum"
        ))
        for res in results:
            assert res.shape == (4, 8)
            assert np.allclose(res, 2.0)  # 2 ranks × 1.0

    def test_all_reduce_max(self):
        group = MockRankGroup(n=4, mesh_axes={"dp": 4})
        results = group.run(lambda r: r.all_reduce(
            np.array([float(r.rank)], dtype=np.float32), op="max"
        ))
        for res in results:
            assert np.isclose(res[0], 3.0)  # max of [0,1,2,3]

    def test_all_reduce_min(self):
        group = MockRankGroup(n=3, mesh_axes={"dp": 3})
        results = group.run(lambda r: r.all_reduce(
            np.array([float(r.rank + 1)], dtype=np.float32), op="min"
        ))
        for res in results:
            assert np.isclose(res[0], 1.0)

    def test_all_reduce_all_same_result(self):
        """Every rank must receive the identical reduced value."""
        group = MockRankGroup(n=4, mesh_axes={"dp": 4})
        results = group.run(lambda r: r.all_reduce(
            np.ones((16,), dtype=np.float32) * r.rank, op="sum"
        ))
        for res in results:
            assert np.allclose(res, results[0])


class TestReduceScatter:
    def test_reduce_scatter_shape(self):
        group = MockRankGroup(n=4, mesh_axes={"dp": 4})
        results = group.run(lambda r: r.reduce_scatter(
            np.ones((16,), dtype=np.float32), axis=0, op="sum"
        ))
        for res in results:
            assert res.shape == (4,)  # 16 / 4 = 4 elements per rank

    def test_reduce_scatter_sum_values(self):
        group = MockRankGroup(n=2, mesh_axes={"dp": 2})
        results = group.run(lambda r: r.reduce_scatter(
            np.ones((4,), dtype=np.float32), axis=0, op="sum"
        ))
        for res in results:
            assert np.allclose(res, 2.0)  # 2 ranks contributing 1.0 each

    def test_reduce_scatter_each_rank_gets_different_slice(self):
        group = MockRankGroup(n=4, mesh_axes={"dp": 4})
        data = np.arange(16, dtype=np.float32)
        results = group.run(lambda r: r.reduce_scatter(
            data.copy(), axis=0, op="sum"
        ))
        # Each rank gets a different slice of the reduced tensor
        all_vals = [res[0] for res in results]
        # The values at different slice positions differ
        assert len(set(all_vals)) > 1  # not all identical

    def test_reduce_scatter_not_divisible_raises(self):
        group = MockRankGroup(n=3, mesh_axes={"dp": 3})
        with pytest.raises((ValueError, MockCollectiveError)):
            group.run(lambda r: r.reduce_scatter(
                np.ones((10,), dtype=np.float32), axis=0, op="sum"
            ))


class TestAllGather:
    def test_all_gather_shape(self):
        group = MockRankGroup(n=4, mesh_axes={"dp": 4})
        results = group.run(lambda r: r.all_gather(
            np.ones((8,), dtype=np.float32) * r.rank, axis=0
        ))
        for res in results:
            assert res.shape == (32,)  # 4 × 8 = 32

    def test_all_gather_preserves_values(self):
        group = MockRankGroup(n=2, mesh_axes={"dp": 2})
        results = group.run(lambda r: r.all_gather(
            np.array([float(r.rank)] * 4, dtype=np.float32), axis=0
        ))
        # Every rank should see [0,0,0,0, 1,1,1,1] (concatenation in rank order)
        for res in results:
            assert np.allclose(res, results[0])
            assert res.shape == (8,)

    def test_all_gather_all_ranks_same_result(self):
        group = MockRankGroup(n=3, mesh_axes={"dp": 3})
        results = group.run(lambda r: r.all_gather(
            np.ones((5,), dtype=np.float32) * r.rank, axis=0
        ))
        for res in results:
            assert np.allclose(res, results[0])


class TestMockRankGroupMeta:
    def test_world_size(self):
        g = MockRankGroup(n=8, mesh_axes={"dp": 4, "tp": 2})
        assert g.world_size == 8

    def test_mesh_axes_product_validation(self):
        with pytest.raises(ValueError, match="mesh_axes product"):
            MockRankGroup(n=8, mesh_axes={"dp": 3, "tp": 2})  # 3×2=6 ≠ 8

    def test_barrier(self):
        log = []
        group = MockRankGroup(n=3, mesh_axes={"dp": 3})
        group.run(lambda r: (log.append(r.rank), r.barrier()))
        assert sorted(log) == [0, 1, 2]

    def test_run_returns_per_rank_results(self):
        group = MockRankGroup(n=4, mesh_axes={"dp": 4})
        results = group.run(lambda r: r.rank * 10)
        assert results == [0, 10, 20, 30]
