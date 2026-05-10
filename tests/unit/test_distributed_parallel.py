"""Phase I — DDP / FSDP wrapper tests.

Uses `tessera.testing.mock_collective.MockRankGroup` to simulate multi-rank
execution in-process. No real NCCL/RCCL dependency.
"""

from __future__ import annotations

import numpy as np
import pytest

import tessera as ts
from tessera.testing.mock_collective import MockRankGroup


# ─────────────────────────────────────────────────────────────────────────────
# DDP
# ─────────────────────────────────────────────────────────────────────────────


class TestDDP:
    def test_construct_only_accepts_module(self):
        with pytest.raises(TypeError, match="tessera.nn.Module"):
            ts.distributed.DDP(lambda x: x, mesh_axis="dp")

    def test_forward_is_pass_through(self):
        m = ts.nn.Linear(4, 8, bias=False)
        ddp = ts.distributed.DDP(m)
        x = np.random.randn(2, 4).astype(np.float32)
        np.testing.assert_allclose(ddp(x), m(x))

    def test_sync_grads_means_across_ranks(self):
        # Each rank fills its grad with its rank index; sync_grads should
        # mean-reduce: result on every rank = (0+1+2+3)/4 = 1.5
        m = ts.nn.Linear(4, 4, bias=False)
        ddp = ts.distributed.DDP(m)

        def worker(rank):
            for p in ddp.module.parameters():
                p.grad = np.full(p.shape, float(rank.rank), dtype=np.float32)
            ddp.sync_grads(rank)
            return [p.grad.numpy().copy() for p in ddp.module.parameters()]

        group = MockRankGroup(n=4, mesh_axes={"dp": 4})
        results = group.run(worker)

        # All four ranks should converge on 1.5 = mean(0,1,2,3)
        for grads in results:
            np.testing.assert_allclose(grads[0], 1.5)

    def test_sync_grads_skips_none(self):
        m = ts.nn.Linear(4, 4, bias=True)
        ddp = ts.distributed.DDP(m)

        def worker(rank):
            # Only the weight has a grad; bias.grad is None
            m.weight.grad = np.ones(m.weight.shape, dtype=np.float32) * (rank.rank + 1)
            ddp.sync_grads(rank)
            return m.weight.grad.numpy().copy(), m.bias.grad

        group = MockRankGroup(n=2, mesh_axes={"dp": 2})
        results = group.run(worker)
        for w_grad, b_grad in results:
            np.testing.assert_allclose(w_grad, 1.5)  # mean(1, 2)
            assert b_grad is None  # never populated → still None

    def test_sync_grads_single_rank_is_noop(self):
        m = ts.nn.Linear(4, 4, bias=False)
        ddp = ts.distributed.DDP(m)
        m.weight.grad = np.ones(m.weight.shape, dtype=np.float32) * 7.0
        # Pass a "rank" with world_size=1 — sync_grads should leave grad untouched
        class _SingleRank:
            world_size = 1
            rank = 0
        ddp.sync_grads(_SingleRank())
        np.testing.assert_allclose(m.weight.grad.numpy(), 7.0)


# ─────────────────────────────────────────────────────────────────────────────
# FSDP
# ─────────────────────────────────────────────────────────────────────────────


class TestFSDP:
    """FSDP requires per-rank Module copies (matches torch FSDP's
    one-process-per-rank model). We construct fresh modules inside each
    worker and seed them deterministically so the comparison across ranks is
    well-defined.
    """

    @staticmethod
    def _make_module(seed: int = 0) -> ts.nn.Module:
        np.random.seed(seed)
        m = ts.nn.Linear(8, 4, bias=False)
        # Seed the weight deterministically so per-rank copies start identical
        m.weight._data._data[...] = np.arange(32, dtype=np.float32).reshape(8, 4)
        return m

    def test_shard_drops_non_local_slice(self):
        # 4 ranks, world_size=4. Each rank gets its OWN module copy and
        # shards it; we compare each shard against the expected slice of the
        # canonical weight matrix.
        canonical = np.arange(32, dtype=np.float32).reshape(8, 4)

        def worker(rank):
            m = self._make_module()
            fsdp = ts.distributed.FSDP(m)
            fsdp.shard(rank)
            return m.weight.numpy().copy(), rank.rank

        group = MockRankGroup(n=4, mesh_axes={"dp": 4})
        results = group.run(worker)
        for shard, rank_idx in results:
            assert shard.shape == (2, 4)
            np.testing.assert_allclose(
                shard, canonical[rank_idx * 2:(rank_idx + 1) * 2]
            )

    def test_shard_rejects_indivisible_leading_dim(self):
        def worker(rank):
            m = ts.nn.Linear(7, 4, bias=False)  # 7 not divisible by 4
            fsdp = ts.distributed.FSDP(m)
            try:
                fsdp.shard(rank)
            except ValueError as e:
                return str(e)
            return None

        group = MockRankGroup(n=4, mesh_axes={"dp": 4})
        results = group.run(worker)
        for r in results:
            assert r is not None
            assert "not divisible" in r

    def test_gather_reshard_round_trip(self):
        canonical = np.arange(32, dtype=np.float32).reshape(8, 4)

        def worker(rank):
            m = self._make_module()
            fsdp = ts.distributed.FSDP(m)
            fsdp.shard(rank)
            assert m.weight.numpy().shape == (2, 4)
            fsdp.gather_for_forward(rank)
            assert m.weight.numpy().shape == (8, 4)
            np.testing.assert_allclose(m.weight.numpy(), canonical)
            fsdp.reshard_after_forward(rank)
            assert m.weight.numpy().shape == (2, 4)

        group = MockRankGroup(n=4, mesh_axes={"dp": 4})
        group.run(worker)

    def test_sync_grads_reduce_scatters_to_local_shard(self):
        def worker(rank):
            m = self._make_module()
            fsdp = ts.distributed.FSDP(m)
            fsdp.shard(rank)
            # Pretend we ran forward on the full gathered weight; grad lives
            # at full shape across all ranks (same value 1, 2, 3, 4 per rank).
            m.weight.grad = np.full((8, 4), float(rank.rank + 1), dtype=np.float32)
            fsdp.sync_grads(rank)
            return m.weight.grad.numpy().copy(), rank.rank

        group = MockRankGroup(n=4, mesh_axes={"dp": 4})
        results = group.run(worker)

        # After reduce_scatter+mean: each local 1/4 slice has mean(1,2,3,4)=2.5
        for grad, rank_idx in results:
            assert grad.shape == (2, 4)
            np.testing.assert_allclose(grad, 2.5)


# ─────────────────────────────────────────────────────────────────────────────
# Deferred-items plan, Item 3 — ZeRO stage 3 unification.
#
# `FSDP(stage=3)` and the `ZeRO3` alias both opt into parameter-sharding
# semantics. The Python wrapper today still holds full params per rank
# between gather/reshard pairs (matching FSDP v1); the ZeROConfig
# annotation is what the IR-side `OptimizerShardPass` reads to emit
# `tessera_sr.params_sharded`. Real NCCL all-gather of parameters
# pre-forward lands in Phase G — this test pins the API contract today.
# ─────────────────────────────────────────────────────────────────────────────


class TestZeRO3:
    def _make_module(self):
        m = ts.nn.Linear(4, 4, bias=False)
        m.weight._data._data = np.arange(16, dtype=np.float32).reshape(4, 4) * 0.1
        return m

    def test_fsdp_default_stage_is_2(self):
        m = self._make_module()
        f = ts.distributed.FSDP(m)
        assert f.stage == 2
        cfg = f.zero_config
        assert cfg.stage == 2
        assert cfg.partition_parameters is False

    def test_fsdp_stage3_sets_partition_parameters(self):
        m = self._make_module()
        f = ts.distributed.FSDP(m, stage=3)
        cfg = f.zero_config
        assert cfg.stage == 3
        assert cfg.partition_parameters is True
        assert cfg.partition_gradients is True
        assert cfg.partition_optimizer_states is True

    def test_zero3_alias_equivalent_to_fsdp_stage3(self):
        m1 = self._make_module()
        m2 = self._make_module()
        z3 = ts.distributed.ZeRO3(m1, mesh_axis="dp")
        f3 = ts.distributed.FSDP(m2, mesh_axis="dp", stage=3)
        # Same stage + same flags + same dp_axis.
        assert z3.stage == f3.stage == 3
        assert z3.zero_config.partition_parameters == f3.zero_config.partition_parameters
        assert z3.mesh_axis == f3.mesh_axis == "dp"

    def test_invalid_stage_rejected(self):
        m = self._make_module()
        with pytest.raises(ValueError, match="stage"):
            ts.distributed.FSDP(m, stage=1)
        with pytest.raises(ValueError, match="stage"):
            ts.distributed.FSDP(m, stage=4)

    def test_zero3_is_a_subclass_of_fsdp(self):
        """Type-check users can write `isinstance(wrapper, FSDP)` and
        catch ZeRO3 too — it's the stricter variant, not a parallel
        hierarchy."""
        m = self._make_module()
        z3 = ts.distributed.ZeRO3(m)
        assert isinstance(z3, ts.distributed.FSDP)

    def test_zero3_forward_passthrough(self):
        """Forward is unchanged from FSDP — ZeRO3 is a stage-3-flagging
        wrapper, not a different runtime path (yet — Phase G upgrades
        the gather/reshard cadence)."""
        m = self._make_module()
        z3 = ts.distributed.ZeRO3(m)
        x = np.random.randn(2, 4).astype(np.float32)
        np.testing.assert_allclose(z3(x), m(x))

    def test_zero3_sync_grads_reduces_correctly_across_ranks(self):
        """End-to-end: ZeRO3 + a 4-rank MockRankGroup. Same correctness
        contract as FSDP stage 2 since the v1 Python wrapper doesn't yet
        actually drop full-param storage — but the gradient sync still
        reduce-scatters into the local shard."""
        def worker(rank):
            m = ts.nn.Linear(4, 4, bias=False)
            m.weight._data._data = np.arange(16, dtype=np.float32).reshape(4, 4) * 0.1
            z3 = ts.distributed.ZeRO3(m)
            z3.shard(rank)
            m.weight.grad = np.full((4, 4), float(rank.rank + 1), dtype=np.float32)
            z3.sync_grads(rank)
            return m.weight.grad.numpy().copy()

        group = MockRankGroup(n=4, mesh_axes={"dp": 4})
        results = group.run(worker)
        for grad in results:
            # Each rank's local 1/4 slice has mean(1,2,3,4) = 2.5
            assert grad.shape == (1, 4)
            np.testing.assert_allclose(grad, 2.5)
