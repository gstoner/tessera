"""Sprint D — memory architecture surface tests.

Locks the three shipped pieces:

  1. ``tessera.sharding.MemoryShardSpec`` — content-addressed
     partitioning of memory banks (KEY_HASH / BUCKET / BLOCK / REPLICATED)
     with deterministic shard_owner() resolution.
  2. ``tessera.cache.MemoryStateHandle`` — persistent state ABI for
     Titans/Atlas-style banks: functional read, append/evict mutation,
     ``checkpoint`` / ``restore`` round-trip on the
     STATE_COLLECTION_SPECS["memory_state"] schema.
  3. ``tessera.memory.vmap_axis_for`` — per-primitive vmap-axis registry
     that flags the bank arg as ``"state"`` so vmap never replicates or
     splits it.

And the primitive_coverage flips:
  - memory_read / memory_write / memory_evict now `complete` on
    batching_rule, transpose_rule (where applicable), and sharding_rule.
"""

from __future__ import annotations

import numpy as np
import pytest

from tessera.cache import MemoryStateHandle
from tessera.compiler.primitive_coverage import coverage_for, is_contract_closed
from tessera.memory import (
    register_vmap_axis,
    vmap_axis_for,
)
from tessera.sharding import (
    MemoryMode,
    MemoryShardSpec,
    NamedMesh,
    get_memory_bucket_fn,
    register_memory_bucket_fn,
)


# ──────────────────────────────────────────────────────────────────────────
#                      MemoryShardSpec
# ──────────────────────────────────────────────────────────────────────────

class TestMemoryShardSpec:
    def test_default_mode_is_key_hash(self):
        spec = MemoryShardSpec(mesh_axis="memory")
        assert spec.mode == MemoryMode.KEY_HASH
        assert spec.eviction == "score"
        assert spec.persistence == "persistent"
        assert spec.bucket_fn is None

    def test_unknown_mode_rejected(self):
        with pytest.raises(ValueError, match="mode must be one of"):
            MemoryShardSpec(mesh_axis="memory", mode="bogus")

    def test_unknown_eviction_rejected(self):
        with pytest.raises(ValueError, match="eviction must be"):
            MemoryShardSpec(mesh_axis="memory", eviction="bogus")

    def test_unknown_persistence_rejected(self):
        with pytest.raises(ValueError, match="persistence must be"):
            MemoryShardSpec(mesh_axis="memory", persistence="bogus")

    def test_bucket_mode_requires_bucket_fn(self):
        with pytest.raises(ValueError, match="requires a `bucket_fn=`"):
            MemoryShardSpec(mesh_axis="memory", mode=MemoryMode.BUCKET)

    def test_non_bucket_mode_rejects_bucket_fn(self):
        with pytest.raises(ValueError, match="only meaningful when mode='bucket'"):
            MemoryShardSpec(
                mesh_axis="memory",
                mode=MemoryMode.KEY_HASH,
                bucket_fn="my_fn",
            )

    def test_validate_against_mesh_axis_membership(self):
        mesh = NamedMesh(axis_names=("memory", "tp"), shape=(4, 2))
        spec = MemoryShardSpec(mesh_axis="memory")
        spec.validate_against(mesh)  # no raise
        bad = MemoryShardSpec(mesh_axis="bogus")
        with pytest.raises(ValueError, match="not in mesh axes"):
            bad.validate_against(mesh)


class TestKeyHashSharding:
    def test_key_hash_is_deterministic(self):
        mesh = NamedMesh(axis_names=("memory",), shape=(4,))
        spec = MemoryShardSpec(mesh_axis="memory", mode=MemoryMode.KEY_HASH)
        key = np.array([1.5, 2.5, 3.5], dtype=np.float32)
        a = spec.shard_owner(key, mesh)
        b = spec.shard_owner(key, mesh)
        assert a == b
        assert 0 <= a < 4

    def test_distinct_keys_distribute(self):
        """Hash distribution should spread distinct keys across shards."""
        mesh = NamedMesh(axis_names=("memory",), shape=(8,))
        spec = MemoryShardSpec(mesh_axis="memory", mode=MemoryMode.KEY_HASH)
        rng = np.random.default_rng(0)
        seen = set()
        for _ in range(256):
            key = rng.standard_normal(8).astype(np.float32)
            seen.add(spec.shard_owner(key, mesh))
        # On 256 distinct keys × 8 shards, we should see most/all shards.
        assert len(seen) >= 6, f"key_hash too clumpy: only {len(seen)}/8 shards used"

    def test_replicated_always_returns_zero(self):
        mesh = NamedMesh(axis_names=("memory",), shape=(4,))
        spec = MemoryShardSpec(mesh_axis="memory", mode=MemoryMode.REPLICATED)
        assert spec.shard_owner(np.array([1.0]), mesh) == 0


class TestBucketSharding:
    def test_bucket_fn_registration(self):
        def parity(key, n):
            return int(key[0] >= 0)

        register_memory_bucket_fn("parity_test", parity)
        assert get_memory_bucket_fn("parity_test") is parity

    def test_bucket_dispatch_uses_registered_fn(self):
        def by_first(key, n):
            return int(abs(key[0]) * 1000) % n

        register_memory_bucket_fn("by_first_test", by_first)
        mesh = NamedMesh(axis_names=("memory",), shape=(8,))
        spec = MemoryShardSpec(
            mesh_axis="memory",
            mode=MemoryMode.BUCKET,
            bucket_fn="by_first_test",
        )
        key = np.array([0.005], dtype=np.float32)
        # Derive the expectation from the registered fn rather than hard-coding
        # an arithmetic result: 0.005 is not exactly representable in float32
        # (it rounds to 0.00499999...), so int(abs(key[0]) * 1000) truncates to
        # 4, not 5. The point of this test is that dispatch routes through the
        # registered bucket fn — assert exactly that.
        expected = by_first(key, mesh.shape[0])
        assert spec.shard_owner(key, mesh) == expected

    def test_unregistered_bucket_fn_errors(self):
        mesh = NamedMesh(axis_names=("memory",), shape=(4,))
        spec = MemoryShardSpec(
            mesh_axis="memory",
            mode=MemoryMode.BUCKET,
            bucket_fn="never_registered",
        )
        with pytest.raises(ValueError, match="not.*registered"):
            spec.shard_owner(np.array([1.0]), mesh)


# ──────────────────────────────────────────────────────────────────────────
#                      MemoryStateHandle
# ──────────────────────────────────────────────────────────────────────────

class TestMemoryStateHandle:
    def _make_handle(self, **kw):
        return MemoryStateHandle(capacity=8, key_dim=4, value_dim=(6,), **kw)

    def test_construction_defaults(self):
        h = self._make_handle()
        assert h.capacity == 8
        assert h.key_dim == 4
        assert h.value_dim == (6,)
        assert h.dtype == "fp32"
        assert h.size == 0
        assert not h.is_full

    def test_dtype_alias_canonicalizes(self):
        h = MemoryStateHandle(capacity=4, key_dim=2, value_dim=(2,), dtype="f32")
        assert h.dtype == "fp32"

    def test_write_grows_size(self):
        h = self._make_handle()
        keys = np.random.randn(3, 4).astype(np.float32)
        vals = np.random.randn(3, 6).astype(np.float32)
        h.write(keys, vals)
        assert h.size == 3
        np.testing.assert_allclose(h.keys, keys, atol=1e-6)
        np.testing.assert_allclose(h.values, vals, atol=1e-6)

    def test_write_triggers_score_eviction_when_full(self):
        h = self._make_handle()
        keys = np.random.randn(8, 4).astype(np.float32)
        vals = np.random.randn(8, 6).astype(np.float32)
        scores = np.arange(8.0).astype(np.float32)
        h.write(keys, vals, scores=scores)
        assert h.is_full

        # Write 2 more — the 2 lowest scores (0.0, 1.0) should be evicted.
        new_keys = np.random.randn(2, 4).astype(np.float32)
        new_vals = np.random.randn(2, 6).astype(np.float32)
        new_scores = np.array([100.0, 200.0], dtype=np.float32)
        h.write(new_keys, new_vals, scores=new_scores)
        assert h.size == 8  # capped at capacity

        # New writes should be present; old low-score entries gone.
        # (We don't lock exact row positions — the eviction shifts.)
        all_scores = h.metadata["score"]
        assert 100.0 in all_scores or 200.0 in all_scores

    def test_evict_n(self):
        h = self._make_handle()
        keys = np.random.randn(5, 4).astype(np.float32)
        vals = np.random.randn(5, 6).astype(np.float32)
        scores = np.arange(5.0).astype(np.float32)
        h.write(keys, vals, scores=scores)
        assert h.size == 5
        h.evict(n=2)
        assert h.size == 3

    def test_fifo_eviction(self):
        h = MemoryStateHandle(capacity=4, key_dim=2, value_dim=(2,), eviction="fifo")
        for i in range(6):
            k = np.full((1, 2), i, dtype=np.float32)
            v = np.full((1, 2), i, dtype=np.float32)
            h.write(k, v)
        # FIFO with capacity 4 should retain the last 4 entries (2..5).
        live = h.keys[:, 0]
        np.testing.assert_array_equal(np.sort(live), np.array([2.0, 3.0, 4.0, 5.0]))

    def test_read_via_handle(self):
        h = self._make_handle()
        keys = np.eye(4, dtype=np.float32) * 3  # well-separated keys
        vals = np.arange(4 * 6).reshape(4, 6).astype(np.float32)
        h.write(keys, vals)
        query = np.array([3.0, 0.0, 0.0, 0.0], dtype=np.float32)
        values, indices, weights, scores = h.read(query, top_k=1, temperature=0.01)
        # With well-separated keys + low temperature, the first row wins.
        assert indices.flatten()[0] == 0

    def test_clone_is_deep(self):
        h = self._make_handle()
        h.write(np.ones((1, 4), dtype=np.float32), np.ones((1, 6), dtype=np.float32))
        h2 = h.clone()
        h.write(np.zeros((1, 4), dtype=np.float32), np.zeros((1, 6), dtype=np.float32))
        assert h.size == 2
        assert h2.size == 1  # the clone didn't see the second write

    def test_checkpoint_restore_round_trip(self):
        h = self._make_handle()
        keys = np.random.randn(5, 4).astype(np.float32)
        vals = np.random.randn(5, 6).astype(np.float32)
        scores = np.random.randn(5).astype(np.float32)
        h.write(keys, vals, scores=scores, step=42)
        sd = h.checkpoint()

        h2 = MemoryStateHandle.restore(sd)
        assert h2.size == 5
        np.testing.assert_allclose(h2.keys, h.keys, atol=1e-6)
        np.testing.assert_allclose(h2.values, h.values, atol=1e-6)
        np.testing.assert_allclose(
            h2.metadata["score"], h.metadata["score"], atol=1e-6
        )

    def test_checkpoint_round_trip_with_shard_spec(self):
        spec = MemoryShardSpec(mesh_axis="memory", mode=MemoryMode.KEY_HASH)
        h = MemoryStateHandle(
            capacity=4, key_dim=2, value_dim=(2,), shard_spec=spec,
        )
        sd = h.checkpoint()
        h2 = MemoryStateHandle.restore(sd)
        assert h2.shard_spec is not None
        assert h2.shard_spec.mesh_axis == "memory"
        assert h2.shard_spec.mode == MemoryMode.KEY_HASH

    def test_shard_for_key(self):
        mesh = NamedMesh(axis_names=("memory",), shape=(4,))
        spec = MemoryShardSpec(mesh_axis="memory", mode=MemoryMode.KEY_HASH)
        h = MemoryStateHandle(capacity=4, key_dim=2, value_dim=(2,), shard_spec=spec)
        key = np.array([1.5, 2.5], dtype=np.float32)
        owner = h.shard_for_key(key, mesh)
        assert 0 <= owner < 4

    def test_shard_for_key_returns_zero_without_spec(self):
        h = MemoryStateHandle(capacity=4, key_dim=2, value_dim=(2,))
        mesh = NamedMesh(axis_names=("memory",), shape=(4,))
        assert h.shard_for_key(np.array([1.0, 2.0]), mesh) == 0


# ──────────────────────────────────────────────────────────────────────────
#                       vmap_axis registry
# ──────────────────────────────────────────────────────────────────────────

class TestVmapAxisRegistry:
    def test_memory_read_marks_state_arg(self):
        axes = vmap_axis_for("memory_read")
        assert axes is not None
        assert axes[0] == "state"
        assert axes[1] == 0

    def test_memory_write_marks_state_arg(self):
        axes = vmap_axis_for("memory_write")
        assert axes is not None
        assert axes[0] == "state"
        # Remaining slots batched at axis 0
        for a in axes[1:]:
            assert a == 0

    def test_memory_evict_marks_state_and_unbatched(self):
        axes = vmap_axis_for("memory_evict")
        assert axes is not None
        assert axes[0] == "state"
        assert axes[1] is None

    def test_unknown_op_returns_none(self):
        assert vmap_axis_for("totally_made_up") is None

    def test_register_overwrites(self):
        register_vmap_axis("memory_read_test_only", ("state", 1))
        assert vmap_axis_for("memory_read_test_only") == ("state", 1)
        register_vmap_axis("memory_read_test_only", ("state", 2))
        assert vmap_axis_for("memory_read_test_only") == ("state", 2)


# ──────────────────────────────────────────────────────────────────────────
#               Registry — memory primitives fully classified
# ──────────────────────────────────────────────────────────────────────────

class TestRegistryMemoryComplete:
    """Sprint D promoted memory_read/write/evict on batching/transpose/sharding."""

    @pytest.mark.parametrize("name", ["memory_read", "memory_write", "memory_evict"])
    def test_batching_rule_complete(self, name):
        assert coverage_for(name).contract_status["batching_rule"] == "complete"

    @pytest.mark.parametrize("name", ["memory_read", "memory_write", "memory_evict"])
    def test_sharding_rule_complete(self, name):
        assert coverage_for(name).contract_status["sharding_rule"] == "complete"

    def test_memory_read_transpose_complete(self):
        # memory_read is differentiable, so transpose_rule is required.
        assert coverage_for("memory_read").contract_status["transpose_rule"] == "complete"

    @pytest.mark.parametrize("name", ["memory_write", "memory_evict"])
    def test_mutating_memory_transpose_is_not_applicable(self, name):
        # write/evict mutate state — no transpose rule applies.
        assert coverage_for(name).contract_status["transpose_rule"] == "no_linear_transpose"

    def test_memory_read_only_backend_kernel_remains(self):
        e = coverage_for("memory_read")
        # Everything else is complete or N/A; backend_kernel is the only gate.
        for axis, status in e.contract_status.items():
            if axis == "backend_kernel":
                assert status in ("partial", "planned")
            else:
                assert is_contract_closed(status), (axis, status)
