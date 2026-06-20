"""P1 distributed-correctness regressions from the full-source code review.

  * DistributedArray.parts() — shards inherited the parent's partitioned
    shard_spec, so a shard reported as still-partitioned and would re-split on
    a second parts(). A concrete per-rank slice must be replicated.  [FIX]
  * plan_all_to_all — source rank was inferred by floor division, dumping the
    remainder of a non-divisible batch onto the last rank (inflated
    send_counts). Must follow the np.array_split contiguous layout.  [FIX]
"""

from __future__ import annotations

import numpy as np

from tessera.distributed.array import DistributedArray
from tessera.distributed.domain import Block, Rect
from tessera.distributed.moe import RoutingResult, plan_all_to_all
from tessera.distributed.shard import MeshSpec


def test_block_shards_are_replicated_not_partitioned():
    X = DistributedArray.from_domain(Rect((8, 4)), dtype="fp32", distribution=Block(("dp",)))
    X._bind_mesh(MeshSpec({"dp": 4}))
    shards = X.parts("dp")

    assert len(shards) == 4
    for s in shards:
        assert s.shard_spec.replicated, "a concrete per-rank shard must be replicated"
        assert s.shard_spec.mesh_axes == ()
    # A shard re-queried returns itself (replicated), not a second split.
    assert shards[0].parts("dp") == [shards[0]]


def test_plan_all_to_all_source_ranks_follow_array_split_for_uneven_batch():
    # 10 tokens across 4 ranks: array_split sizes are [3, 3, 2, 2].
    num_tokens, num_experts, num_ranks, top_k = 10, 8, 4, 1
    assignment = (np.arange(num_tokens) % num_experts).reshape(num_tokens, top_k).astype(np.int64)
    routing = RoutingResult(
        assignment=assignment,
        weights=np.ones((num_tokens, top_k), dtype=np.float32),
        load=np.zeros(num_experts, dtype=np.int64),
        overflow=0,
    )
    plan = plan_all_to_all(routing, num_experts=num_experts, num_ranks=num_ranks)

    # tokens-sent-per-source-rank must match the contiguous split, not the
    # old floor-division behavior that produced [2, 2, 2, 4].
    np.testing.assert_array_equal(plan.send_counts.sum(axis=1), [3, 3, 2, 2])
    # conservation: every routed (token, slot) pair is accounted for once.
    assert int(plan.send_counts.sum()) == num_tokens * top_k
