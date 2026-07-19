"""Host-free contract retained from the NVIDIA MoE transport device family."""
from __future__ import annotations

import numpy as np
import pytest

from tessera.compiler.moe_transport import descriptor_from_dispatch_plan, grouped_expert_metadata
from tessera.stdlib import moe


def test_grouped_gemm_rejects_bad_partition_without_cuda():
    from tessera.compiler.emit.nvidia_cuda import run_grouped_gemm_f32

    with pytest.raises(ValueError, match="K/group_sizes"):
        run_grouped_gemm_f32(
            np.zeros((4, 3), np.float32),
            np.zeros((2, 3, 2), np.float32),
            np.array([1, 1]),
        )


def _plan():
    ids = np.array([[2, 0], [1, 2], [0, 2]], np.int64)
    weights = np.array([[.7, .3], [.4, .6], [.2, .8]], np.float32)
    return moe.plan_dispatch(ids, weights, 3, capacity=2)


def test_canonical_moe_metadata_is_grouped_ragged_and_ordered():
    descriptor = descriptor_from_dispatch_plan(_plan())
    assert descriptor.group_sizes.tolist() == [2, 1, 2]
    assert descriptor.group_offsets.tolist() == [0, 2, 3, 5]
    assert descriptor.ragged
    assert descriptor.token_of_slot.dtype == np.int32
    assert descriptor.expert_of_slot.tolist() == sorted(descriptor.expert_of_slot.tolist())
    assert descriptor.ordering.synchronization == (
        "dispatch_before_expert_compute",
        "expert_compute_before_combine",
        "combine_completion",
    )


def test_canonical_moe_metadata_rejects_duplicate_and_missing_kept_slots():
    from dataclasses import replace

    plan = _plan()
    bad = replace(plan, sort_perm=np.array([0, 0, 2, 3, 4], np.int64))
    with pytest.raises(ValueError, match="duplicate slots"):
        descriptor_from_dispatch_plan(bad)
    bad = replace(plan, sort_perm=plan.sort_perm[:-1])
    with pytest.raises(ValueError, match="exactly the kept slots"):
        descriptor_from_dispatch_plan(bad)


def test_canonical_grouped_expert_metadata_retains_empty_and_ragged_groups():
    metadata = grouped_expert_metadata([2, 0, 3, 1], num_tokens=6)
    assert metadata.group_sizes.tolist() == [2, 0, 3, 1]
    assert metadata.group_offsets.tolist() == [0, 2, 2, 5, 6]
    assert metadata.ragged
    assert metadata.as_metadata_dict()["index_dtype"] == "int32"


def test_canonical_grouped_expert_metadata_rejects_bad_partition():
    with pytest.raises(ValueError, match="partition all tokens"):
        grouped_expert_metadata([2, 1], num_tokens=4)
