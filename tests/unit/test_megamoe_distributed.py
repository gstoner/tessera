"""Distributed MegaMoE — expert-parallel forward with token all-to-all.

megamoe_forward shards experts across ranks (rank r owns expert block
[r·Ep,(r+1)·Ep)) and routes tokens to the owning rank via a 2× all-to-all
(GShard / Switch pattern), running the heavy expert FFN through the fused GPU
moe_swiglu_block. Per Decision #6, multi-rank runs in-process on MockRankGroup
(threads). The correctness anchor: with capacity large enough to drop nothing,
the gathered distributed output equals the single-device nn.functional.moe_layer.
"""

import numpy as np
import pytest

from tessera.distributed.moe import (
    MoEConfig,
    expert_capacity,
    megamoe_forward,
    megamoe_layer,
)
from tessera.nn import functional as F


def _inputs(seed, T=32, K=16, E=8, Fdim=24, N=12):
    rng = np.random.default_rng(seed)
    return (
        rng.standard_normal((T, K)).astype(np.float32),       # x
        rng.standard_normal((K, E)).astype(np.float32),       # W_router
        rng.standard_normal((E, K, Fdim)).astype(np.float32),  # W_gate
        rng.standard_normal((E, K, Fdim)).astype(np.float32),  # W_up
        rng.standard_normal((E, Fdim, N)).astype(np.float32),  # W_down
    )


@pytest.mark.parametrize("world_size,top_k", [(1, 1), (2, 1), (4, 2), (2, 2), (4, 1)])
def test_distributed_matches_single_device(world_size, top_k):
    # Big capacity_factor → no drops → distributed == single-device exactly.
    x, Wr, Wg, Wu, Wd = _inputs(world_size * 10 + top_k, T=32, E=8)
    cfg = MoEConfig(num_experts=8, top_k=top_k, capacity_factor=8.0)
    y_dist, dropped = megamoe_layer(x, Wr, Wg, Wu, Wd, world_size=world_size, config=cfg)
    assert dropped == 0
    y_single = np.asarray(F.moe_layer(x, Wr, Wg, Wu, Wd, top_k=top_k))
    np.testing.assert_allclose(y_dist, y_single, rtol=1e-4, atol=1e-4)


def test_output_shape_and_world_size_one():
    x, Wr, Wg, Wu, Wd = _inputs(1, T=16, E=4, N=12)
    cfg = MoEConfig(num_experts=4, top_k=2, capacity_factor=8.0)
    y, dropped = megamoe_layer(x, Wr, Wg, Wu, Wd, world_size=1, config=cfg)
    assert y.shape == (16, 12)
    assert dropped == 0


def test_expert_capacity_formula():
    # global slots = tokens_per_rank·num_ranks·top_k; per expert × factor.
    c = expert_capacity(tokens_per_rank=8, num_experts=8, num_ranks=4, top_k=2,
                        capacity_factor=1.25)
    assert c == int(np.ceil(1.25 * (8 * 4 * 2) / 8))  # == 10


def test_capacity_drop_is_reported_and_finite():
    # Tiny capacity forces overflow drops; output stays finite, dropped > 0.
    x, Wr, Wg, Wu, Wd = _inputs(7, T=64, E=8)
    cfg = MoEConfig(num_experts=8, top_k=2, capacity_factor=0.25)
    y, dropped = megamoe_layer(x, Wr, Wg, Wu, Wd, world_size=4, config=cfg, capacity=1)
    assert y.shape == (64, 12)
    assert np.isfinite(y).all()
    assert dropped > 0


def test_quantized_distributed_within_budget():
    x, Wr, Wg, Wu, Wd = _inputs(9, T=32, K=64, E=8, Fdim=32, N=16)
    cfg = MoEConfig(num_experts=8, top_k=2, capacity_factor=8.0)
    y_ref, _ = megamoe_layer(x, Wr, Wg, Wu, Wd, world_size=4, config=cfg)
    y_q, _ = megamoe_layer(x, Wr, Wg, Wu, Wd, world_size=4, config=cfg, quant="fp8_e4m3")
    rel = np.linalg.norm(y_q - y_ref) / (np.linalg.norm(y_ref) + 1e-9)
    assert rel < 0.15, f"fp8 distributed MoE rel {rel:.4f}"


def test_experts_not_divisible_by_world_size_raises():
    x, Wr, Wg, Wu, Wd = _inputs(3, T=16, E=8)
    cfg = MoEConfig(num_experts=8, top_k=1, capacity_factor=8.0)
    with pytest.raises(ValueError, match="divisible"):
        megamoe_layer(x, Wr, Wg, Wu, Wd, world_size=3, config=cfg)


def test_two_all_to_all_round_trips_preserve_token_order():
    # A token's combined output must land back at its originating row index —
    # the 2× all-to-all must round-trip cleanly. Verified via the per-rank path.
    from tessera.testing.mock_collective import MockRankGroup
    x, Wr, Wg, Wu, Wd = _inputs(5, T=24, E=6)
    cfg = MoEConfig(num_experts=6, top_k=2, capacity_factor=8.0)
    Ep, Tl = 6 // 3, 24 // 3

    def worker(rank):
        r = rank.rank
        return megamoe_forward(
            rank, x[r * Tl:(r + 1) * Tl], Wr,
            Wg[r * Ep:(r + 1) * Ep], Wu[r * Ep:(r + 1) * Ep],
            Wd[r * Ep:(r + 1) * Ep], config=cfg)

    results = MockRankGroup(n=3).run(worker)
    y = np.concatenate([res.y_local for res in results], axis=0)
    y_single = np.asarray(F.moe_layer(x, Wr, Wg, Wu, Wd, top_k=2))
    np.testing.assert_allclose(y, y_single, rtol=1e-4, atol=1e-4)
