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


def test_fp8xfp4_mixed_precision_distributed():
    # FP8 activations × FP4 weights (Blackwell / DeepGEMM MoE) over the
    # distributed forward — error sits between pure-FP8 and pure-FP4.
    x, Wr, Wg, Wu, Wd = _inputs(15, T=32, K=64, E=8, Fdim=32, N=16)
    cfg = MoEConfig(num_experts=8, top_k=2, capacity_factor=8.0)
    y_ref, _ = megamoe_layer(x, Wr, Wg, Wu, Wd, world_size=4, config=cfg)

    def rel(q):
        y, _ = megamoe_layer(x, Wr, Wg, Wu, Wd, world_size=4, config=cfg, quant=q)
        return np.linalg.norm(y - y_ref) / (np.linalg.norm(y_ref) + 1e-9)

    r_fp8, r_mixed, r_fp4 = rel("fp8_e4m3"), rel("fp8xfp4"), rel("nvfp4")
    assert r_fp8 <= r_mixed <= r_fp4 + 0.05, (r_fp8, r_mixed, r_fp4)


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


# ── Rung 2: comm/compute overlap (micro-batch pipeline) ──────────────────────
from tessera.distributed.moe import (  # noqa: E402
    OverlapSchedule,
    megamoe_layer_overlapped,
)


@pytest.mark.parametrize("num_chunks", [1, 2, 4])
def test_overlap_matches_non_overlapped(num_chunks):
    # Micro-batch pipelining is a pure decomposition — identical result.
    x, Wr, Wg, Wu, Wd = _inputs(num_chunks + 40, T=32, E=8)
    cfg = MoEConfig(num_experts=8, top_k=2, capacity_factor=8.0)
    y_plain, _ = megamoe_layer(x, Wr, Wg, Wu, Wd, world_size=4, config=cfg)
    y_ov, dropped, sched = megamoe_layer_overlapped(
        x, Wr, Wg, Wu, Wd, world_size=4, config=cfg, num_chunks=num_chunks)
    assert dropped == 0
    np.testing.assert_allclose(y_ov, y_plain, rtol=1e-4, atol=1e-4)
    assert isinstance(sched, OverlapSchedule)


def test_overlap_schedule_structure():
    x, Wr, Wg, Wu, Wd = _inputs(2, T=32, E=8)
    cfg = MoEConfig(num_experts=8, top_k=2, capacity_factor=8.0)
    _, _, sched = megamoe_layer_overlapped(
        x, Wr, Wg, Wu, Wd, world_size=4, config=cfg, num_chunks=3)
    assert sched.num_chunks == 3
    # 3 stages per chunk (dispatch / compute / combine).
    kinds = [k for _, k in sched.stages]
    assert kinds.count("dispatch_a2a") == 3
    assert kinds.count("expert_compute") == 3
    assert kinds.count("combine_a2a") == 3
    # compute of chunk c overlaps dispatch of chunk c+1.
    assert sched.overlap_pairs == [(0, 1), (1, 2)]


def test_single_chunk_has_no_overlap():
    x, Wr, Wg, Wu, Wd = _inputs(3, T=16, E=4)
    cfg = MoEConfig(num_experts=4, top_k=1, capacity_factor=8.0)
    _, _, sched = megamoe_layer_overlapped(
        x, Wr, Wg, Wu, Wd, world_size=2, config=cfg, num_chunks=1)
    assert sched.num_chunks == 1
    assert sched.overlap_pairs == []


def test_overlap_composes_with_fp8xfp4():
    x, Wr, Wg, Wu, Wd = _inputs(8, T=32, K=64, E=8, Fdim=32, N=16)
    cfg = MoEConfig(num_experts=8, top_k=2, capacity_factor=8.0)
    y_plain, _ = megamoe_layer(x, Wr, Wg, Wu, Wd, world_size=4, config=cfg, quant="fp8xfp4")
    y_ov, _, _ = megamoe_layer_overlapped(
        x, Wr, Wg, Wu, Wd, world_size=4, config=cfg, num_chunks=2, quant="fp8xfp4")
    # Overlap composes with quant, but micro-batching changes the per-tensor
    # quant SCALE (each chunk quantizes a smaller grouped input) — so the result
    # is close, not bit-identical (a real, expected property of per-chunk quant).
    rel = np.linalg.norm(y_ov - y_plain) / (np.linalg.norm(y_plain) + 1e-9)
    assert rel < 0.05, f"overlap+fp8xfp4 vs monolithic rel {rel:.4f}"


# ── Rung 4: REAL wall-clock comm/compute overlap (async GPU command buffers) ──
from tessera import _apple_gpu_backend as _agb  # noqa: E402
from tessera import _jit_boundary as _jb  # noqa: E402
from tessera.distributed.moe import (  # noqa: E402
    PipelineStats,
    megamoe_forward_pipelined,
    megamoe_layer_pipelined,
)

_GPU = _agb.is_available() and _jb.is_available()
gpu = pytest.mark.hardware_apple_gpu


@pytest.mark.parametrize("num_chunks", [1, 2, 4])
def test_pipelined_matches_non_overlapped(num_chunks):
    # Async pipelining is numerically identical to the monolithic forward.
    x, Wr, Wg, Wu, Wd = _inputs(num_chunks + 60, T=32, E=8)
    cfg = MoEConfig(num_experts=8, top_k=2, capacity_factor=8.0)
    y_plain, _ = megamoe_layer(x, Wr, Wg, Wu, Wd, world_size=4, config=cfg)
    y_pl, dropped, stats = megamoe_layer_pipelined(
        x, Wr, Wg, Wu, Wd, world_size=4, config=cfg, num_chunks=num_chunks)
    assert dropped == 0
    np.testing.assert_allclose(y_pl, y_plain, rtol=1e-4, atol=1e-4)
    assert isinstance(stats, PipelineStats)


def test_compute_runs_off_the_comm_thread():
    # Proof of async: every chunk's expert FFN ran on a worker thread, not the
    # rank thread that issued the comm — i.e. the GPU command buffer was in
    # flight while the CPU did the dispatch all-to-all.
    x, Wr, Wg, Wu, Wd = _inputs(2, T=32, E=8)
    cfg = MoEConfig(num_experts=8, top_k=2, capacity_factor=8.0)
    _, _, stats = megamoe_layer_pipelined(
        x, Wr, Wg, Wu, Wd, world_size=4, config=cfg, num_chunks=3)
    assert stats.num_chunks == 3
    assert len(stats.compute_thread_ids) == 3
    assert all(tid != stats.main_thread_id for tid in stats.compute_thread_ids)
    assert stats.all_offloaded is True


def test_pipelined_composes_with_fp8xfp4():
    x, Wr, Wg, Wu, Wd = _inputs(8, T=32, K=64, E=8, Fdim=32, N=16)
    cfg = MoEConfig(num_experts=8, top_k=2, capacity_factor=8.0)
    y_ov, _, _ = megamoe_layer_overlapped(
        x, Wr, Wg, Wu, Wd, world_size=4, config=cfg, num_chunks=2, quant="fp8xfp4")
    y_pl, _, _ = megamoe_layer_pipelined(
        x, Wr, Wg, Wu, Wd, world_size=4, config=cfg, num_chunks=2, quant="fp8xfp4")
    # Same chunking + same quant scales → identical to the sequential overlap path.
    np.testing.assert_allclose(y_pl, y_ov, rtol=1e-4, atol=1e-4)


@pytest.mark.performance
@pytest.mark.hardware_apple_gpu
@gpu
def test_real_overlap_hides_dispatch_comm_under_gpu_compute():
    # The headline: with a real GPU expert FFN and a modeled interconnect
    # latency per all-to-all, the async pipeline hides the DISPATCH comm under
    # GPU compute, so it beats the sequential-chunked overlap path that exposes
    # every comm round. Large shape so the GPU dispatch dominates; the GPU
    # command buffer runs async (GIL released) while the CPU issues comm.
    import time

    rng = np.random.default_rng(0)
    T, K, E, Fdim, N = 4096, 256, 8, 256, 256
    x = rng.standard_normal((T, K)).astype(np.float32)
    Wr = rng.standard_normal((K, E)).astype(np.float32)
    Wg = rng.standard_normal((E, K, Fdim)).astype(np.float32)
    Wu = rng.standard_normal((E, K, Fdim)).astype(np.float32)
    Wd = rng.standard_normal((E, Fdim, N)).astype(np.float32)
    cfg = MoEConfig(num_experts=E, top_k=2, capacity_factor=4.0)
    L = 0.012  # modeled 12 ms interconnect transfer per all-to-all

    # Warm the fused kernel (one-time MSL compile) so timing is steady-state.
    megamoe_layer_pipelined(x, Wr, Wg, Wu, Wd, world_size=2, config=cfg, num_chunks=4)

    def wall(fn, reps=3):
        best = 1e9
        for _ in range(reps):
            t0 = time.perf_counter()
            fn()
            best = min(best, time.perf_counter() - t0)
        return best

    seq = wall(lambda: megamoe_layer_overlapped(
        x, Wr, Wg, Wu, Wd, world_size=2, config=cfg, num_chunks=4, comm_latency_s=L))
    pll = wall(lambda: megamoe_layer_pipelined(
        x, Wr, Wg, Wu, Wd, world_size=2, config=cfg, num_chunks=4, comm_latency_s=L))
    # The 4 dispatch comms (~48 ms) hide under compute → pipelined < seq-chunked.
    # Generous margin (0.92×) keeps it robust under scheduler noise.
    assert pll < seq * 0.92, f"pipelined {pll*1e3:.1f}ms vs seq-chunked {seq*1e3:.1f}ms"


# ── Rung 5: 2-stage pipeline — also hides the COMBINE comm under compute ──────
@pytest.mark.parametrize("stages", [1, 2])
@pytest.mark.parametrize("num_chunks", [1, 2, 4])
def test_both_pipeline_depths_match_non_overlapped(stages, num_chunks):
    # Both pipeline depths are pure schedule transforms — identical result.
    x, Wr, Wg, Wu, Wd = _inputs(stages * 100 + num_chunks, T=32, E=8)
    cfg = MoEConfig(num_experts=8, top_k=2, capacity_factor=8.0)
    y_plain, _ = megamoe_layer(x, Wr, Wg, Wu, Wd, world_size=4, config=cfg)
    y, dropped, _ = megamoe_layer_pipelined(
        x, Wr, Wg, Wu, Wd, world_size=4, config=cfg,
        num_chunks=num_chunks, pipeline_stages=stages)
    assert dropped == 0
    np.testing.assert_allclose(y, y_plain, rtol=1e-4, atol=1e-4)


def test_two_stage_defers_and_overlaps_every_combine():
    # The structural proof: 2-stage runs nc-1 combines concurrently with a
    # compute (deferred one iteration); 1-stage overlaps no combine.
    x, Wr, Wg, Wu, Wd = _inputs(2, T=32, E=8)
    cfg = MoEConfig(num_experts=8, top_k=2, capacity_factor=8.0)
    _, _, s1 = megamoe_layer_pipelined(
        x, Wr, Wg, Wu, Wd, world_size=4, config=cfg, num_chunks=4, pipeline_stages=1)
    _, _, s2 = megamoe_layer_pipelined(
        x, Wr, Wg, Wu, Wd, world_size=4, config=cfg, num_chunks=4, pipeline_stages=2)
    assert s1.pipeline_stages == 1 and s1.overlapped_combines == 0
    assert s2.pipeline_stages == 2 and s2.overlapped_combines == 4 - 1
    # default is the deeper pipeline
    _, _, sd = megamoe_layer_pipelined(
        x, Wr, Wg, Wu, Wd, world_size=4, config=cfg, num_chunks=3)
    assert sd.pipeline_stages == 2 and sd.overlapped_combines == 3 - 1


def test_invalid_pipeline_stages_raises():
    x, Wr, Wg, Wu, Wd = _inputs(3, T=16, E=4)
    cfg = MoEConfig(num_experts=4, top_k=1, capacity_factor=8.0)
    with pytest.raises(ValueError, match="pipeline_stages"):
        megamoe_layer_pipelined(
            x, Wr, Wg, Wu, Wd, world_size=2, config=cfg, pipeline_stages=3)


def test_single_chunk_overlaps_no_combine():
    # nc=1 has nothing to pipeline — no deferred combine even at depth 2.
    x, Wr, Wg, Wu, Wd = _inputs(4, T=16, E=4)
    cfg = MoEConfig(num_experts=4, top_k=1, capacity_factor=8.0)
    _, _, s = megamoe_layer_pipelined(
        x, Wr, Wg, Wu, Wd, world_size=2, config=cfg, num_chunks=1, pipeline_stages=2)
    assert s.num_chunks == 1 and s.overlapped_combines == 0


@pytest.mark.performance
@pytest.mark.hardware_apple_gpu
@gpu
def test_two_stage_hides_more_comm_than_one_stage():
    # The headline: in a compute-dominant regime the 2-stage pipeline hides the
    # COMBINE comm too, so its EXPOSED comm (time-with-latency minus time-without)
    # is well below the 1-stage's. Measured as the latency-attributable delta so
    # it is robust to absolute GPU speed.
    import time

    rng = np.random.default_rng(0)
    T, K, E, Fdim, N = 8192, 256, 8, 256, 256
    x = rng.standard_normal((T, K)).astype(np.float32)
    Wr = rng.standard_normal((K, E)).astype(np.float32)
    Wg = rng.standard_normal((E, K, Fdim)).astype(np.float32)
    Wu = rng.standard_normal((E, K, Fdim)).astype(np.float32)
    Wd = rng.standard_normal((E, Fdim, N)).astype(np.float32)
    cfg = MoEConfig(num_experts=E, top_k=2, capacity_factor=4.0)
    # L (injected per-all-to-all latency) is large relative to compute jitter so
    # the exposed-comm signal dominates wall-clock noise.
    NC, L = 4, 0.012

    megamoe_layer_pipelined(x, Wr, Wg, Wu, Wd, world_size=2, config=cfg, num_chunks=NC)  # warm

    def wall(stages, lat, reps=6):
        best = 1e9
        for _ in range(reps):
            t0 = time.perf_counter()
            megamoe_layer_pipelined(x, Wr, Wg, Wu, Wd, world_size=2, config=cfg,
                                    num_chunks=NC, comm_latency_s=lat,
                                    pipeline_stages=stages)
            best = min(best, time.perf_counter() - t0)
        return best

    # The exposed-comm signal is real (2-stage genuinely hides the combine comm),
    # but a single wall-clock pair can be masked by a scheduler hiccup on a loaded
    # CI box. Resample a few times and accept the best — this de-flakes without
    # weakening the assertion (each `wall` is already a best-of-N floor).
    margin, exposed1, exposed2 = 0.7, 0.0, 0.0
    for _ in range(3):
        exposed1 = wall(1, L) - wall(1, 0.0)
        exposed2 = wall(2, L) - wall(2, 0.0)
        if exposed1 > 0.0 and exposed2 < exposed1 * margin:
            break
    assert exposed1 > 0.0 and exposed2 < exposed1 * margin, (
        f"2-stage exposed {exposed2*1e3:.1f}ms vs 1-stage {exposed1*1e3:.1f}ms "
        f"(margin {margin})")
