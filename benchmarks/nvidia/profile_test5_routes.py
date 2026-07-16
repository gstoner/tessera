"""One-shot production-route launcher for TEST-5 Nsight resource capture."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "python"))


def main() -> int:
    from tessera import runtime as rt
    from tessera.compiler.emit import nvidia_cuda as nv
    from tessera.compiler.emit.candidate import (
        OP_ATTENTION, OP_FUSED_REGION, OP_GATED_MATMUL, OP_MATMUL,
        candidates_for)
    from tessera.compiler.fusion import (
        AttentionRegion, FusedRegion, GatedMatmulRegion, MatmulRegion)
    from tessera.stdlib import moe

    if rt._nvidia_device_name() != "sm_120":
        return 0
    rng = np.random.default_rng(1205)

    # Tile GEMM direct/shared at a production grid.
    a = (rng.standard_normal((512, 512)) * .1).astype(np.float32)
    b = (rng.standard_normal((512, 512)) * .1).astype(np.float32)
    region = MatmulRegion(dtype="float16")
    for candidate in candidates_for("nvidia", OP_MATMUL):
        if candidate.name.startswith("nvidia_tile_matmul_"):
            candidate.measure_device_latency(region, a, b, reps=1, warmup=0)
        elif candidate.name == "nvidia_mma_gemm_shipped":
            candidate.run(region, a, b)

    # Fused epilogue and forward attention production candidates.
    bias = (rng.standard_normal(512) * .05).astype(np.float32)
    fused = FusedRegion(epilogue=("bias", "gelu"))
    for candidate in candidates_for("nvidia", OP_FUSED_REGION):
        if candidate.name == "nvidia_mma_fused":
            candidate.run(fused, a, b, bias)
    q = (rng.standard_normal((256, 64)) * .1).astype(np.float32)
    k = (rng.standard_normal((256, 64)) * .1).astype(np.float32)
    v = (rng.standard_normal((256, 64)) * .1).astype(np.float32)
    attn = AttentionRegion(scale=64 ** -.5, causal=True)
    for candidate in candidates_for("nvidia", OP_ATTENTION):
        if candidate.name == "nvidia_mma_attn":
            candidate.run(attn, q, k, v)

    # TF32 composed winners retained in the D2 corpus.
    fused_tf32 = FusedRegion(epilogue=("bias", "gelu"), storage_dtype="f32")
    for candidate in candidates_for("nvidia", OP_FUSED_REGION):
        if candidate.name == "nvidia_mma_fused_composed_tf32":
            candidate.measure_device_latency(
                fused_tf32, a, b, bias, reps=1, warmup=0)
    attn_tf32 = AttentionRegion(scale=64 ** -.5, causal=True,
                                storage_dtype="f32")
    for candidate in candidates_for("nvidia", OP_ATTENTION):
        if candidate.name == "nvidia_mma_attn_composed_tf32":
            candidate.measure_device_latency(
                attn_tf32, q, k, v, reps=1, warmup=0)
    gated = GatedMatmulRegion(gate_act="silu", storage_dtype="f32")
    ga = a[:64, :256]; gw = b[:256, :256]
    for candidate in candidates_for("nvidia", OP_GATED_MATMUL):
        if candidate.name == "nvidia_mma_gated_composed_tf32":
            candidate.measure_device_latency(
                gated, ga, gw, gw, reps=1, warmup=0)

    cx = (rng.standard_normal((1, 32, 32, 32)) * .1).astype(np.float32)
    cw = (rng.standard_normal((3, 3, 32, 64)) * .1).astype(np.float32)
    nv.run_conv2d_resident_candidate(
        cx, cw, route="direct", padding=(1, 1), reps=1, warmup=0)

    # Backward attention.
    q4 = q.reshape(1, 1, 256, 64); k4 = k.reshape(1, 1, 256, 64)
    v4 = v.reshape(1, 1, 256, 64)
    do = (rng.standard_normal(q4.shape) * .1).astype(np.float32)
    nv.measure_flash_attention_backward_device(
        do, q4, k4, v4, scale=64 ** -.5, causal=True, reps=1)

    # Reductions and all MoE transport routes.
    red = (rng.standard_normal((257, 1025)) * .1).astype(np.float32)
    nv.measure_row_reduce_device(red, "sum", reps=1)
    x = (rng.standard_normal((257, 193)) * .1).astype(np.float32)
    ids = rng.integers(0, 5, (257, 1), dtype=np.int64)
    weights = np.ones((257, 1), np.float32)
    plan = moe.plan_dispatch(ids, weights, 5, capacity=80)
    slots = np.asarray(plan.sort_perm, np.int64)
    tok = slots // int(plan.top_k)
    packed = nv.run_moe_dispatch_f32(x, tok)
    nv.measure_moe_dispatch_device(x, tok, reps=1)
    combine_weights = np.asarray(plan.weights, np.float32).reshape(-1)[slots]
    nv.measure_moe_combine_device(packed, tok, combine_weights, 257, reps=1)
    groups = np.array([51, 0, 73, 61, 72], np.int64)
    expert_weights = (rng.standard_normal((5, 193, 127)) * .1).astype(np.float32)
    nv.measure_grouped_gemm_device(x, expert_weights, groups, reps=1)

    # Stateful serving candidates, both paged-KV routes and ReplaySSM.
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "nvidia_serving_profile", ROOT / "benchmarks/nvidia/benchmark_serving.py")
    assert spec and spec.loader
    serving = importlib.util.module_from_spec(spec); spec.loader.exec_module(serving)
    serving.run_benchmark(["1x128x64"], tokens=16, chunk=4, slots=4,
                          kv_tokens=[512], heads=8, dim=64,
                          page_size=16, reps=1)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
