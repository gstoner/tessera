--- BEGIN-MERGE: Operator_Benchmarks_Spec ---
# Tessera Operator Benchmarks — Spec (Part 1)

## Scope & Philosophy
- Operator-focused microbenchmarks that map cleanly to Tessera IR and Target-IR.
- Clear metrics: latency (ms), throughput (GFLOP/s), bandwidth (GB/s), numerical error vs reference.

## Operators (initial set)
- Matmul/GEMM (tile-sizes; f16/bf16/fp32; accumulate type)
- Conv2D NHWC (+ fused epilogues: bias, GELU)
- FlashAttention FWD (causal/non-causal; dropout stub)
- Reduce (sum/max), Elementwise (activation mix), Transpose/Gather
- Softmax, LayerNorm, Softmax+LayerNorm

## Metrics
- Average over N iterations (drop first K warmups)
- FLOPs & bytes modeled per op; arithmetic intensity reported
- Optional NVTX ranges to segment kernels
--- END-MERGE: Operator_Benchmarks_Spec ---
