# Public typed FP8 scale-JVP integration

The matching LLVM/MLIR 23.1.1 compiler and native HIP runtime were rebuilt on
Tajasaurus. The runtime reported gfx1201 before tests. Nine public JIT cases
pass: M/N/K=(17,19,256), (32,32,256), (200,19,256), each with sa, sb, or
both scale operands active. Numerical comparisons use an independent float64
block oracle and central finite differences; changed tangent reuse also passes.
The execution receipt requires native_gpu, rocm_gfx1201, tracer authority,
and scaled_product_program. Member execution and private buffers are owned
by native C++, with Graph/Schedule/Tile/Target-generated images.

The first run exposed missing concrete scaled-matmul tracer evaluation.
The repair permits reference evaluation only when requested for frontend
differential certification; production lowering remains compiler-owned.

This packet does not establish generic batching/transpose,
composed AD, sibling physical parity, or a green full suite.
Source archive before the tracer repair:
f337b9a6eaa9fc7dc99fd6b08c4bd829a99c519b6bbeca928d09e450728c4ae3.
Final source/runtime fingerprints and raw logs accompany this packet.

## Separate timing domains

Correctness-gated recorder: benchmarks/rocm/benchmark_public_scaled_jvp.py. Warm calls prohibit compiler subprocesses.

| M/N/K | Public warm ms | Native sequence event ms | Prepared update/invoke/two readbacks ms |
| --- | --- | --- | --- |
| [17, 19, 256] | 6.89969 | 0.02936 | 0.66473 |
| [32, 32, 256] | 6.62063 | 0.02451 | 0.57866 |
| [200, 19, 256] | 6.75952 | 0.03286 | 0.69562 |

Native events include native enqueue gaps, not isolated kernel timing. Public calls include frontend/descriptor work, native allocation/preparation, execution, readback and cleanup. Warm profiling identifies preparation and close as the main overhead (0.079 and 0.032 seconds across 20 M17 calls). Native owner reuse is the next optimization. No speedup claim. The first recorder run double-normalized native event durations; only the regenerated timings.json is evidence.
