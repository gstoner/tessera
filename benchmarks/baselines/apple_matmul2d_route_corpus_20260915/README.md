# Compiled Metal 4 matmul2d route — paired admission corpus, M1 Max, 2026-09-15

Owner E2E-REAL-6 / APPLE-MATMUL2D-1 (follow-through). Two independent
processes (`run_0.json`, `run_1.json`), 20 interleaved repetitions per row
after three warmups, order alternating each repetition, wall time host → ctypes
→ GPU → host. Harness: `benchmarks/apple_gpu/benchmark_matmul2d_route_corpus.py`.

**Candidate:** the compiled route exactly as the compiler emits it — Graph IR
matmul → `tessera-tiling` → `tessera-apple-canonical-gemm-matmul2d` →
`tessera-apple-matmul2d-fuse-epilogue` → `tessera-apple-matmul2d-to-call`, the
value call extracted by `driver.extract_apple_value_calls`, executed by
`runtime._dispatch_gpu_mtl4_matmul2d` on the projected view ABI.
**Incumbent:** what production dispatches for that dtype today: MPS fp16 GEMM
for f16 (the MTL4 f16 GEMV route at M == 1), the MTL4 bf16 route for bf16, and
for the low-precision pairs the fp16 MPS GEMM on the SAME quantized values —
there is no production low-precision route to displace. Both outputs were
checked against a float64 reference before timing (tolerance 3e-2 of the
result scale; the incumbent writes a 16-bit result, the candidate fp32).

Admission rule: the 95% bootstrap lower bound of the median incumbent/candidate
ratio must clear 1.02 in BOTH processes.

| pair | shapes admitted (both runs) | retained | note |
|---|---|---|---|
| f16 × f16 | none | all six (0.53–0.96×) | at M == 1 the incumbent is the same MTL4 kernel: 1.00×, lb 0.89–0.90 |
| bf16 × bf16 | square 512 (lb 1.02 / 1.03), square 1024 (lb 1.04 / 1.07) | 2048 (0.93–0.95), MLP (0.85–0.87), decode (0.89–0.94); ragged 1000 split across runs (lb 1.03 / 1.02) | incumbent is the runtime's own contiguous bf16 entry; gains are same-kernel dispatch differences, not a new kernel |
| e4m3 × e4m3 | decode M == 1 only (1.69× / 1.60×, lb 1.59 / 1.40) | 512, 1024, 2048, MLP (0.73–0.88×) | packing the 4096² weight costs ~28 ms once (vs ~1.8 ms saved per decode step) |
| f16 × e4m3 (weight-only) | decode M == 1 only (1.72× / 1.68×, lb 1.63 / 1.59) | 512, 1024, 2048, MLP (0.75–0.88×) | same amortisation condition |

Decision recorded in the plan log (`2026-09-15 — matmul2d route: ragged
tails, view origins, fused epilogue, paired admission`): the default
`tessera-lower-to-apple_gpu` pipeline admits the family for the 8/4-bit
storage pairs only (`admit=lowp`), because those pairs had no executable route
— the pipeline used to emit an MPSGraph `matmul_contract` claim for an FP8
GEMM that MPSGraph cannot execute — while f16 and bf16 keep their incumbents.
The bf16 ≤ 1024 wins and the FP8 decode win are arbiter-bucket candidates
(Decision #28), not pipeline admissions. No general speedup is claimed.

Reproduce (fresh runtime dylib, unsandboxed on the Mac):

```sh
PYTHONPATH=python:. python3 benchmarks/apple_gpu/benchmark_matmul2d_route_corpus.py --reps 20 --out /tmp/run.json
```
