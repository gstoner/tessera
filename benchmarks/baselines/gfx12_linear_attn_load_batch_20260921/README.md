# gfx1201 D=128 linear-attention load batching

Sync `GFX12-PUBLIC-LINEAR-LOAD-BATCH-2026-09-21`; owner ROCM-2.

This packet closes the RX 9070 XT gate for batching independent D=128
linear-attention loads before their consumers. It does not prove or promote
`gfx1200`.

## Result

Both images were loaded into one HIP context on Tajasarus, used identical
resident buffers for `[B,H,S,D] = [1,1,32,128]` fp16 causal linear attention,
and alternated arm order across 11 trials of 1,000 launches. The candidate and
baseline outputs were bit-identical; the committed device test separately
compares the candidate against the float64 NumPy oracle.

| Gate | Baseline | Batched loads | Change |
|---|---:|---:|---:|
| HIP-event median | 0.037174 ms | 0.028943 ms | **1.284x** |
| Global loads | 160 | 160 | unchanged |
| Load waits | 105 | 70 | -33.3% |
| Full load drains | 81 (77.1%) | 16 (22.9%) | -80.2% count |
| Loads per wait | 1.524 | 2.286 | +50.0% |
| Instructions | 3,189 | 2,903 | -9.0% |
| VGPR / SGPR | 94 / 44 | 114 / 42 | +20 / -2 |
| LDS / scratch | 9,216 B / 0 B | 9,216 B / 0 B | unchanged |

The mechanism is present in the exact HSACO: the load count stays constant,
while the compiler retains nonzero `s_wait_loadcnt` immediates and sharply
reduces full drains. The VGPR increase is accepted for this exact kernel because
LDS remains its occupancy limiter and measured latency improves.

## Reproduction boundary

- Baseline source: `aacee94ba3cda65e6eec05021965483dfea3f0ac`.
- Candidate source: `3a4272d4` (generator implementation first landed in
  `a803c03b`).
- Toolchain: LLVM/MLIR 23.1.1 assertions build and ROCm 10.0.
- Device: Radeon RX 9070 XT (`gfx1201`) on Tajasarus.
- Harness: `benchmarks/rocm/benchmark_rocm_linear_attn_load_batch.py`.
- Raw record: `evidence_gfx1201.json`, including trial samples, image hashes,
  wait immediates, and resource metadata.

This is exact-target evidence only. A `gfx1200` promotion still requires its
own compiler, launch, numerical, resource, and timing packet on matching
hardware.
