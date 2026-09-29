# Next compiler slices — 2026-09-28

Owner: E2E-REAL-6. Synchronization key: `COMPILER-NEXT-SLICES-2026-09-28`.
Base: merged #877 (`4f3396b5a`). Implementation and device evidence are distinct
from promotion. No selectors changed.

## Implemented and validated

- **x86 batched linalg:** Graph verification admits matching rank-2/rank-3
  Cholesky and triangular solve. Native Schedule owns the batch scalar,
  operands, policies and full logical shapes. The retained Python batched
  Tile constructor is removed. Batch mismatch, result mismatch and rank
  mismatch are rejected. The full x86 differential/breadth/operation registry
  sweep passed **1,112** tests on Princess-Luna; both affected lit files passed.
  The differential includes the old image/ABI and independent NumPy oracles.
  The final breadth-entrypoint/census sweep passed **1,108** tests; the owning
  generator now marks x86 breadth as a generic compiled route.
  Follow-up review: Apple CPU explicitly refuses the newly valid rank-3
  Graph forms before executable lowering; its LAPACK ABI remains rank-2.
- **ROCm attention cache:** native Tile-to-Target retains static extents in
  replayed Schedule/Tile and launch guards; the device directive retains head
  dimension, storage and all physical/numerical policy. Batch, query/KV head
  counts and sequence lengths are runtime ABI inputs. The existing checked
  shape-free Target compiler now caches this directive. Tests vary all these
  dimensions and Graph names, require identical images, and require bias to
  miss the cache. gfx1151: **47 passed, 5 skipped** with adjacent consumers;
  gfx1201: **29 passed, 1 skipped**. Skips are other-device gates. Eight additional gfx1201 dtype/bias/causal
  policy combinations passed. Audit/diagnostic/pass/dtype drift: **455 passed**.

## Measurements (host milliseconds unless explicitly device microseconds)

| Route | First cold package | New shape, same image | Exact repeat |
|---|---:|---:|---:|
| gfx1151 attention | 517.57 | 69.63–71.69 | 53.62–56.04 |
| gfx1201 attention | 373.62 | 50.67–52.22 | 36.89–38.07 |

Each architecture compiled **one binary across four query lengths**, all
numerically checked. Replay/tool invocation overhead remains; these timings
are not kernel-time speedups. Full sample arrays and image/Schedule digests
are in `gfx1151_attention_cache.json` and `gfx1201_attention_cache.json`.

Batched x86 Cholesky: cold **59.18 ms**, warm **0.96 ms**. Triangular solve:
cold **59.69 ms**, warm **0.88 ms**. Both warm paths invoked the compiler zero
times. Retired cold calls were 28.03/28.59 ms. This migration increases cold
cost while preserving warm reuse and numerical behavior.

## Device follow-ups and failed performance gates

- **gfx1151 movement:** all **21** selected native MoE/paged-KV tests passed.
  The merged MoE Schedule contract already existed; no duplicate was added.
  Paged KV: compiler **2.81 us** device / **2.23 ms** full call; retained
  **2.50 us** / **0.66 ms**. MoE: compiler **2.32 us** / **1.88 ms** full
  call; retained full call **1.65 ms**. The existing 10% non-regression gate
  fails. No route promotion. General layouts remain open.
- **sm_120 paged KV:** **25** shared packed/state replay tests passed on
  Super-Bear. Both benchmark runs checked all three page-boundary/ragged
  envelopes numerically. The repeat increased sampling to 41 samples,
  300 device repetitions and 30 full-call repetitions. Only the canonical
  2048-token row met the existing stability gate in both runs; the other
  rows remain noisy. Both raw packets are retained. No selector change.
- **Apple:** fresh Mac `tessera-opt` and `TesseraAppleRuntimeShared` builds;
  both checked direct Metal ABI numerical tests passed. Static, file-backed
  `@jit` probes still report `artifact_only`, without a native image or launch
  descriptor. The optional tool-validation path receives noncanonical
  parenthesized Graph text and fails parsing. Eager returned arrays are not
  native proof. `apple_jit_probe.py` reproduces this result; the compact
  provenance record is `apple_jit_gap.json`. Native Schedule/package closure
  for target_verify, scaled ntk_rope and Philox remains open.

## Reproduce

Run from the matching source checkout, with `PYTHONPATH=python:.`,
`TESSERA_BUILD_DIR=$PWD/build` and the owning backend environment script.

- `benchmarks/x86/measure_x86_package_cache.py`
- `benchmarks/rocm/measure_scheduled_attention_cache.py --output <path>`
- `benchmarks/rocm/benchmark_rocm_e2e_movement.py --output <path>`
- `benchmarks/nvidia/record_e2e_spine_paged_kv.py --samples 41 --device-reps 300 --e2e-reps 30 --warmup 40 --output <path>`

Apple proof additionally sets `TESSERA_APPLE_GPU_RUNTIME_LIB` to the fresh
shared library and puts `build/tools/tessera-opt` on PATH. Run outside the
sandbox. ROCm device tests require `TESSERA_ROCM_E2E_DEVICE_TEST=1` on
Princess-Luna or `TESSERA_GFX1201_DEVICE_PROOF=1` on Tajasarus.

## Remaining obligations

Apple compiler-owned execution; x86 ALiBi explicit slopes operand; broader
paged KV and host-overhead reduction; NVIDIA LSE/backward and quantized
matmul routes; scheduled matmul image identity; wider W8A8 short/ragged-K
coverage and correctness-preserving MXFP4 A-stage optimization. The larger
programs retain their own plans. No completion is inferred from these slices.
