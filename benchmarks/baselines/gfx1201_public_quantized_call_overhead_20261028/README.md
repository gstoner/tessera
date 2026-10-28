# Public quantized MLP call overhead on gfx1201 (2026-10-28)

**Sync key:** `GFX1201-PUBLIC-QUANTIZED-CALL-OVERHEAD-20261028`. **Owner:** ROCM-NVFP4-INGEST-1.

**Host:** Tajasarus, an RX 9070 XT (gfx1201) on Ubuntu 26.04 under WSL2, with ROCm 10.0 / HIP 7.15. The compiler is the assertions-ON LLVM/MLIR 23.1.1 build (`build-assertions/`). Everything ran unsandboxed on the device box.

## What changed

- **Retained bounded owner.** A bounded public `@jit` call keeps one prepared capacity owner across calls (`RetainedBoundedOwner`):
  - byte-identical weight roots replace only the activation rows;
  - other weights take the full validated update;
  - admission refusals keep the owner, and native failures close it.

  Contract: `docs/spec/NATIVE_FP8_BLOCK_QUANT_FOUNDATION.md` §"Retained public owner".
- **Memoized location paths.** `graph_ir._loc_path` memoizes source-location path resolution, which the canonical render used as the static call's cache key did per op on every call.

## Timing

`paired.json` was produced by the recorder from #1021, `benchmarks/rocm/benchmark_public_bounded_quantized_mlp.py`, unchanged (same `recorder_sha256`). It ran at commit `33f6a2374` on a clean worktree, with H/K/N 128/12288/256 and capacity 256. Values are host wall per call, as medians. "Before" is the #1021 packet (`../gfx1201_public_bounded_call_20261028/`).

| Active M | Public bounded, before → after | Public static, before → after | Held-owner control, before → after |
|---|---|---|---|
| 1 | 3.695 → **0.918** | 5.887 → 4.316 | 0.790 → 0.772 |
| 16 | 3.464 → **0.674** | 4.753 → 3.879 | 0.545 → 0.539 |
| 64 | 3.907 → **0.762** | 4.809 → 4.033 | 0.639 → 0.642 |
| 256 | 3.770 → **1.027** | 6.218 → 4.198 | 0.880 → 0.888 |

- **Bounded call: about 4–5× faster.** Every window after the change is faster than every window before it: maximum 0.83–1.23 ms against minimum 3.41–3.70 ms. It now sits about 0.13 ms above the held-owner control, and that gap is the weight-byte comparison plus argument binding.
- **The control did not move.** The held-owner arm ran the same code both times, so the setup is comparable across the two runs.
- **Static call: 16–33% lower median.** This compares two separate runs, not interleaved arms, and the windows overlap slightly at M = 16 and 256. Read it as the median change, not a paired result.
- **Bitwise checks hold.** All arms are bitwise equal at every M, before timing.

## Mechanism (cProfile, M = 16, 20 warm calls per arm)

The profiler inflates the absolute numbers.

- **Bounded call:** 8.5 → 0.9 ms per call. `prepare` is gone. What is left is `update_activations` (0.2 ms) and the weight-byte comparison (0.1 ms).
- **Static call:** the canonical `to_mlir` cache-key render went 3.6 → 1.5 ms per call, and the `_loc_path` / `pathlib.resolve` frames are gone. The per-call `prepare` (4.6 ms) is untouched and is now its largest cost.

## Correctness and regression

- **Retention device test.** `tests/device/rocm/test_public_quantized_mlp_row_bound_device.py` gives 5 passed, including `test_public_bound_retains_one_owner_across_calls`. It makes one prepare across these calls:
  - M = 3, 8, 1, 5 on held weights;
  - a capacity refusal;
  - a new weight set;
  - an in-place weight mutation;
  - a return to held weights.

  Each output is bitwise equal to the static capacity rows of the roots it was given.
- **Quantized/NVFP4/gate-up/block-quant device set:** 181 passed.
- **Unit tests.** `tests/unit/test_retained_bounded_owner.py` gives 12 passed (Mac and Tajasarus). It covers:
  - the reuse and refresh policy;
  - refusals that keep the owner;
  - native failures that drop it;
  - fork refusal;
  - `_loc_path` memoization keyed by working directory.
- **Pre-existing failures, not from this change.** On Tajasarus a unit subset (graph_ir/loc/canonical/quantized/nvfp4/jit/trace, `-n 12`) gave 5316 passed, 28 failed, 12 errors. Re-running the five failing files at `origin/main` and at this branch gave the identical set of 209 failing and erroring test IDs. These are attention-reverse files needing the NVIDIA backend, plus NVFP4 owner/resident cases that pass when run alone. On the Mac the same files fail identically at `main`, because its `tessera-opt` has no NVIDIA or ROCm lowering pass.

## Limits

- **Diagnostic only.** Host wall on WSL2 promotes nothing.
- **One shape, gfx1201 only.** There is no gfx1151 evidence.
- **Not covered:** retaining a static program's owner, and the remaining per-call canonical render that keys the static program cache.
