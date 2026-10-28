# Public bounded-row quantized MLP call timing on gfx1201 (2026-10-28)

**Sync key:** `GFX1201-PUBLIC-BOUNDED-CALL-TIMING-20261028`. **Owner:** ROCM-NVFP4-INGEST-1.

**Host:** Tajasarus, an RX 9070 XT (gfx1201) on Ubuntu 26.04 under WSL2, with ROCm 10.0 / HIP 7.15. The compiler is the assertions-ON LLVM/MLIR 23.1.1 build (`build-assertions/`). Everything ran unsandboxed on the device box.

`paired.json` was produced by `benchmarks/rocm/benchmark_public_bounded_quantized_mlp.py` at commit `a94c0062c` on a clean worktree. It records the recorder, compiler and HIP provider hashes.

## What was measured

H/K/N = 128/12288/256 with capacity 256, at active M = 1, 16, 64 and 256. All arms share one weight set, and the activations vary with M. There are three arms, each measured as host wall time per call:

| Arm | What one call does |
|---|---|
| `public_bounded` | A public `@jit` call bound by one `compile_native_quantized_program(..., m_bound=256)`. It prepares, invokes, reads and closes a capacity owner on every call. |
| `public_static` | A public `@jit` call from a separate function traced and compiled at that M. It retraces the function and keys its program cache by canonical MLIR on every call, then prepares an owner at M. |
| `prepared_bounded` | The same bounded program held in one prepared owner: an activation-only row update, then invoke and readback. This is what a public call would cost if it kept its owner. |

**Method:**
- Each sample is a window of 10 calls.
- 12 rounds; the arm order rotates and reverses every round, and the M order alternates.
- Before timing, each arm's output is checked bitwise. Public bounded and prepared bounded must be equal or the recorder aborts. Equality with static-at-M is recorded, not assumed.

## Result (median ms per call)

| Active M | Public bounded | Public static | Prepared bounded | Bounded / static | Prepared / public bounded | Bitwise equal static-at-M | Static first call s |
|---|---|---|---|---|---|---|---|
| 1 | 3.695 | 5.887 | 0.790 | 0.628 | 0.214 | yes | 1.09 |
| 16 | 3.464 | 4.753 | 0.545 | 0.729 | 0.157 | yes | 1.07 |
| 64 | 3.907 | 4.809 | 0.639 | 0.812 | 0.164 | yes | 1.05 |
| 256 | 3.770 | 6.218 | 0.880 | 0.606 | 0.233 | yes | 0.81 |

**Bounded is faster than static.** At every M the slowest public bounded window was faster than the fastest public static window: maximum 4.23–4.38 ms against minimum 4.65–5.00 ms.

**Setup cost.** One bounded compile took 1.48 s, and its owner holds 18 device buffers and 24.7 MB. Each static M costs its own first call of 0.8–1.1 s.

## Where the time goes (cProfile, M = 16, 20 warm calls per arm)

The profiler inflates the absolute numbers; only the split between components is meaningful.

- **Execute is about equal in both arms:** 8.4 ms per call bounded, 8.0 ms static.
- **The static arm adds about 3.7 ms per call before execution.** The extra is `_trace_frontend_capture` plus `GraphIRModule.to_mlir(canonical=True)` plus `verify`, which make the cache key. Most of the `to_mlir` time is spent resolving source-location paths (`_loc_path` → `pathlib.resolve`). The bounded call skips all of it.
- **Inside execute, owner construction dominates.** Each call builds a new owner: `NativeQuantizedMLPProgram.prepare` → `NativeScaledProgram.__init__`, which includes binding-cache snapshots (`quantized_binding_cache.freeze`). It also re-validates the frontend record. The device invoke itself was 0.35 ms per call under the profiler.
- **What keeping the owner would save.** The `prepared_bounded` arm shows a held owner would cut the public bounded call to 16–23% of its current cost. That is the open owner-retention follow-up; this packet measures it but does not implement it.

## Limits

- **Diagnostic only.** Host wall on WSL2 has no admitted kernel-side witness, `rocprofv3` returns no records without `/dev/kfd`, and nothing here is a promotion.
- **One shape, one host, gfx1201 only.** gfx1151 has no evidence and none transfers.
- **Bitwise equality with static-at-M was observed at this shape.** It is not asserted in general.
