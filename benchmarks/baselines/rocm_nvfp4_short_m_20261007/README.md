# gfx1201 packed NVFP4 resident short-M execution — 2026-10-07

Owner ROCM-NVFP4-INGEST-1 / E2E-REAL-6. Sync key NVFP4-SHORT-M-2026-10-07.

The compiler-owned packed folded profile now admits positive M through Graph,
Schedule, Tile, ROCm Target, image lowering, checked runtime ABI and the native
three-stage owner. Its existing full-K schedule, explicit approximate numeric
policy, N16/K64 restrictions, BF16 output, and 256x64 geometry remain intact.
Unpacked folded and legacy hand-emitted profiles retain M>64.

Owning device: Tajasaurus, AMD Radeon RX 9070 XT, gfx1201,
UUID GPU-28d9e7efbf2ef716. The compiler is built on Super-Bear with matching
LLVM/MLIR 23.1.1; immutable tools are replayed on Tajasaurus. The isolated HIP
owner links the existing owning native image library. See fingerprints.json.

Validation:
- 24 host typing/static/runtime-image package cases pass.
- 16 public JIT cases pass for M1/16/32/64, N32/K64 and N80/K256; reordered
  arguments, changed scales, forbidden compiler/host arithmetic and replay.
- 10 owning ragged cases pass M3/7/17/31/63, static and runtime-M images.
- Existing resident/JIT device regressions: 35 passed, one environment failure;
  fresh-process replay then passes after setting the native image library path.
- Existing host regressions: 51 passed, 30 device-gated skips.

Twelve timing shapes M1/16/32/64/65/128 x N32K64/N80K256 pass the existing
BF16 numerical bounds before and after timing. Short-M consumer native graph
per-iteration medians are 0.0183-0.0216 ms; combined graph medians are
0.1089-0.1665 ms. The converter dominates these small envelopes.
HIP event loops include host enqueue gaps. Native graph timings include graph
dispatch; prepared walls include execution/readback. These are characterization
samples, not a prior-route speedup or selector promotion. Raw samples and
exact recorder source are retained.

Earlier Graph/Tile/Target admission failures and the environment failure remain
in their logs. No model-quality, generic AD/layout/dynamic, sibling physical
execution, universal performance or full-suite closure is claimed.
