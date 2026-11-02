# Native quantized MLP — gfx1151, 2026-11-01

**Sync key:** `GFX1151-QUANTIZED-MLP-TIMING-20261101`. **Owner:** ROCM-NVFP4-INGEST-1.
The gfx1151 counterpart of `../rocm_quantized_mlp_20261017/` (gfx1201). Follows
slice 4 (`GFX1151-QUANTIZED-MLP-20261030`, PR #1029).

**Host:** Princess-Luna, a Radeon 8060S (gfx1151, Strix Halo, RDNA 3.5) on Ubuntu
26.04 under WSL2, with ROCm 10.0 / HIP 7.15.26333 and the apt LLVM/MLIR 23.1.1
(assertions OFF) `build/` tree. That tree has an empty `CMAKE_BUILD_TYPE`, so the
runtime libraries (the HIP owner provider) are built without `-O`. This can weigh
on the host-wall scopes below. The seven member kernels are HSACO from
`tessera-opt`'s own LLVM pipeline, which this build setting does not choose. Everything ran unsandboxed on that box. Each recorder ran as
`flock /tmp/tessera-gfx1151-device.lock python ... --device-lock <same path>`, and
checked before timing that the lock was held (`device_lock.held` in every JSON).
The lock is shared with other sessions' gfx1151 device runs, so no two timed
windows overlapped.

`public-<M>-<H>-<K>-<N>.json` were produced by
`benchmarks/rocm/benchmark_public_quantized_mlp.py` at commit `e94fd72a6` on a
clean tree. Each records the recorder, compiler, provider and reference hashes,
all seven receipts, and its portable artifact (`*-artifact.json`). A never-timed
warm-up owner at (1, 16, 16, 1) was prepared first in the process
(`warm_up_owner: true`).

## Route

The authored Graph runs as seven native partitions through Schedule, Tile, ROCm
Target and LLVM/HSACO, inside one checked multi-output HIP owner. Every row
records it as `route`. On gfx1151 the E4M3 product is decoded exactly to F16 and
runs on `v_wmma_f32_16x16x16_f16` with fp32 accumulation, because RDNA 3.5 has no
FP8 WMMA. Every product image is a register-panel macro tile; gfx1151 has none
of gfx1201's LDS images. The artifacts record `[16, 16]` and `[16, 32]` at the
two small shapes and `[16, 32]` for both products at 256 × 128 × 12288 × 256.

## Numerical evidence (before timing)

Each shape passed the staged per-stage checks and the propagated certificate
before any window was timed, and resident results equal the staged native
components bitwise. The full-reference max abs error is 0 at the two small shapes
and 0.0625 at 256 × 128 × 12288 × 256. That shape has 13 outputs outside the flat
relative gate; they are kept, not reported as a flat-oracle pass. These are the
same values gfx1201 recorded at that shape. Numerics are checked by the staged
certificate, never by bitwise equality with gfx1201. This is not model-quality
acceptance.

## Separate timing scopes (median)

| M / H / K / N | Public warm ms | Public cold ms | Prepared invoke/read ms | Program event µs | Captured seven members µs |
|---|---:|---:|---:|---:|---|
| 1 / 16 / 16 / 1 | 3.749 | 1102.2 | 0.328 | 27.30 | 3.93 / 2.55 / 8.58 / 1.44 / 3.94 / 2.32 / 6.33 |
| 3 / 32 / 48 / 19 | 3.938 | 1654.9 | 0.344 | 31.36 | 3.61 / 2.40 / 10.13 / 1.49 / 3.64 / 2.39 / 6.60 |
| 256 / 128 / 12288 / 256 | 11.016 | 1782.5 | 2.969 | 1318.62 | 67.52 / 62.65 / **1043.77** / 2.11 / 4.11 / 2.72 / 16.96 |

Member order: input quantizer, gate/up expansion, gate product + BF16 store, SiLU,
activation quantizer, down expansion, down product. Counts: 7 windows; 50
invocations per event window; 128 repeats per captured member; 5 public calls
per window.

- **Timing sources.** Each scope has its own `timing_source`. Public and prepared
  calls use `host_wall_perf_counter`. The ordinary program event uses
  `hip_event_owner_window`. Captured members use `hip_event_graph_member_window`.
  These are different domains and their medians are not interchangeable.
- **The large shape is the gate product.** About 1.04 ms of the 1.32 ms program
  event is the F16-decode gate product over K = 12288. The rest is µs-scale.
- **No cross-architecture comparison.** gfx1151 and gfx1201 differ in product
  instruction, image, memory system and part class (an iGPU with shared LPDDR5X
  against a discrete GDDR6 card). The two packets sit side by side as two
  separate measurements, not as a parity or speed ratio.

## Limits

- **Diagnostic only; no promotion.** HIP events on WSL2 are not an admitted
  kernel-side witness for gfx1151. Profiler counters are **unverified**: there is
  no `/dev/kfd` under WSL2, so `rocprofv3` returns no records.
- **gfx1151 only.** No gfx1201 evidence transfers here, and none of this transfers
  to gfx1201.
