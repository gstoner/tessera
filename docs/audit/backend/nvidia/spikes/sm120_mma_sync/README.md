---
last_updated: 2026-06-24
audit_role: index
---

# Spike #6 — sm_120 `mma.sync.aligned.m16n8k16` PTX, proven on silicon

**Status:** PASS, end-to-end on real hardware (2026-06-25).
**Box:** NVIDIA GeForce RTX 5070 Ti (consumer Blackwell, CC 12.0), driver 610.62 /
CUDA UMD 13.3, toolkit `nvcc`/`ptxas` 13.3.33, WSL2. 70 SMs.

This spike proves the consumer-Blackwell GEMM instruction path — warp-level
`mma.sync` (NOT Hopper warpgroup `wgmma`) — clears Tessera's full rung ladder:
**emit → assemble → load → launch → execute-and-compare**. It is the
fallback/oracle path for NVIDIA action item #6 and de-risks the productization
in `../../BLACKWELL_SM120_EXECUTION_PLAN.md` (Sequencing, step 3).

The tile: one warp computes `D[16x8] f32 = A[16x16] bf16 (row-major) · B[16x8]
bf16 (col-major)` with f32 accumulation, using
`mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32`.

## Files

| File | Rung | What it proves |
|------|------|----------------|
| `smem_query.cu` | — | Device-query that resolved #14 (per-SM smem = 102400 B = 100 KiB; per-block opt-in = 101376 B = 99 KiB; the 128 KB release-note figure is the unified data cache, not the carve-out). |
| `mma_m16n8k16.cu` | 3–5 | CUDA inline-asm oracle: nails the m16n8k16 fragment layout, executes, matches CPU ref. Also the source for the reference PTX (`nvcc -ptx`). |
| `tessera_mma_m16n8k16.ptx` | 2.5 | **Hand-emitted, Tessera-style raw PTX** (`.version 9.3`, `.target sm_120a`) — what `ptx_emit.py`'s sm_120 path should produce. |
| `run_ptx.cpp` | 3+5 | Driver-API harness: `cuModuleLoadDataEx` (JIT ptxas) + `cuLaunchKernel`, execute-and-compare vs a bf16-rounded CPU reference. Proves the *emitted PTX itself* runs. |
| `nvgemm_proto.cpp` | 3-5 | **General** tiled/K-looped mma.sync GEMM (arbitrary M/N/K, ragged zero-padded) via NVRTC, for bf16 + f16. The basis for the shipped `libtessera_nvidia_gemm.so`. |
| `nvgemm_dtypes.cpp` | 3-5 | Multi-dtype sweep via NVRTC — **bf16, f16, tf32, fp8 e4m3, fp8 e5m2** — each with its own MMA shape + fragment layout, across 7 shapes. Execute-and-compare vs a host reference that quantizes inputs to the same dtype the hardware consumes. |

## Multi-dtype GEMM sweep (nvgemm_dtypes.cpp)

All five Tensor-Core input dtypes the CC 12.0 matrix advertises were validated
end-to-end on the RTX 5070 Ti (NVRTC-compiled `compute_120`, execute-and-compare
across 7 shapes incl. ragged 17x9x31 / 100x50x200):

| dtype | MMA instruction | worst-shape max abs err |
|-------|-----------------|-------------------------|
| bf16  | `mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32` | 9.5e-6 |
| f16   | `mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32`   | 1.7e-5 |
| tf32  | `mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32`  | 2.5e-5 |
| fp8 e4m3 | `mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32` | 0 (bit-exact) |
| fp8 e5m2 | `mma.sync.aligned.m16n8k32.row.col.f32.e5m2.e5m2.f32` | 0 (bit-exact) |

Fragment layouts differ per K (16-bit: m16n8k16, packs 2/reg; tf32: m16n8k8, one
tf32/reg; fp8: m16n8k32, packs 4 bytes/reg). The fp8 paths are bit-exact because
the host reference dequantizes with the same OCP fp8 decode the hardware uses and
the f32 accumulation of those coarse values stays exactly representable.

## NVFP4 block-scaled MMA (m16n8k64) — partial, grounded (action item #9)

The consumer-Blackwell headline: warp-level **block-scaled** NVFP4 (e2m1 data +
ue4m3 per-16-block scale). Files: `nvfp4_probe.cu` (assemble probe),
`nvfp4_gemm.cu` (per-lane fragment harness + scale sweep).

**Grounded on-silicon (ptxas/runtime are the source of truth):**
- The warp instruction
  `mma.sync.aligned.m16n8k64.row.col.kind::mxf4nvf4.block_scale.scale_vec::4X.f32.e2m1.e2m1.f32.ue4m3`
  **assembles and executes on sm_120a** (RTX 5070 Ti). The operand grammar that
  ptxas accepts: `{d4}, {a4}, {b2}, {c4}, {sfa}, {byteid_a, tid_a}, {sfb}, {byteid_b, tid_b}`
  (scale data = one .b32 per thread; byte/thread ids are immediates).
- It is **arch-specific**: only the `sm_120a` SASS target accepts it. The base
  `compute_120`/`sm_120` PTX target rejects `.kind::mxf4nvf4` / `.block_scale` /
  `.scale_vec::4X` — so build with `-gencode arch=compute_120a,code=sm_120a`
  (an `-arch=sm_120a` build that also emits base PTX fails at the PTX stage).

**NOT yet verified (needs the PTX ISA block-scale spec, absent from the on-box
CUDA 13.3 headers — only the datacenter `tcgen05` block-scale variant ships
there):** the numerics. A scale-byte sweep shows the scale is multiplicative
(`0x00` → result ≈ 0) but the `ue4m3` scale **encoding** and the
`{byte-id, thread-id}` **scale-distribution** semantics are non-standard (the
e4m3 code for 1.0, `0x38`, yields ~3e9 not unit), so a guessed layout does not
reproduce a reference. Per the grounding rule we do **not** claim a working
NVFP4 GEMM. Next step: obtain the PTX ISA §"mma with block scaling" operand/scale
mapping, then finish `nvfp4_gemm.cu`'s reference match (the e2m1 data fragment
layout + selector semantics).

## Result

```
mma.sync.m16n8k16 single-tile GEMM (16x16x8, bf16->f32) on sm_120
max abs error vs CPU ref = 4.76837e-07, elements over 1e-2 = 0 / 128
RESULT: PASS (emitted PTX assembled+launched+matched reference)
```

f32 epsilon agreement — the instruction is numerically correct on sm_120.

## Reproduce

```bash
export PATH=/usr/local/cuda-13.3/bin:$PATH
export LD_LIBRARY_PATH=/usr/lib/wsl/lib:/usr/local/cuda-13.3/lib64:$LD_LIBRARY_PATH

# #14 smem device-query
nvcc -arch=sm_120 -o smem_query smem_query.cu && ./smem_query

# Rungs 3-5 via nvcc inline-asm oracle
nvcc -arch=sm_120 -o mma_tile mma_m16n8k16.cu && ./mma_tile

# Rung 3: assemble the hand-emitted PTX
ptxas --gpu-name=sm_120a -o tessera_mma.cubin tessera_mma_m16n8k16.ptx

# Rung 5: load + launch the emitted PTX via the Driver API
g++ -O2 -I/usr/local/cuda-13.3/include -o run_ptx run_ptx.cpp -L/usr/lib/wsl/lib -lcuda
./run_ptx tessera_mma_m16n8k16.ptx
```

## Gotchas (carry into productization)

1. **PTX must be ASCII.** The driver's JIT `ptxas` (`cuModuleLoadDataEx`) aborts on
   a non-ASCII byte (e.g. an em-dash in a comment) that standalone `ptxas`
   silently tolerates. `ptx_emit.py` should emit ASCII-only.
2. **Zero the accumulator.** The f32 `{d0..d3}` registers are the `mma` C operand;
   they must be explicitly `mov.f32 ..., 0f00000000` before the instruction.
3. **CUDA 13.3 `cuCtxCreate` is v4** (takes `CUctxCreateParams*`). Host harnesses
   should use `cuDevicePrimaryCtxRetain` + `cuCtxSetCurrent`.
4. **Fragment layout** (m16n8k16, 16-bit A/B): with A row-major and B col-major,
   each packed `.b32` register is a single contiguous `ld.global.b32` — the two
   sub-elements (A: col,col+1; B: row,row+1) are adjacent in memory, so no
   explicit packing is needed.

## Next (productization — execution-plan step 3, not done here)

- Add `emit_mma_sync_matmul_ptx(...)` sm_120 path to `python/tessera/compiler/ptx_emit.py`.
- Register a CUDA launcher via `tsrRegisterGpuLauncher` (mechanism proven on Metal).
- Land an `mma.sync` hardware-smoke + execute-and-compare oracle test → first real
  NVIDIA `backend_kernel` proof row.
