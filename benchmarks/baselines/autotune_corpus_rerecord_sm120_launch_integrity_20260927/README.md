# sm_120 rows re-recorded after the launch-integrity changes — 2026-09-27

Sync `AUTOTUNE-LAUNCH-INTEGRITY-2026-09-27` ([NVIDIA queue](../../../docs/audit/backend/nvidia/todo.md)).
Follows [`autotune_corpus_rerecord_sm120_followups_20260927/`](../autotune_corpus_rerecord_sm120_followups_20260927/README.md);
same recorder, finalizer and shape lists. Logs are `.txt` because the repo
ignores `*.log`.

**Why every sm_120 row was re-measured (none backfilled).**

1. `NVIDIA-EMITTED-UNCHECKED-LAUNCH`: every emitted CUDA source now reads the
   last-error slot after each launch group and consumes every
   allocation/copy/memset/event status. That changes the emitted source, so
   the identity, of every emitted lane, the resident stages (composed lanes,
   paged-KV, conv2d) included.
2. The shipped `libtessera_nvidia_gemm` no longer embeds a DT_RUNPATH, so its
   bytes (its identity) changed: `sha256:198b85b3…` (was `b2191291…`).
3. `AUTOTUNE-SM120-ROUTE-RESOURCES`: the route-resource manifest gained the 17
   routes that had none, so the finalizer judges those winners on evidence.
4. `AUTOTUNE-KERNEL-IDENTITY-PAGED-KV`: the non-registry rows (`paged_kv_decode`,
   `conv2d`, `ssm_replay_decode`) now stamp route identities.

**Host, commit, trees.** The-Super-Bear (RTX 5070 sm_120, WSL2, driver 610.88,
CUDA 13.4 / nvcc 13.4.59). Own worktree `~/programming/tessera-w-a` detached at
`fcfa3677`, clean (`sm120_host.txt`). `build/` (gcc; `-DTESSERA_ENABLE_CUDA=ON
-DTESSERA_BUILD_NVIDIA_BACKEND=ON -DTESSERA_CUDA_ARCH=sm_120` + EBM, Clifford,
GPU bench, NVTILE, NVIDIA build tools, x86) supplies the shipped GEMM;
`build-nvidia-cuda/` (the release gate's configure: clang 23,
`/usr/local/cuda/bin/nvcc`, `sm_120a`, same toggles) supplies the PTX bridge
(`sha256:1bcb4967…`, unchanged) and `tessera-nvidia-opt`. Both configured from
scratch and fully built; `ninja -n` reported no work in both. 0 compute apps
before recording. All device work ran under `flock /tmp/tessera-timing.lock`.

## Route resources (`AUTOTUNE-SM120-ROUTE-RESOURCES`)

`nvidia_sm120_test5_route_resources.json` attests, per arbiter route, the Nsight
Compute launch facts of the kernels the route launches (registers per thread,
static/dynamic shared memory, theoretical/achieved occupancy, local/shared
spill requests), normalized by `parse_ncu_resources.py` into a fingerprint.
The finalizer marks a stable winner `selector_eligible` only when its route has
an entry. The original capture profiled many routes in one process and mapped
kernels to routes by name, which cannot separate the tf32/fp8_e4m3/fp8_e5m2
builds of one lane (all compile a kernel named
`tessera_nvidia_mma_fused_kernel`) and would have filed the shipped-GEMM
probe's `gemm` launch under a composed route.

Same method, one route per report: `benchmarks/nvidia/capture_route_resources.sh`
runs `profile_route_resources.py` under `ncu --profile-from-start off --set full`
for each route; the target warms the route, then brackets exactly one timed
invocation (`warmup=0, reps=1`, the entry and launch configuration the device
rows time) with `cuProfilerStart/Stop`.
`build_test5_resource_manifest.py --base … --route NAME=payload.json` adds each
report's kernels to its route (refusing an empty report or an existing route).
Now that these routes are committed, a plain re-run of
`capture_route_resources.sh` refuses up front, before any capture (review fix
on PR #868); `capture_route_resources.sh --refresh OUT_DIR` re-captures them
and replaces exactly their entries and tagged sources, keeping every other
route (`tests/unit/test_nvidia_route_resource_manifest.py`).
The normalized per-route payloads are in `route_resources/` (report names and
sha256 in the manifest's `sources`). 17 routes added: the native
`nvidia_mma_{fused,attn,gated}_{tf32,fp8_e4m3,fp8_e5m2}` (9), the composed
`nvidia_mma_{fused,attn,gated}_composed_fp8_{e4m3,e5m2}` (6), and the scalar
`nvidia_flash_attn` / `nvidia_gated`. No values were hand-written; the 19
existing routes are unchanged. Every report has complete spill evidence and no
spills.

Observed while capturing: under `ncu` the processes that loaded a
generic-lane library aborted in glibc at exit ("double free or corruption")
after the report was written; the same process exits 0 without the profiler.
The target now ends with `os._exit` after flushing (commented in the script).

## Re-record

`record_autotune_corpus.py --fused-shapes 64x64x64 256x256x256 128x512x256
127x259x63 128x256x256 --attention-shapes 128x128x64x64 64x512x64x64 64x256x64x64`
twice, each into an empty corpus (`sm120_record_run{1,2}.txt`, rc 0, no
refusal: every timed candidate, conv2d routes included, carried an identity);
`finalize_test5_corpus.py --base <committed> --resources <new manifest>`
(rc 0, so the two runs agreed on the toolchain digest and every identity);
`benchmark_serving.py --update-corpus` at its defaults for the paged-KV and
ReplaySSM rows (`sm120_serving.txt`, `sm120_serving_rows.json`); then
`summarize_rerecord.py` restored the prior record order
(`sm120_finalize_summary.txt`):

- no key lost or added; 108 sm_120 rows changed (96 registry, 2 conv2d, 6
  paged-KV, 4 ReplaySSM); the 16 gfx1151 rows byte-identical here (the 8
  gfx1151 paged-KV rows were re-recorded separately on Princess-Luna,
  [`autotune_corpus_rerecord_gfx1151_paged_kv_20260927/`](../autotune_corpus_rerecord_gfx1151_paged_kv_20260927/README.md));
- 0 registry rows race an unmeasured candidate;
- **selector-eligible registry rows: 91 (was 38)**;
- in the final committed corpus every timed candidate of all 124 rows carries
  an identity (0 unstamped).

`record_autotune_reproducibility.py` against the new corpus and manifest
(`sm120_reproducibility.txt`, `../nvidia_sm120_autotune_reproducibility.json`):
92 strict records considered, 92 admitted, all four stale mutations rejected,
kernel cache reproducible.

## Serve and miss (`check_emitted_identity.py`)

Fresh process, same tree, under the lock (`sm120_serve_and_miss_check.txt`,
rc 0):

- registry identities match 96/96; 0 rows with an untimed live candidate;
  inferred-dims and explicit-dims answers agree on all 96;
- **served by production lookup: 20 registry rows (was 13)**;
- route rows (paged-KV, conv2d, ReplaySSM) match their live route identities
  12/12; the sm_120 paged-KV warm start serves the 128-token device row
  (`fused`), the others are unseparated;
- each emitter perturbed alone with pins unchanged misses some rows (resident
  stages: 68, ReplaySSM ring: 4, …); every one of the 108 rows misses under at
  least one, and with all perturbed 108/108 miss and 0 are served.

**Shipped GEMM from a different tree and configure** (`sm120_serve_check_other_gemm_tree.txt`,
rc 0): the same check with `TESSERA_NVIDIA_GEMM_LIB` pointing at the library
built in `~/programming/tessera-w-a2/build-r1` (another worktree, configured
with `CUDAToolkit_ROOT=/usr/local/cuda` — the configure that used to produce
different bytes) gives the identical result: 96/96 match, 20 served.

### Served registry rows (20)

| op | dtype | timing | bucket | winner |
|---|---|---|---|---|
| matmul | f16, bf16 | device | 512³, 1024³, 2048³ | `nvidia_mma_gemm_emitted` (6 rows) |
| fused_region | f16 | end-to-end | 64³, 256³, 128x512x64, 128x256x256, 128x512x256 | `nvidia_mma_fused` (5 rows) |
| attention | f16 | end-to-end | 128x128x64x64 | `nvidia_mma_attn` |
| attention | fp8_e5m2 | device + end-to-end | 128x128x64x64 | `nvidia_mma_attn_fp8_e5m2` (2) |
| attention | fp8_e4m3 | end-to-end | 64x512x64x64 | `nvidia_flash_attn` |
| fused_region | fp8_e4m3, fp8_e5m2 | device | 64³ | `nvidia_mma_fused_fp8_*` (2) |
| gated_matmul | f32 | device | 128x512x512 | `nvidia_mma_gated_tf32` |
| gated_matmul | fp8_e5m2 | device | 128x512x512, 64x256x256 | `nvidia_mma_gated_fp8_e5m2` (2) |

No longer served: `matmul` f16 and bf16 end-to-end 2048³ (`nvidia_mma_gemm_shipped`,
winner unchanged; re-measured margin 39.6% / 58.8% against noise 33.8% /
41.4%, under the 2x separation factor).

### The 20 formerly partial rows

All 20 now have resource evidence for their winner (the blocker the follow-ups
recorded). **4 are served**: attention fp8_e5m2 device 128x128, gated tf32 device
128x512x512, gated fp8_e5m2 device 128x512x512 and 64x256x256. The other 16
stay out on evidence: 15 are unseparated (margins 3–80% against device-event
noise of 33–332% between whole repeats: 5 fused tf32, 7 attention, 3 gated),
and 1 (attention fp8_e5m2 device 64x512) had the two
runs disagree on the winner. More device repeats, or a device-clock witness,
is what would separate them; they were not re-raced to recover them.

### Winner changes (11 of 96), none of them a served row

| op | dtype | timing | bucket | before -> after |
|---|---|---|---|---|
| attention | f16 | end_to_end | 64x256x64x64 | `nvidia_flash_attn` -> `nvidia_mma_attn` |
| attention | f32 | device | 64x256x64x64 | `nvidia_mma_attn_composed_tf32` -> `nvidia_mma_attn_tf32` |
| matmul | bf16 | device | 128x256x64 | `nvidia_mma_gemm_emitted` -> `nvidia_tile_matmul_direct` |
| matmul | bf16 | end_to_end | 128x256x64 | `nvidia_tile_matmul_direct` -> `nvidia_tile_matmul_shared` |
| matmul | f16 | device | 128x256x64 | `nvidia_tile_matmul_direct` -> `nvidia_mma_gemm_emitted` |
| matmul | f16 | end_to_end | 128x256x64 | `nvidia_tile_matmul_shared` -> `nvidia_mma_gemm_emitted` |
| matmul | bf16 | device | 128x512x64 | `nvidia_tile_matmul_shared` -> `nvidia_tile_matmul_direct` |
| matmul | bf16 | end_to_end | 128x512x64 | `nvidia_tile_matmul_direct` -> `nvidia_tile_matmul_shared` |
| matmul | bf16 | end_to_end | 256³ | `nvidia_tile_matmul_direct` -> `nvidia_mma_gemm_emitted` |
| matmul | bf16 | device | 64³ | `nvidia_mma_gemm_emitted` -> `nvidia_tile_matmul_direct` |
| matmul | f16 | end_to_end | 64³ | `nvidia_tile_matmul_direct` -> `nvidia_mma_gemm_emitted` |

As before, every change is at a small or ragged shape and inadmissible before
and after.

## Shipped GEMM byte-reproducibility

`gemm_build_before_fix.txt`, `gemm_build_trees.txt`: the same source built in
two worktrees with one configure gave identical bytes (`b2191291…`), so tree
paths were never the variable. What differed was how CMake discovered CUDA:
with `CUDAToolkit_ROOT=/usr/local/cuda` the library was `a00c9040…` — exactly
the digest the 2026-09-27 emitted-identity rows stamped. A section-by-section
comparison of the two: `.text`, `.rodata`, `.data`, `.eh_frame`, symbol tables
identical; only `.dynstr` (the DT_RUNPATH string: `/usr/local/cuda/lib64` vs
`/usr/local/cuda-13.4/targets/x86_64-linux/lib`), `.dynamic`, the build-id note
and the offsets after `.dynstr` differed. The runpath came from linking the
imported `CUDA::nvrtc` target. Every loader preloads the driver and NVRTC
before opening the library, so it now builds with `SKIP_BUILD_RPATH`.
`gemm_build_after_fix.txt`: four builds — two worktrees, the default configure
and the `CUDAToolkit_ROOT=/usr/local/cuda` configure — are byte-identical
(`198b85b3…`, no RUNPATH). The identity scheme is unchanged (content digest);
the rows were re-recorded because the bytes changed. A clang-built library
(the release gate's `build-nvidia-cuda/`) is different host code and remains a
different identity; the runtime loads `build/` first.

## Stated limits

- Rows are bound to CUDA 13.4 / driver 610.88 / LLVM 23.1.1, to the bridge and
  GEMM contents above, to `TESSERA_NVIDIA_ARCH` (`sm_120a`) and to the emitted
  sources.
- Route resources are keyed by route, not by code identity or storage: the
  pre-existing entries (e.g. `nvidia_mma_fused`, captured at f16) stand for the
  bf16 build too, and none is re-captured when a kernel changes. The new
  entries were captured from the code these rows timed.
- The shipped f16 GEMM runs the adjacent AOT cubin when present; the library
  identity covers its source (embedded, with its sha256, which the loader
  checks) but not the cubin's nvcc flags.
- CUDA events are selection hints only (`WSL-TIMING-ADMISSION-2026-09-26`).

## Release gate

`scripts/run_nvidia_release_gate.sh --layer device` at `71e1e81e` (this
evidence commit's successor, docs only) on the same box and worktree, under the
timing lock, `TESSERA_NVIDIA_REPORT_DIR=~/gate-reports/a-71e1e81ea`: both
passes **1167 passed, 1 skipped, 0 failed** (junit 1168 tests each); the skip
is NCCL not installed, so the multi-rank topology lane cannot be evaluated
here. `status=success`. The bridge and `build/` GEMM digests are unchanged
after the gate's re-configure.
