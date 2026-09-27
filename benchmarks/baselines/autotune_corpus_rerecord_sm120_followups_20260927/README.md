# sm_120 registry rows re-recorded after the autotune follow-ups — 2026-09-27

Sync `SM120-AUTOTUNE-FOLLOWUPS-2026-09-27`. Follows
[`autotune_corpus_rerecord_sm120_20260927/`](../autotune_corpus_rerecord_sm120_20260927/README.md)
(same recorder, finalizer and check; read that README for the method). Logs are
`.txt` because the repo ignores `*.log`.

**Why a re-record.** Three changes on this branch touch every sm_120 registry row:

1. The emitted CUDA templates follow the stale-error rule (clear the last-error
   slot once on entry), and the arbiter-raced lanes now check each launch group
   through the slot. That changes the emitted source, so the identity, of
   every emitted NVIDIA lane.
2. `nvidia_generic_cuda`, `nvidia_flash_attn` and `nvidia_gated` got a CUDA-event
   device timer. Before, they were recorded `unmeasured` in 20 device rows, and
   production refused those rows as partial-field races.
3. `autotune._infer_dims` got a `gated_matmul` rule, so ordinary dispatch can
   find the 12 gated rows.

The rows were re-measured, not backfilled.

**Host, commit, trees.** The-Super-Bear (RTX 5070 sm_120, WSL2, driver 610.88,
CUDA 13.4 / nvcc 13.4.59 at `/usr/local/cuda`). The recording ran in a fresh
worktree `~/programming/tessera-w-sm120-fu`, detached at `31a58bd4` (branch
`claude/sm120-autotune-followups`) and clean (`dirty=0` in each run log). Both
build trees were configured from scratch in that worktree and built to
completion; `ninja -n` reported "no work to do" in both before recording.

- `build/` uses gcc with `-DTESSERA_ENABLE_CUDA=ON -DTESSERA_BUILD_NVIDIA_BACKEND=ON
  -DTESSERA_CUDA_ARCH=sm_120`, plus EBM, Clifford, GPU bench, NVTILE and NVIDIA
  build tools. It supplies the shipped `libtessera_nvidia_gemm.so`
  (`sha256:b2191291…`). Runtime resolution was checked to point here.
- `build-nvidia-cuda/` uses the release gate's own configure arguments (clang 23,
  `/usr/local/cuda/bin/nvcc`, `sm_120a`) plus the same toggles, so the gate's
  re-configure is a no-op. It supplies `libtessera_nvidia_ptx_launch.so`
  (`sha256:1bcb4967…`, byte-identical to the bridge the previous re-record
  stamped) and `tessera-nvidia-opt`.

The shipped GEMM library here differs in content from the one in the previous
recording tree (`~/programming/tessera-eid`, `sha256:a00c9040…`). Rows that
raced it are bound to this build. A tree with a different GEMM build misses,
which is a false miss at worst.

**Recorder and timer.** `benchmarks/nvidia/record_autotune_corpus.py
--fused-shapes 64x64x64 256x256x256 128x512x256 127x259x63 128x256x256
--attention-shapes 128x128x64x64 64x512x64x64 64x256x64x64`, with other
arguments at their defaults. It ran twice. Each run wrote to its own empty
`TESSERA_AUTOTUNE_CORPUS`, ran under `flock /tmp/tessera-timing.lock` with
`scripts/_nvidia_env.sh` sourced, and started with 0 compute apps in
`nvidia-smi`. Device rows are timed by each candidate's
`measure_device_latency`: CUDA events with operands resident, 100 reps,
10 warm-up and 10 whole repeats. The three scalar lanes now use the same
method through their new `_device_ms` entries, which live in the same artifact
`run` launches. End-to-end rows use the host wall clock (20 reps, 3 warm-up).
These rows are arbiter selection hints, not performance promotions: under
`WSL-TIMING-ADMISSION-2026-09-26`, CUDA events alone never qualify a
promotion. `finalize_test5_corpus.py` ran against the committed corpus and
`nvidia_sm120_test5_route_resources.json` and returned rc 0, so the two runs
agreed on the toolchain digest and every stamped identity. The output was then
re-ordered to the prior record order.

- `sm120_record_run{1,2}.txt` hold the recorder output (rc 0). Neither run
  refused, so every timed candidate carried an identity.
- `sm120_finalize_summary.txt` compares the result, key by key, with the prior
  corpus:
  - no key was lost or added;
  - 98 rows changed (the 96 registry rows plus the 2 `conv2d` rows);
  - the 16 `rocm:gfx1151` rows and all serving rows are byte-identical;
  - 0 timed candidates are unstamped;
  - **0 registry rows have an `unmeasured` candidate** (was 20);
  - 38 registry rows are selector-eligible (was 31).
- `sm120_reproducibility.txt` / `../nvidia_sm120_autotune_reproducibility.json`:
  39 strict records considered, 39 admitted. The generic lane's
  `kernel_cache_key` moved with the template.
- `sm120_serve_and_miss_check.txt` + `check_emitted_identity.py` ran in a fresh
  process in the same tree, under the lock (rc 0):
  - identities match 96/96;
  - **0 rows have a live candidate they did not time**;
  - inferred-dims and explicit-dims lookups agree on all 96 rows;
  - **13 rows are served** (was 15, see below);
  - each single-emitter perturbation, with pins unchanged, makes some rows
    miss, every row misses under at least one, and with all emitters
    perturbed 96/96 miss and 0 are served.
- `sm120_emitted_stale_error_device_test.txt` is
  `tests/device/nvidia/test_emitted_stale_cuda_error.py` on this tree: 19
  passed. For 16 of the 18 lanes, the negative control (entry clears
  stripped) failed under the primed slot, as it must. The two mma.sync
  attention entries are masked, because a successful `cudaFuncSetAttribute`
  resets the slot (measured with a standalone probe on this box).

**Served rows: 15 before, 13 now.** The winners of the 13 served rows are
unchanged:

- 5 `fused_region` f16 end-to-end rows go to `nvidia_mma_fused`;
- 6 `matmul` device rows (f16/bf16 512, 1024, 2048) go to
  `nvidia_mma_gemm_emitted`;
- 2 `matmul` end-to-end 2048 rows go to `nvidia_mma_gemm_shipped`.

Two rows that were served are no longer served. Both keep their winner, and in
both the re-measured margin no longer clears the separation factor over the
noise:

| row | winner | margin / noise before | margin / noise now |
|---|---|---|---|
| `matmul` f16 device 256³ | `nvidia_mma_gemm_emitted` | 18.9% / 7.6% | 5.8% / 9.6% |
| `attention` f16 end-to-end 128x128x64x64 | `nvidia_mma_attn` | 25.2% / 8.2% | 19.5% / 16.4% |

No emitter behind the 256³ matmul race changed on this branch (PTX, Tile
PTX). The shipped GEMM is a different build of unchanged source, as noted
above. The attention row's two host entries gained one
host call each (the slot check). These are fresh measurements, not
re-selected ones; the rows were not re-raced to recover them.

**The 20 formerly partial-field rows are now fully raced, and still not
served.** The scalar lanes were timed in all 20 rows and lost every one by
3–750x. For example, `nvidia_gated` took 15.7 ms at 128x512x512 against
0.021–0.025 ms for the winner, and `nvidia_generic_cuda` took 0.15–3.8 ms against
0.008–0.013 ms. What still blocks serving is not the field:

- 18 of the 20 winners (native tf32/fp8 mma lanes) have no entry in
  `nvidia_sm120_test5_route_resources.json`, so the finalizer marks them
  `selector_eligible: false`;
- the other 2 (`nvidia_mma_attn_composed_tf32` at 64x256 and 64x512) are
  eligible but unseparated from the runner-up.

**The gated rows are reachable, and none is admissible.** Production lookup
without dims now finds all 12 gated rows, and its answer matches the
explicit-dims answer on each. Every gated winner is a native tf32/fp8 lane
with no route-resource entry, or its verdict is unseparated.

**Winner changes (10 of 96)** — none of them is a served row, before or after:

| op | dtype | timing | bucket | before -> after |
|---|---|---|---|---|
| attention | f32 | device | 64x256x64x64 | `nvidia_mma_attn_tf32` -> `nvidia_mma_attn_composed_tf32` |
| attention | fp8_e4m3 | device | 64x512x64x64 | `nvidia_mma_attn_fp8_e4m3` -> `nvidia_mma_attn_composed_fp8_e4m3` |
| fused_region | f32 | end_to_end | 256x256x256 | `nvidia_mma_fused_composed_tf32` -> `nvidia_mma_fused_tf32` |
| gated_matmul | fp8_e4m3 | end_to_end | 128x512x512 | `nvidia_mma_gated_composed_fp8_e4m3` -> `nvidia_mma_gated_fp8_e4m3` |
| matmul | bf16 | end_to_end | 128x512x64 | `nvidia_tile_matmul_shared` -> `nvidia_tile_matmul_direct` |
| matmul | f16 | device | 128x512x64 | `nvidia_mma_gemm_shipped` -> `nvidia_tile_matmul_direct` |
| matmul | f16 | end_to_end | 128x512x64 | `nvidia_tile_matmul_shared` -> `nvidia_tile_matmul_direct` |
| matmul | bf16 | device | 64x64x64 | `nvidia_tile_matmul_direct` -> `nvidia_mma_gemm_emitted` |
| matmul | f16 | device | 64x64x64 | `nvidia_tile_matmul_shared` -> `nvidia_mma_gemm_emitted` |
| matmul | f16 | end_to_end | 64x64x64 | `nvidia_mma_gemm_emitted` -> `nvidia_tile_matmul_direct` |

(`128x512x64` is the power-of-two bucket of the 127x259x63 shape.) As in the
previous re-record, every change is at a small or ragged shape, or is a
native-vs-composed pair whose verdict is unseparated or finalizer-ineligible.

**Release gate.** `scripts/run_nvidia_release_gate.sh --layer device` ran at
`4a275453` (this corpus commit) on the same box and worktree, under the timing
lock, with `TESSERA_NVIDIA_REPORT_DIR=~/gate-reports/sm120-fu-4a275453` (which
holds `device-correctness-{1,2}.xml`). Both passes: **1141 passed, 1 skipped,
0 failed** (junit: 1142 tests, 0 failures, 0 errors, 1 skipped). The skip is
NCCL not installed, so the multi-rank topology lane cannot be evaluated here.
`status=success`. The gate's re-configure of `build-nvidia-cuda/` was a no-op
for the bridge.

**Stated limits.**

- The rows are bound to this toolchain (the CUDA 13.4 / driver 610.88 /
  LLVM 23.1.1 pins), to the bridge and GEMM library contents above, to
  `TESSERA_NVIDIA_ARCH` (default `sm_120a`) and to the emitted source.
- The "fail without the fix" demonstration is the per-lane negative control
  inside the device test, which strips the entry clears from the emitter
  under test. It is not a run of `origin/main`. On main, most of these lanes
  never read the slot at all: their launches were unchecked, a different and
  worse failure. The standalone probe measured that on this box: after an
  invalid-configuration launch, `cudaDeviceSynchronize` returned success and
  the error sat only in the slot.
