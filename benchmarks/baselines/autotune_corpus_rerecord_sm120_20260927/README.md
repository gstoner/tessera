# sm_120 registry rows re-recorded under the emitted-code identity — 2026-09-27

Sync `AUTOTUNE-EMITTED-IDENTITY-2026-09-27`; closes the NVIDIA queue's owed item
`AUTOTUNE-EMITTED-IDENTITY-SM120-RERECORD`. Logs are `.txt` because the repo ignores `*.log`.

**Why.** Every arbiter candidate, of every tier, must now carry an identity of the code it runs,
or a verdict involving it misses (`compiler/emitted_code_identity.py`, normalization
`tessera.emitted_source.v1`; `autotune._record_matches_live_delegates`). The 96 `nvidia:sm_120`
registry rows committed before this (`matmul` / `fused_region` / `attention` / `gated_matmul`)
stamped only `nvidia_mma_gemm_shipped`, so none could be served. They were re-measured, not
backfilled.

**Host, commit, trees.** The-Super-Bear (RTX 5070 sm_120, WSL2, driver 610.88, CUDA 13.4 /
nvcc 13.4.59 at `/usr/local/cuda`), fresh worktree `~/programming/tessera-eid` detached at
`1a737129` (branch `claude/autotune-emitted-identity`), clean (`git status --porcelain` empty),
with two build trees configured from scratch in it and built with `ninja` to completion (rc 0;
`ninja -n` "no work to do" before recording):

- `build/` — `-DTESSERA_ENABLE_CUDA=ON -DTESSERA_BUILD_NVIDIA_BACKEND=ON -DTESSERA_CUDA_ARCH=sm_120`
  (+ EBM, Clifford, GPU bench, NVTILE, NVIDIA build tools). Supplies the shipped
  `libtessera_nvidia_gemm.so` (`runtime._nvidia_gemm_lib_path` looks in `build/` first).
- `build-nvidia-cuda/` — the same with `TESSERA_CUDA_ARCH=sm_120a`, clang 23 host compiler and
  nvcc 13.4. Supplies `libtessera_nvidia_ptx_launch.so` and `tessera-nvidia-opt` (both looked up
  there first).

The bridge library is byte-identical to the one in `~/programming/tessera-timing-nv` (a
different tree at a different commit with the same bridge source), and so is `build/`'s GEMM
library: these content digests are reproducible across trees, not per-build.

**Recorder and timer.** `benchmarks/nvidia/record_autotune_corpus.py` with the NVIDIA queue's
shape lists (`--fused-shapes 64x64x64 256x256x256 128x512x256 127x259x63 128x256x256
--attention-shapes 128x128x64x64 64x512x64x64 64x256x64x64`, other defaults), twice, each run
into its own empty `TESSERA_AUTOTUNE_CORPUS`, each under `flock /tmp/tessera-timing.lock`, with
`scripts/_nvidia_env.sh` sourced and no other GPU process (`nvidia-smi` compute apps empty).
Device rows are timed by each candidate's `measure_device_latency` (CUDA events, operands
resident; 100 reps, 10 warm-up, 10 whole repeats); end-to-end rows by host wall clock
(`measure_latency_samples`, 20 reps, 3 warm-up). These are arbiter selection hints, not
performance promotions: under `WSL-TIMING-ADMISSION-2026-09-26` CUDA events alone never qualify
a promotion. Then `finalize_test5_corpus.py` against the committed corpus and
`nvidia_sm120_test5_route_resources.json` (rc 0: the two runs agreed on the toolchain digest and
every stamped identity at every key, which the finalizer now requires), and the output
re-ordered to the prior record order.

- `sm120_record_run{1,2}.txt` — recorder output. Neither refused: every timed candidate of
  every registry row carried an identity (the recorder refuses to write otherwise).
- `sm120_finalize_summary.txt` — host/toolchain line and the keyed comparison against the
  prior corpus: no key lost or added; 98 rows changed (the 96 registry rows plus the 2 `conv2d`
  rows the recorder also re-measures); the 16 `rocm:gfx1151` rows and the 10 serving rows
  (`paged_kv_decode` / `ssm_replay_decode`, `benchmark_serving.py`) byte-identical; 0 unstamped
  timed candidates; 31 registry rows selector-eligible (was 38).
- `sm120_reproducibility.txt` — `record_autotune_reproducibility.py`: 31 selector-eligible rows
  considered, 31 admitted strictly; `nvidia_sm120_autotune_reproducibility.json` refreshed.
- `sm120_serve_and_miss_check.txt` + `check_emitted_identity.py` — a fresh process in the same
  worktree after the re-record, under the lock. For all 96 rows every timed live candidate's
  identity equals its stamp. **15 rows are served** by `corpus_winner` asked the way
  `run_arbitrated` asks (no explicit dims) — exactly the 15 admissible ones: 7 `matmul` device
  -> `nvidia_mma_gemm_emitted` (f16 256/512/1024/2048, bf16 512/1024/2048), 2 `matmul`
  end-to-end 2048 -> `nvidia_mma_gemm_shipped`, 5 `fused_region` f16 end-to-end ->
  `nvidia_mma_fused`, 1 `attention` f16 end-to-end 128x128x64x64 -> `nvidia_mma_attn`. Each
  NVIDIA emitter is then perturbed in turn with the toolchain digest asserted unchanged
  (`_synthesize_{fused,attention,gated,mma_fused,mma_attn,mma_gated,resident_ops}_cuda`,
  `ptx_emit.emit_mma_sync_gemm_ptx`, the `tessera-nvidia-opt` Tile PTX); every row misses under
  at least one of them, and with all perturbed 96/96 miss and 0 are served.

**Winner changes (15 of 96), all within the re-measured field:**

| op | dtype | timing | bucket | before -> after |
|---|---|---|---|---|
| attention | f16 | end_to_end | 64x256x64x64 | `nvidia_mma_attn` -> `nvidia_flash_attn` |
| attention | f32 | device | 64x256x64x64 | `nvidia_mma_attn_composed_tf32` -> `nvidia_mma_attn_tf32` |
| attention | fp8_e4m3 | device | 64x512x64x64 | `nvidia_mma_attn_composed_fp8_e4m3` -> `nvidia_mma_attn_fp8_e4m3` |
| gated_matmul | fp8_e4m3 | end_to_end | 128x512x512 | `nvidia_mma_gated_fp8_e4m3` -> `nvidia_mma_gated_composed_fp8_e4m3` |
| matmul | bf16 | end_to_end | 128x256x64 | `nvidia_tile_matmul_shared` -> `nvidia_tile_matmul_direct` |
| matmul | f16 | end_to_end | 128x256x64 | `nvidia_tile_matmul_direct` -> `nvidia_tile_matmul_shared` |
| matmul | bf16 | device | 128x512x64 | `nvidia_tile_matmul_direct` -> `nvidia_tile_matmul_shared` |
| matmul | bf16 | end_to_end | 128x512x64 | `nvidia_tile_matmul_direct` -> `nvidia_tile_matmul_shared` |
| matmul | f16 | device | 128x512x64 | `nvidia_tile_matmul_direct` -> `nvidia_mma_gemm_shipped` |
| matmul | f16 | end_to_end | 128x512x64 | `nvidia_tile_matmul_direct` -> `nvidia_tile_matmul_shared` |
| matmul | bf16 | end_to_end | 256x256x256 | `nvidia_mma_gemm_emitted` -> `nvidia_tile_matmul_direct` |
| matmul | bf16 | device | 64x64x64 | `nvidia_mma_gemm_emitted` -> `nvidia_tile_matmul_direct` |
| matmul | bf16 | end_to_end | 64x64x64 | `nvidia_tile_matmul_direct` -> `nvidia_mma_gemm_emitted` |
| matmul | f16 | device | 64x64x64 | `nvidia_tile_matmul_direct` -> `nvidia_tile_matmul_shared` |
| matmul | f16 | end_to_end | 64x64x64 | `nvidia_tile_matmul_shared` -> `nvidia_mma_gemm_emitted` |

(`128x512x64` is the power-of-two bucket of the 127x259x63 shape.) None of the 15 rows whose
winner changed is among the 15 served: every change is at a small or ragged shape whose
verdict is unseparated or finalizer-ineligible, i.e. inside noise, in both the old and new
corpus.

**Stated limits, read before citing these rows.**

- 20 device rows (`attention` / `gated_matmul` for f32/fp8, `fused_region` f32) race a scalar
  lane with no device timer (`nvidia_flash_attn`, `nvidia_gated`, `nvidia_generic_cuda`); it is
  recorded `unmeasured`, never timed and so never stamped, and production refuses those rows
  because they did not race the live field. That predates this change; the check lists them
  as `partial_field` rather than as identity mismatches.
- `gated_matmul` rows are keyed on the recorder's explicit `(M, H, K)` dims, but
  `autotune._infer_dims` has no gated rule, so ordinary `run_arbitrated` dispatch (no dims)
  cannot find them. No gated row is admissible today, so this changes no served count; it is
  recorded as a follow-up in the NVIDIA queue.
- The rows are bound to this toolchain (CUDA 13.4 / driver 610.88 / LLVM 23.1.1 pins), to the
  bridge and shipped-GEMM library contents above, to the host `TESSERA_NVIDIA_ARCH` (default
  `sm_120a`, part of the nvcc build line), and to the emitted source. A tree whose bridge or GEMM
  library differs in content misses (a false miss at worst).
