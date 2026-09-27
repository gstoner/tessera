# Autotune corpus re-record under the toolchain-keyed schema — 2026-09-26

Recorder output behind the `benchmarks/baselines/autotune_corpus.json` rows re-recorded after
Decision #11's toolchain key landed (sync `AUTOTUNE-TOOLCHAIN-KEY-2026-09-26`; see the rocm and
nvidia todos). Logs are `.txt` because the repo ignores `*.log`.

- `gfx1151_paged_kv.{json,txt}` — `benchmarks/rocm/record_paged_kv_corpus.py`, the 8
  `paged_kv_decode` rows. Princess-Luna (gfx1151, WSL2), clean worktree at `82d6f1fa`, every
  timing run under `flock /tmp/tessera-timing.lock`. Winners unchanged from the pre-schema rows.
- `gfx1151_fused_separation.txt` — `benchmarks/rocm/record_autotune_separation.py`, the 8
  `fused_region` rows, same host and commit. **Superseded by the schema review fixes:**
  `rocm_wmma_gemm` is now keyed on the `tessera-opt` binary's digest, which these rows do not
  carry, so they miss and fall back to a live race until re-recorded (owed; see the rocm todo).
- `sm120_record_run{1,2}.txt` — `benchmarks/nvidia/record_autotune_corpus.py`, two fresh runs
  (each to its own `TESSERA_AUTOTUNE_CORPUS` file) with the todo's `--fused-shapes` /
  `--attention-shapes`, finalized by `finalize_test5_corpus.py` against the committed corpus
  and `nvidia_sm120_test5_route_resources.json`: the 92 recorder keys plus 6 new composed
  `end_to_end` keys. The-Super-Bear (RTX 5070, WSL2, CUDA 13.4 / driver 610.88), clean worktree
  at `4e699f7e`, every timing run under `flock /tmp/tessera-timing.lock`.
- `sm120_serving.json` — `benchmarks/nvidia/benchmark_serving.py --update-corpus`, same host
  and commit: the 5 `paged_kv_decode` / `ssm_replay_decode` device keys plus 5 new
  `end_to_end` keys. Fused and staged paged routes are now measured interleaved over 20
  reps, and each row carries its separation verdict, so the paged-attention warm start is
  served again.
- `sm120_reproducibility.txt` — `record_autotune_reproducibility.py`: 38 selector-eligible
  sm_120 rows, all admitted strictly; the refreshed
  `nvidia_sm120_autotune_reproducibility.json` is its output.
- The file was re-ordered to the prior committed record order after finalizing (content
  verbatim), so the 16 gfx1151 rows are textually untouched.
