# Autotune corpus re-record under the toolchain-keyed schema — 2026-09-26

Recorder output behind the `benchmarks/baselines/autotune_corpus.json` rows re-recorded after
Decision #11's toolchain key landed (sync `AUTOTUNE-TOOLCHAIN-KEY-2026-09-26`; see the rocm and
nvidia todos). Logs are `.txt` because the repo ignores `*.log`.

- `gfx1151_paged_kv.{json,txt}` — `benchmarks/rocm/record_paged_kv_corpus.py`, the 8
  `paged_kv_decode` rows. Princess-Luna (gfx1151, WSL2), clean worktree at `82d6f1fa`, every
  timing run under `flock /tmp/tessera-timing.lock`. Winners unchanged from the pre-schema rows.
- `gfx1151_fused_separation.txt` — `benchmarks/rocm/record_autotune_separation.py`, the 8
  `fused_region` rows, same host and commit. Superseded: these rows predate the review fix that
  keys `rocm_wmma_gemm` on the `tessera-opt` binary's digest.
- `gfx1151_fused_separation_opt_identity.txt` — the same recorder re-run for those 8 rows,
  Princess-Luna, same worktree clean at `59ecd215` (`ninja -C build` up to date), under the
  lock. The rows carried `delegate_identities.rocm_wmma_gemm` = that tree's `tessera-opt`
  binary digest (`sha256:701ee098…35f20c87`) and were served only there. Winners unchanged.
  Superseded by the kernel-code identity below.
- `gfx1151_fused_kernel_identity.txt` + `check_kernel_identity.py` — the 8 `fused_region`
  rows re-recorded with the **kernel-code identity** (`compiler/kernel_code_identity.py`,
  normalization `tessera.kernel_code.v1`): `rocm_wmma_gemm` is stamped with the digest of the
  normalized instruction stream and decoded kernel descriptor of the fused image it ran at that
  shape, not the `tessera-opt` binary. Princess-Luna, worktree `~/programming/tessera-kid`
  clean at `acedf002` on `claude/timing-foundation-kernel-identity`, two fresh build trees
  `build-a/` and `build-b/` (same configuration as the box's main tree), `scripts/_rocm_env.sh`
  sourced, every timing and serving run under `flock /tmp/tessera-timing.lock`. The log shows,
  in order: the two `tessera-opt` binaries differ (48 bytes, embedded build-tree paths); the 4
  fused images and their identities are identical across `build-a`, `build-b` and the
  `59ecd215` tree's `tessera-opt` (the images are byte-identical too: no section varied);
  before the re-record the committed rows were not served in `build-b`; the re-record in
  `build-a` (timer `device_event`, winners and separation unchanged); after it all 8 rows are
  served by `corpus_winner` and `measured_arbitrate` (re-measurement disabled) in `build-b`, with
  the `59ecd215` `tessera-opt`, and in `build-a`. Reproduce with the commands in the script's
  docstring.
- `sm120_record_run{1,2}.txt` — `benchmarks/nvidia/record_autotune_corpus.py`, two fresh runs
  (each to its own `TESSERA_AUTOTUNE_CORPUS` file) with the todo's `--fused-shapes` /
  `--attention-shapes`, finalized by `finalize_test5_corpus.py` against the committed corpus
  and `nvidia_sm120_test5_route_resources.json`: the 92 recorder keys plus 6 new composed
  `end_to_end` keys. The-Super-Bear (RTX 5070, WSL2, CUDA 13.4 / driver 610.88), clean worktree
  at `4e699f7e`, every timing run under `flock /tmp/tessera-timing.lock`.
- `sm120_serving.json` — `benchmarks/nvidia/benchmark_serving.py --update-corpus`: the 5
  `paged_kv_decode` / `ssm_replay_decode` device keys plus 5 `end_to_end` keys. **Re-recorded
  2026-09-26 at `9111b1b6`** after the review (untimed warm-up of both routes; one-sample
  spreads no longer read as zero noise); only these 10 rows changed. Fused and staged are
  measured interleaved over 20 reps and each row carries its separation verdict: 128 tokens
  separates (fused), 512 and 2048 do not, so the warm start serves 128 only.
- `sm120_reproducibility.txt` — `record_autotune_reproducibility.py`: 38 selector-eligible
  sm_120 rows, all admitted strictly; the refreshed
  `nvidia_sm120_autotune_reproducibility.json` is its output.
- The file was re-ordered to the prior committed record order after finalizing (content
  verbatim), so the 16 gfx1151 rows are textually untouched.
- `gfx1151_fused_kernel_identity_v2.txt` — review fixes, same host and worktree, clean at
  `41c4feb4` (`claude/timing-foundation-kernel-identity-fixes`), `build-a/`/`build-b/` rebuilt
  there. Supersedes the rows of the file above, which used a 2-D `(M, N)` bucket that ordinary
  `run_arbitrated` dispatch never looked up and the v1 identity (undecodable words dropped,
  `.rodata` tables not digested). Contents: real-tool checks of the v2 fail-closed and data
  rules on assembled gfx1151 kernels; v2 identities equal across the two trees and the
  `59ecd215` `tessera-opt`; the old rows not served when asked without dims; the re-record in
  `build-a` with no explicit dims (winners and separation unchanged); all 8 rows served by
  ordinary `run_arbitrated`, `corpus_winner` and `measured_arbitrate` (dims inferred,
  re-measurement disabled) in `build-b`, with the `59ecd215` `tessera-opt`, and in `build-a`.
