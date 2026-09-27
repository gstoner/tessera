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
- sm_120 rows: owed on The-Super-Bear; their output will be added here.
