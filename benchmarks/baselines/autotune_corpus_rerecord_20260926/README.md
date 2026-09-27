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
  lock. The rows now carry `delegate_identities.rocm_wmma_gemm` (that tree's `tessera-opt`,
  `sha256:701ee098…35f20c87`) and are served on Princess-Luna. Winners unchanged.
- sm_120 rows: owed on The-Super-Bear; their output will be added here.
