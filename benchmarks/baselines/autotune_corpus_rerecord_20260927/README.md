# gfx1151 fused_region re-record under the emitted-code identity — 2026-09-27

Sync `AUTOTUNE-EMITTED-IDENTITY-2026-09-27` (rocm / nvidia / x86 / apple todos). Logs are
`.txt` because the repo ignores `*.log`.

**What changed.** Every arbiter candidate, of every tier, must now carry an identity of the
code it runs, or a verdict involving it misses (`autotune._record_matches_live_delegates`;
`compiler/emitted_code_identity.py`, normalization `tessera.emitted_source.v1`). Before this,
a SYNTHESIZED/EMITTED lane with no identity was served on the toolchain pins alone, so a
changed emitter kept a verdict measured for its old kernel (Codex review P2 on PR #859). The
8 `rocm:gfx1151` `fused_region` rows committed at `1dfad816` stamped only `rocm_wmma_gemm`,
so under the new rule none of them could be served; they were re-measured rather than
backfilled, because a backfilled `rocm_generic_hip` identity would claim a recorded run used
code nobody verified.

**Host and commit.** Princess-Luna (Strix Halo, gfx1151, Ubuntu 26.04 WSL2, ROCm 10.0 /
HIP 7.15, apt LLVM 23.1), fresh worktree `~/programming/tessera-eid` detached at `61ff8b90`
(branch `claude/autotune-emitted-identity`), clean before the run, its own `build/`
configured like the box's main tree (HIP + ROCm + x86 + EBM + Clifford ON) and built with
`ninja -C build` (rc 0), `scripts/_rocm_env.sh` sourced, the recorder under
`flock /tmp/tessera-timing.lock`.

- `gfx1151_fused_emitted_identity.txt` — `benchmarks/rocm/record_autotune_separation.py`
  (defaults: 64/256/512/1024 square, bias+gelu, f16, 12 reps, 3 warm-up, 10 device repeats;
  timer source `device_event`). The 8 rows now stamp **both** candidates:
  `rocm_generic_hip` with its emitted-source identity (HIP source sha256 `7c0d3bcb…`,
  `kernel_cache` key `5fddd098…`, build `hipcc --offload-arch=gfx1151 -O3 -fPIC -shared`)
  and `rocm_wmma_gemm` with its kernel-code v2 identity (unchanged from the `1dfad816` rows
  at all 8 keys: the image did not change). **Winners unchanged**: `rocm_generic_hip` at
  64³ end to end, `rocm_wmma_gemm` everywhere else; all 8 separated. The HIP source digest
  and cache key printed here equal the ones the Mac computes for the same region, which is
  the cross-host determinism the identity needs.
- `gfx1151_serve_and_miss_check.txt` + `check_emitted_identity.py` — a fresh process on
  the same worktree after the re-record. All 8 rows are served by `corpus_winner` with no
  explicit dims (the lookup ordinary `run_arbitrated` dispatch makes); then the HIP emitter
  `rocm_hip._synthesize_fused_hip` is perturbed (one added comment line, toolchain digest
  asserted unchanged) and all 8 miss. Control, same code: the `1dfad816` corpus serves 0/8
  (its rows never stamped `rocm_generic_hip`).

Only the 8 gfx1151 `fused_region` records changed (content compared keyed; record order
unchanged); the 8 gfx1151 `paged_kv_decode` rows and all 108 `nvidia:sm_120` rows are
byte-identical. Corpus format unchanged (v4): `evidence.delegate_identities` simply carries
one more entry per row.

**Not done here.** The `nvidia:sm_120` registry rows stay as recorded: they stamp only
`nvidia_mma_gemm_shipped`, so none is served until re-recorded on The-Super-Bear, which was
offline (owed item `AUTOTUNE-EMITTED-IDENTITY-SM120-RERECORD`, NVIDIA queue). The
non-registry `paged_kv_decode` / `conv2d` / `ssm_replay_decode` rows are read by their own
consumers and are unaffected (`AUTOTUNE-KERNEL-IDENTITY-PAGED-KV` stays open).
