# gfx1151 paged-KV rows re-recorded with route identities — 2026-09-27

Sync `AUTOTUNE-LAUNCH-INTEGRITY-2026-09-27`; closes `AUTOTUNE-KERNEL-IDENTITY-PAGED-KV`
([ROCm queue](../../../docs/audit/backend/rocm/todo.md)). Logs are `.txt` because
the repo ignores `*.log`.

**Why.** The 8 `rocm:gfx1151` `paged_kv_decode` rows and their production
reader (`cache/paged_kv.py::_rocm_paged_attention_corpus_winner`) carried the
toolchain pins and no code identity, so a changed HIP emitter or a rebuilt
compiler that changed the FA-2 image kept serving the old ranking. Each row now
stamps both routes' identities
(`rocm_hip.rocm_paged_attention_route_identities`):

- `direct`: the emitted HIP paged-attention source, its `kernel_cache` key and
  the hipcc line with `--offload-arch=gfx1151`;
- `gather_fa`: the emitted HIP paged-KV gather plus the compiled FA-2 forward
  image it launches, by kernel-code identity (f16, no GQA at 4/4 heads, the
  additive-bias variant because the route expresses the decode's causal mask as
  a bias).

The reader refuses a row whose live identities differ, or that timed a
different set of routes (`autotune.route_record_matches`, fail closed). The
paged HIP artifacts are now cached by the content of the source compiled, so
the identity names the code a launch runs. The same change made the gather,
the direct attention and the ReplaySSM `su` entry read the HIP slot after each
launch (`NVIDIA-EMITTED-UNCHECKED-LAUNCH`, ROCm half), which also changes the
two routes' sources: these rows could not have been kept.

**Host, commit, trees.** Princess-Luna (Strix Halo gfx1151, WSL2, ROCm 10.0 /
HIP 7.15), own worktree `~/programming/tessera-w-a` detached at `176e557e`,
clean, `build/` configured from scratch (ROCm, x86, EBM, Clifford) and fully
built (`gfx1151_host.txt`: `ninja -n` no work to do). A second tree,
`build-alt/`, same source and configure, built `tessera-opt` for the
other-tree check. `scripts/_rocm_env.sh` sourced; recorder and checks under
`flock /tmp/tessera-timing.lock`.

**Recorder.** `benchmarks/rocm/record_paged_kv_corpus.py` at its defaults (4/4
heads, head_dim 32, page 16, tokens 128/512/2048/8192, 7 whole repeats), on a
copy of the committed corpus; `gfx1151_record.txt` rc 0, rows in
`gfx1151_paged_kv_rows.json`. The 8 rows were spliced into the corpus with
`../autotune_corpus_rerecord_sm120_launch_integrity_20260927/summarize_rerecord.py
--devices rocm:gfx1151 --ops paged_kv_decode` (`gfx1151_summary.txt`): 8 rows
changed, the 116 others byte-identical, **no winner changed**.

| tokens | device winner | end-to-end winner | end-to-end separation (margin / noise) |
|---|---|---|---|
| 128 | `gather_fa` | `direct` | separated (69.6% / 9.0%) |
| 512 | `gather_fa` | `direct` | separated (57.8% / 4.1%) |
| 2048 | `gather_fa` | `direct` | separated (39.4% / 1.7%) |
| 8192 | `gather_fa` | `direct` | **not separated** (12.0% / 6.0%; was separated) |

The 22 "timed candidates with no identity" in the summary are sm_120 rows,
re-recorded separately on The-Super-Bear (same sync key).

**Serve and miss (`check_paged_kv_identity.py`).** In a fresh process, once
with the recording tree's `tessera-opt` and once with `build-alt/`'s
(`gfx1151_serve_check_build.txt`, `gfx1151_serve_check_build-alt.txt`, both
rc 0, identical): identities match 8/8; the production warm start serves the
three admissible end-to-end rows (`direct` at 128/512/2048) and refuses 8192
(unseparated); the reader does not consult device rows. Perturbing the gather
emitter, the direct-attention emitter, or the FA-2 image (the head_dim-64
image in place of the head_dim-32 one) with the ROCm pins asserted unchanged
makes all 8 rows miss and nothing is served.

**Before this change** the old reader served all four end-to-end rows on the
pins alone; under the new rule the committed pre-change rows serve 0 (no
stamp).

**Limits.** Host-side Python between the gather and the FA-2 launch
(transposes, the causal bias) is not digested — the stated
`emitted_code_identity` limit. Device (HIP-event) rows are recorded and
stamped but not read by production dispatch.
