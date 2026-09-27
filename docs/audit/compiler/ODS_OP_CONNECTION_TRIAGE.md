---
last_updated: 2026-09-27
audit_role: reference
scope: Connection study for the 84 ODS ops waived by GOV-ODS-CONSUMER-1
---

# ODS op connection triage (GOV-ODS-CONSUMER-1)

**Role.** This is a reference study, not a queue. Ordering belongs to
[GOV-ODS-CONSUMER-1](INTEGRATED_COMPILER_PLAN.md#gov-ods-consumer-1) in the
[integrated plan](INTEGRATED_COMPILER_PLAN.md) and to the owner items named in
each row; the [compiler map](README.md) routes this file. Nothing here changes
compiler code or deletes an op.

**Direction (owner, 2026-09-27).** Ops can tie into capabilities of the
compiler, so the first question for each waived op is *how it should connect*,
not whether to delete it. Every row below therefore names the capability, where
it lives today, and the smallest producer → consumer slice that would connect
it. DELETE appears only where there is positive evidence nothing is lost, and
those rows are listed separately for the owner to decide.

**Input.** The 84 entries of `_WAIVED` in
`tests/unit/test_ods_op_has_consumer.py` (scan:
`python/tessera/compiler/ods_consumer_audit.py`; 623 op records, 539 consumed,
84 waived: 38 fixture-only, 46 unreferenced) at `d8da67f7`. Evidence is a
static code, doc and `git log -S` read on the Mac; **no row rests on a device
run**, and rows whose next step needs a device say which host.

## Recommendation vocabulary

| Label | Meaning |
|---|---|
| **WIRE** | The capability is wanted and the missing link is small and nameable: a producer, a consumer, or both. The row names them and the host that proves the slice. |
| **DECLARED DEBT (#29a)** | Wanted, but the wiring belongs to an existing open item with a gate. The row names that item. To *become* #29a the op still needs the site marker and a behavioural test (conditions 1 and 3); nothing in this PR marks a site. |
| **MERGE/SUPERSEDE** | Another op or route already carries the capability; consolidating is a Decision #31 move. The row names the carrier and anything that must move first (#31 ordering caveat). |
| **DIRECTED-CLOSED** | A contract op for a target closed by owner direction (AMX; ACE is deferred until hardware ships). Keep it, mark it, never propose a lowering. |
| **RE-TIER (consumed)** | The scan missed a real consumer. |
| **DELETE candidate** | Superseded with parity, or never compilable, with the evidence stated. The owner decides. |

## Summary

| Recommendation | Ops |
|---|---:|
| WIRE | 25 |
| DECLARED DEBT (#29a) | 19 |
| MERGE/SUPERSEDE | 17 |
| DELETE candidate | 17 |
| DIRECTED-CLOSED | 5 |
| RE-TIER (consumed) | 1 |
| **Total** | **84** |

By dialect: the 22 Graph `tessera` dialect rows are almost all
WIRE or DEBT, because the capability runs in Python or under another spelling
and only the IR link is missing. The DELETE candidates sit in three places:
two programming-model dialects that were never compiled (`cache`, `moe`), the
never-compiled solver `core` dialects (`trng`), and five ops whose replacement
already landed (`tessera_attn.causal_mask`/`dropout_mask` →
`boundary_mask`/`block_dropout`; `tessera_nvidia.func` → `func.func`;
`tessera_rocm.memcpy` → `async_copy`; `tessera_rocm.emit` → the
`tessera_rocm.kernel` function attribute).

## Scan findings (what the ODS consumer scan got wrong or cannot see)

These are findings about `ods_consumer_audit.py`. **This PR does not change the
scan** -- each is proposed as open follow-up work under GOV-ODS-CONSUMER-1.

1. **The `op_catalog` exclusion hides real frontend producers (4 ops).** The
   scan treats `op_catalog.py` as a name registry, which it is -- but
   `GraphIRBuilder._try_map_call` (`python/tessera/compiler/graph_ir.py` ~2396)
   reads it to *emit IR*: any `ops.X(...)` call in a `@jit` body becomes
   `spec.graph_name`. Tracing `@jit` bodies on 2026-09-27 emitted
   `tessera.ntk_rope`, `tessera.target_verify` and `tessera.cache.commit`
   (`cache.rollback` shares the mechanism). The true tier of those four is
   *produced, no lowering consumer*, not fixture-only.
   **Proposed fix:** a `frontend_produced` tier -- an op is produced when a
   catalog `graph_name` names it, because `_try_map_call` is the consumer of
   the catalog -- still not satisfying #29 on its own (a producer is not a
   consumer), but reported truthfully.
2. **The dialect-implementation exclusion hides an interface producer
   (`tessera.istft_jvp`).** `ISTFTOp::buildTangent`
   (`src/compiler/ir/TangentInterface.cpp:488`) creates `ISTFTJvpOp` under
   `--tessera-autodiff-forward`, and a lit fixture CHECKs it. The file includes
   a generated `*.cpp.inc` and defines op members, so the scan discards it as
   "the dialect's own implementation". **Proposed fix:** exclude only
   verify/print/parse/fold/build members, not interface methods that construct
   *other* ops.
3. **Prefix matches and default branches are invisible (`tile.tmem.store`).**
   `NVIDIALowering.cpp:5727-5735` matches `starts_with("tile.tmem.")` and maps
   anything that is not alloc/load to `tessera_nvidia.tmem_store`. The op is
   consumed. **Proposed fix:** make the store branch explicit (a code change,
   out of scope here); the scan cannot and should not credit a default branch.
4. **Semantics executed under another op's name
   (`tessera.ebm.langevin_step_philox`).** The ROCm/x86 compiled executors
   accept only `"tessera.ebm.langevin_step"` (`runtime.py:18819,18928`) while
   reading Philox key/counter kwargs -- this op's semantics. No scan can see
   that; it is a #31 naming defect.
5. **"Built by CMake" overstates three dialect families.** `cache`, `moe`
   (`src/compiler/programming_model/CMakeLists.txt:14-24`) and the solver
   `core` dialects `trng`/`tsl`/`tss` (`src/solvers/core/dialects/CMakeLists.txt`)
   run TableGen only: no `.cpp` includes the generated headers, nothing is
   linked or registered, and **no tool can parse these ops**. Two "fixtures"
   are dead too: `pm_v1_1_parallel.mlir` (RUN names the unregistered
   `-pm-v1_1-verify`, no `lit.cfg`) and `src/solvers/linalg/test/solver/spd_solve.mlir`
   (no `lit.cfg`; RUN names passes that do not exist). The waiver reason for the
   solver core family is corrected in this PR.
6. **A producer in the wrong namespace (`tessera.arch.*`).** The textual
   frontend emits `tessera.graph.arch.*` (`frontend/parser.py:1137-1138`), so
   ODS op and producer can never meet; an example and a unit test assert the
   wrong name.
7. **Dashboards and specs over-claim some waived ops.**
   `support_table.md` shows `ntk_rope` Tile fused / Target
   device_verified_abi through the `audit.py:230-235` alias `ntk_rope → rope`,
   whose premise (a rewrite exists) is false and is locked by
   `test_compiler_audit.py:285-293`; `primitive_coverage.py:485-510` marks
   `lowering_rule: complete` for the three AttnRes ops;
   `SM120_DIFFERENTIATION_DASHBOARD.md:13-16` cites the fixture-only
   `tessera_nvidia.mma_fused/mma_attention/fpquant` as Target-IR evidence for
   promoted rows (the lanes execute through Python candidates);
   `GRAPH_IR_SPEC.md:398-399` calls `cache.page_lookup`, `ring.create` and the
   DNAS ops implemented/scaffolded. These are Decision #26 reconciliation items
   for their owners; this PR records them and changes no dashboard.
8. **Stale prose that reads like a consumer.** `LegalizeSpectral.cpp:15-18`
   promises one `tessera_spectral.twiddle_table` per stage -- the pass has never
   built one (`git log -S` shows only the op and the comment); the
   `tessera_ebm.partition_z` description says the EBM pipeline lowers it
   (`LowerLangevin.cpp:543-557` refuses it); `python/tessera/distributed/moe.py:21`
   says `GPUCollectiveInsertionPass` sees `tessera.moe.dispatch` -- **false**,
   that pass never mentions MoE; the `tessera_nvidia.func` ODS comment
   (`TesseraNVIDIADialect.td:249-251`) and the `mps_softmax` comment
   (`TesseraAppleOps.td:305-310`) describe producers that no longer exist; and
   `LOWERING_PIPELINE_SPEC.md:245`, `GRAPH_IR_SPEC.md:497` and
   `tessera_compiler_passes_overview.md:58-59` describe the `schedule.async_copy`
   movement classification removed in 45f48565.
9. **Latent defects found in passing (not fixed here).**
   `NVIDIALowering.cpp:5733-5735` erases `tile.tmem.load` without replacing its
   result's uses; any unknown `tile.tmem.*` op silently becomes a `tmem_store`
   contract; `TileBufferArenaPass.cpp:57`, `TileBufferReusePass.cpp:46` and
   `TileMemrefLifetime.h:23` match `"tile.tmem.alloc"` while the registered
   mnemonic is `tmem.allocate`, so they never see the ODS op; and the linalg
   solver passes' `opName.contains("solve")` (`IterativeRefinement.cpp`,
   `MixedPrecision.cpp`) matches every `tessera_solver.*` op because the dialect
   prefix contains "solve" -- a fail-open pseudo-consumer the scan correctly
   rejects.

## Leads checked before recording

| Lead | Finding |
|---|---|
| EBM kernel "mirrors `tessera.ebm.X`" comments (ROCm/x86 generators, `runtime.py`, Apple runtime) | The runtime implements the capability by another route: Python surface (`ebm/energy.py`, `geo_sampling.py`) → runtime helper writes a fixed directive string (`runtime.py:18944,19026,19195`) → `tessera_rocm.ebm_*` Target op → generator pass. The Graph ops are skipped; no Graph → `tessera_ebm` → Target lowering exists. The comments cite the Python API spelling, not the Graph op attributes. Four sibling pilot ops *are* wired Graph → Tile → Apple and are the template. |
| `LegalizeSpectral.cpp:15` twiddle-table comment | Stale since it landed (75789d77); the step was never built. Attributes only. |
| `distributed/moe.py:21` (`GPUCollectiveInsertionPass` sees `tessera.moe.dispatch`) | False today; the pass matches sharded matmul/linear, memory-effect gradient reduction and `schedule.mesh.region` only. |
| Graph solver ops → `tessera_solver.potrf/potrs/trsm/getrf` | No such producer, and adding one would be a second lowering authority: Graph `cholesky/tri_solve/cholesky_solve/lu` already lower `TilingPass.cpp:464-479` → `tile.*` → Apple LAPACK / ROCm / x86 kernels. The dialect's only unique content is its precision policy. |

## Suggested wiring slices, ordered by value for effort

Each slice is a separate future PR under the owner named; none is scheduled by
this document.

1. **Decompose `tessera.target_verify` → `tessera.softmax` and canonicalize
   `tessera.ntk_rope` → `tessera.rope(x, θ/s)`** (host-free lit, S each). The
   frontend already emits both, so one pattern each connects them to every
   backend's existing softmax/rope lowering -- and makes the `ntk_rope`
   dashboard rows true instead of over-claimed.
2. **`tessera.cache.commit/rollback` need a handle-carrying lowering, not a new
   `LowerKVCacheToX86` kind** (M; corrected 2026-09-27 after review). The
   existing pattern is artifact-only by construction: it refuses any op whose
   result has uses (`TileToX86Pass.cpp` "result is consumed"), and its runtime
   call carries only the `kind` i32 -- no cache handle, no accepted/rejected
   count. Commit/rollback always thread their returned handle into later ops, so
   kinds 4/5 would either stay unlowered or emit a call with no commit
   semantics. The slice is: a handle ABI (`tessera_x86_kv_cache_*` taking the
   handle + count and returning the new handle), a runtime consumer on
   `kv_cache_f32.cpp`'s trim/prune, a lowering that threads results, then lit +
   a numeric check against the reference (`__init__.py:5388-5425`). Host-free
   with the x86 backend configured (Mac or Tajasarus).
3. **The `tessera.ebm.*` pilot, on its sibling template:** dotted → flat
   aliases for `inner_step`/`self_verify` (S, host-free); `langevin_step_philox`
   needs a PRODUCER first -- the catalog emits `tessera.ebm.langevin_step` with
   3 operands, the Philox op takes 4 (`y, grad, seed, counter`), and nothing
   constructs it, so repointing the executor alone would be reachable only from
   hand-built fixtures (S-M, host-free then Princess-Luna); then `decode_init` and
   the sphere/bivector steps through TilingPass (M; Princess-Luna, then
   Tajasarus).
4. **`schedule.async_copy`/`await_movement` → `tile.async_copy`/`wait_async`**
   (template `LowerSchedulePrefetchToTileCopy`; S-M) and the
   `schedule.optimizer_shard` consumer in `OptimizerShardPass` (S). Both
   host-free; both restore consumers that were removed or never matched.
5. **`tessera.istft_jvp` consumer in `PMPasses`** beside `spectral_backward`
   (M; host-free lit, then Zen 5 x86 + gfx1151 + gfx1201). Collapses a #31 dual
   authority (C++ forward AD vs the Python JVP plugin) once differential-tested.
6. **Small host-free decompositions:** `guided_denoise_region → score_combine`
   and `tessera_sr.export_manifest` as `ExportDeploymentManifestPass`'s sink
   (S each).
7. **NVIDIA sm_120 `mma_fused`/`mma_attention`/`fpquant`**: Python emitter +
   verifier + `markerForTargetOp` (M together; host-free -- the runtime lanes
   already have Super-Bear evidence), so the SM120 dashboard's Target-IR column
   cites IR a compiler path produces.
8. **`tessera_ebm.partition_z`** (`LowerPartitionZ`, M; CPU JIT then the
   row-program GPUs) and **`tile.tmem.store`** made explicit with an sm_100
   lit plus the three tmem defects (S).
9. **DNAS** (`tessera.arch.*`): namespace fix S, specialize pass M -- needs an
   owner item first.

## DELETE candidates (owner decides)

| Op(s) | Evidence nothing is lost |
|---|---|
| `cache.kv.create`, `cache.page.lookup`, `cache.page.read`, `cache.page.write`, `cache.pt.create`, `cache.ring.create`, `cache.ring.push`, `cache.ring.pop` | Never compiled or registered (TableGen decls only; the one registration ever written was commented out, 8fbc4ebb, and dropped in 0003e9d7), so no user can write them; no attributes, four `hasVerifier` with no `verify()`. Every concept has a consumed carrier: `tessera.kv_cache.*` → `LowerKVCacheToX86`, `tessera.paged_kv_read` → ROCm/NVIDIA, Python `paged_kv.py`; ring push/pop deliberately live handle-side (ROCM-REPLAY-1). Remove `cache/CacheOps` from `programming_model/CMakeLists.txt` and rewrite the two docs that show it. |
| `moe.dispatch` | Never compiled or registered; Graph `tessera.moe_dispatch` executes on ROCm, NVIDIA sm_120 and Apple. (`moe.plan`/`moe.token_limiter.create` are MERGE rows: their attributes move first.) |
| `trng.create_state`, `trng.uniform`, `trng.normal` | Never compiled or registered; nothing names `trng.`; the RNG passes match `tessera_rng.*`; Graph `tessera.rng_philox_*` + `rng.py` carry the capability. Retargeting the RNG passes to the Graph ops is a separate S slice. |
| `tessera_attn.causal_mask`, `tessera_attn.dropout_mask` | Replaced in 1b2b0680 by `boundary_mask` / `block_dropout`, both produced (`TileIRLoweringPass.cpp:688,695`) and consumed (ROCm, Apple). `dropout_mask` never had a producer. |
| `tessera_nvidia.func` | Its only producer was removed in W0.9 (0003e9d7) for `func.func`, documented at `target_ir.py:734-748`; no runtime-built `{dialect}.func` exists. |
| `tessera_rocm.memcpy` | Cerebras scaffold copy (a778907a), never referenced in 13 months; device copies use `tessera_rocm.async_copy` + `wait`; host copies are runtime `hipMemcpy`. |
| `tessera_rocm.emit` | Same scaffold origin; no ROCm region needs the terminator; emission keys on the `tessera_rocm.kernel` function attribute. |

What would change a DELETE to WIRE: a plan item that wants the legacy `cache`
ring as typed IR (then `tessera.ring.create` plus push/pop, not the `cache`
dialect), or an MPS-backed lane that needs a dedicated op. Neither exists
today.

## Per-op table

Anchors are `#<op with . and _ as ->` (e.g. `#tessera-ntk-rope`).

### A. Programming-model `cache` dialect (`src/compiler/programming_model/ir/cache/CacheOps.td`)

| Op | Rec. | Capability | Current state | Where it should connect | Recommendation | Evidence | Owner / host / size |
|---|---|---|---|---|---|---|---|
| <a id="cache-kv-create"></a>`cache.kv.create` | **DELETE candidate** | Create a legacy KV-cache object (no inputs, no attributes). | Unreferenced. TableGen op decls/defs only: no `-gen-dialect-*`, no `.cpp` includes the generated headers, no library links them, `tessera-opt` never registers the dialect -- the op cannot be parsed. Origin 52860a10 (2025-09-17); the only registration ever written was a commented-out line (8fbc4ebb), dropped in 0003e9d7. | Nothing new: `tessera.kv_cache.create` (carries `page_size`/`eviction`) → `LowerKVCacheToX86` (`src/transforms/lib/TileToX86Pass.cpp:474-510`). | SUPERSEDE → delete candidate (#31 duplicate of `tessera.kv_cache.*`). | `src/compiler/programming_model/CMakeLists.txt:14-24`; `Memory_Execution_Model_v1_1.md:49-70` labels it `%kv_legacy`. | Owner decision; host-free, S |
| <a id="cache-page-lookup"></a>`cache.page.lookup` | **DELETE candidate** | Legacy page lookup. | As `cache.kv.create` (never compiled/registered). | `tessera.paged_kv_read(pages, page_table, start, end)` → `NativePagedKV.h` → `schedule.paged_kv_read` → `tile.paged_kv_read_kernel` → ROCm/NVIDIA lowerings. | SUPERSEDE → delete candidate. | `TesseraOps.td:3458` (0398f1ea); `python/tessera/cache/paged_kv.py:77-130`. | Owner decision; host-free, S |
| <a id="cache-page-read"></a>`cache.page.read` | **DELETE candidate** | Legacy page read. | As above. | `tessera.paged_kv_read` / `tessera.kv_cache.read`. | SUPERSEDE → delete candidate. | as above | Owner decision; host-free, S |
| <a id="cache-page-write"></a>`cache.page.write` | **DELETE candidate** | Legacy page write. | As above. | `tessera.kv_cache.append` (consumed by `LowerKVCacheToX86`). | SUPERSEDE → delete candidate. | `TileToX86Pass.cpp:483-485,579-581` | Owner decision; host-free, S |
| <a id="cache-pt-create"></a>`cache.pt.create` | **DELETE candidate** | Legacy page-table object (AnyType result, no page_size). | As above. | `tessera.kv_cache.create{page_size}` + Python `PageTableEntry`/`materialize_paged_kv_abi`. | SUPERSEDE → delete candidate. | `python/tessera/cache/paged_kv.py:77-130,421` | Owner decision; host-free, S |
| <a id="cache-ring-create"></a>`cache.ring.create` | **DELETE candidate** | Legacy ring buffer (no capacity). | As above. | `tessera.ring.create` (row C) and the handle-side ring cursors in `SSMStateHandle`/`DeltaNetStateHandle`. | SUPERSEDE → delete candidate. | `cache/ssm_state.py:311-326`; `cache/delta_state.py:169-187` | Owner decision; host-free, S |
| <a id="cache-ring-push"></a>`cache.ring.push` | **DELETE candidate** | Legacy ring push. | As above. | No IR carrier, by decision: ROCM-REPLAY-1 made ReplaySSM a handle-side serving runtime, not a Graph state type. | SUPERSEDE → delete candidate. Capability lives in the Python ring cursor + `tessera.cache.commit/rollback`. | `docs/audit/backend/rocm/todo.md:6017-6030` | Owner decision; host-free, S |
| <a id="cache-ring-pop"></a>`cache.ring.pop` | **DELETE candidate** | Legacy ring pop. | As above. | As `cache.ring.push`. | SUPERSEDE → delete candidate. | as above | Owner decision; host-free, S |

### B. Programming-model `moe` dialect (`src/compiler/programming_model/ir/moe/MoeOps.td`)

| Op | Rec. | Capability | Current state | Where it should connect | Recommendation | Evidence | Owner / host / size |
|---|---|---|---|---|---|---|---|
| <a id="moe-dispatch"></a>`moe.dispatch` | **DELETE candidate** | Route tokens to experts under a limiter. | Unreferenced. Same build shape as `cache`: TableGen only, never compiled or registered. Origin 52860a10. | Carrier is live: Graph `tessera.moe_dispatch`/`moe_combine` (`TesseraOps.td:1357,1368`; `op_catalog.py:239-240`) → ROCm `TileToROCM.cpp:3186`/`GenerateROCMMoeKernel.cpp`, NVIDIA `NVIDIALowering.cpp:5309`, Apple runtime lanes. | SUPERSEDE → delete candidate (parity: the Graph op executes on three backends). | `src/compiler/programming_model/CMakeLists.txt:14-24` | Owner decision; host-free, S |
| <a id="moe-plan"></a>`moe.plan` | **MERGE/SUPERSEDE** | A2A bucket plan + `pack_cast` wire dtype. | Tier says fixture-only, but the fixture (`pm_v1_1_parallel.mlir`) is dead: its RUN line names `-pm-v1_1-verify`, which no `.cpp` registers, and the directory sits under no `lit.cfg`. | Attributes on `tessera.moe_dispatch`: bucket → `chunk_bytes` (already forwarded by `CollectiveLowering.cpp:90-104`), wire dtype → `tessera_collective.pack_cast` (row G). | MERGE (#31) into dispatch/collective attributes, then delete the dialect. Merge work is DIST-NATIVE-1 debt. | Python `distributed/moe.py` `plan_all_to_all` carries the plan today | DIST-NATIVE-1; host-free |
| <a id="moe-token-limiter-create"></a>`moe.token_limiter.create` | **MERGE/SUPERSEDE** | In-flight token cap + refill for MoE transport. | Fixture-only through the same dead fixture. | `tessera_collective.qos.limit` (row G) is the IR form of the same `TokenLimiter`. | MERGE (#31) into `qos.limit`. | `src/collectives/.../Runtime/Execution.h:102-107` | DIST-NATIVE-1; host-free |

### C. Graph IR (`tessera` dialect, `src/compiler/ir/TesseraOps.td`) — state, spec-decode, position

| Op | Rec. | Capability | Current state | Where it should connect | Recommendation | Evidence | Owner / host / size |
|---|---|---|---|---|---|---|---|
| <a id="tessera-cache-commit"></a>`tessera.cache.commit` | **WIRE** | SD1-3 speculative-decode commit: keep `accepted_length` tokens; threads the KV handle (MemWrite). | **Scan under-count: the frontend produces it.** `op_catalog.py:322` → `GraphIRBuilder._try_map_call` emits `tessera.cache.commit(...)` for `ops.cache_commit` in a `@jit` body (traced 2026-09-27). No pass consumes it after emission; the reference lane runs `__init__.py:5388-5425`. Origin 88b2b46a (#249). | A handle-carrying consumer. **Not** a new kind in `LowerKVCacheToX86` (`TileToX86Pass.cpp:474-510`): that pattern refuses any op whose result has uses ("artifact-only") and its runtime call carries only the `kind` i32, so a commit whose handle feeds later ops -- the normal case -- would stay unlowered, and a lowered one would carry no commit semantics (corrected 2026-09-27, review of this doc). Needed: a handle ABI (handle + `accepted_length` in, new handle out), a runtime consumer on the existing trim/prune (`kv_cache_f32.cpp:39`), and a lowering that threads the result. | WIRE, **after** the handle ABI exists. Then lit on `control_flow/sd1_3_cache_effects.mlir` through `--tessera-tile-to-x86` plus a numeric check against the reference. Artifact-level ABI proof, not a native-execution claim. | `graph_ir.py:3908-3925` (state_handle rule); `audit.py:255-256` marks Tile/Target not_applicable; `TileToX86Pass.cpp` result-use refusal | Host-free lit (x86 backend configured: Mac or Tajasarus); M |
| <a id="tessera-cache-rollback"></a>`tessera.cache.rollback` | **WIRE** | SD1-3 rewind: drop `num_rejected` tokens (KV trim / SSM ring-cursor rewind). | Scan under-count, as `cache.commit` (`op_catalog.py:323`). | As `cache.commit`: the same handle ABI with `num_rejected` in, new handle out (not a kind-5 artifact call). | WIRE, same slice, after the handle ABI. | `__init__.py:5406-5418` | Host-free lit; M |
| <a id="tessera-cache-page-lookup"></a>`tessera.cache.page_lookup` | **MERGE/SUPERSEDE** | Logical position → `!tessera.cache_page` handle. | Unreferenced. No catalog entry, so no frontend producer; `!tessera.cache_page` is produced only here and accepted by no op anywhere (`HANDLE_CACHE_PAGE`, `graph_ir.py:534`, unused). Origin dc95dd47. | Physical: `tessera.paged_kv_read` with an explicit page table. Public: the `kv_cache.read(cache,start,end)` ODS, still undeclared (x86 todo :5035-5042, E2E-REAL-5 / F2-S1). | SUPERSEDE by `paged_kv_read`; delete with `Tessera_CachePageType` only once the public `kv_cache.read` ODS lands (#31 ordering caveat). | `TesseraOps.td:152,2973` | E2E-REAL-6 (F2); host-free, S |
| <a id="tessera-ring-create"></a>`tessera.ring.create` | **MERGE/SUPERSEDE** | Create a `!tessera.ring` with positive capacity. | Fixture-only (one negative verifier fixture). No catalog entry, no push/pop op, `HANDLE_RING` unused. Duplicates `cache.ring.create` (the survivor of the pair: it has capacity + a verifier). | The ring semantics ride `!tessera.kv_cache` + `cache.rollback` (whose ODS text is the ring-cursor rewind); ROCM-REPLAY-1 chose the handle-side route. | SUPERSEDE by `cache.commit/rollback` on the KV handle. If a typed ring is wanted later, it becomes #29a debt with push/pop and a frontend annotation for `SSMStateHandle`. | `TesseraOps.td:2954-2970,2981`; `rocm/todo.md:6017-6030` | Owner decision; host-free, S |
| <a id="tessera-target-verify"></a>`tessera.target_verify` | **WIRE** | SD1-4 target scoring contract: `softmax(logits, -1)` over S=D+1 positions; `tokens` pins S. | **Scan under-count: the frontend produces it** (`op_catalog.py:337` → `_try_map_call`; traced). No lowering; `audit.py:444` forces Target not_applicable. Origin 79ca088a. | Graph decompose `target_verify → tessera.softmax` (last axis) in `CanonicalizeTesseraIR.cpp` (template: `RLLossDecomposePass.cpp`); every backend's softmax family then consumes it, and its output feeds `spec_accept_sample`. | WIRE. Lit: CHECK-NOT `target_verify`, CHECK `tessera.softmax`; numeric check vs `__init__.py:5455`. | `TesseraToLinalgPass.cpp:2692` SoftmaxLowering | Host-free (any box); S |
| <a id="tessera-ntk-rope"></a>`tessera.ntk_rope` | **WIRE** | NTK-scaled RoPE; the reference is literally `rope(x, theta/scale)`. | **Scan under-count: the frontend produces it** (`op_catalog.py:360`; traced; `test_reasoning_model_support.py:166-171`). Every rope consumer matches only the literal `tessera.rope` (`RopeToAppleGPU.cpp:50`, `TileToApple.cpp:106`, x86 `Passes.cpp:171,206`), so an emitted `ntk_rope` reaches none. **The dashboards over-claim it**: `support_table.md` shows Tile fused / Target device_verified_abi via `audit.py:230-235` alias `ntk_rope→rope`, whose stated premise (a rewrite exists) is false. | Canonicalize `ntk_rope(x,θ,s) → rope(x, div(θ, s))` in `CanonicalizeTesseraIR.cpp`; `tessera.div` lowers already. | WIRE as a MERGE-by-lowering into `tessera.rope` (one rope lowering, #31). Until it lands the dashboard alias should read partial. | `nn/functional.py:350-351`; `tests/unit/test_compiler_audit.py:285-293` locks the alias | Host-free lit, Apple execution on Mac; S |

### D. Graph IR — DNAS (`tessera.arch.*`)

| Op | Rec. | Capability | Current state | Where it should connect | Recommendation | Evidence | Owner / host / size |
|---|---|---|---|---|---|---|---|
| <a id="tessera-arch-parameter"></a>`tessera.arch.parameter` | **WIRE** | FP32 architecture logits, kept apart from weights. | Fixture-only (`test_compiler_spec_gap_remediation.py`). **Namespace bug**: the textual frontend emits `tessera.graph.arch.*` (`frontend/parser.py:1137-1138`), which can never match the ODS name; `examples/compiler/dnas/dnas_graphir_sketch.mlir` and `test_textual_frontend.py:215` assert the wrong name. No catalog entries. `GRAPH_IR_SPEC.md:399` claims scaffolded lowering, which is false. Origin b0f15e10 (single commit). | Producer: fix `parser.py:1137` to emit `tessera.arch.*`; add tensor-operand catalog entries. Consumer: a host-free `arch-specialize` pass (search mode: `weighted_sum` → Σ mul(candidate_i, gate_i); export mode: hard `switch`/fixed `mixed` → chosen candidate), mirroring `arch.py` `argmax`/`specialize`. | WIRE (namespace fix S; specialize pass M). No plan item owns DNAS today -- needs a new owner item before #29a debt is possible. | `python/tessera/arch.py:1-8` (reference surface over Python floats) | Host-free; S+M; owner TBD |
| <a id="tessera-arch-weighted-sum"></a>`tessera.arch.weighted_sum` | **WIRE** | Gate-weighted mixture of candidate outputs. | Unreferenced; see `arch.parameter`. | `arch-specialize` search-mode decomposition. | WIRE (same slice). | `arch.py:190` | Host-free; M |
| <a id="tessera-arch-switch"></a>`tessera.arch.switch` | **WIRE** | Select one candidate (soft or hard). | Unreferenced. | `arch-specialize` export mode. | WIRE (same slice). | `arch.py:198` | Host-free; M |
| <a id="tessera-arch-mixed"></a>`tessera.arch.mixed` | **WIRE** | Mixed-op (candidate set + gate). | Unreferenced. | `arch-specialize`. | WIRE (same slice). | `arch.py:210` `MixedOp` | Host-free; M |
| <a id="tessera-arch-ste-one-hot"></a>`tessera.arch.ste_one_hot` | **WIRE** | Straight-through one-hot relaxation. | Unreferenced (its only mention was its own arity table, `TesseraOps.cpp:42`). | Decompose to forward one-hot + identity gradient in `arch-specialize`; VJP via the autodiff layer. | WIRE (same slice). | `arch.py:165` `STEOneHot` | Host-free; M |

### E. Graph IR — EBM value-lane pilot (`tessera.ebm.*`, `TesseraOps.td:770-925`)

| Op | Rec. | Capability | Current state | Where it should connect | Recommendation | Evidence | Owner / host / size |
|---|---|---|---|---|---|---|---|
| <a id="tessera-ebm-inner-step"></a>`tessera.ebm.inner_step` | **WIRE** | y − η·grad (+ optional noise). | Fixture-only. Origin 8b0c94ec; no consumer removed. The capability runs under two other spellings: the flat frontend name `tessera.ebm_inner_step` (`op_catalog.py:506`, CPU/Apple/x86/ROCm compute lanes) and the solver op `tessera_ebm.inner_step` (`LowerLangevin.cpp:104`). Four siblings of this pilot (`energy_quadratic`, `langevin_step`, `refinement`, `partition_exact`) ARE wired Graph → Tile (`TilingPass.cpp:735-871`) → Apple (`TileToApple.cpp:117-135`, `apple_native.py:129-132,496-499`). | Follow the sibling template: dotted → flat alias in `LEGACY_GRAPH_OP_ALIASES` (`op_catalog.py:617` precedent), and a TilingPass rewrite (noise → `langevin_step` Tile pattern; no noise → `refinement` steps=1). | WIRE. | `TesseraOps.td:770-774` (pilot comment) | Host-free lit; Apple execution on Mac; S |
| <a id="tessera-ebm-self-verify"></a>`tessera.ebm.self_verify` | **WIRE** | Hard or soft argmin selection over candidate energies. | Fixture-only. Live as flat `tessera.ebm_self_verify` (`op_catalog.py:504`) and the Apple hard-argmin kernel (`apple_gpu_runtime.mm:16992`). | Alias dotted → flat; TilingPass value pattern + `TileToApple` predicate for the hard-argmin symbol. | WIRE (alias S; Tile lane M). Fallback: #29a debt under W4-PRODUCT-1 (its EBM stream lists Apple open). | `backend_manifest.py:5116-5480` has the Apple symbol | Host-free + Mac; S-M |
| <a id="tessera-ebm-decode-init"></a>`tessera.ebm.decode_init` | **WIRE** | Seed K candidates (`strategy`, optional base/noise/seed). | Fixture-only. Live as eager `ebm/energy.py:378-385` → runtime directive → `tessera_rocm.ebm_decode_init` (`runtime.py:19195-19197`, `GenerateROCMEbmDecodeInitKernel.cpp`). The `mirrors tessera.ebm.decode_init` comments cite the Python API spelling, not this op's attributes. | Route the op into the `ebm_decode_init` family (`pipeline_registry.py:649`), replacing the fixed runtime directive string. | WIRE (or #29a debt under W4-PRODUCT-1). | `GenerateROCMEbmDecodeInitKernel.cpp:5` | Princess-Luna (gfx1151); M |
| <a id="tessera-ebm-langevin-step-philox"></a>`tessera.ebm.langevin_step_philox` | **WIRE** | Langevin step with Philox (seed, counter) noise. | Fixture-only, **but its semantics execute under another op's name**: the ROCm/x86 compiled executors accept only `_EBM_LANGEVIN_OPS = ("tessera.ebm.langevin_step",)` (`runtime.py:18819,18928`) while reading Philox `key`/`counter` kwargs -- the ODS `langevin_step` has a `noise` operand and no seed. Tests build `op_name="tessera.ebm.langevin_step"` (`test_rocm_ebm_langevin_compiled.py:33`). | A **producer first**, then the executor: the canonical catalog emits `tessera.ebm.langevin_step` with 3 operands (`op_catalog.py:159`, min=max=3) while this op requires 4 (`y, grad, seed, counter`, `TesseraOps.td:837`), and nothing constructs it -- so repointing the executor alone would make it reachable only from hand-built fixtures (corrected 2026-09-27, review of this doc). Slice: a catalog/frontend mapping that emits the Philox op when a seed/counter is supplied (with its `OpSpec`, shape/effect rules and registry entries), then point the executor at it, map seed/counter from operands, and fix the two compiled-test fixtures. | WIRE (a #31 name correction plus the missing producer, not new capability). | `runtime.py:18813` comment; `op_catalog.py:159` | Host-free then Princess-Luna; S-M |
| <a id="tessera-ebm-sphere-langevin-step"></a>`tessera.ebm.sphere_langevin_step` | **WIRE** | Manifold Langevin on the sphere (project, step, renormalize). | Fixture-only. Live via eager `geo_sampling.py:192,354` → affine kernels, and via `native_langevin.py` → `tessera_ebm.langevin_step{manifold=sphere}` → row program (gfx1151/gfx1201/sm_120). A direct Graph→`tessera_ebm` pattern is not sound (the solver op takes `energy_fn` + key; this op takes `grad` + `noise`). | TilingPass decomposition: projection + `langevin_step` Tile op + normalize; or a Graph→Target projection feeding `ebm_affine_langevin`. | WIRE (or #29a debt under W4-PRODUCT-1 / W6.4). | `GA_EBM_ARCHITECTURE_REVIEW.md` | Princess-Luna then Tajasarus; M |
| <a id="tessera-ebm-bivector-langevin-step"></a>`tessera.ebm.bivector_langevin_step` | **WIRE** | Manifold Langevin on bivectors (grade-projected). | As `sphere_langevin_step` (manifold=bivector). | As above (projection without normalize). | WIRE (same slice). | `geo_sampling.py:278,431` | Princess-Luna then Tajasarus; M |

### F. Graph IR — Block AttnRes, diffusion, spectral AD

| Op | Rec. | Capability | Current state | Where it should connect | Recommendation | Evidence | Owner / host / size |
|---|---|---|---|---|---|---|---|
| <a id="tessera-attn-with-stats"></a>`tessera.attn_with_stats` | **DECLARED DEBT (#29a)** | Max-shifted depth-attention forward state (o, m, ℓ). | Fixture-only. Origin e958756c. The gfx1151 lane uses the monolithic `tessera.depth_attn` → `schedule.depth_attention` → `tile.depth_attention_kernel` → `tessera_rocm.depth_attention`, with the stats/merge recurrence inline in `GenerateROCMDepthAttentionKernel.cpp:220-261`. `primitive_coverage.py:485-510` marks `lowering_rule: complete` for all three AttnRes ops -- a false green. | (a) Graph canonicalization `softmax_finalize(attn_with_stats(q,V)) → depth_attn`; (b) the query-hoisting schedule transform named in BLOCK_ATTNRES_ROCM_PLAN §III.4 item 1. | #29a DEBT under BLOCK-ATTNRES-1 (the plan's §III.1 names these ops and their consumers). Smallest wire is (a). | `BLOCK_ATTNRES_ROCM_PLAN.md:354-360,380-425`; `PMPasses.cpp:1877,3110` | BLOCK-ATTNRES-1; host-free, S |
| <a id="tessera-softmax-merge"></a>`tessera.softmax_merge` | **DECLARED DEBT (#29a)** | Associative merge of two (o,m,ℓ) states. | Fixture-only. The plan names its consumers (two-phase schedule, sharded prefill, ring/context-parallel attention, split-KV decode); none is built. | Query-hoisting transform (§III.4.1), legality derived from the reassociable trait (#30). | #29a DEBT under BLOCK-ATTNRES-1. | `BLOCK_ATTNRES_ROCM_PLAN.md:358,420-425` | BLOCK-ATTNRES-1; host-free, M |
| <a id="tessera-softmax-finalize"></a>`tessera.softmax_finalize` | **DECLARED DEBT (#29a)** | o / ℓ. | Fixture-only. | Canonicalization (a) above. | #29a DEBT under BLOCK-ATTNRES-1. | `_block_attnres_ops.py:133` | BLOCK-ATTNRES-1; host-free, S |
| <a id="tessera-guided-denoise-region"></a>`tessera.guided_denoise_region` | **WIRE** | CGG guided score: ref + γ·(favored − unfavored) with timestep/schedule. | Fixture-only. Declared as a marker (fcec8945) but it is `Pure` with a `guided_score` result -- value semantics. `tessera.score_combine` is consumed (`TesseraToLinalgPass.cpp:209-235`, Apple runtime). | Decompose `guided_denoise_region → score_combine(ref, sub(fav, unfav), gamma)` in TesseraToLinalgPass or a Graph canonicalizer; keep timestep/schedule as attributes for audit. | WIRE. No owning item found. | `diffusion_guidance.py:238,373-378` | Host-free; S |
| <a id="tessera-istft-jvp"></a>`tessera.istft_jvp` | **WIRE** | Exact ISTFT forward-mode product over spectrum and window. | **Scan under-count: it has a C++ producer.** `ISTFTOp::buildTangent` (`src/compiler/ir/TangentInterface.cpp:488`) builds it under `--tessera-autodiff-forward`, and `autodiff_forward_spectral_products.mlir:37` CHECKs it. The scan skips that file as the dialect's own implementation (it includes `TangentInterface.cpp.inc`). No consumer: nothing lowers it. The JVP executes through the Python native plugin keyed on `tessera.istft` (`native_jvp_plugins.py:410-450` → `schedule.spectral_program`; x86/NVIDIA/ROCm symbols) -- so C++ forward AD and the Python plugin are two authorities for one JVP (#31). | Consumer: `PMPasses.cpp` beside the `tessera.spectral_backward` arm (:2872-2900): lower into `schedule.spectral_program` with product mode `jvp`; retire the Python plugin arm after a differential test. | WIRE. | `runtime.py:14440-14590` | Host-free lit, then Zen 5 x86 + gfx1151 + gfx1201; M |

### G. Collectives (`tessera_collective`, `src/collectives/.../CollectiveOps.td`)

| Op | Rec. | Capability | Current state | Where it should connect | Recommendation | Evidence | Owner / host / size |
|---|---|---|---|---|---|---|---|
| <a id="tessera-collective-qos-limit"></a>`tessera_collective.qos.limit` | **DECLARED DEBT (#29a)** | Cap in-flight collective chunks (`max_inflight`). | Fixture-only (negative verifier fixture). The runtime exists: `tessera_qos_limit_set/acquire/release` on `TokenLimiter` (`Execution.h:102-107` says "C hooks to wire from lowered QoS ops"); only a smoke tool calls them. `CollectiveLowering.cpp` lowers five `tile.*` collectives and emits no QoS. Origin eeb947dd. | Producer: `CollectiveLowering.cpp:73-110` brackets dispatch/await when `chunk_bytes` is set; consumer: a Target→runtime lowering to `func.call @tessera_qos_*`. | #29a DEBT under DIST-NATIVE-1: throttling does nothing while the await sits next to the dispatch (TILERT E3: overlap window zero) and the chunked adapters are mocks. | `TILERT_ASSESSMENT.md:194-195` | DIST-NATIVE-1; Mac C++ lit + link, M |
| <a id="tessera-collective-qos-acquire"></a>`tessera_collective.qos.acquire` | **DECLARED DEBT (#29a)** | Take one in-flight token. | Unreferenced; see `qos.limit`. | As `qos.limit`. | #29a DEBT under DIST-NATIVE-1. | `Execution.cpp:78` | DIST-NATIVE-1 |
| <a id="tessera-collective-qos-release"></a>`tessera_collective.qos.release` | **DECLARED DEBT (#29a)** | Return one token. | Unreferenced. | As `qos.limit`. | #29a DEBT under DIST-NATIVE-1. | `Execution.cpp:84` | DIST-NATIVE-1 |
| <a id="tessera-collective-pack-cast"></a>`tessera_collective.pack_cast` | **DECLARED DEBT (#29a)** | Wire-dtype cast before transport (FP32 → BF16 / FP8 E4M3, optional RLE). | Unreferenced. Runtime exists (`Packing.h:52 pack_cast_fp32`, one demo caller). The op is memref-typed while the collective Target tier is tensor/future-typed (`CollectiveLowering.cpp:78-80`) -- it cannot be emitted where the concept lives. | Producer: `CollectiveLowering.cpp:91` (already forwards `dtype`) emits `pack_cast` when wire ≠ payload dtype; consumer: adapter call to `pack_cast_fp32`. | #29a DEBT under DIST-NATIVE-1, with a retype to tensor-or-memref; alternative MERGE into a `wire_dtype` attribute on the dispatch op. | `MoeOps.td:11` duplicates it as a string | DIST-NATIVE-1; host-free S + device |
| <a id="tessera-collective-shard-view"></a>`tessera_collective.shard_view` | **DECLARED DEBT (#29a)** | Typed logical shard `!tessera_collective.shard<T>` over (mesh_axis, dim). | Unreferenced; only verifiers (`CollectiveOps.cpp:278`). Nothing produces or consumes the shard type. | Producer: `GPUCollectiveInsertionPass.cpp:163-181` (knows row/col-parallel + mesh axis); consumer: `CollectiveLowering.cpp:48-60` reads axis/dim from the type instead of attributes. | #29a DEBT under DIST-NATIVE-1 -- a design choice (sharding as types vs attributes); becomes SUPERSEDE if attributes stay. | `src/compiler/docs/deep_learning_semantic_core.md:71` | DIST-NATIVE-1; host-free M |
| <a id="tessera-collective-materialize-shard"></a>`tessera_collective.materialize_shard` | **DECLARED DEBT (#29a)** | Materialize a shard back to its payload. | Unreferenced; only its verifier (`CollectiveOps.cpp:289`). | As `shard_view`. | #29a DEBT under DIST-NATIVE-1. | as above | DIST-NATIVE-1 |

### H. Schedule IR (`schedule` dialect)

| Op | Rec. | Capability | Current state | Where it should connect | Recommendation | Evidence | Owner / host / size |
|---|---|---|---|---|---|---|---|
| <a id="schedule-optimizer-shard"></a>`schedule.optimizer_shard` | **WIRE** | Per-value ZeRO optimizer-state sharding plan (axis, policy, partitions). | Unreferenced. `OptimizerShardPass` (`tessera-optimizer-shard`) selects ops by substring (`contains("optimizer")`) and writes `tessera_sr.*` attributes from module-level options -- it would tag this op by accident and never read its axis/partitions. Origin 7c230438. | Consumer: `OptimizerShardPass.cpp:73` matches `schedule::OptimizerShardOp` exactly and takes axis/partitions from it; downstream `pipeline_runtime.py` `OptimizerShardTransport` already carries `optimizer_shard` descriptors. | WIRE (consumer slice). Collective mapping stays debt under DIST-NATIVE-1 / rocm todo :7128-7132. | `src/solvers/scaling_resilience/lib/sr/passes/OptimizerShardPass.cpp:73-103` | Host-free lit; S |
| <a id="schedule-async-copy"></a>`schedule.async_copy` | **WIRE** | Schedule-level async movement (src/dst space, stage, overlap) returning a token. | Unreferenced -- **consumers were removed**: 72443ef8 (2026-08-05) deleted the annotation-only Schedule→Tile skeleton that stamped it; 45f48565 (2026-08-11) removed EffectAnnotationPass's movement classification. `LOWERING_PIPELINE_SPEC.md:245`, `GRAPH_IR_SPEC.md:497` and `tessera_compiler_passes_overview.md:58-59` are now stale. | Consumer: `TileIRLoweringPass.cpp:1121` `LowerSchedulePrefetchToTileCopy` is the template -- lower to `tile.async_copy` (live on every backend). Producer: sibling `schedule.prefetch` has one (`AttentionFamilyPasses.cpp:167`); none for this op yet. | WIRE (consumer S-M), and fix the three stale spec lines. | `ScheduleMeshPipelineOps.td:591-607` | Host-free lit; S-M |
| <a id="schedule-await-movement"></a>`schedule.await_movement` | **WIRE** | Wait on a movement token at the true use site. | Unreferenced; same removal history. | Lower to `tile.wait_async` in the same pattern. | WIRE (same slice). | `ScheduleDialect.cpp:881` | Host-free lit; S |

### I. Tile / FA-4 Attn Tile IR

| Op | Rec. | Capability | Current state | Where it should connect | Recommendation | Evidence | Owner / host / size |
|---|---|---|---|---|---|---|---|
| <a id="tile-tmem-store"></a>`tile.tmem.store` | **RE-TIER (consumed)** | Store into Blackwell tensor memory (datacenter sm_100 only). | **Scan false negative: consumed.** `NVIDIALowering.cpp:5727-5735` matches `name.starts_with("tile.tmem.")` and its default `contractName` is `tessera_nvidia.tmem_store`; ROCm/Apple reject the prefix explicitly. A prefix match plus a default branch is invisible to a name scan. | Make the store branch explicit and add a `--lower-tile-to-nvidia` sm=100 lit. Three latent bugs found alongside (see Scan findings). | RE-TIER to consumed; keep. Hardware-gated (sm_100), not a deletion reason. | `NVIDIALowering.cpp:4943-4950` (sm_120 refusal) | Host-free lit; S |
| <a id="tessera-attn-lse-save"></a>`tessera_attn.lse.save` | **DECLARED DEBT (#29a)** | Persist finalized per-row LSE into an explicit memref checkpoint (identity, space, scope). | Fixture-only. **A producer was removed on purpose**: a5c5203a redesigned it as a memref carrier and dropped the destination-less emission. ROCm carries the choice as the `tessera.lse_checkpoint = saved\|recompute` function attribute (`rocm_native.py:861,889`) read by `TileToROCM.cpp:1932,2301`, `ROCMDirectAttention.cpp:468,575`, `NVIDIALowering.cpp:2268`. | `TileIRLoweringPass` emits save/load when a training package binds a real `row_lse` destination (the contract's stated condition); TileToROCM derives the mode from the ops instead of the attribute. | #29a DEBT under the LSE checkpoint contract (needs a queue item ID; x86 todo "Saved LSE and sparse Schedule handoff" is the nearest). | `LSE_CHECKPOINT_CONTRACT.md` | Host-free lit then Princess-Luna; M |
| <a id="tessera-attn-lse-load"></a>`tessera_attn.lse.load` | **DECLARED DEBT (#29a)** | Load LSE from the checkpoint in backward. | Fixture-only; as `lse.save`. | As `lse.save`. | #29a DEBT (same item). | as above | as above |
| <a id="tessera-attn-causal-mask"></a>`tessera_attn.causal_mask` | **DELETE candidate** | Lower-triangular mask on a score tile. | Fixture-only. **Superseded with parity**: 1b2b0680 removed `emitAttnOp("tessera_attn.causal_mask")` from `TileIRLoweringPass` and added `tessera_attn.boundary_mask` (causal BoolAttr + offsets + window), produced at `TileIRLoweringPass.cpp:688` and consumed by `TileToROCM.cpp:1817` and `StreamingAttentionToAppleGPU.cpp:169`. `attn_lower.py` names it only in a docstring. | Nothing -- `boundary_mask` carries it. | SUPERSEDE → delete candidate. | commit 1b2b0680 | Owner decision; host-free, S |
| <a id="tessera-attn-dropout-mask"></a>`tessera_attn.dropout_mask` | **DELETE candidate** | Dropout mask on attention scores. | Unreferenced; never had a producer. Superseded by counter-based `tessera_attn.block_dropout` (1b2b0680): `TileIRLoweringPass.cpp:695` → `TileToROCM.cpp:1819`, Apple :173, `scheduled_attention.py:115`. | Nothing -- `block_dropout` carries it. | SUPERSEDE → delete candidate. | commit 1b2b0680 | Owner decision; host-free, S |

### J. Solver dialects

| Op | Rec. | Capability | Current state | Where it should connect | Recommendation | Evidence | Owner / host / size |
|---|---|---|---|---|---|---|---|
| <a id="trng-create-state"></a>`trng.create_state` | **DELETE candidate** | Counter-based RNG state. | Unreferenced and **unparseable**: `src/solvers/core/dialects/CMakeLists.txt` runs TableGen into an INTERFACE library that nothing compiles or registers (the waiver's "built by CMake" overstates it). The RNG passes (`RNGLegalize.cpp:57`, `RNGStreamAssign.cpp:67`, `RNGQMCPlan.cpp:105`, Decision #18) match a *different*, unregistered name, `tessera_rng.*`. Origin 8d9a8d58. | Carrier: Graph `tessera.rng_philox_uniform/normal` (explicit key; `TesseraOps.td:2453-2500`), Python `rng.py`. | SUPERSEDE → delete candidate; separately retarget the RNG passes from `tessera_rng.*` text to the Graph Philox ops so #18 runs on real IR (S). | `src/solvers/core/dialects/CMakeLists.txt:3-24` | Owner decision; host-free, S |
| <a id="trng-uniform"></a>`trng.uniform` | **DELETE candidate** | Uniform draw. | As `trng.create_state`. | `tessera.rng_philox_uniform`. | SUPERSEDE → delete candidate. | as above | host-free, S |
| <a id="trng-normal"></a>`trng.normal` | **DELETE candidate** | Normal draw. | As above. | `tessera.rng_philox_normal`. | SUPERSEDE → delete candidate. | as above | host-free, S |
| <a id="tsl-root-newton"></a>`tsl.root.newton` | **MERGE/SUPERSEDE** | Newton root solve with autodiff derivative. | Unreferenced and unparseable (same INTERFACE-only build). | `tessera_solver.implicit{residual=@f}` (produced by `implicit_solver.py`, consumed by `NewtonAutodiff.cpp:83` -- which differentiates through a solution; it does not run a forward Newton). | MERGE into `tessera_solver.implicit` with a forward `method` attribute; the forward solve is #29a debt under AD-SOLVER-IFT-1 ("extend residual/predicate/solver envelopes"). | `autodiff/implicit.py:451,547` root_vjp/root_jvp | AD-SOLVER-IFT-1; host-free, M |
| <a id="tsl-root-brent"></a>`tsl.root.brent` | **MERGE/SUPERSEDE** | Bracketed Brent root solve. | As `tsl.root.newton`. No forward Brent exists anywhere. | `tessera_solver.implicit{method=brent}`. | MERGE (same item). | as above | AD-SOLVER-IFT-1 |
| <a id="tsl-solve-trig"></a>`tsl.solve_trig` | **MERGE/SUPERSEDE** | Periodic θ solve. | As above. | `tessera_solver.implicit` + a period/wrap attribute (`TrigInit.cpp` is unrelated: FFT-plan attributes). | MERGE (same item). | as above | AD-SOLVER-IFT-1 |
| <a id="tss-cg"></a>`tss.cg` | **MERGE/SUPERSEDE** | Conjugate gradient with preconditioner region. | Unreferenced and unparseable. | `tessera_solver.linear_solve{linear_solver=cg}` (`implicit_solver.py:712,1047` → `schedule.solver_ift`); execution `emit/nvidia_solver_krylov.py`; `SparseSolverSpecialize` picks cg/gmres/bicgstab by attribute. | MERGE (#31) into `linear_solve`. | `tests/performance/nvidia/test_solver_krylov_ratchet.py` | host-free |
| <a id="tss-gmres"></a>`tss.gmres` | **MERGE/SUPERSEDE** | GMRES. | As `tss.cg`. | `tessera_solver.linear_solve{linear_solver=gmres}`. | MERGE. | as above | host-free |
| <a id="tss-spmm"></a>`tss.spmm` | **MERGE/SUPERSEDE** | Sparse C = A·B. | As above. | Graph `tessera.spmm_csr/spmm_coo` (catalog-only, lowering=sparse) + x86 `avx512_sparse_f32.cpp`. | MERGE; promoting `spmm_*` to ODS is #29a debt under TSOL-PHYS-TAIL-1. | `op_catalog.py:298-299` | TSOL-PHYS-TAIL-1 |
| <a id="tss-spmv"></a>`tss.spmv` | **MERGE/SUPERSEDE** | Sparse y = A·x. | As above. No spmv anywhere else. | `tessera.spmm_csr` with a vector RHS. | MERGE (same item). | as above | TSOL-PHYS-TAIL-1 |
| <a id="tessera-solver-potrf"></a>`tessera_solver.potrf` | **MERGE/SUPERSEDE** | Cholesky factor with a precision policy. | Fixture-only, but the fixture `src/solvers/linalg/test/solver/spd_solve.mlir` is dead (no lit.cfg; RUN names `-tessera-solver-legalize`/`-tessera-mixed-precision-schedule`, which do not exist). **The capability bypasses this dialect**: Graph `tessera.cholesky/tri_solve/cholesky_solve/lu` lower 1:1 to `tile.*` in `TilingPass.cpp:464-479`, then Apple LAPACK (`TileToApple.cpp:206-217`), ROCm `GenerateROCMLuKernel.cpp`, x86 `avx512_lu_qr_f32.cpp`. A Graph→`tessera_solver` lowering would be a second lowering authority. | Carrier `tessera.cholesky`. The dialect's only unique content is the precision policy (fp8 factor, f32 accumulate) -- it must ride `numeric_policy` on the Graph op first. | MERGE (#31); delete only after the precision policy has a carrier (NUMPOL-CARRIER-1). | `Tessera_Linalg_Solvers_Spec.md:29-80` names passes never built | NUMPOL-CARRIER-1; host-free |
| <a id="tessera-solver-potrs"></a>`tessera_solver.potrs` | **MERGE/SUPERSEDE** | Cholesky solve. | As `potrf`. | `tessera.cholesky_solve`. | MERGE (same). | as above | NUMPOL-CARRIER-1 |
| <a id="tessera-solver-getrf"></a>`tessera_solver.getrf` | **MERGE/SUPERSEDE** | LU with pivots. | Unreferenced; as `potrf`. | `tessera.lu`. | MERGE (same). | as above | NUMPOL-CARRIER-1 |
| <a id="tessera-solver-trsm"></a>`tessera_solver.trsm` | **MERGE/SUPERSEDE** | Triangular solve. | Unreferenced; as `potrf`. | `tessera.tri_solve`. | MERGE (same). | as above | NUMPOL-CARRIER-1 |
| <a id="tessera-solver-ir-step"></a>`tessera_solver.ir_step` | **DECLARED DEBT (#29a)** | One mixed-precision iterative-refinement step. | Unreferenced. `IterativeRefinement.cpp:58-88` only tags ops with `ir_max_iter`/`ir_tol` for a "later canonicalization" that does not exist. | IterativeRefinement materializes `ir_step` in an `scf.for` around `tessera.solve`. | #29a DEBT under TSOL-PHYS-TAIL-1 (or MERGE as a refinement policy on `tessera.solve`). | `IterativeRefinement.cpp:58-88` | TSOL-PHYS-TAIL-1; host-free, M |
| <a id="tessera-sr-export-manifest"></a>`tessera_sr.export_manifest` | **WIRE** | Deployment-manifest sink (path, include list). | Unreferenced. The consumer never came along when the op moved out of the archive: `ExportDeploymentManifestPass.cpp:20-56` counts ops module-wide and hard-codes `manifest.json`, ignoring the op; its fixture sits in no lit suite. | The pass reads `tessera_sr.export_manifest{path, include}` as its sink directive (fallback only with a diagnostic, #21a) and erases the op. | WIRE. No item fits; nearest DIST-NATIVE-1. | `archive/src/tessera_scaling_resilience_v1/tests/sr-export-manifest.mlir:2` | Host-free lit; S |
| <a id="tessera-spectral-twiddle-table"></a>`tessera_spectral.twiddle_table` | **DECLARED DEBT (#29a)** | Materialize a (possibly quantized) per-stage twiddle table. | Unreferenced. **`LegalizeSpectral.cpp:15-18` is a stale comment**: it promises one `twiddle_table` per stage, but the pass body (lines ~111-209) only sets attributes (stages, per_axis_len, half_spectrum, norm, direction, legalized); `git log -S` shows no materialization ever existed. Twiddles are computed at run time (`TargetHooks/CPU/StockhamRadix4.cpp:136`) and twiddle identity rides `twiddle_policy`/`twiddle_layout` on the Graph route. | LegalizeSpectral emits one op per static stage; LowerSpectralToTargetIR (and MXP quantization) consume it. | #29a DEBT under TSOL-SCALE-1 (its gate names plan/twiddle/workspace identity); fix the stale comment now. | `scheduled_fft.py:72,312`; `spectral_plan.py:114` | TSOL-SCALE-1; host-free ts-spectral-opt lit, M |
| <a id="tessera-clifford-ext-deriv"></a>`tessera_clifford.ext_deriv` | **DECLARED DEBT (#29a)** | Exterior derivative (central difference, typed algebra/spacing). | Unreferenced; `test_clifford_dialect_wiring.py:203` skips it. Python reference `ga/calculus.py`, JVPs, Apple shim dispatch; catalog `tessera.clifford_ext_deriv` lowering=stencil. | Producer: `native_clifford_gpu.py` `clifford_gpu_skeleton` (add to `CLIFFORD_GPU_OPS`); consumer: `ExpandProductTable.cpp` (Σ e_i ∧ shifted difference, reusing the wedge table). | #29a DEBT under W6.4, which names "the field ops (ext_deriv, codiff, vec_deriv, integral)" as open. | `DOMAIN_AUDIT.md:194` | W6.4; host-free lit + CPU JIT, M |
| <a id="tessera-clifford-vec-deriv"></a>`tessera_clifford.vec_deriv` | **DECLARED DEBT (#29a)** | Geometric (vector) derivative. | As `ext_deriv`. | As above (geometric-product table). | #29a DEBT under W6.4. | as above | W6.4; M |
| <a id="tessera-clifford-codiff"></a>`tessera_clifford.codiff` | **DECLARED DEBT (#29a)** | Codifferential ±⋆d⋆. | As above. | hodge ∘ ext_deriv ∘ hodge in ExpandProductTable. | #29a DEBT under W6.4. | PR #688 (Python `clifford_codiff`) | W6.4; S |
| <a id="tessera-clifford-integral"></a>`tessera_clifford.integral` | **DECLARED DEBT (#29a)** | Riemann integral over a manifold. | As above (not in the catalog). | Weighted reduction in ExpandProductTable. | #29a DEBT under W6.4. | `backend_manifest.py:4917` Apple kernel | W6.4; S |
| <a id="tessera-ebm-partition-z"></a>`tessera_ebm.partition_z` | **WIRE** | Partition function Z by exact / monte_carlo / annealed (AIS). | Fixture-only (parse/print fixtures; benchmarks read fixture text, emit nothing). Its ODS description ("handled by the EBM-pipeline lowering passes") is false: `LowerLangevin.cpp:543-557` refuses it. | `LowerPartitionZ` beside `LowerEnergy`/`LowerInnerStep`: monte_carlo/annealed reuse the energy_fn call + on-device Philox; exact needs a support operand. | WIRE (owner: the EBM stream of W4-PRODUCT-1). | `python/tessera/ebm/partition.py:45-342` | CPU JIT (any box) then gfx1151/gfx1201/sm_120; M |

### K. Target IR — x86 (`tessera_x86`)

| Op | Rec. | Capability | Current state | Where it should connect | Recommendation | Evidence | Owner / host / size |
|---|---|---|---|---|---|---|---|
| <a id="tessera-x86-amx-tile-load"></a>`tessera_x86.amx_tile_load` | **DIRECTED-CLOSED** | Typed AMX tile load (`!tessera_x86.tile`, stride). | Fixture-only. Origin 0003e9d7 (W0.10); never produced. Its only consumer is the dialect's own type constraint (the negative fixture `x86_target_ir_invalid.mlir:17`). | Nothing, by direction: AMX is a closed target, superseded by ACE (deferred until shipping hardware). | DIRECTED-CLOSED CONTRACT. Mark at the site and keep counted; no lowering. | `docs/audit/backend/x86/todo.md:1104-1108` | none |
| <a id="tessera-x86-amx-tile-store"></a>`tessera_x86.amx_tile_store` | **DIRECTED-CLOSED** | Typed AMX tile store. | As above. | As above. | DIRECTED-CLOSED CONTRACT. | as above | none |
| <a id="tessera-x86-amx-tile-zero"></a>`tessera_x86.amx_tile_zero` | **DIRECTED-CLOSED** | Zero an AMX tile. | As above. | As above. | DIRECTED-CLOSED CONTRACT. | as above | none |
| <a id="tessera-x86-amx-dpbf16ps"></a>`tessera_x86.amx_dpbf16ps` | **DIRECTED-CLOSED** | AMX bf16 dot-product accumulate. | As above. | As above. | DIRECTED-CLOSED CONTRACT. | as above | none |
| <a id="tessera-x86-amx-dpbusd"></a>`tessera_x86.amx_dpbusd` | **DIRECTED-CLOSED** | AMX int8 dot-product accumulate. | Unreferenced; its description already records the 2026-08-02 owner direction. | As above. | DIRECTED-CLOSED CONTRACT. | as above | none |
| <a id="tessera-x86-avx512-gemm-microkernel"></a>`tessera_x86.avx512_gemm_microkernel` | **DECLARED DEBT (#29a)** | Register-blocked GEMM shape directive (m, n, k, dtype). | Fixture-only. `LowerMatmulToX86` (`TileToX86Pass.cpp:222-310`) goes straight to `func.call tessera_x86_avx512_gemm_bf16`; the only `tessera_x86` op it builds is `abi_call` (`:1384`). Python sends matmul to the generic `tessera_x86.kernel`. `dtype` is `DefaultValuedAttr "f32"` while the only live GEMM is bf16 -- a #21a default on a semantic key. | Producer: `LowerMatmulToX86` non-AMX path next to the `abi_call` marker; consumer: the future `x86vector.*` AVX-512 lowering (the x86 queue's named follow-on). | #29a DEBT; the `x86vector` follow-on has no item ID -- attach to X86-SPINE-1 ("reconcile C synthesis with the MLIR/LLVM lane"). Make `dtype` required. | `x86/todo.md:3078-3094,3132` | Host-free lit; execute on Princess-Luna/Tajasarus once x86vector exists; M |
| <a id="tessera-x86-pack-b-panel"></a>`tessera_x86.pack_b_panel` | **DECLARED DEBT (#29a)** | B-panel pack directive (kc, nc). | As `avx512_gemm_microkernel`. | As above. | #29a DEBT (same item). | as above | as above |
| <a id="tessera-x86-elementwise"></a>`tessera_x86.elementwise` | **MERGE/SUPERSEDE** | Attribute-less pointwise marker. | Fixture-only. The Python fallback emits `tessera_x86.kernel` (which carries source/abi/status/runtime_lane that `_verify_x86_op` requires, `target_ir.py:814-824`); C++ lowers pointwise to a native call with an `abi_call` marker. | `tessera_x86.kernel` already carries strictly more contract. | MERGE/SUPERSEDE into `tessera_x86.kernel` (alt: WIRE for ROCm symmetry at `target_ir.py:1582`, S). | `target_ir.py:1550-1589` | host-free |

### L. Target IR — NVIDIA, ROCm, Apple

| Op | Rec. | Capability | Current state | Where it should connect | Recommendation | Evidence | Owner / host / size |
|---|---|---|---|---|---|---|---|
| <a id="tessera-nvidia-mma-fused"></a>`tessera_nvidia.mma_fused` | **WIRE** | sm_120 `mma.sync` GEMM with fused epilogue. | Fixture-only (`sm120_differentiation_target_ir.mlir`); the ODS declares no arguments, so the contract lives only in fixture attributes. The capability executes through Python candidates that bypass Target IR (`NvidiaMmaFusedCandidate`, `emit/nvidia_cuda.py:4781`) -- yet `SM120_DIFFERENTIATION_DASHBOARD.md:13-16` cites this op as Target-IR evidence for promoted rows. | Producer: `_lower_nvidia_op` sm_120 branch (`target_ir.py:1592-1705`) emits it for fused-epilogue matmul instead of generic `cuda_kernel`; consumer: `_verify_nvidia_op` cases (mirror `mma_sync`, `target_ir.py:893`) + `markerForTargetOp` (`NVIDIALowering.cpp:5767`). Declare real arguments (#21a). | WIRE. Nearest owner NVIDIA-SPINE-1. | origin 15487733 | Host-free for the IR slice (runtime evidence already exists on Super-Bear); M together |
| <a id="tessera-nvidia-mma-attention"></a>`tessera_nvidia.mma_attention` | **WIRE** | sm_120 two-MMA attention with f32 softmax. | As `mma_fused`; executes via `NvidiaMmaAttnCandidate` (`nvidia_cuda.py:5160`). | Emit for `tessera.flash_attn` on sm_120; same consumers. | WIRE (same slice). | as above | as above |
| <a id="tessera-nvidia-fpquant"></a>`tessera_nvidia.fpquant` | **WIRE** | FP8/FP6/FP4 storage quantize/dequantize. | As `mma_fused`; executes via `run_fpquant_f32` (`nvidia_cuda.py:1949`; `execution_matrix.py:4399-4408`). | Emit for quantize sources on sm_120; same consumers. | WIRE (same slice). | as above | as above |
| <a id="tessera-nvidia-func"></a>`tessera_nvidia.func` | **DELETE candidate** | Per-target function container. | Fixture-only. Registered in 0e4e3cc8 because Python then emitted it; **its producer was removed in W0.9 (0003e9d7)** in favour of `func.func` -- `target_ir.py:734-748` documents the replacement. Not a runtime-assembled false negative: no `{dialect}.func` f-string exists in `python/`. The ODS comment (`TesseraNVIDIADialect.td:249-251`) is stale. | `func.func`. | SUPERSEDE → delete candidate (with its fixture lines and the stale comment). | `target_ir.py:734-748` | Owner decision; host-free, S |
| <a id="tessera-rocm-memcpy"></a>`tessera_rocm.memcpy` | **DELETE candidate** | Synchronous device copy (dst, src, bytes, spaces). | Unreferenced for 13 months. A scaffold copy of the Cerebras `memcpy` from a778907a ("new backends"); never in any ROCm fixture. Host↔device copies are runtime ABI (`hipMemcpy` in `runtime/hip/*.cpp`); device copies use `tessera_rocm.async_copy` (produced `TileToROCM.cpp:3790`, consumed by `LowerROCMAsyncCopyToLoop.cpp`, `TesseraTargetToROCDL.cpp:281,400`). | `tessera_rocm.async_copy` + `wait` (identical operands plus a token). | SUPERSEDE → delete candidate. | `TesseraROCMOps.td:7-18` | Owner decision; host-free, S |
| <a id="tessera-rocm-emit"></a>`tessera_rocm.emit` | **DELETE candidate** | Attribute-only Terminator. | Unreferenced; same Cerebras scaffold origin. No ROCm region op needs this terminator; emission is keyed on the `tessera_rocm.kernel` function attribute by the `tessera_rocm_emit` tool and `emit/rocm_hip.py`. | Nothing -- emission is a driver step, not an op. | DELETE candidate. | `TesseraROCMOps.td:1532-1534`; `tools/tessera_rocm_emit.cpp` | Owner decision; host-free, S |
| <a id="tessera-apple-gpu-mps-softmax"></a>`tessera_apple.gpu.mps_softmax` | **MERGE/SUPERSEDE** | MPS softmax artifact. | Unreferenced. Its own ODS comment says softmax was artifact-only in Phase 8.3 and 8.4 would broaden it; 8.4 chose a custom MSL kernel instead. Today: Python emits `gpu.msl_kernel` + `gpu.mps_dispatch` (`target_ir.py:1926-1945`), C++ `SoftmaxToAppleGPU.cpp` calls `tessera_apple_gpu_softmax_*_status`, and an MPSGraph softmax symbol also exists. | `tessera_apple.gpu.msl_kernel` + `gpu.mps_dispatch`. | MERGE/SUPERSEDE (#31). If an MPS-backed softmax lane is wanted: WIRE at `target_ir.py:1926` when the MPSGraph lane is selected + `_require` at :848 (Mac, S). | origin 3554158b; `TesseraAppleOps.td:305-310` stale comment | Mac if wired |

