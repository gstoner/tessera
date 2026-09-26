---
last_updated: 2026-09-26
audit_role: reference
owning_plan_item: W1.1 / ROCm backend
---

# The ROCm lane map — frontend to hsaco

The current executing routes, their exact-target evidence, and the older
directive-lane snapshot that explains why the scheduled route was built.

## Current lane map (2026-09-21)

| Lane | IR path | Exact targets | Evidence boundary |
|---|---|---|---|
| Generic compiled/directive runtime | Python/runtime-selected Target directives → generated `gpu.func` → ROCDL → hsaco | Broad exact-device evidence on `gfx1151`; many family plugins also execute on `gfx1201` through the promotion gate. | The public execution matrix is still primarily joined at generic `rocm`/gfx1151 grain. |
| Content-addressed family pipeline | typed Graph contract → registered family plugin → replay-bound Tile/Target package → ROCDL → hsaco | Every `FAMILY_PLUGINS` entry is promoted on `gfx1201`; every gfx11-capable entry is promoted on `gfx1151`. `gfx1200` has none. | `rocm_pipeline.promoted_families()` is the fail-closed source of truth. |
| Scheduled native package | Graph → Schedule → Tile → Target → native image + launch descriptor | `gfx1201` owns bounded softmax/reduction, matmul, attention/backward, depth-attention, paged-KV, and sparse-2:4 ABIs; gfx1151 owns its separately proved set. | `runtime._gfx1201_proved_scheduled_abis()` rejects unproved ABI/architecture pairs. The unified RX 9070 XT gate accounts for all 90 ordinarily skipped rows and passes 95/95 with zero skips on a fresh compiler. |
| Exact MXFP4 native package | Named packed W4A8 Graph container contract → Schedule → scaled-partial Tile carrier → self-contained gfx1201 Target ABI → strict materializer → versioned checkpoint/transposed/fragment layout selection → load-time conversion → contiguous lane-word FP8-WMMA HSACO; scalar HIP remains the oracle | `gfx1201` only: exact rows include M1/5/64, N48/80, M64/65 crossover, long K, wide N, poisoned rows, and HIP graph capture on RX 9070 XT. Decode fragment timing is matched against Radiance/libr4d; prefill retains the proved transposed/group-M route. | Checkpoint and AITER-shuffled forms are not launch-compatible with fragment order; refusals are receipted. MXFP4 is not promoted to a first-class Graph storage dtype. Padded multistage prefill and a backend-neutral K-step barrier remain open. |
| Public per-target capability/executor registry | capability row + execution-matrix row → generated target dashboards | Registry join is complete for existing gfx1151 lanes and the bounded proved gfx1201 `matmul`, `flash_attn`, and `softmax` subset. | Proof projection is deliberately narrower than family promotion and scheduled-ABI admission. |
| gfx1200 artifact lane | ISA/profile/feature modeling → compile or artifact validation | `gfx1200` only. | No promoted family, runtime ABI, numerical device proof, topology default, or performance claim. |

The same RDNA 4 feature tables serve `gfx1200` and `gfx1201`, but package
admission and evidence are exact-target. Shared ISA support does not authorize
launching a gfx1201 package on gfx1200 or copying performance conclusions.

The D=128 `linear_attn` gap is closed on gfx1201: the generator batches
independent loads before their consumers, following the `lds-copy-depth`
staging pattern. On RX 9070 XT the exact candidate passes its numerical oracle,
reduces full load drains from 81 to 16, and improves paired device timing by
1.284x. The [evidence packet](../../../../benchmarks/baselines/gfx12_linear_attn_load_batch_20260921/README.md)
owns that exact claim. `gfx1200` remains unpromoted pending matching-device
proof.

## Decision — expander adoption is (a) (owner, 2026-09-26)

Every ROCm family is to be entered from Tile IR, produced by the scheduled
route (Graph → Schedule → Tile) from a Graph contract. **The expander passes
themselves are not being rewritten away:** a `generate-*` pass remains the
Tile → Target generator for its family. What goes is the entry point, a Target
IR directive composed as a string in Python, which skips Graph, Schedule and
Tile. Sequencing follows (b)'s order — GEMM, flash-attention and
linear-attention first (the three expanders that already consume the Tile
fragment types), then the long tail — with progress measured family by family
by the E2E-REAL-6F route census, not asserted here. Recounted 2026-09-26: 78
`Generate*.cpp` expanders (71 in the snapshot below); 3 consume Tile fragment
types (0 in the snapshot).

## Decision — Lane B is retired (owner, 2026-09-26)

Lane B (`runtime._build_canonical_gemm_hsaco`: Graph IR → `tessera-tiling` →
`tessera-tile-ir-lowering` → generator with `via-tile=false`) skipped Schedule
IR, so its schedule arrived as pass options rather than a replayed Schedule
contract, and it was a second Graph → Tile authority for GEMM beside the
scheduled route (Decision #31). Skipping Schedule IR is not a fast path:
compile time is negligible either way (both routes are content-cached), and
no runtime advantage was ever measured — **the two were not measured head to
head**. They do not produce the same kernel: Lane B entered the generator
through the canonical `scf.for` matcher with `via-tile=false`, the scheduled
route enters through the typed `tile.matmul_kernel` adapter with
`via-tile=true`. Resource counts differ too (Lane B's packet: 0 spills; the
scheduled packet: 45-48 on f16/bf16) across a toolchain change (ROCm 7.14 →
10.0), so no regression or speed claim is made in either direction.
Retired in one change:

* **Correct path first.** `runtime.build_canonical_gemm_hsaco` builds a ROCm
  GEMM from a Graph module (the traced frontend reaches the same
  lower-and-package authority through `driver.py`): `lower_scheduled_matmul` (Graph → Schedule →
  Tile, replay-checked) then `package_scheduled_matmul` (Tile → `tessera_rocm`
  → HSACO). The runtime's compiled GEMM now builds through it.
* **Benchmark rebuilt.** `benchmark_rocm_canonical_gemm_kloop.py` measures
  that route, launches from the package's own descriptor, requires an explicit
  `TESSERA_ROCM_CHIP` (gfx1151 or gfx1201), and stamps route, schedule digest
  and image digest per row. Lane B's committed packet
  (`rocm_gfx1151_canonical_gemm_kloop.json`) stays as that route's record and
  is not this one's baseline.
* **Deleted.** `_build_canonical_gemm_hsaco` and the `input=graph` matmul
  branch of `tessera-rocm-executable`; both the Python pipeline config and the
  C++ contract pass now refuse `family=matmul input=graph`, with a negative
  lit fixture (`executable_pipeline_graph_matmul_options.mlir`). Attention
  keeps its Graph entry.
* **Census.** The E2E-REAL-6F package census and `bootstrap_prune_gap` were
  re-run before and after: counts identical (46 graph / 14 scheduled / 15 raw)
  and ROCm GEMM's only package is `package_scheduled_matmul` on the
  `scheduled` boundary, so no family lost its only lowering. **This evidence
  is weak by construction:** Lane B was never a `package_*`, so the census
  cannot see it, and cannot show what Lane B uniquely carried.
* **Physical consumer deleted (#29/#31), 2026-09-26.** Lane B's consumer in
  `GenerateWMMAGemmKernel.cpp` had no production producer left: the only
  producers of `tessera.canonical_k_step` are `TilingPass` and
  `TileIRLoweringPass`, and no ROCm pipeline (C++ `tessera-lower-to-rocm`,
  `tessera-rocm-executable`, or any Python pass list) runs `tessera-tiling`
  ahead of `generate-wmma-gemm-kernel` — the scheduled route arrives as
  `tile.matmul_kernel`, the directive route as `tessera_rocm.wmma_gemm`.
  Deleted: the canonical-step matcher (`matchCanonicalGemmLoop`,
  `WmmaGemmRequest::canonicalKLoop` and the fields only it filled), the
  one-wave LDS comparison body (`emitCanonicalLdsBody` and its arch/knob
  guards), the `canonical_mnk_scf_for` source stamp, and the two diagnostic
  codes only that body emitted (`ROCM_CANONICAL_LDS_{ARCH,KNOB}_UNSUPPORTED`).
  A `canonical_k_step` step that still reaches the generator is now refused
  by name (`ROCM_CANONICAL_GEMM_LOOP_RETIRED`) instead of being matched, and
  `canonical-staging=lds` without `via-tile=true` is refused instead of
  silently producing the register body. Kept: the `canonical-staging` option
  (the typed LDS body and split-K read it), `TilingPass`'s marker (Apple's
  `tessera-apple-canonical-gemm` consumes it), and the typed, directive and
  split-K paths. Tests: `canonical_lds_arch_guard.mlir` deleted;
  `canonical_lds_arch_refused.mlir` → `rocm_canonical_gemm_loop_retired.mlir`
  (the new refusal, both marker carriers); `graph_matmul_generator_knobs.mlir`
  → `tile_matmul_generator_knobs.mlir` (the live `sched-groups` half at Tile
  entry, plus the direct-lane `lds` refusal); in
  `test_rocm_wmma_gemm_generated.py` the two positive canonical-loop tests are
  deleted and the malformed-marker test asserts the new refusal. Lane B's
  benchmark packet stays as that route's record.

## Historical snapshot (2026-08-05)

The remainder records the earlier gfx1151 directive-versus-canonical-lane
measurement. Its caller counts and statements that Schedule/Tile were absent
describe that dated tree; they are not current support claims.

---

### There were two GEMM lanes, and only one ran

### Lane A — the directive lane (this is production)

```
Python runtime
  └─ hand-built one-op MLIR string, already at TARGET IR level:
       "tessera_rocm.wmma_gemm"() {name, m, n, k, mt, nt, dtype, …}
  └─ tessera-opt --pass-pipeline=builtin.module(
       generate-wmma-gemm-kernel,          ← synthesizes gpu.func directly
       lower-tile-to-rocm{arch=gfx1151},   ← NO-OP on the default path
       lower-tessera-target-to-rocdl,
       gpu.module(convert-scf-to-cf, convert-gpu-to-rocdl,
                  reconcile-unrealized-casts),
       rocdl-attach-target{chip=gfx1151},
       gpu-module-to-binary)
  └─ hsaco
```

**Graph IR: none. Schedule IR: none. Tile IR: none** (unless `via-tile=true`).
The pipeline's entry point is a Target-IR directive that Python composes as a
string.

### Lane B — the canonical lane (Graph IR → Tile IR)

```
Python runtime
  └─ real Graph IR:  "tessera.matmul"(%a, %b) : (tensor, tensor) -> tensor
  └─ tessera-opt --pass-pipeline=builtin.module(
       tessera-tiling,
       tessera-tile-ir-lowering,           ← Tile IR appears HERE
       rocm-wave-lds-pipeline,
       rocm-wave-lds-legality,
       generate-wmma-gemm-kernel{canonical-staging=…},
       lower-tile-to-rocm{arch=gfx1151},
       lower-tessera-target-to-rocdl,
       … → gpu-module-to-binary)
  └─ hsaco
```

### Which one runs

| | callers (whole tree, excluding its own definition) |
|---|---|
| `_build_compiled_gemm_hsaco` (Lane A) | **11** |
| `_build_canonical_gemm_hsaco` (Lane B) | **1** — and it is `benchmarks/rocm/benchmark_rocm_canonical_gemm_kloop.py` |

```bash
grep -rn "_build_canonical_gemm_hsaco" --include=*.py . | grep -v "def _build_canonical"
grep -rn "_build_compiled_gemm_hsaco"  --include=*.py . | grep -v "def _build_compiled" | wc -l
```

**The Graph-IR lane has zero production callers. Its only consumer is a
benchmark script.**

---

### The stack in that snapshot

| Layer | Status on the executing ROCm lane |
|---|---|
| Python frontend | Present — but it emits a **Target-IR directive string**, not Graph IR |
| **Graph IR** | **Bypassed.** Only Lane B builds `tessera.matmul`, and Lane B is benchmark-only |
| **Schedule IR** | **Absent entirely.** 0 references to `graph-to-schedule` / `schedule-to-tile` in `runtime.py`, and `TesseraPM` is not linked into `tessera-opt` — consistent with the plan's own note that those passes are annotation-only skeletons in the test binary |
| **Tile IR** | Bypassed on Lane A; present on Lane B via `tessera-tile-ir-lowering` |
| **Target IR** (`tessera_rocm.*`) | **This is where the executing lane starts** |
| ROCDL → hsaco | Present on both |

So on the lane that actually runs, the "four-layer IR stack" is **one layer**:
Target IR. On the benchmark-only lane it is three (Graph → Tile → Target); it is
never four, because Schedule IR is not in any pipeline.

---

### The expander population in that snapshot

The directive lane's work is done by `generate-<op>-kernel` passes, each of which
expands a one-op directive into a `gpu.func` it synthesizes itself.

```bash
ls src/compiler/codegen/Tessera_ROCM_Backend/lib/Conversion/Generate*.cpp | wc -l   # 71
grep -oE "generate-[a-z0-9-]+" python/tessera/runtime.py | sort -u | wc -l          # 58
grep -rl "tile::ViewOp\|tile::FragmentPackOp" \
  src/compiler/codegen/Tessera_ROCM_Backend/lib/Conversion/Generate*.cpp | wc -l    # 0
```

| measure | count |
|---|---:|
| `Generate*.cpp` expanders in the ROCm backend | **71** |
| distinct `generate-*` passes the runtime drives | **58** |
| expanders that consume `tile.view` / `tile.fragment_pack` | **0** |
| runtime pipelines that include `lower-tile-to-rocm` | **2** of 9 |
| runtime pipelines that include `lower-tessera-target-to-rocdl` | **5** of 9 |

> **Corrected after #522 review.** The first version of this table published 4
> and 8. Those were `grep -c` substring counts over `runtime.py`, which also
> match the long explanatory COMMENTS next to the two GEMM builders. Counting
> actual pipeline entries — a quoted, comma-terminated pass name inside a
> pipeline string — gives 2 and 5. Denominator 9 is the number of pipelines that
> terminate in `gpu-module-to-binary`, which is unchanged.
>
> ```bash
> grep -cE '^\s*f?"lower-tile-to-rocm' python/tessera/runtime.py            # 2
> grep -cE '^\s*f?"lower-tessera-target-to-rocdl' python/tessera/runtime.py # 5
> grep -cE '^\s*f?"gpu-module-to-binary' python/tessera/runtime.py          # 9
> ```
>
> The correction **strengthens** the finding rather than weakening it: only two
> of nine runtime pipelines reach `lower-tile-to-rocm` at all, and both are the
> GEMM lanes (A and B).

Every expander does its own lane math and emits `tessera_rocm.*` plus raw
`vector`/`memref` ops. **None consumes the Tile fragment contract.**

---

### Where the W1 work sat — the consequence

W1.1's typed `!tile.fragment` chain (steps 1, 2, 0, 3a) lives inside
`lower-tile-to-rocm`. On the executing lane that pass is present **and is a
verified no-op** — `runtime.py`'s own comment records byte-identical hsaco with
and without it on the default path.

So, stated plainly:

* **W1.1 has no effect on any executing ROCm kernel today.** It is a capability
  with no producer — precisely the shape Decision #29 exists to flag, and the
  reason step 3 is the item that matters rather than one item among five.
* **Step 3 puts the typed contract on the GEMM lane**, by migrating
  `generate-wmma-gemm-kernel`. That is what makes W1.1 affect a running kernel.
  Separately — and *not* a gate on step 3 or step 5 — the other 57 expanders
  would each need the same treatment before "the ROCm backend goes through
  Tile IR" is true. That second statement is the adoption question of §5.
* **Step 5 is NOT gated by the expander population — a correction to this
  document's first version.** It claimed step 5 sat "behind the other 57"
  expanders. That is wrong, and it conflated two independent things.
  `MMAOp::verify()`'s permissive branch governs only producers that emit
  `tile.mma` with bare fragments, and those are enumerable: **5 creation sites
  in 4 files** —

  | site | note |
  |---|---|
  | `GenerateWMMAGemmKernel.cpp:465` | only when `via-tile=true` |
  | `GenerateWMMALinearAttnKernel.cpp:143` | only when `via-tile=true` |
  | `GenerateWMMAFlashAttnKernel.cpp:230` | only when `via-tile=true` |
  | `TileIRLoweringPass.cpp:924`, `:997` | the Graph-IR (Lane B) path |

  plus the Python emitters (step 4). That is exactly the "five construction
  sites plus Python emitters" the W1.1 plan already scoped. An expander that
  never emits `tile.mma` cannot block deleting a `tile.mma` verifier branch.

  Keep the two costs separate:

  | question | cost |
  |---|---|
  | close the Tile **fragment contract** (steps 3–5) | 5 C++ sites + Python emitters |
  | make the ROCm backend **go through Tile IR** | 58 expanders — unpriced, §5 |

  Only the second scales with the expander population. Merging them would defer
  a closeable contract cleanup behind unrelated codegen.

This does not make W1.1 wrong — a composable typed lowering is a precondition
for any of that migration, and steps 0/3a removed two blockers that were
genuinely blocking. It does mean **the W1.1 row's "5w" estimate covers building
the contract, not adopting it**, and the adoption cost scales with the expander
population, not with the number of remaining W1.1 steps.

---

### What that implied for sequencing

1. **Distinguish closing the contract from adopting it.** Steps 3–5 close the
   Tile fragment contract across 5 C++ creation sites plus the Python emitters,
   and are finishable on that scope. Making the ROCm backend actually traverse
   Tile IR is the separate 58-expander question below. "Step 3 of 6" measures
   the first; it says nothing about the second.
2. **Decide adoption policy before mass migration.** Three options, and the
   choice is a project-endpoint decision, not a refactor detail:
   - *(a) migrate all 58 expanders* — the largest option, and the only one that
     makes the layered-stack claim true for ROCm;
   - *(b) migrate the performance-critical family only* (GEMM, flash-attn,
     linear-attn) and leave the long tail as direct Target-IR expanders,
     documented as such;
   - *(c) keep expanders as they are and treat Tile IR as an optional lane* for
     ops that benefit from shared tiling/pipelining.
   Decision #28's three-tier model is compatible with (b) and (c); the current
   plan text implies (a) without pricing it.
3. **The Schedule IR gap is separate and larger.** No pipeline contains a
   Schedule IR pass, and the C++ passes are annotation-only skeletons not linked
   into the production driver. Any claim that the stack is four-layer should
   name this.
4. **Lane B deserves a decision too.** A Graph-IR → Tile-IR lane exists, is
   compiled, and is exercised only by a benchmark. Either it becomes the
   canonical front door (and Lane A becomes a fast path), or it is a declared
   oracle with a differential test (Decision #31), or it should not be carried.

---

### Re-running the historical measurements

```bash
# Lane pipelines
sed -n '/def _build_compiled_gemm_hsaco/,/return hsaco/p' python/tessera/runtime.py
sed -n '/def _build_canonical_gemm_hsaco/,/return hsaco/p' python/tessera/runtime.py

# Stack passes present in the runtime at all
for p in tessera-graph-to-schedule tessera-schedule-to-tile tessera-lower-to-rocm; do
  printf "%-28s %s\n" "$p" "$(grep -c "$p" python/tessera/runtime.py)"
done
```
