---
last_updated: 2026-08-05
audit_role: reference
owning_plan_item: W1.1 / ROCm backend
---

# The ROCm lane map — frontend to hsaco, measured

What actually executes on gfx1151, layer by layer, and where the W1 Tile-typing
work sits relative to it. Every number here was measured on 2026-08-05 against
the working tree; the commands are given so each can be re-run.

This exists because W1.1 has been sized as *"compiler-contract work on the
Tile IR"* without anyone stating which executing lane traverses Tile IR. The
answer changes the sequencing.

---

## 1. There are two GEMM lanes, and only one of them runs

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

## 2. The stack, as executed

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

## 3. The expander population

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

## 4. Where the current W1 work sits — the consequence

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

## 5. What this implies for sequencing

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

## 6. Re-running this

```bash
# Lane pipelines
sed -n '/def _build_compiled_gemm_hsaco/,/return hsaco/p' python/tessera/runtime.py
sed -n '/def _build_canonical_gemm_hsaco/,/return hsaco/p' python/tessera/runtime.py

# Stack passes present in the runtime at all
for p in tessera-graph-to-schedule tessera-schedule-to-tile tessera-lower-to-rocm; do
  printf "%-28s %s\n" "$p" "$(grep -c "$p" python/tessera/runtime.py)"
done
```
