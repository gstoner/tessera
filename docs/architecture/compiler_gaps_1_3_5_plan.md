---
status: Partially landed
classification: Plan
last_updated: 2026-07-14
---

# Compiler Gaps 1, 3, 5 — Execution-Ready Plan

> Status (2026-07-14): **Gaps 1 and 5 have landed**; Gap 3 remains partial.
> `StencilLoopMaterializePass` (Gap 1) and `HaloTransportLowerPass` (Gap 5)
> are implemented, registered in `tessera-opt`, and lit-covered. Gap 3
> (halo + mesh boundary integration) has a lit fixture but its named
> reconciliation helper was never created. Gaps 2 and 4
> (`BoundaryConditionLowerPass` and `attn_local_window_2d`) landed earlier;
> this document retains the original contracts for context.
>
> Each gap below has: precise contract, file list, lit fixture target,
> acceptance criteria, deferral rationale, and a sequencing recommendation.

---

## Sprint context

Gaps 1, 2, 3, 4, 5 were called out together as compiler-correctness items
(not library work). After scoping:

| # | Title                                  | Decision    | Why                                                                          |
|---|----------------------------------------|-------------|------------------------------------------------------------------------------|
| 1 | Stencil loop materialization           | **landed**  | `StencilLoopMaterializePass` emits the scf.for nest with per-axis BC fixups. |
| 2 | Boundary condition lowering            | **landed**  | Pure attribute pass; ships periodic / reflect / dirichlet(v) / neumann(v).   |
| 3 | Halo + mesh boundary integration       | **partial** | Lit fixture present; named `HaloMeshReconcile` helper not created.           |
| 4 | 2D local-window attention              | **landed**  | Pure library work; ~500 LOC including registry + VJP + JVP + 16 tests.       |
| 5 | Async halo transport kernels           | **landed**  | `HaloTransportLowerPass` ships pack / exchange / unpack lowering.            |

The work was **sequenced**: Gap 1 landed first because it defines the loop
nest those halo exchanges (Gaps 3 + 5) wrap and lower through; Gap 5's
`HaloTransportLowerPass` then landed on top of it.

---

## Gap 1 — Stencil Loop Materialization  ✅ LANDED

> **Landed.** Implemented as a single
> `src/compiler/tessera_neighbors/lib/Dialect/Neighbors/Transforms/StencilLoopMaterializePass.cpp`
> (`-tessera-stencil-loop-materialize`, `buildLoopNest` emits an `scf.for`
> tower with per-axis BC fixups, idempotent sentinel `stencil.materialized`),
> registered via `registerStencilLoopMaterializePass()` in
> `tools/tessera-opt/tessera-opt.cpp`. Lit fixtures:
> `tests/tessera-ir/phase7/neighbors_stencil_materialize.mlir` and
> `neighbors_stencil_materialize_rank3.mlir`. (The plan below anticipated a
> separate x86/TMA pass split; the landed pass unifies them.)

### Contract

`StencilLowerPass` today annotates `stencil.apply` with structured
attributes (`stencil.pack_phase`, `stencil.compute_phase`,
`stencil.tap_count`, `stencil.halo_width`, plus the new
`stencil.bc.{modes,values,has_value}` from Gap 2) but **does not emit a
loop nest**. The downstream target lowering needs:

- A concrete `scf.for` (or `affine.for`) loop tower indexed over each
  spatial axis declared by `halo.width`.
- A per-tap load + accumulate body that respects the dialect's
  `DeltaArrayAttr` taps.
- A target hand-off — x86 stays in `scf` until `TileToX86Pass`, NVIDIA
  must annotate the inner loop with `tile.async_copy` / TMA descriptors.

### Files

- New pass:
  `src/compiler/tessera_neighbors/lib/Dialect/Neighbors/Transforms/StencilLoopMaterializePass.cpp`
- Registration:
  - `src/compiler/tessera_neighbors/include/.../Transforms/Passes.h`
  - `src/compiler/tessera_neighbors/CMakeLists.txt`
  - `tools/tessera-opt/tessera-opt.cpp`
- Two target hand-off pieces (separable PRs) — *anticipated split, never
  created; the landed `StencilLoopMaterializePass` unified them (see banner above)*:
  - `StencilToX86Pass.cpp` — scalar `scf.for` body
  - `StencilToTilePass.cpp` — Tile-IR + TMA path
- Lit fixtures — *these split-specific fixtures were never created; the actual
  fixtures are `neighbors_stencil_materialize{,_rank3}.mlir` (cited above)*:
  - `neighbors_stencil_materialize_x86.mlir`
  - `neighbors_stencil_materialize_tma.mlir`
- Python guards:
  - `tests/unit/test_neighbors_stencil_materialize.py`

### Acceptance criteria

1. `tessera-opt -tessera-stencil-lower -tessera-boundary-condition-lower
   -tessera-stencil-loop-materialize` produces visible `scf.for`-nested
   loops with the correct rank.
2. Loop body contains exactly one load per declared tap and one running
   accumulator.
3. Boundary fixups inside the loop honor `stencil.bc.modes[i]`:
   - `periodic` → modular index expression (`(i + N) % N`)
   - `reflect`  → `min(N-1, abs(i))` mirror clamp
   - `dirichlet`→ select against `stencil.bc.values[i]`
   - `neumann`  → `field[clamp(i,0,N-1)] + value` shift
4. The TMA path emits a hoisted `tessera.tma.descriptor` per stencil
   field (one per loop nest, NOT per tile) — Architecture Decision #9.
5. Idempotent (sentinel `stencil.materialized`).
6. Lit fixtures + Python guards pass green; both fixtures FileCheck the
   loop structure and the per-BC index expression.

### Deferral rationale

The loop materialization splits cleanly only after we choose the target's
**index-space type** (scalar `index` on x86, TMA descriptor base + offset
on NVIDIA). Doing both targets in one pass means duplicating the body
twice and re-merging — a known anti-pattern. Better as two passes after
Gap 2 settles the BC ABI.

### Estimated effort

7–10 working days for both paths + their fixtures. Splittable as two PRs.

---

## Gap 3 — Halo + Mesh Boundary Integration  🟡 PARTIAL

> **Partial.** A lit fixture
> `tests/tessera-ir/phase7/neighbors_halo_mesh_integration.mlir` (and
> `halo_mesh_integration_attn_local_window_2d.mlir`) exists, so the halo/mesh
> integration surface is at least partially present, but the named `HaloMeshReconcile.cpp` helper below was never created.

### Contract

`DistributionLoweringPass` (Phase 2,
`src/transforms/lib/DistributionLoweringPass.cpp`) converts
`tessera.shard` arg attributes into `schedule.mesh.define` +
`schedule.mesh.region`. It does **not** look at `tessera.neighbors.halo.region`
ops to know whether a sharded tensor will be consumed by a stencil — so
halo exchanges currently happen only when the user manually wraps a field
in `halo.region`, and they never coordinate with the mesh boundary.

The contract this gap delivers:

1. After `DistributionLoweringPass`, every sharded tensor consumed by a
   `tessera.neighbors.stencil.apply` gets a `halo.exchange` op inserted
   on the **rank-boundary side** of the partition (one exchange per
   neighbor in the mesh's adjacency graph).
2. The exchange's `width` matches the `stencil.halo_width` already set
   by `HaloInferPass`.
3. The mesh's BC interacts with the stencil's BC: if the mesh is
   non-periodic on axis `a` but the stencil declares `bc.modes[a] ==
   "periodic"`, emit a named diagnostic (Architecture Decision #21):

       error: stencil declares periodic boundary on axis 'a' but mesh
              partition is non-periodic; cannot reconcile without an
              explicit ghost-cell strategy.

### Files

- Modify:
  `src/transforms/lib/DistributionLoweringPass.cpp` — add a post-pass
  walk that finds `stencil.apply` consumers and inserts `halo.exchange`
  on every sharded operand.
- New helper — *never created (see banner above); the BC reconciliation table
  was not split out into its own file*:
  `HaloMeshReconcile.cpp` — owns the BC reconciliation table.
- Lit fixture:
  `tests/tessera-ir/phase7/neighbors_halo_mesh_integration.mlir`
- Python guard:
  `tests/unit/test_neighbors_halo_mesh_integration.py`
- New OP_SPECS entry (if `halo.exchange` doesn't already have one for
  the dialect-bound op variant — check before adding).

### Acceptance criteria

1. A function annotated with `tessera.shard(mesh="dp", axis=0)` that
   feeds a stencil produces, after distribution lowering, exactly N+1
   halo.exchange ops (one per inter-rank boundary on axis 0).
2. BC conflicts emit `TesseraConstraintError` (or its MLIR equivalent
   diagnostic) at decoration / pass time, not silently at runtime.
3. The pass leaves a `halo.exchange { width = [w0, w1, …] }` op whose
   width attribute equals the upstream `halo.width` for every axis.
4. Per-axis BC fan-out is correct: a stencil declaring
   `bc = "periodic,dirichlet(0)"` on a 2D mesh emits *2 exchanges on
   axis 0* (periodic = wrap-around) and *0 exchanges on axis 1*
   (dirichlet = local fill, no neighbor traffic).
5. Lit fixture + Python guard pass green.

### Deferral rationale

Touches two dialects (`tessera` + `tessera.neighbors`) and a non-trivial
Phase 2 pass. The right entry point is during Gap 1's loop materialize
work — that's when we'll know exactly what `halo.exchange` needs to
hand the loop. Doing Gap 3 first means designing a contract we'll need
to re-design when Gap 1 lands.

### Estimated effort

5–7 working days, mostly in `DistributionLoweringPass.cpp` and the
reconciliation helper. The lit fixture is the bulk of the test work.

---

## Gap 5 — Async Halo Transport Kernels  ✅ LANDED

> **Landed.** Implemented as
> `src/compiler/tessera_neighbors/lib/Dialect/Neighbors/Transforms/HaloTransportLowerPass.cpp`,
> registered via `registerHaloTransportLowerPass()` in
> `tools/tessera-opt/tessera-opt.cpp`, with lit fixture
> `tests/tessera-ir/phase7/neighbors_halo_transport_lower.mlir`.

### Contract

`PipelineOverlapPass` today writes `comm.stream_id`, `compute.stream_id`,
`comm.async`, and `pipeline.buffer_idx` as attributes on already-existing
`stencil.apply` ops, but emits **no pack / exchange / unpack kernels**.
The actual async transport machinery — kernels that pack ghost cells
into contiguous buffers, post non-blocking sends/recvs, and unpack on
arrival — is missing.

This gap ships three composable kernels:

1. `tessera.neighbors.halo.pack(field, side, axis, width)` — emit a
   contiguous buffer holding the ghost slab.
2. `tessera.neighbors.halo.exchange(buf, peer, stream)` — async send/
   receive on the assigned stream (NCCL `ncclSend`/`ncclRecv` on
   NVIDIA, RCCL on AMD, MPI on x86, mock-collective in tests).
3. `tessera.neighbors.halo.unpack(field, buf, side, axis, width)` —
   write the received slab into the field's ghost region.

Plus the **scheduling**: per Architecture Decision #17, pack runs on
the compute stream and exchange + unpack run on the comm stream, with
the consumer (`stencil.apply` body) waiting on a stream-side barrier
before it reads neighbor cells.

### Files

- New pass:
  `src/compiler/tessera_neighbors/lib/Dialect/Neighbors/Transforms/HaloTransportLowerPass.cpp`
- New ops in ODS:
  `src/compiler/tessera_neighbors/include/tessera/Dialect/Neighbors/IR/tessera_neighbors.td`
  — add `HaloPackOp`, `HaloUnpackOp` (HaloExchangeOp already exists).
- Adapter dispatch wired through `src/collectives/include/.../Adapters.h`.
- Lit fixture:
  `tests/tessera-ir/phase7/neighbors_halo_transport_lower.mlir`
- Python guards (mock-collective-based, no NCCL needed in CI):
  `tests/unit/test_neighbors_halo_transport.py` — uses
  `testing/mock_collective.py` per Architecture Decision #6.

### Acceptance criteria

1. Every `halo.exchange` op present after Gap 3 lowering is replaced
   with a triple `(pack, exchange, unpack)` whose stream assignment
   matches the `comm.stream_id` set by `PipelineOverlapPass`.
2. Buffer reuse honours `pipeline.buffer_idx` — the same slot index
   means the same physical buffer (double-buffered pipeline).
3. Mock-collective tests verify per-rank receive buffer equals per-rank
   send buffer at the right offset, end-to-end, with 4 ranks on a
   1D mesh and 2×2 ranks on a 2D mesh.
4. NVIDIA/AMD paths produce real `ncclSend`/`ncclRecv` / `rcclSend`/
   `rcclRecv` calls in the IR (verifiable via FileCheck — no need to
   actually run on hardware in CI).
5. Architecture Decision #17 compliance: 1F1B-style overlap where
   the next tile's pack starts as soon as the current tile's compute
   begins, on a separate stream.

### Deferral rationale

Has nothing to lower until Gaps 1 + 3 land:

- Without Gap 1's loop materialization, there's no inner loop to
  pipeline against.
- Without Gap 3's `halo.exchange` insertion, the pack/unpack triple
  has no host op to replace.

Doing Gap 5 first means designing a transport ABI for IR that doesn't
exist yet. Pure waste.

### Estimated effort

5–8 working days once Gaps 1 + 3 are in. The bulk is the mock-collective
test harness and proving stream-overlap timing matches the docstring
contract; the ops + dispatch are mechanical.

---

## Sequencing recommendation

```
Gap 1  ──► Gap 3  ──► Gap 5
  │         │           ▲
  │         │           │
  └─────────┴─ Gap 2 ───┘   (Gap 2 already shipped — its BC ABI is what
                             Gap 1's loop fixups consume and what Gap 3's
                             mesh reconciliation cross-references.)
```

Gap 4 (2D local-window attention) was independent of all five and
shipped alongside Gap 2.

## How to revive this plan

1. Re-read `tests/tessera-ir/phase7/neighbors_*.mlir` to refresh on what
   the dialect already supports.
2. Re-read `src/compiler/tessera_neighbors/lib/Dialect/Neighbors/Transforms/StencilLowerPass.cpp`
   and `BoundaryConditionLowerPass.cpp` (sentinel pattern + structured
   attribute idiom).
3. Re-read this document — *all* deferred contracts above are still
   the right contracts; the only thing that needs updating before
   starting is the file-line numbers if the surrounding code has
   moved.
4. Start with Gap 1 (`StencilLoopMaterializePass`). Do x86 path first
   to get the contract right at small scale, then layer the TMA path.

## Related docs

- `src/compiler/tessera_neighbors/lib/Dialect/Neighbors/IR/` — current
  dialect implementation.
- `tests/tessera-ir/phase7/` — every existing lit fixture for the
  Neighbors dialect.
- `docs/spec/COMPILER_REFERENCE.md` — broader compiler design context.
- `docs/audit/roadmap/ROADMAP_AUDIT.md` — Phase A–I roadmap; Gaps 1/3/5
  sit in Phase G–H territory once they land.
