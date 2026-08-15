# Tile sync/memory §5.1 experiments — CAKE Phase 1 step 0

Measured evidence for
[`docs/audit/compiler/compiler_enhancement.md`](../../docs/audit/compiler/compiler_enhancement.md)
§5.1.1 (run 2026-08-15 against `tessera-opt` at `2d05e823`). These are the
hardest-case-first probes W1.1 §2 mandates, run **before** any ODS change.

Run (registered vocabulary — deliberately no `--allow-unregistered-dialect`):

```bash
./build/tools/tessera-opt/tessera-opt \
  --tessera-warpspec-legality --tessera-tile-pipeline-legality \
  --tessera-tile-barrier-reuse-legality <fixture>.mlir
```

| Fixture | Pre-§5.2 (measured 2026-08-15 AM) | Post-§5.2 | Post-§5.3 increment 1 (same day, `--tessera-tile-dataflow-legality`) |
|---|---|---|---|
| `exp1_control.mlir` | `WARPSPEC_MMA_NOT_TOKEN_SYNCED` fires | unchanged | unchanged (control) |
| `exp1_loop_carried.mlir` | **Silent** — block-arg edge invisible | index/no-arrive holes rejected; block-arg cases still silent | **ALL FIRE**: mma block-arg edge → `WARPSPEC_MMA_NOT_TOKEN_SYNCED` (×2 producers); wrong-slot pairing → `TILE_WAIT_SLOT_MISMATCH`; SSA-aliased `expect` → `TILE_TMA_EXPECT_MISMATCH` |
| `exp1_wait_holes.mlir` | **Silent** — bare wait; `i32` dependency | both rejected | pairing probe (c) now *derived* — same slot/barrier verifies green, which is correct |
| `exp2_pipeline_ring.mlir` | **Silent** — stale ring passes | still silent | stale ring → `TILE_PIPELINE_RING_STALE`; well-formed ring stays green |
| `exp2_ring_unclosed.mlir` | **Rejected** (`PipelineAdvanceOp::verify`) | unchanged | unchanged |

**The §5.1.1 fail-open finding is closed: no probe row remains silent.**
Drift gates: `tests/tessera-ir/phase2/tile_sync_typed_invalid.mlir` (§5.2) and
`tests/tessera-ir/phase2/tile_dataflow_legality.mlir` (§5.3, with the
well-formed loop pipeline as the no-diagnostic positive control).

**§5.3/§5.4 fully closed (second increment, same day):** predicates are typed,
the attribute escape hatches are deleted, both legality fixtures run on the
registered vocabulary, and `--tessera-tile-dataflow-legality` runs inside the
NVIDIA pipelines — whose own lowered output passes the derivation gate. See
`compiler_enhancement.md` §5.3.1–§5.3.2. Remaining for Phase 1 exit: the
gfx1151 numerical gate (§5.5 gate 5) and the barrier-at-birth restructure.
