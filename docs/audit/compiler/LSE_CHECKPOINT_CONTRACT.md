---
last_updated: 2026-07-26
audit_role: reference
---

# The LSE checkpoint contract — FlashAttention-2 semantics vs. what Tessera implements

This document records what `tessera_attn.lse.save` / `tessera_attn.lse.load`
are *supposed* to mean, what they actually do in Tessera today, and the
measurement question each hardware backend owns. It exists because the gap
between those first two blocked an Apple lowering
([APPLE-ATTN-STREAM-1](../backend/apple/todo.md)) and because closing it
properly is an HBM-bandwidth question that only NVIDIA and ROCm can answer.

Status truth stays in the generated dashboards and the per-platform audits;
this page owns the *contract* and the open evaluation.

## The FlashAttention-2 design

FlashAttention computes softmax in a single streaming pass over KV blocks
rather than materializing the `[Sq, Sk]` score matrix. Each block updates a
running maximum `m` and a running normalization sum `l`; the per-row
log-sum-exp

```
LSE[q] = log Σ_k exp(scale · Q[q]·K[k]ᵀ)     (numerically: m[q] + log l[q])
```

is the coefficient that rescales partial outputs as blocks are combined. It is
the one piece of softmax state that survives the whole reduction.

The backward pass needs `P = exp(S − LSE)` to form `dS = P ⊙ (dP − D)`. There
are two ways to get `LSE` there, and the choice is the entire design question:

| Strategy | Forward cost | Backward cost | State |
|---|---|---|---|
| **Save** (FA-2 standard) | one `[B·H·Sq]` fp32 store to HBM | one load | a persistent workspace tensor |
| **Recompute** | none | an extra pass over K to rebuild `L` | none |

FA-2 saves. It writes `L` to HBM in the forward and reads it in the backward,
recomputing only `S` and `P` (which are `O(Sq·Sk)` and never worth storing).
The saved vector is small — linear in sequence length, not quadratic — so the
trade is a little bandwidth and a workspace allocation against a full extra
reduction over K per backward launch.

In Triton and other MLIR-based kernel languages this appears as ordinary
pointer traffic: the forward ends in a `tl.store` to an LSE pointer, the
backward opens with a `tl.load` from it. Production kernels keep the running
`m`/`l` in registers or SRAM and commit only the finalized value, so the
`save`/`load` pair is a deliberate, minimized HBM round-trip rather than a
per-block one.

## What Tessera declares

`src/compiler/tile_opt_fa4/include/tessera/Dialect/Attn/Attn.td`:

```tablegen
def LseSaveOp : Op<Tessera_Attn_Dialect, "lse.save", []> {
  let arguments = (ins AnyType:$scores);
  let results   = (outs AnyType:$lse);
}

def LseLoadOp : Op<Tessera_Attn_Dialect, "lse.load", [Pure]> {
  let results   = (outs AnyType:$lse);   // no `let arguments` at all
}
```

These do not implement the contract above. Specifically:

- **No destination.** `lse.save` takes no pointer, memref, or workspace
  operand, so there is nothing for it to store *to*.
- **No source.** `lse.load` declares no `let arguments` block at all — not
  merely an empty one — so it materializes a value from nothing. Nothing links
  a load to the save whose value it should read.
- **No identity.** With neither an SSA edge, a symbol, nor a handle, a backward
  lowering cannot express *which* save its load corresponds to. This is the
  same name-free global-state modeling that `#tile.buffer_ref` → `!tile.buffer`
  and annotation-only `#tile.pipeline_state` → threaded SSA already replaced
  elsewhere in Tile IR.
- **Degenerate shape and a discarded result.** The single emission site
  (`TileIRLoweringPass.cpp`, after `tessera_attn.lse_accumulate`) types the
  result scalar `f32` rather than the per-row `[tile_q]` vector, and drops it.

The only `lse.load` in the tree is a v1.3 example fixture
(`src/compiler/tile_opt_fa4/examples/attention_backward_lse_v1_3.mlir`). No
pass, runtime path, or backward lowering consumes one.

So the pair is a **declared-but-unimplemented contract**: named for the FA-2
pattern, scaffolded in the v1.2/v1.3 design, never wired to memory.

## What every backend does today: recompute

No backend reads a saved LSE. All three with an attention backward
independently chose recompute:

| Backend | Attention backward | LSE source |
|---|---|---|
| **ROCm** gfx1151 | `GenerateWMMAFlashAttnBwdKernel.cpp` | a `_pre` kernel recomputes `L[q] = logsumexp_k(scale·QKᵀ)` online and writes L/D scratch the matmul kernels read; its header states the backward "needs nothing saved from the forward" |
| **NVIDIA** sm_120 | `sm120_attention_backward_kernel.mlir` | `workspace_bytes = 0`, `workspace_owner = "output_element"` |
| **Apple** Apple7 | `flash_attn_bwd_*` (`apple_gpu_runtime.mm`) | `bwd_query_stats` recomputes `m`/`l` per query; the ABI takes no LSE buffer |
| **x86** AVX-512 | none — forward only | n/a |

The convergence is not accidental. Each of those lanes sells a zero-workspace
determinism property — ROCm's "no stored attention matrix", NVIDIA's
`workspace_owner = "output_element"`, Apple's bit-identical repeated launches —
and a saved LSE reintroduces exactly the workspace those contracts exist to
eliminate.

That is a defensible point on the curve. It is not obviously the right one at
long context, which is what the open evaluation below is for.

## `Pure` was tried and rejected

Marking `LseSaveOp` `Pure` is the obvious way to stop a dead value-producer
behaving like a side effect, and it was tried first. It is **not** in the tree,
for two reasons found by testing it:

1. **It changes emitted IR on every backend.** Removing a dead op is a change,
   and `tests/tessera-ir/phase3/flash_attn_full.mlir` asserts the op's presence
   in the shared lowering output. A claim that the trait "changes nothing
   emitted" was wrong.
2. **It is a trap for whoever implements the real contract.** A store must be
   non-`Pure` with `MemWrite`. A future change adding a destination while the
   trait remains would let DCE silently delete the store, and the backward
   would read uninitialized LSE — wrong gradients, no diagnostic.

The Apple consumer does not need it. `tessera-apple-streaming-attention` erases
only a `lse.save` whose own result is unused, as part of re-forming the
recurrence it is already rewriting, and both the Apple fixture and
`flash_attn_full.mlir` pass with the trait absent. That keeps the shared
declaration and every sibling lowering untouched, which is the right trade for
one backend's lowering.

`tests/unit/test_lse_checkpoint_contract.py` stays as a dormant tripwire: if the
op ever acquires a destination-shaped operand, pointer/memref type, or
memory-effect interface *while* declaring `Pure`, the build fails. Today it
passes trivially because the trait is absent.

## Open evaluation — owned by NVIDIA and ROCm

The save-vs-recompute choice is a bandwidth question, and the two lead
performance targets (Decision #28) are the ones with the HBM hierarchy and the
long-context workloads that make it decidable. Apple and x86 inherit whatever
they conclude; neither has the memory system to settle it.

The question is **not** "should we implement `lse.save`" but:

> At what `(sequence length, head count, dtype, architecture)` does storing and
> reloading a `[B·H·Sq]` fp32 LSE vector beat recomputing `L` with an extra
> pass over K in the backward — and does the win survive the workspace and
> determinism cost the current contracts price in?

Evaluation rows:

- NVIDIA: `NVIDIA-LSE-1` in [`backend/nvidia/todo.md`](../backend/nvidia/todo.md)
- ROCm: `ROCM-LSE-1` in [`backend/rocm/todo.md`](../backend/rocm/todo.md)

Both are measurement items, not implementation items. Promotion requires the
same evidence any other route change does: paired two-run medians in a named
timing domain, retained resource evidence, and an explicit statement of the
workspace and determinism the saved path gives up.

### Fix at the source — available now, independent of the measurement

The measurement decides whether to *implement* or *retire*. Neither answer is
needed to remove the defect, and this is the preferred landing:

> Stop `TileIRLoweringPass` emitting a `lse.save` that has no destination, and
> revert `LseSaveOp` to non-`Pure` so the op stays honestly side-effecting and
> ready for a real implementation.

This is strictly better than the `Pure` correction currently in the tree:

- **It removes the cause rather than tolerating the symptom.** The problem is
  not that a side-effecting op is unremovable; it is that the lowering emits a
  store to nowhere on every forward, on every target.
- **It removes the Apple-side erasure too.** With emission fixed there is no
  dead save for any backend to step around, so the one place a backend touches
  a shared effectful op goes away.
- **It benefits every backend, not just Apple.** NVIDIA and ROCm forwards stop
  carrying a dead op too.
- **It leaves the measurement genuinely open.** Retiring the vocabulary or
  implementing it properly both remain available afterwards.

Sequencing: this is a shared-contract change to `TileIRLoweringPass` and
`Attn.td`, so it wants NVIDIA/ROCm sign-off — which is why it is filed in both
backend queues rather than landed from the Apple queue. Once it lands, the
Apple consumer keeps working unchanged: it already erases only a `lse.save`
whose own result is unused, and with emission fixed there will be none to
erase. The `Pure` marking and its guard test should be reverted in the same
change.

### If the answer is "save"

The redesign is a contract change, not a patch, and should follow the pattern
Tile IR already used twice:

1. Give the pair **real identity** — an SSA handle or symbol linking a load to
   its save, so `lse.load` can name which forward it reads (compare
   `#tile.buffer_ref` → `!tile.buffer`).
2. Give `lse.save` a **destination** and the memory effects that go with it,
   dropping `Pure`.
3. Type the value as the per-row `[tile_q]` vector it is, not scalar `f32`.
4. Make emission **conditional** on the LSE actually being consumed, so
   inference-only programs stop carrying a backward checkpoint.

Step 4 is what Apple needs regardless of the outcome: a forward that does not
feed a backward should not emit a checkpoint at all.

### If the answer is "recompute"

Then the vocabulary is genuinely vestigial and should be retired rather than
left as scaffolding that reads like a contract. Retiring it is also a
shared-contract change and needs the same sign-off.

## Source material

- FlashAttention-2 online-softmax formulation and the saved-`L` backward.
- `src/compiler/tile_opt_fa4/docs/AppNote_FA4_in_Tessera_v1_2.md` and the v1.3
  addendum — where the save/load pair was scaffolded.
- `src/transforms/lib/TileIRLoweringPass.cpp` — the single emission site.
- [`backend/apple/todo.md`](../backend/apple/todo.md) APPLE-ATTN-STREAM-1 — the
  lowering the gap blocked, and the trait correction that unblocked it.
