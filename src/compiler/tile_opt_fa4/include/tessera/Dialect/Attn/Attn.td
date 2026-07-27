//===- Attn.td (v2.0) ------------------------------------------------------*- tablegen -*-===//
// Phase 3 additions: ScaledDotProduct, OnlineSoftmax, LseAccumulate,
//                    DropoutMask, CausalMask — plus original LseSave/LseLoad.
include "mlir/IR/OpBase.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

def Tessera_Attn_Dialect : Dialect {
  let name = "tessera_attn";
  let cppNamespace = "::tessera::attn";
}

// ─── Original ops (v1.3) ─────────────────────────────────────────────────────

// `Pure` matches this op's actual signature (2026-07-26). Despite the name and
// the "save" framing, nothing here writes: the op takes one operand, returns
// one value, and owns no memref, symbol, or side-effect interface. `LseLoadOp`
// likewise takes no operands, so no SSA edge, symbol, or handle links a load to
// a save — a backward lowering could not express *which* save its load reads.
//
// The missing trait was the only thing making a dead value-producer
// unremovable, which blocked backends from re-forming the streaming-attention
// recurrence (APPLE-ATTN-STREAM-1). No backend consumes the saved LSE: ROCm's
// `_pre` kernel recomputes `L[q] = logsumexp_k(scale*QK^T)` and states the
// backward "needs nothing saved from the forward"; NVIDIA sm_120 declares
// `workspace_bytes = 0, workspace_owner = "output_element"`; Apple's
// `bwd_query_stats` recomputes m/l per query. All three chose recompute for the
// same reason — a saved LSE reintroduces the workspace their zero-workspace
// determinism contracts exist to eliminate.
//
// This does NOT retire the save/load vocabulary. If long-context measurement
// ever shows recomputing logsumexp costs more than reloading a [tile_q] vector,
// the pair should be redesigned with real identity (an SSA handle or symbol),
// exactly as `#tile.buffer_ref` became `!tile.buffer`. Marking it Pure only
// stops a no-op from behaving like a side effect until then.
def LseSaveOp : Op<Tessera_Attn_Dialect, "lse.save", [Pure]> {
  let summary = "Save per-row log-sum-exp (LSE) from forward attention";
  let arguments = (ins AnyType:$scores);
  let results = (outs AnyType:$lse);
  let assemblyFormat = "$scores attr-dict `:` type($scores) `->` type($lse)";
  let hasVerifier = 1;
}

def LseLoadOp : Op<Tessera_Attn_Dialect, "lse.load", [Pure]> {
  let summary = "Load LSE tensor for backward";
  let results = (outs AnyType:$lse);
  let assemblyFormat = "attr-dict `:` type($lse)";
  let hasVerifier = 1;
}

// ─── Phase 3 FA-4 ops ────────────────────────────────────────────────────────

// ScaledDotProductOp — computes Q·Kᵀ / sqrt(d_k) for one (Q-tile, K-tile) pair.
// Tile sizes are stored as `tessera.tile_q` / `tessera.tile_kv` attributes so
// Phase 5 autotuner can sweep them without recompiling.
def ScaledDotProductOp : Op<Tessera_Attn_Dialect, "scaled_dot_product", [Pure]> {
  let summary = "Scaled dot-product QK^T/sqrt(d_k) for one tile pair";
  let arguments = (ins
    AnyType:$query,   // tile of Q: [tile_q × d_k]
    AnyType:$key,     // tile of K: [tile_kv × d_k]
    F32Attr:$scale    // 1/sqrt(d_k) — constant folded at lowering time
  );
  let results = (outs AnyType:$scores); // [tile_q × tile_kv]
  let assemblyFormat = "$query `,` $key `scale` `=` $scale attr-dict `:` type($query) `,` type($key) `->` type($scores)";
  let hasVerifier = 1;
}

// OnlineSoftmaxOp — FA-2 online softmax on a score tile.
// Maintains running max (m) and running sum (l) for the correction step.
// Output: updated output accumulator, new running_m, new running_l.
def OnlineSoftmaxOp : Op<Tessera_Attn_Dialect, "online_softmax", []> {
  let summary = "FA-2 online softmax — updates running max/sum and output accumulator";
  let arguments = (ins
    AnyType:$scores,    // current score tile [tile_q × tile_kv]
    AnyType:$running_m, // per-row running max [tile_q]
    AnyType:$running_l, // per-row running sum [tile_q]
    AnyType:$acc_out    // accumulated output [tile_q × d_v]
  );
  let results = (outs
    AnyType:$new_acc,   // updated output accumulator
    AnyType:$new_m,     // updated running max
    AnyType:$new_l      // updated running sum
  );
  let assemblyFormat = "$scores `,` $running_m `,` $running_l `,` $acc_out attr-dict `:` type($scores) `,` type($running_m) `,` type($running_l) `,` type($acc_out) `->` type($new_acc) `,` type($new_m) `,` type($new_l)";
  let hasVerifier = 1;
}

// Canonical streaming update. Unlike the original compatibility op above, the
// value tile is explicit: this operation represents the complete online
// softmax/PV recurrence for one KV block, not merely the score statistics.
def StreamingUpdateOp : Op<Tessera_Attn_Dialect, "streaming_update", []> {
  let summary = "Online-softmax and PV update for one KV block";
  let arguments = (ins
    AnyType:$scores,    // current score tile [tile_q × tile_kv]
    AnyType:$value,     // current value tile [tile_kv × d_v]
    AnyType:$running_m, // per-row running max [tile_q]
    AnyType:$running_l, // per-row running sum [tile_q]
    AnyType:$acc_out    // accumulated output [tile_q × d_v]
  );
  let results = (outs
    AnyType:$new_acc,
    AnyType:$new_m,
    AnyType:$new_l
  );
  let assemblyFormat = "$scores `,` $value `,` $running_m `,` $running_l `,` $acc_out attr-dict `:` type($scores) `,` type($value) `,` type($running_m) `,` type($running_l) `,` type($acc_out) `->` type($new_acc) `,` type($new_m) `,` type($new_l)";
  let hasVerifier = 1;
}

// LseAccumulateOp — finalises the FA-2 two-pass algorithm.
// Divides the accumulated output by the final running sum and computes
// log(l) + m (the true LSE value saved for the backward pass).
def LseAccumulateOp : Op<Tessera_Attn_Dialect, "lse_accumulate", [Pure]> {
  let summary = "FA-2 LSE finalisation: out = acc / l,  lse = log(l) + m";
  let arguments = (ins
    AnyType:$acc,       // accumulated output [tile_q × d_v]
    AnyType:$running_m, // final running max [tile_q]
    AnyType:$running_l  // final running sum [tile_q]
  );
  let results = (outs
    AnyType:$output,    // normalised output [tile_q × d_v]
    AnyType:$lse        // log-sum-exp values [tile_q] — saved for backward
  );
  let assemblyFormat = "$acc `,` $running_m `,` $running_l attr-dict `:` type($acc) `,` type($running_m) `,` type($running_l) `->` type($output) `,` type($lse)";
  let hasVerifier = 1;
}

// DropoutMaskOp — stochastic binary mask for attention score dropout.
// Only generated when @jit has a seed and dropout_p > 0; otherwise elided.
// Gates behind target_profile.isa so WGMMA kernels skip it on SM_90 (handled
// in fused epilogue instead).
def DropoutMaskOp : Op<Tessera_Attn_Dialect, "dropout_mask", []> {
  let summary = "Generate dropout mask for attention scores";
  let arguments = (ins
    AnyType:$scores,    // score tile to mask [tile_q × tile_kv]
    F32Attr:$dropout_p, // drop probability in [0, 1)
    I64Attr:$seed       // RNG seed (required when present)
  );
  let results = (outs AnyType:$masked_scores);
  let assemblyFormat = "$scores `p` `=` $dropout_p `seed` `=` $seed attr-dict `:` type($scores) `->` type($masked_scores)";
  let hasVerifier = 1;
}

// CausalMaskOp — applies lower-triangular causal mask to score tile.
// Emitted when @tessera.ops.flash_attn(..., causal=True).
// The mask is structurally zero cost: it becomes a scalar comparison in the
// tiling loop and is folded into the WGMMA epilogue on SM_90.
def CausalMaskOp : Op<Tessera_Attn_Dialect, "causal_mask", [Pure]> {
  let summary = "Apply lower-triangular causal mask to score tile";
  let arguments = (ins
    AnyType:$scores,     // score tile [tile_q × tile_kv]
    I64Attr:$q_offset,   // row offset of this Q tile in the sequence
    I64Attr:$kv_offset   // col offset of this KV tile in the sequence
  );
  let results = (outs AnyType:$masked_scores);
  let assemblyFormat = "$scores `q_off` `=` $q_offset `kv_off` `=` $kv_offset attr-dict `:` type($scores) `->` type($masked_scores)";
  let hasVerifier = 1;
}

// Canonical block-boundary mask. Offsets are SSA values so a real KV scf.for
// can carry and advance them. The window bounds remain compile-time scheduling
// policy; -1 means unbounded.
def BoundaryMaskOp : Op<Tessera_Attn_Dialect, "boundary_mask", [Pure]> {
  let summary = "Apply causal/sliding-window bounds for one KV block";
  let arguments = (ins
    AnyType:$scores,
    Index:$q_offset,
    Index:$kv_offset,
    BoolAttr:$causal,
    I64Attr:$window_left,
    I64Attr:$window_right,
    I64Attr:$logical_sk
  );
  let results = (outs AnyType:$masked_scores);
  let assemblyFormat = "$scores `q_off` `=` $q_offset `kv_off` `=` $kv_offset attr-dict `:` type($scores) `->` type($masked_scores)";
  let hasVerifier = 1;
}

// Counter-based dropout for one score block. The absolute KV offset is an SSA
// operand, preventing every loop iteration from replaying the first mask tile.
def BlockDropoutOp : Op<Tessera_Attn_Dialect, "block_dropout", []> {
  let summary = "Apply deterministic counter-based dropout to one score block";
  let arguments = (ins
    AnyType:$scores,
    Index:$kv_offset,
    F32Attr:$dropout_p,
    I64Attr:$seed
  );
  let results = (outs AnyType:$masked_scores);
  let assemblyFormat = "$scores `kv_off` `=` $kv_offset `p` `=` $dropout_p `seed` `=` $seed attr-dict `:` type($scores) `->` type($masked_scores)";
  let hasVerifier = 1;
}
