//===- Attn.td (v2.0) ------------------------------------------------------*- tablegen -*-===//
// Phase 3 additions: ScaledDotProduct, OnlineSoftmax, LseAccumulate,
//                    DropoutMask, CausalMask — plus original LseSave/LseLoad.
include "mlir/IR/OpBase.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

def Tessera_Attn_Dialect : Dialect {
  let name = "tessera_attn";
  let cppNamespace = "::tessera::attn";
}

// ─── Persistent training checkpoints ─────────────────────────────────────────

// LSE is an explicit memory carrier, not a value-returning marker. The memref
// owns memory-space semantics; identity and scope bind a forward save to the
// corresponding backward load across compiler-owned entry points.
def LseSaveOp : Op<Tessera_Attn_Dialect, "lse.save",
                   [MemoryEffects<[MemWrite]>]> {
  let summary = "Store finalized per-row LSE into a persistent checkpoint";
  let arguments = (ins
    AnyType:$lse,
    AnyMemRef:$destination,
    Index:$row_offset,
    StrAttr:$identity,
    StrAttr:$memory_space,
    StrAttr:$scope,
    OptionalAttr<StrAttr>:$cache_policy
  );
  let assemblyFormat = "$lse `into` $destination `at` $row_offset attr-dict `:` type($lse) `,` type($destination)";
  let hasVerifier = 1;
}

def LseLoadOp : Op<Tessera_Attn_Dialect, "lse.load",
                   [MemoryEffects<[MemRead]>]> {
  let summary = "Load per-row LSE from a persistent checkpoint";
  let arguments = (ins
    AnyMemRef:$source,
    Index:$row_offset,
    StrAttr:$identity,
    StrAttr:$memory_space,
    StrAttr:$scope,
    OptionalAttr<StrAttr>:$cache_policy
  );
  let results = (outs AnyType:$lse);
  let assemblyFormat = "$source `at` $row_offset attr-dict `:` type($source) `->` type($lse)";
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

def ScoreBiasOp : Op<Tessera_Attn_Dialect, "score_bias", [Pure]> {
  let summary = "Add a broadcast-resolved bias block to attention scores";
  let arguments = (ins AnyType:$scores, AnyType:$bias);
  let results = (outs AnyType:$biased_scores);
  let assemblyFormat = "$scores `,` $bias attr-dict `:` type($scores) `,` type($bias) `->` type($biased_scores)";
  let hasVerifier = 1;
}

def SoftcapOp : Op<Tessera_Attn_Dialect, "softcap", [Pure]> {
  let summary = "Apply cap*tanh(score/cap) after scale and additive bias";
  let arguments = (ins AnyType:$scores, F32Attr:$cap);
  let results = (outs AnyType:$capped_scores);
  let assemblyFormat = "$scores `cap` `=` $cap attr-dict `:` type($scores) `->` type($capped_scores)";
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

// Target-neutral tensor contract for deterministic attention backward.  Bias is
// always an operand; callers without bias pass a zero tensor.  The lowering
// materializes launch-owned split partials and fixed-order reductions as
// tensor-valued scf.for iter_args before any backend chooses a physical
// schedule.
def BackwardOp : Op<Tessera_Attn_Dialect, "backward", []> {
  let summary = "Deterministic split/reduced attention VJP tensor contract";
  let arguments = (ins
    AnyType:$d_output,
    AnyType:$query,
    AnyType:$key,
    AnyType:$value,
    AnyType:$bias,
    F32Attr:$scale,
    BoolAttr:$causal,
    I64Attr:$window_left,
    I64Attr:$window_right,
    F32Attr:$softcap,
    F32Attr:$dropout_p,
    I64Attr:$dropout_seed,
    I64Attr:$split_count,
    I64Attr:$query_block,
    I64Attr:$key_block
  );
  let results = (outs AnyType:$d_query, AnyType:$d_key, AnyType:$d_value);
  let assemblyFormat = "$d_output `,` $query `,` $key `,` $value `,` $bias attr-dict `:` type($d_output) `,` type($query) `,` type($key) `,` type($value) `,` type($bias) `->` type($d_query) `,` type($d_key) `,` type($d_value)";
  let hasVerifier = 1;
}

def BackwardDQBlockOp : Op<Tessera_Attn_Dialect, "backward_dq_block", [Pure]> {
  let summary = "Update one deterministic dQ query/KV block";
  let arguments = (ins Variadic<AnyType>:$inputs);
  let results = (outs AnyType:$d_query);
  let assemblyFormat = "$inputs attr-dict `:` type($inputs) `->` type($d_query)";
  let hasVerifier = 1;
}

def BackwardDKDVBlockOp : Op<Tessera_Attn_Dialect, "backward_dkdv_block", [Pure]> {
  let summary = "Accumulate one split-owned dK/dV query/KV block";
  let arguments = (ins Variadic<AnyType>:$inputs);
  let results = (outs AnyType:$partial_d_key, AnyType:$partial_d_value);
  let assemblyFormat = "$inputs attr-dict `:` type($inputs) `->` type($partial_d_key) `,` type($partial_d_value)";
  let hasVerifier = 1;
}

def BackwardReduceSplitOp : Op<Tessera_Attn_Dialect, "backward_reduce_split", [Pure]> {
  let summary = "Reduce one split into dK/dV in ascending split order";
  let arguments = (ins Variadic<AnyType>:$inputs);
  let results = (outs AnyType:$d_key, AnyType:$d_value);
  let assemblyFormat = "$inputs attr-dict `:` type($inputs) `->` type($d_key) `,` type($d_value)";
  let hasVerifier = 1;
}
