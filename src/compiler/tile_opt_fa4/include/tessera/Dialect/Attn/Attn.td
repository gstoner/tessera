//===- Attn.td (v2.0) ------------------------------------------------------*- tablegen -*-===//
// Phase 3 additions: ScaledDotProduct, OnlineSoftmax, LseAccumulate,
//                    DropoutMask, CausalMask — plus original LseSave/LseLoad.
include "mlir/IR/OpBase.td"
include "mlir/IR/DialectSpecification.td"

def Tessera_Attn_Dialect : Dialect {
  let name = "tessera.attn";
  let cppNamespace = "::tessera::attn";
}

// ─── Original ops (v1.3) ─────────────────────────────────────────────────────

def LseSaveOp : Op<Tessera_Attn_Dialect, "lse.save", []> {
  let summary = "Save per-row log-sum-exp (LSE) from forward attention";
  let arguments = (ins AnyType:$scores);
  let results = (outs AnyType:$lse);
  let assemblyFormat = "$scores attr-dict";
  let hasVerifier = 1;
}

def LseLoadOp : Op<Tessera_Attn_Dialect, "lse.load", [Pure]> {
  let summary = "Load LSE tensor for backward";
  let results = (outs AnyType:$lse);
  let assemblyFormat = "attr-dict";
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
  let assemblyFormat = "$scores `,` $running_m `,` $running_l `,` $acc_out attr-dict";
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
  let assemblyFormat = "$acc `,` $running_m `,` $running_l attr-dict";
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
  let assemblyFormat = "$scores `p` `=` $dropout_p `seed` `=` $seed attr-dict";
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
  let assemblyFormat = "$scores `q_off` `=` $q_offset `kv_off` `=` $kv_offset attr-dict";
}
