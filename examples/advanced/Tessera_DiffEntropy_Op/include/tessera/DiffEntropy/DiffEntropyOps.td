//===- DiffEntropyOps.td ------------------------------------*- tablegen -*-===//
// Tessera: Differentiable Entropy Regularization ODS
// Exposes:
//   - tessera.diffentropy.range_entropy_soft
//   - tessera.diffentropy.attn_row_entropy
//------------------------------------------------------------------------------
include "mlir/IR/OpBase.td"
include "mlir/Interfaces/InferTypeOpInterface.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

def Tessera_DiffEntropy_Dialect : Dialect {
  let name = "tessera.diffentropy";
  let cppNamespace = "::tessera::diffentropy";
  let description = "Differentiable entropy regularization ops";
}

def RangeFamilyAttr : StrEnumAttr<"RangeFamily",
  ["balls", "halfspaces", "rects"]> {
  let summary = "Range family used to induce soft partitioning cells";
}

def Tessera_RangeEntropySoftOp : Op<
    "tessera.diffentropy.range_entropy_soft",
    [SameVariadicOperandSize, Pure]> {
  let summary = "Differentiable surrogate for range-partition entropy on point sets";
  let description = [{
Given a batch of point sets X[b, n, d] and either anchors A[b, k, d] (for
`balls`) or halfspace parameters H[b, k, d+1] (w,v where v is bias) for
`halfspaces`, compute a soft partition and return the (optionally reduced)
entropy H = - sum_c p_c * log p_c. Temperature alpha controls sharpness.
}];
  let arguments = (ins
    AnyMemRef:$points,                 // [B,N,D] or [N,D]
    AnyMemRef:$params,                 // anchors [B,K,D] or halfspaces [B,K,D+1]
    F64Attr:$alpha,                    // temperature (>0)
    RangeFamilyAttr:$range_family,     // balls | halfspaces | rects
    StrAttr:$reduction                 // "none" | "mean" | "sum"
  );
  let results = (outs AnyMemRef:$out); // [B] if reduction="none" else []
  let assemblyFormat = [{ $points `,` $params attr-dict `:` type($points) `,` type($params) `->` type($out) }];
}

def Tessera_AttnRowEntropyOp : Op<
    "tessera.diffentropy.attn_row_entropy",
    [Pure]> {
  let summary = "Row-wise entropy penalty for attention weights";
  let description = [{
Given attention scores or probabilities A[b, h, q, k], compute per-row
entropy H_row = -sum_j p_ij log p_ij, where p = softmax(A / tau) if `scores`,
or p = A if `probs` are passed. Reductions: none|mean|sum.
}];
  let arguments = (ins
    AnyMemRef:$attn,                   // [B,H,Q,K]
    F64Attr:$tau,                      // softmax temperature (if scores input)
    StrAttr:$mode,                     // "scores"|"probs"
    StrAttr:$reduction                 // "none"|"mean"|"sum"
  );
  let results = (outs AnyMemRef:$out); // [B,H,Q] if none else []
  let assemblyFormat = [{ $attn attr-dict `:` type($attn) `->` type($out) }];
}
