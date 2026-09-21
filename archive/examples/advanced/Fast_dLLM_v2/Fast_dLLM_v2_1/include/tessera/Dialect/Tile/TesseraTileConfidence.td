//===- TesseraTileConfidence.td - Confidence & LCP ops --------*- tablegen -*-===//
// Minimal extension ops for confidence metrics and prefix LCP reduction.

include "mlir/IR/OpBase.td"

def Tessera_Tile_Dialect : Dialect {
  let name = "tessera.tile";
  let cppNamespace = "::tessera::tile";
}

def ConfidenceStatsOp : Op<Tessera_Tile_Dialect, "confidence_stats", [NoSideEffect]> {
  let summary = "Compute confidence metrics (entropy, top-p margin) per token";
  let arguments = (ins AnyTensor:$logits);
  let results   = (outs AnyTensor:$stats);
  let assemblyFormat = "$logits attr-dict `:` type($logits) `->` type($stats)";
}

def PrefixLCPOp : Op<Tessera_Tile_Dialect, "prefix_lcp", [NoSideEffect]> {
  let summary = "Compute longest common prefix length across K branches";
  let arguments = (ins AnyTensor:$tokens);
  let results   = (outs I32:$lcp_len);
  let assemblyFormat = "$tokens attr-dict `:` type($tokens)";
}
