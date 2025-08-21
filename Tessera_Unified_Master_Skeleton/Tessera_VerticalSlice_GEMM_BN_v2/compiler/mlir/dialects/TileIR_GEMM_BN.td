
//===- TileIR_GEMM_BN.td --------------------------------------*- tablegen -*-===//
include "mlir/IR/OpBase.td"

def TTile_Dialect : Dialect {
  let name = "ttile";
  let cppNamespace = "::tessera::tile";
  let summary = "Block/warp/tensorcore-tile level ops";
}

def TTile_GemmMMAOp : Op<TTile_Dialect, "gemm_mma", []> {
  let summary = "Macro-tile GEMM using MMA instructions";
  let arguments = (ins AnyType:$ATile, AnyType:$BTile,
                        I32Attr:$bm, I32Attr:$bn, I32Attr:$bk,
                        StrAttr:$epilogue);
  let results = (outs AnyType:$CTile);
}

def TTile_BatchNormVecOp : Op<TTile_Dialect, "batch_norm_vec", []> {
  let summary = "Vectorized BN over the last dimension";
  let arguments = (ins AnyType:$XTile, AnyType:$Mean, AnyType:$Var,
                        AnyType:$Gamma, AnyType:$Beta,
                        F32Attr:$eps, I32Attr:$vecWidth);
  let results = (outs AnyType:$YTile);
}
