
//===- ScheduleIR_GEMM_BN.td ----------------------------------*- tablegen -*-===//
include "mlir/IR/OpBase.td"

def TSched_Dialect : Dialect {
  let name = "tsched";
  let cppNamespace = "::tessera::schedule";
  let summary = "Scheduling, tiling, and fusion hints";
}

def TSched_GemmOp : Op<TSched_Dialect, "gemm", []> {
  let summary = "GEMM with explicit tiling and epilogue selection";
  let arguments = (ins AnyType:$A, AnyType:$B,
                        I32Attr:$bm, I32Attr:$bn, I32Attr:$bk,
                        StrAttr:$epilogue); // e.g. "none","bias_relu"
  let results = (outs AnyType:$C);
  let assemblyFormat = "$A `,` $B attr-dict `:` type($C)";
}

def TSched_BatchNormOp : Op<TSched_Dialect, "batch_norm", []> {
  let summary = "BatchNorm scheduled with vector width (last dim)";
  let arguments = (ins AnyType:$X, AnyType:$Mean, AnyType:$Var,
                        AnyType:$Gamma, AnyType:$Beta,
                        F32Attr:$eps, I32Attr:$vecWidth);
  let results = (outs AnyType:$Y);
}
