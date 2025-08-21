
//===- GraphIR_GEMM_BN.td -------------------------------------*- tablegen -*-===//
include "mlir/IR/OpBase.td"

def TGraph_Dialect : Dialect {
  let name = "tgraph";
  let cppNamespace = "::tessera::graph";
  let summary = "High-level ops (autodiff domain)";
}

def TGraph_GemmOp : Op<TGraph_Dialect, "gemm", []> {
  let summary = "C = alpha * (A x B) + beta * C0 (optional epilogue fusion)";
  let arguments = (ins AnyType:$A, AnyType:$B,
                   OptionalAttr<F32Attr>:$alpha,
                   OptionalAttr<F32Attr>:$beta,
                   UnitAttr:$transa, UnitAttr:$transb,
                   Optional<AnyType>:$C0);
  let results = (outs AnyType:$C);
  let assemblyFormat = "$A `,` $B (`,` $C0^)? attr-dict `:` type($C)";
}

def TGraph_BatchNormOp : Op<TGraph_Dialect, "batch_norm", []> {
  let summary = "y = gamma * (x - mean)/sqrt(var+eps) + beta (inference)";
  let arguments = (ins AnyType:$X, AnyType:$Mean, AnyType:$Var,
                        AnyType:$Gamma, AnyType:$Beta, F32Attr:$eps);
  let results = (outs AnyType:$Y);
  let assemblyFormat = "$X `,` $Mean `,` $Var `,` $Gamma `,` $Beta attr-dict `:` type($Y)";
}
