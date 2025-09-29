//===- Attn.td (v1.3) ------------------------------------------------------*- tablegen -*-===//
include "mlir/IR/OpBase.td"
include "mlir/IR/DialectSpecification.td"

def Tessera_Attn_Dialect : Dialect {
  let name = "tessera.attn";
  let cppNamespace = "::tessera::attn";
}

def LseSaveOp : Op<"lse.save", []> {
  let summary = "Save per-row log-sum-exp (LSE) from forward attention";
  let arguments = (ins AnyType:$scores);
  let results = (outs AnyType:$lse);
  let assemblyFormat = "$scores attr-dict";
  let hasVerifier = 1; // shape-aware check in C++
}

def LseLoadOp : Op<"lse.load", [NoSideEffect]> {
  let summary = "Load LSE tensor for backward";
  let results = (outs AnyType:$lse);
  let assemblyFormat = "attr-dict";
}
