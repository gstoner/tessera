//===- Numerics.td - Tessera Numerics policy ops -----------------*- tablegen -*-===//
include "mlir/IR/OpBase.td"
include "mlir/IR/DialectSpecification.td"

def Tessera_Numerics_Dialect : Dialect {
  let name = "tessera.numerics";
  let cppNamespace = "::tessera::numerics";
  let summary = "Numerical policy controls";
}

def SoftmaxPolicyOp : Op<"softmax", [NoSideEffect]> {
  let summary = "Softmax numerics policy: exp approximation and rescale threshold";
  let arguments = (ins StrAttr:$exp, F64Attr:$rescale_threshold);
  let assemblyFormat = "$exp `,` $rescale_threshold attr-dict";
  let hasVerifier = 1;
}
