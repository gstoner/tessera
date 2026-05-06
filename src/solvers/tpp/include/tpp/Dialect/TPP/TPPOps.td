//===- TPPOps.td -----------------------------------------*- tablegen -*-===//
include "mlir/IR/OpBase.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/Interfaces/InferTypeOpInterface.td"
include "mlir/IR/SymbolInterfaces.td"
include "TPP.td"
include "TPPTypes.td"
include "TPPAttrs.td"

def TPP_BCEnforceOp : Op<TPP_Dialect, "bc.enforce", [Pure]> {
  let summary = "Apply boundary condition policy to a field";
  let arguments = (ins AnyType:$field, OptionalAttr<TPP_BCAttr>:$bc);
  let results   = (outs AnyType:$out);
  let assemblyFormat = "$field attr-dict `:` type($field) `->` type($out)";
}

def TPP_GradOp : Op<TPP_Dialect, "grad", [Pure]> {
  let summary = "Gradient of a field";
  let arguments = (ins AnyType:$x);
  let results   = (outs AnyType:$y);
  let assemblyFormat = "$x attr-dict `:` type($x) `->` type($y)";
}

def TPP_StencilApplyOp : Op<TPP_Dialect, "stencil.apply", [Pure]> {
  let summary = "Apply a stencil kernel with a given radius/order";
  let arguments = (ins AnyType:$x, AnyType:$kernel);
  let results   = (outs AnyType:$y);
  let assemblyFormat = "$x `,` $kernel attr-dict `:` type($x) `,` type($kernel) `->` type($y)";
}

def TPP_TimeStepOp : Op<TPP_Dialect, "time.step", [Pure]> {
  let summary = "Temporal step region (e.g., RK schemes)";
  let arguments = (ins Variadic<AnyType>:$state);
  let results   = (outs Variadic<AnyType>:$out);
  let regions = (region AnyRegion:$body);
  let assemblyFormat = "$state attr-dict-with-keyword $body `:` functional-type($state, $out)";
}
