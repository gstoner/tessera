
//===- TPPOps.td -------------------------------------------------*- tablegen -*-===//
#ifndef TESSERA_TPP_OPS
#define TESSERA_TPP_OPS

include "mlir/IR/OpBase.td"
include "mlir/IR/OpImplementation.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/Interfaces/InferTypeOpInterface.td"
include "mlir/IR/SymbolInterfaces.td"
include "TPP.td"

// Types/Attrs can be split later; provide basic skeleton attributes.
def TPP_UnitsAttr : StrAttr<"Units", "tpp.units">;
def TPP_ConserveAttr : StrAttr<"Conserve", "tpp.conservation">;
def TPP_BCAttr : StrAttr<"BC", "tpp.bc">;

// Example basic op: boundary condition enforcement.
def TPP_BCEnforceOp : Op<"tpp.bc.enforce", [NoSideEffect]> {
  let summary = "Apply boundary conditions to a field";
  let arguments = (ins AnyType:$field, OptionalAttr<TPP_BCAttr>:$bc);
  let results   = (outs AnyType:$out);
  let assemblyFormat = "$field attr-dict `:` type($field) `->` type($out)";
}

// Gradient op (sketch)
def TPP_GradOp : Op<"tpp.grad", [NoSideEffect]> {
  let summary = "Field gradient along axes";
  let arguments = (ins AnyType:$x);
  let results   = (outs AnyType:$y);
  let assemblyFormat = "$x attr-dict `:` type($x) `->` type($y)";
}

#endif // TESSERA_TPP_OPS
