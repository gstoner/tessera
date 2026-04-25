
//===- TPP.td - Tessera TPP Dialect ------------------------------*- tablegen -*-===//
#ifndef TESSERA_TPP_DIALECT
#define TESSERA_TPP_DIALECT

include "mlir/IR/OpBase.td"
include "mlir/IR/DialectSpecification.td"

def TPP_Dialect : Dialect {
  let name = "tpp";
  let cppNamespace = "::tessera::tpp";
  let summary = "Tensor Processing Primitives for space–time, physics, and world models";
  let description = [{
    The TPP dialect defines types/attributes/ops to represent space–time fields, meshes,
    stencils, physics-aware solvers, ensembles, RNG, and world-model operations.
  }];
}

#endif // TESSERA_TPP_DIALECT
