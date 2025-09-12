//===- TPP.td --------------------------------------------*- tablegen -*-===//
#ifndef TESSERA_TPP_DIALECT
#define TESSERA_TPP_DIALECT

include "mlir/IR/OpBase.td"
include "mlir/IR/DialectSpecification.td"

def TPP_Dialect : Dialect {
  let name = "tpp";
  let cppNamespace = "::tessera::tpp";
  let summary = "Tensor Processing Primitives (space–time, physics, world models)";
  let description = [{The TPP dialect defines types, attributes, and operations to
represent space–time fields, meshes, stencils, solvers, RNG, ensembles, and world-model ops.}];
}

#endif // TESSERA_TPP_DIALECT
