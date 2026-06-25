//===- CliffordDialect.cpp ----------------------------------------*- C++ -*-===//
#include "tessera/Clifford/CliffordDialect.h"
#include "mlir/IR/DialectImplementation.h"

using namespace mlir;
using namespace tessera::clifford;

// Generated dialect-defs must be at file scope (MLIR 22): the generated
// code references `mlir::detail::TypeIDResolver<...>` which requires the
// `mlir::detail::` qualifier resolve from file scope.
#include "CliffordDialect.cpp.inc"

void CliffordDialect::initialize() {
  // Register all ops (generated from CliffordOps.td).
  addOperations<
#define GET_OP_LIST
#include "CliffordOps.cpp.inc"
      >();
}
