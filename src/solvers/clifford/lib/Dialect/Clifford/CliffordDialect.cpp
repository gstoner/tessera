//===- CliffordDialect.cpp ----------------------------------------*- C++ -*-===//
#include "tessera/Clifford/CliffordDialect.h"
#include "mlir/IR/DialectImplementation.h"

using namespace mlir;

namespace tessera {
namespace clifford {
#include "CliffordDialect.cpp.inc"

void CliffordDialect::initialize() {
  // Register all ops (generated from CliffordOps.td).
  addOperations<
#define GET_OP_LIST
#include "CliffordOps.cpp.inc"
      >();
}

}  // namespace clifford
}  // namespace tessera
