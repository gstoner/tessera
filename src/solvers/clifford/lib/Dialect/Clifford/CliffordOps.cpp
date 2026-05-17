//===- CliffordOps.cpp ----------------------------------------*- C++ -*-===//
#include "tessera/Clifford/CliffordDialect.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/OpImplementation.h"

using namespace mlir;

namespace tessera {
namespace clifford {

#define GET_OP_CLASSES
#include "CliffordOps.cpp.inc"

}  // namespace clifford
}  // namespace tessera
