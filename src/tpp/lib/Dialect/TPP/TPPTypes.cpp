
#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "TPPTypes.h.inc"

using namespace mlir;
namespace tessera { namespace tpp {

// Provide extra verifiers/printers if needed.
static LogicalResult verifyFieldType(::mlir::Location loc) {
  // Minimal placeholder for v0.2: ensure layout/space strings non-empty.
  return success();
}

}} // namespace
#include "TPPTypes.cpp.inc"
