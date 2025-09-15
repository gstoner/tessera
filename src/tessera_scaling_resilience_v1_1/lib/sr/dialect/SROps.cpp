#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/Support/LogicalResult.h"
#include <optional>

using namespace mlir;

namespace mlir { namespace tessera { namespace sr {

// Normally ODS generates classes with verify(), parse(), print().
// Here we sketch additional verifiers you'd wire via ODS 'extraClassDeclaration'.

static LogicalResult verifyCheckpointRegion(Operation *op) {
  // Example: forbid nested export_manifest inside checkpoint
  for (auto &region : op->getRegions()) {
    for (auto &block : region) {
      for (auto &inner : block) {
        if (inner.getName().stripDialect() == "export_manifest") {
          return op->emitError() << "export_manifest not allowed inside checkpoint";
        }
      }
    }
  }
  return success();
}

static LogicalResult verifyResilienceRegion(Operation *op) {
  // Example: ensure exactly one result of ResTokenType
  if (op->getNumResults() != 1)
    return op->emitError() << "resilience_region must yield exactly one token";
  return success();
}

}}} // ns