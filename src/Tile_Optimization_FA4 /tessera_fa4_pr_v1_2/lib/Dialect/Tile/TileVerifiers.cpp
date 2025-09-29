//===- TileVerifiers.cpp (v1.2) --------------------------------------------===//
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/Operation.h"
using namespace mlir;
namespace tessera { namespace tile {

LogicalResult verifyAllocTMEM(Operation *op) { return success(); }

LogicalResult verifyTcgen05(Operation *op, int32_t ctaGroup) {
  if (ctaGroup < 1 || ctaGroup > 4)
    return op->emitOpError("cta_group must be in [1,4]");
  return success();
}

}} // ns
