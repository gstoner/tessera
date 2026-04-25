//===- TileVerifiers.cpp (v1.1) --------------------------------------------===//
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/Operation.h"
using namespace mlir;
namespace tessera { namespace tile {

// TMEM alloc: ensure memref has TMEM memory space via attribute (placeholder check)
LogicalResult verifyAllocTMEM(Operation *op) {
  // In a real impl, inspect MemRefType's memory space or encoding.
  // Here we accept and rely on type parsers; placeholder OK.
  return success();
}

LogicalResult verifyTcgen05(Operation *op, int32_t ctaGroup) {
  if (ctaGroup < 1 || ctaGroup > 4)
    return op->emitOpError("cta_group must be in [1,4]");
  return success();
}

}} // ns
