//===- AttnVerifiers.cpp (v1.2) --------------------------------------------===//
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/Operation.h"
using namespace mlir;
namespace tessera { namespace attn {

LogicalResult verifyLseSave(Operation *op) {
  // Future: check shape agreement (scores row count == lse length)
  return success();
}

}} // ns
