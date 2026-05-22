//===- NumericsVerifiers.cpp (v1.1) ----------------------------------------===//
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/Operation.h"
using namespace mlir;
namespace tessera { namespace numerics {

LogicalResult verifySoftmax(Operation *op, StringRef exp, double threshold) {
  if (exp != "poly3" && exp != "native")
    return op->emitOpError("exp must be 'poly3' or 'native'");
  if (threshold <= 0.0)
    return op->emitOpError("rescale_threshold must be > 0");
  return success();
}

}} // ns
