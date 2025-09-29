//===- AttnVerifiers.cpp (v1.3) --------------------------------------------===//
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/Operation.h"
using namespace mlir;
namespace tessera { namespace attn {

// Ensure that when scores is memref<RxCxf32>, lse is memref<Rxf32> (R rows)
static bool getMemRefShape(Type t, SmallVector<int64_t,2> &shape) {
  if (auto mr = t.dyn_cast<MemRefType>()) {
    for (auto d : mr.getShape()) shape.push_back(d);
    return true;
  }
  return false;
}

LogicalResult verifyLseSave(Operation *op) {
  if (op->getNumOperands() != 1 || op->getNumResults() != 1) return success();
  SmallVector<int64_t,2> sShape, lShape;
  if (!getMemRefShape(op->getOperand(0).getType(), sShape)) return success();
  if (!getMemRefShape(op->getResult(0).getType(),  lShape)) return success();
  if (sShape.size() < 2 || lShape.size() != 1) return op->emitOpError("expect scores rank>=2 and lse rank=1");
  if (sShape[0] != -1 && lShape[0] != -1 && sShape[0] != lShape[0])
    return op->emitOpError("lse length must equal scores rows");
  return success();
}

}} // ns
