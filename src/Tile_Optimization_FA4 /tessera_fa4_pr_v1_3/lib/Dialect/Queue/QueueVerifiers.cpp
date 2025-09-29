//===- QueueVerifiers.cpp (v1.3) -------------------------------------------===//
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/Operation.h"
using namespace mlir;
namespace tessera { namespace queue {
LogicalResult verifyCreate(Operation *op) { return success(); }
LogicalResult verifyPush(Operation *op) { return success(); }
LogicalResult verifyPop(Operation *op)  { return success(); }
}} // ns
