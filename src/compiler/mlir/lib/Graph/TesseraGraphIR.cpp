//===- TesseraGraphIR.cpp — Graph IR dialect registration ------------------===//
//
// Registers the Tessera Graph IR dialect ops and provides the lowering
// entry-point used by GraphToSchedulePass.
//
// Graph IR op namespace: tessera.* (no sub-prefix)
// Key ops:
//   tessera.matmul             — general matrix multiplication
//   tessera.flash_attn         — fused flash-attention kernel descriptor
//   tessera.elementwise        — arbitrary elementwise expression
//   tessera.reduce             — reduction over one or more dimensions
//   tessera.optimizer.shard    — sharding hint (carries tessera.shard attr)
//
// These ops are currently represented as MLIR generic ops (no ODS tables).
// The Graph-layer verifier checks structural invariants via op attribute
// inspection.  When ODS is wired in, replace this file with the generated
// registration code.
//===----------------------------------------------------------------------===//

#include "mlir/IR/DialectRegistry.h"
#include "mlir/IR/Operation.h"
#include "mlir/Support/LogicalResult.h"
#include "llvm/ADT/StringRef.h"

using namespace mlir;

namespace tessera {
namespace graph {

// ---------------------------------------------------------------------------
// Lightweight verifiers for Graph IR ops.
// Called from GraphToSchedulePass before annotation.
// ---------------------------------------------------------------------------

static LogicalResult verifyMatmul(Operation *op) {
  // Requires: lhs, rhs operands (rank ≥ 2 tensors); optional bias
  if (op->getNumOperands() < 2)
    return op->emitOpError("tessera.matmul requires at least 2 operands");
  return success();
}

static LogicalResult verifyFlashAttn(Operation *op) {
  // Requires: q, k, v operands; optional causal mask
  if (op->getNumOperands() < 3)
    return op->emitOpError("tessera.flash_attn requires q, k, v operands");
  auto headDim = op->getAttrOfType<IntegerAttr>("head_dim");
  if (!headDim || headDim.getInt() <= 0)
    return op->emitOpError("'head_dim' must be a positive integer");
  return success();
}

static LogicalResult verifyElementwise(Operation *op) {
  auto fnAttr = op->getAttrOfType<StringAttr>("fn");
  if (!fnAttr || fnAttr.getValue().empty())
    return op->emitOpError("tessera.elementwise requires 'fn' string attribute");
  return success();
}

static LogicalResult verifyOptimizerShard(Operation *op) {
  if (!op->getAttr("tessera.shard"))
    return op->emitOpError(
        "tessera.optimizer.shard requires 'tessera.shard' attribute");
  return success();
}

// ---------------------------------------------------------------------------
// Graph IR op verifier dispatcher
// ---------------------------------------------------------------------------
LogicalResult verifyGraphOp(Operation *op) {
  StringRef name = op->getName().getStringRef();
  if (name == "tessera.matmul")          return verifyMatmul(op);
  if (name == "tessera.flash_attn")      return verifyFlashAttn(op);
  if (name == "tessera.elementwise")     return verifyElementwise(op);
  if (name == "tessera.optimizer.shard") return verifyOptimizerShard(op);
  return success(); // unknown op — not our concern
}

} // namespace graph
} // namespace tessera
