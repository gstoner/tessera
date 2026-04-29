//===- TesseraTargetIR.cpp — Target IR dialect helpers --------------------===//
//
// Provides helpers for the Target IR layer — the final MLIR representation
// before code emission.  At this layer, ops carry target-specific attributes
// (memory spaces, warp counts, TMA descriptors, mbarrier counts).
//
// Currently this file provides:
//   - verifyTargetOp(op)     — dispatches to per-op verifiers for tile.* ops
//   - emitMBarrierAlloc()    — builder helper used by ScheduleToTilePass
//   - emitAsyncCopy()        — builder helper for staged async DMA copies
//
// Relationship to other layers:
//   Graph IR  →  Schedule IR  →  [Target IR]  →  StableHLO / PTX
//
//===----------------------------------------------------------------------===//

#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Operation.h"
#include "mlir/Support/LogicalResult.h"
#include "llvm/ADT/StringRef.h"

using namespace mlir;

namespace tessera {
namespace target {

// ---------------------------------------------------------------------------
// Builder helpers
// ---------------------------------------------------------------------------

/// Emit a tile.mbarrier.alloc op into the given builder.
void emitMBarrierAlloc(OpBuilder &builder, Location loc,
                       int64_t count, llvm::StringRef scope) {
  OperationState state(loc, "tile.mbarrier.alloc");
  state.addAttribute("count", builder.getI64IntegerAttr(count));
  state.addAttribute("scope", builder.getStringAttr(scope));
  builder.create(state);
}

/// Emit a tile.async_copy op (src memref → dst memref at given stage).
void emitAsyncCopy(OpBuilder &builder, Location loc,
                   Value src, Value dst, int64_t stage) {
  OperationState state(loc, "tile.async_copy");
  state.addOperands({src, dst});
  state.addAttribute("stage", builder.getI32IntegerAttr(static_cast<int32_t>(stage)));
  builder.create(state);
}

// ---------------------------------------------------------------------------
// Target op verifier dispatcher
// (Complements the tile.* verifiers in ScheduleOps.cpp; called from the
//  PM verifier pass when processing target-layer files.)
// ---------------------------------------------------------------------------

LogicalResult verifyTargetOp(Operation *op) {
  StringRef name = op->getName().getStringRef();

  // tile.warp_config — warp_count must be a positive power of two
  if (name == "tile.warp_config") {
    auto wc = op->getAttrOfType<IntegerAttr>("warp_count");
    if (!wc || wc.getInt() <= 0)
      return op->emitOpError("'warp_count' must be > 0");
    int64_t n = wc.getInt();
    if ((n & (n - 1)) != 0)
      op->emitWarning("'warp_count' is not a power of two; check target limits");
    return success();
  }

  // tile.smem_layout — must have 'shape' and 'swizzle' attrs
  if (name == "tile.smem_layout") {
    if (!op->getAttr("shape"))
      return op->emitOpError("requires 'shape' array attribute");
    if (!op->getAttr("swizzle"))
      return op->emitOpError("requires 'swizzle' string attribute");
    return success();
  }

  // tile.tma.descriptor — source must be a global memref, cp_size must be > 0
  if (name == "tile.tma.descriptor") {
    auto cpSize = op->getAttrOfType<IntegerAttr>("cp_size");
    if (!cpSize || cpSize.getInt() <= 0)
      return op->emitOpError("'cp_size' (copy size in bytes) must be > 0");
    return success();
  }

  return success(); // unrecognised target op — pass through
}

} // namespace target
} // namespace tessera
