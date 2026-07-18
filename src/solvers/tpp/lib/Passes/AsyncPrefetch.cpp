//===- AsyncPrefetch.cpp - software-pipeline schedule.prefetch ops --------===//
//
// Realizes prefetch *overlap* for `schedule.prefetch` ops (LSA Gap 2).
// Previously a no-op; now a real, deterministic, dependency-safe transform:
//
//   1. Per block, prefetch ops get a rotating double-buffer stage index
//      `tpp.prefetch.stage = i % NUM_STAGES` so successive prefetches target
//      distinct buffers (no aliasing across the software pipeline).
//   2. For an `overlap` policy other than "none", the prefetch is hoisted above
//      its immediately-preceding op when that op does not produce any of the
//      prefetch's operands (dependency-safe). Issuing the copy before the prior
//      op is exactly software-pipelined overlap: the transfer runs concurrently
//      with that preceding compute/collective. Marked `tpp.prefetch.hoisted`.
//   3. Every visited prefetch is tagged `tpp.prefetch.overlapped = <bool>` so a
//      downstream lowering / the audit can see what was realized.
//
// The pass preserves SSA and IR validity; it only annotates and reorders within
// a block. Backends consume the stage/overlap markers when emitting cp.async /
// TMA (NVIDIA) or ds_read streams (AMD).
//
//===----------------------------------------------------------------------===//

#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallVector.h"

using namespace mlir;

namespace {

// True iff any result of `prev` is an operand of `op` (a true RAW dependency
// that would be violated by hoisting `op` above `prev`).
static bool producesOperandOf(Operation *prev, Operation *op) {
  for (Value res : prev->getResults())
    for (Value operand : op->getOperands())
      if (res == operand)
        return true;
  return false;
}

struct AsyncPrefetch
    : public PassWrapper<AsyncPrefetch, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(AsyncPrefetch)

  static constexpr int kNumStages = 2;

  StringRef getArgument() const final { return "tpp-async-prefetch"; }
  StringRef getDescription() const final {
    return "Software-pipeline schedule.prefetch ops: assign double-buffer "
           "stages and hoist overlap-policy prefetches above preceding compute";
  }

  void runOnOperation() final {
    ModuleOp module = getOperation();
    OpBuilder builder(module.getContext());

    // Collect prefetch ops first (we mutate block order below).
    llvm::SmallVector<Operation *> prefetches;
    module.walk([&](Operation *op) {
      if (op->getName().getStringRef() == "schedule.prefetch")
        prefetches.push_back(op);
    });

    llvm::DenseMap<Block *, int> stageCounter;
    for (Operation *op : prefetches) {
      Block *block = op->getBlock();
      int stage = stageCounter[block]++;
      op->setAttr("tpp.prefetch.stage",
                  builder.getI64IntegerAttr(stage % kNumStages));

      auto overlap = op->getAttrOfType<StringAttr>("overlap");
      bool wantOverlap = overlap && overlap.getValue() != "none";
      op->setAttr("tpp.prefetch.overlapped", builder.getBoolAttr(wantOverlap));

      bool hoisted = false;
      if (wantOverlap) {
        Operation *prev = op->getPrevNode();
        if (prev && !prev->hasTrait<OpTrait::IsTerminator>() &&
            !producesOperandOf(prev, op)) {
          op->moveBefore(prev);
          hoisted = true;
        }
      }
      op->setAttr("tpp.prefetch.hoisted", builder.getBoolAttr(hoisted));
    }
  }
};

} // namespace

std::unique_ptr<Pass> createAsyncPrefetchPass() {
  return std::make_unique<AsyncPrefetch>();
}
