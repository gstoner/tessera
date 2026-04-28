//===- InsertRecomputePass.cpp — greedy activation checkpoint insertion --*- C++ -*-===//
//
// InsertRecomputePass scans ops in program order, accumulates the estimated
// live-tensor memory footprint, and inserts a tessera_sr.checkpoint whenever
// the live-set size exceeds --memory-budget-mb.
//
// Only "pure" ops (tessera.effect = "pure" or no effect attr) between two
// checkpoints are tagged with tessera_sr.recompute_hint = true.  Ops with
// side effects are never recomputable.
//
// Output attrs:
//   tessera_sr.checkpoint       — UnitAttr on the boundary op
//   tessera_sr.recompute_hint   — StringAttr("recomputable") on eligible ops
//   tessera_sr.checkpoint_id    — int64 counter per checkpoint
//
// Module attrs:
//   tessera_sr.num_checkpoints
//
//===----------------------------------------------------------------------===//

#include "tessera/sr/Passes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include <cstdint>
#include <numeric>

using namespace mlir;

namespace {

/// Estimate tensor memory in bytes from a shaped type (bf16 = 2 bytes).
static int64_t estimateTensorBytes(Type ty) {
  auto shaped = ty.dyn_cast<ShapedType>();
  if (!shaped || !shaped.hasStaticShape())
    return 4096; // conservative estimate for dynamic shapes
  int64_t elems = 1;
  for (int64_t d : shaped.getShape())
    elems *= d;
  int64_t dtype_bytes = 2; // assume bf16 by default
  if (shaped.getElementType().isF32())
    dtype_bytes = 4;
  else if (shaped.getElementType().isF64())
    dtype_bytes = 8;
  return elems * dtype_bytes;
}

/// True if an op is side-effect-free (eligible for recomputation).
static bool isPureOp(Operation *op) {
  if (auto effect = op->getAttrOfType<StringAttr>("tessera.effect"))
    return effect.getValue() == "pure" || effect.getValue() == "read";
  // Ops without an effect attr are conservatively assumed pure if they have
  // no regions and no memory operands.
  return op->getNumRegions() == 0 &&
         !op->getName().getStringRef().contains("alloc") &&
         !op->getName().getStringRef().contains("store") &&
         !op->getName().getStringRef().contains("dealloc");
}

struct InsertRecomputePass
    : public PassWrapper<InsertRecomputePass, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(InsertRecomputePass)

  Option<int64_t> memoryBudgetMB{
      *this, "memory-budget-mb",
      llvm::cl::desc("Live-tensor memory budget in MiB before inserting a "
                     "checkpoint"),
      llvm::cl::init(4096)}; // 4 GiB default

  StringRef getArgument() const final { return "tessera-insert-recompute"; }
  StringRef getDescription() const final {
    return "Greedy recomputation insertion: checkpoint when live-set exceeds "
           "memory budget";
  }

  void runOnOperation() override {
    ModuleOp mod = getOperation();
    MLIRContext *ctx = mod.getContext();

    int64_t budgetBytes = memoryBudgetMB * 1024LL * 1024LL;
    int64_t ckptId = 0;

    mod.walk([&](func::FuncOp fn) {
      int64_t liveBytes = 0;
      int64_t lastCkptId = -1;

      fn.walk([&](Operation *op) {
        // Skip non-compute ops.
        if (op->getNumResults() == 0)
          return;

        // Accumulate live tensor bytes produced by this op.
        for (Value v : op->getResults()) {
          liveBytes += estimateTensorBytes(v.getType());
        }

        // If existing checkpoint marker, reset counter.
        if (op->hasAttr("tessera_sr.checkpoint")) {
          op->setAttr("tessera_sr.checkpoint_id",
                      IntegerAttr::get(IntegerType::get(ctx, 64), ckptId++));
          liveBytes = 0;
          lastCkptId = ckptId - 1;
          return;
        }

        // Insert checkpoint when budget exceeded.
        if (liveBytes > budgetBytes) {
          op->setAttr("tessera_sr.checkpoint", UnitAttr::get(ctx));
          op->setAttr("tessera_sr.instrumented", UnitAttr::get(ctx));
          op->setAttr("tessera_sr.checkpoint_id",
                      IntegerAttr::get(IntegerType::get(ctx, 64), ckptId++));
          liveBytes = 0;
          lastCkptId = ckptId - 1;
          return;
        }

        // Tag pure ops between checkpoints as recomputable.
        if (isPureOp(op)) {
          op->setAttr("tessera_sr.recompute_hint",
                      StringAttr::get(ctx, "recomputable"));
        }
      });
    });

    mod->setAttr("tessera_sr.num_checkpoints",
                 IntegerAttr::get(IntegerType::get(ctx, 64), ckptId));
  }
};

} // namespace

std::unique_ptr<Pass> mlir::tessera::sr::createInsertRecomputePass() {
  return std::make_unique<InsertRecomputePass>();
}
