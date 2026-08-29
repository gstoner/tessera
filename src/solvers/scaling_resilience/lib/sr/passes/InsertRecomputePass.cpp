//===- InsertRecomputePass.cpp — greedy activation checkpoint insertion --*- C++ -*-===//
//
// InsertRecomputePass scans ops in program order, accumulates the estimated
// live-tensor memory footprint, and inserts a tessera_sr.checkpoint whenever
// the live-set size exceeds --memory-budget-mb.
//
// Only "pure" ops between two checkpoints are tagged with
// tessera_sr.recompute_hint = true.  Purity is either declared
// (tessera.effect = "pure"/"read") or DERIVED from MLIR's effect machinery —
// an op with no effect attribute is recomputable only if it is provably
// memory-effect-free.  "No attribute" does not mean pure: an RNG draw
// recomputed in the backward pass returns different values than the forward
// saw.  Ops with side effects, and ops whose effects cannot be established,
// are never recomputable.
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
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Pass/Pass.h"
#include <cstdint>
#include <numeric>

using namespace mlir;

namespace {

/// Estimate tensor memory in bytes from a shaped type (bf16 = 2 bytes).
static int64_t estimateTensorBytes(Type ty) {
  auto shaped = mlir::dyn_cast<ShapedType>(ty);
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
///
/// Purity is DERIVED, never assumed (Decision #30). The previous fallback
/// described itself as conservative while doing the opposite: an op with no
/// effect attribute was treated as pure unless its *name* happened to contain
/// "alloc", "store", or "dealloc". A `tessera.rng.uniform` or a
/// `func.call @dropout_mask` passes all three substring tests, so it was marked
/// recomputable — and a backward pass that honours the hint re-runs it, drawing
/// different randomness than the forward saw and producing a wrong gradient
/// with nothing to indicate it.
///
/// Unprovable is now unsafe: without an explicit attribute the op must be known
/// effect-free to MLIR's own effect machinery, which reports false for anything
/// it cannot see through (including unregistered ops and opaque calls).
static bool isPureOp(Operation *op) {
  if (auto effect = op->getAttrOfType<StringAttr>("tessera.effect"))
    return effect.getValue() == "pure" || effect.getValue() == "read";
  return op->getNumRegions() == 0 && isMemoryEffectFree(op);
}

struct InsertRecomputePass
    : public PassWrapper<InsertRecomputePass, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(InsertRecomputePass)

  InsertRecomputePass() = default;
  InsertRecomputePass(const InsertRecomputePass& other)
      : PassWrapper<InsertRecomputePass, OperationPass<ModuleOp>>(other) {}

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
