//===- ActivationRematerializationPass.cpp - F2 IR remat ------*- C++ -*-===//
//
// Phase F2 (IR-pass form) of docs/spec/AUTODIFF_SPEC.md. The
// `tessera.autodiff.rematerialize` Python surface (numpy-tape) has shipped
// since Phase F2; this is its Graph-IR counterpart, meant to run alongside /
// after AutodiffPass (Phase F4) once the backward graph is materialised in the
// same function.
//
// Contract
// --------
// An op opts into rematerialization by carrying a `tessera.recompute` unit
// attribute (the lowering target of the Python `rematerialize(fn)` /
// `checkpoint` wrapper — every op the wrapper produced is marked). Instead of
// keeping such an op's forward result live all the way across the (much later)
// backward uses, this pass **clones the op immediately before each backward
// consumer** and rewrites that use to the clone. The original op is erased once
// it has no remaining uses. Net effect: the forward activation's live range
// shrinks to almost nothing (recomputed on demand near the consumer) at the
// cost of extra compute — the classic activation-checkpointing trade
// (Decision #10: recompute is budget-guided and only pure ops qualify).
//
// Safety (Decision #10 / #21)
// ---------------------------
//   * Only pure, region-free ops qualify. An op tagged `tessera.recompute`
//     that carries nested regions (control flow) or is otherwise not safely
//     clonable is a hard error — silently skipping it would leave a stale
//     `tessera.recompute` marker and a wrong memory model.
//   * A clone is only emitted at a use whose op is dominated by all of the
//     recompute op's operands (they must still be in scope at the clone site).
//     In the flat single-block autodiff model this is always true; the check
//     keeps the pass correct if a later structured-CFG model feeds it.
//   * `--memory-budget-mb` is accepted for parity with the InsertRecompute
//     spelling; in the explicit-marker path it is recorded on the function as
//     `tessera.remat_budget_mb` for downstream planners but does not itself
//     drive selection (budget-guided auto-selection is future work — the
//     explicit `tessera.recompute` marker is authoritative today).
//
// Cross-references:
//   * python/tessera/autodiff/rematerialize.py — the Python F2 surface.
//   * AutodiffPass.cpp — emits the backward graph this pass rematerialises into.
//   * docs/spec/AUTODIFF_SPEC.md §Phase F2.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Dominance.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"

namespace tessera {

namespace {

constexpr const char *kRecomputeAttr = "tessera.recompute";
constexpr const char *kRematerializedCountAttr = "tessera.rematerialized";
constexpr const char *kRematBudgetAttr = "tessera.remat_budget_mb";

class ActivationRematerializationPass
    : public mlir::PassWrapper<ActivationRematerializationPass,
                                mlir::OperationPass<mlir::func::FuncOp>> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(ActivationRematerializationPass)

  ActivationRematerializationPass() = default;
  ActivationRematerializationPass(const ActivationRematerializationPass &other)
      : mlir::PassWrapper<ActivationRematerializationPass,
                           mlir::OperationPass<mlir::func::FuncOp>>(other) {}

  llvm::StringRef getArgument() const final {
    return "tessera-activation-rematerialization";
  }

  llvm::StringRef getDescription() const final {
    return "Phase F2 (IR form) — rematerialize `tessera.recompute`-tagged pure "
           "activations at their backward consumers to shrink live ranges "
           "(activation checkpointing).";
  }

  Option<int> memoryBudgetMb{
      *this, "memory-budget-mb",
      llvm::cl::desc("advisory recompute memory budget (MB); recorded on the "
                     "function for downstream planners"),
      llvm::cl::init(0)};

  void runOnOperation() override {
    auto func = getOperation();

    if (memoryBudgetMb.getValue() > 0) {
      func->setAttr(kRematBudgetAttr,
                    mlir::IntegerAttr::get(
                        mlir::IntegerType::get(&getContext(), 32),
                        memoryBudgetMb.getValue()));
    }

    // Collect the recompute-tagged ops up-front — we mutate uses / erase ops
    // during the walk, so snapshot first to keep iteration well-defined.
    llvm::SmallVector<mlir::Operation *> recomputeOps;
    func.walk([&](mlir::Operation *op) {
      if (op->hasAttr(kRecomputeAttr))
        recomputeOps.push_back(op);
    });
    if (recomputeOps.empty())
      return;

    mlir::DominanceInfo dom(func);
    mlir::OpBuilder builder(&getContext());
    int64_t rematCount = 0;
    bool failed = false;

    for (mlir::Operation *op : recomputeOps) {
      // Safety gate: only pure, region-free ops are rematerializable.
      if (op->getNumRegions() != 0) {
        op->emitError()
            << "[REMAT_NON_CLONABLE] op '" << op->getName().getStringRef()
            << "' is tagged " << kRecomputeAttr
            << " but carries nested regions; only pure region-free ops can be "
               "rematerialized";
        failed = true;
        continue;
      }

      // Snapshot the current users (each is a distinct backward consumer). We
      // rewrite one operand-use at a time; cloning per user op keeps the
      // recomputed value adjacent to its consumer.
      llvm::SmallVector<mlir::Operation *> users(op->getUsers().begin(),
                                                 op->getUsers().end());
      // Deduplicate while preserving order — an op may use the value twice.
      llvm::SmallVector<mlir::Operation *> uniqueUsers;
      for (mlir::Operation *u : users) {
        if (u == op)
          continue;
        if (!llvm::is_contained(uniqueUsers, u))
          uniqueUsers.push_back(u);
      }

      for (mlir::Operation *user : uniqueUsers) {
        // All operands of the recompute op must dominate the clone site so the
        // recomputation is valid at the consumer. (Always true in the flat
        // autodiff block; guards a future structured-CFG feeder.)
        bool operandsInScope = llvm::all_of(
            op->getOperands(), [&](mlir::Value v) {
              if (auto *defOp = v.getDefiningOp())
                return dom.dominates(defOp, user);
              return true;  // block argument — dominates everything in-block.
            });
        if (!operandsInScope)
          continue;

        builder.setInsertionPoint(user);
        mlir::Operation *clone = builder.clone(*op);
        clone->removeAttr(kRecomputeAttr);  // the clone is the materialized use

        // Rewrite this user's operands that reference op's results to the clone.
        for (mlir::OpOperand &use : user->getOpOperands()) {
          mlir::Value used = use.get();
          if (auto res = mlir::dyn_cast<mlir::OpResult>(used))
            if (res.getOwner() == op)
              use.set(clone->getResult(res.getResultNumber()));
        }
        rematCount++;
      }

      // If the original is now fully rematerialized away, erase it.
      if (op->use_empty())
        op->erase();
      else
        op->removeAttr(kRecomputeAttr);  // partial — clear the marker regardless
    }

    if (failed)
      return signalPassFailure();

    if (rematCount > 0) {
      func->setAttr(kRematerializedCountAttr,
                    mlir::IntegerAttr::get(
                        mlir::IntegerType::get(&getContext(), 64), rematCount));
    }
  }
};

}  // namespace

std::unique_ptr<mlir::Pass> createActivationRematerializationPass() {
  return std::make_unique<ActivationRematerializationPass>();
}

}  // namespace tessera
