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
//   * Only pure, region-free ops qualify. Two hard gates, each a loud error
//     rather than a silent skip (which would leave a stale `tessera.recompute`
//     marker and a wrong memory model):
//       - nested regions (control flow) → `[REMAT_NON_CLONABLE]`.
//       - not provably side-effect-free (`mlir::isMemoryEffectFree`) →
//         `[REMAT_EFFECTFUL]`. Re-executing an effectful op (RNG like dropout,
//         a collective, a store/copy) on the backward path would change program
//         semantics, not merely trade memory for compute. Tessera Graph IR ops
//         are `[Pure]`, so this admits the real activation ops and rejects the
//         effectful ones; an op that does not model its effects is treated as
//         effectful (conservative — we never recompute what we can't prove pure).
//   * Clone placement is always valid without a dominance query: a user `U`
//     uses the recompute op `P`, so `P` dominates `U`; `P`'s operands dominate
//     `P` (SSA); by transitivity they dominate `U`, hence the clone inserted
//     right before `U`. Producer chains are handled by walking recompute ops in
//     reverse program order (consumers before producers), so a whole tagged
//     chain rematerializes together at the final consumer instead of leaving
//     the earlier producer's clone live from the forward block.
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
#include "mlir/Interfaces/SideEffectInterfaces.h"
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

    mlir::OpBuilder builder(&getContext());
    int64_t rematCount = 0;
    bool failed = false;

    // Walk in REVERSE program order — consumers before producers. For a tagged
    // producer chain (%a feeds %b feeds a backward user, both tagged), handling
    // the consumer %b first sinks its clone to the backward user; %a is then
    // seen with %b's clone as its user, so %a's clone lands next to it — the
    // whole chain rematerializes together at the consumer. Forward order would
    // instead leave %a's clone stranded next to %b in the forward block, still
    // live across to the backward, defeating the checkpoint.
    for (mlir::Operation *op : llvm::reverse(recomputeOps)) {
      // Gate 1: region-free. Cloning a control-flow op is out of scope.
      if (op->getNumRegions() != 0) {
        op->emitError()
            << "[REMAT_NON_CLONABLE] op '" << op->getName().getStringRef()
            << "' is tagged " << kRecomputeAttr
            << " but carries nested regions; only pure region-free ops can be "
               "rematerialized";
        failed = true;
        continue;
      }
      // Gate 2: provably side-effect-free. Re-executing an effectful op (RNG,
      // collective, store/copy) on the backward path would change program
      // semantics — recompute trades memory for *compute*, nothing else. An op
      // that doesn't model its effects is conservatively treated as effectful.
      if (!mlir::isMemoryEffectFree(op)) {
        op->emitError()
            << "[REMAT_EFFECTFUL] op '" << op->getName().getStringRef()
            << "' is tagged " << kRecomputeAttr
            << " but is not provably side-effect-free; rematerializing it would "
               "re-execute its effects and change program semantics — only pure "
               "ops qualify (Decision #10)";
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
        // Clone placement is always valid: `user` uses `op`, so `op` dominates
        // `user`; `op`'s operands dominate `op` (SSA); by transitivity they
        // dominate `user`, so the clone inserted right before `user` sees them.
        // No dominance query needed (and none would be stable across the
        // clones we insert into freshly-relocated chain users).
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
