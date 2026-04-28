//===- SparsePrecond.cpp — select preconditioner for sparse solver ops ---*- C++ -*-===//
//
// Reads tessera_solver.sparse_hint and tessera_solver.fill_fraction attrs set
// by SparseInspectorPass, then attaches tessera_solver.precond = "jacobi" |
// "ilu" | "amg" based on heuristics.
//
// Rules:
//   fill < 1%               → AMG  (very sparse, likely SPD, multigrid wins)
//   fill ∈ [1%, 3%)         → ILU  (moderately sparse, general)
//   fill ∈ [3%, threshold)  → Jacobi (diagonal-dominant, cheap)
//   no sparse_hint          → no precond attr (pass is a no-op for this op)
//
//===----------------------------------------------------------------------===//

#include "SolversPasses.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/LogicalResult.h"

using namespace mlir;

namespace {

struct SparsePrecondPass
    : PassWrapper<SparsePrecondPass, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(SparsePrecondPass)

  StringRef getArgument() const final { return "tessera-sparse-precond"; }
  StringRef getDescription() const final {
    return "Select Jacobi/ILU/AMG preconditioner for sparse-tagged ops";
  }

  void runOnOperation() override {
    ModuleOp mod = getOperation();
    MLIRContext *ctx = mod.getContext();

    mod.walk([&](Operation *op) {
      // Only process ops tagged by SparseInspectorPass.
      if (!op->hasAttr("tessera_solver.sparse_hint"))
        return;

      double fill = 0.005; // conservative default
      if (auto fa = op->getAttrOfType<FloatAttr>("tessera_solver.fill_fraction"))
        fill = fa.getValueAsDouble();

      StringRef precond;
      if (fill < 0.01)
        precond = "amg";
      else if (fill < 0.03)
        precond = "ilu";
      else
        precond = "jacobi";

      op->setAttr("tessera_solver.precond",
                  StringAttr::get(ctx, precond));
    });
  }
};

} // namespace

namespace tessera {
namespace passes {
std::unique_ptr<Pass> createSparsePrecondPass() {
  return std::make_unique<SparsePrecondPass>();
}
} // namespace passes
} // namespace tessera
