//===- SparseSolverSpecialize.cpp — choose solver variant ----------------*- C++ -*-===//
//
// Reads tessera_solver.precond attr (set by SparsePrecondPass) and attaches
// tessera_solver.solver = "cg" | "gmres" | "bicgstab".
//
// Rules:
//   precond = "amg"    → solver = "cg"       (AMG is tuned for SPD/CG)
//   precond = "ilu"    → solver = "gmres"    (general non-symmetric)
//   precond = "jacobi" → solver = "gmres"    (simple precond, gmres is safe)
//   no precond attr    → solver = "gmres"    (safe default)
//
//===----------------------------------------------------------------------===//

#include "SolversPasses.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"

using namespace mlir;

namespace {

struct SparseSolverSpecializePass
    : PassWrapper<SparseSolverSpecializePass, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(SparseSolverSpecializePass)

  StringRef getArgument() const final { return "tessera-sparse-solver-specialize"; }
  StringRef getDescription() const final {
    return "Specialize sparse solver variant (cg/gmres/bicgstab) from precond attr";
  }

  void runOnOperation() override {
    ModuleOp mod = getOperation();
    MLIRContext *ctx = mod.getContext();

    mod.walk([&](Operation *op) {
      // Only annotate sparse ops.
      if (!op->hasAttr("tessera_solver.sparse_hint"))
        return;

      StringRef precond = "none";
      if (auto pa = op->getAttrOfType<StringAttr>("tessera_solver.precond"))
        precond = pa.getValue();

      StringRef solver;
      if (precond == "amg")
        solver = "cg";
      else if (precond == "ilu" || precond == "jacobi")
        solver = "gmres";
      else
        solver = "gmres"; // safe fallback

      op->setAttr("tessera_solver.solver", StringAttr::get(ctx, solver));
      // Mark specialization complete so downstream passes can rely on it.
      op->setAttr("tessera_solver.specialized", UnitAttr::get(ctx));
    });
  }
};

} // namespace

namespace tessera {
namespace passes {
std::unique_ptr<Pass> createSparseSolverSpecializePass() {
  return std::make_unique<SparseSolverSpecializePass>();
}
} // namespace passes
} // namespace tessera
