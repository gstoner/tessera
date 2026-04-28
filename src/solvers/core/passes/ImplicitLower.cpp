//===- ImplicitLower.cpp — lower implicit ops to Newton loop -------------*- C++ -*-===//
//
// Lower tessera_solver.implicit ops to:
//   tessera_solver.residual       — evaluates R(x, u) = 0
//   tessera_solver.newton_step    — applies one Newton update: Δu = -J^{-1} R
//   tessera_solver.converge_check — tests ||R|| < tol
//
// The lowered pattern looks like:
//
//   %r = tessera_solver.residual(%x, %u) { ... }
//   %converged = tessera_solver.converge_check(%r) {tol = 1e-8}
//   if %converged: return %u
//   %du = tessera_solver.newton_step(%r, %J) { max_iter = 500 }
//
// This pass only emits attrs and structural markers; actual kernel selection
// happens in SparseSolverSpecialize.
//
//===----------------------------------------------------------------------===//

#include "SolversPasses.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Pass/Pass.h"

using namespace mlir;

namespace {

struct ImplicitLowerPass
    : PassWrapper<ImplicitLowerPass, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(ImplicitLowerPass)

  Option<int> maxIter{
      *this, "newton-max-iter",
      llvm::cl::desc("Max Newton iterations"),
      llvm::cl::init(500)};

  Option<double> tolerance{
      *this, "newton-tol",
      llvm::cl::desc("Newton convergence tolerance"),
      llvm::cl::init(1e-8)};

  StringRef getArgument() const final { return "tessera-implicit-lower"; }
  StringRef getDescription() const final {
    return "Lower tessera_solver.implicit ops to residual + Newton loop";
  }

  void runOnOperation() override {
    ModuleOp mod = getOperation();
    MLIRContext *ctx = mod.getContext();

    mod.walk([&](Operation *op) {
      if (op->getName().getStringRef() != "tessera_solver.implicit")
        return;

      // Require autodiff annotation from NewtonAutodiffPass.
      if (!op->hasAttr("tessera_solver.autodiff_ready") &&
          !op->hasAttr("tessera_solver.vjp")) {
        op->emitWarning("tessera_solver.implicit op not autodiff-annotated; "
                        "run tessera-newton-autodiff first");
      }

      // Attach lowering descriptor.
      op->setAttr("tessera_solver.lowered_to", StringAttr::get(ctx, "newton"));
      op->setAttr("tessera_solver.newton_max_iter",
                  IntegerAttr::get(IntegerType::get(ctx, 64), maxIter));
      op->setAttr("tessera_solver.newton_tol",
                  FloatAttr::get(Float64Type::get(ctx), tolerance));

      // Decompose into residual + newton_step + converge_check.
      // The actual ops are emitted symbolically via attrs so that a later
      // canonicalization pass can materialize them in the correct region.
      op->setAttr("tessera_solver.residual_op",
                  StringAttr::get(ctx, "tessera_solver.residual"));
      op->setAttr("tessera_solver.newton_step_op",
                  StringAttr::get(ctx, "tessera_solver.newton_step"));
      op->setAttr("tessera_solver.converge_check_op",
                  StringAttr::get(ctx, "tessera_solver.converge_check"));

      // Mark as lowered.
      op->setAttr("tessera_solver.implicit_lowered", UnitAttr::get(ctx));
    });
  }
};

} // namespace

namespace tessera {
namespace passes {
std::unique_ptr<Pass> createImplicitLowerPass() {
  return std::make_unique<ImplicitLowerPass>();
}
} // namespace passes
} // namespace tessera
