//===- IterativeRefinement.cpp — wrap solve regions with IR loop ---------*- C++ -*-===//
//
// Wraps linalg-backed solve regions with iterative refinement:
//
//   for iter in 0..max_iter:
//     r = b - A * x          (fp32 residual)
//     if ||r|| < tol: break
//     dx = solve(A, r)       (fp16 correction solve)
//     x  = x + dx            (fp32 accumulate)
//
// This pass attaches IR attrs to mark the refinement loop parameters.
// The actual loop region is materialized by a later canonicalization.
//
//===----------------------------------------------------------------------===//

#include "tessera/Solvers/LinalgPasses.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Pass/Pass.h"

namespace tessera {
namespace solver {

struct IterativeRefinementPass
    : public mlir::PassWrapper<IterativeRefinementPass,
                               mlir::OperationPass<mlir::func::FuncOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(IterativeRefinementPass)

  mlir::StringRef getArgument() const final {
    return "tessera-linalg-iterative-refinement";
  }
  mlir::StringRef getDescription() const final {
    return "Wrap linalg-backed solver regions with iterative refinement "
           "(residual + correction loop)";
  }

  llvm::cl::opt<int> maxIter{
      "ir-max-iter",
      llvm::cl::desc("Max iterative-refinement iterations"),
      llvm::cl::init(3)};

  llvm::cl::opt<double> tolerance{
      "ir-tol",
      llvm::cl::desc("Convergence tolerance for iterative refinement"),
      llvm::cl::init(1e-6)};

  void runOnOperation() override {
    mlir::func::FuncOp fn = getOperation();
    mlir::MLIRContext *ctx = fn.getContext();

    // Tag each solver op in this function with the refinement loop params.
    fn.walk([&](mlir::Operation *op) {
      mlir::StringRef opName = op->getName().getStringRef();

      bool isSolverOp =
          opName.contains("solve") || opName.contains("factor") ||
          opName == "tessera_solver.implicit" ||
          opName == "tessera_solver.linear_solve";

      if (!isSolverOp)
        return;

      // Attach iterative-refinement loop parameters.
      op->setAttr(
          "tessera_solver.ir_max_iter",
          mlir::IntegerAttr::get(mlir::IntegerType::get(ctx, 64), maxIter));
      op->setAttr(
          "tessera_solver.ir_tol",
          mlir::FloatAttr::get(mlir::Float64Type::get(ctx), tolerance));

      // Mark residual dtype (fp32) and correction dtype (fp16).
      op->setAttr("tessera_solver.residual_dtype",
                  mlir::StringAttr::get(ctx, "f32"));
      op->setAttr("tessera_solver.correction_dtype",
                  mlir::StringAttr::get(ctx, "f16"));

      // Mark the refinement loop pattern.
      op->setAttr("tessera_solver.ir_pattern",
                  mlir::StringAttr::get(ctx,
                                        "residual_correction_accumulate"));
      op->setAttr("tessera_solver.ir_annotated", mlir::UnitAttr::get(ctx));
    });
  }
};

std::unique_ptr<mlir::Pass> createIterativeRefinementPass() {
  return std::make_unique<IterativeRefinementPass>();
}

} // namespace solver
} // namespace tessera
