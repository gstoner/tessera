//===- MixedPrecision.cpp — insert quant/dequant at factor/solve --------*- C++ -*-===//
//
// Walks linalg-backed solver regions and inserts tessera.quantize /
// tessera.dequantize stubs around factor/solve op boundaries.
//
// Policy heuristic:
//   * Factor ops (lu_factor, chol_factor) run in fp32 for stability.
//   * Solve ops (triangular_solve, back_sub) run in fp16 for throughput.
//   * Residual compute runs in fp32.
//
// This pass attaches attrs to mark the desired precision; actual cast ops are
// emitted by a later canonicalization step.
//
//===----------------------------------------------------------------------===//

#include "tessera/Solvers/LinalgPasses.h"
#include "SolversPasses.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include <optional>

namespace tessera {
namespace solver {

struct MixedPrecisionPass
    : public mlir::PassWrapper<MixedPrecisionPass,
                               mlir::OperationPass<mlir::ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(MixedPrecisionPass)

  mlir::StringRef getArgument() const final {
    return "tessera-linalg-mixed-precision";
  }
  mlir::StringRef getDescription() const final {
    return "Attach mixed-precision policies and insert quant/dequant stubs "
           "for linalg-backed solver regions";
  }

  void runOnOperation() override {
    mlir::ModuleOp mod = getOperation();
    mlir::MLIRContext *ctx = mod.getContext();

    mod.walk([&](mlir::Operation *op) {
      mlir::StringRef opName = op->getName().getStringRef();

      // Factor ops: run at fp32 (stability-critical).
      bool isFactor = opName.contains("factor") || opName.contains("lu") ||
                      opName.contains("chol") || opName.contains("qr");

      // Solve ops: run at fp16 (throughput-critical).
      bool isSolve = opName.contains("solve") || opName.contains("back_sub") ||
                     opName.contains("fwd_sub") || opName.contains("triangular");

      // Residual ops: run at fp32 (accuracy-critical).
      bool isResidual = opName.contains("residual");

      if (!isFactor && !isSolve && !isResidual)
        return;

      if (isFactor || isResidual) {
        op->setAttr("tessera.compute_dtype", mlir::StringAttr::get(ctx, "f32"));
        op->setAttr("tessera.quant_before", mlir::UnitAttr::get(ctx));
        op->setAttr("tessera.dequant_after", mlir::UnitAttr::get(ctx));
      } else if (isSolve) {
        op->setAttr("tessera.compute_dtype", mlir::StringAttr::get(ctx, "f16"));
        // Cast inputs to f16 before the solve, back to f32 after.
        op->setAttr("tessera.quant_before", mlir::UnitAttr::get(ctx));
        op->setAttr("tessera.dequant_after", mlir::UnitAttr::get(ctx));
      }

      op->setAttr("tessera.mixed_precision_annotated", mlir::UnitAttr::get(ctx));
    });
  }
};

std::unique_ptr<mlir::Pass> createMixedPrecisionPass() {
  return std::make_unique<MixedPrecisionPass>();
}

void buildTesseraLinalgSolverPipeline(mlir::OpPassManager &pm) {
  pm.addPass(createMixedPrecisionPass());
  tessera::passes::buildTesseraSolverCorePipeline(pm);
  pm.addNestedPass<mlir::func::FuncOp>(createIterativeRefinementPass());
}

void registerTesseraLinalgSolverPasses() {
  mlir::PassRegistration<mlir::Pass> mixedPrecision(
      "tessera-linalg-mixed-precision",
      "Attach mixed-precision policies for linalg-backed solver regions",
      []() { return createMixedPrecisionPass(); });
  mlir::PassRegistration<mlir::Pass> iterativeRefinement(
      "tessera-linalg-iterative-refinement",
      "Wrap linalg-backed solver regions with iterative refinement",
      []() { return createIterativeRefinementPass(); });
}

void registerTesseraLinalgSolverPipeline() {
  registerTesseraLinalgSolverPasses();
  mlir::PassPipelineRegistration<> pipeline(
      "tessera-linalg-solver",
      "Parent linalg solver pipeline: precision policy + canonical solver "
      "stack + refinement",
      [](mlir::OpPassManager &pm) {
        buildTesseraLinalgSolverPipeline(pm);
      });
}

} // namespace solver
} // namespace tessera
