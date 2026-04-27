//===- MixedPrecision.cpp - Insert quant/dequant & set policies ---------*- C++ -*-===//
#include "tessera/Solvers/LinalgPasses.h"
#include "SolversPasses.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include <optional>

namespace tessera { namespace solver {

struct MixedPrecisionPass : public mlir::PassWrapper<MixedPrecisionPass, mlir::OperationPass<mlir::ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(MixedPrecisionPass)

  llvm::StringRef getArgument() const final {
    return "tessera-linalg-mixed-precision";
  }
  llvm::StringRef getDescription() const final {
    return "Attach mixed-precision policies for linalg-backed solver regions";
  }

  void runOnOperation() override {
    // Walk solver ops, attach default precision policies where missing.
    // Insert quantize/dequantize stubs (tessera.quantize/dequantize) around factor/solve boundaries.
    getOperation().walk([&](mlir::Operation *op){
      (void)op; // TODO: implement
    });
  }
};

std::unique_ptr<mlir::Pass> createMixedPrecisionPass() {
  return std::make_unique<MixedPrecisionPass>();
}

void buildTesseraLinalgSolverPipeline(mlir::OpPassManager &pm) {
  // Linalg is the parent solver surface: attach precision policy first, run the
  // canonical solver stack, then close with function-level refinement.
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
      "Parent linalg solver pipeline: precision policy + canonical solver stack + refinement",
      [](mlir::OpPassManager &pm) { buildTesseraLinalgSolverPipeline(pm); });
}

}} // namespace tessera::solver
