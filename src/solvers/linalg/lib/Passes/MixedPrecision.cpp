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
#include "tessera/Dialect/Solver/SolverDialect.h"
#include "SolversPasses.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include <optional>

namespace tessera {
namespace solver {

enum class SolverRole { None, Factor, Solve, Residual };

// Precision role by op identity.
//
// TILE-LATENT-DEFECTS-2026-09-27: this used to classify by substring --
// "factor"/"lu"/"chol"/"qr" as factorizations, "solve"/"triangular"/... as
// solves, "residual" as residuals. Every `tessera_solver.*` name contains
// "solve" through its dialect prefix, and "lu" matches `tessera.gelu`,
// `tessera.relu`, `tessera.silu`, "factor" matches `tessera.adafactor`: the
// pass stamped an fp32 factorization policy on activations and an fp16 solve
// policy on the matrix-free IFT ops (`linear_solve`, `implicit`). The Graph ops
// are compared by exact name because this library does not link the Tessera
// Graph dialect.
static SolverRole classify(mlir::Operation *op) {
  if (mlir::isa<GetrfOp, PotrfOp>(op))
    return SolverRole::Factor;
  if (mlir::isa<TrsmOp, PotrsOp>(op))
    return SolverRole::Solve;
  if (mlir::isa<ResidualOp>(op))
    return SolverRole::Residual;
  mlir::StringRef name = op->getName().getStringRef();
  if (name == "tessera.cholesky" || name == "tessera.lu" ||
      name == "tessera.qr")
    return SolverRole::Factor;
  // Every op_catalog.py entry with lowering="linalg_solver" is a Solve;
  // tests/unit/test_linalg_solver_classifiers.py keeps this list in step
  // with the catalog.
  if (name == "tessera.solve" || name == "tessera.tri_solve" ||
      name == "tessera.cholesky_solve")
    return SolverRole::Solve;
  return SolverRole::None;
}

struct MixedPrecisionPass
    : public mlir::PassWrapper<MixedPrecisionPass,
                               mlir::OperationPass<mlir::ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(MixedPrecisionPass)

  MixedPrecisionPass() = default;
  MixedPrecisionPass(const MixedPrecisionPass &other)
      : mlir::PassWrapper<MixedPrecisionPass,
                          mlir::OperationPass<mlir::ModuleOp>>(other) {}

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
      SolverRole role = classify(op);
      if (role == SolverRole::None)
        return;
      bool isFactor = role == SolverRole::Factor;
      bool isSolve = role == SolverRole::Solve;
      bool isResidual = role == SolverRole::Residual;

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
  mlir::registerPass([]() { return createMixedPrecisionPass(); });
  mlir::registerPass([]() { return createIterativeRefinementPass(); });
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
