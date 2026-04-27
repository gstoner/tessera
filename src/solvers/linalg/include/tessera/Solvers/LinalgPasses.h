#pragma once

#include "mlir/Pass/Pass.h"

namespace mlir {
class OpPassManager;
} // namespace mlir

namespace tessera {
namespace solver {

std::unique_ptr<mlir::Pass> createMixedPrecisionPass();
std::unique_ptr<mlir::Pass> createIterativeRefinementPass();

void buildTesseraLinalgSolverPipeline(mlir::OpPassManager &pm);
void registerTesseraLinalgSolverPasses();
void registerTesseraLinalgSolverPipeline();

} // namespace solver
} // namespace tessera
