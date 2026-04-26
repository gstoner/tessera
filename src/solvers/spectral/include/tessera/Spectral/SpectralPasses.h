
//===- SpectralPasses.h ----------------------------------------*- C++ -*-===//
#pragma once
#include "mlir/Pass/Pass.h"
#include "SpectralPasses.h.inc" // generated pass decls

namespace tessera {
std::unique_ptr<mlir::Pass> createLegalizeSpectralPass();
std::unique_ptr<mlir::Pass> createSpectralMXPPass();
std::unique_ptr<mlir::Pass> createSpectralTransposePlanPass();
std::unique_ptr<mlir::Pass> createSpectralAutotunePass();
std::unique_ptr<mlir::Pass> createLowerSpectralToTargetIRPass();
std::unique_ptr<mlir::Pass> createSpectralDistributedPass();
} // namespace tessera
