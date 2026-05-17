//===- CliffordPasses.h ----------------------------------------*- C++ -*-===//
#pragma once
#include "mlir/Pass/Pass.h"
#include "CliffordPasses.h.inc"  // generated pass decls

namespace tessera {

// GA7 — annotation pass (real implementation).
std::unique_ptr<mlir::Pass> createCliffordAnnotateAlgebraPass();

// GA8 — lowering passes. v1 stubs; full implementations land with GA8.
std::unique_ptr<mlir::Pass> createCliffordExpandProductTablePass();
std::unique_ptr<mlir::Pass> createCliffordGradeFusionPass();
std::unique_ptr<mlir::Pass> createCliffordRotorSandwichFoldPass();

}  // namespace tessera
