//===- CliffordPasses.h ----------------------------------------*- C++ -*-===//
#pragma once
#include "mlir/Pass/Pass.h"
#include "CliffordPasses.h.inc"  // generated pass decls

namespace tessera {

// GA7 — annotation pass (real implementation).
std::unique_ptr<mlir::Pass> createCliffordAnnotateAlgebraPass();

// GA8 — lowering passes. v1 stubs; full implementations land with GA8.
std::unique_ptr<mlir::Pass> createCliffordExpandProductTablePass();
// With `expandRotorSandwich`, rotor_sandwich lowers to gp(gp(R,x), reverse(R))
// instead of surviving as a fused-kernel marker (the MLIR/LLVM JIT lane).
std::unique_ptr<mlir::Pass> createCliffordExpandProductTablePass(bool expandRotorSandwich);
std::unique_ptr<mlir::Pass> createCliffordGradeFusionPass();
std::unique_ptr<mlir::Pass> createCliffordRotorSandwichFoldPass();

}  // namespace tessera
