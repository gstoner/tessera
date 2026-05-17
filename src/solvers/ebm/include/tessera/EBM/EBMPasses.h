//===- EBMPasses.h ----------------------------------------------*- C++ -*-===//
#pragma once
#include "mlir/Pass/Pass.h"
#include "EBMPasses.h.inc"

namespace tessera {

// EBM5 — canonicalization (real).
std::unique_ptr<mlir::Pass> createEBMCanonicalizePass();

// EBM6 — fusion / checkpoint / pipeline passes. v1 stubs.
std::unique_ptr<mlir::Pass> createEBMFuseEnergyGradPass();
std::unique_ptr<mlir::Pass> createEBMCheckpointInnerLoopPass();
std::unique_ptr<mlir::Pass> createEBMPipelineCandidatesPass();

}  // namespace tessera
