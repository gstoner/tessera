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
// 2026-09-16 — the first EBM lowering: energy / inner_step / langevin_step
// (euclidean) to arith/linalg over the compiler-derived gradient.
std::unique_ptr<mlir::Pass> createEBMLowerLangevinPass();

}  // namespace tessera
