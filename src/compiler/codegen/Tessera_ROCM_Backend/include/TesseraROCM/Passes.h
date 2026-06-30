#pragma once
#include "mlir/Pass/Pass.h"
namespace mlir {
class Pass;
class OpPassManager;
class DialectRegistry;
namespace tessera_rocm {
std::unique_ptr<mlir::Pass> createLowerTileToROCMImpl();
std::unique_ptr<mlir::Pass> createLowerTileToROCMPass();
std::unique_ptr<mlir::Pass> createROCMWaveLdsPipelinePass();
std::unique_ptr<mlir::Pass> createROCMWaveLdsLegalityPass();
std::unique_ptr<mlir::Pass> createLowerTesseraToROCDLImpl();
std::unique_ptr<mlir::Pass> createLowerTesseraTargetToROCDLPass();
std::unique_ptr<mlir::Pass> createLowerKernelABIPass();
std::unique_ptr<mlir::Pass> createGenerateWMMAGemmKernelPass();
std::unique_ptr<mlir::Pass> createGenerateWMMAFlashAttnKernelPass();
std::unique_ptr<mlir::Pass> createGenerateWMMAFlashAttnBwdKernelPass();
std::unique_ptr<mlir::Pass> createGenerateWMMALinearAttnKernelPass();
std::unique_ptr<mlir::Pass> createGenerateROCMActivationKernelPass();
std::unique_ptr<mlir::Pass> createGenerateROCMSiluMulKernelPass();
std::unique_ptr<mlir::Pass> createGenerateROCMPointwiseLossKernelPass();
std::unique_ptr<mlir::Pass> createGenerateROCMBinaryLossKernelPass();
std::unique_ptr<mlir::Pass> createGenerateROCMPolicyLossKernelPass();
std::unique_ptr<mlir::Pass> createGenerateROCMFpQuantKernelPass();
std::unique_ptr<mlir::Pass> createGenerateROCMDftKernelPass();
std::unique_ptr<mlir::Pass> createGenerateROCMSpmmKernelPass();
std::unique_ptr<mlir::Pass> createGenerateROCMSddmmKernelPass();
std::unique_ptr<mlir::Pass> createGenerateROCMSelectiveSsmKernelPass();
std::unique_ptr<mlir::Pass> createGenerateROCMCholeskyKernelPass();
std::unique_ptr<mlir::Pass> createGenerateROCMTriSolveKernelPass();
std::unique_ptr<mlir::Pass> createGenerateROCMLuKernelPass();
std::unique_ptr<mlir::Pass> createGenerateROCMQrKernelPass();
std::unique_ptr<mlir::Pass> createGenerateROCMSvdKernelPass();
std::unique_ptr<mlir::Pass> createGenerateROCMOptimizerKernelPass();
std::unique_ptr<mlir::Pass> createGenerateROCMPredicateKernelPass();
std::unique_ptr<mlir::Pass> createGenerateROCMMoeKernelPass();
std::unique_ptr<mlir::Pass> createGenerateROCMAlibiKernelPass();
std::unique_ptr<mlir::Pass> createGenerateROCMDeltaNetKernelPass();
std::unique_ptr<mlir::Pass> createGenerateROCMRopeKernelPass();
std::unique_ptr<mlir::Pass> createGenerateROCMSoftmaxKernelPass();
std::unique_ptr<mlir::Pass> createGenerateROCMNormKernelPass();
std::unique_ptr<mlir::Pass> createGenerateROCMReduceKernelPass();
std::unique_ptr<mlir::Pass> createGenerateROCMArgReduceKernelPass();
std::unique_ptr<mlir::Pass> createGenerateROCMScanKernelPass();
std::unique_ptr<mlir::Pass> createGenerateROCMUnaryKernelPass();
// CF4b — elementwise-body tessera.control_for → one device control-flow kernel.
std::unique_ptr<mlir::Pass> createGenerateROCMControlForKernelPass();
// CF4d-1 — GEMV-recurrence control_for → cooperative-workgroup kernel.
std::unique_ptr<mlir::Pass> createGenerateROCMControlForGemvKernelPass();
// CF4d-2 — norm-in-loop control_for → cooperative-workgroup kernel.
std::unique_ptr<mlir::Pass> createGenerateROCMControlForNormKernelPass();
// CF4d-3 — single-tile WMMA matmul recurrence control_for → one-wave kernel.
std::unique_ptr<mlir::Pass> createGenerateROCMControlForWmmaKernelPass();
// CF4d-4 — multi-tile WMMA matmul recurrence control_for → one-workgroup
// (MT*KT waves) kernel; workgroup barrier handoff, no grid.sync.
std::unique_ptr<mlir::Pass> createGenerateROCMControlForWmmaTileKernelPass();
// CF4e-1 — elementwise-body control_scan (per-step xs in, stacked ys out) →
// one per-thread device kernel.
std::unique_ptr<mlir::Pass> createGenerateROCMControlScanKernelPass();
// CF4d-if — cross-element control_if (rmsnorm/layer_norm branches) →
// cooperative-workgroup kernel; uniform flag selects the norm.
std::unique_ptr<mlir::Pass> createGenerateROCMControlIfNormKernelPass();
// CF4e-2 — linear SSM scan (h_t = h_{t-1}@W + x_t; GEMV body + W capture +
// per-step xs) → cooperative-workgroup kernel.
std::unique_ptr<mlir::Pass> createGenerateROCMControlScanGemvKernelPass();
// CF4e-3 — nonlinear RNN-cell scan (h_t = tanh(h@W + x@U + b); two GEMV
// captures + bias + tanh + per-step xs) → cooperative-workgroup kernel.
std::unique_ptr<mlir::Pass> createGenerateROCMControlScanRnnKernelPass();
std::unique_ptr<mlir::Pass> createGenerateROCMBinaryKernelPass();
std::unique_ptr<mlir::Pass> createGenerateROCMCompareKernelPass();
std::unique_ptr<mlir::Pass> createGenerateROCMLogicalKernelPass();
std::unique_ptr<mlir::Pass> createGenerateROCMBitwiseKernelPass();
std::unique_ptr<mlir::Pass> createGenerateROCMWhereKernelPass();
std::unique_ptr<mlir::Pass> createGenerateROCMPhiloxKernelPass();
std::unique_ptr<mlir::Pass> createGenerateROCMGatherKernelPass();
std::unique_ptr<mlir::Pass> createGenerateROCMSortKernelPass();
std::unique_ptr<mlir::Pass> createGenerateROCMScatterKernelPass();
std::unique_ptr<mlir::Pass> createGenerateROCMEbmLangevinKernelPass();
std::unique_ptr<mlir::Pass> createGenerateROCMCliffordKernelPass();
std::unique_ptr<mlir::Pass> createLowerROCMAsyncCopyToLoopPass();
void buildTesseraROCMBackendPipeline(mlir::OpPassManager &pm);
void registerTesseraROCMPasses();
void registerTesseraROCMDialects(mlir::DialectRegistry &registry);
void registerTesseraROCMBackendPasses();
void registerTesseraROCMBackendDialects(mlir::DialectRegistry &registry);
}}
