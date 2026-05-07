//===- Passes.h - Tessera Apple Silicon Backend Passes -------*- C++ -*-===//
//
// Pass + pipeline registration for the Apple Silicon Target IR backend.
//
//===----------------------------------------------------------------------===//

#ifndef TESSERA_TARGET_APPLE_PASSES_H
#define TESSERA_TARGET_APPLE_PASSES_H

#include <memory>

namespace mlir {
class DialectRegistry;
class Pass;
} // namespace mlir

namespace tessera {
namespace apple {

/// Tile IR → Apple CPU Target IR (Accelerate / vecLib / BNNS artifacts).
std::unique_ptr<::mlir::Pass> createLowerTileToAppleCPUPass();

/// Tile IR → Apple GPU Target IR (Metal / MPS artifacts).
std::unique_ptr<::mlir::Pass> createLowerTileToAppleGPUPass();

/// tessera.matmul (rank-2, f32) → func.call into the Apple CPU runtime
/// shim. Phase 8.2 — the executable counterpart to the artifact-only
/// `tessera-lower-to-apple_cpu` pipeline.
std::unique_ptr<::mlir::Pass> createLowerMatmulToAppleCPUPass();

/// tessera.matmul (rank-2, f32) → func.call into the Apple GPU runtime
/// shim (MTLDevice + MPSMatrixMultiplication). Phase 8.3 — the executable
/// counterpart to the artifact-only `tessera-lower-to-apple_gpu` pipeline.
std::unique_ptr<::mlir::Pass> createLowerMatmulToAppleGPUPass();

/// tessera.rope (rank-2, f32, x.shape == theta.shape) → func.call into the
/// Apple GPU runtime shim's custom-MSL rope kernel. Phase 8.4 — the first
/// concrete custom kernel emitted via the gpu.msl_kernel op contract.
std::unique_ptr<::mlir::Pass> createLowerRopeToAppleGPUPass();

/// tessera.flash_attn (rank-3, f32, head_dim <= 256) → func.call into the
/// Apple GPU runtime shim's custom-MSL flash-attention kernel. Phase 8.4.1.
std::unique_ptr<::mlir::Pass> createLowerFlashAttnToAppleGPUPass();

/// Register the Apple Silicon dialect into a DialectRegistry. Convenience
/// wrapper that forwards to registerAppleDialect from TesseraAppleDialect.h.
void registerTesseraAppleBackendDialects(::mlir::DialectRegistry &registry);

/// Register the Apple Silicon passes (and dialects) for use in tessera-opt.
/// This is the entry point tessera-opt should call.
void registerTesseraAppleBackendPasses(::mlir::DialectRegistry &registry);

/// Force-link the canonical pipelines:
///   - tessera-lower-to-apple_cpu
///   - tessera-lower-to-apple_gpu
/// Pipelines self-register at static init time; calling this from main()
/// keeps the linker from dropping the registration object in static builds.
void registerTesseraAppleBackendPipelines();

} // namespace apple
} // namespace tessera

#endif // TESSERA_TARGET_APPLE_PASSES_H
