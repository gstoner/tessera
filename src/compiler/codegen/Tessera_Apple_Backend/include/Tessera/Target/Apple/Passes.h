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
