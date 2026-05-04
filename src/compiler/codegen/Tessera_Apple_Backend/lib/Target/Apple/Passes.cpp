//===- Passes.cpp - Tessera Apple Silicon Backend Passes -----*- C++ -*-===//
//
// Registration glue for Apple Silicon backend dialects, passes, and the two
// canonical lowering pipelines:
//   - tessera-lower-to-apple_cpu
//   - tessera-lower-to-apple_gpu
//
//===----------------------------------------------------------------------===//

#include "Tessera/Target/Apple/Passes.h"

#include "Tessera/Target/Apple/TesseraAppleDialect.h"

#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Pass/PassRegistry.h"

using namespace ::mlir;

namespace tessera {
namespace apple {

void registerTesseraAppleBackendDialects(::mlir::DialectRegistry &registry) {
  registerAppleDialect(registry);
}

void registerTesseraAppleBackendPasses(::mlir::DialectRegistry &registry) {
  registerTesseraAppleBackendDialects(registry);
  registerTesseraAppleBackendPipelines();
}

//===----------------------------------------------------------------------===//
// Canonical pipeline registration
//
// Both pipelines lower Tile IR to a hardware-free artifact contract. CPU goes
// through Accelerate/vecLib/BNNS spellings; GPU goes through Metal/MPS
// spellings. Backwards-compatible with the Python text path in
// matmul_pipeline.py.
//===----------------------------------------------------------------------===//

namespace {

PassPipelineRegistration<> gAppleCPUPipeline(
    "tessera-lower-to-apple_cpu",
    "Lower Tessera Tile IR to Apple Silicon CPU Target IR (Accelerate)",
    [](OpPassManager &pm) {
      pm.addPass(createLowerTileToAppleCPUPass());
    });

PassPipelineRegistration<> gAppleGPUPipeline(
    "tessera-lower-to-apple_gpu",
    "Lower Tessera Tile IR to Apple Silicon GPU Target IR (Metal)",
    [](OpPassManager &pm) {
      pm.addPass(createLowerTileToAppleGPUPass());
    });

} // namespace

void registerTesseraAppleBackendPipelines() {
  // Touch the static registration objects so the linker keeps them.
  (void)&gAppleCPUPipeline;
  (void)&gAppleGPUPipeline;
}

} // namespace apple
} // namespace tessera
