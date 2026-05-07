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

// Phase 8.2: executable lowering to the Apple CPU runtime shim
// (cblas_sgemm via Accelerate). Distinct from the artifact-only
// `tessera-lower-to-apple_cpu` pipeline above.
PassPipelineRegistration<> gAppleCPURuntimePipeline(
    "tessera-lower-to-apple_cpu-runtime",
    "Lower tessera.matmul (rank-2, f32) to Apple CPU runtime calls "
    "(cblas_sgemm)",
    [](OpPassManager &pm) {
      pm.addPass(createLowerMatmulToAppleCPUPass());
    });

// Phase 8.3 + 8.4: executable lowering to the Apple GPU runtime shim. The
// pipeline composes one or more lowering patterns:
//   - matmul     (rank-2, f32) -> MPSMatrixMultiplication        [Phase 8.3]
//   - rope       (rank-2, f32) -> custom MSL kernel              [Phase 8.4]
//   - flash_attn (rank-3, f32) -> custom MSL kernel              [Phase 8.4.1]
// Distinct from the artifact-only `tessera-lower-to-apple_gpu` pipeline
// above.
PassPipelineRegistration<> gAppleGPURuntimePipeline(
    "tessera-lower-to-apple_gpu-runtime",
    "Lower supported tessera ops (matmul, rope, flash_attn) to Apple GPU "
    "runtime calls (MPS + custom MSL kernels)",
    [](OpPassManager &pm) {
      pm.addPass(createLowerMatmulToAppleGPUPass());
      pm.addPass(createLowerRopeToAppleGPUPass());
      pm.addPass(createLowerFlashAttnToAppleGPUPass());
    });

} // namespace

void registerTesseraAppleBackendPipelines() {
  // Touch the static registration objects so the linker keeps them.
  (void)&gAppleCPUPipeline;
  (void)&gAppleGPUPipeline;
  (void)&gAppleCPURuntimePipeline;
  (void)&gAppleGPURuntimePipeline;
}

} // namespace apple
} // namespace tessera
