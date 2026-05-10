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
//   - matmul -> softmax -> matmul fusion (rank-2, N/P <= 256)     [Phase 8.4.5]
//   - matmul -> softmax fusion           (rank-2, N <= 256)        [Phase 8.4.3]
//   - matmul     (rank-2) -> MPSMatrixMultiplication              [Phase 8.3]
//   - rope       (rank-2) -> custom MSL kernel                    [Phase 8.4]
//   - flash_attn (rank-3) -> custom MSL kernel                    [Phase 8.4.1]
//   - softmax    (rank-2) -> custom MSL kernel                    [Phase 8.4.2]
//   - gelu       (rank-2) -> custom MSL kernel                    [Phase 8.4.2]
//
// Pass ordering matters: longer chain patterns run BEFORE shorter ones so
// that the longest applicable fusion wins. Per-op passes are last so they
// only fire on what's left over.
//
// Distinct from the artifact-only `tessera-lower-to-apple_gpu` pipeline
// above.
PassPipelineRegistration<> gAppleGPURuntimePipeline(
    "tessera-lower-to-apple_gpu-runtime",
    "Lower supported tessera ops (matmul->softmax->matmul + matmul->softmax "
    "fusions, matmul, rope, flash_attn, softmax, gelu) to Apple GPU runtime "
    "calls (MPS + custom MSL kernels)",
    [](OpPassManager &pm) {
      // Phase 8.4.8 (Stage 3) — SwiGLU is an even longer chain (4-op) and
      // already arrives as a single `tessera.swiglu_fused` op courtesy of
      // the Stage 2b Schedule IR fusion recognizer. Run it first so we
      // capture the whole MLP block before per-op patterns can claim
      // pieces of it. No-match (e.g. H > 256) falls through cleanly.
      pm.addPass(createLowerSwigluFusionToAppleGPUPass());
      // Longest in-pipeline 3-op chain.
      pm.addPass(createLowerMatmulSoftmaxMatmulFusionToAppleGPUPass());
      // 2-op fusions next. Order within this group doesn't matter — each
      // matches a different post-matmul op so they don't compete.
      pm.addPass(createLowerMatmulSoftmaxFusionToAppleGPUPass());
      pm.addPass(createLowerMatmulGeluFusionToAppleGPUPass());
      pm.addPass(createLowerMatmulRMSNormFusionToAppleGPUPass());
      pm.addPass(createLowerMatmulToAppleGPUPass());
      pm.addPass(createLowerRopeToAppleGPUPass());
      pm.addPass(createLowerFlashAttnToAppleGPUPass());
      pm.addPass(createLowerSoftmaxToAppleGPUPass());
      pm.addPass(createLowerGeluToAppleGPUPass());
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
