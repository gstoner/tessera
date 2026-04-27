
//===- Passes.cpp - Tessera CPX pass pipeline registration ----------------===//
//
// Registers all four CPX compiler passes and the canonical named pipeline
// `tessera-cpx-pipeline`.
//
// Pipeline order (CPX lowering spec §3):
//   1. tessera-fuse-video-ingest      — fuse video.decode → attn.prefill_fused
//   2. tessera-partition-longcontext  — split CPX/Rubin and insert kv.export/import
//   3. tessera-vectorize-nvfp4        — legalize matmul tiles to NVFP4 MMA forms
//   4. tessera-lower-kv-transport     — lower kv.{export,import,prefetch} to runtime calls
//
//===-----------------------------------------------------------------------===//

#include "tessera/Transforms/Passes.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Pass/PassRegistry.h"

namespace tessera {

void registerCPXPassPipelines() {
  // ── Individual passes ────────────────────────────────────────────────────
  // Each pass self-registers via PassRegistration<> in its own .cpp file.
  // Nothing to do here for individual pass registration.

  // ── Named pipeline: tessera-cpx-pipeline ─────────────────────────────────
  // Registers a pipeline alias that can be invoked as:
  //   tessera-cpx-opt --pass-pipeline='tessera-cpx-pipeline'
  mlir::PassPipelineRegistration<>(
    "tessera-cpx-pipeline",
    "Full Tessera CPX lowering pipeline: video-ingest-fuse → partition-longcontext "
    "→ vectorize-nvfp4 → lower-kv-transport",
    [](mlir::OpPassManager &pm) {
      pm.addPass(createFuseVideoIngestPass());
      pm.addPass(createPartitionLongContextPass());
      pm.addPass(createNVFP4VectorizePass());
      pm.addPass(createLowerKVTransportPass());
    }
  );

  // ── Context-only sub-pipeline ─────────────────────────────────────────────
  // For CPX-side only (no decode dispatch): skip video-ingest-fuse, run the
  // remaining three passes.
  mlir::PassPipelineRegistration<>(
    "tessera-cpx-context-pipeline",
    "CPX context-only pipeline: partition-longcontext → vectorize-nvfp4 → lower-kv-transport",
    [](mlir::OpPassManager &pm) {
      pm.addPass(createPartitionLongContextPass());
      pm.addPass(createNVFP4VectorizePass());
      pm.addPass(createLowerKVTransportPass());
    }
  );
}

} // namespace tessera
