
#include "tessera/Transforms/Passes.h"
#include "mlir/Pass/PassRegistry.h"
#include "mlir/Pass/PassManager.h"

namespace tessera {
void registerCPXPasses() {
  static ::mlir::PassRegistration<mlir::Pass>(createPartitionLongContextPass().release());
  static ::mlir::PassRegistration<mlir::Pass>(createLowerKVTransportPass().release());
  static ::mlir::PassRegistration<mlir::Pass>(createNVFP4VectorizePass().release());
  static ::mlir::PassRegistration<mlir::Pass>(createFuseVideoIngestPass().release());
}
void registerCPXPipeline() {
  ::mlir::PassPipelineRegistration<>(
    "tessera-cpx-pipeline",
    "Partition, lower KV (policy), NVFP4 vectorize, canonicalize, video fuse",
    [](mlir::OpPassManager &pm) {
      pm.addPass(createPartitionLongContextPass());
      pm.addPass(createLowerKVTransportPass());
      pm.addPass(createNVFP4VectorizePass());
      pm.addPass(::mlir::createCanonicalizerPass());
      pm.addPass(createFuseVideoIngestPass());
    });
}
} // namespace tessera
