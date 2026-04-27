#include "tessera/tpu/TesseraTPUPasses.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "stablehlo/dialect/StablehloOps.h"

namespace tessera {
void buildTesseraTPUBackendPipeline(mlir::OpPassManager &pm) {
  pm.addPass(createAnnotateShardingPass());
  pm.addPass(createLowerTesseraAttentionToStableHLOPass());
  pm.addPass(createLowerTesseraConvToStableHLOPass());
  pm.addPass(createLowerTesseraToStableHLOPass());
  pm.addPass(createExportShardyPass());
}

void registerTesseraTPUPasses() {
  ::mlir::PassRegistration<::mlir::Pass>(
      "tessera-lower-to-stablehlo", "Lower Tessera ops to StableHLO",
      []() { return createLowerTesseraToStableHLOPass(); });
  ::mlir::PassRegistration<::mlir::Pass>(
      "tessera-annotate-sharding", "Annotate TPU sharding metadata",
      []() { return createAnnotateShardingPass(); });
  ::mlir::PassRegistration<::mlir::Pass>(
      "tessera-lower-attention-to-stablehlo",
      "Lower Tessera FlashAttention to StableHLO",
      []() { return createLowerTesseraAttentionToStableHLOPass(); });
  ::mlir::PassRegistration<::mlir::Pass>(
      "tessera-lower-conv-to-stablehlo",
      "Lower Tessera convolution epilogues to StableHLO",
      []() { return createLowerTesseraConvToStableHLOPass(); });
  ::mlir::PassRegistration<::mlir::Pass>(
      "tessera-export-shardy", "Export Tessera sharding metadata to Shardy",
      []() { return createExportShardyPass(); });
  ::mlir::PassPipelineRegistration<> pipeline(
      "tessera-tpu-backend", "Lower Tessera Graph IR to TPU StableHLO",
      [](mlir::OpPassManager &pm) { buildTesseraTPUBackendPipeline(pm); });
}

void registerTesseraTPUBackendPasses() { registerTesseraTPUPasses(); }

void registerTesseraTPUBackendDialects(mlir::DialectRegistry &registry) {
  registry.insert<mlir::stablehlo::StablehloDialect>();
}
} // namespace tessera
