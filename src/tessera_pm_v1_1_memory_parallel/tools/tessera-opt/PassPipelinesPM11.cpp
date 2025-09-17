//===- PassPipelinesPM11.cpp ---------------------------------------------*- C++ -*-===//
#include "mlir/Pass/PassManager.h"
#include "mlir/IR/DialectRegistry.h"

using namespace mlir;

namespace tessera {

void registerPMPipelinesV11(DialectRegistry &registry) {
  // TODO: add real dialects here
  // registry.insert<tessera::tile::TileDialect, tessera::schedule::ScheduleDialect, ...>();
}

void buildPMV11VerifyPipeline(OpPassManager &pm) {
  // TODO: add real verifier passes when available
  // pm.addPass(createTileVerifierPass());
  // pm.addPass(createScheduleVerifierPass());
}

void buildPMV11LegalizePipeline(OpPassManager &pm) {
  // TODO: graph -> schedule -> tile -> target
  // pm.addPass(createGraphToSchedulePass());
  // pm.addPass(createTileToTargetLegalizePass());
}

} // namespace tessera
