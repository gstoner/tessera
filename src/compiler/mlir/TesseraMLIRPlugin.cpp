//===- TesseraMLIRPlugin.cpp — Global dialect/pass registration entry ------===//
//
// Implements the central registration entry point declared in
// TesseraMLIRPlugin.h.  Call registerTesseraAll(registry) once from
// tessera-opt main() to wire every Tessera dialect and pass into MLIR.
//
// Build note:
//   This file must link against:
//     TesseraNeighborsDialect   — lib/Dialect/Neighbors/IR/TesseraNeighbors.cpp
//     TesseraNeighborsPasses    — lib/Dialect/Neighbors/Transforms/*.cpp
//     TesseraPMV11Passes        — programming_model/tools/tessera-opt/*.cpp
//     TesseraPMVerifiers        — programming_model/ir/ScheduleOps.cpp
//     TesseraTPUBackend         — codegen/Tessera_TPU_Backend/src/passes/*.cpp
//     MLIRPass, MLIRIR, etc.
//===----------------------------------------------------------------------===//

#include "Tessera/TesseraMLIRPlugin.h"

#include "mlir/IR/DialectRegistry.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Pass/PassRegistry.h"
#include "mlir/Transforms/Passes.h"

// ---------------------------------------------------------------------------
// Forward declarations for per-subsystem registration functions.
// Each subsystem provides its own registration header/function; we collect
// them here so callers have a single stable entry point.
// ---------------------------------------------------------------------------

// Neighbors dialect + passes
namespace tessera {
  void registerNeighborsDialect(mlir::DialectRegistry &registry);

  std::unique_ptr<mlir::Pass> createHaloInferPass();
  std::unique_ptr<mlir::Pass> createStencilLowerPass();
  std::unique_ptr<mlir::Pass> createPipelineOverlapPass();
  std::unique_ptr<mlir::Pass> createDynamicTopologyPass();
} // namespace tessera

// Programming Model v1.1 passes  (declared in PassPipelinesPM11.cpp)
namespace tessera {
  void registerPMV11Passes();
  void buildPMV11VerifyPipeline(mlir::OpPassManager &pm);
  void buildPMV11LegalizePipeline(mlir::OpPassManager &pm);
} // namespace tessera

// TPU backend passes  (declared in RegisterPasses.cpp)
namespace tessera {
  void registerTesseraTPUBackendPasses();
  void registerTesseraTPUBackendDialects(mlir::DialectRegistry &registry);
  void buildTesseraTPUBackendPipeline(mlir::OpPassManager &pm);
} // namespace tessera


namespace tessera {

// ===========================================================================
// registerTesseraDialects
// ===========================================================================

void registerTesseraDialects(mlir::DialectRegistry &registry) {
  // Neighbors dialect (tessera.neighbors.*)
  registerNeighborsDialect(registry);

  // TPU backend brings in stablehlo + any target dialects it needs
  registerTesseraTPUBackendDialects(registry);

  // The Queue / Attn dialects are Python-only at this stage; their MLIR ops
  // are represented as generic `tessera.queue.*` / `tessera.attn.*` strings
  // and round-trip through the unknown-op path until ODS tables exist.
  // When C++ ODS tables are generated, insert:
  //   registry.insert<tessera::queue::QueueDialect,
  //                   tessera::attn::AttnDialect>();
}

// ===========================================================================
// registerTesseraAllPasses
// ===========================================================================

void registerTesseraAllPasses() {
  // ---- Neighbors passes --------------------------------------------------
  mlir::PassRegistration<mlir::Pass>(
      "tessera-halo-infer",
      "Infer halo widths from stencil taps / neighbor.read deltas",
      []() -> std::unique_ptr<mlir::Pass> { return createHaloInferPass(); });

  mlir::PassRegistration<mlir::Pass>(
      "tessera-stencil-lower",
      "Lower tessera.neighbors.stencil.apply to pack/exchange/compute phases",
      []() -> std::unique_ptr<mlir::Pass> { return createStencilLowerPass(); });

  mlir::PassRegistration<mlir::Pass>(
      "tessera-pipeline-overlap",
      "Assign comm/compute stream IDs and double-buffer indices",
      []() -> std::unique_ptr<mlir::Pass> { return createPipelineOverlapPass(); });

  mlir::PassRegistration<mlir::Pass>(
      "tessera-dynamic-topology",
      "Annotate dynamic/adaptive topologies with fence and replan hooks",
      []() -> std::unique_ptr<mlir::Pass> { return createDynamicTopologyPass(); });

  // ---- Programming Model v1.1 passes -------------------------------------
  registerPMV11Passes();

  // ---- TPU backend passes -------------------------------------------------
  registerTesseraTPUBackendPasses();
}

// ===========================================================================
// registerTesseraAllPipelines
// ===========================================================================

void registerTesseraAllPipelines() {
  // Neighbors compilation pipeline
  mlir::PassPipelineRegistration<>(
      "tessera-neighbors-pipeline",
      "Halo inference + stencil lower + pipeline overlap",
      [](mlir::OpPassManager &pm) { buildNeighborsPipeline(pm); });

  // PM v1.1 verify pipeline (re-exported through the plugin layer)
  mlir::PassPipelineRegistration<>(
      "tessera-pm-verify-pipeline",
      "Verify all Schedule / Cache / TileMemory v1.1 ops",
      [](mlir::OpPassManager &pm) { buildPMVerifyPipeline(pm); });

  // PM v1.1 legalize pipeline
  mlir::PassPipelineRegistration<>(
      "tessera-pm-legalize-pipeline",
      "Full Graph IR → Schedule → Tile lowering",
      [](mlir::OpPassManager &pm) { buildPMLegalizePipeline(pm); });

  // TPU backend pipeline
  mlir::PassPipelineRegistration<>(
      "tessera-tpu-backend",
      "Lower Tessera Graph IR to TPU StableHLO + Shardy sharding",
      [](mlir::OpPassManager &pm) { buildTesseraTPUBackendPipeline(pm); });

  // Full end-to-end pipeline
  mlir::PassPipelineRegistration<>(
      "tessera-full-pipeline",
      "Neighbors + PM legalize + TPU backend end-to-end",
      [](mlir::OpPassManager &pm) { buildFullPipeline(pm); });
}

// ===========================================================================
// Per-layer pipeline builders
// ===========================================================================

void buildNeighborsPipeline(mlir::OpPassManager &pm) {
  pm.addPass(createHaloInferPass());
  pm.addPass(createStencilLowerPass());
  pm.addPass(createPipelineOverlapPass());
  pm.addPass(createDynamicTopologyPass());
  pm.addPass(mlir::createCanonicalizerPass());
}

void buildPMVerifyPipeline(mlir::OpPassManager &pm) {
  buildPMV11VerifyPipeline(pm);  // defined in PassPipelinesPM11.cpp
}

void buildPMLegalizePipeline(mlir::OpPassManager &pm) {
  buildPMV11LegalizePipeline(pm); // defined in PassPipelinesPM11.cpp
}

void buildFullPipeline(mlir::OpPassManager &pm) {
  // Phase 1: resolve neighbor stencil ops
  buildNeighborsPipeline(pm);

  // Phase 2: legalize PM v1.1 Graph IR → Schedule → Tile
  buildPMV11LegalizePipeline(pm);

  // Phase 3: lower to StableHLO + attach Shardy sharding annotations
  buildTesseraTPUBackendPipeline(pm);

  // Final cleanup
  pm.addPass(mlir::createCSEPass());
  pm.addPass(mlir::createCanonicalizerPass());
}

} // namespace tessera
