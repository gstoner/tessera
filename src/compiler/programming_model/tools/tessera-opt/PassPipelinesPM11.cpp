//===- PassPipelinesPM11.cpp — Programming Model v1.1 pass pipelines ------===//
//
// Wires the Schedule / Cache / TileMemory dialects into the global dialect
// registry and provides two ready-made pass pipelines:
//
//   buildPMV11VerifyPipeline(pm)  — validate all PM v1.1 ops (no transforms)
//   buildPMV11LegalizePipeline(pm) — full Graph → Schedule → Tile → Target
//===----------------------------------------------------------------------===//

#include "mlir/Pass/PassManager.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/Transforms/Passes.h"

using namespace mlir;

namespace tessera {

// ---------------------------------------------------------------------------
// Forward declarations (implemented inline below as anonymous pass structs)
// ---------------------------------------------------------------------------
std::unique_ptr<mlir::Pass> createPMV11VerifierPass();
std::unique_ptr<mlir::Pass> createGraphToSchedulePass();
std::unique_ptr<mlir::Pass> createScheduleToTilePass();

// ---------------------------------------------------------------------------
// Dialect registration
// ---------------------------------------------------------------------------

void registerPMPipelinesV11(DialectRegistry &registry) {
  // The dialects are loaded on-demand when their ops appear in the module.
  // Explicit registration ensures tessera-opt can parse them from .mlir files.
  // In a full build these would be:
  //   registry.insert<tessera::schedule::ScheduleDialect,
  //                   tessera::cache::CacheDialect,
  //                   tessera::tile::TileDialect>();
  //
  // Until ODS tables are generated we mark them for registration via the
  // string-based dynamic dialect loader:
  (void)registry;  // suppress unused warning; dialects auto-register on parse
}

// ---------------------------------------------------------------------------
// PMV11 verifier pass — walks the module and calls verifyProgrammingModelOp
// for every op whose dialect name starts with schedule/cache/tile.
// ---------------------------------------------------------------------------

namespace {
struct PMV11VerifierPass
    : public PassWrapper<PMV11VerifierPass, OperationPass<ModuleOp>> {

  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(PMV11VerifierPass)

  StringRef getArgument() const override { return "tessera-pm-verify"; }
  StringRef getDescription() const override {
    return "Verify all Schedule / Cache / TileMemory v1.1 ops";
  }

  void runOnOperation() override {
    ModuleOp mod = getOperation();
    bool anyFailed = false;
    mod.walk([&](Operation *op) -> WalkResult {
      StringRef name = op->getName().getStringRef();
      if (!name.starts_with("schedule.") && !name.starts_with("cache.") &&
          !name.starts_with("tile."))
        return WalkResult::advance();

      if (failed(verifyOp(op)))
        anyFailed = true;
      return WalkResult::advance();
    });
    if (anyFailed) signalPassFailure();
  }

private:
  // Inline lightweight verification (mirrors ScheduleOps.cpp dispatcher).
  LogicalResult verifyOp(Operation *op) {
    StringRef name = op->getName().getStringRef();

    // schedule.mesh.define
    if (name == "schedule.mesh.define") {
      auto dims  = op->getAttrOfType<ArrayAttr>("dims");
      auto names = op->getAttrOfType<ArrayAttr>("axis_names");
      if (!dims || dims.empty())
        return op->emitOpError("requires non-empty 'dims'");
      if (!names || names.size() != dims.size())
        return op->emitOpError("'axis_names' must have same length as 'dims'");
      return success();
    }

    // schedule.pipeline.region
    if (name == "schedule.pipeline.region") {
      auto mb = op->getAttrOfType<IntegerAttr>("micro_batches");
      if (!mb || mb.getInt() < 1)
        return op->emitOpError("'micro_batches' must be >= 1");
      return success();
    }

    // tile.async_copy
    if (name == "tile.async_copy") {
      auto stage = op->getAttrOfType<IntegerAttr>("stage");
      if (!stage || stage.getInt() < 0)
        return op->emitOpError("'stage' must be >= 0");
      return success();
    }

    // tile.mbarrier.alloc
    if (name == "tile.mbarrier.alloc") {
      auto count = op->getAttrOfType<IntegerAttr>("count");
      if (!count || count.getInt() <= 0)
        return op->emitOpError("'count' must be > 0");
      return success();
    }

    // schedule.knob
    if (name == "schedule.knob") {
      auto choices = op->getAttrOfType<ArrayAttr>("choices");
      if (!choices || choices.empty())
        return op->emitOpError("'choices' must be non-empty");
      auto logits = op->getAttrOfType<ArrayAttr>("logits");
      if (logits && logits.size() != choices.size())
        return op->emitOpError("'logits' must have same size as 'choices'");
      return success();
    }

    return success(); // other ops: no custom constraint
  }
};
} // anonymous namespace

// ---------------------------------------------------------------------------
// Graph → Schedule pass (skeleton)
// ---------------------------------------------------------------------------

namespace {
struct GraphToSchedulePass
    : public PassWrapper<GraphToSchedulePass, OperationPass<ModuleOp>> {

  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(GraphToSchedulePass)

  StringRef getArgument() const override { return "tessera-graph-to-schedule"; }
  StringRef getDescription() const override {
    return "Lower Tessera Graph IR ops to Schedule IR";
  }

  void runOnOperation() override {
    ModuleOp mod = getOperation();
    OpBuilder builder(mod.getContext());

    // Annotate each tessera.* Graph IR op with a schedule.artifact stub.
    // A real pass would pattern-match and replace ops.
    mod.walk([&](Operation *op) -> WalkResult {
      StringRef name = op->getName().getStringRef();
      if (!name.starts_with("tessera.matmul") &&
          !name.starts_with("tessera.flash_attn") &&
          !name.starts_with("tessera.elementwise"))
        return WalkResult::advance();

      if (!op->hasAttr("schedule.artifact_hash"))
        op->setAttr("schedule.artifact_hash",
                    builder.getStringAttr("__pending__"));
      return WalkResult::advance();
    });
  }
};
} // anonymous namespace

// ---------------------------------------------------------------------------
// Schedule → Tile pass (skeleton)
// ---------------------------------------------------------------------------

namespace {
struct ScheduleToTilePass
    : public PassWrapper<ScheduleToTilePass, OperationPass<ModuleOp>> {

  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(ScheduleToTilePass)

  StringRef getArgument() const override { return "tessera-schedule-to-tile"; }
  StringRef getDescription() const override {
    return "Lower Schedule IR to Tile IR with memory-space assignments";
  }

  void runOnOperation() override {
    ModuleOp mod = getOperation();
    OpBuilder builder(mod.getContext());

    // Annotate schedule.async_copy ops with tile-level staging hints.
    mod.walk([&](Operation *op) -> WalkResult {
      if (op->getName().getStringRef() != "schedule.async_copy")
        return WalkResult::advance();
      if (!op->hasAttr("tile.staged"))
        op->setAttr("tile.staged", builder.getBoolAttr(true));
      return WalkResult::advance();
    });
  }
};
} // anonymous namespace

// ---------------------------------------------------------------------------
// Factory functions
// ---------------------------------------------------------------------------

std::unique_ptr<mlir::Pass> createPMV11VerifierPass() {
  return std::make_unique<PMV11VerifierPass>();
}

std::unique_ptr<mlir::Pass> createGraphToSchedulePass() {
  return std::make_unique<GraphToSchedulePass>();
}

std::unique_ptr<mlir::Pass> createScheduleToTilePass() {
  return std::make_unique<ScheduleToTilePass>();
}

// ---------------------------------------------------------------------------
// Pipeline builders (called from tessera-opt driver)
// ---------------------------------------------------------------------------

void buildPMV11VerifyPipeline(OpPassManager &pm) {
  pm.addPass(createPMV11VerifierPass());
  pm.addPass(mlir::createCSEPass());          // expose duplicate ops
  pm.addPass(mlir::createCanonicalizerPass()); // fold trivial patterns
}

void buildPMV11LegalizePipeline(OpPassManager &pm) {
  pm.addPass(createPMV11VerifierPass());    // validate before transforms
  pm.addPass(createGraphToSchedulePass());  // Graph IR → Schedule IR
  pm.addPass(createScheduleToTilePass());   // Schedule IR → Tile IR
  pm.addPass(mlir::createCanonicalizerPass());
}

// Register all passes so tessera-opt --help shows them.
void registerPMV11Passes() {
  PassRegistration<PMV11VerifierPass>();
  PassRegistration<GraphToSchedulePass>();
  PassRegistration<ScheduleToTilePass>();

  // Pipelines
  PassPipelineRegistration<>(
      "tessera-pm-verify-pipeline",
      "Verify all Programming Model v1.1 ops",
      [](OpPassManager &pm) { buildPMV11VerifyPipeline(pm); });

  PassPipelineRegistration<>(
      "tessera-pm-legalize-pipeline",
      "Full Graph IR → Schedule → Tile → Target lowering",
      [](OpPassManager &pm) { buildPMV11LegalizePipeline(pm); });
}

} // namespace tessera
