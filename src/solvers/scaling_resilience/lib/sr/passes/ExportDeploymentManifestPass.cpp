#include "tessera/sr/Passes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/Attributes.h"
#include <fstream>

using namespace mlir;

namespace {
struct ExportDeploymentManifestPass : public PassWrapper<ExportDeploymentManifestPass, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(ExportDeploymentManifestPass)

  ExportDeploymentManifestPass() = default;
  ExportDeploymentManifestPass(const ExportDeploymentManifestPass& other)
      : PassWrapper<ExportDeploymentManifestPass, OperationPass<ModuleOp>>(other) {}

  StringRef getArgument() const final { return "tessera-export-deployment-manifest"; }
  StringRef getDescription() const final { return "Emit deployment manifest JSON capturing mesh/collectives/optimizer/ckpt"; }

  void runOnOperation() override {
    ModuleOp mod = getOperation();
    int64_t meshOps = 0;
    int64_t collectiveOps = 0;
    int64_t optimizerShards = 0;
    int64_t checkpoints = 0;

    // Walk mesh-like ops (prefix 'tessera.mesh.')
    mod.walk([&](Operation *op){
      auto name = op->getName().getStringRef();
      if (name.starts_with("tessera.mesh.")) {
        ++meshOps;
      }
      if (name.starts_with("tessera.collective.")) {
        ++collectiveOps;
      }
      if (name == "tessera_sr.checkpoint") {
        ++checkpoints;
      }
      if (name.ends_with("optimizer.shard")) {
        ++optimizerShards;
      }
    });

    // Write a dependency-free JSON summary. This keeps the pass usable in
    // minimal MLIR 21 builds where nlohmann/json is not installed.
    std::ofstream out("manifest.json");
    out << "{\n"
        << "  \"version\": \"v1.1\",\n"
        << "  \"mesh_ops\": " << meshOps << ",\n"
        << "  \"collective_ops\": " << collectiveOps << ",\n"
        << "  \"optimizer_shards\": " << optimizerShards << ",\n"
        << "  \"checkpoints\": " << checkpoints << "\n"
        << "}\n";
    mod.emitRemark() << "export manifest wrote manifest.json";
  }
};
}

std::unique_ptr<Pass> mlir::tessera::sr::createExportDeploymentManifestPass() {
  return std::make_unique<ExportDeploymentManifestPass>();
}
