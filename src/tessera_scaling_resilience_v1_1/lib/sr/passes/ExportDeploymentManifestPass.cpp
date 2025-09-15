#include "tessera/sr/Passes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/Attributes.h"
#include <fstream>
#include <nlohmann/json.hpp>

using namespace mlir;
using json = nlohmann::json;

namespace {
struct ExportDeploymentManifestPass : public PassWrapper<ExportDeploymentManifestPass, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(ExportDeploymentManifestPass)
  StringRef getArgument() const final { return "tessera-export-deployment-manifest"; }
  StringRef getDescription() const final { return "Emit deployment manifest JSON capturing mesh/collectives/optimizer/ckpt"; }

  void runOnOperation() override {
    ModuleOp mod = getOperation();
    json manifest;
    manifest["version"] = "v1.1";
    manifest["mesh"] = json::array();
    manifest["collectives"] = json::array();
    manifest["optimizer_shards"] = json::array();
    manifest["checkpoints"] = json::array();

    // Walk mesh-like ops (prefix 'tessera.mesh.')
    mod.walk([&](Operation *op){
      auto name = op->getName().getStringRef();
      if (name.startswith("tessera.mesh.")) {
        json m;
        m["name"] = name.str();
        m["attrs"] = json::object();
        for (auto &it : op->getAttrs()) {
          m["attrs"][it.getName().str()] = it.getValue().dyn_cast_or_null<StringAttr>() ? 
            it.getValue().cast<StringAttr>().getValue().str() : it.getValue().getAsOpaquePointer();
        }
        manifest["mesh"].push_back(m);
      }
      if (name.startswith("tessera.collective.")) {
        json c; c["name"] = name.str(); c["attrs"] = json::object();
        for (auto &it : op->getAttrs()) {
          c["attrs"][it.getName().str()] = it.getValue().isa<StringAttr>() ?
            it.getValue().cast<StringAttr>().getValue().str() : it.getValue().getAsOpaquePointer();
        }
        manifest["collectives"].push_back(c);
      }
      if (name == "tessera_sr.checkpoint") {
        json ck; ck["policy"] = "selective";
        manifest["checkpoints"].push_back(ck);
      }
      if (name.endswith("optimizer.shard")) {
        json s; s["op"] = name.str();
        s["axis"] = "data";
        s["policy"] = "zero2";
        manifest["optimizer_shards"].push_back(s);
      }
    });

    // Write file (default name; could read attr on op in future)
    std::ofstream out("manifest.json");
    out << manifest.dump(2) << std::endl;
  }
};
}

std::unique_ptr<Pass> mlir::tessera::sr::createExportDeploymentManifestPass() {
  return std::make_unique<ExportDeploymentManifestPass>();
}