
#include "tessera/Transforms/Passes.h"
#include "mlir/Pass/Pass.h"
#include "mlir/IR/Builders.h"

using namespace mlir;
namespace {
struct LowerKVTransportPass
    : public PassWrapper<LowerKVTransportPass, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(LowerKVTransportPass)
  StringRef getArgument() const override { return "tessera-lower-kv-transport"; }
  StringRef getDescription() const override {
    return "Lower tessera.target.cpx.kv.{export,import,prefetch} to transport runtime calls (PCIe+CX9 or NVLink)";
  }
  void runOnOperation() override {
    ModuleOp m = getOperation();
    // Skeleton: rewrite ops to calls like @tessera_kv_export_pcie(token, size)
    // Leave as no-op markers for now.
  }
};
}

namespace tessera {
std::unique_ptr<mlir::Pass> createLowerKVTransportPass() {
  return std::make_unique<LowerKVTransportPass>();
}
} // namespace tessera
