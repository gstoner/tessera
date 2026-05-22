
// Lower memory.update/read/reset to Tile-IR/Target-IR kernels and streams.
// cmd: -tessera-atlas-memory-lower
#include "mlir/Pass/Pass.h"
using namespace mlir;
namespace {
struct AtlasMemoryKernelLoweringPass : PassWrapper<AtlasMemoryKernelLoweringPass, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(AtlasMemoryKernelLoweringPass)
  StringRef getArgument() const final { return "tessera-atlas-memory-lower"; }
  StringRef getDescription() const final { return "Lower Atlas memory ops to fused update/read kernels."; }
  void runOnOperation() override {
    // TODO: Emit fusion around (K,V) tile loads, optimizer update (muon/gd), and query readout.
  }
};
}
std::unique_ptr<Pass> createAtlasMemoryKernelLoweringPass() { return std::make_unique<AtlasMemoryKernelLoweringPass>(); }
