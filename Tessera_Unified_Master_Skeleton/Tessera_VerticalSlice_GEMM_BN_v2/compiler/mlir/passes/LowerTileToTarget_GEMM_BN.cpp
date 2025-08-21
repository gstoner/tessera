
//===- LowerTileToTarget_GEMM_BN.cpp -----------------------------------*- C++ -*-===//
// Worked example: translate tile ops into runtime calls (CPU path) or NVVM/ROCDL.
#include "mlir/Pass/Pass.h"

using namespace mlir;

namespace tessera {
namespace {
struct LowerTileToTargetPass : public PassWrapper<LowerTileToTargetPass, OperationPass<ModuleOp>> {
  void runOnOperation() override {
    // In a real pass, you'd pattern-match ttile.gemm_mma and lower to:
    //  - NVVM MMA ops (mma.sync / wgmma) with shared-mem tiling, OR
    //  - A call to a runtime entry (tesseraMatmul) for CPU fallback.
    getOperation().emitRemark("LowerTileToTarget (GEMM/BN) - replace ttile.* with target ops or runtime calls");
  }
};
} // namespace

std::unique_ptr<Pass> createLowerTileToTargetPass() {
  return std::make_unique<LowerTileToTargetPass>();
}
} // namespace tessera
