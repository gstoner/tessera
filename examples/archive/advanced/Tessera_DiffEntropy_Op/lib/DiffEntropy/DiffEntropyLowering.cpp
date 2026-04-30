//===- DiffEntropyLowering.cpp -------------------------------------------===//
// Lower Tessera DiffEntropy ops to Target-IR or LLVM as needed.
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "DiffEntropyOps.h"

using namespace mlir;
using namespace tessera::diffentropy;

namespace {
struct LowerDiffEntropyToLLVMPass
    : public PassWrapper<LowerDiffEntropyToLLVMPass, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(LowerDiffEntropyToLLVMPass)
  void runOnOperation() final {
    // TODO: insert real patterns to map ops to target kernels
    // For now, leave ops as-is (legalized by CPU reference kernel calls).
  }
};
} // namespace

std::unique_ptr<Pass> createLowerDiffEntropyToLLVM() {
  return std::make_unique<LowerDiffEntropyToLLVMPass>();
}
