//===- LowerTileToPTX.cpp --------------------------------------------------===//
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
using namespace mlir;
namespace tessera {
namespace tile {

struct LowerTileToPTXPass : public PassWrapper<LowerTileToPTXPass, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(LowerTileToPTXPass)
  void runOnOperation() override {
    // Lower tile.mma.tcgen05 to NVPTX inline asm or LLVM intrinsics for SM100.
    // Lower ld.tmem/st.tmem to PTX tmem ops once available; fallback to shared/reg shuffle.
  }
};

std::unique_ptr<Pass> createLowerTileToPTXPass() {
  return std::make_unique<LowerTileToPTXPass>();
}

} // namespace tile
} // namespace tessera
