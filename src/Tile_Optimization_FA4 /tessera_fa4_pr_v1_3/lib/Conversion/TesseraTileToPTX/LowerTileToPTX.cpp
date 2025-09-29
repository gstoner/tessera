//===- LowerTileToPTX.cpp (v1.3) -------------------------------------------===//
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "llvm/IR/InlineAsm.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/Module.h"
using namespace mlir;
namespace tessera { namespace tile {

static void emitTcgen05PTX(llvm::IRBuilder<> &B) {
  auto &M = *B.GetInsertBlock()->getModule();
  auto &C = M.getContext();
  auto *FTy = llvm::FunctionType::get(llvm::Type::getVoidTy(C), false);
  // NOTE: This PTX body is a schematic placeholder. Replace with the true tcgen05.mma body when available.
  std::string asmStr =
R"ptx(
{ // --- tcgen05.mma bf16->f32 (schematic) ---
  // .target sm_100
  // inputs: a_frag, b_frag, acc_frag; cta_group encoded externally
  // mma.sync.aligned.m64n64k16.bf16.bf16.f32.f32 {acc}, {a}, {b}, {acc};
}
)ptx";
  std::string constraints = "~{memory}";
  auto *IA = llvm::InlineAsm::get(FTy, asmStr, constraints, /*hasSideEffects*/true);
  B.CreateCall(IA);
}

struct LowerTileToPTXPass : public PassWrapper<LowerTileToPTXPass, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(LowerTileToPTXPass)
  void runOnOperation() override {
    // Detect sm_100 (TODO: read module attr/target); lower mma.tcgen05 to emitTcgen05PTX.
  }
};
std::unique_ptr<Pass> createLowerTileToPTXPass() { return std::make_unique<LowerTileToPTXPass>(); }

}} // ns
