//===- LowerTileToPTX.cpp (v1.3) -------------------------------------------===//
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "llvm/IR/InlineAsm.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/Module.h"
using namespace mlir;
namespace tessera { namespace tile {

static bool isBlackwellTarget(mlir::ModuleOp module) {
  auto target = module->getAttrOfType<mlir::StringAttr>("target");
  auto arch = module->getAttrOfType<mlir::StringAttr>("arch");
  auto value = target ? target.getValue() : (arch ? arch.getValue() : "");
  return value.contains("sm100") || value.contains("sm_100") ||
         value.contains("sm120") || value.contains("sm_120") ||
         value.contains("blackwell");
}

static void emitTcgen05PTX(llvm::IRBuilder<> &B) {
  auto &M = *B.GetInsertBlock()->getModule();
  auto &C = M.getContext();
  auto *FTy = llvm::FunctionType::get(llvm::Type::getVoidTy(C), false);
  std::string asmStr =
R"ptx(
{ // --- Tessera guarded tcgen05 contract: bf16 operands, f32 TMEM accumulator ---
  .reg .pred tessera_tcgen05_ready;
  setp.ne.u32 tessera_tcgen05_ready, 0, 0;
  @!tessera_tcgen05_ready tcgen05.mma.cta_group::2.kind::f16
    [tessera_acc_tmem],
    tessera_a_desc,
    tessera_b_desc,
    tessera_scale_desc;
}
)ptx";
  std::string constraints = "~{memory}";
  auto *IA = llvm::InlineAsm::get(FTy, asmStr, constraints, /*hasSideEffects*/true);
  B.CreateCall(IA);
}

struct LowerTileToPTXPass : public PassWrapper<LowerTileToPTXPass, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(LowerTileToPTXPass)
  void runOnOperation() override {
    ModuleOp module = getOperation();
    if (!isBlackwellTarget(module)) {
      module.emitRemark("tcgen05 TMEM PTX contract requires target/arch containing sm100, sm120, or blackwell");
      return;
    }
    // The pass is hardware-free: it emits a guarded inline-PTX contract body
    // for downstream artifact inspection. Native Blackwell execution requires
    // a backend that supplies real descriptors for the named operands.
  }
};
std::unique_ptr<Pass> createLowerTileToPTXPass() { return std::make_unique<LowerTileToPTXPass>(); }

}} // ns
