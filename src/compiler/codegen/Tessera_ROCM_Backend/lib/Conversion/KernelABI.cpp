#include "TesseraROCM/Passes.h"
#include "TesseraROCM/ABI.h"
#include "TesseraROCM/MemSpace.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/IR/PatternMatch.h"
#include "llvm/ADT/STLExtras.h"

using namespace mlir;

static LLVM::LLVMPointerType ptrInAS(MLIRContext *ctx, unsigned as, Type elemTy){
  return LLVM::LLVMPointerType::get(elemTy, as);
}

namespace {

struct KernelABIPass : PassWrapper<KernelABIPass, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(KernelABIPass)
  StringRef getArgument() const final { return "lower-tessera-kernel-abi"; }
  StringRef getDescription() const final {
    return "Lower Tessera ROCm kernel signatures to the AMDGPU ABI";
  }
  void runOnOperation() override {
    ModuleOp m = getOperation();
    auto ctx = m.getContext();
    for (auto fn : m.getOps<func::FuncOp>()) {
      if (!fn->hasAttr("tessera_rocm.kernel")) continue;

      // Convert memref<*xT> -> !llvm.ptr<T, addrspace(1)>
      SmallVector<Type> newArgTys;
      bool changed = false;
      for (auto t : fn.getFunctionType().getInputs()) {
        if (auto mr = dyn_cast<MemRefType>(t)) {
          auto elem = mr.getElementType();
          auto llvmPtr = ptrInAS(ctx, /*global*/1, elem);
          newArgTys.push_back(llvmPtr);
          changed = true;
        } else {
          newArgTys.push_back(t);
        }
      }
      auto resTys = fn.getFunctionType().getResults();
      if (changed) {
        auto newTy = FunctionType::get(ctx, newArgTys, resTys);
        OpBuilder b(fn.getContext());
        b.setInsertionPoint(fn);
        auto newFn = func::FuncOp::create(fn.getLoc(), fn.getName(), newTy);
        newFn->setAttrs(fn->getAttrDictionary());
        newFn.getBody().takeBody(fn.getBody());
        for (auto [arg, newType] :
             llvm::zip(newFn.getBody().getArguments(), newArgTys))
          arg.setType(newType);
        b.insert(newFn);
        fn.erase();
        fn = newFn;
      }

      // Annotate basic ABI
      mlir::tessera_rocm::ABIConfig cfg;
      cfg.mcpu = "gfx90a";
      cfg.wgX = 256; cfg.wgY = 1; cfg.wgZ = 1;
      cfg.ldsBytes = 0;
      mlir::tessera_rocm::annotateKernelABI(fn, cfg);
    }
  }
};

} // namespace

std::unique_ptr<mlir::Pass> mlir::tessera_rocm::createLowerKernelABIPass() {
  return std::make_unique<KernelABIPass>();
}
