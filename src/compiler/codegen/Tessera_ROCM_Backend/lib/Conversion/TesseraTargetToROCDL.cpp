#include "TesseraROCM/Passes.h"

#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/LLVMIR/ROCDLDialect.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Pass/Pass.h"

using namespace mlir;

namespace {

LLVM::LLVMFuncOp declareVoidMarker(ModuleOp module, StringRef name) {
  if (auto fn = module.lookupSymbol<LLVM::LLVMFuncOp>(name))
    return fn;

  OpBuilder builder(module.getBodyRegion());
  builder.setInsertionPointToStart(module.getBody());
  auto fnType = LLVM::LLVMFunctionType::get(
      LLVM::LLVMVoidType::get(module.getContext()), {}, false);
  return builder.create<LLVM::LLVMFuncOp>(module.getLoc(), name, fnType);
}

void replaceResultUsesWithUndef(Operation *op, PatternRewriter &rewriter) {
  for (Value result : op->getResults()) {
    if (result.use_empty())
      continue;
    auto undef = rewriter.create<LLVM::UndefOp>(op->getLoc(), result.getType());
    result.replaceAllUsesExcept(undef, undef);
  }
}

struct LoweringPass : PassWrapper<LoweringPass, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(LoweringPass)

  StringRef getArgument() const final { return "lower-tessera-target-to-rocdl"; }

  StringRef getDescription() const final {
    return "Lower Tessera ROCm target ops to LLVM/ROCDL artifact markers";
  }

  void runOnOperation() override {
    getContext().loadDialect<LLVM::LLVMDialect, ROCDL::ROCDLDialect>();
    ModuleOp module = getOperation();
    SmallVector<Operation *> waitOps;
    SmallVector<Operation *> rocmOps;

    module.walk([&](Operation *op) {
      StringRef name = op->getName().getStringRef();
      if (name == "tessera_rocm.wait")
        waitOps.push_back(op);
      else if (name == "tessera_rocm.mfma" ||
               name == "tessera_rocm.async_copy")
        rocmOps.push_back(op);
    });
    waitOps.append(rocmOps.begin(), rocmOps.end());
    rocmOps = std::move(waitOps);

    for (Operation *op : rocmOps) {
      PatternRewriter rewriter(op->getContext());
      rewriter.setInsertionPoint(op);

      StringRef opName = op->getName().getStringRef();
      StringRef markerName = "llvm.tessera.rocm.unknown";
      if (opName == "tessera_rocm.mfma")
        markerName = "llvm.amdgcn.mfma.contract";
      else if (opName == "tessera_rocm.async_copy")
        markerName = "llvm.amdgcn.raw.buffer.copy.contract";
      else if (opName == "tessera_rocm.wait")
        markerName = "llvm.amdgcn.s.barrier.contract";

      auto marker = declareVoidMarker(module, markerName);
      rewriter.create<LLVM::CallOp>(op->getLoc(), TypeRange{},
                                    SymbolRefAttr::get(marker), ValueRange{});
      replaceResultUsesWithUndef(op, rewriter);
      rewriter.eraseOp(op);
    }

    bool leakedROCMOp = false;
    module.walk([&](Operation *op) {
      if (op->getName().getStringRef().starts_with("tessera_rocm.")) {
        op->emitError("unsupported ROCm target op after ROCDL lowering");
        leakedROCMOp = true;
      }
    });
    if (leakedROCMOp)
      signalPassFailure();
  }
};

} // namespace

std::unique_ptr<mlir::Pass> mlir::tessera_rocm::createLowerTesseraToROCDLImpl() {
  return std::make_unique<LoweringPass>();
}
