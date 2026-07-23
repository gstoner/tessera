//===- ROCMDynamicLDS.cpp - launch-sized LDS materialization --------------===//
//
// memref.alloca with memory space 3 lowers to llvm.alloca addrspace(3), which
// is private stack allocation with a non-default pointer type, not HIP dynamic
// LDS. Replace the single runtime-sized arena in a kernel with the AMDGPU
// external zero-length workgroup symbol. hipModuleLaunchKernel's sharedMemBytes
// argument supplies the storage for that symbol.
//
// Multiple independent dynamic allocas are rejected for now: correctly packing
// them requires a shared runtime offset expression and a host-visible total.
// TileBufferArenaPass already coalesces the normal single-cohort case.

#include "TesseraROCM/Passes.h"

#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallVector.h"

using namespace mlir;

namespace {

struct ROCMDynamicLDS
    : PassWrapper<ROCMDynamicLDS, OperationPass<gpu::GPUModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(ROCMDynamicLDS)

  StringRef getArgument() const final {
    return "rocm-materialize-dynamic-lds";
  }
  StringRef getDescription() const final {
    return "Replace one runtime-sized addrspace(3) LLVM alloca per ROCm kernel "
           "with launch-sized external LDS";
  }
  void getDependentDialects(DialectRegistry &registry) const final {
    registry.insert<LLVM::LLVMDialect>();
  }

  void runOnOperation() override {
    gpu::GPUModuleOp module = getOperation();
    SmallVector<LLVM::AllocaOp> dynamicLds;
    module.walk([&](LLVM::AllocaOp alloca) {
      auto pointer = dyn_cast<LLVM::LLVMPointerType>(alloca.getType());
      if (pointer && pointer.getAddressSpace() == 3)
        dynamicLds.push_back(alloca);
    });
    if (dynamicLds.empty())
      return;

    // A launch has one dynamic-LDS base and one byte count. Multiple arena
    // allocas need an explicit packed-offset contract; aliasing them here would
    // be silently wrong.
    llvm::DenseMap<Operation *, unsigned> perKernel;
    for (LLVM::AllocaOp alloca : dynamicLds) {
      auto fn = alloca->getParentOfType<LLVM::LLVMFuncOp>();
      if (!fn || ++perKernel[fn.getOperation()] != 1) {
        alloca.emitOpError(
            "ROCM_DYNAMIC_LDS_MULTIPLE_ARENAS: a ROCm kernel may contain only "
            "one runtime-sized LDS arena until packed dynamic-cohort offsets "
            "are represented in the launch contract");
        return signalPassFailure();
      }
    }

    OpBuilder globalBuilder(module.getBodyRegion());
    globalBuilder.setInsertionPointToStart(module.getBody());
    auto arrayType = LLVM::LLVMArrayType::get(
        IntegerType::get(&getContext(), 8), 0);
    constexpr StringLiteral symbol = "__tessera_dynamic_lds";
    auto global = module.lookupSymbol<LLVM::GlobalOp>(symbol);
    if (!global) {
      global = LLVM::GlobalOp::create(
          globalBuilder, module.getLoc(), arrayType, /*isConstant=*/false,
          LLVM::Linkage::External, symbol, Attribute(), /*alignment=*/16,
          /*addrSpace=*/3, /*dsoLocal=*/false, /*threadLocal=*/false,
          SymbolRefAttr(), ArrayRef<NamedAttribute>{},
          ArrayRef<Attribute>{});
    }

    for (LLVM::AllocaOp alloca : dynamicLds) {
      OpBuilder builder(alloca);
      Value base = LLVM::AddressOfOp::create(
          builder, alloca.getLoc(), global);
      alloca.replaceAllUsesWith(base);
      if (auto fn = alloca->getParentOfType<LLVM::LLVMFuncOp>())
        fn->setAttr("tessera.rocm.dynamic_lds_launch_bytes",
                    builder.getUnitAttr());
      alloca.erase();
    }
  }
};

} // namespace

std::unique_ptr<mlir::Pass>
mlir::tessera_rocm::createROCMDynamicLDSPass() {
  return std::make_unique<ROCMDynamicLDS>();
}
