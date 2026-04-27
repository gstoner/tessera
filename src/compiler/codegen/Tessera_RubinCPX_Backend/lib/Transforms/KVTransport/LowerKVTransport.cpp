
//===- LowerKVTransport.cpp - Lower CPX KV ops to runtime calls -----------===//
//
// Lowers tessera.target.cpx.kv.{export,import,prefetch} to calls into the
// Tessera CPX runtime:
//
//   kv.export  src, "pcie+cx9", chunk_bytes
//     → func.call @tessera_kv_export_pcie(src_ptr, total_bytes, chunk_bytes) → token
//   kv.export  src, "nvlink", chunk_bytes
//     → func.call @tessera_kv_export_nvlink(src_ptr, total_bytes, chunk_bytes) → token
//
//   kv.import  token → dst
//     → func.call @tessera_kv_import(token, dst_ptr)
//
//   kv.prefetch kv[start_page:num_pages]
//     → func.call @tessera_kv_prefetch(kv_ptr, start_page, num_pages)
//
// Runtime function signatures (declared via func.func in the module prologue):
//   func @tessera_kv_export_pcie(!llvm.ptr, i64, i64) -> i64
//   func @tessera_kv_export_nvlink(!llvm.ptr, i64, i64) -> i64
//   func @tessera_kv_import(i64, !llvm.ptr)
//   func @tessera_kv_prefetch(!llvm.ptr, i64, i64)
//
//===-----------------------------------------------------------------------===//

#include "tessera/Transforms/Passes.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassRegistry.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"

using namespace mlir;

namespace {

//===----------------------------------------------------------------------===//
// Helper: ensure a runtime function declaration exists in the module
//===----------------------------------------------------------------------===//

/// Returns (creating if absent) a `func.func @name(argTypes) -> resultTypes`
/// declaration at module scope.
static func::FuncOp getOrInsertRuntimeFunc(OpBuilder &b, ModuleOp module,
                                            StringRef name,
                                            TypeRange argTypes,
                                            TypeRange resultTypes) {
  if (auto fn = module.lookupSymbol<func::FuncOp>(name))
    return fn;

  auto fnTy = b.getFunctionType(argTypes, resultTypes);
  OpBuilder::InsertionGuard guard(b);
  b.setInsertionPointToStart(module.getBody());
  auto fn = b.create<func::FuncOp>(module.getLoc(), name, fnTy);
  fn.setPrivate();
  return fn;
}

//===----------------------------------------------------------------------===//
// LowerKVTransportPass
//===----------------------------------------------------------------------===//

struct LowerKVTransportPass
    : public PassWrapper<LowerKVTransportPass, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(LowerKVTransportPass)

  StringRef getArgument() const override { return "tessera-lower-kv-transport"; }
  StringRef getDescription() const override {
    return "Lower tessera.target.cpx.kv.{export,import,prefetch} to "
           "transport runtime calls (PCIe+CX9 or NVLink)";
  }

  void runOnOperation() override {
    ModuleOp module = getOperation();
    OpBuilder b(module.getContext());

    // Common types
    Type i64Ty = b.getI64Type();
    Type ptrTy = b.getType<LLVM::LLVMPointerType>(); // opaque ptr

    // ── Declare runtime functions ──────────────────────────────────────────
    auto kvExportPcieFn = getOrInsertRuntimeFunc(
        b, module, "tessera_kv_export_pcie",
        {ptrTy, i64Ty, i64Ty},   // (src_ptr, total_bytes, chunk_bytes)
        {i64Ty});                  // -> transport token (opaque i64 handle)

    auto kvExportNvlinkFn = getOrInsertRuntimeFunc(
        b, module, "tessera_kv_export_nvlink",
        {ptrTy, i64Ty, i64Ty},
        {i64Ty});

    auto kvImportFn = getOrInsertRuntimeFunc(
        b, module, "tessera_kv_import",
        {i64Ty, ptrTy},   // (token, dst_ptr)
        {});               // void

    auto kvPrefetchFn = getOrInsertRuntimeFunc(
        b, module, "tessera_kv_prefetch",
        {ptrTy, i64Ty, i64Ty},   // (kv_ptr, start_page, num_pages)
        {});

    // ── Walk and lower kv ops ──────────────────────────────────────────────
    // We collect ops first to avoid mutating the IR while walking.
    SmallVector<Operation *> kvExportOps, kvImportOps, kvPrefetchOps;

    module.walk([&](Operation *op) {
      StringRef name = op->getName().getStringRef();
      if (name.endswith("kv.export"))   kvExportOps.push_back(op);
      if (name.endswith("kv.import"))   kvImportOps.push_back(op);
      if (name.endswith("kv.prefetch")) kvPrefetchOps.push_back(op);
    });

    // ── Lower kv.export ────────────────────────────────────────────────────
    for (Operation *op : kvExportOps) {
      b.setInsertionPoint(op);
      Location loc = op->getLoc();

      // Determine transport policy ("pcie+cx9" vs "nvlink")
      auto policyAttr = op->getAttrOfType<StringAttr>("policy");
      StringRef policy = policyAttr ? policyAttr.getValue() : "pcie+cx9";
      func::FuncOp runtimeFn =
          policy == "nvlink" ? kvExportNvlinkFn : kvExportPcieFn;

      // Get the source memref → extract pointer via memref.extract_aligned_pointer_as_index
      Value src = op->getOperand(0);

      // Cast memref to raw pointer (memref → intptr → i64 ptr via index_cast)
      // In a real lowering this goes through -convert-memref-to-llvm first;
      // here we use a placeholder inttoptr so the IR remains well-formed at
      // the -O0 / inspection level.
      Value srcPtr = b.create<memref::ExtractAlignedPointerAsIndexOp>(loc, src);
      srcPtr = b.create<arith::IndexCastOp>(loc, i64Ty, srcPtr);
      srcPtr = b.create<LLVM::IntToPtrOp>(loc, ptrTy, srcPtr);

      // total_bytes: compute from memref type (element_count * element_bytes)
      auto memTy = src.getType().cast<MemRefType>();
      int64_t elemBytes = memTy.getElementType().getIntOrFloatBitWidth() / 8;
      int64_t totalElems = 1;
      for (int64_t d : memTy.getShape()) totalElems *= d;
      Value totalBytes = b.create<arith::ConstantOp>(
          loc, i64Ty, b.getI64IntegerAttr(totalElems * elemBytes));

      // chunk_bytes from op attribute
      auto chunkAttr = op->getAttrOfType<IntegerAttr>("chunk_bytes");
      Value chunkBytes = b.create<arith::ConstantOp>(
          loc, i64Ty,
          b.getI64IntegerAttr(chunkAttr ? chunkAttr.getInt() : 32 * 1024 * 1024));

      // Emit call and replace op result(s)
      auto call = b.create<func::CallOp>(
          loc, runtimeFn, ValueRange{srcPtr, totalBytes, chunkBytes});

      // Replace the kv.export op's token result with the i64 return value
      if (!op->getResults().empty())
        op->getResult(0).replaceAllUsesWith(call.getResult(0));
      op->erase();
    }

    // ── Lower kv.import ────────────────────────────────────────────────────
    for (Operation *op : kvImportOps) {
      b.setInsertionPoint(op);
      Location loc = op->getLoc();

      Value token = op->getOperand(0); // i64 transport token
      Value dst   = op->getOperand(1); // dst memref

      // Cast dst memref to raw pointer
      Value dstPtr = b.create<memref::ExtractAlignedPointerAsIndexOp>(loc, dst);
      dstPtr = b.create<arith::IndexCastOp>(loc, i64Ty, dstPtr);
      dstPtr = b.create<LLVM::IntToPtrOp>(loc, ptrTy, dstPtr);

      b.create<func::CallOp>(loc, kvImportFn, ValueRange{token, dstPtr});
      op->erase();
    }

    // ── Lower kv.prefetch ──────────────────────────────────────────────────
    for (Operation *op : kvPrefetchOps) {
      b.setInsertionPoint(op);
      Location loc = op->getLoc();

      Value kv = op->getOperand(0);

      // Cast kv memref to raw pointer
      Value kvPtr = b.create<memref::ExtractAlignedPointerAsIndexOp>(loc, kv);
      kvPtr = b.create<arith::IndexCastOp>(loc, i64Ty, kvPtr);
      kvPtr = b.create<LLVM::IntToPtrOp>(loc, ptrTy, kvPtr);

      auto startAttr = op->getAttrOfType<IntegerAttr>("start_page");
      auto numAttr   = op->getAttrOfType<IntegerAttr>("num_pages");
      Value startPage = b.create<arith::ConstantOp>(
          loc, i64Ty, b.getI64IntegerAttr(startAttr ? startAttr.getInt() : 0));
      Value numPages = b.create<arith::ConstantOp>(
          loc, i64Ty, b.getI64IntegerAttr(numAttr ? numAttr.getInt() : 1));

      b.create<func::CallOp>(loc, kvPrefetchFn,
                              ValueRange{kvPtr, startPage, numPages});
      op->erase();
    }
  }
};

PassRegistration<LowerKVTransportPass> lowerKVTransportPassReg;

} // namespace

namespace tessera {
std::unique_ptr<mlir::Pass> createLowerKVTransportPass() {
  return std::make_unique<LowerKVTransportPass>();
}
} // namespace tessera
