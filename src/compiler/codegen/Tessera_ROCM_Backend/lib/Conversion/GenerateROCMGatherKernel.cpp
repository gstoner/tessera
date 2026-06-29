//===- GenerateROCMGatherKernel.cpp - strided-copy / gather kernel -------===//
//
// Expands a `tessera_rocm.gather` directive into a flat masked-gather gpu
// kernel — the 0-move lane (P4): one thread per output element,
// `out[i] = (0 <= idx[i] < src_n) ? src[idx[i]] : out[i]`. A negative index
// leaves out[i] (pad fill / cat per-input accumulate). The host computes the
// integer index map from the op's numpy semantics on an arange grid; this
// kernel does the f32 data movement on-device. CPU analog: avx512_gather_f32.
//
// Args: (src : memref<?xf32>, src_n : index, idx : memref<?xi64>,
//        out : memref<?xf32>, N : index).
//===----------------------------------------------------------------------===//

#include "TesseraROCM/Passes.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Pass/Pass.h"

using namespace mlir;

namespace {

static constexpr int64_t BD = 256;

void emitGatherBody(OpBuilder &b, Location loc, gpu::GPUFuncOp f) {
  Type i64 = b.getIntegerType(64);
  Type idxTy = b.getIndexType();
  auto slt = arith::CmpIPredicate::slt;
  auto sge = arith::CmpIPredicate::sge;

  b.setInsertionPointToStart(&f.getBody().front());
  Value SRC = f.getArgument(0), SRCN = f.getArgument(1);
  Value IDX = f.getArgument(2), OUT = f.getArgument(3), N = f.getArgument(4);

  Value bid = b.create<gpu::BlockIdOp>(loc, gpu::Dimension::x);
  Value tid = b.create<gpu::ThreadIdOp>(loc, gpu::Dimension::x);
  Value cBD = b.create<arith::ConstantIndexOp>(loc, BD);
  Value gid = b.create<arith::AddIOp>(loc, b.create<arith::MulIOp>(loc, bid, cBD),
                                      tid);
  Value inb = b.create<arith::CmpIOp>(loc, slt, gid, N);
  auto guard = b.create<scf::IfOp>(loc, inb, /*withElse=*/false);
  b.setInsertionPointToStart(guard.thenBlock());

  Value j = b.create<memref::LoadOp>(loc, IDX, ValueRange{gid});  // i64
  Value zero64 = b.create<arith::ConstantOp>(loc, i64, b.getI64IntegerAttr(0));
  Value srcN64 = b.create<arith::IndexCastOp>(loc, i64, SRCN);
  Value ge0 = b.create<arith::CmpIOp>(loc, sge, j, zero64);
  Value ltN = b.create<arith::CmpIOp>(loc, slt, j, srcN64);
  Value ok = b.create<arith::AndIOp>(loc, ge0, ltN);
  auto wif = b.create<scf::IfOp>(loc, ok, /*withElse=*/false);
  {
    OpBuilder wb = OpBuilder::atBlockBegin(wif.thenBlock());
    Value ji = wb.create<arith::IndexCastOp>(loc, idxTy, j);
    Value v = wb.create<memref::LoadOp>(loc, SRC, ValueRange{ji});
    wb.create<memref::StoreOp>(loc, v, OUT, ValueRange{gid});
  }

  b.setInsertionPointToEnd(&f.getBody().front());
  b.create<gpu::ReturnOp>(loc);
}

struct GenerateROCMGatherKernelPass
    : PassWrapper<GenerateROCMGatherKernelPass, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(GenerateROCMGatherKernelPass)

  StringRef getArgument() const final { return "generate-rocm-gather-kernel"; }
  StringRef getDescription() const final {
    return "Expand a tessera_rocm.gather directive into a flat masked-gather "
           "gpu kernel (out[i]=src[idx[i]] for the 0-move lane)";
  }
  void getDependentDialects(DialectRegistry &registry) const final {
    registry.insert<gpu::GPUDialect, scf::SCFDialect, arith::ArithDialect,
                    memref::MemRefDialect>();
  }

  void runOnOperation() override {
    ModuleOp module = getOperation();
    SmallVector<Operation *> directives;
    module.walk([&](Operation *op) {
      if (op->getName().getStringRef() == "tessera_rocm.gather")
        directives.push_back(op);
    });
    for (Operation *op : directives) {
      auto nameAttr = op->getAttrOfType<StringAttr>("name");
      if (!nameAttr) {
        op->emitError("tessera_rocm.gather missing name");
        return signalPassFailure();
      }
      OpBuilder b(module.getBodyRegion());
      b.setInsertionPointToEnd(module.getBody());
      Location loc = op->getLoc();
      std::string kname = nameAttr.getValue().str();
      Type f32 = b.getF32Type();
      Type i64 = b.getIntegerType(64);
      Type idxTy = b.getIndexType();
      auto memF32 = MemRefType::get({ShapedType::kDynamic}, f32);
      auto memI64 = MemRefType::get({ShapedType::kDynamic}, i64);
      auto fnTy = b.getFunctionType(
          {memF32, idxTy, memI64, memF32, idxTy}, {});
      auto gpuMod = b.create<gpu::GPUModuleOp>(loc, kname + "_mod");
      b.setInsertionPointToStart(&gpuMod.getBodyRegion().front());
      auto gpuFunc = b.create<gpu::GPUFuncOp>(loc, kname, fnTy);
      gpuFunc->setAttr(gpu::GPUDialect::getKernelFuncAttrName(), b.getUnitAttr());
      OpBuilder body(gpuFunc.getContext());
      emitGatherBody(body, loc, gpuFunc);
      op->erase();
    }
  }
};

} // namespace

std::unique_ptr<mlir::Pass>
mlir::tessera_rocm::createGenerateROCMGatherKernelPass() {
  return std::make_unique<GenerateROCMGatherKernelPass>();
}
