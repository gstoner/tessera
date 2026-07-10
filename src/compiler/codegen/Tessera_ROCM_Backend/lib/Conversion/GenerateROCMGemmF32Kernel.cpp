//===- GenerateROCMGemmF32Kernel.cpp - plain f32 GEMM gpu kernel ---------===//
//
// Expands `tessera_rocm.gemm_f32` into a plain single-precision GEMM,
// one thread per output element:
//
//   C[m, n] = Σ_k A[m, k] · B[k, n]      (A: [M,K], B: [K,N], C: [M,N])
//
// RDNA WMMA is f16/bf16 only, so the f32-exact expert GEMM (grouped SwiGLU) can
// not ride the WMMA lane without precision loss. This is the f32 VALU/FMA
// fallback: correctness-first (scalar k-loop, f32 accumulate, no LDS tiling —
// the tiled/blocked perf ladder is a follow-up), matching the numpy f32 matmul.
// M/N/K are runtime index args; the grid folds M*N into a 1-D launch.
//===----------------------------------------------------------------------===//

#include "TesseraROCM/Passes.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"

using namespace mlir;

namespace {

static constexpr int64_t BD = 256;

void emitGemmF32Body(OpBuilder &b, Location loc, gpu::GPUFuncOp f) {
  Type f32 = b.getF32Type();
  auto slt = arith::CmpIPredicate::slt;
  b.setInsertionPointToStart(&f.getBody().front());
  Value A = f.getArgument(0), B = f.getArgument(1), C = f.getArgument(2);
  Value M = f.getArgument(3), N = f.getArgument(4), K = f.getArgument(5);

  Value bid = b.create<gpu::BlockIdOp>(loc, gpu::Dimension::x);
  Value tid = b.create<gpu::ThreadIdOp>(loc, gpu::Dimension::x);
  Value cBD = b.create<arith::ConstantIndexOp>(loc, BD);
  Value gid = b.create<arith::AddIOp>(
      loc, b.create<arith::MulIOp>(loc, bid, cBD), tid);
  Value total = b.create<arith::MulIOp>(loc, M, N);
  Value inb = b.create<arith::CmpIOp>(loc, slt, gid, total);
  auto guard = b.create<scf::IfOp>(loc, inb, /*withElse=*/false);
  b.setInsertionPointToStart(guard.thenBlock());

  Value c0 = b.create<arith::ConstantIndexOp>(loc, 0);
  Value c1 = b.create<arith::ConstantIndexOp>(loc, 1);
  Value zero = b.create<arith::ConstantOp>(loc, f32, b.getF32FloatAttr(0.0f));
  Value m = b.create<arith::DivUIOp>(loc, gid, N);   // row
  Value n = b.create<arith::RemUIOp>(loc, gid, N);   // col
  Value abase = b.create<arith::MulIOp>(loc, m, K);  // m*K
  // acc = Σ_k A[m*K + k] · B[k*N + n]
  auto kl = b.create<scf::ForOp>(loc, c0, K, c1, ValueRange{zero});
  {
    OpBuilder::InsertionGuard g(b);
    b.setInsertionPointToStart(kl.getBody());
    Value k = kl.getInductionVar();
    Value av = b.create<memref::LoadOp>(
        loc, A, ValueRange{b.create<arith::AddIOp>(loc, abase, k)});
    Value boff = b.create<arith::AddIOp>(
        loc, b.create<arith::MulIOp>(loc, k, N), n);   // k*N + n
    Value bv = b.create<memref::LoadOp>(loc, B, ValueRange{boff});
    Value acc = b.create<arith::AddFOp>(loc, kl.getRegionIterArgs()[0],
                                        b.create<arith::MulFOp>(loc, av, bv));
    b.create<scf::YieldOp>(loc, ValueRange{acc});
  }
  b.create<memref::StoreOp>(loc, kl.getResult(0), C, ValueRange{gid});
  b.setInsertionPointToEnd(&f.getBody().front());
  b.create<gpu::ReturnOp>(loc);
}

struct GenerateROCMGemmF32KernelPass
    : PassWrapper<GenerateROCMGemmF32KernelPass, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(GenerateROCMGemmF32KernelPass)

  StringRef getArgument() const final { return "generate-rocm-gemm-f32-kernel"; }
  StringRef getDescription() const final {
    return "Expand a tessera_rocm.gemm_f32 directive into a plain f32 GEMM "
           "kernel (C=A@B, one thread per output element, scalar f32 k-loop)";
  }
  void getDependentDialects(DialectRegistry &registry) const final {
    registry.insert<gpu::GPUDialect, scf::SCFDialect, arith::ArithDialect,
                    memref::MemRefDialect>();
  }

  void runOnOperation() override {
    ModuleOp module = getOperation();
    SmallVector<Operation *> directives;
    module.walk([&](Operation *op) {
      if (op->getName().getStringRef() == "tessera_rocm.gemm_f32")
        directives.push_back(op);
    });
    for (Operation *op : directives) {
      auto nameAttr = op->getAttrOfType<StringAttr>("name");
      if (!nameAttr) {
        op->emitError("tessera_rocm.gemm_f32 missing name");
        return signalPassFailure();
      }
      OpBuilder b(module.getBodyRegion());
      b.setInsertionPointToEnd(module.getBody());
      Location loc = op->getLoc();
      std::string kname = nameAttr.getValue().str();
      Type f32 = b.getF32Type();
      Type idxTy = b.getIndexType();
      auto memF32 = MemRefType::get({ShapedType::kDynamic}, f32);
      auto fnTy = b.getFunctionType(
          {memF32, memF32, memF32, idxTy, idxTy, idxTy}, {});
      auto gpuMod = b.create<gpu::GPUModuleOp>(loc, kname + "_mod");
      b.setInsertionPointToStart(&gpuMod.getBodyRegion().front());
      auto gpuFunc = b.create<gpu::GPUFuncOp>(loc, kname, fnTy);
      gpuFunc->setAttr(gpu::GPUDialect::getKernelFuncAttrName(), b.getUnitAttr());
      OpBuilder body(gpuFunc.getContext());
      emitGemmF32Body(body, loc, gpuFunc);
      op->erase();
    }
  }
};

} // namespace

std::unique_ptr<mlir::Pass>
mlir::tessera_rocm::createGenerateROCMGemmF32KernelPass() {
  return std::make_unique<GenerateROCMGemmF32KernelPass>();
}
