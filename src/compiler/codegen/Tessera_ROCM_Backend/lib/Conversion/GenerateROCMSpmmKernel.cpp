//===- GenerateROCMSpmmKernel.cpp - CSR sparse-dense matmul gpu kernel ----===//
//
// Expands `tessera_rocm.spmm` into a GENUINELY sparse row-wise SpMM kernel —
// C[M,N] = A_csr[M,K] @ B[K,N], one thread per output element (i,n):
//
//   C[i,n] = Σ_{p=indptr[i]}^{indptr[i+1]} values[p] · B[indices[p]·N + n]
//
// It iterates the nonzero structure (the indptr/indices/values CSR arrays), NOT
// densify-then-GEMM — the honest sparse CPU/GPU analog of the GEMM lane. The
// j-sum is an scf.for over the row's nonzeros; indptr/indices are i32 (→index
// cast), values/B/OUT are f32. O(nnz·N). Validated vs numpy on gfx1151.
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

void emitSpmmBody(OpBuilder &b, Location loc, gpu::GPUFuncOp f) {
  Type f32 = b.getF32Type();
  Type idx = b.getIndexType();
  auto slt = arith::CmpIPredicate::slt;
  b.setInsertionPointToStart(&f.getBody().front());
  Value indptr = f.getArgument(0), indices = f.getArgument(1);
  Value values = f.getArgument(2), B = f.getArgument(3), OUT = f.getArgument(4);
  Value M = f.getArgument(5), N = f.getArgument(6);

  Value bid = b.create<gpu::BlockIdOp>(loc, gpu::Dimension::x);
  Value tid = b.create<gpu::ThreadIdOp>(loc, gpu::Dimension::x);
  Value cBD = b.create<arith::ConstantIndexOp>(loc, BD);
  Value gid = b.create<arith::AddIOp>(loc, b.create<arith::MulIOp>(loc, bid, cBD),
                                      tid);
  Value total = b.create<arith::MulIOp>(loc, M, N);
  Value inb = b.create<arith::CmpIOp>(loc, slt, gid, total);
  auto ifo = b.create<scf::IfOp>(loc, inb, /*withElse=*/false);
  b.setInsertionPointToStart(ifo.thenBlock());

  Value i = b.create<arith::DivUIOp>(loc, gid, N);
  Value n = b.create<arith::RemUIOp>(loc, gid, N);
  Value c1 = b.create<arith::ConstantIndexOp>(loc, 1);
  Value zero = b.create<arith::ConstantOp>(loc, f32, b.getF32FloatAttr(0.0f));
  auto toIdx = [&](Value v) {
    return b.create<arith::IndexCastOp>(loc, idx, v);
  };
  // p0 = indptr[i] ; p1 = indptr[i+1]
  Value p0 = toIdx(b.create<memref::LoadOp>(loc, indptr, ValueRange{i}));
  Value ip1 = b.create<arith::AddIOp>(loc, i, c1);
  Value p1 = toIdx(b.create<memref::LoadOp>(loc, indptr, ValueRange{ip1}));

  auto lp = b.create<scf::ForOp>(loc, p0, p1, c1, ValueRange{zero});
  {
    OpBuilder::InsertionGuard g(b);
    b.setInsertionPointToStart(lp.getBody());
    Value p = lp.getInductionVar();
    Value acc = lp.getRegionIterArgs()[0];
    Value col = toIdx(b.create<memref::LoadOp>(loc, indices, ValueRange{p}));
    Value v = b.create<memref::LoadOp>(loc, values, ValueRange{p});
    Value boff = b.create<arith::AddIOp>(loc, b.create<arith::MulIOp>(loc, col, N),
                                         n);
    Value bv = b.create<memref::LoadOp>(loc, B, ValueRange{boff});
    Value nacc = b.create<arith::AddFOp>(loc, acc,
                                         b.create<arith::MulFOp>(loc, v, bv));
    b.create<scf::YieldOp>(loc, ValueRange{nacc});
  }
  b.create<memref::StoreOp>(loc, lp.getResult(0), OUT, ValueRange{gid});
  b.setInsertionPointToEnd(&f.getBody().front());
  b.create<gpu::ReturnOp>(loc);
}

struct GenerateROCMSpmmKernelPass
    : PassWrapper<GenerateROCMSpmmKernelPass, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(GenerateROCMSpmmKernelPass)

  StringRef getArgument() const final { return "generate-rocm-spmm-kernel"; }
  StringRef getDescription() const final {
    return "Expand a tessera_rocm.spmm directive into a row-wise CSR sparse-"
           "dense matmul gpu kernel (one thread per output element)";
  }
  void getDependentDialects(DialectRegistry &registry) const final {
    registry.insert<gpu::GPUDialect, scf::SCFDialect, arith::ArithDialect,
                    memref::MemRefDialect>();
  }

  void runOnOperation() override {
    ModuleOp module = getOperation();
    SmallVector<Operation *> directives;
    module.walk([&](Operation *op) {
      if (op->getName().getStringRef() == "tessera_rocm.spmm")
        directives.push_back(op);
    });
    for (Operation *op : directives) {
      auto nameAttr = op->getAttrOfType<StringAttr>("name");
      if (!nameAttr) {
        op->emitError("tessera_rocm.spmm missing name");
        return signalPassFailure();
      }
      OpBuilder b(module.getBodyRegion());
      b.setInsertionPointToEnd(module.getBody());
      Location loc = op->getLoc();
      std::string kname = nameAttr.getValue().str();
      Type f32 = b.getF32Type();
      Type i32 = b.getI32Type();
      Type idxTy = b.getIndexType();
      auto memF32 = MemRefType::get({ShapedType::kDynamic}, f32);
      auto memI32 = MemRefType::get({ShapedType::kDynamic}, i32);
      auto fnTy = b.getFunctionType(
          {memI32, memI32, memF32, memF32, memF32, idxTy, idxTy}, {});
      auto gpuMod = b.create<gpu::GPUModuleOp>(loc, kname + "_mod");
      b.setInsertionPointToStart(&gpuMod.getBodyRegion().front());
      auto gpuFunc = b.create<gpu::GPUFuncOp>(loc, kname, fnTy);
      gpuFunc->setAttr(gpu::GPUDialect::getKernelFuncAttrName(), b.getUnitAttr());
      OpBuilder body(gpuFunc.getContext());
      emitSpmmBody(body, loc, gpuFunc);
      op->erase();
    }
  }
};

} // namespace

std::unique_ptr<mlir::Pass>
mlir::tessera_rocm::createGenerateROCMSpmmKernelPass() {
  return std::make_unique<GenerateROCMSpmmKernelPass>();
}
