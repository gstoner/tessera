//===- GenerateROCMCholeskyKernel.cpp - batched Cholesky gpu kernel -------===//
//
// Expands `tessera_rocm.cholesky` into a batched Cholesky factorization, one
// thread per matrix — A[b,n,n] SPD → L[b,n,n] lower, A = L·Lᵀ
// (Cholesky–Banachiewicz):
//
//   L[j,j] = sqrt(A[j,j] - Σ_{k<j} L[j,k]²)
//   L[i,j] = (A[i,j] - Σ_{k<j} L[i,k]·L[j,k]) / L[j,j]   (i>j)
//
// Each thread factorizes its own matrix from global memory (sequential O(n³);
// the honest foundation — a blocked/panel version is a follow-up). sqrt via
// math→ROCDL. All f32. CPU analog: avx512_linalg_f32 cholesky.
//===----------------------------------------------------------------------===//

#include "TesseraROCM/Passes.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"

using namespace mlir;

namespace {

static constexpr int64_t BD = 256;

void emitCholBody(OpBuilder &b, Location loc, gpu::GPUFuncOp f) {
  Type f32 = b.getF32Type();
  auto slt = arith::CmpIPredicate::slt;
  b.setInsertionPointToStart(&f.getBody().front());
  Value A = f.getArgument(0), L = f.getArgument(1);
  Value batch = f.getArgument(2), N = f.getArgument(3);

  Value bid = b.create<gpu::BlockIdOp>(loc, gpu::Dimension::x);
  Value tid = b.create<gpu::ThreadIdOp>(loc, gpu::Dimension::x);
  Value cBD = b.create<arith::ConstantIndexOp>(loc, BD);
  Value gid = b.create<arith::AddIOp>(loc, b.create<arith::MulIOp>(loc, bid, cBD),
                                      tid);
  Value inb = b.create<arith::CmpIOp>(loc, slt, gid, batch);
  auto guard = b.create<scf::IfOp>(loc, inb, /*withElse=*/false);
  b.setInsertionPointToStart(guard.thenBlock());

  Value c0 = b.create<arith::ConstantIndexOp>(loc, 0);
  Value c1 = b.create<arith::ConstantIndexOp>(loc, 1);
  Value zero = b.create<arith::ConstantOp>(loc, f32, b.getF32FloatAttr(0.0f));
  Value nn = b.create<arith::MulIOp>(loc, N, N);
  Value base = b.create<arith::MulIOp>(loc, gid, nn);
  auto eltA = [&](Value r, Value c) {
    Value off = b.create<arith::AddIOp>(
        loc, base, b.create<arith::AddIOp>(
                       loc, b.create<arith::MulIOp>(loc, r, N), c));
    return b.create<memref::LoadOp>(loc, A, ValueRange{off}).getResult();
  };
  auto offL = [&](Value r, Value c) {
    return b.create<arith::AddIOp>(
        loc, base, b.create<arith::AddIOp>(
                       loc, b.create<arith::MulIOp>(loc, r, N), c)).getResult();
  };

  // zero L[base : base+n*n]
  auto zl = b.create<scf::ForOp>(loc, c0, nn, c1);
  {
    OpBuilder::InsertionGuard g(b);
    b.setInsertionPointToStart(zl.getBody());
    Value off = b.create<arith::AddIOp>(loc, base, zl.getInductionVar());
    b.create<memref::StoreOp>(loc, zero, L, ValueRange{off});
  }

  auto jl = b.create<scf::ForOp>(loc, c0, N, c1);
  {
    OpBuilder::InsertionGuard g(b);
    b.setInsertionPointToStart(jl.getBody());
    Value j = jl.getInductionVar();
    // s = A[j,j] - Σ_{k<j} L[j,k]²
    auto sl = b.create<scf::ForOp>(loc, c0, j, c1, ValueRange{eltA(j, j)});
    {
      OpBuilder::InsertionGuard g2(b);
      b.setInsertionPointToStart(sl.getBody());
      Value k = sl.getInductionVar();
      Value ljk = b.create<memref::LoadOp>(loc, L, ValueRange{offL(j, k)});
      Value acc = b.create<arith::SubFOp>(
          loc, sl.getRegionIterArgs()[0],
          b.create<arith::MulFOp>(loc, ljk, ljk));
      b.create<scf::YieldOp>(loc, ValueRange{acc});
    }
    Value ljj = b.create<math::SqrtOp>(loc, sl.getResult(0));
    b.create<memref::StoreOp>(loc, ljj, L, ValueRange{offL(j, j)});
    // i in (j+1, n): L[i,j] = (A[i,j] - Σ_{k<j} L[i,k]·L[j,k]) / ljj
    Value jp1 = b.create<arith::AddIOp>(loc, j, c1);
    auto il = b.create<scf::ForOp>(loc, jp1, N, c1);
    {
      OpBuilder::InsertionGuard g2(b);
      b.setInsertionPointToStart(il.getBody());
      Value i = il.getInductionVar();
      auto tl = b.create<scf::ForOp>(loc, c0, j, c1, ValueRange{eltA(i, j)});
      {
        OpBuilder::InsertionGuard g3(b);
        b.setInsertionPointToStart(tl.getBody());
        Value k = tl.getInductionVar();
        Value lik = b.create<memref::LoadOp>(loc, L, ValueRange{offL(i, k)});
        Value ljk = b.create<memref::LoadOp>(loc, L, ValueRange{offL(j, k)});
        Value acc = b.create<arith::SubFOp>(
            loc, tl.getRegionIterArgs()[0],
            b.create<arith::MulFOp>(loc, lik, ljk));
        b.create<scf::YieldOp>(loc, ValueRange{acc});
      }
      Value v = b.create<arith::DivFOp>(loc, tl.getResult(0), ljj);
      b.create<memref::StoreOp>(loc, v, L, ValueRange{offL(i, j)});
    }
  }
  b.setInsertionPointToEnd(&f.getBody().front());
  b.create<gpu::ReturnOp>(loc);
}

struct GenerateROCMCholeskyKernelPass
    : PassWrapper<GenerateROCMCholeskyKernelPass, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(GenerateROCMCholeskyKernelPass)

  StringRef getArgument() const final { return "generate-rocm-cholesky-kernel"; }
  StringRef getDescription() const final {
    return "Expand a tessera_rocm.cholesky directive into a batched Cholesky "
           "factorization gpu kernel (one thread per matrix)";
  }
  void getDependentDialects(DialectRegistry &registry) const final {
    registry.insert<gpu::GPUDialect, scf::SCFDialect, arith::ArithDialect,
                    math::MathDialect, memref::MemRefDialect>();
  }

  void runOnOperation() override {
    ModuleOp module = getOperation();
    SmallVector<Operation *> directives;
    module.walk([&](Operation *op) {
      if (op->getName().getStringRef() == "tessera_rocm.cholesky")
        directives.push_back(op);
    });
    for (Operation *op : directives) {
      auto nameAttr = op->getAttrOfType<StringAttr>("name");
      if (!nameAttr) {
        op->emitError("tessera_rocm.cholesky missing name");
        return signalPassFailure();
      }
      OpBuilder b(module.getBodyRegion());
      b.setInsertionPointToEnd(module.getBody());
      Location loc = op->getLoc();
      std::string kname = nameAttr.getValue().str();
      Type f32 = b.getF32Type();
      Type idxTy = b.getIndexType();
      auto memF32 = MemRefType::get({ShapedType::kDynamic}, f32);
      auto fnTy = b.getFunctionType({memF32, memF32, idxTy, idxTy}, {});
      auto gpuMod = b.create<gpu::GPUModuleOp>(loc, kname + "_mod");
      b.setInsertionPointToStart(&gpuMod.getBodyRegion().front());
      auto gpuFunc = b.create<gpu::GPUFuncOp>(loc, kname, fnTy);
      gpuFunc->setAttr(gpu::GPUDialect::getKernelFuncAttrName(), b.getUnitAttr());
      OpBuilder body(gpuFunc.getContext());
      emitCholBody(body, loc, gpuFunc);
      op->erase();
    }
  }
};

} // namespace

std::unique_ptr<mlir::Pass>
mlir::tessera_rocm::createGenerateROCMCholeskyKernelPass() {
  return std::make_unique<GenerateROCMCholeskyKernelPass>();
}
