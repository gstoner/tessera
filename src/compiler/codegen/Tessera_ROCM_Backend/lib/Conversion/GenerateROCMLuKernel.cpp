//===- GenerateROCMLuKernel.cpp - batched LU (partial pivot) gpu kernel ---===//
//
// Expands `tessera_rocm.lu` into a batched LU factorization with partial
// pivoting (getrf), one thread per matrix — A[b,n,n] → packed LU[b,n,n]
// (unit-lower L below the diagonal, U on/above) + pivots[b,n] (0-based,
// piv[k] = row swapped with k at step k):
//
//   p = argmax_{i≥k} |LU[i,k]| ; swap rows k,p ; piv[k]=p
//   LU[i,k] /= LU[k,k] ; LU[i,j] -= LU[i,k]·LU[k,j]   (i,j > k)
//
// Each thread factorizes its own matrix from global memory (sequential O(n³),
// the honest foundation). pivots are i32. All f32. CPU analog: avx512_lu_qr lu.
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

void emitLuBody(OpBuilder &b, Location loc, gpu::GPUFuncOp f) {
  Type f32 = b.getF32Type();
  Type i32 = b.getI32Type();
  Type idx = b.getIndexType();
  auto slt = arith::CmpIPredicate::slt;
  b.setInsertionPointToStart(&f.getBody().front());
  Value A = f.getArgument(0), LU = f.getArgument(1), PIV = f.getArgument(2);
  Value batch = f.getArgument(3), N = f.getArgument(4);

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
  Value pbase = b.create<arith::MulIOp>(loc, gid, N);
  auto off = [&](Value r, Value c) {
    return b.create<arith::AddIOp>(
        loc, base, b.create<arith::AddIOp>(
                       loc, b.create<arith::MulIOp>(loc, r, N), c)).getResult();
  };

  // copy A -> LU
  auto cp = b.create<scf::ForOp>(loc, c0, nn, c1);
  {
    OpBuilder::InsertionGuard g(b);
    b.setInsertionPointToStart(cp.getBody());
    Value o = b.create<arith::AddIOp>(loc, base, cp.getInductionVar());
    b.create<memref::StoreOp>(loc, b.create<memref::LoadOp>(loc, A, ValueRange{o}),
                              LU, ValueRange{o});
  }

  auto kl = b.create<scf::ForOp>(loc, c0, N, c1);
  {
    OpBuilder::InsertionGuard g(b);
    b.setInsertionPointToStart(kl.getBody());
    Value k = kl.getInductionVar();
    Value kp1 = b.create<arith::AddIOp>(loc, k, c1);
    // argmax |LU[i,k]| over i in [k,n)
    Value bestInit = b.create<math::AbsFOp>(
        loc, b.create<memref::LoadOp>(loc, LU, ValueRange{off(k, k)}));
    auto am = b.create<scf::ForOp>(loc, kp1, N, c1, ValueRange{bestInit, k});
    {
      OpBuilder::InsertionGuard g2(b);
      b.setInsertionPointToStart(am.getBody());
      Value i = am.getInductionVar();
      Value best = am.getRegionIterArgs()[0];
      Value bidx = am.getRegionIterArgs()[1];
      Value v = b.create<math::AbsFOp>(
          loc, b.create<memref::LoadOp>(loc, LU, ValueRange{off(i, k)}));
      Value gt = b.create<arith::CmpFOp>(loc, arith::CmpFPredicate::OGT, v, best);
      Value nb = b.create<arith::SelectOp>(loc, gt, v, best);
      Value ni = b.create<arith::SelectOp>(loc, gt, i, bidx);
      b.create<scf::YieldOp>(loc, ValueRange{nb, ni});
    }
    Value p = am.getResult(1);
    Value poff = b.create<arith::AddIOp>(loc, pbase, k);
    b.create<memref::StoreOp>(loc, b.create<arith::IndexCastOp>(loc, i32, p),
                              PIV, ValueRange{poff});
    // swap rows k,p when p != k
    Value neq = b.create<arith::CmpIOp>(loc, arith::CmpIPredicate::ne, p, k);
    auto sw = b.create<scf::IfOp>(loc, neq, /*withElse=*/false);
    {
      OpBuilder::InsertionGuard g2(b);
      b.setInsertionPointToStart(sw.thenBlock());
      auto jl = b.create<scf::ForOp>(loc, c0, N, c1);
      OpBuilder::InsertionGuard g3(b);
      b.setInsertionPointToStart(jl.getBody());
      Value j = jl.getInductionVar();
      Value ok = off(k, j), op = off(p, j);
      Value tk = b.create<memref::LoadOp>(loc, LU, ValueRange{ok});
      Value tp = b.create<memref::LoadOp>(loc, LU, ValueRange{op});
      b.create<memref::StoreOp>(loc, tp, LU, ValueRange{ok});
      b.create<memref::StoreOp>(loc, tk, LU, ValueRange{op});
    }
    // eliminate when pivot != 0
    Value pivval = b.create<memref::LoadOp>(loc, LU, ValueRange{off(k, k)});
    Value nz = b.create<arith::CmpFOp>(loc, arith::CmpFPredicate::ONE, pivval,
                                       zero);
    auto el = b.create<scf::IfOp>(loc, nz, /*withElse=*/false);
    {
      OpBuilder::InsertionGuard g2(b);
      b.setInsertionPointToStart(el.thenBlock());
      auto il = b.create<scf::ForOp>(loc, kp1, N, c1);
      OpBuilder::InsertionGuard g3(b);
      b.setInsertionPointToStart(il.getBody());
      Value i = il.getInductionVar();
      Value fv = b.create<arith::DivFOp>(
          loc, b.create<memref::LoadOp>(loc, LU, ValueRange{off(i, k)}), pivval);
      b.create<memref::StoreOp>(loc, fv, LU, ValueRange{off(i, k)});
      auto jl = b.create<scf::ForOp>(loc, kp1, N, c1);
      OpBuilder::InsertionGuard g4(b);
      b.setInsertionPointToStart(jl.getBody());
      Value j = jl.getInductionVar();
      Value cur = b.create<memref::LoadOp>(loc, LU, ValueRange{off(i, j)});
      Value ukj = b.create<memref::LoadOp>(loc, LU, ValueRange{off(k, j)});
      Value upd = b.create<arith::SubFOp>(loc, cur,
                                          b.create<arith::MulFOp>(loc, fv, ukj));
      b.create<memref::StoreOp>(loc, upd, LU, ValueRange{off(i, j)});
    }
  }
  (void)idx;
  b.setInsertionPointToEnd(&f.getBody().front());
  b.create<gpu::ReturnOp>(loc);
}

struct GenerateROCMLuKernelPass
    : PassWrapper<GenerateROCMLuKernelPass, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(GenerateROCMLuKernelPass)

  StringRef getArgument() const final { return "generate-rocm-lu-kernel"; }
  StringRef getDescription() const final {
    return "Expand a tessera_rocm.lu directive into a batched LU (partial "
           "pivoting) gpu kernel (one thread per matrix)";
  }
  void getDependentDialects(DialectRegistry &registry) const final {
    registry.insert<gpu::GPUDialect, scf::SCFDialect, arith::ArithDialect,
                    math::MathDialect, memref::MemRefDialect>();
  }

  void runOnOperation() override {
    ModuleOp module = getOperation();
    SmallVector<Operation *> directives;
    module.walk([&](Operation *op) {
      if (op->getName().getStringRef() == "tessera_rocm.lu")
        directives.push_back(op);
    });
    for (Operation *op : directives) {
      auto nameAttr = op->getAttrOfType<StringAttr>("name");
      if (!nameAttr) {
        op->emitError("tessera_rocm.lu missing name");
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
      auto fnTy = b.getFunctionType({memF32, memF32, memI32, idxTy, idxTy}, {});
      auto gpuMod = b.create<gpu::GPUModuleOp>(loc, kname + "_mod");
      b.setInsertionPointToStart(&gpuMod.getBodyRegion().front());
      auto gpuFunc = b.create<gpu::GPUFuncOp>(loc, kname, fnTy);
      gpuFunc->setAttr(gpu::GPUDialect::getKernelFuncAttrName(), b.getUnitAttr());
      OpBuilder body(gpuFunc.getContext());
      emitLuBody(body, loc, gpuFunc);
      op->erase();
    }
  }
};

} // namespace

std::unique_ptr<mlir::Pass>
mlir::tessera_rocm::createGenerateROCMLuKernelPass() {
  return std::make_unique<GenerateROCMLuKernelPass>();
}
