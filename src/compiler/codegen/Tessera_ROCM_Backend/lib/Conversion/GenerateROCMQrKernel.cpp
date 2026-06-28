//===- GenerateROCMQrKernel.cpp - batched Householder QR gpu kernel -------===//
//
// Expands `tessera_rocm.qr` into a batched Householder QR, one thread per matrix
// — A[b,m,n] → full Q[b,m,m] (orthonormal) + R[b,m,n] (upper-trapezoid), A=Q·R.
// The runtime slices the reduced Q[:, :k] / R[:k, :], k=min(m,n).
//
// For each column j: reflector v on R[j:,j] (alpha=-sign(R[j,j])·‖·‖), then
// R ← (I-βvvᵀ)R and Q ← Q(I-βvvᵀ). The per-thread reflector v[m] lives in a
// global scratch buffer V[b·m]. norm via math.sqrt→ROCDL. All f32. CPU analog:
// avx512_lu_qr qr.
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

void emitQrBody(OpBuilder &b, Location loc, gpu::GPUFuncOp f) {
  Type f32 = b.getF32Type();
  auto slt = arith::CmpIPredicate::slt;
  b.setInsertionPointToStart(&f.getBody().front());
  Value A = f.getArgument(0), Q = f.getArgument(1), R = f.getArgument(2);
  Value V = f.getArgument(3);
  Value batch = f.getArgument(4), M = f.getArgument(5), N = f.getArgument(6);

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
  Value f0 = b.create<arith::ConstantOp>(loc, f32, b.getF32FloatAttr(0.0f));
  Value f1 = b.create<arith::ConstantOp>(loc, f32, b.getF32FloatAttr(1.0f));
  Value f2 = b.create<arith::ConstantOp>(loc, f32, b.getF32FloatAttr(2.0f));
  Value mn = b.create<arith::MulIOp>(loc, M, N);
  Value mm = b.create<arith::MulIOp>(loc, M, M);
  Value rbase = b.create<arith::MulIOp>(loc, gid, mn);
  Value qbase = b.create<arith::MulIOp>(loc, gid, mm);
  Value vbase = b.create<arith::MulIOp>(loc, gid, M);
  auto rOff = [&](Value r, Value c) {
    return b.create<arith::AddIOp>(
        loc, rbase, b.create<arith::AddIOp>(
                        loc, b.create<arith::MulIOp>(loc, r, N), c)).getResult();
  };
  auto qOff = [&](Value r, Value c) {
    return b.create<arith::AddIOp>(
        loc, qbase, b.create<arith::AddIOp>(
                        loc, b.create<arith::MulIOp>(loc, r, M), c)).getResult();
  };
  auto vOff = [&](Value i) {
    return b.create<arith::AddIOp>(loc, vbase, i).getResult();
  };

  // R = A
  auto cpr = b.create<scf::ForOp>(loc, c0, mn, c1);
  {
    OpBuilder::InsertionGuard g(b);
    b.setInsertionPointToStart(cpr.getBody());
    Value o = b.create<arith::AddIOp>(loc, rbase, cpr.getInductionVar());
    b.create<memref::StoreOp>(loc, b.create<memref::LoadOp>(loc, A, ValueRange{o}),
                              R, ValueRange{o});
  }
  // Q = I
  auto cpq = b.create<scf::ForOp>(loc, c0, mm, c1);
  {
    OpBuilder::InsertionGuard g(b);
    b.setInsertionPointToStart(cpq.getBody());
    Value o = b.create<arith::AddIOp>(loc, qbase, cpq.getInductionVar());
    b.create<memref::StoreOp>(loc, f0, Q, ValueRange{o});
  }
  auto dq = b.create<scf::ForOp>(loc, c0, M, c1);
  {
    OpBuilder::InsertionGuard g(b);
    b.setInsertionPointToStart(dq.getBody());
    Value i = dq.getInductionVar();
    b.create<memref::StoreOp>(loc, f1, Q, ValueRange{qOff(i, i)});
  }

  // k = min(m,n)
  Value mlt = b.create<arith::CmpIOp>(loc, slt, M, N);
  Value K = b.create<arith::SelectOp>(loc, mlt, M, N);

  auto jl = b.create<scf::ForOp>(loc, c0, K, c1);
  {
    OpBuilder::InsertionGuard g(b);
    b.setInsertionPointToStart(jl.getBody());
    Value j = jl.getInductionVar();
    Value jp1 = b.create<arith::AddIOp>(loc, j, c1);
    // norm² = Σ_{i≥j} R[i,j]²
    auto nl = b.create<scf::ForOp>(loc, j, M, c1, ValueRange{f0});
    {
      OpBuilder::InsertionGuard g2(b);
      b.setInsertionPointToStart(nl.getBody());
      Value i = nl.getInductionVar();
      Value rij = b.create<memref::LoadOp>(loc, R, ValueRange{rOff(i, j)});
      Value acc = b.create<arith::AddFOp>(
          loc, nl.getRegionIterArgs()[0],
          b.create<arith::MulFOp>(loc, rij, rij));
      b.create<scf::YieldOp>(loc, ValueRange{acc});
    }
    Value norm = b.create<math::SqrtOp>(loc, nl.getResult(0));
    Value nzNorm = b.create<arith::CmpFOp>(loc, arith::CmpFPredicate::ONE, norm,
                                           f0);
    auto refl = b.create<scf::IfOp>(loc, nzNorm, /*withElse=*/false);
    OpBuilder::InsertionGuard gIf(b);
    b.setInsertionPointToStart(refl.thenBlock());

    Value rjj = b.create<memref::LoadOp>(loc, R, ValueRange{rOff(j, j)});
    Value rjjNeg = b.create<arith::CmpFOp>(loc, arith::CmpFPredicate::OLT, rjj,
                                           f0);
    // alpha = (rjj >= 0) ? -norm : norm
    Value negNorm = b.create<arith::SubFOp>(loc, f0, norm);
    Value alpha = b.create<arith::SelectOp>(loc, rjjNeg, norm, negNorm);
    // v[i] = R[i,j] for i>j ; v[j] = rjj - alpha
    auto vl = b.create<scf::ForOp>(loc, j, M, c1);
    {
      OpBuilder::InsertionGuard g2(b);
      b.setInsertionPointToStart(vl.getBody());
      Value i = vl.getInductionVar();
      b.create<memref::StoreOp>(
          loc, b.create<memref::LoadOp>(loc, R, ValueRange{rOff(i, j)}),
          V, ValueRange{vOff(i)});
    }
    b.create<memref::StoreOp>(loc, b.create<arith::SubFOp>(loc, rjj, alpha), V,
                              ValueRange{vOff(j)});
    // vtv = Σ_{i≥j} v[i]²
    auto vtvl = b.create<scf::ForOp>(loc, j, M, c1, ValueRange{f0});
    {
      OpBuilder::InsertionGuard g2(b);
      b.setInsertionPointToStart(vtvl.getBody());
      Value i = vtvl.getInductionVar();
      Value vi = b.create<memref::LoadOp>(loc, V, ValueRange{vOff(i)});
      Value acc = b.create<arith::AddFOp>(
          loc, vtvl.getRegionIterArgs()[0],
          b.create<arith::MulFOp>(loc, vi, vi));
      b.create<scf::YieldOp>(loc, ValueRange{acc});
    }
    Value beta = b.create<arith::DivFOp>(loc, f2, vtvl.getResult(0));
    // R[j:, c] -= beta·(vᵀ R[j:, c])·v   for c in [j, n)
    auto rc = b.create<scf::ForOp>(loc, j, N, c1);
    {
      OpBuilder::InsertionGuard g2(b);
      b.setInsertionPointToStart(rc.getBody());
      Value c = rc.getInductionVar();
      auto dl = b.create<scf::ForOp>(loc, j, M, c1, ValueRange{f0});
      {
        OpBuilder::InsertionGuard g3(b);
        b.setInsertionPointToStart(dl.getBody());
        Value i = dl.getInductionVar();
        Value vi = b.create<memref::LoadOp>(loc, V, ValueRange{vOff(i)});
        Value ric = b.create<memref::LoadOp>(loc, R, ValueRange{rOff(i, c)});
        Value acc = b.create<arith::AddFOp>(
            loc, dl.getRegionIterArgs()[0],
            b.create<arith::MulFOp>(loc, vi, ric));
        b.create<scf::YieldOp>(loc, ValueRange{acc});
      }
      Value fac = b.create<arith::MulFOp>(loc, beta, dl.getResult(0));
      auto ul = b.create<scf::ForOp>(loc, j, M, c1);
      OpBuilder::InsertionGuard g3(b);
      b.setInsertionPointToStart(ul.getBody());
      Value i = ul.getInductionVar();
      Value vi = b.create<memref::LoadOp>(loc, V, ValueRange{vOff(i)});
      Value ric = b.create<memref::LoadOp>(loc, R, ValueRange{rOff(i, c)});
      Value upd = b.create<arith::SubFOp>(loc, ric,
                                          b.create<arith::MulFOp>(loc, fac, vi));
      b.create<memref::StoreOp>(loc, upd, R, ValueRange{rOff(i, c)});
    }
    // Q[r, j:] -= beta·(Q[r, j:]·v)·v   for r in [0, m)
    auto qr = b.create<scf::ForOp>(loc, c0, M, c1);
    {
      OpBuilder::InsertionGuard g2(b);
      b.setInsertionPointToStart(qr.getBody());
      Value r = qr.getInductionVar();
      auto dl = b.create<scf::ForOp>(loc, j, M, c1, ValueRange{f0});
      {
        OpBuilder::InsertionGuard g3(b);
        b.setInsertionPointToStart(dl.getBody());
        Value i = dl.getInductionVar();
        Value vi = b.create<memref::LoadOp>(loc, V, ValueRange{vOff(i)});
        Value qri = b.create<memref::LoadOp>(loc, Q, ValueRange{qOff(r, i)});
        Value acc = b.create<arith::AddFOp>(
            loc, dl.getRegionIterArgs()[0],
            b.create<arith::MulFOp>(loc, qri, vi));
        b.create<scf::YieldOp>(loc, ValueRange{acc});
      }
      Value fac = b.create<arith::MulFOp>(loc, beta, dl.getResult(0));
      auto ul = b.create<scf::ForOp>(loc, j, M, c1);
      OpBuilder::InsertionGuard g3(b);
      b.setInsertionPointToStart(ul.getBody());
      Value i = ul.getInductionVar();
      Value vi = b.create<memref::LoadOp>(loc, V, ValueRange{vOff(i)});
      Value qri = b.create<memref::LoadOp>(loc, Q, ValueRange{qOff(r, i)});
      Value upd = b.create<arith::SubFOp>(loc, qri,
                                          b.create<arith::MulFOp>(loc, fac, vi));
      b.create<memref::StoreOp>(loc, upd, Q, ValueRange{qOff(r, i)});
    }
    b.setInsertionPointAfter(refl);
    // zero strict-lower of column j (i>j)
    auto zl = b.create<scf::ForOp>(loc, jp1, M, c1);
    {
      OpBuilder::InsertionGuard g2(b);
      b.setInsertionPointToStart(zl.getBody());
      Value i = zl.getInductionVar();
      b.create<memref::StoreOp>(loc, f0, R, ValueRange{rOff(i, j)});
    }
  }
  b.setInsertionPointToEnd(&f.getBody().front());
  b.create<gpu::ReturnOp>(loc);
}

struct GenerateROCMQrKernelPass
    : PassWrapper<GenerateROCMQrKernelPass, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(GenerateROCMQrKernelPass)

  StringRef getArgument() const final { return "generate-rocm-qr-kernel"; }
  StringRef getDescription() const final {
    return "Expand a tessera_rocm.qr directive into a batched Householder QR gpu "
           "kernel (one thread per matrix)";
  }
  void getDependentDialects(DialectRegistry &registry) const final {
    registry.insert<gpu::GPUDialect, scf::SCFDialect, arith::ArithDialect,
                    math::MathDialect, memref::MemRefDialect>();
  }

  void runOnOperation() override {
    ModuleOp module = getOperation();
    SmallVector<Operation *> directives;
    module.walk([&](Operation *op) {
      if (op->getName().getStringRef() == "tessera_rocm.qr")
        directives.push_back(op);
    });
    for (Operation *op : directives) {
      auto nameAttr = op->getAttrOfType<StringAttr>("name");
      if (!nameAttr) {
        op->emitError("tessera_rocm.qr missing name");
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
          {memF32, memF32, memF32, memF32, idxTy, idxTy, idxTy}, {});
      auto gpuMod = b.create<gpu::GPUModuleOp>(loc, kname + "_mod");
      b.setInsertionPointToStart(&gpuMod.getBodyRegion().front());
      auto gpuFunc = b.create<gpu::GPUFuncOp>(loc, kname, fnTy);
      gpuFunc->setAttr(gpu::GPUDialect::getKernelFuncAttrName(), b.getUnitAttr());
      OpBuilder body(gpuFunc.getContext());
      emitQrBody(body, loc, gpuFunc);
      op->erase();
    }
  }
};

} // namespace

std::unique_ptr<mlir::Pass>
mlir::tessera_rocm::createGenerateROCMQrKernelPass() {
  return std::make_unique<GenerateROCMQrKernelPass>();
}
