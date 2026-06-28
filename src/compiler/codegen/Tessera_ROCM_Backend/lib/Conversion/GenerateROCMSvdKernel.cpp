//===- GenerateROCMSvdKernel.cpp - batched one-sided Jacobi SVD kernel ----===//
//
// Expands `tessera_rocm.svd` into a batched one-sided Jacobi SVD, one thread per
// matrix (requires m≥n; the runtime transposes the wide case):
//
//   A[m,n] = U[m,n]·diag(S[n])·Vᵀ[n,n]
//
// Each thread orthogonalizes the columns of a COLUMN-MAJOR working copy (global
// scratch UC[b·n·m]) by fixed sweeps of 2×2 column rotations, accumulating V in
// scratch VC[b·n·n]; afterwards S = column norms, U = normalized columns, Vt = V
// (already in Vᵀ layout), columns sorted by descending singular value. sqrt via
// math→ROCDL. All f32. CPU analog: avx512_svd_f32.
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
static constexpr int64_t kSweeps = 30;

void emitSvdBody(OpBuilder &b, Location loc, gpu::GPUFuncOp f) {
  Type f32 = b.getF32Type();
  auto slt = arith::CmpIPredicate::slt;
  b.setInsertionPointToStart(&f.getBody().front());
  Value A = f.getArgument(0), U = f.getArgument(1), S = f.getArgument(2);
  Value Vt = f.getArgument(3), UC = f.getArgument(4), VC = f.getArgument(5);
  Value batch = f.getArgument(6), M = f.getArgument(7), N = f.getArgument(8);

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
  Value tol = b.create<arith::ConstantOp>(loc, f32, b.getF32FloatAttr(1e-12f));
  Value abase = b.create<arith::MulIOp>(loc, gid, b.create<arith::MulIOp>(loc, M, N));
  Value ucbase = b.create<arith::MulIOp>(loc, gid, b.create<arith::MulIOp>(loc, N, M));
  Value vcbase = b.create<arith::MulIOp>(loc, gid, b.create<arith::MulIOp>(loc, N, N));
  Value sbase = b.create<arith::MulIOp>(loc, gid, N);

  auto ucOff = [&](Value col, Value i) {  // column-major: col*M + i
    return b.create<arith::AddIOp>(
        loc, ucbase, b.create<arith::AddIOp>(
                         loc, b.create<arith::MulIOp>(loc, col, M), i)).getResult();
  };
  auto vcOff = [&](Value col, Value i) {  // column-major: col*N + i
    return b.create<arith::AddIOp>(
        loc, vcbase, b.create<arith::AddIOp>(
                         loc, b.create<arith::MulIOp>(loc, col, N), i)).getResult();
  };
  // dot of two columns (base offsets ob1, ob2) over len
  auto colDot = [&](Value ob1, Value ob2, Value len) -> Value {
    auto dl = b.create<scf::ForOp>(loc, c0, len, c1, ValueRange{f0});
    OpBuilder::InsertionGuard g(b);
    b.setInsertionPointToStart(dl.getBody());
    Value i = dl.getInductionVar();
    Value x = b.create<memref::LoadOp>(
        loc, UC, ValueRange{b.create<arith::AddIOp>(loc, ob1, i)});
    Value y = b.create<memref::LoadOp>(
        loc, UC, ValueRange{b.create<arith::AddIOp>(loc, ob2, i)});
    Value acc = b.create<arith::AddFOp>(loc, dl.getRegionIterArgs()[0],
                                        b.create<arith::MulFOp>(loc, x, y));
    b.create<scf::YieldOp>(loc, ValueRange{acc});
    return dl.getResult(0);
  };
  // rotate two columns of a buffer BUF: (x,y) <- (c·x - s·y, s·x + c·y)
  auto rotate = [&](Value BUF, Value ob1, Value ob2, Value c, Value s,
                    Value len) {
    auto rl = b.create<scf::ForOp>(loc, c0, len, c1);
    OpBuilder::InsertionGuard g(b);
    b.setInsertionPointToStart(rl.getBody());
    Value i = rl.getInductionVar();
    Value o1 = b.create<arith::AddIOp>(loc, ob1, i);
    Value o2 = b.create<arith::AddIOp>(loc, ob2, i);
    Value x = b.create<memref::LoadOp>(loc, BUF, ValueRange{o1});
    Value y = b.create<memref::LoadOp>(loc, BUF, ValueRange{o2});
    Value nx = b.create<arith::SubFOp>(loc, b.create<arith::MulFOp>(loc, c, x),
                                       b.create<arith::MulFOp>(loc, s, y));
    Value ny = b.create<arith::AddFOp>(loc, b.create<arith::MulFOp>(loc, s, x),
                                       b.create<arith::MulFOp>(loc, c, y));
    b.create<memref::StoreOp>(loc, nx, BUF, ValueRange{o1});
    b.create<memref::StoreOp>(loc, ny, BUF, ValueRange{o2});
  };

  // uc[col j] = A[:, j] (column-major) ; vc = I
  auto initU = b.create<scf::ForOp>(loc, c0, N, c1);
  {
    OpBuilder::InsertionGuard g(b);
    b.setInsertionPointToStart(initU.getBody());
    Value j = initU.getInductionVar();
    auto il = b.create<scf::ForOp>(loc, c0, M, c1);
    OpBuilder::InsertionGuard g2(b);
    b.setInsertionPointToStart(il.getBody());
    Value i = il.getInductionVar();
    Value aoff = b.create<arith::AddIOp>(
        loc, abase, b.create<arith::AddIOp>(
                        loc, b.create<arith::MulIOp>(loc, i, N), j));
    b.create<memref::StoreOp>(loc, b.create<memref::LoadOp>(loc, A, ValueRange{aoff}),
                              UC, ValueRange{ucOff(j, i)});
  }
  auto initV = b.create<scf::ForOp>(loc, c0, N, c1);
  {
    OpBuilder::InsertionGuard g(b);
    b.setInsertionPointToStart(initV.getBody());
    Value j = initV.getInductionVar();
    auto il = b.create<scf::ForOp>(loc, c0, N, c1);
    OpBuilder::InsertionGuard g2(b);
    b.setInsertionPointToStart(il.getBody());
    Value i = il.getInductionVar();
    Value eq = b.create<arith::CmpIOp>(loc, arith::CmpIPredicate::eq, i, j);
    Value val = b.create<arith::SelectOp>(loc, eq, f1, f0);
    b.create<memref::StoreOp>(loc, val, VC, ValueRange{vcOff(j, i)});
  }

  // Jacobi sweeps
  Value cSweeps = b.create<arith::ConstantIndexOp>(loc, kSweeps);
  auto sweepL = b.create<scf::ForOp>(loc, c0, cSweeps, c1);
  {
    OpBuilder::InsertionGuard g(b);
    b.setInsertionPointToStart(sweepL.getBody());
    auto pL = b.create<scf::ForOp>(loc, c0, N, c1);
    OpBuilder::InsertionGuard g2(b);
    b.setInsertionPointToStart(pL.getBody());
    Value p = pL.getInductionVar();
    Value pp1 = b.create<arith::AddIOp>(loc, p, c1);
    auto qL = b.create<scf::ForOp>(loc, pp1, N, c1);
    OpBuilder::InsertionGuard g3(b);
    b.setInsertionPointToStart(qL.getBody());
    Value q = qL.getInductionVar();
    Value obp = ucOff(p, c0), obq = ucOff(q, c0);
    Value app = colDot(obp, obp, M);
    Value aqq = colDot(obq, obq, M);
    Value apq = colDot(obp, obq, M);
    Value thresh = b.create<arith::MulFOp>(
        loc, tol, b.create<math::SqrtOp>(loc, b.create<arith::MulFOp>(loc, app, aqq)));
    Value sig = b.create<arith::CmpFOp>(loc, arith::CmpFPredicate::OGT,
                                        b.create<math::AbsFOp>(loc, apq), thresh);
    auto doRot = b.create<scf::IfOp>(loc, sig, /*withElse=*/false);
    OpBuilder::InsertionGuard g4(b);
    b.setInsertionPointToStart(doRot.thenBlock());
    // tau = (aqq-app)/(2 apq)
    Value tau = b.create<arith::DivFOp>(
        loc, b.create<arith::SubFOp>(loc, aqq, app),
        b.create<arith::MulFOp>(loc, f2, apq));
    // t = sign(tau)/(|tau| + sqrt(tau²+1))
    Value tauNN = b.create<arith::CmpFOp>(loc, arith::CmpFPredicate::OGE, tau, f0);
    Value sgn = b.create<arith::SelectOp>(loc, tauNN, f1,
                                          b.create<arith::SubFOp>(loc, f0, f1));
    Value den = b.create<arith::AddFOp>(
        loc, b.create<math::AbsFOp>(loc, tau),
        b.create<math::SqrtOp>(loc, b.create<arith::AddFOp>(
            loc, b.create<arith::MulFOp>(loc, tau, tau), f1)));
    Value t = b.create<arith::DivFOp>(loc, sgn, den);
    Value c = b.create<arith::DivFOp>(
        loc, f1, b.create<math::SqrtOp>(loc, b.create<arith::AddFOp>(
                     loc, b.create<arith::MulFOp>(loc, t, t), f1)));
    Value s = b.create<arith::MulFOp>(loc, t, c);
    rotate(UC, obp, obq, c, s, M);
    rotate(VC, vcOff(p, c0), vcOff(q, c0), c, s, N);
  }

  // S[j] = ‖uc col j‖
  auto sL = b.create<scf::ForOp>(loc, c0, N, c1);
  {
    OpBuilder::InsertionGuard g(b);
    b.setInsertionPointToStart(sL.getBody());
    Value j = sL.getInductionVar();
    Value ob = ucOff(j, c0);
    Value nrm = b.create<math::SqrtOp>(loc, colDot(ob, ob, M));
    b.create<memref::StoreOp>(loc, nrm, S,
                              ValueRange{b.create<arith::AddIOp>(loc, sbase, j)});
  }
  // selection sort columns by descending S
  auto sortL = b.create<scf::ForOp>(loc, c0, N, c1);
  {
    OpBuilder::InsertionGuard g(b);
    b.setInsertionPointToStart(sortL.getBody());
    Value i = sortL.getInductionVar();
    Value si = b.create<memref::LoadOp>(
        loc, S, ValueRange{b.create<arith::AddIOp>(loc, sbase, i)});
    Value ip1 = b.create<arith::AddIOp>(loc, i, c1);
    // argmax over [i+1, n)
    auto am = b.create<scf::ForOp>(loc, ip1, N, c1, ValueRange{si, i});
    {
      OpBuilder::InsertionGuard g2(b);
      b.setInsertionPointToStart(am.getBody());
      Value j = am.getInductionVar();
      Value sj = b.create<memref::LoadOp>(
          loc, S, ValueRange{b.create<arith::AddIOp>(loc, sbase, j)});
      Value gt = b.create<arith::CmpFOp>(loc, arith::CmpFPredicate::OGT, sj,
                                         am.getRegionIterArgs()[0]);
      Value nb = b.create<arith::SelectOp>(loc, gt, sj, am.getRegionIterArgs()[0]);
      Value ni = b.create<arith::SelectOp>(loc, gt, j, am.getRegionIterArgs()[1]);
      b.create<scf::YieldOp>(loc, ValueRange{nb, ni});
    }
    Value mx = am.getResult(1);
    Value neq = b.create<arith::CmpIOp>(loc, arith::CmpIPredicate::ne, mx, i);
    auto sw = b.create<scf::IfOp>(loc, neq, /*withElse=*/false);
    OpBuilder::InsertionGuard g2(b);
    b.setInsertionPointToStart(sw.thenBlock());
    // swap S[i], S[mx]
    Value oi = b.create<arith::AddIOp>(loc, sbase, i);
    Value omx = b.create<arith::AddIOp>(loc, sbase, mx);
    Value vi = b.create<memref::LoadOp>(loc, S, ValueRange{oi});
    Value vmx = b.create<memref::LoadOp>(loc, S, ValueRange{omx});
    b.create<memref::StoreOp>(loc, vmx, S, ValueRange{oi});
    b.create<memref::StoreOp>(loc, vi, S, ValueRange{omx});
    // swap uc cols i,mx (len m) and vc cols i,mx (len n)
    auto swapCols = [&](Value BUF, Value o1f, Value o2f, Value len) {
      auto l = b.create<scf::ForOp>(loc, c0, len, c1);
      OpBuilder::InsertionGuard g3(b);
      b.setInsertionPointToStart(l.getBody());
      Value r = l.getInductionVar();
      Value a1 = b.create<arith::AddIOp>(loc, o1f, r);
      Value a2 = b.create<arith::AddIOp>(loc, o2f, r);
      Value t1 = b.create<memref::LoadOp>(loc, BUF, ValueRange{a1});
      Value t2 = b.create<memref::LoadOp>(loc, BUF, ValueRange{a2});
      b.create<memref::StoreOp>(loc, t2, BUF, ValueRange{a1});
      b.create<memref::StoreOp>(loc, t1, BUF, ValueRange{a2});
    };
    swapCols(UC, ucOff(i, c0), ucOff(mx, c0), M);
    swapCols(VC, vcOff(i, c0), vcOff(mx, c0), N);
  }
  // U[i,j] = uc[col j][i] / S[j] ; Vt = VC
  auto outL = b.create<scf::ForOp>(loc, c0, N, c1);
  {
    OpBuilder::InsertionGuard g(b);
    b.setInsertionPointToStart(outL.getBody());
    Value j = outL.getInductionVar();
    Value sj = b.create<memref::LoadOp>(
        loc, S, ValueRange{b.create<arith::AddIOp>(loc, sbase, j)});
    Value tiny = b.create<arith::ConstantOp>(loc, f32, b.getF32FloatAttr(1e-20f));
    Value big = b.create<arith::CmpFOp>(loc, arith::CmpFPredicate::OGT, sj, tiny);
    Value inv = b.create<arith::SelectOp>(loc, big,
                                          b.create<arith::DivFOp>(loc, f1, sj), f0);
    auto il = b.create<scf::ForOp>(loc, c0, M, c1);
    OpBuilder::InsertionGuard g2(b);
    b.setInsertionPointToStart(il.getBody());
    Value i = il.getInductionVar();
    Value uval = b.create<arith::MulFOp>(
        loc, b.create<memref::LoadOp>(loc, UC, ValueRange{ucOff(j, i)}), inv);
    Value uoff = b.create<arith::AddIOp>(
        loc, abase, b.create<arith::AddIOp>(
                        loc, b.create<arith::MulIOp>(loc, i, N), j));
    b.create<memref::StoreOp>(loc, uval, U, ValueRange{uoff});
  }
  auto vtL = b.create<scf::ForOp>(loc, c0, b.create<arith::MulIOp>(loc, N, N), c1);
  {
    OpBuilder::InsertionGuard g(b);
    b.setInsertionPointToStart(vtL.getBody());
    Value idx = vtL.getInductionVar();
    Value o = b.create<arith::AddIOp>(loc, vcbase, idx);
    b.create<memref::StoreOp>(loc, b.create<memref::LoadOp>(loc, VC, ValueRange{o}),
                              Vt, ValueRange{o});
  }
  b.setInsertionPointToEnd(&f.getBody().front());
  b.create<gpu::ReturnOp>(loc);
}

struct GenerateROCMSvdKernelPass
    : PassWrapper<GenerateROCMSvdKernelPass, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(GenerateROCMSvdKernelPass)

  StringRef getArgument() const final { return "generate-rocm-svd-kernel"; }
  StringRef getDescription() const final {
    return "Expand a tessera_rocm.svd directive into a batched one-sided Jacobi "
           "SVD gpu kernel (one thread per matrix, m>=n)";
  }
  void getDependentDialects(DialectRegistry &registry) const final {
    registry.insert<gpu::GPUDialect, scf::SCFDialect, arith::ArithDialect,
                    math::MathDialect, memref::MemRefDialect>();
  }

  void runOnOperation() override {
    ModuleOp module = getOperation();
    SmallVector<Operation *> directives;
    module.walk([&](Operation *op) {
      if (op->getName().getStringRef() == "tessera_rocm.svd")
        directives.push_back(op);
    });
    for (Operation *op : directives) {
      auto nameAttr = op->getAttrOfType<StringAttr>("name");
      if (!nameAttr) {
        op->emitError("tessera_rocm.svd missing name");
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
          {memF32, memF32, memF32, memF32, memF32, memF32, idxTy, idxTy, idxTy},
          {});
      auto gpuMod = b.create<gpu::GPUModuleOp>(loc, kname + "_mod");
      b.setInsertionPointToStart(&gpuMod.getBodyRegion().front());
      auto gpuFunc = b.create<gpu::GPUFuncOp>(loc, kname, fnTy);
      gpuFunc->setAttr(gpu::GPUDialect::getKernelFuncAttrName(), b.getUnitAttr());
      OpBuilder body(gpuFunc.getContext());
      emitSvdBody(body, loc, gpuFunc);
      op->erase();
    }
  }
};

} // namespace

std::unique_ptr<mlir::Pass>
mlir::tessera_rocm::createGenerateROCMSvdKernelPass() {
  return std::make_unique<GenerateROCMSvdKernelPass>();
}
