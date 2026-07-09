//===- GenerateROCMSelectiveSsmBwdKernel.cpp - Mamba2 backward scan -------===//
//
// Expands `tessera_rocm.selective_ssm_bwd` into the reverse-mode adjoint of the
// selective_ssm scan (matches autodiff/vjp.py::vjp_selective_ssm), one thread
// per (b,d) channel. First fills the forward trajectory h_traj (h_traj[0] = the
// caller's initial state), then walks t = S-1 → 0 accumulating (dx, dA2d, dB,
// dC, ddelta). The scan adjoint dh[b,d,:] is a per-thread global scratch slice.
// dx/ddelta are unique per (b,t,d) (written once); dC/dB reduce over channels d
// and dA2d reduces over b,t, so those use memref.atomic_rmw addf. All f32.
// CPU analog: tessera_x86_selective_ssm_bwd_f32.
//
// Args: (X, A2d, B, C, DELTA, DY : f32, H_TRAJ, DH : f32 scratch,
//        DX, DA2D, DB, DC, DDELTA : f32, Bsz, S, D, N : index).
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

void emitSsmBwdBody(OpBuilder &b, Location loc, gpu::GPUFuncOp f) {
  Type f32 = b.getF32Type();
  auto slt = arith::CmpIPredicate::slt;
  b.setInsertionPointToStart(&f.getBody().front());
  Value X = f.getArgument(0), A = f.getArgument(1), B = f.getArgument(2);
  Value C = f.getArgument(3), DELTA = f.getArgument(4), DY = f.getArgument(5);
  Value HT = f.getArgument(6), DH = f.getArgument(7);
  Value DX = f.getArgument(8), DA = f.getArgument(9), DB = f.getArgument(10);
  Value DC = f.getArgument(11), DDL = f.getArgument(12);
  Value Bsz = f.getArgument(13), S = f.getArgument(14), D = f.getArgument(15);
  Value N = f.getArgument(16);

  auto ci = [&](int64_t v) { return b.create<arith::ConstantIndexOp>(loc, v); };
  auto muli = [&](Value x, Value y) { return b.create<arith::MulIOp>(loc, x, y); };
  auto addi = [&](Value x, Value y) { return b.create<arith::AddIOp>(loc, x, y); };
  auto mulf = [&](Value x, Value y) { return b.create<arith::MulFOp>(loc, x, y); };
  auto addf = [&](Value x, Value y) { return b.create<arith::AddFOp>(loc, x, y); };
  auto ld = [&](Value m, Value i) {
    return b.create<memref::LoadOp>(loc, m, ValueRange{i}).getResult();
  };
  auto atomAdd = [&](Value v, Value m, Value i) {
    b.create<memref::AtomicRMWOp>(loc, arith::AtomicRMWKind::addf, v, m,
                                  ValueRange{i});
  };

  Value c0 = ci(0), c1 = ci(1);
  Value zero = b.create<arith::ConstantOp>(loc, f32, b.getF32FloatAttr(0.0f));
  Value bid = b.create<gpu::BlockIdOp>(loc, gpu::Dimension::x);
  Value tid = b.create<gpu::ThreadIdOp>(loc, gpu::Dimension::x);
  Value gid = addi(muli(bid, ci(BD)), tid);
  Value total = muli(Bsz, D);
  auto guard = b.create<scf::IfOp>(loc, b.create<arith::CmpIOp>(loc, slt, gid, total),
                                   /*withElse=*/false);
  b.setInsertionPointToStart(guard.thenBlock());

  Value bb = b.create<arith::DivUIOp>(loc, gid, D);   // batch
  Value dd = b.create<arith::RemUIOp>(loc, gid, D);   // channel
  Value BDN = muli(total, N);                          // Bsz*D*N
  Value gN = muli(gid, N);                             // gid*N
  Value abase = muli(dd, N);                           // d*N

  // ── Phase 1 — forward fill h_traj[t+1] = A_bar·h_traj[t] + B_bar·x ──
  {
    auto tl = b.create<scf::ForOp>(loc, c0, S, c1);
    OpBuilder::InsertionGuard g(b);
    b.setInsertionPointToStart(tl.getBody());
    Value t = tl.getInductionVar();
    Value row = addi(muli(bb, S), t);
    Value xoff = addi(muli(row, D), dd);
    Value dt = ld(DELTA, xoff), xt = ld(X, xoff);
    Value bcbase = muli(row, N);
    Value hb0 = addi(muli(t, BDN), gN);
    Value hb1 = addi(muli(addi(t, c1), BDN), gN);
    auto nl = b.create<scf::ForOp>(loc, c0, N, c1);
    OpBuilder::InsertionGuard g2(b);
    b.setInsertionPointToStart(nl.getBody());
    Value n = nl.getInductionVar();
    Value abar = b.create<math::ExpOp>(loc, mulf(dt, ld(A, addi(abase, n))));
    Value bbar = mulf(dt, ld(B, addi(bcbase, n)));
    Value hv = addf(mulf(abar, ld(HT, addi(hb0, n))), mulf(bbar, xt));
    b.create<memref::StoreOp>(loc, hv, HT, ValueRange{addi(hb1, n)});
  }

  // ── Phase 2 — reverse: tp = 0..S-1, actual t = (S-1) - tp ──
  {
    Value Sm1 = b.create<arith::SubIOp>(loc, S, c1);
    auto tl = b.create<scf::ForOp>(loc, c0, S, c1);
    OpBuilder::InsertionGuard g(b);
    b.setInsertionPointToStart(tl.getBody());
    Value t = b.create<arith::SubIOp>(loc, Sm1, tl.getInductionVar());
    Value row = addi(muli(bb, S), t);
    Value xoff = addi(muli(row, D), dd);
    Value dt = ld(DELTA, xoff), xt = ld(X, xoff), dyt = ld(DY, xoff);
    Value bcbase = muli(row, N);
    Value hb0 = addi(muli(t, BDN), gN);              // h[t-1]
    Value hb1 = addi(muli(addi(t, c1), BDN), gN);    // h[t]
    auto nl = b.create<scf::ForOp>(loc, c0, N, c1, ValueRange{zero, zero});
    {
      OpBuilder::InsertionGuard g2(b);
      b.setInsertionPointToStart(nl.getBody());
      Value n = nl.getInductionVar();
      Value dxacc = nl.getRegionIterArgs()[0];
      Value ddlacc = nl.getRegionIterArgs()[1];
      Value a = ld(A, addi(abase, n));
      Value bv = ld(B, addi(bcbase, n));
      Value cv = ld(C, addi(bcbase, n));
      Value abar = b.create<math::ExpOp>(loc, mulf(dt, a));
      Value bbar = mulf(dt, bv);
      Value dhc = addf(ld(DH, addi(gN, n)), mulf(cv, dyt));
      // dC += h[t]·dy   (reduce over d)
      atomAdd(mulf(ld(HT, addi(hb1, n)), dyt), DC, addi(bcbase, n));
      Value dAbar = mulf(dhc, ld(HT, addi(hb0, n)));
      Value dhprev = mulf(dhc, abar);
      Value dBbar = mulf(dhc, xt);
      Value ndx = addf(dxacc, mulf(dhc, bbar));
      // dB += dBbar·delta   (reduce over d)
      atomAdd(mulf(dBbar, dt), DB, addi(bcbase, n));
      Value ddl = addf(ddlacc, mulf(dBbar, bv));
      Value dz = mulf(dAbar, abar);
      // dA2d += delta·dz   (reduce over b,t)
      atomAdd(mulf(dt, dz), DA, addi(abase, n));
      ddl = addf(ddl, mulf(dz, a));
      b.create<memref::StoreOp>(loc, dhprev, DH, ValueRange{addi(gN, n)});
      b.create<scf::YieldOp>(loc, ValueRange{ndx, ddl});
    }
    b.create<memref::StoreOp>(loc, nl.getResult(0), DX, ValueRange{xoff});
    b.create<memref::StoreOp>(loc, nl.getResult(1), DDL, ValueRange{xoff});
  }

  b.setInsertionPointToEnd(&f.getBody().front());
  b.create<gpu::ReturnOp>(loc);
}

struct GenerateROCMSelectiveSsmBwdKernelPass
    : PassWrapper<GenerateROCMSelectiveSsmBwdKernelPass, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(
      GenerateROCMSelectiveSsmBwdKernelPass)

  StringRef getArgument() const final {
    return "generate-rocm-selective-ssm-bwd-kernel";
  }
  StringRef getDescription() const final {
    return "Expand a tessera_rocm.selective_ssm_bwd directive into the Mamba2 "
           "selective-scan reverse-mode adjoint gpu kernel (one thread per "
           "(b,d), atomic cross-channel reductions)";
  }
  void getDependentDialects(DialectRegistry &registry) const final {
    registry.insert<gpu::GPUDialect, scf::SCFDialect, arith::ArithDialect,
                    math::MathDialect, memref::MemRefDialect>();
  }

  void runOnOperation() override {
    ModuleOp module = getOperation();
    SmallVector<Operation *> directives;
    module.walk([&](Operation *op) {
      if (op->getName().getStringRef() == "tessera_rocm.selective_ssm_bwd")
        directives.push_back(op);
    });
    for (Operation *op : directives) {
      auto nameAttr = op->getAttrOfType<StringAttr>("name");
      if (!nameAttr) {
        op->emitError("tessera_rocm.selective_ssm_bwd missing name");
        return signalPassFailure();
      }
      OpBuilder b(module.getBodyRegion());
      b.setInsertionPointToEnd(module.getBody());
      Location loc = op->getLoc();
      std::string kname = nameAttr.getValue().str();
      Type f32 = b.getF32Type();
      Type idxTy = b.getIndexType();
      auto m = MemRefType::get({ShapedType::kDynamic}, f32);
      // 13 f32 memrefs + 4 index dims.
      SmallVector<Type> argTys(13, m);
      argTys.append({idxTy, idxTy, idxTy, idxTy});
      auto fnTy = b.getFunctionType(argTys, {});
      auto gpuMod = b.create<gpu::GPUModuleOp>(loc, kname + "_mod");
      b.setInsertionPointToStart(&gpuMod.getBodyRegion().front());
      auto gpuFunc = b.create<gpu::GPUFuncOp>(loc, kname, fnTy);
      gpuFunc->setAttr(gpu::GPUDialect::getKernelFuncAttrName(), b.getUnitAttr());
      OpBuilder body(gpuFunc.getContext());
      emitSsmBwdBody(body, loc, gpuFunc);
      op->erase();
    }
  }
};

} // namespace

std::unique_ptr<mlir::Pass>
mlir::tessera_rocm::createGenerateROCMSelectiveSsmBwdKernelPass() {
  return std::make_unique<GenerateROCMSelectiveSsmBwdKernelPass>();
}
