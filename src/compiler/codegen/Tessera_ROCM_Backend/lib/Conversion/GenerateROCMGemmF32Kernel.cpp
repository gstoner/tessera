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
// Output-tile register blocking: each thread computes a TM×TN tile of C. Per
// k-step it loads TM A-values + TN B-values from global and reuses them across
// TM*TN FMAs (each A elt reused TN times, each B elt TM times) — the arithmetic-
// intensity / register-budget lever that wins on Strix Halo's unified memory
// (STRIX_HALO_EXECUTION_PLAN Stage F: "register-budget tiling is the lever").
void emitGemmF32Body(OpBuilder &b, Location loc, gpu::GPUFuncOp f,
                     int64_t TM, int64_t TN) {
  Type f32 = b.getF32Type();
  auto slt = arith::CmpIPredicate::slt;
  b.setInsertionPointToStart(&f.getBody().front());
  Value A = f.getArgument(0), B = f.getArgument(1), C = f.getArgument(2);
  Value M = f.getArgument(3), N = f.getArgument(4), K = f.getArgument(5);

  auto ci = [&](int64_t v) { return b.create<arith::ConstantIndexOp>(loc, v); };
  Value c0 = ci(0), c1 = ci(1);
  Value bid = b.create<gpu::BlockIdOp>(loc, gpu::Dimension::x);
  Value tid = b.create<gpu::ThreadIdOp>(loc, gpu::Dimension::x);
  Value gid = b.create<arith::AddIOp>(
      loc, b.create<arith::MulIOp>(loc, bid, ci(BD)), tid);
  // One thread per TM×TN output tile: nTilesN = ceil(N/TN), total = ceil(M/TM)*.
  Value nTilesN = b.create<arith::DivUIOp>(
      loc, b.create<arith::AddIOp>(loc, N, ci(TN - 1)), ci(TN));
  Value nTilesM = b.create<arith::DivUIOp>(
      loc, b.create<arith::AddIOp>(loc, M, ci(TM - 1)), ci(TM));
  Value total = b.create<arith::MulIOp>(loc, nTilesM, nTilesN);
  Value inb = b.create<arith::CmpIOp>(loc, slt, gid, total);
  auto guard = b.create<scf::IfOp>(loc, inb, /*withElse=*/false);
  b.setInsertionPointToStart(guard.thenBlock());

  Value zero = b.create<arith::ConstantOp>(loc, f32, b.getF32FloatAttr(0.0f));
  Value tr = b.create<arith::DivUIOp>(loc, gid, nTilesN);   // tile row
  Value tc = b.create<arith::RemUIOp>(loc, gid, nTilesN);   // tile col
  Value m0 = b.create<arith::MulIOp>(loc, tr, ci(TM));
  Value n0 = b.create<arith::MulIOp>(loc, tc, ci(TN));

  // Per-row (i) and per-col (j) bounds + safe (clamped) indices, hoisted out of
  // the k-loop. OOB rows/cols contribute 0 (masked load) and are not stored.
  SmallVector<Value> mi(TM), inbM(TM), aRowBase(TM);
  for (int64_t i = 0; i < TM; ++i) {
    mi[i] = b.create<arith::AddIOp>(loc, m0, ci(i));
    inbM[i] = b.create<arith::CmpIOp>(loc, slt, mi[i], M);
    Value miSafe = b.create<arith::SelectOp>(loc, inbM[i], mi[i], c0);
    aRowBase[i] = b.create<arith::MulIOp>(loc, miSafe, K);
  }
  SmallVector<Value> nj(TN), inbN(TN), njSafe(TN);
  for (int64_t j = 0; j < TN; ++j) {
    nj[j] = b.create<arith::AddIOp>(loc, n0, ci(j));
    inbN[j] = b.create<arith::CmpIOp>(loc, slt, nj[j], N);
    njSafe[j] = b.create<arith::SelectOp>(loc, inbN[j], nj[j], c0);
  }

  // k-loop with TM*TN register accumulators.
  SmallVector<Value> initAcc(TM * TN, zero);
  auto kl = b.create<scf::ForOp>(loc, c0, K, c1, initAcc);
  {
    OpBuilder::InsertionGuard g(b);
    b.setInsertionPointToStart(kl.getBody());
    Value k = kl.getInductionVar();
    SmallVector<Value> acc(kl.getRegionIterArgs().begin(),
                           kl.getRegionIterArgs().end());
    SmallVector<Value> a(TM), bcol(TN);
    for (int64_t i = 0; i < TM; ++i) {
      Value v = b.create<memref::LoadOp>(
          loc, A, ValueRange{b.create<arith::AddIOp>(loc, aRowBase[i], k)});
      a[i] = b.create<arith::SelectOp>(loc, inbM[i], v, zero);
    }
    Value kN = b.create<arith::MulIOp>(loc, k, N);
    for (int64_t j = 0; j < TN; ++j) {
      Value v = b.create<memref::LoadOp>(
          loc, B, ValueRange{b.create<arith::AddIOp>(loc, kN, njSafe[j])});
      bcol[j] = b.create<arith::SelectOp>(loc, inbN[j], v, zero);
    }
    SmallVector<Value> newAcc(TM * TN);
    for (int64_t i = 0; i < TM; ++i)
      for (int64_t j = 0; j < TN; ++j)
        newAcc[i * TN + j] = b.create<arith::AddFOp>(
            loc, acc[i * TN + j],
            b.create<arith::MulFOp>(loc, a[i], bcol[j]));
    b.create<scf::YieldOp>(loc, newAcc);
  }
  // Store the tile, guarded on both bounds.
  auto res = kl.getResults();
  for (int64_t i = 0; i < TM; ++i)
    for (int64_t j = 0; j < TN; ++j) {
      Value inBoth = b.create<arith::AndIOp>(loc, inbM[i], inbN[j]);
      auto st = b.create<scf::IfOp>(loc, inBoth, /*withElse=*/false);
      OpBuilder::InsertionGuard g(b);
      b.setInsertionPointToStart(st.thenBlock());
      Value cidx = b.create<arith::AddIOp>(
          loc, b.create<arith::MulIOp>(loc, mi[i], N), nj[j]);
      b.create<memref::StoreOp>(loc, res[i * TN + j], C, ValueRange{cidx});
    }
  b.setInsertionPointToEnd(&f.getBody().front());
  b.create<gpu::ReturnOp>(loc);
}

struct GenerateROCMGemmF32KernelPass
    : PassWrapper<GenerateROCMGemmF32KernelPass, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(GenerateROCMGemmF32KernelPass)

  StringRef getArgument() const final { return "generate-rocm-gemm-f32-kernel"; }
  StringRef getDescription() const final {
    return "Expand a tessera_rocm.gemm_f32 directive into an f32 GEMM kernel "
           "(C=A@B, configurable register-blocked TMxTN output tile per thread, "
           "f32 k-loop)";
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
      int64_t tm = 4, tn = 4;
      if (auto attr = op->getAttrOfType<IntegerAttr>("tm"))
        tm = attr.getInt();
      if (auto attr = op->getAttrOfType<IntegerAttr>("tn"))
        tn = attr.getInt();
      if (tm < 1 || tn < 1 || tm * tn > 32) {
        op->emitError("tessera_rocm.gemm_f32 requires tm/tn >= 1 and "
                      "tm*tn <= 32 (got ")
            << tm << "x" << tn << ")";
        return signalPassFailure();
      }
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
      gpuFunc->setAttr("tessera.rocm.tm", b.getI64IntegerAttr(tm));
      gpuFunc->setAttr("tessera.rocm.tn", b.getI64IntegerAttr(tn));
      emitGemmF32Body(body, loc, gpuFunc, tm, tn);
      op->erase();
    }
  }
};

} // namespace

std::unique_ptr<mlir::Pass>
mlir::tessera_rocm::createGenerateROCMGemmF32KernelPass() {
  return std::make_unique<GenerateROCMGemmF32KernelPass>();
}
