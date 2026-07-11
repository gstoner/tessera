//===- GenerateROCMBatchedGemmF32Kernel.cpp - batched f32 GEMM gpu kernel -===//
//
// Expands `tessera_rocm.batched_gemm_f32` into a SINGLE-LAUNCH batched f32 GEMM,
// one thread per TM×TN output tile across ALL batches:
//
//   C[b, m, n] = Σ_k A[b, m, k] · B[b, k, n]
//   A: [Batch, M, K]  B: [Batch, K, N]  C: [Batch, M, N]   (contiguous)
//
// The single-GEMM `generate-rocm-gemm-f32-kernel` had to be looped per batch
// (one hipMalloc/H2D/launch/D2H round-trip each) — catastrophic for the chunked
// SSD scan, which issues many small bmms. This folds the batch into the grid:
// gid ∈ [0, Batch·ceil(M/TM)·ceil(N/TN)); the batch index selects per-batch base
// offsets (b·M·K / b·K·N / b·M·N) and the existing register-blocked tile logic
// runs unchanged. One launch for the whole batch → one H2D + one D2H at the
// caller. Correctness-first f32 (scalar k-loop, f32 accumulate). Batch/M/N/K are
// runtime index args. Validated vs numpy batched matmul on gfx1151.
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
static constexpr int64_t TM = 4;
static constexpr int64_t TN = 4;

void emitBatchedGemmF32Body(OpBuilder &b, Location loc, gpu::GPUFuncOp f) {
  Type f32 = b.getF32Type();
  auto slt = arith::CmpIPredicate::slt;
  b.setInsertionPointToStart(&f.getBody().front());
  Value A = f.getArgument(0), B = f.getArgument(1), C = f.getArgument(2);
  Value Batch = f.getArgument(3), M = f.getArgument(4), N = f.getArgument(5),
        K = f.getArgument(6);

  auto ci = [&](int64_t v) { return b.create<arith::ConstantIndexOp>(loc, v); };
  Value c0 = ci(0), c1 = ci(1);
  Value bid = b.create<gpu::BlockIdOp>(loc, gpu::Dimension::x);
  Value tid = b.create<gpu::ThreadIdOp>(loc, gpu::Dimension::x);
  Value gid = b.create<arith::AddIOp>(
      loc, b.create<arith::MulIOp>(loc, bid, ci(BD)), tid);
  Value nTilesN = b.create<arith::DivUIOp>(
      loc, b.create<arith::AddIOp>(loc, N, ci(TN - 1)), ci(TN));
  Value nTilesM = b.create<arith::DivUIOp>(
      loc, b.create<arith::AddIOp>(loc, M, ci(TM - 1)), ci(TM));
  Value tilesPerBatch = b.create<arith::MulIOp>(loc, nTilesM, nTilesN);
  Value total = b.create<arith::MulIOp>(loc, Batch, tilesPerBatch);
  Value inb = b.create<arith::CmpIOp>(loc, slt, gid, total);
  auto guard = b.create<scf::IfOp>(loc, inb, /*withElse=*/false);
  b.setInsertionPointToStart(guard.thenBlock());

  Value zero = b.create<arith::ConstantOp>(loc, f32, b.getF32FloatAttr(0.0f));
  // Decode the batch, then the tile within the batch.
  Value batch = b.create<arith::DivUIOp>(loc, gid, tilesPerBatch);
  Value local = b.create<arith::RemUIOp>(loc, gid, tilesPerBatch);
  // Per-batch base offsets into the flat A/B/C buffers.
  Value aBase = b.create<arith::MulIOp>(loc, batch,
                                        b.create<arith::MulIOp>(loc, M, K));
  Value bBase = b.create<arith::MulIOp>(loc, batch,
                                        b.create<arith::MulIOp>(loc, K, N));
  Value cBase = b.create<arith::MulIOp>(loc, batch,
                                        b.create<arith::MulIOp>(loc, M, N));

  Value tr = b.create<arith::DivUIOp>(loc, local, nTilesN);   // tile row
  Value tc = b.create<arith::RemUIOp>(loc, local, nTilesN);   // tile col
  Value m0 = b.create<arith::MulIOp>(loc, tr, ci(TM));
  Value n0 = b.create<arith::MulIOp>(loc, tc, ci(TN));

  // Per-row / per-col bounds + clamped A-row bases (batch-offset), hoisted.
  SmallVector<Value> mi(TM), inbM(TM), aRowBase(TM);
  for (int64_t i = 0; i < TM; ++i) {
    mi[i] = b.create<arith::AddIOp>(loc, m0, ci(i));
    inbM[i] = b.create<arith::CmpIOp>(loc, slt, mi[i], M);
    Value miSafe = b.create<arith::SelectOp>(loc, inbM[i], mi[i], c0);
    aRowBase[i] = b.create<arith::AddIOp>(
        loc, aBase, b.create<arith::MulIOp>(loc, miSafe, K));
  }
  SmallVector<Value> nj(TN), inbN(TN), njSafe(TN);
  for (int64_t j = 0; j < TN; ++j) {
    nj[j] = b.create<arith::AddIOp>(loc, n0, ci(j));
    inbN[j] = b.create<arith::CmpIOp>(loc, slt, nj[j], N);
    njSafe[j] = b.create<arith::SelectOp>(loc, inbN[j], nj[j], c0);
  }

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
    // B[b, k, n]: bBase + k*N + n.
    Value bRow = b.create<arith::AddIOp>(loc, bBase,
                                         b.create<arith::MulIOp>(loc, k, N));
    for (int64_t j = 0; j < TN; ++j) {
      Value v = b.create<memref::LoadOp>(
          loc, B, ValueRange{b.create<arith::AddIOp>(loc, bRow, njSafe[j])});
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
  auto res = kl.getResults();
  for (int64_t i = 0; i < TM; ++i)
    for (int64_t j = 0; j < TN; ++j) {
      Value inBoth = b.create<arith::AndIOp>(loc, inbM[i], inbN[j]);
      auto st = b.create<scf::IfOp>(loc, inBoth, /*withElse=*/false);
      OpBuilder::InsertionGuard g(b);
      b.setInsertionPointToStart(st.thenBlock());
      // C[b, m, n]: cBase + m*N + n.
      Value cidx = b.create<arith::AddIOp>(
          loc, cBase,
          b.create<arith::AddIOp>(
              loc, b.create<arith::MulIOp>(loc, mi[i], N), nj[j]));
      b.create<memref::StoreOp>(loc, res[i * TN + j], C, ValueRange{cidx});
    }
  b.setInsertionPointToEnd(&f.getBody().front());
  b.create<gpu::ReturnOp>(loc);
}

struct GenerateROCMBatchedGemmF32KernelPass
    : PassWrapper<GenerateROCMBatchedGemmF32KernelPass,
                  OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(
      GenerateROCMBatchedGemmF32KernelPass)

  StringRef getArgument() const final {
    return "generate-rocm-batched-gemm-f32-kernel";
  }
  StringRef getDescription() const final {
    return "Expand a tessera_rocm.batched_gemm_f32 directive into a single-launch "
           "batched f32 GEMM kernel (C[b]=A[b]@B[b], batch folded into the grid, "
           "register-blocked TMxTN tile per thread)";
  }
  void getDependentDialects(DialectRegistry &registry) const final {
    registry.insert<gpu::GPUDialect, scf::SCFDialect, arith::ArithDialect,
                    memref::MemRefDialect>();
  }

  void runOnOperation() override {
    ModuleOp module = getOperation();
    SmallVector<Operation *> directives;
    module.walk([&](Operation *op) {
      if (op->getName().getStringRef() == "tessera_rocm.batched_gemm_f32")
        directives.push_back(op);
    });
    for (Operation *op : directives) {
      auto nameAttr = op->getAttrOfType<StringAttr>("name");
      if (!nameAttr) {
        op->emitError("tessera_rocm.batched_gemm_f32 missing name");
        return signalPassFailure();
      }
      OpBuilder b(module.getBodyRegion());
      b.setInsertionPointToEnd(module.getBody());
      Location loc = op->getLoc();
      std::string kname = nameAttr.getValue().str();
      Type f32 = b.getF32Type();
      Type idxTy = b.getIndexType();
      auto memF32 = MemRefType::get({ShapedType::kDynamic}, f32);
      // (A, B, C : memref<?xf32>, Batch, M, N, K : index)
      auto fnTy = b.getFunctionType(
          {memF32, memF32, memF32, idxTy, idxTy, idxTy, idxTy}, {});
      auto gpuMod = b.create<gpu::GPUModuleOp>(loc, kname + "_mod");
      b.setInsertionPointToStart(&gpuMod.getBodyRegion().front());
      auto gpuFunc = b.create<gpu::GPUFuncOp>(loc, kname, fnTy);
      gpuFunc->setAttr(gpu::GPUDialect::getKernelFuncAttrName(), b.getUnitAttr());
      OpBuilder body(gpuFunc.getContext());
      emitBatchedGemmF32Body(body, loc, gpuFunc);
      op->erase();
    }
  }
};

} // namespace

std::unique_ptr<mlir::Pass>
mlir::tessera_rocm::createGenerateROCMBatchedGemmF32KernelPass() {
  return std::make_unique<GenerateROCMBatchedGemmF32KernelPass>();
}
