//===- GenerateROCMSortKernel.cpp - cooperative bitonic sort -------------===//
//
// Expands a `tessera_rocm.sort` directive into a cooperative bitonic sort gpu
// kernel — the sort lane (P9 of S_SERIES_GAP_CLOSURE_PLAN) backing
// sort / argsort / top_k. One workgroup sorts one row of `pn` (power-of-two)
// f32 keys ASCENDING, carrying the parallel i64 index array so the same
// compare-exchange schedule realizes argsort / top_k. The host pads each row to
// `pn` with +INF sentinels (so the real elements land in the first L slots) and
// flips the result host-side for descending — matching the numpy reference's
// flip semantics — so this kernel only ever sorts ascending.
//
// The bitonic network is data-independent (the schedule depends only on `pn`,
// never on the data) — the identical schedule the x86 AVX-512 kernel runs. The
// host passes `nstages = log2(pn)`; the kernel walks the schedule as two nested
// `scf.for` loops with a `gpu.barrier` after each phase:
//   for s in [0, nstages):  k = 1 << (s+1)
//     for t in [0, s+1):    j = 1 << (s-t)        // k/2, k/4, ..., 1
//       for i = tid; i < pn; i += blockDim:
//         l = i ^ j;  if (l > i) compare-exchange (i,l), ascending iff (i&k)==0
//       barrier
// Within one phase every index is touched in exactly one (i,l) pair owned by a
// single thread, so the phase is race-free; the barrier orders phase t before
// phase t+1. Args: (keys : memref<?xf32>, idx : memref<?xi64>, pn : index,
// nstages : index). Grid = one block per row. CPU analog: avx512_sort_f32.
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

void emitSortBody(OpBuilder &b, Location loc, gpu::GPUFuncOp f) {
  Type i64 = b.getIntegerType(64);
  Type idxTy = b.getIndexType();
  auto sgt = arith::CmpIPredicate::sgt;
  auto eq = arith::CmpIPredicate::eq;
  auto ogt = arith::CmpFPredicate::OGT;

  b.setInsertionPointToStart(&f.getBody().front());
  Value KEYS = f.getArgument(0), IDX = f.getArgument(1);
  Value PN = f.getArgument(2), NSTAGES = f.getArgument(3);

  auto ci = [&](int64_t v) { return b.create<arith::ConstantIndexOp>(loc, v); };
  Value c0 = ci(0), c1 = ci(1), cBD = ci(BD);
  Value one64 = b.create<arith::ConstantOp>(loc, i64, b.getI64IntegerAttr(1));
  Value zero64 = b.create<arith::ConstantOp>(loc, i64, b.getI64IntegerAttr(0));

  // pow2(e) = index(1 << e) for an index-typed exponent.
  auto pow2 = [&](Value e) -> Value {
    Value e64 = b.create<arith::IndexCastOp>(loc, i64, e);
    Value sh = b.create<arith::ShLIOp>(loc, one64, e64);
    return b.create<arith::IndexCastOp>(loc, idxTy, sh);
  };

  Value bid = b.create<gpu::BlockIdOp>(loc, gpu::Dimension::x);
  Value tid = b.create<gpu::ThreadIdOp>(loc, gpu::Dimension::x);
  Value base = b.create<arith::MulIOp>(loc, bid, PN);

  // for s in [0, nstages)
  auto sLoop = b.create<scf::ForOp>(loc, c0, NSTAGES, c1);
  b.setInsertionPointToStart(sLoop.getBody());
  Value s = sLoop.getInductionVar();
  Value sp1 = b.create<arith::AddIOp>(loc, s, c1);
  Value k = pow2(sp1);                       // k = 1 << (s+1)
  Value k64 = b.create<arith::IndexCastOp>(loc, i64, k);

  // for t in [0, s+1)
  auto tLoop = b.create<scf::ForOp>(loc, c0, sp1, c1);
  b.setInsertionPointToStart(tLoop.getBody());
  Value t = tLoop.getInductionVar();
  Value smt = b.create<arith::SubIOp>(loc, s, t);
  Value j = pow2(smt);                        // j = 1 << (s-t)
  Value j64 = b.create<arith::IndexCastOp>(loc, i64, j);

  // for i = tid; i < pn; i += blockDim
  auto iLoop = b.create<scf::ForOp>(loc, tid, PN, cBD);
  b.setInsertionPointToStart(iLoop.getBody());
  Value i = iLoop.getInductionVar();
  Value i64v = b.create<arith::IndexCastOp>(loc, i64, i);
  Value l64 = b.create<arith::XOrIOp>(loc, i64v, j64);
  Value l = b.create<arith::IndexCastOp>(loc, idxTy, l64);
  Value doSwap = b.create<arith::CmpIOp>(loc, sgt, l, i);   // l > i
  auto pairIf = b.create<scf::IfOp>(loc, doSwap, /*withElse=*/false);
  {
    OpBuilder pb = OpBuilder::atBlockBegin(pairIf.thenBlock());
    Value gi = pb.create<arith::AddIOp>(loc, base, i);
    Value gl = pb.create<arith::AddIOp>(loc, base, l);
    Value ki = pb.create<memref::LoadOp>(loc, KEYS, ValueRange{gi});
    Value kl = pb.create<memref::LoadOp>(loc, KEYS, ValueRange{gl});
    // up = (i & k) == 0  → ascending block; else descending.
    Value ik = pb.create<arith::AndIOp>(loc, i64v, k64);
    Value up = pb.create<arith::CmpIOp>(loc, eq, ik, zero64);
    // swap iff (up ? ki>kl : ki<kl). ki<kl == kl>ki.
    Value kiGt = pb.create<arith::CmpFOp>(loc, ogt, ki, kl);
    Value klGt = pb.create<arith::CmpFOp>(loc, ogt, kl, ki);
    Value swap = pb.create<arith::SelectOp>(loc, up, kiGt, klGt);
    auto swIf = pb.create<scf::IfOp>(loc, swap, /*withElse=*/false);
    {
      OpBuilder sb = OpBuilder::atBlockBegin(swIf.thenBlock());
      Value xi = sb.create<memref::LoadOp>(loc, IDX, ValueRange{gi});
      Value xl = sb.create<memref::LoadOp>(loc, IDX, ValueRange{gl});
      sb.create<memref::StoreOp>(loc, kl, KEYS, ValueRange{gi});
      sb.create<memref::StoreOp>(loc, ki, KEYS, ValueRange{gl});
      sb.create<memref::StoreOp>(loc, xl, IDX, ValueRange{gi});
      sb.create<memref::StoreOp>(loc, xi, IDX, ValueRange{gl});
    }
  }

  // barrier after each phase: orders phase t before phase t+1.
  b.setInsertionPointAfter(iLoop);
  b.create<gpu::BarrierOp>(loc);

  b.setInsertionPointToEnd(&f.getBody().front());
  b.create<gpu::ReturnOp>(loc);
}

struct GenerateROCMSortKernelPass
    : PassWrapper<GenerateROCMSortKernelPass, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(GenerateROCMSortKernelPass)

  StringRef getArgument() const final { return "generate-rocm-sort-kernel"; }
  StringRef getDescription() const final {
    return "Expand a tessera_rocm.sort directive into a cooperative bitonic "
           "sort gpu kernel (sort/argsort/top_k lane)";
  }
  void getDependentDialects(DialectRegistry &registry) const final {
    registry.insert<gpu::GPUDialect, scf::SCFDialect, arith::ArithDialect,
                    memref::MemRefDialect>();
  }

  void runOnOperation() override {
    ModuleOp module = getOperation();
    SmallVector<Operation *> directives;
    module.walk([&](Operation *op) {
      if (op->getName().getStringRef() == "tessera_rocm.sort")
        directives.push_back(op);
    });
    for (Operation *op : directives) {
      auto nameAttr = op->getAttrOfType<StringAttr>("name");
      if (!nameAttr) {
        op->emitError("tessera_rocm.sort missing name");
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
      auto fnTy = b.getFunctionType({memF32, memI64, idxTy, idxTy}, {});
      auto gpuMod = b.create<gpu::GPUModuleOp>(loc, kname + "_mod");
      b.setInsertionPointToStart(&gpuMod.getBodyRegion().front());
      auto gpuFunc = b.create<gpu::GPUFuncOp>(loc, kname, fnTy);
      gpuFunc->setAttr(gpu::GPUDialect::getKernelFuncAttrName(), b.getUnitAttr());
      OpBuilder body(gpuFunc.getContext());
      emitSortBody(body, loc, gpuFunc);
      op->erase();
    }
  }
};

} // namespace

std::unique_ptr<mlir::Pass>
mlir::tessera_rocm::createGenerateROCMSortKernelPass() {
  return std::make_unique<GenerateROCMSortKernelPass>();
}
