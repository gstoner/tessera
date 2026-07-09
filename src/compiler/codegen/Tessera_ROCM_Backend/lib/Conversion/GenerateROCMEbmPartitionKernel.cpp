//===- GenerateROCMEbmPartitionKernel.cpp - exact-partition log-sum-exp ---===//
//
// Expands a `tessera_rocm.ebm_partition` directive into a full-array reduction
// gpu kernel — the EBM3 exact-partition lane. For per-state energies E (n
// elements) at temperature T it computes the numerically-stable partition value
//
//   Z = Σ_i exp(-E_i / T) = exp(m) · Σ_i exp(-E_i/T - m),  m = max_i(-E_i/T).
//
//   ONE workgroup, blockDim = 256. The 256 lanes stride over n and reduce via
//   the same CUB/rocPRIM-style warp-shuffle as the softmax kernel (gpu.shuffle
//   xor within a 32-lane subgroup → 8 LDS partials → combine):
//     pass 1: local max of (-E_i/T) → warp-reduce max → m.
//     pass 2: local Σ exp(-E_i/T - m) → warp-reduce add → s.
//     thread 0: out[0] = exp(m) · s.
//   Reductions run in f32; `exp` lowers through convert-math-to-rocdl (OCML),
//   the same path softmax/flash_attn use. n is a runtime index arg. Matches
//   tessera.ebm.partition_exact_from_energies. CPU analog:
//   tessera_x86_ebm_partition_exact_f32.
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
#include "mlir/Pass/Pass.h"

using namespace mlir;

namespace {

static constexpr int64_t BD = 256;
static constexpr int64_t SG = 32;           // shuffle subgroup width
static constexpr int64_t NGROUPS = BD / SG; // per-subgroup partials (= 8)

void emitPartitionBody(OpBuilder &b, Location loc, gpu::GPUFuncOp f) {
  MLIRContext *ctx = b.getContext();
  Type f32 = b.getF32Type();
  Type i32 = b.getI32Type();

  auto ws = gpu::AddressSpaceAttr::get(ctx, gpu::AddressSpace::Workgroup);
  Value red = f.addWorkgroupAttribution(
      MemRefType::get({NGROUPS}, f32, MemRefLayoutAttrInterface(), ws), loc);

  b.setInsertionPointToStart(&f.getBody().front());
  Value E = f.getArgument(0), N = f.getArgument(1);
  Value T = f.getArgument(2), OUT = f.getArgument(3);

  auto ci = [&](int64_t v) { return b.create<arith::ConstantIndexOp>(loc, v); };
  Value c0 = ci(0), cBD = ci(BD), cSG = ci(SG);
  Value negInf =
      b.create<arith::ConstantOp>(loc, f32, b.getF32FloatAttr(-3.4028235e38f));
  Value zerof = b.create<arith::ConstantOp>(loc, f32, b.getF32FloatAttr(0.0f));
  Value onef = b.create<arith::ConstantOp>(loc, f32, b.getF32FloatAttr(1.0f));

  Value tid = b.create<gpu::ThreadIdOp>(loc, gpu::Dimension::x);
  Value wSG = b.create<arith::ConstantIntOp>(loc, i32, SG);
  Value group = b.create<arith::DivUIOp>(loc, tid, cSG);
  Value laneInSg = b.create<arith::RemUIOp>(loc, tid, cSG);
  Value isLeader =
      b.create<arith::CmpIOp>(loc, arith::CmpIPredicate::eq, laneInSg, c0);

  // negInvT = -1 / T  (so  -E_i/T = E_i * negInvT).
  Value negInvT =
      b.create<arith::DivFOp>(loc, b.create<arith::NegFOp>(loc, onef), T);

  // CUB/rocPRIM warp-shuffle reduce (identical to the softmax kernel): butterfly
  // within a 32-lane subgroup, stage 8 per-subgroup partials to `red`, then every
  // thread combines them to the broadcast total. Barriers bracket `red` reuse.
  auto warpReduce = [&](Value acc, bool isMax) -> Value {
    auto comb = [&](Value a, Value c) -> Value {
      return isMax ? b.create<arith::MaximumFOp>(loc, a, c).getResult()
                   : b.create<arith::AddFOp>(loc, a, c).getResult();
    };
    for (int64_t off = SG / 2; off > 0; off >>= 1) {
      Value offC = b.create<arith::ConstantIntOp>(loc, i32, off);
      auto sh = b.create<gpu::ShuffleOp>(loc, acc, offC, wSG,
                                         gpu::ShuffleMode::XOR);
      acc = comb(acc, sh.getShuffleResult());
    }
    auto ldIf = b.create<scf::IfOp>(loc, isLeader, /*withElse=*/false);
    {
      OpBuilder::InsertionGuard g(b);
      b.setInsertionPointToStart(ldIf.thenBlock());
      b.create<memref::StoreOp>(loc, acc, red, ValueRange{group});
    }
    b.create<gpu::BarrierOp>(loc);
    Value total = b.create<memref::LoadOp>(loc, red, ValueRange{c0});
    for (int64_t gi = 1; gi < NGROUPS; ++gi)
      total = comb(total,
                   b.create<memref::LoadOp>(loc, red, ValueRange{ci(gi)}));
    b.create<gpu::BarrierOp>(loc);
    return total;
  };

  // pass 1 — local max of the negative scaled energies, then warp-reduce.
  Value localMax;
  {
    auto lp = b.create<scf::ForOp>(loc, tid, N, cBD, ValueRange{negInf});
    OpBuilder::InsertionGuard g(b);
    b.setInsertionPointToStart(lp.getBody());
    Value i = lp.getInductionVar();
    Value acc = lp.getRegionIterArgs()[0];
    Value ev = b.create<memref::LoadOp>(loc, E, ValueRange{i});
    Value neg = b.create<arith::MulFOp>(loc, ev, negInvT);
    b.create<scf::YieldOp>(
        loc, ValueRange{b.create<arith::MaximumFOp>(loc, acc, neg)});
    localMax = lp.getResult(0);
  }
  Value m = warpReduce(localMax, /*isMax=*/true);

  // pass 2 — local Σ exp(neg - m), then warp-reduce.
  Value localSum;
  {
    auto lp = b.create<scf::ForOp>(loc, tid, N, cBD, ValueRange{zerof});
    OpBuilder::InsertionGuard g(b);
    b.setInsertionPointToStart(lp.getBody());
    Value i = lp.getInductionVar();
    Value acc = lp.getRegionIterArgs()[0];
    Value ev = b.create<memref::LoadOp>(loc, E, ValueRange{i});
    Value neg = b.create<arith::MulFOp>(loc, ev, negInvT);
    Value e = b.create<math::ExpOp>(loc, b.create<arith::SubFOp>(loc, neg, m));
    b.create<scf::YieldOp>(loc,
                           ValueRange{b.create<arith::AddFOp>(loc, acc, e)});
    localSum = lp.getResult(0);
  }
  Value s = warpReduce(localSum, /*isMax=*/false);

  // thread 0 writes Z = exp(m) · s.
  Value isT0 = b.create<arith::CmpIOp>(loc, arith::CmpIPredicate::eq, tid, c0);
  auto wIf = b.create<scf::IfOp>(loc, isT0, /*withElse=*/false);
  {
    OpBuilder::InsertionGuard g(b);
    b.setInsertionPointToStart(wIf.thenBlock());
    Value z = b.create<arith::MulFOp>(loc, b.create<math::ExpOp>(loc, m), s);
    b.create<memref::StoreOp>(loc, z, OUT, ValueRange{c0});
  }

  b.setInsertionPointToEnd(&f.getBody().front());
  b.create<gpu::ReturnOp>(loc);
}

struct GenerateROCMEbmPartitionKernelPass
    : PassWrapper<GenerateROCMEbmPartitionKernelPass, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(GenerateROCMEbmPartitionKernelPass)

  StringRef getArgument() const final {
    return "generate-rocm-ebm-partition-kernel";
  }
  StringRef getDescription() const final {
    return "Expand a tessera_rocm.ebm_partition directive into a full-array "
           "log-sum-exp partition reduction gpu kernel (compiler-generated)";
  }
  void getDependentDialects(DialectRegistry &registry) const final {
    registry.insert<gpu::GPUDialect, scf::SCFDialect, arith::ArithDialect,
                    math::MathDialect, memref::MemRefDialect>();
  }

  void runOnOperation() override {
    ModuleOp module = getOperation();
    SmallVector<Operation *> directives;
    module.walk([&](Operation *op) {
      if (op->getName().getStringRef() == "tessera_rocm.ebm_partition")
        directives.push_back(op);
    });
    for (Operation *op : directives) {
      auto nameAttr = op->getAttrOfType<StringAttr>("name");
      if (!nameAttr) {
        op->emitError("tessera_rocm.ebm_partition missing name");
        return signalPassFailure();
      }
      OpBuilder b(module.getBodyRegion());
      b.setInsertionPointToEnd(module.getBody());
      Location loc = op->getLoc();
      std::string kname = nameAttr.getValue().str();
      Type f32 = b.getF32Type();
      Type idxTy = b.getIndexType();
      auto memF32 = MemRefType::get({ShapedType::kDynamic}, f32);
      // (E : memref<?xf32>, n : index, T : f32, out : memref<?xf32>)
      auto fnTy = b.getFunctionType({memF32, idxTy, f32, memF32}, {});
      auto gpuMod = b.create<gpu::GPUModuleOp>(loc, kname + "_mod");
      b.setInsertionPointToStart(&gpuMod.getBodyRegion().front());
      auto gpuFunc = b.create<gpu::GPUFuncOp>(loc, kname, fnTy);
      gpuFunc->setAttr(gpu::GPUDialect::getKernelFuncAttrName(),
                       b.getUnitAttr());
      OpBuilder body(gpuFunc.getContext());
      emitPartitionBody(body, loc, gpuFunc);
      op->erase();
    }
  }
};

} // namespace

std::unique_ptr<mlir::Pass>
mlir::tessera_rocm::createGenerateROCMEbmPartitionKernelPass() {
  return std::make_unique<GenerateROCMEbmPartitionKernelPass>();
}
