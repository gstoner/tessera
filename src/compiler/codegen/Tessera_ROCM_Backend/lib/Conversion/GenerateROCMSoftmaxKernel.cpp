//===- GenerateROCMSoftmaxKernel.cpp - compiler-generated row softmax -----===//
//
// Expands a `tessera_rocm.softmax` directive into a real **row-reduction** gpu
// kernel — the first non-matmul/non-WMMA compiler-generated ROCm kernel. For a
// rank-2 input [M, K] it computes the numerically-stable softmax over the last
// axis:  O[m,:] = exp(X[m,:] - max_k X[m,k]) / Σ_k exp(X[m,k] - max).
//
//   One workgroup per row (blockIdx.x = m), blockDim = 256. The 256 lanes
//   stride over K and reduce via a CUB/rocPRIM-style warp-shuffle (gpu.shuffle
//   xor within a 32-lane subgroup → 8 LDS partials → combine; no 256-wide tree):
//     pass 1: local max → warp-reduce → row max.
//     pass 2: e = exp(x - max) staged into O, local sum → warp-reduce → row sum.
//     pass 3: O[m,c] /= row sum.
//   Reductions run in f32 for stability regardless of storage dtype; `exp`
//   lowers through `convert-math-to-rocdl` (OCML), the same path flash_attn
//   uses. M/K are runtime index args (problem-generic). Validated vs a numpy
//   softmax reference on gfx1151.
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

// Workgroup size (one wavefront-multiple block per row). Multiple of the
// 32-lane shuffle subgroup.
static constexpr int64_t BD = 256;
static constexpr int64_t SG = 32;           // shuffle subgroup width
static constexpr int64_t NGROUPS = BD / SG; // per-subgroup partials (= 8)

void emitSoftmaxBody(OpBuilder &b, Location loc, gpu::GPUFuncOp f, Type storeTy) {
  MLIRContext *ctx = b.getContext();
  Type f32 = b.getF32Type();
  bool isF32 = storeTy.isF32();
  auto slt = arith::CmpIPredicate::slt;

  auto ws = gpu::AddressSpaceAttr::get(ctx, gpu::AddressSpace::Workgroup);
  // LDS holds only the per-subgroup partials (NGROUPS = 8), reused per pass.
  Value red = f.addWorkgroupAttribution(
      MemRefType::get({NGROUPS}, f32, MemRefLayoutAttrInterface(), ws), loc);

  b.setInsertionPointToStart(&f.getBody().front());
  Value X = f.getArgument(0), O = f.getArgument(1);
  Value M = f.getArgument(2), K = f.getArgument(3);

  auto ci = [&](int64_t v) { return b.create<arith::ConstantIndexOp>(loc, v); };
  Value c0 = ci(0), c1 = ci(1), cBD = ci(BD);
  Value negInf =
      b.create<arith::ConstantOp>(loc, f32, b.getF32FloatAttr(-3.4028235e38f));
  Value zerof = b.create<arith::ConstantOp>(loc, f32, b.getF32FloatAttr(0.0));

  Value m = b.create<gpu::BlockIdOp>(loc, gpu::Dimension::x);
  Value tid = b.create<gpu::ThreadIdOp>(loc, gpu::Dimension::x);
  Type i32 = b.getI32Type();
  Value wSG = b.create<arith::ConstantIntOp>(loc, i32, SG);
  Value cSG = ci(SG);
  Value group = b.create<arith::DivUIOp>(loc, tid, cSG);
  Value laneInSg = b.create<arith::RemUIOp>(loc, tid, cSG);
  Value isLeader =
      b.create<arith::CmpIOp>(loc, arith::CmpIPredicate::eq, laneInSg, c0);
  Value rowInb = b.create<arith::CmpIOp>(loc, slt, m, M);
  // Guard the whole body on a valid row (grid is exactly M, so always true, but
  // keep it total).
  auto rowIf = b.create<scf::IfOp>(loc, rowInb, /*withElse=*/false);
  b.setInsertionPointToStart(rowIf.thenBlock());
  Value base = b.create<arith::MulIOp>(loc, m, K);

  auto loadF32From = [&](Value mem, Value idx) -> Value {
    Value v = b.create<memref::LoadOp>(loc, mem, ValueRange{idx});
    return isF32 ? v : b.create<arith::ExtFOp>(loc, f32, v);
  };
  auto loadF32 = [&](Value idx) -> Value { return loadF32From(X, idx); };
  auto storeFromF32 = [&](Value val, Value idx) {
    Value sv = isF32 ? val : b.create<arith::TruncFOp>(loc, storeTy, val);
    b.create<memref::StoreOp>(loc, sv, O, ValueRange{idx});
  };

  // CUB/rocPRIM warp-shuffle reduce: butterfly within a 32-lane subgroup (no
  // LDS) over `acc`, stage the 8 per-subgroup partials to `red`, then every
  // thread combines the partials to the broadcast total. `red` is reused per
  // pass, so barriers bracket its read. `isMax` selects max vs add.
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

  // pass 1 — local max over strided cols, then warp-reduce to the row max.
  Value localMax;
  {
    auto lp = b.create<scf::ForOp>(loc, tid, K, cBD, ValueRange{negInf});
    OpBuilder::InsertionGuard g(b);
    b.setInsertionPointToStart(lp.getBody());
    Value c = lp.getInductionVar();
    Value acc = lp.getRegionIterArgs()[0];
    Value v = loadF32(b.create<arith::AddIOp>(loc, base, c));
    b.create<scf::YieldOp>(
        loc, ValueRange{b.create<arith::MaximumFOp>(loc, acc, v)});
    localMax = lp.getResult(0);
  }
  Value rmax = warpReduce(localMax, /*isMax=*/true);

  // pass 2 — e = exp(x - max) staged into O; local sum, then warp-reduce.
  Value localSum;
  {
    auto lp = b.create<scf::ForOp>(loc, tid, K, cBD, ValueRange{zerof});
    OpBuilder::InsertionGuard g(b);
    b.setInsertionPointToStart(lp.getBody());
    Value c = lp.getInductionVar();
    Value acc = lp.getRegionIterArgs()[0];
    Value idx = b.create<arith::AddIOp>(loc, base, c);
    Value e = b.create<math::ExpOp>(
        loc, b.create<arith::SubFOp>(loc, loadF32(idx), rmax));
    storeFromF32(e, idx);
    b.create<scf::YieldOp>(loc,
                           ValueRange{b.create<arith::AddFOp>(loc, acc, e)});
    localSum = lp.getResult(0);
  }
  Value rsum = warpReduce(localSum, /*isMax=*/false);

  // pass 3 — divide O by the row sum.
  {
    auto lp = b.create<scf::ForOp>(loc, tid, K, cBD);
    OpBuilder::InsertionGuard g(b);
    b.setInsertionPointToStart(lp.getBody());
    Value c = lp.getInductionVar();
    Value idx = b.create<arith::AddIOp>(loc, base, c);
    // Read the exp value staged into O by pass 2 (NOT X), divide by the sum.
    Value cur = loadF32From(O, idx);
    storeFromF32(b.create<arith::DivFOp>(loc, cur, rsum), idx);
  }
  // (c1 is referenced to keep the builder honest about the index unit.)
  (void)c1;

  b.setInsertionPointToEnd(&f.getBody().front());
  b.create<gpu::ReturnOp>(loc);
}

struct GenerateROCMSoftmaxKernelPass
    : PassWrapper<GenerateROCMSoftmaxKernelPass, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(GenerateROCMSoftmaxKernelPass)

  StringRef getArgument() const final { return "generate-rocm-softmax-kernel"; }
  StringRef getDescription() const final {
    return "Expand a tessera_rocm.softmax directive into a row-reduction "
           "(stable softmax over the last axis) gpu kernel (compiler-generated)";
  }
  void getDependentDialects(DialectRegistry &registry) const final {
    registry.insert<gpu::GPUDialect, scf::SCFDialect, arith::ArithDialect,
                    math::MathDialect, memref::MemRefDialect>();
  }

  void runOnOperation() override {
    ModuleOp module = getOperation();
    SmallVector<Operation *> directives;
    module.walk([&](Operation *op) {
      if (op->getName().getStringRef() == "tessera_rocm.softmax")
        directives.push_back(op);
    });
    for (Operation *op : directives) {
      auto nameAttr = op->getAttrOfType<StringAttr>("name");
      if (!nameAttr) {
        op->emitError("tessera_rocm.softmax missing name");
        return signalPassFailure();
      }
      OpBuilder b(module.getBodyRegion());
      b.setInsertionPointToEnd(module.getBody());
      Location loc = op->getLoc();
      std::string kname = nameAttr.getValue().str();

      Type storeTy = b.getF32Type();
      if (auto a = op->getAttrOfType<StringAttr>("dtype")) {
        StringRef dt = a.getValue();
        if (dt == "f16" || dt == "float16")
          storeTy = b.getF16Type();
        else if (dt == "bf16" || dt == "bfloat16")
          storeTy = b.getBF16Type();
        else if (dt != "f32" && dt != "float32") {
          op->emitError("generate-rocm-softmax-kernel: dtype must be f32, f16, "
                        "or bf16 (got '")
              << dt << "')";
          return signalPassFailure();
        }
      }

      auto gpuMod = b.create<gpu::GPUModuleOp>(loc, kname + "_mod");
      b.setInsertionPointToStart(&gpuMod.getBodyRegion().front());
      Type idxTy = b.getIndexType();
      auto memTy = MemRefType::get({ShapedType::kDynamic}, storeTy);
      // (X, O : memref<?xstore>, M, K : index)
      auto fnTy = b.getFunctionType({memTy, memTy, idxTy, idxTy}, {});
      auto gpuFunc = b.create<gpu::GPUFuncOp>(loc, kname, fnTy);
      gpuFunc->setAttr(gpu::GPUDialect::getKernelFuncAttrName(),
                       b.getUnitAttr());
      OpBuilder body(gpuFunc.getContext());
      emitSoftmaxBody(body, loc, gpuFunc, storeTy);
      op->erase();
    }
  }
};

} // namespace

std::unique_ptr<mlir::Pass>
mlir::tessera_rocm::createGenerateROCMSoftmaxKernelPass() {
  return std::make_unique<GenerateROCMSoftmaxKernelPass>();
}
