//===- GenerateROCMSoftmaxKernel.cpp - compiler-generated row softmax -----===//
//
// Expands a `tessera_rocm.softmax` directive into a real **row-reduction** gpu
// kernel — the first non-matmul/non-WMMA compiler-generated ROCm kernel. For a
// rank-2 input [M, K] it computes the numerically-stable softmax over the last
// axis:  O[m,:] = exp(X[m,:] - max_k X[m,k]) / Σ_k exp(X[m,k] - max).
//
//   One workgroup per row (blockIdx.x = m), blockDim = 256. The 256 lanes
//   stride over K and tree-reduce through LDS:
//     pass 1: local max → LDS tree-reduce → row max.
//     pass 2: e = exp(x - max) staged into O, local sum → tree-reduce → row sum.
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

// Workgroup size (one wavefront-multiple block per row). Power of two so the
// LDS tree-reduction halving is exact.
static constexpr int64_t BD = 256;

void emitSoftmaxBody(OpBuilder &b, Location loc, gpu::GPUFuncOp f, Type storeTy) {
  MLIRContext *ctx = b.getContext();
  Type f32 = b.getF32Type();
  bool isF32 = storeTy.isF32();
  auto slt = arith::CmpIPredicate::slt;

  auto ws = gpu::AddressSpaceAttr::get(ctx, gpu::AddressSpace::Workgroup);
  Value red = f.addWorkgroupAttribution(
      MemRefType::get({BD}, f32, MemRefLayoutAttrInterface(), ws), loc);

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

  // Unrolled LDS tree-reduction over red[0..BD): combine red[t] with red[t+s]
  // for s = BD/2, BD/4, ..., 1, with a barrier between levels. `isMax` selects
  // max vs add. Leaves the result in red[0].
  auto treeReduce = [&](bool isMax) {
    for (int64_t s = BD / 2; s > 0; s >>= 1) {
      Value cs = ci(s);
      Value lt = b.create<arith::CmpIOp>(loc, slt, tid, cs);
      auto ifo = b.create<scf::IfOp>(loc, lt, /*withElse=*/false);
      {
        OpBuilder::InsertionGuard g(b);
        b.setInsertionPointToStart(ifo.thenBlock());
        Value a = b.create<memref::LoadOp>(loc, red, ValueRange{tid});
        Value other = b.create<memref::LoadOp>(
            loc, red, ValueRange{b.create<arith::AddIOp>(loc, tid, cs)});
        Value comb = isMax ? b.create<arith::MaximumFOp>(loc, a, other).getResult()
                           : b.create<arith::AddFOp>(loc, a, other).getResult();
        b.create<memref::StoreOp>(loc, comb, red, ValueRange{tid});
      }
      b.create<gpu::BarrierOp>(loc);
    }
  };

  // pass 1 — local max over strided cols, then tree-reduce to the row max.
  {
    auto lp = b.create<scf::ForOp>(loc, tid, K, cBD, ValueRange{negInf});
    OpBuilder::InsertionGuard g(b);
    b.setInsertionPointToStart(lp.getBody());
    Value c = lp.getInductionVar();
    Value acc = lp.getRegionIterArgs()[0];
    Value v = loadF32(b.create<arith::AddIOp>(loc, base, c));
    Value nmax = b.create<arith::MaximumFOp>(loc, acc, v);
    b.create<scf::YieldOp>(loc, ValueRange{nmax});
    b.setInsertionPointAfter(lp);
    b.create<memref::StoreOp>(loc, lp.getResult(0), red, ValueRange{tid});
  }
  b.create<gpu::BarrierOp>(loc);
  treeReduce(/*isMax=*/true);
  Value rmax = b.create<memref::LoadOp>(loc, red, ValueRange{c0});
  b.create<gpu::BarrierOp>(loc);

  // pass 2 — e = exp(x - max) staged into O; local sum, then tree-reduce.
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
    Value nsum = b.create<arith::AddFOp>(loc, acc, e);
    b.create<scf::YieldOp>(loc, ValueRange{nsum});
    b.setInsertionPointAfter(lp);
    b.create<memref::StoreOp>(loc, lp.getResult(0), red, ValueRange{tid});
  }
  b.create<gpu::BarrierOp>(loc);
  treeReduce(/*isMax=*/false);
  Value rsum = b.create<memref::LoadOp>(loc, red, ValueRange{c0});
  b.create<gpu::BarrierOp>(loc);

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
