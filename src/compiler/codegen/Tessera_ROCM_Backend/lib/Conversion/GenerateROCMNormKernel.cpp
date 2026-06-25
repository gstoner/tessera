//===- GenerateROCMNormKernel.cpp - compiler-generated row normalization --===//
//
// Expands a `tessera_rocm.norm` directive into a row-reduction gpu kernel for
// the **unweighted** row normalizations over the last axis (the sibling of the
// softmax reduction kernel):
//
//   kind = "rmsnorm"    : O[m,:] = X[m,:] / sqrt(mean_k X[m,k]² + eps)
//   kind = "layer_norm" : O[m,:] = (X[m,:] − μ) / sqrt(var + eps),
//                         μ = mean_k X,  var = mean_k (X − μ)²
//
//   One workgroup per row (blockIdx.x = m), blockDim = 256, LDS tree-reduce.
//   rmsnorm is one reduction (Σx²). layer_norm is TWO reductions — Σx for the
//   mean, then Σ(x−μ)² for the variance (the stable squared-deviation form,
//   NOT E[x²]−E[x]², which cancels for large-offset/small-variance rows). Then
//   a write pass applies the per-row normalize.
//   Reductions run in f32 regardless of storage dtype; `sqrt` lowers through
//   convert-math-to-rocdl. eps is a trailing f32 runtime arg; M/K are runtime
//   index args. Validated vs the numpy reference (`_apple_gpu_rowop_numpy`).
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

void emitNormBody(OpBuilder &b, Location loc, gpu::GPUFuncOp f, Type storeTy,
                  bool isLayerNorm) {
  MLIRContext *ctx = b.getContext();
  Type f32 = b.getF32Type();
  bool isF32 = storeTy.isF32();
  auto slt = arith::CmpIPredicate::slt;

  auto ws = gpu::AddressSpaceAttr::get(ctx, gpu::AddressSpace::Workgroup);
  auto ldsT = MemRefType::get({BD}, f32, MemRefLayoutAttrInterface(), ws);
  Value red = f.addWorkgroupAttribution(ldsT, loc);  // reused per reduction pass

  b.setInsertionPointToStart(&f.getBody().front());
  Value X = f.getArgument(0), O = f.getArgument(1);
  Value M = f.getArgument(2), K = f.getArgument(3), eps = f.getArgument(4);

  auto ci = [&](int64_t v) { return b.create<arith::ConstantIndexOp>(loc, v); };
  Value c0 = ci(0), cBD = ci(BD);
  Value zerof = b.create<arith::ConstantOp>(loc, f32, b.getF32FloatAttr(0.0));

  Value m = b.create<gpu::BlockIdOp>(loc, gpu::Dimension::x);
  Value tid = b.create<gpu::ThreadIdOp>(loc, gpu::Dimension::x);
  Value rowInb = b.create<arith::CmpIOp>(loc, slt, m, M);
  auto rowIf = b.create<scf::IfOp>(loc, rowInb, /*withElse=*/false);
  b.setInsertionPointToStart(rowIf.thenBlock());
  Value base = b.create<arith::MulIOp>(loc, m, K);
  // Kf = (f32) K  (for the means).
  Value Kf = b.create<arith::SIToFPOp>(
      loc, f32, b.create<arith::IndexCastOp>(loc, b.getI64Type(), K));

  auto loadF32 = [&](Value idx) -> Value {
    Value v = b.create<memref::LoadOp>(loc, X, ValueRange{idx});
    return isF32 ? v : b.create<arith::ExtFOp>(loc, f32, v);
  };
  auto storeFromF32 = [&](Value val, Value idx) {
    Value sv = isF32 ? val : b.create<arith::TruncFOp>(loc, storeTy, val);
    b.create<memref::StoreOp>(loc, sv, O, ValueRange{idx});
  };
  // Unrolled LDS sum tree-reduction over `buf[0..BD)`, result in buf[0].
  auto treeReduceSum = [&](Value buf) {
    for (int64_t s = BD / 2; s > 0; s >>= 1) {
      Value cs = ci(s);
      Value lt = b.create<arith::CmpIOp>(loc, slt, tid, cs);
      auto ifo = b.create<scf::IfOp>(loc, lt, /*withElse=*/false);
      {
        OpBuilder::InsertionGuard g(b);
        b.setInsertionPointToStart(ifo.thenBlock());
        Value a = b.create<memref::LoadOp>(loc, buf, ValueRange{tid});
        Value o = b.create<memref::LoadOp>(
            loc, buf, ValueRange{b.create<arith::AddIOp>(loc, tid, cs)});
        b.create<memref::StoreOp>(loc, b.create<arith::AddFOp>(loc, a, o), buf,
                                  ValueRange{tid});
      }
      b.create<gpu::BarrierOp>(loc);
    }
  };

  // One full row reduction: accumulate a per-element f32 value over the strided
  // cols, stage to red[tid], tree-reduce, and return the row total (red[0]).
  // `red` is reused across passes, so a barrier brackets the read of red[0].
  auto reduceRow =
      [&](function_ref<Value(Value /*idx*/)> localOf) -> Value {
    auto lp = b.create<scf::ForOp>(loc, tid, K, cBD, ValueRange{zerof});
    {
      OpBuilder::InsertionGuard g(b);
      b.setInsertionPointToStart(lp.getBody());
      Value c = lp.getInductionVar();
      Value acc = lp.getRegionIterArgs()[0];
      Value v = localOf(b.create<arith::AddIOp>(loc, base, c));
      b.create<scf::YieldOp>(loc,
                             ValueRange{b.create<arith::AddFOp>(loc, acc, v)});
    }
    b.create<memref::StoreOp>(loc, lp.getResult(0), red, ValueRange{tid});
    b.create<gpu::BarrierOp>(loc);
    treeReduceSum(red);
    Value total = b.create<memref::LoadOp>(loc, red, ValueRange{c0});
    b.create<gpu::BarrierOp>(loc);  // all read red[0] before the next pass reuses red
    return total;
  };

  // denom = sqrt(stat + eps), matching numpy's x / √(...). For layer_norm the
  // variance is computed as a SECOND reduction of the squared deviations
  // (mean((x−μ)²)) rather than E[x²]−E[x]² — the latter cancels catastrophically
  // for rows with a large common offset and small variance (PR#123 review).
  Value mean, denom;
  if (isLayerNorm) {
    mean = b.create<arith::DivFOp>(loc, reduceRow(loadF32), Kf);
    Value vsum = reduceRow([&](Value idx) {
      Value d = b.create<arith::SubFOp>(loc, loadF32(idx), mean);
      return b.create<arith::MulFOp>(loc, d, d).getResult();
    });
    Value var = b.create<arith::DivFOp>(loc, vsum, Kf);
    denom = b.create<math::SqrtOp>(loc, b.create<arith::AddFOp>(loc, var, eps));
  } else {  // rmsnorm: denom = sqrt(mean(x²) + eps)
    Value sumsq = reduceRow([&](Value idx) {
      Value v = loadF32(idx);
      return b.create<arith::MulFOp>(loc, v, v).getResult();
    });
    Value ms = b.create<arith::DivFOp>(loc, sumsq, Kf);
    denom = b.create<math::SqrtOp>(loc, b.create<arith::AddFOp>(loc, ms, eps));
  }

  // write pass — O = (x [− μ]) / denom.
  {
    auto lp = b.create<scf::ForOp>(loc, tid, K, cBD);
    OpBuilder::InsertionGuard g(b);
    b.setInsertionPointToStart(lp.getBody());
    Value c = lp.getInductionVar();
    Value idx = b.create<arith::AddIOp>(loc, base, c);
    Value v = loadF32(idx);
    if (isLayerNorm)
      v = b.create<arith::SubFOp>(loc, v, mean);
    storeFromF32(b.create<arith::DivFOp>(loc, v, denom), idx);
  }

  b.setInsertionPointToEnd(&f.getBody().front());
  b.create<gpu::ReturnOp>(loc);
}

struct GenerateROCMNormKernelPass
    : PassWrapper<GenerateROCMNormKernelPass, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(GenerateROCMNormKernelPass)

  StringRef getArgument() const final { return "generate-rocm-norm-kernel"; }
  StringRef getDescription() const final {
    return "Expand a tessera_rocm.norm directive into a row-reduction "
           "(rmsnorm / layer_norm over the last axis) gpu kernel "
           "(compiler-generated)";
  }
  void getDependentDialects(DialectRegistry &registry) const final {
    registry.insert<gpu::GPUDialect, scf::SCFDialect, arith::ArithDialect,
                    math::MathDialect, memref::MemRefDialect>();
  }

  void runOnOperation() override {
    ModuleOp module = getOperation();
    SmallVector<Operation *> directives;
    module.walk([&](Operation *op) {
      if (op->getName().getStringRef() == "tessera_rocm.norm")
        directives.push_back(op);
    });
    for (Operation *op : directives) {
      auto nameAttr = op->getAttrOfType<StringAttr>("name");
      if (!nameAttr) {
        op->emitError("tessera_rocm.norm missing name");
        return signalPassFailure();
      }
      StringRef kind = "rmsnorm";
      if (auto a = op->getAttrOfType<StringAttr>("kind"))
        kind = a.getValue();
      if (kind != "rmsnorm" && kind != "layer_norm") {
        op->emitError("generate-rocm-norm-kernel: kind must be rmsnorm or "
                      "layer_norm (got '")
            << kind << "')";
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
          op->emitError("generate-rocm-norm-kernel: dtype must be f32, f16, or "
                        "bf16 (got '")
              << dt << "')";
          return signalPassFailure();
        }
      }

      auto gpuMod = b.create<gpu::GPUModuleOp>(loc, kname + "_mod");
      b.setInsertionPointToStart(&gpuMod.getBodyRegion().front());
      Type f32 = b.getF32Type();
      Type idxTy = b.getIndexType();
      auto memTy = MemRefType::get({ShapedType::kDynamic}, storeTy);
      // (X, O : memref<?xstore>, M, K : index, eps : f32)
      auto fnTy = b.getFunctionType({memTy, memTy, idxTy, idxTy, f32}, {});
      auto gpuFunc = b.create<gpu::GPUFuncOp>(loc, kname, fnTy);
      gpuFunc->setAttr(gpu::GPUDialect::getKernelFuncAttrName(),
                       b.getUnitAttr());
      OpBuilder body(gpuFunc.getContext());
      emitNormBody(body, loc, gpuFunc, storeTy, kind == "layer_norm");
      op->erase();
    }
  }
};

} // namespace

std::unique_ptr<mlir::Pass>
mlir::tessera_rocm::createGenerateROCMNormKernelPass() {
  return std::make_unique<GenerateROCMNormKernelPass>();
}
