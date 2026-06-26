//===- GenerateROCMReduceKernel.cpp - row reduction (sum/mean/max/min) ----===//
//
// Expands a `tessera_rocm.reduce` directive into a row-reduction gpu kernel —
// the ROCm analog of the x86 AVX-512 reduction lane. Reduces each row of a
// [M, K] input over the last axis to a single value O[M] (the runtime folds an
// arbitrary reduced axis to [outer=M, inner=K] by transposing the reduced axes
// to the end, matching `_apple_gpu_dispatch_reduce`):
//
//   kind = "sum"  : O[m] = Σ_k X[m,k]
//   kind = "mean" : O[m] = (Σ_k X[m,k]) / K
//   kind = "max"  : O[m] = max_k X[m,k]
//   kind = "min"  : O[m] = min_k X[m,k]
//
// One workgroup per row (blockIdx.x = m), blockDim = 256, LDS tree-reduce.
// Reductions run in f32 regardless of storage dtype; M/K are runtime index args.
// Validated vs numpy (np.sum/mean/amax/amin) on gfx1151.
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

#include <limits>

using namespace mlir;

namespace {

static constexpr int64_t BD = 256;

enum class Red { Sum, Mean, Max, Min };

void emitReduceBody(OpBuilder &b, Location loc, gpu::GPUFuncOp f, Type storeTy,
                    Red red) {
  MLIRContext *ctx = b.getContext();
  Type f32 = b.getF32Type();
  bool isF32 = storeTy.isF32();
  auto slt = arith::CmpIPredicate::slt;
  bool isMax = red == Red::Max, isMin = red == Red::Min;
  bool isMinMax = isMax || isMin;

  auto ws = gpu::AddressSpaceAttr::get(ctx, gpu::AddressSpace::Workgroup);
  auto ldsT = MemRefType::get({BD}, f32, MemRefLayoutAttrInterface(), ws);
  Value buf = f.addWorkgroupAttribution(ldsT, loc);

  b.setInsertionPointToStart(&f.getBody().front());
  Value X = f.getArgument(0), O = f.getArgument(1);
  Value M = f.getArgument(2), K = f.getArgument(3);

  auto ci = [&](int64_t v) { return b.create<arith::ConstantIndexOp>(loc, v); };
  auto cf = [&](float v) {
    return b.create<arith::ConstantOp>(loc, f32, b.getF32FloatAttr(v));
  };
  Value c0 = ci(0), cBD = ci(BD);
  float ninf = -std::numeric_limits<float>::infinity();
  float pinf = std::numeric_limits<float>::infinity();
  Value ident = isMax ? cf(ninf) : isMin ? cf(pinf) : cf(0.0f);

  auto combine = [&](Value a, Value c) -> Value {
    if (isMax) return b.create<arith::MaximumFOp>(loc, a, c);
    if (isMin) return b.create<arith::MinimumFOp>(loc, a, c);
    return b.create<arith::AddFOp>(loc, a, c);
  };

  Value m = b.create<gpu::BlockIdOp>(loc, gpu::Dimension::x);
  Value tid = b.create<gpu::ThreadIdOp>(loc, gpu::Dimension::x);
  Value rowInb = b.create<arith::CmpIOp>(loc, slt, m, M);
  auto rowIf = b.create<scf::IfOp>(loc, rowInb, /*withElse=*/false);
  b.setInsertionPointToStart(rowIf.thenBlock());
  Value base = b.create<arith::MulIOp>(loc, m, K);

  auto loadF32 = [&](Value idx) -> Value {
    Value v = b.create<memref::LoadOp>(loc, X, ValueRange{idx});
    return isF32 ? v : b.create<arith::ExtFOp>(loc, f32, v);
  };

  // per-thread strided accumulate over the row (identity-seeded)
  auto lp = b.create<scf::ForOp>(loc, tid, K, cBD, ValueRange{ident});
  {
    OpBuilder::InsertionGuard g(b);
    b.setInsertionPointToStart(lp.getBody());
    Value c = lp.getInductionVar();
    Value acc = lp.getRegionIterArgs()[0];
    Value v = loadF32(b.create<arith::AddIOp>(loc, base, c));
    b.create<scf::YieldOp>(loc, ValueRange{combine(acc, v)});
  }
  b.create<memref::StoreOp>(loc, lp.getResult(0), buf, ValueRange{tid});
  b.create<gpu::BarrierOp>(loc);

  // unrolled LDS tree-reduce over buf[0..BD) with `combine`
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
      b.create<memref::StoreOp>(loc, combine(a, o), buf, ValueRange{tid});
    }
    b.create<gpu::BarrierOp>(loc);
  }

  // thread 0 writes O[m] (mean divides by K)
  Value isT0 = b.create<arith::CmpIOp>(loc, arith::CmpIPredicate::eq, tid, c0);
  auto t0if = b.create<scf::IfOp>(loc, isT0, /*withElse=*/false);
  {
    OpBuilder::InsertionGuard g(b);
    b.setInsertionPointToStart(t0if.thenBlock());
    Value total = b.create<memref::LoadOp>(loc, buf, ValueRange{c0});
    if (red == Red::Mean) {
      Value Kf = b.create<arith::SIToFPOp>(
          loc, f32, b.create<arith::IndexCastOp>(loc, b.getI64Type(), K));
      total = b.create<arith::DivFOp>(loc, total, Kf);
    }
    (void)isMinMax;
    Value sv = isF32 ? total : b.create<arith::TruncFOp>(loc, storeTy, total);
    b.create<memref::StoreOp>(loc, sv, O, ValueRange{m});
  }

  b.setInsertionPointToEnd(&f.getBody().front());
  b.create<gpu::ReturnOp>(loc);
}

struct GenerateROCMReduceKernelPass
    : PassWrapper<GenerateROCMReduceKernelPass, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(GenerateROCMReduceKernelPass)

  StringRef getArgument() const final { return "generate-rocm-reduce-kernel"; }
  StringRef getDescription() const final {
    return "Expand a tessera_rocm.reduce directive into a row-reduction "
           "(sum/mean/max/min) gpu kernel (compiler-generated)";
  }
  void getDependentDialects(DialectRegistry &registry) const final {
    registry.insert<gpu::GPUDialect, scf::SCFDialect, arith::ArithDialect,
                    memref::MemRefDialect>();
  }

  void runOnOperation() override {
    ModuleOp module = getOperation();
    SmallVector<Operation *> directives;
    module.walk([&](Operation *op) {
      if (op->getName().getStringRef() == "tessera_rocm.reduce")
        directives.push_back(op);
    });
    for (Operation *op : directives) {
      auto nameAttr = op->getAttrOfType<StringAttr>("name");
      if (!nameAttr) {
        op->emitError("tessera_rocm.reduce missing name");
        return signalPassFailure();
      }
      StringRef kindStr = "sum";
      if (auto a = op->getAttrOfType<StringAttr>("kind")) kindStr = a.getValue();
      Red red;
      if (kindStr == "sum") red = Red::Sum;
      else if (kindStr == "mean") red = Red::Mean;
      else if (kindStr == "max") red = Red::Max;
      else if (kindStr == "min") red = Red::Min;
      else {
        op->emitError("generate-rocm-reduce-kernel: kind must be sum, mean, "
                      "max, or min (got '") << kindStr << "')";
        return signalPassFailure();
      }
      OpBuilder b(module.getBodyRegion());
      Type storeTy = b.getF32Type();
      if (auto a = op->getAttrOfType<StringAttr>("dtype")) {
        StringRef dt = a.getValue();
        if (dt == "f16" || dt == "float16") storeTy = b.getF16Type();
        else if (dt == "bf16" || dt == "bfloat16") storeTy = b.getBF16Type();
        else if (dt != "f32" && dt != "float32") {
          op->emitError("generate-rocm-reduce-kernel: dtype must be f32, f16, "
                        "or bf16 (got '") << dt << "')";
          return signalPassFailure();
        }
      }
      b.setInsertionPointToEnd(module.getBody());
      Location loc = op->getLoc();
      std::string kname = nameAttr.getValue().str();
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
      emitReduceBody(body, loc, gpuFunc, storeTy, red);
      op->erase();
    }
  }
};

} // namespace

std::unique_ptr<mlir::Pass>
mlir::tessera_rocm::createGenerateROCMReduceKernelPass() {
  return std::make_unique<GenerateROCMReduceKernelPass>();
}
