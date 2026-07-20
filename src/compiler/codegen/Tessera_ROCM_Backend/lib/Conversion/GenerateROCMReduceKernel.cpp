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
// One workgroup per row (blockIdx.x = m), blockDim = 256. The intra-row reduce
// uses a CUB/rocPRIM-style WARP-SHUFFLE reduction: each thread strided-
// accumulates its elements in-register, then a `gpu.shuffle xor` butterfly
// reduces within a 32-lane subgroup (no LDS), so only the 8 per-subgroup
// partials touch LDS for the final combine — vs the old 8-step LDS tree-reduce
// (8 barriers, 256-wide LDS traffic). Reductions run in f32 regardless of
// storage dtype; M/K are runtime index args. FP add reorders, so results match
// numpy within tolerance (not bit-exact); max/min stay NaN-propagating.
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
static constexpr int64_t SG = 32;          // shuffle subgroup width
static constexpr int64_t NGROUPS = BD / SG; // per-subgroup partials (= 8)

enum class Red { Sum, Mean, Max, Min, Prod };

void emitReduceBody(OpBuilder &b, Location loc, gpu::GPUFuncOp f, Type inputTy,
                    Type outputTy, Red red, bool outerAxisInner,
                    bool innerIsOne) {
  MLIRContext *ctx = b.getContext();
  Type f32 = b.getF32Type();
  bool inputIsF32 = inputTy.isF32();
  bool outputIsF32 = outputTy.isF32();
  auto slt = arith::CmpIPredicate::slt;
  bool isMax = red == Red::Max, isMin = red == Red::Min;
  bool isMinMax = isMax || isMin;

  auto ws = gpu::AddressSpaceAttr::get(ctx, gpu::AddressSpace::Workgroup);
  // LDS holds only the per-subgroup partials (NGROUPS = 8), not all BD threads.
  auto ldsT = MemRefType::get({NGROUPS}, f32, MemRefLayoutAttrInterface(), ws);
  Value buf = f.addWorkgroupAttribution(ldsT, loc);

  b.setInsertionPointToStart(&f.getBody().front());
  Value X = f.getArgument(0), O = f.getArgument(1);
  Value outer = f.getArgument(2), K = f.getArgument(3);
  Value inner = outerAxisInner ? f.getArgument(4) : Value();
  Value M = outerAxisInner ? b.create<arith::MulIOp>(loc, outer, inner)
                           : outer;

  auto ci = [&](int64_t v) { return b.create<arith::ConstantIndexOp>(loc, v); };
  auto cf = [&](float v) {
    return b.create<arith::ConstantOp>(loc, f32, b.getF32FloatAttr(v));
  };
  Value c0 = ci(0), cBD = ci(BD);
  float ninf = -std::numeric_limits<float>::infinity();
  float pinf = std::numeric_limits<float>::infinity();
  bool isProd = red == Red::Prod;
  Value ident = isMax ? cf(ninf) : isMin ? cf(pinf)
                : isProd ? cf(1.0f) : cf(0.0f);

  auto combine = [&](Value a, Value c) -> Value {
    if (isMax) return b.create<arith::MaximumFOp>(loc, a, c);
    if (isMin) return b.create<arith::MinimumFOp>(loc, a, c);
    if (isProd) return b.create<arith::MulFOp>(loc, a, c);
    return b.create<arith::AddFOp>(loc, a, c);
  };

  Value m = b.create<gpu::BlockIdOp>(loc, gpu::Dimension::x);
  Value tid = b.create<gpu::ThreadIdOp>(loc, gpu::Dimension::x);
  Value rowInb = b.create<arith::CmpIOp>(loc, slt, m, M);
  auto rowIf = b.create<scf::IfOp>(loc, rowInb, /*withElse=*/false);
  b.setInsertionPointToStart(rowIf.thenBlock());
  Value base = b.create<arith::MulIOp>(loc, m, K);
  Value stride = ci(1);
  if (outerAxisInner && !innerIsOne) {
    Value outerIndex = b.create<arith::DivUIOp>(loc, m, inner);
    Value innerIndex = b.create<arith::RemUIOp>(loc, m, inner);
    base = b.create<arith::AddIOp>(
        loc,
        b.create<arith::MulIOp>(
            loc, b.create<arith::MulIOp>(loc, outerIndex, K), inner),
        innerIndex);
    stride = inner;
  }

  auto loadF32 = [&](Value idx) -> Value {
    Value v = b.create<memref::LoadOp>(loc, X, ValueRange{idx});
    return inputIsF32 ? v : b.create<arith::ExtFOp>(loc, f32, v);
  };

  // per-thread strided accumulate over the row (identity-seeded)
  auto lp = b.create<scf::ForOp>(loc, tid, K, cBD, ValueRange{ident});
  {
    OpBuilder::InsertionGuard g(b);
    b.setInsertionPointToStart(lp.getBody());
    Value c = lp.getInductionVar();
    Value acc = lp.getRegionIterArgs()[0];
    Value linear = b.create<arith::AddIOp>(
        loc, base, b.create<arith::MulIOp>(loc, c, stride));
    Value v = loadF32(linear);
    b.create<scf::YieldOp>(loc, ValueRange{combine(acc, v)});
  }

  // ── warp-shuffle butterfly reduce within a 32-lane subgroup (no LDS) ──
  // After log2(SG) xor-shuffles every lane holds its 32-lane-group total.
  Type i32 = b.getI32Type();
  Value wSG = b.create<arith::ConstantIntOp>(loc, i32, SG);
  Value acc = lp.getResult(0);
  for (int64_t off = SG / 2; off > 0; off >>= 1) {
    Value offC = b.create<arith::ConstantIntOp>(loc, i32, off);
    auto sh = b.create<gpu::ShuffleOp>(loc, acc, offC, wSG,
                                       gpu::ShuffleMode::XOR);
    acc = combine(acc, sh.getShuffleResult());
  }

  // lane 0 of each subgroup writes its partial to LDS buf[group]
  Value cSG = ci(SG);
  Value group = b.create<arith::DivUIOp>(loc, tid, cSG);
  Value laneInSg = b.create<arith::RemUIOp>(loc, tid, cSG);
  Value isLeader =
      b.create<arith::CmpIOp>(loc, arith::CmpIPredicate::eq, laneInSg, c0);
  auto ldIf = b.create<scf::IfOp>(loc, isLeader, /*withElse=*/false);
  {
    OpBuilder::InsertionGuard g(b);
    b.setInsertionPointToStart(ldIf.thenBlock());
    b.create<memref::StoreOp>(loc, acc, buf, ValueRange{group});
  }
  b.create<gpu::BarrierOp>(loc);

  // thread 0 combines the NGROUPS partials + writes O[m] (mean divides by K)
  Value isT0 = b.create<arith::CmpIOp>(loc, arith::CmpIPredicate::eq, tid, c0);
  auto t0if = b.create<scf::IfOp>(loc, isT0, /*withElse=*/false);
  {
    OpBuilder::InsertionGuard g(b);
    b.setInsertionPointToStart(t0if.thenBlock());
    Value total = b.create<memref::LoadOp>(loc, buf, ValueRange{c0});
    for (int64_t gi = 1; gi < NGROUPS; ++gi)
      total = combine(total,
                      b.create<memref::LoadOp>(loc, buf, ValueRange{ci(gi)}));
    if (red == Red::Mean) {
      Value Kf = b.create<arith::SIToFPOp>(
          loc, f32, b.create<arith::IndexCastOp>(loc, b.getI64Type(), K));
      total = b.create<arith::DivFOp>(loc, total, Kf);
    }
    (void)isMinMax;
    Value sv = outputIsF32
                   ? total
                   : b.create<arith::TruncFOp>(loc, outputTy, total);
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
      else if (kindStr == "prod") red = Red::Prod;
      else {
        op->emitError("generate-rocm-reduce-kernel: kind must be sum, mean, "
                      "max, min, or prod (got '") << kindStr << "')";
        return signalPassFailure();
      }
      OpBuilder b(module.getBodyRegion());
      bool outerAxisInner = false;
      if (auto layout = op->getAttrOfType<StringAttr>("layout")) {
        if (layout.getValue() != "outer_axis_inner") {
          op->emitError("generate-rocm-reduce-kernel: layout must be "
                        "outer_axis_inner when present");
          return signalPassFailure();
        }
        outerAxisInner = true;
      }
      bool innerIsOne = false;
      if (auto attr = op->getAttrOfType<BoolAttr>("inner_is_one"))
        innerIsOne = attr.getValue();
      Type inputTy = b.getF32Type();
      if (auto a = op->getAttrOfType<StringAttr>("dtype")) {
        StringRef dt = a.getValue();
        if (dt == "f16" || dt == "float16") inputTy = b.getF16Type();
        else if (dt == "bf16" || dt == "bfloat16") inputTy = b.getBF16Type();
        else if (dt != "f32" && dt != "float32") {
          op->emitError("generate-rocm-reduce-kernel: dtype must be f32, f16, "
                        "or bf16 (got '") << dt << "')";
          return signalPassFailure();
        }
      }
      Type outputTy = inputTy;
      if (auto a = op->getAttrOfType<StringAttr>("output_dtype")) {
        if (!outerAxisInner || a.getValue() != "f32") {
          op->emitError("generate-rocm-reduce-kernel: typed reduction output "
                        "must be f32");
          return signalPassFailure();
        }
        outputTy = b.getF32Type();
      }
      b.setInsertionPointToEnd(module.getBody());
      Location loc = op->getLoc();
      std::string kname = nameAttr.getValue().str();
      auto gpuMod = b.create<gpu::GPUModuleOp>(loc, kname + "_mod");
      b.setInsertionPointToStart(&gpuMod.getBodyRegion().front());
      Type idxTy = b.getIndexType();
      auto inputMemTy = MemRefType::get({ShapedType::kDynamic}, inputTy);
      auto outputMemTy = MemRefType::get({ShapedType::kDynamic}, outputTy);
      // Legacy: (X, O, M, K). Typed carrier: (X, O, Outer, AxisExtent, Inner).
      SmallVector<Type> arguments{inputMemTy, outputMemTy, idxTy, idxTy};
      if (outerAxisInner)
        arguments.push_back(idxTy);
      auto fnTy = b.getFunctionType(arguments, {});
      auto gpuFunc = b.create<gpu::GPUFuncOp>(loc, kname, fnTy);
      gpuFunc->setAttr(gpu::GPUDialect::getKernelFuncAttrName(),
                       b.getUnitAttr());
      OpBuilder body(gpuFunc.getContext());
      emitReduceBody(body, loc, gpuFunc, inputTy, outputTy, red,
                     outerAxisInner, innerIsOne);
      op->erase();
    }
  }
};

} // namespace

std::unique_ptr<mlir::Pass>
mlir::tessera_rocm::createGenerateROCMReduceKernelPass() {
  return std::make_unique<GenerateROCMReduceKernelPass>();
}
