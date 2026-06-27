//===- GenerateROCMArgReduceKernel.cpp - argmax / argmin row reduction ----===//
//
// Expands a `tessera_rocm.argreduce` directive into a row arg-reduction gpu
// kernel — the CUB/rocPRIM `DeviceReduce::ArgMax` / `ArgMin` pattern: each
// thread carries the best (value, index) pair over its strided columns, then a
// `gpu.shuffle xor` butterfly reduces the PAIR within a 32-lane subgroup (no
// LDS), and the 8 per-subgroup partials combine to the row winner:
//
//   kind = "argmax" : O[m] = argmax_k X[m,k]   (index of the row max)
//   kind = "argmin" : O[m] = argmin_k X[m,k]   (index of the row min)
//
// numpy tie-break: the FIRST occurrence wins → on equal value keep the LOWER
// index (min). One workgroup per row, blockDim = 256. Values compared in f32
// regardless of storage dtype; the output is an i32 index. M/K are runtime
// index args. Validated vs np.argmax / np.argmin on gfx1151.
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
#include <utility>

using namespace mlir;

namespace {

static constexpr int64_t BD = 256;
static constexpr int64_t SG = 32;           // shuffle subgroup width
static constexpr int64_t NGROUPS = BD / SG; // per-subgroup partials (= 8)

void emitArgReduceBody(OpBuilder &b, Location loc, gpu::GPUFuncOp f,
                       Type storeTy, bool isMax) {
  MLIRContext *ctx = b.getContext();
  Type f32 = b.getF32Type();
  Type i32 = b.getI32Type();
  bool isF32 = storeTy.isF32();
  auto slt = arith::CmpIPredicate::slt;
  auto eqI = arith::CmpIPredicate::eq;

  auto ws = gpu::AddressSpaceAttr::get(ctx, gpu::AddressSpace::Workgroup);
  auto fT = MemRefType::get({NGROUPS}, f32, MemRefLayoutAttrInterface(), ws);
  auto iT = MemRefType::get({NGROUPS}, i32, MemRefLayoutAttrInterface(), ws);
  Value valBuf = f.addWorkgroupAttribution(fT, loc);
  Value idxBuf = f.addWorkgroupAttribution(iT, loc);

  b.setInsertionPointToStart(&f.getBody().front());
  Value X = f.getArgument(0), O = f.getArgument(1);
  Value M = f.getArgument(2), K = f.getArgument(3);

  auto ci = [&](int64_t v) { return b.create<arith::ConstantIndexOp>(loc, v); };
  auto cI32 = [&](int64_t v) {
    return b.create<arith::ConstantIntOp>(loc, i32, v);
  };
  Value c0 = ci(0), cBD = ci(BD), cSG = ci(SG);
  float ninf = -std::numeric_limits<float>::infinity();
  float pinf = std::numeric_limits<float>::infinity();
  Value identV =
      b.create<arith::ConstantOp>(loc, f32, b.getF32FloatAttr(isMax ? ninf : pinf));
  Value identI = cI32(0);

  // combine (bv,bi) with candidate (v,i): pick the better value; on a tie keep
  // the lower index (numpy first-occurrence).
  auto combine = [&](Value bv, Value bi, Value v,
                     Value i) -> std::pair<Value, Value> {
    auto pred = isMax ? arith::CmpFPredicate::OGT : arith::CmpFPredicate::OLT;
    Value better = b.create<arith::CmpFOp>(loc, pred, v, bv);
    Value eq = b.create<arith::CmpFOp>(loc, arith::CmpFPredicate::OEQ, v, bv);
    Value nv = b.create<arith::SelectOp>(loc, better, v, bv);
    Value minIdx = b.create<arith::MinUIOp>(loc, bi, i);
    Value idxEq = b.create<arith::SelectOp>(loc, eq, minIdx, bi);
    Value ni = b.create<arith::SelectOp>(loc, better, i, idxEq);
    return {nv, ni};
  };

  Value m = b.create<gpu::BlockIdOp>(loc, gpu::Dimension::x);
  Value tid = b.create<gpu::ThreadIdOp>(loc, gpu::Dimension::x);
  Value wSG = cI32(SG);
  Value group = b.create<arith::DivUIOp>(loc, tid, cSG);
  Value laneInSg = b.create<arith::RemUIOp>(loc, tid, cSG);
  Value isLeader = b.create<arith::CmpIOp>(loc, eqI, laneInSg, c0);
  Value rowInb = b.create<arith::CmpIOp>(loc, slt, m, M);
  auto rowIf = b.create<scf::IfOp>(loc, rowInb, /*withElse=*/false);
  b.setInsertionPointToStart(rowIf.thenBlock());
  Value base = b.create<arith::MulIOp>(loc, m, K);

  auto loadF32 = [&](Value idx) -> Value {
    Value v = b.create<memref::LoadOp>(loc, X, ValueRange{idx});
    return isF32 ? v : b.create<arith::ExtFOp>(loc, f32, v);
  };

  // per-thread strided arg-accumulate over the row (identity-seeded pair)
  auto lp = b.create<scf::ForOp>(loc, tid, K, cBD, ValueRange{identV, identI});
  {
    OpBuilder::InsertionGuard g(b);
    b.setInsertionPointToStart(lp.getBody());
    Value c = lp.getInductionVar();
    Value bv = lp.getRegionIterArgs()[0], bi = lp.getRegionIterArgs()[1];
    Value v = loadF32(b.create<arith::AddIOp>(loc, base, c));
    Value ci32 = b.create<arith::IndexCastOp>(loc, i32, c);
    auto [nv, ni] = combine(bv, bi, v, ci32);
    b.create<scf::YieldOp>(loc, ValueRange{nv, ni});
  }

  // warp-shuffle butterfly reduce of the (val,idx) pair within a 32-lane group
  Value accV = lp.getResult(0), accI = lp.getResult(1);
  for (int64_t off = SG / 2; off > 0; off >>= 1) {
    Value offC = cI32(off);
    Value shV = b.create<gpu::ShuffleOp>(loc, accV, offC, wSG,
                                         gpu::ShuffleMode::XOR)
                    .getShuffleResult();
    Value shI = b.create<gpu::ShuffleOp>(loc, accI, offC, wSG,
                                         gpu::ShuffleMode::XOR)
                    .getShuffleResult();
    auto [nv, ni] = combine(accV, accI, shV, shI);
    accV = nv;
    accI = ni;
  }

  // subgroup leaders stage their partial pair to LDS
  auto ldIf = b.create<scf::IfOp>(loc, isLeader, /*withElse=*/false);
  {
    OpBuilder::InsertionGuard g(b);
    b.setInsertionPointToStart(ldIf.thenBlock());
    b.create<memref::StoreOp>(loc, accV, valBuf, ValueRange{group});
    b.create<memref::StoreOp>(loc, accI, idxBuf, ValueRange{group});
  }
  b.create<gpu::BarrierOp>(loc);

  // thread 0 combines the NGROUPS partial pairs + writes the winning index
  Value isT0 = b.create<arith::CmpIOp>(loc, eqI, tid, c0);
  auto t0if = b.create<scf::IfOp>(loc, isT0, /*withElse=*/false);
  {
    OpBuilder::InsertionGuard g(b);
    b.setInsertionPointToStart(t0if.thenBlock());
    Value bv = b.create<memref::LoadOp>(loc, valBuf, ValueRange{c0});
    Value bi = b.create<memref::LoadOp>(loc, idxBuf, ValueRange{c0});
    for (int64_t gi = 1; gi < NGROUPS; ++gi) {
      Value v = b.create<memref::LoadOp>(loc, valBuf, ValueRange{ci(gi)});
      Value i = b.create<memref::LoadOp>(loc, idxBuf, ValueRange{ci(gi)});
      auto [nv, ni] = combine(bv, bi, v, i);
      bv = nv;
      bi = ni;
    }
    b.create<memref::StoreOp>(loc, bi, O, ValueRange{m});
  }

  b.setInsertionPointToEnd(&f.getBody().front());
  b.create<gpu::ReturnOp>(loc);
}

struct GenerateROCMArgReduceKernelPass
    : PassWrapper<GenerateROCMArgReduceKernelPass, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(GenerateROCMArgReduceKernelPass)

  StringRef getArgument() const final { return "generate-rocm-argreduce-kernel"; }
  StringRef getDescription() const final {
    return "Expand a tessera_rocm.argreduce directive into a row argmax/argmin "
           "gpu kernel (warp-shuffle arg-reduce, compiler-generated)";
  }
  void getDependentDialects(DialectRegistry &registry) const final {
    registry.insert<gpu::GPUDialect, scf::SCFDialect, arith::ArithDialect,
                    memref::MemRefDialect>();
  }

  void runOnOperation() override {
    ModuleOp module = getOperation();
    SmallVector<Operation *> directives;
    module.walk([&](Operation *op) {
      if (op->getName().getStringRef() == "tessera_rocm.argreduce")
        directives.push_back(op);
    });
    for (Operation *op : directives) {
      auto nameAttr = op->getAttrOfType<StringAttr>("name");
      if (!nameAttr) {
        op->emitError("tessera_rocm.argreduce missing name");
        return signalPassFailure();
      }
      StringRef kindStr = "argmax";
      if (auto a = op->getAttrOfType<StringAttr>("kind")) kindStr = a.getValue();
      bool isMax;
      if (kindStr == "argmax") isMax = true;
      else if (kindStr == "argmin") isMax = false;
      else {
        op->emitError("generate-rocm-argreduce-kernel: kind must be argmax or "
                      "argmin (got '") << kindStr << "')";
        return signalPassFailure();
      }
      OpBuilder b(module.getBodyRegion());
      Type storeTy = b.getF32Type();
      if (auto a = op->getAttrOfType<StringAttr>("dtype")) {
        StringRef dt = a.getValue();
        if (dt == "f16" || dt == "float16") storeTy = b.getF16Type();
        else if (dt == "bf16" || dt == "bfloat16") storeTy = b.getBF16Type();
        else if (dt != "f32" && dt != "float32") {
          op->emitError("generate-rocm-argreduce-kernel: dtype must be f32, f16, "
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
      Type i32 = b.getI32Type();
      auto inTy = MemRefType::get({ShapedType::kDynamic}, storeTy);
      auto outTy = MemRefType::get({ShapedType::kDynamic}, i32);
      // (X : memref<?xstore>, O : memref<?xi32>, M, K : index)
      auto fnTy = b.getFunctionType({inTy, outTy, idxTy, idxTy}, {});
      auto gpuFunc = b.create<gpu::GPUFuncOp>(loc, kname, fnTy);
      gpuFunc->setAttr(gpu::GPUDialect::getKernelFuncAttrName(),
                       b.getUnitAttr());
      OpBuilder body(gpuFunc.getContext());
      emitArgReduceBody(body, loc, gpuFunc, storeTy, isMax);
      op->erase();
    }
  }
};

} // namespace

std::unique_ptr<mlir::Pass>
mlir::tessera_rocm::createGenerateROCMArgReduceKernelPass() {
  return std::make_unique<GenerateROCMArgReduceKernelPass>();
}
