//===- GenerateROCMScanKernel.cpp - inclusive prefix scan (cumsum/...) ----===//
//
// Expands a `tessera_rocm.scan` directive into a row inclusive-scan gpu kernel —
// the CUB/rocPRIM `BlockScan` technique reimplemented in Tessera codegen:
//
//   kind = "cumsum"  : O[m,k] = Σ_{j<=k} X[m,j]
//   kind = "cumprod" : O[m,k] = Π_{j<=k} X[m,j]
//   kind = "cummax"  : O[m,k] = max_{j<=k} X[m,j]
//   kind = "cummin"  : O[m,k] = min_{j<=k} X[m,j]
//
// One workgroup per row, blockDim = 256. The row is processed in BD-element
// tiles with a running `row_carry` (prefix of the preceding tiles). Per tile:
//   * warp inclusive scan via `gpu.shuffle up` (Kogge-Stone) within a 32-lane
//     subgroup → each lane holds its in-subgroup prefix;
//   * the per-subgroup totals (LDS, NGROUPS=8) are exclusive-scanned to give
//     each subgroup its offset;
//   * v = combine(v, subgroup_offset, row_carry); write O; advance row_carry by
//     the tile total.
// `combine`/identity per kind (add/0, mul/1, max/-inf, min/+inf). Scan runs in
// f32 regardless of storage dtype. M/K are runtime index args. Validated vs
// np.cumsum / cumprod / maximum.accumulate / minimum.accumulate on gfx1151.
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
static constexpr int64_t SG = 32;
static constexpr int64_t NGROUPS = BD / SG;

enum class Scan { Sum, Prod, Max, Min };

void emitScanBody(OpBuilder &b, Location loc, gpu::GPUFuncOp f, Type storeTy,
                  Scan scan) {
  MLIRContext *ctx = b.getContext();
  Type f32 = b.getF32Type();
  Type i32 = b.getI32Type();
  bool isF32 = storeTy.isF32();
  auto slt = arith::CmpIPredicate::slt;
  auto eqI = arith::CmpIPredicate::eq;
  auto sge = arith::CmpIPredicate::sge;

  auto combine = [&](Value a, Value c) -> Value {
    switch (scan) {
    case Scan::Sum: return b.create<arith::AddFOp>(loc, a, c);
    case Scan::Prod: return b.create<arith::MulFOp>(loc, a, c);
    case Scan::Max: return b.create<arith::MaximumFOp>(loc, a, c);
    case Scan::Min: return b.create<arith::MinimumFOp>(loc, a, c);
    }
    return a;
  };

  auto ws = gpu::AddressSpaceAttr::get(ctx, gpu::AddressSpace::Workgroup);
  auto ldsT = MemRefType::get({NGROUPS}, f32, MemRefLayoutAttrInterface(), ws);
  Value sgTot = f.addWorkgroupAttribution(ldsT, loc);

  b.setInsertionPointToStart(&f.getBody().front());
  Value X = f.getArgument(0), O = f.getArgument(1);
  Value M = f.getArgument(2), K = f.getArgument(3);

  auto ci = [&](int64_t v) { return b.create<arith::ConstantIndexOp>(loc, v); };
  auto cI32 = [&](int64_t v) {
    return b.create<arith::ConstantIntOp>(loc, i32, v);
  };
  Value c0 = ci(0), c1 = ci(1), cBD = ci(BD), cSG = ci(SG);
  Value wSG = cI32(SG);
  float ninf = -std::numeric_limits<float>::infinity();
  float pinf = std::numeric_limits<float>::infinity();
  float idf = scan == Scan::Sum    ? 0.0f
              : scan == Scan::Prod ? 1.0f
              : scan == Scan::Max  ? ninf
                                   : pinf;
  Value ident = b.create<arith::ConstantOp>(loc, f32, b.getF32FloatAttr(idf));

  Value m = b.create<gpu::BlockIdOp>(loc, gpu::Dimension::x);
  Value tid = b.create<gpu::ThreadIdOp>(loc, gpu::Dimension::x);
  Value group = b.create<arith::DivUIOp>(loc, tid, cSG);
  Value laneInSg = b.create<arith::RemUIOp>(loc, tid, cSG);
  Value isLast =
      b.create<arith::CmpIOp>(loc, eqI, laneInSg, ci(SG - 1));
  Value rowInb = b.create<arith::CmpIOp>(loc, slt, m, M);
  auto rowIf = b.create<scf::IfOp>(loc, rowInb, /*withElse=*/false);
  b.setInsertionPointToStart(rowIf.thenBlock());
  Value base = b.create<arith::MulIOp>(loc, m, K);

  // for tile_base in [0, K) step BD, carrying row_carry across tiles.
  auto tileLoop = b.create<scf::ForOp>(loc, c0, K, cBD, ValueRange{ident});
  {
    OpBuilder::InsertionGuard g(b);
    b.setInsertionPointToStart(tileLoop.getBody());
    Value tileBase = tileLoop.getInductionVar();
    Value rowCarry = tileLoop.getRegionIterArgs()[0];
    Value c = b.create<arith::AddIOp>(loc, tileBase, tid);
    Value inb = b.create<arith::CmpIOp>(loc, slt, c, K);

    // v = inb ? load(base+c) : identity
    auto loadIf = b.create<scf::IfOp>(loc, f32, inb, /*withElse=*/true);
    {
      OpBuilder::InsertionGuard g2(b);
      b.setInsertionPointToStart(loadIf.thenBlock());
      Value raw = b.create<memref::LoadOp>(
          loc, X, ValueRange{b.create<arith::AddIOp>(loc, base, c)});
      Value v = isF32 ? raw : b.create<arith::ExtFOp>(loc, f32, raw);
      b.create<scf::YieldOp>(loc, v);
      b.setInsertionPointToStart(loadIf.elseBlock());
      b.create<scf::YieldOp>(loc, ident);
    }
    Value v = loadIf.getResult(0);

    // warp inclusive scan (Kogge-Stone) via gpu.shuffle up within the subgroup.
    // Guard explicitly on laneInSg >= off (don't rely on the shuffle valid bit):
    // lane i combines the value from lane i-off only when that source is in the
    // subgroup; lanes below `off` keep their value.
    for (int64_t off = 1; off < SG; off <<= 1) {
      Value offC = cI32(off);
      auto sh = b.create<gpu::ShuffleOp>(loc, v, offC, wSG,
                                         gpu::ShuffleMode::UP);
      Value doAdd = b.create<arith::CmpIOp>(loc, arith::CmpIPredicate::uge,
                                            laneInSg, ci(off));
      Value add = combine(v, sh.getShuffleResult());
      v = b.create<arith::SelectOp>(loc, doAdd, add, v);
    }

    // last lane of each subgroup publishes the subgroup total
    auto pubIf = b.create<scf::IfOp>(loc, isLast, /*withElse=*/false);
    {
      OpBuilder::InsertionGuard g2(b);
      b.setInsertionPointToStart(pubIf.thenBlock());
      b.create<memref::StoreOp>(loc, v, sgTot, ValueRange{group});
    }
    b.create<gpu::BarrierOp>(loc);

    // exclusive prefix offset of preceding subgroups (loop 0..group), plus the
    // full tile total (loop 0..NGROUPS) for the carry update.
    auto offLoop = b.create<scf::ForOp>(loc, c0, group, c1, ValueRange{ident});
    {
      OpBuilder::InsertionGuard g2(b);
      b.setInsertionPointToStart(offLoop.getBody());
      Value gi = offLoop.getInductionVar();
      Value acc = offLoop.getRegionIterArgs()[0];
      Value t = b.create<memref::LoadOp>(loc, sgTot, ValueRange{gi});
      b.create<scf::YieldOp>(loc, combine(acc, t));
    }
    Value offset = offLoop.getResult(0);
    Value tileTotal = b.create<memref::LoadOp>(loc, sgTot, ValueRange{c0});
    for (int64_t gi = 1; gi < NGROUPS; ++gi)
      tileTotal = combine(
          tileTotal, b.create<memref::LoadOp>(loc, sgTot, ValueRange{ci(gi)}));

    Value scanned = combine(combine(v, offset), rowCarry);
    auto stIf = b.create<scf::IfOp>(loc, inb, /*withElse=*/false);
    {
      OpBuilder::InsertionGuard g2(b);
      b.setInsertionPointToStart(stIf.thenBlock());
      Value sv =
          isF32 ? scanned : b.create<arith::TruncFOp>(loc, storeTy, scanned);
      b.create<memref::StoreOp>(loc, sv, O,
                                ValueRange{b.create<arith::AddIOp>(loc, base, c)});
    }
    Value newCarry = combine(rowCarry, tileTotal);
    b.create<gpu::BarrierOp>(loc);  // before next tile overwrites sgTot
    b.create<scf::YieldOp>(loc, newCarry);
  }
  (void)sge;

  b.setInsertionPointToEnd(&f.getBody().front());
  b.create<gpu::ReturnOp>(loc);
}

struct GenerateROCMScanKernelPass
    : PassWrapper<GenerateROCMScanKernelPass, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(GenerateROCMScanKernelPass)

  StringRef getArgument() const final { return "generate-rocm-scan-kernel"; }
  StringRef getDescription() const final {
    return "Expand a tessera_rocm.scan directive into a row inclusive-scan "
           "(cumsum/cumprod/cummax/cummin) gpu kernel (block-scan, "
           "compiler-generated)";
  }
  void getDependentDialects(DialectRegistry &registry) const final {
    registry.insert<gpu::GPUDialect, scf::SCFDialect, arith::ArithDialect,
                    memref::MemRefDialect>();
  }

  void runOnOperation() override {
    ModuleOp module = getOperation();
    SmallVector<Operation *> directives;
    module.walk([&](Operation *op) {
      if (op->getName().getStringRef() == "tessera_rocm.scan")
        directives.push_back(op);
    });
    for (Operation *op : directives) {
      auto nameAttr = op->getAttrOfType<StringAttr>("name");
      if (!nameAttr) {
        op->emitError("tessera_rocm.scan missing name");
        return signalPassFailure();
      }
      StringRef kindStr = "cumsum";
      if (auto a = op->getAttrOfType<StringAttr>("kind")) kindStr = a.getValue();
      Scan scan;
      if (kindStr == "cumsum") scan = Scan::Sum;
      else if (kindStr == "cumprod") scan = Scan::Prod;
      else if (kindStr == "cummax") scan = Scan::Max;
      else if (kindStr == "cummin") scan = Scan::Min;
      else {
        op->emitError("generate-rocm-scan-kernel: kind must be cumsum, cumprod, "
                      "cummax, or cummin (got '") << kindStr << "')";
        return signalPassFailure();
      }
      OpBuilder b(module.getBodyRegion());
      Type storeTy = b.getF32Type();
      if (auto a = op->getAttrOfType<StringAttr>("dtype")) {
        StringRef dt = a.getValue();
        if (dt == "f16" || dt == "float16") storeTy = b.getF16Type();
        else if (dt == "bf16" || dt == "bfloat16") storeTy = b.getBF16Type();
        else if (dt != "f32" && dt != "float32") {
          op->emitError("generate-rocm-scan-kernel: dtype must be f32, f16, or "
                        "bf16 (got '") << dt << "')";
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
      emitScanBody(body, loc, gpuFunc, storeTy, scan);
      op->erase();
    }
  }
};

} // namespace

std::unique_ptr<mlir::Pass>
mlir::tessera_rocm::createGenerateROCMScanKernelPass() {
  return std::make_unique<GenerateROCMScanKernelPass>();
}
