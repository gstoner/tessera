//===- GenerateROCMEbmEnergyQuadraticKernel.cpp - per-row 0.5*||x-y||^2 --===//
//
// Expands a `tessera_rocm.ebm_energy_quadratic` directive into a per-row
// reduction gpu kernel computing the dominant EBT / diffusion energy form
//
//   out[b] = 0.5 * Σ_d (x[b,d] - y[b,d])^2
//
// over B rows of D elements (x, y row-major [B, D]). ONE workgroup per row
// (grid = B): blockIdx.x selects the row b; the 256 lanes stride over D,
// accumulate the squared differences, and warp-shuffle sum-reduce via the same
// CUB/rocPRIM butterfly as the softmax / partition kernels (gpu.shuffle xor
// within a 32-lane subgroup → 8 LDS partials → combine). Thread 0 writes
// out[b] = 0.5·s. Reductions run in f32. Matches
// tessera.ebm.energy_quadratic — the concrete energy shared by `ebm_energy` /
// `ebm_energy_quadratic`. CPU analog: tessera_x86_ebm_energy_quadratic_f32.
// Args: (x : memref<?xf32>, y : memref<?xf32>, B : index, D : index,
// out : memref<?xf32>).
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
static constexpr int64_t SG = 32;           // shuffle subgroup width
static constexpr int64_t NGROUPS = BD / SG; // per-subgroup partials (= 8)

void emitEnergyQuadraticBody(OpBuilder &b, Location loc, gpu::GPUFuncOp f) {
  MLIRContext *ctx = b.getContext();
  Type f32 = b.getF32Type();
  Type i32 = b.getI32Type();

  auto ws = gpu::AddressSpaceAttr::get(ctx, gpu::AddressSpace::Workgroup);
  Value red = f.addWorkgroupAttribution(
      MemRefType::get({NGROUPS}, f32, MemRefLayoutAttrInterface(), ws), loc);

  b.setInsertionPointToStart(&f.getBody().front());
  Value X = f.getArgument(0), Y = f.getArgument(1);
  Value B = f.getArgument(2), D = f.getArgument(3), OUT = f.getArgument(4);

  auto ci = [&](int64_t v) { return b.create<arith::ConstantIndexOp>(loc, v); };
  Value c0 = ci(0), cBD = ci(BD), cSG = ci(SG);
  Value zerof = b.create<arith::ConstantOp>(loc, f32, b.getF32FloatAttr(0.0f));
  Value halff = b.create<arith::ConstantOp>(loc, f32, b.getF32FloatAttr(0.5f));

  // One workgroup per row: b = blockIdx.x. Guard b < B (grid may round up).
  Value bid = b.create<gpu::BlockIdOp>(loc, gpu::Dimension::x);
  Value rowInb =
      b.create<arith::CmpIOp>(loc, arith::CmpIPredicate::slt, bid, B);
  auto rowGuard = b.create<scf::IfOp>(loc, rowInb, /*withElse=*/false);
  b.setInsertionPointToStart(rowGuard.thenBlock());

  Value rowOff = b.create<arith::MulIOp>(loc, bid, D); // b * D

  Value tid = b.create<gpu::ThreadIdOp>(loc, gpu::Dimension::x);
  Value wSG = b.create<arith::ConstantIntOp>(loc, i32, SG);
  Value group = b.create<arith::DivUIOp>(loc, tid, cSG);
  Value laneInSg = b.create<arith::RemUIOp>(loc, tid, cSG);
  Value isLeader =
      b.create<arith::CmpIOp>(loc, arith::CmpIPredicate::eq, laneInSg, c0);

  // Local sum of squared differences over this row's D elements.
  Value localSum;
  {
    auto lp = b.create<scf::ForOp>(loc, tid, D, cBD, ValueRange{zerof});
    OpBuilder::InsertionGuard g(b);
    b.setInsertionPointToStart(lp.getBody());
    Value d = lp.getInductionVar();
    Value acc = lp.getRegionIterArgs()[0];
    Value idx = b.create<arith::AddIOp>(loc, rowOff, d);
    Value xv = b.create<memref::LoadOp>(loc, X, ValueRange{idx});
    Value yv = b.create<memref::LoadOp>(loc, Y, ValueRange{idx});
    Value diff = b.create<arith::SubFOp>(loc, xv, yv);
    Value sq = b.create<arith::MulFOp>(loc, diff, diff);
    b.create<scf::YieldOp>(loc,
                           ValueRange{b.create<arith::AddFOp>(loc, acc, sq)});
    localSum = lp.getResult(0);
  }

  // CUB/rocPRIM warp-shuffle sum-reduce (identical to the partition kernel):
  // butterfly within a 32-lane subgroup, stage 8 per-subgroup partials to
  // `red`, then every thread combines them to the broadcast total.
  Value acc = localSum;
  for (int64_t off = SG / 2; off > 0; off >>= 1) {
    Value offC = b.create<arith::ConstantIntOp>(loc, i32, off);
    auto sh =
        b.create<gpu::ShuffleOp>(loc, acc, offC, wSG, gpu::ShuffleMode::XOR);
    acc = b.create<arith::AddFOp>(loc, acc, sh.getShuffleResult());
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
    total = b.create<arith::AddFOp>(
        loc, total, b.create<memref::LoadOp>(loc, red, ValueRange{ci(gi)}));

  // thread 0 writes out[b] = 0.5 · total.
  Value isT0 =
      b.create<arith::CmpIOp>(loc, arith::CmpIPredicate::eq, tid, c0);
  auto wIf = b.create<scf::IfOp>(loc, isT0, /*withElse=*/false);
  {
    OpBuilder::InsertionGuard g(b);
    b.setInsertionPointToStart(wIf.thenBlock());
    Value e = b.create<arith::MulFOp>(loc, halff, total);
    b.create<memref::StoreOp>(loc, e, OUT, ValueRange{bid});
  }

  b.setInsertionPointToEnd(&f.getBody().front());
  b.create<gpu::ReturnOp>(loc);
}

struct GenerateROCMEbmEnergyQuadraticKernelPass
    : PassWrapper<GenerateROCMEbmEnergyQuadraticKernelPass,
                  OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(
      GenerateROCMEbmEnergyQuadraticKernelPass)

  StringRef getArgument() const final {
    return "generate-rocm-ebm-energy-quadratic-kernel";
  }
  StringRef getDescription() const final {
    return "Expand a tessera_rocm.ebm_energy_quadratic directive into a per-row "
           "0.5*||x-y||^2 reduction gpu kernel (compiler-generated)";
  }
  void getDependentDialects(DialectRegistry &registry) const final {
    registry.insert<gpu::GPUDialect, scf::SCFDialect, arith::ArithDialect,
                    memref::MemRefDialect>();
  }

  void runOnOperation() override {
    ModuleOp module = getOperation();
    SmallVector<Operation *> directives;
    module.walk([&](Operation *op) {
      if (op->getName().getStringRef() == "tessera_rocm.ebm_energy_quadratic")
        directives.push_back(op);
    });
    for (Operation *op : directives) {
      auto nameAttr = op->getAttrOfType<StringAttr>("name");
      if (!nameAttr) {
        op->emitError("tessera_rocm.ebm_energy_quadratic missing name");
        return signalPassFailure();
      }
      OpBuilder b(module.getBodyRegion());
      b.setInsertionPointToEnd(module.getBody());
      Location loc = op->getLoc();
      std::string kname = nameAttr.getValue().str();
      Type f32 = b.getF32Type();
      Type idxTy = b.getIndexType();
      auto memF32 = MemRefType::get({ShapedType::kDynamic}, f32);
      // (x : memref<?xf32>, y : memref<?xf32>, B : index, D : index,
      //  out : memref<?xf32>)
      auto fnTy = b.getFunctionType({memF32, memF32, idxTy, idxTy, memF32}, {});
      auto gpuMod = b.create<gpu::GPUModuleOp>(loc, kname + "_mod");
      b.setInsertionPointToStart(&gpuMod.getBodyRegion().front());
      auto gpuFunc = b.create<gpu::GPUFuncOp>(loc, kname, fnTy);
      gpuFunc->setAttr(gpu::GPUDialect::getKernelFuncAttrName(),
                       b.getUnitAttr());
      OpBuilder body(gpuFunc.getContext());
      emitEnergyQuadraticBody(body, loc, gpuFunc);
      op->erase();
    }
  }
};

} // namespace

std::unique_ptr<mlir::Pass>
mlir::tessera_rocm::createGenerateROCMEbmEnergyQuadraticKernelPass() {
  return std::make_unique<GenerateROCMEbmEnergyQuadraticKernelPass>();
}
