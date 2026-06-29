//===- GenerateROCMPhiloxKernel.cpp - counter-based Philox-4x32-10 RNG ----===//
//
// Expands a `tessera_rocm.philox` directive into a flat counter-based RNG gpu
// kernel: one thread per counter block b writes 4 uniform-f32 in [0,1) to
// out[4b..4b+3] from Philox-4x32-10 keyed by (seed_lo, seed_hi) with counter
// (cbase+b, 0, 0, 0). Bit-identical to the x86 kernel + the tessera.rng_device
// numpy reference (Salmon et al. 2011; the JAX / cuRAND algorithm). The host
// applies the distribution transform (uniform scale / Box-Muller / dropout).
//
// Args: (out : memref<?xf32>, N : index, seed_lo i32, seed_hi i32,
//        cbase_lo i32, cbase_hi i32). umulhi is i64-extend * then shift-32.
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

void emitPhiloxBody(OpBuilder &b, Location loc, gpu::GPUFuncOp f) {
  Type i32 = b.getIntegerType(32);
  Type i64 = b.getIntegerType(64);
  Type f32 = b.getF32Type();
  auto slt = arith::CmpIPredicate::slt;

  b.setInsertionPointToStart(&f.getBody().front());
  Value O = f.getArgument(0), N = f.getArgument(1);
  Value seedLo = f.getArgument(2), seedHi = f.getArgument(3);
  Value cbaseLo = f.getArgument(4), cbaseHi = f.getArgument(5);

  auto ci32 = [&](uint32_t v) {
    return b.create<arith::ConstantOp>(loc, i32,
                                       b.getIntegerAttr(i32, static_cast<int32_t>(v)));
  };
  Value M0 = ci32(0xD2511F53u), M1 = ci32(0xCD9E8D57u);
  Value W0 = ci32(0x9E3779B9u), W1 = ci32(0xBB67AE85u);
  Value c32mask = b.create<arith::ConstantOp>(loc, i64, b.getIntegerAttr(i64, 32));

  Value bid = b.create<gpu::BlockIdOp>(loc, gpu::Dimension::x);
  Value tid = b.create<gpu::ThreadIdOp>(loc, gpu::Dimension::x);
  Value cBD = b.create<arith::ConstantIndexOp>(loc, BD);
  Value blk = b.create<arith::AddIOp>(
      loc, b.create<arith::MulIOp>(loc, bid, cBD), tid);  // block index b
  Value four = b.create<arith::ConstantIndexOp>(loc, 4);
  Value base = b.create<arith::MulIOp>(loc, blk, four);   // 4b
  // guard: 4b < N (at least one element in this block)
  Value inb = b.create<arith::CmpIOp>(loc, slt, base, N);
  auto ifo = b.create<scf::IfOp>(loc, inb, /*withElse=*/false);
  b.setInsertionPointToStart(ifo.thenBlock());

  // counter (c0,c1,c2,c3) = (cbase+b)_lo/hi, 0, 0  — add b (as i64) to cbase
  Value blkI64 = b.create<arith::IndexCastOp>(loc, i64, blk);
  Value cbase64 = b.create<arith::OrIOp>(
      loc,
      b.create<arith::ShLIOp>(
          loc, b.create<arith::ExtUIOp>(loc, i64, cbaseHi), c32mask),
      b.create<arith::ExtUIOp>(loc, i64, cbaseLo));
  Value ctr64 = b.create<arith::AddIOp>(loc, cbase64, blkI64);
  Value c0 = b.create<arith::TruncIOp>(loc, i32, ctr64);
  Value c1 = b.create<arith::TruncIOp>(
      loc, i32, b.create<arith::ShRUIOp>(loc, ctr64, c32mask));
  Value zero32 = ci32(0u);
  Value c2 = zero32, c3 = zero32;
  Value k0 = seedLo, k1 = seedHi;

  for (int r = 0; r < 10; ++r) {
    if (r > 0) {
      k0 = b.create<arith::AddIOp>(loc, k0, W0);
      k1 = b.create<arith::AddIOp>(loc, k1, W1);
    }
    // p0 = (u64)c0 * M0 ; p1 = (u64)c2 * M1
    Value p0 = b.create<arith::MulIOp>(
        loc, b.create<arith::ExtUIOp>(loc, i64, c0),
        b.create<arith::ExtUIOp>(loc, i64, M0));
    Value p1 = b.create<arith::MulIOp>(
        loc, b.create<arith::ExtUIOp>(loc, i64, c2),
        b.create<arith::ExtUIOp>(loc, i64, M1));
    Value hi0 = b.create<arith::TruncIOp>(
        loc, i32, b.create<arith::ShRUIOp>(loc, p0, c32mask));
    Value lo0 = b.create<arith::TruncIOp>(loc, i32, p0);
    Value hi1 = b.create<arith::TruncIOp>(
        loc, i32, b.create<arith::ShRUIOp>(loc, p1, c32mask));
    Value lo1 = b.create<arith::TruncIOp>(loc, i32, p1);
    Value n0 = b.create<arith::XOrIOp>(loc, b.create<arith::XOrIOp>(loc, hi1, c1), k0);
    Value n2 = b.create<arith::XOrIOp>(loc, b.create<arith::XOrIOp>(loc, hi0, c3), k1);
    c0 = n0; c1 = lo1; c2 = n2; c3 = lo0;
  }

  // write 4 uniforms: out[base+j] = uitofp(c_j) * 2^-32, bounds-checked
  Value inv = b.create<arith::ConstantOp>(loc, f32,
                                          b.getF32FloatAttr(1.0f / 4294967296.0f));
  Value outs[4] = {c0, c1, c2, c3};
  for (int j = 0; j < 4; ++j) {
    Value idx = b.create<arith::AddIOp>(
        loc, base, b.create<arith::ConstantIndexOp>(loc, j));
    Value ok = b.create<arith::CmpIOp>(loc, slt, idx, N);
    auto g = b.create<scf::IfOp>(loc, ok, /*withElse=*/false);
    OpBuilder gb = OpBuilder::atBlockBegin(g.thenBlock());
    Value u = gb.create<arith::MulFOp>(
        loc, gb.create<arith::UIToFPOp>(loc, f32, outs[j]), inv);
    gb.create<memref::StoreOp>(loc, u, O, ValueRange{idx});
  }

  b.setInsertionPointToEnd(&f.getBody().front());
  b.create<gpu::ReturnOp>(loc);
}

struct GenerateROCMPhiloxKernelPass
    : PassWrapper<GenerateROCMPhiloxKernelPass, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(GenerateROCMPhiloxKernelPass)

  StringRef getArgument() const final { return "generate-rocm-philox-kernel"; }
  StringRef getDescription() const final {
    return "Expand a tessera_rocm.philox directive into a counter-based "
           "Philox-4x32-10 uniform-RNG gpu kernel (compiler-generated)";
  }
  void getDependentDialects(DialectRegistry &registry) const final {
    registry.insert<gpu::GPUDialect, scf::SCFDialect, arith::ArithDialect,
                    memref::MemRefDialect>();
  }

  void runOnOperation() override {
    ModuleOp module = getOperation();
    SmallVector<Operation *> directives;
    module.walk([&](Operation *op) {
      if (op->getName().getStringRef() == "tessera_rocm.philox")
        directives.push_back(op);
    });
    for (Operation *op : directives) {
      auto nameAttr = op->getAttrOfType<StringAttr>("name");
      if (!nameAttr) {
        op->emitError("tessera_rocm.philox missing name");
        return signalPassFailure();
      }
      OpBuilder b(module.getBodyRegion());
      b.setInsertionPointToEnd(module.getBody());
      Location loc = op->getLoc();
      std::string kname = nameAttr.getValue().str();
      Type i32 = b.getIntegerType(32);
      Type idxTy = b.getIndexType();
      Type f32 = b.getF32Type();
      auto memF32 = MemRefType::get({ShapedType::kDynamic}, f32);
      auto fnTy = b.getFunctionType(
          {memF32, idxTy, i32, i32, i32, i32}, {});
      auto gpuMod = b.create<gpu::GPUModuleOp>(loc, kname + "_mod");
      b.setInsertionPointToStart(&gpuMod.getBodyRegion().front());
      auto gpuFunc = b.create<gpu::GPUFuncOp>(loc, kname, fnTy);
      gpuFunc->setAttr(gpu::GPUDialect::getKernelFuncAttrName(), b.getUnitAttr());
      OpBuilder body(gpuFunc.getContext());
      emitPhiloxBody(body, loc, gpuFunc);
      op->erase();
    }
  }
};

} // namespace

std::unique_ptr<mlir::Pass>
mlir::tessera_rocm::createGenerateROCMPhiloxKernelPass() {
  return std::make_unique<GenerateROCMPhiloxKernelPass>();
}
