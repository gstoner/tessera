//===- GenerateROCMPhiloxKernel.cpp - counter-based Philox-4x32-10 RNG ----===//
//
// Expands a `tessera_rocm.philox` directive into a flat counter-based RNG gpu
// kernel: one thread per counter block b writes 4 uniform-f32 in [0,1) to
// out[4b..4b+3] from Philox-4x32-10 keyed by (seed_lo, seed_hi) with counter
// (cbase+b, 0, 0, 0). Bit-identical to the x86 kernel + the tessera.rng_device
// numpy reference (Salmon et al. 2011; the JAX / cuRAND algorithm). The host
// owns the distribution transform (uniform scale / Box-Muller / dropout).
//
// Args: (out : memref<?xf32>, N : index, seed_lo i32, seed_hi i32,
//        cbase_lo i32, cbase_hi i32). umulhi is i64-extend * then shift-32.
//===----------------------------------------------------------------------===//

#include "TesseraROCM/Passes.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Pass/Pass.h"

using namespace mlir;

namespace {

static constexpr int64_t BD = 256;

static Value emitUniformWord(OpBuilder &b, Location loc, Value seedLo,
                             Value seedHi, Value counterBase,
                             Value wordIndex) {
  Type i32 = b.getI32Type(), i64 = b.getI64Type(), f32 = b.getF32Type();
  auto ci32 = [&](uint32_t v) {
    return b.create<arith::ConstantOp>(
        loc, i32, b.getIntegerAttr(i32, static_cast<int32_t>(v)));
  };
  Value four = b.create<arith::ConstantIndexOp>(loc, 4);
  Value block = b.create<arith::DivUIOp>(loc, wordIndex, four);
  Value lane = b.create<arith::RemUIOp>(loc, wordIndex, four);
  Value ctr64 = b.create<arith::AddIOp>(
      loc, counterBase, b.create<arith::IndexCastOp>(loc, i64, block));
  Value shift = b.create<arith::ConstantIntOp>(loc, 32, 64);
  Value c0 = b.create<arith::TruncIOp>(loc, i32, ctr64);
  Value c1 = b.create<arith::TruncIOp>(
      loc, i32, b.create<arith::ShRUIOp>(loc, ctr64, shift));
  Value c2 = ci32(0), c3 = ci32(0), k0 = seedLo, k1 = seedHi;
  Value M0 = ci32(0xD2511F53u), M1 = ci32(0xCD9E8D57u);
  Value W0 = ci32(0x9E3779B9u), W1 = ci32(0xBB67AE85u);
  for (int round = 0; round < 10; ++round) {
    if (round) {
      k0 = b.create<arith::AddIOp>(loc, k0, W0);
      k1 = b.create<arith::AddIOp>(loc, k1, W1);
    }
    Value p0 = b.create<arith::MulIOp>(
        loc, b.create<arith::ExtUIOp>(loc, i64, c0),
        b.create<arith::ExtUIOp>(loc, i64, M0));
    Value p1 = b.create<arith::MulIOp>(
        loc, b.create<arith::ExtUIOp>(loc, i64, c2),
        b.create<arith::ExtUIOp>(loc, i64, M1));
    Value hi0 = b.create<arith::TruncIOp>(
        loc, i32, b.create<arith::ShRUIOp>(loc, p0, shift));
    Value hi1 = b.create<arith::TruncIOp>(
        loc, i32, b.create<arith::ShRUIOp>(loc, p1, shift));
    Value lo0 = b.create<arith::TruncIOp>(loc, i32, p0);
    Value lo1 = b.create<arith::TruncIOp>(loc, i32, p1);
    Value n0 = b.create<arith::XOrIOp>(
        loc, b.create<arith::XOrIOp>(loc, hi1, c1), k0);
    Value n2 = b.create<arith::XOrIOp>(
        loc, b.create<arith::XOrIOp>(loc, hi0, c3), k1);
    c0 = n0; c1 = lo1; c2 = n2; c3 = lo0;
  }
  Value selected = c3;
  Value words[3] = {c0, c1, c2};
  for (int j = 2; j >= 0; --j) {
    Value isLane = b.create<arith::CmpIOp>(
        loc, arith::CmpIPredicate::eq, lane,
        b.create<arith::ConstantIndexOp>(loc, j));
    selected = b.create<arith::SelectOp>(loc, isLane, words[j], selected);
  }
  Value converted = b.create<arith::UIToFPOp>(loc, f32, selected);
  return b.create<arith::MulFOp>(
      loc, converted,
      b.create<arith::ConstantOp>(loc, f32,
                                  b.getF32FloatAttr(1.0f / 4294967296.0f)));
}

static void emitDistributionBody(OpBuilder &b, Location loc, gpu::GPUFuncOp f,
                                 StringRef mode) {
  Type i64 = b.getI64Type(), f32 = b.getF32Type();
  b.setInsertionPointToStart(&f.getBody().front());
  unsigned offset = mode == "dropout" ? 1 : 0;
  Value input = mode == "dropout" ? f.getArgument(0) : Value{};
  Value output = f.getArgument(offset), n = f.getArgument(offset + 1);
  Value seedLo = f.getArgument(offset + 2), seedHi = f.getArgument(offset + 3);
  Value baseLo = f.getArgument(offset + 4), baseHi = f.getArgument(offset + 5);
  Value p0 = f.getArgument(offset + 6), p1 =
      mode == "dropout" ? Value{} : f.getArgument(offset + 7);
  Value shift = b.create<arith::ConstantIntOp>(loc, 32, 64);
  Value base = b.create<arith::OrIOp>(
      loc, b.create<arith::ShLIOp>(
               loc, b.create<arith::ExtUIOp>(loc, i64, baseHi), shift),
      b.create<arith::ExtUIOp>(loc, i64, baseLo));
  Value bid = b.create<gpu::BlockIdOp>(loc, gpu::Dimension::x);
  Value tid = b.create<gpu::ThreadIdOp>(loc, gpu::Dimension::x);
  Value index = b.create<arith::AddIOp>(
      loc, b.create<arith::MulIOp>(
               loc, bid, b.create<arith::ConstantIndexOp>(loc, BD)), tid);
  Value inBounds = b.create<arith::CmpIOp>(
      loc, arith::CmpIPredicate::slt, index, n);
  auto guarded = b.create<scf::IfOp>(loc, inBounds, false);
  b.setInsertionPointToStart(guarded.thenBlock());
  Value result;
  if (mode == "uniform_range") {
    Value u = emitUniformWord(b, loc, seedLo, seedHi, base, index);
    result = b.create<arith::AddFOp>(
        loc, p0, b.create<arith::MulFOp>(
                     loc, b.create<arith::SubFOp>(loc, p1, p0), u));
  } else if (mode == "dropout") {
    Value u = emitUniformWord(b, loc, seedLo, seedHi, base, index);
    Value keep = b.create<arith::CmpFOp>(loc, arith::CmpFPredicate::OGE, u, p0);
    Value one = b.create<arith::ConstantOp>(loc, f32, b.getF32FloatAttr(1.0f));
    Value scale = b.create<arith::DivFOp>(
        loc, one, b.create<arith::SubFOp>(loc, one, p0));
    Value masked = b.create<arith::SelectOp>(
        loc, keep, scale,
        b.create<arith::ConstantOp>(loc, f32, b.getF32FloatAttr(0.0f)));
    result = b.create<arith::MulFOp>(
        loc, b.create<memref::LoadOp>(loc, input, index), masked);
  } else {
    Value two = b.create<arith::ConstantIndexOp>(loc, 2);
    Value pair = b.create<arith::DivUIOp>(loc, index, two);
    Value m = b.create<arith::DivUIOp>(
        loc, b.create<arith::AddIOp>(
                 loc, n, b.create<arith::ConstantIndexOp>(loc, 1)), two);
    Value secondOffset = b.create<arith::AddIOp>(
        loc, b.create<arith::DivUIOp>(
                 loc, b.create<arith::AddIOp>(
                          loc, m, b.create<arith::ConstantIndexOp>(loc, 3)),
                 b.create<arith::ConstantIndexOp>(loc, 4)),
        b.create<arith::ConstantIndexOp>(loc, 1));
    Value secondBase = b.create<arith::AddIOp>(
        loc, base, b.create<arith::IndexCastOp>(loc, i64, secondOffset));
    Value u1 = emitUniformWord(b, loc, seedLo, seedHi, base, pair);
    Value u2 = emitUniformWord(b, loc, seedLo, seedHi, secondBase, pair);
    Value floor = b.create<arith::ConstantOp>(loc, f32,
                                              b.getF32FloatAttr(1.0e-7f));
    u1 = b.create<arith::MaxNumFOp>(loc, u1, floor);
    Value radius = b.create<math::SqrtOp>(
        loc, b.create<arith::MulFOp>(
                 loc, b.create<arith::ConstantOp>(
                          loc, f32, b.getF32FloatAttr(-2.0f)),
                 b.create<math::LogOp>(loc, u1)));
    Value theta = b.create<arith::MulFOp>(
        loc, b.create<arith::ConstantOp>(
                 loc, f32, b.getF32FloatAttr(6.283185307179586f)), u2);
    Value odd = b.create<arith::CmpIOp>(
        loc, arith::CmpIPredicate::eq,
        b.create<arith::RemUIOp>(loc, index, two),
        b.create<arith::ConstantIndexOp>(loc, 1));
    Value trig = b.create<arith::SelectOp>(
        loc, odd, b.create<math::SinOp>(loc, theta),
        b.create<math::CosOp>(loc, theta));
    result = b.create<arith::AddFOp>(
        loc, p0, b.create<arith::MulFOp>(
                     loc, p1, b.create<arith::MulFOp>(loc, radius, trig)));
  }
  b.create<memref::StoreOp>(loc, result, output, index);
  b.setInsertionPointToEnd(&f.getBody().front());
  b.create<gpu::ReturnOp>(loc);
}

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
                    math::MathDialect, memref::MemRefDialect>();
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
      auto modeAttr = op->getAttrOfType<StringAttr>("mode");
      StringRef mode = modeAttr ? modeAttr.getValue() : "uniform_core";
      if (mode != "uniform_core" && mode != "uniform_range" &&
          mode != "normal" && mode != "dropout") {
        op->emitError("tessera_rocm.philox has unknown mode");
        return signalPassFailure();
      }
      Type i32 = b.getIntegerType(32);
      Type idxTy = b.getIndexType();
      Type f32 = b.getF32Type();
      auto memF32 = MemRefType::get({ShapedType::kDynamic}, f32);
      SmallVector<Type> arguments;
      if (mode == "dropout")
        arguments.push_back(memF32);
      arguments.append({memF32, idxTy, i32, i32, i32, i32});
      if (mode != "uniform_core")
        arguments.push_back(f32);
      if (mode == "uniform_range" || mode == "normal")
        arguments.push_back(f32);
      auto fnTy = b.getFunctionType(arguments, {});
      auto gpuMod = b.create<gpu::GPUModuleOp>(loc, kname + "_mod");
      b.setInsertionPointToStart(&gpuMod.getBodyRegion().front());
      auto gpuFunc = b.create<gpu::GPUFuncOp>(loc, kname, fnTy);
      gpuFunc->setAttr(gpu::GPUDialect::getKernelFuncAttrName(), b.getUnitAttr());
      OpBuilder body(gpuFunc.getContext());
      if (mode == "uniform_core")
        emitPhiloxBody(body, loc, gpuFunc);
      else
        emitDistributionBody(body, loc, gpuFunc, mode);
      op->erase();
    }
  }
};

} // namespace

std::unique_ptr<mlir::Pass>
mlir::tessera_rocm::createGenerateROCMPhiloxKernelPass() {
  return std::make_unique<GenerateROCMPhiloxKernelPass>();
}
