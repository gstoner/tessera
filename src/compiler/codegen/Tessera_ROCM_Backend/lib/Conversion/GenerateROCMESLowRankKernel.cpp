//===- GenerateROCMESLowRankKernel.cpp - EGGROLL rank-1 correction -------===//

#include "TesseraROCM/Passes.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/APInt.h"

using namespace mlir;

namespace {

static constexpr int64_t kBlockSize = 256;
static constexpr int64_t kSubgroupSize = 32;
static constexpr int64_t kSubgroups = kBlockSize / kSubgroupSize;

static Value i64Constant(OpBuilder &b, Location loc, uint64_t value) {
  Type i64 = b.getI64Type();
  return b.create<arith::ConstantOp>(
      loc, i64, b.getIntegerAttr(i64, llvm::APInt(64, value)));
}

static Value splitmix64(OpBuilder &b, Location loc, Value value) {
  Value z = b.create<arith::AddIOp>(
      loc, value, i64Constant(b, loc, 0x9E3779B97F4A7C15ULL));
  auto mix = [&](Value input, uint64_t multiplier, int shift) {
    Value shifted = b.create<arith::ShRUIOp>(
        loc, input, i64Constant(b, loc, shift));
    Value x = b.create<arith::XOrIOp>(loc, input, shifted);
    return b.create<arith::MulIOp>(
        loc, x, i64Constant(b, loc, multiplier));
  };
  z = mix(z, 0xBF58476D1CE4E5B9ULL, 30);
  z = mix(z, 0x94D049BB133111EBULL, 27);
  return b.create<arith::XOrIOp>(
      loc, z,
      b.create<arith::ShRUIOp>(loc, z, i64Constant(b, loc, 31)));
}

static Value memberSeed(OpBuilder &b, Location loc, Value high, Value low,
                        int64_t epoch, Value pair) {
  Value root = splitmix64(
      b, loc, b.create<arith::XOrIOp>(loc, high, splitmix64(b, loc, low)));
  Value epochKey = splitmix64(
      b, loc,
      b.create<arith::XOrIOp>(
          loc, b.create<arith::XOrIOp>(
                   loc, root, i64Constant(b, loc, 0x45535F45504F4348ULL)),
          i64Constant(b, loc, static_cast<uint64_t>(epoch))));
  return splitmix64(
      b, loc,
      b.create<arith::XOrIOp>(
          loc, b.create<arith::XOrIOp>(
                   loc, epochKey,
                   i64Constant(b, loc, 0x45535F504149525FULL)),
          pair));
}

static Value philoxUniform(OpBuilder &b, Location loc, Value seed,
                           Value uniformIndex, int64_t counterBase) {
  Type i32 = b.getI32Type();
  Type i64 = b.getI64Type();
  Type f32 = b.getF32Type();
  auto ci32 = [&](uint32_t value) {
    return b.create<arith::ConstantOp>(
        loc, i32,
        b.getIntegerAttr(i32, static_cast<int32_t>(value)));
  };
  Value four = b.create<arith::ConstantIndexOp>(loc, 4);
  Value block = b.create<arith::DivUIOp>(loc, uniformIndex, four);
  block = b.create<arith::AddIOp>(
      loc, block, b.create<arith::ConstantIndexOp>(loc, counterBase));
  Value counter = b.create<arith::IndexCastOp>(loc, i64, block);
  Value c0 = b.create<arith::TruncIOp>(loc, i32, counter);
  Value c1 = b.create<arith::TruncIOp>(
      loc, i32,
      b.create<arith::ShRUIOp>(loc, counter, i64Constant(b, loc, 32)));
  Value c2 = ci32(0), c3 = ci32(0);
  Value k0 = b.create<arith::TruncIOp>(loc, i32, seed);
  Value k1 = b.create<arith::TruncIOp>(
      loc, i32,
      b.create<arith::ShRUIOp>(loc, seed, i64Constant(b, loc, 32)));
  Value m0 = ci32(0xD2511F53u), m1 = ci32(0xCD9E8D57u);
  Value w0 = ci32(0x9E3779B9u), w1 = ci32(0xBB67AE85u);
  for (int round = 0; round < 10; ++round) {
    if (round > 0) {
      k0 = b.create<arith::AddIOp>(loc, k0, w0);
      k1 = b.create<arith::AddIOp>(loc, k1, w1);
    }
    Value p0 = b.create<arith::MulIOp>(
        loc, b.create<arith::ExtUIOp>(loc, i64, c0),
        b.create<arith::ExtUIOp>(loc, i64, m0));
    Value p1 = b.create<arith::MulIOp>(
        loc, b.create<arith::ExtUIOp>(loc, i64, c2),
        b.create<arith::ExtUIOp>(loc, i64, m1));
    Value hi0 = b.create<arith::TruncIOp>(
        loc, i32,
        b.create<arith::ShRUIOp>(loc, p0, i64Constant(b, loc, 32)));
    Value hi1 = b.create<arith::TruncIOp>(
        loc, i32,
        b.create<arith::ShRUIOp>(loc, p1, i64Constant(b, loc, 32)));
    Value lo0 = b.create<arith::TruncIOp>(loc, i32, p0);
    Value lo1 = b.create<arith::TruncIOp>(loc, i32, p1);
    Value next0 = b.create<arith::XOrIOp>(
        loc, b.create<arith::XOrIOp>(loc, hi1, c1), k0);
    Value next2 = b.create<arith::XOrIOp>(
        loc, b.create<arith::XOrIOp>(loc, hi0, c3), k1);
    c0 = next0;
    c1 = lo1;
    c2 = next2;
    c3 = lo0;
  }
  Value wordIndex = b.create<arith::RemUIOp>(loc, uniformIndex, four);
  Value word = c3;
  for (int candidate = 2; candidate >= 0; --candidate) {
    Value match = b.create<arith::CmpIOp>(
        loc, arith::CmpIPredicate::eq, wordIndex,
        b.create<arith::ConstantIndexOp>(loc, candidate));
    Value selected = candidate == 0 ? c0 : (candidate == 1 ? c1 : c2);
    word = b.create<arith::SelectOp>(loc, match, selected, word);
  }
  Value inverse = b.create<arith::ConstantOp>(
      loc, f32, b.getF32FloatAttr(1.0f / 4294967296.0f));
  return b.create<arith::MulFOp>(
      loc, b.create<arith::UIToFPOp>(loc, f32, word), inverse);
}

static Value normalAt(OpBuilder &b, Location loc, Value seed, Value index,
                      int64_t normalCount) {
  Value two = b.create<arith::ConstantIndexOp>(loc, 2);
  Value pair = b.create<arith::DivUIOp>(loc, index, two);
  Value parity = b.create<arith::RemUIOp>(loc, index, two);
  int64_t half = (normalCount + 1) / 2;
  int64_t secondCounterBase = (half + 3) / 4 + 1;
  Value u1 = philoxUniform(b, loc, seed, pair, 0);
  Value u2 = philoxUniform(b, loc, seed, pair, secondCounterBase);
  Value lower = b.create<arith::ConstantOp>(
      loc, b.getF32Type(), b.getF32FloatAttr(1.0e-7f));
  Value small = b.create<arith::CmpFOp>(
      loc, arith::CmpFPredicate::OLT, u1, lower);
  u1 = b.create<arith::SelectOp>(loc, small, lower, u1);
  Value minusTwo = b.create<arith::ConstantOp>(
      loc, b.getF32Type(), b.getF32FloatAttr(-2.0f));
  Value radius = b.create<math::SqrtOp>(
      loc, b.create<arith::MulFOp>(loc, minusTwo,
                                   b.create<math::LogOp>(loc, u1)));
  Value twoPi = b.create<arith::ConstantOp>(
      loc, b.getF32Type(), b.getF32FloatAttr(6.2831853071795864769f));
  Value theta = b.create<arith::MulFOp>(loc, twoPi, u2);
  Value cosine = b.create<arith::MulFOp>(
      loc, radius, b.create<math::CosOp>(loc, theta));
  Value sine = b.create<arith::MulFOp>(
      loc, radius, b.create<math::SinOp>(loc, theta));
  Value even = b.create<arith::CmpIOp>(
      loc, arith::CmpIPredicate::eq, parity,
      b.create<arith::ConstantIndexOp>(loc, 0));
  return b.create<arith::SelectOp>(loc, even, cosine, sine);
}

static void emitBody(OpBuilder &b, Location loc, gpu::GPUFuncOp function,
                     int64_t population, int64_t rows, int64_t inDim,
                     int64_t outDim, int64_t epoch, float sigma,
                     bool antithetic) {
  MLIRContext *ctx = b.getContext();
  Type f32 = b.getF32Type();
  Type i64 = b.getI64Type();
  auto workgroup =
      gpu::AddressSpaceAttr::get(ctx, gpu::AddressSpace::Workgroup);
  auto ldsType = [&](int64_t extent, Type type) {
    return MemRefType::get({extent}, type, MemRefLayoutAttrInterface(),
                           workgroup);
  };
  Value partials =
      function.addWorkgroupAttribution(ldsType(kSubgroups, f32), loc);
  Value seedBuffer = function.addWorkgroupAttribution(ldsType(1, i64), loc);
  Value signBuffer = function.addWorkgroupAttribution(ldsType(1, f32), loc);

  b.setInsertionPointToStart(&function.getBody().front());
  Value x = function.getArgument(0), members = function.getArgument(1);
  Value key = function.getArgument(2), output = function.getArgument(3);
  Value block = b.create<gpu::BlockIdOp>(loc, gpu::Dimension::x);
  Value thread = b.create<gpu::ThreadIdOp>(loc, gpu::Dimension::x);
  Value c0 = b.create<arith::ConstantIndexOp>(loc, 0);
  Value cBlock = b.create<arith::ConstantIndexOp>(loc, kBlockSize);
  Value cSubgroup = b.create<arith::ConstantIndexOp>(loc, kSubgroupSize);
  Value total = b.create<arith::ConstantIndexOp>(loc, population * rows);
  Value inBounds = b.create<arith::CmpIOp>(
      loc, arith::CmpIPredicate::slt, block, total);
  auto guarded = b.create<scf::IfOp>(loc, inBounds, /*withElse=*/false);
  b.setInsertionPointToStart(guarded.thenBlock());

  Value rowExtent = b.create<arith::ConstantIndexOp>(loc, rows);
  Value populationIndex = b.create<arith::DivUIOp>(loc, block, rowExtent);

  // Member/key derivation is uniform for the whole row. Do it once rather
  // than once per output element, then publish the seed/sign through LDS.
  Value isThreadZero = b.create<arith::CmpIOp>(
      loc, arith::CmpIPredicate::eq, thread, c0);
  auto derive = b.create<scf::IfOp>(loc, isThreadZero, /*withElse=*/false);
  {
    OpBuilder::InsertionGuard insertionGuard(b);
    b.setInsertionPointToStart(derive.thenBlock());
    Value member =
        b.create<memref::LoadOp>(loc, members, ValueRange{populationIndex});
    Value pair = member;
    Value sign = b.create<arith::ConstantOp>(
        loc, f32, b.getF32FloatAttr(1.0f));
    if (antithetic) {
      pair = b.create<arith::ShRUIOp>(loc, member, i64Constant(b, loc, 1));
      Value oddBits = b.create<arith::AndIOp>(
          loc, member, i64Constant(b, loc, 1));
      Value even = b.create<arith::CmpIOp>(
          loc, arith::CmpIPredicate::eq, oddBits, i64Constant(b, loc, 0));
      Value minusOne = b.create<arith::ConstantOp>(
          loc, f32, b.getF32FloatAttr(-1.0f));
      sign = b.create<arith::SelectOp>(loc, even, sign, minusOne);
    }
    Value keyHigh = b.create<memref::LoadOp>(loc, key, ValueRange{c0});
    Value keyLow = b.create<memref::LoadOp>(
        loc, key,
        ValueRange{b.create<arith::ConstantIndexOp>(loc, 1)});
    b.create<memref::StoreOp>(
        loc, memberSeed(b, loc, keyHigh, keyLow, epoch, pair), seedBuffer,
        ValueRange{c0});
    b.create<memref::StoreOp>(loc, sign, signBuffer, ValueRange{c0});
  }
  b.create<gpu::BarrierOp>(loc);
  Value seed = b.create<memref::LoadOp>(loc, seedBuffer, ValueRange{c0});
  Value sign = b.create<memref::LoadOp>(loc, signBuffer, ValueRange{c0});

  Value zero = b.create<arith::ConstantOp>(
      loc, f32, b.getF32FloatAttr(0.0f));
  Value xRowBase = b.create<arith::MulIOp>(
      loc, block, b.create<arith::ConstantIndexOp>(loc, inDim));
  auto loop = b.create<scf::ForOp>(
      loc, thread, b.create<arith::ConstantIndexOp>(loc, inDim), cBlock,
      ValueRange{zero});
  {
    OpBuilder lb = OpBuilder::atBlockBegin(loop.getBody());
    Value inputColumn = loop.getInductionVar();
    Value xIndex =
        lb.create<arith::AddIOp>(loc, xRowBase, inputColumn);
    Value inputValue = lb.create<memref::LoadOp>(loc, x, ValueRange{xIndex});
    Value factor = normalAt(lb, loc, seed, inputColumn, inDim + outDim);
    Value product = lb.create<arith::MulFOp>(loc, inputValue, factor);
    Value sum = lb.create<arith::AddFOp>(
        loc, loop.getRegionIterArgs().front(), product);
    lb.create<scf::YieldOp>(loc, sum);
  }

  // Wave32 shuffle reduction, followed by eight LDS subgroup partials. Every
  // output lane receives the one x@B scalar for this (population,row) pair.
  Value reduced = loop.getResult(0);
  Value subgroupWidth =
      b.create<arith::ConstantIntOp>(loc, b.getI32Type(), kSubgroupSize);
  for (int64_t offset = kSubgroupSize / 2; offset > 0; offset >>= 1) {
    Value offsetValue =
        b.create<arith::ConstantIntOp>(loc, b.getI32Type(), offset);
    auto shuffled = b.create<gpu::ShuffleOp>(
        loc, reduced, offsetValue, subgroupWidth, gpu::ShuffleMode::XOR);
    reduced =
        b.create<arith::AddFOp>(loc, reduced, shuffled.getShuffleResult());
  }
  Value subgroup = b.create<arith::DivUIOp>(loc, thread, cSubgroup);
  Value lane = b.create<arith::RemUIOp>(loc, thread, cSubgroup);
  Value isLeader = b.create<arith::CmpIOp>(
      loc, arith::CmpIPredicate::eq, lane, c0);
  auto publish = b.create<scf::IfOp>(loc, isLeader, /*withElse=*/false);
  {
    OpBuilder::InsertionGuard insertionGuard(b);
    b.setInsertionPointToStart(publish.thenBlock());
    b.create<memref::StoreOp>(loc, reduced, partials, ValueRange{subgroup});
  }
  b.create<gpu::BarrierOp>(loc);
  Value dot = b.create<memref::LoadOp>(loc, partials, ValueRange{c0});
  for (int64_t group = 1; group < kSubgroups; ++group)
    dot = b.create<arith::AddFOp>(
        loc, dot,
        b.create<memref::LoadOp>(
            loc, partials,
            ValueRange{b.create<arith::ConstantIndexOp>(loc, group)}));

  Value scale =
      b.create<arith::ConstantOp>(loc, f32, b.getF32FloatAttr(sigma));
  Value signedScale = b.create<arith::MulFOp>(loc, sign, scale);
  auto outputs = b.create<scf::ForOp>(
      loc, thread, b.create<arith::ConstantIndexOp>(loc, outDim), cBlock);
  {
    OpBuilder outputBuilder = OpBuilder::atBlockBegin(outputs.getBody());
    Value outputColumn = outputs.getInductionVar();
    Value aIndex = outputBuilder.create<arith::AddIOp>(
        loc, outputColumn,
        outputBuilder.create<arith::ConstantIndexOp>(loc, inDim));
    Value a = normalAt(outputBuilder, loc, seed, aIndex, inDim + outDim);
    Value value = outputBuilder.create<arith::MulFOp>(
        loc, signedScale, outputBuilder.create<arith::MulFOp>(loc, dot, a));
    Value outputIndex = outputBuilder.create<arith::AddIOp>(
        loc,
        outputBuilder.create<arith::MulIOp>(
            loc, block,
            outputBuilder.create<arith::ConstantIndexOp>(loc, outDim)),
        outputColumn);
    outputBuilder.create<memref::StoreOp>(
        loc, value, output, ValueRange{outputIndex});
  }
  b.setInsertionPointAfter(guarded);
  b.create<gpu::ReturnOp>(loc);
}

struct GenerateROCMESLowRankKernelPass
    : PassWrapper<GenerateROCMESLowRankKernelPass, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(GenerateROCMESLowRankKernelPass)

  StringRef getArgument() const final {
    return "generate-rocm-es-low-rank-kernel";
  }
  StringRef getDescription() const final {
    return "Generate the gfx1151 member-keyed EGGROLL rank-1 correction";
  }
  void getDependentDialects(DialectRegistry &registry) const final {
    registry.insert<gpu::GPUDialect, scf::SCFDialect, arith::ArithDialect,
                    math::MathDialect, memref::MemRefDialect>();
  }

  void runOnOperation() override {
    SmallVector<Operation *> directives;
    getOperation().walk([&](Operation *op) {
      if (op->getName().getStringRef() ==
          "tessera_rocm.es_low_rank_correction")
        directives.push_back(op);
    });
    for (Operation *op : directives) {
      auto name = op->getAttrOfType<StringAttr>("name");
      auto hash = op->getAttrOfType<StringAttr>("artifact_hash");
      auto population = op->getAttrOfType<IntegerAttr>("population");
      auto rows = op->getAttrOfType<IntegerAttr>("rows_per_member");
      auto inDim = op->getAttrOfType<IntegerAttr>("in_dim");
      auto outDim = op->getAttrOfType<IntegerAttr>("out_dim");
      auto rank = op->getAttrOfType<IntegerAttr>("rank");
      auto epoch = op->getAttrOfType<IntegerAttr>("epoch");
      auto sigma = op->getAttrOfType<FloatAttr>("sigma");
      auto antithetic = op->getAttrOfType<BoolAttr>("antithetic");
      auto rng = op->getAttrOfType<StringAttr>("rng_algorithm");
      auto version = op->getAttrOfType<IntegerAttr>("rng_version");
      if (!name || !hash || hash.getValue().size() != 64 || !population ||
          population.getInt() <= 0 || !rows || rows.getInt() <= 0 || !inDim ||
          inDim.getInt() <= 0 || !outDim || outDim.getInt() <= 0 || !rank ||
          rank.getInt() != 1 || !epoch || epoch.getInt() < 0 || !sigma ||
          !sigma.getValue().isFinite() || sigma.getValueAsDouble() <= 0.0 ||
          !antithetic || !rng ||
          rng.getValue() != "splitmix64-philox4x32-boxmuller" || !version ||
          version.getInt() != 1) {
        op->emitError("generate-rocm-es-low-rank-kernel: incomplete rank-1 contract");
        return signalPassFailure();
      }
      OpBuilder builder(getOperation().getBodyRegion());
      builder.setInsertionPointToEnd(getOperation().getBody());
      Location loc = op->getLoc();
      auto gpuModule = builder.create<gpu::GPUModuleOp>(
          loc, name.getValue().str() + "_mod");
      builder.setInsertionPointToStart(&gpuModule.getBodyRegion().front());
      Type f32 = builder.getF32Type();
      Type i64 = builder.getI64Type();
      Type f32Buffer = MemRefType::get({ShapedType::kDynamic}, f32);
      Type i64Buffer = MemRefType::get({ShapedType::kDynamic}, i64);
      auto functionType = builder.getFunctionType(
          {f32Buffer, i64Buffer, i64Buffer, f32Buffer}, {});
      auto function = builder.create<gpu::GPUFuncOp>(
          loc, name.getValue(), functionType);
      function->setAttr(gpu::GPUDialect::getKernelFuncAttrName(),
                        builder.getUnitAttr());
      OpBuilder body(function.getContext());
      emitBody(body, loc, function, population.getInt(), rows.getInt(),
               inDim.getInt(), outDim.getInt(), epoch.getInt(),
               static_cast<float>(sigma.getValueAsDouble()),
               antithetic.getValue());
      op->erase();
    }
  }
};

} // namespace

std::unique_ptr<mlir::Pass>
mlir::tessera_rocm::createGenerateROCMESLowRankKernelPass() {
  return std::make_unique<GenerateROCMESLowRankKernelPass>();
}
