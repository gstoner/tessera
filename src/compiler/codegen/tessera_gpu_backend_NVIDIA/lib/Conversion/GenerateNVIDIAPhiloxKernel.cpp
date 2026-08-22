//===- GenerateNVIDIAPhiloxKernel.cpp - stateless Philox4x32-10 RNG -------===//
//
// Expands `tessera_nvidia.philox` into a GPU kernel. One thread owns one
// 128-bit counter block and writes up to four uniform-f32 words. The key and
// counter are explicit ABI arguments: invocation order and host state cannot
// affect the generated stream.
//
// Args: (out : memref<?xf32>, N : index, seed_lo : i32, seed_hi : i32,
//        counter_lo : i32, counter_hi : i32)
//
//===----------------------------------------------------------------------===//

#include "tessera/gpu/BackendRegistration.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"

using namespace mlir;

namespace tessera {
namespace {

static constexpr int64_t kBlockSize = 256;

static Value emitUniformWord(OpBuilder &builder, Location loc, Value key0,
                             Value key1, Value counterBase, Value wordIndex) {
  Type i32 = builder.getI32Type();
  Type i64 = builder.getI64Type();
  Type f32 = builder.getF32Type();
  auto constantI32 = [&](uint32_t value) {
    return builder.create<arith::ConstantOp>(
        loc, i32,
        builder.getIntegerAttr(i32, static_cast<int32_t>(value)));
  };
  Value four = builder.create<arith::ConstantIndexOp>(loc, 4);
  Value block = builder.create<arith::DivUIOp>(loc, wordIndex, four);
  Value lane = builder.create<arith::RemUIOp>(loc, wordIndex, four);
  Value shift32 = builder.create<arith::ConstantIntOp>(loc, 32, 64);
  Value counter64 = builder.create<arith::AddIOp>(
      loc, counterBase, builder.create<arith::IndexCastOp>(loc, i64, block));
  Value word0 = builder.create<arith::TruncIOp>(loc, i32, counter64);
  Value word1 = builder.create<arith::TruncIOp>(
      loc, i32, builder.create<arith::ShRUIOp>(loc, counter64, shift32));
  Value word2 = constantI32(0);
  Value word3 = constantI32(0);
  Value multiplier0 = constantI32(0xD2511F53u);
  Value multiplier1 = constantI32(0xCD9E8D57u);
  Value increment0 = constantI32(0x9E3779B9u);
  Value increment1 = constantI32(0xBB67AE85u);
  for (int round = 0; round < 10; ++round) {
    if (round != 0) {
      key0 = builder.create<arith::AddIOp>(loc, key0, increment0);
      key1 = builder.create<arith::AddIOp>(loc, key1, increment1);
    }
    Value product0 = builder.create<arith::MulIOp>(
        loc, builder.create<arith::ExtUIOp>(loc, i64, word0),
        builder.create<arith::ExtUIOp>(loc, i64, multiplier0));
    Value product1 = builder.create<arith::MulIOp>(
        loc, builder.create<arith::ExtUIOp>(loc, i64, word2),
        builder.create<arith::ExtUIOp>(loc, i64, multiplier1));
    Value high0 = builder.create<arith::TruncIOp>(
        loc, i32, builder.create<arith::ShRUIOp>(loc, product0, shift32));
    Value high1 = builder.create<arith::TruncIOp>(
        loc, i32, builder.create<arith::ShRUIOp>(loc, product1, shift32));
    Value low0 = builder.create<arith::TruncIOp>(loc, i32, product0);
    Value low1 = builder.create<arith::TruncIOp>(loc, i32, product1);
    Value next0 = builder.create<arith::XOrIOp>(
        loc, builder.create<arith::XOrIOp>(loc, high1, word1), key0);
    Value next2 = builder.create<arith::XOrIOp>(
        loc, builder.create<arith::XOrIOp>(loc, high0, word3), key1);
    word0 = next0;
    word1 = low1;
    word2 = next2;
    word3 = low0;
  }
  Value selected = word3;
  Value words[] = {word0, word1, word2};
  for (int candidate = 2; candidate >= 0; --candidate) {
    Value matches = builder.create<arith::CmpIOp>(
        loc, arith::CmpIPredicate::eq, lane,
        builder.create<arith::ConstantIndexOp>(loc, candidate));
    selected = builder.create<arith::SelectOp>(loc, matches, words[candidate],
                                                selected);
  }
  Value converted = builder.create<arith::UIToFPOp>(loc, f32, selected);
  return builder.create<arith::MulFOp>(
      loc, converted,
      builder.create<arith::ConstantOp>(
          loc, f32, builder.getF32FloatAttr(1.0f / 4294967296.0f)));
}

static void emitDistribution(OpBuilder &builder, Location loc,
                             gpu::GPUFuncOp function, StringRef mode) {
  Type i64 = builder.getI64Type();
  Type f32 = builder.getF32Type();
  builder.setInsertionPointToStart(&function.getBody().front());
  unsigned offset = mode == "dropout" ? 1 : 0;
  Value input = mode == "dropout" ? function.getArgument(0) : Value{};
  Value output = function.getArgument(offset);
  Value count = function.getArgument(offset + 1);
  Value key0 = function.getArgument(offset + 2);
  Value key1 = function.getArgument(offset + 3);
  Value counterLo = function.getArgument(offset + 4);
  Value counterHi = function.getArgument(offset + 5);
  Value parameter0 = function.getArgument(offset + 6);
  Value parameter1 = mode == "dropout" ? Value{}
                                         : function.getArgument(offset + 7);
  Value shift32 = builder.create<arith::ConstantIntOp>(loc, 32, 64);
  Value counterBase = builder.create<arith::OrIOp>(
      loc,
      builder.create<arith::ShLIOp>(
          loc, builder.create<arith::ExtUIOp>(loc, i64, counterHi), shift32),
      builder.create<arith::ExtUIOp>(loc, i64, counterLo));
  Value index = builder.create<arith::AddIOp>(
      loc,
      builder.create<arith::MulIOp>(
          loc, builder.create<gpu::BlockIdOp>(loc, gpu::Dimension::x),
          builder.create<arith::ConstantIndexOp>(loc, kBlockSize)),
      builder.create<gpu::ThreadIdOp>(loc, gpu::Dimension::x));
  Value inBounds = builder.create<arith::CmpIOp>(
      loc, arith::CmpIPredicate::slt, index, count);
  auto guarded = builder.create<scf::IfOp>(loc, inBounds, false);
  builder.setInsertionPointToStart(guarded.thenBlock());

  Value result;
  if (mode == "uniform_range") {
    Value uniform =
        emitUniformWord(builder, loc, key0, key1, counterBase, index);
    result = builder.create<arith::AddFOp>(
        loc, parameter0,
        builder.create<arith::MulFOp>(
            loc, builder.create<arith::SubFOp>(loc, parameter1, parameter0),
            uniform));
  } else if (mode == "dropout") {
    Value uniform =
        emitUniformWord(builder, loc, key0, key1, counterBase, index);
    Value keep = builder.create<arith::CmpFOp>(
        loc, arith::CmpFPredicate::OGE, uniform, parameter0);
    Value one = builder.create<arith::ConstantOp>(
        loc, f32, builder.getF32FloatAttr(1.0f));
    Value scale = builder.create<arith::DivFOp>(
        loc, one, builder.create<arith::SubFOp>(loc, one, parameter0));
    Value mask = builder.create<arith::SelectOp>(
        loc, keep, scale,
        builder.create<arith::ConstantOp>(
            loc, f32, builder.getF32FloatAttr(0.0f)));
    result = builder.create<arith::MulFOp>(
        loc, builder.create<memref::LoadOp>(loc, input, index), mask);
  } else {
    Value two = builder.create<arith::ConstantIndexOp>(loc, 2);
    Value pair = builder.create<arith::DivUIOp>(loc, index, two);
    Value pairCount = builder.create<arith::DivUIOp>(
        loc,
        builder.create<arith::AddIOp>(
            loc, count, builder.create<arith::ConstantIndexOp>(loc, 1)),
        two);
    Value secondOffset = builder.create<arith::AddIOp>(
        loc,
        builder.create<arith::DivUIOp>(
            loc,
            builder.create<arith::AddIOp>(
                loc, pairCount,
                builder.create<arith::ConstantIndexOp>(loc, 3)),
            builder.create<arith::ConstantIndexOp>(loc, 4)),
        builder.create<arith::ConstantIndexOp>(loc, 1));
    Value secondBase = builder.create<arith::AddIOp>(
        loc, counterBase,
        builder.create<arith::IndexCastOp>(loc, i64, secondOffset));
    Value uniform1 =
        emitUniformWord(builder, loc, key0, key1, counterBase, pair);
    Value uniform2 =
        emitUniformWord(builder, loc, key0, key1, secondBase, pair);
    uniform1 = builder.create<arith::MaxNumFOp>(
        loc, uniform1,
        builder.create<arith::ConstantOp>(
            loc, f32, builder.getF32FloatAttr(1.0e-7f)));
    Value radius = builder.create<math::SqrtOp>(
        loc, builder.create<arith::MulFOp>(
                 loc,
                 builder.create<arith::ConstantOp>(
                     loc, f32, builder.getF32FloatAttr(-2.0f)),
                 builder.create<math::LogOp>(loc, uniform1)));
    Value theta = builder.create<arith::MulFOp>(
        loc,
        builder.create<arith::ConstantOp>(
            loc, f32, builder.getF32FloatAttr(6.283185307179586f)),
        uniform2);
    Value odd = builder.create<arith::CmpIOp>(
        loc, arith::CmpIPredicate::eq,
        builder.create<arith::RemUIOp>(loc, index, two),
        builder.create<arith::ConstantIndexOp>(loc, 1));
    Value trig = builder.create<arith::SelectOp>(
        loc, odd, builder.create<math::SinOp>(loc, theta),
        builder.create<math::CosOp>(loc, theta));
    result = builder.create<arith::AddFOp>(
        loc, parameter0,
        builder.create<arith::MulFOp>(
            loc, parameter1,
            builder.create<arith::MulFOp>(loc, radius, trig)));
  }
  builder.create<memref::StoreOp>(loc, result, output, index);
  builder.setInsertionPointToEnd(&function.getBody().front());
  builder.create<gpu::ReturnOp>(loc);
}

static void emitUniformCore(OpBuilder &builder, Location loc,
                            gpu::GPUFuncOp function) {
  Type i32 = builder.getI32Type();
  Type i64 = builder.getI64Type();
  Type f32 = builder.getF32Type();
  builder.setInsertionPointToStart(&function.getBody().front());

  Value output = function.getArgument(0);
  Value elementCount = function.getArgument(1);
  Value key0 = function.getArgument(2);
  Value key1 = function.getArgument(3);
  Value counterLo = function.getArgument(4);
  Value counterHi = function.getArgument(5);
  auto constantI32 = [&](uint32_t value) {
    return builder.create<arith::ConstantOp>(
        loc, i32,
        builder.getIntegerAttr(i32, static_cast<int32_t>(value)));
  };

  Value multiplier0 = constantI32(0xD2511F53u);
  Value multiplier1 = constantI32(0xCD9E8D57u);
  Value keyIncrement0 = constantI32(0x9E3779B9u);
  Value keyIncrement1 = constantI32(0xBB67AE85u);
  Value shift32 = builder.create<arith::ConstantIntOp>(loc, 32, 64);

  Value blockId = builder.create<gpu::BlockIdOp>(loc, gpu::Dimension::x);
  Value threadId = builder.create<gpu::ThreadIdOp>(loc, gpu::Dimension::x);
  Value counterBlock = builder.create<arith::AddIOp>(
      loc,
      builder.create<arith::MulIOp>(
          loc, blockId,
          builder.create<arith::ConstantIndexOp>(loc, kBlockSize)),
      threadId);
  Value four = builder.create<arith::ConstantIndexOp>(loc, 4);
  Value outputBase =
      builder.create<arith::MulIOp>(loc, counterBlock, four);
  Value hasOutput = builder.create<arith::CmpIOp>(
      loc, arith::CmpIPredicate::slt, outputBase, elementCount);
  auto guarded = builder.create<scf::IfOp>(loc, hasOutput, false);
  builder.setInsertionPointToStart(guarded.thenBlock());

  Value counter64 = builder.create<arith::OrIOp>(
      loc,
      builder.create<arith::ShLIOp>(
          loc, builder.create<arith::ExtUIOp>(loc, i64, counterHi), shift32),
      builder.create<arith::ExtUIOp>(loc, i64, counterLo));
  counter64 = builder.create<arith::AddIOp>(
      loc, counter64,
      builder.create<arith::IndexCastOp>(loc, i64, counterBlock));
  Value word0 = builder.create<arith::TruncIOp>(loc, i32, counter64);
  Value word1 = builder.create<arith::TruncIOp>(
      loc, i32, builder.create<arith::ShRUIOp>(loc, counter64, shift32));
  Value word2 = constantI32(0);
  Value word3 = constantI32(0);

  for (int round = 0; round < 10; ++round) {
    if (round != 0) {
      key0 = builder.create<arith::AddIOp>(loc, key0, keyIncrement0);
      key1 = builder.create<arith::AddIOp>(loc, key1, keyIncrement1);
    }
    Value product0 = builder.create<arith::MulIOp>(
        loc, builder.create<arith::ExtUIOp>(loc, i64, word0),
        builder.create<arith::ExtUIOp>(loc, i64, multiplier0));
    Value product1 = builder.create<arith::MulIOp>(
        loc, builder.create<arith::ExtUIOp>(loc, i64, word2),
        builder.create<arith::ExtUIOp>(loc, i64, multiplier1));
    Value high0 = builder.create<arith::TruncIOp>(
        loc, i32, builder.create<arith::ShRUIOp>(loc, product0, shift32));
    Value high1 = builder.create<arith::TruncIOp>(
        loc, i32, builder.create<arith::ShRUIOp>(loc, product1, shift32));
    Value low0 = builder.create<arith::TruncIOp>(loc, i32, product0);
    Value low1 = builder.create<arith::TruncIOp>(loc, i32, product1);
    Value next0 = builder.create<arith::XOrIOp>(
        loc, builder.create<arith::XOrIOp>(loc, high1, word1), key0);
    Value next2 = builder.create<arith::XOrIOp>(
        loc, builder.create<arith::XOrIOp>(loc, high0, word3), key1);
    word0 = next0;
    word1 = low1;
    word2 = next2;
    word3 = low0;
  }

  Value inverseTwoTo32 = builder.create<arith::ConstantOp>(
      loc, f32, builder.getF32FloatAttr(1.0f / 4294967296.0f));
  Value words[] = {word0, word1, word2, word3};
  for (int lane = 0; lane < 4; ++lane) {
    Value index = builder.create<arith::AddIOp>(
        loc, outputBase, builder.create<arith::ConstantIndexOp>(loc, lane));
    Value inBounds = builder.create<arith::CmpIOp>(
        loc, arith::CmpIPredicate::slt, index, elementCount);
    auto laneGuard = builder.create<scf::IfOp>(loc, inBounds, false);
    OpBuilder laneBuilder = OpBuilder::atBlockBegin(laneGuard.thenBlock());
    Value uniform = laneBuilder.create<arith::MulFOp>(
        loc, laneBuilder.create<arith::UIToFPOp>(loc, f32, words[lane]),
        inverseTwoTo32);
    laneBuilder.create<memref::StoreOp>(loc, uniform, output,
                                        ValueRange{index});
  }

  builder.setInsertionPointToEnd(&function.getBody().front());
  builder.create<gpu::ReturnOp>(loc);
}

struct GenerateNVIDIAPhiloxKernelPass
    : PassWrapper<GenerateNVIDIAPhiloxKernelPass, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(GenerateNVIDIAPhiloxKernelPass)

  StringRef getArgument() const final {
    return "generate-nvidia-philox-kernel";
  }
  StringRef getDescription() const final {
    return "Expand a typed tessera_nvidia.philox directive into a stateless "
           "Philox4x32-10 uniform GPU kernel";
  }
  void getDependentDialects(DialectRegistry &registry) const final {
    registry.insert<arith::ArithDialect, gpu::GPUDialect,
                    memref::MemRefDialect, scf::SCFDialect>();
  }

  void runOnOperation() override {
    ModuleOp module = getOperation();
    // The standalone opt driver lazily loads only dialects present in the
    // input. Distribution directives contain no Math ops until this pass
    // creates Box-Muller, so make that generated-IR dependency explicit.
    module.getContext()->getOrLoadDialect<math::MathDialect>();
    SmallVector<Operation *> directives;
    module.walk([&](Operation *operation) {
      if (operation->getName().getStringRef() == "tessera_nvidia.philox")
        directives.push_back(operation);
    });
    for (Operation *directive : directives) {
      auto name = directive->getAttrOfType<StringAttr>("name");
      auto mode = directive->getAttrOfType<StringAttr>("mode");
      if (!name || name.getValue().empty()) {
        directive->emitError("requires a non-empty kernel name");
        return signalPassFailure();
      }
      if (!mode || (mode.getValue() != "uniform_core" &&
                    mode.getValue() != "uniform_range" &&
                    mode.getValue() != "normal" &&
                    mode.getValue() != "dropout")) {
        directive->emitError("requires a supported Philox mode");
        return signalPassFailure();
      }

      OpBuilder builder(module.getBodyRegion());
      builder.setInsertionPointToEnd(module.getBody());
      Location loc = directive->getLoc();
      Type i32 = builder.getI32Type();
      Type dynamicF32 = MemRefType::get(
          {ShapedType::kDynamic}, builder.getF32Type());
      SmallVector<Type> arguments;
      if (mode.getValue() == "dropout")
        arguments.push_back(dynamicF32);
      arguments.append(
          {dynamicF32, builder.getIndexType(), i32, i32, i32, i32});
      if (mode.getValue() != "uniform_core")
        arguments.push_back(builder.getF32Type());
      if (mode.getValue() == "uniform_range" || mode.getValue() == "normal")
        arguments.push_back(builder.getF32Type());
      auto functionType = builder.getFunctionType(arguments, {});
      auto gpuModule = builder.create<gpu::GPUModuleOp>(
          loc, name.getValue().str() + "_module");
      builder.setInsertionPointToStart(&gpuModule.getBodyRegion().front());
      auto function = builder.create<gpu::GPUFuncOp>(
          loc, name.getValue(), functionType);
      function->setAttr(gpu::GPUDialect::getKernelFuncAttrName(),
                        builder.getUnitAttr());
      OpBuilder body(function.getContext());
      if (mode.getValue() == "uniform_core")
        emitUniformCore(body, loc, function);
      else
        emitDistribution(body, loc, function, mode.getValue());
      directive->erase();
    }
  }
};

} // namespace

std::unique_ptr<mlir::Pass> createGenerateNVIDIAPhiloxKernelPass() {
  return std::make_unique<GenerateNVIDIAPhiloxKernelPass>();
}

} // namespace tessera
