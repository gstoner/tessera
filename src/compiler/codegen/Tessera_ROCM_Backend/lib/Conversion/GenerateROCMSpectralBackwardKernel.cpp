//===- GenerateROCMSpectralBackwardKernel.cpp ----------------------------===//
// Native gfx1151 consumers for the bounded compound-spectral VJP contract.
//===----------------------------------------------------------------------===//

#include "TesseraROCM/Passes.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"

using namespace mlir;

namespace {

static constexpr int64_t kBlockSize = 256;

static Value globalIndex(OpBuilder &b, Location loc) {
  Value block = b.create<gpu::BlockIdOp>(loc, gpu::Dimension::x);
  Value thread = b.create<gpu::ThreadIdOp>(loc, gpu::Dimension::x);
  Value width = b.create<arith::ConstantIndexOp>(loc, kBlockSize);
  return b.create<arith::AddIOp>(
      loc, b.create<arith::MulIOp>(loc, block, width), thread);
}

static void emitFilterBody(OpBuilder &b, Location loc, gpu::GPUFuncOp fn,
                           int64_t elements) {
  b.setInsertionPointToStart(&fn.getBody().front());
  Value dy = fn.getArgument(0), input = fn.getArgument(1);
  Value filter = fn.getArgument(2), dx = fn.getArgument(3);
  Value dfilter = fn.getArgument(4);
  Value gid = globalIndex(b, loc);
  Value limit = b.create<arith::ConstantIndexOp>(loc, elements);
  Value inBounds = b.create<arith::CmpIOp>(
      loc, arith::CmpIPredicate::slt, gid, limit);
  auto ifOp = b.create<scf::IfOp>(loc, inBounds, false);
  b.setInsertionPointToStart(ifOp.thenBlock());
  Value two = b.create<arith::ConstantIndexOp>(loc, 2);
  Value one = b.create<arith::ConstantIndexOp>(loc, 1);
  Value realIndex = b.create<arith::MulIOp>(loc, gid, two);
  Value imagIndex = b.create<arith::AddIOp>(loc, realIndex, one);
  auto load = [&](Value buffer, Value index) {
    return b.create<memref::LoadOp>(loc, buffer, ValueRange{index}).getResult();
  };
  Value dyr = load(dy, realIndex), dyi = load(dy, imagIndex);
  Value xr = load(input, realIndex), xi = load(input, imagIndex);
  Value fr = load(filter, realIndex), fi = load(filter, imagIndex);
  Value dxr = b.create<arith::AddFOp>(
      loc, b.create<arith::MulFOp>(loc, dyr, fr),
      b.create<arith::MulFOp>(loc, dyi, fi));
  Value dxi = b.create<arith::SubFOp>(
      loc, b.create<arith::MulFOp>(loc, dyi, fr),
      b.create<arith::MulFOp>(loc, dyr, fi));
  Value dfr = b.create<arith::AddFOp>(
      loc, b.create<arith::MulFOp>(loc, dyr, xr),
      b.create<arith::MulFOp>(loc, dyi, xi));
  Value dfi = b.create<arith::SubFOp>(
      loc, b.create<arith::MulFOp>(loc, dyi, xr),
      b.create<arith::MulFOp>(loc, dyr, xi));
  b.create<memref::StoreOp>(loc, dxr, dx, ValueRange{realIndex});
  b.create<memref::StoreOp>(loc, dxi, dx, ValueRange{imagIndex});
  b.create<memref::StoreOp>(loc, dfr, dfilter, ValueRange{realIndex});
  b.create<memref::StoreOp>(loc, dfi, dfilter, ValueRange{imagIndex});
  b.setInsertionPointAfter(ifOp);
  b.create<gpu::ReturnOp>(loc);
}

static void emitConvBody(OpBuilder &b, Location loc, gpu::GPUFuncOp fn,
                         int64_t batch, int64_t outputLength,
                         int64_t inputLength, int64_t kernelLength,
                         float scale) {
  b.setInsertionPointToStart(&fn.getBody().front());
  Value dy = fn.getArgument(0), input = fn.getArgument(1);
  Value kernel = fn.getArgument(2), dx = fn.getArgument(3);
  Value dkernel = fn.getArgument(4);
  Value gid = globalIndex(b, loc);
  Value rowWidth = b.create<arith::ConstantIndexOp>(
      loc, inputLength + kernelLength);
  Value limit = b.create<arith::ConstantIndexOp>(
      loc, batch * (inputLength + kernelLength));
  Value inBounds = b.create<arith::CmpIOp>(
      loc, arith::CmpIPredicate::slt, gid, limit);
  auto ifOp = b.create<scf::IfOp>(loc, inBounds, false);
  b.setInsertionPointToStart(ifOp.thenBlock());
  Value row = b.create<arith::DivUIOp>(loc, gid, rowWidth);
  Value local = b.create<arith::RemUIOp>(loc, gid, rowWidth);
  Value inputLimit = b.create<arith::ConstantIndexOp>(loc, inputLength);
  Value isInput = b.create<arith::CmpIOp>(
      loc, arith::CmpIPredicate::slt, local, inputLimit);
  Value zero = b.create<arith::ConstantOp>(
      loc, b.getF32Type(), b.getF32FloatAttr(0.0f));
  Value lb = b.create<arith::ConstantIndexOp>(loc, 0);
  Value kernelLimit = b.create<arith::ConstantIndexOp>(loc, kernelLength);
  Value step = b.create<arith::ConstantIndexOp>(loc, 1);
  auto branch = b.create<scf::IfOp>(loc, TypeRange{b.getF32Type()}, isInput,
                                    true);
  b.setInsertionPointToStart(branch.thenBlock());
  auto dxLoop = b.create<scf::ForOp>(loc, lb, kernelLimit, step,
                                     ValueRange{zero});
  b.setInsertionPointToStart(dxLoop.getBody());
  Value j = dxLoop.getInductionVar();
  Value dyBase = b.create<arith::MulIOp>(
      loc, row, b.create<arith::ConstantIndexOp>(loc, outputLength));
  Value dyIndex = b.create<arith::AddIOp>(
      loc, dyBase, b.create<arith::AddIOp>(loc, local, j));
  Value kernelBase = b.create<arith::MulIOp>(loc, row, kernelLimit);
  Value kernelIndex = b.create<arith::AddIOp>(loc, kernelBase, j);
  Value product = b.create<arith::MulFOp>(
      loc, b.create<memref::LoadOp>(loc, dy, ValueRange{dyIndex}),
      b.create<memref::LoadOp>(loc, kernel, ValueRange{kernelIndex}));
  Value sum = b.create<arith::AddFOp>(loc, dxLoop.getRegionIterArgs()[0],
                                      product);
  b.create<scf::YieldOp>(loc, sum);
  b.setInsertionPointToEnd(branch.thenBlock());
  b.create<scf::YieldOp>(loc, dxLoop.getResult(0));

  b.setInsertionPointToStart(branch.elseBlock());
  Value kernelLocal = b.create<arith::SubIOp>(loc, local, inputLimit);
  auto dkLoop = b.create<scf::ForOp>(loc, lb, inputLimit, step,
                                     ValueRange{zero});
  b.setInsertionPointToStart(dkLoop.getBody());
  Value i = dkLoop.getInductionVar();
  Value dyBase2 = b.create<arith::MulIOp>(
      loc, row, b.create<arith::ConstantIndexOp>(loc, outputLength));
  Value dyIndex2 = b.create<arith::AddIOp>(
      loc, dyBase2, b.create<arith::AddIOp>(loc, i, kernelLocal));
  Value inputBase = b.create<arith::MulIOp>(loc, row, inputLimit);
  Value inputIndex = b.create<arith::AddIOp>(loc, inputBase, i);
  Value product2 = b.create<arith::MulFOp>(
      loc, b.create<memref::LoadOp>(loc, dy, ValueRange{dyIndex2}),
      b.create<memref::LoadOp>(loc, input, ValueRange{inputIndex}));
  Value sum2 = b.create<arith::AddFOp>(loc, dkLoop.getRegionIterArgs()[0],
                                       product2);
  b.create<scf::YieldOp>(loc, sum2);
  b.setInsertionPointToEnd(branch.elseBlock());
  b.create<scf::YieldOp>(loc, dkLoop.getResult(0));

  b.setInsertionPointAfter(branch);
  Value scaleValue = b.create<arith::ConstantOp>(
      loc, b.getF32Type(), b.getF32FloatAttr(scale));
  Value result = b.create<arith::MulFOp>(loc, branch.getResult(0), scaleValue);
  auto storeBranch = b.create<scf::IfOp>(loc, isInput, false);
  b.setInsertionPointToStart(storeBranch.thenBlock());
  Value dxIndex = b.create<arith::AddIOp>(
      loc, b.create<arith::MulIOp>(loc, row, inputLimit), local);
  b.create<memref::StoreOp>(loc, result, dx, ValueRange{dxIndex});
  b.setInsertionPointAfter(storeBranch);
  Value isKernel = b.create<arith::XOrIOp>(
      loc, isInput,
      b.create<arith::ConstantIntOp>(loc, 1, 1));
  auto kernelStore = b.create<scf::IfOp>(loc, isKernel, false);
  b.setInsertionPointToStart(kernelStore.thenBlock());
  Value kernelLocal2 = b.create<arith::SubIOp>(loc, local, inputLimit);
  Value dkIndex = b.create<arith::AddIOp>(
      loc, b.create<arith::MulIOp>(loc, row,
                                   b.create<arith::ConstantIndexOp>(loc, kernelLength)),
      kernelLocal2);
  b.create<memref::StoreOp>(loc, result, dkernel, ValueRange{dkIndex});
  b.setInsertionPointAfter(kernelStore);
  b.setInsertionPointAfter(ifOp);
  b.create<gpu::ReturnOp>(loc);
}

struct GenerateROCMSpectralBackwardKernelPass
    : PassWrapper<GenerateROCMSpectralBackwardKernelPass,
                  OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(
      GenerateROCMSpectralBackwardKernelPass)
  StringRef getArgument() const final {
    return "generate-rocm-spectral-backward-kernel";
  }
  StringRef getDescription() const final {
    return "Generate bounded native gfx1151 compound spectral adjoints";
  }
  void getDependentDialects(DialectRegistry &registry) const final {
    registry.insert<gpu::GPUDialect, scf::SCFDialect, arith::ArithDialect,
                    memref::MemRefDialect>();
  }
  void runOnOperation() override {
    SmallVector<Operation *> directives;
    getOperation().walk([&](Operation *op) {
      if (op->getName().getStringRef() == "tessera_rocm.spectral_backward")
        directives.push_back(op);
    });
    for (Operation *op : directives) {
      auto name = op->getAttrOfType<StringAttr>("name");
      auto hash = op->getAttrOfType<StringAttr>("artifact_hash");
      auto kind = op->getAttrOfType<StringAttr>("kind");
      if (!name || !hash || hash.getValue().size() != 64 || !kind) {
        op->emitError("native ROCm spectral adjoint contract is incomplete");
        return signalPassFailure();
      }
      int64_t elements = op->getAttrOfType<IntegerAttr>("elements").getInt();
      int64_t batch = op->getAttrOfType<IntegerAttr>("batch").getInt();
      int64_t outputLength =
          op->getAttrOfType<IntegerAttr>("cotangent_length").getInt();
      int64_t inputLength =
          op->getAttrOfType<IntegerAttr>("input_length").getInt();
      int64_t kernelLength =
          op->getAttrOfType<IntegerAttr>("parameter_length").getInt();
      float scale = op->getAttrOfType<FloatAttr>("normalization_scale")
                        .getValueAsDouble();
      bool filter = kind.getValue() == "tessera.spectral_filter";
      bool conv = kind.getValue() == "tessera.spectral_conv";
      if ((!filter && !conv) || (filter && elements <= 0) ||
          (conv && (batch <= 0 || outputLength <= 0 || inputLength <= 0 ||
                    kernelLength <= 0))) {
        op->emitError("compound spectral adjoint kind has no gfx1151 native package");
        return signalPassFailure();
      }
      OpBuilder b(getOperation().getBodyRegion());
      b.setInsertionPointToEnd(getOperation().getBody());
      Location loc = op->getLoc();
      auto gpuModule =
          b.create<gpu::GPUModuleOp>(loc, name.getValue().str() + "_mod");
      b.setInsertionPointToStart(&gpuModule.getBodyRegion().front());
      Type buffer = MemRefType::get({ShapedType::kDynamic}, b.getF32Type());
      auto fn = b.create<gpu::GPUFuncOp>(
          loc, name.getValue(),
          b.getFunctionType({buffer, buffer, buffer, buffer, buffer}, {}));
      fn->setAttr(gpu::GPUDialect::getKernelFuncAttrName(), b.getUnitAttr());
      OpBuilder body(fn.getContext());
      if (filter)
        emitFilterBody(body, loc, fn, elements);
      else
        emitConvBody(body, loc, fn, batch, outputLength, inputLength,
                     kernelLength, scale);
      op->erase();
    }
  }
};

} // namespace

std::unique_ptr<mlir::Pass>
mlir::tessera_rocm::createGenerateROCMSpectralBackwardKernelPass() {
  return std::make_unique<GenerateROCMSpectralBackwardKernelPass>();
}
