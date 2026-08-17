//===- GenerateROCMCoalitionButterflyKernel.cpp - shared Yates butterfly --===//

#include "TesseraROCM/Passes.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"
#include "llvm/Support/MathExtras.h"

using namespace mlir;

namespace {
static constexpr int64_t MaxLdsBytes = 64 * 1024;

static void emitButterfly(OpBuilder &builder, Location loc,
                          gpu::GPUFuncOp function, int64_t size, int64_t half,
                          int64_t sign, int64_t workgroupSize) {
  auto workgroupSpace = gpu::AddressSpaceAttr::get(
      function.getContext(), gpu::AddressSpace::Workgroup);
  Value values = function.addWorkgroupAttribution(
      MemRefType::get({size}, builder.getF64Type(),
                      MemRefLayoutAttrInterface(), workgroupSpace),
      loc);
  builder.setInsertionPointToStart(&function.getBody().front());
  Value input = function.getArgument(0), output = function.getArgument(1);
  Value lane = builder.create<gpu::ThreadIdOp>(loc, gpu::Dimension::x);
  Value system = builder.create<gpu::BlockIdOp>(loc, gpu::Dimension::x);
  Value cSize = builder.create<arith::ConstantIndexOp>(loc, size);
  Value cWorkgroup =
      builder.create<arith::ConstantIndexOp>(loc, workgroupSize);
  Value base = builder.create<arith::MulIOp>(loc, system, cSize);
  auto copy = builder.create<scf::ForOp>(loc, lane, cSize, cWorkgroup);
  {
    OpBuilder::InsertionGuard guard(builder);
    builder.setInsertionPointToStart(copy.getBody());
    Value index = copy.getInductionVar();
    Value global = builder.create<arith::AddIOp>(loc, base, index);
    Value value = builder.create<memref::LoadOp>(loc, input, global);
    Value wide =
        builder.create<arith::ExtFOp>(loc, builder.getF64Type(), value);
    builder.create<memref::StoreOp>(loc, wide, values, index);
  }
  builder.setInsertionPointAfter(copy);
  builder.create<gpu::BarrierOp>(loc);

  for (int64_t stride = 1; stride < size; stride <<= 1) {
    auto loop = builder.create<scf::ForOp>(loc, lane, cSize, cWorkgroup);
    {
      OpBuilder::InsertionGuard guard(builder);
      builder.setInsertionPointToStart(loop.getBody());
      Value index = loop.getInductionVar();
      Value cStride = builder.create<arith::ConstantIndexOp>(loc, stride);
      Value cMask = builder.create<arith::ConstantIndexOp>(loc, stride);
      Value bit = builder.create<arith::AndIOp>(loc, index, cMask);
      Value selected = builder.create<arith::CmpIOp>(
          loc, half ? arith::CmpIPredicate::ne : arith::CmpIPredicate::eq,
          bit, builder.create<arith::ConstantIndexOp>(loc, 0));
      auto update = builder.create<scf::IfOp>(loc, selected, false);
      builder.setInsertionPointToStart(update.thenBlock());
      Value other;
      if (half)
        other = builder.create<arith::SubIOp>(loc, index, cStride);
      else
        other = builder.create<arith::AddIOp>(loc, index, cStride);
      Value destination64 =
          builder.create<memref::LoadOp>(loc, values, index);
      Value source64 = builder.create<memref::LoadOp>(loc, values, other);
      Value result64;
      if (sign > 0)
        result64 =
            builder.create<arith::AddFOp>(loc, destination64, source64);
      else
        result64 =
            builder.create<arith::SubFOp>(loc, destination64, source64);
      builder.create<memref::StoreOp>(loc, result64, values, index);
    }
    builder.setInsertionPointAfter(loop);
    builder.create<gpu::BarrierOp>(loc);
  }
  auto store = builder.create<scf::ForOp>(loc, lane, cSize, cWorkgroup);
  {
    OpBuilder::InsertionGuard guard(builder);
    builder.setInsertionPointToStart(store.getBody());
    Value index = store.getInductionVar();
    Value global = builder.create<arith::AddIOp>(loc, base, index);
    Value wide = builder.create<memref::LoadOp>(loc, values, index);
    Value value =
        builder.create<arith::TruncFOp>(loc, builder.getF32Type(), wide);
    builder.create<memref::StoreOp>(loc, value, output, global);
  }
  builder.setInsertionPointAfter(store);
  builder.create<gpu::ReturnOp>(loc);
}

struct GenerateROCMCoalitionButterflyKernelPass
    : PassWrapper<GenerateROCMCoalitionButterflyKernelPass,
                  OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(
      GenerateROCMCoalitionButterflyKernelPass)
  StringRef getArgument() const final {
    return "generate-rocm-coalition-butterfly-kernel";
  }
  StringRef getDescription() const final {
    return "Expand the shared coalition-lattice Yates Target record";
  }
  void getDependentDialects(DialectRegistry &registry) const final {
    registry.insert<gpu::GPUDialect, scf::SCFDialect, arith::ArithDialect,
                    memref::MemRefDialect>();
  }
  void runOnOperation() override {
    ModuleOp module = getOperation();
    SmallVector<Operation *> directives;
    module.walk([&](Operation *op) {
      if (op->getName().getStringRef() == "tessera_rocm.coalition_butterfly")
        directives.push_back(op);
    });
    for (Operation *op : directives) {
      auto name = op->getAttrOfType<StringAttr>("name");
      auto arch = op->getAttrOfType<StringAttr>("arch");
      auto hash = op->getAttrOfType<StringAttr>("artifact_hash");
      auto batch = op->getAttrOfType<IntegerAttr>("batch");
      auto size = op->getAttrOfType<IntegerAttr>("lattice_size");
      auto players = op->getAttrOfType<IntegerAttr>("players");
      auto half = op->getAttrOfType<IntegerAttr>("half");
      auto sign = op->getAttrOfType<IntegerAttr>("sign");
      auto workgroup = op->getAttrOfType<IntegerAttr>("workgroup_size");
      auto order = op->getAttrOfType<StringAttr>("stage_order");
      auto dtype = op->getAttrOfType<StringAttr>("dtype");
      auto accum = op->getAttrOfType<StringAttr>("accum");
      if (!name || !arch || arch.getValue() != "gfx1151" || !hash ||
          hash.getValue().size() != 64 || !batch || batch.getInt() <= 0 ||
          !size || size.getInt() <= 0 || !llvm::isPowerOf2_64(size.getInt()) ||
          size.getInt() > MaxLdsBytes / static_cast<int64_t>(sizeof(double)) ||
          !players || players.getInt() != llvm::Log2_64(size.getInt()) ||
          !half || (half.getInt() != 0 && half.getInt() != 1) || !sign ||
          (sign.getInt() != -1 && sign.getInt() != 1) || !order ||
          order.getValue() != "ascending_bit_yates_v1" || !workgroup ||
          workgroup.getInt() != std::min<int64_t>(256, size.getInt()) || !dtype ||
          dtype.getValue() != "f32" || !accum || accum.getValue() != "f64") {
        op->emitError("invalid gfx1151 coalition butterfly contract");
        return signalPassFailure();
      }
      OpBuilder builder(module.getBodyRegion());
      builder.setInsertionPointToEnd(module.getBody());
      Location loc = op->getLoc();
      auto gpuModule = builder.create<gpu::GPUModuleOp>(
          loc, name.getValue().str() + "_mod");
      builder.setInsertionPointToStart(&gpuModule.getBodyRegion().front());
      auto memref = MemRefType::get({ShapedType::kDynamic}, builder.getF32Type());
      auto type = builder.getFunctionType({memref, memref}, {});
      auto function =
          builder.create<gpu::GPUFuncOp>(loc, name.getValue(), type);
      function->setAttr(gpu::GPUDialect::getKernelFuncAttrName(),
                        builder.getUnitAttr());
      function->setAttr("tessera.schedule_hash", hash);
      function->setAttr("tessera.block_size",
                        workgroup);
      function->setAttr("tessera.grid_size", batch);
      OpBuilder body(function.getContext());
      emitButterfly(body, loc, function, size.getInt(), half.getInt(),
                    sign.getInt(), workgroup.getInt());
      op->erase();
    }
  }
};
} // namespace

std::unique_ptr<mlir::Pass>
mlir::tessera_rocm::createGenerateROCMCoalitionButterflyKernelPass() {
  return std::make_unique<GenerateROCMCoalitionButterflyKernelPass>();
}
