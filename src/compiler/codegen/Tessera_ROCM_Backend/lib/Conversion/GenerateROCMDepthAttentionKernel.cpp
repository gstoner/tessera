//===- GenerateROCMDepthAttentionKernel.cpp -------------------------------===//
//
// BLOCK-ATTNRES-1 Phase 5: materialize the exact gfx1151 depth-attention
// statistics/merge contract. One workgroup owns one flattened token row.
// Lanes [0, source_count) compute RMS-normalized logits, then output lanes
// replay fixed source tiles and apply the declared max-shifted pairwise merge.
// This intentionally favors a transparent, auditable first physical package;
// exact-device evidence decides subsequent subgroup/vector refinements.
//
//===----------------------------------------------------------------------===//

#include "TesseraROCM/Passes.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"

#include <algorithm>

using namespace mlir;

namespace {

static constexpr int64_t WorkgroupSize = 256;

static Value sourceIndex(OpBuilder &builder, Location loc, Value source,
                         Value row, Value column, Value rows, Value width) {
  Value sourceRow = builder.create<arith::AddIOp>(
      loc, builder.create<arith::MulIOp>(loc, source, rows), row);
  return builder.create<arith::AddIOp>(
      loc, builder.create<arith::MulIOp>(loc, sourceRow, width), column);
}

static void emitDepthAttentionBody(OpBuilder &builder, Location loc,
                                   gpu::GPUFuncOp function,
                                   int64_t sourceCount, int64_t rows,
                                   int64_t width, int64_t sourceTile,
                                   double epsilon) {
  MLIRContext *context = builder.getContext();
  Type f32 = builder.getF32Type();
  auto workgroup = gpu::AddressSpaceAttr::get(
      context, gpu::AddressSpace::Workgroup);
  Value logits = function.addWorkgroupAttribution(
      MemRefType::get({sourceCount}, f32, MemRefLayoutAttrInterface(),
                      workgroup),
      loc);

  builder.setInsertionPointToStart(&function.getBody().front());
  Value query = function.getArgument(0);
  Value sources = function.getArgument(1);
  Value output = function.getArgument(2);
  auto ci = [&](int64_t value) {
    return builder.create<arith::ConstantIndexOp>(loc, value).getResult();
  };
  auto cf = [&](double value) {
    return builder
        .create<arith::ConstantOp>(
            loc, f32, builder.getF32FloatAttr(static_cast<float>(value)))
        .getResult();
  };
  Value c0 = ci(0);
  Value c1 = ci(1);
  Value cRows = ci(rows);
  Value cWidth = ci(width);
  Value cSources = ci(sourceCount);
  Value cWorkgroup = ci(WorkgroupSize);
  Value zero = cf(0.0);
  Value negativeInfinity = cf(-3.4028235e38);
  Value epsilonValue = cf(epsilon);
  Value inverseWidth = cf(1.0 / static_cast<double>(width));
  Value row = builder.create<gpu::BlockIdOp>(loc, gpu::Dimension::x);
  Value lane = builder.create<gpu::ThreadIdOp>(loc, gpu::Dimension::x);

  Value rowInBounds = builder.create<arith::CmpIOp>(
      loc, arith::CmpIPredicate::slt, row, cRows);
  auto rowIf = builder.create<scf::IfOp>(loc, rowInBounds,
                                         /*withElseRegion=*/false);
  builder.setInsertionPointToStart(rowIf.thenBlock());

  Value sourceLane = builder.create<arith::CmpIOp>(
      loc, arith::CmpIPredicate::slt, lane, cSources);
  auto logitIf = builder.create<scf::IfOp>(loc, sourceLane,
                                           /*withElseRegion=*/false);
  {
    OpBuilder::InsertionGuard guard(builder);
    builder.setInsertionPointToStart(logitIf.thenBlock());
    auto squares = builder.create<scf::ForOp>(
        loc, c0, cWidth, c1, ValueRange{zero});
    {
      OpBuilder::InsertionGuard loopGuard(builder);
      builder.setInsertionPointToStart(squares.getBody());
      Value column = squares.getInductionVar();
      Value value = builder.create<memref::LoadOp>(
          loc, sources,
          ValueRange{sourceIndex(builder, loc, lane, row, column, cRows,
                                 cWidth)});
      Value square = builder.create<arith::MulFOp>(loc, value, value);
      Value sum = builder.create<arith::AddFOp>(
          loc, squares.getRegionIterArgs()[0], square);
      builder.create<scf::YieldOp>(loc, ValueRange{sum});
    }
    Value meanSquare = builder.create<arith::MulFOp>(
        loc, squares.getResult(0), inverseWidth);
    Value rms = builder.create<math::SqrtOp>(
        loc, builder.create<arith::AddFOp>(loc, meanSquare, epsilonValue));
    auto dot = builder.create<scf::ForOp>(
        loc, c0, cWidth, c1, ValueRange{zero});
    {
      OpBuilder::InsertionGuard loopGuard(builder);
      builder.setInsertionPointToStart(dot.getBody());
      Value column = dot.getInductionVar();
      Value q = builder.create<memref::LoadOp>(loc, query,
                                               ValueRange{column});
      Value value = builder.create<memref::LoadOp>(
          loc, sources,
          ValueRange{sourceIndex(builder, loc, lane, row, column, cRows,
                                 cWidth)});
      Value product = builder.create<arith::MulFOp>(loc, q, value);
      Value sum = builder.create<arith::AddFOp>(
          loc, dot.getRegionIterArgs()[0], product);
      builder.create<scf::YieldOp>(loc, ValueRange{sum});
    }
    Value normalized =
        builder.create<arith::DivFOp>(loc, dot.getResult(0), rms);
    builder.create<memref::StoreOp>(loc, normalized, logits,
                                    ValueRange{lane});
  }
  builder.create<gpu::BarrierOp>(loc);

  auto columns = builder.create<scf::ForOp>(loc, lane, cWidth, cWorkgroup);
  {
    OpBuilder::InsertionGuard guard(builder);
    builder.setInsertionPointToStart(columns.getBody());
    Value column = columns.getInductionVar();
    Value mergedMaximum = negativeInfinity;
    Value mergedNormalizer = zero;
    Value mergedOutput = zero;

    for (int64_t base = 0; base < sourceCount; base += sourceTile) {
      int64_t end = std::min(base + sourceTile, sourceCount);
      Value chunkMaximum = negativeInfinity;
      for (int64_t source = base; source < end; ++source) {
        Value logit = builder.create<memref::LoadOp>(
            loc, logits, ValueRange{ci(source)});
        chunkMaximum = builder.create<arith::MaximumFOp>(
            loc, chunkMaximum, logit);
      }
      Value chunkNormalizer = zero;
      Value chunkOutput = zero;
      for (int64_t source = base; source < end; ++source) {
        Value logit = builder.create<memref::LoadOp>(
            loc, logits, ValueRange{ci(source)});
        Value weight = builder.create<math::ExpOp>(
            loc, builder.create<arith::SubFOp>(loc, logit, chunkMaximum));
        Value value = builder.create<memref::LoadOp>(
            loc, sources,
            ValueRange{sourceIndex(builder, loc, ci(source), row, column,
                                   cRows, cWidth)});
        chunkNormalizer = builder.create<arith::AddFOp>(
            loc, chunkNormalizer, weight);
        chunkOutput = builder.create<arith::AddFOp>(
            loc, chunkOutput,
            builder.create<arith::MulFOp>(loc, weight, value));
      }
      Value nextMaximum = builder.create<arith::MaximumFOp>(
          loc, mergedMaximum, chunkMaximum);
      Value oldScale = builder.create<math::ExpOp>(
          loc, builder.create<arith::SubFOp>(loc, mergedMaximum,
                                             nextMaximum));
      Value chunkScale = builder.create<math::ExpOp>(
          loc, builder.create<arith::SubFOp>(loc, chunkMaximum,
                                             nextMaximum));
      mergedOutput = builder.create<arith::AddFOp>(
          loc, builder.create<arith::MulFOp>(loc, oldScale, mergedOutput),
          builder.create<arith::MulFOp>(loc, chunkScale, chunkOutput));
      mergedNormalizer = builder.create<arith::AddFOp>(
          loc,
          builder.create<arith::MulFOp>(loc, oldScale, mergedNormalizer),
          builder.create<arith::MulFOp>(loc, chunkScale, chunkNormalizer));
      mergedMaximum = nextMaximum;
    }
    Value finalized = builder.create<arith::DivFOp>(
        loc, mergedOutput, mergedNormalizer);
    Value outputIndex = builder.create<arith::AddIOp>(
        loc, builder.create<arith::MulIOp>(loc, row, cWidth), column);
    builder.create<memref::StoreOp>(loc, finalized, output,
                                    ValueRange{outputIndex});
  }

  builder.setInsertionPointToEnd(&function.getBody().front());
  builder.create<gpu::ReturnOp>(loc);
}

struct GenerateROCMDepthAttentionKernelPass
    : PassWrapper<GenerateROCMDepthAttentionKernelPass,
                  OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(
      GenerateROCMDepthAttentionKernelPass)

  StringRef getArgument() const final {
    return "generate-rocm-depth-attention-kernel";
  }
  StringRef getDescription() const final {
    return "Expand the content-addressed gfx1151 depth-attention Target record";
  }
  void getDependentDialects(DialectRegistry &registry) const final {
    registry.insert<gpu::GPUDialect, scf::SCFDialect, arith::ArithDialect,
                    math::MathDialect, memref::MemRefDialect>();
  }

  void runOnOperation() override {
    ModuleOp module = getOperation();
    SmallVector<Operation *> directives;
    module.walk([&](Operation *op) {
      if (op->getName().getStringRef() == "tessera_rocm.depth_attention")
        directives.push_back(op);
    });
    for (Operation *op : directives) {
      auto name = op->getAttrOfType<StringAttr>("name");
      auto arch = op->getAttrOfType<StringAttr>("arch");
      auto hash = op->getAttrOfType<StringAttr>("artifact_hash");
      auto eps = op->getAttrOfType<FloatAttr>("eps");
      auto sourceCount = op->getAttrOfType<IntegerAttr>("source_count");
      auto rows = op->getAttrOfType<IntegerAttr>("rows");
      auto width = op->getAttrOfType<IntegerAttr>("width");
      auto sourceTile = op->getAttrOfType<IntegerAttr>("source_tile");
      auto dtype = op->getAttrOfType<StringAttr>("dtype");
      auto statistics =
          op->getAttrOfType<StringAttr>("statistics_recurrence");
      auto merge = op->getAttrOfType<StringAttr>("merge_recurrence");
      if (!name || !arch || arch.getValue() != "gfx1151" || !hash ||
          hash.getValue().size() != 64 || !eps ||
          !eps.getValue().isFinite() || eps.getValueAsDouble() <= 0.0 ||
          !sourceCount || sourceCount.getInt() <= 0 ||
          sourceCount.getInt() > WorkgroupSize || !rows || rows.getInt() <= 0 ||
          !width || width.getInt() <= 0 || !sourceTile ||
          sourceTile.getInt() <= 0 || sourceTile.getInt() > sourceCount.getInt() ||
          !dtype || dtype.getValue() != "f32" || !statistics ||
          statistics.getValue() != "rms_key_online_softmax_stats_v1" ||
          !merge || merge.getValue() != "max_shifted_pairwise_merge_v1") {
        op->emitError("invalid BLOCK-ATTNRES-1 gfx1151 Target contract");
        return signalPassFailure();
      }

      OpBuilder builder(module.getBodyRegion());
      builder.setInsertionPointToEnd(module.getBody());
      Location loc = op->getLoc();
      std::string kernelName = name.getValue().str();
      auto gpuModule =
          builder.create<gpu::GPUModuleOp>(loc, kernelName + "_mod");
      builder.setInsertionPointToStart(&gpuModule.getBodyRegion().front());
      auto memref = MemRefType::get({ShapedType::kDynamic}, builder.getF32Type());
      auto functionType = builder.getFunctionType({memref, memref, memref}, {});
      auto function =
          builder.create<gpu::GPUFuncOp>(loc, kernelName, functionType);
      function->setAttr(gpu::GPUDialect::getKernelFuncAttrName(),
                        builder.getUnitAttr());
      function->setAttr("tessera.schedule_hash", hash);
      function->setAttr("tessera.block_size",
                        builder.getI64IntegerAttr(WorkgroupSize));
      function->setAttr("tessera.grid_rows", rows);
      OpBuilder body(function.getContext());
      emitDepthAttentionBody(
          body, loc, function, sourceCount.getInt(), rows.getInt(),
          width.getInt(), sourceTile.getInt(), eps.getValueAsDouble());
      op->erase();
    }
  }
};

} // namespace

std::unique_ptr<mlir::Pass>
mlir::tessera_rocm::createGenerateROCMDepthAttentionKernelPass() {
  return std::make_unique<GenerateROCMDepthAttentionKernelPass>();
}
