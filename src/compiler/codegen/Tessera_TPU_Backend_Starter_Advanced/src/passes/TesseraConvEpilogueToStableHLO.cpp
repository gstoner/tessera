#include "tessera/tpu/TesseraTPUPasses.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "stablehlo/dialect/StablehloOps.h"

using namespace mlir;

namespace {
static Value broadcastBiasToNHWC(PatternRewriter &rewriter, Location loc,
                                 Value bias, Value activ) {
  // bias is [C], activ is [N,H,W,C] -> broadcast along N,H,W
  auto actTy = dyn_cast<RankedTensorType>(activ.getType());
  if (!actTy) return bias;
  auto shape = actTy.getShape();
  SmallVector<int64_t,4> bshape = {shape[0], shape[1], shape[2], shape[3]};
  auto biasTy = dyn_cast<RankedTensorType>(bias.getType());
  if (!biasTy) return bias;
  // reshape bias [C] -> [1,1,1,C] then broadcast to [N,H,W,C]
  auto elemTy = biasTy.getElementType();
  auto bias411C = RankedTensorType::get({1,1,1, biasTy.getShape()[0]}, elemTy);
  auto cst1 = rewriter.create<stablehlo::ReshapeOp>(loc, bias411C, bias);
  auto bcast = rewriter.create<stablehlo::BroadcastInDimOp>(
      loc, actTy, cst1,
      /*broadcast_dimensions=*/rewriter.getI64TensorAttr({0,1,2,3}));
  return bcast.getResult();
}

struct LowerConv2DPattern : OpRewritePattern<Operation> {
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(Operation *op, PatternRewriter &rewriter) const override {
    if (op->getName().getStringRef() != "tessera.conv2d") return failure();
    // Expect operands: input[N,H,W,C], weight[KH,KW,C,OC]; attrs: strides, dilations, padding
    if (op->getNumOperands() != 2 || op->getNumResults() != 1) return failure();
    Value input = op->getOperand(0);
    Value weight = op->getOperand(1);
    auto outTy = op->getResult(0).getType();

    // Parse window attrs if present
    DictionaryAttr attrs = op->getAttrDictionary();
    DenseI64ArrayAttr strides = attrs.getAs<DenseI64ArrayAttr>("strides");
    DenseI64ArrayAttr dilations = attrs.getAs<DenseI64ArrayAttr>("dilations");
    DenseI64ArrayAttr padding = attrs.getAs<DenseI64ArrayAttr>("padding"); // flattened [padh0,padh1,padw0,padw1]

    SmallVector<int64_t> strideVals = {1,1};
    SmallVector<int64_t> dilationVals = {1,1};
    SmallVector<int64_t> padVals = {0,0,0,0};
    if (strides)    strideVals.assign(strides.asArrayRef().begin(), strides.asArrayRef().end());
    if (dilations)  dilationVals.assign(dilations.asArrayRef().begin(), dilations.asArrayRef().end());
    if (padding)    padVals.assign(padding.asArrayRef().begin(), padding.asArrayRef().end());

    auto conv = rewriter.create<stablehlo::ConvolutionOp>(
        op->getLoc(), outTy, input, weight,
        /*window_strides=*/rewriter.getI64TensorAttr(strideVals),
        /*padding=*/rewriter.getI64TensorAttr(padVals),
        /*lhs_dilation=*/rewriter.getI64TensorAttr({1,1}),
        /*rhs_dilation=*/rewriter.getI64TensorAttr(dilationVals),
        /*window_reversal=*/rewriter.getI64TensorAttr({0,0}),
        /*dimension_numbers=*/stablehlo::ConvDimensionNumbersAttr::get(
            op->getContext(),
            /*input_batch_dimension=*/0, /*input_feature_dimension=*/3,
            /*input_spatial_dimensions=*/{1,2},
            /*kernel_input_feature_dimension=*/2, /*kernel_output_feature_dimension=*/3,
            /*kernel_spatial_dimensions=*/{0,1},
            /*output_batch_dimension=*/0, /*output_feature_dimension=*/3,
            /*output_spatial_dimensions=*/{1,2}),
        /*feature_group_count=*/rewriter.getI64IntegerAttr(1),
        /*batch_group_count=*/rewriter.getI64IntegerAttr(1),
        /*precision_config=*/ArrayAttr());

    rewriter.replaceOp(op, conv.getResult());
    return success();
  }
};

/// Fused epilogue: conv + bias + GELU (approx)
struct LowerConv2DBiasGELUPattern : OpRewritePattern<Operation> {
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(Operation *op, PatternRewriter &rewriter) const override {
    if (op->getName().getStringRef() != "tessera.conv2d_bias_gelu") return failure();
    if (op->getNumOperands() != 3 || op->getNumResults() != 1) return failure();
    Value input = op->getOperand(0);
    Value weight = op->getOperand(1);
    Value bias = op->getOperand(2);
    auto outTy = op->getResult(0).getType();
    Location loc = op->getLoc();

    // 1) Convolution (NHWC x HWIO -> NHWC)
    auto convTy = outTy;
    auto conv = rewriter.create<stablehlo::ConvolutionOp>(
        loc, convTy, input, weight,
        rewriter.getI64TensorAttr({1,1}), /*padding*/rewriter.getI64TensorAttr({0,0,0,0}),
        rewriter.getI64TensorAttr({1,1}), rewriter.getI64TensorAttr({1,1}),
        rewriter.getI64TensorAttr({0,0}),
        stablehlo::ConvDimensionNumbersAttr::get(
            op->getContext(), 0,3,{1,2}, 2,3,{0,1}, 0,3,{1,2}),
        rewriter.getI64IntegerAttr(1), rewriter.getI64IntegerAttr(1), ArrayAttr());

    // 2) Bias (broadcast [C] to NHWC)
    Value bcast = broadcastBiasToNHWC(rewriter, loc, bias, conv.getResult());
    auto add = rewriter.create<stablehlo::AddOp>(loc, conv.getType(), conv, bcast);

    // 3) GELU approx: 0.5*x*(1+tanh(√(2/π)*(x + 0.044715*x^3)))
    auto ty = dyn_cast<RankedTensorType>(conv.getType());
    auto elemTy = ty ? ty.getElementType() : rewriter.getF32Type();
    auto c05 = rewriter.create<stablehlo::ConstantOp>(loc, DenseElementsAttr::get(ty, rewriter.getFloatAttr(elemTy, 0.5)));
    auto c044715 = rewriter.create<stablehlo::ConstantOp>(loc, DenseElementsAttr::get(ty, rewriter.getFloatAttr(elemTy, 0.044715)));
    auto cSqrt2OverPi = rewriter.create<stablehlo::ConstantOp>(loc, DenseElementsAttr::get(ty, rewriter.getFloatAttr(elemTy, 0.79788458347320556640625))); // ~sqrt(2/pi)

    auto x = add.getResult();
    auto x3 = rewriter.create<stablehlo::MulOp>(loc, ty, rewriter.create<stablehlo::MulOp>(loc, ty, x, x), x);
    auto inner = rewriter.create<stablehlo::AddOp>(loc, ty, x, rewriter.create<stablehlo::MulOp>(loc, ty, c044715, x3));
    auto t = rewriter.create<stablehlo::TanhOp>(loc, ty, rewriter.create<stablehlo::MulOp>(loc, ty, cSqrt2OverPi, inner));
    auto one = rewriter.create<stablehlo::ConstantOp>(loc, DenseElementsAttr::get(ty, rewriter.getFloatAttr(elemTy, 1.0)));
    auto gelu = rewriter.create<stablehlo::MulOp>(loc, ty, c05,
                  rewriter.create<stablehlo::MulOp>(loc, ty, x,
                    rewriter.create<stablehlo::AddOp>(loc, ty, one, t)));
    rewriter.replaceOp(op, gelu.getResult());
    return success();
  }
};

struct TesseraConvEpilogueToStableHLOPass
    : PassWrapper<TesseraConvEpilogueToStableHLOPass, OperationPass<ModuleOp>> {
  StringRef getArgument() const override { return "tessera-lower-conv-to-stablehlo"; }
  StringRef getDescription() const override { return "Lower Tessera conv2d (+epilogues) to StableHLO NHWC forms."; }
  void runOnOperation() override {
    MLIRContext *ctx = &getContext();
    RewritePatternSet patterns(ctx);
    patterns.add<LowerConv2DPattern, LowerConv2DBiasGELUPattern>(ctx);
    if (failed(applyPatternsAndFoldGreedily(getOperation(), std::move(patterns))))
      signalPassFailure();
  }
};
} // namespace

namespace tessera {
std::unique_ptr<mlir::Pass> createLowerTesseraConvToStableHLOPass() {
  return std::make_unique<TesseraConvEpilogueToStableHLOPass>();
}
} // namespace tessera
