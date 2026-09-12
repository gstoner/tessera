// Byte-backed FP8 conversion legalization. Arithmetic remains explicitly f32;
// this is not an FP8 matrix-instruction or public frontend capability claim.
#include "Tessera/Transforms/Passes.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include <cmath>
using namespace mlir;
namespace {
Type element(Type t) { if (auto v = dyn_cast<VectorType>(t)) return v.getElementType(); return t; }
Type shape(Type t, Type e) { if (auto v = dyn_cast<VectorType>(t)) return v.clone(e); return e; }
bool format(Type t, int &m, int &bias) {
  if (auto v = dyn_cast<VectorType>(t); v && v.isScalable()) return false;
  t = element(t);
  if (isa<Float8E4M3FNType>(t)) { m = 3; bias = 7; return true; }
  if (isa<Float8E5M2Type>(t)) { m = 2; bias = 15; return true; }
  return false;
}
struct Math {
  PatternRewriter &r; Location l; Type i, f;
  Value op(StringRef name, Type t, ValueRange args) {
    OperationState s(l, name); s.addOperands(args); s.addTypes(t);
    return r.create(s)->getResult(0);
  }
  Value ci(int64_t x) {
    Attribute a = r.getIntegerAttr(element(i), x);
    if (auto v = dyn_cast<VectorType>(i)) a = DenseElementsAttr::get(v, a);
    return r.create<arith::ConstantOp>(l, i, cast<TypedAttr>(a));
  }
  Value cf(double x) {
    Attribute a = r.getFloatAttr(element(f), x);
    if (auto v = dyn_cast<VectorType>(f)) a = DenseElementsAttr::get(v, a);
    return r.create<arith::ConstantOp>(l, f, cast<TypedAttr>(a));
  }
  Value bin(StringRef n, Value a, Value b) { return op(n, a.getType(), {a,b}); }
  Value cmp(arith::CmpIPredicate p, Value a, Value b) { return r.create<arith::CmpIOp>(l,p,a,b); }
  Value sel(Value c, Value a, Value b) { return r.create<arith::SelectOp>(l,c,a,b); }
  Value bit(Value a, Type t) { return op("arith.bitcast",t,a); }
  Value decode(Value byte, int m, int bias) {
    auto x = op("arith.extui",i,byte);
    auto sign = bin("arith.shli",bin("arith.andi",x,ci(128)),ci(24));
    auto exponent = bin("arith.andi",bin("arith.shrui",x,ci(m)),ci(m==3?15:31));
    auto mantissa = bin("arith.andi",x,ci((1<<m)-1));
    auto normal = bin("arith.ori",bin("arith.shli",bin("arith.addi",exponent,ci(127-bias)),ci(23)),bin("arith.shli",mantissa,ci(23-m)));
    auto sub = bin("arith.mulf",op("arith.uitofp",f,mantissa),cf(std::ldexp(1.,1-bias-m)));
    auto bits = sel(cmp(arith::CmpIPredicate::eq,exponent,ci(0)),bit(sub,i),normal);
    auto magnitude = bin("arith.andi",x,ci(127));
    if (m==3) bits = sel(cmp(arith::CmpIPredicate::eq,magnitude,ci(127)),ci(0x7fc00000),bits);
    else {
      bits = sel(cmp(arith::CmpIPredicate::eq,magnitude,ci(124)),ci(0x7f800000),bits);
      bits = sel(cmp(arith::CmpIPredicate::ugt,magnitude,ci(124)),ci(0x7fc00000),bits);
    }
    return bit(bin("arith.ori",bits,sign),f);
  }
  Value encode(Value value, Type output, int m, int bias) {
    auto bits = bit(value,i);
    auto sign = bin("arith.andi",bin("arith.shrui",bits,ci(24)),ci(128));
    auto abs = bin("arith.andi",bits,ci(0x7fffffff));
    int shift=23-m;
    auto tie = bin("arith.andi",bin("arith.shrui",abs,ci(shift)),ci(1));
    auto rounded = bin("arith.shrui",bin("arith.addi",abs,bin("arith.addi",ci((1<<(shift-1))-1),tie)),ci(shift));
    auto normal = bin("arith.subi",rounded,ci((127-bias)<<m));
    auto small = cmp(arith::CmpIPredicate::ult,abs,ci((128-bias)<<23));
    // Both arms execute: sanitize before fptoui so NaNs/overflow cannot poison
    // the selected normal result. Small scaled values are bounded by 2^m.
    auto safe = bit(sel(small,abs,ci(0)),f);
    auto scaled = bin("arith.mulf",safe,cf(std::ldexp(1.,bias+m-1)));
    auto floor = op("arith.fptoui",i,scaled);
    auto frac = bin("arith.subf",scaled,op("arith.uitofp",f,floor));
    auto greater = r.create<arith::CmpFOp>(l,arith::CmpFPredicate::OGT,frac,cf(.5));
    auto equal = r.create<arith::CmpFOp>(l,arith::CmpFPredicate::OEQ,frac,cf(.5));
    auto odd = cmp(arith::CmpIPredicate::ne,bin("arith.andi",floor,ci(1)),ci(0));
    auto up = bin("arith.ori",greater,bin("arith.andi",equal,odd));
    auto sub = bin("arith.addi",floor,sel(up,ci(1),ci(0)));
    auto code = sel(small,sub,normal);
    int limit = m==3?127:124;
    code = sel(cmp(arith::CmpIPredicate::uge,code,ci(limit)),ci(limit),code);
    code = sel(cmp(arith::CmpIPredicate::ugt,abs,ci(0x7f800000)),ci(127),code);
    return op("arith.trunci",output,bin("arith.ori",code,sign));
  }
};
struct Decode : OpRewritePattern<arith::ExtFOp> {
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(arith::ExtFOp op, PatternRewriter &r) const override {
    int m,bias; auto input=op.getIn().getDefiningOp<arith::BitcastOp>();
    if (!input || !format(op.getIn().getType(),m,bias) || !element(op.getType()).isF32() || !element(input.getIn().getType()).isInteger(8)) return failure();
    Math math{r,op.getLoc(),shape(op.getType(),r.getI32Type()),op.getType()};
    r.replaceOp(op,math.decode(input.getIn(),m,bias));
    if (input->use_empty()) r.eraseOp(input);
    return success();
  }
};
struct Encode : OpRewritePattern<arith::BitcastOp> {
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(arith::BitcastOp op, PatternRewriter &r) const override {
    int m,bias; auto input=op.getIn().getDefiningOp<arith::TruncFOp>();
    if (!input || !format(input.getType(),m,bias) || !element(input.getIn().getType()).isF32() || !element(op.getType()).isInteger(8)) return failure();
    if (auto mode=input->getAttrOfType<arith::RoundingModeAttr>("roundingmode"))
      if (mode.getValue()!=arith::RoundingMode::to_nearest_even) return failure();
    Math math{r,op.getLoc(),shape(input.getIn().getType(),r.getI32Type()),input.getIn().getType()};
    r.replaceOp(op,math.encode(input.getIn(),op.getType(),m,bias));
    if (input->use_empty()) r.eraseOp(input);
    return success();
  }
};
struct LowpConversions : PassWrapper<LowpConversions,OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(LowpConversions)
  StringRef getArgument() const final { return "tessera-expand-lowp-conversions"; }
  StringRef getDescription() const final { return "Legalize byte-backed scalar/vector E4M3FN/E5M2 f32 conversions with nearest-even rounding"; }
  void getDependentDialects(DialectRegistry &r) const override { r.insert<arith::ArithDialect>(); }
  void runOnOperation() override {
    RewritePatternSet patterns(&getContext()); patterns.add<Decode,Encode>(&getContext());
    if (failed(applyPatternsGreedily(getOperation(),std::move(patterns)))) signalPassFailure();
  }
};
}
namespace tessera { std::unique_ptr<Pass> createLowpConversionsPass() { return std::make_unique<LowpConversions>(); } }
