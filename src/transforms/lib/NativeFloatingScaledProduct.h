// Compiler-owned continuous exact block-scaled product expansion.
#pragma once
#include "Tessera/IR/ScaledBatchContract.h"

namespace tessera {
using namespace mlir;
static bool isNativeFloatingScaledProduct(Operation *op) {
  if (!isa<ScaledMatmulOp>(op) || op->getNumOperands() != 4 ||
      op->hasAttr("physical_contract")) return false;
  auto module = op->getParentOfType<ModuleOp>();
  auto target = module ? module->getAttrOfType<StringAttr>("tessera.target") : StringAttr{};
  auto arch = module ? module->getAttrOfType<StringAttr>("tessera.arch") : StringAttr{};
  if (!target || target.getValue() != "rocm" || !arch || arch.getValue() != "gfx1201")
    return false;
  for (Value input : op->getOperands()) {
    auto type = dyn_cast<RankedTensorType>(input.getType());
    if (!type || !type.hasStaticShape() || type.getEncoding() ||
        !type.getElementType().isF32()) return false;
  }
  return true;
}

static LogicalResult expandNativeFloatingScaledProduct(Operation *operation) {
  using namespace mlir;
  auto product = cast<ScaledMatmulOp>(operation);
  auto output = dyn_cast<RankedTensorType>(product.getResult().getType());
  SmallVector<RankedTensorType> types;
  for (Value value : product->getOperands())
    types.push_back(cast<RankedTensorType>(value.getType()));
  auto layout = product.getScaleLayoutAttr();
  auto policy = product.getNumericPolicyAttr();
  auto granularity = layout.getAs<StringAttr>("granularity");
  auto format = layout.getAs<StringAttr>("format");
  auto block = layout.getAs<ArrayAttr>("block");
  auto mode = policy.getAs<StringAttr>("execution_mode");
  auto accum = policy.getAs<StringAttr>("accum");
  if (!output || !output.hasStaticShape() || output.getEncoding() ||
      !output.getElementType().isF32() || output.getRank() < 2 ||
      output.getRank() > 8 || output.getNumElements() <= 0 ||
      output.getNumElements() > INT32_MAX ||
      !hasExactScaledBroadcastPrefix(types, output) ||
      !granularity || granularity.getValue() != "block" ||
      !format || format.getValue() != "fp32" || !block || block.size() != 2 ||
      !isa<IntegerAttr>(block[0]) || !isa<IntegerAttr>(block[1]) ||
      !mode || mode.getValue() != "exact_per_block" ||
      !accum || accum.getValue() != "fp32")
    return product.emitError("continuous scaled product requires static exact fp32 block semantics");
  int64_t sn = cast<IntegerAttr>(block[0]).getInt();
  int64_t sk = cast<IntegerAttr>(block[1]).getInt();
  int64_t prefix = output.getRank() - 2;
  int64_t m = output.getDimSize(prefix), n = output.getDimSize(prefix+1);
  bool ta = product.getTransposeA(), tb = product.getTransposeB();
  int64_t k = types[0].getDimSize(types[0].getRank()-(ta ? 2 : 1));
  if (sn <= 0 || sk <= 0 || k <= 0 ||
      types[0].getShape().take_back(2) != (ta ? ArrayRef<int64_t>({k,m}) : ArrayRef<int64_t>({m,k})) ||
      types[1].getShape().take_back(2) != (tb ? ArrayRef<int64_t>({n,k}) : ArrayRef<int64_t>({k,n})) ||
      types[2].getShape().take_back(2) != ArrayRef<int64_t>({m,k/sk+(k%sk!=0)}) ||
      types[3].getShape().take_back(2) != ArrayRef<int64_t>({k/sk+(k%sk!=0),n/sn+(n%sn!=0)}))
    return product.emitError("continuous scaled product suffixes differ from declared block semantics");
  OpBuilder builder(product);
  auto generated = tensor::GenerateOp::create(builder, product.getLoc(), output, ValueRange{},
      [&](OpBuilder &g, Location loc, ValueRange indices) {
    auto ci = [&](int64_t value) -> Value {
      return arith::ConstantIndexOp::create(g, loc, value);
    };
    Value c0=ci(0), c1=ci(1), groups=ci(k/sk+(k%sk!=0));
    Value zero=arith::ConstantOp::create(g, loc, g.getF32FloatAttr(0));
    Value row=indices[prefix], column=indices[prefix+1];
    auto coordinates = [&](unsigned slot, Value x, Value y) {
      SmallVector<Value> result;
      int64_t own=types[slot].getRank()-2;
      for (int64_t axis=0;axis<own;++axis)
        result.push_back(types[slot].getDimSize(axis)==1 ? c0 : indices[prefix-own+axis]);
      result.push_back(x); result.push_back(y);
      return result;
    };
    auto outer=scf::ForOp::create(g, loc, c0, groups, c1, ValueRange{zero},
        [&](OpBuilder &body, Location at, Value group, ValueRange carried) {
      Value lo=arith::MulIOp::create(body, at, group, arith::ConstantIndexOp::create(body, at, sk));
      Value remaining=arith::SubIOp::create(body, at, arith::ConstantIndexOp::create(body, at, k), lo);
      Value end=arith::AddIOp::create(body, at, lo,
          arith::MinUIOp::create(body, at, arith::ConstantIndexOp::create(body, at, sk), remaining));
      auto dot=scf::ForOp::create(body, at, lo, end, c1, ValueRange{zero},
          [&](OpBuilder &inner, Location il, Value contraction, ValueRange partial) {
        Value a=tensor::ExtractOp::create(inner, il, product.getLhs(),
            coordinates(0,ta ? contraction : row,ta ? row : contraction));
        Value b=tensor::ExtractOp::create(inner, il, product.getRhs(),
            coordinates(1,tb ? column : contraction,tb ? contraction : column));
        Value term=arith::MulFOp::create(inner, il, a, b);
        scf::YieldOp::create(inner, il,
            arith::AddFOp::create(inner, il, partial[0],term).getResult());
      });
      Value columnGroup=arith::DivUIOp::create(body, at, column,arith::ConstantIndexOp::create(body, at, sn));
      Value sa=tensor::ExtractOp::create(body, at, product.getLhsScale(),coordinates(2,row,group));
      Value sb=tensor::ExtractOp::create(body, at, product.getRhsScale(),coordinates(3,group,columnGroup));
      Value scale=arith::MulFOp::create(body, at,sa,sb);
      Value contribution=arith::MulFOp::create(body, at,dot.getResult(0),scale);
      scf::YieldOp::create(body, at,
          arith::AddFOp::create(body, at,carried[0],contribution).getResult());
    });
    tensor::YieldOp::create(g, loc,outer.getResult(0));
  });
  generated->setAttr("tessera.native.scaled_product",builder.getUnitAttr());
  generated->setAttr("tessera.native.scaled_product_semantics",builder.getDictionaryAttr(product->getAttrs()));
  product.getResult().replaceAllUsesWith(generated.getResult());
  product.erase();
  return success();
}
} // namespace tessera
