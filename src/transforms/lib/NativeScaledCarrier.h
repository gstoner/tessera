// Native materialization and transpose-reduction of explicit static broadcasts.
#pragma once

namespace tessera {
static bool isNativeScaledCarrier(mlir::Operation *op) {
  return mlir::isa<BroadcastOp, ReduceOp>(op);
}

static mlir::LogicalResult expandNativeScaledCarrier(mlir::Operation *op) {
  using namespace mlir;
  if (!isNativeScaledCarrier(op)) return failure();
  auto input = dyn_cast<RankedTensorType>(op->getOperand(0).getType());
  auto output = dyn_cast<RankedTensorType>(op->getResult(0).getType());
  if (!input || !output || !input.hasStaticShape() || !output.hasStaticShape() ||
      input.getEncoding() || output.getEncoding() ||
      !input.getElementType().isF32() || !output.getElementType().isF32() ||
      input.getRank() < 1 || input.getRank() > 8 ||
      output.getRank() < 1 || output.getRank() > 8 ||
      output.getNumElements() <= 0 || output.getNumElements() > INT32_MAX)
    return op->emitError("native scaled carrier requires plain static rank-1..8 f32 tensors");
  bool reduce = isa<ReduceOp>(op);
  int64_t axis = -1;
  if (reduce) {
    auto kind = op->getAttrOfType<StringAttr>("kind");
    auto attr = op->getAttrOfType<IntegerAttr>("axis");
    if (!kind || kind.getValue() != "sum" || !attr ||
        attr.getInt() < -input.getRank() || attr.getInt() >= input.getRank())
      return op->emitError("native scaled carrier requires an exact sum axis");
    axis = attr.getInt() < 0 ? attr.getInt()+input.getRank() : attr.getInt();
    SmallVector<int64_t> shape(input.getShape());
    shape.erase(shape.begin()+axis);
    if (output.getShape() != ArrayRef<int64_t>(shape))
      return op->emitError("native scaled carrier reduction shape differs");
  } else {
    int64_t offset = output.getRank()-input.getRank();
    if (offset < 0) return op->emitError("native scaled broadcast cannot lower rank");
    for (int64_t i=0; i<input.getRank(); ++i)
      if (input.getDimSize(i) != 1 && input.getDimSize(i) != output.getDimSize(i+offset))
        return op->emitError("native scaled broadcast dimensions differ");
  }
  Value source = op->getOperand(0);
  OpBuilder b(op);
  auto generator = tensor::GenerateOp::create(b, op->getLoc(), output, ValueRange{},
      [&](OpBuilder &g, Location loc, ValueRange indices) {
    Value zeroIndex = arith::ConstantIndexOp::create(g, loc, 0);
    Value one = arith::ConstantIndexOp::create(g, loc, 1);
    Value upper = arith::ConstantIndexOp::create(g, loc, reduce ? input.getDimSize(axis) : 1);
    Value zero = arith::ConstantOp::create(g, loc, g.getF32FloatAttr(0));
    if (!reduce) {
      SmallVector<Value> coordinates;
      int64_t offset=output.getRank()-input.getRank();
      for (int64_t i=0; i<input.getRank(); ++i)
        coordinates.push_back(input.getDimSize(i)==1 ? zeroIndex : indices[i+offset]);
      tensor::YieldOp::create(g, loc, tensor::ExtractOp::create(g, loc, source, coordinates));
      return;
    }
    auto loop = scf::ForOp::create(g, loc, zeroIndex, upper, one, ValueRange{zero},
        [&](OpBuilder &body, Location at, Value iv, ValueRange acc) {
      SmallVector<Value> coordinates;
      if (reduce) {
        unsigned next=0;
        for (int64_t i=0; i<input.getRank(); ++i)
          coordinates.push_back(i==axis ? iv : indices[next++]);
      } else {
        int64_t offset=output.getRank()-input.getRank();
        for (int64_t i=0; i<input.getRank(); ++i)
          coordinates.push_back(input.getDimSize(i)==1 ? zeroIndex : indices[i+offset]);
      }
      Value element=tensor::ExtractOp::create(body, at, source, coordinates);
      Value joined=arith::AddFOp::create(body, at, acc[0], element);
      scf::YieldOp::create(body, at, joined);
    });
    tensor::YieldOp::create(g, loc, loop.getResult(0));
  });
  generator->setAttr("tessera.native.scaled_carrier", b.getStringAttr(reduce ? "sum" : "broadcast"));
  op->getResult(0).replaceAllUsesWith(generator.getResult());
  op->erase();
  return success();
}
} // namespace tessera
