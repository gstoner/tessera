// Shared scalar SSD baseline. Target tiling and device admission are separate.
// Included inside namespace tessera after the common MLIR headers.
namespace {
static LogicalResult lowerNativeSSD(ModuleOp module) {
  SmallVector<schedule::SSDOp> work;
  module.walk([&](schedule::SSDOp op) { work.push_back(op); });
  for (auto op : work) {
    if (failed(op.verify())) return failure();
    OpBuilder b(op);
    Location loc = op.getLoc();
    auto xType = cast<RankedTensorType>(op.getX().getType());
    auto bType = cast<RankedTensorType>(op.getB().getType());
    auto index = [&](int64_t v) -> Value {
      return b.create<arith::ConstantIndexOp>(loc, v);
    };
    Value zero = index(0), one = index(1);
    Value time = index(xType.getDimSize(0)), heads = index(xType.getDimSize(1));
    Value width = index(xType.getDimSize(2)), states = index(bType.getDimSize(2));
    Value chunk = index(op.getChunkSize());
    Value fzero = b.create<arith::ConstantFloatOp>(loc, b.getF32Type(), APFloat(0.0f));
    auto empty = [&](Value result) -> Value {
      auto type = cast<RankedTensorType>(result.getType());
      return b.create<tensor::EmptyOp>(loc, type.getShape(), type.getElementType());
    };
    Value output = empty(op.getOutput()), checkpoints = empty(op.getCheckpoints());
    auto loop = [&](OpBuilder &builder, Value upper, ValueRange initial, auto body) {
      auto forOp = builder.create<scf::ForOp>(loc, zero, upper, one, initial,
          [&](OpBuilder &nested, Location at, Value iv, ValueRange args) {
            auto values = body(nested, iv, args);
            nested.create<scf::YieldOp>(at, values);
          });
      return SmallVector<Value>(forOp.getResults());
    };
    // Every Y element is written once, every checkpoint element at least once.
    // Rewriting a chunk's checkpoint at each step preserves the last state even
    // for a tail chunk, without allocating a full per-time residual tape.
    auto results = loop(b, time, ValueRange{op.getInitial(), output, checkpoints},
      [&](OpBuilder &tb, Value t, ValueRange ta) {
        Value ci = tb.create<arith::DivUIOp>(loc, t, chunk);
        return loop(tb, heads, ta, [&](OpBuilder &hb, Value h, ValueRange ha) {
          Value decay = hb.create<tensor::ExtractOp>(loc, op.getDecay(), ValueRange{t,h});
          return loop(hb, width, ha, [&](OpBuilder &pb, Value p, ValueRange pa) {
            Value x = pb.create<tensor::ExtractOp>(loc, op.getX(), ValueRange{t,h,p});
            auto ns = loop(pb, states, ValueRange{pa[0], pa[2], fzero},
              [&](OpBuilder &nb, Value n, ValueRange na) {
                Value old = nb.create<tensor::ExtractOp>(loc, na[0], ValueRange{h,n,p});
                Value bv = nb.create<tensor::ExtractOp>(loc, op.getB(), ValueRange{t,h,n});
                Value cv = nb.create<tensor::ExtractOp>(loc, op.getC(), ValueRange{t,h,n});
                Value carried = nb.create<arith::MulFOp>(loc, decay, old);
                Value added = nb.create<arith::MulFOp>(loc, bv, x);
                Value next = nb.create<arith::AddFOp>(loc, carried, added);
                Value state = nb.create<tensor::InsertOp>(loc, next, na[0], ValueRange{h,n,p});
                Value saved = nb.create<tensor::InsertOp>(loc, next, na[1], ValueRange{ci,h,n,p});
                Value weighted = nb.create<arith::MulFOp>(loc, cv, next);
                Value sum = nb.create<arith::AddFOp>(loc, na[2], weighted);
                return SmallVector<Value>{state, saved, sum};
              });
            Value y = pb.create<tensor::InsertOp>(loc, ns[2], pa[1], ValueRange{t,h,p});
            return SmallVector<Value>{ns[0], y, ns[1]};
          });
        });
      });
    op.getOutput().replaceAllUsesWith(results[1]);
    op.getCarry().replaceAllUsesWith(results[0]);
    op.getCheckpoints().replaceAllUsesWith(results[2]);
    op.erase();
  }
  return success();
}
} // namespace
