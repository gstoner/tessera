// Native physical leaf for ROCM-NVFP4-INGEST-1. One thread owns one K32
// destination block. Six flat memrefs: source bytes, source E4M3 scale bytes,
// projection globals (f64), destination bytes, destination E8M0 scales,
// [signal,error] f64 pairs. Source and destination cannot alias.
static mlir::LogicalResult emitNativeNVFP4Ingest(mlir::ModuleOp module,
                                                mlir::Operation *op) {
  using namespace mlir;
  auto arch = module->getAttrOfType<StringAttr>("tessera.arch");
  if (!arch || arch.getValue() != "gfx1201") {
    op->emitError("native NVFP4 ingest requires an exact gfx1201 module");
    return failure();
  }
  int64_t n = op->getAttrOfType<IntegerAttr>("n").getInt();
  int64_t k = op->getAttrOfType<IntegerAttr>("k").getInt();
  int64_t groups = k / 32;
  auto offsets = op->getAttrOfType<ArrayAttr>("row_offsets");
  std::string name = op->getAttrOfType<StringAttr>("name").getValue().str();
  OpBuilder b(module.getBodyRegion());
  b.setInsertionPointToEnd(module.getBody());
  Location l = op->getLoc();
  auto gm = b.create<gpu::GPUModuleOp>(l, name + "_mod");
  b.setInsertionPointToStart(&gm.getBodyRegion().front());
  auto bytes = MemRefType::get({ShapedType::kDynamic}, b.getI8Type());
  auto doubles = MemRefType::get({ShapedType::kDynamic}, b.getF64Type());
  auto f = b.create<gpu::GPUFuncOp>(
      l, name, b.getFunctionType({bytes, bytes, doubles, bytes, bytes, doubles}, {}));
  f.setKernelAttr(b.getUnitAttr());
  f->setAttr("gpu.known_block_size", b.getDenseI32ArrayAttr({256, 1, 1}));
  b.setInsertionPointToStart(&f.getBody().front());
  auto ci = [&](int64_t v) -> Value { return b.create<arith::ConstantIndexOp>(l, v); };
  auto c32 = [&](int64_t v) -> Value { return b.create<arith::ConstantIntOp>(l, v, 32); };
  auto c64 = [&](int64_t v) -> Value { return b.create<arith::ConstantIntOp>(l, v, 64); };
  auto cf = [&](double v) -> Value {
    return b.create<arith::ConstantOp>(l, b.getF64Type(), b.getF64FloatAttr(v));
  };
  auto add = [&](Value a, Value v) -> Value { return b.create<arith::AddFOp>(l, a, v); };
  auto sub = [&](Value a, Value v) -> Value { return b.create<arith::SubFOp>(l, a, v); };
  auto mul = [&](Value a, Value v) -> Value { return b.create<arith::MulFOp>(l, a, v); };
  auto div = [&](Value a, Value v) -> Value { return b.create<arith::DivFOp>(l, a, v); };
  auto eq = [&](Value a, Value v) -> Value {
    return b.create<arith::CmpIOp>(l, arith::CmpIPredicate::eq, a, v);
  };
  auto choose = [&](Value predicate, Value a, Value v) -> Value {
    return b.create<arith::SelectOp>(l, predicate, a, v);
  };
  auto pow2 = [&](Value exponent) -> Value {
    Value wide = b.create<arith::ExtSIOp>(l, b.getI64Type(), exponent);
    Value bits = b.create<arith::ShLIOp>(
        l, b.create<arith::AddIOp>(l, wide, c64(1023)), c64(52));
    return b.create<arith::BitcastOp>(l, b.getF64Type(), bits);
  };
  // Match the contiguous 32-element oracle's eight-accumulator reduction.
  auto sum32 = [&](ArrayRef<Value> values) -> Value {
    SmallVector<Value, 8> partial;
    for (int j = 0; j < 8; ++j) {
      Value value = values[j];
      for (int q = 8; q < 32; q += 8)
        value = add(value, values[j + q]);
      partial.push_back(value);
    }
    return add(add(add(partial[0], partial[1]), add(partial[2], partial[3])),
               add(add(partial[4], partial[5]), add(partial[6], partial[7])));
  };
  Value block = b.create<gpu::BlockIdOp>(l, gpu::Dimension::x);
  Value thread = b.create<gpu::ThreadIdOp>(l, gpu::Dimension::x);
  Value index = b.create<arith::AddIOp>(
      l, b.create<arith::MulIOp>(l, block, ci(256)), thread);
  Value active = b.create<arith::CmpIOp>(l, arith::CmpIPredicate::ult, index, ci(n * groups));
  auto guard = b.create<scf::IfOp>(l, active, false);
  b.setInsertionPointToStart(guard.thenBlock());
  Value row = b.create<arith::DivUIOp>(l, index, ci(groups));
  Value group = b.create<arith::RemUIOp>(l, index, ci(groups));
  Value projection = ci(0);
  for (unsigned p = 1; p + 1 < offsets.size(); ++p) {
    Value after = b.create<arith::CmpIOp>(
        l, arith::CmpIPredicate::uge, row, ci(cast<IntegerAttr>(offsets[p]).getInt()));
    projection = choose(after, ci(p), projection);
  }
  Value global = b.create<memref::LoadOp>(l, f.getArgument(2), ValueRange{projection});
  SmallVector<Value, 2> scales;
  for (int half = 0; half < 2; ++half) {
    Value scaleIndex = b.create<arith::AddIOp>(
        l, b.create<arith::MulIOp>(l, row, ci(k / 16)),
        b.create<arith::AddIOp>(l, b.create<arith::MulIOp>(l, group, ci(2)), ci(half)));
    Value raw = b.create<memref::LoadOp>(l, f.getArgument(1), ValueRange{scaleIndex});
    Value bits = b.create<arith::ExtUIOp>(l, b.getI32Type(), raw);
    Value exponent = b.create<arith::AndIOp>(
        l, b.create<arith::ShRUIOp>(l, bits, c32(3)), c32(15));
    Value fraction = b.create<arith::AndIOp>(l, bits, c32(7));
    Value normal = mul(b.create<arith::UIToFPOp>(
        l, b.getF64Type(), b.create<arith::AddIOp>(l, fraction, c32(8))),
        pow2(b.create<arith::SubIOp>(l, exponent, c32(10))));
    Value subnormal = mul(b.create<arith::UIToFPOp>(l, b.getF64Type(), fraction), cf(0x1p-9));
    scales.push_back(mul(choose(eq(exponent, c32(0)), subnormal, normal), global));
  }
  constexpr double levels[8] = {0., .5, 1., 1.5, 2., 3., 4., 6.};
  constexpr double midpoints[7] = {.25, .75, 1.25, 1.75, 2.5, 3.5, 5.};
  auto magnitudeValue = [&](Value code) -> Value {
    Value value = cf(0);
    for (int j = 1; j < 8; ++j)
      value = choose(eq(code, c32(j)), cf(levels[j]), value);
    return value;
  };
  SmallVector<Value, 16> sourceBytes;
  Value packedBase = b.create<arith::AddIOp>(
      l, b.create<arith::MulIOp>(l, row, ci(k / 2)),
      b.create<arith::MulIOp>(l, group, ci(16)));
  for (int j = 0; j < 16; ++j) {
    Value address = b.create<arith::AddIOp>(l, packedBase, ci(j));
    sourceBytes.push_back(b.create<arith::ExtUIOp>(
        l, b.getI32Type(), b.create<memref::LoadOp>(l, f.getArgument(0), ValueRange{address})));
  }
  SmallVector<Value, 32> values, weights, weightedScale, signal;
  for (int j = 0; j < 32; ++j) {
    Value code = b.create<arith::AndIOp>(
        l, b.create<arith::ShRUIOp>(l, sourceBytes[j / 2], c32((j % 2) * 4)), c32(15));
    Value magnitude = magnitudeValue(b.create<arith::AndIOp>(l, code, c32(7)));
    Value negative = b.create<arith::CmpIOp>(
        l, arith::CmpIPredicate::ne, b.create<arith::AndIOp>(l, code, c32(8)), c32(0));
    Value value = mul(choose(negative, b.create<arith::NegFOp>(l, magnitude), magnitude), scales[j / 16]);
    Value energy = mul(magnitude, magnitude);
    values.push_back(value);
    weights.push_back(energy);
    weightedScale.push_back(mul(energy, scales[j / 16]));
    signal.push_back(mul(value, value));
  }
  Value total = sum32(weights);
  Value maxScale = b.create<arith::MaximumFOp>(l, scales[0], scales[1]);
  Value mean = choose(b.create<arith::CmpFOp>(l, arith::CmpFPredicate::OGT, total, cf(0)),
                      div(sum32(weightedScale), total), maxScale);
  Value meanBits = b.create<arith::BitcastOp>(l, b.getI64Type(), mean);
  Value floor = b.create<arith::TruncIOp>(
      l, b.getI32Type(), b.create<arith::SubIOp>(
          l, b.create<arith::AndIOp>(
              l, b.create<arith::ShRUIOp>(l, meanBits, c64(52)), c64(2047)), c64(1023)));
  auto clamp = [&](Value exponent) -> Value {
    return b.create<arith::MinSIOp>(
        l, b.create<arith::MaxSIOp>(l, exponent, c32(-126)), c32(127));
  };
  Value lower = clamp(floor), upper = clamp(b.create<arith::AddIOp>(l, floor, c32(1)));
  auto seedError = [&](Value exponent) -> Value {
    SmallVector<Value, 32> terms;
    Value candidate = pow2(exponent);
    for (int j = 0; j < 32; ++j) {
      Value difference = sub(scales[j / 16], candidate);
      terms.push_back(mul(weights[j], mul(difference, difference)));
    }
    return sum32(terms);
  };
  Value seed = choose(b.create<arith::CmpFOp>(
      l, arith::CmpFPredicate::OLT, seedError(upper), seedError(lower)), upper, lower);
  seed = choose(b.create<arith::CmpFOp>(l, arith::CmpFPredicate::OGT, mean, cf(0)), seed, c32(0));
  Value bestError = cf(std::numeric_limits<double>::infinity());
  Value bestDistance = c32(99), bestExponent = seed;
  for (int delta = -4; delta <= 4; ++delta) {
    Value exponent = b.create<arith::AddIOp>(l, seed, c32(delta));
    Value valid = b.create<arith::AndIOp>(
        l, b.create<arith::CmpIOp>(l, arith::CmpIPredicate::sge, exponent, c32(-126)),
        b.create<arith::CmpIOp>(l, arith::CmpIPredicate::sle, exponent, c32(127)));
    Value boundedExponent = clamp(exponent);
    Value scale = pow2(boundedExponent);
    // Compare in source units. Each midpoint times scale is an exact normal
    // f64 dyadic number for exponent [-126,127]. Near a midpoint, scaling a
    // normal input by the reciprocal power of two is exact; underflow/overflow
    // occurs only outside all midpoint decisions. Ordered strict comparisons
    // therefore preserve ties without a normalization multiply per value.
    SmallVector<Value, 7> thresholds;
    for (double midpoint : midpoints)
      thresholds.push_back(mul(cf(midpoint), scale));
    SmallVector<Value, 32> errors;
    for (int j = 0; j < 32; ++j) {
      Value absolute = b.create<math::AbsFOp>(l, values[j]);
      Value magnitude = c32(0);
      for (Value threshold : thresholds)
        magnitude = b.create<arith::AddIOp>(
            l, magnitude, b.create<arith::ExtUIOp>(
                l, b.getI32Type(), b.create<arith::CmpFOp>(
                    l, arith::CmpFPredicate::OGT, absolute, threshold)));
      Value negative = b.create<arith::CmpFOp>(l, arith::CmpFPredicate::OLT, values[j], cf(0));
      Value decoded = mul(choose(negative, b.create<arith::NegFOp>(
          l, magnitudeValue(magnitude)), magnitudeValue(magnitude)), scale);
      Value difference = sub(values[j], decoded);
      errors.push_back(mul(difference, difference));
    }
    Value error = sum32(errors);
    Value equal = b.create<arith::CmpFOp>(l, arith::CmpFPredicate::OEQ, error, bestError);
    Value tie = b.create<arith::OrIOp>(
        l, b.create<arith::CmpIOp>(l, arith::CmpIPredicate::slt, c32(std::abs(delta)), bestDistance),
        b.create<arith::AndIOp>(
            l, eq(c32(std::abs(delta)), bestDistance),
            b.create<arith::CmpIOp>(l, arith::CmpIPredicate::slt, exponent, bestExponent)));
    Value better = b.create<arith::AndIOp>(
        l, valid, b.create<arith::OrIOp>(
            l, b.create<arith::CmpFOp>(l, arith::CmpFPredicate::OLT, error, bestError),
            b.create<arith::AndIOp>(l, equal, tie)));
    bestError = choose(better, error, bestError);
    bestDistance = choose(better, c32(std::abs(delta)), bestDistance);
    bestExponent = choose(better, exponent, bestExponent);
  }
  // Candidate codes are used only to score their exponent. Carrying all 32
  // through each best-candidate select extends their live ranges across the
  // nine-way search. Encode once from the selected exponent instead. Reuse
  // exactly the ordered strict midpoint decisions above, including ties,
  // signed zero and the clamped exponent envelope.
  Value winningScale = pow2(bestExponent);
  SmallVector<Value, 7> winningThresholds;
  for (double midpoint : midpoints)
    winningThresholds.push_back(mul(cf(midpoint), winningScale));
  SmallVector<Value, 32> bestCodes;
  for (int j = 0; j < 32; ++j) {
    Value absolute = b.create<math::AbsFOp>(l, values[j]);
    Value magnitude = c32(0);
    for (Value threshold : winningThresholds)
      magnitude = b.create<arith::AddIOp>(
          l, magnitude, b.create<arith::ExtUIOp>(
              l, b.getI32Type(), b.create<arith::CmpFOp>(
                  l, arith::CmpFPredicate::OGT, absolute, threshold)));
    Value negative = b.create<arith::CmpFOp>(
        l, arith::CmpFPredicate::OLT, values[j], cf(0));
    bestCodes.push_back(b.create<arith::OrIOp>(
        l, magnitude, choose(negative, c32(8), c32(0))));
  }
  Value signalSum = sum32(signal);
  Value nonzero = b.create<arith::CmpFOp>(l, arith::CmpFPredicate::OGT, signalSum, cf(0));
  for (int j = 0; j < 16; ++j) {
    Value code = b.create<arith::OrIOp>(
        l, bestCodes[j * 2],
        b.create<arith::ShLIOp>(l, bestCodes[j * 2 + 1], c32(4)));
    code = choose(nonzero, code, c32(0));
    b.create<memref::StoreOp>(l, b.create<arith::TruncIOp>(l, b.getI8Type(), code),
        f.getArgument(3), ValueRange{b.create<arith::AddIOp>(l, packedBase, ci(j))});
  }
  Value anyScale = b.create<arith::CmpFOp>(l, arith::CmpFPredicate::OGT, maxScale, cf(0));
  Value encoded = choose(anyScale, b.create<arith::AddIOp>(
      l, choose(nonzero, bestExponent, c32(0)), c32(127)), c32(0));
  Value scaleIndex = b.create<arith::AddIOp>(
      l, b.create<arith::MulIOp>(l, group, ci(n)), row);
  b.create<memref::StoreOp>(l, b.create<arith::TruncIOp>(l, b.getI8Type(), encoded),
                          f.getArgument(4), ValueRange{scaleIndex});
  Value statIndex = b.create<arith::MulIOp>(l, index, ci(2));
  b.create<memref::StoreOp>(l, signalSum, f.getArgument(5), ValueRange{statIndex});
  b.create<memref::StoreOp>(l, choose(nonzero, bestError, cf(0)), f.getArgument(5),
                          ValueRange{b.create<arith::AddIOp>(l, statIndex, ci(1))});
  b.setInsertionPointToEnd(&f.getBody().front());
  b.create<gpu::ReturnOp>(l);
  return success();
}
