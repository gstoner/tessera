// Native Schedule/Tile projection of actual scalar scale-adjoint regions.

static std::string nativeScaleTransposeDigest(tensor::GenerateOp generator) {
  Operation *copy = generator->clone();
  copy->removeAttr("schedule.artifact_hash");
  // Schedule algorithm and width remain in the content-addressed identity.
  auto digest = structuredReductionBodyHash(copy);
  copy->destroy();
  return digest;
}

static FailureOr<tensor::GenerateOp> nativeScaleTransposeRoot(ModuleOp mod) {
  auto member = mod->getAttrOfType<DictionaryAttr>("tessera.autodiff.scaled_member");
  auto kind = member ? member.getAs<StringAttr>("kind") : StringAttr{};
  if (!kind || kind.getValue() != "scale_vjp")
    return tensor::GenerateOp{};
  auto operation = member.getAs<StringAttr>("operation");
  if (operation && operation.getValue() == "tessera.transpose") {
    // An inverse output seed is a movement member, not a scale reduction.
    // Keep the isolated original Graph for the result-permutation consumer.
    auto functions = llvm::to_vector(mod.getOps<func::FuncOp>());
    if (functions.size() != 1 || !functions[0].getBody().hasOneBlock() ||
        functions[0].getNumArguments() != 1 || functions[0].getNumResults() != 1 ||
        !isa<tessera::TransposeOp>(functions[0].getBody().front().front()) ||
        !isa<func::ReturnOp>(functions[0].getBody().front().back()))
      return mod.emitError("native inverse cotangent lost its isolated Graph member"), failure();
    for (Operation &op : functions[0].getBody().front())
      if (&op != &functions[0].getBody().front().front() &&
          !isa<func::ReturnOp>(op) && op.getName().getStringRef() != "schedule.artifact")
        return mod.emitError("native inverse cotangent has an unexpected Schedule member"), failure();
    return tensor::GenerateOp{};
  }
  if (operation && operation.getValue() == "tessera.add") {
    // Accumulated scale contributions use the ordinary native sum lane.
    // This reduction-specific projection must not claim that sum member.
    auto functions = llvm::to_vector(mod.getOps<func::FuncOp>());
    if (functions.size() != 1 || !functions[0].getBody().hasOneBlock() ||
        functions[0].getNumArguments() != 2 || functions[0].getNumResults() != 1 ||
        !isa<tessera::AddOp>(functions[0].getBody().front().front()) ||
        !isa<func::ReturnOp>(functions[0].getBody().front().back()))
      return mod.emitError("native scale sum lost its isolated Graph member"), failure();
    for (Operation &op : functions[0].getBody().front())
      if (&op != &functions[0].getBody().front().front() &&
          !isa<func::ReturnOp>(op) && op.getName().getStringRef() != "schedule.artifact")
        return mod.emitError("native scale sum has an unexpected Schedule member"), failure();
    return tensor::GenerateOp{};
  }
  auto target = mod->getAttrOfType<StringAttr>("tessera.target");
  auto arch = mod->getAttrOfType<StringAttr>("tessera.arch");
  auto funcs = llvm::to_vector(mod.getOps<func::FuncOp>());
  if (!target || target.getValue() != "rocm" || !arch || arch.getValue() != "gfx1201" ||
      funcs.size() != 1 || !funcs[0].getBody().hasOneBlock() ||
      funcs[0].getNumArguments() != 4 || funcs[0].getNumResults() != 1 ||
      funcs[0].getBody().front().getOperations().size() != 2)
    return mod.emitError("native scale transpose needs one isolated gfx1201 reduction member"), failure();
  auto generator = dyn_cast<tensor::GenerateOp>(funcs[0].getBody().front().front());
  auto role = generator ? generator->getAttrOfType<StringAttr>("tessera.autodiff.scale_adjoint")
                        : StringAttr{};
  auto type = generator ? dyn_cast<RankedTensorType>(generator.getType()) : RankedTensorType{};
  if (!generator || !role || (role.getValue() != "lhs_scale" && role.getValue() != "rhs_scale") ||
      !type || !type.hasStaticShape() || !type.getElementType().isF32() ||
      type.getNumElements() <= 0 || type.getNumElements() > INT32_MAX)
    return mod.emitError("native scale transpose lost its static f32 generated reduction"), failure();
  unsigned bytes = 0, floats = 0;
  for (Type input : funcs[0].getArgumentTypes()) {
    auto tensor = dyn_cast<RankedTensorType>(input);
    if (!tensor || !tensor.hasStaticShape() || tensor.getEncoding())
      return mod.emitError("native scale transpose requires contiguous static tensor storage"), failure();
    bytes += isa<Float8E4M3FNType>(tensor.getElementType());
    floats += tensor.getElementType().isF32();
  }
  if (bytes != 2 || floats != 2)
    return mod.emitError("native scale transpose requires two E4M3 and two f32 captured inputs"), failure();
  bool valid = true;
  generator.getBody().walk([&](Operation *op) {
    auto name = op->getName().getStringRef();
    valid &= name.starts_with("arith.") || name == "scf.for" || name == "scf.yield" ||
             name == "tensor.extract" || name == "tensor.yield";
    if (auto extract = dyn_cast<tensor::ExtractOp>(op)) {
      auto input = dyn_cast<BlockArgument>(extract.getTensor());
      valid &= input && input.getOwner() == &funcs[0].getBody().front();
      if (isa<Float8E4M3FNType>(extract.getType()))
        for (auto *user : extract.getResult().getUsers())
          valid &= isa<arith::ExtFOp>(user) && user->getResult(0).getType().isF32();
    }
  });
  if (!valid)
    return mod.emitError("native scale transpose body has unsupported captures or scalar operations"), failure();
  return generator;
}

static LogicalResult scheduleNativeScaleTranspose(ModuleOp mod, bool &selected, bool wave) {
  auto root = nativeScaleTransposeRoot(mod);
  if (failed(root)) return failure();
  auto generator = *root;
  selected = bool(generator);
  if (!selected) return success();
  if (generator->hasAttr("schedule.artifact_hash"))
    return generator.emitError("native scale transpose is already scheduled");
  OpBuilder b(mod.getContext());
  int64_t width = wave ? 32 : 128;
  StringRef algorithm = wave ? "wave_per_scale_element" : "serial_per_scale_element";
  generator->setAttr("schedule.workgroup_size", b.getI64IntegerAttr(width));
  generator->setAttr("schedule.algorithm", b.getStringAttr(algorithm));
  generator->setAttr("schedule.outer_accumulation", b.getStringAttr("compensated_fp32"));
  std::string digest = nativeScaleTransposeDigest(generator);
  generator->setAttr("schedule.artifact_hash", b.getStringAttr(digest));
  b.setInsertionPointAfter(generator->getParentOfType<func::FuncOp>());
  OperationState state(generator.getLoc(), "schedule.artifact");
  state.addAttribute("hash", b.getStringAttr(digest));
  state.addAttribute("arch", b.getStringAttr("gfx1201"));
  state.addAttribute("shape_key", b.getStringAttr("family=scale_adjoint;count=" +
      std::to_string(cast<RankedTensorType>(generator.getType()).getNumElements())));
  state.addAttribute("tile", b.getDictionaryAttr({
      b.getNamedAttr("workgroup_size", b.getI64IntegerAttr(width)),
      b.getNamedAttr("algorithm", b.getStringAttr(algorithm)),
      b.getNamedAttr("outer_accumulation", b.getStringAttr("compensated_fp32"))}));
  state.addAttribute("numeric_policy", b.getStringAttr("E4M3FN coefficients;fp32 scale adjoint;exact_per_block"));
  b.create(state);
  return success();
}

// Portable exact OCP E4M3FN scalar conversion. IEEE signed zero and the two
// reserved NaN codes survive; no FN/FNUZ reinterpretation or host decode.
static Value decodeNativeE4M3FN(OpBuilder &b, Location loc, Value byte) {
  auto ci = [&](int64_t v) -> Value { return arith::ConstantIntOp::create(b, loc, v, 32); };
  Value bits = arith::ExtUIOp::create(b, loc, b.getI32Type(), byte);
  Value sign = arith::ShLIOp::create(b, loc,
      arith::AndIOp::create(b, loc, bits, ci(128)), ci(24));
  Value mantissa = arith::AndIOp::create(b, loc, bits, ci(7));
  Value exponent = arith::AndIOp::create(b, loc,
      arith::ShRUIOp::create(b, loc, bits, ci(3)), ci(15));
  Value normalBits = arith::OrIOp::create(b, loc, sign,
      arith::OrIOp::create(b, loc,
          arith::ShLIOp::create(b, loc, arith::AddIOp::create(b, loc, exponent, ci(120)), ci(23)),
          arith::ShLIOp::create(b, loc, mantissa, ci(20))));
  Value normal = arith::BitcastOp::create(b, loc, b.getF32Type(), normalBits);
  Value subnormal = arith::MulFOp::create(b, loc,
      arith::UIToFPOp::create(b, loc, b.getF32Type(), mantissa),
      arith::ConstantOp::create(b, loc, b.getF32FloatAttr(0.001953125)));
  Value negative = arith::CmpIOp::create(b, loc, arith::CmpIPredicate::ne, sign, ci(0));
  subnormal = arith::SelectOp::create(b, loc, negative,
      arith::NegFOp::create(b, loc, subnormal), subnormal);
  Value zeroExponent = arith::CmpIOp::create(b, loc, arith::CmpIPredicate::eq, exponent, ci(0));
  Value value = arith::SelectOp::create(b, loc, zeroExponent, subnormal, normal);
  Value isNan = arith::CmpIOp::create(b, loc, arith::CmpIPredicate::eq,
      arith::AndIOp::create(b, loc, bits, ci(127)), ci(127));
  Value nan = arith::BitcastOp::create(b, loc, b.getF32Type(),
      arith::OrIOp::create(b, loc, sign, ci(0x7fc00000)));
  return arith::SelectOp::create(b, loc, isNan, nan, value);
}

// Prove that the carried value contributes exactly once, including nested
// serial reductions initialized with the outer accumulator. Contributions
// cannot depend on that accumulator through a second captured use.
static bool additiveScaleReduction(scf::ForOp loop) {
  if (!loop || loop.getNumResults() != 1 || loop.getInitArgs().size() != 1 ||
      !loop.getResult(0).getType().isF32())
    return false;
  Value accumulator = loop.getRegionIterArgs()[0];
  if (!accumulator.hasOneUse()) return false;
  auto yield = dyn_cast<scf::YieldOp>(loop.getBody()->getTerminator());
  if (!yield || yield.getNumOperands() != 1) return false;
  Value value = yield.getOperand(0);
  if (auto join = value.getDefiningOp<arith::AddFOp>())
    return join.getLhs() == accumulator || join.getRhs() == accumulator;
  auto nested = value.getDefiningOp<scf::ForOp>();
  return nested && nested.getInitArgs().size() == 1 &&
         nested.getInitArgs()[0] == accumulator && additiveScaleReduction(nested);
}

// Rebuild only the proved additive accumulator chain with a companion FP32
// rounding residual. Dot-product loops are cloned unchanged. Nested reductions
// retain the residual across their parent M/batch iterations.
static scf::ForOp compensatedScaleReduction(OpBuilder &b, scf::ForOp loop,
                                            Value residual, IRMapping &outer) {
  auto loc = loop.getLoc();
  Value initial = outer.lookupOrDefault(loop.getInitArgs()[0]);
  auto replacement = scf::ForOp::create(b, loc,
      outer.lookupOrDefault(loop.getLowerBound()),
      outer.lookupOrDefault(loop.getUpperBound()),
      outer.lookupOrDefault(loop.getStep()), ValueRange{initial, residual});
  OpBuilder::InsertionGuard guard(b);
  b.setInsertionPointToStart(replacement.getBody());
  IRMapping mapping = outer;
  mapping.map(loop.getInductionVar(), replacement.getInductionVar());
  mapping.map(loop.getRegionIterArgs()[0], replacement.getRegionIterArgs()[0]);
  Value correction = replacement.getRegionIterArgs()[1];
  auto yield = cast<scf::YieldOp>(loop.getBody()->getTerminator());
  Operation *last = yield.getOperand(0).getDefiningOp();
  Value sum;
  for (Operation &op : loop.getBody()->without_terminator()) {
    if (&op == last) {
      if (auto nested = dyn_cast<scf::ForOp>(&op)) {
        auto next = compensatedScaleReduction(b, nested, correction, mapping);
        sum = next.getResult(0);
        correction = next.getResult(1);
      } else {
        auto join = cast<arith::AddFOp>(&op);
        Value term = mapping.lookupOrDefault(
            join.getLhs() == loop.getRegionIterArgs()[0] ? join.getRhs() : join.getLhs());
        Value adjusted = arith::SubFOp::create(b, loc, term, correction);
        Value current = replacement.getRegionIterArgs()[0];
        sum = arith::AddFOp::create(b, loc, current, adjusted);
        correction = arith::SubFOp::create(b, loc,
            arith::SubFOp::create(b, loc, sum, current), adjusted);
      }
      mapping.map(op.getResult(0), sum);
    } else {
      b.clone(op, mapping);
    }
  }
  scf::YieldOp::create(b, loc, ValueRange{sum, correction});
  return replacement;
}

static LogicalResult lowerNativeScaleTranspose(ModuleOp mod, bool &selected) {
  auto root = nativeScaleTransposeRoot(mod);
  if (failed(root)) return failure();
  auto generator = *root;
  selected = bool(generator);
  if (!selected) return success();
  auto func = generator->getParentOfType<func::FuncOp>();
  std::string digest = nativeScaleTransposeDigest(generator);
  auto hash = generator->getAttrOfType<StringAttr>("schedule.artifact_hash");
  auto threads = generator->getAttrOfType<IntegerAttr>("schedule.workgroup_size");
  SmallVector<Operation *> artifacts;
  mod.walk([&](Operation *op) {
    if (op->getName().getStringRef() == "schedule.artifact") artifacts.push_back(op);
  });
  if (!hash || hash.getValue() != digest || !threads || (threads.getInt() != 128 && threads.getInt() != 32) ||
      artifacts.size() != 1 ||
      artifacts[0]->getAttrOfType<StringAttr>("hash") != hash ||
      artifacts[0]->getAttrOfType<StringAttr>("arch").getValue() != "gfx1201")
    return generator.emitError("native scale transpose Schedule differs from its Graph reduction");
  auto knobs = artifacts[0]->getAttrOfType<DictionaryAttr>("tile");
  auto width = knobs ? knobs.getAs<IntegerAttr>("workgroup_size") : IntegerAttr{};
  auto algorithm = knobs ? knobs.getAs<StringAttr>("algorithm") : StringAttr{};
  auto accumulation = knobs ? knobs.getAs<StringAttr>("outer_accumulation") : StringAttr{};
  bool wave = algorithm && algorithm.getValue() == "wave_per_scale_element";
  if (!accumulation || accumulation.getValue() != "compensated_fp32" ||
      generator->getAttrOfType<StringAttr>("schedule.outer_accumulation") != accumulation ||
      !width || width.getInt() != threads.getInt() || !algorithm ||
      generator->getAttrOfType<StringAttr>("schedule.algorithm") != algorithm ||
      (!wave && algorithm.getValue() != "serial_per_scale_element") ||
      width.getInt() != (wave ? 32 : 128))
    return generator.emitError("native scale transpose physical Schedule knobs differ");

  auto output = cast<RankedTensorType>(generator.getType());
  int64_t count = output.getNumElements();
  OpBuilder b(mod.getContext());
  b.setInsertionPointToEnd(mod.getBody());
  std::string name = "tessera_scale_transpose_" + digest.substr(0, 24);
  OperationState state(generator.getLoc(), "tile.structured_reduction_kernel");
  state.addAttribute("name", b.getStringAttr(name));
  state.addAttribute("arch", b.getStringAttr("gfx1201"));
  state.addAttribute("count", b.getI64IntegerAttr(count));
  state.addAttribute("input_count", b.getI64IntegerAttr(4));
  state.addAttribute("workgroup_size", width);
  state.addAttribute("algorithm", algorithm);
  state.addAttribute("artifact_hash", hash);
  state.addRegion();
  Operation *carrier = b.create(state);
  carrier->getRegion(0).push_back(new Block);
  b.setInsertionPointToStart(&carrier->getRegion(0).front());
  auto gpuModule = gpu::GPUModuleOp::create(b, generator.getLoc(), name + "_gpu");
  b.setInsertionPointToStart(&gpuModule.getBodyRegion().front());
  SmallVector<Type> inputs;
  for (Type type : func.getArgumentTypes()) {
    auto element = cast<RankedTensorType>(type).getElementType();
    inputs.push_back(MemRefType::get({ShapedType::kDynamic},
        isa<Float8E4M3FNType>(element) ? b.getI8Type() : element));
  }
  inputs.push_back(MemRefType::get({ShapedType::kDynamic}, b.getF32Type()));
  inputs.push_back(b.getI64Type());
  auto kernel = gpu::GPUFuncOp::create(b, generator.getLoc(), name, b.getFunctionType(inputs, {}));
  kernel.setKernel(true); // Canonical inherent property survives GPU-to-LLVM lowering.
  b.setInsertionPointToStart(&kernel.getBody().front());
  auto loc = generator.getLoc();
  Value c128 = arith::ConstantIndexOp::create(b, loc, 128);
  Value thread = gpu::ThreadIdOp::create(b, loc, gpu::Dimension::x);
  Value block = gpu::BlockIdOp::create(b, loc, gpu::Dimension::x);
  Value linear = wave ? block : arith::AddIOp::create(b, loc,
      arith::MulIOp::create(b, loc, block, c128), thread).getResult();
  Value extent = arith::IndexCastOp::create(b, loc, b.getIndexType(), kernel.getArgument(5));
  auto inBounds = arith::CmpIOp::create(b, loc, arith::CmpIPredicate::ult, linear, extent);
  auto branch = scf::IfOp::create(b, loc, inBounds, false);
  b.setInsertionPointToStart(branch.thenBlock());
  IRMapping mapping;
  for (auto [arg, buffer] : llvm::zip(func.getArguments(), kernel.getArguments()))
    mapping.map(arg, buffer);
  Value remainder = linear;
  for (int64_t i = output.getRank()-1; i >= 0; --i) {
    Value dim = arith::ConstantIndexOp::create(b, loc, output.getDimSize(i));
    mapping.map(generator.getBody().front().getArgument(i),
                arith::RemUIOp::create(b, loc, remainder, dim));
    remainder = arith::DivUIOp::create(b, loc, remainder, dim);
  }
  auto &body = generator.getBody().front();
  for (Operation &op : body.without_terminator()) b.clone(op, mapping);
  auto yield = cast<tensor::YieldOp>(body.getTerminator());
  Value value = mapping.lookup(yield.getValue());
  auto reduction = value.getDefiningOp<scf::ForOp>();
  auto zero = reduction && reduction.getNumResults() == 1
      ? reduction.getInitArgs()[0].getDefiningOp<arith::ConstantOp>()
      : arith::ConstantOp{};
  auto initial = zero ? dyn_cast<FloatAttr>(zero.getValue()) : FloatAttr{};
  if (!reduction || !initial || !initial.getValue().isZero() ||
      !additiveScaleReduction(reduction))
    return generator.emitError("scale transpose requires an additive zero-initialized reduction");
  if (wave) {
    // Partition the innermost additive contribution loop, not the K-group dot.
    auto partition = reduction;
    while (auto nested = cast<scf::YieldOp>(partition.getBody()->getTerminator())
                             .getOperand(0).getDefiningOp<scf::ForOp>())
      partition = nested;
    b.setInsertionPoint(partition);
    Value laneOffset = arith::MulIOp::create(b, loc, thread, partition.getStep());
    Value lower = arith::AddIOp::create(b, loc, partition.getLowerBound(), laneOffset);
    Value step = arith::MulIOp::create(b, loc, partition.getStep(),
        arith::ConstantIndexOp::create(b, loc, 32));
    partition.getLowerBoundMutable().assign(lower);
    partition.getStepMutable().assign(step);
  }
  b.setInsertionPoint(reduction);
  IRMapping compensationMapping;
  auto compensated = compensatedScaleReduction(
      b, reduction, reduction.getInitArgs()[0], compensationMapping);
  reduction.getResult(0).replaceAllUsesWith(compensated.getResult(0));
  reduction.erase();
  value = compensated.getResult(0);
  b.setInsertionPointAfter(compensated);
  if (wave) {
    Value shuffleWidth = arith::ConstantIntOp::create(b, loc, 32, 32);
    for (int64_t offset = 16; offset > 0; offset >>= 1) {
      Value off = arith::ConstantIntOp::create(b, loc, offset, 32);
      auto shuffled = gpu::ShuffleOp::create(b, loc, value, off, shuffleWidth, gpu::ShuffleMode::XOR);
      value = arith::AddFOp::create(b, loc, value, shuffled.getShuffleResult());
    }
    auto firstLane = arith::CmpIOp::create(b, loc, arith::CmpIPredicate::eq, thread,
        arith::ConstantIndexOp::create(b, loc, 0));
    auto store = scf::IfOp::create(b, loc, firstLane, false);
    b.setInsertionPointToStart(store.thenBlock());
  }
  memref::StoreOp::create(b, loc, value, kernel.getArgument(4), ValueRange{linear});
  b.setInsertionPointToEnd(&kernel.getBody().front());
  gpu::ReturnOp::create(b, loc);

  SmallVector<tensor::ExtractOp> extracts;
  kernel.walk([&](tensor::ExtractOp op) { extracts.push_back(op); });
  for (auto extract : extracts) {
    auto arg = cast<BlockArgument>(extract->getOperand(0));
    auto logical = cast<RankedTensorType>(func.getArgument(arg.getArgNumber()).getType());
    b.setInsertionPoint(extract);
    Value offset = arith::ConstantIndexOp::create(b, loc, 0);
    for (auto [dim, index] : llvm::zip(logical.getShape(), extract.getIndices()))
      offset = arith::AddIOp::create(b, loc,
          arith::MulIOp::create(b, loc, offset, arith::ConstantIndexOp::create(b, loc, dim)), index);
    Value loaded = memref::LoadOp::create(b, loc, arg, ValueRange{offset});
    if (logical.getElementType().isF32()) {
      extract.getResult().replaceAllUsesWith(loaded);
    } else {
      Value decoded = decodeNativeE4M3FN(b, loc, loaded);
      for (auto *user : llvm::make_early_inc_range(extract.getResult().getUsers())) {
        user->getResult(0).replaceAllUsesWith(decoded);
        user->erase();
      }
    }
    extract.erase();
  }
  carrier->setAttr("body_hash", b.getStringAttr(structuredReductionBodyHash(gpuModule)));
  func.erase();
  for (auto artifact : artifacts) artifact->erase();
  return verifyStructuredReductionCarrier(carrier);
}
