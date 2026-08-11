#include "Tessera/Transforms/RegionAdjointInterface.h"

#include "Tessera/Transforms/SemanticEffects.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/TypeUtilities.h"
#include "mlir/Transforms/RegionUtils.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/SetVector.h"

using namespace mlir;

namespace tessera {
namespace {

static bool isDifferentiableType(Type type) {
  return isa<FloatType>(getElementTypeOrSelf(type));
}

static Value buildZeroLike(OpBuilder &builder, Value primal) {
  Type type = primal.getType();
  Type elementType = getElementTypeOrSelf(type);
  TypedAttr zero = isa<FloatType>(elementType)
                       ? TypedAttr(FloatAttr::get(elementType, 0.0))
                       : TypedAttr(IntegerAttr::get(elementType, 0));
  if (auto shaped = dyn_cast<ShapedType>(type);
      shaped && shaped.hasStaticShape())
    return arith::ConstantOp::create(
        builder, primal.getLoc(), DenseElementsAttr::get(shaped, zero));
  if (!isa<ShapedType>(type))
    return arith::ConstantOp::create(builder, primal.getLoc(), zero);
  OperationState state(primal.getLoc(), "tessera.custom_adjoint_call");
  state.addOperands(primal);
  state.addTypes(type);
  state.addAttribute("name", builder.getStringAttr("zeros_like"));
  return builder.create(state)->getResult(0);
}

static Value addCotangents(OpBuilder &builder, Location loc, Value lhs,
                           Value rhs) {
  return arith::AddFOp::create(builder, loc, lhs, rhs);
}

static bool isReplayable(Region &region) {
  bool replayable = true;
  region.walk([&](Operation *op) {
    if (op->hasTrait<OpTrait::IsTerminator>())
      return;
    if (getRegisteredSemanticEffect(op) != SemanticEffectLevel::Pure)
      replayable = false;
  });
  return replayable;
}

static SmallVector<Value> collectDifferentiableCaptures(Region &region) {
  llvm::SetVector<Value> usedAbove;
  getUsedValuesDefinedAbove(region, region, usedAbove);
  SmallVector<Value> captures;
  for (Value value : usedAbove)
    if (isDifferentiableType(value.getType()))
      captures.push_back(value);
  return captures;
}

static SmallVector<Value> collectIfCaptures(scf::IfOp ifOp) {
  SmallVector<Value> captures;
  llvm::SmallDenseSet<Value, 8> seen;
  auto append = [&](Region &region) {
    for (Value value : collectDifferentiableCaptures(region))
      if (seen.insert(value).second)
        captures.push_back(value);
  };
  append(ifOp.getThenRegion());
  append(ifOp.getElseRegion());
  return captures;
}

static bool isCanonicalCountedFor(scf::ForOp forOp) {
  if (!forOp.getRegion().hasOneBlock())
    return false;
  llvm::APInt step;
  return matchPattern(forOp.getStep(), m_ConstantInt(&step)) &&
         step.isStrictlyPositive();
}

static LogicalResult buildIfAdjoint(
    scf::IfOp ifOp, OpBuilder &builder, ValueRange outputCotangents,
    RegionPullbackBuilder buildRegionPullback,
    SmallVectorImpl<RegionCotangent> &captureCotangents) {
  if (ifOp.getElseRegion().empty() ||
      outputCotangents.size() != ifOp.getNumResults())
    return failure();

  SmallVector<Value> captures = collectIfCaptures(ifOp);
  SmallVector<Type> resultTypes;
  for (Value capture : captures)
    resultTypes.push_back(capture.getType());

  auto backwardIf = scf::IfOp::create(builder, ifOp.getLoc(), resultTypes,
                                      ifOp.getCondition(),
                                      /*withElseRegion=*/true);
  auto buildBranch = [&](Region &source, Block *destination) -> LogicalResult {
    OpBuilder::InsertionGuard guard(builder);
    builder.setInsertionPointToStart(destination);
    SmallVector<Value> branchCotangents;
    SmallVector<Value> blockCotangents;
    if (failed(buildRegionPullback(source, outputCotangents, {}, captures,
                                   builder, blockCotangents,
                                   branchCotangents)) ||
        !blockCotangents.empty() ||
        branchCotangents.size() != captures.size())
      return failure();
    scf::YieldOp::create(builder, ifOp.getLoc(), branchCotangents);
    return success();
  };

  if (failed(buildBranch(ifOp.getThenRegion(), backwardIf.thenBlock())) ||
      failed(buildBranch(ifOp.getElseRegion(), backwardIf.elseBlock()))) {
    backwardIf.erase();
    return failure();
  }
  for (auto [capture, cotangent] :
       llvm::zip_equal(captures, backwardIf.getResults()))
    captureCotangents.push_back({capture, cotangent});
  return success();
}

static LogicalResult buildForAdjoint(
    scf::ForOp forOp, OpBuilder &builder, ValueRange outputCotangents,
    RegionPullbackBuilder buildRegionPullback,
    SmallVectorImpl<RegionCotangent> &regionCotangents) {
  Region &region = forOp.getRegion();
  if (!isCanonicalCountedFor(forOp) || !isReplayable(region) ||
      outputCotangents.size() != forOp.getNumResults())
    return failure();
  auto yield = dyn_cast<scf::YieldOp>(region.front().getTerminator());
  if (!yield || yield.getNumOperands() != forOp.getNumResults())
    return failure();

  SmallVector<Value> captures = collectDifferentiableCaptures(region);
  SmallVector<Value> initialCotangents;
  for (auto [result, cotangent] :
       llvm::zip_equal(forOp.getResults(), outputCotangents))
    initialCotangents.push_back(cotangent ? cotangent
                                          : buildZeroLike(builder, result));
  for (Value capture : captures)
    initialCotangents.push_back(buildZeroLike(builder, capture));

  Location loc = forOp.getLoc();
  Value span = arith::SubIOp::create(builder, loc, forOp.getUpperBound(),
                                     forOp.getLowerBound());
  Value tripCount = arith::CeilDivSIOp::create(
      builder, loc, span, forOp.getStep());
  Value one = arith::ConstantIndexOp::create(builder, loc, 1);
  Value lastOrdinal = arith::SubIOp::create(builder, loc, tripCount, one);
  Value lastOffset = arith::MulIOp::create(builder, loc, lastOrdinal,
                                           forOp.getStep());
  Value lastIv = arith::AddIOp::create(builder, loc, forOp.getLowerBound(),
                                       lastOffset);

  auto backward = scf::ForOp::create(
      builder, loc, forOp.getLowerBound(), forOp.getUpperBound(),
      forOp.getStep(), initialCotangents);
  {
    OpBuilder::InsertionGuard guard(builder);
    builder.setInsertionPointToStart(backward.getBody());
    Value reverseOffset = arith::SubIOp::create(
        builder, loc, backward.getInductionVar(), forOp.getLowerBound());
    Value reverseIv =
        arith::SubIOp::create(builder, loc, lastIv, reverseOffset);

    // RECOMPUTE_ALL baseline: replay [lb, reverseIv) to recover the primal
    // carried state at the selected iteration. Treeverse replaces this inner
    // replay schedule without changing RegionAdjointInterface.
    auto replay = scf::ForOp::create(
        builder, loc, forOp.getLowerBound(), reverseIv, forOp.getStep(),
        forOp.getInitArgs());
    {
      OpBuilder::InsertionGuard replayGuard(builder);
      builder.setInsertionPointToStart(replay.getBody());
      IRMapping mapping;
      mapping.map(forOp.getInductionVar(), replay.getInductionVar());
      for (auto [source, destination] : llvm::zip_equal(
               forOp.getRegionIterArgs(), replay.getRegionIterArgs()))
        mapping.map(source, destination);
      for (Operation &source : region.front().without_terminator())
        builder.clone(source, mapping);
      SmallVector<Value> replayYields;
      for (Value value : yield.getOperands())
        replayYields.push_back(mapping.lookupOrDefault(value));
      scf::YieldOp::create(builder, loc, replayYields);
    }

    SmallVector<Value> blockValues{reverseIv};
    llvm::append_range(blockValues, replay.getResults());
    ValueRange carried = backward.getRegionIterArgs();
    ValueRange stateCotangents = carried.take_front(forOp.getNumResults());
    ValueRange captureAccumulators = carried.drop_front(forOp.getNumResults());
    SmallVector<Value> blockCotangents;
    SmallVector<Value> captureStepCotangents;
    if (failed(buildRegionPullback(
            region, stateCotangents, blockValues, captures, builder,
            blockCotangents, captureStepCotangents)) ||
        blockCotangents.size() != blockValues.size() ||
        captureStepCotangents.size() != captures.size()) {
      backward.erase();
      return failure();
    }

    SmallVector<Value> nextCotangents;
    // The induction variable is integer-valued and deliberately has no
    // cotangent. Remaining block arguments are loop-carried primals.
    nextCotangents.append(blockCotangents.begin() + 1,
                          blockCotangents.end());
    for (auto [accumulator, stepCotangent] : llvm::zip_equal(
             captureAccumulators, captureStepCotangents))
      nextCotangents.push_back(
          addCotangents(builder, loc, accumulator, stepCotangent));
    scf::YieldOp::create(builder, loc, nextCotangents);
  }

  for (auto [init, cotangent] : llvm::zip_equal(
           forOp.getInitArgs(),
           backward.getResults().take_front(forOp.getNumResults())))
    if (isDifferentiableType(init.getType()))
      regionCotangents.push_back({init, cotangent});
  for (auto [capture, cotangent] : llvm::zip_equal(
           captures, backward.getResults().drop_front(forOp.getNumResults())))
    regionCotangents.push_back({capture, cotangent});
  return success();
}

static bool isCanonicalBoundedWhile(scf::WhileOp whileOp) {
  if (!whileOp.getBefore().hasOneBlock() || !whileOp.getAfter().hasOneBlock() ||
      whileOp.getInits().empty() ||
      !isa<IndexType>(whileOp.getInits().front().getType()) ||
      whileOp.getNumResults() != whileOp.getInits().size())
    return false;
  auto condition = dyn_cast<scf::ConditionOp>(
      whileOp.getBefore().front().getTerminator());
  auto yield =
      dyn_cast<scf::YieldOp>(whileOp.getAfter().front().getTerminator());
  if (!condition || !yield ||
      condition.getArgs().size() != whileOp.getInits().size() ||
      yield.getNumOperands() != whileOp.getInits().size())
    return false;

  // The reverse implementation uses result 0 as the exact number of body
  // executions. Admit only the tracer's bounded counted-while normal form:
  // the before region forwards every state unchanged and the after region
  // increments the first (index) state by exactly one. Anything more general
  // remains fail-closed until it carries an explicit iteration tape.
  Block &before = whileOp.getBefore().front();
  for (auto [forwarded, argument] :
       llvm::zip_equal(condition.getArgs(), before.getArguments()))
    if (forwarded != argument)
      return false;

  Block &after = whileOp.getAfter().front();
  auto increment = yield.getOperand(0).getDefiningOp<arith::AddIOp>();
  if (!increment)
    return false;
  llvm::APInt constant;
  return (increment.getLhs() == after.getArgument(0) &&
          matchPattern(increment.getRhs(), m_ConstantInt(&constant)) &&
          constant.isOne()) ||
         (increment.getRhs() == after.getArgument(0) &&
          matchPattern(increment.getLhs(), m_ConstantInt(&constant)) &&
          constant.isOne());
}

static LogicalResult buildWhileAdjoint(
    scf::WhileOp whileOp, OpBuilder &builder, ValueRange outputCotangents,
    RegionPullbackBuilder buildRegionPullback,
    SmallVectorImpl<RegionCotangent> &regionCotangents) {
  if (!isCanonicalBoundedWhile(whileOp) ||
      !isReplayable(whileOp.getBefore()) ||
      !isReplayable(whileOp.getAfter()) ||
      outputCotangents.size() != whileOp.getNumResults())
    return failure();

  Region &body = whileOp.getAfter();
  auto yield = dyn_cast<scf::YieldOp>(body.front().getTerminator());
  SmallVector<unsigned> differentiableState;
  for (auto [index, init] : llvm::enumerate(whileOp.getInits()))
    if (isDifferentiableType(init.getType()))
      differentiableState.push_back(index);
  if (differentiableState.empty())
    return failure();

  SmallVector<Value> captures = collectDifferentiableCaptures(body);
  SmallVector<Value> initialCotangents;
  for (unsigned index : differentiableState) {
    Value cotangent = outputCotangents[index];
    initialCotangents.push_back(
        cotangent ? cotangent
                  : buildZeroLike(builder, whileOp.getResult(index)));
  }
  for (Value capture : captures)
    initialCotangents.push_back(buildZeroLike(builder, capture));

  Location loc = whileOp.getLoc();
  Value zero = arith::ConstantIndexOp::create(builder, loc, 0);
  Value one = arith::ConstantIndexOp::create(builder, loc, 1);
  Value tripCount = whileOp.getResult(0);
  auto backward = scf::ForOp::create(builder, loc, zero, tripCount, one,
                                     initialCotangents);
  {
    OpBuilder::InsertionGuard guard(builder);
    builder.setInsertionPointToStart(backward.getBody());
    Value last = arith::SubIOp::create(builder, loc, tripCount, one);
    Value reverseOrdinal = arith::SubIOp::create(
        builder, loc, last, backward.getInductionVar());

    auto replay = scf::ForOp::create(builder, loc, zero, reverseOrdinal, one,
                                     whileOp.getInits());
    {
      OpBuilder::InsertionGuard replayGuard(builder);
      builder.setInsertionPointToStart(replay.getBody());
      IRMapping mapping;
      for (auto [source, destination] : llvm::zip_equal(
               body.front().getArguments(), replay.getRegionIterArgs()))
        mapping.map(source, destination);
      for (Operation &source : body.front().without_terminator())
        builder.clone(source, mapping);
      SmallVector<Value> replayYields;
      for (Value value : yield.getOperands())
        replayYields.push_back(mapping.lookupOrDefault(value));
      scf::YieldOp::create(builder, loc, replayYields);
    }

    ValueRange carried = backward.getRegionIterArgs();
    ValueRange stateCotangents =
        carried.take_front(differentiableState.size());
    ValueRange captureAccumulators =
        carried.drop_front(differentiableState.size());
    SmallVector<Value> outputSeeds(whileOp.getNumResults());
    for (auto [ordinal, stateIndex] : llvm::enumerate(differentiableState))
      outputSeeds[stateIndex] = stateCotangents[ordinal];
    SmallVector<Value> blockCotangents;
    SmallVector<Value> captureStepCotangents;
    if (failed(buildRegionPullback(
            body, outputSeeds, replay.getResults(), captures, builder,
            blockCotangents, captureStepCotangents)) ||
        blockCotangents.size() != whileOp.getInits().size() ||
        captureStepCotangents.size() != captures.size()) {
      backward.erase();
      return failure();
    }

    SmallVector<Value> nextCotangents;
    for (unsigned stateIndex : differentiableState)
      nextCotangents.push_back(blockCotangents[stateIndex]);
    for (auto [accumulator, stepCotangent] : llvm::zip_equal(
             captureAccumulators, captureStepCotangents))
      nextCotangents.push_back(
          addCotangents(builder, loc, accumulator, stepCotangent));
    scf::YieldOp::create(builder, loc, nextCotangents);
  }

  for (auto [ordinal, stateIndex] : llvm::enumerate(differentiableState))
    regionCotangents.push_back(
        {whileOp.getInits()[stateIndex], backward.getResult(ordinal)});
  ValueRange captureResults =
      backward.getResults().drop_front(differentiableState.size());
  for (auto [capture, cotangent] :
       llvm::zip_equal(captures, captureResults))
    regionCotangents.push_back({capture, cotangent});
  return success();
}

} // namespace

bool RegionAdjointInterface::supports(Operation *op) {
  if (auto whileOp = dyn_cast<scf::WhileOp>(op))
    return isCanonicalBoundedWhile(whileOp);
  if (auto forOp = dyn_cast<scf::ForOp>(op))
    return isCanonicalCountedFor(forOp);
  return isa<scf::IfOp>(op);
}

LogicalResult RegionAdjointInterface::buildAdjoint(
    Operation *op, OpBuilder &builder, ValueRange outputCotangents,
    RegionPullbackBuilder buildRegionPullback,
    SmallVectorImpl<RegionCotangent> &captureCotangents) {
  if (auto ifOp = dyn_cast<scf::IfOp>(op))
    return buildIfAdjoint(ifOp, builder, outputCotangents,
                          buildRegionPullback, captureCotangents);
  if (auto forOp = dyn_cast<scf::ForOp>(op))
    return buildForAdjoint(forOp, builder, outputCotangents,
                           buildRegionPullback, captureCotangents);
  if (auto whileOp = dyn_cast<scf::WhileOp>(op))
    return buildWhileAdjoint(whileOp, builder, outputCotangents,
                             buildRegionPullback, captureCotangents);
  return failure();
}

} // namespace tessera
