#include "Tessera/Transforms/RegionAdjointInterface.h"

#include "Tessera/Transforms/LoopBodyYield.h"
#include "Tessera/Transforms/SemanticEffects.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/TypeUtilities.h"
#include "mlir/Transforms/RegionUtils.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/DenseMap.h"
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
    // Structured containers are replayable exactly when their recursively
    // visited contents are replayable. Treating the container itself as an
    // opaque effect rejects every canonicalized multi-block state machine even
    // though each selected block is pure and its path is explicit SSA.
    if (isa<scf::IfOp, scf::ForOp, scf::WhileOp>(op))
      return;
    // Compiler-generated assertions are idempotent observations of replayed
    // SSA and carry no mutable state. Admit only the named cf.assert contract;
    // arbitrary operations cannot self-declare away their effects.
    if (op->getName().getStringRef() == "cf.assert") {
      auto guard = op->getAttrOfType<BoolAttr>(
          "tessera.autodiff.replay_safe_guard");
      if (guard && guard.getValue())
        return;
    }
    // W4-EFFECTS-1 (PR #630 review). An effectful op that carries a VERIFIED
    // recorded product is replayable by construction — that is what the
    // product means. Without this, a keyed dropout admitted at the top level
    // was still rejected the moment it appeared inside a supported scf.if /
    // scf.for / scf.while, so the newly admitted family failed with
    // AUTODIFF_REGION_ADJOINT under exactly the control flow W4 exists to
    // support. The check is the SHARED one, so admission cannot diverge
    // between a top-level op and the same op nested in a region (#31).
    if (getRegisteredSemanticEffect(op) == SemanticEffectLevel::Random &&
        carriesVerifiedRecordedProduct(op, "keyed_rng"))
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
    ValueRange explicitResiduals,
    RegionPullbackBuilder buildRegionPullback,
    SmallVectorImpl<RegionCotangent> &captureCotangents) {
  if (ifOp.getElseRegion().empty() ||
      !isReplayable(ifOp.getThenRegion()) ||
      !isReplayable(ifOp.getElseRegion()) ||
      outputCotangents.size() != ifOp.getNumResults() ||
      (!explicitResiduals.empty() &&
       !explicitResiduals.front().getType().isInteger(1)))
    return failure();

  // Top-level data-dependent branches receive their forward predicate through
  // the public paired residual ABI. Nested pure branches created by bounded CFG
  // structurization are deterministically recomputed with their enclosing
  // state step, so their local SSA condition is the exact path identity.
  Value predicate = explicitResiduals.empty() ? ifOp.getCondition()
                                               : explicitResiduals.front();

  SmallVector<Value> captures = collectIfCaptures(ifOp);
  SmallVector<Value> thenSaved, elseSaved;
  auto thenCount = ifOp->getAttrOfType<IntegerAttr>(
      "tessera.autodiff.then_saved_count");
  auto elseCount = ifOp->getAttrOfType<IntegerAttr>(
      "tessera.autodiff.else_saved_count");
  if (explicitResiduals.size() > 1) {
    if (!thenCount || !elseCount || thenCount.getInt() < 0 ||
        elseCount.getInt() < 0 ||
        static_cast<size_t>(1 + thenCount.getInt() + elseCount.getInt()) !=
            explicitResiduals.size())
      return failure();
    thenSaved.append(explicitResiduals.begin() + 1,
                     explicitResiduals.begin() + 1 + thenCount.getInt());
    elseSaved.append(explicitResiduals.begin() + 1 + thenCount.getInt(),
                     explicitResiduals.end());
  }
  int64_t primalResultCount =
      ifOp.getNumResults() - thenSaved.size() - elseSaved.size();
  if (primalResultCount < 0)
    return failure();
  SmallVector<Type> resultTypes;
  for (Value capture : captures)
    resultTypes.push_back(capture.getType());

  auto backwardIf = scf::IfOp::create(builder, ifOp.getLoc(), resultTypes,
                                      predicate,
                                      /*withElseRegion=*/true);
  auto buildBranch = [&](Region &source, Block *destination, ValueRange saved,
                         int64_t savedOffset) -> LogicalResult {
    OpBuilder::InsertionGuard guard(builder);
    builder.setInsertionPointToStart(destination);
    SmallVector<Value> branchCotangents;
    SmallVector<Value> blockCotangents;
    llvm::DenseMap<Value, Value> savedValues;
    auto sourceYield = dyn_cast<scf::YieldOp>(source.front().getTerminator());
    if (!sourceYield || savedOffset < 0 ||
        static_cast<size_t>(savedOffset) + saved.size() >
            sourceYield.getNumOperands())
      return failure();
    ValueRange candidates =
        sourceYield.getOperands().slice(savedOffset, saved.size());
    for (auto [sourceValue, savedValue] : llvm::zip_equal(candidates, saved))
      savedValues.try_emplace(sourceValue, savedValue);
    if (failed(buildRegionPullback(source, outputCotangents, {}, captures,
                                   savedValues, builder, blockCotangents,
                                   branchCotangents)) ||
        !blockCotangents.empty() ||
        branchCotangents.size() != captures.size())
      return failure();
    scf::YieldOp::create(builder, ifOp.getLoc(), branchCotangents);
    return success();
  };

  if (failed(buildBranch(ifOp.getThenRegion(), backwardIf.thenBlock(),
                         thenSaved, primalResultCount)) ||
      failed(buildBranch(ifOp.getElseRegion(), backwardIf.elseBlock(),
                         elseSaved,
                         primalResultCount + thenSaved.size()))) {
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
    ValueRange explicitResiduals,
    RegionPullbackBuilder buildRegionPullback,
    SmallVectorImpl<RegionCotangent> &regionCotangents) {
  Region &region = forOp.getRegion();
  if (!isCanonicalCountedFor(forOp) || !isReplayable(region) ||
      outputCotangents.size() != forOp.getNumResults())
    return failure();
  auto yield = dyn_cast<scf::YieldOp>(region.front().getTerminator());
  if (!yield || yield.getNumOperands() != forOp.getNumResults())
    return failure();

  auto policy = forOp->getAttrOfType<StringAttr>(
      "tessera.autodiff.checkpoint_policy");
  const bool materializedPolicy =
      policy && (policy.getValue() == "save" || policy.getValue() == "hybrid");
  unsigned primalResultCount = forOp.getNumResults();
  DenseI64ArrayAttr checkpointIndices;
  DenseI64ArrayAttr shapeTapeOrdinals;
  SmallVector<int64_t> residualStateIndices;
  StringRef residualOwner;
  if (materializedPolicy) {
    auto materialized = forOp->getAttrOfType<BoolAttr>(
        "tessera.autodiff.residual_materialized");
    auto owner = forOp->getAttrOfType<StringAttr>(
        "tessera.autodiff.residual_owner");
    auto resultIndices = forOp->getAttrOfType<DenseI64ArrayAttr>(
        "tessera.autodiff.residual_result_indices");
    checkpointIndices = forOp->getAttrOfType<DenseI64ArrayAttr>(
        "tessera.autodiff.checkpoint_indices");
    auto primalIndices = forOp->getAttrOfType<DenseI64ArrayAttr>(
        "tessera.autodiff.residual_primal_iter_arg_indices");
    shapeTapeOrdinals = forOp->getAttrOfType<DenseI64ArrayAttr>(
        "tessera.autodiff.residual_shape_tape_ordinals");
    if (!materialized || !materialized.getValue() || !owner || !resultIndices ||
        !checkpointIndices || checkpointIndices.empty() || !primalIndices ||
        explicitResiduals.size() != static_cast<size_t>(resultIndices.size()))
      return failure();
    residualOwner = owner.getValue();
    residualStateIndices.assign(primalIndices.asArrayRef().begin(),
                                primalIndices.asArrayRef().end());
    if (residualOwner == "control_scan") {
      if (resultIndices.size() != 1 || primalIndices.size() != 1 ||
          resultIndices[0] != forOp.getNumResults() - 1 ||
          residualStateIndices != SmallVector<int64_t>{0})
        return failure();
      primalResultCount = forOp.getNumResults() - 1;
    } else if (residualOwner == "generic_for") {
      if (resultIndices.empty() || primalIndices.empty() ||
          !shapeTapeOrdinals ||
          shapeTapeOrdinals.size() != primalIndices.size() ||
          resultIndices.size() < primalIndices.size() ||
          resultIndices.size() >= forOp.getNumResults())
        return failure();
      primalResultCount = forOp.getNumResults() - resultIndices.size();
      llvm::SmallDenseSet<int64_t> seenStates;
      for (auto [ordinal, resultIndex] : llvm::enumerate(
               resultIndices.asArrayRef()))
        if (resultIndex != static_cast<int64_t>(primalResultCount + ordinal))
          return failure();
      llvm::SmallDenseSet<int64_t> seenShapeTapes;
      for (auto [ordinal, stateIndex] :
           llvm::enumerate(residualStateIndices)) {
        int64_t shapeOrdinal = shapeTapeOrdinals[ordinal];
        if (stateIndex < 0 ||
            residualStateIndices[ordinal] >=
                static_cast<int64_t>(primalResultCount) ||
            !seenStates.insert(stateIndex).second || shapeOrdinal < -1 ||
            shapeOrdinal >= static_cast<int64_t>(resultIndices.size()) ||
            (shapeOrdinal >= 0 &&
             (!seenShapeTapes.insert(shapeOrdinal).second ||
              shapeOrdinal < static_cast<int64_t>(primalIndices.size()))))
          return failure();
      }
    } else {
      return failure();
    }
  } else if (!explicitResiduals.empty()) {
    return failure();
  }

  SmallVector<Value> captures = collectDifferentiableCaptures(region);
  SmallVector<Value> initialCotangents;
  for (auto [result, cotangent] :
       llvm::zip_equal(forOp.getResults().take_front(primalResultCount),
                       outputCotangents.take_front(primalResultCount)))
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

    SmallVector<Value> blockValues{reverseIv};
    if (materializedPolicy) {
      Value forwardOffset = arith::SubIOp::create(
          builder, loc, reverseIv, forOp.getLowerBound());
      Value ordinal = arith::DivSIOp::create(
          builder, loc, forwardOffset, forOp.getStep());
      Value checkpointOrdinal = arith::ConstantIndexOp::create(builder, loc, 0);
      for (auto [slot, checkpoint] :
           llvm::enumerate(checkpointIndices.asArrayRef())) {
        Value checkpointValue = arith::ConstantIndexOp::create(
            builder, loc, checkpoint);
        Value available = arith::CmpIOp::create(
            builder, loc, arith::CmpIPredicate::ule, checkpointValue, ordinal);
        checkpointOrdinal = arith::SelectOp::create(
            builder, loc, available, checkpointValue, checkpointOrdinal);
      }
      SmallVector<Value> primalValues(
          forOp.getResults().take_front(primalResultCount).begin(),
          forOp.getResults().take_front(primalResultCount).end());
      for (auto [tapeOrdinal, stateIndex] :
           llvm::enumerate(residualStateIndices)) {
        Type stateType = forOp.getInitArgs()[stateIndex].getType();
        auto rankedStateType = dyn_cast<RankedTensorType>(stateType);
        auto tapeType =
            dyn_cast<RankedTensorType>(explicitResiduals[tapeOrdinal].getType());
        if (!tapeType ||
            tapeType.getRank() != (rankedStateType ? rankedStateType.getRank() + 1
                                                   : 1) ||
            tapeType.getDimSize(0) !=
                static_cast<int64_t>(checkpointIndices.size()) ||
            tapeType.getElementType() != getElementTypeOrSelf(stateType))
          return failure();
        Value shapeTape;
        // control_scan predates the generic per-state shape-tape ABI and its
        // retained state is statically shaped.  Only generic_for requires the
        // total ordinal vector; never index a null DenseArrayAttr.
        int64_t shapeOrdinal = shapeTapeOrdinals
                                   ? shapeTapeOrdinals[tapeOrdinal]
                                   : -1;
        if (shapeOrdinal >= 0) {
          shapeTape = explicitResiduals[shapeOrdinal];
          auto shapeTapeType = dyn_cast<RankedTensorType>(shapeTape.getType());
          if (!rankedStateType || !shapeTapeType ||
              shapeTapeType.getShape() !=
                  ArrayRef<int64_t>{
                      static_cast<int64_t>(checkpointIndices.size()),
                      rankedStateType.getRank()} ||
              !shapeTapeType.getElementType().isIndex())
            return failure();
        }
        for (int64_t dim = 0;
             rankedStateType && dim < rankedStateType.getRank(); ++dim)
          if (!rankedStateType.isDynamicDim(dim) &&
              tapeType.getDimSize(dim + 1) !=
                  rankedStateType.getDimSize(dim))
            return failure();
        Value checkpointState = forOp.getInitArgs()[stateIndex];
        // Under a dense interior checkpoint set (1..trip-1, which is what the
        // "save" policy always produces) the chain below reduces to arithmetic:
        // the largest checkpoint <= ordinal is exactly ordinal - 1. Emitting
        // one dynamic-offset slice instead of a per-slot select chain turns
        // O(trip) emitted IR and O(trip) whole-tensor selects per backward
        // iteration into O(1).
        bool denseInterior = !checkpointIndices.empty();
        for (auto [slot, checkpoint] :
             llvm::enumerate(checkpointIndices.asArrayRef()))
          denseInterior &= checkpoint == static_cast<int64_t>(slot) + 1;

        // Decide the one way this path can fail before building any of it, so a
        // refusal does not leave a half-emitted select behind.
        if (denseInterior && rankedStateType && !shapeTape)
          for (int64_t dim : rankedStateType.getShape())
            if (ShapedType::isDynamic(dim))
              return failure();

        if (denseInterior) {
          Value one = arith::ConstantIndexOp::create(builder, loc, 1);
          Value available = arith::CmpIOp::create(
              builder, loc, arith::CmpIPredicate::uge, ordinal, one);
          auto selectState = scf::IfOp::create(
              builder, loc, TypeRange{stateType}, available, true);
          {
            OpBuilder::InsertionGuard stateGuard(builder);
            builder.setInsertionPointToStart(selectState.thenBlock());
            // The chain this replaces stopped at its last slot for an ordinal
            // past the final checkpoint; clamp so the slice cannot run off the
            // tape if an ordinal ever exceeds the checkpoint count.
            Value lastSlot = arith::ConstantIndexOp::create(
                builder, loc,
                static_cast<int64_t>(checkpointIndices.size()) - 1);
            Value slotIndex = arith::MinUIOp::create(
                builder, loc,
                arith::SubIOp::create(builder, loc, ordinal, one), lastSlot);
            Value retained;
            if (rankedStateType) {
              SmallVector<OpFoldResult> offsets{OpFoldResult(slotIndex)};
              SmallVector<OpFoldResult> sizes{builder.getIndexAttr(1)};
              SmallVector<OpFoldResult> strides(rankedStateType.getRank() + 1,
                                                builder.getIndexAttr(1));
              for (auto [dimIndex, dim] :
                   llvm::enumerate(rankedStateType.getShape())) {
                offsets.push_back(builder.getIndexAttr(0));
                if (ShapedType::isDynamic(dim)) {
                  Value dimIndexValue = arith::ConstantIndexOp::create(
                      builder, loc, dimIndex);
                  sizes.push_back(OpFoldResult(tensor::ExtractOp::create(
                      builder, loc, shapeTape,
                      ValueRange{slotIndex, dimIndexValue})));
                } else {
                  sizes.push_back(OpFoldResult(builder.getIndexAttr(dim)));
                }
              }
              retained = tensor::ExtractSliceOp::create(
                  builder, loc, rankedStateType,
                  explicitResiduals[tapeOrdinal], offsets, sizes, strides);
            } else {
              retained = tensor::ExtractOp::create(
                  builder, loc, explicitResiduals[tapeOrdinal], slotIndex);
            }
            scf::YieldOp::create(builder, loc, retained);
          }
          {
            OpBuilder::InsertionGuard stateGuard(builder);
            builder.setInsertionPointToStart(selectState.elseBlock());
            scf::YieldOp::create(builder, loc, checkpointState);
          }
          primalValues[stateIndex] = selectState.getResult(0);
          continue;
        }

        for (auto [slot, checkpoint] :
             llvm::enumerate(checkpointIndices.asArrayRef())) {
          Value checkpointValue = arith::ConstantIndexOp::create(
              builder, loc, checkpoint);
          Value available = arith::CmpIOp::create(
              builder, loc, arith::CmpIPredicate::ule, checkpointValue,
              ordinal);
          Value retained;
          if (rankedStateType) {
            SmallVector<OpFoldResult> offsets{
                builder.getIndexAttr(static_cast<int64_t>(slot))};
            SmallVector<OpFoldResult> sizes{builder.getIndexAttr(1)};
            SmallVector<OpFoldResult> strides(rankedStateType.getRank() + 1,
                                              builder.getIndexAttr(1));
            for (auto [dimIndex, dim] :
                 llvm::enumerate(rankedStateType.getShape())) {
              offsets.push_back(builder.getIndexAttr(0));
              if (ShapedType::isDynamic(dim)) {
                if (!shapeTape)
                  return failure();
                Value slotIndex = arith::ConstantIndexOp::create(
                    builder, loc, static_cast<int64_t>(slot));
                Value dimIndexValue = arith::ConstantIndexOp::create(
                    builder, loc, dimIndex);
                sizes.push_back(OpFoldResult(tensor::ExtractOp::create(
                    builder, loc, shapeTape,
                    ValueRange{slotIndex, dimIndexValue})));
              } else {
                sizes.push_back(OpFoldResult(builder.getIndexAttr(dim)));
              }
            }
            retained = tensor::ExtractSliceOp::create(
                builder, loc, rankedStateType,
                explicitResiduals[tapeOrdinal], offsets, sizes, strides);
          } else {
            Value slotIndex = arith::ConstantIndexOp::create(
                builder, loc, static_cast<int64_t>(slot));
            retained = tensor::ExtractOp::create(
                builder, loc, explicitResiduals[tapeOrdinal], slotIndex);
          }
          auto selectState = scf::IfOp::create(
              builder, loc, TypeRange{stateType}, available, true);
          {
            OpBuilder::InsertionGuard stateGuard(builder);
            builder.setInsertionPointToStart(selectState.thenBlock());
            scf::YieldOp::create(builder, loc, retained);
          }
          {
            OpBuilder::InsertionGuard stateGuard(builder);
            builder.setInsertionPointToStart(selectState.elseBlock());
            scf::YieldOp::create(builder, loc, checkpointState);
          }
          checkpointState = selectState.getResult(0);
        }
        primalValues[stateIndex] = checkpointState;
      }
      if (policy.getValue() == "save") {
        llvm::append_range(blockValues, primalValues);
        llvm::append_range(blockValues, explicitResiduals);
      } else {
        Value checkpointOffset = arith::MulIOp::create(
            builder, loc, checkpointOrdinal, forOp.getStep());
        Value checkpointIv = arith::AddIOp::create(
            builder, loc, forOp.getLowerBound(), checkpointOffset);
        SmallVector<Value> replayInits(primalValues.begin(), primalValues.end());
        llvm::append_range(replayInits, explicitResiduals);
        auto replay = scf::ForOp::create(builder, loc, checkpointIv, reverseIv,
                                         forOp.getStep(), replayInits);
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
          closeBodyWithYield(builder, loc, replay.getBody(), replayYields);
        }
        llvm::append_range(blockValues, replay.getResults());
      }
    } else {
      // RECOMPUTE_ALL baseline: replay [lb, reverseIv) to recover the primal
      // carried state at the selected iteration.
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
        closeBodyWithYield(builder, loc, replay.getBody(), replayYields);
      }
      llvm::append_range(blockValues, replay.getResults());
    }
    ValueRange carried = backward.getRegionIterArgs();
    ValueRange stateCotangents = carried.take_front(primalResultCount);
    ValueRange captureAccumulators = carried.drop_front(primalResultCount);
    SmallVector<Value> regionSeeds(stateCotangents.begin(),
                                   stateCotangents.end());
    if (materializedPolicy)
      regionSeeds.append(explicitResiduals.size(), Value{});
    SmallVector<Value> blockCotangents;
    SmallVector<Value> captureStepCotangents;
    llvm::DenseMap<Value, Value> noSavedValues;
    if (failed(buildRegionPullback(
            region, regionSeeds, blockValues, captures, noSavedValues, builder,
            blockCotangents, captureStepCotangents)) ||
        blockCotangents.size() != blockValues.size() ||
        captureStepCotangents.size() != captures.size()) {
      backward.erase();
      return failure();
    }

    // Nested region pullbacks may leave the shared builder in their final
    // branch. Continue construction in the reverse loop body, and make the
    // mixed-state contract total: control/index slots deliberately receive no
    // analytical cotangent, but scf.for still requires a typed value for every
    // carried slot.
    builder.setInsertionPointToEnd(backward.getBody());
    SmallVector<Value> nextCotangents;
    // The induction variable is integer-valued and deliberately has no
    // cotangent. Remaining block arguments are loop-carried primals.
    for (auto [stateIndex, cotangent] : llvm::enumerate(
             llvm::ArrayRef(blockCotangents).slice(1, primalResultCount)))
      nextCotangents.push_back(
          cotangent ? cotangent
                    : buildZeroLike(builder, forOp.getInitArgs()[stateIndex]));
    for (auto [accumulator, stepCotangent] : llvm::zip_equal(
             captureAccumulators, captureStepCotangents))
      nextCotangents.push_back(
          addCotangents(builder, loc, accumulator, stepCotangent));
    closeBodyWithYield(builder, loc, backward.getBody(), nextCotangents);
  }

  for (auto [init, cotangent] : llvm::zip_equal(
           forOp.getInitArgs().take_front(primalResultCount),
           backward.getResults().take_front(primalResultCount)))
    if (isDifferentiableType(init.getType()))
      regionCotangents.push_back({init, cotangent});
  for (auto [capture, cotangent] : llvm::zip_equal(
           captures, backward.getResults().drop_front(primalResultCount)))
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

static FailureOr<Value> extractWhileTapeState(OpBuilder &builder, Location loc,
                                              Value tape,
                                              RankedTensorType stateType,
                                              OpFoldResult slot) {
  auto tapeType = dyn_cast<RankedTensorType>(tape.getType());
  if (!tapeType || tapeType.getRank() != stateType.getRank() + 1)
    return failure();
  SmallVector<OpFoldResult> offsets{slot};
  SmallVector<OpFoldResult> sizes{builder.getIndexAttr(1)};
  SmallVector<OpFoldResult> strides(stateType.getRank() + 1,
                                    builder.getIndexAttr(1));
  for (auto [dimIndex, dim] : llvm::enumerate(stateType.getShape())) {
    offsets.push_back(builder.getIndexAttr(0));
    sizes.push_back(ShapedType::isDynamic(dim)
                        ? OpFoldResult(tensor::DimOp::create(
                                           builder, loc, tape, dimIndex + 1)
                                           .getResult())
                        : OpFoldResult(builder.getIndexAttr(dim)));
  }
  return tensor::ExtractSliceOp::create(builder, loc, stateType, tape,
                                        offsets, sizes, strides)
      .getResult();
}

static LogicalResult buildWhileAdjoint(
    scf::WhileOp whileOp, OpBuilder &builder, ValueRange outputCotangents,
    ValueRange explicitResiduals,
    RegionPullbackBuilder buildRegionPullback,
    SmallVectorImpl<RegionCotangent> &regionCotangents) {
  auto materialized = whileOp->getAttrOfType<BoolAttr>(
      "tessera.autodiff.residual_materialized");
  auto owner = whileOp->getAttrOfType<StringAttr>(
      "tessera.autodiff.residual_owner");
  auto resultIndices = whileOp->getAttrOfType<DenseI64ArrayAttr>(
      "tessera.autodiff.residual_result_indices");
  auto primalIndices = whileOp->getAttrOfType<DenseI64ArrayAttr>(
      "tessera.autodiff.residual_primal_iter_arg_indices");
  auto checkpointPolicy = whileOp->getAttrOfType<StringAttr>(
      "tessera.autodiff.checkpoint_policy");
  auto checkpoints = whileOp->getAttrOfType<DenseI64ArrayAttr>(
      "tessera.autodiff.checkpoint_indices");
  const bool savedStateTape = materialized && materialized.getValue();
  const bool hybridStateTape =
      savedStateTape && checkpointPolicy &&
      checkpointPolicy.getValue() == "hybrid";
  unsigned primalResultCount = whileOp.getNumResults();
  if (savedStateTape) {
    if (!owner || owner.getValue() != "scf_while" || !resultIndices ||
        !primalIndices || resultIndices.size() != primalIndices.size() ||
        explicitResiduals.size() !=
            static_cast<size_t>(resultIndices.size() + 1) ||
        whileOp.getNumResults() < resultIndices.size() ||
        (hybridStateTape && (!checkpoints || checkpoints.empty())))
      return failure();
    primalResultCount = whileOp.getNumResults() - resultIndices.size();
    for (auto [ordinal, resultIndex] :
         llvm::enumerate(resultIndices.asArrayRef()))
      if (resultIndex != static_cast<int64_t>(primalResultCount + ordinal) ||
          primalIndices[ordinal] <= 0 ||
          primalIndices[ordinal] >= primalResultCount)
        return failure();
  }
  if (!isCanonicalBoundedWhile(whileOp) ||
      !isReplayable(whileOp.getBefore()) ||
      !isReplayable(whileOp.getAfter()) ||
      outputCotangents.size() != whileOp.getNumResults() ||
      (!savedStateTape && explicitResiduals.size() != 1) ||
      !isa<IndexType>(explicitResiduals.front().getType()))
    return failure();

  Region &body = whileOp.getAfter();
  auto yield = dyn_cast<scf::YieldOp>(body.front().getTerminator());
  SmallVector<unsigned> differentiableState;
  for (auto [index, init] : llvm::enumerate(
           whileOp.getInits().take_front(primalResultCount)))
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
  Value tripCount = explicitResiduals.front();
  auto backward = scf::ForOp::create(builder, loc, zero, tripCount, one,
                                     initialCotangents);
  {
    OpBuilder::InsertionGuard guard(builder);
    builder.setInsertionPointToStart(backward.getBody());
    Value last = arith::SubIOp::create(builder, loc, tripCount, one);
    Value reverseOrdinal = arith::SubIOp::create(
        builder, loc, last, backward.getInductionVar());

    SmallVector<Value> primalBlockValues;
    if (savedStateTape) {
      llvm::SmallDenseMap<unsigned, unsigned, 4> stateToTape;
      for (auto [tapeOrdinal, stateIndex] :
           llvm::enumerate(primalIndices.asArrayRef()))
        stateToTape.try_emplace(stateIndex, tapeOrdinal);
      auto extractState = [&](unsigned stateIndex,
                              OpFoldResult slot) -> FailureOr<Value> {
        auto tapeIt = stateToTape.find(stateIndex);
        auto stateType = dyn_cast<RankedTensorType>(
            whileOp.getInits()[stateIndex].getType());
        if (tapeIt == stateToTape.end() || !stateType)
          return failure();
        return extractWhileTapeState(
            builder, loc, explicitResiduals[1 + tapeIt->second], stateType,
            slot);
      };
      if (!hybridStateTape) {
        primalBlockValues.resize(primalResultCount);
        primalBlockValues[0] = reverseOrdinal;
        for (unsigned stateIndex = 1; stateIndex < primalResultCount;
             ++stateIndex) {
          FailureOr<Value> state = extractState(stateIndex, reverseOrdinal);
          if (failed(state)) {
            backward.erase();
            return failure();
          }
          primalBlockValues[stateIndex] = *state;
        }
      } else {
        SmallVector<Value> selected(
            whileOp.getInits().take_front(primalResultCount));
        for (auto [slot, checkpoint] :
             llvm::enumerate(checkpoints.asArrayRef())) {
          Value checkpointValue =
              arith::ConstantIndexOp::create(builder, loc, checkpoint);
          Value checkpointApplies = arith::CmpIOp::create(
              builder, loc, arith::CmpIPredicate::ule, checkpointValue,
              reverseOrdinal);
          SmallVector<Type> selectedTypes;
          for (Value value : selected)
            selectedTypes.push_back(value.getType());
          auto select = scf::IfOp::create(builder, loc, selectedTypes,
                                          checkpointApplies, true);
          {
            OpBuilder::InsertionGuard selectGuard(builder);
            builder.setInsertionPointToStart(select.thenBlock());
            SmallVector<Value> retained{checkpointValue};
            for (unsigned stateIndex = 1; stateIndex < primalResultCount;
                 ++stateIndex) {
              FailureOr<Value> state = extractState(
                  stateIndex,
                  builder.getIndexAttr(static_cast<int64_t>(slot)));
              if (failed(state)) {
                select.erase();
                backward.erase();
                return failure();
              }
              retained.push_back(*state);
            }
            scf::YieldOp::create(builder, loc, retained);
          }
          {
            OpBuilder::InsertionGuard selectGuard(builder);
            builder.setInsertionPointToStart(select.elseBlock());
            scf::YieldOp::create(builder, loc, selected);
          }
          selected.assign(select.getResults().begin(),
                          select.getResults().end());
        }

        auto primalOpCount = whileOp->getAttrOfType<IntegerAttr>(
            "tessera.autodiff.residual_primal_op_count");
        if (!primalOpCount || primalOpCount.getInt() < 0 ||
            primalOpCount.getInt() >
                std::distance(body.front().begin(),
                              body.front().getTerminator()->getIterator())) {
          backward.erase();
          return failure();
        }
        auto replay = scf::ForOp::create(builder, loc, selected.front(),
                                         reverseOrdinal, one, selected);
        {
          OpBuilder::InsertionGuard replayGuard(builder);
          builder.setInsertionPointToStart(replay.getBody());
          IRMapping mapping;
          for (auto [source, destination] : llvm::zip_equal(
                   body.front().getArguments().take_front(primalResultCount),
                   replay.getRegionIterArgs()))
            mapping.map(source, destination);
          auto source = body.front().begin();
          for (int64_t ordinal = 0; ordinal < primalOpCount.getInt();
               ++ordinal, ++source)
            builder.clone(*source, mapping);
          SmallVector<Value> replayYields;
          for (Value value :
               yield.getOperands().take_front(primalResultCount))
            replayYields.push_back(mapping.lookupOrDefault(value));
          scf::YieldOp::create(builder, loc, replayYields);
        }
        llvm::append_range(primalBlockValues, replay.getResults());
      }
    } else {
      auto replay = scf::ForOp::create(
          builder, loc, zero, reverseOrdinal, one,
          whileOp.getInits().take_front(primalResultCount));
      {
        OpBuilder::InsertionGuard replayGuard(builder);
        builder.setInsertionPointToStart(replay.getBody());
        IRMapping mapping;
        for (auto [source, destination] : llvm::zip_equal(
                 body.front().getArguments().take_front(primalResultCount),
                 replay.getRegionIterArgs()))
          mapping.map(source, destination);
        for (Operation &source : body.front().without_terminator())
          builder.clone(source, mapping);
        SmallVector<Value> replayYields;
        for (Value value : yield.getOperands().take_front(primalResultCount))
          replayYields.push_back(mapping.lookupOrDefault(value));
        scf::YieldOp::create(builder, loc, replayYields);
      }
      llvm::append_range(primalBlockValues, replay.getResults());
    }
    SmallVector<Value> blockValues(primalBlockValues.begin(),
                                   primalBlockValues.end());
    if (savedStateTape)
      llvm::append_range(blockValues, explicitResiduals.drop_front());

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
    llvm::DenseMap<Value, Value> noSavedValues;
    if (failed(buildRegionPullback(
            body, outputSeeds, blockValues, captures, noSavedValues,
            builder,
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
    ValueRange explicitResiduals,
    RegionPullbackBuilder buildRegionPullback,
    SmallVectorImpl<RegionCotangent> &captureCotangents) {
  if (auto ifOp = dyn_cast<scf::IfOp>(op)) {
    return buildIfAdjoint(ifOp, builder, outputCotangents, explicitResiduals,
                          buildRegionPullback, captureCotangents);
  }
  if (auto forOp = dyn_cast<scf::ForOp>(op))
    return buildForAdjoint(forOp, builder, outputCotangents, explicitResiduals,
                           buildRegionPullback, captureCotangents);
  if (auto whileOp = dyn_cast<scf::WhileOp>(op)) {
    return buildWhileAdjoint(whileOp, builder, outputCotangents,
                             explicitResiduals,
                             buildRegionPullback, captureCotangents);
  }
  return failure();
}

} // namespace tessera
