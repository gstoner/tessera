//===- AutodiffPairedPass.cpp - Paired forward/backward autodiff --*- C++ -*-===//
//
// Phase 2 of docs/audit/compiler/AUTODIFF_UNIFICATION_PLAN.md. Where the
// in-place `--tessera-autodiff` pass fuses the backward into the forward
// function's return (a bootstrap), this pass emits the **paired-program model**:
//
//   forward(inputs) -> (primals, explicit residuals...)
//   @f__bwd(inputs, out_cotangents..., residuals...)
//                   -> input_cotangents
//
// This is the deterministic forward/backward/residual ABI the rest of the plan
// (runtime binding in Phase 4, per-op-family expansion in Phase 5, distributed +
// accelerator promotion in Phase 6) keys off. It is verifiable independently of
// Python tape state — a lit fixture checks the backward signature + body.
//
// RECOMPUTE_ALL remains the default. SAVE regions append typed, named residual
// values to the paired ABI; backward must consume those values and cannot
// silently relabel SAVE/HYBRID as recomputation. The backward still clones the
// forward cone for unsaved intermediates (CSE later collapses redundant work).
// This is not a toy default: the shipped ROCm gfx1151
// flash-attention backward lane (`_execute_rocm_compiled_flash_attn_bwd`) takes
// `(dO, Q, K, V)` and likewise *recomputes* the softmax rather than saving the
// logsumexp. SAVE currently carries scan state tapes, branch identity, and
// executed while trip counts; scan-form HYBRID performs bounded replay from
// the nearest explicitly retained checkpoint.
//
// The paired backward is an **ABI, not an implementation**: a hand-emitted
// backward kernel (ROCm WMMA flash-attn bwd) satisfies the same
// `@f__bwd(inputs, out_cotangents) -> input_cotangents` contract and is a
// first-class arbiter candidate (Decision #28). This pass is the compiler-
// generated implementation of that contract.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/TypeUtilities.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/STLExtras.h"

#include <algorithm>
#include <functional>

#include "Tessera/AdjointInterface.h.inc"
#include "Tessera/LinearTransposeInterface.h.inc"
#include "Tessera/Transforms/GraphDataflow.h"
#include "Tessera/Transforms/RegionAdjointInterface.h"
#include "Tessera/Transforms/SemanticEffects.h"

namespace tessera {

namespace {

constexpr const char *kAutodiffMarker = "tessera.autodiff";

using CotangentMap = llvm::DenseMap<mlir::Value, mlir::Value>;
bool hasStochasticEffect(mlir::Operation *op) {
  return getRegisteredSemanticEffect(op) == SemanticEffectLevel::Random;
}

void eraseStopGradientBarriers(mlir::func::FuncOp func) {
  llvm::SmallVector<mlir::Operation *> barriers;
  func.walk([&](mlir::Operation *op) {
    if (op->getName().getStringRef() == "tessera.stop_gradient")
      barriers.push_back(op);
  });
  for (mlir::Operation *op : barriers) {
    if (op->getNumOperands() == 1 && op->getNumResults() == 1) {
      op->getResult(0).replaceAllUsesWith(op->getOperand(0));
      op->erase();
    }
  }
}

static mlir::FailureOr<mlir::RankedTensorType>
getResidualTapeType(mlir::Type stateType, int64_t slots) {
  if (slots <= 0)
    return mlir::failure();
  if (auto ranked = mlir::dyn_cast<mlir::RankedTensorType>(stateType)) {
    llvm::SmallVector<int64_t> shape{slots};
    llvm::append_range(shape, ranked.getShape());
    return mlir::RankedTensorType::get(shape, ranked.getElementType(),
                                       ranked.getEncoding());
  }
  if (stateType.isIntOrIndexOrFloat())
    return mlir::RankedTensorType::get({slots}, stateType);
  return mlir::failure();
}

using SavedSlotShapeEnvelopes =
    llvm::DenseMap<unsigned, llvm::SmallVector<int64_t>>;

// Dynamic saved values need two independent facts: a maximum allocation
// envelope and the actual extent at each retained checkpoint.  The former is
// carried by these three flat, total attributes; the latter is materialized as
// a shape tape beside the data tape below.  Indices address the primal loop
// iter-argument ABI, so mixed scalar/static/dynamic state cannot be confused by
// ordinal compression.
static mlir::FailureOr<SavedSlotShapeEnvelopes>
readSavedSlotShapeEnvelopes(mlir::Operation *owner, mlir::TypeRange stateTypes,
                            bool requireEveryDynamic) {
  auto indices = owner->getAttrOfType<mlir::DenseI64ArrayAttr>(
      "tessera.autodiff.saved_slot_shape_envelope_indices");
  auto ranks = owner->getAttrOfType<mlir::DenseI64ArrayAttr>(
      "tessera.autodiff.saved_slot_shape_envelope_ranks");
  auto flatBounds = owner->getAttrOfType<mlir::DenseI64ArrayAttr>(
      "tessera.autodiff.saved_slot_shape_envelope_bounds");
  if (!indices && !ranks && !flatBounds) {
    if (requireEveryDynamic && llvm::any_of(stateTypes, [](mlir::Type type) {
          auto ranked = mlir::dyn_cast<mlir::RankedTensorType>(type);
          return ranked && !ranked.hasStaticShape();
        }))
      return mlir::failure();
    return SavedSlotShapeEnvelopes{};
  }
  if (!indices || !ranks || !flatBounds || indices.size() != ranks.size())
    return mlir::failure();

  SavedSlotShapeEnvelopes envelopes;
  size_t boundOffset = 0;
  const size_t flatBoundCount = static_cast<size_t>(flatBounds.size());
  for (auto [entry, rawIndex] : llvm::enumerate(indices.asArrayRef())) {
    int64_t rawRank = ranks[entry];
    if (rawIndex < 0 || rawIndex >= static_cast<int64_t>(stateTypes.size()) ||
        rawRank <= 0 ||
        boundOffset + static_cast<size_t>(rawRank) > flatBoundCount ||
        envelopes.contains(static_cast<unsigned>(rawIndex)))
      return mlir::failure();
    auto ranked = mlir::dyn_cast<mlir::RankedTensorType>(
        stateTypes[static_cast<unsigned>(rawIndex)]);
    if (!ranked || ranked.hasStaticShape() || ranked.getRank() != rawRank)
      return mlir::failure();
    llvm::SmallVector<int64_t> bounds;
    for (int64_t dim = 0; dim < rawRank; ++dim) {
      int64_t bound = flatBounds[boundOffset++];
      int64_t staticExtent = ranked.getDimSize(dim);
      if (bound <= 0 ||
          (!mlir::ShapedType::isDynamic(staticExtent) &&
           bound != staticExtent))
        return mlir::failure();
      bounds.push_back(bound);
    }
    envelopes.try_emplace(static_cast<unsigned>(rawIndex), std::move(bounds));
  }
  if (boundOffset != flatBoundCount)
    return mlir::failure();
  if (requireEveryDynamic)
    for (auto [index, type] : llvm::enumerate(stateTypes)) {
      auto ranked = mlir::dyn_cast<mlir::RankedTensorType>(type);
      if (ranked && !ranked.hasStaticShape() && !envelopes.contains(index))
        return mlir::failure();
    }
  return envelopes;
}

static llvm::SmallVector<mlir::Value>
getDynamicTensorSizes(mlir::OpBuilder &builder, mlir::Location loc,
                      mlir::Value source,
                      mlir::RankedTensorType sourceType) {
  llvm::SmallVector<mlir::Value> sizes;
  for (int64_t dim = 0; dim < sourceType.getRank(); ++dim)
    if (sourceType.isDynamicDim(dim))
      sizes.push_back(
          mlir::tensor::DimOp::create(builder, loc, source, dim));
  return sizes;
}

static mlir::FailureOr<mlir::Value>
buildInactiveResidualZero(mlir::OpBuilder &builder, mlir::Location loc,
                          mlir::Type type) {
  mlir::Type elementType = mlir::getElementTypeOrSelf(type);
  if (!llvm::isa<mlir::FloatType>(elementType))
    return mlir::failure();
  mlir::TypedAttr zero = mlir::FloatAttr::get(elementType, 0.0);
  if (auto shaped = mlir::dyn_cast<mlir::ShapedType>(type)) {
    if (shaped.hasStaticShape())
      return mlir::arith::ConstantOp::create(
                 builder, loc, mlir::DenseElementsAttr::get(shaped, zero))
          .getResult();
    auto ranked = mlir::dyn_cast<mlir::RankedTensorType>(shaped);
    if (!ranked)
      return mlir::failure();
    // The inactive branch has no trustworthy runtime extents for a value that
    // exists only in the taken branch.  A zero-extent tensor is a type-correct,
    // initialized sentinel for every dynamic dimension.  Predicate replay
    // guarantees this value is never selected by the pullback.
    llvm::SmallVector<mlir::Value> dynamicSizes;
    for (int64_t dim : ranked.getShape())
      if (mlir::ShapedType::isDynamic(dim))
        dynamicSizes.push_back(
            mlir::arith::ConstantIndexOp::create(builder, loc, 0));
    mlir::Value empty = mlir::tensor::EmptyOp::create(
        builder, loc, ranked.getShape(), ranked.getElementType(),
        dynamicSizes);
    mlir::Value zeroValue =
        mlir::arith::ConstantOp::create(builder, loc, zero);
    return mlir::linalg::FillOp::create(builder, loc,
                                        mlir::ValueRange{zeroValue},
                                        mlir::ValueRange{empty})
        .getResult(0);
  }
  return mlir::arith::ConstantOp::create(builder, loc, zero).getResult();
}

static llvm::SmallVector<mlir::Value>
collectBranchResidualCandidates(mlir::Region &region) {
  llvm::SmallVector<mlir::Value> candidates;
  for (mlir::Operation &nested : region.front().without_terminator())
    for (mlir::Value result : nested.getResults())
      if (llvm::isa<mlir::FloatType>(
              mlir::getElementTypeOrSelf(result.getType())))
        candidates.push_back(result);
  return candidates;
}

// Prefer the compact canonical scf.if form for a verified acyclic diamond.
// Remaining bounded multi-block graphs are handled by the general state-machine
// structurizer below; unbounded/effectful graphs still fail closed.
static mlir::LogicalResult structurizeNativeDiamonds(
    mlir::func::FuncOp function) {
  llvm::SmallVector<mlir::scf::ExecuteRegionOp> regions;
  // Collect innermost regions first: structurizing an outer execute_region
  // erases its original body and would otherwise invalidate queued nested ops.
  function.walk<mlir::WalkOrder::PostOrder>(
      [&](mlir::scf::ExecuteRegionOp region) {
        if (!region.getRegion().hasOneBlock())
          regions.push_back(region);
      });
  for (mlir::scf::ExecuteRegionOp execute : regions) {
    mlir::Region &region = execute.getRegion();
    if (region.getBlocks().size() != 4)
      continue;
    mlir::Block &entry = region.front();
    auto branch = mlir::dyn_cast<mlir::cf::CondBranchOp>(
        entry.getTerminator());
    if (!branch || branch.getTrueDest() == branch.getFalseDest())
      continue;
    mlir::Block *thenBlock = branch.getTrueDest();
    mlir::Block *elseBlock = branch.getFalseDest();
    auto thenExit = mlir::dyn_cast<mlir::cf::BranchOp>(
        thenBlock->getTerminator());
    auto elseExit = mlir::dyn_cast<mlir::cf::BranchOp>(
        elseBlock->getTerminator());
    if (!thenExit || !elseExit || thenExit.getDest() != elseExit.getDest())
      continue;
    if (thenBlock->getNumArguments() !=
            branch.getTrueDestOperands().size() ||
        elseBlock->getNumArguments() !=
            branch.getFalseDestOperands().size())
      continue;
    mlir::Block *merge = thenExit.getDest();
    if (merge == &entry || merge == thenBlock || merge == elseBlock ||
        thenExit.getDestOperands().size() != merge->getNumArguments() ||
        elseExit.getDestOperands().size() != merge->getNumArguments())
      continue;
    auto finalYield =
        mlir::dyn_cast<mlir::scf::YieldOp>(merge->getTerminator());
    if (!finalYield || finalYield.getNumOperands() != execute.getNumResults())
      continue;

    mlir::OpBuilder builder(execute);
    mlir::IRMapping entryMapping;
    for (mlir::Operation &source : entry.without_terminator())
      builder.clone(source, entryMapping);
    mlir::Value condition =
        entryMapping.lookupOrDefault(branch.getCondition());
    llvm::SmallVector<mlir::Type> mergeTypes(merge->getArgumentTypes().begin(),
                                             merge->getArgumentTypes().end());
    auto replacement = mlir::scf::IfOp::create(
        builder, execute.getLoc(), mergeTypes, condition, true);
    for (mlir::NamedAttribute attr : execute->getAttrs())
      if (attr.getName().getValue().starts_with("tessera."))
        replacement->setAttr(attr.getName(), attr.getValue());
    replacement->setAttr("tessera.autodiff.native_multiblock_structurized",
                         builder.getBoolAttr(true));

    auto cloneArm = [&](mlir::Block *source, mlir::Block *destination,
                        mlir::ValueRange incoming,
                        mlir::cf::BranchOp exit) -> mlir::LogicalResult {
      if (source->getNumArguments() != incoming.size())
        return mlir::failure();
      mlir::OpBuilder::InsertionGuard guard(builder);
      builder.setInsertionPointToStart(destination);
      mlir::IRMapping mapping(entryMapping);
      for (auto [argument, value] :
           llvm::zip_equal(source->getArguments(), incoming))
        mapping.map(argument, entryMapping.lookupOrDefault(value));
      for (mlir::Operation &nested : source->without_terminator())
        builder.clone(nested, mapping);
      llvm::SmallVector<mlir::Value> yields;
      for (mlir::Value value : exit.getDestOperands())
        yields.push_back(mapping.lookupOrDefault(value));
      mlir::scf::YieldOp::create(builder, execute.getLoc(), yields);
      return mlir::success();
    };
    if (mlir::failed(cloneArm(thenBlock, replacement.thenBlock(),
                              branch.getTrueDestOperands(), thenExit)) ||
        mlir::failed(cloneArm(elseBlock, replacement.elseBlock(),
                              branch.getFalseDestOperands(), elseExit))) {
      replacement.erase();
      return mlir::failure();
    }

    builder.setInsertionPointAfter(replacement);
    mlir::IRMapping mergeMapping(entryMapping);
    for (auto [argument, value] :
         llvm::zip_equal(merge->getArguments(), replacement.getResults()))
      mergeMapping.map(argument, value);
    for (mlir::Operation &source : merge->without_terminator())
      builder.clone(source, mergeMapping);
    llvm::SmallVector<mlir::Value> results;
    for (mlir::Value value : finalYield.getOperands())
      results.push_back(mergeMapping.lookupOrDefault(value));
    execute->replaceAllUsesWith(results);
    execute.erase();
  }
  return mlir::success();
}

static mlir::FailureOr<mlir::Value>
buildCFGStateSentinel(mlir::OpBuilder &builder, mlir::Location loc,
                      mlir::Type type) {
  if (type.isIndex())
    return mlir::arith::ConstantIndexOp::create(builder, loc, 0).getResult();
  mlir::Type elementType = mlir::getElementTypeOrSelf(type);
  mlir::TypedAttr zero;
  if (auto floatType = mlir::dyn_cast<mlir::FloatType>(elementType))
    zero = mlir::FloatAttr::get(floatType, 0.0);
  else if (auto integerType = mlir::dyn_cast<mlir::IntegerType>(elementType))
    zero = mlir::IntegerAttr::get(integerType, 0);
  else
    return mlir::failure();
  if (auto shaped = mlir::dyn_cast<mlir::ShapedType>(type)) {
    auto ranked = mlir::dyn_cast<mlir::RankedTensorType>(shaped);
    if (!ranked)
      return mlir::failure();
    if (ranked.hasStaticShape())
      return mlir::arith::ConstantOp::create(
                 builder, loc, mlir::DenseElementsAttr::get(ranked, zero))
          .getResult();
    llvm::SmallVector<mlir::Value> dynamicSizes;
    for (int64_t dim : ranked.getShape())
      if (mlir::ShapedType::isDynamic(dim))
        dynamicSizes.push_back(
            mlir::arith::ConstantIndexOp::create(builder, loc, 0));
    mlir::Value empty = mlir::tensor::EmptyOp::create(
        builder, loc, ranked.getShape(), ranked.getElementType(),
        dynamicSizes);
    // A dynamic state-machine slot is read only after its owning CFG edge has
    // assigned it. Keeping this zero-extent sentinel uninitialized avoids a
    // synthetic active linalg region; PC/done dispatch proves it unobservable.
    return empty;
  }
  return mlir::arith::ConstantOp::create(builder, loc, zero).getResult();
}

static bool isReplayableCFGBodyOperation(mlir::Operation &operation) {
  if (operation.hasTrait<mlir::OpTrait::IsTerminator>())
    return true;
  if (operation.getNumRegions() == 0)
    return getRegisteredSemanticEffect(&operation) == SemanticEffectLevel::Pure;
  if (!llvm::isa<mlir::scf::IfOp, mlir::scf::ForOp, mlir::scf::WhileOp>(
          operation))
    return false;
  for (mlir::Region &nestedRegion : operation.getRegions()) {
    if (!nestedRegion.hasOneBlock())
      return false;
    for (mlir::Operation &nested : nestedRegion.front())
      if (!isReplayableCFGBodyOperation(nested))
        return false;
  }
  return true;
}

// Lower any bounded, pure multi-block execute_region into one canonical SCF
// state machine.  Program-counter dispatch treats reducible and irreducible
// CFGs uniformly; a required maximum-step bound makes cycles total and keeps
// reverse replay finite.  Each block argument owns a distinct typed state slot,
// so mixed tensor shapes/dtypes never alias merely because their types match.
static mlir::LogicalResult structurizeBoundedNativeCFGs(
    mlir::func::FuncOp function) {
  llvm::SmallVector<mlir::scf::ExecuteRegionOp> regions;
  function.walk<mlir::WalkOrder::PostOrder>(
      [&](mlir::scf::ExecuteRegionOp region) {
        if (!region.getRegion().hasOneBlock())
          regions.push_back(region);
      });
  for (mlir::scf::ExecuteRegionOp execute : regions) {
    mlir::Region &region = execute.getRegion();
    auto maxSteps = execute->getAttrOfType<mlir::IntegerAttr>(
        "tessera.structured_cfg.max_steps");
    auto cfgDigest = execute->getAttrOfType<mlir::StringAttr>(
        "tessera.structured_cfg.digest");
    if (!maxSteps || maxSteps.getInt() <= 0 || maxSteps.getInt() > 1'000'000 ||
        !cfgDigest || cfgDigest.getValue().size() != 64) {
      execute.emitError()
          << "general native CFG requires a positive bounded "
             "tessera.structured_cfg.max_steps and SHA-256 CFG identity";
      return mlir::failure();
    }
    if (region.empty() || region.front().getNumArguments() != 0) {
      execute.emitError()
          << "general native CFG requires an argument-free entry block";
      return mlir::failure();
    }
    auto checkpointPolicy = execute->getAttrOfType<mlir::StringAttr>(
        "tessera.autodiff.checkpoint_policy");
    const bool retainsState =
        checkpointPolicy && (checkpointPolicy.getValue() == "save" ||
                             checkpointPolicy.getValue() == "hybrid");
    llvm::SmallVector<mlir::Block *> blocks;
    llvm::DenseMap<mlir::Block *, unsigned> blockOrdinals;
    llvm::DenseMap<mlir::BlockArgument, unsigned> argumentSlots;
    llvm::SmallVector<mlir::Type> slotTypes;
    for (mlir::Block &block : region) {
      blockOrdinals.try_emplace(&block, blocks.size());
      blocks.push_back(&block);
      for (mlir::BlockArgument argument : block.getArguments()) {
        argumentSlots.try_emplace(argument, slotTypes.size());
        slotTypes.push_back(argument.getType());
      }
      for (mlir::Operation &nested : block.without_terminator()) {
        if (!isReplayableCFGBodyOperation(nested)) {
          execute.emitError()
              << "general native CFG state-machine replay requires pure, "
                 "canonical structured block operations";
          return mlir::failure();
        }
      }
      mlir::Operation *terminator = block.getTerminator();
      if (!llvm::isa<mlir::cf::BranchOp, mlir::cf::CondBranchOp,
                     mlir::scf::YieldOp>(terminator)) {
        execute.emitError()
            << "general native CFG supports cf.br, cf.cond_br, and scf.yield "
               "terminators";
        return mlir::failure();
      }
      if (auto yield = mlir::dyn_cast<mlir::scf::YieldOp>(terminator);
          yield && yield.getNumOperands() != execute.getNumResults()) {
        execute.emitError() << "native CFG yield/result cardinality mismatch";
        return mlir::failure();
      }
    }
    llvm::SmallVector<mlir::Type> envelopeStateTypes{
        mlir::IndexType::get(execute.getContext()),
        mlir::IntegerType::get(execute.getContext(), 1)};
    llvm::append_range(envelopeStateTypes, slotTypes);
    llvm::append_range(envelopeStateTypes, execute.getResultTypes());
    auto shapeEnvelopes = readSavedSlotShapeEnvelopes(
        execute, mlir::TypeRange(envelopeStateTypes), retainsState);
    if (mlir::failed(shapeEnvelopes)) {
      execute.emitError()
          << "saved native CFG dynamic state requires total, positive per-slot "
             "shape-envelope indices/ranks/bounds matching the state ABI";
      return mlir::failure();
    }

    auto validateEdge = [&](mlir::Block *target,
                            mlir::ValueRange values) -> mlir::LogicalResult {
      if (!blockOrdinals.contains(target) ||
          target->getNumArguments() != values.size())
        return mlir::failure();
      for (auto [argument, value] :
           llvm::zip_equal(target->getArguments(), values))
        if (argument.getType() != value.getType())
          return mlir::failure();
      return mlir::success();
    };
    for (mlir::Block *block : blocks) {
      if (auto branch =
              mlir::dyn_cast<mlir::cf::BranchOp>(block->getTerminator())) {
        if (mlir::failed(
                validateEdge(branch.getDest(), branch.getDestOperands()))) {
          execute.emitError() << "native CFG has an invalid cf.br edge ABI";
          return mlir::failure();
        }
      } else if (auto branch = mlir::dyn_cast<mlir::cf::CondBranchOp>(
                     block->getTerminator())) {
        if (mlir::failed(validateEdge(branch.getTrueDest(),
                                      branch.getTrueDestOperands())) ||
            mlir::failed(validateEdge(branch.getFalseDest(),
                                      branch.getFalseDestOperands()))) {
          execute.emitError()
              << "native CFG has an invalid cf.cond_br edge ABI";
          return mlir::failure();
        }
      }
    }

    mlir::OpBuilder builder(execute);
    mlir::Location loc = execute.getLoc();
    mlir::Value zero =
        mlir::arith::ConstantIndexOp::create(builder, loc, 0);
    mlir::Value one =
        mlir::arith::ConstantIndexOp::create(builder, loc, 1);
    mlir::Value upper = mlir::arith::ConstantIndexOp::create(
        builder, loc, maxSteps.getInt());
    mlir::Value initialDone = mlir::arith::ConstantIntOp::create(
        builder, loc, 0, 1);
    llvm::SmallVector<mlir::Value> initialState{zero, initialDone};
    for (mlir::Type type : slotTypes) {
      auto sentinel = buildCFGStateSentinel(builder, loc, type);
      if (mlir::failed(sentinel)) {
        execute.emitError()
            << "native CFG block state requires scalar or ranked-tensor "
               "integer/float types";
        return mlir::failure();
      }
      initialState.push_back(*sentinel);
    }
    for (mlir::Type type : execute.getResultTypes()) {
      auto sentinel = buildCFGStateSentinel(builder, loc, type);
      if (mlir::failed(sentinel)) {
        execute.emitError()
            << "native CFG results require scalar or ranked-tensor "
               "integer/float types";
        return mlir::failure();
      }
      initialState.push_back(*sentinel);
    }
    llvm::SmallVector<mlir::Type> stateTypes;
    for (mlir::Value value : initialState)
      stateTypes.push_back(value.getType());
    const unsigned slotBase = 2;
    const unsigned resultBase = slotBase + slotTypes.size();

    auto stateMachine = mlir::scf::ForOp::create(
        builder, loc, zero, upper, one, initialState);
    for (mlir::NamedAttribute attr : execute->getAttrs())
      if (attr.getName().getValue().starts_with("tessera."))
        stateMachine->setAttr(attr.getName(), attr.getValue());
    stateMachine->setAttr("tessera.autodiff.native_multiblock_structurized",
                          builder.getBoolAttr(true));
    stateMachine->setAttr("tessera.structured_cfg.execution",
                          builder.getStringAttr("bounded_state_machine_v1"));

    using StateBuilder = std::function<mlir::FailureOr<
        llvm::SmallVector<mlir::Value>>(mlir::Block *, mlir::ValueRange)>;
    StateBuilder executeBlock;
    executeBlock = [&](mlir::Block *block, mlir::ValueRange state)
        -> mlir::FailureOr<llvm::SmallVector<mlir::Value>> {
      mlir::IRMapping mapping;
      for (mlir::BlockArgument argument : block->getArguments())
        mapping.map(argument, state[slotBase + argumentSlots.lookup(argument)]);
      for (mlir::Operation &nested : block->without_terminator())
        builder.clone(nested, mapping);

      auto edgeState = [&](mlir::Block *target,
                           mlir::ValueRange operands) {
        llvm::SmallVector<mlir::Value> next(state.begin(), state.end());
        next[0] = mlir::arith::ConstantIndexOp::create(
            builder, loc, blockOrdinals.lookup(target));
        next[1] = mlir::arith::ConstantIntOp::create(
            builder, loc, 0, 1);
        for (auto [argument, value] :
             llvm::zip_equal(target->getArguments(), operands))
          next[slotBase + argumentSlots.lookup(argument)] =
              mapping.lookupOrDefault(value);
        return next;
      };
      if (auto branch =
              mlir::dyn_cast<mlir::cf::BranchOp>(block->getTerminator()))
        return edgeState(branch.getDest(), branch.getDestOperands());
      if (auto branch = mlir::dyn_cast<mlir::cf::CondBranchOp>(
              block->getTerminator())) {
        auto select = mlir::scf::IfOp::create(
            builder, loc, stateTypes,
            mapping.lookupOrDefault(branch.getCondition()), true);
        {
          mlir::OpBuilder::InsertionGuard guard(builder);
          builder.setInsertionPointToStart(select.thenBlock());
          llvm::SmallVector<mlir::Value> next = edgeState(
              branch.getTrueDest(), branch.getTrueDestOperands());
          mlir::scf::YieldOp::create(builder, loc, next);
        }
        {
          mlir::OpBuilder::InsertionGuard guard(builder);
          builder.setInsertionPointToStart(select.elseBlock());
          llvm::SmallVector<mlir::Value> next = edgeState(
              branch.getFalseDest(), branch.getFalseDestOperands());
          mlir::scf::YieldOp::create(builder, loc, next);
        }
        return llvm::SmallVector<mlir::Value>(select.getResults().begin(),
                                               select.getResults().end());
      }
      auto yield =
          mlir::cast<mlir::scf::YieldOp>(block->getTerminator());
      llvm::SmallVector<mlir::Value> next(state.begin(), state.end());
      next[1] = mlir::arith::ConstantIntOp::create(
          builder, loc, 1, 1);
      for (auto [ordinal, value] : llvm::enumerate(yield.getOperands()))
        next[resultBase + ordinal] = mapping.lookupOrDefault(value);
      return next;
    };

    {
      mlir::OpBuilder::InsertionGuard loopGuard(builder);
      builder.setInsertionPointToStart(stateMachine.getBody());
      mlir::Value notDone = mlir::arith::CmpIOp::create(
          builder, loc, mlir::arith::CmpIPredicate::eq,
          stateMachine.getRegionIterArg(1), initialDone);
      auto active = mlir::scf::IfOp::create(
          builder, loc, stateTypes, notDone, true);
      {
        mlir::OpBuilder::InsertionGuard activeGuard(builder);
        builder.setInsertionPointToStart(active.thenBlock());
        mlir::ValueRange state = stateMachine.getRegionIterArgs();
        std::function<mlir::FailureOr<llvm::SmallVector<mlir::Value>>(
            unsigned)> dispatch;
        dispatch = [&](unsigned ordinal)
            -> mlir::FailureOr<llvm::SmallVector<mlir::Value>> {
          mlir::Value ordinalValue = mlir::arith::ConstantIndexOp::create(
              builder, loc, ordinal);
          mlir::Value selected = mlir::arith::CmpIOp::create(
              builder, loc, mlir::arith::CmpIPredicate::eq, state.front(),
              ordinalValue);
          auto select = mlir::scf::IfOp::create(builder, loc, stateTypes,
                                                selected, true);
          {
            mlir::OpBuilder::InsertionGuard selectGuard(builder);
            builder.setInsertionPointToStart(select.thenBlock());
            auto next = executeBlock(blocks[ordinal], state);
            if (mlir::failed(next))
              return mlir::failure();
            mlir::scf::YieldOp::create(builder, loc, *next);
          }
          {
            mlir::OpBuilder::InsertionGuard selectGuard(builder);
            builder.setInsertionPointToStart(select.elseBlock());
            llvm::SmallVector<mlir::Value> next;
            if (ordinal + 1 < blocks.size()) {
              auto nested = dispatch(ordinal + 1);
              if (mlir::failed(nested))
                return mlir::failure();
              next = std::move(*nested);
            } else {
              next.assign(state.begin(), state.end());
            }
            mlir::scf::YieldOp::create(builder, loc, next);
          }
          return llvm::SmallVector<mlir::Value>(select.getResults().begin(),
                                                 select.getResults().end());
        };
        auto dispatched = dispatch(0);
        if (mlir::failed(dispatched)) {
          stateMachine.erase();
          return mlir::failure();
        }
        mlir::scf::YieldOp::create(builder, loc, *dispatched);
      }
      {
        mlir::OpBuilder::InsertionGuard activeGuard(builder);
        builder.setInsertionPointToStart(active.elseBlock());
        mlir::scf::YieldOp::create(builder, loc,
                                   stateMachine.getRegionIterArgs());
      }
      llvm::SmallVector<mlir::Value> activeResults(active.getResults().begin(),
                                                    active.getResults().end());
      if (auto loopYield = mlir::dyn_cast<mlir::scf::YieldOp>(
              stateMachine.getBody()->getTerminator())) {
        loopYield.getResultsMutable().assign(activeResults);
      } else {
        builder.setInsertionPointToEnd(stateMachine.getBody());
        mlir::scf::YieldOp::create(builder, loc, activeResults);
      }
    }

    builder.setInsertionPointAfter(stateMachine);
    mlir::cf::AssertOp::create(
        builder, loc, stateMachine.getResult(1),
        builder.getStringAttr("bounded native CFG exhausted max_steps"));
    llvm::SmallVector<mlir::Value> results;
    for (unsigned ordinal = 0; ordinal < execute.getNumResults(); ++ordinal)
      results.push_back(stateMachine.getResult(resultBase + ordinal));
    execute->replaceAllUsesWith(results);
    execute.erase();
  }
  return mlir::success();
}

// A selected SAVE/HYBRID scf.if exposes every differentiable branch-local SSA
// value as a typed result.  The untaken branch yields inert zero placeholders;
// the predicate residual guarantees its placeholders are never consumed by the
// pullback. Dynamic residuals use initialized zero-extent sentinels in the
// untaken branch; their runtime shape is deliberately not fabricated from an
// unrelated branch value.
static mlir::LogicalResult materializeIfResiduals(
    mlir::func::FuncOp function) {
  llvm::SmallVector<mlir::scf::IfOp> branches;
  function.getBody().walk([&](mlir::scf::IfOp ifOp) {
    auto policy = ifOp->getAttrOfType<mlir::StringAttr>(
        "tessera.autodiff.checkpoint_policy");
    if (policy && (policy.getValue() == "save" ||
                   policy.getValue() == "hybrid") &&
        !ifOp->hasAttr("tessera.autodiff.residual_materialized"))
      branches.push_back(ifOp);
  });
  for (mlir::scf::IfOp ifOp : branches) {
    if (ifOp.getElseRegion().empty() || !ifOp.getThenRegion().hasOneBlock() ||
        !ifOp.getElseRegion().hasOneBlock()) {
      ifOp.emitError() << "saved scf.if requires single-block then/else regions";
      return mlir::failure();
    }
    llvm::SmallVector<mlir::Value> thenValues =
        collectBranchResidualCandidates(ifOp.getThenRegion());
    llvm::SmallVector<mlir::Value> elseValues =
        collectBranchResidualCandidates(ifOp.getElseRegion());
    if (thenValues.empty() && elseValues.empty()) {
      ifOp.emitError() << "saved scf.if has no differentiable branch-local SSA";
      return mlir::failure();
    }
    llvm::SmallVector<mlir::Type> resultTypes(ifOp.getResultTypes().begin(),
                                              ifOp.getResultTypes().end());
    for (mlir::Value value : thenValues)
      resultTypes.push_back(value.getType());
    for (mlir::Value value : elseValues)
      resultTypes.push_back(value.getType());

    mlir::OpBuilder builder(ifOp);
    auto replacement = mlir::scf::IfOp::create(
        builder, ifOp.getLoc(), resultTypes, ifOp.getCondition(), true);
    replacement->setAttrs(ifOp->getAttrs());
    auto cloneBranch = [&](mlir::Region &source, mlir::Block *destination,
                           llvm::ArrayRef<mlir::Value> activeValues,
                           llvm::ArrayRef<mlir::Value> inactiveValues,
                           bool activeIsThen) -> mlir::LogicalResult {
      mlir::OpBuilder::InsertionGuard guard(builder);
      builder.setInsertionPointToStart(destination);
      mlir::IRMapping mapping;
      for (mlir::Operation &nested : source.front().without_terminator())
        builder.clone(nested, mapping);
      auto oldYield = mlir::cast<mlir::scf::YieldOp>(
          source.front().getTerminator());
      llvm::SmallVector<mlir::Value> yields;
      for (mlir::Value value : oldYield.getOperands())
        yields.push_back(mapping.lookupOrDefault(value));
      auto appendActive = [&]() {
        for (mlir::Value value : activeValues)
          yields.push_back(mapping.lookupOrDefault(value));
      };
      auto appendInactive = [&]() -> mlir::LogicalResult {
        for (mlir::Value value : inactiveValues) {
          auto zero =
              buildInactiveResidualZero(builder, ifOp.getLoc(), value.getType());
          if (mlir::failed(zero)) {
            ifOp.emitError()
                << "saved scf.if requires ranked-tensor or scalar branch "
                   "residuals";
            return mlir::failure();
          }
          yields.push_back(*zero);
        }
        return mlir::success();
      };
      if (activeIsThen) {
        appendActive();
        if (mlir::failed(appendInactive()))
          return mlir::failure();
      } else {
        if (mlir::failed(appendInactive()))
          return mlir::failure();
        appendActive();
      }
      mlir::scf::YieldOp::create(builder, ifOp.getLoc(), yields);
      return mlir::success();
    };
    if (mlir::failed(cloneBranch(ifOp.getThenRegion(), replacement.thenBlock(),
                                 thenValues, elseValues, true)) ||
        mlir::failed(cloneBranch(ifOp.getElseRegion(), replacement.elseBlock(),
                                 elseValues, thenValues, false))) {
      replacement.erase();
      return mlir::failure();
    }
    unsigned primalCount = ifOp.getNumResults();
    for (auto [oldResult, newResult] : llvm::zip_equal(
             ifOp.getResults(), replacement.getResults().take_front(primalCount)))
      oldResult.replaceAllUsesWith(newResult);
    llvm::SmallVector<int64_t> residualIndices;
    for (unsigned index = primalCount; index < replacement.getNumResults();
         ++index)
      residualIndices.push_back(index);
    replacement->setAttr("tessera.autodiff.residual_materialized",
                         builder.getBoolAttr(true));
    replacement->setAttr("tessera.autodiff.residual_owner",
                         builder.getStringAttr("scf_if"));
    replacement->setAttr("tessera.autodiff.residual_result_indices",
                         builder.getDenseI64ArrayAttr(residualIndices));
    replacement->setAttr("tessera.autodiff.then_saved_count",
                         builder.getI64IntegerAttr(thenValues.size()));
    replacement->setAttr("tessera.autodiff.else_saved_count",
                         builder.getI64IntegerAttr(elseValues.size()));
    ifOp.erase();
  }
  return mlir::success();
}

// SAVE for the canonical bounded scf.while records every differentiable
// predecessor state at its executed ordinal. HYBRID records only the declared
// interior states; reverse mode selects the nearest predecessor and performs a
// bounded replay. Neither policy silently falls back to whole-prefix replay.
static mlir::LogicalResult materializeWhileResiduals(
    mlir::func::FuncOp function) {
  llvm::SmallVector<mlir::scf::WhileOp> loops;
  function.getBody().walk([&](mlir::scf::WhileOp whileOp) {
    auto policy = whileOp->getAttrOfType<mlir::StringAttr>(
        "tessera.autodiff.checkpoint_policy");
    if (policy && (policy.getValue() == "save" ||
                   policy.getValue() == "hybrid") &&
        !whileOp->hasAttr("tessera.autodiff.residual_materialized"))
      loops.push_back(whileOp);
  });
  for (mlir::scf::WhileOp whileOp : loops) {
    auto policy = whileOp->getAttrOfType<mlir::StringAttr>(
        "tessera.autodiff.checkpoint_policy");
    const bool hybrid = policy.getValue() == "hybrid";
    auto maxIters = whileOp->getAttrOfType<mlir::IntegerAttr>(
        "tessera.autodiff.max_iters");
    if (!maxIters || maxIters.getInt() <= 0 ||
        !whileOp.getBefore().hasOneBlock() ||
        !whileOp.getAfter().hasOneBlock() || whileOp.getInits().empty() ||
        !llvm::isa<mlir::IndexType>(whileOp.getInits().front().getType())) {
      whileOp.emitError()
          << "saved scf.while requires max_iters and canonical counted state";
      return mlir::failure();
    }
    auto oldCondition = mlir::dyn_cast<mlir::scf::ConditionOp>(
        whileOp.getBefore().front().getTerminator());
    auto oldYield = mlir::dyn_cast<mlir::scf::YieldOp>(
        whileOp.getAfter().front().getTerminator());
    const int64_t primalBodyOpCount = static_cast<int64_t>(
        std::distance(whileOp.getAfter().front().begin(),
                      whileOp.getAfter().front().getTerminator()->getIterator()));
    if (!oldCondition || !oldYield ||
        oldCondition.getArgs().size() != whileOp.getInits().size() ||
        oldYield.getNumOperands() != whileOp.getInits().size()) {
      whileOp.emitError() << "saved scf.while has non-canonical region ABI";
      return mlir::failure();
    }

    auto checkpoints = whileOp->getAttrOfType<mlir::DenseI64ArrayAttr>(
        "tessera.autodiff.checkpoint_indices");
    llvm::ArrayRef<int64_t> checkpointIndices;
    if (hybrid) {
      if (!checkpoints || checkpoints.empty()) {
        whileOp.emitError()
            << "HYBRID scf.while requires explicit checkpoint_indices";
        return mlir::failure();
      }
      checkpointIndices = checkpoints.asArrayRef();
      if (!llvm::is_sorted(checkpointIndices) ||
          std::adjacent_find(checkpointIndices.begin(),
                             checkpointIndices.end()) !=
              checkpointIndices.end() ||
          llvm::any_of(checkpointIndices, [&](int64_t checkpoint) {
            return checkpoint <= 0 || checkpoint >= maxIters.getInt();
          })) {
        whileOp.emitError()
            << "HYBRID scf.while checkpoints must be sorted, unique, and "
               "inside (0, max_iters)";
        return mlir::failure();
      }
    }
    const int64_t tapeSlots =
        hybrid ? static_cast<int64_t>(checkpointIndices.size())
               : maxIters.getInt();

    llvm::SmallVector<unsigned> stateIndices;
    llvm::SmallVector<mlir::RankedTensorType> tapeTypes;
    llvm::SmallVector<mlir::Value> tapeInits;
    mlir::OpBuilder builder(whileOp);
    for (auto [index, init] : llvm::enumerate(whileOp.getInits())) {
      if (!llvm::isa<mlir::FloatType>(
              mlir::getElementTypeOrSelf(init.getType())))
        continue;
      auto tapeType = getResidualTapeType(init.getType(), tapeSlots);
      if (mlir::failed(tapeType)) {
        whileOp.emitError()
            << "saved scf.while requires ranked-tensor differentiable state";
        return mlir::failure();
      }
      auto stateType = mlir::cast<mlir::RankedTensorType>(init.getType());
      stateIndices.push_back(index);
      tapeTypes.push_back(*tapeType);
      tapeInits.push_back(mlir::tensor::EmptyOp::create(
          builder, whileOp.getLoc(), tapeType->getShape(),
          tapeType->getElementType(),
          getDynamicTensorSizes(builder, whileOp.getLoc(), init, stateType)));
    }
    if (stateIndices.empty() ||
        stateIndices.size() + 1 != whileOp.getInits().size()) {
      whileOp.emitError()
          << "saved scf.while requires every non-counter state to be a "
             "differentiable ranked tensor";
      return mlir::failure();
    }

    llvm::SmallVector<mlir::Value> inits(whileOp.getInits().begin(),
                                         whileOp.getInits().end());
    llvm::append_range(inits, tapeInits);
    llvm::SmallVector<mlir::Type> stateTypes;
    for (mlir::Value init : inits)
      stateTypes.push_back(init.getType());
    auto replacement = mlir::scf::WhileOp::create(
        builder, whileOp.getLoc(), stateTypes, inits);
    replacement->setAttrs(whileOp->getAttrs());

    llvm::SmallVector<mlir::Location> locations(stateTypes.size(),
                                                whileOp.getLoc());
    {
      mlir::OpBuilder::InsertionGuard guard(builder);
      mlir::Block *before = builder.createBlock(&replacement.getBefore());
      before->addArguments(stateTypes, locations);
      builder.setInsertionPointToStart(before);
      mlir::IRMapping mapping;
      for (auto [source, destination] : llvm::zip_equal(
               whileOp.getBefore().front().getArguments(),
               before->getArguments().take_front(whileOp.getInits().size())))
        mapping.map(source, destination);
      for (mlir::Operation &nested :
           whileOp.getBefore().front().without_terminator())
        builder.clone(nested, mapping);
      llvm::SmallVector<mlir::Value> forwarded;
      for (mlir::Value value : oldCondition.getArgs())
        forwarded.push_back(mapping.lookupOrDefault(value));
      llvm::append_range(
          forwarded,
          before->getArguments().drop_front(whileOp.getInits().size()));
      mlir::scf::ConditionOp::create(
          builder, whileOp.getLoc(),
          mapping.lookupOrDefault(oldCondition.getCondition()), forwarded);
    }
    {
      mlir::OpBuilder::InsertionGuard guard(builder);
      mlir::Block *after = builder.createBlock(&replacement.getAfter());
      after->addArguments(stateTypes, locations);
      builder.setInsertionPointToStart(after);
      mlir::IRMapping mapping;
      for (auto [source, destination] : llvm::zip_equal(
               whileOp.getAfter().front().getArguments(),
               after->getArguments().take_front(whileOp.getInits().size())))
        mapping.map(source, destination);
      for (mlir::Operation &nested :
           whileOp.getAfter().front().without_terminator())
        builder.clone(nested, mapping);
      llvm::SmallVector<mlir::Value> yields;
      for (mlir::Value value : oldYield.getOperands())
        yields.push_back(mapping.lookupOrDefault(value));
      mlir::Value ordinal = after->getArgument(0);
      for (auto [tapeOrdinal, stateIndex] : llvm::enumerate(stateIndices)) {
        mlir::Value predecessor = after->getArgument(stateIndex);
        mlir::Value successor = yields[stateIndex];
        mlir::Value tape = after->getArgument(whileOp.getInits().size() +
                                              tapeOrdinal);
        auto stateType =
            mlir::cast<mlir::RankedTensorType>(predecessor.getType());
        auto insertState = [&](mlir::Value source, mlir::Value destination,
                               mlir::OpFoldResult slot) {
          llvm::SmallVector<mlir::OpFoldResult> offsets{slot};
          llvm::SmallVector<mlir::OpFoldResult> sizes{
              builder.getIndexAttr(1)};
          llvm::SmallVector<mlir::OpFoldResult> strides(
              stateType.getRank() + 1, builder.getIndexAttr(1));
          for (auto [dimIndex, dim] : llvm::enumerate(stateType.getShape())) {
            offsets.push_back(builder.getIndexAttr(0));
            sizes.push_back(mlir::ShapedType::isDynamic(dim)
                                ? mlir::OpFoldResult(
                                      mlir::tensor::DimOp::create(
                                          builder, whileOp.getLoc(), source,
                                          dimIndex)
                                          .getResult())
                                : mlir::OpFoldResult(
                                      builder.getIndexAttr(dim)));
          }
          return mlir::tensor::InsertSliceOp::create(
              builder, whileOp.getLoc(), source, destination, offsets, sizes,
              strides);
        };
        if (!hybrid) {
          yields.push_back(insertState(predecessor, tape, ordinal));
          continue;
        }
        mlir::Value nextOrdinal = yields.front();
        for (auto [slot, checkpoint] :
             llvm::enumerate(checkpointIndices)) {
          mlir::Value checkpointValue = mlir::arith::ConstantIndexOp::create(
              builder, whileOp.getLoc(), checkpoint);
          mlir::Value retain = mlir::arith::CmpIOp::create(
              builder, whileOp.getLoc(), mlir::arith::CmpIPredicate::eq,
              nextOrdinal, checkpointValue);
          auto retainIf = mlir::scf::IfOp::create(
              builder, whileOp.getLoc(), mlir::TypeRange{tape.getType()},
              retain, true);
          {
            mlir::OpBuilder::InsertionGuard retainGuard(builder);
            builder.setInsertionPointToStart(retainIf.thenBlock());
            mlir::Value retained = insertState(
                successor, tape,
                builder.getIndexAttr(static_cast<int64_t>(slot)));
            mlir::scf::YieldOp::create(builder, whileOp.getLoc(), retained);
          }
          {
            mlir::OpBuilder::InsertionGuard retainGuard(builder);
            builder.setInsertionPointToStart(retainIf.elseBlock());
            mlir::scf::YieldOp::create(builder, whileOp.getLoc(), tape);
          }
          tape = retainIf.getResult(0);
        }
        yields.push_back(tape);
      }
      mlir::scf::YieldOp::create(builder, whileOp.getLoc(), yields);
    }

    unsigned primalCount = whileOp.getNumResults();
    for (auto [oldResult, newResult] : llvm::zip_equal(
             whileOp.getResults(),
             replacement.getResults().take_front(primalCount)))
      oldResult.replaceAllUsesWith(newResult);
    llvm::SmallVector<int64_t> resultIndices;
    for (unsigned index = primalCount; index < replacement.getNumResults();
         ++index)
      resultIndices.push_back(index);
    llvm::SmallVector<int64_t> primalIndices(stateIndices.begin(),
                                             stateIndices.end());
    replacement->setAttr("tessera.autodiff.residual_materialized",
                         builder.getBoolAttr(true));
    replacement->setAttr("tessera.autodiff.residual_owner",
                         builder.getStringAttr("scf_while"));
    replacement->setAttr("tessera.autodiff.residual_result_indices",
                         builder.getDenseI64ArrayAttr(resultIndices));
    replacement->setAttr("tessera.autodiff.residual_primal_iter_arg_indices",
                         builder.getDenseI64ArrayAttr(primalIndices));
    replacement->setAttr("tessera.autodiff.residual_primal_op_count",
                         builder.getI64IntegerAttr(primalBodyOpCount));
    whileOp.erase();
  }
  return mlir::success();
}

// Materialize the generic counted-loop residual contract before activity
// analysis. Every replay-relevant scalar or ranked-tensor slot receives its
// own typed tape. This includes program counters, completion bits, and integer
// counters: they do not acquire analytical cotangents, but HYBRID suffix replay
// must restart from their checkpoint values rather than from final state. The
// explicit state-index map keeps mixed state lossless and makes unknown storage
// fail closed.
static mlir::LogicalResult materializeGenericForResiduals(
    mlir::func::FuncOp function) {
  llvm::SmallVector<mlir::scf::ForOp> loops;
  function.getBody().walk([&](mlir::scf::ForOp loop) {
    auto policy = loop->getAttrOfType<mlir::StringAttr>(
        "tessera.autodiff.checkpoint_policy");
    if (policy && (policy.getValue() == "save" ||
                   policy.getValue() == "hybrid") &&
        !loop->hasAttr("tessera.autodiff.residual_materialized"))
      loops.push_back(loop);
  });
  for (mlir::scf::ForOp loop : loops) {
    llvm::APInt lbValue, ubValue, stepValue;
    if (!mlir::matchPattern(loop.getLowerBound(),
                            mlir::m_ConstantInt(&lbValue)) ||
        !mlir::matchPattern(loop.getUpperBound(),
                            mlir::m_ConstantInt(&ubValue)) ||
        !mlir::matchPattern(loop.getStep(),
                            mlir::m_ConstantInt(&stepValue)) ||
        !stepValue.isStrictlyPositive() ||
        ubValue.getSExtValue() <= lbValue.getSExtValue()) {
      loop.emitError() << "generic saved scf.for requires positive static "
                          "bounds and step";
      return mlir::failure();
    }
    int64_t span = ubValue.getSExtValue() - lbValue.getSExtValue();
    int64_t step = stepValue.getSExtValue();
    int64_t trip = (span + step - 1) / step;
    auto checkpoints = loop->getAttrOfType<mlir::DenseI64ArrayAttr>(
        "tessera.autodiff.checkpoint_indices");
    if (!checkpoints || checkpoints.empty()) {
      loop.emitError() << "generic saved scf.for requires checkpoint_indices";
      return mlir::failure();
    }
    llvm::ArrayRef<int64_t> indices = checkpoints.asArrayRef();
    if (!llvm::is_sorted(indices) ||
        std::adjacent_find(indices.begin(), indices.end()) != indices.end() ||
        llvm::any_of(indices,
                     [trip](int64_t index) { return index <= 0 || index >= trip; })) {
      loop.emitError() << "generic scf.for checkpoints must be sorted, unique, "
                          "interior ordinals";
      return mlir::failure();
    }
    auto policy = loop->getAttrOfType<mlir::StringAttr>(
        "tessera.autodiff.checkpoint_policy");
    if ((policy.getValue() == "save" &&
         indices.size() != static_cast<size_t>(trip - 1)) ||
        (policy.getValue() == "hybrid" &&
         indices.size() >= static_cast<size_t>(trip - 1))) {
      loop.emitError() << "generic scf.for checkpoint cardinality disagrees "
                          "with its policy";
      return mlir::failure();
    }

    llvm::SmallVector<unsigned> tapedStateIndices;
    llvm::SmallVector<mlir::RankedTensorType> tapeTypes;
    llvm::SmallVector<mlir::Value> tapeInits;
    llvm::SmallVector<int64_t> shapeTapeOrdinals;
    llvm::SmallVector<mlir::RankedTensorType> shapeTapeTypes;
    llvm::SmallVector<mlir::Value> shapeTapeInits;
    mlir::OpBuilder builder(loop);
    llvm::SmallVector<mlir::Type> primalStateTypes;
    for (mlir::Value init : loop.getInitArgs())
      primalStateTypes.push_back(init.getType());
    auto shapeEnvelopes = readSavedSlotShapeEnvelopes(
        loop, mlir::TypeRange(primalStateTypes), /*requireEveryDynamic=*/true);
    if (mlir::failed(shapeEnvelopes)) {
      loop.emitError()
          << "generic saved scf.for dynamic state requires total, positive "
             "per-slot shape-envelope indices/ranks/bounds";
      return mlir::failure();
    }
    for (auto [stateIndex, init] : llvm::enumerate(loop.getInitArgs())) {
      auto tapeType = getResidualTapeType(init.getType(), indices.size());
      if (mlir::failed(tapeType)) {
        loop.emitError() << "generic saved scf.for cannot materialize state "
                         << stateIndex << " with type " << init.getType();
        return mlir::failure();
      }
      tapedStateIndices.push_back(stateIndex);
      auto stateType =
          mlir::dyn_cast<mlir::RankedTensorType>(init.getType());
      auto envelope = shapeEnvelopes->find(stateIndex);
      if (stateType && !stateType.hasStaticShape()) {
        if (envelope == shapeEnvelopes->end()) {
          loop.emitError() << "dynamic saved state " << stateIndex
                           << " has no shape envelope";
          return mlir::failure();
        }
        llvm::SmallVector<int64_t> boundedShape{
            static_cast<int64_t>(indices.size())};
        llvm::append_range(boundedShape, envelope->second);
        tapeType = mlir::RankedTensorType::get(
            boundedShape, stateType.getElementType(), stateType.getEncoding());
        shapeTapeOrdinals.push_back(shapeTapeTypes.size());
        shapeTapeTypes.push_back(mlir::RankedTensorType::get(
            {static_cast<int64_t>(indices.size()), stateType.getRank()},
            builder.getIndexType()));
        shapeTapeInits.push_back(mlir::tensor::EmptyOp::create(
            builder, loop.getLoc(), shapeTapeTypes.back().getShape(),
            builder.getIndexType()));
      } else {
        shapeTapeOrdinals.push_back(-1);
      }
      tapeTypes.push_back(*tapeType);
      llvm::SmallVector<mlir::Value> dynamicSizes;
      if (stateType && stateType.hasStaticShape())
        dynamicSizes =
            getDynamicTensorSizes(builder, loop.getLoc(), init, stateType);
      tapeInits.push_back(mlir::tensor::EmptyOp::create(
          builder, loop.getLoc(), tapeType->getShape(),
          tapeType->getElementType(), dynamicSizes));
    }
    llvm::SmallVector<mlir::Value> inits(loop.getInitArgs().begin(),
                                         loop.getInitArgs().end());
    llvm::append_range(inits, tapeInits);
    llvm::append_range(inits, shapeTapeInits);
    auto replacement = mlir::scf::ForOp::create(
        builder, loop.getLoc(), loop.getLowerBound(), loop.getUpperBound(),
        loop.getStep(), inits);
    replacement->setAttrs(loop->getAttrs());
    mlir::Block &oldBody = loop.getRegion().front();
    mlir::Block &newBody = replacement.getRegion().front();
    mlir::IRMapping mapping;
    mapping.map(loop.getInductionVar(), replacement.getInductionVar());
    for (auto [source, destination] : llvm::zip_equal(
             loop.getRegionIterArgs().take_front(loop.getNumResults()),
             replacement.getRegionIterArgs().take_front(loop.getNumResults())))
      mapping.map(source, destination);
    builder.setInsertionPointToStart(&newBody);
    for (mlir::Operation &source : oldBody.without_terminator())
      builder.clone(source, mapping);
    auto oldYield = mlir::cast<mlir::scf::YieldOp>(oldBody.getTerminator());
    llvm::SmallVector<mlir::Value> yields;
    for (mlir::Value value : oldYield.getOperands())
      yields.push_back(mapping.lookupOrDefault(value));
    llvm::SmallVector<mlir::Value> shapeTapeValues;
    for (unsigned shapeOrdinal = 0; shapeOrdinal < shapeTapeTypes.size();
         ++shapeOrdinal)
      shapeTapeValues.push_back(replacement.getRegionIterArg(
          loop.getNumResults() + tapeTypes.size() + shapeOrdinal));
    for (auto [tapeOrdinal, tapeType] : llvm::enumerate(tapeTypes)) {
      unsigned stateIndex = tapedStateIndices[tapeOrdinal];
      mlir::Value tape = replacement.getRegionIterArg(
          loop.getNumResults() + tapeOrdinal);
      auto stateType = mlir::dyn_cast<mlir::RankedTensorType>(
          loop.getInitArgs()[stateIndex].getType());
      int64_t shapeOrdinal = shapeTapeOrdinals[tapeOrdinal];
      mlir::Value shapeTape =
          shapeOrdinal >= 0 ? shapeTapeValues[shapeOrdinal] : mlir::Value{};
      for (auto [slot, checkpoint] : llvm::enumerate(indices)) {
        mlir::Value retainIv = mlir::arith::ConstantIndexOp::create(
            builder, loop.getLoc(),
            lbValue.getSExtValue() + (checkpoint - 1) * step);
        mlir::Value retain = mlir::arith::CmpIOp::create(
            builder, loop.getLoc(), mlir::arith::CmpIPredicate::eq,
            replacement.getInductionVar(), retainIv);
        llvm::SmallVector<mlir::Type> retainedTypes{tapeType};
        if (shapeTape)
          retainedTypes.push_back(shapeTape.getType());
        auto retainIf = mlir::scf::IfOp::create(
            builder, loop.getLoc(), retainedTypes, retain, true);
        {
          mlir::OpBuilder::InsertionGuard guard(builder);
          builder.setInsertionPointToStart(retainIf.thenBlock());
          mlir::Value retained;
          if (stateType) {
            llvm::SmallVector<mlir::OpFoldResult> offsets{
                builder.getIndexAttr(static_cast<int64_t>(slot))};
            llvm::SmallVector<mlir::OpFoldResult> sizes{builder.getIndexAttr(1)};
            llvm::SmallVector<mlir::OpFoldResult> strides(
                stateType.getRank() + 1, builder.getIndexAttr(1));
            for (auto [dimIndex, dim] :
                 llvm::enumerate(stateType.getShape())) {
              offsets.push_back(builder.getIndexAttr(0));
              if (mlir::ShapedType::isDynamic(dim)) {
                mlir::Value actual = mlir::tensor::DimOp::create(
                    builder, loop.getLoc(), yields[stateIndex], dimIndex);
                int64_t bound =
                    shapeEnvelopes->lookup(stateIndex)[dimIndex];
                mlir::Value boundValue =
                    mlir::arith::ConstantIndexOp::create(
                        builder, loop.getLoc(), bound);
                mlir::Value withinEnvelope = mlir::arith::CmpIOp::create(
                    builder, loop.getLoc(),
                    mlir::arith::CmpIPredicate::ule, actual, boundValue);
                auto assertion = mlir::cf::AssertOp::create(
                    builder, loop.getLoc(), withinEnvelope,
                    builder.getStringAttr(
                        "saved dynamic state exceeds its slot envelope"));
                assertion->setAttr("tessera.autodiff.replay_safe_guard",
                                   builder.getBoolAttr(true));
                sizes.push_back(mlir::OpFoldResult(actual));
              } else {
                sizes.push_back(mlir::OpFoldResult(
                    builder.getIndexAttr(dim)));
              }
            }
            retained = mlir::tensor::InsertSliceOp::create(
                builder, loop.getLoc(), yields[stateIndex], tape, offsets,
                sizes, strides);
          } else {
            mlir::Value slotIndex = mlir::arith::ConstantIndexOp::create(
                builder, loop.getLoc(), static_cast<int64_t>(slot));
            retained = mlir::tensor::InsertOp::create(
                builder, loop.getLoc(), yields[stateIndex], tape, slotIndex);
          }
          llvm::SmallVector<mlir::Value> retainedValues{retained};
          if (shapeTape) {
            mlir::Value retainedShape = shapeTape;
            mlir::Value slotValue = mlir::arith::ConstantIndexOp::create(
                builder, loop.getLoc(), static_cast<int64_t>(slot));
            for (int64_t dimIndex = 0; dimIndex < stateType.getRank();
                 ++dimIndex) {
              mlir::Value dimValue = mlir::tensor::DimOp::create(
                  builder, loop.getLoc(), yields[stateIndex], dimIndex);
              mlir::Value dimOrdinal = mlir::arith::ConstantIndexOp::create(
                  builder, loop.getLoc(), dimIndex);
              retainedShape = mlir::tensor::InsertOp::create(
                  builder, loop.getLoc(), dimValue, retainedShape,
                  mlir::ValueRange{slotValue, dimOrdinal});
            }
            retainedValues.push_back(retainedShape);
          }
          mlir::scf::YieldOp::create(builder, loop.getLoc(), retainedValues);
        }
        {
          mlir::OpBuilder::InsertionGuard guard(builder);
          builder.setInsertionPointToStart(retainIf.elseBlock());
          llvm::SmallVector<mlir::Value> retainedValues{tape};
          if (shapeTape)
            retainedValues.push_back(shapeTape);
          mlir::scf::YieldOp::create(builder, loop.getLoc(), retainedValues);
        }
        tape = retainIf.getResult(0);
        if (shapeTape)
          shapeTape = retainIf.getResult(1);
      }
      yields.push_back(tape);
      if (shapeOrdinal >= 0)
        shapeTapeValues[shapeOrdinal] = shapeTape;
    }
    llvm::append_range(yields, shapeTapeValues);
    mlir::scf::YieldOp::create(builder, loop.getLoc(), yields);
    unsigned primalCount = loop.getNumResults();
    for (auto [oldResult, newResult] : llvm::zip_equal(
             loop.getResults(), replacement.getResults().take_front(primalCount)))
      oldResult.replaceAllUsesWith(newResult);
    llvm::SmallVector<int64_t> resultIndices, primalIndices;
    for (auto [tapeOrdinal, stateIndex] :
         llvm::enumerate(tapedStateIndices)) {
      resultIndices.push_back(primalCount + tapeOrdinal);
      primalIndices.push_back(stateIndex);
    }
    for (unsigned shapeOrdinal = 0; shapeOrdinal < shapeTapeTypes.size();
         ++shapeOrdinal)
      resultIndices.push_back(primalCount + tapeTypes.size() + shapeOrdinal);
    llvm::SmallVector<int64_t> shapeResidualOrdinals;
    for (int64_t shapeOrdinal : shapeTapeOrdinals)
      shapeResidualOrdinals.push_back(
          shapeOrdinal < 0
              ? -1
              : static_cast<int64_t>(tapeTypes.size()) + shapeOrdinal);
    replacement->setAttr("tessera.autodiff.residual_materialized",
                         builder.getBoolAttr(true));
    replacement->setAttr("tessera.autodiff.residual_owner",
                         builder.getStringAttr("generic_for"));
    replacement->setAttr("tessera.autodiff.residual_result_indices",
                         builder.getDenseI64ArrayAttr(resultIndices));
    replacement->setAttr("tessera.autodiff.residual_primal_iter_arg_indices",
                         builder.getDenseI64ArrayAttr(primalIndices));
    replacement->setAttr("tessera.autodiff.residual_shape_tape_ordinals",
                         builder.getDenseI64ArrayAttr(shapeResidualOrdinals));
    loop.erase();
  }
  return mlir::success();
}

/// Accumulate `g` into `cotan[v]` (float → addf, integer → addi). Shared shape
/// with AutodiffPass.cpp; kept local so the two passes stay independent.
void accumulateCotangent(mlir::OpBuilder &builder, CotangentMap &cotan,
                         mlir::Value v, mlir::Value g) {
  if (!g)
    return;
  auto it = cotan.find(v);
  if (it == cotan.end()) {
    cotan[v] = g;
    return;
  }
  auto loc = g.getLoc();
  mlir::Type elemTy = mlir::getElementTypeOrSelf(g.getType());
  mlir::Value sum =
      llvm::isa<mlir::FloatType>(elemTy)
          ? builder.create<mlir::arith::AddFOp>(loc, it->second, g).getResult()
          : builder.create<mlir::arith::AddIOp>(loc, it->second, g).getResult();
  cotan[v] = sum;
}

class AutodiffPairedPass
    : public mlir::PassWrapper<AutodiffPairedPass,
                               mlir::OperationPass<mlir::ModuleOp>> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(AutodiffPairedPass)

  llvm::StringRef getArgument() const final {
    return "tessera-autodiff-paired";
  }
  llvm::StringRef getDescription() const final {
    return "Paired forward/backward autodiff — emits @f__bwd(inputs, "
           "out_cotangents) -> input_cotangents (recompute-all residual "
           "policy). Phase 2 of AUTODIFF_UNIFICATION_PLAN.md.";
  }
  void getDependentDialects(mlir::DialectRegistry &registry) const override {
    registry.insert<mlir::arith::ArithDialect, mlir::cf::ControlFlowDialect,
                    mlir::func::FuncDialect, mlir::linalg::LinalgDialect,
                    mlir::scf::SCFDialect, mlir::tensor::TensorDialect>();
  }

  void runOnOperation() override {
    auto module = getOperation();
    llvm::SmallVector<mlir::func::FuncOp> targets;
    module.walk([&](mlir::func::FuncOp fn) {
      auto marker = fn->getAttrOfType<mlir::StringAttr>(kAutodiffMarker);
      // Skip functions we already produced (role=backward) or that aren't marked.
      if (marker && marker.getValue() == "reverse" &&
          !fn->hasAttr("tessera.autodiff.role"))
        targets.push_back(fn);
    });
    for (auto fn : targets)
      if (failed(buildBackward(fn)))
        return signalPassFailure();
  }

private:
  llvm::DenseMap<mlir::Operation *, llvm::SmallVector<mlir::Value>>
      explicitRegionResiduals;

  mlir::LogicalResult buildBackward(mlir::func::FuncOp fwd) {
    explicitRegionResiduals.clear();
    auto module = fwd->getParentOfType<mlir::ModuleOp>();
    mlir::MLIRContext *ctx = &getContext();

    if (fwd.getBody().empty()) {
      fwd.emitError() << "[AUTODIFF_PAIRED] cannot differentiate a declaration";
      return mlir::failure();
    }
    if (mlir::failed(structurizeNativeDiamonds(fwd)) ||
        mlir::failed(structurizeBoundedNativeCFGs(fwd)) ||
        mlir::failed(materializeIfResiduals(fwd)) ||
        mlir::failed(materializeWhileResiduals(fwd)) ||
        mlir::failed(materializeGenericForResiduals(fwd)))
      return mlir::failure();
    mlir::Block &fwdBlock = fwd.getBody().front();

    auto returnOp = mlir::dyn_cast<mlir::func::ReturnOp>(fwdBlock.getTerminator());
    if (!returnOp) {
      fwd.emitError() << "[AUTODIFF_PAIRED] forward has no return terminator";
      return mlir::failure();
    }
    GraphDataflowAnalysis dataflow(fwd);
    if (mlir::failed(dataflow.run())) {
      fwd.emitError()
          << "[AUTODIFF_PAIRED] Graph IR dataflow analysis failed";
      return mlir::failure();
    }
    GraphDataflowAnalysis::ActiveOpSet activeOps =
        dataflow.computeActivity(returnOp.getOperands());

    // Collect only the backward-reachable forward cone. Inactive side
    // computations and inactive nested regions are neither cloned nor rejected.
    llvm::SmallVector<mlir::Operation *> forwardOps;
    for (mlir::Operation &opRef : fwdBlock) {
      mlir::Operation *op = &opRef;
      if (mlir::isa<mlir::func::ReturnOp>(op))
        continue;
      op->setAttr("tessera.autodiff.activity",
                  mlir::StringAttr::get(ctx, activeOps.contains(op)
                                                ? "active"
                                                : "inactive"));
      if (!activeOps.contains(op))
        continue;
      if (op->getNumRegions() != 0 &&
          !RegionAdjointInterface::supports(op)) {
        op->emitError() << "[AUTODIFF_NESTED_REGION] active paired reverse-mode "
                           "path contains unsupported nested-region op ('"
                        << op->getName().getStringRef() << "')";
        return mlir::failure();
      }
      if (auto policy = op->getAttrOfType<mlir::StringAttr>(
              "tessera.autodiff.checkpoint_policy");
          policy && policy.getValue() != "recompute_all") {
        auto materialized = op->getAttrOfType<mlir::BoolAttr>(
            "tessera.autodiff.residual_materialized");
        if ((policy.getValue() != "save" && policy.getValue() != "hybrid") ||
            !materialized ||
            !materialized.getValue()) {
          op->emitError()
              << "paired reverse-mode cannot consume checkpoint policy '"
              << policy.getValue()
              << "' until its explicit residual operands/results have been "
                 "materialized";
          return mlir::failure();
        }
      }
      if (hasStochasticEffect(op)) {
        op->emitError()
            << "AUTODIFF_STOCHASTIC_EFFECT: active stochastic op "
            << op->getName()
            << " requires an explicit pathwise or score-function adjoint";
        return mlir::failure();
      }
      forwardOps.push_back(op);
    }

    // Backward signature: (forward inputs..., out_cotangents...) ->
    // (input_cotangents...). One out-cotangent per forward result; the input
    // cotangent types mirror the forward argument types.
    llvm::SmallVector<mlir::Type> fwdInTypes(fwd.getArgumentTypes().begin(),
                                             fwd.getArgumentTypes().end());
    llvm::SmallVector<mlir::Type> fwdResTypes(
        fwd.getResultTypes().begin(), fwd.getResultTypes().end());

    llvm::SmallVector<mlir::Value> forwardResiduals;
    llvm::SmallVector<mlir::Attribute> residualSources;
    bool hasHybridResidual = false;
    llvm::DenseMap<mlir::Operation *, llvm::SmallVector<mlir::Value>>
        residualValues;
    for (mlir::Operation *op : forwardOps) {
      if (auto checkpointPolicy = op->getAttrOfType<mlir::StringAttr>(
              "tessera.autodiff.checkpoint_policy"))
        hasHybridResidual |= checkpointPolicy.getValue() == "hybrid";
      auto materialized = op->getAttrOfType<mlir::BoolAttr>(
          "tessera.autodiff.residual_materialized");
      llvm::SmallVector<mlir::Value> values;
      if (materialized && materialized.getValue()) {
        auto indices = op->getAttrOfType<mlir::DenseI64ArrayAttr>(
            "tessera.autodiff.residual_result_indices");
        if (!indices || indices.empty()) {
          op->emitError()
              << "materialized region residual has no result indices";
          return mlir::failure();
        }
        auto owner = op->getAttrOfType<mlir::StringAttr>(
            "tessera.autodiff.residual_owner");
        if (owner && owner.getValue() == "scf_if") {
          auto ifOp = mlir::dyn_cast<mlir::scf::IfOp>(op);
          if (!ifOp) {
            op->emitError() << "scf_if residual owner requires scf.if";
            return mlir::failure();
          }
          values.push_back(ifOp.getCondition());
          residualSources.push_back(
              mlir::StringAttr::get(ctx, "scf.if:predicate"));
        } else if (owner && owner.getValue() == "scf_while") {
          auto whileOp = mlir::dyn_cast<mlir::scf::WhileOp>(op);
          if (!whileOp || whileOp.getNumResults() == 0 ||
              !llvm::isa<mlir::IndexType>(whileOp.getResult(0).getType())) {
            op->emitError()
                << "scf_while residual owner requires counted scf.while";
            return mlir::failure();
          }
          values.push_back(whileOp.getResult(0));
          residualSources.push_back(
              mlir::StringAttr::get(ctx, "scf.while:trip_count"));
        }
        for (auto [residualOrdinal, index] :
             llvm::enumerate(indices.asArrayRef())) {
          if (index < 0 || index >= op->getNumResults()) {
            op->emitError() << "region residual result index " << index
                            << " is outside the operation result range";
            return mlir::failure();
          }
          values.push_back(op->getResult(index));
          residualSources.push_back(mlir::StringAttr::get(
              ctx, owner && owner.getValue() == "scf_if"
                       ? ("scf.if:branch_value:" +
                          llvm::Twine(residualOrdinal))
                             .str()
                       : owner && owner.getValue() == "scf_while"
                             ? ("scf.while:state_tape:" +
                                llvm::Twine(residualOrdinal))
                                   .str()
                       : owner ? (owner.getValue() + ":state_tape" +
                            (indices.size() == 1
                                 ? llvm::Twine()
                                 : ":" + llvm::Twine(residualOrdinal)))
                               .str()
                         : (op->getName().getStringRef() + ":result:" +
                            llvm::Twine(index))
                               .str()));
        }
      } else if (auto ifOp = mlir::dyn_cast<mlir::scf::IfOp>(op)) {
        values.push_back(ifOp.getCondition());
        residualSources.push_back(
            mlir::StringAttr::get(ctx, "scf.if:predicate"));
      } else if (auto whileOp = mlir::dyn_cast<mlir::scf::WhileOp>(op)) {
        // Canonical bounded while result zero is the exact number of executed
        // body iterations.  Expose it rather than trusting backward replay to
        // rediscover a data-dependent path.
        values.push_back(whileOp.getResult(0));
        residualSources.push_back(
            mlir::StringAttr::get(ctx, "scf.while:trip_count"));
      }
      if (!values.empty()) {
        llvm::append_range(forwardResiduals, values);
        residualValues.try_emplace(op, std::move(values));
      }
    }

    if (!forwardResiduals.empty()) {
      llvm::SmallVector<mlir::Type> publicResultTypes(fwdResTypes);
      for (mlir::Value residual : forwardResiduals)
        publicResultTypes.push_back(residual.getType());
      fwd.setType(mlir::FunctionType::get(ctx, fwdInTypes, publicResultTypes));
      returnOp->insertOperands(returnOp.getNumOperands(), forwardResiduals);
    }

    llvm::SmallVector<mlir::Type> bwdInTypes(fwdInTypes);
    for (mlir::Type rt : fwdResTypes)
      bwdInTypes.push_back(rt);
    for (mlir::Value residual : forwardResiduals)
      bwdInTypes.push_back(residual.getType());
    // Input cotangents mirror the input types (one per forward argument).
    llvm::SmallVector<mlir::Type> bwdResTypes(fwdInTypes);

    mlir::OpBuilder builder(ctx);
    llvm::StringRef pairedResidualPolicy =
        forwardResiduals.empty()
            ? "recompute_all"
            : hasHybridResidual ? "hybrid" : "save";
    builder.setInsertionPointToEnd(module.getBody());
    auto bwdName = (fwd.getName() + "__bwd").str();
    auto bwdType = builder.getFunctionType(bwdInTypes, bwdResTypes);
    auto bwd = builder.create<mlir::func::FuncOp>(fwd.getLoc(), bwdName, bwdType);
    bwd->setAttr("tessera.autodiff.role", builder.getStringAttr("backward"));
    bwd->setAttr("tessera.autodiff.forward",
                 mlir::FlatSymbolRefAttr::get(ctx, fwd.getName()));
    bwd->setAttr("tessera.autodiff.residual_policy",
                 builder.getStringAttr(pairedResidualPolicy));
    if (!residualSources.empty())
      bwd->setAttr("tessera.autodiff.residual_sources",
                   builder.getArrayAttr(residualSources));

    mlir::Block *bwdBlock = bwd.addEntryBlock();
    builder.setInsertionPointToStart(bwdBlock);

    unsigned nIn = fwd.getNumArguments();
    unsigned nRes = fwdResTypes.size();

    // Map forward SSA values into the backward function: forward argument i →
    // backward argument i (the recompute-all residual = the forward inputs).
    mlir::IRMapping map;
    for (unsigned i = 0; i < nIn; ++i)
      map.map(fwd.getArgument(i), bwdBlock->getArgument(i));

    // Recompute the forward ops inside the backward body (clones), so each
    // adjoint's `getX()` resolves to a value that lives in this function.
    llvm::SmallVector<mlir::Operation *> clones;
    unsigned residualArgument = nIn + nRes;
    for (mlir::Operation *op : forwardOps) {
      // Recompute-all can preserve a stopped primal only when its operand is
      // already a backward argument.  Recomputing an inactive producer cone
      // here would be wrong for stateful or stochastic producers; a future
      // saved-residual policy can lift this restriction explicitly.
      if (op->getName().getStringRef() == "tessera.stop_gradient" &&
          !map.contains(op->getOperand(0))) {
        op->emitError()
            << "AUTODIFF_STOP_GRADIENT_RESIDUAL_REQUIRED: paired "
               "recompute-all cannot preserve a stopped intermediate; save "
               "the stopped primal as an explicit residual";
        return mlir::failure();
      }
      mlir::Operation *clone = builder.clone(*op, map);
      clones.push_back(clone);
      auto residualIt = residualValues.find(op);
      if (residualIt != residualValues.end()) {
        llvm::SmallVector<mlir::Value> values;
        for ([[maybe_unused]] mlir::Value residual : residualIt->second)
          values.push_back(bwdBlock->getArgument(residualArgument++));
        explicitRegionResiduals.try_emplace(clone, std::move(values));
      }
    }

    // Seed cotangents: forward result j ↦ backward out-cotangent argument
    // (nIn + j). The clone of the op producing that result carries the seed.
    CotangentMap cotan;
    for (unsigned j = 0; j < nRes; ++j) {
      mlir::Value fwdRes = returnOp.getOperand(j);
      mlir::Value cloneRes = map.lookupOrNull(fwdRes);
      if (!cloneRes) {
        // A forward result that is itself an argument (identity return): route
        // the cotangent straight to that input slot.
        if (auto ba = llvm::dyn_cast<mlir::BlockArgument>(fwdRes))
          accumulateCotangent(builder, cotan, bwdBlock->getArgument(ba.getArgNumber()),
                              bwdBlock->getArgument(nIn + j));
        continue;
      }
      accumulateCotangent(builder, cotan, cloneRes, bwdBlock->getArgument(nIn + j));
    }

    // Reverse walk the clones.
    for (auto it = clones.rbegin(); it != clones.rend(); ++it) {
      mlir::Operation *op = *it;
      llvm::SmallVector<mlir::Value> outCotans;
      bool any = false;
      for (mlir::Value r : op->getResults()) {
        mlir::Value c = cotan.lookup(r);
        outCotans.push_back(c);
        if (c)
          any = true;
      }
      if (!any) {
        if (op->use_empty() &&
            getRegisteredSemanticEffect(op) == SemanticEffectLevel::Pure)
          op->erase();
        continue;
      }

      auto residualOwner = op->getAttrOfType<mlir::StringAttr>(
          "tessera.autodiff.residual_owner");
      bool eraseSavedPrimal =
          explicitRegionResiduals.contains(op) && op->use_empty() &&
          residualOwner &&
          (residualOwner.getValue() == "scf_if" ||
           residualOwner.getValue() == "scf_while");
      if (failed(differentiateOperation(op, outCotans, builder, cotan)))
        return mlir::failure();
      if (eraseSavedPrimal)
        op->erase();
    }

    // Return input cotangents (zero-splat for inputs off the gradient path so
    // the signature is total and the buffer binding in Phase 4 is uniform).
    llvm::SmallVector<mlir::Value> results;
    for (unsigned i = 0; i < nIn; ++i) {
      mlir::Value g = cotan.lookup(bwdBlock->getArgument(i));
      if (!g) {
        mlir::Type ty = fwdInTypes[i];
        mlir::Value zero;
        if (auto shaped = llvm::dyn_cast<mlir::ShapedType>(ty)) {
          auto elem = shaped.getElementType();
          mlir::Attribute z = llvm::isa<mlir::FloatType>(elem)
                                  ? (mlir::Attribute)mlir::FloatAttr::get(elem, 0.0)
                                  : (mlir::Attribute)mlir::IntegerAttr::get(elem, 0);
          zero = builder.create<mlir::arith::ConstantOp>(
              fwd.getLoc(), mlir::DenseElementsAttr::get(shaped, z));
        } else {
          zero = builder.create<mlir::arith::ConstantOp>(
              fwd.getLoc(), builder.getZeroAttr(ty));
        }
        g = zero;
      }
      results.push_back(g);
    }
    builder.create<mlir::func::ReturnOp>(fwd.getLoc(), results);

    // Link the forward to its paired backward (residuals empty under
    // recompute-all — the forward stays primals-only).
    fwd->setAttr("tessera.autodiff.paired",
                 mlir::FlatSymbolRefAttr::get(ctx, bwdName));
    fwd->setAttr("tessera.autodiff.residual_policy",
                 builder.getStringAttr(pairedResidualPolicy));
    if (!residualSources.empty())
      fwd->setAttr("tessera.autodiff.residual_sources",
                   builder.getArrayAttr(residualSources));
    eraseStopGradientBarriers(bwd);
    eraseStopGradientBarriers(fwd);
    return mlir::success();
  }

  mlir::Value buildZeroLike(mlir::OpBuilder &builder, mlir::Value primal) {
    mlir::Type type = primal.getType();
    mlir::Type elementType = mlir::getElementTypeOrSelf(type);
    mlir::TypedAttr zero = llvm::isa<mlir::FloatType>(elementType)
                               ? mlir::TypedAttr(
                                     mlir::FloatAttr::get(elementType, 0.0))
                               : mlir::TypedAttr(
                                     mlir::IntegerAttr::get(elementType, 0));
    if (auto shaped = llvm::dyn_cast<mlir::ShapedType>(type);
        shaped && shaped.hasStaticShape()) {
      return builder
          .create<mlir::arith::ConstantOp>(
              primal.getLoc(), mlir::DenseElementsAttr::get(shaped, zero))
          .getResult();
    }
    if (!llvm::isa<mlir::ShapedType>(type))
      return builder
          .create<mlir::arith::ConstantOp>(primal.getLoc(), zero)
          .getResult();

    mlir::OperationState state(primal.getLoc(),
                               "tessera.custom_adjoint_call");
    state.addOperands(primal);
    state.addTypes(type);
    state.addAttribute("name", builder.getStringAttr("zeros_like"));
    return builder.create(state)->getResult(0);
  }

  mlir::LogicalResult buildRegionPullback(
      mlir::Region &region, mlir::ValueRange outputCotangents,
      mlir::ValueRange blockArgumentValues,
      llvm::ArrayRef<mlir::Value> captures,
      const llvm::DenseMap<mlir::Value, mlir::Value> &savedValues,
      mlir::OpBuilder &builder,
      llvm::SmallVectorImpl<mlir::Value> &blockArgumentCotangents,
      llvm::SmallVectorImpl<mlir::Value> &captureCotangents) {
    if (!region.hasOneBlock() ||
        region.front().getNumArguments() != blockArgumentValues.size())
      return mlir::failure();
    auto yield = mlir::dyn_cast<mlir::scf::YieldOp>(
        region.front().getTerminator());
    if (!yield || yield.getNumOperands() != outputCotangents.size())
      return mlir::failure();

    mlir::IRMapping mapping;
    for (auto [argument, value] : llvm::zip_equal(
             region.front().getArguments(), blockArgumentValues))
      mapping.map(argument, value);
    for (mlir::Value capture : captures)
      mapping.map(capture, capture);
    // Pre-map saved branch-local SSA so downstream clones consume the residual
    // rather than the recomputed value.  We still create the defining clone so
    // its registered derivative model remains the source of the pullback; any
    // primal-result uses emitted by that model are redirected to the residual
    // below and the now-dead pure clone is erased.
    for (const auto &entry : savedValues)
      mapping.map(entry.first, entry.second);
    llvm::SmallVector<mlir::Operation *> clones;
    llvm::DenseMap<mlir::Value, mlir::Value> cloneResultToSaved;
    for (mlir::Operation &source : region.front().without_terminator()) {
      mlir::Operation *clone = builder.clone(source, mapping);
      clones.push_back(clone);
      for (auto [sourceResult, cloneResult] :
           llvm::zip_equal(source.getResults(), clone->getResults()))
        if (mlir::Value saved = savedValues.lookup(sourceResult)) {
          cloneResultToSaved.try_emplace(cloneResult, saved);
          mapping.map(sourceResult, saved);
        }
    }

    CotangentMap cotangents;
    for (auto [yielded, seed] :
         llvm::zip_equal(yield.getOperands(), outputCotangents))
      if (seed)
        accumulateCotangent(builder, cotangents,
                            mapping.lookupOrDefault(yielded), seed);

    for (mlir::Operation *clone : llvm::reverse(clones)) {
      llvm::SmallVector<mlir::Value> resultCotangents;
      bool active = false;
      for (mlir::Value result : clone->getResults()) {
        mlir::Value key = cloneResultToSaved.lookup(result);
        mlir::Value cotangent = cotangents.lookup(key ? key : result);
        resultCotangents.push_back(cotangent);
        active |= static_cast<bool>(cotangent);
      }
      if (!active && clone->use_empty() &&
          getRegisteredSemanticEffect(clone) == SemanticEffectLevel::Pure) {
        clone->erase();
        continue;
      }
      if (active && failed(differentiateOperation(
                        clone, resultCotangents, builder, cotangents)))
        return mlir::failure();
      bool allResultsSaved = clone->getNumResults() != 0;
      for (mlir::Value result : clone->getResults()) {
        mlir::Value saved = cloneResultToSaved.lookup(result);
        allResultsSaved &= static_cast<bool>(saved);
        if (saved)
          result.replaceAllUsesWith(saved);
      }
      if (allResultsSaved && clone->use_empty() &&
          getRegisteredSemanticEffect(clone) == SemanticEffectLevel::Pure)
        clone->erase();
    }

    for (auto [argument, value] : llvm::zip_equal(
             region.front().getArguments(), blockArgumentValues)) {
      if (!llvm::isa<mlir::FloatType>(
              mlir::getElementTypeOrSelf(argument.getType()))) {
        blockArgumentCotangents.push_back({});
        continue;
      }
      mlir::Value cotangent = cotangents.lookup(value);
      blockArgumentCotangents.push_back(
          cotangent ? cotangent : buildZeroLike(builder, value));
    }
    for (mlir::Value capture : captures) {
      mlir::Value cotangent = cotangents.lookup(capture);
      captureCotangents.push_back(cotangent ? cotangent
                                            : buildZeroLike(builder, capture));
    }
    return mlir::success();
  }

  mlir::LogicalResult differentiateOperation(
      mlir::Operation *op, mlir::ValueRange outputCotangents,
      mlir::OpBuilder &builder, CotangentMap &cotangents) {
    if (op->getNumRegions() != 0) {
      if (!RegionAdjointInterface::supports(op)) {
        op->emitError() << "[AUTODIFF_NESTED_REGION] no registered "
                           "RegionAdjointInterface model for "
                        << op->getName();
        return mlir::failure();
      }
      llvm::SmallVector<RegionCotangent> regionCotangents;
      auto callback = [&](mlir::Region &region, mlir::ValueRange seeds,
                          mlir::ValueRange blockArgumentValues,
                          llvm::ArrayRef<mlir::Value> captures,
                          const llvm::DenseMap<mlir::Value, mlir::Value>
                              &savedValues,
                          mlir::OpBuilder &nestedBuilder,
                          llvm::SmallVectorImpl<mlir::Value> &blockResults,
                          llvm::SmallVectorImpl<mlir::Value> &results) {
        return buildRegionPullback(region, seeds, blockArgumentValues,
                                   captures, savedValues, nestedBuilder,
                                   blockResults,
                                   results);
      };
      mlir::ValueRange residuals;
      auto residualIt = explicitRegionResiduals.find(op);
      if (residualIt != explicitRegionResiduals.end())
        residuals = residualIt->second;
      if (failed(RegionAdjointInterface::buildAdjoint(
              op, builder, outputCotangents, residuals, callback,
              regionCotangents))) {
        op->emitError() << "[AUTODIFF_REGION_ADJOINT] structured pullback "
                           "construction failed";
        return mlir::failure();
      }
      for (const RegionCotangent &entry : regionCotangents)
        accumulateCotangent(builder, cotangents, entry.primal,
                            entry.cotangent);
      return mlir::success();
    }

    // These tensor slice operations are introduced by control_scan lowering.
    // Keep their linear transpose here instead of teaching the Tessera
    // operation interface about foreign-dialect operations.  extract_slice^T
    // scatters into a zero source; insert_slice^T gathers the overwritten
    // source region and masks that region out of the destination cotangent.
    if (auto extract = mlir::dyn_cast<mlir::tensor::ExtractSliceOp>(op)) {
      if (outputCotangents.size() != 1 || !outputCotangents.front())
        return mlir::failure();
      mlir::Value sourceZero = buildZeroLike(builder, extract.getSource());
      mlir::Value sourceCotangent =
          mlir::tensor::InsertSliceOp::create(
              builder, op->getLoc(), outputCotangents.front(), sourceZero,
              extract.getMixedOffsets(), extract.getMixedSizes(),
              extract.getMixedStrides());
      accumulateCotangent(builder, cotangents, extract.getSource(),
                          sourceCotangent);
      return mlir::success();
    }
    if (auto insert = mlir::dyn_cast<mlir::tensor::InsertSliceOp>(op)) {
      if (outputCotangents.size() != 1 || !outputCotangents.front())
        return mlir::failure();
      mlir::Value sourceCotangent =
          mlir::tensor::ExtractSliceOp::create(
              builder, op->getLoc(), insert.getSourceType(),
              outputCotangents.front(), insert.getMixedOffsets(),
              insert.getMixedSizes(), insert.getMixedStrides());
      mlir::Value sourceZero = buildZeroLike(builder, insert.getSource());
      mlir::Value destinationCotangent =
          mlir::tensor::InsertSliceOp::create(
              builder, op->getLoc(), sourceZero, outputCotangents.front(),
              insert.getMixedOffsets(), insert.getMixedSizes(),
              insert.getMixedStrides());
      accumulateCotangent(builder, cotangents, insert.getSource(),
                          sourceCotangent);
      accumulateCotangent(builder, cotangents, insert.getDest(),
                          destinationCotangent);
      return mlir::success();
    }

    // tensor.empty carries allocation shape, not mathematical data.  Its
    // operands are index-valued extents and therefore have no cotangent.  The
    // contents become differentiable only after a subsequent insert/fill op;
    // treating the allocation itself as an unknown differentiable operation
    // incorrectly rejects dynamically shaped residual/state buffers.
    if (mlir::isa<mlir::tensor::EmptyOp>(op))
      return mlir::success();

    llvm::SmallVector<mlir::Value> inputCotangents;
    if (auto adjoint = mlir::dyn_cast<AdjointInterface>(op)) {
      if (!adjoint.isDifferentiable()) {
        op->emitError() << "[AUTODIFF_PAIRED] op " << op->getName()
                        << " declares AdjointInterface but isDifferentiable() "
                           "is false";
        return mlir::failure();
      }
      inputCotangents = adjoint.buildAdjoint(builder, outputCotangents);
    } else if (auto linear =
                   mlir::dyn_cast<LinearTransposeInterface>(op)) {
      inputCotangents =
          linear.buildLinearTranspose(builder, outputCotangents);
      for (auto [index, cotangent] : llvm::enumerate(inputCotangents)) {
        if (cotangent && !linear.isLinearInOperand(index)) {
          op->emitError()
              << "[AUTODIFF_PAIRED] LinearTransposeInterface produced a "
                 "cotangent for non-linear operand "
              << index;
          return mlir::failure();
        }
      }
    } else {
      if (op->getNumOperands() > 0) {
        op->emitError() << "[AUTODIFF_OP_NOT_DIFFERENTIABLE] op "
                        << op->getName()
                        << " is on the gradient path but implements neither "
                           "AdjointInterface nor LinearTransposeInterface";
        return mlir::failure();
      }
      return mlir::success();
    }
    if (inputCotangents.size() != op->getNumOperands()) {
      op->emitError() << "[AUTODIFF_PAIRED] derivative interface returned "
                      << inputCotangents.size() << " cotangents, expected "
                      << op->getNumOperands();
      return mlir::failure();
    }
    for (auto [operand, cotangent] :
         llvm::zip_equal(op->getOperands(), inputCotangents))
      accumulateCotangent(builder, cotangents, operand, cotangent);
    return mlir::success();
  }
};

}  // namespace

std::unique_ptr<mlir::Pass> createAutodiffPairedPass() {
  return std::make_unique<AutodiffPairedPass>();
}

}  // namespace tessera
