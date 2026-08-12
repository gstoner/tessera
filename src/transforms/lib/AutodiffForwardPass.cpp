//===- AutodiffForwardPass.cpp - Paired Graph IR JVP -----------*- C++ -*-===//

#include "Tessera/Transforms/Passes.h"
#include "Tessera/IR/TesseraOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/SmallVector.h"

namespace tessera {
namespace {

static bool isAllowedStochasticTangent(mlir::Operation *op) {
  auto effect = op->getAttrOfType<mlir::StringAttr>("tessera.effect_kind");
  if (!effect || effect.getValue() != "random")
    return true;
  auto estimator = op->getAttrOfType<mlir::StringAttr>("estimator");
  if (estimator && estimator.getValue() != "constant_noise")
    return false;
  // Constant-noise differentiation must replay the exact primal sample.
  // Dropout's first contract uses a counter-based seed/counter identity.
  if (mlir::isa<DropoutOp>(op))
    return op->getAttr("seed") != nullptr;
  return false;
}

static mlir::Value buildStaticZero(mlir::OpBuilder &builder,
                                   mlir::Location loc, mlir::Type type) {
  if (auto shaped = mlir::dyn_cast<mlir::ShapedType>(type)) {
    if (!shaped.hasStaticShape())
      return {};
    mlir::Type element = shaped.getElementType();
    mlir::Attribute zero = builder.getZeroAttr(element);
    if (!zero)
      return {};
    return builder
        .create<mlir::arith::ConstantOp>(
            loc, mlir::DenseElementsAttr::get(shaped, zero))
        .getResult();
  }
  if (auto floatType = mlir::dyn_cast<mlir::FloatType>(type))
    return builder
        .create<mlir::arith::ConstantOp>(loc,
                                         builder.getFloatAttr(floatType, 0.0))
        .getResult();
  if (auto integerType = mlir::dyn_cast<mlir::IntegerType>(type))
    return builder
        .create<mlir::arith::ConstantOp>(
            loc, builder.getIntegerAttr(integerType, 0))
        .getResult();
  if (mlir::isa<mlir::IndexType>(type))
    return builder.create<mlir::arith::ConstantIndexOp>(loc, 0).getResult();
  return {};
}

struct ForwardState {
  mlir::IRMapping primals;
  llvm::DenseMap<mlir::Value, mlir::Value> tangents;
  llvm::SmallDenseSet<mlir::Value> active;
};

/// Build primal and tangent SSA together. Structured control flow must not be
/// cloned once for the primal and again for the tangent: doing so duplicates
/// effects and can select a different stochastic/control path. This builder
/// instead extends each admitted SCF op with tangent-carried results.
class RegionTangentBuilder {
 public:
  mlir::LogicalResult buildOperation(mlir::Operation &operation,
                                     mlir::OpBuilder &builder,
                                     ForwardState &state) {
    if (auto ifOp = mlir::dyn_cast<mlir::scf::IfOp>(&operation))
      return buildIf(ifOp, builder, state);
    if (auto forOp = mlir::dyn_cast<mlir::scf::ForOp>(&operation))
      return buildFor(forOp, builder, state);
    if (auto whileOp = mlir::dyn_cast<mlir::scf::WhileOp>(&operation))
      return buildWhile(whileOp, builder, state);
    if (operation.getNumRegions() != 0) {
      operation.emitError(
          "tessera-autodiff-forward: unsupported active nested region; "
          "expected scf.if, scf.for, or canonical scf.while");
      return mlir::failure();
    }
    return buildLeaf(operation, builder, state);
  }

 private:
  mlir::Value tangentOrZero(mlir::Value source, mlir::Value primal,
                            mlir::OpBuilder &builder) {
    auto found = current_->tangents.find(source);
    if (found != current_->tangents.end() && found->second)
      return found->second;
    return buildStaticZero(builder, source.getLoc(), primal.getType());
  }

  mlir::LogicalResult buildLeaf(mlir::Operation &operation,
                                mlir::OpBuilder &builder,
                                ForwardState &state) {
    mlir::Operation *primal = builder.clone(operation, state.primals);
    for (auto [source, cloned] :
         llvm::zip(operation.getResults(), primal->getResults()))
      state.primals.map(source, cloned);

    bool active = false;
    llvm::SmallVector<mlir::Value> inputTangents;
    ForwardState *saved = current_;
    current_ = &state;
    for (mlir::Value operand : operation.getOperands()) {
      mlir::Value mapped = state.primals.lookupOrDefault(operand);
      auto found = state.tangents.find(operand);
      if (found != state.tangents.end())
        inputTangents.push_back(found->second);
      else
        inputTangents.push_back(tangentOrZero(operand, mapped, builder));
      active |= state.active.contains(operand);
    }
    current_ = saved;

    llvm::SmallVector<mlir::Value> resultTangents;
    if (active) {
      if (!isAllowedStochasticTangent(primal)) {
        operation.emitError(
            "tessera-autodiff-forward: active stochastic operation requires "
            "explicit constant_noise replay or a pathwise/score-function "
            "operation");
        return mlir::failure();
      }
      if (mlir::isa<mlir::tensor::ExtractSliceOp,
                    mlir::tensor::InsertSliceOp>(primal)) {
        llvm::SmallVector<mlir::Value> operands(primal->getOperands());
        if (mlir::isa<mlir::tensor::ExtractSliceOp>(primal)) {
          if (inputTangents.empty() || !inputTangents[0])
            return mlir::failure();
          operands[0] = inputTangents[0];
        } else {
          if (inputTangents.size() < 2 || !inputTangents[0] ||
              !inputTangents[1])
            return mlir::failure();
          operands[0] = inputTangents[0];
          operands[1] = inputTangents[1];
        }
        mlir::OperationState tangentState(primal->getLoc(), primal->getName());
        tangentState.addOperands(operands);
        tangentState.addTypes(primal->getResultTypes());
        tangentState.addAttributes(primal->getAttrs());
        resultTangents.push_back(builder.create(tangentState)->getResult(0));
      } else if (auto tangent = mlir::dyn_cast<TangentInterface>(primal)) {
        resultTangents = tangent.buildTangent(builder, inputTangents);
      } else {
        operation.emitError(
            "tessera-autodiff-forward: active operation has no "
            "TangentInterface");
        return mlir::failure();
      }
      if (resultTangents.size() != operation.getNumResults()) {
        operation.emitError(
            "tessera-autodiff-forward: TangentInterface rejected the active "
            "operand combination or returned the wrong result arity");
        return mlir::failure();
      }
    } else {
      resultTangents.resize(operation.getNumResults());
    }
    for (auto [source, tangent] :
         llvm::zip(operation.getResults(), resultTangents)) {
      if (!tangent && active) {
        operation.emitError(
            "tessera-autodiff-forward: cannot materialize a result tangent");
        return mlir::failure();
      }
      state.tangents[source] = tangent;
      if (active && !mlir::isa<StopGradientOp>(primal))
        state.active.insert(source);
    }
    return mlir::success();
  }

  mlir::LogicalResult buildBlock(mlir::Block &source, mlir::Block &target,
                                 mlir::OpBuilder &builder,
                                 ForwardState &state,
                                 mlir::Operation *terminator) {
    mlir::OpBuilder::InsertionGuard guard(builder);
    builder.setInsertionPointToStart(&target);
    for (mlir::Operation &nested : source.without_terminator())
      if (mlir::failed(buildOperation(nested, builder, state)))
        return mlir::failure();

    llvm::SmallVector<mlir::Value> yields;
    for (mlir::Value operand : terminator->getOperands())
      yields.push_back(state.primals.lookupOrDefault(operand));
    ForwardState *saved = current_;
    current_ = &state;
    for (mlir::Value operand : terminator->getOperands()) {
      mlir::Value primal = state.primals.lookupOrDefault(operand);
      mlir::Value tangent = tangentOrZero(operand, primal, builder);
      if (!tangent)
        return mlir::failure();
      yields.push_back(tangent);
    }
    current_ = saved;
    mlir::scf::YieldOp::create(builder, terminator->getLoc(), yields);
    return mlir::success();
  }

  mlir::LogicalResult buildIf(mlir::scf::IfOp source,
                              mlir::OpBuilder &builder,
                              ForwardState &state) {
    if (source.getElseRegion().empty()) {
      source.emitError(
          "tessera-autodiff-forward: result-bearing scf.if requires an else "
          "region");
      return mlir::failure();
    }
    llvm::SmallVector<mlir::Type> resultTypes(source.getResultTypes());
    resultTypes.append(source.getResultTypes().begin(),
                       source.getResultTypes().end());
    auto combined = mlir::scf::IfOp::create(
        builder, source.getLoc(), resultTypes,
        state.primals.lookupOrDefault(source.getCondition()),
        /*withElseRegion=*/true);

    bool resultActive = false;
    auto buildBranch = [&](mlir::Region &sourceRegion,
                           mlir::Block *target) -> mlir::LogicalResult {
      ForwardState branch = state;
      auto yield = mlir::dyn_cast<mlir::scf::YieldOp>(
          sourceRegion.front().getTerminator());
      if (!yield || mlir::failed(buildBlock(sourceRegion.front(), *target,
                                            builder, branch, yield)))
        return mlir::failure();
      for (mlir::Value operand : yield.getOperands())
        resultActive |= branch.active.contains(operand);
      return mlir::success();
    };
    if (mlir::failed(buildBranch(source.getThenRegion(), combined.thenBlock())) ||
        mlir::failed(buildBranch(source.getElseRegion(), combined.elseBlock()))) {
      combined.erase();
      return mlir::failure();
    }
    unsigned count = source.getNumResults();
    for (unsigned index = 0; index < count; ++index) {
      state.primals.map(source.getResult(index), combined.getResult(index));
      state.tangents[source.getResult(index)] =
          combined.getResult(index + count);
      if (resultActive)
        state.active.insert(source.getResult(index));
    }
    return mlir::success();
  }

  mlir::LogicalResult buildFor(mlir::scf::ForOp source,
                               mlir::OpBuilder &builder,
                               ForwardState &state) {
    llvm::APInt step;
    if (!source.getRegion().hasOneBlock() ||
        !mlir::matchPattern(source.getStep(), mlir::m_ConstantInt(&step)) ||
        !step.isStrictlyPositive()) {
      source.emitError(
          "tessera-autodiff-forward: scf.for requires one block and a positive "
          "constant step");
      return mlir::failure();
    }
    llvm::SmallVector<mlir::Value> inits;
    for (mlir::Value init : source.getInitArgs())
      inits.push_back(state.primals.lookupOrDefault(init));
    ForwardState *saved = current_;
    current_ = &state;
    for (mlir::Value init : source.getInitArgs()) {
      mlir::Value primal = state.primals.lookupOrDefault(init);
      mlir::Value tangent = tangentOrZero(init, primal, builder);
      if (!tangent)
        return mlir::failure();
      inits.push_back(tangent);
    }
    current_ = saved;
    auto combined = mlir::scf::ForOp::create(
        builder, source.getLoc(),
        state.primals.lookupOrDefault(source.getLowerBound()),
        state.primals.lookupOrDefault(source.getUpperBound()),
        state.primals.lookupOrDefault(source.getStep()), inits);

    ForwardState bodyState = state;
    bodyState.primals.map(source.getInductionVar(),
                          combined.getInductionVar());
    unsigned count = source.getInitArgs().size();
    for (unsigned index = 0; index < count; ++index) {
      mlir::Value sourceArg = source.getRegionIterArg(index);
      bodyState.primals.map(sourceArg, combined.getRegionIterArg(index));
      // Every differentiable loop-carried value has a real tangent iter_arg,
      // even when its initial tangent is zero.  A later body operation may
      // activate that value (control_scan's initially-empty stacked output is
      // the canonical example), so dropping the mapping here loses tangent
      // state across iterations.
      bodyState.tangents[sourceArg] =
          combined.getRegionIterArg(index + count);
      if (state.active.contains(source.getInitArgs()[index])) {
        bodyState.active.insert(sourceArg);
      }
    }
    auto yield = mlir::cast<mlir::scf::YieldOp>(
        source.getRegion().front().getTerminator());
    if (mlir::failed(buildBlock(source.getRegion().front(),
                                *combined.getBody(), builder, bodyState,
                                yield))) {
      combined.erase();
      return mlir::failure();
    }
    for (unsigned index = 0; index < count; ++index) {
      state.primals.map(source.getResult(index), combined.getResult(index));
      state.tangents[source.getResult(index)] =
          combined.getResult(index + count);
      if (bodyState.active.contains(yield.getOperand(index)))
        state.active.insert(source.getResult(index));
    }
    return mlir::success();
  }

  mlir::LogicalResult buildWhile(mlir::scf::WhileOp source,
                                 mlir::OpBuilder &builder,
                                 ForwardState &state) {
    if (!source.getBefore().hasOneBlock() ||
        !source.getAfter().hasOneBlock()) {
      source.emitError(
          "tessera-autodiff-forward: scf.while requires one before and one "
          "after block");
      return mlir::failure();
    }
    llvm::SmallVector<mlir::Value> inits;
    llvm::SmallVector<mlir::Type> types(source.getResultTypes());
    for (mlir::Value init : source.getInits())
      inits.push_back(state.primals.lookupOrDefault(init));
    ForwardState *saved = current_;
    current_ = &state;
    for (mlir::Value init : source.getInits()) {
      mlir::Value primal = state.primals.lookupOrDefault(init);
      mlir::Value tangent = tangentOrZero(init, primal, builder);
      if (!tangent)
        return mlir::failure();
      inits.push_back(tangent);
      types.push_back(init.getType());
    }
    current_ = saved;
    auto combined = mlir::scf::WhileOp::create(builder, source.getLoc(), types,
                                               inits);
    llvm::SmallVector<mlir::Location> locations(types.size(), source.getLoc());

    auto buildBefore = [&]() -> mlir::LogicalResult {
      mlir::Block *target = builder.createBlock(&combined.getBefore());
      target->addArguments(types, locations);
      ForwardState beforeState = state;
      unsigned count = source.getInits().size();
      for (unsigned index = 0; index < count; ++index) {
        mlir::Value sourceArg = source.getBefore().front().getArgument(index);
        beforeState.primals.map(sourceArg, target->getArgument(index));
        if (state.active.contains(source.getInits()[index])) {
          beforeState.tangents[sourceArg] = target->getArgument(index + count);
          beforeState.active.insert(sourceArg);
        } else {
          beforeState.tangents[sourceArg] = mlir::Value{};
        }
      }
      mlir::OpBuilder::InsertionGuard guard(builder);
      builder.setInsertionPointToStart(target);
      for (mlir::Operation &nested :
           source.getBefore().front().without_terminator())
        if (mlir::failed(buildOperation(nested, builder, beforeState)))
          return mlir::failure();
      auto condition = mlir::dyn_cast<mlir::scf::ConditionOp>(
          source.getBefore().front().getTerminator());
      if (!condition)
        return mlir::failure();
      llvm::SmallVector<mlir::Value> forwarded;
      for (mlir::Value value : condition.getArgs())
        forwarded.push_back(beforeState.primals.lookupOrDefault(value));
      ForwardState *old = current_;
      current_ = &beforeState;
      for (mlir::Value value : condition.getArgs()) {
        mlir::Value primal = beforeState.primals.lookupOrDefault(value);
        mlir::Value tangent = tangentOrZero(value, primal, builder);
        if (!tangent)
          return mlir::failure();
        forwarded.push_back(tangent);
      }
      current_ = old;
      mlir::scf::ConditionOp::create(
          builder, condition.getLoc(),
          beforeState.primals.lookupOrDefault(condition.getCondition()),
          forwarded);
      return mlir::success();
    };

    auto buildAfter = [&]() -> mlir::LogicalResult {
      mlir::Block *target = builder.createBlock(&combined.getAfter());
      target->addArguments(types, locations);
      ForwardState afterState = state;
      unsigned count = source.getInits().size();
      for (unsigned index = 0; index < count; ++index) {
        mlir::Value sourceArg = source.getAfter().front().getArgument(index);
        afterState.primals.map(sourceArg, target->getArgument(index));
        if (state.active.contains(source.getInits()[index])) {
          afterState.tangents[sourceArg] = target->getArgument(index + count);
          afterState.active.insert(sourceArg);
        } else {
          afterState.tangents[sourceArg] = mlir::Value{};
        }
      }
      auto yield = mlir::dyn_cast<mlir::scf::YieldOp>(
          source.getAfter().front().getTerminator());
      return yield ? buildBlock(source.getAfter().front(), *target, builder,
                                afterState, yield)
                   : mlir::failure();
    };
    if (mlir::failed(buildBefore()) || mlir::failed(buildAfter())) {
      combined.erase();
      return mlir::failure();
    }
    // createBlock() moves the builder into the new region. Restore the outer
    // insertion point so following function operations are not accidentally
    // appended to the while body.
    builder.setInsertionPointAfter(combined);
    unsigned count = source.getNumResults();
    for (unsigned index = 0; index < count; ++index) {
      state.primals.map(source.getResult(index), combined.getResult(index));
      state.tangents[source.getResult(index)] =
          combined.getResult(index + count);
      if (state.active.contains(source.getInits()[index]))
        state.active.insert(source.getResult(index));
    }
    return mlir::success();
  }

  ForwardState *current_ = nullptr;
};

class AutodiffForwardPass
    : public mlir::PassWrapper<AutodiffForwardPass,
                               mlir::OperationPass<mlir::ModuleOp>> {
 public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(AutodiffForwardPass)

  llvm::StringRef getArgument() const final {
    return "tessera-autodiff-forward";
  }
  llvm::StringRef getDescription() const final {
    return "Emit a paired Graph IR JVP through TangentInterface";
  }
  void getDependentDialects(mlir::DialectRegistry &registry) const override {
    registry.insert<mlir::arith::ArithDialect, mlir::func::FuncDialect,
                    mlir::scf::SCFDialect, mlir::tensor::TensorDialect>();
  }

  void runOnOperation() override {
    mlir::ModuleOp module = getOperation();
    llvm::SmallVector<mlir::func::FuncOp> forwards;
    module.walk([&](mlir::func::FuncOp func) {
      auto mode = func->getAttrOfType<mlir::StringAttr>("tessera.autodiff");
      if (mode && mode.getValue() == "forward")
        forwards.push_back(func);
    });

    for (mlir::func::FuncOp forward : forwards) {
      if (forward.isDeclaration() || !forward.getBody().hasOneBlock()) {
        forward.emitError("tessera-autodiff-forward: requires one defined block");
        return signalPassFailure();
      }
      for (mlir::Type type : forward.getArgumentTypes()) {
        auto tensor = mlir::dyn_cast<mlir::RankedTensorType>(type);
        bool admissibleTensor =
            tensor && tensor.hasStaticShape() &&
            mlir::isa<mlir::FloatType, mlir::IntegerType, mlir::ComplexType>(
                tensor.getElementType());
        bool controlScalar =
            mlir::isa<mlir::IntegerType, mlir::IndexType>(type);
        if (!admissibleTensor && !controlScalar) {
          forward.emitError(
              "tessera-autodiff-forward: requires static ranked numeric "
              "tensors plus nondifferentiable integer/index control scalars");
          return signalPassFailure();
        }
      }

      std::string jvpName = (forward.getName() + "__jvp").str();
      if (module.lookupSymbol(jvpName)) {
        forward.emitError("tessera-autodiff-forward: JVP symbol already exists");
        return signalPassFailure();
      }

      llvm::SmallVector<unsigned> wrtIndices;
      llvm::SmallDenseSet<unsigned> wrtSet;
      if (auto wrt = forward->getAttrOfType<mlir::ArrayAttr>(
              "tessera.autodiff.wrt_indices")) {
        for (mlir::Attribute item : wrt) {
          auto indexAttr = mlir::dyn_cast<mlir::IntegerAttr>(item);
          if (!indexAttr || indexAttr.getInt() < 0 ||
              indexAttr.getInt() >= forward.getNumArguments()) {
            forward.emitError(
                "tessera-autodiff-forward: invalid wrt_indices entry");
            return signalPassFailure();
          }
          unsigned index = static_cast<unsigned>(indexAttr.getInt());
          auto tensor = mlir::dyn_cast<mlir::RankedTensorType>(
              forward.getArgumentTypes()[index]);
          if (!tensor ||
              !mlir::isa<mlir::FloatType, mlir::ComplexType>(
                  tensor.getElementType())) {
            forward.emitError(
                "tessera-autodiff-forward: wrt_indices may select only floating or complex tensor arguments");
            return signalPassFailure();
          }
          if (!wrtSet.insert(index).second) {
            forward.emitError(
                "tessera-autodiff-forward: duplicate wrt_indices entry");
            return signalPassFailure();
          }
          wrtIndices.push_back(index);
        }
        if (wrtIndices.empty()) {
          forward.emitError(
              "tessera-autodiff-forward: wrt_indices must not be empty");
          return signalPassFailure();
        }
      } else {
        for (unsigned index = 0; index < forward.getNumArguments(); ++index) {
          auto tensor = mlir::dyn_cast<mlir::RankedTensorType>(
              forward.getArgumentTypes()[index]);
          if (!tensor ||
              !mlir::isa<mlir::FloatType, mlir::ComplexType>(
                  tensor.getElementType()))
            continue;
          wrtIndices.push_back(index);
          wrtSet.insert(index);
        }
        if (wrtIndices.empty()) {
          forward.emitError(
              "tessera-autodiff-forward: no differentiable floating or complex arguments");
          return signalPassFailure();
        }
      }

      llvm::SmallVector<mlir::Type> inputTypes(forward.getArgumentTypes());
      for (unsigned index : wrtIndices)
        inputTypes.push_back(forward.getArgumentTypes()[index]);
      llvm::SmallVector<mlir::Type> resultTypes(forward.getResultTypes());
      resultTypes.append(forward.getResultTypes().begin(),
                         forward.getResultTypes().end());
      auto jvpType = mlir::FunctionType::get(&getContext(), inputTypes,
                                             resultTypes);
      auto jvp = mlir::func::FuncOp::create(forward.getLoc(), jvpName, jvpType);
      jvp.setPrivate();
      jvp->setAttr("tessera.autodiff.role",
                   mlir::StringAttr::get(&getContext(), "jvp"));
      jvp->setAttr("tessera.autodiff.forward",
                   mlir::FlatSymbolRefAttr::get(&getContext(), forward.getName()));
      mlir::Block *body = jvp.addEntryBlock();
      mlir::OpBuilder builder(body, body->begin());

      ForwardState state;
      unsigned argumentCount = forward.getNumArguments();
      unsigned tangentIndex = argumentCount;
      for (auto [index, argument] : llvm::enumerate(forward.getArguments())) {
        state.primals.map(argument, body->getArgument(index));
        if (wrtSet.contains(index)) {
          state.tangents[argument] = body->getArgument(tangentIndex++);
          state.active.insert(argument);
        } else {
          state.tangents[argument] = mlir::Value{};
        }
      }

      auto originalReturn = mlir::cast<mlir::func::ReturnOp>(
          forward.getBody().front().getTerminator());
      RegionTangentBuilder tangentBuilder;
      for (mlir::Operation &operation : forward.getBody().front()) {
        if (mlir::isa<mlir::func::ReturnOp>(operation))
          break;
        if (mlir::failed(
                tangentBuilder.buildOperation(operation, builder, state)))
          return signalPassFailure();
      }

      llvm::SmallVector<mlir::Value> returns;
      for (mlir::Value result : originalReturn.getOperands())
        returns.push_back(state.primals.lookup(result));
      for (mlir::Value result : originalReturn.getOperands()) {
        auto tangent = state.tangents.find(result);
        mlir::Value value =
            tangent == state.tangents.end() ? mlir::Value{} : tangent->second;
        if (!value)
          value = buildStaticZero(builder, forward.getLoc(),
                                  state.primals.lookup(result).getType());
        if (!value) {
          forward.emitError(
              "tessera-autodiff-forward: cannot materialize return tangent");
          return signalPassFailure();
        }
        returns.push_back(value);
      }
      builder.create<mlir::func::ReturnOp>(forward.getLoc(), returns);
      module.push_back(jvp);
      forward->setAttr("tessera.autodiff.jvp",
                       mlir::FlatSymbolRefAttr::get(&getContext(), jvpName));
    }
  }
};

}  // namespace

std::unique_ptr<mlir::Pass> createAutodiffForwardPass() {
  return std::make_unique<AutodiffForwardPass>();
}

}  // namespace tessera
