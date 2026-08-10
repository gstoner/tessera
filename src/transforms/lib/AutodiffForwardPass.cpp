//===- AutodiffForwardPass.cpp - Paired Graph IR JVP -----------*- C++ -*-===//

#include "Tessera/Transforms/Passes.h"
#include "Tessera/IR/TesseraOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
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
  return {};
}

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
    registry.insert<mlir::arith::ArithDialect, mlir::func::FuncDialect>();
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
        if (!tensor || !tensor.hasStaticShape() ||
            !mlir::isa<mlir::FloatType, mlir::IntegerType,
                       mlir::ComplexType>(
                tensor.getElementType())) {
          forward.emitError("tessera-autodiff-forward: first cut requires static ranked floating, complex, or integer tensor arguments");
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
          auto tensor = mlir::cast<mlir::RankedTensorType>(
              forward.getArgumentTypes()[index]);
          if (!mlir::isa<mlir::FloatType, mlir::ComplexType>(
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
          auto tensor = mlir::cast<mlir::RankedTensorType>(
              forward.getArgumentTypes()[index]);
          if (!mlir::isa<mlir::FloatType, mlir::ComplexType>(
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

      mlir::IRMapping primalMap;
      llvm::DenseMap<mlir::Value, mlir::Value> tangentMap;
      llvm::SmallDenseSet<mlir::Value> activeValues;
      unsigned argumentCount = forward.getNumArguments();
      unsigned tangentIndex = argumentCount;
      for (auto [index, argument] : llvm::enumerate(forward.getArguments())) {
        primalMap.map(argument, body->getArgument(index));
        if (wrtSet.contains(index)) {
          tangentMap[argument] = body->getArgument(tangentIndex++);
          activeValues.insert(argument);
        } else {
          tangentMap[argument] = mlir::Value{};
        }
      }

      auto originalReturn = mlir::cast<mlir::func::ReturnOp>(
          forward.getBody().front().getTerminator());
      for (mlir::Operation &operation : forward.getBody().front()) {
        if (mlir::isa<mlir::func::ReturnOp>(operation))
          break;
        if (operation.getNumRegions() != 0) {
          operation.emitError("tessera-autodiff-forward: active nested regions require RegionTangentInterface");
          return signalPassFailure();
        }

        mlir::Operation *primal = builder.clone(operation, primalMap);
        for (auto [source, cloned] :
             llvm::zip(operation.getResults(), primal->getResults()))
          primalMap.map(source, cloned);

        bool active = false;
        llvm::SmallVector<mlir::Value> inputTangents;
        for (mlir::Value operand : operation.getOperands()) {
          auto found = tangentMap.find(operand);
          if (found != tangentMap.end()) {
            inputTangents.push_back(found->second);
            active |= activeValues.contains(operand);
          } else {
            mlir::Value zero = buildStaticZero(
                builder, operation.getLoc(), primalMap.lookup(operand).getType());
            inputTangents.push_back(zero);
          }
        }

        llvm::SmallVector<mlir::Value> resultTangents;
        if (active) {
          if (!isAllowedStochasticTangent(primal)) {
            operation.emitError(
                "tessera-autodiff-forward: active stochastic operation requires explicit constant_noise replay or a pathwise/score-function operation");
            return signalPassFailure();
          }
          auto tangent = mlir::dyn_cast<TangentInterface>(primal);
          if (!tangent) {
            operation.emitError("tessera-autodiff-forward: active operation has no TangentInterface");
            return signalPassFailure();
          }
          resultTangents = tangent.buildTangent(builder, inputTangents);
          if (resultTangents.size() != operation.getNumResults()) {
            operation.emitError("tessera-autodiff-forward: TangentInterface returned the wrong result arity");
            return signalPassFailure();
          }
        } else {
          resultTangents.resize(operation.getNumResults());
        }
        for (auto [source, tangent] :
             llvm::zip(operation.getResults(), resultTangents)) {
          if (!tangent && active) {
            operation.emitError("tessera-autodiff-forward: cannot materialize a result tangent");
            return signalPassFailure();
          }
          tangentMap[source] = tangent;
          if (active && !mlir::isa<StopGradientOp>(primal))
            activeValues.insert(source);
        }
      }

      llvm::SmallVector<mlir::Value> returns;
      for (mlir::Value result : originalReturn.getOperands())
        returns.push_back(primalMap.lookup(result));
      for (mlir::Value result : originalReturn.getOperands()) {
        auto tangent = tangentMap.find(result);
        mlir::Value value =
            tangent == tangentMap.end() ? mlir::Value{} : tangent->second;
        if (!value)
          value = buildStaticZero(builder, forward.getLoc(),
                                  primalMap.lookup(result).getType());
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
