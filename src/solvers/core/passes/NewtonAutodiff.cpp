//===- NewtonAutodiff.cpp - value-producing implicit VJP/JVP ------*- C++ -*-===//
//
// Lowers the differentiation contract of `tessera_solver.implicit` with the
// implicit-function theorem.  An implicit op must identify the function that
// evaluates its residual with:
//
//   residual = @residual
//
// where @residual accepts `(parameters..., solution...)` and returns one
// residual for every solution result.  For each implicit op this pass creates
// a private VJP function with the ABI
//
//   (parameters..., solution..., solution_cotangents...)
//       -> parameter_cotangents...
//
// and the value-producing body
//
//   residual(parameters, solution)
//   lambda = linear_solve((D_solution residual)^T, solution_cotangents)
//   parameter_cotangents = -(D_parameters residual)^T lambda
//
// The residual value is deliberately threaded through the solve.  It makes
// the linearization point explicit and prevents a later lowering from silently
// differentiating a different residual or solution.  The generated function
// is linked from the forward implicit op by a FlatSymbolRefAttr; there is no
// runtime string lookup and no annotation-only success state.
//
// The solver operations remain target-independent.  Architecture-owned
// lowering selects the matrix-free linear solver and residual VJP kernels.
//
//===----------------------------------------------------------------------===//

#include "SolversPasses.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/SmallVector.h"

using namespace mlir;

namespace {

static bool sameTypes(TypeRange lhs, TypeRange rhs) {
  return lhs.size() == rhs.size() &&
         llvm::equal(lhs, rhs, [](Type a, Type b) { return a == b; });
}

static Operation *createSolverOp(OpBuilder &builder, Location loc,
                                 StringRef name, ValueRange operands,
                                 TypeRange results,
                                 ArrayRef<NamedAttribute> attrs) {
  OperationState state(loc, name);
  state.addOperands(operands);
  state.addTypes(results);
  state.addAttributes(attrs);
  return builder.create(state);
}

struct NewtonAutodiffPass
    : PassWrapper<NewtonAutodiffPass, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(NewtonAutodiffPass)

  NewtonAutodiffPass() = default;
  NewtonAutodiffPass(const NewtonAutodiffPass &other) : PassWrapper(other) {}

  Option<bool> generateJVP{
      *this, "generate-jvp",
      llvm::cl::desc("Also generate a value-producing implicit JVP function"),
      llvm::cl::init(false)};

  StringRef getArgument() const final { return "tessera-newton-autodiff"; }
  StringRef getDescription() const final {
    return "Generate value-producing IFT residual/linear-solve/adjoint IR";
  }

  void runOnOperation() override {
    ModuleOp module = getOperation();
    MLIRContext *ctx = module.getContext();
    SmallVector<Operation *> implicitOps;
    module.walk([&](Operation *op) {
      if (op->getName().getStringRef() == "tessera_solver.implicit")
        implicitOps.push_back(op);
    });

    SymbolTable symbols(module);
    unsigned ordinal = 0;
    for (Operation *implicit : implicitOps) {
      auto parent = implicit->getParentOfType<func::FuncOp>();
      if (!parent) {
        implicit->emitError("implicit differentiation requires a containing "
                            "func.func");
        signalPassFailure();
        return;
      }
      if (implicit->getNumOperands() == 0 || implicit->getNumResults() == 0) {
        implicit->emitError("implicit differentiation requires at least one "
                            "parameter operand and one solution result");
        signalPassFailure();
        return;
      }

      auto residualRef =
          implicit->getAttrOfType<FlatSymbolRefAttr>("residual");
      if (!residualRef) {
        implicit->emitError("requires a `residual = @symbol` "
                            "contract before IFT differentiation");
        signalPassFailure();
        return;
      }
      auto residualFn = symbols.lookup<func::FuncOp>(residualRef.getValue());
      if (!residualFn) {
        implicit->emitError() << "residual symbol @" << residualRef.getValue()
                              << " does not name a func.func";
        signalPassFailure();
        return;
      }

      SmallVector<Type> parameterTypes(implicit->getOperandTypes());
      SmallVector<Type> solutionTypes(implicit->getResultTypes());
      SmallVector<Type> residualInputs(parameterTypes);
      llvm::append_range(residualInputs, solutionTypes);
      auto residualType = residualFn.getFunctionType();
      if (!sameTypes(residualType.getInputs(), residualInputs) ||
          !sameTypes(residualType.getResults(), solutionTypes)) {
        implicit->emitError()
            << "residual @" << residualRef.getValue()
            << " must have type (parameters..., solution...) -> residuals "
               "with one result matching each solution; got "
            << residualType;
        signalPassFailure();
        return;
      }

      std::string baseName =
          (parent.getSymName() + "__implicit_" + Twine(ordinal++)).str();
      std::string vjpName = baseName + "_vjp";
      if (symbols.lookup(vjpName)) {
        implicit->emitError() << "generated IFT symbol collision: @" << vjpName;
        signalPassFailure();
        return;
      }

      SmallVector<Type> vjpInputs(parameterTypes);
      llvm::append_range(vjpInputs, solutionTypes);
      llvm::append_range(vjpInputs, solutionTypes);
      auto vjpType = FunctionType::get(ctx, vjpInputs, parameterTypes);
      auto vjp = func::FuncOp::create(implicit->getLoc(), vjpName, vjpType);
      vjp.setPrivate();
      vjp->setAttr("tessera.solver.ift_role",
                   StringAttr::get(ctx, "implicit_vjp"));
      vjp->setAttr("tessera.solver.residual", residualRef);
      Block *entry = vjp.addEntryBlock();
      OpBuilder builder = OpBuilder::atBlockBegin(entry);

      const unsigned numParameters = parameterTypes.size();
      const unsigned numSolutions = solutionTypes.size();
      ValueRange parameters = entry->getArguments().take_front(numParameters);
      ValueRange solutions = entry->getArguments().slice(numParameters,
                                                         numSolutions);
      ValueRange cotangents = entry->getArguments().take_back(numSolutions);
      SmallVector<Value> primals(parameters);
      llvm::append_range(primals, solutions);

      NamedAttribute residualCallee = builder.getNamedAttr(
          "callee", residualRef);
      Operation *residual = createSolverOp(
          builder, implicit->getLoc(), "tessera_solver.residual", primals,
          solutionTypes, {residualCallee});

      SmallVector<Value> solveInputs(primals);
      llvm::append_range(solveInputs, residual->getResults());
      llvm::append_range(solveInputs, cotangents);
      SmallVector<NamedAttribute> solveAttrs{
          builder.getNamedAttr("residual", residualRef),
          builder.getNamedAttr("transpose", builder.getBoolAttr(true)),
          builder.getNamedAttr("linearization",
                               builder.getStringAttr("solution"))};
      Operation *lambda = createSolverOp(
          builder, implicit->getLoc(), "tessera_solver.linear_solve",
          solveInputs, solutionTypes, solveAttrs);

      SmallVector<Value> adjointInputs(primals);
      llvm::append_range(adjointInputs, lambda->getResults());
      SmallVector<NamedAttribute> adjointAttrs{
          builder.getNamedAttr("residual", residualRef),
          builder.getNamedAttr("wrt", builder.getStringAttr("parameters")),
          builder.getNamedAttr("scale", builder.getF64FloatAttr(-1.0))};
      Operation *adjoint = createSolverOp(
          builder, implicit->getLoc(), "tessera_solver.residual_adjoint",
          adjointInputs, parameterTypes, adjointAttrs);
      func::ReturnOp::create(builder, implicit->getLoc(),
                             adjoint->getResults());

      symbols.insert(vjp);
      implicit->setAttr("tessera.solver.vjp",
                        FlatSymbolRefAttr::get(ctx, vjpName));
      implicit->setAttr(
          "tessera.solver.vjp_arity",
          IntegerAttr::get(IntegerType::get(ctx, 64), parameterTypes.size()));
      implicit->setAttr("tessera.solver.autodiff_ready", UnitAttr::get(ctx));
      implicit->setAttr(
          "tessera.solver.diff_method",
          StringAttr::get(ctx, "implicit_function_theorem"));
      implicit->setAttr("tessera.solver.ift_contract_version",
                        IntegerAttr::get(IntegerType::get(ctx, 32), 1));

      if (generateJVP) {
        std::string jvpName = baseName + "_jvp";
        SmallVector<Type> jvpInputs(parameterTypes);
        llvm::append_range(jvpInputs, solutionTypes);
        llvm::append_range(jvpInputs, parameterTypes);
        auto jvpType = FunctionType::get(ctx, jvpInputs, solutionTypes);
        auto jvp = func::FuncOp::create(implicit->getLoc(), jvpName, jvpType);
        jvp.setPrivate();
        jvp->setAttr("tessera.solver.ift_role",
                     StringAttr::get(ctx, "implicit_jvp"));
        jvp->setAttr("tessera.solver.residual", residualRef);
        Block *jvpEntry = jvp.addEntryBlock();
        OpBuilder jvpBuilder = OpBuilder::atBlockBegin(jvpEntry);
        ValueRange jvpParameters =
            jvpEntry->getArguments().take_front(numParameters);
        ValueRange jvpSolutions = jvpEntry->getArguments().slice(
            numParameters, numSolutions);
        ValueRange parameterTangents =
            jvpEntry->getArguments().take_back(numParameters);
        SmallVector<Value> jvpPrimals(jvpParameters);
        llvm::append_range(jvpPrimals, jvpSolutions);
        Operation *jvpResidual = createSolverOp(
            jvpBuilder, implicit->getLoc(), "tessera_solver.residual",
            jvpPrimals, solutionTypes,
            {jvpBuilder.getNamedAttr("callee", residualRef)});
        SmallVector<Value> rhsInputs(jvpPrimals);
        llvm::append_range(rhsInputs, parameterTangents);
        Operation *rhs = createSolverOp(
            jvpBuilder, implicit->getLoc(), "tessera_solver.residual_jvp",
            rhsInputs, solutionTypes,
            {jvpBuilder.getNamedAttr("residual", residualRef),
             jvpBuilder.getNamedAttr("wrt",
                                     jvpBuilder.getStringAttr("parameters"))});
        SmallVector<Value> jvpSolveInputs(jvpPrimals);
        llvm::append_range(jvpSolveInputs, jvpResidual->getResults());
        llvm::append_range(jvpSolveInputs, rhs->getResults());
        Operation *solutionTangent = createSolverOp(
            jvpBuilder, implicit->getLoc(), "tessera_solver.linear_solve",
            jvpSolveInputs, solutionTypes,
            {jvpBuilder.getNamedAttr("residual", residualRef),
             jvpBuilder.getNamedAttr("transpose", jvpBuilder.getBoolAttr(false)),
             jvpBuilder.getNamedAttr("rhs_scale",
                                     jvpBuilder.getF64FloatAttr(-1.0)),
             jvpBuilder.getNamedAttr(
                 "linearization", jvpBuilder.getStringAttr("solution"))});
        func::ReturnOp::create(jvpBuilder, implicit->getLoc(),
                               solutionTangent->getResults());
        symbols.insert(jvp);
        implicit->setAttr("tessera.solver.jvp",
                          FlatSymbolRefAttr::get(ctx, jvpName));
        implicit->setAttr(
            "tessera.solver.jvp_arity",
            IntegerAttr::get(IntegerType::get(ctx, 64), solutionTypes.size()));
      }
    }
  }
};

} // namespace

namespace tessera {
namespace passes {
std::unique_ptr<Pass> createNewtonAutodiffPass() {
  return std::make_unique<NewtonAutodiffPass>();
}
} // namespace passes
} // namespace tessera
