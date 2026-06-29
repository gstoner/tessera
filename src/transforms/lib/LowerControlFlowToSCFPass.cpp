// LowerControlFlowToSCFPass.cpp — CF2 control-flow → scf lowering
//
// CF2 of docs/audit/roadmap/CONTROL_FLOW_AND_DEEPSEEK_ACCELERATION_PLAN.md and
// docs/spec/CONTROL_FLOW_CONTRACT.md. The first portable, hardware-free step of
// the CUDA/ROCm control-flow path: lower the Graph IR bounded loop
// `tessera.control_for` to a standard `scf.for` carrying its state in
// `iter_args`, so the downstream tile/backend chain (and, in CF3/CF4, CUDA and
// ROCm) codegens it as ONE loop wrapper rather than one launch per iteration.
//
// The loop body is a symbol-referenced `func.func` (`body = @loop_body`) —
// `control_for` is a value-semantic leaf. The lowering keeps it a `func.call`
// inside the loop region (CF3/CF4 inline / device-codegen it); the calling
// convention is fixed here:
//
//     @body(<carried/iter_args + loop-invariant captures, in original operand
//            order>) -> <carried result type(s)>
//
// Two operand forms, both handled (this is where pytree carries fold in — the
// legacy all-carried form becomes a multi-`iter_args` scf.for):
//
//   * carry_arg_index form: operand `carry_arg_index` is the one loop-carried
//     value; the rest are loop-invariant captures. → scf.for with 1 iter_arg.
//   * legacy form (no carry_arg_index): every operand is loop-carried, one
//     result per operand. → scf.for with N iter_args (the pytree-carry shape).
//
// Standalone `--tessera-control-flow-to-scf`. Runs BEFORE the CF0
// control-flow-target-guard in a backend pipeline, so a successfully lowered
// loop never trips the guard; anything this pass leaves (control_if / while, or
// a malformed for) is still caught loudly. control_if / control_while lowering
// (scf.if / scf.while) is the CF2b follow-up.

#include "Tessera/Transforms/Passes.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/SmallVector.h"

using namespace mlir;

namespace {

struct LowerControlFlowToSCF
    : public PassWrapper<LowerControlFlowToSCF, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(LowerControlFlowToSCF)

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<scf::SCFDialect, func::FuncDialect, arith::ArithDialect,
                    tensor::TensorDialect>();
  }

  StringRef getArgument() const override {
    return "tessera-control-flow-to-scf";
  }
  StringRef getDescription() const override {
    return "CF2: lower tessera.control_{for,if,while} to scf.{for,if,while} "
           "(state in iter_args / branch+loop bodies kept as func.calls), the "
           "portable hardware-free step of the CUDA/ROCm control-flow path.";
  }

  // Outcome of trying to lower one control op.
  enum class Outcome { Lowered, Skipped, Malformed };

  // True iff @sym resolves to a func.func whose signature is exactly
  // (argTypes) -> (resultTypes). Unknown symbols (extern) return true (we trust
  // the declared call). A mismatch means the executable-payload form (a stub
  // signature) — caller skips rather than build an ill-typed func.call.
  bool calleeMatches(Operation *op, FlatSymbolRefAttr sym,
                     TypeRange argTypes, TypeRange resultTypes) {
    auto *callee = SymbolTable::lookupNearestSymbolFrom(op, sym);
    auto fn = dyn_cast_or_null<func::FuncOp>(callee);
    if (!fn)
      return true;
    FunctionType ft = fn.getFunctionType();
    return TypeRange(ft.getInputs()) == argTypes &&
           TypeRange(ft.getResults()) == resultTypes;
  }

  // Reduce a predicate tensor to an i1: `element[0,..,0] > 0`. Handles a 0-d
  // tensor<f32> (no indices) and a rank-r tensor (first element on each axis),
  // matching the control_if flag / control_while cond `>0` selector contract.
  Value extractPredicateI1(OpBuilder &b, Location loc, Value predTensor) {
    auto tt = dyn_cast<RankedTensorType>(predTensor.getType());
    SmallVector<Value> idx;
    if (tt) {
      Value z = arith::ConstantIndexOp::create(b, loc, 0);
      for (int64_t d = 0; d < tt.getRank(); ++d)
        idx.push_back(z);
    }
    Value scalar = tensor::ExtractOp::create(b, loc, predTensor, idx);
    Type et = scalar.getType();
    Value zero = arith::ConstantOp::create(b, loc, et, b.getFloatAttr(et, 0.0));
    return arith::CmpFOp::create(b, loc, arith::CmpFPredicate::OGT, scalar,
                                 zero);
  }

  // tessera.control_for → scf.for. `Malformed` ops (missing/invalid attrs) are
  // reported; `Skipped` ops (a form this pass can't lower CORRECTLY yet — see
  // the payload note) are left untouched for the CF0 guard / a later decoder.
  Outcome lowerControlFor(Operation *op) {
    OpBuilder b(op);
    Location loc = op->getLoc();

    auto bodySym = op->getAttrOfType<FlatSymbolRefAttr>("body");
    auto startA = op->getAttrOfType<IntegerAttr>("start");
    auto stopA = op->getAttrOfType<IntegerAttr>("stop");
    auto stepA = op->getAttrOfType<IntegerAttr>("step");
    if (!bodySym || !startA || !stopA || !stepA)
      return Outcome::Malformed;

    SmallVector<Value> operands(op->getOperands().begin(),
                                op->getOperands().end());
    int64_t n = static_cast<int64_t>(operands.size());

    // Determine which operands are loop-carried (become iter_args).
    SmallVector<int64_t> carriedPos;
    if (auto idxA = op->getAttrOfType<IntegerAttr>("carry_arg_index")) {
      int64_t idx = idxA.getInt();
      if (idx < 0 || idx >= n)
        return Outcome::Malformed;
      carriedPos.push_back(idx);
    } else {
      for (int64_t i = 0; i < n; ++i)
        carriedPos.push_back(i);
    }
    if (static_cast<int64_t>(op->getNumResults()) !=
        static_cast<int64_t>(carriedPos.size()))
      return Outcome::Malformed;

    // The executable-PAYLOAD form (Apple run_graph ABI): the real body is
    // encoded in body_opcodes/body_in0/... and @body is a CARRY-ONLY stub —
    // the loop-invariant captures live in the payload, not in @body's
    // signature. Forwarding the captures to func.call @body would build a
    // malformed call (e.g. a 2-arg call to a 1-arg @loop_body). We can't lower
    // this to scf.for without decoding the payload into real body ops, so leave
    // it for the CF0 guard (and the CF3/CF4 payload decoder).
    if (op->getAttr("body_opcodes"))
      return Outcome::Skipped;

    // Defensive sibling of the payload check: only lower when @body's declared
    // arity matches the call we would build (every operand forwarded in order).
    // A carry-only stub (arity 1) against an n>1 operand list is the payload
    // form above; skip rather than emit an ill-typed call.
    if (auto *callee = SymbolTable::lookupNearestSymbolFrom(op, bodySym)) {
      if (auto fn = dyn_cast<func::FuncOp>(callee)) {
        if (static_cast<int64_t>(fn.getFunctionType().getNumInputs()) != n)
          return Outcome::Skipped;
      }
    }

    // Loop bounds as index constants.
    Value lb = arith::ConstantIndexOp::create(b, loc, startA.getInt());
    Value ub = arith::ConstantIndexOp::create(b, loc, stopA.getInt());
    Value step = arith::ConstantIndexOp::create(b, loc, stepA.getInt());

    SmallVector<Value> iterInits;
    for (int64_t p : carriedPos)
      iterInits.push_back(operands[p]);

    auto forOp = scf::ForOp::create(b, loc, lb, ub, step, iterInits);
    {
      OpBuilder::InsertionGuard g(b);
      b.setInsertionPointToStart(forOp.getBody());

      // Reassemble the @body call args in ORIGINAL operand order: a carried
      // position takes its iter_arg; an invariant capture takes the original
      // (loop-dominating) value.
      SmallVector<int64_t> posToIter(n, -1);
      for (size_t k = 0; k < carriedPos.size(); ++k)
        posToIter[carriedPos[k]] = static_cast<int64_t>(k);

      SmallVector<Value> callArgs;
      for (int64_t p = 0; p < n; ++p) {
        if (posToIter[p] >= 0)
          callArgs.push_back(forOp.getRegionIterArg(posToIter[p]));
        else
          callArgs.push_back(operands[p]);
      }

      // Result types = the carried (iter_arg) types, in order.
      SmallVector<Type> resTypes;
      for (int64_t p : carriedPos)
        resTypes.push_back(operands[p].getType());

      auto call = func::CallOp::create(b, loc, bodySym.getValue(), resTypes,
                                       callArgs);
      scf::YieldOp::create(b, loc, call.getResults());
    }

    op->replaceAllUsesWith(forOp.getResults());
    op->erase();
    return Outcome::Lowered;
  }

  // tessera.control_if → scf.if. flag = operands[flag_arg_index] (a predicate
  // tensor; flag[0] > 0 selects `then`). Both branches are kept as func.calls
  // over the NON-flag data operands (original order), returning the merged
  // result type(s). The executable-payload form (then_opcodes/else_opcodes) and
  // any signature-mismatch stub are skipped.
  Outcome lowerControlIf(Operation *op) {
    OpBuilder b(op);
    Location loc = op->getLoc();

    auto thenSym = op->getAttrOfType<FlatSymbolRefAttr>("then_branch");
    auto elseSym = op->getAttrOfType<FlatSymbolRefAttr>("else_branch");
    auto flagA = op->getAttrOfType<IntegerAttr>("flag_arg_index");
    if (!thenSym || !elseSym || !flagA)
      return Outcome::Malformed;

    SmallVector<Value> operands(op->getOperands().begin(),
                                op->getOperands().end());
    int64_t n = static_cast<int64_t>(operands.size());
    int64_t flagIdx = flagA.getInt();
    if (flagIdx < 0 || flagIdx >= n)
      return Outcome::Malformed;

    if (op->getAttr("then_opcodes") || op->getAttr("else_opcodes"))
      return Outcome::Skipped;

    SmallVector<Value> callArgs;
    SmallVector<Type> argTypes;
    for (int64_t p = 0; p < n; ++p)
      if (p != flagIdx) {
        callArgs.push_back(operands[p]);
        argTypes.push_back(operands[p].getType());
      }
    SmallVector<Type> resTypes(op->getResultTypes().begin(),
                               op->getResultTypes().end());

    if (!calleeMatches(op, thenSym, argTypes, resTypes) ||
        !calleeMatches(op, elseSym, argTypes, resTypes))
      return Outcome::Skipped;

    Value cond = extractPredicateI1(b, loc, operands[flagIdx]);
    // With non-empty result types the builder creates both blocks WITHOUT
    // terminators — we add the func.call + scf.yield to each.
    auto ifOp = scf::IfOp::create(b, loc, resTypes, cond,
                                  /*withElseRegion=*/true);
    {
      OpBuilder::InsertionGuard g(b);
      b.setInsertionPointToStart(ifOp.thenBlock());
      auto call = func::CallOp::create(b, loc, thenSym.getValue(), resTypes,
                                       callArgs);
      scf::YieldOp::create(b, loc, call.getResults());
    }
    {
      OpBuilder::InsertionGuard g(b);
      b.setInsertionPointToStart(ifOp.elseBlock());
      auto call = func::CallOp::create(b, loc, elseSym.getValue(), resTypes,
                                       callArgs);
      scf::YieldOp::create(b, loc, call.getResults());
    }
    op->replaceAllUsesWith(ifOp.getResults());
    op->erase();
    return Outcome::Lowered;
  }

  // tessera.control_while → bounded scf.while. carry = operands[carry_arg_index];
  // the loop state is (counter : index, carry). The before region computes
  // `(i < max_iters) && (cond(carry)[0] > 0)`; the after region runs
  // `carry = body(carry)` and increments the counter. cond/body kept as
  // func.calls. Payload (body_opcodes/cond_opcodes) / signature-mismatch forms
  // are skipped.
  Outcome lowerControlWhile(Operation *op) {
    OpBuilder b(op);
    Location loc = op->getLoc();

    auto bodySym = op->getAttrOfType<FlatSymbolRefAttr>("body");
    auto condSym = op->getAttrOfType<FlatSymbolRefAttr>("cond");
    auto idxA = op->getAttrOfType<IntegerAttr>("carry_arg_index");
    auto maxA = op->getAttrOfType<IntegerAttr>("max_iters");
    if (!bodySym || !condSym || !idxA || !maxA)
      return Outcome::Malformed;

    SmallVector<Value> operands(op->getOperands().begin(),
                                op->getOperands().end());
    int64_t n = static_cast<int64_t>(operands.size());
    int64_t carryIdx = idxA.getInt();
    if (carryIdx < 0 || carryIdx >= n || op->getNumResults() != 1)
      return Outcome::Malformed;

    if (op->getAttr("body_opcodes") || op->getAttr("cond_opcodes"))
      return Outcome::Skipped;

    Value carryInit = operands[carryIdx];
    Type carryTy = carryInit.getType();
    if (op->getResult(0).getType() != carryTy)
      return Outcome::Malformed;

    // @body must be (carry) -> carry; @cond must be (carry) -> predicate tensor.
    // We must know @cond's result type to build the call, so an unresolved /
    // mismatched cond is skipped (left for the guard).
    if (!calleeMatches(op, bodySym, {carryTy}, {carryTy}))
      return Outcome::Skipped;
    auto condFn = dyn_cast_or_null<func::FuncOp>(
        SymbolTable::lookupNearestSymbolFrom(op, condSym));
    if (!condFn)
      return Outcome::Skipped;
    FunctionType condFt = condFn.getFunctionType();
    if (condFt.getNumInputs() != 1 || condFt.getInput(0) != carryTy ||
        condFt.getNumResults() != 1)
      return Outcome::Skipped;
    Type predTy = condFt.getResult(0);

    Value c1 = arith::ConstantIndexOp::create(b, loc, 1);
    Value maxV = arith::ConstantIndexOp::create(b, loc, maxA.getInt());
    Value i0 = arith::ConstantIndexOp::create(b, loc, 0);
    Type idxTy = b.getIndexType();

    SmallVector<Type> stateTys{idxTy, carryTy};
    SmallVector<Value> inits{i0, carryInit};
    SmallVector<Location> locs(stateTys.size(), loc);
    auto whileOp = scf::WhileOp::create(b, loc, stateTys, inits);

    {
      OpBuilder::InsertionGuard g(b);
      Block *before = b.createBlock(&whileOp.getBefore());
      before->addArguments(stateTys, locs);
      b.setInsertionPointToStart(before);
      Value i = before->getArgument(0);
      Value c = before->getArgument(1);
      Value within = arith::CmpIOp::create(b, loc, arith::CmpIPredicate::ult, i,
                                           maxV);
      Type i1Ty = b.getI1Type();
      // SHORT-CIRCUIT the bound: only evaluate @cond when still within
      // max_iters, so an always-true condition is invoked at most max_iters
      // times (never the extra time at i == max_iters). arith.andi is eager, so
      // the bound check must gate the @cond call via an scf.if, not an &&.
      auto contIf = scf::IfOp::create(b, loc, TypeRange{i1Ty}, within,
                                      /*withElseRegion=*/true);
      {
        OpBuilder::InsertionGuard g2(b);
        b.setInsertionPointToStart(contIf.thenBlock());
        auto condCall = func::CallOp::create(b, loc, condSym.getValue(),
                                             TypeRange{predTy}, ValueRange{c});
        Value p = extractPredicateI1(b, loc, condCall.getResult(0));
        scf::YieldOp::create(b, loc, ValueRange{p});
      }
      {
        OpBuilder::InsertionGuard g2(b);
        b.setInsertionPointToStart(contIf.elseBlock());
        Value f = arith::ConstantOp::create(b, loc, b.getBoolAttr(false));
        scf::YieldOp::create(b, loc, ValueRange{f});
      }
      scf::ConditionOp::create(b, loc, contIf.getResult(0), ValueRange{i, c});
    }
    {
      OpBuilder::InsertionGuard g(b);
      Block *after = b.createBlock(&whileOp.getAfter());
      after->addArguments(stateTys, locs);
      b.setInsertionPointToStart(after);
      Value i = after->getArgument(0);
      Value c = after->getArgument(1);
      auto bodyCall = func::CallOp::create(b, loc, bodySym.getValue(),
                                           TypeRange{carryTy}, ValueRange{c});
      Value i2 = arith::AddIOp::create(b, loc, i, c1);
      scf::YieldOp::create(b, loc, ValueRange{i2, bodyCall.getResult(0)});
    }

    op->replaceAllUsesWith(ValueRange{whileOp.getResult(1)});
    op->erase();
    return Outcome::Lowered;
  }

  void runOnOperation() override {
    ModuleOp module = getOperation();
    // Collect first (we erase ops as we lower them).
    SmallVector<Operation *> ctrl;
    module.walk([&](Operation *op) {
      StringRef nm = op->getName().getStringRef();
      if (nm == "tessera.control_for" || nm == "tessera.control_if" ||
          nm == "tessera.control_while")
        ctrl.push_back(op);
    });
    for (Operation *op : ctrl) {
      StringRef nm = op->getName().getStringRef();
      Outcome r;
      if (nm == "tessera.control_for")
        r = lowerControlFor(op);
      else if (nm == "tessera.control_if")
        r = lowerControlIf(op);
      else
        r = lowerControlWhile(op);
      // Skipped (e.g. the executable-payload form) is intentional and silent —
      // the op is left for the CF0 guard / the CF3/CF4 payload decoder.
      if (r == Outcome::Malformed)
        op->emitWarning()
            << nm
            << " left unlowered by control-flow-to-scf (malformed: missing/"
               "invalid attrs or carry/result-count mismatch); the "
               "control-flow target guard will report it";
    }
  }
};

}  // namespace

namespace tessera {
std::unique_ptr<Pass> createLowerControlFlowToSCFPass() {
  return std::make_unique<LowerControlFlowToSCF>();
}
}  // namespace tessera
