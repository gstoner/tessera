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
    registry.insert<scf::SCFDialect, func::FuncDialect, arith::ArithDialect>();
  }

  StringRef getArgument() const override {
    return "tessera-control-flow-to-scf";
  }
  StringRef getDescription() const override {
    return "CF2: lower tessera.control_for to scf.for (carry in iter_args; body "
           "kept as a func.call), the portable hardware-free step of the "
           "CUDA/ROCm control-flow path. control_if/while → CF2b.";
  }

  // Outcome of trying to lower one control_for.
  enum class Outcome { Lowered, Skipped, Malformed };

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

  void runOnOperation() override {
    ModuleOp module = getOperation();
    SmallVector<Operation *> fors;
    module.walk([&](Operation *op) {
      if (op->getName().getStringRef() == "tessera.control_for")
        fors.push_back(op);
    });
    for (Operation *op : fors) {
      if (lowerControlFor(op) == Outcome::Malformed)
        op->emitWarning("tessera.control_for left unlowered by "
                        "control-flow-to-scf (malformed: missing body / "
                        "start / stop / step or carry/result-count mismatch); "
                        "the control-flow target guard will report it");
      // Skipped (e.g. the executable-payload form) is intentional and silent —
      // the op is left for the CF0 guard / the CF3/CF4 payload decoder.
    }
  }
};

}  // namespace

namespace tessera {
std::unique_ptr<Pass> createLowerControlFlowToSCFPass() {
  return std::make_unique<LowerControlFlowToSCF>();
}
}  // namespace tessera
