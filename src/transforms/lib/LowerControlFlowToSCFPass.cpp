// LowerControlFlowToSCFPass.cpp — CF2 control-flow → scf lowering
//
// CF2 of docs/audit/roadmap/archive/CONTROL_FLOW_AND_DEEPSEEK_ACCELERATION_PLAN.md and
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
// loop never trips the guard; anything this pass leaves is still caught
// loudly. The pass now also lowers the supported control_if, bounded
// control_while, and control_scan forms. SAVE scan lowering materializes a
// compact interior carry-state tape; unsupported payload forms remain guarded.

#include "Tessera/Transforms/Passes.h"

#include <algorithm>

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/StringExtras.h"

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
    return "CF2: lower tessera.control_{for,if,while,scan} to structured SCF "
           "(state in iter_args; scan bodies inlined for differentiation), the "
           "portable hardware-free step of the CUDA/ROCm control-flow path.";
  }

  // Outcome of trying to lower one control op.
  enum class Outcome { Lowered, Skipped, Malformed };

  struct ResidualContract {
    StringRef policy = "recompute_all";
    DenseI64ArrayAttr checkpointIndices;
    StringAttr schema;
    StringAttr cfgDigest;
    StringAttr residualDigest;
  };

  FailureOr<ResidualContract> readResidualContract(Operation *op,
                                                    int64_t trip) {
    ResidualContract contract;
    auto policy = op->getAttrOfType<StringAttr>(
        "tessera.autodiff.checkpoint_policy");
    if (!policy)
      return contract;
    contract.policy = policy.getValue();
    if (contract.policy == "recompute")
      contract.policy = "recompute_all";
    if (contract.policy != "recompute_all" && contract.policy != "save" &&
        contract.policy != "hybrid") {
      op->emitError() << "unsupported control-region checkpoint policy '"
                      << policy.getValue()
                      << "'; expected recompute_all, save, or hybrid";
      return failure();
    }

    contract.checkpointIndices = op->getAttrOfType<DenseI64ArrayAttr>(
        "tessera.autodiff.checkpoint_indices");
    if (contract.policy == "recompute_all") {
      if (contract.checkpointIndices && !contract.checkpointIndices.empty()) {
        op->emitError() << "recompute_all control region cannot retain checkpoint "
                          "indices";
        return failure();
      }
      return contract;
    }

    contract.schema = op->getAttrOfType<StringAttr>(
        "tessera.autodiff.residual_schema");
    contract.cfgDigest = op->getAttrOfType<StringAttr>(
        "tessera.structured_cfg.digest");
    contract.residualDigest = op->getAttrOfType<StringAttr>(
        "tessera.autodiff.residual_digest");
    auto validDigest = [](StringAttr digest) {
      return digest && digest.getValue().size() == 64 &&
             llvm::all_of(digest.getValue(), llvm::isHexDigit);
    };
    if (!contract.schema ||
        contract.schema.getValue() != "tessera.region_residual_abi.v1" ||
        !validDigest(contract.cfgDigest) ||
        !validDigest(contract.residualDigest) ||
        !contract.checkpointIndices) {
      op->emitError()
          << contract.policy
          << " control region requires residual_schema=v1, CFG/residual "
             "SHA-256 digests, and explicit checkpoint_indices";
      return failure();
    }

    ArrayRef<int64_t> indices = contract.checkpointIndices.asArrayRef();
    if (!llvm::is_sorted(indices) ||
        std::adjacent_find(indices.begin(), indices.end()) != indices.end() ||
        llvm::any_of(indices, [trip](int64_t index) {
          return index <= 0 || index >= trip;
        })) {
      op->emitError() << "control-region checkpoint_indices must be sorted, "
                        "unique, interior step indices";
      return failure();
    }
    const size_t allInterior = static_cast<size_t>(trip - 1);
    if ((contract.policy == "save" &&
         (allInterior == 0 || indices.size() != allInterior)) ||
        (contract.policy == "hybrid" &&
         (indices.empty() || indices.size() >= allInterior))) {
      op->emitError() << "control-region checkpoint policy '" << contract.policy
                      << "' disagrees with its retained checkpoint set";
      return failure();
    }
    return contract;
  }

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

  // Whether `extractPredicateI1` can reduce this type to an i1. Checked BEFORE
  // any IR is created: a failed lowering is not rolled back, so refusing at the
  // point of use would leave a half-built scf.while behind.
  static bool isReduciblePredicateType(Type type) {
    auto tt = dyn_cast<RankedTensorType>(type);
    Type element = tt ? tt.getElementType() : type;
    // Signedness matters here. `arith.cmpi` requires SIGNLESS operands —
    // feeding it a `ui8` is not merely mis-signed, it does not verify
    // ("'lhs' must be signless-non-zero-bitwidth-integer-like"). So an
    // explicitly signed or unsigned predicate cannot be lowered by this pass
    // at all; refuse it here rather than build invalid IR or silently read
    // `ui8` 255 as -1 through a signed compare.
    if (auto integer = dyn_cast<IntegerType>(element))
      return integer.isSignless();
    return isa<FloatType>(element);
  }

  // Reduce a predicate to an i1: `element[0,..,0] > 0`. Handles a 0-d
  // tensor<f32> (no indices) and a rank-r tensor (first element on each axis),
  // matching the control_if flag / control_while cond `>0` selector contract.
  //
  // Also handles the integer forms, which used to crash: a boolean condition
  // (`tensor<i1>`, or a bare `i1`) is the most natural thing for a user to
  // write, and it reached `getFloatAttr` on an integer type. A non-tensor
  // predicate likewise reached `tensor.extract` on a non-tensor. Callers gate
  // on `isReduciblePredicateType` first, so a null return is unreachable.
  Value extractPredicateI1(OpBuilder &b, Location loc, Value predTensor) {
    Type type = predTensor.getType();
    auto tt = dyn_cast<RankedTensorType>(type);
    Type elementType = tt ? tt.getElementType() : type;
    if (!elementType.isIntOrFloat())
      return {};

    Value scalar = predTensor;
    if (tt) {
      SmallVector<Value> idx;
      Value z = arith::ConstantIndexOp::create(b, loc, 0);
      for (int64_t d = 0; d < tt.getRank(); ++d)
        idx.push_back(z);
      scalar = tensor::ExtractOp::create(b, loc, predTensor, idx);
    }
    Type et = scalar.getType();
    if (et.isInteger(1))
      return scalar;  // already the i1 the caller wants
    if (isa<IntegerType>(et)) {
      Value zero = arith::ConstantOp::create(b, loc, et, b.getIntegerAttr(et, 0));
      return arith::CmpIOp::create(b, loc, arith::CmpIPredicate::sgt, scalar,
                                   zero);
    }
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

    if (!isReduciblePredicateType(operands[flagIdx].getType()))
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
    FailureOr<ResidualContract> residual =
        readResidualContract(op, maxA.getInt());
    if (failed(residual))
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
    if (!isReduciblePredicateType(predTy))
      return Outcome::Skipped;

    Value c1 = arith::ConstantIndexOp::create(b, loc, 1);
    Value maxV = arith::ConstantIndexOp::create(b, loc, maxA.getInt());
    Value i0 = arith::ConstantIndexOp::create(b, loc, 0);
    Type idxTy = b.getIndexType();

    SmallVector<Type> stateTys{idxTy, carryTy};
    SmallVector<Value> inits{i0, carryInit};
    SmallVector<Location> locs(stateTys.size(), loc);
    auto whileOp = scf::WhileOp::create(b, loc, stateTys, inits);
    whileOp->setAttr("tessera.autodiff.max_iters", maxA);
    whileOp->setAttr("tessera.autodiff.checkpoint_policy",
                     b.getStringAttr(residual->policy));
    if (residual->checkpointIndices) {
      whileOp->setAttr("tessera.autodiff.checkpoint_indices",
                       residual->checkpointIndices);
      whileOp->setAttr("tessera.autodiff.residual_schema", residual->schema);
      whileOp->setAttr("tessera.structured_cfg.digest", residual->cfgDigest);
      whileOp->setAttr("tessera.autodiff.residual_digest",
                       residual->residualDigest);
    }

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

  // tessera.control_scan -> one scf.for carrying both recurrent state and the
  // stacked output tensor.  The body function is inlined so region JVP/VJP
  // sees the actual differentiable operations rather than an opaque func.call.
  Outcome lowerControlScan(Operation *op) {
    if (op->getNumOperands() < 2 || op->getNumResults() != 2 ||
        op->getAttr("body_opcodes"))
      return Outcome::Skipped;
    auto bodySym = op->getAttrOfType<FlatSymbolRefAttr>("body");
    auto tripAttr = op->getAttrOfType<IntegerAttr>("trip");
    auto bodyFn = dyn_cast_or_null<func::FuncOp>(
        bodySym ? SymbolTable::lookupNearestSymbolFrom(op, bodySym) : nullptr);
    if (!bodySym || !tripAttr || tripAttr.getInt() <= 0 || !bodyFn ||
        bodyFn.isDeclaration() || !bodyFn.getBody().hasOneBlock())
      return Outcome::Skipped;
    FailureOr<ResidualContract> residual =
        readResidualContract(op, tripAttr.getInt());
    if (failed(residual)) {
      signalPassFailure();
      return Outcome::Malformed;
    }

    auto xsType = dyn_cast<RankedTensorType>(op->getOperand(1).getType());
    auto ysType = dyn_cast<RankedTensorType>(op->getResult(1).getType());
    if (!xsType || !ysType || !xsType.hasStaticShape() ||
        !ysType.hasStaticShape() || xsType.getRank() < 1 ||
        ysType.getRank() < 1 || xsType.getDimSize(0) != tripAttr.getInt() ||
        ysType.getDimSize(0) != tripAttr.getInt())
      return Outcome::Skipped;

    SmallVector<Type> expectedInputs{op->getOperand(0).getType()};
    auto xtType = RankedTensorType::get(xsType.getShape().drop_front(),
                                        xsType.getElementType(),
                                        xsType.getEncoding());
    expectedInputs.push_back(xtType);
    for (Value capture : op->getOperands().drop_front(2))
      expectedInputs.push_back(capture.getType());
    auto yType = RankedTensorType::get(ysType.getShape().drop_front(),
                                       ysType.getElementType(),
                                       ysType.getEncoding());
    if (TypeRange(bodyFn.getArgumentTypes()) != TypeRange(expectedInputs) ||
        bodyFn.getNumResults() != 2 ||
        bodyFn.getResultTypes()[0] != op->getResult(0).getType() ||
        bodyFn.getResultTypes()[1] != yType)
      return Outcome::Skipped;
    auto bodyReturn = dyn_cast<func::ReturnOp>(
        bodyFn.getBody().front().getTerminator());
    if (!bodyReturn || bodyReturn.getNumOperands() != 2)
      return Outcome::Skipped;

    OpBuilder b(op);
    Location loc = op->getLoc();
    Value empty = tensor::EmptyOp::create(
        b, loc, ysType.getShape(), ysType.getElementType());
    auto carryType = dyn_cast<RankedTensorType>(op->getOperand(0).getType());
    const bool materializeResidual =
        residual->policy == "save" || residual->policy == "hybrid";
    if (materializeResidual && (!carryType || !carryType.hasStaticShape())) {
      op->emitError()
          << residual->policy
          << " control_scan requires a statically shaped tensor carry";
      signalPassFailure();
      return Outcome::Malformed;
    }
    Value residualTape;
    RankedTensorType residualTapeType;
    if (materializeResidual) {
      SmallVector<int64_t> tapeShape{
          static_cast<int64_t>(residual->checkpointIndices.size())};
      llvm::append_range(tapeShape, carryType.getShape());
      residualTapeType = RankedTensorType::get(
          tapeShape, carryType.getElementType(), carryType.getEncoding());
      residualTape = tensor::EmptyOp::create(
          b, loc, residualTapeType.getShape(), residualTapeType.getElementType());
    }
    Value lb = arith::ConstantIndexOp::create(b, loc, 0);
    Value ub = arith::ConstantIndexOp::create(b, loc, tripAttr.getInt());
    Value step = arith::ConstantIndexOp::create(b, loc, 1);
    SmallVector<Value> initArgs{op->getOperand(0), empty};
    if (materializeResidual)
      initArgs.push_back(residualTape);
    auto loop = scf::ForOp::create(b, loc, lb, ub, step, initArgs);
    loop->setAttr("tessera.autodiff.checkpoint_policy",
                  b.getStringAttr(residual->policy));
    if (residual->checkpointIndices) {
      loop->setAttr("tessera.autodiff.checkpoint_indices",
                    residual->checkpointIndices);
      loop->setAttr("tessera.autodiff.residual_schema", residual->schema);
      loop->setAttr("tessera.structured_cfg.digest", residual->cfgDigest);
      loop->setAttr("tessera.autodiff.residual_digest",
                    residual->residualDigest);
    }
    if (materializeResidual) {
      loop->setAttr("tessera.autodiff.residual_materialized",
                    b.getBoolAttr(true));
      loop->setAttr("tessera.autodiff.residual_owner",
                    b.getStringAttr("control_scan"));
      loop->setAttr("tessera.autodiff.residual_result_indices",
                    b.getDenseI64ArrayAttr({2}));
      loop->setAttr("tessera.autodiff.residual_primal_iter_arg_indices",
                    b.getDenseI64ArrayAttr({0}));
    }
    {
      OpBuilder::InsertionGuard guard(b);
      b.setInsertionPointToStart(loop.getBody());
      SmallVector<OpFoldResult> xOffsets{loop.getInductionVar()};
      SmallVector<OpFoldResult> xSizes{b.getIndexAttr(1)};
      SmallVector<OpFoldResult> xStrides(xsType.getRank(), b.getIndexAttr(1));
      for (int64_t dim : xsType.getShape().drop_front()) {
        xOffsets.push_back(b.getIndexAttr(0));
        xSizes.push_back(b.getIndexAttr(dim));
      }
      Value xt = tensor::ExtractSliceOp::create(
          b, loc, xtType, op->getOperand(1), xOffsets, xSizes, xStrides);

      IRMapping mapping;
      mapping.map(bodyFn.getArgument(0), loop.getRegionIterArg(0));
      mapping.map(bodyFn.getArgument(1), xt);
      for (auto [argument, capture] : llvm::zip_equal(
               bodyFn.getArguments().drop_front(2),
               op->getOperands().drop_front(2)))
        mapping.map(argument, capture);
      for (Operation &nested : bodyFn.getBody().front().without_terminator())
        b.clone(nested, mapping);
      Value nextCarry = mapping.lookupOrDefault(bodyReturn.getOperand(0));
      Value y = mapping.lookupOrDefault(bodyReturn.getOperand(1));

      SmallVector<OpFoldResult> yOffsets{loop.getInductionVar()};
      SmallVector<OpFoldResult> ySizes{b.getIndexAttr(1)};
      SmallVector<OpFoldResult> yStrides(ysType.getRank(), b.getIndexAttr(1));
      for (int64_t dim : ysType.getShape().drop_front()) {
        yOffsets.push_back(b.getIndexAttr(0));
        ySizes.push_back(b.getIndexAttr(dim));
      }
      Value nextYs = tensor::InsertSliceOp::create(
          b, loc, y, loop.getRegionIterArg(1), yOffsets, ySizes, yStrides);
      SmallVector<Value> yields{nextCarry, nextYs};
      if (materializeResidual) {
        Value nextTape = loop.getRegionIterArg(2);
        for (auto [slot, checkpoint] :
             llvm::enumerate(residual->checkpointIndices.asArrayRef())) {
          Value retainAt = arith::ConstantIndexOp::create(
              b, loc, checkpoint - 1);
          Value retain = arith::CmpIOp::create(
              b, loc, arith::CmpIPredicate::eq, loop.getInductionVar(),
              retainAt);
          auto retainIf = scf::IfOp::create(
              b, loc, TypeRange{residualTapeType}, retain,
              /*withElseRegion=*/true);
          {
            OpBuilder::InsertionGuard retainGuard(b);
            b.setInsertionPointToStart(retainIf.thenBlock());
            SmallVector<OpFoldResult> offsets{
                b.getIndexAttr(static_cast<int64_t>(slot))};
            SmallVector<OpFoldResult> sizes{b.getIndexAttr(1)};
            SmallVector<OpFoldResult> strides(carryType.getRank() + 1,
                                             b.getIndexAttr(1));
            for (int64_t dim : carryType.getShape()) {
              offsets.push_back(b.getIndexAttr(0));
              sizes.push_back(b.getIndexAttr(dim));
            }
            Value retained = tensor::InsertSliceOp::create(
                b, loc, nextCarry, nextTape, offsets, sizes, strides);
            scf::YieldOp::create(b, loc, retained);
          }
          {
            OpBuilder::InsertionGuard retainGuard(b);
            b.setInsertionPointToStart(retainIf.elseBlock());
            scf::YieldOp::create(b, loc, nextTape);
          }
          nextTape = retainIf.getResult(0);
        }
        yields.push_back(nextTape);
      }
      scf::YieldOp::create(b, loc, yields);
    }
    for (auto [oldResult, newResult] :
         llvm::zip_equal(op->getResults(), loop.getResults().take_front(2)))
      oldResult.replaceAllUsesWith(newResult);
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
          nm == "tessera.control_while" || nm == "tessera.control_scan")
        ctrl.push_back(op);
    });
    for (Operation *op : ctrl) {
      StringRef nm = op->getName().getStringRef();
      Outcome r;
      if (nm == "tessera.control_for")
        r = lowerControlFor(op);
      else if (nm == "tessera.control_if")
        r = lowerControlIf(op);
      else if (nm == "tessera.control_while")
        r = lowerControlWhile(op);
      else
        r = lowerControlScan(op);
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
