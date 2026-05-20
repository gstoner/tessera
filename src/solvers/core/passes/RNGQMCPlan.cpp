//===- RNGQMCPlan.cpp — plan QMC sequences for tessera_rng.* ops ---*- C++ -*-===//
//
// Walks ``tessera_rng.*`` operations and attaches a planned QMC
// (quasi-Monte-Carlo) sampling sequence based on the **enclosing
// solver scope**:
//
//   * If the op is nested inside an op carrying
//     ``tessera.solver.method = "quasi_monte_carlo"`` (or its alias
//     ``"qmc"``), attach
//         ``rng.qmc_plan = {seq="sobol", dim_offset=<N>}``
//     where ``N`` is the next free dimension in a module-scope
//     monotonically-increasing counter.  Sobol is the canonical QMC
//     sequence for Tessera's Monte-Carlo solver lane (low-discrepancy,
//     log-N convergence).
//
//   * If the enclosing scope carries
//     ``tessera.solver.method = "monte_carlo"`` (or no scope tag at
//     all), attach
//         ``rng.qmc_plan = {seq="philox"}``
//     so downstream codegen can fall through to the legacy stochastic
//     RNG path.
//
//   * If the op already carries ``rng.qmc_plan``, leave it alone
//     (explicit user / earlier-pass declaration wins).
//
// This pass is **idempotent**: re-running it on a module with all
// ops already tagged is a no-op.  It runs in the solver pipeline
// between ``RNGLegalize`` (which assigns ``rng.stream_id``) and
// ``RNGStreamAssign`` (which materializes per-stream state).
//
//===----------------------------------------------------------------------===//

#include "SolversPasses.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Operation.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/LLVM.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/CommandLine.h"

using namespace mlir;

namespace {

// Look up the nearest enclosing op (excluding ``rng_op`` itself) that
// carries a ``tessera.solver.method`` attribute and return its string
// value.  Returns an empty StringRef if no such ancestor exists.
static llvm::StringRef findEnclosingMethod(Operation *rng_op) {
  for (Operation *p = rng_op->getParentOp(); p; p = p->getParentOp()) {
    if (auto attr = p->getAttrOfType<StringAttr>("tessera.solver.method"))
      return attr.getValue();
    // Some lowering passes nest the tag on ``func.func`` via a
    // discardable attribute named ``solver_method`` — accept either
    // spelling so the pass is forgiving about which producer set it.
    if (auto attr = p->getAttrOfType<StringAttr>("solver_method"))
      return attr.getValue();
  }
  return {};
}

// Translate the method string into the QMC sequence label this pass
// will attach.  Unknown / missing methods fall back to "philox" so the
// downstream RNG codegen always has a defined target.
static llvm::StringRef sequenceForMethod(llvm::StringRef method) {
  if (method == "quasi_monte_carlo" || method == "qmc" || method == "sobol")
    return "sobol";
  if (method == "halton")
    return "halton";
  return "philox";
}

struct RNGQMCPlanPass
    : PassWrapper<RNGQMCPlanPass, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(RNGQMCPlanPass)

  RNGQMCPlanPass() = default;
  RNGQMCPlanPass(const RNGQMCPlanPass &other) : PassWrapper(other) {}

  Option<std::string> defaultSequence{
      *this, "rng-qmc-default-sequence",
      llvm::cl::desc(
          "Sequence to attach when no enclosing tessera.solver.method "
          "scope is set.  Valid values: philox | sobol | halton."),
      llvm::cl::init(std::string("philox"))};

  StringRef getArgument() const final { return "tessera-rng-qmcplan"; }
  StringRef getDescription() const final {
    return ("Attach rng.qmc_plan to tessera_rng.* ops based on the "
            "enclosing tessera.solver.method scope.");
  }

  void runOnOperation() override {
    ModuleOp mod = getOperation();
    MLIRContext *ctx = mod.getContext();
    OpBuilder b(ctx);

    int64_t dimCounter = 0;
    int64_t plansAttached = 0;

    mod.walk([&](Operation *op) {
      if (!op->getName().getStringRef().starts_with("tessera_rng."))
        return;

      // Honour an existing plan (idempotent / user override).
      if (op->hasAttr("rng.qmc_plan"))
        return;

      llvm::StringRef method = findEnclosingMethod(op);
      llvm::StringRef seq = sequenceForMethod(
          method.empty() ? llvm::StringRef(defaultSequence) : method);

      // Build a dictionary attribute  ``{seq = "sobol", dim_offset = N}``.
      // ``dim_offset`` only applies to low-discrepancy sequences (Sobol,
      // Halton); Philox is a stateless stream so we omit it there to
      // keep the attribute readable.
      SmallVector<NamedAttribute, 2> fields;
      fields.push_back(b.getNamedAttr("seq", b.getStringAttr(seq)));
      if (seq == "sobol" || seq == "halton") {
        fields.push_back(b.getNamedAttr(
            "dim_offset",
            IntegerAttr::get(IntegerType::get(ctx, 64), dimCounter++)));
      }
      op->setAttr("rng.qmc_plan", DictionaryAttr::get(ctx, fields));
      ++plansAttached;
    });

    // Stamp a module-level summary so downstream passes / tests can
    // verify the planner actually ran (vs. having silently no-op'd
    // before this rewrite).  This mirrors how ``RNGLegalize`` marks
    // ops with ``rng.legalized``.
    SmallVector<NamedAttribute, 2> summaryFields;
    summaryFields.push_back(b.getNamedAttr(
        "plans_attached",
        IntegerAttr::get(IntegerType::get(ctx, 64), plansAttached)));
    summaryFields.push_back(b.getNamedAttr(
        "sobol_dims",
        IntegerAttr::get(IntegerType::get(ctx, 64), dimCounter)));
    mod->setAttr("tessera.rng.qmc_plan_summary",
                 DictionaryAttr::get(ctx, summaryFields));
  }
};

} // namespace

namespace tessera {
namespace passes {
std::unique_ptr<Pass> createRNGQMCPlanPass() {
  return std::make_unique<RNGQMCPlanPass>();
}
} // namespace passes
} // namespace tessera
