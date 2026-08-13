//===- LowerTPPToTargetIR.cpp - bridge TPP ops to the hardware-free Target-IR ==//
//
// Classifies each TPP space-time op against the C ABI Target-IR capability
// table selected by the module-level `tessera.target` attribute
// (cpu | nvidia | amd).  This is
// the same hardware-free Target-IR seam the rest of Tessera uses (Decision
// #19): the pass never emits PTX/HIP directly, it annotates every op with the
// backend + the abstract call the codegen pass will materialise, so TPP
// shares one bottom end with the other solvers instead of owning a bespoke
// sketch. A call is emitted only when the implementation is linkable;
// otherwise the record remains explicitly artifact-only.
//
// Per op we attach a backend and truthful status.  Executable records also
// carry a linkable call symbol and `lowered`; intended-but-unimplemented
// records are `artifact_only` and never advertise a callable symbol.
//
// Op -> primitive mapping:
//   tpp.grad           -> ts_stencil_grad_<backend>      (finite-diff stencil)
//   tpp.stencil.apply  -> ts_stencil_apply_<backend>     (general stencil)
//   tpp.bc.enforce     -> ts_bc_enforce_<backend>        (masked boundary store)
//   tpp.halo.exchange  -> ts_halo_exchange_<backend>     (neighbour comm)
//
// `tpp.bc.enforce` additionally keeps the historical `lowered.bc.masked`
// marker so existing lowering tests / consumers keep working.
//
//===----------------------------------------------------------------------===//

#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Operation.h"
#include "mlir/Pass/Pass.h"

using namespace mlir;

namespace {

// The abstract Target-IR primitive for a TPP op name, or "" if none.
static StringRef primitiveFor(StringRef opName) {
  if (opName.ends_with("tpp.grad"))
    return "ts_stencil_grad";
  if (opName.ends_with("tpp.stencil.apply"))
    return "ts_stencil_apply";
  if (opName.ends_with("tpp.bc.enforce"))
    return "ts_bc_enforce";
  if (opName.ends_with("tpp.halo.exchange"))
    return "ts_halo_exchange";
  return "";
}

static bool hasExecutableImplementation(StringRef primitive,
                                        StringRef backend) {
  return primitive == "ts_stencil_grad" && backend == "cpu";
}

struct LowerTPPToTargetIR
    : public PassWrapper<LowerTPPToTargetIR, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(LowerTPPToTargetIR)

  StringRef getArgument() const final { return "lower-tpp-to-target-ir"; }
  StringRef getDescription() const final {
    return "Classify TPP ops against hardware-free Target-IR capabilities "
           "and emit calls only for linkable implementations";
  }

  void runOnOperation() final {
    ModuleOp m = getOperation();
    MLIRContext *ctx = m.getContext();
    OpBuilder b(ctx);

    StringRef backend = "cpu";
    if (auto t = m->getAttrOfType<StringAttr>("tessera.target"))
      backend = t.getValue();

    bool failed = false;
    m.walk([&](Operation *op) {
      StringRef prim = primitiveFor(op->getName().getStringRef());
      if (prim.empty())
        return;
      op->setAttr("tessera.target_ir.backend", b.getStringAttr(backend));

      StringRef nm = op->getName().getStringRef();
      bool isStencilCompute = nm.ends_with("tpp.grad") ||
                              nm.ends_with("tpp.stencil.apply");
      if (isStencilCompute &&
          (!op->hasAttr("tpp.scheme.normalized") ||
           !op->hasAttr("tpp.spacing.validated"))) {
        op->emitError("tpp: target lowering requires validated scheme, order, "
                      "and spacing; run -tpp-legalize-space-time first");
        failed = true;
        return;
      }

      if (hasExecutableImplementation(prim, backend)) {
        op->setAttr("tessera.target_ir.call",
                    b.getStringAttr((prim + "_" + backend).str()));
        op->setAttr("tessera.target_ir.status",
                    b.getStringAttr("executable"));
        op->setAttr("tessera.target_ir.lowered", b.getUnitAttr());
        op->removeAttr("tessera.target_ir.unavailable_reason");
      } else {
        op->removeAttr("tessera.target_ir.call");
        op->removeAttr("tessera.target_ir.lowered");
        op->setAttr("tessera.target_ir.status",
                    b.getStringAttr("artifact_only"));
        op->setAttr("tessera.target_ir.unavailable_reason",
                    b.getStringAttr("no_linkable_target_implementation"));
      }

      // Stencil-family ops route through the D1 arbiter (op-kind "tpp_stencil",
      // emit/tpp_candidates.py) — the arbiter owns candidate selection per
      // (op, target) instead of the pass hard-wiring one symbol.
      if (nm.ends_with("tpp.grad") || nm.ends_with("tpp.stencil.apply") ||
          nm.ends_with("tpp.div"))
        op->setAttr("tessera.target_ir.arbiter_op",
                    b.getStringAttr("tpp_stencil"));

      // Preserve the historical marker for tpp.bc.enforce lowering.
      if (op->getName().getStringRef().ends_with("tpp.bc.enforce"))
        op->setAttr("lowered.bc.masked", b.getUnitAttr());
    });
    if (failed)
      signalPassFailure();
  }
};

} // namespace

std::unique_ptr<Pass> createLowerTPPToTargetIRPass() {
  return std::make_unique<LowerTPPToTargetIR>();
}
