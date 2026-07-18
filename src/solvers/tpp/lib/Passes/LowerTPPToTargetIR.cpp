//===- LowerTPPToTargetIR.cpp - bridge TPP ops to the hardware-free Target-IR ==//
//
// Dispatches each TPP space-time op to a C ABI Target-IR symbol, selected by
// the module-level `tessera.target` attribute (cpu | nvidia | amd).  This is
// the same hardware-free Target-IR seam the rest of Tessera uses (Decision
// #19): the pass never emits PTX/HIP directly, it annotates every op with the
// backend + the abstract call the codegen pass will materialise, so TPP
// shares one bottom end with the other solvers instead of owning a bespoke
// sketch.
//
// Per op we attach:
//   tessera.target_ir.backend : "cpu" | "nvidia" | "amd"
//   tessera.target_ir.call    : the stage/primitive symbol, e.g.
//                               @ts_stencil_grad_cpu, @ts_halo_exchange_amd
//   tessera.target_ir.lowered : unit marker
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

struct LowerTPPToTargetIR
    : public PassWrapper<LowerTPPToTargetIR, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(LowerTPPToTargetIR)

  StringRef getArgument() const final { return "lower-tpp-to-target-ir"; }
  StringRef getDescription() const final {
    return "Annotate TPP ops with hardware-free Target-IR call symbols "
           "(cpu / nvidia / amd), selected by the module tessera.target";
  }

  void runOnOperation() final {
    ModuleOp m = getOperation();
    MLIRContext *ctx = m.getContext();
    OpBuilder b(ctx);

    StringRef backend = "cpu";
    if (auto t = m->getAttrOfType<StringAttr>("tessera.target"))
      backend = t.getValue();

    m.walk([&](Operation *op) {
      StringRef prim = primitiveFor(op->getName().getStringRef());
      if (prim.empty())
        return;
      op->setAttr("tessera.target_ir.backend", b.getStringAttr(backend));
      op->setAttr("tessera.target_ir.call",
                  b.getStringAttr((prim + "_" + backend).str()));
      op->setAttr("tessera.target_ir.lowered", b.getUnitAttr());

      // Stencil-family ops route through the D1 arbiter (op-kind "tpp_stencil",
      // emit/tpp_candidates.py) — the arbiter owns candidate selection per
      // (op, target) instead of the pass hard-wiring one symbol.
      StringRef nm = op->getName().getStringRef();
      if (nm.ends_with("tpp.grad") || nm.ends_with("tpp.stencil.apply") ||
          nm.ends_with("tpp.div"))
        op->setAttr("tessera.target_ir.arbiter_op",
                    b.getStringAttr("tpp_stencil"));

      // Preserve the historical marker for tpp.bc.enforce lowering.
      if (op->getName().getStringRef().ends_with("tpp.bc.enforce"))
        op->setAttr("lowered.bc.masked", b.getUnitAttr());
    });
  }
};

} // namespace

std::unique_ptr<Pass> createLowerTPPToTargetIRPass() {
  return std::make_unique<LowerTPPToTargetIR>();
}
