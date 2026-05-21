//===- HaloTransportLowerPass.cpp — Halo transport (Sub-4) ----------------===//
//
// Lowers a ``tessera.neighbors.halo.exchange`` op into the three-stage
// transport sequence used by every real halo implementation:
//
//   tessera.neighbors.halo.pack      — gather ghost slabs from the local
//                                       tile into a contiguous send buffer
//                                       per (axis, side).
//   tessera.neighbors.halo.transport — async send/recv against the peer
//                                       rank determined by mesh topology
//                                       (mocked in the test harness;
//                                       NCCL/RCCL adapter selected by
//                                       collective dispatch at codegen).
//   tessera.neighbors.halo.unpack    — scatter received slabs back into
//                                       the field's ghost region.
//
// This is the consumer for the `halo.exchange` ops that
// HaloMeshIntegrationPass inserts.  After this pass runs every exchange
// has been resolved into three structured ops the runtime can serve.
//
// Pass argument: ``-tessera-halo-transport-lower``.
//
// Per-axis × per-side semantics: for an N-D halo.exchange with
// ``halo.width = [w0, w1, …]`` the pass emits 2*N triples (one per
// (axis, side) ∈ axes × {"lo", "hi"}).  ``width = 0`` on an axis is
// elided.  ``inserted_by = "halo-transport-lower"`` provenance lets
// downstream passes know who synthesised the ops.
//===----------------------------------------------------------------------===//

#include "tessera/Dialect/Neighbors/Transforms/Passes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/Visitors.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/SmallVector.h"

using namespace mlir;

namespace tessera {
namespace neighbors {

namespace {

struct HaloTransportLowerPass
    : public PassWrapper<HaloTransportLowerPass,
                         OperationPass<ModuleOp>> {

  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(HaloTransportLowerPass)

  StringRef getArgument() const final {
    return "tessera-halo-transport-lower";
  }
  StringRef getDescription() const final {
    return "Lower halo.exchange into pack/transport/unpack triples "
           "(one per axis-side; mock-collective transport for tests)";
  }

  void runOnOperation() override {
    ModuleOp mod = getOperation();

    SmallVector<Operation *> targets;
    mod.walk([&](Operation *op) {
      if (op->getName().getStringRef()
              == "tessera.neighbors.halo.exchange"
          && !op->hasAttr("halo.transport_lowered"))
        targets.push_back(op);
    });

    for (Operation *op : targets)
      lowerOne(op);
  }

  void lowerOne(Operation *op) {
    OpBuilder b(op);
    Location loc = op->getLoc();
    MLIRContext *ctx = op->getContext();

    Value field = op->getOperand(0);
    auto width = op->getAttrOfType<ArrayAttr>("halo.width");
    if (!width) {
      op->emitWarning() << "halo.exchange has no halo.width; skipping "
                           "transport lowering";
      return;
    }
    int64_t rank = static_cast<int64_t>(width.size());

    // Carry mesh.axis if present.
    auto meshAxis = op->getAttrOfType<StringAttr>("mesh.axis");

    // Threaded SSA value: pack/transport/unpack chain produces a new
    // field-shaped tensor that becomes the input for the next side.
    Value threaded = field;
    SmallVector<NamedAttribute, 4> commonAttrs;
    commonAttrs.push_back({
        StringAttr::get(ctx, "inserted_by"),
        StringAttr::get(ctx, "halo-transport-lower")
    });
    if (meshAxis)
      commonAttrs.push_back({StringAttr::get(ctx, "mesh.axis"), meshAxis});

    // For each axis: skip width=0, otherwise emit (lo, hi) pack/transport/
    // unpack triples that thread the field through.  The two sides on
    // an axis are independent of each other so order doesn't matter; we
    // pick lo then hi for readability.
    for (int64_t a = 0; a < rank; ++a) {
      int64_t w = 0;
      if (auto i = llvm::dyn_cast<IntegerAttr>(width[a]))
        w = i.getInt();
      if (w == 0) continue;

      for (StringRef side : {"lo", "hi"}) {
        // ── halo.pack ────────────────────────────────────────────
        OperationState ps(loc, "tessera.neighbors.halo.pack");
        ps.addOperands({threaded});
        ps.addTypes({threaded.getType()});
        ps.addAttribute("axis",  b.getI64IntegerAttr(a));
        ps.addAttribute("side",  StringAttr::get(ctx, side));
        ps.addAttribute("width", b.getI64IntegerAttr(w));
        for (auto &na : commonAttrs) ps.addAttribute(na.getName(), na.getValue());
        Operation *packOp = b.create(ps);
        Value packed = packOp->getResult(0);

        // ── halo.transport ───────────────────────────────────────
        OperationState ts(loc, "tessera.neighbors.halo.transport");
        ts.addOperands({packed});
        ts.addTypes({threaded.getType()});
        ts.addAttribute("axis",  b.getI64IntegerAttr(a));
        ts.addAttribute("side",  StringAttr::get(ctx, side));
        ts.addAttribute("width", b.getI64IntegerAttr(w));
        // Peer rank is mesh-topology-derived; we record the rule the
        // mock-collective will resolve.  Convention:
        //   side == "lo" → peer is rank-1 (left neighbour, axis-periodic)
        //   side == "hi" → peer is rank+1 (right neighbour)
        ts.addAttribute("peer_rule",
                         StringAttr::get(ctx,
                             side == "lo" ? "neg1" : "pos1"));
        for (auto &na : commonAttrs) ts.addAttribute(na.getName(), na.getValue());
        Operation *xferOp = b.create(ts);
        Value received = xferOp->getResult(0);

        // ── halo.unpack ──────────────────────────────────────────
        OperationState us(loc, "tessera.neighbors.halo.unpack");
        us.addOperands({threaded, received});
        us.addTypes({threaded.getType()});
        us.addAttribute("axis",  b.getI64IntegerAttr(a));
        us.addAttribute("side",  StringAttr::get(ctx, side));
        us.addAttribute("width", b.getI64IntegerAttr(w));
        for (auto &na : commonAttrs) us.addAttribute(na.getName(), na.getValue());
        Operation *unpackOp = b.create(us);
        threaded = unpackOp->getResult(0);
      }
    }

    // Replace uses of halo.exchange with the threaded field result and
    // drop the original op.
    op->getResult(0).replaceAllUsesWith(threaded);
    // Mark the *outermost* result-producing op so a re-run is idempotent.
    if (auto *def = threaded.getDefiningOp())
      def->setAttr("halo.transport_lowered", b.getBoolAttr(true));
    op->erase();
  }
};

} // anonymous namespace

void registerHaloTransportLowerPass() {
  PassRegistration<HaloTransportLowerPass>();
}

} // namespace neighbors
} // namespace tessera
