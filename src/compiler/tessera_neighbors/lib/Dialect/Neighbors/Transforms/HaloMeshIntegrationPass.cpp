//===- HaloMeshIntegrationPass.cpp — Halo+Mesh integration (Ask 3) --------===//
//
// Joins halo inference (Neighbors dialect) with mesh partitioning
// (Schedule dialect).  After ``DistributionLoweringPass`` has wrapped a
// function body in a ``schedule.mesh.region``, this pass walks the
// region body looking for ``tessera.neighbors.stencil.apply`` ops and:
//
//   1. Inserts a ``tessera.neighbors.halo.exchange`` immediately before
//      each ``stencil.apply`` whose field operand traces back to a
//      function argument (the "sharded input" case).  The exchange
//      carries:
//
//          width      = stencil.halo_width   (from HaloInferPass)
//          mesh.axes  = enclosing mesh.region's axis name
//          source     = the field SSA value
//
//   2. Cross-checks the stencil's per-axis BC against the mesh's axis
//      policy (Architecture Decision #21 — named diagnostic, not a
//      silent no-op).  A periodic BC on a non-periodic mesh axis emits
//      a warning attribute ``mesh.bc_conflict`` carrying the conflict
//      reason; consumers can elevate to error.
//
// This pass is idempotent: the sentinel ``halo.mesh_integrated`` is set
// on every stencil.apply it touches.
//
// Pass argument: ``-tessera-halo-mesh-integration``.
//===----------------------------------------------------------------------===//

#include "tessera/Dialect/Neighbors/Transforms/Passes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/Visitors.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/raw_ostream.h"

using namespace mlir;

namespace tessera {
namespace neighbors {

namespace {

// Walk parent ops to find the enclosing schedule.mesh.region (if any).
// Returns nullptr if the op is not inside one.
static Operation *findEnclosingMeshRegion(Operation *op) {
  for (Operation *p = op->getParentOp(); p; p = p->getParentOp()) {
    if (p->getName().getStringRef() == "schedule.mesh.region")
      return p;
  }
  return nullptr;
}

// Pretty-print a per-axis BC vs. mesh-policy reconciliation conflict.
// Returns a populated string-attr conflict message, or null if no
// conflict is detected.
static StringAttr reconcileBCMesh(MLIRContext *ctx,
                                  ArrayAttr bcModes,
                                  StringRef meshAxisPolicy) {
  if (!bcModes) return nullptr;
  llvm::SmallVector<std::string, 4> issues;
  for (size_t i = 0; i < bcModes.size(); ++i) {
    auto sa = llvm::dyn_cast<StringAttr>(bcModes[i]);
    if (!sa) continue;
    StringRef mode = sa.getValue();
    // The interesting conflict: stencil declares periodic on axis `i`,
    // but the mesh axis is open.  All other combinations are safe:
    //   reflect / dirichlet / neumann work on any mesh policy because
    //   they're locally derivable from the rank-local tile.
    if (mode == "periodic" && meshAxisPolicy != "periodic") {
      llvm::SmallString<128> buf;
      llvm::raw_svector_ostream os(buf);
      os << "axis " << i << ": stencil BC 'periodic' incompatible with "
         << "mesh axis policy '" << meshAxisPolicy << "'";
      issues.push_back(std::string(os.str()));
    }
  }
  if (issues.empty()) return nullptr;
  llvm::SmallString<256> joined;
  for (size_t i = 0; i < issues.size(); ++i) {
    if (i) joined += "; ";
    joined += issues[i];
  }
  return StringAttr::get(ctx, joined);
}

struct HaloMeshIntegrationPass
    : public PassWrapper<HaloMeshIntegrationPass,
                         OperationPass<ModuleOp>> {

  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(HaloMeshIntegrationPass)

  HaloMeshIntegrationPass() = default;
  HaloMeshIntegrationPass(const HaloMeshIntegrationPass &other)
      : PassWrapper(other) {}

  // The mesh's per-axis policy.  Default "open"; pipelines that target a
  // toroidal/wrap-around topology pass "periodic".
  Option<std::string> meshPolicyOpt{
      *this, "mesh-axis-policy",
      llvm::cl::desc("Mesh axis policy for BC reconciliation "
                     "('open' or 'periodic'); default 'open'"),
      llvm::cl::init("open")};

  StringRef getArgument() const final {
    return "tessera-halo-mesh-integration";
  }
  StringRef getDescription() const final {
    return "Insert halo.exchange before sharded stencil.apply ops and "
           "reconcile per-axis BC against the mesh axis policy";
  }

  void runOnOperation() override {
    ModuleOp mod = getOperation();
    MLIRContext *ctx = mod.getContext();
    StringRef meshPolicy = meshPolicyOpt;

    SmallVector<Operation *> applyOps;
    mod.walk([&](Operation *op) {
      if (op->getName().getStringRef() == "tessera.neighbors.stencil.apply"
          && !op->hasAttr("halo.mesh_integrated"))
        applyOps.push_back(op);
    });

    for (Operation *op : applyOps)
      processOne(op, ctx, meshPolicy);
  }

  void processOne(Operation *op, MLIRContext *ctx, StringRef meshPolicy) {
    OpBuilder b(op);
    Location loc = op->getLoc();

    Operation *meshRegion = findEnclosingMeshRegion(op);

    // Trace operand 1 (the field) back through halo.region (if any) to
    // see if it ultimately comes from a func.func argument.
    Value field = op->getOperand(1);
    Value tracedField = field;
    while (Operation *def = tracedField.getDefiningOp()) {
      if (def->getName().getStringRef() == "tessera.neighbors.halo.region") {
        tracedField = def->getOperand(0);
        continue;
      }
      break;
    }
    bool fromArg = llvm::isa<BlockArgument>(tracedField);

    // Read BC info.
    auto bcModes = op->getAttrOfType<ArrayAttr>("stencil.bc.modes");
    auto halo    = op->getAttrOfType<ArrayAttr>("stencil.halo_width");

    // ── 1. Emit halo.exchange before stencil.apply when sharded ──
    if (meshRegion && fromArg && halo) {
      // Build a halo.exchange op carrying width + axes + the field.
      // Conservative: emit it even if the mesh has only one rank — at
      // codegen time the chunk planner can elide a zero-width swap.
      OperationState xSt(loc, "tessera.neighbors.halo.exchange");
      xSt.addOperands({field});
      xSt.addTypes({field.getType()});
      xSt.addAttribute("halo.width", halo);
      // Carry the mesh axis name from the mesh.region.
      if (auto axis = meshRegion->getAttrOfType<StringAttr>("axis"))
        xSt.addAttribute("mesh.axis", axis);
      // Mark provenance so the transport-lowering pass (Gap 5) knows
      // this exchange was inserted by the integration pass.
      xSt.addAttribute("inserted_by",
                       b.getStringAttr("halo-mesh-integration"));
      Operation *xOp = b.create(xSt);
      // Rewire stencil.apply's field operand to the exchange result.
      op->setOperand(1, xOp->getResult(0));
    }

    // ── 2. BC vs mesh-policy reconciliation ──
    if (bcModes) {
      if (StringAttr conflict = reconcileBCMesh(ctx, bcModes, meshPolicy)) {
        op->setAttr("mesh.bc_conflict", conflict);
        op->emitWarning() << "halo-mesh integration: "
                          << conflict.getValue();
      }
    }

    // ── 3. Sentinel ──
    op->setAttr("halo.mesh_integrated", b.getBoolAttr(true));
    op->setAttr("halo.mesh_policy", b.getStringAttr(meshPolicy));
  }
};

} // anonymous namespace

void registerHaloMeshIntegrationPass() {
  PassRegistration<HaloMeshIntegrationPass>();
}

} // namespace neighbors
} // namespace tessera
