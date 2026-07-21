
// DistributionLoweringPass.cpp
//
// Converts tessera.shard argument attributes into schedule.mesh.define +
// schedule.mesh.region ops that wrap the function body.
//
// Before:
//   func.func @step(%a: tensor<128x256xbf16>
//                       {tessera.shard = {axes = ["dp"], dims = [0]}}) {
//     %0 = tessera.matmul %a, %b : ...
//     return
//   }
//
// After:
//   func.func @step(%a: tensor<128x256xbf16>) {
//     schedule.mesh.define {dims = [4], axis_names = ["dp"]}
//     schedule.mesh.region {mesh = @dp, axis = "dp"} {
//       %0 = tessera.matmul %a, %b : ...
//       schedule.yield
//     }
//     return
//   }
//
// Pass options
//   --mesh-axes  : comma-separated axis names, e.g. "dp,tp"
//   --mesh-sizes : comma-separated axis sizes, e.g. "4,4"
//
// If a func has no shard attrs AND no pass options are provided, the func is
// left unchanged.

#include "Tessera/Transforms/Passes.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Dialect.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/raw_ostream.h"

using namespace mlir;

namespace {

// Schedule programming-model ops are deliberately generic until their ODS
// dialect is fully wired.  Keep that permissiveness scoped to schedule.*;
// every other dialect remains strict.
class ScheduleDialect final : public Dialect {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(ScheduleDialect)

  explicit ScheduleDialect(MLIRContext *context)
      : Dialect(getDialectNamespace(), context, TypeID::get<ScheduleDialect>()) {
    allowUnknownOperations(true);
  }

  static StringRef getDialectNamespace() { return "schedule"; }
};

struct DistributionLowering
    : public PassWrapper<DistributionLowering, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(DistributionLowering)

  DistributionLowering() = default;
  DistributionLowering(const DistributionLowering &other)
      : PassWrapper(other) {}

  Option<std::string> meshAxesOpt{
      *this, "mesh-axes",
      llvm::cl::desc("Comma-separated mesh axis names (e.g. 'dp,tp')"),
      llvm::cl::init("")};
  Option<std::string> meshSizesOpt{
      *this, "mesh-sizes",
      llvm::cl::desc("Comma-separated mesh axis sizes (e.g. '4,4')"),
      llvm::cl::init("")};

  StringRef getArgument() const override {
    return "tessera-distribution-lowering";
  }
  StringRef getDescription() const override {
    return "Lower tessera.shard attrs to schedule.mesh.define + mesh.region";
  }

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<ScheduleDialect>();
  }

  // Split a comma-separated string into trimmed tokens.
  static SmallVector<std::string> splitComma(StringRef s) {
    SmallVector<std::string> result;
    SmallVector<StringRef> parts;
    s.split(parts, ',', -1, /*KeepEmpty=*/false);
    for (auto p : parts) result.push_back(p.trim().str());
    return result;
  }

  void runOnOperation() override {
    ModuleOp mod = getOperation();
    MLIRContext *ctx = mod.getContext();
    OpBuilder b(ctx);

    // Parse pass-option mesh config.
    SmallVector<std::string> optAxes = splitComma(meshAxesOpt);
    SmallVector<int64_t>     optSizes;
    for (auto &s : splitComma(meshSizesOpt)) {
      int64_t v = 1;
      StringRef(s).getAsInteger(10, v);
      optSizes.push_back(v);
    }
    while (optSizes.size() < optAxes.size()) optSizes.push_back(1);

    mod.walk([&](func::FuncOp func) {
      // Collect tessera.shard axes from function arguments.
      SmallVector<std::string> shardAxes;
      SmallVector<int64_t>     shardSizes;
      for (unsigned i = 0; i < func.getNumArguments(); ++i) {
        auto shardAttr =
            func.getArgAttrOfType<DictionaryAttr>(i, "tessera.shard");
        if (!shardAttr) continue;
        if (auto axAttr = shardAttr.getAs<ArrayAttr>("axes")) {
          for (auto a : axAttr)
            if (auto sa = llvm::dyn_cast<StringAttr>(a))
              shardAxes.push_back(sa.getValue().str());
        }
        if (auto szAttr = shardAttr.getAs<ArrayAttr>("sizes")) {
          for (auto a : szAttr)
            if (auto ia = llvm::dyn_cast<IntegerAttr>(a))
              shardSizes.push_back(ia.getInt());
        }
      }

      // Choose axes + sizes from ONE source so they stay paired. Pass options
      // win only when axes are supplied there; selecting axes and sizes from
      // independent sources (e.g. --mesh-sizes without --mesh-axes) desynced
      // the axis↔size pairing, and the pad loop below silently masked it.
      bool useOpt = !optAxes.empty();
      SmallVector<std::string> &axes  = useOpt ? optAxes  : shardAxes;
      SmallVector<int64_t>     &sizes = useOpt ? optSizes : shardSizes;
      while (sizes.size() < axes.size()) sizes.push_back(1);

      if (axes.empty()) return; // nothing to do

      auto &bodyBlock = func.getBody().front();
      auto *ret       = bodyBlock.getTerminator();
      auto  loc       = func.getLoc();

      // ── 1. Insert schedule.mesh.define at function entry ──────────────────
      b.setInsertionPointToStart(&bodyBlock);
      {
        SmallVector<Attribute> dimAttrs, nameAttrs;
        for (size_t i = 0; i < axes.size(); ++i) {
          dimAttrs.push_back(b.getI64IntegerAttr(sizes[i]));
          nameAttrs.push_back(b.getStringAttr(axes[i]));
        }
        OperationState st(loc, "schedule.mesh.define");
        st.addAttribute("dims",       b.getArrayAttr(dimAttrs));
        st.addAttribute("axis_names", b.getArrayAttr(nameAttrs));
        b.create(st);
      }

      // ── 2. Collect computation ops to move into the mesh.region body ──────
      // Everything in bodyBlock except the newly-created mesh.define and the
      // function terminator will be moved.
      SmallVector<Operation *> toMove;
      for (auto &op : bodyBlock) {
        if (&op == ret) break;
        StringRef n = op.getName().getStringRef();
        if (n == "schedule.mesh.define") continue;
        toMove.push_back(&op);
      }

      // ── 3. Compute escaping values ────────────────────────────────────────
      // Results of moved ops that are used *outside* the set of moved ops must
      // be yielded out of the region and re-bound, otherwise the func.return
      // (and any other external user) would reference a value defined inside
      // the region — a dominance violation.  This is the fix that lets the
      // Schedule layer produce verifier-clean IR (previously XFAIL).
      llvm::SmallPtrSet<Operation *, 8> movedSet(toMove.begin(), toMove.end());
      llvm::SetVector<Value> escaping;
      for (auto *op : toMove)
        for (Value res : op->getResults())
          for (OpOperand &use : res.getUses())
            if (!movedSet.contains(use.getOwner())) {
              escaping.insert(res);
              break;
            }
      SmallVector<Value> yielded(escaping.begin(), escaping.end());
      SmallVector<Type> resultTypes;
      resultTypes.reserve(yielded.size());
      for (Value v : yielded) resultTypes.push_back(v.getType());

      // ── 4. Create schedule.mesh.region (with result types) before return ──
      b.setInsertionPoint(ret);
      Operation *regionOp = nullptr;
      Block *regionBlock = nullptr;
      {
        OperationState st(loc, "schedule.mesh.region");
        // Use the first axis as the logical mesh identifier.
        st.addAttribute("mesh", FlatSymbolRefAttr::get(ctx, axes[0]));
        st.addAttribute("axis", b.getStringAttr(axes[0]));
        st.addTypes(resultTypes);
        Region *rgn = st.addRegion();
        regionBlock = new Block();
        rgn->push_back(regionBlock);
        regionOp = b.create(st);
      }

      // ── 5. Move collected ops into the region body ────────────────────────
      for (auto *op : toMove)
        op->moveBefore(regionBlock, regionBlock->end());

      // ── 6. Terminate the region body with schedule.yield <escaping...> ────
      {
        OpBuilder rb(regionBlock, regionBlock->end());
        OperationState ySt(loc, "schedule.yield");
        ySt.addOperands(yielded);
        rb.create(ySt);
      }

      // ── 7. Re-bind external uses to the region's results ──────────────────
      // Replace uses of each escaping value that live *outside* the region
      // (e.g. func.return) with the corresponding region result.  Uses inside
      // the region — including the schedule.yield we just created — are left
      // untouched.
      Region &meshRegion = regionOp->getRegion(0);
      auto useIsInsideRegion = [&](Operation *user) {
        for (Region *r = user->getParentRegion(); r; r = r->getParentRegion())
          if (r == &meshRegion) return true;
        return false;
      };
      for (auto pair : llvm::enumerate(yielded)) {
        Value oldV = pair.value();
        Value newV = regionOp->getResult(pair.index());
        oldV.replaceUsesWithIf(newV, [&](OpOperand &use) {
          return !useIsInsideRegion(use.getOwner());
        });
      }

      // ── 6. Strip tessera.shard attrs from function arguments ──────────────
      for (unsigned i = 0; i < func.getNumArguments(); ++i)
        func.removeArgAttr(i, "tessera.shard");
    });
  }
};

} // namespace

namespace tessera {
std::unique_ptr<Pass> createDistributionLoweringPass() {
  return std::make_unique<DistributionLowering>();
}
} // namespace tessera
