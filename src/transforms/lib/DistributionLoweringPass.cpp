
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
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/Support/raw_ostream.h"

using namespace mlir;

namespace {

struct DistributionLowering
    : public PassWrapper<DistributionLowering, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(DistributionLowering)

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
            if (auto sa = a.dyn_cast<StringAttr>())
              shardAxes.push_back(sa.getValue().str());
        }
        if (auto szAttr = shardAttr.getAs<ArrayAttr>("sizes")) {
          for (auto a : szAttr)
            if (auto ia = a.dyn_cast<IntegerAttr>())
              shardSizes.push_back(ia.getInt());
        }
      }

      // Prefer pass-option axes; fall back to shard attrs.
      SmallVector<std::string> &axes  = optAxes.empty()  ? shardAxes  : optAxes;
      SmallVector<int64_t>     &sizes = optSizes.empty() ? shardSizes : optSizes;
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

      // ── 3. Create schedule.mesh.region (empty body) before the return ─────
      b.setInsertionPoint(ret);
      Block *regionBlock = nullptr;
      {
        OperationState st(loc, "schedule.mesh.region");
        // Use the first axis as the logical mesh identifier.
        st.addAttribute("mesh", FlatSymbolRefAttr::get(ctx, axes[0]));
        st.addAttribute("axis", b.getStringAttr(axes[0]));
        Region *rgn = st.addRegion();
        regionBlock = new Block();
        rgn->push_back(regionBlock);
        b.create(st);
      }

      // ── 4. Move collected ops into the region body ────────────────────────
      for (auto *op : toMove)
        op->moveBefore(regionBlock, regionBlock->end());

      // ── 5. Terminate the region body with schedule.yield ──────────────────
      {
        OpBuilder rb(regionBlock, regionBlock->end());
        OperationState ySt(loc, "schedule.yield");
        rb.create(ySt);
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
