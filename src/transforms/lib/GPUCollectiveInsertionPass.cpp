//===- GPUCollectiveInsertionPass.cpp — Phase 4 ───────────────────────────===//
//
// Inserts tessera_collective.reduce_scatter and tessera_collective.all_gather
// ops at
// data-parallel and tensor-parallel mesh boundaries.
//
// The pass reads two sources of truth:
//   1. tessera.distributed_plan attr on the module (emitted by
//      DistributedPlan::to_mlir_attrs() in Python)
//   2. tessera.effect = "memory" on func.func args that carry gradients
//
// Insertion rules (match CLAUDE.md §Phase 4 design contracts):
//
//   Column-parallel linear (weight_sharding = "col_parallel"):
//     → insert collective.reduce_scatter(output, "sum", tp_axis, dim=0)
//       after the op  (each TP rank has a partial sum; scatter one shard/rank)
//
//   Row-parallel linear (weight_sharding = "row_parallel"):
//     → insert collective.all_gather(output, tp_axis, dim=0)
//       after the op  (assemble the split output across TP ranks)
//
//   Data-parallel backward boundary (tessera.effect = "memory" on arg):
//     → insert collective.reduce_scatter(grad, "sum", dp_axis, dim=0)
//       at function exit  (reduce-scatter gradients before optimizer step)
//
// This pass must run AFTER EffectAnnotationPass (to see tessera.effect attrs)
// and AFTER DistributionLoweringPass (to see schedule.mesh.region wrappers).
//
// Registration: --tessera-gpu-collective-insertion
// Options:
//   --dp-axis  mesh axis name for data parallelism   (default "dp")
//   --tp-axis  mesh axis name for tensor parallelism (default "tp")
//
//===----------------------------------------------------------------------===//

#include "Tessera/Transforms/Passes.h"
#include "tessera/Dialect/Collective/IR/CollectiveDialect.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Operation.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/LogicalResult.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "gpu-collective-insertion"

using namespace mlir;

namespace {

//===----------------------------------------------------------------------===//
// Helpers
//===----------------------------------------------------------------------===//

/// Returns true if \p op is a tessera linear/matmul op whose output needs
/// a collective based on its `tessera.weight_sharding` attribute.
static StringRef getWeightSharding(Operation *op) {
  if (auto attr = op->getAttrOfType<StringAttr>("tessera.weight_sharding"))
    return attr.getValue();
  return "";
}

/// Returns the mesh axis tagged on an op or its enclosing region.
static StringRef getMeshAxis(Operation *op, StringRef kind) {
  StringRef attrName = (kind == "tp") ? "tessera.tp_axis" : "tessera.dp_axis";
  if (auto attr = op->getAttrOfType<StringAttr>(attrName))
    return attr.getValue();
  // Walk up to enclosing schedule.mesh.region for the annotation
  Operation *parent = op->getParentOp();
  while (parent) {
    if (auto attr = parent->getAttrOfType<StringAttr>(attrName))
      return attr.getValue();
    parent = parent->getParentOp();
  }
  return "";
}

/// Emit a registered async collective and await its payload.
static Value insertCollective(OpBuilder &b, Operation *insertAfter, Value input,
                              StringRef kind, StringRef meshAxis,
                              int64_t tensorAxis, StringRef reduction) {
  b.setInsertionPointAfter(insertAfter);
  Location loc = insertAfter->getLoc();
  std::string opName = ("tessera_collective." + kind).str();
  OperationState dispatchState(loc, opName);
  dispatchState.addOperands(input);
  dispatchState.addTypes(
      tessera::collective::FutureType::get(b.getContext(), input.getType()));
  dispatchState.addAttribute("mesh_axis", b.getStringAttr(meshAxis));
  dispatchState.addAttribute("tensor_axis", b.getI64IntegerAttr(tensorAxis));
  dispatchState.addAttribute("reduction", b.getStringAttr(reduction));
  dispatchState.addAttribute("tessera.collective.abi", b.getStringAttr("v1"));
  Operation *dispatch = b.create(dispatchState);

  b.setInsertionPointAfter(dispatch);
  OperationState awaitState(loc, "tessera_collective.await");
  awaitState.addOperands(dispatch->getResult(0));
  awaitState.addTypes(input.getType());
  Operation *await = b.create(awaitState);
  LLVM_DEBUG(llvm::dbgs()
             << "[collective-insert] " << kind << " on " << meshAxis
             << " after " << insertAfter->getName() << "\n");
  return await->getResult(0);
}

//===----------------------------------------------------------------------===//
// GPUCollectiveInsertionPass
//===----------------------------------------------------------------------===//

struct GPUCollectiveInsertionPass
    : public PassWrapper<GPUCollectiveInsertionPass, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(GPUCollectiveInsertionPass)

  GPUCollectiveInsertionPass() = default;
  GPUCollectiveInsertionPass(const GPUCollectiveInsertionPass &other)
      : PassWrapper(other) {}

  Option<std::string> dpAxisOpt{
      *this, "dp-axis",
      llvm::cl::desc("Mesh axis name for data parallelism"),
      llvm::cl::init("dp")};
  Option<std::string> tpAxisOpt{
      *this, "tp-axis",
      llvm::cl::desc("Mesh axis name for tensor parallelism"),
      llvm::cl::init("tp")};

  StringRef getArgument() const override {
    return "tessera-gpu-collective-insertion";
  }
  StringRef getDescription() const override {
    return "Insert collective.reduce_scatter / collective.all_gather at "
           "DP/TP mesh boundaries for distributed training";
  }

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<tessera::collective::TesseraCollectiveDialect>();
  }

  void runOnOperation() override {
    ModuleOp module = getOperation();
    OpBuilder b(module.getContext());

    StringRef dpAxis = dpAxisOpt.getValue();
    StringRef tpAxis = tpAxisOpt.getValue();

    unsigned rsInserted = 0, agInserted = 0;

    // ── Pass 1: TP boundaries for linear/matmul ops ───────────────────────
    // Scan for ops tagged with tessera.weight_sharding and insert the right
    // collective at the output boundary.

    SmallVector<Operation *> linearOps;
    module.walk([&](Operation *op) {
      StringRef name = op->getName().getStringRef();
      if ((name.contains("tessera.matmul") || name.contains("linalg.matmul") ||
           name.contains("stablehlo.dot") || name.contains("tessera.linear")) &&
          !getWeightSharding(op).empty())
        linearOps.push_back(op);
    });

    for (Operation *op : linearOps) {
      if (op->getResults().empty()) continue;
      Value output = op->getResult(0);
      StringRef sharding = getWeightSharding(op);
      StringRef axis = getMeshAxis(op, "tp");
      if (axis.empty()) axis = tpAxis;

      if (sharding == "col_parallel") {
        // Column-parallel: each rank has partial dot product → reduce_scatter
        Value reduced = insertCollective(b, op, output, "reduce_scatter", axis,
                                         /*tensorAxis=*/0, "sum");
        output.replaceAllUsesExcept(reduced, reduced.getDefiningOp()->getPrevNode());
        ++rsInserted;
      } else if (sharding == "row_parallel") {
        // Row-parallel: activations split across TP ranks → all_gather
        Value gathered = insertCollective(b, op, output, "all_gather", axis,
                                          /*tensorAxis=*/0, "none");
        output.replaceAllUsesExcept(gathered, gathered.getDefiningOp()->getPrevNode());
        ++agInserted;
      }
    }

    // ── Pass 2: DP backward boundary (grad tensors with "memory" effect) ───
    // For each func.func with a "memory" arg that is written (reduce_sum
    // region mode or explicit tessera.effect = "memory" on the function),
    // insert reduce_scatter at the last use of that grad arg.

    module.walk([&](func::FuncOp fn) {
      auto effectAttr = fn->getAttrOfType<StringAttr>("tessera.effect");
      if (!effectAttr || effectAttr.getValue() != "memory") return;

      // Collect gradient-writing ops FIRST, then insert collectives — mutating
      // the IR while walking it is the iterator-invalidation anti-pattern (the
      // new reduce_scatter happens not to re-match today, but this is brittle).
      SmallVector<Operation *> gradOps;
      fn.walk([&](Operation *op) {
        if (op->getName().getStringRef().contains("tessera.reduce") ||
            op->getAttrOfType<UnitAttr>("tessera.is_grad")) {
          if (!op->getResults().empty())
            gradOps.push_back(op);
        }
      });
      for (Operation *op : gradOps) {
        StringRef axis = getMeshAxis(op, "dp");
        if (axis.empty()) axis = dpAxis;
        Value output = op->getResult(0);
        Value reduced = insertCollective(b, op, output, "reduce_scatter", axis,
                                         /*tensorAxis=*/0, "sum");
        output.replaceAllUsesExcept(reduced, reduced.getDefiningOp()->getPrevNode());
        ++rsInserted;
      }
    });

    // ── Pass 3: Annotate schedule.mesh.region ops with collective counts ───
    // NOTE: these are module-wide totals (every region gets the same count),
    // not per-region tallies — consumers should read them as a module summary.
    module.walk([&](Operation *op) {
      if (op->getName().getStringRef().contains("schedule.mesh.region")) {
        op->setAttr("tessera.collective_rs_count",
                    b.getI64IntegerAttr(rsInserted));
        op->setAttr("tessera.collective_ag_count",
                    b.getI64IntegerAttr(agInserted));
      }
    });

    if (rsInserted + agInserted > 0)
      module.emitRemark("gpu-collective-insertion: inserted ")
          << rsInserted << " reduce_scatter + "
          << agInserted << " all_gather op(s)";
  }
};

} // namespace

namespace tessera {
std::unique_ptr<mlir::Pass> createGPUCollectiveInsertionPass() {
  return std::make_unique<GPUCollectiveInsertionPass>();
}
} // namespace tessera
