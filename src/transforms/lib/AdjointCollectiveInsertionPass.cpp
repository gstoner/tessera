//===- AdjointCollectiveInsertionPass.cpp - F5 adjoint collectives -*-C++-*===//
//
// Phase F5 of docs/audit/execution_roadmap.md. Runs **after** the
// `AutodiffPass` (Phase F4) on a `func.func` carrying both
// `tessera.autodiff = "reverse"` and `tessera.weight_sharding = ...`
// attributes. Inserts the appropriate distributed-gradient collective on
// each cotangent SSA value reaching a sharded parameter:
//
//   * Cotangent of a data-parallel-replicated parameter — `reduce_scatter`
//     across the DP mesh axis (sums per-rank gradients into the local
//     shard).
//   * Cotangent of a tensor-parallel-sharded parameter — `all_gather` on
//     the TP mesh axis when the consumer needs the full gradient (rare).
//   * Cotangent of a fully-replicated parameter — `all_reduce` (default).
//
// The AutodiffPass multi-output rewrite (Phase F4) exposes each argument's
// cotangent as an additional function result. This pass identifies those
// trailing results, looks up the associated arg's sharding declaration,
// and inserts the matching `tessera.collective.*` op on each.
//
// Cross-references:
//   * GPUCollectiveInsertionPass.cpp — forward-pass collective insertion.
//   * AutodiffPass.cpp — must run before this pass; provides the
//     `tessera.autodiff.arg_cotangents` attribute and the multi-output
//     return rewrite that gives us SSA handles to each cotangent.
//   * docs/spec/AUTODIFF_SPEC.md §Phase F5
//
//===----------------------------------------------------------------------===//

#include "Tessera/IR/TesseraOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/SmallVector.h"

namespace tessera {

namespace {

constexpr const char *kAutodiffMarker = "tessera.autodiff";
constexpr const char *kArgCotangentsAttr = "tessera.autodiff.arg_cotangents";
constexpr const char *kWeightShardingAttr = "tessera.weight_sharding";
constexpr const char *kCollectivePlanAttr = "tessera.adjoint_collective_plan";
constexpr const char *kCollectiveInsertedAttr = "tessera.adjoint_collective_inserted";

class AdjointCollectiveInsertionPass
    : public mlir::PassWrapper<AdjointCollectiveInsertionPass,
                                mlir::OperationPass<mlir::func::FuncOp>> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(AdjointCollectiveInsertionPass)

  AdjointCollectiveInsertionPass() = default;
  AdjointCollectiveInsertionPass(const AdjointCollectiveInsertionPass &other)
      : mlir::PassWrapper<AdjointCollectiveInsertionPass,
                           mlir::OperationPass<mlir::func::FuncOp>>(other) {}

  llvm::StringRef getArgument() const final {
    return "tessera-adjoint-collective-insertion";
  }

  llvm::StringRef getDescription() const final {
    return "Phase F5 — emit reduce_scatter / all_gather / all_reduce on the "
           "cotangent SSA values produced by AutodiffPass for each sharded "
           "function argument.";
  }

  /// Options — match the forward-pass GPUCollectiveInsertionPass spelling so
  /// pipelines can pass the same `--dp-axis=` / `--tp-axis=` to both.
  Option<std::string> dpAxis{*this, "dp-axis",
                             llvm::cl::desc("mesh axis for data parallelism"),
                             llvm::cl::init("dp")};
  Option<std::string> tpAxis{*this, "tp-axis",
                             llvm::cl::desc("mesh axis for tensor parallelism"),
                             llvm::cl::init("tp")};

  void runOnOperation() override {
    auto func = getOperation();

    // Only run on functions that AutodiffPass already transformed.
    auto markerAttr = func->getAttrOfType<mlir::StringAttr>(kAutodiffMarker);
    if (!markerAttr || markerAttr.getValue() != "reverse")
      return;

    auto cotanArrayAttr =
        func->getAttrOfType<mlir::ArrayAttr>(kArgCotangentsAttr);
    if (!cotanArrayAttr) return;

    auto weightShardingAttr =
        func->getAttrOfType<mlir::DictionaryAttr>(kWeightShardingAttr);
    if (!weightShardingAttr) return;

    // Locate the function's terminator and the cotangent SSA values it
    // returns (appended by AutodiffPass after the original results).
    auto returnOp = mlir::dyn_cast<mlir::func::ReturnOp>(
        func.getBody().front().getTerminator());
    if (!returnOp) return;

    // Original-result count = full result count - number of populated
    // cotangent slots in the array attr.
    int populatedCotans = 0;
    for (auto attr : cotanArrayAttr) {
      if (auto s = mlir::dyn_cast<mlir::StringAttr>(attr)) {
        if (!s.getValue().empty())
          populatedCotans++;
      }
    }
    int origResultCount =
        static_cast<int>(returnOp.getNumOperands()) - populatedCotans;
    if (origResultCount < 0) {
      // The array attr is out of sync with the actual return rewrite;
      // defensive bail-out keeps the IR valid.
      return;
    }

    mlir::OpBuilder builder(&getContext());
    builder.setInsertionPoint(returnOp);
    bool changed = false;

    // Walk argument-cotangent pairs. Each populated entry in the array attr
    // corresponds to one of the trailing return operands (in arg-index
    // order, skipping unpopulated slots).
    int cotanIndex = origResultCount;
    for (auto arg : func.getArguments()) {
      auto argSlotAttr = cotanArrayAttr[arg.getArgNumber()];
      auto slotName = mlir::dyn_cast<mlir::StringAttr>(argSlotAttr);
      if (!slotName || slotName.getValue().empty())
        continue;

      // What sharding does this arg have? Read from
      // `tessera.weight_sharding[arg_<N>]` (mirrors the forward pass's
      // convention).
      std::string shardingKey =
          ("arg_" + llvm::Twine(arg.getArgNumber())).str();
      auto shardingForArg = weightShardingAttr.get(shardingKey);
      if (!shardingForArg) {
        cotanIndex++;
        continue;
      }
      auto kindAttr = mlir::dyn_cast<mlir::StringAttr>(shardingForArg);
      if (!kindAttr) {
        cotanIndex++;
        continue;
      }
      llvm::StringRef kind = kindAttr.getValue();

      // Pull the cotangent SSA value out of the (now-rewritten) return.
      mlir::Value cotanValue = returnOp.getOperand(cotanIndex);

      // Emit the matching `tessera.collective.*` op via the generic
      // OperationState path (matches GPUCollectiveInsertionPass — the
      // collective ops are registered in a separate dialect that the
      // transforms library doesn't link directly).
      mlir::Value reduced;
      auto loc = cotanValue.getLoc();
      mlir::OperationState state(loc, llvm::StringRef());
      state.addOperands(cotanValue);
      state.addTypes(cotanValue.getType());
      if (kind == "dp") {
        state.name = mlir::OperationName("tessera.collective.reduce_scatter",
                                          &getContext());
        state.addAttribute("axis", builder.getStringAttr(dpAxis.getValue()));
        state.addAttribute("op", builder.getStringAttr("sum"));
      } else if (kind == "tp") {
        state.name = mlir::OperationName("tessera.collective.all_gather",
                                          &getContext());
        state.addAttribute("axis", builder.getStringAttr(tpAxis.getValue()));
      } else if (kind == "replicated") {
        state.name = mlir::OperationName("tessera.collective.all_reduce",
                                          &getContext());
        state.addAttribute("axis", builder.getStringAttr(dpAxis.getValue()));
        state.addAttribute("op", builder.getStringAttr("sum"));
      } else {
        cotanIndex++;
        continue;
      }
      state.addAttribute("tessera.collective",
                         mlir::UnitAttr::get(&getContext()));
      auto *collectiveOp = builder.create(state);
      reduced = collectiveOp->getResult(0);

      // Splice the collective's result into the return operand list and
      // record the choice as a per-arg attribute for downstream tools.
      returnOp.setOperand(cotanIndex, reduced);

      auto plan = builder.getStringAttr(
          kind == "dp" ? "reduce_scatter:" + dpAxis.getValue()
          : kind == "tp" ? "all_gather:" + tpAxis.getValue()
                          : "all_reduce");
      func.setArgAttr(arg.getArgNumber(),
                       kCollectivePlanAttr, plan);
      changed = true;
      cotanIndex++;
    }

    if (changed) {
      func->setAttr(kCollectiveInsertedAttr, builder.getUnitAttr());
    }
  }
};

}  // namespace

std::unique_ptr<mlir::Pass> createAdjointCollectiveInsertionPass() {
  return std::make_unique<AdjointCollectiveInsertionPass>();
}

}  // namespace tessera
