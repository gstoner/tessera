//===- ActivationRematerializationPass.cpp - F2 IR remat ------*- C++ -*-===//
//
// Phase F2 (IR-pass form) of docs/spec/AUTODIFF_SPEC.md. The
// `tessera.autodiff.rematerialize` Python surface (numpy-tape) has shipped
// since Phase F2; this is its Graph-IR counterpart, meant to run alongside /
// after AutodiffPass (Phase F4) once the backward graph is materialised in the
// same function.
//
// Contract
// --------
// An op opts into rematerialization by carrying a `tessera.recompute` unit
// attribute (the lowering target of the Python `rematerialize(fn)` /
// `checkpoint` wrapper — every op the wrapper produced is marked). Instead of
// keeping such an op's forward result live all the way across the (much later)
// backward uses, this pass **clones the op immediately before each backward
// consumer** and rewrites that use to the clone. The original op is erased once
// it has no remaining uses. Net effect: the forward activation's live range
// shrinks to almost nothing (recomputed on demand near the consumer) at the
// cost of extra compute — the classic activation-checkpointing trade
// (Decision #10: recompute is budget-guided and only pure ops qualify).
//
// Safety (Decision #10 / #21)
// ---------------------------
//   * Only pure, region-free ops qualify. Two hard gates, each a loud error
//     rather than a silent skip (which would leave a stale `tessera.recompute`
//     marker and a wrong memory model):
//       - nested regions (control flow) → `REMAT_NON_CLONABLE`.
//       - not provably side-effect-free (`mlir::isMemoryEffectFree`) →
//         `REMAT_EFFECTFUL`. Re-executing an effectful op (RNG like dropout,
//         a collective, a store/copy) on the backward path would change program
//         semantics, not merely trade memory for compute. Tessera Graph IR ops
//         are `[Pure]`, so this admits the real activation ops and rejects the
//         effectful ones; an op that does not model its effects is treated as
//         effectful (conservative — we never recompute what we can't prove pure).
//   * Clone placement is always valid without a dominance query: a user `U`
//     uses the recompute op `P`, so `P` dominates `U`; `P`'s operands dominate
//     `P` (SSA); by transitivity they dominate `U`, hence the clone inserted
//     right before `U`. Producer chains are handled by walking recompute ops in
//     reverse program order (consumers before producers), so a whole tagged
//     chain rematerializes together at the final consumer instead of leaving
//     the earlier producer's clone live from the forward block.
//   * `--memory-budget-mb` or a function's `tessera.remat_budget_mb` drives a
//     deterministic liveness-aware global selection when no explicit marker is
//     present. The largest long-lived pure activation intervals are selected
//     until the estimated peak fits. Explicit markers remain authoritative.
//
// Cross-references:
//   * python/tessera/autodiff/rematerialize.py — the Python F2 surface.
//   * AutodiffPass.cpp — emits the backward graph this pass rematerialises into.
//   * docs/spec/AUTODIFF_SPEC.md §Phase F2.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"

#include <algorithm>
#include <limits>
#include <optional>

namespace tessera {

namespace {

constexpr const char *kRecomputeAttr = "tessera.recompute";
constexpr const char *kRematerializedCountAttr = "tessera.rematerialized";
constexpr const char *kRematBudgetAttr = "tessera.remat_budget_mb";
constexpr const char *kRematBudgetBytesAttr = "tessera.remat_budget_bytes";
constexpr const char *kRematBudgetSourceAttr = "tessera.remat_budget_source";
constexpr const char *kRematAutoSelectedAttr = "tessera.remat_auto_selected";
constexpr const char *kRecomputeScopeAttr = "tessera.recompute_scope";
constexpr const char *kAutodiffPhaseAttr = "tessera.autodiff.phase";
constexpr const char *kMeasuredCostAttr = "tessera.remat_cost_ns";
constexpr const char *kPeakBeforeAttr = "tessera.remat_peak_before_bytes";
constexpr const char *kPeakAfterAttr = "tessera.remat_peak_after_bytes";
constexpr const char *kSelectedCostAttr = "tessera.remat_selected_cost_ns";
constexpr const char *kDeviceCapacityAttr =
    "tessera.device_memory_capacity_bytes";
constexpr const char *kDeviceReserveBasisPointsAttr =
    "tessera.device_memory_reserve_basis_points";
constexpr const char *kModelParameterAttr = "tessera.model.parameter";
constexpr const char *kModelParameterBytesBoundAttr =
    "tessera.model.parameter_bytes_bound";
constexpr const char *kModelGradientCopiesAttr =
    "tessera.model_gradient_copies";
constexpr const char *kModelOptimizerStateCopiesAttr =
    "tessera.model_optimizer_state_copies";
constexpr const char *kModelPersistentBytesAttr =
    "tessera.model_persistent_bytes";
constexpr const char *kModelParameterBytesAttr =
    "tessera.model_parameter_bytes";
constexpr const char *kModelStateBytesAttr = "tessera.model_state_bytes";

static bool isBackwardOperation(mlir::Operation *op) {
  auto phase = op->getAttrOfType<mlir::StringAttr>(kAutodiffPhaseAttr);
  return phase && phase.getValue() == "backward";
}

static std::optional<int64_t> checkedAdd(int64_t lhs, int64_t rhs) {
  if (lhs < 0 || rhs < 0 ||
      lhs > std::numeric_limits<int64_t>::max() - rhs)
    return std::nullopt;
  return lhs + rhs;
}

static std::optional<int64_t> checkedMultiply(int64_t lhs, int64_t rhs) {
  if (lhs < 0 || rhs < 0 ||
      (rhs != 0 && lhs > std::numeric_limits<int64_t>::max() / rhs))
    return std::nullopt;
  return lhs * rhs;
}

static std::optional<int64_t> staticShapedBytes(mlir::Type type) {
  auto shaped = mlir::dyn_cast<mlir::ShapedType>(type);
  if (!shaped || !shaped.hasStaticShape())
    return std::nullopt;
  int64_t elements = 1;
  for (int64_t extent : shaped.getShape()) {
    auto product = checkedMultiply(elements, extent);
    if (!product)
      return std::nullopt;
    elements = *product;
  }
  int64_t bits = shaped.getElementType().getIntOrFloatBitWidth();
  if (bits <= 0)
    return std::nullopt;
  return checkedMultiply(elements, (bits + 7) / 8);
}

struct DerivedBudget {
  int64_t budgetBytes;
  int64_t parameterBytes;
  int64_t stateBytes;
};

static std::optional<DerivedBudget>
deriveModelMemoryBudget(mlir::func::FuncOp func, bool &invalid) {
  invalid = false;
  auto capacity =
      func->getAttrOfType<mlir::IntegerAttr>(kDeviceCapacityAttr);
  if (!capacity)
    return std::nullopt;

  int64_t capacityBytes = capacity.getInt();
  int64_t reserveBasisPoints = 1000;
  int64_t gradientCopies = 1;
  int64_t optimizerStateCopies = 2;
  int64_t persistentBytes = 0;
  if (auto attr =
          func->getAttrOfType<mlir::IntegerAttr>(kDeviceReserveBasisPointsAttr))
    reserveBasisPoints = attr.getInt();
  if (auto attr =
          func->getAttrOfType<mlir::IntegerAttr>(kModelGradientCopiesAttr))
    gradientCopies = attr.getInt();
  if (auto attr = func->getAttrOfType<mlir::IntegerAttr>(
          kModelOptimizerStateCopiesAttr))
    optimizerStateCopies = attr.getInt();
  if (auto attr =
          func->getAttrOfType<mlir::IntegerAttr>(kModelPersistentBytesAttr))
    persistentBytes = attr.getInt();

  if (capacityBytes < 0 || reserveBasisPoints < 0 ||
      reserveBasisPoints > 10000 || gradientCopies < 0 ||
      optimizerStateCopies < 0 || persistentBytes < 0) {
    func.emitError()
        << "REMAT_MODEL_BUDGET_INVALID: model-derived memory-budget inputs "
           "must be non-negative and reserve basis points must be <= 10000";
    invalid = true;
    return std::nullopt;
  }

  int64_t parameterBytes = 0;
  for (unsigned index = 0; index < func.getNumArguments(); ++index) {
    if (!func.getArgAttr(index, kModelParameterAttr))
      continue;
    std::optional<int64_t> bytes =
        staticShapedBytes(func.getArgument(index).getType());
    if (!bytes) {
      if (auto bound = func.getArgAttrOfType<mlir::IntegerAttr>(
              index, kModelParameterBytesBoundAttr))
        if (bound.getInt() >= 0)
          bytes = bound.getInt();
    }
    if (!bytes) {
      func.emitError()
          << "REMAT_MODEL_BUDGET_INVALID: model parameter argument " << index
          << " has a dynamic or unsupported type and requires a non-negative "
          << kModelParameterBytesBoundAttr;
      invalid = true;
      return std::nullopt;
    }
    auto total = checkedAdd(parameterBytes, *bytes);
    if (!total) {
      func.emitError()
          << "REMAT_MODEL_BUDGET_INVALID: model parameter byte total "
             "overflows signed i64";
      invalid = true;
      return std::nullopt;
    }
    parameterBytes = *total;
  }

  auto extraCopies = checkedAdd(gradientCopies, optimizerStateCopies);
  auto stateCopies =
      extraCopies ? checkedAdd(*extraCopies, 1) : std::nullopt;
  auto replicatedStateBytes =
      stateCopies ? checkedMultiply(parameterBytes, *stateCopies)
                  : std::nullopt;
  auto stateBytes =
      replicatedStateBytes
          ? checkedAdd(*replicatedStateBytes, persistentBytes)
          : std::nullopt;
  auto retainedBasisPoints =
      checkedMultiply(capacityBytes, 10000 - reserveBasisPoints);
  if (!stateCopies || !stateBytes || !retainedBasisPoints) {
    func.emitError()
        << "REMAT_MODEL_BUDGET_INVALID: model-derived memory-budget "
           "arithmetic overflows signed i64";
    invalid = true;
    return std::nullopt;
  }

  int64_t usableBytes = *retainedBasisPoints / 10000;
  return DerivedBudget{
      std::max<int64_t>(usableBytes - *stateBytes, 0), parameterBytes,
      *stateBytes};
}

static int64_t estimateResultBytes(mlir::Operation *op) {
  int64_t bytes = 0;
  for (mlir::Value result : op->getResults()) {
    auto shaped = mlir::dyn_cast<mlir::ShapedType>(result.getType());
    if (!shaped)
      continue;
    if (!shaped.hasStaticShape()) {
      bytes += 4096; // explicit conservative dynamic-shape planning unit
      continue;
    }
    int64_t elements = 1;
    for (int64_t extent : shaped.getShape()) {
      if (extent > 0 &&
          elements > std::numeric_limits<int64_t>::max() / extent)
        return std::numeric_limits<int64_t>::max();
      elements *= extent;
    }
    int64_t bits = shaped.getElementType().getIntOrFloatBitWidth();
    int64_t elementBytes = bits > 0 ? (bits + 7) / 8 : 1;
    if (elements >
        (std::numeric_limits<int64_t>::max() - bytes) / elementBytes)
      return std::numeric_limits<int64_t>::max();
    bytes += elements * elementBytes;
  }
  return bytes;
}

// Target benchmark ingestion is deliberately an attribute contract rather than
// a target lookup in this shared Graph pass. A benchmark/selector may stamp
// `tessera.remat_cost_ns` on a producer; host-free compilation falls back to a
// stable operation-work estimate. The fallback is only a ranking unit, not a
// latency claim.
static int64_t estimateRecomputeCost(mlir::Operation *op) {
  if (auto measured = op->getAttrOfType<mlir::IntegerAttr>(kMeasuredCostAttr))
    return std::max<int64_t>(measured.getInt(), 1);

  int64_t resultBytes = estimateResultBytes(op);
  int64_t resultElements = 1;
  if (auto shaped = mlir::dyn_cast<mlir::ShapedType>(
          op->getNumResults() ? op->getResult(0).getType() : mlir::Type{})) {
    if (shaped.hasStaticShape()) {
      resultElements = 1;
      for (int64_t extent : shaped.getShape()) {
        if (extent <= 0 ||
            resultElements > std::numeric_limits<int64_t>::max() / extent) {
          resultElements = std::max<int64_t>(resultBytes, 1);
          break;
        }
        resultElements *= extent;
      }
    } else {
      resultElements = std::max<int64_t>(resultBytes, 1);
    }
  }

  llvm::StringRef name = op->getName().getStringRef();
  int64_t multiplier = 1;
  if (name == "tessera.matmul" || name == "tessera.batched_gemm")
    multiplier = 32;
  else if (name == "tessera.softmax" || name == "tessera.layer_norm" ||
           name == "tessera.rms_norm")
    multiplier = 8;
  else if (name == "tessera.exp" || name == "tessera.log" ||
           name == "tessera.gelu" || name == "tessera.silu")
    multiplier = 4;
  if (resultElements >
      std::numeric_limits<int64_t>::max() / multiplier)
    return std::numeric_limits<int64_t>::max();
  return std::max<int64_t>(resultElements * multiplier, 1);
}

struct RematCandidate {
  mlir::Operation *op;
  int64_t begin;
  int64_t end;
  int64_t bytes;
  int64_t recomputeCost;
};

class ActivationRematerializationPass
    : public mlir::PassWrapper<ActivationRematerializationPass,
                                mlir::OperationPass<mlir::func::FuncOp>> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(ActivationRematerializationPass)

  ActivationRematerializationPass() = default;
  ActivationRematerializationPass(const ActivationRematerializationPass &other)
      : mlir::PassWrapper<ActivationRematerializationPass,
                           mlir::OperationPass<mlir::func::FuncOp>>(other) {}

  llvm::StringRef getArgument() const final {
    return "tessera-activation-rematerialization";
  }

  llvm::StringRef getDescription() const final {
    return "Phase F2 (IR form) — rematerialize `tessera.recompute`-tagged pure "
           "activations at their backward consumers to shrink live ranges "
           "(activation checkpointing).";
  }

  Option<int> memoryBudgetMb{
      *this, "memory-budget-mb",
      llvm::cl::desc("recompute memory budget (MB); selects long-lived pure "
                     "activations when explicit markers are absent"),
      llvm::cl::init(0)};

  void runOnOperation() override {
    auto func = getOperation();

    std::optional<int64_t> effectiveBudgetBytes;
    llvm::StringRef budgetSource;
    int64_t effectiveBudgetMb = memoryBudgetMb.getValue();
    if (effectiveBudgetMb > 0) {
      budgetSource = "explicit_cli";
    } else if (auto attr =
                   func->getAttrOfType<mlir::IntegerAttr>(kRematBudgetAttr)) {
      effectiveBudgetMb = attr.getInt();
      if (effectiveBudgetMb > 0)
        budgetSource = "explicit_function";
    }
    if (effectiveBudgetMb > 0) {
      auto bytes = checkedMultiply(effectiveBudgetMb, 1024LL * 1024LL);
      if (!bytes) {
        func.emitError()
            << "REMAT_MODEL_BUDGET_INVALID: explicit memory budget "
               "overflows signed i64 bytes";
        return signalPassFailure();
      }
      effectiveBudgetBytes = bytes;
      func->setAttr(kRematBudgetAttr,
                    mlir::IntegerAttr::get(
                        mlir::IntegerType::get(&getContext(), 32),
                        effectiveBudgetMb));
    } else {
      bool invalid = false;
      if (auto derived = deriveModelMemoryBudget(func, invalid)) {
        effectiveBudgetBytes = derived->budgetBytes;
        budgetSource = "model_device_envelope";
        func->setAttr(
            kModelParameterBytesAttr,
            mlir::IntegerAttr::get(
                mlir::IntegerType::get(&getContext(), 64),
                derived->parameterBytes));
        func->setAttr(
            kModelStateBytesAttr,
            mlir::IntegerAttr::get(mlir::IntegerType::get(&getContext(), 64),
                                   derived->stateBytes));
      } else if (invalid) {
        return signalPassFailure();
      }
    }
    if (effectiveBudgetBytes) {
      func->setAttr(
          kRematBudgetBytesAttr,
          mlir::IntegerAttr::get(mlir::IntegerType::get(&getContext(), 64),
                                 *effectiveBudgetBytes));
      func->setAttr(kRematBudgetSourceAttr,
                    mlir::StringAttr::get(&getContext(), budgetSource));
    }

    // Collect the recompute-tagged ops up-front — we mutate uses / erase ops
    // during the walk, so snapshot first to keep iteration well-defined.
    llvm::SmallVector<mlir::Operation *> recomputeOps;
    func.walk([&](mlir::Operation *op) {
      if (op->hasAttr(kRecomputeAttr))
        recomputeOps.push_back(op);
    });

    // Production-path global selection: the named autodiff pipeline invokes
    // this pass after building the backward graph. A function-level budget now
    // drives a deterministic liveness-aware choice when the frontend did not
    // provide explicit markers. We remove the longest, largest pure activation
    // intervals until the estimated peak fits; the existing clone/sink logic
    // below then realizes those choices.
    if (recomputeOps.empty() && effectiveBudgetBytes) {
      llvm::SmallVector<mlir::Operation *> ordered;
      func.walk([&](mlir::Operation *op) {
        if (op != func.getOperation() && op->getNumResults() > 0)
          ordered.push_back(op);
      });
      llvm::DenseMap<mlir::Operation *, int64_t> ordinal;
      for (auto [index, op] : llvm::enumerate(ordered))
        ordinal[op] = static_cast<int64_t>(index);
      bool hasAutodiffPhases =
          llvm::any_of(ordered, [](mlir::Operation *op) {
            return isBackwardOperation(op);
          });

      llvm::SmallVector<RematCandidate> candidates;
      for (mlir::Operation *op : ordered) {
        if (op->getNumRegions() != 0 || !mlir::isMemoryEffectFree(op) ||
            (hasAutodiffPhases && isBackwardOperation(op)))
          continue;
        int64_t begin = ordinal[op], end = begin;
        bool hasBackwardUse = false;
        for (mlir::Operation *user : op->getUsers()) {
          auto it = ordinal.find(user);
          if (it != ordinal.end()) {
            end = std::max(end, it->second);
            hasBackwardUse |= isBackwardOperation(user);
          }
        }
        int64_t bytes = estimateResultBytes(op);
        if (bytes > 0 && end > begin &&
            (!hasAutodiffPhases || hasBackwardUse))
          candidates.push_back(
              {op, begin, end, bytes, estimateRecomputeCost(op)});
      }

      int64_t budgetBytes = *effectiveBudgetBytes;
      auto estimatedPeak = [&](llvm::ArrayRef<RematCandidate> active) {
        int64_t peak = 0;
        for (int64_t point = 0;
             point < static_cast<int64_t>(ordered.size()); ++point) {
          int64_t live = 0;
          for (const RematCandidate &candidate : active)
            if (candidate.begin <= point && point <= candidate.end) {
              if (candidate.bytes >
                  std::numeric_limits<int64_t>::max() - live)
                live = std::numeric_limits<int64_t>::max();
              else
                live += candidate.bytes;
            }
          peak = std::max(peak, live);
        }
        return peak;
      };
      llvm::SmallVector<RematCandidate> active(candidates);
      llvm::SmallVector<mlir::Operation *> selected;
      int64_t peakBefore = estimatedPeak(active);
      int64_t selectedCost = 0;
      while (!active.empty() && estimatedPeak(active) > budgetBytes) {
        auto best = std::max_element(
            active.begin(), active.end(),
            [](const RematCandidate &lhs, const RematCandidate &rhs) {
              // Maximize memory-pressure relief per nanosecond/work unit.
              // Cross multiplication avoids floating-point instability.
              __int128 lhsBenefit =
                  static_cast<__int128>(lhs.bytes) * (lhs.end - lhs.begin);
              __int128 rhsBenefit =
                  static_cast<__int128>(rhs.bytes) * (rhs.end - rhs.begin);
              __int128 lhsWeighted =
                  lhsBenefit * std::max<int64_t>(rhs.recomputeCost, 1);
              __int128 rhsWeighted =
                  rhsBenefit * std::max<int64_t>(lhs.recomputeCost, 1);
              if (lhsWeighted != rhsWeighted)
                return lhsWeighted < rhsWeighted;
              return lhs.begin > rhs.begin;
            });
        selected.push_back(best->op);
        if (best->recomputeCost >
            std::numeric_limits<int64_t>::max() - selectedCost)
          selectedCost = std::numeric_limits<int64_t>::max();
        else
          selectedCost += best->recomputeCost;
        active.erase(best);
      }
      func->setAttr(kPeakBeforeAttr,
                    mlir::IntegerAttr::get(
                        mlir::IntegerType::get(&getContext(), 64), peakBefore));
      func->setAttr(kPeakAfterAttr,
                    mlir::IntegerAttr::get(
                        mlir::IntegerType::get(&getContext(), 64),
                        estimatedPeak(active)));
      func->setAttr(kSelectedCostAttr,
                    mlir::IntegerAttr::get(
                        mlir::IntegerType::get(&getContext(), 64),
                        selectedCost));
      for (mlir::Operation *op : selected) {
        op->setAttr(kRecomputeAttr, mlir::UnitAttr::get(&getContext()));
        if (hasAutodiffPhases)
          op->setAttr(kRecomputeScopeAttr,
                      mlir::StringAttr::get(&getContext(), "backward"));
        recomputeOps.push_back(op);
      }
      if (!selected.empty())
        func->setAttr(
            kRematAutoSelectedAttr,
            mlir::IntegerAttr::get(
                mlir::IntegerType::get(&getContext(), 64),
                static_cast<int64_t>(selected.size())));
    }
    if (recomputeOps.empty())
      return;

    mlir::OpBuilder builder(&getContext());
    int64_t rematCount = 0;
    bool failed = false;

    // Walk in REVERSE program order — consumers before producers. For a tagged
    // producer chain (%a feeds %b feeds a backward user, both tagged), handling
    // the consumer %b first sinks its clone to the backward user; %a is then
    // seen with %b's clone as its user, so %a's clone lands next to it — the
    // whole chain rematerializes together at the consumer. Forward order would
    // instead leave %a's clone stranded next to %b in the forward block, still
    // live across to the backward, defeating the checkpoint.
    for (mlir::Operation *op : llvm::reverse(recomputeOps)) {
      bool backwardOnly = false;
      if (auto scope =
              op->getAttrOfType<mlir::StringAttr>(kRecomputeScopeAttr))
        backwardOnly = scope.getValue() == "backward";
      // Gate 1: region-free. Cloning a control-flow op is out of scope.
      if (op->getNumRegions() != 0) {
        op->emitError()
            << "REMAT_NON_CLONABLE: op '" << op->getName().getStringRef()
            << "' is tagged " << kRecomputeAttr
            << " but carries nested regions; only pure region-free ops can be "
               "rematerialized";
        failed = true;
        continue;
      }
      // Gate 2: provably side-effect-free. Re-executing an effectful op (RNG,
      // collective, store/copy) on the backward path would change program
      // semantics — recompute trades memory for *compute*, nothing else. An op
      // that doesn't model its effects is conservatively treated as effectful.
      if (!mlir::isMemoryEffectFree(op)) {
        op->emitError()
            << "REMAT_EFFECTFUL: op '" << op->getName().getStringRef()
            << "' is tagged " << kRecomputeAttr
            << " but is not provably side-effect-free; rematerializing it would "
               "re-execute its effects and change program semantics — only pure "
               "ops qualify (Decision #10)";
        failed = true;
        continue;
      }

      // Snapshot the current users (each is a distinct backward consumer). We
      // rewrite one operand-use at a time; cloning per user op keeps the
      // recomputed value adjacent to its consumer.
      llvm::SmallVector<mlir::Operation *> users(op->getUsers().begin(),
                                                 op->getUsers().end());
      // Deduplicate while preserving order — an op may use the value twice.
      llvm::SmallVector<mlir::Operation *> uniqueUsers;
      for (mlir::Operation *u : users) {
        if (u == op)
          continue;
        if (backwardOnly && !isBackwardOperation(u))
          continue;
        if (!llvm::is_contained(uniqueUsers, u))
          uniqueUsers.push_back(u);
      }

      for (mlir::Operation *user : uniqueUsers) {
        // Clone placement is always valid: `user` uses `op`, so `op` dominates
        // `user`; `op`'s operands dominate `op` (SSA); by transitivity they
        // dominate `user`, so the clone inserted right before `user` sees them.
        // No dominance query needed (and none would be stable across the
        // clones we insert into freshly-relocated chain users).
        builder.setInsertionPoint(user);
        mlir::Operation *clone = builder.clone(*op);
        clone->removeAttr(kRecomputeAttr);  // the clone is the materialized use
        clone->removeAttr(kRecomputeScopeAttr);

        // Rewrite this user's operands that reference op's results to the clone.
        for (mlir::OpOperand &use : user->getOpOperands()) {
          mlir::Value used = use.get();
          if (auto res = mlir::dyn_cast<mlir::OpResult>(used))
            if (res.getOwner() == op)
              use.set(clone->getResult(res.getResultNumber()));
        }
        rematCount++;
      }

      // If the original is now fully rematerialized away, erase it.
      if (op->use_empty())
        op->erase();
      else {
        op->removeAttr(kRecomputeAttr);  // partial — clear the marker regardless
        op->removeAttr(kRecomputeScopeAttr);
      }
    }

    if (failed)
      return signalPassFailure();

    if (rematCount > 0) {
      func->setAttr(kRematerializedCountAttr,
                    mlir::IntegerAttr::get(
                        mlir::IntegerType::get(&getContext(), 64), rematCount));
    }
  }
};

}  // namespace

std::unique_ptr<mlir::Pass> createActivationRematerializationPass() {
  return std::make_unique<ActivationRematerializationPass>();
}

}  // namespace tessera
