//===- OptimizerShardPass.cpp — ZeRO-2 optimizer state partitioning ------*- C++ -*-===//
//
// Implements ZeRO stage-2 partitioning: momentum and variance (optimizer state)
// are evenly divided across the DP mesh axis.
//
// For each tessera.optimizer.* op or function arg tagged
// tessera_sr.optimizer_state = "momentum" | "variance":
//
//   tessera_sr.sharded         — UnitAttr (marks as partitioned)
//   tessera_sr.shard_axis      — axis name (default "dp")
//   tessera_sr.shard_rank      — which slice this rank holds (set to * for IR)
//   tessera_sr.zero_stage      — 2 (int64)
//   tessera_sr.partition_count — num_dp_ranks (int64)
//
// On the module:
//   tessera_sr.zero_stage = 2
//   tessera_sr.dp_axis    = "dp"
//   tessera_sr.num_dp_ranks = N
//
// Configuration contract (TILE-LATENT-DEFECTS-2026-09-27). The one producer is
// Python `ZeROConfig.to_ir_attr()` (`compiler/solver_config.py`), which emits
// the module attribute
//
//   tessera_sr.zero_config = {stage = S, dp_axis = "A", num_ranks = N}
//
// and that dictionary is what this pass reads. It used to read
// `tessera.num_dp_ranks` / `tessera.dp_axis`, which nothing produces, so a
// user's ZeRO configuration never reached the pass and it silently sharded with
// its option defaults (num_ranks = 1). Stage, axis and partition count are
// semantic keys (Decision #21a): they decide which slice of optimizer state
// each rank owns. So the pass takes them from the dictionary, or from all
// three explicit `--tessera-optimizer-shard` options, and never defaults them:
//
//   * a malformed dictionary (missing/mistyped key, stage outside {1,2,3},
//     num_ranks < 1, empty axis)                   -> SR_ZERO_CONFIG_MALFORMED
//   * an explicit option that disagrees with it    -> SR_ZERO_CONFIG_CONFLICT
//   * a `tessera.distributed_plan` mesh whose size for the axis disagrees with
//     num_ranks (the count is derivable there, #30) -> SR_ZERO_CONFIG_CONFLICT
//   * optimizer state to shard but no configuration -> SR_ZERO_CONFIG_MISSING
//
// A module with no configuration and nothing to shard is left untouched: the
// pass records no ZeRO facts it was never given.
//
//===----------------------------------------------------------------------===//

#include "tessera/sr/Passes.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"
#include <optional>

using namespace mlir;

namespace {

struct OptimizerShardPass
    : public PassWrapper<OptimizerShardPass, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(OptimizerShardPass)

  OptimizerShardPass() = default;
  OptimizerShardPass(const OptimizerShardPass& other)
      : PassWrapper<OptimizerShardPass, OperationPass<ModuleOp>>(other) {}

  // CLI form of the configuration. The init values are placeholders, never
  // used as defaults: without `tessera_sr.zero_config` all three must be given
  // explicitly (hasValue()), and with it any given option must agree.
  Option<std::string> dpAxis{
      *this, "dp-axis",
      llvm::cl::desc("Data-parallel mesh axis name"),
      llvm::cl::init(std::string("dp"))};

  Option<int> numDPRanks{
      *this, "num-dp-ranks",
      llvm::cl::desc("Number of data-parallel ranks for ZeRO partitioning"),
      llvm::cl::init(1)};

  Option<int> zeroStage{
      *this, "zero-stage",
      llvm::cl::desc("ZeRO stage: 1, 2, or 3"),
      llvm::cl::init(2)};

  struct ZeroSettings {
    int64_t stage;
    std::string axis;
    int64_t ranks;
  };

  // Read the Python-emitted dictionary; std::nullopt + diagnostic on any
  // malformed field. Returns an empty optional with `present == false` when
  // the module carries no dictionary.
  std::optional<ZeroSettings> readZeroConfig(ModuleOp mod, bool &present,
                                             bool &ok) {
    present = false;
    ok = true;
    Attribute raw = mod->getAttr("tessera_sr.zero_config");
    if (!raw)
      return std::nullopt;
    present = true;
    auto malformed = [&](const Twine &why) {
      mod.emitError("SR_ZERO_CONFIG_MALFORMED: tessera_sr.zero_config ")
          << why << "; expected {stage = 1|2|3, dp_axis = \"<axis>\", "
          << "num_ranks = N >= 1} as emitted by ZeROConfig.to_ir_attr()";
      ok = false;
      return std::nullopt;
    };
    auto dict = dyn_cast<DictionaryAttr>(raw);
    if (!dict)
      return malformed("is not a dictionary");
    auto stage = dict.getAs<IntegerAttr>("stage");
    auto axis = dict.getAs<StringAttr>("dp_axis");
    auto ranks = dict.getAs<IntegerAttr>("num_ranks");
    if (!stage)
      return malformed("has no integer 'stage'");
    if (!axis)
      return malformed("has no string 'dp_axis'");
    if (!ranks)
      return malformed("has no integer 'num_ranks'");
    int64_t s = stage.getInt();
    if (s < 1 || s > 3)
      return malformed("has stage = " + Twine(s) + " outside {1, 2, 3}");
    if (axis.getValue().empty())
      return malformed("has an empty 'dp_axis'");
    if (ranks.getInt() < 1)
      return malformed("has num_ranks = " + Twine(ranks.getInt()) + " < 1");
    return ZeroSettings{s, axis.getValue().str(), ranks.getInt()};
  }

  StringRef getArgument() const final { return "tessera-optimizer-shard"; }
  StringRef getDescription() const final {
    return "ZeRO-2: partition optimizer states (momentum/variance) across DP "
           "mesh";
  }

  static bool isOptimizerOp(Operation *op) {
    StringRef opName = op->getName().getStringRef();
    return opName.contains("optimizer") || opName.contains("momentum") ||
           opName.contains("variance") || opName.contains("adam") ||
           opName.ends_with("optimizer.shard") ||
           op->hasAttr("tessera_sr.optimizer_state");
  }

  void runOnOperation() override {
    ModuleOp mod = getOperation();
    MLIRContext *ctx = mod.getContext();

    bool present = false, ok = true;
    std::optional<ZeroSettings> config = readZeroConfig(mod, present, ok);
    if (!ok) {
      signalPassFailure();
      return;
    }

    ZeroSettings settings{zeroStage, dpAxis, numDPRanks};
    if (config) {
      // Options may restate the dictionary but never override it.
      auto conflict = [&](StringRef key, const Twine &option,
                          const Twine &attr) {
        mod.emitError("SR_ZERO_CONFIG_CONFLICT: --tessera-optimizer-shard ")
            << key << "=" << option << " disagrees with tessera_sr.zero_config "
            << key << " = " << attr;
        signalPassFailure();
      };
      if (zeroStage.hasValue() && zeroStage != config->stage)
        return conflict("zero-stage", Twine(int(zeroStage)),
                        Twine(config->stage));
      if (dpAxis.hasValue() && dpAxis != config->axis)
        return conflict("dp-axis", dpAxis, config->axis);
      if (numDPRanks.hasValue() && numDPRanks != config->ranks)
        return conflict("num-dp-ranks", Twine(int(numDPRanks)),
                        Twine(config->ranks));
      settings = *config;
    } else {
      bool anyToShard = false;
      mod.walk([&](Operation *op) {
        if (isOptimizerOp(op))
          anyToShard = true;
      });
      if (!anyToShard)
        return; // nothing configured, nothing to shard: record nothing
      if (!zeroStage.hasValue() || !dpAxis.hasValue() ||
          !numDPRanks.hasValue()) {
        mod.emitError("SR_ZERO_CONFIG_MISSING: optimizer state to shard but "
                      "no ZeRO configuration; attach "
                      "tessera_sr.zero_config (ZeROConfig.to_ir_attr()) or "
                      "pass zero-stage, dp-axis and num-dp-ranks explicitly -- "
                      "the partition count is never defaulted");
        signalPassFailure();
        return;
      }
      if (settings.stage < 1 || settings.stage > 3 || settings.ranks < 1 ||
          settings.axis.empty()) {
        mod.emitError("SR_ZERO_CONFIG_MALFORMED: --tessera-optimizer-shard "
                      "requires zero-stage in {1, 2, 3}, num-dp-ranks >= 1 "
                      "and a non-empty dp-axis");
        signalPassFailure();
        return;
      }
    }

    // Derive, don't ask (#30): when the module states its mesh, the partition
    // count for the axis is already known and must agree.
    if (auto plan = mod->getAttrOfType<DictionaryAttr>("tessera.distributed_plan"))
      if (auto mesh = plan.getAs<DictionaryAttr>("mesh"))
        if (auto size = mesh.getAs<IntegerAttr>(settings.axis))
          if (size.getInt() != settings.ranks) {
            mod.emitError("SR_ZERO_CONFIG_CONFLICT: ZeRO num_ranks = ")
                << settings.ranks << " but tessera.distributed_plan mesh axis '"
                << settings.axis << "' has " << size.getInt() << " ranks";
            signalPassFailure();
            return;
          }

    int64_t nRanks = settings.ranks;
    StringRef axis = settings.axis;
    int64_t stage = settings.stage;

    mod.walk([&](Operation *op) {
      if (!isOptimizerOp(op))
        return;

      op->setAttr("tessera_sr.sharded", UnitAttr::get(ctx));
      op->setAttr("tessera_sr.shard_axis", StringAttr::get(ctx, axis));
      op->setAttr("tessera_sr.zero_stage",
                  IntegerAttr::get(IntegerType::get(ctx, 64), stage));
      op->setAttr("tessera_sr.partition_count",
                  IntegerAttr::get(IntegerType::get(ctx, 64), nRanks));

      // Stage 3: also tag parameter tensors.
      if (stage >= 3) {
        op->setAttr("tessera_sr.params_sharded", UnitAttr::get(ctx));
      }
    });

    // Annotate module.
    mod->setAttr("tessera_sr.zero_stage",
                 IntegerAttr::get(IntegerType::get(ctx, 64), stage));
    mod->setAttr("tessera_sr.dp_axis", StringAttr::get(ctx, axis));
    mod->setAttr("tessera_sr.num_dp_ranks",
                 IntegerAttr::get(IntegerType::get(ctx, 64), nRanks));
  }
};

} // namespace

std::unique_ptr<Pass> mlir::tessera::sr::createOptimizerShardPass() {
  return std::make_unique<OptimizerShardPass>();
}
