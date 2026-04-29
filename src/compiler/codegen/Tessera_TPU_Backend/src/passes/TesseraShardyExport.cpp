//===- TesseraShardyExport.cpp — Translate tessera.shard → sdy.* attrs ----===//
//
// Translates Tessera sharding metadata (attached as `tessera.shard` attributes
// by the Graph/Schedule IR) into Shardy-native IR annotations understood by
// the XLA/StableHLO compilation pipeline:
//
//   tessera.shard = "replicated"
//     → sdy.tensor_sharding = #sdy.sharding<@mesh, [{}, {}, ...]>
//       (all dimensions open/replicated)
//
//   tessera.shard = {kind = "block", axes = ["dp"], dims = [0]}
//     → sdy.tensor_sharding = #sdy.sharding<@mesh, [{"dp"}, {}, ...]>
//       (dim 0 is closed-sharded over mesh axis "dp")
//
//   tessera.shard = {kind = "cyclic", axes = ["dp","tp"], dims = [0,1]}
//     → sdy.tensor_sharding = #sdy.sharding<@mesh, [{"dp"}, {"tp"}, ...]>
//       (cyclic treated as block at the IR level; scheduler handles striping)
//
// The pass also emits one `sdy.mesh` module-level attribute consolidating
// every axis/size pair seen across all `schedule.mesh.define` ops in the
// module.  If no mesh.define ops are found, a default 1-device "global" mesh
// is emitted so downstream tools always have a valid mesh to reference.
//
// Preconditions:
//   - `tessera.shard` attribute must be a StringAttr or DictionaryAttr whose
//     schema matches the Python ShardSpec.to_ir_attr() output.
//   - For per-result sharding, a `tessera.shard_results` ArrayAttr can be
//     attached to multi-result ops; each element is a per-result shard attr.
//
//===----------------------------------------------------------------------===//

#include "tessera/tpu/TesseraTPUPasses.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Operation.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/StringRef.h"

#include <sstream>
#include <string>
#include <vector>

using namespace mlir;

namespace {

// ---------------------------------------------------------------------------
// Helper: parse the tessera.shard attribute string into components.
//
// The Python ShardSpec.to_ir_attr() emits one of:
//   "{tessera.shard = \"replicated\"}"
//   "{tessera.shard = {kind = \"block\", axes = [\"dp\",\"tp\"], dims = [0,1]}}"
//
// We receive the VALUE of the `tessera.shard` key (already extracted from the
// outer dict), which is either:
//   (a) StringAttr "replicated"
//   (b) DictionaryAttr {kind="block"|"cyclic", axes=[...], dims=[...]}
// ---------------------------------------------------------------------------

struct ParsedShard {
  bool replicated = true;
  std::string kind;                     // "block" or "cyclic"
  std::vector<std::string> axes;        // mesh axis names
  std::vector<int64_t> dims;            // logical tensor dimensions
};

static ParsedShard parseTesseraShardAttr(Attribute attr) {
  ParsedShard result;

  // Case (a): StringAttr "replicated"
  if (auto strAttr = attr.dyn_cast<StringAttr>()) {
    result.replicated = true;
    return result;
  }

  // Case (b): DictionaryAttr
  auto dictAttr = attr.dyn_cast<DictionaryAttr>();
  if (!dictAttr) return result; // unknown format → treat as replicated

  result.replicated = false;

  if (auto kindAttr = dictAttr.getAs<StringAttr>("kind"))
    result.kind = kindAttr.getValue().str();

  if (auto axesAttr = dictAttr.getAs<ArrayAttr>("axes")) {
    for (auto elem : axesAttr) {
      if (auto s = elem.dyn_cast<StringAttr>())
        result.axes.push_back(s.getValue().str());
    }
  }

  if (auto dimsAttr = dictAttr.getAs<ArrayAttr>("dims")) {
    for (auto elem : dimsAttr) {
      if (auto i = elem.dyn_cast<IntegerAttr>())
        result.dims.push_back(i.getInt());
    }
  }

  return result;
}

// ---------------------------------------------------------------------------
// Helper: build the sdy.tensor_sharding attribute string.
//
// Shardy sharding syntax (as understood by the XLA Shardy bridge):
//   #sdy.sharding<@mesh_name, [per_dim_sharding, ...]>
//
// where per_dim_sharding is:
//   {}          — dimension is fully open (replicated / unsharded)
//   {"dp"}      — dimension is block-sharded over mesh axis "dp" (closed)
//   {"dp", ?}   — dimension is sharded over "dp" with a sub-axis factor
//
// For now we emit closed shardings (no sub-axis factors) and open dims for
// unpartitioned dimensions.  The tensor rank is inferred from the op's
// result type when available; otherwise we emit the per-dim shardings for
// only the sharded dims (valid when rank is unknown).
// ---------------------------------------------------------------------------

static std::string buildSdySharding(
    const ParsedShard &shard,
    int64_t tensorRank,
    llvm::StringRef meshName) {

  if (shard.replicated || shard.axes.empty()) {
    // All dims open
    std::string open;
    for (int64_t i = 0; i < tensorRank; ++i) {
      if (i > 0) open += ", ";
      open += "{}";
    }
    return std::string("#sdy.sharding<@") + meshName.str() +
           ", [" + open + "]>";
  }

  // Build per-dim sharding array (tensorRank entries)
  std::vector<std::string> dimShardings(
      tensorRank > 0 ? static_cast<size_t>(tensorRank) : shard.dims.size(),
      "{}");

  for (size_t i = 0; i < shard.dims.size() && i < shard.axes.size(); ++i) {
    int64_t dim = shard.dims[i];
    if (dim >= 0 && dim < static_cast<int64_t>(dimShardings.size()))
      dimShardings[dim] = "{\"" + shard.axes[i] + "\"}";
  }

  std::string result = std::string("#sdy.sharding<@") + meshName.str() + ", [";
  for (size_t i = 0; i < dimShardings.size(); ++i) {
    if (i > 0) result += ", ";
    result += dimShardings[i];
  }
  result += "]>";
  return result;
}

// ---------------------------------------------------------------------------
// Helper: infer the rank of a value from its type (0 if not a tensor/memref).
// ---------------------------------------------------------------------------
static int64_t inferRank(Value v) {
  if (!v) return 0;
  if (auto rtt = v.getType().dyn_cast<RankedTensorType>())
    return rtt.getRank();
  if (auto mrt = v.getType().dyn_cast<MemRefType>())
    return mrt.getRank();
  return 0;
}

// ---------------------------------------------------------------------------
// Helper: collect mesh axis/size pairs from schedule.mesh.define ops.
//
// The op has:
//   dims      : I64ArrayAttr — axis sizes
//   axis_names: ArrayAttr of StringAttr — axis names
// ---------------------------------------------------------------------------
static llvm::StringMap<int64_t> collectMeshAxes(ModuleOp mod) {
  llvm::StringMap<int64_t> axes;
  mod.walk([&](Operation *op) {
    if (op->getName().getStringRef() != "schedule.mesh.define")
      return WalkResult::advance();
    auto dims  = op->getAttrOfType<ArrayAttr>("dims");
    auto names = op->getAttrOfType<ArrayAttr>("axis_names");
    if (!dims || !names || dims.size() != names.size())
      return WalkResult::advance();
    for (size_t i = 0; i < dims.size(); ++i) {
      auto nameAttr = names[i].dyn_cast<StringAttr>();
      auto sizeAttr = dims[i].dyn_cast<IntegerAttr>();
      if (nameAttr && sizeAttr)
        axes[nameAttr.getValue()] = sizeAttr.getInt();
    }
    return WalkResult::advance();
  });
  return axes;
}

// ---------------------------------------------------------------------------
// Helper: emit the sdy.mesh module-level attribute string.
//
// Shardy mesh syntax:
//   #sdy.mesh<["dp"=4, "tp"=8, "pp"=2]>
// ---------------------------------------------------------------------------
static std::string buildSdyMesh(const llvm::StringMap<int64_t> &axes) {
  std::string result = "#sdy.mesh<[";
  bool first = true;
  for (auto &kv : axes) {
    if (!first) result += ", ";
    result += "\"" + kv.first().str() + "\"=" + std::to_string(kv.second);
    first = false;
  }
  result += "]>";
  return result;
}

// ===========================================================================
// ShardyExportPass
// ===========================================================================

struct ShardyExportPass
    : public PassWrapper<ShardyExportPass, OperationPass<ModuleOp>> {

  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(ShardyExportPass)

  StringRef getArgument() const override { return "tessera-export-shardy"; }
  StringRef getDescription() const override {
    return "Translate tessera.shard attrs to sdy.tensor_sharding (Shardy IR)";
  }

  void runOnOperation() override {
    ModuleOp mod = getOperation();
    MLIRContext *ctx = mod.getContext();
    OpBuilder builder(ctx);

    // ------------------------------------------------------------------
    // Step 1 — collect all mesh axes from schedule.mesh.define ops and
    //          emit a single consolidated sdy.mesh module attribute.
    // ------------------------------------------------------------------
    llvm::StringMap<int64_t> meshAxes = collectMeshAxes(mod);
    if (meshAxes.empty()) {
      // No explicit mesh → default: one axis "global" of size 1
      meshAxes["global"] = 1;
    }

    const std::string kMeshName = "mesh";
    std::string sdyMeshStr = buildSdyMesh(meshAxes);
    mod->setAttr("sdy.mesh",
                 StringAttr::get(ctx, sdyMeshStr));

    // ------------------------------------------------------------------
    // Step 2 — walk every op that carries tessera.shard / sdy annotations
    //          and replace them with proper sdy.tensor_sharding strings.
    // ------------------------------------------------------------------
    mod.walk([&](Operation *op) -> WalkResult {
      // ---- Handle per-op single shard (single-result ops) ---------------
      if (auto shardAttr = op->getAttr("tessera.shard")) {
        ParsedShard shard = parseTesseraShardAttr(shardAttr);

        // Infer rank from first result if present
        int64_t rank = 0;
        if (op->getNumResults() > 0)
          rank = inferRank(op->getResult(0));

        std::string sdyStr = buildSdySharding(shard, rank, kMeshName);
        op->setAttr("sdy.tensor_sharding", StringAttr::get(ctx, sdyStr));
        op->removeAttr("tessera.shard");
      }

      // ---- Handle multi-result ops with tessera.shard_results -----------
      if (auto shardResultsAttr =
              op->getAttrOfType<ArrayAttr>("tessera.shard_results")) {
        SmallVector<Attribute> sdyAttrs;
        for (size_t i = 0; i < shardResultsAttr.size(); ++i) {
          Attribute elem = shardResultsAttr[i];
          ParsedShard shard = parseTesseraShardAttr(elem);
          int64_t rank = (i < op->getNumResults())
                             ? inferRank(op->getResult(i))
                             : 0;
          std::string sdyStr = buildSdySharding(shard, rank, kMeshName);
          sdyAttrs.push_back(StringAttr::get(ctx, sdyStr));
        }
        op->setAttr("sdy.tensor_sharding_per_value",
                    ArrayAttr::get(ctx, sdyAttrs));
        op->removeAttr("tessera.shard_results");
      }

      // ---- Upgrade stale placeholder shardings emitted by old pass ------
      if (auto oldSharding = op->getAttrOfType<StringAttr>("sdy.tensor_sharding")) {
        llvm::StringRef val = oldSharding.getValue();
        // Replace bare "{sharding = replicated}" placeholder with proper syntax
        if (val == "{sharding = replicated}") {
          int64_t rank = 0;
          if (op->getNumResults() > 0)
            rank = inferRank(op->getResult(0));
          ParsedShard rep;
          rep.replicated = true;
          std::string sdyStr = buildSdySharding(rep, rank, kMeshName);
          op->setAttr("sdy.tensor_sharding", StringAttr::get(ctx, sdyStr));
        }
      }

      return WalkResult::advance();
    });

    // ------------------------------------------------------------------
    // Step 3 — annotate FuncOp argument/result shardings if the function
    //          carries tessera.arg_shardings / tessera.res_shardings attrs.
    //
    //   tessera.arg_shardings = [{tessera.shard = ...}, ...]
    //   → sdy.in_shardings = ["#sdy.sharding<...>", ...]
    //   → sdy.out_shardings = ["#sdy.sharding<...>", ...]
    // ------------------------------------------------------------------
    mod.walk([&](FuncOp func) -> WalkResult {
      // Argument shardings
      if (auto argShardings =
              func->getAttrOfType<ArrayAttr>("tessera.arg_shardings")) {
        SmallVector<Attribute> sdyIn;
        for (size_t i = 0; i < argShardings.size(); ++i) {
          ParsedShard shard = parseTesseraShardAttr(argShardings[i]);
          int64_t rank = 0;
          if (i < func.getNumArguments()) {
            auto argType = func.getArgumentTypes()[i];
            if (auto rtt = argType.dyn_cast<RankedTensorType>())
              rank = rtt.getRank();
          }
          std::string sdyStr = buildSdySharding(shard, rank, kMeshName);
          sdyIn.push_back(StringAttr::get(ctx, sdyStr));
        }
        func->setAttr("sdy.in_shardings", ArrayAttr::get(ctx, sdyIn));
        func->removeAttr("tessera.arg_shardings");
      }

      // Result shardings
      if (auto resShardings =
              func->getAttrOfType<ArrayAttr>("tessera.res_shardings")) {
        SmallVector<Attribute> sdyOut;
        FunctionType ftype = func.getFunctionType();
        for (size_t i = 0; i < resShardings.size(); ++i) {
          ParsedShard shard = parseTesseraShardAttr(resShardings[i]);
          int64_t rank = 0;
          if (i < ftype.getNumResults()) {
            if (auto rtt = ftype.getResult(i).dyn_cast<RankedTensorType>())
              rank = rtt.getRank();
          }
          std::string sdyStr = buildSdySharding(shard, rank, kMeshName);
          sdyOut.push_back(StringAttr::get(ctx, sdyStr));
        }
        func->setAttr("sdy.out_shardings", ArrayAttr::get(ctx, sdyOut));
        func->removeAttr("tessera.res_shardings");
      }

      return WalkResult::advance();
    });
  }
};

} // anonymous namespace

// ===========================================================================
// Factory
// ===========================================================================

namespace tessera {
std::unique_ptr<mlir::Pass> createExportShardyPass() {
  return std::make_unique<ShardyExportPass>();
}
} // namespace tessera
