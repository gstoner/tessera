// DtypeLegalizePass.cpp — C4 (2026-06-23)
//
// Operationalizes Decision #15a (storage dtype ≠ accumulator) as *pass
// ordering*, the way TIRx splits BF16/FP8 ComputeLegalize (early, rewrite math
// to wide-accumulate form) from …StorageLegalize (terminal, pack sub-byte
// values into a container). Two complementary rewrite passes:
//
//   --tessera-compute-legalize  (EARLY)
//     For any op whose `numeric_policy.storage` is reduced-precision and which
//     lacks an accumulator, stamp `numeric_policy.accum` with the natural wide
//     accumulator (fp32 for float storages, int32 for int4/int8). This makes the
//     storage/accum split explicit *before* fusion/codegen, and the result then
//     passes IRContractLegalityPass (DTYPE_LEGALITY_LOWP_WITHOUT_WIDE_ACCUM).
//
//   --tessera-storage-legalize  (TERMINAL)
//     For any op whose `numeric_policy.storage` is sub-byte / block-scaled
//     (fp4_e2m1 / nvfp4 / fp6_* / int4), stamp `tessera.storage_packed` +
//     `tessera.storage_container` recording the byte container the packed values
//     live in. This is the late packing step, run after all compute rewrites.
//
// Both are idempotent and additive (a value-preserving annotation today; the
// container marker becomes a real repack when a low-precision backend consumes
// it). Decision: never silently skip — an op with a reduced-precision storage
// and no recognizable policy is left untouched for IRContractLegalityPass to
// flag, not quietly "legalized".

#include "Tessera/Transforms/Passes.h"

#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/StringSet.h"

using namespace mlir;

namespace {

// Reduced-precision float + low-precision storages whose math must accumulate in
// a wider type (Decision #15a). Mirrors IRContractLegalityPass's sets.
static const llvm::StringSet<> &reducedPrecisionStorages() {
  static const llvm::StringSet<> kSet = {
      "bf16",     "fp16",     "fp8_e4m3", "fp8_e5m2", "fp6_e2m3",
      "fp6_e3m2", "fp4_e2m1", "nvfp4",    "int4",     "int8"};
  return kSet;
}

static bool isIntegerStorage(StringRef s) { return s == "int4" || s == "int8"; }

// Sub-byte / block-scaled storages that need terminal packing into a container.
// (fp8 / int8 are byte-sized — not packed.)
static StringRef packedContainerFor(StringRef storage) {
  if (storage == "fp4_e2m1" || storage == "nvfp4" || storage == "fp6_e2m3" ||
      storage == "fp6_e3m2" || storage == "int4")
    return "int8"; // two/one packed values per byte container (v1 marker)
  return {};
}

static DictionaryAttr policyOf(Operation *op) {
  return op->getAttrOfType<DictionaryAttr>("numeric_policy");
}

//===----------------------------------------------------------------------===//
// Compute-legalize (early)
//===----------------------------------------------------------------------===//

struct ComputeLegalize
    : public PassWrapper<ComputeLegalize, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(ComputeLegalize)

  StringRef getArgument() const override { return "tessera-compute-legalize"; }
  StringRef getDescription() const override {
    return "C4 compute-legalize — reduced-precision storage without an "
           "accumulator gets numeric_policy.accum = fp32 (int32 for int4/int8), "
           "the wide-accumulate form. Run early, before fusion/codegen.";
  }

  void runOnOperation() override {
    ModuleOp module = getOperation();
    MLIRContext *ctx = &getContext();
    module.walk([&](Operation *op) {
      DictionaryAttr policy = policyOf(op);
      if (!policy)
        return;
      auto storageAttr = policy.getAs<StringAttr>("storage");
      if (!storageAttr)
        return;
      StringRef storage = storageAttr.getValue();
      if (!reducedPrecisionStorages().contains(storage))
        return;
      auto accumAttr = policy.getAs<StringAttr>("accum");
      if (accumAttr && !accumAttr.getValue().empty())
        return; // already has an accumulator — idempotent.
      StringRef accum = isIntegerStorage(storage) ? "int32" : "fp32";
      NamedAttrList entries(policy);
      entries.set("accum", StringAttr::get(ctx, accum));
      op->setAttr("numeric_policy", entries.getDictionary(ctx));
    });
  }
};

//===----------------------------------------------------------------------===//
// Storage-legalize (terminal)
//===----------------------------------------------------------------------===//

struct StorageLegalize
    : public PassWrapper<StorageLegalize, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(StorageLegalize)

  StringRef getArgument() const override { return "tessera-storage-legalize"; }
  StringRef getDescription() const override {
    return "C4 storage-legalize — sub-byte / block-scaled storage "
           "(fp4/nvfp4/fp6/int4) gets tessera.storage_packed + "
           "tessera.storage_container. Run terminally, after all compute "
           "rewrites.";
  }

  void runOnOperation() override {
    ModuleOp module = getOperation();
    MLIRContext *ctx = &getContext();
    module.walk([&](Operation *op) {
      DictionaryAttr policy = policyOf(op);
      if (!policy)
        return;
      auto storageAttr = policy.getAs<StringAttr>("storage");
      if (!storageAttr)
        return;
      StringRef container = packedContainerFor(storageAttr.getValue());
      if (container.empty())
        return;
      if (op->hasAttr("tessera.storage_packed"))
        return; // idempotent.
      op->setAttr("tessera.storage_packed", BoolAttr::get(ctx, true));
      op->setAttr("tessera.storage_container", StringAttr::get(ctx, container));
    });
  }
};

} // namespace

namespace tessera {
std::unique_ptr<Pass> createComputeLegalizePass() {
  return std::make_unique<ComputeLegalize>();
}
std::unique_ptr<Pass> createStorageLegalizePass() {
  return std::make_unique<StorageLegalize>();
}
} // namespace tessera
