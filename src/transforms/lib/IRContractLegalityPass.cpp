// IRContractLegalityPass.cpp — dtype / aliasing / buffer-binding contracts (2026-06-19)
//
// Closes the "Layout and binding contracts are uneven" item in
// docs/audit/compiler/COMPILER_AUDIT.md. LayoutLegalityPass already covers the
// *layout* half (cast accept-set + producer/consumer + scale_layout). This pass
// is its sibling for the three remaining contract families the audit names —
// **dtype, aliasing, buffer-binding** — folded into one ModuleOp walk so the
// rule set lives in one place (the same shape as LayoutLegalityPass's
// cast/matmul/conv/scale rules).
//
// ── Dtype contracts (enforce CANONICAL_API Decision #15a) ──
//   DTYPE_LEGALITY_TF32_AS_STORAGE
//     numeric_policy.storage = "tf32" is illegal — TF32 is a math_mode on fp32
//     storage, not a storage dtype.
//   DTYPE_LEGALITY_UNKNOWN_STORAGE
//     numeric_policy.storage names a dtype outside the canonical + known-gated
//     set.
//   DTYPE_LEGALITY_LOWP_WITHOUT_WIDE_ACCUM
//     A low-precision storage (fp8*/fp6*/fp4*/nvfp4/int4/int8) must declare a
//     *wider* accumulator (fp32/fp16/bf16/int32). Storage and accumulator are
//     distinct contracts (Decision #15a) — a fused single dtype is illegal for
//     these ops.
//
// ── Aliasing contracts ──
//   ALIAS_LEGALITY_MISSING_ALIASES
//     An op marked `tessera.inplace = true` must declare `tessera.aliases`
//     (the operand index its result aliases) — an undeclared in-place mutation
//     has no aliasing contract the scheduler can honor.
//   ALIAS_LEGALITY_OPERAND_OOB
//     `tessera.aliases` indexes past the operand list.
//
// ── Buffer-binding contracts ──
//   BUFFER_BINDING_UNKNOWN_ROLE
//     `tessera.buffer_role` outside {input, output, scratch, accumulator, weight}.
//   BUFFER_BINDING_CONFLICT
//     Two ops bind the same `tessera.binding` id to *different* roles — a buffer
//     can't be both (e.g.) an input and a scratch in one program.
//
// Diagnostic codes are stable for COMPILER_AUDIT / SHAPE_SYSTEM cross-linking.
// Registered standalone as `--tessera-ir-contracts` (parallel to
// `--tessera-layout-legality`) and wired into the named lowering pipelines
// (tessera-lower-to-x86, -to-gpu, and the CUDA13 chain) right after
// LayoutLegalityPass, so dtype/aliasing/buffer-binding violations surface with
// the other early structural diagnostics on every backend.

#include "Tessera/Transforms/Passes.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/StringSet.h"

using namespace mlir;

namespace {

// Canonical dtype names (docs/reference/tessera_tensor_attributes.md) plus the
// known planned-gated quant storages that appear as numeric_policy.storage in
// real IR (int4 on grouped_gemm / dequant paths).
static const llvm::StringSet<> &knownStorageDtypes() {
  static const llvm::StringSet<> kSet = {
      "fp64", "fp32", "fp16", "bf16",
      "fp8_e4m3", "fp8_e5m2", "fp6_e2m3", "fp6_e3m2", "fp4_e2m1", "nvfp4",
      "int8", "int16", "int32", "int64", "bool",
      // planned-gated, but real storages in current IR:
      "int4",
  };
  return kSet;
}

// Low-precision storages that REQUIRE a separate wider accumulator (Decision #15a).
static const llvm::StringSet<> &lowPrecisionStorages() {
  static const llvm::StringSet<> kSet = {
      "fp8_e4m3", "fp8_e5m2", "fp6_e2m3", "fp6_e3m2", "fp4_e2m1", "nvfp4",
      "int4", "int8",
  };
  return kSet;
}

// Dtypes wide enough to be a legal accumulator for a low-precision storage.
static const llvm::StringSet<> &wideAccumDtypes() {
  static const llvm::StringSet<> kSet = {"fp32", "fp16", "bf16", "int32"};
  return kSet;
}

static const llvm::StringSet<> &bufferRoles() {
  static const llvm::StringSet<> kSet = {
      "input", "output", "scratch", "accumulator", "weight"};
  return kSet;
}

struct IRContractLegality
    : public PassWrapper<IRContractLegality, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(IRContractLegality)

  StringRef getArgument() const override { return "tessera-ir-contracts"; }
  StringRef getDescription() const override {
    return "IR contract legality — dtype (numeric_policy storage/accum coupling, "
           "TF32-as-storage, unknown storage; Decision #15a), aliasing "
           "(tessera.inplace requires tessera.aliases, in-range), and "
           "buffer-binding (tessera.buffer_role accept-set + no conflicting "
           "role for one tessera.binding) contracts.";
  }

  // ── Dtype: numeric_policy storage/accum coupling ──
  static LogicalResult checkNumericPolicy(Operation *op) {
    auto policy = op->getAttrOfType<DictionaryAttr>("numeric_policy");
    if (!policy) return success();
    auto storageAttr = policy.getAs<StringAttr>("storage");
    if (!storageAttr) return success();  // policy without storage → nothing to check
    StringRef storage = storageAttr.getValue();

    if (storage == "tf32")
      return op->emitOpError(
          "DTYPE_LEGALITY_TF32_AS_STORAGE: numeric_policy.storage=\"tf32\" is "
          "illegal; TF32 is a math_mode on fp32 storage, not a storage dtype "
          "(set numeric_policy.math_mode=\"tf32\" on fp32). See Decision #15a.");

    if (!knownStorageDtypes().contains(storage))
      return op->emitOpError(
                 "DTYPE_LEGALITY_UNKNOWN_STORAGE: numeric_policy.storage=\"")
             << storage << "\" is not a known storage dtype.";

    if (lowPrecisionStorages().contains(storage)) {
      auto accumAttr = policy.getAs<StringAttr>("accum");
      if (!accumAttr || accumAttr.getValue().empty())
        return op->emitOpError(
                   "DTYPE_LEGALITY_LOWP_WITHOUT_WIDE_ACCUM: low-precision "
                   "storage \"")
               << storage
               << "\" must declare numeric_policy.accum (a wider accumulator: "
                  "fp32/fp16/bf16/int32). Storage and accumulator are distinct "
                  "contracts (Decision #15a).";
      StringRef accum = accumAttr.getValue();
      if (!wideAccumDtypes().contains(accum))
        return op->emitOpError(
                   "DTYPE_LEGALITY_LOWP_WITHOUT_WIDE_ACCUM: low-precision "
                   "storage \"")
               << storage << "\" has accum \"" << accum
               << "\" which is not a wider accumulator (fp32/fp16/bf16/int32).";
    }
    return success();
  }

  // ── Aliasing: tessera.inplace requires a valid tessera.aliases index ──
  static LogicalResult checkAliasing(Operation *op) {
    auto inplace = op->getAttrOfType<BoolAttr>("tessera.inplace");
    if (!inplace || !inplace.getValue()) return success();
    auto aliases = op->getAttrOfType<IntegerAttr>("tessera.aliases");
    if (!aliases)
      return op->emitOpError(
          "ALIAS_LEGALITY_MISSING_ALIASES: op is tessera.inplace=true but does "
          "not declare `tessera.aliases` (the operand index its result aliases).");
    int64_t idx = aliases.getInt();
    if (idx < 0 || idx >= static_cast<int64_t>(op->getNumOperands()))
      return op->emitOpError("ALIAS_LEGALITY_OPERAND_OOB: tessera.aliases=")
             << idx << " is out of range [0, " << op->getNumOperands() << ").";
    return success();
  }

  // ── Buffer-binding: role accept-set (per-op) ──
  static LogicalResult checkBufferRole(Operation *op) {
    auto roleAttr = op->getAttrOfType<StringAttr>("tessera.buffer_role");
    if (!roleAttr) return success();
    if (bufferRoles().contains(roleAttr.getValue())) return success();
    return op->emitOpError(
               "BUFFER_BINDING_UNKNOWN_ROLE: tessera.buffer_role=\"")
           << roleAttr.getValue()
           << "\" is not in {input, output, scratch, accumulator, weight}.";
  }

  void runOnOperation() override {
    ModuleOp module = getOperation();
    bool anyError = false;

    // Per-op contract checks.
    module.walk([&](Operation *op) {
      if (failed(checkNumericPolicy(op))) anyError = true;
      if (failed(checkAliasing(op))) anyError = true;
      if (failed(checkBufferRole(op))) anyError = true;
    });

    // Cross-op buffer-binding conflict check: one binding id must not be bound
    // to two different roles anywhere in the module.
    llvm::DenseMap<StringRef, StringRef> bindingRole;
    module.walk([&](Operation *op) {
      auto bind = op->getAttrOfType<StringAttr>("tessera.binding");
      auto role = op->getAttrOfType<StringAttr>("tessera.buffer_role");
      if (!bind || !role) return;
      auto it = bindingRole.find(bind.getValue());
      if (it == bindingRole.end()) {
        bindingRole[bind.getValue()] = role.getValue();
      } else if (it->second != role.getValue()) {
        op->emitOpError("BUFFER_BINDING_CONFLICT: tessera.binding=\"")
            << bind.getValue() << "\" is bound as both \"" << it->second
            << "\" and \"" << role.getValue() << "\".";
        anyError = true;
      }
    });

    if (anyError) signalPassFailure();
  }
};

}  // namespace

namespace tessera {
std::unique_ptr<Pass> createIRContractLegalityPass() {
  return std::make_unique<IRContractLegality>();
}
}  // namespace tessera
