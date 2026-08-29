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
#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/StringSet.h"
#include "llvm/ADT/StringSwitch.h"

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


// ── NUMPOL-CARRIER-1 (queue row 3b) — the policy gets a SCHEMA ──
//
// Measured on this tree 2026-08-25, before any of the checks below existed:
// five malformed policies were all accepted (exit 0), while the documented
// TF32-as-storage violation correctly failed — so the pass was running and
// simply had nothing to say about them.
//
//   {storage="fp32", accum="fp16"}        accumulator NARROWER than storage
//   {storage="bf16", accumulator="fp32"}  a TYPO: the real key is absent
//   {storage="bf16", accum="float128"}    a dtype that does not exist
//   {storage="bf16", accum=32 : i64}      not even a string
//   {storage="bf16", ..., math_mode="tf32"}  TF32 is fp32-only (#15a)
//
// The typo is the sharpest: `getAs<StringAttr>("accum")` returns null for a
// misspelled key exactly as it does for an absent one, so the op carried a
// policy that LOOKED like it stated an accumulator contract and stated none.
// A carrier cannot be built on a payload with no schema — that is why this
// slice comes first.
//
// ── Why mantissa bits, not storage width ──
//
// The accumulator's contract is how precisely a running sum is held, and that
// is its MANTISSA, not its total width. fp16 and bf16 are both 16 bits but
// carry 11 and 8 mantissa bits; an fp16-storage program accumulating in bf16
// loses three bits of every operand it already paid to keep. Comparing total
// width would call that pair legal.
//
// ── Why a narrower accumulator is refused rather than warned ──
//
// Measured, K=4096 dot products, median relative error over 48 trials against
// a float64 reference:
//
//   storage=fp32 accum=fp32   8.44e-07        (baseline)
//   storage=fp32 accum=fp16   6.19e-03        7334x worse
//   storage=fp16 accum=fp32   2.40e-04         285x worse
//
// Both of the last two spend the same 48 dtype bits. Spending them on the
// ACCUMULATOR is 25.8x more accurate (18.2x for the bf16 pair). So a narrowing
// policy is not merely unusual — at its own bit budget it is strictly
// dominated by the policy that swaps the two.
//
// Stronger, and the reason the diagnostic says what it says: with accum=bf16
// the fp32-storage result is BIT-IDENTICAL to the all-bf16 result (verified:
// both 0xBBEE). Under a narrower accumulator the wider storage is
// unobservable, so it buys nothing but memory traffic. There is no program
// for which this policy is the right answer, which is what makes refusing it
// safe.
//
// FORGE §1.3 is the same fact at training scale: whether an fp32 accumulator's
// precision benefit is realizable (913x / 1.1x / 1.0x) is a function of
// `numeric_policy.accum` x state dtype. A compiler can only decide that if the
// policy is well-formed and survives to where the decision is made.

//: Keys a numeric_policy may contain, and the attribute kind each carries.
//:
//: Taken from the NORMATIVE definition — `NumericPolicy` in
//: `python/tessera/compiler/primitive_coverage.py` and the row in
//: `docs/reference/tessera_tensor_attributes.md` that spells it
//: `NumericPolicy(storage, accum, rounding, scale, quant_axis, deterministic[,
//: math_mode])`. The first version of this checker derived its key set from
//: the policies that happen to appear in fixtures instead, and so invented
//: `rounding_mode` for the canonical `rounding` and omitted `scale`,
//: `quant_axis`, `deterministic`, and `scale_layout` entirely (PR #631
//: review). Every in-tree fixture passed, because none of them carries a
//: quantization or determinism policy — but the production legality pipeline
//: would have rejected the first one that did.
//:
//: The value KIND is part of the schema for the same reason the key set is:
//: `quant_axis` is an integer, `deterministic` a boolean, and `scale_layout` a
//: nested dictionary, so a blanket "every value is a string" rule refuses
//: three canonical fields. Checking the declared kind per key is what makes
//: the earlier NON_STRING_VALUE rule correct rather than merely strict.
enum class PolicyValueKind { Str, Int, Bool, Dict };

static const llvm::StringMap<PolicyValueKind> &numericPolicySchema() {
  static const llvm::StringMap<PolicyValueKind> kSchema = {
      {"storage", PolicyValueKind::Str},
      {"accum", PolicyValueKind::Str},
      {"rounding", PolicyValueKind::Str},
      {"scale", PolicyValueKind::Str},
      {"quant_axis", PolicyValueKind::Int},
      {"deterministic", PolicyValueKind::Bool},
      {"math_mode", PolicyValueKind::Str},
      {"scale_layout", PolicyValueKind::Dict},
      // Not in the dataclass, but 12 in-tree fixtures carry it: the attention
      // family accumulates its softmax statistics separately from the matmul,
      // so the stage has its own dtype. Listed explicitly rather than tolerated
      // by a permissive default — that is the whole point of a closed set.
      {"softmax", PolicyValueKind::Str},
  };
  return kSchema;
}

static llvm::StringRef policyValueKindName(PolicyValueKind kind) {
  switch (kind) {
  case PolicyValueKind::Str: return "a string";
  case PolicyValueKind::Int: return "an integer";
  case PolicyValueKind::Bool: return "a boolean";
  case PolicyValueKind::Dict: return "a dictionary";
  }
  return "a value";
}

static bool policyValueMatches(PolicyValueKind kind, mlir::Attribute value) {
  switch (kind) {
  case PolicyValueKind::Str: return llvm::isa<StringAttr>(value);
  case PolicyValueKind::Int: return llvm::isa<IntegerAttr>(value);
  case PolicyValueKind::Bool: return llvm::isa<BoolAttr>(value);
  case PolicyValueKind::Dict: return llvm::isa<DictionaryAttr>(value);
  }
  return false;
}

//: Significand bits INCLUDING the implicit leading one; for integers, the
//: representable width. This is the precision an accumulator actually offers.
static int numericPolicyMantissaBits(llvm::StringRef dtype) {
  return llvm::StringSwitch<int>(dtype)
      .Case("fp64", 53).Case("fp32", 24).Case("tf32", 11)
      .Case("fp16", 11).Case("bf16", 8)
      .Case("fp8_e4m3", 4).Case("fp8_e5m2", 3)
      .Case("fp6_e2m3", 4).Case("fp6_e3m2", 3)
      .Case("fp4_e2m1", 2).Case("nvfp4", 2)
      .Case("int64", 64).Case("int32", 32).Case("int16", 16)
      .Case("int8", 8).Case("int4", 4).Case("bool", 1)
      .Default(-1);
}

static bool numericPolicyIsFloat(llvm::StringRef dtype) {
  return dtype.starts_with("fp") || dtype == "bf16" || dtype == "nvfp4" ||
         dtype == "tf32";
}

//: Accumulator dtypes. Deliberately a SUPERSET of storage dtypes minus the
//: sub-8-bit formats: nothing accumulates into fp4.
static const llvm::StringSet<> &knownAccumDtypes() {
  static const llvm::StringSet<> kSet = {
      "fp64", "fp32", "fp16", "bf16", "int64", "int32", "int16", "int8"};
  return kSet;
}

//: math_mode names a REDUCED-precision arithmetic on a wider storage. It is a
//: semantic key (#21a), so the legal set is stated rather than assumed.
static const llvm::StringSet<> &knownMathModes() {
  static const llvm::StringSet<> kSet = {"ieee", "default", "tf32", "bf16x3",
                                         "fp16x2"};
  return kSet;
}

static const llvm::StringSet<> &knownRoundingModes() {
  static const llvm::StringSet<> kSet = {
      "round_to_nearest_even", "round_to_nearest_away", "round_toward_zero",
      "round_toward_positive", "round_toward_negative", "stochastic"};
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
    Attribute rawPolicy = op->getAttr("numeric_policy");
    if (!rawPolicy) return success();
    auto policy = llvm::dyn_cast<DictionaryAttr>(rawPolicy);
    if (!policy) {
      // NUMPOL-CARRIER-1 (queue row 3b). `getAttrOfType<DictionaryAttr>`
      // returns null for a WRONGLY TYPED attribute exactly as it does for an
      // absent one, so a numeric_policy that was not a dictionary used to be
      // skipped in silence by this checker AND by every consumer of the
      // attribute. Measured: the spectral scheduler emitted
      // `numeric_policy = "f32;ortho"` — a StringAttr holding a private
      // semicolon-delimited encoding — and it was invisible to all of them.
      // (That contract has been renamed to tessera.spectral_accumulation /
      // tessera.spectral_normalization, since a reduction-ORDER contract is
      // not a Decision #15a numeric_policy: its value could be
      // "deterministic_f32_ascending_frames", which is not a dtype at all.)
      //
      // Refusing here is what stops the collision recurring. One attribute
      // name must mean one thing, or its consumers are reading a different
      // contract than its producers wrote (#31).
      auto diag = op->emitOpError(
          "NUMERIC_POLICY_NOT_A_DICTIONARY: numeric_policy must be a "
          "dictionary of {storage, accum, math_mode, rounding_mode, softmax}");
      diag << ", got " << rawPolicy << ".";
      diag.attachNote()
          << "A wrongly typed attribute reads back as ABSENT through every "
             "consumer's DictionaryAttr lookup, so an unrelated contract "
             "parked under this name is not merely unchecked — it is "
             "invisible. Give a different contract a different name.";
      return failure();
    }

    // ── Schema first: every key known, every value a string ──
    // These run BEFORE the storage lookup, because the old early return on a
    // missing `storage` is precisely what let a typo'd key through untouched.
    for (NamedAttribute entry : policy) {
      StringRef key = entry.getName().getValue();
      auto declared = numericPolicySchema().find(key);
      if (declared == numericPolicySchema().end()) {
        auto diag = op->emitOpError(
                        "NUMERIC_POLICY_UNKNOWN_KEY: numeric_policy has key \"")
                    << key << "\", which no Tessera contract defines.";
        diag.attachNote() << "A key nobody reads is not ignored here: it is how "
                             "a misspelling becomes a silently ABSENT semantic "
                             "contract (a typo'd `accum` leaves the op with no "
                             "accumulator contract while appearing to state "
                             "one). Legal keys: storage, accum, rounding, "
                             "scale, quant_axis, deterministic, math_mode, "
                             "scale_layout, softmax — the NumericPolicy fields "
                             "in docs/reference/tessera_tensor_attributes.md. "
                             "See Decisions #15a/#21a.";
        return failure();
      }
      if (!policyValueMatches(declared->second, entry.getValue()))
        return op->emitOpError(
                   "NUMERIC_POLICY_NON_STRING_VALUE: numeric_policy.")
               << key << " must be "
               << policyValueKindName(declared->second)
               << "; a wrongly typed value reads back as absent through the "
                  "consumer's typed lookup, so the contract silently "
                  "disappears.";
    }

    // ── Mode names are semantic keys: state the legal set (#21a) ──
    if (auto mode = policy.getAs<StringAttr>("math_mode")) {
      if (!knownMathModes().contains(mode.getValue()))
        return op->emitOpError("NUMERIC_POLICY_UNKNOWN_MATH_MODE: math_mode=\"")
               << mode.getValue()
               << "\" is not a known math mode (ieee, default, tf32, bf16x3, "
                  "fp16x2).";
    }
    if (auto rmode = policy.getAs<StringAttr>("rounding")) {
      if (!knownRoundingModes().contains(rmode.getValue()))
        return op->emitOpError(
                   "NUMERIC_POLICY_UNKNOWN_ROUNDING_MODE: rounding=\"")
               << rmode.getValue() << "\" is not a known rounding mode.";
    }

    auto accumAttrEarly = policy.getAs<StringAttr>("accum");
    auto storageAttr = policy.getAs<StringAttr>("storage");

    // A bad accumulator dtype is refused wherever it appears — including on a
    // policy that states no storage, which the coupling checks below skip.
    // DEFERRED for low-precision storage: that path already has the older and
    // more specific DTYPE_LEGALITY_LOWP_WITHOUT_WIDE_ACCUM, and shadowing a
    // stable diagnostic code with a newer generic one silently rewrites the
    // contract that existing IR and its fixtures were written against.
    bool lowPrecisionOwnsTheAccum =
        storageAttr && lowPrecisionStorages().contains(storageAttr.getValue());
    if (accumAttrEarly && !lowPrecisionOwnsTheAccum &&
        !knownAccumDtypes().contains(accumAttrEarly.getValue()))
      return op->emitOpError("NUMERIC_POLICY_UNKNOWN_ACCUM: numeric_policy.accum=\"")
             << accumAttrEarly.getValue()
             << "\" is not a known accumulator dtype (fp64/fp32/fp16/bf16/"
                "int64/int32/int16/int8).";

    if (!storageAttr) return success();  // no storage stated → nothing to couple
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

    // ── The accumulator may not be narrower than the storage ──
    // See the measurement in this file's header: at a fixed dtype-bit budget
    // the narrowing policy is strictly dominated by the one that swaps
    // storage and accumulator (25.8x for the fp16/fp32 pair), and the wider
    // storage is BIT-IDENTICALLY unobservable under the narrower accumulator.
    if (accumAttrEarly) {
      StringRef accum = accumAttrEarly.getValue();
      int storageBits = numericPolicyMantissaBits(storage);
      int accumBits = numericPolicyMantissaBits(accum);
      bool storageIsFloat = numericPolicyIsFloat(storage);
      bool accumIsFloat = numericPolicyIsFloat(accum);
      if (storageBits > 0 && accumBits > 0) {
        // An integer accumulator cannot hold a floating-point product at all;
        // the reverse (integer storage, float accumulator) is the ordinary
        // dequantized-weight path and stays legal.
        if (storageIsFloat && !accumIsFloat) {
          auto diag = op->emitOpError(
              "NUMERIC_POLICY_NARROWING_ACCUM: floating-point storage \"");
          diag << storage << "\" cannot accumulate into integer \"" << accum
               << "\".";
          return failure();
        }
        // The comparison below is only meaningful within one domain:
        // numericPolicyMantissaBits returns significand bits for a float and
        // representable width for an integer, so int16-into-fp16 (a routine
        // dequantized-weight policy, declared legal three lines up) reads as
        // 16 > 11 and was refused. The measurement quoted in the note is a
        // float-into-float result and does not transfer to dequantization,
        // where the storage bits are integer codes rather than running-sum
        // precision.
        if (accumBits < storageBits && storageIsFloat == accumIsFloat) {
          auto diag = op->emitOpError(
              "NUMERIC_POLICY_NARROWING_ACCUM: numeric_policy declares storage "
              "\"");
          diag << storage << "\" (" << storageBits
               << (storageIsFloat ? " significand bits" : " bits")
               << ") accumulating into \"" << accum << "\" (" << accumBits
               << " bits), which is NARROWER.";
          // The dominance measurement below was taken on float/float pairs.
          // Quoting it on an int/int policy would assert a number nobody
          // measured, so that case gets the structural argument only.
          if (storageIsFloat)
            diag.attachNote()
                << "This is refused rather than warned because no program "
                   "wants it: at the same dtype-bit budget, spending the bits "
                   "on the accumulator instead is 25.8x more accurate "
                   "(measured, K=4096 dot product vs an fp64 reference), and "
                   "the result under the narrow accumulator is BIT-IDENTICAL "
                   "to also narrowing the storage — so the wider storage is "
                   "unobservable and buys only memory traffic. Either widen "
                   "accum to at least \"" << storage
                << "\", or narrow storage to \"" << accum
                << "\" and keep the bandwidth.";
          else
            diag.attachNote()
                << "An accumulator narrower than the storage it sums cannot "
                   "represent every value it loads, so the wider storage buys "
                   "only memory traffic. Either widen accum to at least \""
                << storage << "\", or narrow storage to \"" << accum << "\".";
          return failure();
        }
      }
    }

    // ── math_mode names a reduced arithmetic ON a wider storage ──
    // TF32 has an 11-bit significand: on bf16 storage (8 bits) it can round
    // nothing, so declaring it is either a no-op or a false statement about
    // the accumulate path. Decision #15a states TF32 as an fp32 math_mode.
    if (auto mode = policy.getAs<StringAttr>("math_mode")) {
      int modeBits = numericPolicyMantissaBits(mode.getValue());
      int storageBits = numericPolicyMantissaBits(storage);
      if (modeBits > 0 && storageBits > 0 && modeBits >= storageBits)
        return op->emitOpError("NUMERIC_POLICY_MATH_MODE_NOT_REDUCING: math_mode=\"")
               << mode.getValue() << "\" (" << modeBits
               << " significand bits) does not reduce storage \"" << storage
               << "\" (" << storageBits
               << " bits); a math mode names a NARROWER arithmetic performed on "
                  "wider storage. TF32 is an fp32 math mode (Decision #15a).";
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
