// MetadataObligationPass.cpp — Decision #32 boundary verifier (W1.3, 2026-08-03)
//
// Decision #32: "Information loss across a level boundary must be declared. A
// lowering either carries each Decision #15a attribute forward, or records a
// named reason it dropped it. A boundary verifier fails on silent loss."
//
// The driving defect, re-measured on 2026-08-03 against this tree:
//
//     attribute        Graph IR ODS   Schedule IR   Tile IR
//     numeric_policy        30             1           0
//     distribution           5             0           0
//
// The accumulator contract is stated at Graph IR and no longer exists by the
// time codegen picks an instruction. Nothing detected that, because nothing was
// looking: a grep for a drop-reason mechanism anywhere in `src/` or
// `python/tessera/compiler/` returned nothing before this file.
//
// ── Why two passes and not one ──
//
// A boundary has two sides, and an MLIR pass sees one module. The obvious
// implementation is a `PassInstrumentation` straddling `runBeforePass` /
// `runAfterPass`, but an instrumentation is registered in the driver rather
// than named in a pipeline, so it cannot be exercised by a lit fixture — and an
// unfixturable verifier is exactly the kind of declaration Decision #29 rejects.
//
// Instead the snapshot rides IN the IR:
//
//     tessera-opt --tessera-record-metadata \
//                 --pass-pipeline=<the boundary lowering> \
//                 --tessera-verify-metadata-obligation
//
// One invocation, fully testable, and the snapshot is inspectable in the dump
// when a fixture fails. Both passes are ALSO scheduled around
// `TileIRLoweringPass` inside the production pipelines (`Passes.cpp`), so
// ordinary compilation is checked and not only the fixtures — raised in PR #500
// review, and correctly: standalone registration alone would have made this a
// tool nobody runs.
//
// ── What is compared: VALUES, not just names ──
//
// Per function, a MULTISET of (attribute name -> printed value) for the
// Decision #15a attributes.
//
// The first version of this pass recorded only the SET OF NAMES present, which
// PR #500 review correctly called a false-negative generator. Two cases it
// waved through:
//
//   * two matmuls carry `numeric_policy` before the boundary and only one
//     resulting `tile.mma` keeps it — the name is still somewhere in the
//     function, so the surviving occurrence covered for the lost one;
//   * a policy REPLACED (`accum = "fp32"` becomes `accum = "fp16"`) — the name
//     never moved, so nothing fired.
//
// The second is the worse one: it is exactly the instruction-selection
// corruption this verifier exists to prevent, and it was invisible. Counting
// values fixes both.
//
// Names are normalized to the last dot-component, so `tessera.layout`,
// `tile.layout` and `layout` are one fact. #32 requires the information to
// survive, not the spelling: a lowering may re-express an attribute in the
// target level's vocabulary, and a verifier keyed on the exact string would
// report a false drop every time a level renamed one.
//
// Re-expressing the VALUE is the same kind of legitimate move (Graph IR's
// `layout = "row_major"` becomes Tile's `#tile.layout<shard = ...>`), so it has
// its own reason, `re_expressed` — accepted only while the NAME still survives.
// If the name is gone too then nothing was re-expressed, and the reason is
// refused.
//
// `shape` and `dtype` are deliberately NOT tracked. They live in MLIR types,
// not attributes, so they cannot silently vanish the way a discardable
// attribute can, and the type system already checks them. Tracking them here
// would produce noise, not coverage.
//
// ── Declaring a drop ──
//
//   func.func @f() attributes {
//     tessera.lowering.dropped = { numeric_policy = "represented_in_type" }
//   }
//
// The reason is a SEMANTIC key, so per Decision #21a it fails closed: an
// unrecognised reason is an error, never a permissive default. `not_yet_carried`
// is the honest escape hatch and is the only one that must name a plan item
// (`not_yet_carried:W1.1`), so declared debt stays attributable.
//
// ── Stale declarations are errors too ──
//
// A declaration that explains nothing fails with
// METADATA_OBLIGATION_STALE_DECLARATION — whether the attribute is still fully
// present, or was never there to begin with. Decision #29's rule applied to
// this file's own mechanism: such a record reads in review as a considered
// exception while carrying nothing, and it silently licenses a REAL future drop
// of that attribute. The never-present case was added in PR #500 review; it is
// the more dangerous of the two, because a declaration for an attribute the
// function does not have looks harmless right up until the function acquires
// one.

#include "Tessera/Transforms/Passes.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/StringSet.h"
#include "llvm/Support/raw_ostream.h"

#include <map>
#include <string>

using namespace mlir;

namespace {

//: Module attribute the record pass writes and the verify pass reads.
constexpr llvm::StringLiteral kSnapshotAttr = "tessera.metadata_snapshot";
//: Per-function (or module) dictionary of {attribute name -> reason}.
constexpr llvm::StringLiteral kDroppedAttr = "tessera.lowering.dropped";
//: Snapshot key for ops that live outside any function.
constexpr llvm::StringLiteral kModuleScope = "<module>";

// The Decision #15a attributes that ride as ATTRIBUTES. `shape` and `dtype`
// live in types and are checked by the type system; see the header comment.
bool isTrackedName(StringRef shortName) {
  return shortName == "numeric_policy" || shortName == "layout" ||
         shortName == "distribution" || shortName == "target";
}

// `tessera.layout` / `tile.layout` / `layout` all normalize to `layout`.
StringRef shortName(StringRef full) {
  auto pos = full.rfind('.');
  return pos == StringRef::npos ? full : full.drop_front(pos + 1);
}

// Legal drop reasons. Closed set, fails closed (Decision #21a).
//
// `re_expressed` is validated separately: it is legal only while the attribute
// NAME still survives, since it claims the value was re-encoded rather than
// lost. `not_yet_carried` requires a `:<plan item>` suffix.
bool isKnownReason(StringRef reason) {
  return reason == "represented_in_type" ||  // moved into the level's type
         reason == "target_invariant" ||     // one behaviour; attribute vacuous
         reason == "consumed_by_pass";       // acted on; no downstream meaning
}

// The scope an op belongs to: its enclosing function's symbol name, or
// `<module>`. Attribute survival is compared per scope rather than per op
// because a lowering REPLACES ops -- op identity does not survive a boundary,
// which is the whole reason a naive before/after diff does not work here.
StringRef scopeOf(Operation *op) {
  for (Operation *cur = op; cur; cur = cur->getParentOp())
    if (auto fn = dyn_cast<func::FuncOp>(cur))
      return fn.getName();
  return kModuleScope;
}

//: attribute short-name -> printed value -> how many ops carry it.
using ValueCounts = std::map<std::string, std::map<std::string, int64_t>>;

std::string printAttr(Attribute attr) {
  std::string text;
  llvm::raw_string_ostream os(text);
  attr.print(os);
  return text;
}

// Collect, per scope, the multiset of tracked (name, value) pairs.
// `std::map`, not `llvm::MapVector`: the latter is DenseMap-backed and there is
// no DenseMapInfo<std::string>. Sorted order is wanted here anyway -- the
// snapshot is compared as text.
std::map<std::string, ValueCounts> collect(ModuleOp module) {
  std::map<std::string, ValueCounts> found;
  module.walk([&](Operation *op) {
    for (NamedAttribute attr : op->getAttrs()) {
      StringRef sn = shortName(attr.getName().strref());
      if (!isTrackedName(sn))
        continue;
      found[scopeOf(op).str()][sn.str()][printAttr(attr.getValue())] += 1;
    }
  });
  return found;
}

// ── Pass 1 — record ────────────────────────────────────────────────────────

struct RecordMetadata
    : public PassWrapper<RecordMetadata, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(RecordMetadata)

  StringRef getArgument() const final { return "tessera-record-metadata"; }
  StringRef getDescription() const final {
    return "Record the Decision #15a attribute inventory (names AND values) "
           "before a boundary lowering, for "
           "--tessera-verify-metadata-obligation to check.";
  }

  void runOnOperation() override {
    ModuleOp module = getOperation();
    OpBuilder builder(module.getContext());

    // Encoding: {scope: {attr_name: ["<printed value>", <count>, ...]}}.
    // Flat pairs rather than a nested dictionary because a printed attribute is
    // not a legal MLIR identifier and so cannot be a dictionary KEY.
    SmallVector<NamedAttribute> scopes;
    auto collected = collect(module);
    for (auto &scopeEntry : collected) {
      SmallVector<NamedAttribute> names;
      for (auto &nameEntry : scopeEntry.second) {
        SmallVector<Attribute> flat;
        for (auto &valueEntry : nameEntry.second) {  // std::map => sorted
          flat.push_back(builder.getStringAttr(valueEntry.first));
          flat.push_back(builder.getI64IntegerAttr(valueEntry.second));
        }
        names.push_back(builder.getNamedAttr(nameEntry.first,
                                             builder.getArrayAttr(flat)));
      }
      scopes.push_back(builder.getNamedAttr(scopeEntry.first,
                                            builder.getDictionaryAttr(names)));
    }
    llvm::sort(scopes, [](const NamedAttribute &a, const NamedAttribute &b) {
      return a.getName().strref() < b.getName().strref();
    });
    module->setAttr(kSnapshotAttr, builder.getDictionaryAttr(scopes));
  }
};

// ── Pass 2 — verify ────────────────────────────────────────────────────────

struct VerifyMetadataObligation
    : public PassWrapper<VerifyMetadataObligation,
                         OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(VerifyMetadataObligation)

  StringRef getArgument() const final {
    return "tessera-verify-metadata-obligation";
  }
  StringRef getDescription() const final {
    return "Decision #32: fail when a Decision #15a attribute vanishes across a "
           "level boundary without a recorded reason.";
  }

  // Where a scope's drop declaration may live: on the function itself, or on
  // the module for whole-module lowerings.
  DictionaryAttr droppedFor(ModuleOp module, StringRef scope) {
    if (scope != kModuleScope) {
      if (auto fn = module.lookupSymbol<func::FuncOp>(scope))
        if (auto d = fn->getAttrOfType<DictionaryAttr>(kDroppedAttr))
          return d;
    }
    return module->getAttrOfType<DictionaryAttr>(kDroppedAttr);
  }

  // Decode one scope's recorded {name -> {value -> count}}.
  ValueCounts decode(DictionaryAttr scopeDict) {
    ValueCounts out;
    for (NamedAttribute nameEntry : scopeDict) {
      auto flat = dyn_cast<ArrayAttr>(nameEntry.getValue());
      if (!flat) continue;
      for (size_t i = 0; i + 1 < flat.size(); i += 2) {
        auto value = dyn_cast<StringAttr>(flat[i]);
        auto count = dyn_cast<IntegerAttr>(flat[i + 1]);
        if (value && count)
          out[nameEntry.getName().str()][value.getValue().str()] =
              count.getInt();
      }
    }
    return out;
  }

  // Validate one declared reason. `nameSurvives` gates `re_expressed`.
  bool checkReason(ModuleOp module, StringRef scope, StringRef name,
                   Attribute reasonAttr, bool nameSurvives) {
    auto reason = dyn_cast<StringAttr>(reasonAttr);
    if (!reason) {
      module.emitError()
          << "METADATA_OBLIGATION_UNKNOWN_REASON: the reason for dropping `"
          << name << "` in @" << scope << " must be a string.";
      return false;
    }
    StringRef text = reason.getValue();

    if (text.starts_with("not_yet_carried")) {
      // Declared debt has to name its owner, or it is just a silent drop with
      // extra syntax.
      StringRef item = text.drop_front(StringRef("not_yet_carried").size());
      if (!item.consume_front(":") || item.trim().empty()) {
        module.emitError()
            << "METADATA_OBLIGATION_DEBT_UNATTRIBUTED: `" << name << "` in @"
            << scope
            << " is declared `not_yet_carried` with no plan item. Write "
               "`not_yet_carried:<item>` (e.g. not_yet_carried:W1.1) so the "
               "debt has an owner.";
        return false;
      }
      return true;
    }

    if (text == "re_expressed") {
      // `re_expressed` claims the VALUE was re-encoded in the target level's
      // vocabulary. That claim is only coherent while the name is still there;
      // if the whole attribute is gone, nothing was re-expressed.
      if (!nameSurvives) {
        module.emitError()
            << "METADATA_OBLIGATION_UNKNOWN_REASON: `" << name << "` in @"
            << scope
            << " is declared `re_expressed`, but the attribute is absent after "
               "the boundary — nothing was re-expressed. Use "
               "represented_in_type, consumed_by_pass, target_invariant, or "
               "not_yet_carried:<plan item>.";
        return false;
      }
      return true;
    }

    if (!isKnownReason(text)) {
      module.emitError()
          << "METADATA_OBLIGATION_UNKNOWN_REASON: `" << text
          << "` is not a legal drop reason for `" << name << "` in @" << scope
          << ". Legal: represented_in_type, target_invariant, "
             "consumed_by_pass, re_expressed, not_yet_carried:<plan item>.";
      return false;
    }
    return true;
  }

  void runOnOperation() override {
    ModuleOp module = getOperation();
    bool anyError = false;

    auto snapshot = module->getAttrOfType<DictionaryAttr>(kSnapshotAttr);
    if (!snapshot) {
      // Fail closed. A verify with no snapshot proves nothing, and silently
      // succeeding would make the gate green on every pipeline that forgot to
      // record -- the precise failure this project keeps finding.
      module.emitError(
          "METADATA_OBLIGATION_NO_SNAPSHOT: --tessera-verify-metadata-obligation "
          "requires a `" + kSnapshotAttr + "` recorded by "
          "--tessera-record-metadata before the boundary lowering.");
      return signalPassFailure();
    }

    auto current = collect(module);

    for (NamedAttribute scopeEntry : snapshot) {
      StringRef scope = scopeEntry.getName().strref();
      auto scopeDict = dyn_cast<DictionaryAttr>(scopeEntry.getValue());
      if (!scopeDict) continue;

      ValueCounts before = decode(scopeDict);
      auto currentIt = current.find(scope.str());
      ValueCounts after =
          currentIt != current.end() ? currentIt->second : ValueCounts{};
      DictionaryAttr dropped = droppedFor(module, scope);
      llvm::StringSet<> declarationExplainedSomething;

      for (auto &nameEntry : before) {
        const std::string &name = nameEntry.first;
        auto afterName = after.find(name);
        bool nameSurvives = afterName != after.end();

        // Which recorded values are missing, and by how much.
        SmallVector<std::string> lost;
        for (auto &valueEntry : nameEntry.second) {
          int64_t had = valueEntry.second;
          int64_t has = 0;
          if (nameSurvives) {
            auto v = afterName->second.find(valueEntry.first);
            if (v != afterName->second.end()) has = v->second;
          }
          if (has < had)
            lost.push_back(valueEntry.first);
        }
        if (lost.empty()) continue;

        Attribute reasonAttr = dropped ? dropped.get(name) : Attribute();
        if (reasonAttr) {
          declarationExplainedSomething.insert(name);
          if (!checkReason(module, scope, name, reasonAttr, nameSurvives))
            anyError = true;
          continue;
        }

        if (!nameSurvives) {
          module.emitError()
              << "METADATA_OBLIGATION_SILENT_DROP: `" << name
              << "` was present in @" << scope
              << " before this boundary and is gone after it, with no reason "
                 "recorded. Decision #32: carry it forward in the target "
                 "level's vocabulary, or declare `" << kDroppedAttr << " = { "
              << name << " = \"<reason>\" }` on the function or module. Legal "
                 "reasons: represented_in_type, target_invariant, "
                 "consumed_by_pass, not_yet_carried:<plan item>.";
        } else {
          // The name survived on some other op, which is why a name-only
          // snapshot could not see this. Either an occurrence was lost or a
          // value was replaced.
          module.emitError()
              << "METADATA_OBLIGATION_VALUE_DROP: `" << name << "` in @" << scope
              << " still exists after this boundary, but the value "
              << lost.front() << " it carried before does not (" << lost.size()
              << " value(s) lost). A surviving occurrence elsewhere in the "
                 "function does not carry this one's contract — this is how a "
                 "replaced accumulator policy stays invisible. Carry the value "
                 "forward, or declare a reason (`re_expressed` if the value was "
                 "re-encoded in the target level's vocabulary).";
        }
        anyError = true;
      }

      // A declaration that explains nothing. Two shapes, both refused: the
      // attribute is still fully present, or it was never recorded at all. The
      // second is the more dangerous -- it looks harmless right up until the
      // function acquires that attribute, at which point it silently licenses a
      // real drop nobody reviewed.
      if (dropped) {
        for (NamedAttribute d : dropped) {
          StringRef name = d.getName().strref();
          if (declarationExplainedSomething.contains(name)) continue;
          bool wasRecorded = before.find(name.str()) != before.end();
          module.emitError()
              << "METADATA_OBLIGATION_STALE_DECLARATION: @" << scope
              << " declares `" << name << "` dropped, but "
              << (wasRecorded
                      ? "it is still present after the boundary"
                      : "it was never present before the boundary either")
              << ". Remove the declaration — an unused exception licenses a "
                 "future drop nobody reviewed.";
          anyError = true;
        }
      }
    }

    if (anyError) signalPassFailure();
  }
};

}  // namespace

namespace tessera {
std::unique_ptr<Pass> createRecordMetadataPass() {
  return std::make_unique<RecordMetadata>();
}
std::unique_ptr<Pass> createVerifyMetadataObligationPass() {
  return std::make_unique<VerifyMetadataObligation>();
}
}  // namespace tessera
