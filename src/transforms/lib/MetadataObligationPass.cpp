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
// when a fixture fails.
//
// ── What is compared ──
//
// Per function, the SET of Decision #15a attribute names reachable on any op.
// Normalized to the last dot-component, so `tessera.layout`, `tile.layout` and
// `layout` are one fact: #32 requires the information to survive, not the
// spelling — a lowering is explicitly allowed to re-express an attribute in the
// target level's vocabulary, and a verifier keyed on the exact string would
// report a false drop every time a level renamed one.
//
// `shape` and `dtype` are deliberately NOT tracked. They live in MLIR types,
// not attributes, so they cannot silently vanish the way a discardable
// attribute can, and the type system already checks them. Tracking them here
// would produce noise, not coverage. The four tracked names are the ones that
// ride as attributes and do in fact disappear.
//
// ── Declaring a drop ──
//
//   func.func @f() attributes {
//     tessera.lowering.dropped = { numeric_policy = "represented_in_type" }
//   }
//
// The reason is a SEMANTIC key, so per Decision #21a it fails closed: an
// unrecognised reason is an error, never a permissive default. The closed set
// is deliberately small; `not_yet_carried` is the honest escape hatch and is
// the only one that must name a plan item (`not_yet_carried:W1.1`), so declared
// debt stays attributable instead of becoming a shrug.
//
// ── Stale declarations are errors too ──
//
// Declaring a drop that did not happen fails with
// METADATA_OBLIGATION_STALE_DECLARATION. Decision #29's rule applied to this
// file's own mechanism: a drop record with nothing to explain reads in review
// as a considered exception while carrying nothing, and it would silently
// license a REAL future drop of that attribute. This is the negative half the
// eligibility-marking discipline (#10a) asks for.

#include "Tessera/Transforms/Passes.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/StringSet.h"

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
// `not_yet_carried` is handled separately: it requires a `:<plan item>` suffix,
// so declared debt names its owner.
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

// Collect, per scope, the set of tracked attribute short-names present.
llvm::DenseMap<StringRef, llvm::StringSet<>> collect(ModuleOp module) {
  llvm::DenseMap<StringRef, llvm::StringSet<>> found;
  module.walk([&](Operation *op) {
    for (NamedAttribute attr : op->getAttrs()) {
      StringRef sn = shortName(attr.getName().strref());
      if (isTrackedName(sn))
        found[scopeOf(op)].insert(sn);
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
    return "Record the Decision #15a attribute inventory before a boundary "
           "lowering, for --tessera-verify-metadata-obligation to check.";
  }

  void runOnOperation() override {
    ModuleOp module = getOperation();
    OpBuilder builder(module.getContext());

    SmallVector<NamedAttribute> scopes;
    for (auto &entry : collect(module)) {
      SmallVector<StringRef> names(entry.second.keys().begin(),
                                   entry.second.keys().end());
      llvm::sort(names);  // deterministic: the snapshot is compared as text
      SmallVector<Attribute> values;
      for (StringRef n : names)
        values.push_back(builder.getStringAttr(n));
      scopes.push_back(builder.getNamedAttr(entry.first,
                                            builder.getArrayAttr(values)));
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
      auto before = dyn_cast<ArrayAttr>(scopeEntry.getValue());
      if (!before) continue;

      auto it = current.find(scope);
      DictionaryAttr dropped = droppedFor(module, scope);
      llvm::StringSet<> declaredAndUsed;

      for (Attribute a : before) {
        auto nameAttr = dyn_cast<StringAttr>(a);
        if (!nameAttr) continue;
        StringRef name = nameAttr.getValue();

        bool survives = it != current.end() && it->second.contains(name);
        if (survives) continue;

        Attribute reasonAttr = dropped ? dropped.get(name) : Attribute();
        if (!reasonAttr) {
          module.emitError()
              << "METADATA_OBLIGATION_SILENT_DROP: `" << name
              << "` was present in @" << scope
              << " before this boundary and is gone after it, with no reason "
                 "recorded. Decision #32: carry it forward in the target "
                 "level's vocabulary, or declare `" << kDroppedAttr << " = { "
              << name << " = \"<reason>\" }` on the function or module. Legal "
                 "reasons: represented_in_type, target_invariant, "
                 "consumed_by_pass, not_yet_carried:<plan item>.";
          anyError = true;
          continue;
        }

        auto reason = dyn_cast<StringAttr>(reasonAttr);
        if (!reason) {
          module.emitError()
              << "METADATA_OBLIGATION_UNKNOWN_REASON: the reason for dropping `"
              << name << "` in @" << scope << " must be a string.";
          anyError = true;
          continue;
        }
        declaredAndUsed.insert(name);

        StringRef text = reason.getValue();
        if (text.starts_with("not_yet_carried")) {
          // Declared debt has to name its owner, or it is just a silent drop
          // with extra syntax.
          StringRef item = text.drop_front(StringRef("not_yet_carried").size());
          if (!item.consume_front(":") || item.trim().empty()) {
            module.emitError()
                << "METADATA_OBLIGATION_DEBT_UNATTRIBUTED: `" << name
                << "` in @" << scope
                << " is declared `not_yet_carried` with no plan item. Write "
                   "`not_yet_carried:<item>` (e.g. not_yet_carried:W1.1) so the "
                   "debt has an owner.";
            anyError = true;
          }
          continue;
        }
        if (!isKnownReason(text)) {
          module.emitError()
              << "METADATA_OBLIGATION_UNKNOWN_REASON: `" << text
              << "` is not a legal drop reason for `" << name << "` in @"
              << scope
              << ". Legal: represented_in_type, target_invariant, "
                 "consumed_by_pass, not_yet_carried:<plan item>.";
          anyError = true;
        }
      }

      // A declared drop for an attribute that did not drop. See the header:
      // this would license a real future drop, and reads in review as a
      // considered exception while carrying nothing.
      if (dropped) {
        for (NamedAttribute d : dropped) {
          StringRef name = d.getName().strref();
          if (declaredAndUsed.contains(name)) continue;
          bool stillPresent = it != current.end() && it->second.contains(name);
          if (!stillPresent) continue;
          module.emitError()
              << "METADATA_OBLIGATION_STALE_DECLARATION: @" << scope
              << " declares `" << name
              << "` dropped, but it is still present after the boundary. Remove "
                 "the declaration -- an unused exception licenses a future drop "
                 "nobody reviewed.";
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
