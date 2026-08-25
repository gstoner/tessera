#include "Tessera/Transforms/SemanticEffects.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/Support/SHA256.h"

#include "mlir/Interfaces/CastInterfaces.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Interfaces/ViewLikeInterface.h"

using namespace mlir;

namespace tessera {

SemanticEffectLevel joinSemanticEffects(SemanticEffectLevel lhs,
                                        SemanticEffectLevel rhs) {
  return static_cast<int>(lhs) >= static_cast<int>(rhs) ? lhs : rhs;
}

llvm::StringRef stringifySemanticEffect(SemanticEffectLevel effect) {
  switch (effect) {
  case SemanticEffectLevel::Pure: return "pure";
  case SemanticEffectLevel::Random: return "random";
  case SemanticEffectLevel::Movement: return "movement";
  case SemanticEffectLevel::State: return "state";
  case SemanticEffectLevel::Collective: return "collective";
  case SemanticEffectLevel::Memory: return "memory";
  case SemanticEffectLevel::IO: return "io";
  case SemanticEffectLevel::Top: return "top";
  }
  return "top";
}

bool parseSemanticEffect(llvm::StringRef value,
                         SemanticEffectLevel &effect) {
  if (value == "pure") effect = SemanticEffectLevel::Pure;
  else if (value == "random") effect = SemanticEffectLevel::Random;
  else if (value == "movement") effect = SemanticEffectLevel::Movement;
  else if (value == "state") effect = SemanticEffectLevel::State;
  else if (value == "collective") effect = SemanticEffectLevel::Collective;
  else if (value == "memory") effect = SemanticEffectLevel::Memory;
  else if (value == "io") effect = SemanticEffectLevel::IO;
  else if (value == "top") effect = SemanticEffectLevel::Top;
  else return false;
  return true;
}

SemanticEffectLevel getRegisteredSemanticEffect(Operation *op) {
  if (auto attr = op->getAttrOfType<StringAttr>("tessera.effect_kind")) {
    SemanticEffectLevel effect;
    if (!parseSemanticEffect(attr.getValue(), effect))
      return SemanticEffectLevel::Top;
    // A textual attribute may refine a registered side effect (memory write
    // into state/random/collective), but it cannot erase one by claiming pure.
    if (effect == SemanticEffectLevel::Pure && !isMemoryEffectFree(op))
      return SemanticEffectLevel::Top;
    return effect;
  }

  // MemoryEffectOpInterface is the registered fallback for non-Graph
  // dialects. `isMemoryEffectFree` also recognizes the Pure trait.
  if (isMemoryEffectFree(op))
    return SemanticEffectLevel::Pure;
  if (isa<MemoryEffectOpInterface>(op))
    return SemanticEffectLevel::Memory;
  return SemanticEffectLevel::Top;
}

bool isSemanticSchedulingBarrier(Operation *op) {
  if (op->getNumRegions() != 0)
    return true;

  if (auto alias = op->getAttrOfType<StringAttr>("tessera.aliasing"))
    if (alias.getValue() != "none")
      return true;
  if (isa<CastOpInterface, ViewLikeOpInterface>(op))
    return true;

  // Even explicitly keyed RNG retains a sample identity. Moving waits across
  // it is deferred until a consumer has a proof that the key/counter lineage
  // makes the transformation legal.
  if (auto stochastic =
          op->getAttrOfType<StringAttr>("tessera.stochastic_identity"))
    if (stochastic.getValue() != "none")
      return true;

  return getRegisteredSemanticEffect(op) != SemanticEffectLevel::Pure;
}


static constexpr const char *kRecordedProductClass =
    "tessera.recorded_product.effect_class";
static constexpr const char *kRecordedProductDigest =
    "tessera.recorded_product.digest";
static constexpr const char *kRecordedProductSchemaAttr =
    "tessera.recorded_product.schema";
static constexpr const char *kRecordedProductPayload =
    "tessera.recorded_product.payload";
static constexpr const char *kRecordedProductSchemaValue =
    "tessera.recorded_product.v1";

static bool isLowercaseHex64(llvm::StringRef text) {
  if (text.size() != 64)
    return false;
  for (char c : text)
    if (!((c >= '0' && c <= '9') || (c >= 'a' && c <= 'f')))
      return false;
  return true;
}

std::string recordedProductFailure(mlir::Operation *op,
                                   llvm::StringRef requiredClass) {
  auto cls = op->getAttrOfType<mlir::StringAttr>(kRecordedProductClass);
  if (!cls)
    return "carries no recorded product";
  if (cls.getValue() != requiredClass)
    return ("carries a '" + cls.getValue() + "' product, not '" +
            requiredClass + "'")
        .str();
  auto schema = op->getAttrOfType<mlir::StringAttr>(kRecordedProductSchemaAttr);
  if (!schema || schema.getValue() != kRecordedProductSchemaValue)
    return "declares no supported recorded-product schema";
  auto digest = op->getAttrOfType<mlir::StringAttr>(kRecordedProductDigest);
  if (!digest || !isLowercaseHex64(digest.getValue()))
    return "has no lowercase 64-hex content digest";
  auto payload = op->getAttrOfType<mlir::StringAttr>(kRecordedProductPayload);
  if (!payload || payload.getValue().empty())
    return "carries a digest but not the payload it addresses, so the "
           "product cannot be verified";
  llvm::SHA256 hasher;
  hasher.update(payload.getValue());
  if (llvm::toHex(hasher.final(), /*LowerCase=*/true) != digest.getValue())
    return "payload does not hash to its declared digest; the recorded "
           "product is not the one addressed";
  // The payload must describe THIS operation, in THIS class: otherwise a
  // valid product could be copied from another op and still verify.
  std::string opNeedle = ("\"op\":\"" + op->getName().getStringRef() + "\"").str();
  if (!payload.getValue().contains(opNeedle))
    return ("payload does not name " + op->getName().getStringRef() +
            "; a product recorded for another operation cannot admit this one")
        .str();
  std::string classNeedle =
      ("\"effect_class\":\"" + requiredClass + "\"").str();
  if (!payload.getValue().contains(classNeedle))
    return "payload's effect class disagrees with the declared attribute";
  return {};
}

bool carriesVerifiedRecordedProduct(mlir::Operation *op,
                                    llvm::StringRef requiredClass) {
  return recordedProductFailure(op, requiredClass).empty();
}

} // namespace tessera
