//===- Autotune.cpp --------------------------------------------*- C++ -*-===//
//
// SpectralAutotunePass: computes a deterministic cache key for each spectral
// exec op (forward / inverse / conv FFT) so that an offline / SQLite-backed
// autotuner can persist best (radix_seq, tile_shape, pipeline_stages) for
// each (axes × per_axis_len × elem_precision × acc_precision × target) tuple.
//
// The key derivation here mirrors the v2 schema used by
// `compiler/autotune_v2.py`:
//
//     cache_key = fnv1a_64("axes={...}|len={...}|elem=X|acc=Y|target=Z|"
//                          "stages={...}|tile={...}")
//
// We attach:
//   tessera.autotune.cache_key   : i64StringAttr   (printable hex form)
//   tessera.autotune.cached      : BoolAttr        (always false here —
//                                                   real cache lookups happen
//                                                   at runtime invocation)
//   tessera.autotune.knobs       : DictionaryAttr  (radix_seq, tile, pipe)
//
// The pass is deterministic and side-effect-free.  When the autotuner
// callback is hooked up later, it just consults this key.
//
//===----------------------------------------------------------------------===//

#include "tessera/Spectral/SpectralPasses.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"

#include "llvm/ADT/Hashing.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/Support/Format.h" // format_hex_no_prefix
#include "llvm/Support/raw_ostream.h"

#include <cstdint>
#include <sstream>

using namespace mlir;

namespace tessera {
namespace {

static uint64_t fnv1a64(StringRef s) {
  uint64_t h = 0xcbf29ce484222325ULL;
  for (unsigned char c : s) {
    h ^= c;
    h *= 0x100000001b3ULL;
  }
  return h;
}

static std::string toFlatString(ArrayAttr a) {
  std::string out;
  llvm::raw_string_ostream os(out);
  os << "[";
  for (size_t i = 0; i < a.size(); ++i) {
    if (i)
      os << ",";
    if (auto ia = dyn_cast<IntegerAttr>(a[i]))
      os << ia.getInt();
    else if (auto sa = dyn_cast<StringAttr>(a[i]))
      os << sa.getValue();
    else if (auto aa = dyn_cast<ArrayAttr>(a[i]))
      os << toFlatString(aa);
  }
  os << "]";
  return out;
}

struct AutotunePass
    : public PassWrapper<AutotunePass, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(AutotunePass)

  StringRef getArgument() const final { return "tessera-spectral-autotune"; }
  StringRef getDescription() const final {
    return "Compute deterministic autotune cache keys + default knobs for "
           "tessera_spectral exec ops.";
  }

  void runOnOperation() override {
    ModuleOp mod = getOperation();
    MLIRContext *ctx = &getContext();
    OpBuilder builder(ctx);

    // Read target backend hint from module attr if present.
    StringRef targetHint = "cpu";
    if (auto a = mod->getAttrOfType<StringAttr>("tessera.target"))
      targetHint = a.getValue();

    mod.walk([&](Operation *op) {
      StringRef name = op->getName().getStringRef();
      if (name != "tessera_spectral.fft" && name != "tessera_spectral.ifft" &&
          name != "tessera_spectral.conv_fft")
        return WalkResult::advance();
      if (op->getNumOperands() < 1)
        return WalkResult::advance();
      Operation *planDef = op->getOperand(0).getDefiningOp();
      if (!planDef || planDef->getName().getStringRef() != "tessera_spectral.plan")
        return WalkResult::advance();

      std::string keyMaterial;
      llvm::raw_string_ostream os(keyMaterial);
      if (auto axes = planDef->getAttrOfType<ArrayAttr>("axes"))
        os << "axes=" << toFlatString(axes);
      if (auto perAxis =
              op->getAttrOfType<ArrayAttr>("tessera.spectral.per_axis_len"))
        os << "|len=" << toFlatString(perAxis);
      if (auto elem =
              planDef->getAttrOfType<StringAttr>("elem_precision"))
        os << "|elem=" << elem.getValue();
      if (auto acc = planDef->getAttrOfType<StringAttr>("acc_precision"))
        os << "|acc=" << acc.getValue();
      os << "|target=" << targetHint;
      if (auto stages =
              op->getAttrOfType<ArrayAttr>("tessera.spectral.stages"))
        os << "|stages=" << toFlatString(stages);
      if (auto tiles =
              op->getAttrOfType<ArrayAttr>("tessera.transpose.tile_shapes"))
        os << "|tile=" << toFlatString(tiles);
      os << "|kind=" << name;

      uint64_t h = fnv1a64(os.str());
      std::string keyHex;
      {
        llvm::raw_string_ostream ks(keyHex);
        ks << "ts-fft-";
        ks << llvm::format_hex_no_prefix(h, 16);
      }
      op->setAttr("tessera.autotune.cache_key",
                  StringAttr::get(ctx, keyHex));
      op->setAttr("tessera.autotune.cached", builder.getBoolAttr(false));

      // Default knobs.  Real autotuner overwrites these from cache.
      NamedAttrList knobs;
      knobs.append("pipeline_stages", builder.getI64IntegerAttr(2));
      knobs.append("warp_specialized", builder.getBoolAttr(targetHint != "cpu"));
      knobs.append("persistent", builder.getBoolAttr(false));
      op->setAttr("tessera.autotune.knobs",
                  DictionaryAttr::get(ctx, knobs));
      return WalkResult::advance();
    });
  }
};

} // namespace

std::unique_ptr<mlir::Pass> createSpectralAutotunePass() {
  return std::make_unique<AutotunePass>();
}

} // namespace tessera
