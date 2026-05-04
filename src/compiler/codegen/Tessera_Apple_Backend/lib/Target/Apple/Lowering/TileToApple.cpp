//===- TileToApple.cpp - Tile IR -> Apple Silicon Target IR --*- C++ -*-===//
//
// Two passes that walk a Tile IR module and emit hardware-free Apple Silicon
// Target IR. Mirrors the text-based pipeline in
// python/tessera/compiler/matmul_pipeline.py:
//
//   CPU side  (-tessera-lower-to-apple_cpu):
//     tessera.matmul / tessera.gemm        -> tessera_apple.cpu.accelerate_gemm
//     tessera.softmax / tessera.softmax_safe -> tessera_apple.cpu.vector_reduce
//     tessera.kv_cache.*                   -> tessera_apple.diagnostic
//     anything else                        -> tessera_apple.cpu.vector_op
//
//   GPU side  (-tessera-lower-to-apple_gpu):
//     tessera.flash_attn                   -> metal_kernel("flash_attn_contract")
//                                             + tessera_apple.gpu.dispatch
//     tessera.kv_cache.*                   -> tessera_apple.diagnostic
//     anything else                        -> tessera_apple.gpu.metal_kernel
//                                             + tessera_apple.gpu.dispatch
//
// Lowering matches on op-name strings so it can consume both the registered
// Tile IR dialect and the unregistered text-form artifacts the Python pipeline
// emits, without binding to Tile IR op C++ classes.
//
//===----------------------------------------------------------------------===//

#include "Tessera/Target/Apple/Passes.h"
#include "Tessera/Target/Apple/TesseraAppleDialect.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/OperationSupport.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/LLVM.h"

#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"

#include <string>

using namespace ::mlir;

namespace tessera {
namespace apple {

namespace {

//===----------------------------------------------------------------------===//
// Helpers
//===----------------------------------------------------------------------===//

constexpr llvm::StringLiteral kCPUFunc = "tessera_apple.cpu.func";
constexpr llvm::StringLiteral kCPUAccelerateGemm =
    "tessera_apple.cpu.accelerate_gemm";
constexpr llvm::StringLiteral kCPUVectorReduce =
    "tessera_apple.cpu.vector_reduce";
constexpr llvm::StringLiteral kCPUVectorOp = "tessera_apple.cpu.vector_op";

constexpr llvm::StringLiteral kGPUFunc = "tessera_apple.gpu.func";
constexpr llvm::StringLiteral kGPUMetalKernel =
    "tessera_apple.gpu.metal_kernel";
constexpr llvm::StringLiteral kGPUDispatch = "tessera_apple.gpu.dispatch";

constexpr llvm::StringLiteral kDiagnostic = "tessera_apple.diagnostic";

// Op-name predicates that mirror the Python pipeline.
bool isMatmul(llvm::StringRef name) {
  return name == "tessera.matmul" || name == "tessera.gemm";
}

bool isReduction(llvm::StringRef name) {
  return name == "tessera.softmax" || name == "tessera.softmax_safe";
}

bool isFlashAttn(llvm::StringRef name) { return name == "tessera.flash_attn"; }

bool isKVCache(llvm::StringRef name) {
  return name.starts_with("tessera.kv_cache.");
}

// Tile-level op that the lowering should consume. The Python text pipeline
// keeps the Graph IR op name as the `source` attribute on the Tile IR op, so
// we match either the Graph IR spelling directly or any tile.* op carrying a
// `source` string attribute.
bool isLowerable(Operation *op) {
  llvm::StringRef name = op->getName().getStringRef();
  if (name.starts_with("tessera_apple."))
    return false; // already lowered
  if (isMatmul(name) || isReduction(name) || isFlashAttn(name) ||
      isKVCache(name))
    return true;
  if (name.starts_with("tessera.tile.") || name.starts_with("tile."))
    return op->hasAttr("source");
  return false;
}

// Resolve the canonical "source" name. Either the original Graph IR op name
// (preserved on Tile IR ops as an attribute) or the op's own MLIR name.
std::string canonicalSource(Operation *op) {
  if (auto src = op->getAttrOfType<StringAttr>("source"))
    return src.getValue().str();
  return op->getName().getStringRef().str();
}

std::string resolveResult(Operation *op, int64_t ordinal) {
  if (auto r = op->getAttrOfType<StringAttr>("result"))
    return r.getValue().str();
  return ("v" + llvm::Twine(ordinal)).str();
}

int64_t resolveOrdinal(Operation *op, int64_t fallback) {
  if (auto o = op->getAttrOfType<IntegerAttr>("ordinal"))
    return o.getInt();
  return fallback;
}

// Build an op of the given name with the standard
// {source, result, ordinal} attributes plus any extra attributes provided.
// Inserts before `op`.
Operation *buildAppleOp(OpBuilder &b, Operation *op, llvm::StringRef opName,
                        int64_t ordinal,
                        llvm::ArrayRef<NamedAttribute> extra = {}) {
  MLIRContext *ctx = op->getContext();
  OperationState state(op->getLoc(), opName);
  state.addAttribute("source", b.getStringAttr(canonicalSource(op)));
  state.addAttribute("result", b.getStringAttr(resolveResult(op, ordinal)));
  state.addAttribute("ordinal",
                     IntegerAttr::get(IntegerType::get(ctx, 64), ordinal));
  for (auto &na : extra)
    state.addAttribute(na.getName(), na.getValue());
  b.setInsertionPoint(op);
  return b.create(state);
}

// Build a tessera_apple.diagnostic op with severity / reason for unsupported
// ops on a given target.
Operation *buildDiagnostic(OpBuilder &b, Operation *op, int64_t ordinal,
                           llvm::StringRef severity, llvm::StringRef reason) {
  llvm::SmallVector<NamedAttribute, 2> extra;
  extra.emplace_back(b.getStringAttr("severity"), b.getStringAttr(severity));
  extra.emplace_back(b.getStringAttr("reason"), b.getStringAttr(reason));
  return buildAppleOp(b, op, kDiagnostic, ordinal, extra);
}

// Walk all "lowerable" ops in the module. Collected up front so that op
// erasure during rewriting does not invalidate the iterator.
llvm::SmallVector<Operation *> collectLowerableOps(ModuleOp module) {
  llvm::SmallVector<Operation *> ops;
  module.walk([&](Operation *op) {
    if (isLowerable(op))
      ops.push_back(op);
  });
  return ops;
}

//===----------------------------------------------------------------------===//
// CPU pass
//===----------------------------------------------------------------------===//

struct LowerTileToAppleCPUPass
    : public PassWrapper<LowerTileToAppleCPUPass, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(LowerTileToAppleCPUPass)

  llvm::StringRef getArgument() const final {
    return "tile-to-apple_cpu";
  }
  llvm::StringRef getDescription() const final {
    return "Lower Tessera Tile IR to Apple Silicon CPU Target IR "
           "(Accelerate / vecLib / BNNS artifacts)";
  }
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<TesseraApple_Dialect>();
  }

  void runOnOperation() override {
    ModuleOp module = getOperation();
    OpBuilder builder(&getContext());

    int64_t ordinal = 0;
    for (Operation *op : collectLowerableOps(module)) {
      llvm::StringRef name = op->getName().getStringRef();
      llvm::StringRef src =
          op->getAttrOfType<StringAttr>("source")
              ? op->getAttrOfType<StringAttr>("source").getValue()
              : name;

      if (isKVCache(name) || isKVCache(src)) {
        buildDiagnostic(builder, op, ordinal, "unsupported",
                        "KV-cache target lowering is not implemented for "
                        "Apple CPU in this phase");
      } else if (isMatmul(name) || isMatmul(src)) {
        llvm::SmallVector<NamedAttribute, 2> extra;
        extra.emplace_back(builder.getStringAttr("framework"),
                           builder.getStringAttr("Accelerate"));
        extra.emplace_back(builder.getStringAttr("abi"),
                           builder.getStringAttr("cblas_sgemm"));
        buildAppleOp(builder, op, kCPUAccelerateGemm, ordinal, extra);
      } else if (isReduction(name) || isReduction(src)) {
        llvm::SmallVector<NamedAttribute, 2> extra;
        extra.emplace_back(builder.getStringAttr("framework"),
                           builder.getStringAttr("Accelerate"));
        extra.emplace_back(builder.getStringAttr("abi"),
                           builder.getStringAttr("vDSP"));
        buildAppleOp(builder, op, kCPUVectorReduce, ordinal, extra);
      } else {
        llvm::SmallVector<NamedAttribute, 2> extra;
        extra.emplace_back(builder.getStringAttr("framework"),
                           builder.getStringAttr("Accelerate"));
        extra.emplace_back(builder.getStringAttr("abi"),
                           builder.getStringAttr("vecLib"));
        buildAppleOp(builder, op, kCPUVectorOp, ordinal, extra);
      }
      op->erase();
      ++ordinal;
    }
  }
};

//===----------------------------------------------------------------------===//
// GPU pass
//===----------------------------------------------------------------------===//

struct LowerTileToAppleGPUPass
    : public PassWrapper<LowerTileToAppleGPUPass, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(LowerTileToAppleGPUPass)

  llvm::StringRef getArgument() const final {
    return "tile-to-apple_gpu";
  }
  llvm::StringRef getDescription() const final {
    return "Lower Tessera Tile IR to Apple Silicon GPU Target IR "
           "(Metal / MPS artifacts)";
  }
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<TesseraApple_Dialect>();
  }

  void runOnOperation() override {
    ModuleOp module = getOperation();
    OpBuilder builder(&getContext());

    int64_t ordinal = 0;
    for (Operation *op : collectLowerableOps(module)) {
      llvm::StringRef name = op->getName().getStringRef();
      llvm::StringRef src =
          op->getAttrOfType<StringAttr>("source")
              ? op->getAttrOfType<StringAttr>("source").getValue()
              : name;

      bool emittedKernel = false;
      if (isKVCache(name) || isKVCache(src)) {
        buildDiagnostic(builder, op, ordinal, "unsupported",
                        "KV-cache target lowering is not implemented for "
                        "Apple GPU in this phase");
      } else if (isFlashAttn(name) || isFlashAttn(src)) {
        llvm::SmallVector<NamedAttribute, 3> extra;
        extra.emplace_back(builder.getStringAttr("kernel"),
                           builder.getStringAttr("flash_attn_contract"));
        extra.emplace_back(builder.getStringAttr("framework"),
                           builder.getStringAttr("Metal"));
        extra.emplace_back(builder.getStringAttr("status"),
                           builder.getStringAttr("artifact_only"));
        buildAppleOp(builder, op, kGPUMetalKernel, ordinal, extra);
        emittedKernel = true;
      } else {
        llvm::SmallVector<NamedAttribute, 2> extra;
        extra.emplace_back(builder.getStringAttr("framework"),
                           builder.getStringAttr("Metal"));
        extra.emplace_back(builder.getStringAttr("threadgroup_memory"),
                           builder.getStringAttr("auto"));
        buildAppleOp(builder, op, kGPUMetalKernel, ordinal, extra);
        emittedKernel = true;
      }

      if (emittedKernel) {
        // Pair every kernel with a dispatch artifact, matching the Python
        // text pipeline.
        MLIRContext *ctx = &getContext();
        OperationState dispatchState(op->getLoc(), kGPUDispatch);
        dispatchState.addAttribute(
            "ordinal", IntegerAttr::get(IntegerType::get(ctx, 64), ordinal));
        dispatchState.addAttribute("queue",
                                   builder.getStringAttr("MTLCommandQueue"));
        dispatchState.addAttribute("artifact",
                                   builder.getStringAttr("metallib"));
        builder.setInsertionPoint(op);
        builder.create(dispatchState);
      }

      op->erase();
      ++ordinal;
    }
  }
};

} // namespace

//===----------------------------------------------------------------------===//
// Factory functions (declared in Passes.h)
//===----------------------------------------------------------------------===//

std::unique_ptr<Pass> createLowerTileToAppleCPUPass() {
  return std::make_unique<LowerTileToAppleCPUPass>();
}

std::unique_ptr<Pass> createLowerTileToAppleGPUPass() {
  return std::make_unique<LowerTileToAppleGPUPass>();
}

} // namespace apple
} // namespace tessera
