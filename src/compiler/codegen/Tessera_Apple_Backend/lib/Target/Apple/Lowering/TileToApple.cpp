//===- TileToApple.cpp - Tile IR -> Apple Silicon Target IR --*- C++ -*-===//
//
// Two passes that walk a Tile IR module and emit hardware-free Apple Silicon
// Target IR. Mirrors the text-based pipeline in
// python/tessera/compiler/matmul_pipeline.py:
//
//   CPU side  (-tessera-lower-to-apple_cpu):
//     tessera.matmul / tessera.gemm        -> tessera_apple.cpu.accelerate_gemm
//     tessera.softmax / tessera.softmax_safe -> tessera_apple.cpu.vector_reduce
//     tessera.kv_cache.*                   -> tessera_apple.cpu.kv_cache_op
//     anything else                        -> tessera_apple.cpu.vector_op
//
//   GPU side  (-tessera-lower-to-apple_gpu):
//     tessera.flash_attn                   -> metal_kernel("flash_attn_contract")
//                                             + tessera_apple.gpu.dispatch
//     tessera.kv_cache.*                   -> tessera_apple.gpu.kv_cache_op
//     anything else                        -> tessera_apple.gpu.metal_kernel
//                                             + tessera_apple.gpu.dispatch
//
// kv_cache_coverage_matrix.md (2026-05-10): the kv_cache.* lowerings
// previously emitted `tessera_apple.diagnostic("unsupported")`. They
// now emit real `tessera_apple.{cpu,gpu}.kv_cache_op` artifacts that
// carry the original Graph IR op as a `kind` attribute; the Python
// runtime dispatches on (target, kind) and routes to
// `tessera.cache.KVCacheHandle.{append,read,prune,...}` — same
// reference path that backs the numpy execution route.
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

constexpr llvm::StringLiteral kCPUAccelerateGemm =
    "tessera_apple.cpu.accelerate_gemm";
constexpr llvm::StringLiteral kCPUVectorReduce =
    "tessera_apple.cpu.vector_reduce";
constexpr llvm::StringLiteral kCPUVectorOp = "tessera_apple.cpu.vector_op";

constexpr llvm::StringLiteral kGPUMetalKernel =
    "tessera_apple.gpu.metal_kernel";
constexpr llvm::StringLiteral kGPUDispatch = "tessera_apple.gpu.dispatch";

constexpr llvm::StringLiteral kDiagnostic = "tessera_apple.diagnostic";

// kv_cache_coverage_matrix.md — Apple CPU + GPU KV-cache lowering targets.
// `kv_cache.{create,append,prune,read}` map to a single tessera_apple.*
// op carrying the original Graph IR op as a `kind` attribute. The Python
// runtime path dispatches on (target, kind) and routes to
// `tessera.cache.KVCacheHandle.{append,read,prune,...}` — the same
// reference path that backs the numpy execution route.
constexpr llvm::StringLiteral kCPUKVCacheOp = "tessera_apple.cpu.kv_cache_cpu";
constexpr llvm::StringLiteral kGPUKVCacheOp = "tessera_apple.gpu.kv_cache_gpu";

// Op-name predicates that mirror the Python pipeline.
bool isMatmul(llvm::StringRef name) {
  return name == "tessera.matmul" || name == "tessera.gemm";
}

bool isReduction(llvm::StringRef name) {
  return name == "tessera.softmax" || name == "tessera.softmax_safe";
}

bool isFlashAttn(llvm::StringRef name) { return name == "tessera.flash_attn"; }

bool isRoPE(llvm::StringRef name) { return name == "tessera.rope"; }

// L-series linalg pilot (2026-06-02): Cholesky factorization.  The Tile layer
// emits `tile.cholesky` (carrying source = "tessera.cholesky"); match either.
bool isCholesky(llvm::StringRef name) {
  return name == "tessera.cholesky" || name == "tile.cholesky";
}

bool isKVCache(llvm::StringRef name) {
  return name.starts_with("tessera.kv_cache.");
}

// G3 — the full set of Graph IR ops the Apple GPU runtime executes natively.
// This MIRRORS the Python runtime envelope
// driver._APPLE_GPU_{MPS,MSL,MPSGRAPH}_OPS (the single source of truth): MPS
// (matmul/gemm/batched_gemm), custom MSL (rope/flash_attn/softmax[_safe]/gelu),
// and the MPSGraph Tier-1 activations/norms. The drift test
// `test_apple_gpu_tile_pass_status_matches_envelope` runs this pass over every
// envelope op and fails if the two ever diverge — so adding a runtime op in
// driver.py forces a matching update here.
bool isAppleGpuRuntimeOp(llvm::StringRef n) {
  static constexpr llvm::StringLiteral kRuntimeOps[] = {
      // MPS
      "tessera.matmul", "tessera.gemm", "tessera.batched_gemm",
      // custom MSL
      "tessera.rope", "tessera.flash_attn", "tessera.softmax",
      "tessera.softmax_safe", "tessera.gelu",
      // MPSGraph Tier-1 activations / norms
      "tessera.abs", "tessera.absolute", "tessera.exp", "tessera.layer_norm",
      "tessera.log", "tessera.log_softmax", "tessera.neg", "tessera.negative",
      "tessera.relu", "tessera.rmsnorm", "tessera.rmsnorm_safe", "tessera.rsqrt",
      "tessera.sigmoid", "tessera.sigmoid_safe", "tessera.silu",
      "tessera.silu_mul", "tessera.softplus", "tessera.sqrt", "tessera.tanh",
      // Task C (2026-06-01) — conv2d / conv3d. Project 5 wired the
      // encode-session lane for conv2d (`tessera_apple_gpu_conv2d_dev_f32_enc`);
      // Sprint A extended it to {f16, bf16}. conv3d uses an im2col + GPU
      // batched matmul decomposition. Both belong on the metal_runtime
      // rung — without them, the C++ pass silently demoted conv to
      // artifact_only even though the runtime executes it.
      "tessera.conv2d", "tessera.conv3d",
      // L-series linalg pilot (2026-06-02) — cholesky executes on the GPU via
      // the real MSL kernel `tessera_apple_gpu_cholesky_f32`
      // (driver._APPLE_GPU_LINALG_OPS).  Closes APPLE_AUDIT glass-jaw #10 for
      // cholesky; tri_solve is the remaining LINALG member (same template).
      "tessera.cholesky"};
  for (const auto &r : kRuntimeOps)
    if (n == r)
      return true;
  return false;
}

// Tile-level op that the lowering should consume. The Python text pipeline
// keeps the Graph IR op name as the `source` attribute on the Tile IR op, so
// we match either the Graph IR spelling directly or any tile.* op carrying a
// `source` string attribute.
bool isLowerable(Operation *op) {
  llvm::StringRef name = op->getName().getStringRef();
  if (name.starts_with("tessera_apple."))
    return false; // already lowered
  if (isMatmul(name) || isReduction(name) || isFlashAttn(name) || isRoPE(name) ||
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

// Erase a lowered tile op without leaving dangling SSA uses.
//
// The artifact projection emits *value-less* tessera_apple.* ops, so historically
// the original tile op had no results/uses and a bare `op->erase()` was safe.
// Real SSA Tile IR (e.g. the C++ TilingPass output, `tile.cholesky : (T) -> T`)
// does carry a used result; erasing it blindly leaves `func.return` pointing at
// freed IR — a verifier crash.  This rebinds each used result to a same-typed
// operand when one exists (cholesky's result type matches its input), keeping
// the module valid.  The executable semantics live in the emitted artifact's
// `symbol` (consumed by the runtime per L6), not in this dataflow husk.  If no
// type-compatible replacement exists (e.g. matmul, where result ≠ operand
// shapes), the op is left in place rather than erased — never crash.
// L-series linalg pilot (2026-06-02).
bool safeEraseLowered(Operation *op) {
  for (Value res : op->getResults()) {
    if (res.use_empty())
      continue;
    Value repl;
    for (Value operand : op->getOperands())
      if (operand.getType() == res.getType()) {
        repl = operand;
        break;
      }
    if (!repl)
      return false; // cannot safely erase; leave the op in place
    res.replaceAllUsesWith(repl);
  }
  op->erase();
  return true;
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
    registry.insert<TesseraAppleDialect>();
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
        // kv_cache_coverage_matrix.md (2026-05-10) — replace the
        // historical "unsupported" diagnostic with a real lowering to
        // tessera_apple.cpu.kv_cache_op. The op carries the original
        // Graph IR op spelling as `kind` so the Python runtime can
        // dispatch correctly without re-parsing the source string.
        llvm::StringRef kind = isKVCache(name) ? name : src;
        llvm::SmallVector<NamedAttribute, 3> extra;
        extra.emplace_back(builder.getStringAttr("framework"),
                           builder.getStringAttr("Tessera"));
        extra.emplace_back(builder.getStringAttr("abi"),
                           builder.getStringAttr("kv_cache_handle"));
        extra.emplace_back(builder.getStringAttr("kind"),
                           builder.getStringAttr(kind));
        buildAppleOp(builder, op, kCPUKVCacheOp, ordinal, extra);
      } else if (isCholesky(name) || isCholesky(src)) {
        // L-series linalg pilot — Apple CPU Cholesky via Accelerate's LAPACK
        // (spotrf).  Reuses the registered generic `tessera_apple.cpu.vector_op`
        // (the dialect rejects unregistered ops); the `abi` + `symbol` attrs
        // carry the linalg identity.  The seam-closure executor (L6) reads
        // `symbol` straight off this op to pick the C ABI entry.
        llvm::SmallVector<NamedAttribute, 4> extra;
        extra.emplace_back(builder.getStringAttr("framework"),
                           builder.getStringAttr("Accelerate"));
        extra.emplace_back(builder.getStringAttr("abi"),
                           builder.getStringAttr("lapack_spotrf"));
        extra.emplace_back(builder.getStringAttr("op_kind"),
                           builder.getStringAttr("cholesky"));
        extra.emplace_back(builder.getStringAttr("symbol"),
                           builder.getStringAttr("tessera_apple_cpu_cholesky_f32"));
        buildAppleOp(builder, op, kCPUVectorOp, ordinal, extra);
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
      } else if (isRoPE(name) || isRoPE(src)) {
        llvm::SmallVector<NamedAttribute, 3> extra;
        extra.emplace_back(builder.getStringAttr("framework"),
                           builder.getStringAttr("Accelerate"));
        extra.emplace_back(builder.getStringAttr("abi"),
                           builder.getStringAttr("vecLib"));
        extra.emplace_back(builder.getStringAttr("pattern"),
                           builder.getStringAttr("rotary_pairs"));
        buildAppleOp(builder, op, kCPUVectorOp, ordinal, extra);
      } else {
        llvm::SmallVector<NamedAttribute, 2> extra;
        extra.emplace_back(builder.getStringAttr("framework"),
                           builder.getStringAttr("Accelerate"));
        extra.emplace_back(builder.getStringAttr("abi"),
                           builder.getStringAttr("vecLib"));
        buildAppleOp(builder, op, kCPUVectorOp, ordinal, extra);
      }
      safeEraseLowered(op);
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
    registry.insert<TesseraAppleDialect>();
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

      // G3 — execution status MUST mirror the Python runtime envelope
      // (driver._APPLE_GPU_{MPS,MSL}_OPS), the single source of truth: matmul /
      // gemm (MPS) and rope / flash_attn / softmax[_safe] / gelu (custom MSL)
      // execute on the GPU at runtime ("metal_runtime"); everything else stays an
      // inspection-only artifact ("artifact_only"). The drift test
      // test_apple_gpu_tile_pass_status_matches_envelope enforces this agreement.
      const bool runtimeClass =
          isAppleGpuRuntimeOp(name) || isAppleGpuRuntimeOp(src);
      const llvm::StringRef execStatus =
          runtimeClass ? "metal_runtime" : "artifact_only";

      bool emittedKernel = false;
      if (isKVCache(name) || isKVCache(src)) {
        // kv_cache_coverage_matrix.md (2026-05-10) — Apple GPU mirrors
        // Apple CPU: emit a tessera_apple.gpu.kv_cache_op carrying the
        // original Graph IR `kind`. The Python runtime dispatches to
        // `tessera.cache.KVCacheHandle` for execution; a future native
        // Metal kernel for the in-cache score-matrix path is gated on
        // Phase G.
        llvm::StringRef kind = isKVCache(name) ? name : src;
        llvm::SmallVector<NamedAttribute, 3> extra;
        extra.emplace_back(builder.getStringAttr("framework"),
                           builder.getStringAttr("Metal"));
        extra.emplace_back(builder.getStringAttr("abi"),
                           builder.getStringAttr("kv_cache_handle"));
        extra.emplace_back(builder.getStringAttr("kind"),
                           builder.getStringAttr(kind));
        buildAppleOp(builder, op, kGPUKVCacheOp, ordinal, extra);
      } else if (isFlashAttn(name) || isFlashAttn(src)) {
        llvm::SmallVector<NamedAttribute, 6> extra;
        extra.emplace_back(builder.getStringAttr("kernel"),
                           builder.getStringAttr("flash_attn_contract"));
        extra.emplace_back(builder.getStringAttr("framework"),
                           builder.getStringAttr("Metal"));
        extra.emplace_back(builder.getStringAttr("status"),
                           builder.getStringAttr(execStatus));
        extra.emplace_back(builder.getStringAttr("grid"),
                           builder.getStringAttr("bhn"));
        extra.emplace_back(builder.getStringAttr("threadgroup"),
                           builder.getStringAttr("64x1x1"));
        extra.emplace_back(builder.getStringAttr("temporary_memory"),
                           builder.getStringAttr("scores_lse"));
        buildAppleOp(builder, op, kGPUMetalKernel, ordinal, extra);
        emittedKernel = true;
      } else if (isCholesky(name) || isCholesky(src)) {
        // L-series linalg pilot — Apple GPU Cholesky via the custom MSL kernel
        // `tessera_apple_gpu_cholesky_f32`.  `status` mirrors the Python
        // envelope (isAppleGpuRuntimeOp ⇒ metal_runtime); `symbol` is the C ABI
        // entry the seam-closure executor (L6) reads directly off this op.
        llvm::SmallVector<NamedAttribute, 7> extra;
        extra.emplace_back(builder.getStringAttr("kernel"),
                           builder.getStringAttr("cholesky_contract"));
        extra.emplace_back(builder.getStringAttr("framework"),
                           builder.getStringAttr("Metal"));
        extra.emplace_back(builder.getStringAttr("status"),
                           builder.getStringAttr(execStatus));
        extra.emplace_back(builder.getStringAttr("symbol"),
                           builder.getStringAttr("tessera_apple_gpu_cholesky_f32"));
        extra.emplace_back(builder.getStringAttr("grid"),
                           builder.getStringAttr("single_threadgroup"));
        extra.emplace_back(builder.getStringAttr("threadgroup"),
                           builder.getStringAttr("32x1x1"));
        extra.emplace_back(builder.getStringAttr("temporary_memory"),
                           builder.getStringAttr("panel"));
        buildAppleOp(builder, op, kGPUMetalKernel, ordinal, extra);
        emittedKernel = true;
      } else if (isMatmul(name) || isMatmul(src)) {
        llvm::SmallVector<NamedAttribute, 6> extra;
        extra.emplace_back(builder.getStringAttr("kernel"),
                           builder.getStringAttr("matmul_contract"));
        extra.emplace_back(builder.getStringAttr("framework"),
                           builder.getStringAttr("MPSGraph"));
        extra.emplace_back(builder.getStringAttr("status"),
                           builder.getStringAttr(execStatus));
        extra.emplace_back(builder.getStringAttr("grid"),
                           builder.getStringAttr("mn_tiles"));
        extra.emplace_back(builder.getStringAttr("threadgroup"),
                           builder.getStringAttr("16x16x1"));
        extra.emplace_back(builder.getStringAttr("temporary_memory"),
                           builder.getStringAttr("none"));
        buildAppleOp(builder, op, kGPUMetalKernel, ordinal, extra);
        emittedKernel = true;
      } else if (isReduction(name) || isReduction(src)) {
        llvm::SmallVector<NamedAttribute, 6> extra;
        extra.emplace_back(builder.getStringAttr("kernel"),
                           builder.getStringAttr("softmax_contract"));
        extra.emplace_back(builder.getStringAttr("framework"),
                           builder.getStringAttr("MPSGraph"));
        extra.emplace_back(builder.getStringAttr("status"),
                           builder.getStringAttr(execStatus));
        extra.emplace_back(builder.getStringAttr("grid"),
                           builder.getStringAttr("rows"));
        extra.emplace_back(builder.getStringAttr("threadgroup"),
                           builder.getStringAttr("256x1x1"));
        extra.emplace_back(builder.getStringAttr("temporary_memory"),
                           builder.getStringAttr("row_max_sum"));
        buildAppleOp(builder, op, kGPUMetalKernel, ordinal, extra);
        emittedKernel = true;
      } else if (isRoPE(name) || isRoPE(src)) {
        llvm::SmallVector<NamedAttribute, 6> extra;
        extra.emplace_back(builder.getStringAttr("kernel"),
                           builder.getStringAttr("rope_contract"));
        extra.emplace_back(builder.getStringAttr("framework"),
                           builder.getStringAttr("Metal"));
        extra.emplace_back(builder.getStringAttr("status"),
                           builder.getStringAttr(execStatus));
        extra.emplace_back(builder.getStringAttr("grid"),
                           builder.getStringAttr("tokens_heads"));
        extra.emplace_back(builder.getStringAttr("threadgroup"),
                           builder.getStringAttr("128x1x1"));
        extra.emplace_back(builder.getStringAttr("temporary_memory"),
                           builder.getStringAttr("none"));
        buildAppleOp(builder, op, kGPUMetalKernel, ordinal, extra);
        emittedKernel = true;
      } else {
        llvm::SmallVector<NamedAttribute, 6> extra;
        extra.emplace_back(builder.getStringAttr("kernel"),
                           builder.getStringAttr("elementwise_contract"));
        extra.emplace_back(builder.getStringAttr("framework"),
                           builder.getStringAttr("Metal"));
        extra.emplace_back(builder.getStringAttr("threadgroup_memory"),
                           builder.getStringAttr("auto"));
        extra.emplace_back(builder.getStringAttr("status"),
                           builder.getStringAttr(execStatus));
        extra.emplace_back(builder.getStringAttr("grid"),
                           builder.getStringAttr("elements"));
        extra.emplace_back(builder.getStringAttr("temporary_memory"),
                           builder.getStringAttr("none"));
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
        dispatchState.addAttribute("execution_mode",
                                   builder.getStringAttr("metal_artifact"));
        builder.setInsertionPoint(op);
        builder.create(dispatchState);
      }

      safeEraseLowered(op);
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
