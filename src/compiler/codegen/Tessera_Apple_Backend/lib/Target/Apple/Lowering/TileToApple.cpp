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

#include "mlir/Dialect/UB/IR/UBOps.h"
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

// L-series linalg family (2026-06-02) — table-driven Tile→Apple lowering.
//
// Each linalg op carries the Tile-layer name `tile.<suffix>` plus
// `source = "tessera.<suffix>"` (set by TilingPass).  This single spec table is
// the source of truth that drives both the CPU branch (Accelerate LAPACK) and
// the GPU branch (custom MSL).  Adding a linalg op = one row here + one entry in
// TilingPass.cpp's kLinalgGraphOps + the runtime symbol.
//
//   cpuSymbol  — Apple CPU C ABI entry (Accelerate LAPACK); always present.
//   gpuSymbol  — Apple GPU MSL C ABI entry, or "" when no GPU kernel exists yet
//                (the op then lowers GPU-side as an inspection artifact).
struct LinalgSpec {
  llvm::StringLiteral graphName; // "tessera.cholesky"
  llvm::StringLiteral cpuAbi;    // "lapack_spotrf"
  llvm::StringLiteral cpuSymbol; // "tessera_apple_cpu_cholesky_f32"
  llvm::StringLiteral gpuSymbol; // "tessera_apple_gpu_cholesky_f32" or ""
};

constexpr LinalgSpec kLinalgSpecs[] = {
    {"tessera.cholesky", "lapack_spotrf", "tessera_apple_cpu_cholesky_f32",
     "tessera_apple_gpu_cholesky_f32"},
    {"tessera.tri_solve", "lapack_strtrs", "tessera_apple_cpu_tri_solve_f32",
     "tessera_apple_gpu_tri_solve_f32"},
    {"tessera.cholesky_solve", "lapack_spotrs",
     "tessera_apple_cpu_cholesky_solve_f32",
     "tessera_apple_gpu_solve_cholesky_f32"},
    {"tessera.lu", "lapack_sgetrf", "tessera_apple_cpu_lu_f32", ""},
    {"tessera.qr", "lapack_sgeqrf", "tessera_apple_cpu_qr_f32", ""},
    {"tessera.svd", "lapack_sgesvd", "tessera_apple_cpu_svd_f32",
     "tessera_apple_gpu_svd_f32"},
};

// Resolve the spec by either the Tile-IR name (tile.<suffix>) or the canonical
// Graph-IR `source` spelling (tessera.<suffix>).
const LinalgSpec *linalgSpecFor(llvm::StringRef name) {
  for (const LinalgSpec &s : kLinalgSpecs) {
    if (name == s.graphName)
      return &s;
    llvm::StringRef suffix = s.graphName;
    suffix.consume_front("tessera.");
    if (name == ("tile." + suffix).str())
      return &s;
  }
  return nullptr;
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
      // L-series linalg family (2026-06-02) — cholesky + tri_solve execute on
      // the GPU via real MSL kernels (driver._APPLE_GPU_LINALG_OPS).  The other
      // family members (cholesky_solve/lu/qr/svd) lower GPU-side as inspection
      // artifacts for now (their CPU LAPACK path is real + tested); their GPU
      // runtime wiring is a follow-on, at which point each joins this list, the
      // driver envelope, and the drift gate together.
      "tessera.cholesky", "tessera.tri_solve"};
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
// The tessera_apple.* ops this pass emits are *value-less inspection artifacts*
// (ODS: attribute-only, no results) — execution happens through the runtime
// `symbol` recorded on the artifact (the seam-closure contract), NOT through
// the lowered module's dataflow.  The lowered module is therefore a dataflow
// "husk".  But it must still be VALID IR with the original tile op erased, so
// every used result of the tile op needs a stand-in value.
//
// Earlier this rebound a single result to a same-typed *operand* (e.g.
// tri_solve's X → its B input).  That kept the IR valid but **falsely implied
// result == input**, and it could not handle multi-result ops (lu/qr/svd),
// which it left orphaned in the module (review glass-jaws R1 + R2, 2026-06-03).
//
// The honest, uniform husk is `ub.poison` — an explicitly *poisoned* (undefined)
// value of the result's type: "this SSA value is intentionally not produced by
// visible IR; its contents come from the runtime `symbol`."  Unlike
// `tensor.empty` it carries no "allocated tensor" implication and no data
// dependence, and unlike an operand rebind it works for any result type
// (static or dynamic, ranked or not) and any arity — so this is fully
// transactional: one poison per used result, replace, erase.
//
// NOTE (follow-on): a *semantics-preserving* hand-off would require the
// tessera_apple target ops to carry real operands + results in ODS (a
// dialect-wide change) so the lowering can `replaceOp` with produced values.
// Until then this is an artifact projection; the seam-closure tests prove the
// named symbol computes the correct result.
bool safeEraseLowered(Operation *op, OpBuilder &builder) {
  for (Value res : op->getResults()) {
    if (res.use_empty())
      continue;
    builder.setInsertionPoint(op);
    Value poison = ub::PoisonOp::create(builder, op->getLoc(), res.getType());
    res.replaceAllUsesWith(poison);
  }
  op->erase();
  return true;
}

// Emit a *value-producing* Apple target op (Apple Value Target IR sprint,
// 2026-06-03) and replace the Tile op directly with it.  Unlike the artifact
// projection, the value op carries the Tile op's real SSA operands and result
// types, so `op->replaceAllUsesWith(call)` is a genuine semantics-preserving
// hand-off (multi-result included).  `valueOpName` is one of
// tessera_apple.cpu.call / gpu.kernel_call / gpu.package_call; `symbol` is the
// C ABI entry the runtime dispatches to; `status` is "executable" when a
// runtime dispatcher exists.
void emitAppleValueCall(OpBuilder &b, Operation *op,
                        llvm::StringRef valueOpName, llvm::StringRef opKind,
                        llvm::StringRef symbol, llvm::StringRef abi,
                        llvm::StringRef status, llvm::StringRef framework) {
  OperationState st(op->getLoc(), valueOpName);
  st.addOperands(op->getOperands());
  st.addTypes(op->getResultTypes());
  st.addAttribute("op_kind", b.getStringAttr(opKind));
  st.addAttribute("symbol", b.getStringAttr(symbol));
  st.addAttribute("abi", b.getStringAttr(abi));
  st.addAttribute("status", b.getStringAttr(status));
  st.addAttribute("framework", b.getStringAttr(framework));
  // Sprint 3: preserve linalg semantic attrs from the source op so runtime
  // dispatch never silently assumes a default (lower/trans/unit_diag for
  // solves; full_matrices for qr/svd).  Copied verbatim when present.
  for (llvm::StringRef name :
       {"lower", "trans", "unit_diag", "full_matrices"}) {
    if (Attribute a = op->getAttr(name))
      st.addAttribute(name, a);
  }
  b.setInsertionPoint(op);
  Operation *call = b.create(st);
  op->replaceAllUsesWith(call);
  op->erase();
}

//===----------------------------------------------------------------------===//
// CPU pass
//===----------------------------------------------------------------------===//

struct LowerTileToAppleCPUPass
    : public PassWrapper<LowerTileToAppleCPUPass, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(LowerTileToAppleCPUPass)

  LowerTileToAppleCPUPass() = default;
  explicit LowerTileToAppleCPUPass(bool valueMode) : valueMode(valueMode) {}
  LowerTileToAppleCPUPass(const LowerTileToAppleCPUPass &other)
      : PassWrapper(other), valueMode(other.valueMode) {}

  // Apple Value Target IR sprint: artifact mode (default) emits metadata ops +
  // ub.poison husks; value mode emits value-producing tessera_apple.cpu.call
  // ops and replaces the Tile op directly (used by the `-full` pipeline).
  bool valueMode = false;

  llvm::StringRef getArgument() const final {
    return "tile-to-apple_cpu";
  }
  llvm::StringRef getDescription() const final {
    return "Lower Tessera Tile IR to Apple Silicon CPU Target IR "
           "(Accelerate / vecLib / BNNS artifacts)";
  }
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<TesseraAppleDialect, mlir::ub::UBDialect>();
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

      // Apple Value Target IR sprint: the `-full` pipeline lowers to
      // value-producing target ops (semantics-preserving SSA hand-off).
      if (valueMode) {
        const LinalgSpec *sp = linalgSpecFor(name);
        if (!sp)
          sp = linalgSpecFor(src);
        if (!sp) {
          op->emitError("apple_cpu value lowering: no value-producing CPU "
                        "target op for '")
              << src << "' (only the linalg family is value-converted so far)";
          signalPassFailure();
          return;
        }
        llvm::StringRef opKind = sp->graphName;
        opKind.consume_front("tessera.");
        emitAppleValueCall(builder, op, "tessera_apple.cpu.call", opKind,
                           sp->cpuSymbol, sp->cpuAbi, "executable",
                           "Accelerate");
        ++ordinal;
        continue;
      }

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
      } else if (const LinalgSpec *sp =
                     linalgSpecFor(name) ? linalgSpecFor(name)
                                         : linalgSpecFor(src)) {
        // L-series linalg family — Apple CPU via Accelerate's LAPACK.  Reuses
        // the registered generic `tessera_apple.cpu.vector_op` (the dialect
        // rejects unregistered ops); the `abi` + `symbol` attrs carry the linalg
        // identity.  The seam-closure executor reads `symbol` straight off this
        // op to pick the C ABI entry.  Table-driven via kLinalgSpecs.
        llvm::StringRef opKind = sp->graphName;
        opKind.consume_front("tessera.");
        llvm::SmallVector<NamedAttribute, 4> extra;
        extra.emplace_back(builder.getStringAttr("framework"),
                           builder.getStringAttr("Accelerate"));
        extra.emplace_back(builder.getStringAttr("abi"),
                           builder.getStringAttr(sp->cpuAbi));
        extra.emplace_back(builder.getStringAttr("op_kind"),
                           builder.getStringAttr(opKind));
        extra.emplace_back(builder.getStringAttr("symbol"),
                           builder.getStringAttr(sp->cpuSymbol));
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
      safeEraseLowered(op, builder);
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

  LowerTileToAppleGPUPass() = default;
  explicit LowerTileToAppleGPUPass(bool valueMode) : valueMode(valueMode) {}
  LowerTileToAppleGPUPass(const LowerTileToAppleGPUPass &other)
      : PassWrapper(other), valueMode(other.valueMode) {}

  bool valueMode = false;

  llvm::StringRef getArgument() const final {
    return "tile-to-apple_gpu";
  }
  llvm::StringRef getDescription() const final {
    return "Lower Tessera Tile IR to Apple Silicon GPU Target IR "
           "(Metal / MPS artifacts)";
  }
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<TesseraAppleDialect, mlir::ub::UBDialect>();
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

      // Apple Value Target IR sprint: the `-full` pipeline lowers to
      // value-producing target ops.  Only ops with executable GPU dispatch
      // (the runtime envelope) become gpu.kernel_call; everything else fails
      // with a named diagnostic (the full pipeline is value-only and must not
      // silently degrade to an artifact).
      if (valueMode) {
        const LinalgSpec *sp = linalgSpecFor(name);
        if (!sp)
          sp = linalgSpecFor(src);
        if (!sp) {
          op->emitError("apple_gpu value lowering: no value-producing GPU "
                        "target op for '")
              << src << "'";
          signalPassFailure();
          return;
        }
        if (sp->gpuSymbol.empty() || !isAppleGpuRuntimeOp(sp->graphName)) {
          op->emitError("apple_gpu value lowering: '")
              << sp->graphName
              << "' has no executable GPU dispatch yet (artifact-only); use "
                 "the artifact pipeline (tessera-lower-to-apple_gpu) instead";
          signalPassFailure();
          return;
        }
        llvm::StringRef opKind = sp->graphName;
        opKind.consume_front("tessera.");
        emitAppleValueCall(builder, op, "tessera_apple.gpu.kernel_call", opKind,
                           sp->gpuSymbol, "msl", "executable", "Metal");
        ++ordinal;
        continue;
      }

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
      } else if (const LinalgSpec *sp =
                     linalgSpecFor(name) ? linalgSpecFor(name)
                                         : linalgSpecFor(src)) {
        // L-series linalg family — Apple GPU via custom MSL kernels.  `status`
        // mirrors the Python envelope (isAppleGpuRuntimeOp ⇒ metal_runtime).
        // `symbol` is the C ABI entry the seam-closure executor reads directly
        // off this op.  Review glass-jaw R3 (2026-06-03): emit `symbol` ONLY
        // when this op actually executes on the GPU (status == metal_runtime).
        // A table `gpuSymbol` is necessary but not sufficient — cholesky_solve
        // and svd have GPU kernels in the .mm runtime but are not yet in the
        // dispatch envelope (isAppleGpuRuntimeOp / driver._APPLE_GPU_LINALG_OPS
        // / runtime.py), so they lower as artifact_only and must NOT advertise a
        // GPU symbol (that would imply a dispatch path that doesn't exist).
        // They join the runtime set — symbol + envelope + drift — together.
        llvm::StringRef opKind = sp->graphName;
        opKind.consume_front("tessera.");
        const bool gpuRuntime = (execStatus == "metal_runtime");
        llvm::SmallVector<NamedAttribute, 7> extra;
        extra.emplace_back(builder.getStringAttr("kernel"),
                           builder.getStringAttr((opKind + "_contract").str()));
        extra.emplace_back(builder.getStringAttr("framework"),
                           builder.getStringAttr("Metal"));
        extra.emplace_back(builder.getStringAttr("status"),
                           builder.getStringAttr(execStatus));
        if (gpuRuntime && !sp->gpuSymbol.empty())
          extra.emplace_back(builder.getStringAttr("symbol"),
                             builder.getStringAttr(sp->gpuSymbol));
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

      safeEraseLowered(op, builder);
      ++ordinal;
    }
  }
};

} // namespace

//===----------------------------------------------------------------------===//
// Factory functions (declared in Passes.h)
//===----------------------------------------------------------------------===//

std::unique_ptr<Pass> createLowerTileToAppleCPUPass(bool valueMode) {
  return std::make_unique<LowerTileToAppleCPUPass>(valueMode);
}

std::unique_ptr<Pass> createLowerTileToAppleGPUPass(bool valueMode) {
  return std::make_unique<LowerTileToAppleGPUPass>(valueMode);
}

} // namespace apple
} // namespace tessera
