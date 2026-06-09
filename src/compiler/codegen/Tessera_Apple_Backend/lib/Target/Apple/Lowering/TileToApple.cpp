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
#include "mlir/IR/BuiltinAttributes.h"
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

// Sprint 6: batched matmul. Kept distinct from isMatmul so the value-mode
// dispatch can route the vetted tile.batched_gemm to the batched C ABI while a
// *raw* tessera.batched_gemm (out-of-envelope: dynamic / non-f32) that survived
// tiling is still collected here and gated by the value-mode diagnostic instead
// of silently leaking through as an unlowered Graph-IR op.
bool isBatchedMatmul(llvm::StringRef name) {
  return name == "tessera.batched_gemm";
}

bool isReduction(llvm::StringRef name) {
  return name == "tessera.softmax" || name == "tessera.softmax_safe";
}

bool isFlashAttn(llvm::StringRef name) { return name == "tessera.flash_attn"; }

bool isRoPE(llvm::StringRef name) { return name == "tessera.rope"; }

bool isNativeSparseAttnFused(llvm::StringRef name) {
  return name == "tessera.native_sparse_attn_fused";
}

bool isPPOPolicyLoss(llvm::StringRef name) {
  return name == "tessera.rl.ppo_policy_loss" ||
         name == "tile.ppo_policy_loss";
}

bool isEBMEnergyQuadratic(llvm::StringRef name) {
  return name == "tessera.ebm.energy_quadratic" ||
         name == "tile.ebm_energy_quadratic";
}

bool isEBMLangevinStep(llvm::StringRef name) {
  return name == "tessera.ebm.langevin_step" ||
         name == "tile.ebm_langevin_step";
}

bool isEBMRefinement(llvm::StringRef name) {
  return name == "tessera.ebm.refinement" ||
         name == "tile.ebm_refinement";
}

bool isEBMPartitionExact(llvm::StringRef name) {
  return name == "tessera.ebm.partition_exact" ||
         name == "tile.ebm_partition_exact";
}

bool isCliffordGeometricProduct(llvm::StringRef name) {
  return name == "tessera.clifford.geometric_product" ||
         name == "tile.clifford_geometric_product";
}

bool isCliffordValueSeamOp(llvm::StringRef name) {
  return name == "tessera.clifford.geometric_product" ||
         name == "tessera.clifford.outer_product" ||
         name == "tessera.clifford.inner_product" ||
         name == "tessera.clifford.reverse" ||
         name == "tessera.clifford.grade_project" ||
         name == "tessera.clifford.norm" ||
         name == "tessera.clifford.rotor_sandwich" ||
         name == "tile.clifford_geometric_product" ||
         name == "tile.clifford_outer_product" ||
         name == "tile.clifford_inner_product" ||
         name == "tile.clifford_reverse" ||
         name == "tile.clifford_grade_project" ||
         name == "tile.clifford_norm" ||
         name == "tile.clifford_rotor_sandwich";
}

bool boolAttr(Operation *op, llvm::StringRef name) {
  if (auto attr = op->getAttrOfType<BoolAttr>(name))
    return attr.getValue();
  return false;
}

bool operandSegmentPresent(Operation *op, unsigned index) {
  auto segments = op->getAttrOfType<DenseI32ArrayAttr>("operand_segment_sizes");
  if (!segments)
    segments = op->getAttrOfType<DenseI32ArrayAttr>("operandSegmentSizes");
  if (!segments || index >= segments.asArrayRef().size())
    return false;
  return segments.asArrayRef()[index] != 0;
}

struct PPOOperandFlags {
  bool hasMask = false;
  bool hasRefKL = false;
  bool hasEntropy = false;
};

PPOOperandFlags ppoOperandFlags(Operation *op) {
  PPOOperandFlags flags;
  flags.hasMask = boolAttr(op, "has_mask") || operandSegmentPresent(op, 3);
  flags.hasRefKL = boolAttr(op, "has_ref_kl") || operandSegmentPresent(op, 4);
  flags.hasEntropy = boolAttr(op, "has_entropy") || operandSegmentPresent(op, 5);
  return flags;
}

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
      // Batch 1 (2026-06-08) — float-output elementwise math + comparison on the
      // MPSGraph unary/binary opcode lane (driver._APPLE_GPU_MPSGRAPH_OPS).
      "tessera.sin", "tessera.cos", "tessera.tan", "tessera.asin", "tessera.acos",
      "tessera.atan", "tessera.sinh", "tessera.cosh", "tessera.erf", "tessera.erfc",
      "tessera.expm1", "tessera.log1p", "tessera.reciprocal", "tessera.sign",
      "tessera.floor", "tessera.ceil", "tessera.round", "tessera.trunc",
      "tessera.add", "tessera.sub", "tessera.mul", "tessera.div",
      "tessera.maximum", "tessera.minimum", "tessera.pow", "tessera.atan2",
      "tessera.mod", "tessera.floor_div", "tessera.eq", "tessera.ne",
      "tessera.lt", "tessera.le", "tessera.gt", "tessera.ge",
      // Batch 2 (2026-06-08) — reduce/scan opcode completions.
      "tessera.logsumexp", "tessera.cummax", "tessera.cummin",
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
  if (isNativeSparseAttnFused(n))
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
  if (isMatmul(name) || isBatchedMatmul(name) || isReduction(name) ||
      isFlashAttn(name) || isRoPE(name) || isKVCache(name) ||
      isNativeSparseAttnFused(name) || isPPOPolicyLoss(name) ||
      isEBMEnergyQuadratic(name) || isEBMLangevinStep(name) ||
      isEBMRefinement(name) || isEBMPartitionExact(name) ||
      isCliffordValueSeamOp(name))
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

bool isStaticRank4F32Tensor(Type ty) {
  auto rt = llvm::dyn_cast<RankedTensorType>(ty);
  return rt && rt.getRank() == 4 && rt.hasStaticShape() &&
         rt.getElementType().isF32();
}

bool verifyNativeSparseValueEnvelope(Operation *op) {
  if (op->getNumOperands() != 4 || op->getNumResults() != 1)
    return false;
  auto qTy = llvm::dyn_cast<RankedTensorType>(op->getOperand(0).getType());
  auto kTy = llvm::dyn_cast<RankedTensorType>(op->getOperand(1).getType());
  auto vTy = llvm::dyn_cast<RankedTensorType>(op->getOperand(2).getType());
  auto gateTy = llvm::dyn_cast<RankedTensorType>(op->getOperand(3).getType());
  auto outTy = llvm::dyn_cast<RankedTensorType>(op->getResult(0).getType());
  if (!qTy || !kTy || !vTy || !gateTy || !outTy)
    return false;
  if (!isStaticRank4F32Tensor(qTy) || !isStaticRank4F32Tensor(kTy) ||
      !isStaticRank4F32Tensor(vTy) || !isStaticRank4F32Tensor(gateTy) ||
      !isStaticRank4F32Tensor(outTy))
    return false;
  if (qTy.getShape() != kTy.getShape() || qTy.getShape() != vTy.getShape() ||
      qTy.getShape() != outTy.getShape())
    return false;
  auto blockAttr = op->getAttrOfType<IntegerAttr>("block_size");
  auto windowAttr = op->getAttrOfType<IntegerAttr>("window_size");
  auto topKAttr = op->getAttrOfType<IntegerAttr>("top_k");
  if (!blockAttr || !windowAttr || !topKAttr)
    return false;
  int64_t block = blockAttr.getInt();
  int64_t window = windowAttr.getInt();
  int64_t topK = topKAttr.getInt();
  int64_t seq = qTy.getDimSize(2);
  if (block <= 0 || window <= 0 || topK <= 0 || seq % block != 0)
    return false;
  int64_t numBlocks = seq / block;
  if (topK > numBlocks)
    return false;
  return gateTy.getDimSize(0) == qTy.getDimSize(0) &&
         gateTy.getDimSize(1) == qTy.getDimSize(1) &&
         gateTy.getDimSize(2) == qTy.getDimSize(2) &&
         gateTy.getDimSize(3) == numBlocks;
}

bool verifyPPOPolicyLossValueEnvelope(Operation *op) {
  if (op->getNumOperands() < 3 || op->getNumOperands() > 6 ||
      op->getNumResults() != 1)
    return false;
  auto nextTy = llvm::dyn_cast<RankedTensorType>(op->getOperand(0).getType());
  auto oldTy = llvm::dyn_cast<RankedTensorType>(op->getOperand(1).getType());
  auto advTy = llvm::dyn_cast<RankedTensorType>(op->getOperand(2).getType());
  auto outTy = llvm::dyn_cast<RankedTensorType>(op->getResult(0).getType());
  if (!nextTy || !oldTy || !advTy || !outTy)
    return false;
  if (!nextTy.hasStaticShape() || !oldTy.hasStaticShape() ||
      !advTy.hasStaticShape() || !outTy.hasStaticShape())
    return false;
  if (!nextTy.getElementType().isF32() || !oldTy.getElementType().isF32() ||
      !advTy.getElementType().isF32() || !outTy.getElementType().isF32())
    return false;
  if (nextTy.getShape() != oldTy.getShape() ||
      nextTy.getShape() != advTy.getShape())
    return false;
  for (Value side : op->getOperands().drop_front(3)) {
    auto sideTy = llvm::dyn_cast<RankedTensorType>(side.getType());
    if (!sideTy || !sideTy.hasStaticShape() ||
        !sideTy.getElementType().isF32() ||
        sideTy.getShape() != nextTy.getShape())
      return false;
  }
  if (outTy.getRank() != 0)
    return false;
  if (auto reduction = op->getAttrOfType<StringAttr>("reduction");
      reduction && reduction.getValue() != "mean")
    return false;
  if (auto clip = op->getAttrOfType<FloatAttr>("clip_epsilon");
      clip && clip.getValueAsDouble() <= 0.0)
    return false;
  if (auto kl = op->getAttrOfType<FloatAttr>("kl_coef");
      kl && kl.getValueAsDouble() < 0.0)
    return false;
  if (auto entropy = op->getAttrOfType<FloatAttr>("entropy_coef");
      entropy && entropy.getValueAsDouble() < 0.0)
    return false;
  PPOOperandFlags flags = ppoOperandFlags(op);
  int64_t expectedOperands = 3 + (flags.hasMask ? 1 : 0) +
                             (flags.hasRefKL ? 1 : 0) +
                             (flags.hasEntropy ? 1 : 0);
  if (expectedOperands != op->getNumOperands())
    return false;
  if (auto kl = op->getAttrOfType<FloatAttr>("kl_coef");
      kl && kl.getValueAsDouble() != 0.0 && !flags.hasRefKL)
    return false;
  if (auto entropy = op->getAttrOfType<FloatAttr>("entropy_coef");
      entropy && entropy.getValueAsDouble() != 0.0 && !flags.hasEntropy)
    return false;
  return true;
}

bool verifyEBMEnergyQuadraticValueEnvelope(Operation *op) {
  if (op->getNumOperands() != 2 || op->getNumResults() != 1)
    return false;
  auto xTy = llvm::dyn_cast<RankedTensorType>(op->getOperand(0).getType());
  auto yTy = llvm::dyn_cast<RankedTensorType>(op->getOperand(1).getType());
  auto eTy = llvm::dyn_cast<RankedTensorType>(op->getResult(0).getType());
  if (!xTy || !yTy || !eTy)
    return false;
  if (!xTy.hasStaticShape() || !yTy.hasStaticShape() || !eTy.hasStaticShape())
    return false;
  if (!xTy.getElementType().isF32() || !yTy.getElementType().isF32() ||
      !eTy.getElementType().isF32())
    return false;
  if (xTy.getRank() != 2 || yTy.getRank() != 2 || eTy.getRank() != 1)
    return false;
  return xTy.getShape() == yTy.getShape() &&
         eTy.getDimSize(0) == xTy.getDimSize(0);
}

bool verifyEBMLangevinStepValueEnvelope(Operation *op) {
  if (op->getNumOperands() != 3 || op->getNumResults() != 1)
    return false;
  auto yTy = llvm::dyn_cast<RankedTensorType>(op->getOperand(0).getType());
  auto gTy = llvm::dyn_cast<RankedTensorType>(op->getOperand(1).getType());
  auto nTy = llvm::dyn_cast<RankedTensorType>(op->getOperand(2).getType());
  auto oTy = llvm::dyn_cast<RankedTensorType>(op->getResult(0).getType());
  if (!yTy || !gTy || !nTy || !oTy)
    return false;
  if (!yTy.hasStaticShape() || !gTy.hasStaticShape() ||
      !nTy.hasStaticShape() || !oTy.hasStaticShape())
    return false;
  if (!yTy.getElementType().isF32() || !gTy.getElementType().isF32() ||
      !nTy.getElementType().isF32() || !oTy.getElementType().isF32())
    return false;
  if (yTy.getShape() != gTy.getShape() || yTy.getShape() != nTy.getShape() ||
      yTy.getShape() != oTy.getShape())
    return false;
  auto eta = op->getAttrOfType<FloatAttr>("eta");
  if (!eta || eta.getValueAsDouble() <= 0.0)
    return false;
  if (auto scale = op->getAttrOfType<FloatAttr>("noise_scale");
      scale && scale.getValueAsDouble() < 0.0)
    return false;
  return true;
}

bool verifyEBMRefinementValueEnvelope(Operation *op) {
  if (op->getNumOperands() != 2 || op->getNumResults() != 1)
    return false;
  auto yTy = llvm::dyn_cast<RankedTensorType>(op->getOperand(0).getType());
  auto gTy = llvm::dyn_cast<RankedTensorType>(op->getOperand(1).getType());
  auto oTy = llvm::dyn_cast<RankedTensorType>(op->getResult(0).getType());
  if (!yTy || !gTy || !oTy)
    return false;
  if (!yTy.hasStaticShape() || !gTy.hasStaticShape() || !oTy.hasStaticShape())
    return false;
  if (!yTy.getElementType().isF32() || !gTy.getElementType().isF32() ||
      !oTy.getElementType().isF32())
    return false;
  if (yTy.getShape() != gTy.getShape() || yTy.getShape() != oTy.getShape())
    return false;
  auto eta = op->getAttrOfType<FloatAttr>("eta");
  auto steps = op->getAttrOfType<IntegerAttr>("steps");
  return eta && eta.getValueAsDouble() > 0.0 && steps && steps.getInt() > 0 &&
         !op->hasAttr("temperature") && !op->hasAttr("noise_scale");
}

bool verifyEBMPartitionExactValueEnvelope(Operation *op) {
  if (op->getNumOperands() != 1 || op->getNumResults() != 1)
    return false;
  auto eTy = llvm::dyn_cast<RankedTensorType>(op->getOperand(0).getType());
  auto oTy = llvm::dyn_cast<RankedTensorType>(op->getResult(0).getType());
  if (!eTy || !oTy)
    return false;
  if (!eTy.hasStaticShape() || !oTy.hasStaticShape())
    return false;
  if (!eTy.getElementType().isF32() || !oTy.getElementType().isF32())
    return false;
  if (oTy.getRank() != 0)
    return false;
  if (auto temperature = op->getAttrOfType<FloatAttr>("temperature");
      temperature && temperature.getValueAsDouble() <= 0.0)
    return false;
  if (auto reduction = op->getAttrOfType<StringAttr>("reduction");
      reduction && reduction.getValue() != "logsumexp")
    return false;
  return true;
}

bool verifyCliffordGeometricProductValueEnvelope(Operation *op) {
  if (op->getNumOperands() != 2 || op->getNumResults() != 1)
    return false;
  auto lhsTy = llvm::dyn_cast<RankedTensorType>(op->getOperand(0).getType());
  auto rhsTy = llvm::dyn_cast<RankedTensorType>(op->getOperand(1).getType());
  auto resTy = llvm::dyn_cast<RankedTensorType>(op->getResult(0).getType());
  if (!lhsTy || !rhsTy || !resTy)
    return false;
  if (!lhsTy.hasStaticShape() || !rhsTy.hasStaticShape() ||
      !resTy.hasStaticShape())
    return false;
  if (!lhsTy.getElementType().isF32() || !rhsTy.getElementType().isF32() ||
      !resTy.getElementType().isF32())
    return false;
  if (lhsTy.getRank() < 1 || rhsTy.getRank() != lhsTy.getRank() ||
      resTy.getRank() != lhsTy.getRank())
    return false;
  if (lhsTy.getDimSize(lhsTy.getRank() - 1) != 8 ||
      rhsTy.getDimSize(rhsTy.getRank() - 1) != 8 ||
      resTy.getDimSize(resTy.getRank() - 1) != 8)
    return false;
  if (lhsTy.getShape() != rhsTy.getShape() ||
      lhsTy.getShape() != resTy.getShape())
    return false;
  auto p = op->getAttrOfType<IntegerAttr>("p");
  auto q = op->getAttrOfType<IntegerAttr>("q");
  return p && q && p.getInt() == 3 && q.getInt() == 0;
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
       {"lower", "trans", "unit_diag", "full_matrices", "window_size",
        "block_size", "top_k", "causal", "clip_epsilon", "kl_coef",
        "entropy_coef", "has_mask", "has_ref_kl", "has_entropy",
        "reduction", "eta", "temperature", "noise_scale", "has_noise",
        "steps", "p", "q", "coefficient_layout", "has_signature",
        "has_grade_mask"}) {
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
        if (sp) {
          llvm::StringRef opKind = sp->graphName;
          opKind.consume_front("tessera.");
          emitAppleValueCall(builder, op, "tessera_apple.cpu.call", opKind,
                             sp->cpuSymbol, sp->cpuAbi, "executable",
                             "Accelerate");
          ++ordinal;
          continue;
        }
        // Sprint 5: dense fp32 rank-2 matmul/gemm executes as a single
        // Accelerate GEMM value call. ONLY the vetted tile op (tile.matmul /
        // tile.gemm, emitted by TilingPass's TileMatmulValue for the static
        // rank-2 f32 envelope) is executable here. A raw `tessera.matmul` that
        // reaches this pass was *rejected* by that envelope (e.g. f16/bf16, or
        // dynamic shape) — it must fall through to the named diagnostic and be
        // gated, never silently dispatched as f32. op_kind distinguishes
        // matmul vs gemm; both use the one tessera_apple_cpu_gemm_f32 symbol.
        if (name == "tile.matmul" || name == "tile.gemm") {
          bool isGemm = name == "tile.gemm" || src == "tessera.gemm";
          // Sprint 7: pick the dtype-specific GEMM symbol from the result
          // element type. f32 → cblas_sgemm; f16/bf16 → BNNS. TilingPass's
          // TileMatmulValue guarantees a single shared float element type, so
          // reading the result type is sufficient.
          llvm::StringRef symbol = "tessera_apple_cpu_gemm_f32";
          llvm::StringRef abi = "cblas_sgemm";
          if (auto rt = llvm::dyn_cast<RankedTensorType>(
                  op->getResult(0).getType())) {
            Type et = rt.getElementType();
            if (et.isF16()) {
              symbol = "tessera_apple_cpu_gemm_f16";
              abi = "bnns_matmul_f16";
            } else if (et.isBF16()) {
              symbol = "tessera_apple_cpu_gemm_bf16";
              abi = "bnns_matmul_bf16";
            }
          }
          emitAppleValueCall(builder, op, "tessera_apple.cpu.call",
                             isGemm ? "gemm" : "matmul", symbol, abi,
                             "executable", "Accelerate");
          ++ordinal;
          continue;
        }
        // Sprint 6: static rank-3 *f32* batched matmul → Accelerate batched GEMM
        // value call. CPU batched is fp32-only (no batched f16/bf16 on CPU yet —
        // Sprint 8 wired batched f16/bf16 on the GPU lane only). TileMatmulValue
        // may now produce an f16/bf16 tile.batched_gemm (shared with the GPU
        // pipeline), so gate non-f32 here → it falls to the named diagnostic.
        if (name == "tile.batched_gemm") {
          bool isF32Batched = true;
          if (auto rt = llvm::dyn_cast<RankedTensorType>(
                  op->getResult(0).getType()))
            isF32Batched = rt.getElementType().isF32();
          if (isF32Batched) {
            emitAppleValueCall(builder, op, "tessera_apple.cpu.call",
                               "batched_gemm",
                               "tessera_apple_cpu_gemm_f32_batched",
                               "cblas_sgemm_batched_loop", "executable",
                               "Accelerate");
            ++ordinal;
            continue;
          }
          // non-f32 batched on CPU → gated (fall through to diagnostic).
        }
        op->emitError("apple_cpu value lowering: no value-producing CPU "
                      "target op for '")
            << src << "' (executable envelope: linalg family + static rank-2 "
                      "f32 matmul + static rank-3 f32 batched_gemm)";
        signalPassFailure();
        return;
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
      } else if (isMatmul(name) || isMatmul(src) || isBatchedMatmul(name) ||
                 isBatchedMatmul(src)) {
        llvm::SmallVector<NamedAttribute, 2> extra;
        extra.emplace_back(builder.getStringAttr("framework"),
                           builder.getStringAttr("Accelerate"));
        extra.emplace_back(builder.getStringAttr("abi"),
                           builder.getStringAttr(
                               isBatchedMatmul(name) || isBatchedMatmul(src)
                                   ? "cblas_sgemm_batched_loop"
                                   : "cblas_sgemm"));
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
        if (sp) {
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
          emitAppleValueCall(builder, op, "tessera_apple.gpu.kernel_call",
                             opKind, sp->gpuSymbol, "msl", "executable", "Metal");
          ++ordinal;
          continue;
        }
        // Sprint 8: static rank-3 batched matmul executes on the GPU value lane
        // for f32/f16/bf16 via the MPSGraph-backed bmm symbols. Only the vetted
        // tile op (TileBatchedMatmulValue, strict static rank-3 envelope)
        // reaches here; the dtype-specific symbol is read from the result type.
        if (name == "tile.batched_gemm") {
          llvm::StringRef symbol = "tessera_apple_gpu_bmm_f32";
          if (auto rt = llvm::dyn_cast<RankedTensorType>(
                  op->getResult(0).getType())) {
            Type et = rt.getElementType();
            if (et.isF16())
              symbol = "tessera_apple_gpu_bmm_f16";
            else if (et.isBF16())
              symbol = "tessera_apple_gpu_bmm_bf16";
          }
          emitAppleValueCall(builder, op, "tessera_apple.gpu.kernel_call",
                             "batched_gemm", symbol, "mps", "executable",
                             "Metal");
          ++ordinal;
          continue;
        }
        if (isPPOPolicyLoss(name) || isPPOPolicyLoss(src)) {
          if (!verifyPPOPolicyLossValueEnvelope(op)) {
            op->emitError("apple_gpu value lowering: ppo_policy_loss requires "
                          "3 to 6 static fp32 operands with identical shapes, "
                          "rank-0 fp32 result, reduction=\"mean\", positive "
                          "clip_epsilon, valid optional mask/ref_logp/entropy "
                          "flags, and non-negative side-term coefficients");
            signalPassFailure();
            return;
          }
          PPOOperandFlags flags = ppoOperandFlags(op);
          bool extended = flags.hasMask || flags.hasRefKL || flags.hasEntropy;
          emitAppleValueCall(builder, op, "tessera_apple.gpu.kernel_call",
                             "ppo_policy_loss",
                             extended
                                 ? "tessera_apple_gpu_ppo_policy_loss_ex_f32"
                                 : "tessera_apple_gpu_ppo_policy_loss_f32",
                             extended ? "mpsgraph_ppo_policy_loss_ex_f32"
                                      : "mpsgraph_ppo_policy_loss_f32",
                             "executable", "MPSGraph");
          ++ordinal;
          continue;
        }
        if (isEBMEnergyQuadratic(name) || isEBMEnergyQuadratic(src)) {
          if (!verifyEBMEnergyQuadraticValueEnvelope(op)) {
            op->emitError("apple_gpu value lowering: ebm.energy_quadratic "
                          "requires static fp32 rank-2 x/y with matching "
                          "shape and rank-1 fp32 energies[B]");
            signalPassFailure();
            return;
          }
          emitAppleValueCall(builder, op, "tessera_apple.gpu.kernel_call",
                             "ebm_energy_quadratic",
                             "tessera_apple_gpu_ebm_energy_quadratic_value_f32",
                             "msl_ebm_energy_quadratic_value_f32",
                             "executable", "Metal");
          ++ordinal;
          continue;
        }
        if (isEBMLangevinStep(name) || isEBMLangevinStep(src)) {
          if (!verifyEBMLangevinStepValueEnvelope(op)) {
            op->emitError("apple_gpu value lowering: ebm.langevin_step "
                          "requires static fp32 y/grad/noise/result with "
                          "matching shapes, positive eta, and non-negative "
                          "noise_scale");
            signalPassFailure();
            return;
          }
          emitAppleValueCall(builder, op, "tessera_apple.gpu.kernel_call",
                             "ebm_langevin_step",
                             "tessera_apple_gpu_ebm_langevin_step_value_f32",
                             "msl_ebm_langevin_step_value_f32", "executable",
                             "Metal");
          ++ordinal;
          continue;
        }
        if (isEBMRefinement(name) || isEBMRefinement(src)) {
          if (!verifyEBMRefinementValueEnvelope(op)) {
            op->emitError("apple_gpu value lowering: ebm.refinement requires "
                          "static fp32 y/grad/result with matching shapes, "
                          "positive eta, positive steps, and no "
                          "temperature/noise_scale side semantics");
            signalPassFailure();
            return;
          }
          emitAppleValueCall(builder, op, "tessera_apple.gpu.kernel_call",
                             "ebm_refinement",
                             "tessera_apple_gpu_ebm_refinement_value_f32",
                             "msl_ebm_refinement_value_f32", "executable",
                             "Metal");
          ++ordinal;
          continue;
        }
        if (isEBMPartitionExact(name) || isEBMPartitionExact(src)) {
          if (!verifyEBMPartitionExactValueEnvelope(op)) {
            op->emitError("apple_gpu value lowering: ebm.partition_exact "
                          "requires static fp32 energies, scalar fp32 result, "
                          "positive temperature, and reduction=\"logsumexp\"");
            signalPassFailure();
            return;
          }
          emitAppleValueCall(builder, op, "tessera_apple.gpu.kernel_call",
                             "ebm_partition_exact",
                             "tessera_apple_gpu_ebm_partition_exact_value_f32",
                             "msl_ebm_partition_exact_value_f32",
                             "executable", "Metal");
          ++ordinal;
          continue;
        }
        if ((isCliffordGeometricProduct(name) ||
             isCliffordGeometricProduct(src))) {
          if (!verifyCliffordGeometricProductValueEnvelope(op)) {
            op->emitError("apple_gpu value lowering: "
                          "clifford.geometric_product requires static fp32 "
                          "cl30 tensors with matching shapes, p=3, q=0, and "
                          "last dimension 8");
            signalPassFailure();
            return;
          }
          emitAppleValueCall(builder, op, "tessera_apple.gpu.kernel_call",
                             "clifford_geometric_product",
                             "tessera_apple_gpu_clifford_geo_product_cl30_value_f32",
                             "msl_clifford_geo_product_cl30_value_f32",
                             "executable", "Metal");
          ++ordinal;
          continue;
        }
        if (isNativeSparseAttnFused(name) || isNativeSparseAttnFused(src)) {
          if (!verifyNativeSparseValueEnvelope(op)) {
            op->emitError("apple_gpu value lowering: native_sparse_attn_fused "
                          "requires static rank-4 fp32 Q/K/V/O, static rank-4 "
                          "fp32 gate logits [B,H,S,S/block_size], positive "
                          "window_size/block_size/top_k, S divisible by "
                          "block_size, and top_k <= S/block_size");
            signalPassFailure();
            return;
          }
          emitAppleValueCall(builder, op, "tessera_apple.gpu.kernel_call",
                             "native_sparse_attn_fused",
                             "tessera_apple_gpu_native_sparse_attn_f32",
                             "msl_native_sparse_attn_f32", "executable",
                             "Metal");
          ++ordinal;
          continue;
        }
        op->emitError("apple_gpu value lowering: no value-producing GPU "
                      "target op for '")
            << src << "'";
        signalPassFailure();
        return;
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
