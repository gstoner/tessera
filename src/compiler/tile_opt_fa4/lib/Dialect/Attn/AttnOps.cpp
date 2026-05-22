//===- AttnOps.cpp — FA-4 attention op verifiers + helpers ───────────────===//
//
// Implements verifiers for the Phase 3 attention ops defined in Attn.td:
//   LseSaveOp, ScaledDotProductOp, OnlineSoftmaxOp (FA-2 invariants)
//
// All ops use ODS-generated accessors; this file only fills hasVerifier=1 slots
// and provides the FA-2 online softmax documentation helpers.
//
// FA-2 online softmax algorithm (Dao et al. 2022):
//
//   For each KV tile j:
//     S_j  = Q · K_j^T / sqrt(d_k)               [ScaledDotProductOp]
//     m_j  = max(m_{j-1}, rowmax(S_j))
//     l_j  = exp(m_{j-1} - m_j) * l_{j-1}
//            + rowsum(exp(S_j - m_j))
//     O_j  = diag(exp(m_{j-1} - m_j)) * O_{j-1}
//            + exp(S_j - m_j) * V_j               [OnlineSoftmaxOp]
//   Final: O = diag(1/l) * O,  LSE = m + log(l)  [LseAccumulateOp]
//
//===----------------------------------------------------------------------===//

#include "mlir/Bytecode/BytecodeOpInterface.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Support/LogicalResult.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/StringSwitch.h"

// ODS-generated declarations (produced by mlir-tblgen from Attn.td).
// Include the generated header; actual path depends on CMake binary dir.
#include "AttnDialect.h.inc"
#define GET_OP_CLASSES
#include "AttnOps.h.inc"

namespace tessera {
namespace attn {

// ── LseSaveOp verifier ────────────────────────────────────────────────────
// scores must be a ranked tensor; LSE result has one fewer dimension (row vec).
mlir::LogicalResult LseSaveOp::verify() {
  auto srcType = mlir::dyn_cast<mlir::RankedTensorType>(getScores().getType());
  if (!srcType)
    return emitOpError("scores must be a ranked tensor");
  if (srcType.getRank() < 2)
    return emitOpError("scores tensor must have rank >= 2 (at least [rows, cols])");
  return mlir::success();
}

// Sprint V6c (2026-05-22) — target-aware tile size limits.
//
// Per-SM (tile_q_max, tile_kv_max) for the FA-4 ScaledDotProduct kernel.
// Generalizes the V3 FlashAttnOp head_dim pattern to the FA-4 Tile IR
// op family.  Numbers come from the canonical FA-2/FA-3/FA-4 kernel
// shapes documented in `docs/nvidia_cuda13_kernel_inventory.md` +
// CLAUDE.md Phase 3 description.
//
// sm_70 / sm_75 / sm_80 / sm_86 / sm_89  — FA-2 baseline: 64 × 128
// sm_90 / sm_90a / sm_100 / sm_100a / sm_120 / sm_120a — FA-3/FA-4: 128 × 256
// no SM tag — CPU reference; no tile-size limit applied.
//
// Returns {0, 0} when the SM is unknown (so the verifier skips the
// check rather than rejecting unrecognized strings).
static std::pair<int64_t, int64_t>
maxTileSizesForTargetSm(llvm::StringRef sm) {
  return llvm::StringSwitch<std::pair<int64_t, int64_t>>(sm)
      .Case("sm_70",   {64, 128})
      .Case("sm_75",   {64, 128})
      .Case("sm_80",   {64, 128})
      .Case("sm_86",   {64, 128})
      .Case("sm_89",   {64, 128})
      .Case("sm_90",   {128, 256})
      .Case("sm_90a",  {128, 256})
      .Case("sm_100",  {128, 256})
      .Case("sm_100a", {128, 256})
      .Case("sm_120",  {128, 256})
      .Case("sm_120a", {128, 256})
      .Default({0, 0});  // unknown / no limit applied
}

// ── ScaledDotProductOp verifier ───────────────────────────────────────────
// query: [tile_q × d_k]  key: [tile_kv × d_k]  → scores: [tile_q × tile_kv]
//
// Sprint V6c (2026-05-22): also walks the parent op chain for a
// `tessera.target_sm` attribute and enforces the per-SM (tile_q, tile_kv)
// ceiling.  The Q tile size is read from query.dim(0); the KV tile
// size from key.dim(0).  Functions without the attribute (CPU
// reference path) skip the target-aware check.
mlir::LogicalResult ScaledDotProductOp::verify() {
  auto qType = mlir::dyn_cast<mlir::RankedTensorType>(getQuery().getType());
  auto kType = mlir::dyn_cast<mlir::RankedTensorType>(getKey().getType());
  if (!qType || !kType)
    return emitOpError("query and key must be ranked tensors");
  if (qType.getRank() != 2 || kType.getRank() != 2)
    return emitOpError("query and key must be 2-D tensors [tile × head_dim]");
  // d_k dimension must match (dim 1 of Q == dim 1 of K).
  if (!qType.isDynamicDim(1) && !kType.isDynamicDim(1) &&
      qType.getDimSize(1) != kType.getDimSize(1))
    return emitOpError("query and key head_dim mismatch: ")
           << qType.getDimSize(1) << " vs " << kType.getDimSize(1);
  double scale = getScale().convertToDouble();
  if (scale <= 0.0 && scale != -1.0)
    return emitOpError("scale must be positive (or -1.0 as sentinel)");
  // Sprint V6c — target-aware tile size limits.  Walk parent op chain
  // for `tessera.target_sm`; enforce per-SM (tile_q_max, tile_kv_max).
  mlir::Operation *parent = (*this)->getParentOp();
  while (parent && !parent->hasAttr("tessera.target_sm"))
    parent = parent->getParentOp();
  if (parent) {
    if (auto attr = mlir::dyn_cast<mlir::StringAttr>(
            parent->getAttr("tessera.target_sm"))) {
      auto [tileQMax, tileKvMax] = maxTileSizesForTargetSm(attr.getValue());
      if (tileQMax > 0 && !qType.isDynamicDim(0)
          && qType.getDimSize(0) > tileQMax)
        return emitOpError("tile_q=")
               << qType.getDimSize(0)
               << " exceeds the SM " << attr.getValue()
               << " ScaledDotProduct kernel limit of " << tileQMax
               << " (Sprint V6c FA-4 tile size table)";
      if (tileKvMax > 0 && !kType.isDynamicDim(0)
          && kType.getDimSize(0) > tileKvMax)
        return emitOpError("tile_kv=")
               << kType.getDimSize(0)
               << " exceeds the SM " << attr.getValue()
               << " ScaledDotProduct kernel limit of " << tileKvMax
               << " (Sprint V6c FA-4 tile size table)";
    }
  }
  return mlir::success();
}

// ── OnlineSoftmaxOp verifier ──────────────────────────────────────────────
// Verifies the FA-2 loop-invariant shapes are consistent.
mlir::LogicalResult OnlineSoftmaxOp::verify() {
  // scores: [tile_q × tile_kv]
  auto sType = mlir::dyn_cast<mlir::RankedTensorType>(getScores().getType());
  if (!sType || sType.getRank() != 2)
    return emitOpError("scores must be a 2-D ranked tensor");

  // running_m / running_l: scalar f32 or 1-D [tile_q] — accept both.
  // acc_out: [tile_q × d_v] — opaque check only.
  return mlir::success();
}

void TesseraAttnDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "AttnOps.cpp.inc"
      >();
}

} // namespace attn
} // namespace tessera

// ── Register dialect ops from ODS-generated definitions ──────────────────
#include "AttnDialect.cpp.inc"
#define GET_OP_CLASSES
#include "AttnOps.cpp.inc"
